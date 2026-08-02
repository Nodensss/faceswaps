package com.faceswaplocal.app.inference

import kotlin.math.ceil
import kotlin.math.exp
import kotlin.math.floor
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/** Integer target-image bounds. [right] and [bottom] are exclusive. */
data class CompositeRoi(
    val left: Int,
    val top: Int,
    val right: Int,
    val bottom: Int,
) {
    val width: Int get() = right - left
    val height: Int get() = bottom - top
}

/** Per-channel affine colour correction applied before pasting the crop. */
data class FaceColorAdjustment(
    val redGain: Double,
    val greenGain: Double,
    val blueGain: Double,
    val redOffset: Double,
    val greenOffset: Double,
    val blueOffset: Double,
    val strength: Double,
)

data class ColorMatchedCrop(
    val pixels: IntArray,
    val adjustment: FaceColorAdjustment,
)

/** Result of [FaceCompositor.pasteBack]: the blended image and the warped alpha mask. */
data class PasteBackResult(
    val pixels: IntArray,
    val warpedMask: FloatArray,
    val roi: CompositeRoi,
)

/**
 * Pure-array Stage C output. [warpedMask] has the same dimensions as [pixels], while
 * [cropMask] and [colorMatchedCrop] use the swapper crop dimensions supplied to
 * [FaceCompositor.composite].
 */
data class FaceCompositeResult(
    val pixels: IntArray,
    val width: Int,
    val height: Int,
    val cropMask: FloatArray,
    val warpedMask: FloatArray,
    val colorMatchedCrop: IntArray,
    val colorAdjustment: FaceColorAdjustment,
    val roi: CompositeRoi,
)

/**
 * FaceFusion-compatible affine box-mask compositor without Android graphics types.
 *
 * Inputs and outputs are packed ARGB pixels. The target alpha channel is retained;
 * only RGB is colour-matched and blended.
 */
object FaceCompositor {
    /**
     * Colour-matches [swappedCropPixels] to the already aligned target crop, warps the
     * crop and box mask back through [targetToCrop], and blends them into the target.
     */
    fun composite(
        targetPixels: IntArray,
        targetWidth: Int,
        targetHeight: Int,
        targetCropPixels: IntArray,
        swappedCropPixels: IntArray,
        cropWidth: Int,
        cropHeight: Int,
        targetToCrop: AffineMatrix,
        /** Optional aligned-crop mask that can only reduce the box-mask paste alpha. */
        blendConstraintMask: FloatArray? = null,
    ): FaceCompositeResult {
        requireImage(targetPixels, targetWidth, targetHeight, "Target")
        requireImage(targetCropPixels, cropWidth, cropHeight, "Aligned target crop")
        requireImage(swappedCropPixels, cropWidth, cropHeight, "Swapped crop")
        requireFinite(targetToCrop)

        val boxMask = createBoxMask(cropWidth, cropHeight)
        blendConstraintMask?.let { constraint ->
            require(constraint.size == boxMask.size) {
                "Blend constraint mask size must match crop dimensions"
            }
            require(constraint.all { it.isFinite() && it in 0f..1f }) {
                "Blend constraint mask values must be finite and within 0..1"
            }
        }
        val effectiveCropMask = if (blendConstraintMask == null) {
            // Preserve the pre-parser path exactly, including the generated mask values.
            boxMask
        } else {
            FloatArray(boxMask.size) { index ->
                min(boxMask[index], blendConstraintMask[index])
            }
        }
        val colorMatch = matchCropColors(
            targetCropPixels = targetCropPixels,
            swappedCropPixels = swappedCropPixels,
            // Parser coverage must not change the established Stage C colour statistics.
            mask = boxMask,
            width = cropWidth,
            height = cropHeight,
        )
        val paste = pasteBack(
            basePixels = targetPixels,
            baseWidth = targetWidth,
            baseHeight = targetHeight,
            cropPixels = colorMatch.pixels,
            cropMask = effectiveCropMask,
            cropWidth = cropWidth,
            cropHeight = cropHeight,
            baseToCrop = targetToCrop,
        )

        return FaceCompositeResult(
            pixels = paste.pixels,
            width = targetWidth,
            height = targetHeight,
            cropMask = effectiveCropMask,
            warpedMask = paste.warpedMask,
            colorMatchedCrop = colorMatch.pixels,
            colorAdjustment = colorMatch.adjustment,
            roi = paste.roi,
        )
    }

    /**
     * Warps an arbitrary aligned [cropPixels] and [cropMask] back through [baseToCrop],
     * then alpha-blends them into a copy of [basePixels]. This is shared by the swap
     * compositor and the later FFHQ-512 restoration pass.
     *
     * The base alpha channel is preserved, and pixels outside the inverse crop bounds
     * are left bit-for-bit unchanged.
     */
    fun pasteBack(
        basePixels: IntArray,
        baseWidth: Int,
        baseHeight: Int,
        cropPixels: IntArray,
        cropMask: FloatArray,
        cropWidth: Int,
        cropHeight: Int,
        baseToCrop: AffineMatrix,
        /** Full-image regions that this paste is forbidden to alter. */
        protectedBaseRois: List<CompositeRoi> = emptyList(),
    ): PasteBackResult {
        requireImage(basePixels, baseWidth, baseHeight, "Base")
        requireImage(cropPixels, cropWidth, cropHeight, "Paste crop")
        require(cropMask.size == checkedPixelCount(cropWidth, cropHeight)) {
            "Mask size must match crop dimensions"
        }
        require(cropMask.all(Float::isFinite)) { "Mask values must be finite" }
        requireFinite(baseToCrop)
        require(protectedBaseRois.all { roi -> roi.width >= 0 && roi.height >= 0 }) {
            "Protected ROI bounds must be ordered"
        }

        val roi = calculateRoi(
            targetWidth = baseWidth,
            targetHeight = baseHeight,
            cropWidth = cropWidth,
            cropHeight = cropHeight,
            targetToCrop = baseToCrop,
        )
        val resultPixels = basePixels.copyOf()
        val warpedMask = FloatArray(basePixels.size)

        for (baseY in roi.top until roi.bottom) {
            for (baseX in roi.left until roi.right) {
                if (protectedBaseRois.any { protected ->
                        baseX in protected.left until protected.right &&
                            baseY in protected.top until protected.bottom
                    }
                ) {
                    continue
                }
                val cropPoint = baseToCrop.map(Point2(baseX.toDouble(), baseY.toDouble()))
                val alpha = sampleMaskConstantZero(
                    mask = cropMask,
                    width = cropWidth,
                    height = cropHeight,
                    x = cropPoint.x,
                    y = cropPoint.y,
                ).coerceIn(0.0, 1.0)
                val baseIndex = baseY * baseWidth + baseX
                warpedMask[baseIndex] = alpha.toFloat()
                if (alpha <= 0.0) continue

                val cropColor = sampleArgbEdgeReplicate(
                    pixels = cropPixels,
                    width = cropWidth,
                    height = cropHeight,
                    x = cropPoint.x,
                    y = cropPoint.y,
                )
                val baseColor = basePixels[baseIndex]
                val inverseAlpha = 1.0 - alpha
                val red = (
                    channel(baseColor, RED_SHIFT) * inverseAlpha + cropColor.red * alpha
                    ).coerceIn(0.0, 255.0).toInt()
                val green = (
                    channel(baseColor, GREEN_SHIFT) * inverseAlpha + cropColor.green * alpha
                    ).coerceIn(0.0, 255.0).toInt()
                val blue = (
                    channel(baseColor, BLUE_SHIFT) * inverseAlpha + cropColor.blue * alpha
                    ).coerceIn(0.0, 255.0).toInt()
                resultPixels[baseIndex] =
                    (baseColor and ALPHA_MASK) or
                        (red shl RED_SHIFT) or
                        (green shl GREEN_SHIFT) or
                        blue
            }
        }

        return PasteBackResult(
            pixels = resultPixels,
            warpedMask = warpedMask,
            roi = roi,
        )
    }

    /** FaceFusion 3.7.1 box mask with blur 0.3 and zero padding. */
    fun createBoxMask(width: Int, height: Int): FloatArray {
        val pixelCount = checkedPixelCount(width, height)
        val blurAmount = (width * 0.5 * BOX_MASK_BLUR).toInt()
        val blurArea = max(blurAmount / 2, 1)
        val mask = FloatArray(pixelCount) { 1f }
        val verticalBorder = min(blurArea, height)
        val horizontalBorder = min(blurArea, width)

        for (y in 0 until verticalBorder) {
            mask.fill(0f, y * width, (y + 1) * width)
        }
        for (y in max(height - verticalBorder, 0) until height) {
            mask.fill(0f, y * width, (y + 1) * width)
        }
        for (y in 0 until height) {
            val row = y * width
            for (x in 0 until horizontalBorder) mask[row + x] = 0f
            for (x in max(width - horizontalBorder, 0) until width) mask[row + x] = 0f
        }

        if (blurAmount == 0) return mask
        return gaussianBlurReflect101(
            input = mask,
            width = width,
            height = height,
            sigma = blurAmount * BOX_MASK_SIGMA_FACTOR,
        )
    }

    /**
     * Matches per-channel mean and standard deviation inside the supported face mask.
     * Pixels below 0.5 are excluded; retained samples are weighted by their mask value.
     */
    fun matchCropColors(
        targetCropPixels: IntArray,
        swappedCropPixels: IntArray,
        mask: FloatArray,
        width: Int,
        height: Int,
    ): ColorMatchedCrop {
        requireImage(targetCropPixels, width, height, "Aligned target crop")
        requireImage(swappedCropPixels, width, height, "Swapped crop")
        require(mask.size == targetCropPixels.size) {
            "Mask size must match crop dimensions"
        }
        require(mask.all(Float::isFinite)) { "Mask values must be finite" }

        val targetStats = weightedRgbStats(targetCropPixels, mask)
        val swappedStats = weightedRgbStats(swappedCropPixels, mask)
        val adjustment = FaceColorAdjustment(
            redGain = calculateGain(targetStats.redStd, swappedStats.redStd),
            greenGain = calculateGain(targetStats.greenStd, swappedStats.greenStd),
            blueGain = calculateGain(targetStats.blueStd, swappedStats.blueStd),
            redOffset = 0.0,
            greenOffset = 0.0,
            blueOffset = 0.0,
            strength = COLOR_MATCH_STRENGTH,
        ).withOffsets(targetStats, swappedStats)

        val output = IntArray(swappedCropPixels.size)
        swappedCropPixels.forEachIndexed { index, pixel ->
            val red = adjustChannel(
                value = channel(pixel, RED_SHIFT),
                gain = adjustment.redGain,
                offset = adjustment.redOffset,
            )
            val green = adjustChannel(
                value = channel(pixel, GREEN_SHIFT),
                gain = adjustment.greenGain,
                offset = adjustment.greenOffset,
            )
            val blue = adjustChannel(
                value = channel(pixel, BLUE_SHIFT),
                gain = adjustment.blueGain,
                offset = adjustment.blueOffset,
            )
            output[index] =
                (pixel and ALPHA_MASK) or
                    (red shl RED_SHIFT) or
                    (green shl GREEN_SHIFT) or
                    blue
        }
        return ColorMatchedCrop(output, adjustment)
    }

    private fun FaceColorAdjustment.withOffsets(
        target: RgbStats,
        source: RgbStats,
    ): FaceColorAdjustment = copy(
        redOffset = calculateOffset(target.redMean, source.redMean, redGain),
        greenOffset = calculateOffset(target.greenMean, source.greenMean, greenGain),
        blueOffset = calculateOffset(target.blueMean, source.blueMean, blueGain),
    )

    private fun weightedRgbStats(pixels: IntArray, mask: FloatArray): RgbStats {
        var weightSum = 0.0
        var redSum = 0.0
        var greenSum = 0.0
        var blueSum = 0.0
        pixels.forEachIndexed { index, pixel ->
            val weight = mask[index].toDouble()
            if (weight >= MASK_STAT_THRESHOLD) {
                weightSum += weight
                redSum += channel(pixel, RED_SHIFT) * weight
                greenSum += channel(pixel, GREEN_SHIFT) * weight
                blueSum += channel(pixel, BLUE_SHIFT) * weight
            }
        }
        require(weightSum > 0.0) { "Mask has no samples at or above 0.5" }

        val redMean = redSum / weightSum
        val greenMean = greenSum / weightSum
        val blueMean = blueSum / weightSum
        var redVariance = 0.0
        var greenVariance = 0.0
        var blueVariance = 0.0
        pixels.forEachIndexed { index, pixel ->
            val weight = mask[index].toDouble()
            if (weight >= MASK_STAT_THRESHOLD) {
                redVariance += squared(channel(pixel, RED_SHIFT) - redMean) * weight
                greenVariance += squared(channel(pixel, GREEN_SHIFT) - greenMean) * weight
                blueVariance += squared(channel(pixel, BLUE_SHIFT) - blueMean) * weight
            }
        }
        return RgbStats(
            redMean = redMean,
            greenMean = greenMean,
            blueMean = blueMean,
            redStd = sqrt(max(redVariance / weightSum, 0.0)),
            greenStd = sqrt(max(greenVariance / weightSum, 0.0)),
            blueStd = sqrt(max(blueVariance / weightSum, 0.0)),
        )
    }

    private fun calculateGain(targetStd: Double, sourceStd: Double): Double =
        if (sourceStd <= MIN_STANDARD_DEVIATION) {
            1.0
        } else {
            (targetStd / sourceStd).coerceIn(MIN_GAIN, MAX_GAIN)
        }

    private fun calculateOffset(targetMean: Double, sourceMean: Double, gain: Double): Double =
        (targetMean - sourceMean * gain).coerceIn(MIN_OFFSET, MAX_OFFSET)

    private fun adjustChannel(value: Int, gain: Double, offset: Double): Int {
        val fullyAdjusted = value * gain + offset
        return (value + COLOR_MATCH_STRENGTH * (fullyAdjusted - value))
            .coerceIn(0.0, 255.0)
            .toInt()
    }

    internal fun calculateRoi(
        targetWidth: Int,
        targetHeight: Int,
        cropWidth: Int,
        cropHeight: Int,
        targetToCrop: AffineMatrix,
    ): CompositeRoi {
        val cropToTarget = targetToCrop.inverse()
        val corners = listOf(
            cropToTarget.map(Point2(0.0, 0.0)),
            cropToTarget.map(Point2(cropWidth.toDouble(), 0.0)),
            cropToTarget.map(Point2(cropWidth.toDouble(), cropHeight.toDouble())),
            cropToTarget.map(Point2(0.0, cropHeight.toDouble())),
        )
        require(corners.all { it.x.isFinite() && it.y.isFinite() }) {
            "Affine ROI must be finite"
        }
        val left = floor(corners.minOf(Point2::x)).coerceIn(0.0, targetWidth.toDouble()).toInt()
        val top = floor(corners.minOf(Point2::y)).coerceIn(0.0, targetHeight.toDouble()).toInt()
        val right = ceil(corners.maxOf(Point2::x)).coerceIn(0.0, targetWidth.toDouble()).toInt()
        val bottom = ceil(corners.maxOf(Point2::y)).coerceIn(0.0, targetHeight.toDouble()).toInt()
        return CompositeRoi(left, top, right, bottom)
    }

    internal fun gaussianBlurReflect101(
        input: FloatArray,
        width: Int,
        height: Int,
        sigma: Double,
    ): FloatArray {
        require(input.size == checkedPixelCount(width, height)) {
            "Input size must match mask dimensions"
        }
        require(input.all(Float::isFinite)) { "Mask values must be finite" }
        require(sigma > 0.0 && sigma.isFinite()) { "Gaussian sigma must be positive and finite" }
        val radius = ceil(3.0 * sigma).toInt()
        val kernel = gaussianKernel(radius, sigma)
        val horizontal = FloatArray(input.size)
        val output = FloatArray(input.size)

        for (y in 0 until height) {
            for (x in 0 until width) {
                var sum = 0.0
                for (offset in -radius..radius) {
                    val reflectedX = reflect101(x + offset, width)
                    sum += input[y * width + reflectedX] * kernel[offset + radius]
                }
                horizontal[y * width + x] = sum.toFloat()
            }
        }
        for (y in 0 until height) {
            for (x in 0 until width) {
                var sum = 0.0
                for (offset in -radius..radius) {
                    val reflectedY = reflect101(y + offset, height)
                    sum += horizontal[reflectedY * width + x] * kernel[offset + radius]
                }
                output[y * width + x] = sum.coerceIn(0.0, 1.0).toFloat()
            }
        }
        return output
    }

    private fun gaussianKernel(radius: Int, sigma: Double): DoubleArray {
        val kernel = DoubleArray(radius * 2 + 1)
        var sum = 0.0
        for (offset in -radius..radius) {
            val value = exp(-(offset.toDouble() * offset) / (2.0 * sigma * sigma))
            kernel[offset + radius] = value
            sum += value
        }
        for (index in kernel.indices) kernel[index] /= sum
        return kernel
    }

    private fun reflect101(index: Int, size: Int): Int {
        if (size == 1) return 0
        val period = 2 * size - 2
        val wrapped = ((index % period) + period) % period
        return if (wrapped < size) wrapped else period - wrapped
    }

    private fun sampleMaskConstantZero(
        mask: FloatArray,
        width: Int,
        height: Int,
        x: Double,
        y: Double,
    ): Double {
        val x0 = floor(x).toInt()
        val y0 = floor(y).toInt()
        val x1 = x0 + 1
        val y1 = y0 + 1
        val fractionX = x - x0
        val fractionY = y - y0
        val top = maskAt(mask, width, height, x0, y0) * (1.0 - fractionX) +
            maskAt(mask, width, height, x1, y0) * fractionX
        val bottom = maskAt(mask, width, height, x0, y1) * (1.0 - fractionX) +
            maskAt(mask, width, height, x1, y1) * fractionX
        return top * (1.0 - fractionY) + bottom * fractionY
    }

    private fun maskAt(mask: FloatArray, width: Int, height: Int, x: Int, y: Int): Double =
        if (x in 0 until width && y in 0 until height) {
            mask[y * width + x].toDouble()
        } else {
            0.0
        }

    private fun sampleArgbEdgeReplicate(
        pixels: IntArray,
        width: Int,
        height: Int,
        x: Double,
        y: Double,
    ): RgbDouble {
        val boundedX = x.coerceIn(0.0, (width - 1).toDouble())
        val boundedY = y.coerceIn(0.0, (height - 1).toDouble())
        val x0 = floor(boundedX).toInt()
        val y0 = floor(boundedY).toInt()
        val x1 = min(x0 + 1, width - 1)
        val y1 = min(y0 + 1, height - 1)
        val fractionX = boundedX - x0
        val fractionY = boundedY - y0
        val topLeft = pixels[y0 * width + x0]
        val topRight = pixels[y0 * width + x1]
        val bottomLeft = pixels[y1 * width + x0]
        val bottomRight = pixels[y1 * width + x1]
        return RgbDouble(
            red = bilinearChannel(topLeft, topRight, bottomLeft, bottomRight, RED_SHIFT, fractionX, fractionY),
            green = bilinearChannel(topLeft, topRight, bottomLeft, bottomRight, GREEN_SHIFT, fractionX, fractionY),
            blue = bilinearChannel(topLeft, topRight, bottomLeft, bottomRight, BLUE_SHIFT, fractionX, fractionY),
        )
    }

    private fun bilinearChannel(
        topLeft: Int,
        topRight: Int,
        bottomLeft: Int,
        bottomRight: Int,
        shift: Int,
        fractionX: Double,
        fractionY: Double,
    ): Double {
        val top = channel(topLeft, shift) * (1.0 - fractionX) + channel(topRight, shift) * fractionX
        val bottom = channel(bottomLeft, shift) * (1.0 - fractionX) + channel(bottomRight, shift) * fractionX
        return top * (1.0 - fractionY) + bottom * fractionY
    }

    private fun requireImage(pixels: IntArray, width: Int, height: Int, name: String) {
        val expectedSize = checkedPixelCount(width, height)
        require(pixels.size == expectedSize) {
            "$name pixel count ${pixels.size} does not match ${width}x$height"
        }
    }

    private fun checkedPixelCount(width: Int, height: Int): Int {
        require(width > 0 && height > 0) { "Image dimensions must be positive" }
        require(width <= Int.MAX_VALUE / height) { "Image dimensions are too large" }
        return width * height
    }

    private fun requireFinite(matrix: AffineMatrix) {
        require(
            matrix.a.isFinite() && matrix.b.isFinite() && matrix.c.isFinite() &&
                matrix.d.isFinite() && matrix.e.isFinite() && matrix.f.isFinite(),
        ) { "Affine matrix values must be finite" }
    }

    private fun channel(pixel: Int, shift: Int): Int = (pixel ushr shift) and CHANNEL_MASK

    private fun squared(value: Double): Double = value * value

    private data class RgbStats(
        val redMean: Double,
        val greenMean: Double,
        val blueMean: Double,
        val redStd: Double,
        val greenStd: Double,
        val blueStd: Double,
    )

    private data class RgbDouble(
        val red: Double,
        val green: Double,
        val blue: Double,
    )

    private const val BOX_MASK_BLUR = 0.3
    private const val BOX_MASK_SIGMA_FACTOR = 0.25
    private const val MASK_STAT_THRESHOLD = 0.5
    private const val COLOR_MATCH_STRENGTH = 0.65
    private const val MIN_GAIN = 0.85
    private const val MAX_GAIN = 1.15
    private const val MIN_OFFSET = -24.0
    private const val MAX_OFFSET = 24.0
    private const val MIN_STANDARD_DEVIATION = 1e-8
    private const val RED_SHIFT = 16
    private const val GREEN_SHIFT = 8
    private const val BLUE_SHIFT = 0
    private const val CHANNEL_MASK = 0xff
    private const val ALPHA_MASK = -0x1000000
}
