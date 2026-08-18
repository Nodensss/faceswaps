package com.faceswaplocal.app.inference

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Test

class FaceCompositorTest {
    @Test
    fun `box mask uses FaceFusion border blur and is symmetric`() {
        val width = 16
        val height = 12

        val mask = FaceCompositor.createBoxMask(width, height)

        assertEquals(width * height, mask.size)
        assertEquals(0.045552f, mask[index(0, 0, width)], 1e-5f)
        assertEquals(0.798430f, mask[index(1, 1, width)], 1e-5f)
        assertEquals(1f, mask[index(8, 6, width)], 1e-5f)
        for (y in 0 until height) {
            for (x in 0 until width) {
                val value = mask[index(x, y, width)]
                assertTrue(value in 0f..1f)
                assertEquals(value, mask[index(width - 1 - x, y, width)], 1e-6f)
                assertEquals(value, mask[index(x, height - 1 - y, width)], 1e-6f)
            }
        }
    }

    @Test
    fun `small box mask keeps a hard one-pixel border when blur amount is zero`() {
        val mask = FaceCompositor.createBoxMask(width = 4, height = 4)

        assertArrayEquals(
            floatArrayOf(
                0f, 0f, 0f, 0f,
                0f, 1f, 1f, 0f,
                0f, 1f, 1f, 0f,
                0f, 0f, 0f, 0f,
            ),
            mask,
            0f,
        )
    }

    @Test
    fun `color match applies bounded gain offset and sixty-five percent strength`() {
        val swapped = intArrayOf(
            argb(255, 20, 100, 200),
            argb(255, 60, 140, 220),
        )
        val target = intArrayOf(
            argb(255, 40, 80, 200),
            argb(255, 80, 100, 240),
        )

        val result = FaceCompositor.matchCropColors(
            targetCropPixels = target,
            swappedCropPixels = swapped,
            mask = floatArrayOf(1f, 1f),
            width = 2,
            height = 1,
        )

        assertEquals(1.0, result.adjustment.redGain, 1e-12)
        assertEquals(0.85, result.adjustment.greenGain, 1e-12)
        assertEquals(1.15, result.adjustment.blueGain, 1e-12)
        assertEquals(20.0, result.adjustment.redOffset, 1e-12)
        assertEquals(-12.0, result.adjustment.greenOffset, 1e-12)
        assertEquals(-21.5, result.adjustment.blueOffset, 1e-12)
        assertEquals(0.65, result.adjustment.strength, 0.0)
        assertArrayEquals(
            intArrayOf(
                argb(255, 33, 82, 205),
                argb(255, 73, 118, 227),
            ),
            result.pixels,
        )
    }

    @Test
    fun `color statistics ignore samples below threshold and weight accepted samples`() {
        val first = FaceCompositor.matchCropColors(
            targetCropPixels = intArrayOf(
                argb(255, 20, 60, 100),
                argb(255, 40, 80, 120),
                argb(255, 0, 0, 0),
            ),
            swappedCropPixels = intArrayOf(
                argb(255, 10, 30, 50),
                argb(255, 30, 50, 70),
                argb(255, 255, 255, 255),
            ),
            mask = floatArrayOf(1f, 0.5f, 0.49f),
            width = 3,
            height = 1,
        )
        val withoutIgnoredSample = FaceCompositor.matchCropColors(
            targetCropPixels = intArrayOf(
                argb(255, 20, 60, 100),
                argb(255, 40, 80, 120),
            ),
            swappedCropPixels = intArrayOf(
                argb(255, 10, 30, 50),
                argb(255, 30, 50, 70),
            ),
            mask = floatArrayOf(1f, 0.5f),
            width = 2,
            height = 1,
        )

        assertEquals(withoutIgnoredSample.adjustment, first.adjustment)
        assertEquals(1.0, first.adjustment.redGain, 1e-12)
        assertEquals(10.0, first.adjustment.redOffset, 1e-12)
    }

    @Test
    fun `color match retains alpha and handles zero source variance`() {
        val result = FaceCompositor.matchCropColors(
            targetCropPixels = intArrayOf(argb(255, 100, 110, 120)),
            swappedCropPixels = intArrayOf(argb(73, 80, 90, 100)),
            mask = floatArrayOf(1f),
            width = 1,
            height = 1,
        )

        assertEquals(1.0, result.adjustment.redGain, 0.0)
        assertEquals(20.0, result.adjustment.redOffset, 0.0)
        assertEquals(argb(73, 93, 103, 113), result.pixels.single())
    }

    @Test
    fun `identity affine replaces only the unfeathered crop interior`() {
        val width = 4
        val target = IntArray(width * width) { index -> argb(41 + index, 7, 8, 9) }
        val targetCrop = target.copyOf().also {
            it[index(1, 1, width)] = argb(255, 10, 20, 30)
            it[index(2, 1, width)] = argb(255, 20, 30, 40)
            it[index(1, 2, width)] = argb(255, 30, 40, 50)
            it[index(2, 2, width)] = argb(255, 40, 50, 60)
        }
        val swapped = targetCrop.copyOf().also {
            it[index(1, 1, width)] = argb(255, 40, 50, 60)
            it[index(2, 1, width)] = argb(255, 30, 40, 50)
            it[index(1, 2, width)] = argb(255, 20, 30, 40)
            it[index(2, 2, width)] = argb(255, 10, 20, 30)
        }
        val targetBefore = target.copyOf()
        val targetCropBefore = targetCrop.copyOf()
        val swappedBefore = swapped.copyOf()

        val result = FaceCompositor.composite(
            targetPixels = target,
            targetWidth = width,
            targetHeight = width,
            targetCropPixels = targetCrop,
            swappedCropPixels = swapped,
            cropWidth = width,
            cropHeight = width,
            targetToCrop = IDENTITY,
        )

        assertEquals(CompositeRoi(0, 0, 4, 4), result.roi)
        assertEquals(argb(46, 40, 50, 60), result.pixels[index(1, 1, width)])
        assertEquals(argb(47, 30, 40, 50), result.pixels[index(2, 1, width)])
        assertEquals(argb(50, 20, 30, 40), result.pixels[index(1, 2, width)])
        assertEquals(argb(51, 10, 20, 30), result.pixels[index(2, 2, width)])
        for (y in 0 until width) {
            assertEquals(target[index(0, y, width)], result.pixels[index(0, y, width)])
            assertEquals(target[index(3, y, width)], result.pixels[index(3, y, width)])
        }
        assertArrayEquals(targetBefore, target)
        assertArrayEquals(targetCropBefore, targetCrop)
        assertArrayEquals(swappedBefore, swapped)
    }

    @Test
    fun `blend constraint limits paste alpha and is returned as the effective crop mask`() {
        val width = 4
        val target = IntArray(width * width) { argb(255, 5, 10, 15) }
        val targetCrop = IntArray(width * width) { index ->
            argb(255, 40 + index, 60 + index, 80 + index)
        }
        val swappedCrop = IntArray(width * width) { index ->
            argb(255, 180 - index, 160 - index, 140 - index)
        }
        val constraint = FloatArray(width * width) { 1f }.also {
            it[index(1, 1, width)] = 0f
            it[index(2, 1, width)] = 0.25f
        }

        val unconstrained = FaceCompositor.composite(
            targetPixels = target,
            targetWidth = width,
            targetHeight = width,
            targetCropPixels = targetCrop,
            swappedCropPixels = swappedCrop,
            cropWidth = width,
            cropHeight = width,
            targetToCrop = IDENTITY,
        )
        val constrained = FaceCompositor.composite(
            targetPixels = target,
            targetWidth = width,
            targetHeight = width,
            targetCropPixels = targetCrop,
            swappedCropPixels = swappedCrop,
            cropWidth = width,
            cropHeight = width,
            targetToCrop = IDENTITY,
            blendConstraintMask = constraint,
            collectWarpedMask = true,
        )

        assertEquals(0f, constrained.cropMask[index(1, 1, width)], 0f)
        assertEquals(0.25f, constrained.cropMask[index(2, 1, width)], 0f)
        assertEquals(0f, requireNotNull(constrained.warpedMask)[index(1, 1, width)], 0f)
        assertEquals(0.25f, requireNotNull(constrained.warpedMask)[index(2, 1, width)], 0f)
        assertEquals(target[index(1, 1, width)], constrained.pixels[index(1, 1, width)])
        assertNotEquals(target[index(1, 1, width)], unconstrained.pixels[index(1, 1, width)])
        assertNotEquals(
            unconstrained.pixels[index(2, 1, width)],
            constrained.pixels[index(2, 1, width)],
        )
        assertEquals(
            unconstrained.pixels[index(1, 2, width)],
            constrained.pixels[index(1, 2, width)],
        )
    }

    @Test
    fun `blend constraint leaves box-mask color adjustment unchanged`() {
        val width = 4
        val target = IntArray(width * width) { argb(255, 10, 20, 30) }
        val targetCrop = IntArray(width * width) { index ->
            argb(255, 30 + index * 2, 70 + index, 110 - index)
        }
        val swappedCrop = IntArray(width * width) { index ->
            argb(255, 150 - index, 40 + index * 3, 60 + index)
        }

        val withoutConstraint = FaceCompositor.composite(
            targetPixels = target,
            targetWidth = width,
            targetHeight = width,
            targetCropPixels = targetCrop,
            swappedCropPixels = swappedCrop,
            cropWidth = width,
            cropHeight = width,
            targetToCrop = IDENTITY,
        )
        val withConstraint = FaceCompositor.composite(
            targetPixels = target,
            targetWidth = width,
            targetHeight = width,
            targetCropPixels = targetCrop,
            swappedCropPixels = swappedCrop,
            cropWidth = width,
            cropHeight = width,
            targetToCrop = IDENTITY,
            blendConstraintMask = FloatArray(width * width) { index ->
                if (index % 2 == 0) 0f else 0.4f
            },
        )

        assertEquals(withoutConstraint.colorAdjustment, withConstraint.colorAdjustment)
        assertArrayEquals(withoutConstraint.colorMatchedCrop, withConstraint.colorMatchedCrop)
    }

    @Test
    fun `translated inverse affine limits changes and mask to its target ROI`() {
        val targetWidth = 10
        val targetHeight = 8
        val cropWidth = 4
        val cropHeight = 4
        val target = IntArray(targetWidth * targetHeight) { argb(255, 4, 5, 6) }
        val crop = intArrayOf(
            argb(255, 1, 2, 3), argb(255, 2, 3, 4), argb(255, 3, 4, 5), argb(255, 4, 5, 6),
            argb(255, 5, 6, 7), argb(255, 10, 20, 30), argb(255, 20, 30, 40), argb(255, 8, 9, 10),
            argb(255, 9, 10, 11), argb(255, 30, 40, 50), argb(255, 40, 50, 60), argb(255, 12, 13, 14),
            argb(255, 13, 14, 15), argb(255, 14, 15, 16), argb(255, 15, 16, 17), argb(255, 16, 17, 18),
        )
        val swapped = crop.copyOf().also {
            it[index(1, 1, cropWidth)] = argb(255, 40, 50, 60)
            it[index(2, 1, cropWidth)] = argb(255, 30, 40, 50)
            it[index(1, 2, cropWidth)] = argb(255, 20, 30, 40)
            it[index(2, 2, cropWidth)] = argb(255, 10, 20, 30)
        }

        val result = FaceCompositor.composite(
            targetPixels = target,
            targetWidth = targetWidth,
            targetHeight = targetHeight,
            targetCropPixels = crop,
            swappedCropPixels = swapped,
            cropWidth = cropWidth,
            cropHeight = cropHeight,
            targetToCrop = AffineMatrix(1.0, 0.0, -3.0, 0.0, 1.0, -2.0),
            collectWarpedMask = true,
        )

        assertEquals(CompositeRoi(3, 2, 7, 6), result.roi)
        assertEquals(1f, requireNotNull(result.warpedMask)[index(4, 3, targetWidth)], 0f)
        assertEquals(0f, requireNotNull(result.warpedMask)[index(3, 3, targetWidth)], 0f)
        assertEquals(argb(255, 40, 50, 60), result.pixels[index(4, 3, targetWidth)])
        for (y in 0 until targetHeight) {
            for (x in 0 until targetWidth) {
                if (x !in 3 until 7 || y !in 2 until 6) {
                    assertEquals(target[index(x, y, targetWidth)], result.pixels[index(x, y, targetWidth)])
                    assertEquals(0f, requireNotNull(result.warpedMask)[index(x, y, targetWidth)], 0f)
                }
            }
        }
    }

    @Test
    fun `fractional inverse sampling uses zero mask border edge-replicated crop and truncation`() {
        val width = 16
        val height = 12
        val black = argb(255, 0, 0, 0)
        val gray = argb(255, 100, 100, 100)
        val result = FaceCompositor.composite(
            targetPixels = IntArray(width * height) { black },
            targetWidth = width,
            targetHeight = height,
            targetCropPixels = IntArray(width * height) { gray },
            swappedCropPixels = IntArray(width * height) { gray },
            cropWidth = width,
            cropHeight = height,
            targetToCrop = AffineMatrix(1.0, 0.0, -0.5, 0.0, 1.0, 0.0),
            collectWarpedMask = true,
        )

        val alpha = requireNotNull(result.warpedMask)[index(0, 0, width)]
        assertEquals(result.cropMask[index(0, 0, width)] * 0.5f, alpha, 1e-7f)
        val expected = (100.0 * alpha).toInt()
        assertEquals(argb(255, expected, expected, expected), result.pixels[index(0, 0, width)])
        assertEquals(2, expected)
    }

    @Test
    fun `crop fully outside target produces an unchanged image and zero warped mask`() {
        val target = IntArray(36) { index -> argb(100 + index, index, index + 1, index + 2) }
        val crop = IntArray(16) { index -> argb(255, index * 2, index * 3, index * 4) }

        val result = FaceCompositor.composite(
            targetPixels = target,
            targetWidth = 6,
            targetHeight = 6,
            targetCropPixels = crop,
            swappedCropPixels = crop,
            cropWidth = 4,
            cropHeight = 4,
            targetToCrop = AffineMatrix(1.0, 0.0, 100.0, 0.0, 1.0, 100.0),
            collectWarpedMask = true,
        )

        assertEquals(CompositeRoi(0, 0, 0, 0), result.roi)
        assertArrayEquals(target, result.pixels)
        assertTrue(requireNotNull(result.warpedMask).all { it == 0f })
    }

    @Test
    fun `paste back supports arbitrary crop mask and preserves base alpha and inputs`() {
        val base = IntArray(4) { argb(77 + it, 10, 20, 30) }
        val crop = IntArray(4) { argb(255, 110, 120, 130) }
        val mask = floatArrayOf(0f, 0.25f, 0.5f, 1f)
        val baseBefore = base.copyOf()
        val cropBefore = crop.copyOf()
        val maskBefore = mask.copyOf()

        val result = FaceCompositor.pasteBack(
            basePixels = base,
            baseWidth = 2,
            baseHeight = 2,
            cropPixels = crop,
            cropMask = mask,
            cropWidth = 2,
            cropHeight = 2,
            baseToCrop = IDENTITY,
            collectWarpedMask = true,
        )

        assertEquals(CompositeRoi(0, 0, 2, 2), result.roi)
        assertArrayEquals(mask, requireNotNull(result.warpedMask), 0f)
        assertArrayEquals(
            intArrayOf(
                argb(77, 10, 20, 30),
                argb(78, 35, 45, 55),
                argb(79, 60, 70, 80),
                argb(80, 110, 120, 130),
            ),
            result.pixels,
        )
        assertArrayEquals(baseBefore, base)
        assertArrayEquals(cropBefore, crop)
        assertArrayEquals(maskBefore, mask, 0f)
    }

    @Test
    fun `paste back leaves protected full-image roi bit identical`() {
        val base = IntArray(16) { index -> argb(80 + index, 10, 20, 30) }
        val crop = IntArray(16) { argb(255, 240, 230, 220) }
        val protected = CompositeRoi(left = 1, top = 1, right = 3, bottom = 3)

        val result = FaceCompositor.pasteBack(
            basePixels = base,
            baseWidth = 4,
            baseHeight = 4,
            cropPixels = crop,
            cropMask = FloatArray(16) { 1f },
            cropWidth = 4,
            cropHeight = 4,
            baseToCrop = IDENTITY,
            protectedBaseRois = listOf(protected),
            collectWarpedMask = true,
        )

        for (y in protected.top until protected.bottom) {
            for (x in protected.left until protected.right) {
                val pixel = index(x, y, 4)
                assertEquals(base[pixel], result.pixels[pixel])
                assertEquals(0f, requireNotNull(result.warpedMask)[pixel])
            }
        }
        assertNotEquals(base[0], result.pixels[0])
    }

    @Test
    fun `gaussian blur preserves a constant mask and is symmetric for an impulse`() {
        val constant = FaceCompositor.gaussianBlurReflect101(
            input = FloatArray(25) { 1f },
            width = 5,
            height = 5,
            sigma = 1.0,
        )
        assertTrue(constant.all { kotlin.math.abs(it - 1f) <= 1e-6f })

        val impulse = FloatArray(25).also { it[index(2, 2, 5)] = 1f }
        val blurred = FaceCompositor.gaussianBlurReflect101(impulse, 5, 5, 1.0)

        for (y in 0 until 5) {
            for (x in 0 until 5) {
                val value = blurred[index(x, y, 5)]
                assertTrue(value in 0f..1f)
                assertEquals(value, blurred[index(4 - x, y, 5)], 1e-7f)
                assertEquals(value, blurred[index(x, 4 - y, 5)], 1e-7f)
            }
        }
        assertTrue(blurred[index(2, 2, 5)] > blurred[index(1, 2, 5)])
        assertTrue(blurred[index(1, 2, 5)] > blurred[index(0, 2, 5)])
    }

    @Test
    fun `close face overlap composites second paste over accumulated first photo pixels`() {
        // The target stands in for a close-people photo: two face ROIs overlap at x=3..4.
        val originalPhoto = IntArray(8 * 6) { argb(255, 30, 40, 50) }
        val neutralCrop = IntArray(4 * 4) { argb(255, 30, 40, 50) }
        val first = FaceCompositor.composite(
            originalPhoto, 8, 6, neutralCrop, IntArray(16) { argb(255, 210, 20, 20) }, 4, 4,
            AffineMatrix(1.0, 0.0, 0.0, 0.0, 1.0, -1.0),
        )
        val second = FaceCompositor.composite(
            first.pixels, 8, 6, neutralCrop, IntArray(16) { argb(255, 20, 210, 20) }, 4, 4,
            AffineMatrix(1.0, 0.0, -1.0, 0.0, 1.0, -1.0),
        )
        // First-only interior remains; overlap uses the second face, proving no reset to original.
        val firstOnly = second.pixels[index(1, 2, 8)]
        val overlap = second.pixels[index(2, 2, 8)]
        assertTrue(((firstOnly ushr 16) and 0xff) > ((firstOnly ushr 8) and 0xff))
        assertTrue(((overlap ushr 8) and 0xff) > ((overlap ushr 16) and 0xff))
    }

    @Test
    fun `invalid inputs fail before compositing`() {
        assertThrows(IllegalArgumentException::class.java) {
            FaceCompositor.createBoxMask(0, 8)
        }
        assertThrows(IllegalArgumentException::class.java) {
            FaceCompositor.matchCropColors(
                targetCropPixels = intArrayOf(argb(255, 1, 2, 3)),
                swappedCropPixels = intArrayOf(argb(255, 1, 2, 3)),
                mask = floatArrayOf(0.49f),
                width = 1,
                height = 1,
            )
        }
        assertThrows(IllegalArgumentException::class.java) {
            FaceCompositor.composite(
                targetPixels = IntArray(3),
                targetWidth = 2,
                targetHeight = 2,
                targetCropPixels = IntArray(16),
                swappedCropPixels = IntArray(16),
                cropWidth = 4,
                cropHeight = 4,
                targetToCrop = IDENTITY,
            )
        }
        assertThrows(IllegalArgumentException::class.java) {
            FaceCompositor.composite(
                targetPixels = IntArray(16),
                targetWidth = 4,
                targetHeight = 4,
                targetCropPixels = IntArray(16),
                swappedCropPixels = IntArray(16),
                cropWidth = 4,
                cropHeight = 4,
                targetToCrop = AffineMatrix(1.0, 0.0, 0.0, 2.0, 0.0, 0.0),
            )
        }
    }

    /**
     * A parser mask on a real photograph reaches 0.999... rather than a clean 1.0, so a
     * fully protected face still received a residual alpha. The blend then truncated the
     * base value itself - `200 * (1 - 0.0002)` floors to 199 - and shifted the pixel by
     * one level. On the live 4 MP acceptance run this altered 1054 pixels of a face the
     * user had marked "do not change". Sub-threshold alphas must skip the write entirely.
     */
    @Test
    fun `residual protection alpha leaves a protected face bit-identical`() {
        val width = 4
        val height = 4
        val base = IntArray(width * height) { argb(255, 200, 150, 100) }
        val crop = IntArray(width * height) { argb(255, 10, 20, 30) }
        val baseBefore = base.copyOf()
        // 0.9998 protection against a full paste leaves alpha = 2e-4, far below 1/512.
        val protection = ProtectedFaceRegion(
            bounds = CompositeRoi(0, 0, width, height),
            mask = FloatArray(width * height) { 0.9998f },
            cropWidth = width,
            cropHeight = height,
            baseToCrop = IDENTITY,
        )

        val result = FaceCompositor.pasteBack(
            basePixels = base,
            baseWidth = width,
            baseHeight = height,
            cropPixels = crop,
            cropMask = FloatArray(width * height) { 1f },
            cropWidth = width,
            cropHeight = height,
            baseToCrop = IDENTITY,
            protectedFaceRegions = listOf(protection),
        )

        assertArrayEquals("input must not be mutated", baseBefore, base)
        assertArrayEquals(
            "a residual alpha below the write threshold must not shift any channel",
            baseBefore,
            result.pixels,
        )
    }

    /** The threshold must not silently swallow a blend that is genuinely visible. */
    @Test
    fun `an alpha just above the threshold still blends`() {
        val width = 2
        val height = 1
        val base = IntArray(width * height) { argb(255, 200, 200, 200) }
        val crop = IntArray(width * height) { argb(255, 0, 0, 0) }
        val justBelow = (FaceCompositor.MINIMUM_BLEND_ALPHA / 2).toFloat()
        val justAbove = (FaceCompositor.MINIMUM_BLEND_ALPHA * 4).toFloat()

        val result = FaceCompositor.pasteBack(
            basePixels = base,
            baseWidth = width,
            baseHeight = height,
            cropPixels = crop,
            cropMask = floatArrayOf(justBelow, justAbove),
            cropWidth = width,
            cropHeight = height,
            baseToCrop = IDENTITY,
        )

        assertEquals("below the threshold stays untouched", base[0], result.pixels[0])
        assertTrue("above the threshold must still change", result.pixels[1] != base[1])
    }

    /**
     * The full-frame alpha mask is 4 bytes per pixel of the target — 64 MB on a 16 MP
     * photo, a quarter of the per-pixel budget — and no production path ever read it.
     * It must stay unallocated unless a caller explicitly asks.
     */
    @Test
    fun `the warped mask is not allocated unless requested`() {
        val width = 4
        val target = IntArray(width * width) { argb(255, 5, 10, 15) }
        val crop = IntArray(width * width) { argb(255, 100, 110, 120) }

        val default = FaceCompositor.composite(
            targetPixels = target,
            targetWidth = width,
            targetHeight = width,
            targetCropPixels = crop,
            swappedCropPixels = crop,
            cropWidth = width,
            cropHeight = width,
            targetToCrop = IDENTITY,
        )
        val requested = FaceCompositor.composite(
            targetPixels = target,
            targetWidth = width,
            targetHeight = width,
            targetCropPixels = crop,
            swappedCropPixels = crop,
            cropWidth = width,
            cropHeight = width,
            targetToCrop = IDENTITY,
            collectWarpedMask = true,
        )

        assertNull("production must not pay for a mask it never reads", default.warpedMask)
        assertEquals(target.size, requireNotNull(requested.warpedMask).size)
        // Opting in must not change a single pixel of the composite.
        assertArrayEquals(default.pixels, requested.pixels)
    }

    private fun index(x: Int, y: Int, width: Int): Int = y * width + x

    private fun argb(alpha: Int, red: Int, green: Int, blue: Int): Int =
        (alpha shl 24) or (red shl 16) or (green shl 8) or blue

    private companion object {
        val IDENTITY = AffineMatrix(1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    }
}
