package com.faceswaplocal.app.inference

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import java.io.File
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.security.MessageDigest
import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.exp
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min
import kotlin.math.pow
import kotlin.math.roundToInt
import kotlin.math.sin
import kotlin.math.sqrt
import kotlinx.coroutines.runBlocking
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

/**
 * Diagnostic checkpoint for the remaining face-to-neck tone mismatch on pair_02/03.
 *
 * This test deliberately leaves the production colour algorithm unchanged. It freezes
 * the exact box-mask statistics contract, records a parser-masked swap snapshot, then
 * feeds those same pixels directly to the production GFPGAN enhancer at strength 0.8.
 */
@RunWith(AndroidJUnit4::class)
class StageEColorDiagnosticsInstrumentedTest {
    private val context = ApplicationProvider.getApplicationContext<Context>()
    private val assets = InstrumentationRegistry.getInstrumentation().context.assets

    @Test
    fun pair02AndPair03RecordToneBeforeAndAfterGfpgan() = runBlocking {
        assertEquals(
            "CIEDE2000 implementation must match the Sharma reference pair",
            2.0425,
            deltaE2000(
                Lab(50.0, 2.6772, -79.7751),
                Lab(50.0, 0.0, -82.7485),
            ),
            0.0001,
        )
        val store = ModelStore(context)
        requireModels(store)
        val rawPipeline = OnnxRawFaceSwapPipeline(store)
        val parserPipeline = OnnxFaceParserPipeline(store)
        val enhancerPipeline = OnnxFaceEnhancerPipeline(store)
        val outputDirectory = outputDirectory()
        val boxMask = FaceCompositor.createBoxMask(SWAP_CROP_SIZE, SWAP_CROP_SIZE)
        val contract = colorContract(boxMask)
        val pairs = JSONArray()

        try {
            parserPipeline.withSession(RequestedInferenceBackend.CPU_ONLY) { parserSession ->
                FIXTURES.forEach { fixture ->
                    pairs.put(
                        diagnosePair(
                            fixture = fixture,
                            rawPipeline = rawPipeline,
                            enhancerPipeline = enhancerPipeline,
                            parserSession = parserSession,
                            boxMask = boxMask,
                            outputDirectory = outputDirectory,
                        ),
                    )
                }
            }
        } finally {
            boxMask.fill(0f)
        }

        File(outputDirectory, RESULTS_FILE).writeText(
            JSONObject()
                .put("checkpoint", "E1 checkpoint 3: color diagnostics")
                .put("device_api", 35)
                .put("backend", "CPU")
                .put("restoration_strength", RESTORATION_STRENGTH.toDouble())
                .put("measurement_space", "sRGB IEC 61966-2-1 -> CIE Lab D65")
                .put("polygon_rule", "pixel center (x+0.5,y+0.5), half-open ray crossing")
                .put("color_match_contract", contract)
                .put("pairs", pairs)
                .put(
                    "interpretation",
                    "Absolute face-neck delta is not expected to be zero; compare each output to the original target relation",
                )
                .toString(2),
        )
    }

    private suspend fun diagnosePair(
        fixture: FixtureSpec,
        rawPipeline: OnnxRawFaceSwapPipeline,
        enhancerPipeline: OnnxFaceEnhancerPipeline,
        parserSession: FaceParserSession,
        boxMask: FloatArray,
        outputDirectory: File,
    ): JSONObject {
        val sourcePath = "inputs/${fixture.pair}_source.png"
        val targetPath = "inputs/${fixture.pair}_target.png"
        val committedPath =
            "android/api35-x86_64/checkpoint_2/${fixture.pair}_parser_after.png"
        val source = bitmap(sourcePath)
        val target = bitmap(targetPath)
        val committed = bitmap(committedPath)
        var raw: RawFaceSwapResult? = null
        var enhanced: FaceEnhancementResult? = null
        var postSwapBitmap: Bitmap? = null
        var enhancedBitmap: Bitmap? = null
        var targetPixels: IntArray? = null
        var targetCropPixels: IntArray? = null
        var swappedCropPixels: IntArray? = null
        var committedPixels: IntArray? = null
        var parserMask: FloatArray? = null
        var composite: FaceCompositeResult? = null
        try {
            assertEquals(IMAGE_SIZE, target.width)
            assertEquals(IMAGE_SIZE, target.height)
            raw = rawPipeline.process(
                RawFaceSwapRequest(
                    source = source,
                    target = target,
                    swapper = SwapperModel.INSWAPPER_128_FP16,
                    backend = RequestedInferenceBackend.CPU_ONLY,
                ),
            )
            assertEquals(SWAP_CROP_SIZE, raw.swapper.cropSize)
            targetPixels = target.pixels()
            targetCropPixels = raw.alignedTarget.pixels()
            swappedCropPixels = raw.rawOutputBitmap.pixels()

            val parsed = parserSession.createRegionMask(
                cropPixels = requireNotNull(swappedCropPixels),
                cropWidth = SWAP_CROP_SIZE,
                cropHeight = SWAP_CROP_SIZE,
            )
            parserMask = parsed.mask
            composite = FaceCompositor.composite(
                targetPixels = requireNotNull(targetPixels),
                targetWidth = target.width,
                targetHeight = target.height,
                targetCropPixels = requireNotNull(targetCropPixels),
                swappedCropPixels = requireNotNull(swappedCropPixels),
                cropWidth = SWAP_CROP_SIZE,
                cropHeight = SWAP_CROP_SIZE,
                targetToCrop = raw.targetToSwapperCrop,
                blendConstraintMask = requireNotNull(parserMask),
            )

            committedPixels = committed.pixels()
            assertArrayEquals(
                "${fixture.pair} diagnostic swap snapshot must equal checkpoint 2",
                requireNotNull(committedPixels),
                composite.pixels,
            )

            enhanced = enhancerPipeline.enhance(
                basePixels = composite.pixels,
                baseWidth = target.width,
                baseHeight = target.height,
                targetFace = raw.targetFace,
                strength = RESTORATION_STRENGTH,
                backend = RequestedInferenceBackend.CPU_ONLY,
                parserSession = parserSession,
            )

            val faceIndices = polygonIndices(
                target.width,
                target.height,
                fixture.facePolygon,
            )
            val neckIndices = polygonIndices(
                target.width,
                target.height,
                fixture.neckPolygon,
            )
            assertEquals(fixture.expectedFaceSamples, faceIndices.size)
            assertEquals(fixture.expectedNeckSamples, neckIndices.size)
            assertEquals(
                "${fixture.pair} neck must remain unchanged after swap",
                0,
                changedAt(requireNotNull(targetPixels), composite.pixels, neckIndices),
            )
            assertEquals(
                "${fixture.pair} neck must remain unchanged after GFPGAN",
                0,
                changedAt(requireNotNull(targetPixels), enhanced.pixels, neckIndices),
            )
            assertTrue(
                "${fixture.pair} face diagnostic must observe the swap",
                changedAt(requireNotNull(targetPixels), composite.pixels, faceIndices) > 0,
            )
            assertTrue(
                "${fixture.pair} face diagnostic must observe GFPGAN",
                changedAt(composite.pixels, enhanced.pixels, faceIndices) > 0,
            )

            val originalTone = toneStage(
                requireNotNull(targetPixels),
                faceIndices,
                neckIndices,
            )
            val postSwapTone = toneStage(composite.pixels, faceIndices, neckIndices)
            val postGfpganTone = toneStage(enhanced.pixels, faceIndices, neckIndices)

            val currentIndices = boxMask.indices.filter { boxMask[it] >= MASK_STAT_THRESHOLD }
            val overlap = parserOverlap(boxMask, requireNotNull(parserMask), currentIndices)
            val currentStats = JSONObject()
                .put(
                    "aligned_target",
                    weightedRgbStatsJson(
                        requireNotNull(targetCropPixels),
                        boxMask,
                        currentIndices,
                    ),
                )
                .put(
                    "raw_swap",
                    weightedRgbStatsJson(
                        requireNotNull(swappedCropPixels),
                        boxMask,
                        currentIndices,
                    ),
                )
                .put(
                    "color_matched_swap",
                    weightedRgbStatsJson(
                        composite.colorMatchedCrop,
                        boxMask,
                        currentIndices,
                    ),
                )

            postSwapBitmap = bitmapFromPixels(composite.pixels, target.width, target.height)
            enhancedBitmap = bitmapFromPixels(enhanced.pixels, target.width, target.height)
            savePng(
                requireNotNull(postSwapBitmap),
                File(outputDirectory, "${fixture.pair}_post_swap_color.png"),
            )
            savePng(
                requireNotNull(enhancedBitmap),
                File(outputDirectory, "${fixture.pair}_post_gfpgan_0_8.png"),
            )
            saveTriptych(
                original = requireNotNull(targetPixels),
                postSwap = composite.pixels,
                postGfpgan = enhanced.pixels,
                width = target.width,
                height = target.height,
                facePolygon = fixture.facePolygon,
                neckPolygon = fixture.neckPolygon,
                file = File(outputDirectory, "${fixture.pair}_tone_triptych.png"),
            )

            return JSONObject()
                .put("pair", fixture.pair)
                .put("source_file", sourcePath)
                .put("source_sha256", sha256Asset(sourcePath))
                .put("target_file", targetPath)
                .put("target_sha256", sha256Asset(targetPath))
                .put("checkpoint_2_snapshot", committedPath)
                .put("checkpoint_2_snapshot_matched", true)
                .put("runtime_target_landmarks", landmarksJson(raw.targetFace.landmarks))
                .put("target_to_swapper_crop", affineJson(raw.targetToSwapperCrop))
                .put(
                    "statistics_frame_bounds_polygon",
                    polygonJson(statisticsFrameBounds(raw.targetToSwapperCrop)),
                )
                .put("current_box_rgb_statistics", currentStats)
                .put("current_statistics_parser_overlap", overlap)
                .put("color_adjustment", adjustmentJson(composite.colorAdjustment))
                .put(
                    "tone_rois",
                    JSONObject()
                        .put(
                            "face",
                            JSONObject()
                                .put("description", "fixture-annotated forehead skin")
                                .put("polygon", polygonJson(fixture.facePolygon))
                                .put("samples", faceIndices.size),
                        )
                        .put(
                            "neck",
                            JSONObject()
                                .put("description", "fixture-annotated unchanged neck skin")
                                .put("polygon", polygonJson(fixture.neckPolygon))
                                .put("samples", neckIndices.size),
                        ),
                )
                .put(
                    "stages",
                    JSONObject()
                        .put("target_original", originalTone.toJson())
                        .put("post_swap_color", postSwapTone.toJson())
                        .put("post_gfpgan_0_8", postGfpganTone.toJson()),
                )
                .put(
                    "comparisons",
                    JSONObject()
                        .put(
                            "post_swap_vs_original_relation",
                            relationComparison(originalTone, postSwapTone),
                        )
                        .put(
                            "post_gfpgan_vs_original_relation",
                            relationComparison(originalTone, postGfpganTone),
                        )
                        .put(
                            "gfpgan_drift_from_post_swap",
                            stageDrift(postSwapTone, postGfpganTone),
                        ),
                )
                .put(
                    "changed_pixels",
                    JSONObject()
                        .put(
                            "face_post_swap_vs_original",
                            changedAt(requireNotNull(targetPixels), composite.pixels, faceIndices),
                        )
                        .put(
                            "face_post_gfpgan_vs_post_swap",
                            changedAt(composite.pixels, enhanced.pixels, faceIndices),
                        )
                        .put("neck_post_swap_vs_original", 0)
                        .put("neck_post_gfpgan_vs_original", 0),
                )
                .put("parser_backend", parsed.backend.name)
                .put("parser_inference_ms", parsed.inferenceMs)
                .put("enhancer_backend", enhanced.enhancerBackend.name)
                .put("enhancer_parser_backend", enhanced.parserBackend.name)
                .put("enhancer_inference_ms", enhanced.enhancerMs)
                .put("enhancer_parser_ms", enhanced.parserMs)
                .put("enhancer_compositing_ms", enhanced.compositingMs)
        } finally {
            composite?.cropMask?.fill(0f)
            composite?.warpedMask?.fill(0f)
            composite?.colorMatchedCrop?.fill(0)
            composite?.pixels?.fill(0)
            parserMask?.fill(0f)
            enhanced?.pixels?.fill(0)
            targetPixels?.fill(0)
            targetCropPixels?.fill(0)
            swappedCropPixels?.fill(0)
            committedPixels?.fill(0)
            postSwapBitmap?.recycleSafely()
            enhancedBitmap?.recycleSafely()
            raw?.release()
            source.recycleSafely()
            target.recycleSafely()
            committed.recycleSafely()
        }
    }

    private suspend fun requireModels(store: ModelStore) {
        val statuses = store.refreshStatuses()
        REQUIRED_MODELS.forEach { id ->
            assertTrue("${id.stableId} must be imported", statuses[id] is ModelStatus.Ready)
        }
    }

    private fun colorContract(mask: FloatArray): JSONObject {
        val indices = mask.indices.filter { mask[it] >= MASK_STAT_THRESHOLD }
        val xs = indices.map { it % SWAP_CROP_SIZE }
        val ys = indices.map { it / SWAP_CROP_SIZE }
        assertEquals(EXPECTED_STAT_SAMPLES, indices.size)
        assertEquals(STAT_LEFT, xs.min())
        assertEquals(STAT_TOP, ys.min())
        assertEquals(STAT_RIGHT_EXCLUSIVE - 1, xs.max())
        assertEquals(STAT_BOTTOM_EXCLUSIVE - 1, ys.max())
        return JSONObject()
            .put("crop_size", SWAP_CROP_SIZE)
            .put("statistics_mask", "FaceCompositor.createBoxMask")
            .put("parser_mask_used_for_statistics", false)
            .put("box_blur", 0.3)
            .put("blur_amount", 19)
            .put("zero_border", 9)
            .put("gaussian_sigma", 4.75)
            .put("gaussian_kernel_radius", 15)
            .put("sample_threshold", MASK_STAT_THRESHOLD)
            .put("selected_samples", indices.size)
            .put(
                "selected_bounds_xyxy_exclusive",
                JSONArray()
                    .put(STAT_LEFT)
                    .put(STAT_TOP)
                    .put(STAT_RIGHT_EXCLUSIVE)
                    .put(STAT_BOTTOM_EXCLUSIVE),
            )
            .put("selected_weight_sum", indices.sumOf { mask[it].toDouble() })
            .put("mask_minimum", mask.min())
            .put("mask_maximum", mask.max())
            .put("mask_mean", mask.sumOf(Float::toDouble) / mask.size)
            .put("mask_sha256_f32le", sha256FloatLittleEndian(mask))
            .put("color_match_strength", 0.65)
            .put("gain_limits", JSONArray().put(0.85).put(1.15))
            .put("offset_limits", JSONArray().put(-24.0).put(24.0))
    }

    private fun parserOverlap(
        boxMask: FloatArray,
        parserMask: FloatArray,
        currentIndices: List<Int>,
    ): JSONObject {
        val totalWeight = currentIndices.sumOf { boxMask[it].toDouble() }
        val parserWeighted = currentIndices.sumOf { index ->
            boxMask[index].toDouble() * parserMask[index].toDouble()
        }
        return JSONObject()
            .put("selected_samples", currentIndices.size)
            .put(
                "samples_parser_alpha_at_least_0_5",
                currentIndices.count { parserMask[it] >= 0.5f },
            )
            .put(
                "samples_parser_alpha_at_most_0_05",
                currentIndices.count { parserMask[it] <= 0.05f },
            )
            .put("box_weight_sum", totalWeight)
            .put("box_times_parser_weight_sum", parserWeighted)
            .put("parser_weight_fraction", parserWeighted / totalWeight)
    }

    private fun weightedRgbStatsJson(
        pixels: IntArray,
        mask: FloatArray,
        selectedIndices: List<Int>,
    ): JSONObject {
        var weightSum = 0.0
        val sums = DoubleArray(3)
        selectedIndices.forEach { index ->
            val weight = mask[index].toDouble()
            weightSum += weight
            sums[0] += channel(pixels[index], 16) * weight
            sums[1] += channel(pixels[index], 8) * weight
            sums[2] += channel(pixels[index], 0) * weight
        }
        val means = DoubleArray(3) { sums[it] / weightSum }
        val variances = DoubleArray(3)
        selectedIndices.forEach { index ->
            val weight = mask[index].toDouble()
            variances[0] += (channel(pixels[index], 16) - means[0]).pow(2) * weight
            variances[1] += (channel(pixels[index], 8) - means[1]).pow(2) * weight
            variances[2] += (channel(pixels[index], 0) - means[2]).pow(2) * weight
        }
        return JSONObject()
            .put("samples", selectedIndices.size)
            .put("weight_sum", weightSum)
            .put("mean_rgb", doubleArrayJson(means))
            .put(
                "std_rgb",
                doubleArrayJson(DoubleArray(3) { sqrt(max(variances[it] / weightSum, 0.0)) }),
            )
    }

    private fun adjustmentJson(adjustment: FaceColorAdjustment): JSONObject = JSONObject()
        .put("red_gain", adjustment.redGain)
        .put("green_gain", adjustment.greenGain)
        .put("blue_gain", adjustment.blueGain)
        .put("red_offset", adjustment.redOffset)
        .put("green_offset", adjustment.greenOffset)
        .put("blue_offset", adjustment.blueOffset)
        .put("strength", adjustment.strength)
        .put(
            "gain_clipped",
            JSONObject()
                .put("red", isAtLimit(adjustment.redGain, 0.85, 1.15))
                .put("green", isAtLimit(adjustment.greenGain, 0.85, 1.15))
                .put("blue", isAtLimit(adjustment.blueGain, 0.85, 1.15)),
        )
        .put(
            "offset_clipped",
            JSONObject()
                .put("red", isAtLimit(adjustment.redOffset, -24.0, 24.0))
                .put("green", isAtLimit(adjustment.greenOffset, -24.0, 24.0))
                .put("blue", isAtLimit(adjustment.blueOffset, -24.0, 24.0)),
        )

    private fun isAtLimit(value: Double, lower: Double, upper: Double): Boolean =
        abs(value - lower) <= LIMIT_EPSILON || abs(value - upper) <= LIMIT_EPSILON

    private fun toneStage(
        pixels: IntArray,
        faceIndices: IntArray,
        neckIndices: IntArray,
    ): ToneStage {
        val face = summarize(pixels, faceIndices)
        val neck = summarize(pixels, neckIndices)
        val relation = face.median - neck.median
        return ToneStage(
            face = face,
            neck = neck,
            faceMinusNeck = relation,
            deltaE00 = deltaE2000(face.median, neck.median),
        ).also { stage -> requireFinite(stage) }
    }

    private fun summarize(pixels: IntArray, indices: IntArray): ColorSummary {
        val labs = ArrayList<Lab>(indices.size)
        val reds = DoubleArray(indices.size)
        val greens = DoubleArray(indices.size)
        val blues = DoubleArray(indices.size)
        indices.forEachIndexed { outputIndex, pixelIndex ->
            val pixel = pixels[pixelIndex]
            val red = channel(pixel, 16).toDouble()
            val green = channel(pixel, 8).toDouble()
            val blue = channel(pixel, 0).toDouble()
            reds[outputIndex] = red
            greens[outputIndex] = green
            blues[outputIndex] = blue
            labs += rgbToLab(red, green, blue)
        }
        val lValues = labs.map(Lab::l).toDoubleArray()
        val aValues = labs.map(Lab::a).toDoubleArray()
        val bValues = labs.map(Lab::b).toDoubleArray()
        val median = Lab(median(lValues), median(aValues), median(bValues))
        return ColorSummary(
            samples = indices.size,
            median = median,
            mad = Lab(
                median(DoubleArray(lValues.size) { abs(lValues[it] - median.l) }),
                median(DoubleArray(aValues.size) { abs(aValues[it] - median.a) }),
                median(DoubleArray(bValues.size) { abs(bValues[it] - median.b) }),
            ),
            meanRgb = Rgb(mean(reds), mean(greens), mean(blues)),
            stdRgb = Rgb(std(reds), std(greens), std(blues)),
        )
    }

    private fun relationComparison(original: ToneStage, output: ToneStage): JSONObject {
        val residual = output.faceMinusNeck - original.faceMinusNeck
        return JSONObject()
            .put("relation_residual_lab", residual.toJson())
            .put("relation_residual_delta_e76", residual.norm())
            .put(
                "face_delta_e00_from_original",
                deltaE2000(original.face.median, output.face.median),
            )
            .put(
                "face_neck_delta_e00_change",
                output.deltaE00 - original.deltaE00,
            )
    }

    private fun stageDrift(before: ToneStage, after: ToneStage): JSONObject {
        val relationDrift = after.faceMinusNeck - before.faceMinusNeck
        return JSONObject()
            .put("relation_drift_lab", relationDrift.toJson())
            .put("relation_drift_delta_e76", relationDrift.norm())
            .put("face_delta_e00", deltaE2000(before.face.median, after.face.median))
            .put("face_neck_delta_e00_change", after.deltaE00 - before.deltaE00)
    }

    private fun requireFinite(stage: ToneStage) {
        val values = listOf(
            stage.face.median.l,
            stage.face.median.a,
            stage.face.median.b,
            stage.neck.median.l,
            stage.neck.median.a,
            stage.neck.median.b,
            stage.faceMinusNeck.l,
            stage.faceMinusNeck.a,
            stage.faceMinusNeck.b,
            stage.deltaE00,
        )
        assertTrue("Tone metrics must be finite: $values", values.all(Double::isFinite))
    }

    private fun rgbToLab(red: Double, green: Double, blue: Double): Lab {
        val r = linearSrgb(red / 255.0)
        val g = linearSrgb(green / 255.0)
        val b = linearSrgb(blue / 255.0)
        val x = 0.4124564 * r + 0.3575761 * g + 0.1804375 * b
        val y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b
        val z = 0.0193339 * r + 0.1191920 * g + 0.9503041 * b
        val fx = labPivot(x / D65_X)
        val fy = labPivot(y / D65_Y)
        val fz = labPivot(z / D65_Z)
        return Lab(
            l = 116.0 * fy - 16.0,
            a = 500.0 * (fx - fy),
            b = 200.0 * (fy - fz),
        )
    }

    private fun linearSrgb(value: Double): Double = if (value <= 0.04045) {
        value / 12.92
    } else {
        ((value + 0.055) / 1.055).pow(2.4)
    }

    private fun labPivot(value: Double): Double = if (value > LAB_EPSILON) {
        value.pow(1.0 / 3.0)
    } else {
        (LAB_KAPPA * value + 16.0) / 116.0
    }

    /** Sharma et al. CIEDE2000 with unit weighting factors. */
    private fun deltaE2000(first: Lab, second: Lab): Double {
        val c1 = hypot(first.a, first.b)
        val c2 = hypot(second.a, second.b)
        val meanC = (c1 + c2) / 2.0
        val meanC7 = meanC.pow(7)
        val g = 0.5 * (1.0 - sqrt(meanC7 / (meanC7 + TWENTY_FIVE_POW_7)))
        val a1Prime = (1.0 + g) * first.a
        val a2Prime = (1.0 + g) * second.a
        val c1Prime = hypot(a1Prime, first.b)
        val c2Prime = hypot(a2Prime, second.b)
        val h1Prime = hueDegrees(first.b, a1Prime)
        val h2Prime = hueDegrees(second.b, a2Prime)
        val deltaLPrime = second.l - first.l
        val deltaCPrime = c2Prime - c1Prime
        val deltaHueDegrees = when {
            c1Prime * c2Prime == 0.0 -> 0.0
            abs(h2Prime - h1Prime) <= 180.0 -> h2Prime - h1Prime
            h2Prime <= h1Prime -> h2Prime - h1Prime + 360.0
            else -> h2Prime - h1Prime - 360.0
        }
        val deltaHPrime =
            2.0 * sqrt(c1Prime * c2Prime) * sin(degreesToRadians(deltaHueDegrees / 2.0))
        val meanLPrime = (first.l + second.l) / 2.0
        val meanCPrime = (c1Prime + c2Prime) / 2.0
        val meanHPrime = when {
            c1Prime * c2Prime == 0.0 -> h1Prime + h2Prime
            abs(h1Prime - h2Prime) <= 180.0 -> (h1Prime + h2Prime) / 2.0
            h1Prime + h2Prime < 360.0 -> (h1Prime + h2Prime + 360.0) / 2.0
            else -> (h1Prime + h2Prime - 360.0) / 2.0
        }
        val t = 1.0 - 0.17 * cos(degreesToRadians(meanHPrime - 30.0)) +
            0.24 * cos(degreesToRadians(2.0 * meanHPrime)) +
            0.32 * cos(degreesToRadians(3.0 * meanHPrime + 6.0)) -
            0.20 * cos(degreesToRadians(4.0 * meanHPrime - 63.0))
        val deltaTheta = 30.0 * exp(-((meanHPrime - 275.0) / 25.0).pow(2))
        val meanCPrime7 = meanCPrime.pow(7)
        val rc = 2.0 * sqrt(meanCPrime7 / (meanCPrime7 + TWENTY_FIVE_POW_7))
        val sl = 1.0 + 0.015 * (meanLPrime - 50.0).pow(2) /
            sqrt(20.0 + (meanLPrime - 50.0).pow(2))
        val sc = 1.0 + 0.045 * meanCPrime
        val sh = 1.0 + 0.015 * meanCPrime * t
        val rt = -sin(degreesToRadians(2.0 * deltaTheta)) * rc
        val lTerm = deltaLPrime / sl
        val cTerm = deltaCPrime / sc
        val hTerm = deltaHPrime / sh
        return sqrt(max(lTerm.pow(2) + cTerm.pow(2) + hTerm.pow(2) + rt * cTerm * hTerm, 0.0))
    }

    private fun hueDegrees(b: Double, aPrime: Double): Double {
        val degrees = atan2(b, aPrime) * 180.0 / PI
        return if (degrees >= 0.0) degrees else degrees + 360.0
    }

    private fun degreesToRadians(degrees: Double): Double = degrees * PI / 180.0

    private fun polygonIndices(width: Int, height: Int, polygon: List<Point2>): IntArray {
        require(polygon.size >= 3)
        val left = kotlin.math.floor(polygon.minOf(Point2::x)).toInt().coerceIn(0, width)
        val top = kotlin.math.floor(polygon.minOf(Point2::y)).toInt().coerceIn(0, height)
        val right = kotlin.math.ceil(polygon.maxOf(Point2::x)).toInt().coerceIn(0, width)
        val bottom = kotlin.math.ceil(polygon.maxOf(Point2::y)).toInt().coerceIn(0, height)
        val output = ArrayList<Int>()
        for (y in top until bottom) {
            for (x in left until right) {
                if (pointInPolygon(x + 0.5, y + 0.5, polygon)) output += y * width + x
            }
        }
        return output.toIntArray()
    }

    private fun pointInPolygon(x: Double, y: Double, polygon: List<Point2>): Boolean {
        var inside = false
        var previous = polygon.last()
        polygon.forEach { current ->
            if (
                (current.y > y) != (previous.y > y) &&
                x < (previous.x - current.x) * (y - current.y) /
                (previous.y - current.y) + current.x
            ) {
                inside = !inside
            }
            previous = current
        }
        return inside
    }

    private fun changedAt(first: IntArray, second: IntArray, indices: IntArray): Int =
        indices.count { first[it] != second[it] }

    private fun statisticsFrameBounds(targetToCrop: AffineMatrix): List<Point2> {
        val cropToTarget = targetToCrop.inverse()
        return listOf(
            cropToTarget.map(Point2(STAT_LEFT.toDouble(), STAT_TOP.toDouble())),
            cropToTarget.map(Point2(STAT_RIGHT_EXCLUSIVE.toDouble(), STAT_TOP.toDouble())),
            cropToTarget.map(
                Point2(STAT_RIGHT_EXCLUSIVE.toDouble(), STAT_BOTTOM_EXCLUSIVE.toDouble()),
            ),
            cropToTarget.map(Point2(STAT_LEFT.toDouble(), STAT_BOTTOM_EXCLUSIVE.toDouble())),
        )
    }

    private fun saveTriptych(
        original: IntArray,
        postSwap: IntArray,
        postGfpgan: IntArray,
        width: Int,
        height: Int,
        facePolygon: List<Point2>,
        neckPolygon: List<Point2>,
        file: File,
    ) {
        val panels = listOf(original.copyOf(), postSwap.copyOf(), postGfpgan.copyOf())
        val outputWidth = width * panels.size + TRIPTYCH_GAP * (panels.size - 1)
        val output = IntArray(outputWidth * height) { OPAQUE_WHITE }
        try {
            panels.forEachIndexed { panelIndex, panel ->
                drawPolygon(panel, width, height, facePolygon, FACE_OUTLINE)
                drawPolygon(panel, width, height, neckPolygon, NECK_OUTLINE)
                val destinationX = panelIndex * (width + TRIPTYCH_GAP)
                for (y in 0 until height) {
                    panel.copyInto(
                        output,
                        destinationOffset = y * outputWidth + destinationX,
                        startIndex = y * width,
                        endIndex = (y + 1) * width,
                    )
                }
            }
            val bitmap = bitmapFromPixels(output, outputWidth, height)
            try {
                savePng(bitmap, file)
            } finally {
                bitmap.recycleSafely()
            }
        } finally {
            panels.forEach { it.fill(0) }
            output.fill(0)
        }
    }

    private fun drawPolygon(
        pixels: IntArray,
        width: Int,
        height: Int,
        polygon: List<Point2>,
        color: Int,
    ) {
        polygon.indices.forEach { index ->
            val start = polygon[index]
            val end = polygon[(index + 1) % polygon.size]
            drawLine(
                pixels,
                width,
                height,
                start.x.roundToInt(),
                start.y.roundToInt(),
                end.x.roundToInt(),
                end.y.roundToInt(),
                color,
            )
        }
    }

    private fun drawLine(
        pixels: IntArray,
        width: Int,
        height: Int,
        startX: Int,
        startY: Int,
        endX: Int,
        endY: Int,
        color: Int,
    ) {
        var x = startX
        var y = startY
        val dx = abs(endX - startX)
        val stepX = if (startX < endX) 1 else -1
        val dy = -abs(endY - startY)
        val stepY = if (startY < endY) 1 else -1
        var error = dx + dy
        while (true) {
            for (offsetY in -OUTLINE_RADIUS..OUTLINE_RADIUS) {
                for (offsetX in -OUTLINE_RADIUS..OUTLINE_RADIUS) {
                    val drawX = x + offsetX
                    val drawY = y + offsetY
                    if (drawX in 0 until width && drawY in 0 until height) {
                        pixels[drawY * width + drawX] = color
                    }
                }
            }
            if (x == endX && y == endY) break
            val doubled = 2 * error
            if (doubled >= dy) {
                error += dy
                x += stepX
            }
            if (doubled <= dx) {
                error += dx
                y += stepY
            }
        }
    }

    private fun outputDirectory(): File = File(context.filesDir, OUTPUT_DIRECTORY).apply {
        if (exists()) check(deleteRecursively()) { "Could not clear $absolutePath" }
        check(mkdirs()) { "Could not create $absolutePath" }
    }

    private fun bitmap(path: String): Bitmap = assets.open(path).use { input ->
        requireNotNull(BitmapFactory.decodeStream(input)) { "Cannot decode $path" }
            .copy(Bitmap.Config.ARGB_8888, false)
    }

    private fun bitmapFromPixels(pixels: IntArray, width: Int, height: Int): Bitmap =
        Bitmap.createBitmap(pixels, width, height, Bitmap.Config.ARGB_8888)

    private fun Bitmap.pixels(): IntArray = IntArray(width * height).also { destination ->
        getPixels(destination, 0, width, 0, 0, width, height)
    }

    private fun savePng(bitmap: Bitmap, file: File) {
        FileOutputStream(file).use { output ->
            check(bitmap.compress(Bitmap.CompressFormat.PNG, 100, output))
        }
    }

    private fun sha256Asset(path: String): String {
        val digest = MessageDigest.getInstance("SHA-256")
        assets.open(path).use { input ->
            val buffer = ByteArray(DEFAULT_BUFFER_SIZE)
            while (true) {
                val count = input.read(buffer)
                if (count < 0) break
                digest.update(buffer, 0, count)
            }
            buffer.fill(0)
        }
        return digest.digest().toHex()
    }

    private fun sha256FloatLittleEndian(values: FloatArray): String {
        val digest = MessageDigest.getInstance("SHA-256")
        val buffer = ByteBuffer.allocate(values.size * Float.SIZE_BYTES)
            .order(ByteOrder.LITTLE_ENDIAN)
        values.forEach(buffer::putFloat)
        digest.update(buffer.array())
        buffer.array().fill(0)
        return digest.digest().toHex()
    }

    private fun ByteArray.toHex(): String =
        joinToString("") { byte -> "%02x".format(byte.toInt() and 0xff) }

    private fun landmarksJson(points: List<Point2>): JSONArray = JSONArray().also { array ->
        points.forEach { point -> array.put(JSONArray().put(point.x).put(point.y)) }
    }

    private fun affineJson(matrix: AffineMatrix): JSONArray = JSONArray()
        .put(JSONArray().put(matrix.a).put(matrix.b).put(matrix.c))
        .put(JSONArray().put(matrix.d).put(matrix.e).put(matrix.f))

    private fun polygonJson(points: List<Point2>): JSONArray = JSONArray().also { array ->
        points.forEach { point -> array.put(JSONArray().put(point.x).put(point.y)) }
    }

    private fun doubleArrayJson(values: DoubleArray): JSONArray = JSONArray().also { array ->
        values.forEach(array::put)
    }

    private fun channel(pixel: Int, shift: Int): Int = (pixel ushr shift) and 0xff

    private fun mean(values: DoubleArray): Double = values.sum() / values.size

    private fun std(values: DoubleArray): Double {
        val mean = mean(values)
        return sqrt(values.sumOf { (it - mean).pow(2) } / values.size)
    }

    private fun median(values: DoubleArray): Double {
        require(values.isNotEmpty())
        val ordered = values.sortedArray()
        val middle = ordered.size / 2
        return if (ordered.size % 2 == 0) {
            (ordered[middle - 1] + ordered[middle]) / 2.0
        } else {
            ordered[middle]
        }
    }

    private fun RawFaceSwapResult.release() {
        alignedSource112.recycleSafely()
        alignedTarget.recycleSafely()
        rawOutputBitmap.recycleSafely()
        rawOutput.fill(0f)
        rawMask?.fill(0f)
        sourceEmbedding.fill(0f)
    }

    private fun Bitmap.recycleSafely() {
        if (!isRecycled) recycle()
    }

    private data class FixtureSpec(
        val pair: String,
        val facePolygon: List<Point2>,
        val neckPolygon: List<Point2>,
        val expectedFaceSamples: Int,
        val expectedNeckSamples: Int,
    )

    private data class Rgb(val red: Double, val green: Double, val blue: Double) {
        fun toJson(): JSONObject = JSONObject()
            .put("red", red)
            .put("green", green)
            .put("blue", blue)
    }

    private data class Lab(val l: Double, val a: Double, val b: Double) {
        operator fun minus(other: Lab): Lab = Lab(l - other.l, a - other.a, b - other.b)
        fun norm(): Double = sqrt(l * l + a * a + b * b)
        fun toJson(): JSONObject = JSONObject().put("l", l).put("a", a).put("b", b)
    }

    private data class ColorSummary(
        val samples: Int,
        val median: Lab,
        val mad: Lab,
        val meanRgb: Rgb,
        val stdRgb: Rgb,
    ) {
        fun toJson(): JSONObject = JSONObject()
            .put("samples", samples)
            .put("median_lab", median.toJson())
            .put("mad_lab", mad.toJson())
            .put("mean_rgb", meanRgb.toJson())
            .put("std_rgb", stdRgb.toJson())
    }

    private data class ToneStage(
        val face: ColorSummary,
        val neck: ColorSummary,
        val faceMinusNeck: Lab,
        val deltaE00: Double,
    ) {
        fun toJson(): JSONObject = JSONObject()
            .put("face", face.toJson())
            .put("neck", neck.toJson())
            .put("face_minus_neck_lab", faceMinusNeck.toJson())
            .put("face_neck_delta_e00", deltaE00)
    }

    private companion object {
        const val OUTPUT_DIRECTORY = "stage-e-checkpoint-3"
        const val RESULTS_FILE = "checkpoint_3_color_diagnostics.json"
        const val IMAGE_SIZE = 1254
        const val SWAP_CROP_SIZE = 128
        const val RESTORATION_STRENGTH = 0.8f
        const val MASK_STAT_THRESHOLD = 0.5f
        const val EXPECTED_STAT_SAMPLES = 12_024
        const val STAT_LEFT = 9
        const val STAT_TOP = 9
        const val STAT_RIGHT_EXCLUSIVE = 119
        const val STAT_BOTTOM_EXCLUSIVE = 119
        const val LIMIT_EPSILON = 1e-12
        const val D65_X = 0.95047
        const val D65_Y = 1.0
        const val D65_Z = 1.08883
        const val LAB_EPSILON = 216.0 / 24389.0
        const val LAB_KAPPA = 24389.0 / 27.0
        const val TWENTY_FIVE_POW_7 = 6_103_515_625.0
        const val TRIPTYCH_GAP = 8
        const val OPAQUE_WHITE = -0x1
        const val FACE_OUTLINE = -0xff0100
        const val NECK_OUTLINE = -0xffff01
        const val OUTLINE_RADIUS = 1

        val REQUIRED_MODELS = setOf(
            ModelId.YOLOFACE_8N,
            ModelId.ARCFACE_W600K_R50,
            ModelId.INSWAPPER_128_FP16,
            ModelId.GFPGAN_1_4,
            ModelId.BISENET_RESNET_34,
        )

        val FIXTURES = listOf(
            FixtureSpec(
                pair = "pair_02",
                facePolygon = listOf(
                    Point2(685.467, 401.595),
                    Point2(731.964, 396.008),
                    Point2(735.689, 427.006),
                    Point2(689.193, 432.593),
                ),
                neckPolygon = listOf(
                    Point2(590.0, 940.0),
                    Point2(680.0, 940.0),
                    Point2(680.0, 985.0),
                    Point2(590.0, 985.0),
                ),
                expectedFaceSamples = 1_462,
                expectedNeckSamples = 4_050,
            ),
            FixtureSpec(
                pair = "pair_03",
                facePolygon = listOf(
                    Point2(622.348, 473.065),
                    Point2(668.455, 463.684),
                    Point2(674.709, 494.422),
                    Point2(628.603, 503.803),
                ),
                neckPolygon = listOf(
                    Point2(718.0, 983.0),
                    Point2(779.0, 970.0),
                    Point2(786.0, 1001.0),
                    Point2(724.0, 1014.0),
                ),
                expectedFaceSamples = 1_475,
                expectedNeckSamples = 1_991,
            ),
        )
    }
}
