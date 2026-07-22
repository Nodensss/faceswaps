package com.faceswaplocal.app.inference

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import java.io.File
import java.io.FileOutputStream
import kotlin.math.max
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class FaceFusionFinalFrameParityInstrumentedTest {
    private val context = ApplicationProvider.getApplicationContext<Context>()
    private val assets = InstrumentationRegistry.getInstrumentation().context.assets

    @Test
    fun finalInSwapperFramesMatchFaceFusionVisualReferenceContract() {
        val modelStore = ModelStore(context)
        val statuses = kotlinx.coroutines.runBlocking { modelStore.refreshStatuses() }
        listOf(
            ModelId.YOLOFACE_8N,
            ModelId.ARCFACE_W600K_R50,
            ModelId.INSWAPPER_128_FP16,
        ).forEach { modelId ->
            assertTrue(
                "Import and verify ${modelId.stableId} through the app before Stage C parity",
                statuses[modelId] is ModelStatus.Ready,
            )
        }

        val pipeline = OnnxPhotoFaceSwapPipeline(modelStore)
        val outputDirectory = File(context.filesDir, "stage-c-parity-output").apply {
            deleteRecursively()
            mkdirs()
        }
        val allMetrics = JSONArray()

        for (pairNumber in 1..3) {
            val pair = "pair_%02d".format(pairNumber)
            val source = decodeBitmap("inputs/${pair}_source.png")
            val target = decodeBitmap("inputs/${pair}_target.png")
            val reference = decodeBitmap(
                "reference/facefusion-3.7.1/$pair/inswapper_final_box_03.png",
            )
            val result = kotlinx.coroutines.runBlocking {
                pipeline.process(
                    PhotoFaceSwapRequest(
                        source = source,
                        target = target,
                        backend = RequestedInferenceBackend.CPU_ONLY,
                    ),
                )
            }

            try {
                assertEquals(target.width, result.finalBitmap.width)
                assertEquals(target.height, result.finalBitmap.height)
                assertEquals(reference.width, result.finalBitmap.width)
                assertEquals(reference.height, result.finalBitmap.height)
                assertEquals(InferenceBackend.CPU, result.detectorBackend)
                assertEquals(InferenceBackend.CPU, result.recognizerBackend)
                assertEquals(InferenceBackend.CPU, result.swapperBackend)
                assertTrue(result.cropMask.all { it.isFinite() && it in 0f..1f })
                assertTrue(result.cropMask.first() < 0.1f)
                val maskSize = SwapperModel.INSWAPPER_128_FP16.cropSize
                assertTrue(result.cropMask[(maskSize / 2) * maskSize + maskSize / 2] > 0.9f)

                val targetPixels = target.readPixels()
                val actualPixels = result.finalBitmap.readPixels()
                val referencePixels = reference.readPixels()
                val outsideChanges = countOutsideChanges(
                    targetPixels,
                    actualPixels,
                    target.width,
                    target.height,
                    result.pasteRoi,
                )
                val insideChanges = countInsideChanges(
                    targetPixels,
                    actualPixels,
                    target.width,
                    result.pasteRoi,
                )
                val fullMetrics = comparePixels(
                    actualPixels,
                    referencePixels,
                    target.width,
                    CompositeRoi(0, 0, target.width, target.height),
                )
                val roiMetrics = comparePixels(
                    actualPixels,
                    referencePixels,
                    target.width,
                    result.pasteRoi,
                )

                assertEquals("$pair modified pixels outside inverse paste ROI", 0, outsideChanges)
                assertTrue("$pair did not materially change the selected face ROI", insideChanges > 100)
                assertTrue("$pair full-frame SSIM ${fullMetrics.globalSsim}", fullMetrics.globalSsim >= 0.95)
                assertTrue("$pair face-ROI SSIM ${roiMetrics.globalSsim}", roiMetrics.globalSsim >= 0.95)

                val metric = JSONObject()
                    .put("pair", pair)
                    .put("reference", "FaceFusion 3.7.1 production swap_face, box blur 0.3")
                    .put("mobile_color_match", "masked RGB mean/std, strength 0.65")
                    .put("full_frame_global_ssim", fullMetrics.globalSsim)
                    .put("full_frame_mae_255", fullMetrics.meanAbsoluteError)
                    .put("face_roi_global_ssim", roiMetrics.globalSsim)
                    .put("face_roi_mae_255", roiMetrics.meanAbsoluteError)
                    .put("outside_roi_changed_pixels", outsideChanges)
                    .put("inside_roi_changed_pixels", insideChanges)
                    .put("roi", result.pasteRoi.toJson())
                    .put("color_adjustment", result.colorAdjustment.toJson())
                    .put("detector_backend", result.detectorBackend.name)
                    .put("recognizer_backend", result.recognizerBackend.name)
                    .put("swapper_backend", result.swapperBackend.name)
                    .put("detector_ms", result.timings.detectorMs)
                    .put("recognizer_ms", result.timings.recognizerMs)
                    .put("swapper_ms", result.timings.swapperMs)
                    .put("compositing_ms", result.timings.compositingMs)
                    .put("total_ms", result.timings.totalMs)
                allMetrics.put(metric)
                File(outputDirectory, "stage_c_results.json").writeText(allMetrics.toString(2))
                FileOutputStream(File(outputDirectory, "${pair}_inswapper_final.png")).use { stream ->
                    result.finalBitmap.compress(Bitmap.CompressFormat.PNG, 100, stream)
                }
                writeCropMask(
                    File(outputDirectory, "${pair}_box_mask_03.png"),
                    result.cropMask,
                    SwapperModel.INSWAPPER_128_FP16.cropSize,
                )
            } finally {
                source.recycle()
                target.recycle()
                reference.recycle()
                result.finalBitmap.recycle()
            }
        }
    }

    private fun decodeBitmap(path: String): Bitmap = assets.open(path).use { stream ->
        requireNotNull(BitmapFactory.decodeStream(stream)) { "Cannot decode parity asset $path" }
            .copy(Bitmap.Config.ARGB_8888, false)
    }

    private fun Bitmap.readPixels(): IntArray = IntArray(width * height).also { pixels ->
        getPixels(pixels, 0, width, 0, 0, width, height)
    }

    private fun countOutsideChanges(
        expected: IntArray,
        actual: IntArray,
        width: Int,
        height: Int,
        roi: CompositeRoi,
    ): Int {
        var changed = 0
        for (y in 0 until height) {
            for (x in 0 until width) {
                if (x !in roi.left until roi.right || y !in roi.top until roi.bottom) {
                    val index = y * width + x
                    if (expected[index] != actual[index]) changed++
                }
            }
        }
        return changed
    }

    private fun countInsideChanges(
        expected: IntArray,
        actual: IntArray,
        width: Int,
        roi: CompositeRoi,
    ): Int {
        var changed = 0
        for (y in roi.top until roi.bottom) {
            for (x in roi.left until roi.right) {
                val index = y * width + x
                if (expected[index] != actual[index]) changed++
            }
        }
        return changed
    }

    private fun comparePixels(
        actual: IntArray,
        expected: IntArray,
        width: Int,
        roi: CompositeRoi,
    ): ImageMetrics {
        val count = max(1, roi.width * roi.height)
        var ssimSum = 0.0
        var maeSum = 0.0
        for (shift in intArrayOf(16, 8, 0)) {
            var actualSum = 0.0
            var expectedSum = 0.0
            for (y in roi.top until roi.bottom) {
                for (x in roi.left until roi.right) {
                    val index = y * width + x
                    actualSum += (actual[index] ushr shift) and 0xff
                    expectedSum += (expected[index] ushr shift) and 0xff
                }
            }
            val actualMean = actualSum / count
            val expectedMean = expectedSum / count
            var actualVariance = 0.0
            var expectedVariance = 0.0
            var covariance = 0.0
            for (y in roi.top until roi.bottom) {
                for (x in roi.left until roi.right) {
                    val index = y * width + x
                    val actualChannel = ((actual[index] ushr shift) and 0xff).toDouble()
                    val expectedChannel = ((expected[index] ushr shift) and 0xff).toDouble()
                    val actualDelta = actualChannel - actualMean
                    val expectedDelta = expectedChannel - expectedMean
                    actualVariance += actualDelta * actualDelta
                    expectedVariance += expectedDelta * expectedDelta
                    covariance += actualDelta * expectedDelta
                    maeSum += kotlin.math.abs(actualChannel - expectedChannel)
                }
            }
            actualVariance /= count
            expectedVariance /= count
            covariance /= count
            val c1 = (0.01 * 255.0) * (0.01 * 255.0)
            val c2 = (0.03 * 255.0) * (0.03 * 255.0)
            ssimSum +=
                ((2 * actualMean * expectedMean + c1) * (2 * covariance + c2)) /
                ((actualMean * actualMean + expectedMean * expectedMean + c1) *
                    (actualVariance + expectedVariance + c2))
        }
        return ImageMetrics(
            globalSsim = ssimSum / 3.0,
            meanAbsoluteError = maeSum / (count * 3.0),
        )
    }

    private fun writeCropMask(file: File, mask: FloatArray, size: Int) {
        val pixels = IntArray(mask.size) { index ->
            val value = (mask[index].coerceIn(0f, 1f) * 255f).toInt()
            (0xff shl 24) or (value shl 16) or (value shl 8) or value
        }
        val bitmap = Bitmap.createBitmap(pixels, size, size, Bitmap.Config.ARGB_8888)
        try {
            FileOutputStream(file).use { stream -> bitmap.compress(Bitmap.CompressFormat.PNG, 100, stream) }
        } finally {
            bitmap.recycle()
        }
    }

    private fun CompositeRoi.toJson(): JSONObject = JSONObject()
        .put("left", left)
        .put("top", top)
        .put("right", right)
        .put("bottom", bottom)

    private fun FaceColorAdjustment.toJson(): JSONObject = JSONObject()
        .put("red_gain", redGain)
        .put("green_gain", greenGain)
        .put("blue_gain", blueGain)
        .put("red_offset", redOffset)
        .put("green_offset", greenOffset)
        .put("blue_offset", blueOffset)
        .put("strength", strength)

    private data class ImageMetrics(
        val globalSsim: Double,
        val meanAbsoluteError: Double,
    )
}
