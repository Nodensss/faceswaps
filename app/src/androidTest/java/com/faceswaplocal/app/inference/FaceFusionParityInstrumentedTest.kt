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
import kotlin.math.abs
import kotlin.math.max
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class FaceFusionParityInstrumentedTest {
    private val context = ApplicationProvider.getApplicationContext<Context>()
    private val assets = InstrumentationRegistry.getInstrumentation().context.assets

    @Test
    fun rawPipelinesMatchFaceFusion371CpuReference() {
        val modelStore = ModelStore(context)
        val statuses = kotlinx.coroutines.runBlocking { modelStore.refreshStatuses() }
        ModelCatalog.all.forEach { descriptor ->
            assertTrue(
                "Import and verify ${descriptor.fileName} through the app before parity",
                statuses[descriptor.id] is ModelStatus.Ready,
            )
        }

        val pipeline = OnnxRawFaceSwapPipeline(modelStore)
        val outputDirectory = File(context.filesDir, "parity-output").apply { mkdirs() }
        val allMetrics = JSONArray()

        for (pairNumber in 1..3) {
            val pair = "pair_%02d".format(pairNumber)
            val metadata = readJson("reference/facefusion-3.7.1/$pair/metadata.json")
            for (swapper in SwapperModel.entries) {
                val source = decodeBitmap("inputs/${pair}_source.png")
                val target = decodeBitmap("inputs/${pair}_target.png")
                val result = kotlinx.coroutines.runBlocking {
                    pipeline.process(
                        RawFaceSwapRequest(
                            source = source,
                            target = target,
                            swapper = swapper,
                            backend = RequestedInferenceBackend.CPU_ONLY,
                        ),
                    )
                }

                try {
                    val geometry = compareGeometry(metadata, result, swapper)
                    val referenceRawPath = when (swapper) {
                        SwapperModel.HYPERSWAP_1A_256 ->
                            "reference/facefusion-3.7.1/$pair/raw_output_f32le.bin"

                        SwapperModel.INSWAPPER_128_FP16 ->
                            "reference/facefusion-3.7.1/$pair/inswapper_raw_output_f32le.bin"
                    }
                    val referenceRaw = readFloatAsset(referenceRawPath)
                    assertEquals(referenceRaw.size, result.rawOutput.size)
                    val normalizedActual = normalizeRawForSsim(result.rawOutput, swapper)
                    val normalizedReference = normalizeRawForSsim(referenceRaw, swapper)
                    val rawSsim = structuralSimilarity(
                        normalizedActual,
                        normalizedReference,
                        width = swapper.cropSize,
                        height = swapper.cropSize,
                    )

                    val metric = JSONObject()
                        .put("pair", pair)
                        .put("swapper", swapper.name)
                        .put("landmark_max_error_px", geometry.landmarkMaxError)
                        .put("source_matrix_max_abs_error", geometry.sourceMatrixMaxError)
                        .put("target_matrix_max_abs_error", geometry.targetMatrixMaxError)
                        .put("source_aligned_ssim", geometry.sourceAlignedSsim)
                        .put("target_aligned_ssim", geometry.targetAlignedSsim)
                        .put("raw_output_ssim", rawSsim)
                        .put("detector_backend", result.detectorBackend.name)
                        .put("recognizer_backend", result.recognizerBackend.name)
                        .put("swapper_backend", result.swapperBackend.name)
                        .put("detector_ms", result.timings.detectorMs)
                        .put("recognizer_ms", result.timings.recognizerMs)
                        .put("swapper_ms", result.timings.swapperMs)
                        .put("total_ms", result.timings.totalMs)
                    allMetrics.put(metric)
                    writeMetrics(outputDirectory, allMetrics)
                    FileOutputStream(File(outputDirectory, "${pair}_${swapper.name.lowercase()}.png")).use {
                        result.rawOutputBitmap.compress(Bitmap.CompressFormat.PNG, 100, it)
                    }

                    assertTrue("$pair/$swapper landmark error ${geometry.landmarkMaxError}", geometry.landmarkMaxError <= 2.0)
                    assertTrue("$pair/$swapper source crop SSIM ${geometry.sourceAlignedSsim}", geometry.sourceAlignedSsim >= 0.95)
                    assertTrue("$pair/$swapper target crop SSIM ${geometry.targetAlignedSsim}", geometry.targetAlignedSsim >= 0.95)
                    assertTrue("$pair/$swapper raw SSIM $rawSsim", rawSsim >= 0.95)
                    assertEquals(InferenceBackend.CPU, result.detectorBackend)
                    assertEquals(InferenceBackend.CPU, result.recognizerBackend)
                    assertEquals(InferenceBackend.CPU, result.swapperBackend)
                } finally {
                    source.recycle()
                    target.recycle()
                    result.alignedSource112.recycle()
                    result.alignedTarget.recycle()
                    result.rawOutputBitmap.recycle()
                }
            }
        }
    }

    private fun compareGeometry(
        metadata: JSONObject,
        result: RawFaceSwapResult,
        swapper: SwapperModel,
    ): GeometryMetrics {
        val sourceMetadata = metadata.getJSONObject("source")
        val targetMetadata = metadata.getJSONObject("target")
        val sourceLandmarks = readPoints(sourceMetadata.getJSONArray("landmarks_5_xy"))
        val targetLandmarks = readPoints(targetMetadata.getJSONArray("landmarks_5_xy"))
        val landmarkMaxError = max(
            maxPointError(sourceLandmarks, result.sourceFace.landmarks),
            maxPointError(targetLandmarks, result.targetFace.landmarks),
        )

        val expectedSourceMatrix = readMatrix(sourceMetadata.getJSONArray("affine_source_to_crop"))
        val targetMatrixKey = when (swapper) {
            SwapperModel.HYPERSWAP_1A_256 -> "affine_source_to_crop"
            SwapperModel.INSWAPPER_128_FP16 -> "fallback_affine_source_to_crop"
        }
        val expectedTargetMatrix = readMatrix(targetMetadata.getJSONArray(targetMatrixKey))

        val pair = metadata.getString("pair")
        val expectedSourceCrop = decodeBitmap(
            "reference/facefusion-3.7.1/$pair/source_aligned_112.png",
        )
        val targetCropFile = when (swapper) {
            SwapperModel.HYPERSWAP_1A_256 -> "target_aligned_256.png"
            SwapperModel.INSWAPPER_128_FP16 -> "target_aligned_128.png"
        }
        val expectedTargetCrop = decodeBitmap(
            "reference/facefusion-3.7.1/$pair/$targetCropFile",
        )
        return try {
            GeometryMetrics(
                landmarkMaxError = landmarkMaxError,
                sourceMatrixMaxError = matrixMaxError(expectedSourceMatrix, result.sourceToRecognizerCrop),
                targetMatrixMaxError = matrixMaxError(expectedTargetMatrix, result.targetToSwapperCrop),
                sourceAlignedSsim = bitmapSsim(result.alignedSource112, expectedSourceCrop),
                targetAlignedSsim = bitmapSsim(result.alignedTarget, expectedTargetCrop),
            )
        } finally {
            expectedSourceCrop.recycle()
            expectedTargetCrop.recycle()
        }
    }

    private fun decodeBitmap(path: String): Bitmap = assets.open(path).use { stream ->
        requireNotNull(BitmapFactory.decodeStream(stream)) { "Cannot decode parity asset $path" }
            .copy(Bitmap.Config.ARGB_8888, false)
    }

    private fun readJson(path: String): JSONObject = assets.open(path).bufferedReader().use { reader ->
        JSONObject(reader.readText())
    }

    private fun readFloatAsset(path: String): FloatArray = assets.open(path).use { input ->
        val bytes = input.readBytes()
        require(bytes.size % Float.SIZE_BYTES == 0)
        val buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
        FloatArray(buffer.remaining()).also(buffer::get)
    }

    private fun writeMetrics(directory: File, metrics: JSONArray) {
        File(directory, "android_results.json").writeText(metrics.toString(2))
    }

    private fun readPoints(array: JSONArray): List<Point2> = (0 until array.length()).map { index ->
        val point = array.getJSONArray(index)
        Point2(point.getDouble(0), point.getDouble(1))
    }

    private fun readMatrix(array: JSONArray): AffineMatrix {
        val first = array.getJSONArray(0)
        val second = array.getJSONArray(1)
        return AffineMatrix(
            a = first.getDouble(0),
            b = first.getDouble(1),
            c = first.getDouble(2),
            d = second.getDouble(0),
            e = second.getDouble(1),
            f = second.getDouble(2),
        )
    }

    private fun maxPointError(expected: List<Point2>, actual: List<Point2>): Double =
        expected.zip(actual).maxOf { (expectedPoint, actualPoint) ->
            max(abs(expectedPoint.x - actualPoint.x), abs(expectedPoint.y - actualPoint.y))
        }

    private fun matrixMaxError(expected: AffineMatrix, actual: AffineMatrix): Double =
        listOf(
            abs(expected.a - actual.a),
            abs(expected.b - actual.b),
            abs(expected.c - actual.c),
            abs(expected.d - actual.d),
            abs(expected.e - actual.e),
            abs(expected.f - actual.f),
        ).max()

    private fun bitmapSsim(actual: Bitmap, expected: Bitmap): Double {
        assertEquals(expected.width, actual.width)
        assertEquals(expected.height, actual.height)
        return structuralSimilarity(
            bitmapToRgbTensor(actual),
            bitmapToRgbTensor(expected),
            actual.width,
            actual.height,
        )
    }

    private fun bitmapToRgbTensor(bitmap: Bitmap): FloatArray {
        val pixels = IntArray(bitmap.width * bitmap.height)
        bitmap.getPixels(pixels, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)
        val planeSize = pixels.size
        return FloatArray(3 * planeSize).also { tensor ->
            pixels.forEachIndexed { index, pixel ->
                tensor[index] = ((pixel ushr 16) and 0xff) / 255f
                tensor[planeSize + index] = ((pixel ushr 8) and 0xff) / 255f
                tensor[2 * planeSize + index] = (pixel and 0xff) / 255f
            }
        }
    }

    private fun normalizeRawForSsim(values: FloatArray, swapper: SwapperModel): FloatArray =
        when (swapper) {
            SwapperModel.HYPERSWAP_1A_256 -> FloatArray(values.size) { index -> values[index] * 0.5f + 0.5f }
            SwapperModel.INSWAPPER_128_FP16 -> values
        }

    /** Skimage-compatible 7x7 uniform-window SSIM, averaged over RGB channels. */
    private fun structuralSimilarity(
        actual: FloatArray,
        expected: FloatArray,
        width: Int,
        height: Int,
    ): Double {
        assertEquals(actual.size, expected.size)
        assertEquals(3 * width * height, actual.size)
        val radius = 3
        val count = 49
        val covarianceScale = count.toDouble() / (count - 1).toDouble()
        val c1 = 0.01 * 0.01
        val c2 = 0.03 * 0.03
        val planeSize = width * height
        var total = 0.0
        var windows = 0

        for (channel in 0 until 3) {
            val planeOffset = channel * planeSize
            for (centerY in radius until height - radius) {
                for (centerX in radius until width - radius) {
                    var sumActual = 0.0
                    var sumExpected = 0.0
                    var sumActualSquared = 0.0
                    var sumExpectedSquared = 0.0
                    var sumProduct = 0.0
                    for (y in centerY - radius..centerY + radius) {
                        for (x in centerX - radius..centerX + radius) {
                            val index = planeOffset + y * width + x
                            val a = actual[index].toDouble()
                            val e = expected[index].toDouble()
                            sumActual += a
                            sumExpected += e
                            sumActualSquared += a * a
                            sumExpectedSquared += e * e
                            sumProduct += a * e
                        }
                    }
                    val meanActual = sumActual / count
                    val meanExpected = sumExpected / count
                    val varianceActual = (sumActualSquared / count - meanActual * meanActual) * covarianceScale
                    val varianceExpected = (sumExpectedSquared / count - meanExpected * meanExpected) * covarianceScale
                    val covariance = (sumProduct / count - meanActual * meanExpected) * covarianceScale
                    val numerator = (2 * meanActual * meanExpected + c1) * (2 * covariance + c2)
                    val denominator =
                        (meanActual * meanActual + meanExpected * meanExpected + c1) *
                            (varianceActual + varianceExpected + c2)
                    total += numerator / denominator
                    windows++
                }
            }
        }
        return total / windows
    }

    private data class GeometryMetrics(
        val landmarkMaxError: Double,
        val sourceMatrixMaxError: Double,
        val targetMatrixMaxError: Double,
        val sourceAlignedSsim: Double,
        val targetAlignedSsim: Double,
    )
}
