package com.faceswaplocal.app.inference

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import java.io.File
import java.io.FileOutputStream
import java.security.MessageDigest
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.sqrt
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

/** Android-only ffhq_512 geometry check. No ONNX model or inference session is opened. */
@RunWith(AndroidJUnit4::class)
class FaceQualityGeometryInstrumentedTest {
    private val context = ApplicationProvider.getApplicationContext<Context>()
    private val assets = InstrumentationRegistry.getInstrumentation().context.assets

    @Test
    fun androidFfhq512WarpMatchesCanonicalDesktopCrop() {
        val stageCBytes = readAsset(STAGE_C_FRAME_ASSET)
        val canonicalBytes = readAsset(CANONICAL_CROP_ASSET)
        assertEquals(STAGE_C_FRAME_SHA256, sha256(stageCBytes))
        assertEquals(OnnxFaceQualityParityCore.CANONICAL_INPUT_SHA256, sha256(canonicalBytes))

        val stageCFrame = decodeArgb(stageCBytes, STAGE_C_FRAME_ASSET)
        val canonicalCrop = decodeArgb(canonicalBytes, CANONICAL_CROP_ASSET)
        val pairMetadata = JSONObject(readAsset(PAIR_METADATA_ASSET).toString(Charsets.UTF_8))
        val qualityMetadata = JSONObject(readAsset(QUALITY_METADATA_ASSET).toString(Charsets.UTF_8))
        val sourceLandmarks = readPoints(
            pairMetadata.getJSONObject("target").getJSONArray("landmarks_5_xy"),
        )
        val desktopMatrix = readMatrix(
            qualityMetadata
                .getJSONObject("canonical_input")
                .getJSONObject("provenance")
                .getJSONArray("affine_matrix"),
        )

        val androidMatrix = FaceGeometry.estimateSimilarity(
            source = sourceLandmarks,
            template = WarpTemplate.FFHQ_512,
            cropWidth = CROP_SIZE,
            cropHeight = CROP_SIZE,
        )
        val landmarkProjectionError = sourceLandmarks.maxOf { landmark ->
            pointDistance(androidMatrix.map(landmark), desktopMatrix.map(landmark))
        }
        val matrixMaxError = matrixMaxAbsError(androidMatrix, desktopMatrix)
        val androidCrop = BitmapSampling.warpAffine(
            source = stageCFrame,
            sourceToDestination = androidMatrix,
            destinationWidth = CROP_SIZE,
            destinationHeight = CROP_SIZE,
        )

        try {
            val cropSsim = bitmapSsim(androidCrop, canonicalCrop)
            val cropMae = bitmapMeanAbsoluteError(androidCrop, canonicalCrop)
            val outputDirectory = cleanOutputDirectory()
            val outputPng = File(outputDirectory, "android_ffhq_512_crop.png")
            FileOutputStream(outputPng).use { output ->
                require(androidCrop.compress(Bitmap.CompressFormat.PNG, 100, output)) {
                    "Could not encode Android ffhq_512 crop"
                }
            }
            val metrics = JSONObject()
                .put("stage_c_frame_sha256", sha256(stageCBytes))
                .put("canonical_crop_sha256", sha256(canonicalBytes))
                .put("landmark_projection_max_error_px", landmarkProjectionError)
                .put("matrix_max_abs_error", matrixMaxError)
                .put("crop_ssim", cropSsim)
                .put("crop_rgb_mae_255", cropMae)
                .put("android_matrix", matrixJson(androidMatrix))
                .put("desktop_matrix", matrixJson(desktopMatrix))
                .put("sampler", "BitmapSampling.warpAffine bilinear edge-replicate")
            File(outputDirectory, "face_quality_geometry_results.json")
                .writeText(metrics.toString(2) + "\n")

            assertTrue(
                "ffhq_512 landmark projection error $landmarkProjectionError px must be <= 2 px",
                landmarkProjectionError <= LANDMARK_ERROR_THRESHOLD_PX,
            )
            assertTrue("ffhq_512 crop SSIM $cropSsim must be >= 0.95", cropSsim >= SSIM_THRESHOLD)
        } finally {
            androidCrop.recycle()
            canonicalCrop.recycle()
            stageCFrame.recycle()
        }
    }

    private fun cleanOutputDirectory(): File {
        val privateRoot = context.filesDir.canonicalFile
        val outputDirectory = File(privateRoot, OUTPUT_DIRECTORY_NAME).canonicalFile
        require(outputDirectory.parentFile == privateRoot) { "Geometry output escaped filesDir" }
        if (outputDirectory.isDirectory) {
            outputDirectory.listFiles().orEmpty().forEach { file ->
                require(file.isFile && file.delete()) { "Could not clean ${file.name}" }
            }
        }
        require(outputDirectory.isDirectory || outputDirectory.mkdirs()) {
            "Could not create geometry output directory"
        }
        return outputDirectory
    }

    private fun readAsset(path: String): ByteArray = assets.open(path).use { it.readBytes() }

    private fun decodeArgb(bytes: ByteArray, path: String): Bitmap {
        val decoded = requireNotNull(BitmapFactory.decodeByteArray(bytes, 0, bytes.size)) {
            "Could not decode $path"
        }
        val bitmap = decoded.copy(Bitmap.Config.ARGB_8888, false)
        if (bitmap !== decoded) decoded.recycle()
        return bitmap
    }

    private fun readPoints(array: JSONArray): List<Point2> =
        List(array.length()) { index ->
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

    private fun matrixJson(matrix: AffineMatrix): JSONArray = JSONArray()
        .put(JSONArray().put(matrix.a).put(matrix.b).put(matrix.c))
        .put(JSONArray().put(matrix.d).put(matrix.e).put(matrix.f))

    private fun matrixMaxAbsError(actual: AffineMatrix, expected: AffineMatrix): Double =
        listOf(
            abs(actual.a - expected.a),
            abs(actual.b - expected.b),
            abs(actual.c - expected.c),
            abs(actual.d - expected.d),
            abs(actual.e - expected.e),
            abs(actual.f - expected.f),
        ).max()

    private fun pointDistance(first: Point2, second: Point2): Double {
        val x = first.x - second.x
        val y = first.y - second.y
        return sqrt(x * x + y * y)
    }

    private fun bitmapMeanAbsoluteError(actual: Bitmap, expected: Bitmap): Double {
        assertEquals(expected.width, actual.width)
        assertEquals(expected.height, actual.height)
        val actualPixels = IntArray(actual.width * actual.height)
        val expectedPixels = IntArray(expected.width * expected.height)
        actual.getPixels(actualPixels, 0, actual.width, 0, 0, actual.width, actual.height)
        expected.getPixels(expectedPixels, 0, expected.width, 0, 0, expected.width, expected.height)
        var total = 0.0
        for (index in actualPixels.indices) {
            total += abs(((actualPixels[index] ushr 16) and 0xff) - ((expectedPixels[index] ushr 16) and 0xff))
            total += abs(((actualPixels[index] ushr 8) and 0xff) - ((expectedPixels[index] ushr 8) and 0xff))
            total += abs((actualPixels[index] and 0xff) - (expectedPixels[index] and 0xff))
        }
        return total / (actualPixels.size * 3)
    }

    private fun bitmapSsim(actual: Bitmap, expected: Bitmap): Double {
        assertEquals(expected.width, actual.width)
        assertEquals(expected.height, actual.height)
        return structuralSimilarity(
            actual = bitmapToRgbTensor(actual),
            expected = bitmapToRgbTensor(expected),
            width = actual.width,
            height = actual.height,
        )
    }

    private fun bitmapToRgbTensor(bitmap: Bitmap): FloatArray {
        val pixels = IntArray(bitmap.width * bitmap.height)
        bitmap.getPixels(pixels, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)
        val planeSize = pixels.size
        return FloatArray(planeSize * 3).also { output ->
            pixels.forEachIndexed { index, pixel ->
                output[index] = ((pixel ushr 16) and 0xff) / 255f
                output[planeSize + index] = ((pixel ushr 8) and 0xff) / 255f
                output[2 * planeSize + index] = (pixel and 0xff) / 255f
            }
        }
    }

    /** Skimage-compatible 7x7 uniform-window SSIM, averaged over planar RGB. */
    private fun structuralSimilarity(
        actual: FloatArray,
        expected: FloatArray,
        width: Int,
        height: Int,
    ): Double {
        require(actual.size == expected.size && actual.size == 3 * width * height)
        val radius = 3
        val sampleCount = 49
        val covarianceScale = sampleCount.toDouble() / (sampleCount - 1).toDouble()
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
                    val meanActual = sumActual / sampleCount
                    val meanExpected = sumExpected / sampleCount
                    val varianceActual =
                        (sumActualSquared / sampleCount - meanActual * meanActual) * covarianceScale
                    val varianceExpected =
                        (sumExpectedSquared / sampleCount - meanExpected * meanExpected) * covarianceScale
                    val covariance =
                        (sumProduct / sampleCount - meanActual * meanExpected) * covarianceScale
                    total += ((2 * meanActual * meanExpected + c1) * (2 * covariance + c2)) /
                        ((meanActual * meanActual + meanExpected * meanExpected + c1) *
                            (varianceActual + varianceExpected + c2))
                    windows++
                }
            }
        }
        return total / windows
    }

    private fun sha256(bytes: ByteArray): String =
        MessageDigest.getInstance("SHA-256").digest(bytes)
            .joinToString("") { byte -> "%02x".format(byte) }

    private companion object {
        const val CROP_SIZE = 512
        const val LANDMARK_ERROR_THRESHOLD_PX = 2.0
        const val SSIM_THRESHOLD = 0.95
        const val OUTPUT_DIRECTORY_NAME = "stage-e1-geometry-output"
        const val STAGE_C_FRAME_ASSET =
            "reference/facefusion-3.7.1/pair_01/inswapper_final_box_03.png"
        const val PAIR_METADATA_ASSET = "reference/facefusion-3.7.1/pair_01/metadata.json"
        const val QUALITY_METADATA_ASSET =
            "reference/facefusion-3.7.1/face_quality/pair_01/metadata.json"
        const val CANONICAL_CROP_ASSET = "inputs/pair_01_face_quality_input_512.png"
        const val STAGE_C_FRAME_SHA256 =
            "eeada935b979fd02c34504a777681d618fd062ec3949117fa336c25d2b026afe"
    }
}
