package com.faceswaplocal.app.inference

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.SystemClock
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import java.io.ByteArrayInputStream
import java.io.File
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.security.MessageDigest
import kotlin.math.abs
import kotlin.math.max
import kotlinx.coroutines.runBlocking
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

/**
 * Numeric E1 parity gate on one byte-identical desktop FaceFusion 512 crop.
 * No detector, alignment, swapper, compositor, coordinator, picker, or UI runs here.
 */
@RunWith(AndroidJUnit4::class)
class FaceQualityParityInstrumentedTest {
    private val context = ApplicationProvider.getApplicationContext<Context>()
    private val assets = InstrumentationRegistry.getInstrumentation().context.assets

    @Test
    fun gfpganAndBisenetMatchDesktopOnCanonicalCrop() = runBlocking {
        val inputBytes = readAsset(CANONICAL_INPUT_ASSET)
        assertEquals(
            "Canonical input bytes changed",
            OnnxFaceQualityParityCore.CANONICAL_INPUT_SHA256,
            sha256(inputBytes),
        )
        val crop = decodeOpaqueArgb(inputBytes)
        val privateModels = File(context.filesDir, MODEL_DIRECTORY_NAME).canonicalFile
        val privateRoot = context.filesDir.canonicalFile
        assertTrue(
            "Parity weights must be staged below app-private filesDir",
            privateModels.path.startsWith(privateRoot.path + File.separator),
        )
        val outputDirectory = cleanOutputDirectory(privateRoot)
        val core = OnnxFaceQualityParityCore(privateModels)

        try {
            val gfpganStarted = SystemClock.elapsedRealtime()
            val actualGfpgan = core.runGfpgan512Cpu(crop)
            val gfpganMs = SystemClock.elapsedRealtime() - gfpganStarted
            val expectedGfpgan = readFloatAsset(GFPGAN_REFERENCE_ASSET)
            assertEquals(expectedGfpgan.size, actualGfpgan.size)
            val gfpganSsim = structuralSimilarity(
                actual = linearUnit(actualGfpgan),
                expected = linearUnit(expectedGfpgan),
                width = OnnxFaceQualityParityCore.CROP_SIZE,
                height = OnnxFaceQualityParityCore.CROP_SIZE,
            )
            val gfpganMae = meanAbsoluteError(actualGfpgan, expectedGfpgan)
            val gfpganMaxError = maxAbsoluteError(actualGfpgan, expectedGfpgan)
            writeGfpganPng(actualGfpgan, File(outputDirectory, "android_gfpgan_output.png"))

            val bisenetStarted = SystemClock.elapsedRealtime()
            val actualClasses = core.runBisenet512Cpu(crop)
            val bisenetMs = SystemClock.elapsedRealtime() - bisenetStarted
            val expectedClasses = readUnsignedByteAsset(BISENET_REFERENCE_ASSET)
            assertEquals(expectedClasses.size, actualClasses.size)
            val classAgreement = classAgreement(actualClasses, expectedClasses)
            val regionIou = protectedRegionIou(actualClasses, expectedClasses)
            writeRegionMaskPng(actualClasses, File(outputDirectory, "android_bisenet_region_mask.png"))

            val metrics = JSONObject()
                .put("canonical_input_sha256", sha256(inputBytes))
                .put("backend", "CPU")
                .put("gfpgan_raw_ssim", gfpganSsim)
                .put("gfpgan_raw_mae", gfpganMae)
                .put("gfpgan_raw_max_abs_error", gfpganMaxError)
                .put("gfpgan_ms", gfpganMs)
                .put("bisenet_class_agreement", classAgreement)
                .put("bisenet_protected_region_iou", regionIou)
                .put("bisenet_ms", bisenetMs)
            File(outputDirectory, "face_quality_parity_results.json")
                .writeText(metrics.toString(2) + "\n")

            assertTrue("GFPGAN raw SSIM $gfpganSsim must be >= 0.95", gfpganSsim >= 0.95)
            assertTrue(
                "BiSeNet class agreement $classAgreement must be >= 0.95",
                classAgreement >= 0.95,
            )
            assertTrue("BiSeNet protected-region IoU $regionIou must be >= 0.95", regionIou >= 0.95)
        } finally {
            crop.recycle()
        }
    }

    private fun cleanOutputDirectory(privateRoot: File): File {
        val outputDirectory = File(privateRoot, OUTPUT_DIRECTORY_NAME).canonicalFile
        require(outputDirectory.parentFile == privateRoot) { "Parity output escaped filesDir" }
        if (outputDirectory.isDirectory) {
            outputDirectory.listFiles().orEmpty().forEach { file ->
                require(file.isFile && file.delete()) { "Could not clean ${file.name}" }
            }
        }
        require(outputDirectory.isDirectory || outputDirectory.mkdirs()) {
            "Could not create parity output directory"
        }
        return outputDirectory
    }

    private fun decodeOpaqueArgb(bytes: ByteArray): Bitmap {
        val decoded = ByteArrayInputStream(bytes).use { input ->
            requireNotNull(BitmapFactory.decodeStream(input)) { "Could not decode canonical crop" }
        }
        val bitmap = decoded.copy(Bitmap.Config.ARGB_8888, false)
        if (bitmap !== decoded) decoded.recycle()
        assertEquals(OnnxFaceQualityParityCore.CROP_SIZE, bitmap.width)
        assertEquals(OnnxFaceQualityParityCore.CROP_SIZE, bitmap.height)
        return bitmap
    }

    private fun readAsset(path: String): ByteArray = assets.open(path).use { it.readBytes() }

    private fun readFloatAsset(path: String): FloatArray {
        val bytes = readAsset(path)
        require(bytes.size % Float.SIZE_BYTES == 0)
        val floats = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
        return FloatArray(floats.remaining()).also(floats::get)
    }

    private fun readUnsignedByteAsset(path: String): IntArray {
        val bytes = readAsset(path)
        return IntArray(bytes.size) { index -> bytes[index].toInt() and 0xff }
    }

    /** Keep raw tails (for example desktop min -1.012) visible to SSIM; do not clamp. */
    private fun linearUnit(values: FloatArray): FloatArray =
        FloatArray(values.size) { index -> (values[index] + 1f) / 2f }

    private fun meanAbsoluteError(actual: FloatArray, expected: FloatArray): Double {
        require(actual.size == expected.size)
        var total = 0.0
        for (index in actual.indices) total += abs(actual[index] - expected[index]).toDouble()
        return total / actual.size
    }

    private fun maxAbsoluteError(actual: FloatArray, expected: FloatArray): Double {
        require(actual.size == expected.size)
        var result = 0.0
        for (index in actual.indices) {
            result = max(result, abs(actual[index] - expected[index]).toDouble())
        }
        return result
    }

    private fun classAgreement(actual: IntArray, expected: IntArray): Double {
        var equal = 0
        for (index in actual.indices) if (actual[index] == expected[index]) equal++
        return equal.toDouble() / actual.size
    }

    private fun protectedRegionIou(actual: IntArray, expected: IntArray): Double {
        var intersection = 0
        var union = 0
        for (index in actual.indices) {
            val actualRegion = actual[index] in OnnxFaceQualityParityCore.REGION_CLASS_IDS
            val expectedRegion = expected[index] in OnnxFaceQualityParityCore.REGION_CLASS_IDS
            if (actualRegion || expectedRegion) union++
            if (actualRegion && expectedRegion) intersection++
        }
        return if (union == 0) 1.0 else intersection.toDouble() / union
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

    private fun writeGfpganPng(raw: FloatArray, destination: File) {
        val planeSize = OnnxFaceQualityParityCore.CROP_SIZE * OnnxFaceQualityParityCore.CROP_SIZE
        val pixels = IntArray(planeSize) { index ->
            val red = toByte(raw[index])
            val green = toByte(raw[planeSize + index])
            val blue = toByte(raw[2 * planeSize + index])
            (0xff shl 24) or (red shl 16) or (green shl 8) or blue
        }
        writeBitmap(
            Bitmap.createBitmap(
                pixels,
                OnnxFaceQualityParityCore.CROP_SIZE,
                OnnxFaceQualityParityCore.CROP_SIZE,
                Bitmap.Config.ARGB_8888,
            ),
            destination,
        )
    }

    private fun writeRegionMaskPng(classes: IntArray, destination: File) {
        val pixels = IntArray(classes.size) { index ->
            if (classes[index] in OnnxFaceQualityParityCore.REGION_CLASS_IDS) -0x1 else -0x1000000
        }
        writeBitmap(
            Bitmap.createBitmap(
                pixels,
                OnnxFaceQualityParityCore.CROP_SIZE,
                OnnxFaceQualityParityCore.CROP_SIZE,
                Bitmap.Config.ARGB_8888,
            ),
            destination,
        )
    }

    private fun writeBitmap(bitmap: Bitmap, destination: File) {
        try {
            FileOutputStream(destination).use { output ->
                require(bitmap.compress(Bitmap.CompressFormat.PNG, 100, output)) {
                    "Could not encode ${destination.name}"
                }
            }
        } finally {
            bitmap.recycle()
        }
    }

    private fun toByte(value: Float): Int =
        (((value.coerceIn(-1f, 1f) + 1f) / 2f) * 255f).toInt().coerceIn(0, 255)

    private fun sha256(bytes: ByteArray): String =
        MessageDigest.getInstance("SHA-256").digest(bytes)
            .joinToString("") { byte -> "%02x".format(byte) }

    private companion object {
        const val MODEL_DIRECTORY_NAME = "models"
        const val OUTPUT_DIRECTORY_NAME = "stage-e1-parity-output"
        const val CANONICAL_INPUT_ASSET = "inputs/pair_01_face_quality_input_512.png"
        const val GFPGAN_REFERENCE_ASSET =
            "reference/facefusion-3.7.1/face_quality/pair_01/gfpgan_output_f32le.bin"
        const val BISENET_REFERENCE_ASSET =
            "reference/facefusion-3.7.1/face_quality/pair_01/bisenet_argmax_u8.bin"
    }
}
