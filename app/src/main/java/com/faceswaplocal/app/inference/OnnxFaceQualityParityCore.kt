package com.faceswaplocal.app.inference

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.TensorInfo
import android.graphics.Bitmap
import java.io.File
import java.io.FileInputStream
import java.nio.FloatBuffer
import java.security.MessageDigest
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

/**
 * Isolated Stage E1 parity kernel for one canonical 512x512 crop.
 *
 * This class deliberately has no dependency on the swapper, photo coordinator,
 * compositor, ModelStore, or UI. It verifies developer-staged private model files,
 * opens one CPU session at a time, and exposes only raw GFPGAN output and BiSeNet
 * class IDs. Product integration is intentionally outside this parity slice.
 */
class OnnxFaceQualityParityCore(
    private val privateModelDirectory: File,
    private val ioDispatcher: CoroutineDispatcher = Dispatchers.IO,
    private val workerDispatcher: CoroutineDispatcher = Dispatchers.Default,
    private val environment: OrtEnvironment = OrtEnvironment.getEnvironment(),
) {
    suspend fun runGfpgan512Cpu(crop512: Bitmap): FloatArray {
        requireCanonicalCrop(crop512)
        val modelFile = verifiedModel(GFPGAN_SPEC)
        return withContext(workerDispatcher) {
            val input = rgbTensor(crop512, GFPGAN_MEAN, GFPGAN_STD)
            createCpuSessionOptions().use { options ->
                environment.createSession(modelFile.absolutePath, options).use { session ->
                    requireTensorContract(
                        session = session,
                        expectedOutputChannels = RGB_CHANNELS,
                        modelName = GFPGAN_SPEC.fileName,
                    )
                    OnnxTensor.createTensor(
                        environment,
                        FloatBuffer.wrap(input),
                        longArrayOf(1, RGB_CHANNELS.toLong(), CROP_SIZE.toLong(), CROP_SIZE.toLong()),
                    ).use { tensor ->
                        session.run(
                            mapOf(INPUT_NAME to tensor),
                            setOf(OUTPUT_NAME),
                        ).use { result ->
                            val output = result.get(OUTPUT_NAME).orElseThrow {
                                IllegalStateException("GFPGAN output '$OUTPUT_NAME' is missing")
                            } as? OnnxTensor ?: error("GFPGAN output is not a tensor")
                            copyFloatOutput(output, RGB_CHANNELS, "GFPGAN")
                        }
                    }
                }
            }
        }
    }

    suspend fun runBisenet512Cpu(crop512: Bitmap): IntArray {
        requireCanonicalCrop(crop512)
        val modelFile = verifiedModel(BISENET_SPEC)
        return withContext(workerDispatcher) {
            val input = rgbTensor(crop512, IMAGENET_MEAN, IMAGENET_STD)
            createCpuSessionOptions().use { options ->
                environment.createSession(modelFile.absolutePath, options).use { session ->
                    requireTensorContract(
                        session = session,
                        expectedOutputChannels = BISENET_CLASS_COUNT,
                        modelName = BISENET_SPEC.fileName,
                    )
                    OnnxTensor.createTensor(
                        environment,
                        FloatBuffer.wrap(input),
                        longArrayOf(1, RGB_CHANNELS.toLong(), CROP_SIZE.toLong(), CROP_SIZE.toLong()),
                    ).use { tensor ->
                        // BiSeNet has three ~19 MiB outputs. Request only the production
                        // main output so auxiliary tensors are never materialized.
                        session.run(
                            mapOf(INPUT_NAME to tensor),
                            setOf(OUTPUT_NAME),
                        ).use { result ->
                            val output = result.get(OUTPUT_NAME).orElseThrow {
                                IllegalStateException("BiSeNet output '$OUTPUT_NAME' is missing")
                            } as? OnnxTensor ?: error("BiSeNet output is not a tensor")
                            argmaxClasses(output)
                        }
                    }
                }
            }
        }
    }

    private suspend fun verifiedModel(spec: PrivateParityModelSpec): File = withContext(ioDispatcher) {
        val canonicalDirectory = privateModelDirectory.canonicalFile
        require(canonicalDirectory.isDirectory) {
            "Private parity model directory is missing: ${canonicalDirectory.name}"
        }
        val modelFile = File(canonicalDirectory, spec.fileName).canonicalFile
        require(modelFile.parentFile == canonicalDirectory) { "Model path escaped the private directory" }
        require(modelFile.isFile) { "Missing private parity model: ${spec.fileName}" }
        require(modelFile.length() == spec.sizeBytes) {
            "Size mismatch for ${spec.fileName}: ${modelFile.length()} != ${spec.sizeBytes}"
        }
        val actualSha256 = sha256(modelFile)
        require(actualSha256 == spec.sha256) {
            "SHA-256 mismatch for ${spec.fileName}: $actualSha256 != ${spec.sha256}"
        }
        modelFile
    }

    private fun requireCanonicalCrop(bitmap: Bitmap) {
        require(bitmap.width == CROP_SIZE && bitmap.height == CROP_SIZE) {
            "Parity input must be exactly ${CROP_SIZE}x$CROP_SIZE"
        }
        val pixels = IntArray(CROP_SIZE * CROP_SIZE)
        bitmap.getPixels(pixels, 0, CROP_SIZE, 0, 0, CROP_SIZE, CROP_SIZE)
        require(pixels.all { pixel -> (pixel ushr ALPHA_SHIFT) == OPAQUE_ALPHA }) {
            "Parity input must be fully opaque"
        }
    }

    private fun rgbTensor(bitmap: Bitmap, mean: FloatArray, std: FloatArray): FloatArray {
        require(mean.size == RGB_CHANNELS && std.size == RGB_CHANNELS)
        require(std.all { it != 0f })
        val pixels = IntArray(CROP_SIZE * CROP_SIZE)
        bitmap.getPixels(pixels, 0, CROP_SIZE, 0, 0, CROP_SIZE, CROP_SIZE)
        val planeSize = pixels.size
        val output = FloatArray(RGB_CHANNELS * planeSize)
        pixels.forEachIndexed { index, pixel ->
            val red = ((pixel ushr RED_SHIFT) and CHANNEL_MASK) / 255f
            val green = ((pixel ushr GREEN_SHIFT) and CHANNEL_MASK) / 255f
            val blue = (pixel and CHANNEL_MASK) / 255f
            output[index] = (red - mean[0]) / std[0]
            output[planeSize + index] = (green - mean[1]) / std[1]
            output[2 * planeSize + index] = (blue - mean[2]) / std[2]
        }
        return output
    }

    private fun createCpuSessionOptions(): OrtSession.SessionOptions =
        OrtSession.SessionOptions().apply {
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
            setInterOpNumThreads(1)
            setIntraOpNumThreads(1)
            addConfigEntry("session.intra_op.allow_spinning", "0")
        }

    private fun requireTensorContract(
        session: OrtSession,
        expectedOutputChannels: Int,
        modelName: String,
    ) {
        val input = requireNotNull(session.inputInfo[INPUT_NAME]) {
            "$modelName input '$INPUT_NAME' is missing"
        }.info as? TensorInfo ?: error("$modelName input is not a tensor")
        require(shapeMatches(input.shape, RGB_CHANNELS)) {
            "Unexpected $modelName input shape: ${input.shape.contentToString()}"
        }
        val output = requireNotNull(session.outputInfo[OUTPUT_NAME]) {
            "$modelName output '$OUTPUT_NAME' is missing"
        }.info as? TensorInfo ?: error("$modelName output is not a tensor")
        require(shapeMatches(output.shape, expectedOutputChannels)) {
            "Unexpected $modelName output shape: ${output.shape.contentToString()}"
        }
    }

    private fun shapeMatches(shape: LongArray, channels: Int): Boolean =
        shape.size == 4 &&
            (shape[0] <= 0L || shape[0] == 1L) &&
            (shape[1] <= 0L || shape[1] == channels.toLong()) &&
            (shape[2] <= 0L || shape[2] == CROP_SIZE.toLong()) &&
            (shape[3] <= 0L || shape[3] == CROP_SIZE.toLong())

    private fun copyFloatOutput(output: OnnxTensor, channels: Int, label: String): FloatArray {
        val shape = (output.info as TensorInfo).shape
        val expectedShape = longArrayOf(1, channels.toLong(), CROP_SIZE.toLong(), CROP_SIZE.toLong())
        require(shape.contentEquals(expectedShape)) {
            "Unexpected $label runtime shape: ${shape.contentToString()}"
        }
        val buffer = output.floatBuffer
        buffer.rewind()
        return FloatArray(buffer.remaining()).also(buffer::get)
    }

    private fun argmaxClasses(output: OnnxTensor): IntArray {
        val shape = (output.info as TensorInfo).shape
        val expectedShape = longArrayOf(1, BISENET_CLASS_COUNT.toLong(), CROP_SIZE.toLong(), CROP_SIZE.toLong())
        require(shape.contentEquals(expectedShape)) {
            "Unexpected BiSeNet runtime shape: ${shape.contentToString()}"
        }
        val logits = output.floatBuffer
        val planeSize = CROP_SIZE * CROP_SIZE
        return IntArray(planeSize) { pixel ->
            var bestClass = 0
            var bestValue = logits.get(pixel)
            for (channel in 1 until BISENET_CLASS_COUNT) {
                val value = logits.get(channel * planeSize + pixel)
                if (value > bestValue) {
                    bestValue = value
                    bestClass = channel
                }
            }
            bestClass
        }
    }

    private fun sha256(file: File): String {
        val digest = MessageDigest.getInstance("SHA-256")
        FileInputStream(file).use { input ->
            val buffer = ByteArray(HASH_BUFFER_SIZE)
            while (true) {
                val count = input.read(buffer)
                if (count == -1) break
                if (count > 0) digest.update(buffer, 0, count)
            }
        }
        return digest.digest().joinToString("") { byte -> "%02x".format(byte) }
    }

    private data class PrivateParityModelSpec(
        val fileName: String,
        val sizeBytes: Long,
        val sha256: String,
    )

    companion object {
        const val CROP_SIZE = 512
        const val BISENET_CLASS_COUNT = 19
        const val CANONICAL_INPUT_SHA256 =
            "5987781f96010ceddbf7445b26bb5420b56e20138d9603a352a48a57f0fb2ec8"
        val REGION_CLASS_IDS = setOf(1, 2, 3, 4, 5, 6, 10, 11, 12, 13)

        private val GFPGAN_SPEC = PrivateParityModelSpec(
            fileName = "gfpgan_1.4.onnx",
            sizeBytes = 340_299_087L,
            sha256 = "accc4757b26bdb89b32b4d3500d4f79c9dff97c1dd7c7104bf9dcb95e3311385",
        )
        private val BISENET_SPEC = PrivateParityModelSpec(
            fileName = "bisenet_resnet_34.onnx",
            sizeBytes = 93_632_546L,
            sha256 = "4a0b8c958a3c938913bd06a8365dbb3c8761afba6ecbf0d14b3b1f77eb230c96",
        )
        private const val INPUT_NAME = "input"
        private const val OUTPUT_NAME = "output"
        private const val RGB_CHANNELS = 3
        private const val HASH_BUFFER_SIZE = 1024 * 1024
        private const val ALPHA_SHIFT = 24
        private const val OPAQUE_ALPHA = 0xff
        private const val RED_SHIFT = 16
        private const val GREEN_SHIFT = 8
        private const val CHANNEL_MASK = 0xff
        private val GFPGAN_MEAN = floatArrayOf(0.5f, 0.5f, 0.5f)
        private val GFPGAN_STD = floatArrayOf(0.5f, 0.5f, 0.5f)
        private val IMAGENET_MEAN = floatArrayOf(0.485f, 0.456f, 0.406f)
        private val IMAGENET_STD = floatArrayOf(0.229f, 0.224f, 0.225f)
    }
}
