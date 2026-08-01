package com.faceswaplocal.app.inference

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OnnxValue
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtException
import ai.onnxruntime.OrtSession
import android.graphics.Bitmap
import android.os.Build
import java.io.File
import java.nio.FloatBuffer
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

/**
 * Stage E1 face-quality inference: GFPGAN 1.4 restoration and BiSeNet face parsing.
 *
 * Both sessions are opened only for their step and closed immediately after (`use`), so
 * neither is held together with the swapper (§5.2). Preprocessing constants are ported
 * from FaceFusion 3.7.1 (`face_enhancer.prepare_crop_frame` and
 * `face_masker.create_region_mask`); parity is proven against the desktop reference on a
 * fixed 512 crop before any orchestration relies on it.
 */
class OnnxFaceEnhancerPipeline(
    private val modelStore: ModelStore,
    private val workerDispatcher: CoroutineDispatcher = Dispatchers.Default,
    private val environment: OrtEnvironment = OrtEnvironment.getEnvironment(),
) {
    /** GFPGAN raw output on an ffhq_512-aligned crop: RGB CHW float in [-1, 1], [3,512,512]. */
    suspend fun enhanceRawOutput(
        crop512: Bitmap,
        backend: RequestedInferenceBackend,
    ): Pair<FloatArray, InferenceBackend> = withContext(workerDispatcher) {
        require(crop512.width == ENHANCER_SIZE && crop512.height == ENHANCER_SIZE) {
            "GFPGAN expects a ${ENHANCER_SIZE}x$ENHANCER_SIZE crop"
        }
        val modelFile = modelStore.requireVerifiedModel(ModelId.GFPGAN_1_4)
        val input = BitmapSampling.rgbTensor(crop512, mean = 0.5f, standardDeviation = 0.5f)
        runWithBackendFallback(modelFile, backend) { session ->
            OnnxTensor.createTensor(
                environment,
                FloatBuffer.wrap(input),
                longArrayOf(1, 3, ENHANCER_SIZE.toLong(), ENHANCER_SIZE.toLong()),
            ).use { tensor ->
                session.run(mapOf(ENHANCER_INPUT_NAME to tensor)).use { result ->
                    copyFloatTensor(result[0]).also { output ->
                        require(output.size == 3 * ENHANCER_SIZE * ENHANCER_SIZE) {
                            "Unexpected GFPGAN output element count: ${output.size}"
                        }
                    }
                }
            }
        }
    }

    /** BiSeNet per-pixel argmax class map at 512x512 (FaceFusion `create_region_mask`). */
    suspend fun parseArgmax(
        crop: Bitmap,
        backend: RequestedInferenceBackend,
    ): Pair<IntArray, InferenceBackend> = withContext(workerDispatcher) {
        val modelFile = modelStore.requireVerifiedModel(ModelId.BISENET_RESNET_34)
        val input = BitmapSampling.rgbTensorResized(crop, PARSER_SIZE, IMAGENET_MEAN, IMAGENET_STD)
        runWithBackendFallback(modelFile, backend) { session ->
            OnnxTensor.createTensor(
                environment,
                FloatBuffer.wrap(input),
                longArrayOf(1, 3, PARSER_SIZE.toLong(), PARSER_SIZE.toLong()),
            ).use { tensor ->
                session.run(mapOf(PARSER_INPUT_NAME to tensor)).use { result ->
                    argmaxClasses(copyFloatTensor(result[0]))
                }
            }
        }
    }

    private fun argmaxClasses(logits: FloatArray): IntArray {
        val planeSize = PARSER_SIZE * PARSER_SIZE
        val channels = logits.size / planeSize
        require(channels * planeSize == logits.size && channels >= 2) {
            "Unexpected BiSeNet output element count: ${logits.size}"
        }
        return IntArray(planeSize) { pixel ->
            var bestClass = 0
            var bestValue = logits[pixel]
            for (channel in 1 until channels) {
                val value = logits[channel * planeSize + pixel]
                if (value > bestValue) {
                    bestValue = value
                    bestClass = channel
                }
            }
            bestClass
        }
    }

    private fun copyFloatTensor(value: OnnxValue): FloatArray {
        val tensor = value as? OnnxTensor ?: error("Expected an ONNX tensor output")
        val buffer = tensor.floatBuffer
        buffer.rewind()
        return FloatArray(buffer.remaining()).also(buffer::get)
    }

    private fun <T> runWithBackendFallback(
        modelFile: File,
        requestedBackend: RequestedInferenceBackend,
        operation: (OrtSession) -> T,
    ): Pair<T, InferenceBackend> {
        if (requestedBackend == RequestedInferenceBackend.XNNPACK_WITH_CPU_FALLBACK &&
            mayAttemptXnnpack(Build.SUPPORTED_ABIS)
        ) {
            try {
                return runWithSession(modelFile, xnnpack = true, operation) to InferenceBackend.XNNPACK
            } catch (_: OrtException) {
                return runWithSession(modelFile, xnnpack = false, operation) to InferenceBackend.CPU_FALLBACK
            }
        }
        val backend = if (requestedBackend == RequestedInferenceBackend.CPU_ONLY) {
            InferenceBackend.CPU
        } else {
            InferenceBackend.CPU_FALLBACK
        }
        return runWithSession(modelFile, xnnpack = false, operation) to backend
    }

    private fun <T> runWithSession(
        modelFile: File,
        xnnpack: Boolean,
        operation: (OrtSession) -> T,
    ): T {
        createSessionOptions(xnnpack).use { options ->
            environment.createSession(modelFile.absolutePath, options).use { session ->
                return operation(session)
            }
        }
    }

    private fun createSessionOptions(xnnpack: Boolean): OrtSession.SessionOptions {
        val options = OrtSession.SessionOptions()
        try {
            options.apply {
                setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
                setInterOpNumThreads(1)
                addConfigEntry("session.intra_op.allow_spinning", "0")
                if (xnnpack) {
                    setIntraOpNumThreads(1)
                    addXnnpack(mapOf("intra_op_num_threads" to inferenceThreadCount().toString()))
                } else {
                    setIntraOpNumThreads(inferenceThreadCount())
                }
            }
            return options
        } catch (error: Throwable) {
            options.close()
            throw error
        }
    }

    private fun inferenceThreadCount(): Int =
        (Runtime.getRuntime().availableProcessors() - 1).coerceIn(1, MAX_INFERENCE_THREADS)

    companion object {
        const val ENHANCER_SIZE = 512
        const val PARSER_SIZE = 512

        /** FaceFusion Stage E1 protected regions: skin, brows, eyes, glasses, nose, mouth, lips. */
        val REGION_CLASS_IDS = intArrayOf(1, 2, 3, 4, 5, 6, 10, 11, 12, 13)

        private const val ENHANCER_INPUT_NAME = "input"
        private const val PARSER_INPUT_NAME = "input"
        private const val MAX_INFERENCE_THREADS = 4
        private val IMAGENET_MEAN = floatArrayOf(0.485f, 0.456f, 0.406f)
        private val IMAGENET_STD = floatArrayOf(0.229f, 0.224f, 0.225f)
    }
}
