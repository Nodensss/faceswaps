package com.faceswaplocal.app.inference

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtException
import ai.onnxruntime.OrtSession
import ai.onnxruntime.TensorInfo
import android.graphics.Bitmap
import android.os.Build
import android.os.SystemClock
import java.io.File
import java.nio.FloatBuffer
import kotlin.coroutines.coroutineContext
import kotlin.math.min
import kotlin.math.roundToInt
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.withContext

data class FaceEnhancementResult(
    val pixels: IntArray,
    val targetToEnhancerCrop: AffineMatrix,
    val roi: CompositeRoi,
    val enhancerBackend: InferenceBackend,
    val parserBackend: InferenceBackend,
    val enhancerMs: Long,
    val parserMs: Long,
    val compositingMs: Long,
)

/**
 * Production Stage-E restoration for one already-swapped target face.
 *
 * The caller supplies pixels accumulated by the swap pass and the original target's
 * five landmarks. The coordinator owns one reusable BiSeNet session for the complete
 * task; this method opens only GFPGAN, so the restoration-pass peak is GFPGAN+BiSeNet.
 * The returned buffer differs from [basePixels] only inside [FaceEnhancementResult.roi].
 */
class OnnxFaceEnhancerPipeline(
    private val modelStore: ModelStore,
    private val workerDispatcher: CoroutineDispatcher = Dispatchers.Default,
    private val environment: OrtEnvironment = OrtEnvironment.getEnvironment(),
    private val sessionLifecycle: InferenceSessionLifecycleListener =
        NoOpInferenceSessionLifecycleListener,
) {
    suspend fun enhance(
        basePixels: IntArray,
        baseWidth: Int,
        baseHeight: Int,
        targetFace: DetectedFace5,
        strength: Float,
        backend: RequestedInferenceBackend,
        parserSession: FaceParserSession,
        protectedBaseRois: List<CompositeRoi> = emptyList(),
        protectedFaceRegions: List<ProtectedFaceRegion> = emptyList(),
    ): FaceEnhancementResult {
        var undeliveredPixels: IntArray? = null
        try {
            val result = withContext(workerDispatcher) {
        require(baseWidth > 0 && baseHeight > 0 && baseWidth <= Int.MAX_VALUE / baseHeight)
        require(basePixels.size == baseWidth * baseHeight) {
            "Base pixel count does not match image dimensions"
        }
        require(strength.isFinite() && strength > 0f && strength <= 1f) {
            "Restoration strength must be in (0, 1]"
        }
        coroutineContext.ensureActive()

        val targetToCrop = FaceGeometry.estimateSimilarity(
            source = targetFace.landmarks,
            template = WarpTemplate.FFHQ_512,
            cropWidth = CROP_SIZE,
            cropHeight = CROP_SIZE,
        )
        val crop = BitmapSampling.warpAffine(
            sourcePixels = basePixels,
            sourceWidth = baseWidth,
            sourceHeight = baseHeight,
            sourceToDestination = targetToCrop,
            destinationWidth = CROP_SIZE,
            destinationHeight = CROP_SIZE,
        )
        var rawOutput: FloatArray? = null
        var restoredPixels: IntArray? = null
        var parserPixels: IntArray? = null
        var regionMask: FloatArray? = null
        var boxMask: FloatArray? = null
        var blendMask: FloatArray? = null
        try {
            val enhancerStarted = elapsedRealtimeMs()
            val enhancerResult = runGfpgan(crop, backend)
            rawOutput = enhancerResult.first
            val enhancerMs = elapsedRealtimeMs() - enhancerStarted
            coroutineContext.ensureActive()

            restoredPixels = gfpganOutputToPixels(enhancerResult.first)
            enhancerResult.first.fill(0f)
            rawOutput = null

            val parserStarted = elapsedRealtimeMs()
            parserPixels = IntArray(CROP_SIZE * CROP_SIZE).also { destination ->
                crop.getPixels(destination, 0, CROP_SIZE, 0, 0, CROP_SIZE, CROP_SIZE)
            }
            val parserResult = parserSession.createRegionMask(
                cropPixels = requireNotNull(parserPixels),
                cropWidth = CROP_SIZE,
                cropHeight = CROP_SIZE,
            )
            val parserMs = elapsedRealtimeMs() - parserStarted
            coroutineContext.ensureActive()

            regionMask = parserResult.mask
            boxMask = FaceCompositor.createBoxMask(CROP_SIZE, CROP_SIZE)
            blendMask = FloatArray(requireNotNull(regionMask).size) { index ->
                min(requireNotNull(boxMask)[index], requireNotNull(regionMask)[index]) * strength
            }

            val compositingStarted = elapsedRealtimeMs()
            val paste = FaceCompositor.pasteBack(
                basePixels = basePixels,
                baseWidth = baseWidth,
                baseHeight = baseHeight,
                cropPixels = requireNotNull(restoredPixels),
                cropMask = requireNotNull(blendMask),
                cropWidth = CROP_SIZE,
                cropHeight = CROP_SIZE,
                baseToCrop = targetToCrop,
                protectedBaseRois = protectedBaseRois,
                protectedFaceRegions = protectedFaceRegions,
            )
            try {
                val compositingMs = elapsedRealtimeMs() - compositingStarted
                FaceEnhancementResult(
                    pixels = paste.pixels,
                    targetToEnhancerCrop = targetToCrop,
                    roi = paste.roi,
                    enhancerBackend = enhancerResult.second,
                    parserBackend = parserResult.backend,
                    enhancerMs = enhancerMs,
                    parserMs = parserMs,
                    compositingMs = compositingMs,
                ).also { undeliveredPixels = it.pixels }
            } finally {
                paste.warpedMask?.fill(0f)
            }
        } finally {
            rawOutput?.fill(0f)
            restoredPixels?.fill(0)
            parserPixels?.fill(0)
            regionMask?.fill(0f)
            boxMask?.fill(0f)
            blendMask?.fill(0f)
            if (!crop.isRecycled) crop.recycle()
        }
            }
            undeliveredPixels = null
            return result
        } finally {
            undeliveredPixels?.fill(0)
        }
    }

    private suspend fun runGfpgan(
        crop: Bitmap,
        backend: RequestedInferenceBackend,
    ): Pair<FloatArray, InferenceBackend> {
        val modelFile = modelStore.requireVerifiedModel(ModelId.GFPGAN_1_4)
        val input = BitmapSampling.rgbTensor(crop, mean = 0.5f, standardDeviation = 0.5f)
        try {
            return runWithBackendFallback(modelFile, backend) { session ->
                requireTensorContract(session, RGB_CHANNELS, "GFPGAN")
                OnnxTensor.createTensor(
                    environment,
                    FloatBuffer.wrap(input),
                    longArrayOf(1, RGB_CHANNELS.toLong(), CROP_SIZE.toLong(), CROP_SIZE.toLong()),
                ).use { tensor ->
                    session.run(mapOf(INPUT_NAME to tensor), setOf(OUTPUT_NAME)).use { result ->
                        val output = result.get(OUTPUT_NAME).orElseThrow {
                            IllegalStateException("GFPGAN output '$OUTPUT_NAME' is missing")
                        } as? OnnxTensor ?: error("GFPGAN output is not a tensor")
                        copyFloatOutput(output, RGB_CHANNELS, "GFPGAN")
                    }
                }
            }
        } finally {
            input.fill(0f)
        }
    }

    private fun gfpganOutputToPixels(output: FloatArray): IntArray {
        val planeSize = CROP_SIZE * CROP_SIZE
        require(output.size == RGB_CHANNELS * planeSize)
        return IntArray(planeSize) { index ->
            fun channel(plane: Int): Int =
                (((output[plane * planeSize + index].coerceIn(-1f, 1f) + 1f) * 0.5f) * 255f)
                    .roundToInt()
                    .coerceIn(0, 255)
            (OPAQUE_ALPHA shl ALPHA_SHIFT) or
                (channel(0) shl RED_SHIFT) or
                (channel(1) shl GREEN_SHIFT) or
                channel(2)
        }
    }

    private fun requireTensorContract(
        session: OrtSession,
        expectedOutputChannels: Int,
        label: String,
    ) {
        val input = requireNotNull(session.inputInfo[INPUT_NAME]) {
            "$label input '$INPUT_NAME' is missing"
        }.info as? TensorInfo ?: error("$label input is not a tensor")
        require(shapeMatches(input.shape, RGB_CHANNELS)) {
            "Unexpected $label input shape: ${input.shape.contentToString()}"
        }
        val output = requireNotNull(session.outputInfo[OUTPUT_NAME]) {
            "$label output '$OUTPUT_NAME' is missing"
        }.info as? TensorInfo ?: error("$label output is not a tensor")
        require(shapeMatches(output.shape, expectedOutputChannels)) {
            "Unexpected $label output shape: ${output.shape.contentToString()}"
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
        val expected = longArrayOf(1, channels.toLong(), CROP_SIZE.toLong(), CROP_SIZE.toLong())
        require(shape.contentEquals(expected)) {
            "Unexpected $label runtime shape: ${shape.contentToString()}"
        }
        val buffer = output.floatBuffer
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
                return runWithSession(modelFile, xnnpack = false, operation) to
                    InferenceBackend.CPU_FALLBACK
            }
        }
        val actualBackend = if (requestedBackend == RequestedInferenceBackend.CPU_ONLY) {
            InferenceBackend.CPU
        } else {
            InferenceBackend.CPU_FALLBACK
        }
        return runWithSession(modelFile, xnnpack = false, operation) to actualBackend
    }

    private fun <T> runWithSession(
        modelFile: File,
        xnnpack: Boolean,
        operation: (OrtSession) -> T,
    ): T {
        createSessionOptions(xnnpack).use { options ->
            val session = environment.createSession(modelFile.absolutePath, options)
            var openedEventSent = false
            try {
                sessionLifecycle.onSessionOpened(modelFile.name)
                openedEventSent = true
                return operation(session)
            } finally {
                try {
                    session.close()
                } finally {
                    if (openedEventSent) {
                        sessionLifecycle.onSessionClosed(modelFile.name)
                    }
                }
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

    private fun elapsedRealtimeMs(): Long = SystemClock.elapsedRealtimeNanos() / 1_000_000L

    private companion object {
        const val CROP_SIZE = 512
        const val RGB_CHANNELS = 3
        const val INPUT_NAME = "input"
        const val OUTPUT_NAME = "output"
        const val MAX_INFERENCE_THREADS = 4
        const val OPAQUE_ALPHA = 0xff
        const val ALPHA_SHIFT = 24
        const val RED_SHIFT = 16
        const val GREEN_SHIFT = 8
    }
}
