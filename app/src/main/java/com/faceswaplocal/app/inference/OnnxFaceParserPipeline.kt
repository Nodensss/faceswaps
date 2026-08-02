package com.faceswaplocal.app.inference

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtException
import ai.onnxruntime.OrtSession
import ai.onnxruntime.TensorInfo
import android.os.Build
import android.os.SystemClock
import java.io.Closeable
import java.io.File
import java.nio.FloatBuffer
import kotlin.math.floor
import kotlin.math.min
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

data class FaceParserMaskResult(
    val mask: FloatArray,
    val backend: InferenceBackend,
    val inferenceMs: Long,
)

/**
 * Owns one reusable BiSeNet session for a complete photo task.
 *
 * The coordinator keeps this session alive across every assigned face in both passes:
 * InSwapper may coexist with it during swap, and GFPGAN may coexist with it during
 * restoration. The opaque [FaceParserSession] prevents ONNX Runtime types from escaping
 * the inference package.
 */
class OnnxFaceParserPipeline(
    private val modelStore: ModelStore,
    private val workerDispatcher: CoroutineDispatcher = Dispatchers.Default,
    private val environment: OrtEnvironment = OrtEnvironment.getEnvironment(),
    private val sessionLifecycle: InferenceSessionLifecycleListener =
        NoOpInferenceSessionLifecycleListener,
) {
    suspend fun <T> withSession(
        backend: RequestedInferenceBackend,
        operation: suspend (FaceParserSession) -> T,
    ): T {
        val modelFile = modelStore.requireVerifiedModel(ModelId.BISENET_RESNET_34)
        return withContext(workerDispatcher) {
            val session = createParserSession(modelFile, backend)
            try {
                operation(session)
            } finally {
                session.close()
            }
        }
    }

    private fun createParserSession(
        modelFile: File,
        requestedBackend: RequestedInferenceBackend,
    ): FaceParserSession {
        val mayUseXnnpack =
            requestedBackend == RequestedInferenceBackend.XNNPACK_WITH_CPU_FALLBACK &&
                mayAttemptXnnpack(Build.SUPPORTED_ABIS)
        val initial = if (mayUseXnnpack) {
            try {
                openHandle(modelFile, ParserSessionKind.XNNPACK, InferenceBackend.XNNPACK)
            } catch (_: OrtException) {
                openHandle(modelFile, ParserSessionKind.CPU, InferenceBackend.CPU_FALLBACK)
            }
        } else {
            val actualBackend = if (requestedBackend == RequestedInferenceBackend.CPU_ONLY) {
                InferenceBackend.CPU
            } else {
                InferenceBackend.CPU_FALLBACK
            }
            openHandle(modelFile, ParserSessionKind.CPU, actualBackend)
        }
        val fallbackFactory = if (initial.backend == InferenceBackend.XNNPACK) {
            {
                openHandle(
                    modelFile,
                    ParserSessionKind.CPU,
                    InferenceBackend.CPU_FALLBACK,
                )
            }
        } else {
            null
        }
        return FaceParserSession(initial, fallbackFactory)
    }

    private fun openHandle(
        modelFile: File,
        kind: ParserSessionKind,
        backend: InferenceBackend,
    ): ParserSessionHandle {
        val options = createSessionOptions(kind)
        var session: OrtSession? = null
        var openedEventSent = false
        try {
            session = environment.createSession(modelFile.absolutePath, options)
            sessionLifecycle.onSessionOpened(modelFile.name)
            openedEventSent = true
            return ParserSessionHandle(
                session = session,
                options = options,
                backend = backend,
                modelFileName = modelFile.name,
                sessionLifecycle = sessionLifecycle,
            )
        } catch (error: Throwable) {
            try {
                session?.close()
            } finally {
                if (openedEventSent) sessionLifecycle.onSessionClosed(modelFile.name)
                options.close()
            }
            throw error
        }
    }

    private fun createSessionOptions(kind: ParserSessionKind): OrtSession.SessionOptions {
        val options = OrtSession.SessionOptions()
        try {
            options.apply {
                setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
                setInterOpNumThreads(1)
                addConfigEntry("session.intra_op.allow_spinning", "0")
                when (kind) {
                    ParserSessionKind.XNNPACK -> {
                        setIntraOpNumThreads(1)
                        addXnnpack(
                            mapOf(
                                "intra_op_num_threads" to inferenceThreadCount().toString(),
                            ),
                        )
                    }

                    ParserSessionKind.CPU -> setIntraOpNumThreads(inferenceThreadCount())
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

    private enum class ParserSessionKind {
        XNNPACK,
        CPU,
    }

    private companion object {
        const val MAX_INFERENCE_THREADS = 4
    }
}

/** Public only as an opaque inference-package capability; it exposes no ORT object. */
class FaceParserSession internal constructor(
    initialHandle: ParserSessionHandle,
    private val cpuFallbackFactory: (() -> ParserSessionHandle)?,
) : Closeable {
    private var handle: ParserSessionHandle? = initialHandle

    @Synchronized
    internal fun createRegionMask(
        cropPixels: IntArray,
        cropWidth: Int,
        cropHeight: Int,
    ): FaceParserMaskResult {
        require(cropWidth > 0 && cropHeight > 0 && cropWidth <= Int.MAX_VALUE / cropHeight)
        require(cropPixels.size == cropWidth * cropHeight) {
            "Parser crop pixel count does not match its dimensions"
        }
        val input = FaceRegionMask.rgbTensor(cropPixels, cropWidth, cropHeight)
        var classes: IntArray? = null
        try {
            val started = elapsedRealtimeMs()
            classes = try {
                runParser(requireOpenHandle(), input)
            } catch (error: OrtException) {
                val active = requireOpenHandle()
                val fallback = cpuFallbackFactory
                if (active.backend != InferenceBackend.XNNPACK || fallback == null) throw error
                active.close()
                handle = fallback()
                runParser(requireOpenHandle(), input)
            }
            val inferenceMs = elapsedRealtimeMs() - started
            val activeBackend = requireOpenHandle().backend
            val mask = FaceRegionMask.fromClasses(
                classes = requireNotNull(classes),
                classWidth = MODEL_SIZE,
                classHeight = MODEL_SIZE,
                outputWidth = cropWidth,
                outputHeight = cropHeight,
            )
            return FaceParserMaskResult(mask, activeBackend, inferenceMs)
        } finally {
            input.fill(0f)
            classes?.fill(0)
        }
    }

    @Synchronized
    override fun close() {
        val active = handle ?: return
        handle = null
        active.close()
    }

    private fun requireOpenHandle(): ParserSessionHandle =
        checkNotNull(handle) { "Face parser session is closed" }

    private fun runParser(active: ParserSessionHandle, input: FloatArray): IntArray {
        requireTensorContract(active.session)
        OnnxTensor.createTensor(
            OrtEnvironment.getEnvironment(),
            FloatBuffer.wrap(input),
            longArrayOf(1, RGB_CHANNELS.toLong(), MODEL_SIZE.toLong(), MODEL_SIZE.toLong()),
        ).use { tensor ->
            active.session.run(mapOf(INPUT_NAME to tensor), setOf(OUTPUT_NAME)).use { result ->
                val output = result.get(OUTPUT_NAME).orElseThrow {
                    IllegalStateException("BiSeNet output '$OUTPUT_NAME' is missing")
                } as? OnnxTensor ?: error("BiSeNet output is not a tensor")
                return argmaxClasses(output)
            }
        }
    }

    private fun requireTensorContract(session: OrtSession) {
        val input = requireNotNull(session.inputInfo[INPUT_NAME]) {
            "BiSeNet input '$INPUT_NAME' is missing"
        }.info as? TensorInfo ?: error("BiSeNet input is not a tensor")
        require(shapeMatches(input.shape, RGB_CHANNELS)) {
            "Unexpected BiSeNet input shape: ${input.shape.contentToString()}"
        }
        val output = requireNotNull(session.outputInfo[OUTPUT_NAME]) {
            "BiSeNet output '$OUTPUT_NAME' is missing"
        }.info as? TensorInfo ?: error("BiSeNet output is not a tensor")
        require(shapeMatches(output.shape, BISENET_CLASS_COUNT)) {
            "Unexpected BiSeNet output shape: ${output.shape.contentToString()}"
        }
    }

    private fun shapeMatches(shape: LongArray, channels: Int): Boolean =
        shape.size == 4 &&
            (shape[0] <= 0L || shape[0] == 1L) &&
            (shape[1] <= 0L || shape[1] == channels.toLong()) &&
            (shape[2] <= 0L || shape[2] == MODEL_SIZE.toLong()) &&
            (shape[3] <= 0L || shape[3] == MODEL_SIZE.toLong())

    private fun argmaxClasses(output: OnnxTensor): IntArray {
        val shape = (output.info as TensorInfo).shape
        val expected = longArrayOf(
            1,
            BISENET_CLASS_COUNT.toLong(),
            MODEL_SIZE.toLong(),
            MODEL_SIZE.toLong(),
        )
        require(shape.contentEquals(expected)) {
            "Unexpected BiSeNet runtime shape: ${shape.contentToString()}"
        }
        val logits = output.floatBuffer
        val planeSize = MODEL_SIZE * MODEL_SIZE
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

    private fun elapsedRealtimeMs(): Long = SystemClock.elapsedRealtimeNanos() / 1_000_000L

    private companion object {
        const val MODEL_SIZE = 512
        const val BISENET_CLASS_COUNT = 19
        const val RGB_CHANNELS = 3
        const val INPUT_NAME = "input"
        const val OUTPUT_NAME = "output"
    }
}

internal class ParserSessionHandle(
    val session: OrtSession,
    private val options: OrtSession.SessionOptions,
    val backend: InferenceBackend,
    private val modelFileName: String,
    private val sessionLifecycle: InferenceSessionLifecycleListener,
) : Closeable {
    private var closed = false

    @Synchronized
    override fun close() {
        if (closed) return
        closed = true
        try {
            session.close()
        } finally {
            try {
                sessionLifecycle.onSessionClosed(modelFileName)
            } finally {
                options.close()
            }
        }
    }
}

/** FaceFusion 3.7.1 region-mask preprocessing and postprocessing. */
internal object FaceRegionMask {
    fun rgbTensor(cropPixels: IntArray, cropWidth: Int, cropHeight: Int): FloatArray {
        val planeSize = MODEL_SIZE * MODEL_SIZE
        val output = FloatArray(RGB_CHANNELS * planeSize)
        for (destinationY in 0 until MODEL_SIZE) {
            val sourceY = (destinationY + 0.5) * cropHeight / MODEL_SIZE - 0.5
            for (destinationX in 0 until MODEL_SIZE) {
                val sourceX = (destinationX + 0.5) * cropWidth / MODEL_SIZE - 0.5
                val index = destinationY * MODEL_SIZE + destinationX
                output[index] = (sampleChannel(
                    cropPixels,
                    cropWidth,
                    cropHeight,
                    sourceX,
                    sourceY,
                    RED_SHIFT,
                ) / 255f - IMAGENET_MEAN[0]) / IMAGENET_STD[0]
                output[planeSize + index] = (sampleChannel(
                    cropPixels,
                    cropWidth,
                    cropHeight,
                    sourceX,
                    sourceY,
                    GREEN_SHIFT,
                ) / 255f - IMAGENET_MEAN[1]) / IMAGENET_STD[1]
                output[2 * planeSize + index] = (sampleChannel(
                    cropPixels,
                    cropWidth,
                    cropHeight,
                    sourceX,
                    sourceY,
                    BLUE_SHIFT,
                ) / 255f - IMAGENET_MEAN[2]) / IMAGENET_STD[2]
            }
        }
        return output
    }

    fun fromClasses(
        classes: IntArray,
        classWidth: Int,
        classHeight: Int,
        outputWidth: Int,
        outputHeight: Int,
    ): FloatArray {
        require(classWidth > 0 && classHeight > 0 && classWidth <= Int.MAX_VALUE / classHeight)
        require(classes.size == classWidth * classHeight)
        require(outputWidth > 0 && outputHeight > 0 && outputWidth <= Int.MAX_VALUE / outputHeight)
        val selected = FloatArray(classes.size) { index ->
            if (classes[index] in REGION_CLASS_IDS) 1f else 0f
        }
        var resized: FloatArray? = null
        var blurred: FloatArray? = null
        try {
            resized = resizeMaskBilinear(
                selected,
                classWidth,
                classHeight,
                outputWidth,
                outputHeight,
            )
            blurred = FaceCompositor.gaussianBlurReflect101(
                input = requireNotNull(resized),
                width = outputWidth,
                height = outputHeight,
                sigma = REGION_MASK_SIGMA,
            )
            return FloatArray(requireNotNull(blurred).size) { index ->
                ((requireNotNull(blurred)[index].coerceIn(0.5f, 1f) - 0.5f) * 2f)
                    .coerceIn(0f, 1f)
            }
        } finally {
            selected.fill(0f)
            resized?.fill(0f)
            blurred?.fill(0f)
        }
    }

    private fun resizeMaskBilinear(
        source: FloatArray,
        sourceWidth: Int,
        sourceHeight: Int,
        destinationWidth: Int,
        destinationHeight: Int,
    ): FloatArray {
        if (sourceWidth == destinationWidth && sourceHeight == destinationHeight) {
            return source.copyOf()
        }
        return FloatArray(destinationWidth * destinationHeight).also { destination ->
            for (y in 0 until destinationHeight) {
                val sourceY = ((y + 0.5) * sourceHeight / destinationHeight - 0.5)
                    .coerceIn(0.0, (sourceHeight - 1).toDouble())
                val y0 = floor(sourceY).toInt()
                val y1 = min(y0 + 1, sourceHeight - 1)
                val fractionY = sourceY - y0
                for (x in 0 until destinationWidth) {
                    val sourceX = ((x + 0.5) * sourceWidth / destinationWidth - 0.5)
                        .coerceIn(0.0, (sourceWidth - 1).toDouble())
                    val x0 = floor(sourceX).toInt()
                    val x1 = min(x0 + 1, sourceWidth - 1)
                    val fractionX = sourceX - x0
                    val top = source[y0 * sourceWidth + x0] * (1.0 - fractionX) +
                        source[y0 * sourceWidth + x1] * fractionX
                    val bottom = source[y1 * sourceWidth + x0] * (1.0 - fractionX) +
                        source[y1 * sourceWidth + x1] * fractionX
                    destination[y * destinationWidth + x] =
                        (top * (1.0 - fractionY) + bottom * fractionY).toFloat()
                }
            }
        }
    }

    private fun sampleChannel(
        pixels: IntArray,
        width: Int,
        height: Int,
        x: Double,
        y: Double,
        shift: Int,
    ): Float {
        val boundedX = x.coerceIn(0.0, (width - 1).toDouble())
        val boundedY = y.coerceIn(0.0, (height - 1).toDouble())
        val x0 = floor(boundedX).toInt()
        val y0 = floor(boundedY).toInt()
        val x1 = min(x0 + 1, width - 1)
        val y1 = min(y0 + 1, height - 1)
        val fractionX = boundedX - x0
        val fractionY = boundedY - y0
        fun value(sampleX: Int, sampleY: Int): Double =
            ((pixels[sampleY * width + sampleX] ushr shift) and CHANNEL_MASK).toDouble()
        val top = value(x0, y0) * (1.0 - fractionX) + value(x1, y0) * fractionX
        val bottom = value(x0, y1) * (1.0 - fractionX) + value(x1, y1) * fractionX
        return (top * (1.0 - fractionY) + bottom * fractionY).toFloat()
    }

    private const val MODEL_SIZE = 512
    private const val RGB_CHANNELS = 3
    private const val REGION_MASK_SIGMA = 5.0
    private const val RED_SHIFT = 16
    private const val GREEN_SHIFT = 8
    private const val BLUE_SHIFT = 0
    private const val CHANNEL_MASK = 0xff
    private val REGION_CLASS_IDS = setOf(1, 2, 3, 4, 5, 6, 10, 11, 12, 13)
    private val IMAGENET_MEAN = floatArrayOf(0.485f, 0.456f, 0.406f)
    private val IMAGENET_STD = floatArrayOf(0.229f, 0.224f, 0.225f)
}
