package com.faceswaplocal.app.inference

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtException
import ai.onnxruntime.OrtSession
import android.graphics.Bitmap
import android.os.Build
import android.os.SystemClock
import java.io.File
import java.nio.FloatBuffer
import kotlin.coroutines.coroutineContext
import kotlin.math.floor
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.withContext

enum class SwapperModel(
    val modelId: ModelId,
    val cropSize: Int,
) {
    HYPERSWAP_1A_256(ModelId.HYPERSWAP_1A_256, 256),
    INSWAPPER_128_FP16(ModelId.INSWAPPER_128_FP16, 128),
}

enum class RequestedInferenceBackend {
    XNNPACK_WITH_CPU_FALLBACK,
    CPU_ONLY,
}

enum class InferenceBackend {
    XNNPACK,
    CPU,
    CPU_FALLBACK,
}

data class FaceBox(
    val left: Double,
    val top: Double,
    val right: Double,
    val bottom: Double,
)

data class DetectedFace5(
    val score: Float,
    val box: FaceBox,
    val landmarks: List<Point2>,
)

data class RawFaceSwapRequest(
    val source: Bitmap,
    val target: Bitmap,
    val swapper: SwapperModel = SwapperModel.HYPERSWAP_1A_256,
    val backend: RequestedInferenceBackend = RequestedInferenceBackend.CPU_ONLY,
    val sourceFaceHint: FaceBox? = null,
    val targetFaceHint: FaceBox? = null,
)

data class RawFaceSwapTimings(
    val detectorMs: Long,
    val recognizerMs: Long,
    val swapperMs: Long,
    val totalMs: Long,
)

/**
 * Stage B output. The square [rawOutputBitmap] is deliberately not pasted into the
 * target photo; inverse transform, masks, colour matching, and blending belong to Stage C.
 */
data class RawFaceSwapResult(
    val swapper: SwapperModel,
    val sourceFace: DetectedFace5,
    val targetFace: DetectedFace5,
    val sourceToRecognizerCrop: AffineMatrix,
    val targetToSwapperCrop: AffineMatrix,
    val alignedSource112: Bitmap,
    val alignedTarget: Bitmap,
    val rawOutput: FloatArray,
    val rawMask: FloatArray?,
    val rawOutputBitmap: Bitmap,
    val detectorBackend: InferenceBackend,
    val recognizerBackend: InferenceBackend,
    val swapperBackend: InferenceBackend,
    val timings: RawFaceSwapTimings,
)

class NoNeuralFaceFoundException(
    imageRole: String,
) : IllegalArgumentException("The 5-point neural detector found no face in the $imageRole image")

/**
 * FaceFusion 3.7.1-compatible Stage B pipeline:
 * yoloface_8n -> arcface_w600k_r50 -> HyperSwap/InSwapper raw crop.
 *
 * Every heavyweight session is opened only for its pipeline step and closed before the
 * next one. ML Kit landmarks are never accepted by this class.
 */
class OnnxRawFaceSwapPipeline(
    private val modelStore: ModelStore,
    private val workerDispatcher: CoroutineDispatcher = Dispatchers.Default,
    private val environment: OrtEnvironment = OrtEnvironment.getEnvironment(),
) {
    suspend fun process(request: RawFaceSwapRequest): RawFaceSwapResult =
        withContext(workerDispatcher) {
            val totalStarted = elapsedRealtimeMs()
            coroutineContext.ensureActive()

            val detectorStarted = elapsedRealtimeMs()
            val detectorFile = modelStore.requireVerifiedModel(ModelId.YOLOFACE_8N)
            val (detectedPair, detectorBackend) = runWithBackendFallback(
                modelFile = detectorFile,
                requestedBackend = request.backend,
            ) { session ->
                val sourceFaces = detect(session, request.source)
                coroutineContext.ensureActive()
                val targetFaces = detect(session, request.target)
                sourceFaces to targetFaces
            }
            val sourceFace = selectDetectedFace(detectedPair.first, request.sourceFaceHint)
                ?: throw NoNeuralFaceFoundException("source")
            val targetFace = selectDetectedFace(detectedPair.second, request.targetFaceHint)
                ?: throw NoNeuralFaceFoundException("target")
            val detectorMs = elapsedRealtimeMs() - detectorStarted

            coroutineContext.ensureActive()
            val sourceMatrix = FaceGeometry.estimateSimilarity(
                source = sourceFace.landmarks,
                template = WarpTemplate.ARCFACE_112_V2,
                cropWidth = RECOGNIZER_SIZE,
                cropHeight = RECOGNIZER_SIZE,
            )
            val alignedSource = BitmapSampling.warpAffine(
                source = request.source,
                sourceToDestination = sourceMatrix,
                destinationWidth = RECOGNIZER_SIZE,
                destinationHeight = RECOGNIZER_SIZE,
            )
            var alignedTargetForCleanup: Bitmap? = null
            var ownershipTransferred = false
            try {
                val recognizerStarted = elapsedRealtimeMs()
                val recognizerFile = modelStore.requireVerifiedModel(ModelId.ARCFACE_W600K_R50)
                val (embedding, recognizerBackend) = runWithBackendFallback(
                    modelFile = recognizerFile,
                    requestedBackend = request.backend,
                ) { session -> recognize(session, alignedSource) }
                val recognizerMs = elapsedRealtimeMs() - recognizerStarted

                coroutineContext.ensureActive()
                val targetMatrix = FaceGeometry.estimateSimilarity(
                    source = targetFace.landmarks,
                    template = WarpTemplate.ARCFACE_128,
                    cropWidth = request.swapper.cropSize,
                    cropHeight = request.swapper.cropSize,
                )
                val alignedTarget = BitmapSampling.warpAffine(
                    source = request.target,
                    sourceToDestination = targetMatrix,
                    destinationWidth = request.swapper.cropSize,
                    destinationHeight = request.swapper.cropSize,
                )
                alignedTargetForCleanup = alignedTarget

                val swapperStarted = elapsedRealtimeMs()
                val swapperFile = modelStore.requireVerifiedModel(request.swapper.modelId)
                val sourceEmbedding = when (request.swapper) {
                    SwapperModel.HYPERSWAP_1A_256 -> normalizeL2(embedding)
                    SwapperModel.INSWAPPER_128_FP16 -> {
                        val emap = OnnxInitializerReader.readFloatTensor(
                            modelFile = swapperFile,
                            initializerName = INSWAPPER_EMAP_NAME,
                            expectedDimensions = longArrayOf(EMBEDDING_SIZE.toLong(), EMBEDDING_SIZE.toLong()),
                        )
                        convertInSwapperEmbedding(embedding, emap)
                    }
                }
                coroutineContext.ensureActive()
                val (swapperOutput, swapperBackend) = runWithBackendFallback(
                    modelFile = swapperFile,
                    requestedBackend = request.backend,
                ) { session -> runSwapper(session, request.swapper, sourceEmbedding, alignedTarget) }
                val swapperMs = elapsedRealtimeMs() - swapperStarted

                val rawBitmap = rawOutputToBitmap(swapperOutput.output, request.swapper)
                RawFaceSwapResult(
                    swapper = request.swapper,
                    sourceFace = sourceFace,
                    targetFace = targetFace,
                    sourceToRecognizerCrop = sourceMatrix,
                    targetToSwapperCrop = targetMatrix,
                    alignedSource112 = alignedSource,
                    alignedTarget = alignedTarget,
                    rawOutput = swapperOutput.output,
                    rawMask = swapperOutput.mask,
                    rawOutputBitmap = rawBitmap,
                    detectorBackend = detectorBackend,
                    recognizerBackend = recognizerBackend,
                    swapperBackend = swapperBackend,
                    timings = RawFaceSwapTimings(
                        detectorMs = detectorMs,
                        recognizerMs = recognizerMs,
                        swapperMs = swapperMs,
                        totalMs = elapsedRealtimeMs() - totalStarted,
                    ),
                ).also { ownershipTransferred = true }
            } finally {
                if (!ownershipTransferred) {
                    alignedSource.recycle()
                    alignedTargetForCleanup?.recycle()
                }
            }
        }

    private fun detect(session: OrtSession, bitmap: Bitmap): List<DetectedFace5> {
        val detectorInput = BitmapSampling.detectorInput(bitmap, DETECTOR_SIZE)
        OnnxTensor.createTensor(
            environment,
            FloatBuffer.wrap(detectorInput),
            longArrayOf(1, 3, DETECTOR_SIZE.toLong(), DETECTOR_SIZE.toLong()),
        ).use { input ->
            session.run(mapOf(DETECTOR_INPUT_NAME to input)).use { result ->
                val output = copyFloatTensor(result[0])
                require(output.size == DETECTOR_CHANNELS * DETECTOR_CANDIDATES) {
                    "Unexpected yoloface output element count: ${output.size}"
                }
                return decodeYoloFace(output, bitmap.width, bitmap.height)
            }
        }
    }

    private fun recognize(session: OrtSession, alignedSource: Bitmap): FloatArray {
        val inputData = BitmapSampling.rgbTensor(alignedSource, mean = 0.5f, standardDeviation = 0.5f)
        OnnxTensor.createTensor(
            environment,
            FloatBuffer.wrap(inputData),
            longArrayOf(1, 3, RECOGNIZER_SIZE.toLong(), RECOGNIZER_SIZE.toLong()),
        ).use { input ->
            session.run(mapOf(RECOGNIZER_INPUT_NAME to input)).use { result ->
                return copyFloatTensor(result[0]).also { embedding ->
                    require(embedding.size == EMBEDDING_SIZE) {
                        "Unexpected ArcFace embedding size: ${embedding.size}"
                    }
                }
            }
        }
    }

    private fun runSwapper(
        session: OrtSession,
        swapper: SwapperModel,
        sourceEmbedding: FloatArray,
        alignedTarget: Bitmap,
    ): SwapperOutput {
        val (mean, standardDeviation) = when (swapper) {
            SwapperModel.HYPERSWAP_1A_256 -> 0.5f to 0.5f
            SwapperModel.INSWAPPER_128_FP16 -> 0f to 1f
        }
        val targetData = BitmapSampling.rgbTensor(alignedTarget, mean, standardDeviation)
        OnnxTensor.createTensor(
            environment,
            FloatBuffer.wrap(sourceEmbedding),
            longArrayOf(1, EMBEDDING_SIZE.toLong()),
        ).use { sourceTensor ->
            OnnxTensor.createTensor(
                environment,
                FloatBuffer.wrap(targetData),
                longArrayOf(1, 3, swapper.cropSize.toLong(), swapper.cropSize.toLong()),
            ).use { targetTensor ->
                session.run(
                    mapOf(
                        SWAPPER_SOURCE_INPUT_NAME to sourceTensor,
                        SWAPPER_TARGET_INPUT_NAME to targetTensor,
                    ),
                ).use { result ->
                    val output = copyFloatTensor(
                        result[SWAPPER_OUTPUT_NAME].orElseThrow {
                            IllegalStateException("Swapper has no '$SWAPPER_OUTPUT_NAME' output")
                        },
                    )
                    val expectedOutputSize = 3 * swapper.cropSize * swapper.cropSize
                    require(output.size == expectedOutputSize) {
                        "Unexpected swapper output element count: ${output.size}"
                    }
                    val mask = result[SWAPPER_MASK_OUTPUT_NAME]
                        .map(::copyFloatTensor)
                        .orElse(null)
                    return SwapperOutput(output, mask)
                }
            }
        }
    }

    private fun decodeYoloFace(
        output: FloatArray,
        originalWidth: Int,
        originalHeight: Int,
    ): List<DetectedFace5> {
        val scale = min(
            DETECTOR_SIZE.toDouble() / originalHeight.toDouble(),
            DETECTOR_SIZE.toDouble() / originalWidth.toDouble(),
        ).coerceAtMost(1.0)
        val resizedWidth = max(1, (originalWidth * scale).toInt())
        val resizedHeight = max(1, (originalHeight * scale).toInt())
        val ratioWidth = originalWidth.toDouble() / resizedWidth.toDouble()
        val ratioHeight = originalHeight.toDouble() / resizedHeight.toDouble()
        val candidates = ArrayList<DetectedFace5>()

        for (candidate in 0 until DETECTOR_CANDIDATES) {
            fun channel(index: Int): Float = output[index * DETECTOR_CANDIDATES + candidate]
            val score = channel(4)
            if (score <= DETECTOR_SCORE_THRESHOLD) continue

            val centerX = channel(0).toDouble()
            val centerY = channel(1).toDouble()
            val width = channel(2).toDouble()
            val height = channel(3).toDouble()
            val landmarks = (0 until LANDMARK_COUNT).map { landmark ->
                val offset = 5 + landmark * 3
                Point2(
                    x = channel(offset).toDouble() * ratioWidth,
                    y = channel(offset + 1).toDouble() * ratioHeight,
                )
            }
            candidates += DetectedFace5(
                score = score,
                box = FaceBox(
                    left = (centerX - width / 2.0) * ratioWidth,
                    top = (centerY - height / 2.0) * ratioHeight,
                    right = (centerX + width / 2.0) * ratioWidth,
                    bottom = (centerY + height / 2.0) * ratioHeight,
                ),
                landmarks = landmarks,
            )
        }
        return nonMaximumSuppression(candidates, NMS_THRESHOLD)
    }

    private fun nonMaximumSuppression(
        candidates: List<DetectedFace5>,
        threshold: Double,
    ): List<DetectedFace5> {
        val kept = ArrayList<DetectedFace5>()
        candidates.sortedByDescending(DetectedFace5::score).forEach { candidate ->
            if (kept.none { faceBoxIntersectionOverUnion(candidate.box, it.box) > threshold }) {
                kept += candidate
            }
        }
        return kept
    }

    private fun convertInSwapperEmbedding(
        embedding: FloatArray,
        emap: FloatArray,
    ): FloatArray {
        require(embedding.size == EMBEDDING_SIZE)
        require(emap.size == EMBEDDING_SIZE * EMBEDDING_SIZE)
        val norm = l2Norm(embedding)
        require(norm > 0f && norm.isFinite()) { "ArcFace returned a zero or invalid embedding" }
        return FloatArray(EMBEDDING_SIZE) { column ->
            var sum = 0.0
            for (row in 0 until EMBEDDING_SIZE) {
                sum += embedding[row].toDouble() * emap[row * EMBEDDING_SIZE + column].toDouble()
            }
            (sum / norm).toFloat()
        }
    }

    private fun normalizeL2(values: FloatArray): FloatArray {
        val norm = l2Norm(values)
        require(norm > 0f && norm.isFinite()) { "ArcFace returned a zero or invalid embedding" }
        return FloatArray(values.size) { index -> values[index] / norm }
    }

    private fun l2Norm(values: FloatArray): Float {
        var squaredSum = 0.0
        values.forEach { value -> squaredSum += value.toDouble() * value.toDouble() }
        return sqrt(squaredSum).toFloat()
    }

    private fun rawOutputToBitmap(output: FloatArray, swapper: SwapperModel): Bitmap {
        val size = swapper.cropSize
        val planeSize = size * size
        val pixels = IntArray(planeSize)
        for (index in 0 until planeSize) {
            fun visualValue(channel: Int): Int {
                val raw = output[channel * planeSize + index]
                val normalized = when (swapper) {
                    SwapperModel.HYPERSWAP_1A_256 -> raw * 0.5f + 0.5f
                    SwapperModel.INSWAPPER_128_FP16 -> raw
                }
                return (normalized.coerceIn(0f, 1f) * 255f).toInt()
            }
            val red = visualValue(0)
            val green = visualValue(1)
            val blue = visualValue(2)
            pixels[index] = (0xff shl 24) or (red shl 16) or (green shl 8) or blue
        }
        return Bitmap.createBitmap(pixels, size, size, Bitmap.Config.ARGB_8888)
    }

    private fun copyFloatTensor(value: ai.onnxruntime.OnnxValue): FloatArray {
        val tensor = value as? OnnxTensor
            ?: error("Expected an ONNX tensor output")
        val buffer = tensor.floatBuffer
        buffer.rewind()
        return FloatArray(buffer.remaining()).also(buffer::get)
    }

    private fun <T> runWithBackendFallback(
        modelFile: File,
        requestedBackend: RequestedInferenceBackend,
        operation: (OrtSession) -> T,
    ): Pair<T, InferenceBackend> {
        if (requestedBackend == RequestedInferenceBackend.XNNPACK_WITH_CPU_FALLBACK) {
            if (!mayAttemptXnnpack(Build.SUPPORTED_ABIS)) {
                return runWithSession(modelFile, SessionKind.CPU, operation) to InferenceBackend.CPU_FALLBACK
            }
            try {
                return runWithSession(modelFile, SessionKind.XNNPACK, operation) to InferenceBackend.XNNPACK
            } catch (_: OrtException) {
                return runWithSession(modelFile, SessionKind.CPU, operation) to InferenceBackend.CPU_FALLBACK
            }
        }
        return runWithSession(modelFile, SessionKind.CPU, operation) to InferenceBackend.CPU
    }

    private fun <T> runWithSession(
        modelFile: File,
        kind: SessionKind,
        operation: (OrtSession) -> T,
    ): T {
        createSessionOptions(kind).use { options ->
            environment.createSession(modelFile.absolutePath, options).use { session ->
                return operation(session)
            }
        }
    }

    private fun createSessionOptions(kind: SessionKind): OrtSession.SessionOptions {
        val options = OrtSession.SessionOptions()
        try {
            options.apply {
                setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
                setInterOpNumThreads(1)
                addConfigEntry("session.intra_op.allow_spinning", "0")
                when (kind) {
                    SessionKind.XNNPACK -> {
                        setIntraOpNumThreads(1)
                        addXnnpack(
                            mapOf(
                                "intra_op_num_threads" to inferenceThreadCount().toString(),
                            ),
                        )
                    }

                    SessionKind.CPU -> setIntraOpNumThreads(inferenceThreadCount())
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

    private data class SwapperOutput(
        val output: FloatArray,
        val mask: FloatArray?,
    )

    private enum class SessionKind {
        XNNPACK,
        CPU,
    }

    private companion object {
        const val DETECTOR_SIZE = 640
        const val DETECTOR_CHANNELS = 20
        const val DETECTOR_CANDIDATES = 8_400
        const val DETECTOR_SCORE_THRESHOLD = 0.5f
        const val NMS_THRESHOLD = 0.4
        const val LANDMARK_COUNT = 5
        const val RECOGNIZER_SIZE = 112
        const val EMBEDDING_SIZE = 512
        const val MAX_INFERENCE_THREADS = 4
        const val DETECTOR_INPUT_NAME = "input"
        const val RECOGNIZER_INPUT_NAME = "input"
        const val SWAPPER_SOURCE_INPUT_NAME = "source"
        const val SWAPPER_TARGET_INPUT_NAME = "target"
        const val SWAPPER_OUTPUT_NAME = "output"
        const val SWAPPER_MASK_OUTPUT_NAME = "mask"
        const val INSWAPPER_EMAP_NAME = "initializer"

        fun elapsedRealtimeMs(): Long = SystemClock.elapsedRealtimeNanos() / 1_000_000L
    }
}

/**
 * ONNX Runtime 1.26.0 aborts the whole process while creating the InSwapper
 * XNNPACK session on Android x86/x86_64. A native SIGABRT cannot reach the
 * OrtException fallback, so known-unsafe emulator ABIs are routed to CPU first.
 */
internal fun mayAttemptXnnpack(supportedAbis: Array<String>): Boolean =
    supportedAbis.none { abi -> abi.startsWith("x86", ignoreCase = true) }

/**
 * Resolves the neural detector result that corresponds to the face selected in the UI.
 * Without a hint, Stage B keeps its highest-confidence behaviour. With a hint, confidence
 * ordering must not override the user's assignment: the overlapping box with the greatest
 * IoU wins, and a non-overlapping result is rejected instead of silently swapping a stranger.
 */
internal fun selectDetectedFace(
    candidates: List<DetectedFace5>,
    hint: FaceBox?,
): DetectedFace5? {
    if (hint == null) return candidates.firstOrNull()
    return candidates
        .map { candidate -> candidate to faceBoxIntersectionOverUnion(candidate.box, hint) }
        .filter { (_, overlap) -> overlap > 0.0 }
        .maxByOrNull { (_, overlap) -> overlap }
        ?.first
}

internal fun faceBoxIntersectionOverUnion(first: FaceBox, second: FaceBox): Double {
    val intersectionWidth = max(0.0, min(first.right, second.right) - max(first.left, second.left))
    val intersectionHeight = max(0.0, min(first.bottom, second.bottom) - max(first.top, second.top))
    val intersection = intersectionWidth * intersectionHeight
    val firstArea = max(0.0, first.right - first.left) * max(0.0, first.bottom - first.top)
    val secondArea = max(0.0, second.right - second.left) * max(0.0, second.bottom - second.top)
    val union = firstArea + secondArea - intersection
    return if (union > 0.0) intersection / union else 0.0
}

/** Pixel transforms kept independent from Canvas/GPU to make parity deterministic. */
private object BitmapSampling {
    fun detectorInput(bitmap: Bitmap, detectorSize: Int): FloatArray {
        val source = PixelSource(bitmap)
        val scale = min(
            detectorSize.toDouble() / source.height.toDouble(),
            detectorSize.toDouble() / source.width.toDouble(),
        ).coerceAtMost(1.0)
        val resizedWidth = max(1, (source.width * scale).toInt())
        val resizedHeight = max(1, (source.height * scale).toInt())
        val planeSize = detectorSize * detectorSize
        val output = FloatArray(3 * planeSize)

        for (destinationY in 0 until resizedHeight) {
            val sourceY = (destinationY + 0.5) * source.height / resizedHeight - 0.5
            for (destinationX in 0 until resizedWidth) {
                val sourceX = (destinationX + 0.5) * source.width / resizedWidth - 0.5
                val color = source.sample(sourceX, sourceY)
                val index = destinationY * detectorSize + destinationX
                // FaceFusion receives OpenCV BGR and transposes it to NCHW.
                output[index] = color.blue / 255f
                output[planeSize + index] = color.green / 255f
                output[2 * planeSize + index] = color.red / 255f
            }
        }
        return output
    }

    fun rgbTensor(
        bitmap: Bitmap,
        mean: Float,
        standardDeviation: Float,
    ): FloatArray {
        require(standardDeviation != 0f)
        val pixels = IntArray(bitmap.width * bitmap.height)
        bitmap.getPixels(pixels, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)
        val planeSize = pixels.size
        val output = FloatArray(3 * planeSize)
        pixels.forEachIndexed { index, pixel ->
            val red = ((pixel ushr 16) and 0xff) / 255f
            val green = ((pixel ushr 8) and 0xff) / 255f
            val blue = (pixel and 0xff) / 255f
            output[index] = (red - mean) / standardDeviation
            output[planeSize + index] = (green - mean) / standardDeviation
            output[2 * planeSize + index] = (blue - mean) / standardDeviation
        }
        return output
    }

    fun warpAffine(
        source: Bitmap,
        sourceToDestination: AffineMatrix,
        destinationWidth: Int,
        destinationHeight: Int,
    ): Bitmap {
        val sourcePixels = PixelSource(source)
        val destinationToSource = sourceToDestination.inverse()
        val output = IntArray(destinationWidth * destinationHeight)
        for (y in 0 until destinationHeight) {
            for (x in 0 until destinationWidth) {
                val sourcePoint = destinationToSource.map(Point2(x.toDouble(), y.toDouble()))
                val color = sourcePixels.sample(sourcePoint.x, sourcePoint.y)
                output[y * destinationWidth + x] =
                    (0xff shl 24) or
                        (color.red.toInt().coerceIn(0, 255) shl 16) or
                        (color.green.toInt().coerceIn(0, 255) shl 8) or
                        color.blue.toInt().coerceIn(0, 255)
            }
        }
        return Bitmap.createBitmap(output, destinationWidth, destinationHeight, Bitmap.Config.ARGB_8888)
    }

    private class PixelSource(bitmap: Bitmap) {
        val width: Int = bitmap.width
        val height: Int = bitmap.height
        private val pixels = IntArray(width * height).also { destination ->
            bitmap.getPixels(destination, 0, width, 0, 0, width, height)
        }

        fun sample(x: Double, y: Double): RgbColor {
            val boundedX = x.coerceIn(0.0, (width - 1).toDouble())
            val boundedY = y.coerceIn(0.0, (height - 1).toDouble())
            val x0 = floor(boundedX).toInt()
            val y0 = floor(boundedY).toInt()
            val x1 = min(x0 + 1, width - 1)
            val y1 = min(y0 + 1, height - 1)
            val fractionX = boundedX - x0
            val fractionY = boundedY - y0

            val topLeft = pixels[y0 * width + x0]
            val topRight = pixels[y0 * width + x1]
            val bottomLeft = pixels[y1 * width + x0]
            val bottomRight = pixels[y1 * width + x1]
            return RgbColor(
                red = bilinear(topLeft ushr 16, topRight ushr 16, bottomLeft ushr 16, bottomRight ushr 16, fractionX, fractionY),
                green = bilinear(topLeft ushr 8, topRight ushr 8, bottomLeft ushr 8, bottomRight ushr 8, fractionX, fractionY),
                blue = bilinear(topLeft, topRight, bottomLeft, bottomRight, fractionX, fractionY),
            )
        }

        private fun bilinear(
            topLeft: Int,
            topRight: Int,
            bottomLeft: Int,
            bottomRight: Int,
            fractionX: Double,
            fractionY: Double,
        ): Float {
            val tl = (topLeft and 0xff).toDouble()
            val tr = (topRight and 0xff).toDouble()
            val bl = (bottomLeft and 0xff).toDouble()
            val br = (bottomRight and 0xff).toDouble()
            val top = tl + (tr - tl) * fractionX
            val bottom = bl + (br - bl) * fractionX
            return (top + (bottom - top) * fractionY).toFloat()
        }
    }

    private data class RgbColor(
        val red: Float,
        val green: Float,
        val blue: Float,
    )
}
