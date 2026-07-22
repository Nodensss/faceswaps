package com.faceswaplocal.app.inference

import android.graphics.Bitmap
import android.os.SystemClock
import kotlin.coroutines.coroutineContext
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.withContext

data class PhotoFaceSwapRequest(
    val source: Bitmap,
    val target: Bitmap,
    val sourceFaceHint: FaceBox? = null,
    val targetFaceHint: FaceBox? = null,
    val resolvedTargetFaces: List<DetectedFace5>? = null,
    /** Packed pixels of preceding compositing steps; geometry always uses [target]. */
    val basePixels: IntArray? = null,
    val cachedSourceEmbedding: FloatArray? = null,
    val backend: RequestedInferenceBackend = RequestedInferenceBackend.XNNPACK_WITH_CPU_FALLBACK,
)

data class PhotoFaceSwapTimings(
    val detectorMs: Long,
    val recognizerMs: Long,
    val swapperMs: Long,
    val compositingMs: Long,
    val totalMs: Long,
)

/**
 * Stage C result. It owns only the target-sized final bitmap and lightweight diagnostics;
 * aligned/raw Stage B bitmaps are released before this object is returned.
 */
data class PhotoFaceSwapResult(
    val finalBitmap: Bitmap,
    val targetToSwapperCrop: AffineMatrix,
    val cropMask: FloatArray,
    val pasteRoi: CompositeRoi,
    val colorAdjustment: FaceColorAdjustment,
    val detectorBackend: InferenceBackend,
    val recognizerBackend: InferenceBackend,
    val swapperBackend: InferenceBackend,
    val timings: PhotoFaceSwapTimings,
    val sourceEmbedding: FloatArray,
)

/**
 * Stage C InSwapper pipeline: verified Stage B inference followed by deterministic CPU
 * colour matching, inverse affine mask warp and soft alpha compositing.
 */
class OnnxPhotoFaceSwapPipeline(
    modelStore: ModelStore,
    private val workerDispatcher: CoroutineDispatcher = Dispatchers.Default,
    private val rawPipeline: OnnxRawFaceSwapPipeline = OnnxRawFaceSwapPipeline(modelStore),
) {
    suspend fun process(request: PhotoFaceSwapRequest): PhotoFaceSwapResult {
        // withContext has prompt cancellation on the way back to the caller. Keep a
        // reference outside its block so a completed bitmap is still recycled if the
        // result is cancelled before ownership can be delivered to the caller.
        var undeliveredBitmap: Bitmap? = null
        try {
            val result = withContext(workerDispatcher) {
                val totalStarted = elapsedRealtimeMs()
                val raw = rawPipeline.process(
                    RawFaceSwapRequest(
                        source = request.source,
                        target = request.target,
                        swapper = SwapperModel.INSWAPPER_128_FP16,
                        backend = request.backend,
                        sourceFaceHint = request.sourceFaceHint,
                        targetFaceHint = request.targetFaceHint,
                        resolvedTargetFaces = request.resolvedTargetFaces,
                        cachedSourceEmbedding = request.cachedSourceEmbedding,
                    ),
                )
                try {
                    coroutineContext.ensureActive()
                    val compositingStarted = elapsedRealtimeMs()
                    val targetPixels = request.basePixels ?: request.target.readPixels()
                    val targetCropPixels = raw.alignedTarget.readPixels()
                    val swappedCropPixels = raw.rawOutputBitmap.readPixels()
                    val composite = FaceCompositor.composite(
                        targetPixels = targetPixels,
                        targetWidth = request.target.width,
                        targetHeight = request.target.height,
                        targetCropPixels = targetCropPixels,
                        swappedCropPixels = swappedCropPixels,
                        cropWidth = raw.swapper.cropSize,
                        cropHeight = raw.swapper.cropSize,
                        targetToCrop = raw.targetToSwapperCrop,
                    )
                    coroutineContext.ensureActive()
                    val resultBitmap = Bitmap.createBitmap(
                        composite.pixels,
                        composite.width,
                        composite.height,
                        Bitmap.Config.ARGB_8888,
                    ).also { bitmap -> undeliveredBitmap = bitmap }
                    val compositingMs = elapsedRealtimeMs() - compositingStarted
                    PhotoFaceSwapResult(
                        finalBitmap = resultBitmap,
                        targetToSwapperCrop = raw.targetToSwapperCrop,
                        cropMask = composite.cropMask,
                        pasteRoi = composite.roi,
                        colorAdjustment = composite.colorAdjustment,
                        detectorBackend = raw.detectorBackend,
                        recognizerBackend = raw.recognizerBackend,
                        swapperBackend = raw.swapperBackend,
                        timings = PhotoFaceSwapTimings(
                            detectorMs = raw.timings.detectorMs,
                            recognizerMs = raw.timings.recognizerMs,
                            swapperMs = raw.timings.swapperMs,
                            compositingMs = compositingMs,
                            totalMs = elapsedRealtimeMs() - totalStarted,
                        ),
                        sourceEmbedding = raw.sourceEmbedding,
                    )
                } finally {
                    raw.recycleBitmaps()
                }
            }
            undeliveredBitmap = null
            return result
        } finally {
            undeliveredBitmap?.recycleSafely()
        }
    }

    private fun Bitmap.readPixels(): IntArray = IntArray(width * height).also { pixels ->
        getPixels(pixels, 0, width, 0, 0, width, height)
    }

    private fun RawFaceSwapResult.recycleBitmaps() {
        alignedSource112.recycleSafely()
        alignedTarget.recycleSafely()
        rawOutputBitmap.recycleSafely()
    }

    private fun Bitmap.recycleSafely() {
        if (!isRecycled) recycle()
    }

    private fun elapsedRealtimeMs(): Long = SystemClock.elapsedRealtimeNanos() / 1_000_000L
}
