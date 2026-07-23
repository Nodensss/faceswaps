package com.faceswaplocal.app.inference

import android.graphics.Bitmap
import android.os.SystemClock
import com.faceswaplocal.app.domain.FaceId
import kotlinx.coroutines.ensureActive
import kotlin.coroutines.coroutineContext

data class MultiPhotoSource(val id: FaceId, val bitmap: Bitmap, val faceHint: FaceBox)
data class MultiPhotoTarget(val id: FaceId, val faceHint: FaceBox)
data class MultiPhotoAssignment(val targetId: FaceId, val sourceId: FaceId)

/**
 * Stage D coordinator output delivered to the UI layer. It intentionally excludes the
 * source identity embedding: the ViewModel never needs it, so it is not carried past the
 * pipeline boundary. [pasteRois] lists the composited paste region of every applied swap
 * in stable target order.
 */
data class MultiPhotoFaceSwapResult(
    val finalBitmap: Bitmap,
    val pasteRois: List<CompositeRoi>,
    val detectorBackend: InferenceBackend,
    val recognizerBackend: InferenceBackend,
    val swapperBackend: InferenceBackend,
    val timings: PhotoFaceSwapTimings,
)

/**
 * Stage D coordinator. It detects the target exactly once from the untouched image,
 * then composites stable target order onto the accumulated pixel buffer.
 */
class OnnxMultiPhotoFaceSwapPipeline(
    private val rawPipeline: OnnxRawFaceSwapPipeline,
    private val photoPipeline: OnnxPhotoFaceSwapPipeline,
) {
    suspend fun process(
        target: Bitmap,
        sources: List<MultiPhotoSource>,
        targetsInStableOrder: List<MultiPhotoTarget>,
        assignments: List<MultiPhotoAssignment>,
        backend: RequestedInferenceBackend,
    ): MultiPhotoFaceSwapResult? {
        val started = SystemClock.elapsedRealtime()
        val (resolvedTargets, _) = rawPipeline.detectFaces(target, backend)
        var accumulated: IntArray? = null
        val sourceEmbeddings = mutableMapOf<FaceId, FloatArray>()
        var last: PhotoFaceSwapResult? = null
        val pasteRois = mutableListOf<CompositeRoi>()
        try {
        for (targetFace in targetsInStableOrder) {
            coroutineContext.ensureActive()
            val assignment = assignments.firstOrNull { it.targetId == targetFace.id } ?: continue
            val source = sources.firstOrNull { it.id == assignment.sourceId } ?: continue
            val next = photoPipeline.process(
                PhotoFaceSwapRequest(
                    source = source.bitmap,
                    target = target,
                    sourceFaceHint = source.faceHint,
                    targetFaceHint = targetFace.faceHint,
                    resolvedTargetFaces = resolvedTargets,
                    basePixels = accumulated,
                    cachedSourceEmbedding = sourceEmbeddings[source.id],
                    backend = backend,
                ),
            )
            last?.finalBitmap?.recycleSafely()
            last = next
            sourceEmbeddings.putIfAbsent(source.id, next.sourceEmbedding)
            pasteRois += next.pasteRoi
            accumulated = IntArray(next.finalBitmap.width * next.finalBitmap.height).also {
                next.finalBitmap.getPixels(it, 0, next.finalBitmap.width, 0, 0, next.finalBitmap.width, next.finalBitmap.height)
            }
        }
        return last?.let { result ->
            MultiPhotoFaceSwapResult(
                finalBitmap = result.finalBitmap,
                pasteRois = pasteRois.toList(),
                detectorBackend = result.detectorBackend,
                recognizerBackend = result.recognizerBackend,
                swapperBackend = result.swapperBackend,
                timings = result.timings.copy(totalMs = SystemClock.elapsedRealtime() - started),
            )
        }
        } finally {
            // Zero every identity embedding this task held: the per-source cache and the
            // final result copy that is not handed to the ViewModel. Runs on success,
            // error and cancellation.
            sourceEmbeddings.values.forEach { it.fill(0f) }
            sourceEmbeddings.clear()
            last?.sourceEmbedding?.fill(0f)
        }
    }

    private fun Bitmap.recycleSafely() { if (!isRecycled) recycle() }
}
