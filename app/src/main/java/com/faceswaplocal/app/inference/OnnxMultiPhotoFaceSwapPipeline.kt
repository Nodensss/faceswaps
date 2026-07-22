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
    ): PhotoFaceSwapResult? {
        val started = SystemClock.elapsedRealtime()
        val (resolvedTargets, _) = rawPipeline.detectFaces(target, backend)
        var accumulated: IntArray? = null
        var last: PhotoFaceSwapResult? = null
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
                    backend = backend,
                ),
            )
            last?.finalBitmap?.recycleSafely()
            last = next
            accumulated = IntArray(next.finalBitmap.width * next.finalBitmap.height).also {
                next.finalBitmap.getPixels(it, 0, next.finalBitmap.width, 0, 0, next.finalBitmap.width, next.finalBitmap.height)
            }
        }
        return last?.let { result ->
            result.copy(timings = result.timings.copy(totalMs = SystemClock.elapsedRealtime() - started))
        }
    }

    private fun Bitmap.recycleSafely() { if (!isRecycled) recycle() }
}
