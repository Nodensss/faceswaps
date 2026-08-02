package com.faceswaplocal.app.inference

import android.graphics.Bitmap
import android.os.SystemClock
import com.faceswaplocal.app.domain.FaceId
import kotlin.coroutines.coroutineContext
import kotlinx.coroutines.ensureActive

data class MultiPhotoSource(val id: FaceId, val bitmap: Bitmap, val faceHint: FaceBox)
data class MultiPhotoTarget(val id: FaceId, val faceHint: FaceBox)
data class MultiPhotoAssignment(val targetId: FaceId, val sourceId: FaceId)

/** A write ROI bound to the assignment that produced it. */
data class AppliedFaceRoi(
    val targetId: FaceId,
    val sourceId: FaceId,
    val bounds: CompositeRoi,
)

/**
 * Two-pass photo result delivered without identity embeddings or runtime session types.
 * [swapRois] and [enhanceRois] are separately retained so parity can verify the exact
 * union of every region the two passes were allowed to change.
 */
data class MultiPhotoFaceSwapResult(
    val finalBitmap: Bitmap,
    val swapRois: List<AppliedFaceRoi>,
    val enhanceRois: List<AppliedFaceRoi>,
    val detectorBackend: InferenceBackend,
    val recognizerBackend: InferenceBackend,
    val swapperBackend: InferenceBackend,
    val enhancerBackends: List<InferenceBackend>,
    val parserBackends: List<InferenceBackend>,
    val protectedUnassignedRois: List<CompositeRoi>,
    val restorationStrength: Float,
    val enhancementMs: Long,
    val timings: PhotoFaceSwapTimings,
) {
    /** Stage-D compatibility view. */
    val pasteRois: List<CompositeRoi> get() = swapRois.map(AppliedFaceRoi::bounds)
}

/**
 * Photo coordinator with an explicit heavyweight-session barrier:
 *
 * 1. detect the untouched target once and complete every assigned InSwapper operation;
 * 2. only after the entire swap loop has returned, restore those successful assignments.
 *
 * Each concrete inference method owns and closes its ONNX session before returning, so
 * no swapper session can coexist with GFPGAN or BiSeNet. Unassigned detected faces never
 * enter either loop.
 */
class OnnxMultiPhotoFaceSwapPipeline(
    private val rawPipeline: OnnxRawFaceSwapPipeline,
    private val photoPipeline: OnnxPhotoFaceSwapPipeline,
    private val enhancerPipeline: OnnxFaceEnhancerPipeline? = null,
) {
    suspend fun process(
        target: Bitmap,
        sources: List<MultiPhotoSource>,
        targetsInStableOrder: List<MultiPhotoTarget>,
        assignments: List<MultiPhotoAssignment>,
        backend: RequestedInferenceBackend,
        restorationStrength: Float = 0f,
    ): MultiPhotoFaceSwapResult? {
        require(restorationStrength.isFinite() && restorationStrength in 0f..1f) {
            "Restoration strength must be in [0, 1]"
        }
        requireUniqueIds(sources.map(MultiPhotoSource::id), "source")
        requireUniqueIds(targetsInStableOrder.map(MultiPhotoTarget::id), "target")
        requireUniqueIds(assignments.map(MultiPhotoAssignment::targetId), "assigned target")

        val sourcesById = sources.associateBy(MultiPhotoSource::id)
        val assignmentsByTarget = assignments.associateBy(MultiPhotoAssignment::targetId)
        val workItems = targetsInStableOrder.mapNotNull { targetFace ->
            val assignment = assignmentsByTarget[targetFace.id] ?: return@mapNotNull null
            val source = sourcesById[assignment.sourceId] ?: return@mapNotNull null
            WorkItem(targetFace, source)
        }
        if (workItems.isEmpty()) return null

        val started = SystemClock.elapsedRealtime()
        val (resolvedTargets, detectorBackend) = rawPipeline.detectFaces(target, backend)
        var accumulated: IntArray? = null
        val sourceEmbeddings = mutableMapOf<FaceId, FloatArray>()
        val producedEmbeddings = mutableListOf<FloatArray>()
        val appliedSwaps = mutableListOf<AppliedSwap>()
        val swapRois = mutableListOf<AppliedFaceRoi>()
        val enhanceRois = mutableListOf<AppliedFaceRoi>()
        val enhancerBackends = mutableListOf<InferenceBackend>()
        val parserBackends = mutableListOf<InferenceBackend>()
        var protectedUnassignedRois = emptyList<CompositeRoi>()
        var enhancementMs = 0L
        var lastSwap: PhotoFaceSwapResult? = null
        var workingSwapBitmap: Bitmap? = null
        var resultBitmap: Bitmap? = null
        var delivered = false
        try {
            // Pass 1. No restoration call is reachable from inside this loop.
            for (workItem in workItems) {
                coroutineContext.ensureActive()
                val next = photoPipeline.process(
                    PhotoFaceSwapRequest(
                        source = workItem.source.bitmap,
                        target = target,
                        sourceFaceHint = workItem.source.faceHint,
                        targetFaceHint = workItem.target.faceHint,
                        resolvedTargetFaces = resolvedTargets,
                        basePixels = accumulated,
                        cachedSourceEmbedding = sourceEmbeddings[workItem.source.id],
                        backend = backend,
                    ),
                )
                producedEmbeddings += next.sourceEmbedding
                sourceEmbeddings.putIfAbsent(workItem.source.id, next.sourceEmbedding)

                val nextPixels = next.finalBitmap.readPixels()
                workingSwapBitmap?.recycleSafely()
                workingSwapBitmap = next.finalBitmap
                accumulated = nextPixels
                lastSwap = next
                appliedSwaps += AppliedSwap(workItem, next.targetFace)
                swapRois += AppliedFaceRoi(
                    targetId = workItem.target.id,
                    sourceId = workItem.source.id,
                    bounds = next.pasteRoi,
                )
            }

            val swapResult = lastSwap ?: return null
            var finalPixels = requireNotNull(accumulated)

            // Pass boundary: every photoPipeline call above has returned, therefore its
            // swapper session has been closed. Strength zero bypasses both model checks
            // and both restoration sessions.
            if (restorationStrength > 0f) {
                val enhancer = checkNotNull(enhancerPipeline) {
                    "Restoration was requested but no enhancer pipeline is configured"
                }
                val appliedTargetFaces = appliedSwaps.map(AppliedSwap::targetFace).toSet()
                protectedUnassignedRois = resolvedTargets
                    .filterNot(appliedTargetFaces::contains)
                    .map { face -> ffhqRoi(face, target.width, target.height) }
                val enhancementStarted = SystemClock.elapsedRealtime()
                for (applied in appliedSwaps) {
                    coroutineContext.ensureActive()
                    val enhanced = enhancer.enhance(
                        basePixels = finalPixels,
                        baseWidth = target.width,
                        baseHeight = target.height,
                        targetFace = applied.targetFace,
                        strength = restorationStrength,
                        backend = backend,
                        protectedBaseRois = protectedUnassignedRois,
                    )
                    finalPixels = enhanced.pixels
                    enhanceRois += AppliedFaceRoi(
                        targetId = applied.workItem.target.id,
                        sourceId = applied.workItem.source.id,
                        bounds = enhanced.roi,
                    )
                    enhancerBackends += enhanced.enhancerBackend
                    parserBackends += enhanced.parserBackend
                }
                enhancementMs = SystemClock.elapsedRealtime() - enhancementStarted
            }

            resultBitmap = if (restorationStrength > 0f) {
                Bitmap.createBitmap(
                    finalPixels,
                    target.width,
                    target.height,
                    Bitmap.Config.ARGB_8888,
                ).also {
                    workingSwapBitmap?.recycleSafely()
                    workingSwapBitmap = null
                }
            } else {
                requireNotNull(workingSwapBitmap).also { workingSwapBitmap = null }
            }

            val result = MultiPhotoFaceSwapResult(
                finalBitmap = requireNotNull(resultBitmap),
                swapRois = swapRois.toList(),
                enhanceRois = enhanceRois.toList(),
                detectorBackend = detectorBackend,
                recognizerBackend = swapResult.recognizerBackend,
                swapperBackend = swapResult.swapperBackend,
                enhancerBackends = enhancerBackends.toList(),
                parserBackends = parserBackends.toList(),
                protectedUnassignedRois = protectedUnassignedRois,
                restorationStrength = restorationStrength,
                enhancementMs = enhancementMs,
                timings = swapResult.timings.copy(
                    totalMs = SystemClock.elapsedRealtime() - started,
                ),
            )
            delivered = true
            return result
        } finally {
            // Every produced copy is cleared, including copies rejected by putIfAbsent.
            // This runs after success, inference errors and cancellation.
            producedEmbeddings.forEach { embedding -> embedding.fill(0f) }
            sourceEmbeddings.clear()
            workingSwapBitmap?.recycleSafely()
            if (!delivered) resultBitmap?.recycleSafely()
        }
    }

    private data class WorkItem(
        val target: MultiPhotoTarget,
        val source: MultiPhotoSource,
    )

    private data class AppliedSwap(
        val workItem: WorkItem,
        val targetFace: DetectedFace5,
    )

    private fun requireUniqueIds(ids: List<FaceId>, label: String) {
        require(ids.size == ids.toSet().size) { "Duplicate $label identifiers are not allowed" }
    }

    private fun ffhqRoi(face: DetectedFace5, width: Int, height: Int): CompositeRoi {
        val targetToCrop = FaceGeometry.estimateSimilarity(
            source = face.landmarks,
            template = WarpTemplate.FFHQ_512,
            cropWidth = ENHANCER_CROP_SIZE,
            cropHeight = ENHANCER_CROP_SIZE,
        )
        return FaceCompositor.calculateRoi(
            targetWidth = width,
            targetHeight = height,
            cropWidth = ENHANCER_CROP_SIZE,
            cropHeight = ENHANCER_CROP_SIZE,
            targetToCrop = targetToCrop,
        )
    }

    private fun Bitmap.readPixels(): IntArray = IntArray(width * height).also { pixels ->
        getPixels(pixels, 0, width, 0, 0, width, height)
    }

    private fun Bitmap.recycleSafely() {
        if (!isRecycled) recycle()
    }

    private companion object {
        const val ENHANCER_CROP_SIZE = 512
    }
}
