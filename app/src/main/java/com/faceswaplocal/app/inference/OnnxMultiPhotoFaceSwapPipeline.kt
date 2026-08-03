package com.faceswaplocal.app.inference

import android.graphics.Bitmap
import android.os.SystemClock
import com.faceswaplocal.app.domain.FaceId
import com.faceswaplocal.app.domain.ProcessingProgress
import com.faceswaplocal.app.domain.ProcessingStage
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
    val swapParserBackends: List<InferenceBackend>,
    val enhancementParserBackends: List<InferenceBackend>,
    val protectedUnassignedRois: List<CompositeRoi>,
    val restorationStrength: Float,
    val swapParserMs: Long,
    val enhancementMs: Long,
    val timings: PhotoFaceSwapTimings,
) {
    /** Stage-D compatibility view. */
    val pasteRois: List<CompositeRoi> get() = swapRois.map(AppliedFaceRoi::bounds)

    /** Checkpoint-1 compatibility view; new code should select the pass explicitly. */
    val parserBackends: List<InferenceBackend> get() = enhancementParserBackends
}

/**
 * Photo coordinator with an explicit heavyweight-session barrier:
 *
 * 1. detect the untouched target once and complete every assigned InSwapper operation;
 * 2. only after the entire swap loop has returned, restore those successful assignments.
 *
 * One BiSeNet session is reused across the task. It may coexist with InSwapper in pass 1
 * and GFPGAN in pass 2, while the pass barrier guarantees InSwapper and GFPGAN never
 * coexist. Unassigned detected faces never enter either loop.
 */
class OnnxMultiPhotoFaceSwapPipeline(
    private val rawPipeline: OnnxRawFaceSwapPipeline,
    private val photoPipeline: OnnxPhotoFaceSwapPipeline,
    private val enhancerPipeline: OnnxFaceEnhancerPipeline? = null,
    private val parserPipeline: OnnxFaceParserPipeline? = null,
) {
    suspend fun process(
        target: Bitmap,
        sources: List<MultiPhotoSource>,
        targetsInStableOrder: List<MultiPhotoTarget>,
        assignments: List<MultiPhotoAssignment>,
        backend: RequestedInferenceBackend,
        restorationStrength: Float = 0f,
        swapBlendMaskMode: SwapBlendMaskMode = SwapBlendMaskMode.PARSER_REGION,
        onProgress: (ProcessingProgress) -> Unit = {},
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

        val progress = ProgressReporter(
            onProgress = onProgress,
            totalFaces = workItems.size,
            restorationPlanned = restorationStrength > 0f,
        )
        progress.report(ProcessingStage.PREPARING)

        val started = SystemClock.elapsedRealtime()
        progress.report(ProcessingStage.DETECTING)
        val (resolvedTargets, detectorBackend) = rawPipeline.detectFaces(target, backend)
        val needsParser =
            swapBlendMaskMode == SwapBlendMaskMode.PARSER_REGION || restorationStrength > 0f
        var undeliveredResult: MultiPhotoFaceSwapResult? = null
        try {
            val result = if (needsParser) {
                val parser = checkNotNull(parserPipeline) {
                    "Parser blending/restoration was requested but no parser pipeline is configured"
                }
                parser.withSession(backend) { parserSession ->
                    processWithParserSession(
                        target = target,
                        workItems = workItems,
                        resolvedTargets = resolvedTargets,
                        detectorBackend = detectorBackend,
                        backend = backend,
                        restorationStrength = restorationStrength,
                        swapBlendMaskMode = swapBlendMaskMode,
                        parserSession = parserSession,
                        started = started,
                        progress = progress,
                    ).also { undeliveredResult = it }
                }
            } else {
                processWithParserSession(
                    target = target,
                    workItems = workItems,
                    resolvedTargets = resolvedTargets,
                    detectorBackend = detectorBackend,
                    backend = backend,
                    restorationStrength = restorationStrength,
                    swapBlendMaskMode = swapBlendMaskMode,
                    parserSession = null,
                    started = started,
                    progress = progress,
                ).also { undeliveredResult = it }
            }
            undeliveredResult = null
            return result
        } finally {
            undeliveredResult?.finalBitmap?.recycleSafely()
        }
    }

    private suspend fun processWithParserSession(
        target: Bitmap,
        workItems: List<WorkItem>,
        resolvedTargets: List<DetectedFace5>,
        detectorBackend: InferenceBackend,
        backend: RequestedInferenceBackend,
        restorationStrength: Float,
        swapBlendMaskMode: SwapBlendMaskMode,
        parserSession: FaceParserSession?,
        started: Long,
        progress: ProgressReporter,
    ): MultiPhotoFaceSwapResult {
        var accumulated: IntArray? = null
        var finalPixels: IntArray? = null
        val sourceEmbeddings = mutableMapOf<FaceId, FloatArray>()
        val producedEmbeddings = mutableListOf<FloatArray>()
        val appliedSwaps = mutableListOf<AppliedSwap>()
        val swapRois = mutableListOf<AppliedFaceRoi>()
        val enhanceRois = mutableListOf<AppliedFaceRoi>()
        val enhancerBackends = mutableListOf<InferenceBackend>()
        val swapParserBackends = mutableListOf<InferenceBackend>()
        val enhancementParserBackends = mutableListOf<InferenceBackend>()
        var protectedUnassignedRois = emptyList<CompositeRoi>()
        var swapParserMs = 0L
        var enhancementMs = 0L
        var lastSwap: PhotoFaceSwapResult? = null
        var workingSwapBitmap: Bitmap? = null
        var resultBitmap: Bitmap? = null
        var delivered = false
        try {
            // Pass 1. No restoration call is reachable from inside this loop.
            for ((index, workItem) in workItems.withIndex()) {
                coroutineContext.ensureActive()
                progress.report(ProcessingStage.SWAPPING, completedFaces = index)
                val next = photoPipeline.process(
                    PhotoFaceSwapRequest(
                        source = workItem.source.bitmap,
                        target = target,
                        sourceFaceHint = workItem.source.faceHint,
                        targetFaceHint = workItem.target.faceHint,
                        resolvedTargetFaces = resolvedTargets,
                        basePixels = accumulated,
                        cachedSourceEmbedding = sourceEmbeddings[workItem.source.id],
                        swapBlendMaskMode = swapBlendMaskMode,
                        parserSession = parserSession,
                        backend = backend,
                    ),
                )
                producedEmbeddings += next.sourceEmbedding
                sourceEmbeddings.putIfAbsent(workItem.source.id, next.sourceEmbedding)

                val nextPixels = next.finalBitmap.readPixels()
                accumulated?.fill(0)
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
                next.parserBackend?.let(swapParserBackends::add)
                swapParserMs += next.timings.parserMs
                next.cropMask.fill(0f)
                progress.report(ProcessingStage.SWAPPING, completedFaces = index + 1)
            }

            val swapResult = checkNotNull(lastSwap) { "No assigned face produced a swap" }
            finalPixels = requireNotNull(accumulated)

            // Pass boundary: every photoPipeline call above has returned, therefore its
            // swapper session has been closed. Strength zero bypasses both model checks
            // and both restoration sessions.
            if (restorationStrength > 0f) {
                val enhancer = checkNotNull(enhancerPipeline) {
                    "Restoration was requested but no enhancer pipeline is configured"
                }
                val parser = checkNotNull(parserSession) {
                    "Restoration requires an open face parser session"
                }
                val appliedTargetFaces = appliedSwaps.map(AppliedSwap::targetFace).toSet()
                protectedUnassignedRois = resolvedTargets
                    .filterNot(appliedTargetFaces::contains)
                    .map { face -> ffhqRoi(face, target.width, target.height) }
                val enhancementStarted = SystemClock.elapsedRealtime()
                for ((index, applied) in appliedSwaps.withIndex()) {
                    coroutineContext.ensureActive()
                    progress.report(ProcessingStage.RESTORING, completedFaces = index)
                    val enhanced = enhancer.enhance(
                        basePixels = requireNotNull(finalPixels),
                        baseWidth = target.width,
                        baseHeight = target.height,
                        targetFace = applied.targetFace,
                        strength = restorationStrength,
                        backend = backend,
                        parserSession = parser,
                        protectedBaseRois = protectedUnassignedRois,
                    )
                    finalPixels?.fill(0)
                    finalPixels = enhanced.pixels
                    enhanceRois += AppliedFaceRoi(
                        targetId = applied.workItem.target.id,
                        sourceId = applied.workItem.source.id,
                        bounds = enhanced.roi,
                    )
                    enhancerBackends += enhanced.enhancerBackend
                    enhancementParserBackends += enhanced.parserBackend
                    progress.report(ProcessingStage.RESTORING, completedFaces = index + 1)
                }
                enhancementMs = SystemClock.elapsedRealtime() - enhancementStarted
            }

            resultBitmap = if (restorationStrength > 0f) {
                Bitmap.createBitmap(
                    requireNotNull(finalPixels),
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
                swapParserBackends = swapParserBackends.toList(),
                enhancementParserBackends = enhancementParserBackends.toList(),
                protectedUnassignedRois = protectedUnassignedRois,
                restorationStrength = restorationStrength,
                swapParserMs = swapParserMs,
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
            finalPixels?.fill(0)
            accumulated?.fill(0)
            workingSwapBitmap?.recycleSafely()
            if (!delivered) resultBitmap?.recycleSafely()
        }
    }

    /**
     * Converts coordinator positions into the domain progress model. Only assigned faces
     * are counted, because only they enter either pass; a duplicate event is dropped so
     * the UI is not woken up between two identical states.
     */
    private class ProgressReporter(
        private val onProgress: (ProcessingProgress) -> Unit,
        private val totalFaces: Int,
        private val restorationPlanned: Boolean,
    ) {
        private var last: ProcessingProgress? = null

        fun report(stage: ProcessingStage, completedFaces: Int = 0) {
            val next = ProcessingProgress(
                stage = stage,
                completedFaces = completedFaces.coerceIn(0, totalFaces),
                totalFaces = totalFaces,
                restorationPlanned = restorationPlanned,
            )
            if (next == last) return
            last = next
            onProgress(next)
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
