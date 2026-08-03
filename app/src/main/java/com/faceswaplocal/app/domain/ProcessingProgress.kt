package com.faceswaplocal.app.domain

/** Pipeline steps the user is allowed to see (FR-PHOTO-07, §9.4). */
enum class ProcessingStage(
    internal val spanStart: Float,
    internal val spanEnd: Float,
    /** True when the stage advances once per assigned face. */
    val countsFaces: Boolean,
) {
    PREPARING(0f, 0.05f, countsFaces = false),
    DETECTING(0.05f, 0.15f, countsFaces = false),
    SWAPPING(0.15f, 0.60f, countsFaces = true),
    RESTORING(0.60f, 0.90f, countsFaces = true),
    EXPORTING(0.90f, 1f, countsFaces = false),
    COMPLETED(1f, 1f, countsFaces = false),
}

/**
 * Deterministic progress model. Every stage owns a fixed span of the bar and the face
 * counter only moves inside its own span, so the reported fraction never goes backwards
 * and never depends on a wall-clock estimate (§9.4 forbids promising remaining time).
 *
 * When restoration is not planned, the swap stage absorbs the restoration span so a
 * completed run still reaches the export stage at a full swap bar.
 */
data class ProcessingProgress(
    val stage: ProcessingStage,
    val completedFaces: Int = 0,
    val totalFaces: Int = 0,
    val restorationPlanned: Boolean = false,
) {
    init {
        require(completedFaces >= 0) { "Completed faces must not be negative" }
        require(totalFaces >= 0) { "Total faces must not be negative" }
        require(completedFaces <= totalFaces) { "Completed faces must not exceed the total" }
    }

    /** 1-based number of the face being worked on, or 0 outside a per-face stage. */
    val currentFace: Int
        get() = if (stage.countsFaces && totalFaces > 0) {
            (completedFaces + 1).coerceAtMost(totalFaces)
        } else {
            0
        }

    val fraction: Float
        get() {
            val end = if (stage == ProcessingStage.SWAPPING && !restorationPlanned) {
                ProcessingStage.RESTORING.spanEnd
            } else {
                stage.spanEnd
            }
            if (!stage.countsFaces || totalFaces <= 0) return stage.spanStart
            val done = completedFaces.toFloat() / totalFaces.toFloat()
            return stage.spanStart + (end - stage.spanStart) * done
        }
}
