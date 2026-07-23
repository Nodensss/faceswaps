package com.faceswaplocal.app.domain

import kotlin.math.max
import kotlin.math.min

@JvmInline
value class FaceId(val value: String)

data class NormalizedRect(
    val left: Float,
    val top: Float,
    val right: Float,
    val bottom: Float,
) {
    init {
        require(left in 0f..1f)
        require(top in 0f..1f)
        require(right in 0f..1f)
        require(bottom in 0f..1f)
        require(left <= right)
        require(top <= bottom)
    }

    val width: Float get() = right - left
    val height: Float get() = bottom - top

    companion object {
        fun clamped(left: Float, top: Float, right: Float, bottom: Float): NormalizedRect {
            val safeLeft = min(left, right).coerceIn(0f, 1f)
            val safeRight = max(left, right).coerceIn(0f, 1f)
            val safeTop = min(top, bottom).coerceIn(0f, 1f)
            val safeBottom = max(top, bottom).coerceIn(0f, 1f)
            return NormalizedRect(safeLeft, safeTop, safeRight, safeBottom)
        }
    }
}

data class DetectedFace(
    val id: FaceId,
    val bounds: NormalizedRect,
    val yawDegrees: Float,
    val rollDegrees: Float,
    val smileProbability: Float?,
)

data class SwapAssignment(
    val targetFaceId: FaceId,
    val sourceFaceId: FaceId,
)

object FaceAssignmentPlanner {
    fun defaults(
        sourceFaces: List<DetectedFace>,
        targetFaces: List<DetectedFace>,
    ): List<SwapAssignment> {
        val firstSource = sourceFaces.firstOrNull() ?: return emptyList()
        val firstTarget = targetFaces.firstOrNull() ?: return emptyList()
        return listOf(
            SwapAssignment(
                targetFaceId = firstTarget.id,
                sourceFaceId = firstSource.id,
            ),
        )
    }

    fun replaceSource(
        assignments: List<SwapAssignment>,
        targetFaceId: FaceId,
        sourceFaceId: FaceId,
    ): List<SwapAssignment> {
        var replaced = false
        val updated = assignments.map { assignment ->
            if (assignment.targetFaceId == targetFaceId) {
                replaced = true
                assignment.copy(sourceFaceId = sourceFaceId)
            } else {
                assignment
            }
        }

        return if (replaced) {
            updated
        } else {
            updated + SwapAssignment(targetFaceId, sourceFaceId)
        }
    }

    fun clearSource(assignments: List<SwapAssignment>, sourceFaceId: FaceId): List<SwapAssignment> =
        assignments.filterNot { it.sourceFaceId == sourceFaceId }

    fun applySourceToAll(
        targetFaces: List<DetectedFace>,
        sourceFaceId: FaceId,
    ): List<SwapAssignment> = targetFaces.map { SwapAssignment(it.id, sourceFaceId) }

    fun retainValidSources(
        assignments: List<SwapAssignment>,
        sourceFaces: List<DetectedFace>,
    ): List<SwapAssignment> {
        val ids = sourceFaces.mapTo(mutableSetOf()) { it.id }
        return assignments.filter { it.sourceFaceId in ids }
    }
}

object AssignmentStateCodec {
    fun encode(assignments: List<SwapAssignment>): ArrayList<String> = ArrayList(
        assignments.map { "${it.targetFaceId.value}|${it.sourceFaceId.value}" },
    )

    fun restore(serialized: List<String>, sourceFaces: List<DetectedFace>, targetFaces: List<DetectedFace>): List<SwapAssignment> {
        val sources = sourceFaces.mapTo(mutableSetOf()) { it.id.value }
        val targets = targetFaces.mapTo(mutableSetOf()) { it.id.value }
        return serialized.mapNotNull { value ->
            val parts = value.split('|', limit = 2)
            if (parts.size == 2 && parts[0] in targets && parts[1] in sources) SwapAssignment(FaceId(parts[0]), FaceId(parts[1])) else null
        }
    }
}

