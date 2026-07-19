package com.faceswaplocal.app.domain

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class FaceAssignmentPlannerTest {
    private fun face(id: String) = DetectedFace(
        id = FaceId(id),
        bounds = NormalizedRect(0.1f, 0.1f, 0.4f, 0.5f),
        yawDegrees = 0f,
        rollDegrees = 0f,
        smileProbability = null,
    )

    @Test
    fun `defaults map every target to the first source`() {
        val assignments = FaceAssignmentPlanner.defaults(
            sourceFaces = listOf(face("source-1"), face("source-2")),
            targetFaces = listOf(face("target-1"), face("target-2")),
        )

        assertEquals(2, assignments.size)
        assertTrue(assignments.all { it.sourceFaceId == FaceId("source-1") })
    }

    @Test
    fun `replaceSource changes only the requested target`() {
        val assignments = FaceAssignmentPlanner.defaults(
            sourceFaces = listOf(face("source-1")),
            targetFaces = listOf(face("target-1"), face("target-2")),
        )

        val updated = FaceAssignmentPlanner.replaceSource(
            assignments = assignments,
            targetFaceId = FaceId("target-2"),
            sourceFaceId = FaceId("source-2"),
        )

        assertEquals(FaceId("source-1"), updated[0].sourceFaceId)
        assertEquals(FaceId("source-2"), updated[1].sourceFaceId)
    }

    @Test
    fun `defaults are empty without a source face`() {
        val assignments = FaceAssignmentPlanner.defaults(
            sourceFaces = emptyList(),
            targetFaces = listOf(face("target-1")),
        )

        assertTrue(assignments.isEmpty())
    }
}

