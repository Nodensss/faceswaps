package com.faceswaplocal.app.inference

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

class FaceSelectionTest {
    @Test
    fun highestIouFollowsUiHintInsteadOfDetectorConfidence() {
        val wrongHighConfidence = face(
            score = 0.99f,
            box = FaceBox(0.0, 0.0, 40.0, 40.0),
        )
        val selectedLowerConfidence = face(
            score = 0.70f,
            box = FaceBox(90.0, 90.0, 150.0, 150.0),
        )

        val actual = selectDetectedFace(
            candidates = listOf(wrongHighConfidence, selectedLowerConfidence),
            hint = FaceBox(100.0, 100.0, 160.0, 160.0),
        )

        assertEquals(selectedLowerConfidence, actual)
    }

    @Test
    fun absentHintPreservesHighestConfidenceOrdering() {
        val first = face(0.9f, FaceBox(0.0, 0.0, 10.0, 10.0))
        val second = face(0.8f, FaceBox(20.0, 20.0, 30.0, 30.0))

        assertEquals(first, selectDetectedFace(listOf(first, second), hint = null))
    }

    @Test
    fun nonOverlappingHintIsRejected() {
        val candidate = face(0.9f, FaceBox(0.0, 0.0, 10.0, 10.0))

        assertNull(
            selectDetectedFace(
                candidates = listOf(candidate),
                hint = FaceBox(20.0, 20.0, 30.0, 30.0),
            ),
        )
    }

    @Test
    fun iouUsesUnionArea() {
        assertEquals(
            25.0 / 175.0,
            faceBoxIntersectionOverUnion(
                FaceBox(0.0, 0.0, 10.0, 10.0),
                FaceBox(5.0, 5.0, 15.0, 15.0),
            ),
            1e-12,
        )
    }

    private fun face(score: Float, box: FaceBox): DetectedFace5 = DetectedFace5(
        score = score,
        box = box,
        landmarks = List(5) { index -> Point2(index.toDouble(), index.toDouble()) },
    )
}
