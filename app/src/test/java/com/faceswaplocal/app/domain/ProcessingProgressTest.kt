package com.faceswaplocal.app.domain

import org.junit.Assert.assertEquals
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Test

class ProcessingProgressTest {
    @Test
    fun `face counter is one-based and only reported inside per-face stages`() {
        assertEquals(
            0,
            ProcessingProgress(ProcessingStage.DETECTING, totalFaces = 3).currentFace,
        )
        assertEquals(
            1,
            ProcessingProgress(ProcessingStage.SWAPPING, completedFaces = 0, totalFaces = 3).currentFace,
        )
        assertEquals(
            3,
            ProcessingProgress(ProcessingStage.SWAPPING, completedFaces = 2, totalFaces = 3).currentFace,
        )
        assertEquals(
            "the last finished face must not report a fourth face",
            3,
            ProcessingProgress(ProcessingStage.SWAPPING, completedFaces = 3, totalFaces = 3).currentFace,
        )
        assertEquals(
            0,
            ProcessingProgress(ProcessingStage.EXPORTING, completedFaces = 3, totalFaces = 3).currentFace,
        )
    }

    @Test
    fun `fraction never decreases along a restoring run`() {
        val sequence = buildList {
            add(ProcessingProgress(ProcessingStage.PREPARING, 0, 2, restorationPlanned = true))
            add(ProcessingProgress(ProcessingStage.DETECTING, 0, 2, restorationPlanned = true))
            (0..2).forEach { done ->
                add(ProcessingProgress(ProcessingStage.SWAPPING, done, 2, restorationPlanned = true))
            }
            (0..2).forEach { done ->
                add(ProcessingProgress(ProcessingStage.RESTORING, done, 2, restorationPlanned = true))
            }
            add(ProcessingProgress(ProcessingStage.EXPORTING, 2, 2, restorationPlanned = true))
            add(ProcessingProgress(ProcessingStage.COMPLETED, 2, 2, restorationPlanned = true))
        }

        sequence.zipWithNext { previous, next ->
            assertTrue(
                "${previous.stage}:${previous.completedFaces} -> ${next.stage}:${next.completedFaces}",
                next.fraction >= previous.fraction,
            )
        }
        assertEquals(0f, sequence.first().fraction, 0.0001f)
        assertEquals(1f, sequence.last().fraction, 0.0001f)
    }

    @Test
    fun `swapping absorbs the restoration span when restoration is off`() {
        val withRestoration =
            ProcessingProgress(ProcessingStage.SWAPPING, completedFaces = 2, totalFaces = 2, restorationPlanned = true)
        val withoutRestoration =
            ProcessingProgress(ProcessingStage.SWAPPING, completedFaces = 2, totalFaces = 2, restorationPlanned = false)

        assertEquals(0.60f, withRestoration.fraction, 0.0001f)
        assertEquals(
            "a run without restoration must still reach the export span",
            0.90f,
            withoutRestoration.fraction,
            0.0001f,
        )
    }

    @Test
    fun `partial face completion splits the stage span evenly`() {
        val half = ProcessingProgress(ProcessingStage.SWAPPING, completedFaces = 1, totalFaces = 2, restorationPlanned = true)

        assertEquals(0.15f + (0.60f - 0.15f) / 2f, half.fraction, 0.0001f)
    }

    @Test
    fun `a stage without faces reports its span start`() {
        assertEquals(0.05f, ProcessingProgress(ProcessingStage.DETECTING).fraction, 0.0001f)
        assertEquals(0.90f, ProcessingProgress(ProcessingStage.EXPORTING).fraction, 0.0001f)
        assertEquals(
            "no assigned face must not divide by zero",
            0.15f,
            ProcessingProgress(ProcessingStage.SWAPPING, totalFaces = 0).fraction,
            0.0001f,
        )
    }

    @Test
    fun `impossible counters are rejected instead of being clamped silently`() {
        assertThrows(IllegalArgumentException::class.java) {
            ProcessingProgress(ProcessingStage.SWAPPING, completedFaces = 3, totalFaces = 2)
        }
        assertThrows(IllegalArgumentException::class.java) {
            ProcessingProgress(ProcessingStage.SWAPPING, completedFaces = -1, totalFaces = 2)
        }
    }
}
