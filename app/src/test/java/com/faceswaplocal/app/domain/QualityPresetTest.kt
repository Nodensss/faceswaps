package com.faceswaplocal.app.domain

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class QualityPresetTest {
    @Test
    fun `each preset maps onto settings the pipeline already supports`() {
        val fast = requireNotNull(QualityPreset.FAST.settings)
        val balanced = requireNotNull(QualityPreset.BALANCED.settings)
        val maximum = requireNotNull(QualityPreset.MAXIMUM.settings)

        // Fast is the only mode without a restoration pass, which is the expensive one.
        assertEquals(0f, fast.effectiveRestorationStrength)
        assertEquals(FaceQualitySettings.DEFAULT_RESTORATION_STRENGTH, balanced.effectiveRestorationStrength)
        assertEquals(FaceQualitySettings.DEFAULT_RESTORATION_STRENGTH, maximum.effectiveRestorationStrength)

        // Maximum is Balanced plus the parser blend mask, and nothing else.
        assertFalse(balanced.parserSwapMaskEnabled)
        assertTrue(maximum.parserSwapMaskEnabled)
        assertEquals(balanced, maximum.copy(parserSwapMaskEnabled = false))
    }

    @Test
    fun `settings resolve back to the preset that produced them`() {
        QualityPreset.selectable.forEach { preset ->
            assertEquals(preset, QualityPreset.of(requireNotNull(preset.settings)))
        }
    }

    @Test
    fun `a hand-tuned strength is reported as custom rather than attributed to a preset`() {
        val balanced = requireNotNull(QualityPreset.BALANCED.settings)

        val nudged = balanced.copy(restorationStrength = 0.5f)

        assertEquals(QualityPreset.CUSTOM, QualityPreset.of(nudged))
        assertEquals(QualityPreset.BALANCED, QualityPreset.of(balanced))
    }

    @Test
    fun `custom is never offered as a choice and never quotes a time`() {
        assertFalse(QualityPreset.CUSTOM in QualityPreset.selectable)
        assertNull(QualityPreset.CUSTOM.settings)
        assertNull(QualityPreset.CUSTOM.emulatorSecondsPerFace)
        QualityPreset.selectable.forEach { preset ->
            assertNotNull("${preset.name} must quote a measured time", preset.emulatorSecondsPerFace)
        }
    }

    @Test
    fun `quoted times follow the measured ordering`() {
        val fast = requireNotNull(QualityPreset.FAST.emulatorSecondsPerFace)
        val balanced = requireNotNull(QualityPreset.BALANCED.emulatorSecondsPerFace)
        val maximum = requireNotNull(QualityPreset.MAXIMUM.emulatorSecondsPerFace)

        // Measured 34.8 s, 80.9 s and 81.3 s: restoration roughly doubles the cost, and
        // the parser mask on top of it is noise.
        assertTrue("restoration must be the expensive axis", balanced > fast)
        assertEquals("the parser mask costs about 3 s, below the rounding", balanced, maximum)
    }
}
