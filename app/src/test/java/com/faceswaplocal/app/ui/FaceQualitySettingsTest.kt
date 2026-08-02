package com.faceswaplocal.app.ui

import androidx.lifecycle.SavedStateHandle
import com.faceswaplocal.app.domain.FaceQualitySettings
import com.faceswaplocal.app.inference.ModelId
import com.faceswaplocal.app.inference.SwapBlendMaskMode
import kotlinx.coroutines.CancellationException
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Test

class FaceQualitySettingsTest {
    @Test
    fun `defaults enable restoration at point eight and parser swap mask`() {
        val settings = FaceQualitySettings()

        assertTrue(settings.restorationEnabled)
        assertEquals(0.8f, settings.restorationStrength, 0f)
        assertEquals(0.8f, settings.effectiveRestorationStrength, 0f)
        assertTrue(settings.parserSwapMaskEnabled)
        assertEquals(SwapBlendMaskMode.PARSER_REGION, settings.swapBlendMaskMode())
    }

    @Test
    fun `disabled restoration passes effective zero and affine mode follows parser toggle`() {
        val settings = FaceQualitySettings(
            restorationEnabled = false,
            restorationStrength = 0.8f,
            parserSwapMaskEnabled = false,
        )

        assertEquals(0f, settings.effectiveRestorationStrength, 0f)
        assertEquals(SwapBlendMaskMode.AFFINE_BOX, settings.swapBlendMaskMode())
    }

    @Test
    fun `persisted values are sanitized and round trip through SavedStateHandle primitives`() {
        val handle = SavedStateHandle()
        val expected = FaceQualitySettings(
            restorationEnabled = false,
            restorationStrength = 0.4f,
            parserSwapMaskEnabled = false,
        )

        FaceQualitySettingsSavedState.write(handle, expected)

        assertEquals(expected, FaceQualitySettingsSavedState.read(handle))
        assertEquals(
            FaceQualitySettings(),
            FaceQualitySettings.fromPersisted(
                restorationEnabled = null,
                restorationStrength = Float.NaN,
                parserSwapMaskEnabled = null,
            ),
        )
        assertEquals(
            0.8f,
            FaceQualitySettings.fromPersisted(true, 1.01f, true).restorationStrength,
            0f,
        )
        assertThrows(IllegalArgumentException::class.java) {
            FaceQualitySettings(restorationStrength = Float.POSITIVE_INFINITY)
        }
    }

    @Test
    fun `required models follow parser and effective restoration independently`() {
        val base = setOf(
            ModelId.YOLOFACE_8N,
            ModelId.ARCFACE_W600K_R50,
            ModelId.INSWAPPER_128_FP16,
        )

        val fastBox = FaceQualitySettings(
            restorationEnabled = false,
            parserSwapMaskEnabled = false,
        )
        val parserOnly = fastBox.copy(parserSwapMaskEnabled = true)
        val restorationOnly = fastBox.copy(restorationEnabled = true)
        val both = restorationOnly.copy(parserSwapMaskEnabled = true)

        assertEquals(base, fastBox.requiredModelIds())
        assertEquals(base + ModelId.BISENET_RESNET_34, parserOnly.requiredModelIds())
        assertEquals(
            base + setOf(ModelId.BISENET_RESNET_34, ModelId.GFPGAN_1_4),
            restorationOnly.requiredModelIds(),
        )
        assertEquals(restorationOnly.requiredModelIds(), both.requiredModelIds())
        assertFalse(ModelId.GFPGAN_1_4 in parserOnly.requiredModelIds())
        assertTrue(ModelId.BISENET_RESNET_34 in restorationOnly.requiredModelIds())
    }

    @Test
    fun `owned result is released when cancellation wins before publication`() {
        val released = mutableListOf<String>()
        var published = false

        assertThrows(CancellationException::class.java) {
            publishOrReleaseOwnedResult(
                result = "unpublished",
                checkCanPublish = { throw CancellationException("late cancellation") },
                publish = { published = true },
                release = released::add,
            )
        }

        assertFalse(published)
        assertEquals(listOf("unpublished"), released)
    }

    @Test
    fun `published result is not released by ownership guard`() {
        val released = mutableListOf<String>()
        var published: String? = null

        publishOrReleaseOwnedResult(
            result = "published",
            checkCanPublish = {},
            publish = { published = it },
            release = released::add,
        )

        assertEquals("published", published)
        assertTrue(released.isEmpty())
    }
}
