package com.faceswaplocal.app.ui

import android.graphics.Bitmap
import androidx.compose.ui.semantics.SemanticsActions
import androidx.compose.ui.test.assertIsDisplayed
import androidx.compose.ui.test.assertIsEnabled
import androidx.compose.ui.test.assertIsNotEnabled
import androidx.compose.ui.test.assertIsOff
import androidx.compose.ui.test.assertIsOn
import androidx.compose.ui.test.assertTextContains
import androidx.compose.ui.test.hasTestTag
import androidx.compose.ui.test.junit4.createAndroidComposeRule
import androidx.compose.ui.test.onNodeWithTag
import androidx.compose.ui.test.performClick
import androidx.compose.ui.test.performScrollTo
import androidx.compose.ui.test.performScrollToNode
import androidx.compose.ui.test.performSemanticsAction
import androidx.test.ext.junit.runners.AndroidJUnit4
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class StageEFaceQualityUiInstrumentedTest {
    @get:Rule
    val compose = createAndroidComposeRule<StageDUiTestActivity>()

    @Test
    fun settingsInvalidateAndRetireOldResultThenSurviveRecreate() {
        lateinit var oldResult: Bitmap
        compose.activityRule.scenario.onActivity { activity ->
            oldResult = activity.initialResultBitmap
        }
        compose.onNodeWithTag("comparison-after").fetchSemanticsNode()

        scrollTo("restoration-enabled")
        compose.onNodeWithTag("restoration-enabled").assertIsOn().performClick()
        compose.waitUntil(timeoutMillis = 5_000) { oldResult.isRecycled }
        compose.onNodeWithTag("restoration-enabled").assertIsOff().performClick()

        scrollTo("restoration-strength")
        compose.onNodeWithTag("restoration-strength")
            .assertIsEnabled()
            .performSemanticsAction(SemanticsActions.SetProgress) { setProgress ->
                assertTrue(setProgress(0.4f))
            }
        scrollTo("parser-swap-mask-enabled")
        compose.onNodeWithTag("parser-swap-mask-enabled").assertIsOn().performClick()
        compose.waitForIdle()

        assertSettings(restorationEnabled = true, strength = 0.4f, parserEnabled = false)
        compose.activityRule.scenario.recreate()
        compose.waitForIdle()
        assertSettings(restorationEnabled = true, strength = 0.4f, parserEnabled = false)
        scrollTo("parser-swap-mask-enabled")
        compose.onNodeWithTag("parser-swap-mask-enabled").assertIsOff()
        scrollTo("restoration-strength-label")
        compose.onNodeWithTag("restoration-strength-label")
            .assertTextContains("40%", substring = true)
    }

    @Test
    fun beforeAndAfterToggleUsesOneResultViewportAndReportsAppliedSettings() {
        scrollTo("comparison-after")
        compose.onNodeWithTag("comparison-after").assertIsDisplayed()
        compose.onNodeWithTag("compare-show-before").performScrollTo().performClick()
        compose.onNodeWithTag("comparison-before").assertIsDisplayed()
        compose.onNodeWithTag("compare-show-after").performScrollTo().performClick()
        compose.onNodeWithTag("comparison-after").assertIsDisplayed()

        scrollTo("applied-quality-settings")
        compose.onNodeWithTag("applied-quality-settings")
            .assertTextContains("80%", substring = true)
            .assertTextContains("включена", substring = true)
    }

    @Test
    fun qualityControlsAndViewModelMutationsAreDisabledWhileRunning() {
        compose.activityRule.scenario.onActivity { activity ->
            val current = activity.faceSwapViewModel.state.value
            activity.faceSwapViewModel.injectUiStateForTest(
                current.copy(photoSwapPhase = PhotoSwapPhase.RUNNING),
            )
            activity.faceSwapViewModel.setRestorationEnabled(false)
            activity.faceSwapViewModel.setRestorationStrength(0.2f)
            activity.faceSwapViewModel.setParserSwapMaskEnabled(false)
        }
        compose.waitForIdle()

        scrollTo("restoration-enabled")
        compose.onNodeWithTag("restoration-enabled").assertIsNotEnabled().assertIsOn()
        scrollTo("restoration-strength")
        compose.onNodeWithTag("restoration-strength").assertIsNotEnabled()
        scrollTo("parser-swap-mask-enabled")
        compose.onNodeWithTag("parser-swap-mask-enabled").assertIsNotEnabled().assertIsOn()
        assertSettings(restorationEnabled = true, strength = 0.8f, parserEnabled = true)
    }

    private fun assertSettings(
        restorationEnabled: Boolean,
        strength: Float,
        parserEnabled: Boolean,
    ) {
        compose.activityRule.scenario.onActivity { activity ->
            val settings = activity.faceSwapViewModel.state.value.qualitySettings
            assertEquals(restorationEnabled, settings.restorationEnabled)
            assertEquals(strength, settings.restorationStrength, 0.001f)
            assertEquals(parserEnabled, settings.parserSwapMaskEnabled)
            assertFalse(settings.restorationStrength.isNaN())
        }
    }

    private fun scrollTo(tag: String) {
        compose.onNodeWithTag("main-scroll").performScrollToNode(hasTestTag(tag))
    }
}
