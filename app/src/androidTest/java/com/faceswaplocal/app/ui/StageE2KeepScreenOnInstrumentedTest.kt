package com.faceswaplocal.app.ui

import android.app.Activity
import android.view.View
import android.view.ViewGroup
import androidx.compose.ui.test.junit4.createAndroidComposeRule
import androidx.test.ext.junit.runners.AndroidJUnit4
import com.faceswaplocal.app.domain.ProcessingProgress
import com.faceswaplocal.app.domain.ProcessingStage
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith

/** §9.4: the screen stays awake during a photo run and is released as soon as it ends. */
@RunWith(AndroidJUnit4::class)
class StageE2KeepScreenOnInstrumentedTest {
    @get:Rule
    val compose = createAndroidComposeRule<StageDUiTestActivity>()

    @Test
    fun screenIsHeldAwakeOnlyWhileProcessingRuns() {
        assertKeepScreenOn("an idle screen must not hold the device awake", expected = false)

        setPhase(PhotoSwapPhase.RUNNING)
        assertKeepScreenOn("a running swap must hold the screen awake", expected = true)

        setPhase(PhotoSwapPhase.CANCELLING)
        assertKeepScreenOn(
            "the screen stays awake until the safe boundary is reached",
            expected = true,
        )

        setPhase(PhotoSwapPhase.READY)
        assertKeepScreenOn("a finished run must release the screen", expected = false)
    }

    @Test
    fun anErrorAlsoReleasesTheScreen() {
        setPhase(PhotoSwapPhase.RUNNING)
        assertKeepScreenOn(expected = true)

        setPhase(PhotoSwapPhase.ERROR)
        assertKeepScreenOn("a failed run must not leave the screen pinned on", expected = false)
    }

    private fun setPhase(phase: PhotoSwapPhase) {
        compose.activityRule.scenario.onActivity { activity ->
            val viewModel = activity.faceSwapViewModel
            viewModel.injectUiStateForTest(
                viewModel.state.value.copy(
                    photoSwapPhase = phase,
                    processingProgress = ProcessingProgress(
                        stage = ProcessingStage.SWAPPING,
                        completedFaces = 0,
                        totalFaces = 2,
                        restorationPlanned = true,
                    ),
                ),
            )
        }
        compose.waitForIdle()
    }

    private fun assertKeepScreenOn(message: String = "", expected: Boolean) {
        compose.waitForIdle()
        var held = false
        compose.activityRule.scenario.onActivity { activity: Activity ->
            held = anyViewKeepsScreenOn(activity.window.decorView)
        }
        if (expected) assertTrue(message, held) else assertFalse(message, held)
    }

    /**
     * `View.setKeepScreenOn` is applied by the ViewRootImpl rather than as a window
     * layout flag, so the hierarchy itself is the observable state.
     */
    private fun anyViewKeepsScreenOn(view: View): Boolean {
        if (view.keepScreenOn) return true
        if (view !is ViewGroup) return false
        return (0 until view.childCount).any { index -> anyViewKeepsScreenOn(view.getChildAt(index)) }
    }
}
