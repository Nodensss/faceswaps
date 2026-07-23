package com.faceswaplocal.app.ui

import androidx.compose.ui.test.assertIsDisplayed
import androidx.compose.ui.test.hasTestTag
import androidx.compose.ui.test.junit4.createAndroidComposeRule
import androidx.compose.ui.test.onNodeWithTag
import androidx.compose.ui.test.performClick
import androidx.compose.ui.test.performScrollTo
import androidx.compose.ui.test.performScrollToNode
import androidx.test.ext.junit.runners.AndroidJUnit4
import com.faceswaplocal.app.domain.FaceId
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class StageDComposeUiInstrumentedTest {
    @get:Rule
    val compose = createAndroidComposeRule<StageDUiTestActivity>()

    @Test
    fun assignmentsConfirmationInvalidationAndRecreateAreStable() {
        compose.onNodeWithTag("target-0").performScrollTo().assertIsDisplayed()
        compose.onNodeWithTag("target-3").performScrollTo().assertIsDisplayed()
        compose.onNodeWithTag("assign-1-1").performScrollTo().performClick()
        compose.onNodeWithTag("assign-2-2").performScrollTo().performClick()
        compose.onNodeWithTag("unchanged-3").performScrollTo().performClick()
        assertSource("target-1", "source-1")
        assertSource("target-2", "source-2")
        assertSource("target-3", "source-3")
        assertSource("target-4", null)

        scrollTo("apply-all-0")
        compose.onNodeWithTag("apply-all-0").performClick()
        compose.onNodeWithTag("apply-all-dialog").fetchSemanticsNode()
        compose.onNodeWithTag("confirm-apply-all").performClick()
        (1..4).forEach { assertSource("target-$it", "source-1") }

        compose.onNodeWithTag("assign-1-1").performScrollTo().performClick()
        scrollTo("remove-source-1")
        compose.onNodeWithTag("remove-sources-row").performScrollToNode(hasTestTag("remove-source-1"))
        compose.onNodeWithTag("remove-source-1").fetchSemanticsNode()
        compose.activityRule.scenario.onActivity {
            it.faceSwapViewModel.removeSource(FaceId("source-2"))
        }
        compose.waitForIdle()
        assertSource("target-2", null)
        compose.activityRule.scenario.recreate()
        compose.waitForIdle()
        assertSource("target-1", "source-1")
        assertSource("target-2", null)
        compose.onNodeWithTag("target-3").performScrollTo().assertIsDisplayed()
    }

    private fun assertSource(target: String, expected: String?) {
        compose.activityRule.scenario.onActivity { activity ->
            val assignment = activity.faceSwapViewModel.state.value.assignments
                .firstOrNull { it.targetFaceId == FaceId(target) }
            if (expected == null) {
                assertNull(assignment)
            } else {
                assertEquals(FaceId(expected), assignment?.sourceFaceId)
            }
        }
    }

    private fun scrollTo(tag: String) {
        compose.onNodeWithTag("main-scroll").performScrollToNode(hasTestTag(tag))
    }
}
