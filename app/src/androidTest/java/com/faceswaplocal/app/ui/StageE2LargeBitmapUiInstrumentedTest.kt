package com.faceswaplocal.app.ui

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import androidx.compose.ui.test.assertIsDisplayed
import androidx.compose.ui.test.hasTestTag
import androidx.compose.ui.test.junit4.createAndroidComposeRule
import androidx.compose.ui.test.onNodeWithTag
import androidx.compose.ui.test.performClick
import androidx.compose.ui.test.performScrollToNode
import androidx.test.ext.junit.runners.AndroidJUnit4
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith

/**
 * Removing the 2 560 px decode cap means the before/after viewport now receives
 * full-resolution bitmaps. This checks that a 12 MP pair really renders instead of
 * hitting a texture limit, and that the export card reports the processed size.
 */
@RunWith(AndroidJUnit4::class)
class StageE2LargeBitmapUiInstrumentedTest {
    @get:Rule
    val compose = createAndroidComposeRule<StageDUiTestActivity>()

    /** Bitmap ownership stays with the ViewModel, which releases both on teardown. */
    @Test
    fun aTwelveMegapixelResultRendersInTheComparisonViewport() {
        compose.activityRule.scenario.onActivity { activity ->
            val viewModel = activity.faceSwapViewModel
            val current = viewModel.state.value
            viewModel.injectUiStateForTest(
                current.copy(
                    targetBitmap = photo(Color.rgb(30, 60, 90)),
                    targetSourceSize = ImageSize(WIDTH, HEIGHT),
                    photoSwapResult = current.photoSwapResult?.copy(
                        finalBitmap = photo(Color.rgb(120, 80, 40)),
                    ),
                ),
            )
        }
        compose.waitForIdle()

        scrollTo("comparison-after")
        compose.onNodeWithTag("comparison-after").assertIsDisplayed()
        compose.onNodeWithTag("compare-show-before").performClick()
        compose.waitForIdle()
        compose.onNodeWithTag("comparison-before").assertIsDisplayed()

        compose.activityRule.scenario.onActivity { activity ->
            val state = activity.faceSwapViewModel.state.value
            assertEquals(WIDTH, state.targetBitmap?.width)
            assertEquals(HEIGHT, state.targetBitmap?.height)
            assertEquals(WIDTH, state.photoSwapResult?.finalBitmap?.width)
            assertFalse(
                "a target decoded at its own size must not be flagged as downscaled",
                state.exportIsDownscaled,
            )
        }
    }

    @Test
    fun aDownscaledTargetIsAnnouncedBeforeSaving() {
        compose.activityRule.scenario.onActivity { activity ->
            val viewModel = activity.faceSwapViewModel
            val current = viewModel.state.value
            viewModel.injectUiStateForTest(
                current.copy(
                    targetBitmap = smallPhoto(),
                    targetSourceSize = ImageSize(WIDTH, HEIGHT),
                ),
            )
        }
        compose.waitForIdle()

        scrollTo("export-downscale-notice")
        compose.onNodeWithTag("export-downscale-notice").assertIsDisplayed()
        compose.activityRule.scenario.onActivity { activity ->
            val state = activity.faceSwapViewModel.state.value
            assertEquals(true, state.exportIsDownscaled)
        }
    }

    private fun photo(color: Int): Bitmap =
        Bitmap.createBitmap(WIDTH, HEIGHT, Bitmap.Config.ARGB_8888).also { bitmap ->
            Canvas(bitmap).apply {
                drawColor(color)
                drawCircle(
                    WIDTH / 2f,
                    HEIGHT / 2f,
                    HEIGHT / 4f,
                    Paint().apply { this.color = Color.WHITE },
                )
            }
        }

    private fun smallPhoto(): Bitmap =
        Bitmap.createBitmap(1_600, 1_200, Bitmap.Config.ARGB_8888).also { bitmap ->
            Canvas(bitmap).drawColor(Color.DKGRAY)
        }

    private fun scrollTo(tag: String) {
        compose.onNodeWithTag("main-scroll").performScrollToNode(hasTestTag(tag))
    }

    private companion object {
        const val WIDTH = 4_000
        const val HEIGHT = 3_000
    }
}
