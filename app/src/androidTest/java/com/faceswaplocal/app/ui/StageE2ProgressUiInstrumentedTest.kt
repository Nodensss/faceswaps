package com.faceswaplocal.app.ui

import androidx.compose.ui.test.assertIsDisplayed
import androidx.compose.ui.test.assertIsEnabled
import androidx.compose.ui.test.assertIsNotEnabled
import androidx.compose.ui.test.assertIsOff
import androidx.compose.ui.test.assertIsOn
import androidx.compose.ui.test.assertIsSelected
import androidx.compose.ui.test.assertTextContains
import androidx.compose.ui.test.hasTestTag
import androidx.compose.ui.test.junit4.createAndroidComposeRule
import androidx.compose.ui.test.onNodeWithTag
import androidx.compose.ui.test.performClick
import androidx.compose.ui.test.performScrollToNode
import androidx.test.ext.junit.runners.AndroidJUnit4
import com.faceswaplocal.app.domain.ExportFormat
import com.faceswaplocal.app.domain.ExportSettings
import com.faceswaplocal.app.domain.ProcessingProgress
import com.faceswaplocal.app.domain.ProcessingStage
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith

/** FR-PHOTO-07 progress/cancel surface and FR-PHOTO-09 export controls on API 35. */
@RunWith(AndroidJUnit4::class)
class StageE2ProgressUiInstrumentedTest {
    @get:Rule
    val compose = createAndroidComposeRule<StageDUiTestActivity>()

    @Test
    fun runningStateShowsStageAndFaceNumberAndBlocksASecondLaunch() {
        setRunning(ProcessingStage.SWAPPING, completedFaces = 1, totalFaces = 3)

        scrollTo("processing-status")
        compose.onNodeWithTag("processing-status")
            .assertTextContains("Замена лица", substring = true)
            .assertTextContains("лицо 2 из 3", substring = true)
        compose.onNodeWithTag("processing-progress").assertIsDisplayed()
        scrollTo("run-photo-swap")
        compose.onNodeWithTag("run-photo-swap").assertIsNotEnabled()
        scrollTo("cancel-photo-swap")
        compose.onNodeWithTag("cancel-photo-swap").assertIsEnabled()
        scrollTo("export-save")
        compose.onNodeWithTag("export-save")
            .assertIsNotEnabled()
    }

    @Test
    fun restorationStageIsNamedAndSurvivesActivityRecreation() {
        setRunning(ProcessingStage.RESTORING, completedFaces = 0, totalFaces = 2)

        scrollTo("processing-status")
        compose.onNodeWithTag("processing-status")
            .assertTextContains("Восстановление деталей", substring = true)
            .assertTextContains("лицо 1 из 2", substring = true)

        compose.activityRule.scenario.recreate()
        compose.waitForIdle()

        // The ViewModel outlives the Activity, so a rotation cannot restart processing.
        compose.activityRule.scenario.onActivity { activity ->
            val state = activity.faceSwapViewModel.state.value
            assertEquals(PhotoSwapPhase.RUNNING, state.photoSwapPhase)
            assertEquals(ProcessingStage.RESTORING, state.processingProgress?.stage)
            assertEquals(1, state.processingProgress?.currentFace)
        }
        scrollTo("processing-status")
        compose.onNodeWithTag("processing-status")
            .assertTextContains("лицо 1 из 2", substring = true)
        scrollTo("run-photo-swap")
        compose.onNodeWithTag("run-photo-swap").assertIsNotEnabled()
    }

    @Test
    fun cancellingStateDisablesBothActionsUntilTheSafeBoundary() {
        compose.activityRule.scenario.onActivity { activity ->
            val current = activity.faceSwapViewModel.state.value
            activity.faceSwapViewModel.injectUiStateForTest(
                current.copy(photoSwapPhase = PhotoSwapPhase.CANCELLING),
            )
        }
        compose.waitForIdle()

        scrollTo("processing-status")
        compose.onNodeWithTag("processing-status")
            .assertTextContains("Отмена", substring = true)
        scrollTo("cancel-photo-swap")
        compose.onNodeWithTag("cancel-photo-swap").assertIsNotEnabled()
        scrollTo("run-photo-swap")
        compose.onNodeWithTag("run-photo-swap").assertIsNotEnabled()
    }

    @Test
    fun exportDefaultsAreJpegWithTheWatermarkOnAndSurviveRecreation() {
        scrollTo("export-watermark")
        compose.onNodeWithTag("export-watermark").assertIsOn()
        scrollTo("export-format-jpeg")
        compose.onNodeWithTag("export-format-jpeg").assertIsSelected()
        scrollTo("export-quality-label")
        compose.onNodeWithTag("export-quality-label")
            .assertTextContains("${ExportSettings.DEFAULT_JPEG_QUALITY}", substring = true)
        scrollTo("export-jpeg-quality")
        compose.onNodeWithTag("export-jpeg-quality").assertIsEnabled()

        scrollTo("export-format-png")
        compose.onNodeWithTag("export-format-png").performClick()
        scrollTo("export-watermark")
        compose.onNodeWithTag("export-watermark").performClick()
        compose.waitForIdle()

        scrollTo("export-jpeg-quality")
        compose.onNodeWithTag("export-jpeg-quality")
            .assertIsNotEnabled()
        scrollTo("export-quality-label")
        compose.onNodeWithTag("export-quality-label")
            .assertTextContains("без потерь", substring = true)

        compose.activityRule.scenario.recreate()
        compose.waitForIdle()

        compose.activityRule.scenario.onActivity { activity ->
            val settings = activity.faceSwapViewModel.state.value.exportSettings
            assertEquals(ExportFormat.PNG, settings.format)
            assertEquals(false, settings.watermarkEnabled)
            assertEquals(ExportSettings.DEFAULT_JPEG_QUALITY, settings.jpegQuality)
        }
        scrollTo("export-format-png")
        compose.onNodeWithTag("export-format-png").assertIsSelected()
        scrollTo("export-watermark")
        compose.onNodeWithTag("export-watermark").assertIsOff()
    }

    @Test
    fun exportSettingsAreLockedWhileTheExportRuns() {
        compose.activityRule.scenario.onActivity { activity ->
            val viewModel = activity.faceSwapViewModel
            viewModel.injectUiStateForTest(
                viewModel.state.value.copy(exportPhase = ExportPhase.RUNNING),
            )
            viewModel.setExportFormat(ExportFormat.PNG)
            viewModel.setWatermarkEnabled(false)
            viewModel.setJpegQuality(60)
        }
        compose.waitForIdle()

        compose.activityRule.scenario.onActivity { activity ->
            val state = activity.faceSwapViewModel.state.value
            assertEquals(ExportSettings(), state.exportSettings)
            assertNull(state.savedExport)
        }
        scrollTo("export-format-png")
        compose.onNodeWithTag("export-format-png").assertIsNotEnabled()
        scrollTo("export-watermark")
        compose.onNodeWithTag("export-watermark").assertIsNotEnabled()
        scrollTo("export-save")
        compose.onNodeWithTag("export-save").assertIsNotEnabled()
    }

    private fun setRunning(stage: ProcessingStage, completedFaces: Int, totalFaces: Int) {
        compose.activityRule.scenario.onActivity { activity ->
            val current = activity.faceSwapViewModel.state.value
            activity.faceSwapViewModel.injectUiStateForTest(
                current.copy(
                    photoSwapPhase = PhotoSwapPhase.RUNNING,
                    processingProgress = ProcessingProgress(
                        stage = stage,
                        completedFaces = completedFaces,
                        totalFaces = totalFaces,
                        restorationPlanned = true,
                    ),
                ),
            )
        }
        compose.waitForIdle()
    }

    private fun scrollTo(tag: String) {
        compose.onNodeWithTag("main-scroll").performScrollToNode(hasTestTag(tag))
    }
}
