package com.faceswaplocal.app.ui

import android.graphics.Bitmap
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.viewModels
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.faceswaplocal.app.domain.DetectedFace
import com.faceswaplocal.app.domain.FaceAssignmentPlanner
import com.faceswaplocal.app.domain.FaceId
import com.faceswaplocal.app.domain.NormalizedRect
import com.faceswaplocal.app.domain.ProcessingProgress
import com.faceswaplocal.app.domain.ProcessingStage
import com.faceswaplocal.app.inference.InferenceBackend
import com.faceswaplocal.app.inference.MultiPhotoFaceSwapResult
import com.faceswaplocal.app.inference.PhotoFaceSwapTimings
import com.faceswaplocal.app.ui.theme.FaceSwapLocalTheme

class StageDUiTestActivity : ComponentActivity() {
    val faceSwapViewModel: FaceSwapViewModel by viewModels()
    lateinit var initialResultBitmap: Bitmap
        private set

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        // `-e stage e2-progress` renders the running screen for report screenshots.
        val forcedProgressStage = intent?.getStringExtra(EXTRA_PROGRESS_STAGE)
        if (faceSwapViewModel.state.value.phase != AnalysisPhase.MAPPING) {
            val sources = (1..3).map { face("source-$it", it * 0.1f) }
            val targets = (1..4).map { face("target-$it", it * 0.08f) }
            val sourceBitmaps = sources.associate { it.id to bitmap(0xff445566.toInt()) }
            val targetBitmap = bitmap(0xff223344.toInt())
            initialResultBitmap = bitmap(0xff8899aa.toInt())
            faceSwapViewModel.injectUiStateForTest(
                FaceSwapUiState(
                    sourceBitmap = sourceBitmaps.getValue(sources.first().id),
                    targetBitmap = targetBitmap,
                    sourceFaces = sources,
                    sourceBitmaps = sourceBitmaps,
                    targetFaces = targets,
                    assignments = FaceAssignmentPlanner.defaults(sources, targets),
                    phase = AnalysisPhase.MAPPING,
                    photoSwapPhase = PhotoSwapPhase.READY,
                    photoSwapResult = fakeResult(initialResultBitmap),
                ),
            )
        }
        forcedProgressStage?.let { stageName ->
            val stage = ProcessingStage.valueOf(stageName)
            faceSwapViewModel.injectUiStateForTest(
                faceSwapViewModel.state.value.copy(
                    photoSwapPhase = PhotoSwapPhase.RUNNING,
                    processingProgress = ProcessingProgress(
                        stage = stage,
                        completedFaces = 1,
                        totalFaces = 3,
                        restorationPlanned = true,
                    ),
                ),
            )
        }
        setContent {
            val state = faceSwapViewModel.state.collectAsStateWithLifecycle().value
            FaceSwapLocalTheme {
                FaceSwapScreen(
                    state = state,
                    onPickSource = {},
                    onPickTarget = {},
                    onImportModel = {},
                    onAnalyze = {},
                    onAssignSource = faceSwapViewModel::assignSource,
                    onSetUnchanged = faceSwapViewModel::setUnchanged,
                    onApplySourceToAll = faceSwapViewModel::applySourceToAll,
                    onRemoveSource = faceSwapViewModel::removeSource,
                    onQualityPresetChange = faceSwapViewModel::setQualityPreset,
                    onRestorationEnabledChange = faceSwapViewModel::setRestorationEnabled,
                    onRestorationStrengthChange = faceSwapViewModel::setRestorationStrength,
                    onParserSwapMaskEnabledChange = faceSwapViewModel::setParserSwapMaskEnabled,
                    onRunPhotoSwap = {},
                    onCancelPhotoSwap = faceSwapViewModel::cancelPhotoSwap,
                    onExportFormatChange = faceSwapViewModel::setExportFormat,
                    onJpegQualityChange = faceSwapViewModel::setJpegQuality,
                    onWatermarkEnabledChange = faceSwapViewModel::setWatermarkEnabled,
                    onExport = faceSwapViewModel::exportResult,
                    onDismissExportError = faceSwapViewModel::dismissExportError,
                    onPhotoResultDisposed = faceSwapViewModel::onPhotoResultDisposed,
                    onDismissError = {},
                )
            }
        }
    }

    companion object {
        const val EXTRA_PROGRESS_STAGE = "stage_e2_progress_stage"
    }

    private fun face(id: String, offset: Float) = DetectedFace(
        FaceId(id),
        NormalizedRect(offset, offset, (offset + 0.2f).coerceAtMost(1f), (offset + 0.25f).coerceAtMost(1f)),
        0f,
        0f,
        null,
    )

    private fun bitmap(color: Int) =
        Bitmap.createBitmap(IntArray(64 * 64) { color }, 64, 64, Bitmap.Config.ARGB_8888)

    private fun fakeResult(bitmap: Bitmap) = MultiPhotoFaceSwapResult(
        finalBitmap = bitmap,
        swapRois = emptyList(),
        enhanceRois = emptyList(),
        detectorBackend = InferenceBackend.CPU,
        recognizerBackend = InferenceBackend.CPU,
        swapperBackend = InferenceBackend.CPU,
        enhancerBackends = listOf(InferenceBackend.CPU),
        swapParserBackends = listOf(InferenceBackend.CPU),
        enhancementParserBackends = listOf(InferenceBackend.CPU),
        protectedUnassignedRois = emptyList(),
        restorationStrength = 0.8f,
        swapParserMs = 12L,
        enhancementMs = 34L,
        timings = PhotoFaceSwapTimings(
            detectorMs = 1L,
            recognizerMs = 2L,
            swapperMs = 3L,
            parserMs = 4L,
            compositingMs = 5L,
            totalMs = 45L,
        ),
    )
}
