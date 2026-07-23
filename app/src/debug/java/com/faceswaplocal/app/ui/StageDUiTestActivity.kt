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
import com.faceswaplocal.app.ui.theme.FaceSwapLocalTheme

class StageDUiTestActivity : ComponentActivity() {
    val faceSwapViewModel: FaceSwapViewModel by viewModels()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        if (faceSwapViewModel.state.value.phase != AnalysisPhase.MAPPING) {
            val sources = (1..3).map { face("source-$it", it * 0.1f) }
            val targets = (1..4).map { face("target-$it", it * 0.08f) }
            val sourceBitmaps = sources.associate { it.id to bitmap(0xff445566.toInt()) }
            faceSwapViewModel.injectUiStateForTest(
                FaceSwapUiState(
                    sourceBitmap = sourceBitmaps.getValue(sources.first().id),
                    targetBitmap = bitmap(0xff223344.toInt()),
                    sourceFaces = sources,
                    sourceBitmaps = sourceBitmaps,
                    targetFaces = targets,
                    assignments = FaceAssignmentPlanner.defaults(sources, targets),
                    phase = AnalysisPhase.MAPPING,
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
                    onRunPhotoSwap = {},
                    onDismissError = {},
                )
            }
        }
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
}
