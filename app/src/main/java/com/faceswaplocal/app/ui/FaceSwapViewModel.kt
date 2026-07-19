package com.faceswaplocal.app.ui

import android.app.Application
import android.graphics.Bitmap
import android.net.Uri
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.faceswaplocal.app.data.BitmapLoader
import com.faceswaplocal.app.data.MlKitLocalFaceDetector
import com.faceswaplocal.app.domain.DetectedFace
import com.faceswaplocal.app.domain.FaceAssignmentPlanner
import com.faceswaplocal.app.domain.FaceId
import com.faceswaplocal.app.domain.SwapAssignment
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch

enum class AnalysisPhase {
    WAITING_FOR_MEDIA,
    READY,
    ANALYZING,
    MAPPING,
    ERROR,
}

data class FaceSwapUiState(
    val sourceUri: Uri? = null,
    val targetUri: Uri? = null,
    val sourceBitmap: Bitmap? = null,
    val targetBitmap: Bitmap? = null,
    val sourceFaces: List<DetectedFace> = emptyList(),
    val targetFaces: List<DetectedFace> = emptyList(),
    val assignments: List<SwapAssignment> = emptyList(),
    val phase: AnalysisPhase = AnalysisPhase.WAITING_FOR_MEDIA,
    val errorMessage: String? = null,
) {
    val canAnalyze: Boolean
        get() = sourceUri != null && targetUri != null && phase != AnalysisPhase.ANALYZING
}

class FaceSwapViewModel(application: Application) : AndroidViewModel(application) {
    private val bitmapLoader = BitmapLoader(application.contentResolver)
    private val faceDetector = MlKitLocalFaceDetector()
    private val mutableState = MutableStateFlow(FaceSwapUiState())
    private var analysisJob: Job? = null

    val state: StateFlow<FaceSwapUiState> = mutableState.asStateFlow()

    fun selectSource(uri: Uri) {
        analysisJob?.cancel()
        val current = mutableState.value
        resetSelections(sourceUri = uri, targetUri = current.targetUri)
    }

    fun selectTarget(uri: Uri) {
        analysisJob?.cancel()
        val current = mutableState.value
        resetSelections(sourceUri = current.sourceUri, targetUri = uri)
    }

    fun analyze() {
        val sourceUri = mutableState.value.sourceUri ?: return
        val targetUri = mutableState.value.targetUri ?: return

        analysisJob?.cancel()
        analysisJob = viewModelScope.launch {
            mutableState.update {
                it.copy(
                    sourceBitmap = null,
                    targetBitmap = null,
                    sourceFaces = emptyList(),
                    targetFaces = emptyList(),
                    assignments = emptyList(),
                    phase = AnalysisPhase.ANALYZING,
                    errorMessage = null,
                )
            }

            try {
                val sourceBitmap = bitmapLoader.load(sourceUri)
                val targetBitmap = bitmapLoader.load(targetUri)
                val sourceFaces = faceDetector.detect(sourceBitmap, idPrefix = "source")
                val targetFaces = faceDetector.detect(targetBitmap, idPrefix = "target")

                require(sourceFaces.isNotEmpty()) {
                    "На фотографии-источнике лицо не найдено. Выберите более чёткий снимок."
                }
                require(targetFaces.isNotEmpty()) {
                    "На целевой фотографии лицо не найдено. Выберите другой снимок."
                }

                mutableState.update {
                    it.copy(
                        sourceBitmap = sourceBitmap,
                        targetBitmap = targetBitmap,
                        sourceFaces = sourceFaces,
                        targetFaces = targetFaces,
                        assignments = FaceAssignmentPlanner.defaults(sourceFaces, targetFaces),
                        phase = AnalysisPhase.MAPPING,
                    )
                }
            } catch (cancelled: CancellationException) {
                throw cancelled
            } catch (error: Throwable) {
                mutableState.update {
                    it.copy(
                        phase = AnalysisPhase.ERROR,
                        errorMessage = error.message
                            ?: "Не удалось обработать фотографии. Попробуйте другие файлы.",
                    )
                }
            }
        }
    }

    fun assignSource(targetFaceId: FaceId, sourceFaceId: FaceId) {
        mutableState.update {
            it.copy(
                assignments = FaceAssignmentPlanner.replaceSource(
                    assignments = it.assignments,
                    targetFaceId = targetFaceId,
                    sourceFaceId = sourceFaceId,
                ),
            )
        }
    }

    fun dismissError() {
        mutableState.update {
            it.copy(
                phase = if (it.sourceUri != null && it.targetUri != null) {
                    AnalysisPhase.READY
                } else {
                    AnalysisPhase.WAITING_FOR_MEDIA
                },
                errorMessage = null,
            )
        }
    }

    private fun resetSelections(sourceUri: Uri?, targetUri: Uri?) {
        mutableState.value = FaceSwapUiState(
            sourceUri = sourceUri,
            targetUri = targetUri,
            phase = if (sourceUri != null && targetUri != null) {
                AnalysisPhase.READY
            } else {
                AnalysisPhase.WAITING_FOR_MEDIA
            },
        )
    }

    override fun onCleared() {
        faceDetector.close()
        super.onCleared()
    }
}

