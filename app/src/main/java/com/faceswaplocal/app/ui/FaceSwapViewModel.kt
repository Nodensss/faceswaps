package com.faceswaplocal.app.ui

import android.app.Application
import android.graphics.Bitmap
import android.net.Uri
import android.os.Handler
import android.os.Looper
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.faceswaplocal.app.data.BitmapLoader
import com.faceswaplocal.app.data.MlKitLocalFaceDetector
import com.faceswaplocal.app.domain.DetectedFace
import com.faceswaplocal.app.domain.FaceAssignmentPlanner
import com.faceswaplocal.app.domain.FaceId
import com.faceswaplocal.app.domain.SwapAssignment
import com.faceswaplocal.app.inference.ModelCatalog
import com.faceswaplocal.app.inference.ModelId
import com.faceswaplocal.app.inference.ModelImportResult
import com.faceswaplocal.app.inference.ModelStatus
import com.faceswaplocal.app.inference.ModelStore
import com.faceswaplocal.app.inference.OnnxRawFaceSwapPipeline
import com.faceswaplocal.app.inference.RawFaceSwapRequest
import com.faceswaplocal.app.inference.RawFaceSwapResult
import com.faceswaplocal.app.inference.RequestedInferenceBackend
import com.faceswaplocal.app.inference.SwapperModel
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Job
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.flow.collectLatest
import kotlinx.coroutines.launch

enum class AnalysisPhase {
    WAITING_FOR_MEDIA,
    READY,
    ANALYZING,
    MAPPING,
    ERROR,
}

enum class RawSwapPhase {
    IDLE,
    RUNNING,
    READY,
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
    val modelStatuses: Map<ModelId, ModelStatus> = ModelCatalog.all.associate { it.id to ModelStatus.Missing },
    val modelMessage: String? = null,
    // HyperSwap remains the primary candidate, but the verified InSwapper fallback is
    // the Stage B default because it transfers source identity visibly on all parity pairs.
    val selectedSwapper: SwapperModel = SwapperModel.INSWAPPER_128_FP16,
    val rawSwapPhase: RawSwapPhase = RawSwapPhase.IDLE,
    val rawSwapResult: RawFaceSwapResult? = null,
    val rawSwapError: String? = null,
) {
    val canAnalyze: Boolean
        get() = sourceUri != null && targetUri != null && phase != AnalysisPhase.ANALYZING

    val canRunRawSwap: Boolean
        get() {
            val requiredModels = setOf(
                ModelId.YOLOFACE_8N,
                ModelId.ARCFACE_W600K_R50,
                selectedSwapper.modelId,
            )
            return phase == AnalysisPhase.MAPPING &&
                rawSwapPhase != RawSwapPhase.RUNNING &&
                sourceBitmap != null &&
                targetBitmap != null &&
                requiredModels.all { modelStatuses[it] is ModelStatus.Ready }
        }
}

class FaceSwapViewModel(application: Application) : AndroidViewModel(application) {
    private val bitmapLoader = BitmapLoader(application.contentResolver)
    private val faceDetector = MlKitLocalFaceDetector()
    private val modelStore = ModelStore(application)
    private val rawSwapPipeline = OnnxRawFaceSwapPipeline(modelStore)
    private val mutableState = MutableStateFlow(FaceSwapUiState())
    private val mainHandler = Handler(Looper.getMainLooper())
    private var analysisJob: Job? = null
    private var rawSwapJob: Job? = null

    val state: StateFlow<FaceSwapUiState> = mutableState.asStateFlow()

    init {
        viewModelScope.launch {
            modelStore.statuses.collectLatest { statuses ->
                mutableState.update { it.copy(modelStatuses = statuses) }
            }
        }
        viewModelScope.launch {
            modelStore.cleanupInterruptedImports()
            modelStore.refreshStatuses()
        }
    }

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
        val runningRawSwap = rawSwapJob
        runningRawSwap?.cancel()
        val previousState = mutableState.value
        mutableState.update {
            it.copy(
                sourceBitmap = null,
                targetBitmap = null,
                sourceFaces = emptyList(),
                targetFaces = emptyList(),
                assignments = emptyList(),
                phase = AnalysisPhase.ANALYZING,
                errorMessage = null,
                rawSwapPhase = RawSwapPhase.IDLE,
                rawSwapResult = null,
                rawSwapError = null,
            )
        }
        releaseRawResult(previousState.rawSwapResult)
        releaseInputBitmapsAfter(
            job = runningRawSwap,
            source = previousState.sourceBitmap,
            target = previousState.targetBitmap,
        )

        analysisJob = viewModelScope.launch {
            var decodedSource: Bitmap? = null
            var decodedTarget: Bitmap? = null
            var stateOwnsDecodedBitmaps = false
            try {
                val sourceBitmap = bitmapLoader.load(sourceUri).also { decodedSource = it }
                val targetBitmap = bitmapLoader.load(targetUri).also { decodedTarget = it }
                val sourceFaces = faceDetector.detect(sourceBitmap, idPrefix = "source")
                val targetFaces = faceDetector.detect(targetBitmap, idPrefix = "target")

                require(sourceFaces.isNotEmpty()) {
                    "На фотографии-источнике лицо не найдено. Выберите более чёткий снимок."
                }
                require(targetFaces.isNotEmpty()) {
                    "На целевой фотографии лицо не найдено. Выберите другой снимок."
                }

                currentCoroutineContext().ensureActive()
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
                stateOwnsDecodedBitmaps = true
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
            } finally {
                if (!stateOwnsDecodedBitmaps) {
                    releaseInputBitmaps(decodedSource, decodedTarget)
                }
            }
        }
    }

    fun importModel(id: ModelId, uri: Uri) {
        viewModelScope.launch {
            mutableState.update {
                it.copy(modelMessage = "Проверяю размер и SHA-256 файла ${id.stableId}…")
            }
            val message = when (val result = modelStore.importModel(id, uri)) {
                is ModelImportResult.Imported ->
                    "${result.id.stableId}: модель проверена и сохранена только в приватном хранилище приложения."

                is ModelImportResult.Rejected -> when (result.details.reason) {
                    com.faceswaplocal.app.inference.ModelValidationFailure.SIZE_MISMATCH ->
                        "${result.id.stableId}: неверный размер файла (${result.details.actualSizeBytes} вместо ${result.details.expectedSizeBytes} байт)."

                    com.faceswaplocal.app.inference.ModelValidationFailure.CHECKSUM_MISMATCH ->
                        "${result.id.stableId}: SHA-256 не совпадает с первоисточником. Файл отклонён."
                }

                is ModelImportResult.Failed ->
                    "${result.id.stableId}: импорт не выполнен (${result.reason.name})."
            }
            mutableState.update { it.copy(modelMessage = message) }
        }
    }

    fun selectSwapper(swapper: SwapperModel) {
        if (mutableState.value.selectedSwapper == swapper) return
        rawSwapJob?.cancel()
        releaseRawResult(mutableState.value.rawSwapResult)
        mutableState.update {
            it.copy(
                selectedSwapper = swapper,
                rawSwapPhase = RawSwapPhase.IDLE,
                rawSwapResult = null,
                rawSwapError = null,
            )
        }
    }

    fun runRawSwap() {
        val current = mutableState.value
        if (!current.canRunRawSwap) return
        val source = current.sourceBitmap ?: return
        val target = current.targetBitmap ?: return
        val swapper = current.selectedSwapper

        rawSwapJob?.cancel()
        rawSwapJob = viewModelScope.launch {
            releaseRawResult(mutableState.value.rawSwapResult)
            mutableState.update {
                it.copy(
                    rawSwapPhase = RawSwapPhase.RUNNING,
                    rawSwapResult = null,
                    rawSwapError = null,
                )
            }
            try {
                val result = rawSwapPipeline.process(
                    RawFaceSwapRequest(
                        source = source,
                        target = target,
                        swapper = swapper,
                        // CPU is the guaranteed Stage B path. ONNX Runtime 1.26.0
                        // aborts natively (SIGABRT, outside Kotlin exception handling)
                        // while creating an XNNPACK InSwapper session on API 35 x86_64.
                        // XNNPACK remains available through RawFaceSwapRequest for an
                        // explicit, reference-device-qualified caller.
                        backend = RequestedInferenceBackend.CPU_ONLY,
                    ),
                )
                mutableState.update {
                    it.copy(
                        rawSwapPhase = RawSwapPhase.READY,
                        rawSwapResult = result,
                    )
                }
            } catch (cancelled: CancellationException) {
                throw cancelled
            } catch (error: Throwable) {
                mutableState.update {
                    it.copy(
                        rawSwapPhase = RawSwapPhase.ERROR,
                        rawSwapError = error.toUserMessage(),
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
        val runningRawSwap = rawSwapJob
        runningRawSwap?.cancel()
        val current = mutableState.value
        mutableState.value = FaceSwapUiState(
            sourceUri = sourceUri,
            targetUri = targetUri,
            modelStatuses = current.modelStatuses,
            modelMessage = current.modelMessage,
            selectedSwapper = current.selectedSwapper,
            phase = if (sourceUri != null && targetUri != null) {
                AnalysisPhase.READY
            } else {
                AnalysisPhase.WAITING_FOR_MEDIA
            },
        )
        releaseRawResult(current.rawSwapResult)
        releaseInputBitmapsAfter(
            job = runningRawSwap,
            source = current.sourceBitmap,
            target = current.targetBitmap,
        )
    }

    private fun Throwable.toUserMessage(): String = when (this) {
        is com.faceswaplocal.app.inference.ModelUnavailableException ->
            "Модель ${id.stableId} отсутствует или не прошла повторную проверку SHA-256."

        is com.faceswaplocal.app.inference.NoNeuralFaceFoundException ->
            "5-точечный нейродетектор не нашёл подходящее лицо. Выберите более чёткое фото."

        is OutOfMemoryError ->
            "Недостаточно оперативной памяти для этой модели. Освободите память и повторите попытку."

        else -> "Локальный ONNX inference завершился ошибкой. Проверьте модели и повторите попытку."
    }

    private fun releaseRawResult(result: RawFaceSwapResult?) {
        result ?: return
        recycleBitmap(result.alignedSource112)
        recycleBitmap(result.alignedTarget)
        recycleBitmap(result.rawOutputBitmap)
    }

    private fun releaseInputBitmapsAfter(job: Job?, source: Bitmap?, target: Bitmap?) {
        if (source == null && target == null) return

        val release = {
            mainHandler.post {
                releaseInputBitmaps(source, target)
            }
            Unit
        }
        if (job == null || job.isCompleted) {
            release()
        } else {
            job.invokeOnCompletion { release() }
        }
    }

    private fun releaseInputBitmaps(source: Bitmap?, target: Bitmap?) {
        recycleBitmap(source)
        if (target !== source) recycleBitmap(target)
    }

    private fun recycleBitmap(bitmap: Bitmap?) {
        if (bitmap != null && !bitmap.isRecycled) bitmap.recycle()
    }

    override fun onCleared() {
        analysisJob?.cancel()
        val runningRawSwap = rawSwapJob
        val current = mutableState.value
        runningRawSwap?.cancel()
        releaseRawResult(current.rawSwapResult)
        releaseInputBitmapsAfter(
            job = runningRawSwap,
            source = current.sourceBitmap,
            target = current.targetBitmap,
        )
        faceDetector.close()
        super.onCleared()
    }
}

