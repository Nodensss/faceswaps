package com.faceswaplocal.app.ui

import android.app.Application
import android.graphics.Bitmap
import android.net.Uri
import android.os.Handler
import android.os.Looper
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.SavedStateHandle
import androidx.lifecycle.viewModelScope
import com.faceswaplocal.app.data.BitmapLoader
import com.faceswaplocal.app.data.MlKitLocalFaceDetector
import com.faceswaplocal.app.domain.DetectedFace
import com.faceswaplocal.app.domain.FaceAssignmentPlanner
import com.faceswaplocal.app.domain.AssignmentStateCodec
import com.faceswaplocal.app.domain.FaceId
import com.faceswaplocal.app.domain.SwapAssignment
import com.faceswaplocal.app.inference.ModelCatalog
import com.faceswaplocal.app.inference.ModelId
import com.faceswaplocal.app.inference.ModelImportResult
import com.faceswaplocal.app.inference.ModelStatus
import com.faceswaplocal.app.inference.ModelStore
import com.faceswaplocal.app.inference.FaceBox
import com.faceswaplocal.app.inference.OnnxPhotoFaceSwapPipeline
import com.faceswaplocal.app.inference.OnnxRawFaceSwapPipeline
import com.faceswaplocal.app.inference.OnnxMultiPhotoFaceSwapPipeline
import com.faceswaplocal.app.inference.MultiPhotoAssignment
import com.faceswaplocal.app.inference.MultiPhotoSource
import com.faceswaplocal.app.inference.MultiPhotoTarget
import com.faceswaplocal.app.inference.PhotoFaceSwapRequest
import com.faceswaplocal.app.inference.PhotoFaceSwapResult
import com.faceswaplocal.app.inference.RequestedInferenceBackend
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

enum class PhotoSwapPhase {
    IDLE,
    RUNNING,
    READY,
    ERROR,
}

data class FaceSwapUiState(
    val sourceUris: List<Uri> = emptyList(),
    val targetUri: Uri? = null,
    val sourceBitmap: Bitmap? = null,
    val targetBitmap: Bitmap? = null,
    val sourceFaces: List<DetectedFace> = emptyList(),
    val sourceBitmaps: Map<FaceId, Bitmap> = emptyMap(),
    val targetFaces: List<DetectedFace> = emptyList(),
    val assignments: List<SwapAssignment> = emptyList(),
    val phase: AnalysisPhase = AnalysisPhase.WAITING_FOR_MEDIA,
    val errorMessage: String? = null,
    val modelStatuses: Map<ModelId, ModelStatus> = ModelCatalog.all.associate { it.id to ModelStatus.Missing },
    val modelMessage: String? = null,
    val photoSwapPhase: PhotoSwapPhase = PhotoSwapPhase.IDLE,
    val photoSwapResult: PhotoFaceSwapResult? = null,
    val photoSwapError: String? = null,
) {
    val canAnalyze: Boolean
        get() = sourceUris.isNotEmpty() && targetUri != null && phase != AnalysisPhase.ANALYZING

    val canRunPhotoSwap: Boolean
        get() {
            val requiredModels = setOf(
                ModelId.YOLOFACE_8N,
                ModelId.ARCFACE_W600K_R50,
                ModelId.INSWAPPER_128_FP16,
            )
            return phase == AnalysisPhase.MAPPING &&
                photoSwapPhase != PhotoSwapPhase.RUNNING &&
                targetBitmap != null &&
                assignments.isNotEmpty() &&
                requiredModels.all { modelStatuses[it] is ModelStatus.Ready }
        }
}

class FaceSwapViewModel(application: Application, private val savedStateHandle: SavedStateHandle) : AndroidViewModel(application) {
    private val bitmapLoader = BitmapLoader(application.contentResolver)
    private val faceDetector = MlKitLocalFaceDetector()
    private val modelStore = ModelStore(application)
    private val photoSwapPipeline = OnnxPhotoFaceSwapPipeline(modelStore)
    private val rawPipeline = OnnxRawFaceSwapPipeline(modelStore)
    private val multiPhotoSwapPipeline = OnnxMultiPhotoFaceSwapPipeline(rawPipeline, photoSwapPipeline)
    private val mutableState = MutableStateFlow(FaceSwapUiState())
    private val mainHandler = Handler(Looper.getMainLooper())
    private var analysisJob: Job? = null
    private var photoSwapJob: Job? = null

    val state: StateFlow<FaceSwapUiState> = mutableState.asStateFlow()

    internal fun injectUiStateForTest(testState: FaceSwapUiState) {
        mutableState.value = testState
        persistAssignments()
    }

    init {
        val restored = savedStateHandle.get<ArrayList<String>>(SAVED_ASSIGNMENTS).orEmpty()
        if (restored.isNotEmpty()) {
            mutableState.value = mutableState.value.copy(
                errorMessage = "Выбранные назначения восстановлены, но изображения после смерти процесса нужно выбрать заново.",
            )
        }
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

    fun selectSources(uris: List<Uri>) {
        analysisJob?.cancel()
        val current = mutableState.value
        resetSelections(sourceUris = uris.distinct().take(8), targetUri = current.targetUri)
    }

    fun selectTarget(uri: Uri) {
        analysisJob?.cancel()
        val current = mutableState.value
        resetSelections(sourceUris = current.sourceUris, targetUri = uri)
    }

    fun analyze() {
        val sourceUris = mutableState.value.sourceUris
        val targetUri = mutableState.value.targetUri ?: return
        if (sourceUris.isEmpty()) return

        analysisJob?.cancel()
        val runningPhotoSwap = photoSwapJob
        runningPhotoSwap?.cancel()
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
                photoSwapPhase = PhotoSwapPhase.IDLE,
                photoSwapResult = null,
                photoSwapError = null,
            )
        }
        releasePhotoResultDeferred(previousState.photoSwapResult)
        releaseInputBitmapsAfter(
            job = runningPhotoSwap,
            source = previousState.sourceBitmap,
            target = previousState.targetBitmap,
        )

        analysisJob = viewModelScope.launch {
            var decodedSource: Bitmap? = null
            var decodedTarget: Bitmap? = null
            var stateOwnsDecodedBitmaps = false
            try {
                val targetBitmap = bitmapLoader.load(targetUri).also { decodedTarget = it }
                val decodedSources = sourceUris.mapIndexed { index, uri ->
                    val bitmap = bitmapLoader.load(uri)
                    val faces = faceDetector.detect(bitmap, idPrefix = "source-$index")
                    bitmap to faces
                }
                val sourceFaces = decodedSources.flatMap { it.second }
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
                        sourceBitmap = decodedSources.first().first,
                        sourceBitmaps = decodedSources.flatMap { (bitmap, faces) -> faces.map { it.id to bitmap } }.toMap(),
                        targetBitmap = targetBitmap,
                        sourceFaces = sourceFaces,
                        targetFaces = targetFaces,
                        assignments = restoreAssignments(sourceFaces, targetFaces)
                            ?: FaceAssignmentPlanner.defaults(sourceFaces, targetFaces),
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
                    decodedSource?.let { recycleBitmap(it) }
                    releaseInputBitmaps(null, decodedTarget)
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

    fun runPhotoSwap() {
        val current = mutableState.value
        if (!current.canRunPhotoSwap) return
        val target = current.targetBitmap ?: return
        val selectedAssignments = current.assignments
        val sources = current.sourceFaces.mapNotNull { face ->
            current.sourceBitmaps[face.id]?.let { bitmap -> MultiPhotoSource(face.id, bitmap, face.toPixelBox(bitmap)) }
        }

        photoSwapJob?.cancel()
        photoSwapJob = viewModelScope.launch {
            val previousResult = mutableState.value.photoSwapResult
            mutableState.update {
                it.copy(
                    photoSwapPhase = PhotoSwapPhase.RUNNING,
                    photoSwapResult = null,
                    photoSwapError = null,
                )
            }
            releasePhotoResultDeferred(previousResult)
            try {
                val result = multiPhotoSwapPipeline.process(
                    target = target,
                    sources = sources,
                    targetsInStableOrder = current.targetFaces.map { face -> MultiPhotoTarget(face.id, face.toPixelBox(target)) },
                    assignments = selectedAssignments.map { MultiPhotoAssignment(it.targetFaceId, it.sourceFaceId) },
                    backend = RequestedInferenceBackend.XNNPACK_WITH_CPU_FALLBACK,
                ) ?: throw IllegalStateException("No target face is assigned")
                mutableState.update {
                    it.copy(
                        photoSwapPhase = PhotoSwapPhase.READY,
                        photoSwapResult = result,
                    )
                }
            } catch (cancelled: CancellationException) {
                throw cancelled
            } catch (error: Throwable) {
                mutableState.update {
                    it.copy(
                        photoSwapPhase = PhotoSwapPhase.ERROR,
                        photoSwapError = error.toUserMessage(),
                    )
                }
            }
        }
    }

    fun assignSource(targetFaceId: FaceId, sourceFaceId: FaceId) {
        val runningPhotoSwap = photoSwapJob
        runningPhotoSwap?.cancel()
        val previousResult = mutableState.value.photoSwapResult
        mutableState.update {
            it.copy(
                assignments = FaceAssignmentPlanner.replaceSource(
                    assignments = it.assignments,
                    targetFaceId = targetFaceId,
                    sourceFaceId = sourceFaceId,
                ),
                photoSwapPhase = PhotoSwapPhase.IDLE,
                photoSwapResult = null,
                photoSwapError = null,
            )
        }
        releasePhotoResultDeferred(previousResult)
        persistAssignments()
    }

    fun setUnchanged(targetFaceId: FaceId) {
        mutableState.update { it.copy(assignments = it.assignments.filterNot { assignment -> assignment.targetFaceId == targetFaceId }) }
        persistAssignments()
    }

    fun applySourceToAll(sourceFaceId: FaceId) {
        mutableState.update { it.copy(assignments = FaceAssignmentPlanner.applySourceToAll(it.targetFaces, sourceFaceId)) }
        persistAssignments()
    }

    fun removeSource(sourceFaceId: FaceId) {
        mutableState.update { state ->
            val keptFaces = state.sourceFaces.filterNot { it.id == sourceFaceId }
            state.copy(
                sourceFaces = keptFaces,
                sourceBitmaps = state.sourceBitmaps - sourceFaceId,
                assignments = FaceAssignmentPlanner.clearSource(state.assignments, sourceFaceId),
            )
        }
        persistAssignments()
    }

    fun dismissError() {
        mutableState.update {
            it.copy(
                    phase = if (it.sourceUris.isNotEmpty() && it.targetUri != null) {
                    AnalysisPhase.READY
                } else {
                    AnalysisPhase.WAITING_FOR_MEDIA
                },
                errorMessage = null,
            )
        }
    }

    private fun resetSelections(sourceUris: List<Uri>, targetUri: Uri?) {
        val runningPhotoSwap = photoSwapJob
        runningPhotoSwap?.cancel()
        val current = mutableState.value
        mutableState.value = FaceSwapUiState(
            sourceUris = sourceUris,
            targetUri = targetUri,
            modelStatuses = current.modelStatuses,
            modelMessage = current.modelMessage,
            phase = if (sourceUris.isNotEmpty() && targetUri != null) {
                AnalysisPhase.READY
            } else {
                AnalysisPhase.WAITING_FOR_MEDIA
            },
        )
        releasePhotoResultDeferred(current.photoSwapResult)
        savedStateHandle.remove<ArrayList<String>>(SAVED_ASSIGNMENTS)
        releaseInputBitmapsAfter(
            job = runningPhotoSwap,
            source = current.sourceBitmap,
            target = current.targetBitmap,
        )
        current.sourceBitmaps.values.distinct().filter { it !== current.sourceBitmap }.forEach(::recycleBitmap)
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

    private fun DetectedFace.toPixelBox(bitmap: Bitmap): FaceBox = FaceBox(
        left = bounds.left * bitmap.width.toDouble(),
        top = bounds.top * bitmap.height.toDouble(),
        right = bounds.right * bitmap.width.toDouble(),
        bottom = bounds.bottom * bitmap.height.toDouble(),
    )

    private fun releasePhotoResultDeferred(result: PhotoFaceSwapResult?) {
        result ?: return
        mainHandler.post { recycleBitmap(result.finalBitmap) }
    }

    private fun releasePhotoResultNow(result: PhotoFaceSwapResult?) {
        recycleBitmap(result?.finalBitmap)
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

    private fun persistAssignments() {
        savedStateHandle[SAVED_ASSIGNMENTS] = AssignmentStateCodec.encode(mutableState.value.assignments)
    }

    private fun restoreAssignments(sourceFaces: List<DetectedFace>, targetFaces: List<DetectedFace>): List<SwapAssignment>? {
        val restored = AssignmentStateCodec.restore(savedStateHandle.get<ArrayList<String>>(SAVED_ASSIGNMENTS).orEmpty(), sourceFaces, targetFaces)
        return restored.takeIf { it.isNotEmpty() }
    }

    private companion object { const val SAVED_ASSIGNMENTS = "stage_d_assignment_ids" }

    override fun onCleared() {
        analysisJob?.cancel()
        val runningPhotoSwap = photoSwapJob
        val current = mutableState.value
        runningPhotoSwap?.cancel()
        releasePhotoResultNow(current.photoSwapResult)
        releaseInputBitmapsAfter(
            job = runningPhotoSwap,
            source = current.sourceBitmap,
            target = current.targetBitmap,
        )
        current.sourceBitmaps.values.distinct().filter { it !== current.sourceBitmap }.forEach(::recycleBitmap)
        faceDetector.close()
        super.onCleared()
    }
}

