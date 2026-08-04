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
import com.faceswaplocal.app.data.ImageMemoryBudget
import com.faceswaplocal.app.data.ExportFailure
import com.faceswaplocal.app.data.ExportOutcome
import com.faceswaplocal.app.data.MlKitLocalFaceDetector
import com.faceswaplocal.app.data.ResultExporter
import com.faceswaplocal.app.domain.DetectedFace
import com.faceswaplocal.app.domain.ExportFormat
import com.faceswaplocal.app.domain.ExportSettings
import com.faceswaplocal.app.domain.FaceAssignmentPlanner
import com.faceswaplocal.app.domain.AssignmentStateCodec
import com.faceswaplocal.app.domain.FaceQualitySettings
import com.faceswaplocal.app.domain.FaceId
import com.faceswaplocal.app.domain.ProcessingProgress
import com.faceswaplocal.app.domain.ProcessingStage
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
import com.faceswaplocal.app.inference.OnnxFaceEnhancerPipeline
import com.faceswaplocal.app.inference.OnnxFaceParserPipeline
import com.faceswaplocal.app.inference.MultiPhotoAssignment
import com.faceswaplocal.app.inference.MultiPhotoSource
import com.faceswaplocal.app.inference.MultiPhotoTarget
import com.faceswaplocal.app.inference.MultiPhotoFaceSwapResult
import com.faceswaplocal.app.inference.RequestedInferenceBackend
import com.faceswaplocal.app.inference.SwapBlendMaskMode
import java.util.Collections
import java.util.IdentityHashMap
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

    /** Cancel was requested; the coordinator stops at its next safe boundary. */
    CANCELLING,
    READY,
    ERROR,
}

enum class ExportPhase {
    IDLE,
    RUNNING,
    SAVED,
    ERROR,
}

/** UI-safe description of a saved file: a content URI, never a filesystem path. */
data class SavedExport(
    val uri: Uri,
    val displayName: String,
    val album: String?,
    val width: Int,
    val height: Int,
)

/** Pixel size of the picked file, kept to tell the user when the export is smaller. */
data class ImageSize(val width: Int, val height: Int)

/** API 28 has no pending MediaStore row, so the user names the file through SAF. */
data class ExportDestinationRequest(
    val suggestedName: String,
    val mimeType: String,
)

internal object ExportSettingsSavedState {
    private const val FORMAT = "photo_export_format"
    private const val JPEG_QUALITY = "photo_export_jpeg_quality"
    private const val WATERMARK_ENABLED = "photo_export_watermark_enabled"

    fun read(handle: SavedStateHandle): ExportSettings = ExportSettings.fromPersisted(
        formatName = handle[FORMAT],
        jpegQuality = handle[JPEG_QUALITY],
        watermarkEnabled = handle[WATERMARK_ENABLED],
    )

    fun write(handle: SavedStateHandle, settings: ExportSettings) {
        handle[FORMAT] = settings.format.name
        handle[JPEG_QUALITY] = settings.jpegQuality
        handle[WATERMARK_ENABLED] = settings.watermarkEnabled
    }
}

internal object FaceQualitySettingsSavedState {
    private const val RESTORATION_ENABLED = "photo_restoration_enabled"
    private const val RESTORATION_STRENGTH = "photo_restoration_strength"
    private const val PARSER_SWAP_MASK_ENABLED = "photo_parser_swap_mask_enabled"

    fun read(handle: SavedStateHandle): FaceQualitySettings = FaceQualitySettings.fromPersisted(
        restorationEnabled = handle[RESTORATION_ENABLED],
        restorationStrength = handle[RESTORATION_STRENGTH],
        parserSwapMaskEnabled = handle[PARSER_SWAP_MASK_ENABLED],
    )

    fun write(handle: SavedStateHandle, settings: FaceQualitySettings) {
        handle[RESTORATION_ENABLED] = settings.restorationEnabled
        handle[RESTORATION_STRENGTH] = settings.restorationStrength
        handle[PARSER_SWAP_MASK_ENABLED] = settings.parserSwapMaskEnabled
    }
}

internal fun FaceQualitySettings.requiredModelIds(): Set<ModelId> = buildSet {
    add(ModelId.YOLOFACE_8N)
    add(ModelId.ARCFACE_W600K_R50)
    add(ModelId.INSWAPPER_128_FP16)
    if (parserSwapMaskEnabled || effectiveRestorationStrength > 0f) {
        add(ModelId.BISENET_RESNET_34)
    }
    if (effectiveRestorationStrength > 0f) {
        add(ModelId.GFPGAN_1_4)
    }
}

internal fun FaceQualitySettings.swapBlendMaskMode(): SwapBlendMaskMode =
    if (parserSwapMaskEnabled) SwapBlendMaskMode.PARSER_REGION else SwapBlendMaskMode.AFFINE_BOX

internal inline fun <T> publishOrReleaseOwnedResult(
    result: T,
    checkCanPublish: () -> Unit,
    publish: (T) -> Unit,
    release: (T) -> Unit,
) {
    var published = false
    try {
        checkCanPublish()
        publish(result)
        published = true
    } finally {
        if (!published) release(result)
    }
}

data class FaceSwapUiState(
    val sourceUris: List<Uri> = emptyList(),
    val targetUri: Uri? = null,
    val sourceBitmap: Bitmap? = null,
    val targetBitmap: Bitmap? = null,
    /** Size of the picked target file; null until it has been decoded. */
    val targetSourceSize: ImageSize? = null,
    val sourceFaces: List<DetectedFace> = emptyList(),
    val sourceBitmaps: Map<FaceId, Bitmap> = emptyMap(),
    val targetFaces: List<DetectedFace> = emptyList(),
    val assignments: List<SwapAssignment> = emptyList(),
    val phase: AnalysisPhase = AnalysisPhase.WAITING_FOR_MEDIA,
    val errorMessage: String? = null,
    val modelStatuses: Map<ModelId, ModelStatus> = ModelCatalog.all.associate { it.id to ModelStatus.Missing },
    val modelMessage: String? = null,
    val qualitySettings: FaceQualitySettings = FaceQualitySettings(),
    val exportSettings: ExportSettings = ExportSettings(),
    val photoSwapPhase: PhotoSwapPhase = PhotoSwapPhase.IDLE,
    val photoSwapResult: MultiPhotoFaceSwapResult? = null,
    val photoSwapError: String? = null,
    val processingProgress: ProcessingProgress? = null,
    val exportPhase: ExportPhase = ExportPhase.IDLE,
    val savedExport: SavedExport? = null,
    val exportError: String? = null,
    val exportDestinationRequest: ExportDestinationRequest? = null,
) {
    val canAnalyze: Boolean
        get() = sourceUris.isNotEmpty() && targetUri != null && phase != AnalysisPhase.ANALYZING &&
            !isProcessing

    /** A run in progress, including the window between cancel request and safe stop. */
    val isProcessing: Boolean
        get() = photoSwapPhase == PhotoSwapPhase.RUNNING || photoSwapPhase == PhotoSwapPhase.CANCELLING

    val canRunPhotoSwap: Boolean
        get() {
            return phase == AnalysisPhase.MAPPING &&
                !isProcessing &&
                exportPhase != ExportPhase.RUNNING &&
                targetBitmap != null &&
                assignments.isNotEmpty() &&
                qualitySettings.requiredModelIds().all { modelStatuses[it] is ModelStatus.Ready }
        }

    val canCancelPhotoSwap: Boolean
        get() = photoSwapPhase == PhotoSwapPhase.RUNNING

    /**
     * True when the memory budget forced a smaller decode, so the export cannot match
     * the picked file's own size and the UI has to say so.
     */
    val exportIsDownscaled: Boolean
        get() {
            val source = targetSourceSize ?: return false
            val decoded = targetBitmap ?: return false
            return decoded.width < source.width || decoded.height < source.height
        }

    val canExport: Boolean
        get() = photoSwapResult != null &&
            photoSwapPhase == PhotoSwapPhase.READY &&
            exportPhase != ExportPhase.RUNNING &&
            exportDestinationRequest == null
}

class FaceSwapViewModel(application: Application, private val savedStateHandle: SavedStateHandle) : AndroidViewModel(application) {
    private val bitmapLoader = BitmapLoader(
        contentResolver = application.contentResolver,
        budget = ImageMemoryBudget.forContext(application),
    )
    private val faceDetector = MlKitLocalFaceDetector()
    private val modelStore = ModelStore(application)
    private val rawPipeline = OnnxRawFaceSwapPipeline(modelStore)
    private val photoSwapPipeline = OnnxPhotoFaceSwapPipeline(modelStore, rawPipeline = rawPipeline)
    private val faceEnhancerPipeline = OnnxFaceEnhancerPipeline(modelStore)
    private val faceParserPipeline = OnnxFaceParserPipeline(modelStore)
    private val multiPhotoSwapPipeline = OnnxMultiPhotoFaceSwapPipeline(
        rawPipeline,
        photoSwapPipeline,
        faceEnhancerPipeline,
        faceParserPipeline,
    )
    private val resultExporter = ResultExporter(application)
    private val mutableState = MutableStateFlow(
        FaceSwapUiState(
            qualitySettings = FaceQualitySettingsSavedState.read(savedStateHandle),
            exportSettings = ExportSettingsSavedState.read(savedStateHandle),
        ),
    )
    private val mainHandler = Handler(Looper.getMainLooper())
    private val retiredPhotoResults = Collections.newSetFromMap(
        IdentityHashMap<MultiPhotoFaceSwapResult, Boolean>(),
    )
    private var analysisJob: Job? = null
    private var photoSwapJob: Job? = null
    private var exportJob: Job? = null

    /**
     * Monotonic id of the current swap run. A cancelled run may finish its unwinding after
     * a newer run has already started, and must not overwrite the newer run's state.
     */
    private var photoSwapRunId = 0L

    val state: StateFlow<FaceSwapUiState> = mutableState.asStateFlow()

    internal fun injectUiStateForTest(testState: FaceSwapUiState) {
        val previousResult = mutableState.value.photoSwapResult
        if (previousResult !== testState.photoSwapResult) retirePhotoResult(previousResult)
        mutableState.value = testState
        persistAssignments()
        FaceQualitySettingsSavedState.write(savedStateHandle, testState.qualitySettings)
        ExportSettingsSavedState.write(savedStateHandle, testState.exportSettings)
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
        // Staging files and pending MediaStore rows an earlier process could not clean up
        // after a crash are removed before the first export of this session (§5.1).
        viewModelScope.launch { resultExporter.sweepAbandonedData() }
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
        photoSwapRunId++
        runningPhotoSwap?.cancel()
        exportJob?.cancel()
        val previousState = mutableState.value
        retirePhotoResult(previousState.photoSwapResult)
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
            ).withoutExportState()
        }
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
                val decodedTargetImage = bitmapLoader.loadTarget(targetUri)
                val targetBitmap = decodedTargetImage.bitmap.also { decodedTarget = it }
                val decodedSources = sourceUris.mapIndexed { index, uri ->
                    val bitmap = bitmapLoader.loadSource(uri).bitmap
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
                        targetSourceSize = ImageSize(
                            width = decodedTargetImage.sourceWidth,
                            height = decodedTargetImage.sourceHeight,
                        ),
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

        // Claim the new id before cancelling, so the outgoing run can never win the race
        // and reset the state this run is about to publish.
        val runId = ++photoSwapRunId
        photoSwapJob?.cancel()
        photoSwapJob = viewModelScope.launch {
            mutableState.update {
                it.copy(
                    photoSwapPhase = PhotoSwapPhase.RUNNING,
                    photoSwapError = null,
                    processingProgress = ProcessingProgress(
                        stage = ProcessingStage.PREPARING,
                        totalFaces = selectedAssignments.size,
                        restorationPlanned = current.qualitySettings.effectiveRestorationStrength > 0f,
                    ),
                    exportPhase = ExportPhase.IDLE,
                    savedExport = null,
                    exportError = null,
                    exportDestinationRequest = null,
                )
            }
            try {
                val result = multiPhotoSwapPipeline.process(
                    target = target,
                    sources = sources,
                    targetsInStableOrder = current.targetFaces.map { face -> MultiPhotoTarget(face.id, face.toPixelBox(target)) },
                    assignments = selectedAssignments.map { MultiPhotoAssignment(it.targetFaceId, it.sourceFaceId) },
                    backend = RequestedInferenceBackend.XNNPACK_WITH_CPU_FALLBACK,
                    restorationStrength = current.qualitySettings.effectiveRestorationStrength,
                    swapBlendMaskMode = current.qualitySettings.swapBlendMaskMode(),
                    onProgress = { progress -> publishProgress(runId, progress) },
                ) ?: throw IllegalStateException("No target face is assigned")
                publishOrReleaseOwnedResult(
                    result = result,
                    checkCanPublish = { currentCoroutineContext().ensureActive() },
                    publish = { publishedResult ->
                        val previousResult = mutableState.value.photoSwapResult
                        if (previousResult !== publishedResult) retirePhotoResult(previousResult)
                        mutableState.update {
                            it.copy(
                                photoSwapPhase = PhotoSwapPhase.READY,
                                photoSwapResult = publishedResult,
                                processingProgress = ProcessingProgress(
                                    stage = ProcessingStage.COMPLETED,
                                    completedFaces = selectedAssignments.size,
                                    totalFaces = selectedAssignments.size,
                                    restorationPlanned = current.qualitySettings.effectiveRestorationStrength > 0f,
                                ),
                            )
                        }
                    },
                    release = { unpublishedResult -> recycleBitmap(unpublishedResult.finalBitmap) },
                )
            } catch (cancelled: CancellationException) {
                // The coordinator already released its sessions and bitmaps in its own
                // finally block; only the UI-visible run state is reset here.
                if (photoSwapRunId == runId) {
                    mutableState.update {
                        it.copy(
                            photoSwapPhase = PhotoSwapPhase.IDLE,
                            processingProgress = null,
                        )
                    }
                }
                throw cancelled
            } catch (error: Throwable) {
                mutableState.update {
                    it.copy(
                        photoSwapPhase = PhotoSwapPhase.ERROR,
                        photoSwapError = error.toUserMessage(),
                        processingProgress = null,
                    )
                }
            } finally {
                if (photoSwapRunId == runId) sweepTemporaryFiles()
            }
        }
    }

    /** Requests a stop; the coordinator observes it at its next `ensureActive` boundary. */
    fun cancelPhotoSwap() {
        val running = photoSwapJob ?: return
        if (!mutableState.value.canCancelPhotoSwap) return
        mutableState.update { it.copy(photoSwapPhase = PhotoSwapPhase.CANCELLING) }
        running.cancel()
    }

    private fun publishProgress(runId: Long, progress: ProcessingProgress) {
        if (photoSwapRunId != runId) return
        mutableState.update { state ->
            if (state.photoSwapPhase != PhotoSwapPhase.RUNNING) state
            else state.copy(processingProgress = progress)
        }
    }

    private fun sweepTemporaryFiles() {
        viewModelScope.launch { resultExporter.sweepStagingFiles() }
    }

    fun assignSource(targetFaceId: FaceId, sourceFaceId: FaceId) {
        mutateMapping { state ->
            state.copy(
                assignments = FaceAssignmentPlanner.replaceSource(
                    assignments = state.assignments,
                    targetFaceId = targetFaceId,
                    sourceFaceId = sourceFaceId,
                ),
            )
        }
    }

    fun setUnchanged(targetFaceId: FaceId) {
        mutateMapping { state ->
            state.copy(
                assignments = state.assignments.filterNot { assignment ->
                    assignment.targetFaceId == targetFaceId
                },
            )
        }
    }

    fun applySourceToAll(sourceFaceId: FaceId) {
        mutateMapping { state ->
            state.copy(
                assignments = FaceAssignmentPlanner.applySourceToAll(state.targetFaces, sourceFaceId),
            )
        }
    }

    fun removeSource(sourceFaceId: FaceId) {
        mutateMapping { state ->
            val keptFaces = state.sourceFaces.filterNot { it.id == sourceFaceId }
            state.copy(
                sourceFaces = keptFaces,
                sourceBitmaps = state.sourceBitmaps - sourceFaceId,
                assignments = FaceAssignmentPlanner.clearSource(state.assignments, sourceFaceId),
            )
        }
    }

    fun setRestorationEnabled(enabled: Boolean) {
        updateQualitySettings { settings -> settings.copy(restorationEnabled = enabled) }
    }

    fun setRestorationStrength(strength: Float) {
        val sanitized = strength
            .takeIf { value -> value.isFinite() && value in 0f..1f }
            ?: FaceQualitySettings.DEFAULT_RESTORATION_STRENGTH
        updateQualitySettings { settings -> settings.copy(restorationStrength = sanitized) }
    }

    fun setParserSwapMaskEnabled(enabled: Boolean) {
        updateQualitySettings { settings -> settings.copy(parserSwapMaskEnabled = enabled) }
    }

    fun setExportFormat(format: ExportFormat) {
        updateExportSettings { settings -> settings.copy(format = format) }
    }

    fun setJpegQuality(quality: Int) {
        updateExportSettings { settings ->
            settings.copy(jpegQuality = ExportSettings.sanitizedQuality(quality))
        }
    }

    fun setWatermarkEnabled(enabled: Boolean) {
        updateExportSettings { settings -> settings.copy(watermarkEnabled = enabled) }
    }

    /**
     * Saves the current result. On API 29+ the destination is a pending MediaStore row; on
     * API 28 the exporter asks for a SAF document first, which the UI answers through
     * [provideExportDestination].
     */
    fun exportResult() {
        val current = mutableState.value
        if (!current.canExport) return
        val bitmap = current.photoSwapResult?.finalBitmap ?: return
        if (bitmap.isRecycled) return
        startExport(bitmap, current.exportSettings, destination = null)
    }

    fun provideExportDestination(destination: Uri) {
        val current = mutableState.value
        val bitmap = current.photoSwapResult?.finalBitmap ?: return
        if (current.exportDestinationRequest == null || bitmap.isRecycled) return
        mutableState.update { it.copy(exportDestinationRequest = null) }
        startExport(bitmap, current.exportSettings, destination = destination)
    }

    fun cancelExportDestinationRequest() {
        mutableState.update {
            it.copy(exportDestinationRequest = null, exportPhase = ExportPhase.IDLE)
        }
    }

    fun dismissExportError() {
        mutableState.update { it.copy(exportError = null, exportPhase = ExportPhase.IDLE) }
    }

    private fun startExport(bitmap: Bitmap, settings: ExportSettings, destination: Uri?) {
        exportJob?.cancel()
        exportJob = viewModelScope.launch {
            mutableState.update {
                it.copy(
                    exportPhase = ExportPhase.RUNNING,
                    exportError = null,
                    savedExport = null,
                    processingProgress = it.processingProgress?.copy(stage = ProcessingStage.EXPORTING),
                )
            }
            try {
                when (val outcome = resultExporter.export(bitmap, settings, destination)) {
                    is ExportOutcome.Saved -> mutableState.update {
                        it.copy(
                            exportPhase = ExportPhase.SAVED,
                            processingProgress = it.processingProgress
                                ?.copy(stage = ProcessingStage.COMPLETED),
                            savedExport = SavedExport(
                                uri = outcome.uri,
                                displayName = outcome.displayName,
                                album = outcome.album,
                                width = outcome.width,
                                height = outcome.height,
                            ),
                        )
                    }

                    is ExportOutcome.NeedsDestination -> mutableState.update {
                        it.copy(
                            exportPhase = ExportPhase.IDLE,
                            exportDestinationRequest = ExportDestinationRequest(
                                suggestedName = outcome.suggestedName,
                                mimeType = outcome.mimeType,
                            ),
                        )
                    }

                    is ExportOutcome.Failed -> mutableState.update {
                        it.copy(
                            exportPhase = ExportPhase.ERROR,
                            exportError = outcome.reason.toUserMessage(),
                        )
                    }
                }
            } catch (cancelled: CancellationException) {
                mutableState.update { it.copy(exportPhase = ExportPhase.IDLE) }
                throw cancelled
            } catch (error: Throwable) {
                mutableState.update {
                    it.copy(
                        exportPhase = ExportPhase.ERROR,
                        exportError = ExportFailure.WRITE_FAILED.toUserMessage(),
                    )
                }
            }
        }
    }

    private fun ExportFailure.toUserMessage(): String = when (this) {
        ExportFailure.CACHE_UNAVAILABLE ->
            "Недостаточно свободного места во временном хранилище приложения."

        ExportFailure.ENCODE_FAILED ->
            "Не удалось закодировать результат в выбранный формат."

        ExportFailure.DESTINATION_UNAVAILABLE ->
            "Галерея недоступна для записи. Незавершённая запись удалена."

        ExportFailure.WRITE_FAILED ->
            "Сохранение в галерею не удалось. Незавершённая запись удалена."
    }

    /** A saved-file badge must never survive the result it describes. */
    private fun FaceSwapUiState.withoutExportState(): FaceSwapUiState = copy(
        processingProgress = null,
        exportPhase = ExportPhase.IDLE,
        savedExport = null,
        exportError = null,
        exportDestinationRequest = null,
    )

    private inline fun mutateMapping(transform: (FaceSwapUiState) -> FaceSwapUiState) {
        val current = mutableState.value
        val transformed = transform(current)
        if (transformed == current) return

        photoSwapJob?.cancel()
        retirePhotoResult(current.photoSwapResult)
        mutableState.value = transformed.copy(
            photoSwapPhase = PhotoSwapPhase.IDLE,
            photoSwapResult = null,
            photoSwapError = null,
        ).withoutExportState()
        persistAssignments()
    }

    private inline fun updateQualitySettings(
        transform: (FaceQualitySettings) -> FaceQualitySettings,
    ) {
        val current = mutableState.value
        if (current.photoSwapPhase == PhotoSwapPhase.RUNNING) return
        val updated = transform(current.qualitySettings)
        if (updated == current.qualitySettings) return

        retirePhotoResult(current.photoSwapResult)
        mutableState.value = current.copy(
            qualitySettings = updated,
            photoSwapPhase = PhotoSwapPhase.IDLE,
            photoSwapResult = null,
            photoSwapError = null,
        ).withoutExportState()
        FaceQualitySettingsSavedState.write(savedStateHandle, updated)
    }

    /**
     * Export settings do not invalidate the processed bitmap, so the result stays on
     * screen; only the "already saved" state is dropped because it no longer describes
     * what the next save would produce.
     */
    private inline fun updateExportSettings(
        transform: (ExportSettings) -> ExportSettings,
    ) {
        val current = mutableState.value
        if (current.exportPhase == ExportPhase.RUNNING || current.exportDestinationRequest != null) return
        val updated = transform(current.exportSettings)
        if (updated == current.exportSettings) return

        mutableState.value = current.copy(
            exportSettings = updated,
            exportPhase = ExportPhase.IDLE,
            savedExport = null,
            exportError = null,
        )
        ExportSettingsSavedState.write(savedStateHandle, updated)
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
        photoSwapRunId++
        runningPhotoSwap?.cancel()
        exportJob?.cancel()
        val current = mutableState.value
        retirePhotoResult(current.photoSwapResult)
        mutableState.value = FaceSwapUiState(
            sourceUris = sourceUris,
            targetUri = targetUri,
            modelStatuses = current.modelStatuses,
            modelMessage = current.modelMessage,
            qualitySettings = current.qualitySettings,
            exportSettings = current.exportSettings,
            phase = if (sourceUris.isNotEmpty() && targetUri != null) {
                AnalysisPhase.READY
            } else {
                AnalysisPhase.WAITING_FOR_MEDIA
            },
        )
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

    /** Called by Compose only after the corresponding result leaves the composition. */
    fun onPhotoResultDisposed(result: MultiPhotoFaceSwapResult) {
        if (!retiredPhotoResults.remove(result)) return
        recycleBitmap(result.finalBitmap)
    }

    private fun retirePhotoResult(result: MultiPhotoFaceSwapResult?) {
        if (result != null) retiredPhotoResults.add(result)
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
        exportJob?.cancel()
        val runningPhotoSwap = photoSwapJob
        val current = mutableState.value
        photoSwapRunId++
        runningPhotoSwap?.cancel()
        recycleBitmap(current.photoSwapResult?.finalBitmap)
        retiredPhotoResults.forEach { result -> recycleBitmap(result.finalBitmap) }
        retiredPhotoResults.clear()
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
