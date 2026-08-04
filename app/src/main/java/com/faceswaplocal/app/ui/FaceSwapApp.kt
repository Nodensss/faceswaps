package com.faceswaplocal.app.ui

import android.content.ActivityNotFoundException
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.Paint
import android.net.Uri
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts.CreateDocument
import androidx.activity.result.contract.ActivityResultContracts.OpenDocument
import androidx.activity.result.contract.ActivityResultContracts.PickVisualMedia
import androidx.activity.result.contract.ActivityResultContracts.PickMultipleVisualMedia
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.FilterChip
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Slider
import androidx.compose.material3.Surface
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalView
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.geometry.CornerRadius
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.nativeCanvas
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import com.faceswaplocal.app.domain.DetectedFace
import com.faceswaplocal.app.domain.ExportFormat
import com.faceswaplocal.app.domain.ExportSettings
import com.faceswaplocal.app.domain.FaceId
import com.faceswaplocal.app.domain.ProcessingProgress
import com.faceswaplocal.app.domain.ProcessingStage
import com.faceswaplocal.app.domain.SwapAssignment
import com.faceswaplocal.app.inference.ModelCatalog
import com.faceswaplocal.app.inference.ModelDescriptor
import com.faceswaplocal.app.inference.ModelId
import com.faceswaplocal.app.inference.ModelStatus
import com.faceswaplocal.app.inference.MultiPhotoFaceSwapResult
import kotlin.math.min

@Composable
fun FaceSwapRoute(viewModel: FaceSwapViewModel = viewModel()) {
    val state by viewModel.state.collectAsStateWithLifecycle()
    val sourcePicker = rememberLauncherForActivityResult(PickMultipleVisualMedia(8)) { uris ->
        if (uris.isNotEmpty()) viewModel.selectSources(uris)
    }
    val targetPicker = rememberLauncherForActivityResult(PickVisualMedia()) { uri ->
        uri?.let(viewModel::selectTarget)
    }
    var pendingModelId by remember { mutableStateOf<ModelId?>(null) }
    val modelPicker = rememberLauncherForActivityResult(OpenDocument()) { uri ->
        val modelId = pendingModelId
        pendingModelId = null
        if (uri != null && modelId != null) {
            viewModel.importModel(modelId, uri)
        }
    }

    // Only reached on API 28, where MediaStore has no pending row and inserting into the
    // images collection would need the forbidden full-storage permission (§5.1).
    val exportDestinationRequest = state.exportDestinationRequest
    val destinationPicker = rememberLauncherForActivityResult(
        CreateDocument(exportDestinationRequest?.mimeType ?: "image/jpeg"),
    ) { uri ->
        if (uri != null) viewModel.provideExportDestination(uri) else viewModel.cancelExportDestinationRequest()
    }
    LaunchedEffect(exportDestinationRequest) {
        exportDestinationRequest?.let { request -> destinationPicker.launch(request.suggestedName) }
    }

    FaceSwapScreen(
        state = state,
        onPickSource = {
            sourcePicker.launch(PickVisualMediaRequest(PickVisualMedia.ImageOnly))
        },
        onPickTarget = {
            targetPicker.launch(PickVisualMediaRequest(PickVisualMedia.ImageOnly))
        },
        onImportModel = { modelId ->
            pendingModelId = modelId
            modelPicker.launch(
                arrayOf(
                    "application/octet-stream",
                    "application/onnx",
                    "*/*",
                ),
            )
        },
        onAnalyze = viewModel::analyze,
        onAssignSource = viewModel::assignSource,
        onSetUnchanged = viewModel::setUnchanged,
        onApplySourceToAll = viewModel::applySourceToAll,
        onRemoveSource = viewModel::removeSource,
        onRestorationEnabledChange = viewModel::setRestorationEnabled,
        onRestorationStrengthChange = viewModel::setRestorationStrength,
        onParserSwapMaskEnabledChange = viewModel::setParserSwapMaskEnabled,
        onRunPhotoSwap = viewModel::runPhotoSwap,
        onCancelPhotoSwap = viewModel::cancelPhotoSwap,
        onExportFormatChange = viewModel::setExportFormat,
        onJpegQualityChange = viewModel::setJpegQuality,
        onWatermarkEnabledChange = viewModel::setWatermarkEnabled,
        onExport = viewModel::exportResult,
        onDismissExportError = viewModel::dismissExportError,
        onPhotoResultDisposed = viewModel::onPhotoResultDisposed,
        onDismissError = viewModel::dismissError,
    )
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
internal fun FaceSwapScreen(
    state: FaceSwapUiState,
    onPickSource: () -> Unit,
    onPickTarget: () -> Unit,
    onImportModel: (ModelId) -> Unit,
    onAnalyze: () -> Unit,
    onAssignSource: (FaceId, FaceId) -> Unit,
    onSetUnchanged: (FaceId) -> Unit,
    onApplySourceToAll: (FaceId) -> Unit,
    onRemoveSource: (FaceId) -> Unit,
    onRestorationEnabledChange: (Boolean) -> Unit,
    onRestorationStrengthChange: (Float) -> Unit,
    onParserSwapMaskEnabledChange: (Boolean) -> Unit,
    onRunPhotoSwap: () -> Unit,
    onCancelPhotoSwap: () -> Unit,
    onExportFormatChange: (ExportFormat) -> Unit,
    onJpegQualityChange: (Int) -> Unit,
    onWatermarkEnabledChange: (Boolean) -> Unit,
    onExport: () -> Unit,
    onDismissExportError: () -> Unit,
    onPhotoResultDisposed: (MultiPhotoFaceSwapResult) -> Unit,
    onDismissError: () -> Unit,
) {
    Scaffold(
        topBar = {
            TopAppBar(
                title = {
                    Column {
                        Text("FaceSwap Local", fontWeight = FontWeight.SemiBold)
                        Text(
                            text = "Фото · офлайн",
                            style = MaterialTheme.typography.labelMedium,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                        )
                    }
                },
            )
        },
    ) { scaffoldPadding ->
        // §9.4: the screen must not fall asleep during a short photo run.
        KeepScreenOnWhile(state.isProcessing)

        Column(
            modifier = Modifier
                .fillMaxSize()
                .testTag("main-scroll")
                .padding(scaffoldPadding)
                .padding(horizontal = 16.dp, vertical = 12.dp)
                .verticalScroll(rememberScrollState()),
            verticalArrangement = Arrangement.spacedBy(14.dp),
        ) {
            PrivacyBanner()

            ModelSetupCard(
                statuses = state.modelStatuses,
                message = state.modelMessage,
                onImportModel = onImportModel,
            )

            MediaPickerCard(
                title = "1. Лица-источники",
                description = "Выберите фото с одним или несколькими лицами, которые будем переносить.",
                isSelected = state.sourceUris.isNotEmpty(),
                bitmap = state.sourceBitmap,
                faces = state.sourceFaces,
                onPick = onPickSource,
            )

            MediaPickerCard(
                title = "2. Целевая фотография",
                description = "Лица на этом фото можно будет заменить независимо друг от друга.",
                isSelected = state.targetUri != null,
                bitmap = state.targetBitmap,
                faces = state.targetFaces,
                onPick = onPickTarget,
            )

            if (state.phase == AnalysisPhase.ANALYZING) {
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(vertical = 10.dp),
                    horizontalArrangement = Arrangement.Center,
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    CircularProgressIndicator(modifier = Modifier.size(24.dp), strokeWidth = 3.dp)
                    Spacer(Modifier.size(12.dp))
                    Text("Ищу лица на устройстве…")
                }
            } else {
                Button(
                    onClick = onAnalyze,
                    enabled = state.canAnalyze,
                    modifier = Modifier.fillMaxWidth(),
                ) {
                    Text(if (state.phase == AnalysisPhase.MAPPING) "Найти лица заново" else "Найти лица")
                }
            }

            state.errorMessage?.let { message ->
                ErrorCard(message = message, onDismiss = onDismissError)
            }

            if (state.phase == AnalysisPhase.MAPPING) {
                AssignmentCard(
                    sourceFaces = state.sourceFaces,
                    targetFaces = state.targetFaces,
                    assignments = state.assignments,
                    onAssignSource = onAssignSource,
                    onSetUnchanged = onSetUnchanged,
                    onApplySourceToAll = onApplySourceToAll,
                    onRemoveSource = onRemoveSource,
                )

                PhotoSwapCard(
                    state = state,
                    onRestorationEnabledChange = onRestorationEnabledChange,
                    onRestorationStrengthChange = onRestorationStrengthChange,
                    onParserSwapMaskEnabledChange = onParserSwapMaskEnabledChange,
                    onRunPhotoSwap = onRunPhotoSwap,
                    onCancelPhotoSwap = onCancelPhotoSwap,
                    onPhotoResultDisposed = onPhotoResultDisposed,
                )

                if (state.photoSwapResult != null) {
                    ExportCard(
                        state = state,
                        onExportFormatChange = onExportFormatChange,
                        onJpegQualityChange = onJpegQualityChange,
                        onWatermarkEnabledChange = onWatermarkEnabledChange,
                        onExport = onExport,
                        onDismissExportError = onDismissExportError,
                    )
                }
            }

            Spacer(Modifier.height(24.dp))
        }
    }
}

/**
 * Holds the screen awake only while a run is in flight. The flag is bound to the
 * composition's own view, so it is released on dispose even if processing ends through
 * cancellation or an error.
 */
@Composable
private fun KeepScreenOnWhile(active: Boolean) {
    val view = LocalView.current
    DisposableEffect(view, active) {
        view.keepScreenOn = active
        onDispose { view.keepScreenOn = false }
    }
}

@Composable
private fun PrivacyBanner() {
    Surface(
        color = MaterialTheme.colorScheme.secondaryContainer,
        contentColor = MaterialTheme.colorScheme.onSecondaryContainer,
        shape = RoundedCornerShape(16.dp),
    ) {
        Column(Modifier.padding(16.dp)) {
            Text("Только на телефоне", fontWeight = FontWeight.SemiBold)
            Spacer(Modifier.height(4.dp))
            Text(
                "У приложения нет сетевого разрешения. Выбранные фотографии никуда не отправляются.",
                style = MaterialTheme.typography.bodyMedium,
            )
        }
    }
}

@Composable
private fun ModelSetupCard(
    statuses: Map<ModelId, ModelStatus>,
    message: String?,
    onImportModel: (ModelId) -> Unit,
) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.tertiaryContainer),
    ) {
        Column(
            modifier = Modifier.padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            Text(
                text = "Локальные модели · фото-пайплайн",
                style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.Bold,
            )
            Text(
                text = "Веса не входят в APK и не отправляются в сеть. Выберите официальные ONNX-файлы: приложение скопирует их в приватное хранилище только после проверки полного SHA-256.",
                style = MaterialTheme.typography.bodyMedium,
            )

            ModelCatalog.all.forEach { descriptor ->
                ModelImportRow(
                    descriptor = descriptor,
                    status = statuses[descriptor.id] ?: ModelStatus.Missing,
                    onImport = { onImportModel(descriptor.id) },
                )
            }

            Text(
                text = "Рабочий swapper: InSwapper fp16 · 128×128. BiSeNet защищает границы лица, а GFPGAN восстанавливает детали с выбранной ниже силой.",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )

            message?.let {
                Surface(
                    color = MaterialTheme.colorScheme.surface.copy(alpha = 0.7f),
                    shape = RoundedCornerShape(10.dp),
                ) {
                    Text(it, modifier = Modifier.padding(10.dp), style = MaterialTheme.typography.bodySmall)
                }
            }
        }
    }
}

@Composable
private fun ModelImportRow(
    descriptor: ModelDescriptor,
    status: ModelStatus,
    onImport: () -> Unit,
) {
    Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Column(modifier = Modifier.weight(1f)) {
                Text(modelDisplayName(descriptor.id), fontWeight = FontWeight.SemiBold)
                Text(
                    text = "${descriptor.expectedSizeBytes / (1024 * 1024)} МиБ · ${descriptor.expectedSha256.take(12)}…",
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
            OutlinedButton(
                onClick = onImport,
                enabled = status !is ModelStatus.Importing,
            ) {
                Text(if (status is ModelStatus.Ready) "Заменить" else "Импорт")
            }
        }
        Text(
            text = modelStatusText(status),
            style = MaterialTheme.typography.bodySmall,
            color = when (status) {
                is ModelStatus.Ready -> MaterialTheme.colorScheme.primary
                is ModelStatus.Invalid, is ModelStatus.Failed -> MaterialTheme.colorScheme.error
                else -> MaterialTheme.colorScheme.onSurfaceVariant
            },
        )
    }
}

private fun modelDisplayName(id: ModelId): String = when (id) {
    ModelId.YOLOFACE_8N -> "YOLOFace 8n · 5 точек"
    ModelId.ARCFACE_W600K_R50 -> "ArcFace w600k_r50 · embedding"
    ModelId.HYPERSWAP_1A_256 -> "HyperSwap 1a · основной кандидат"
    ModelId.INSWAPPER_128_FP16 -> "InSwapper 128 fp16 · активный fallback"
    ModelId.GFPGAN_1_4 -> "GFPGAN 1.4 · восстановление деталей"
    ModelId.BISENET_RESNET_34 -> "BiSeNet ResNet-34 · маска лица"
}

private fun modelStatusText(status: ModelStatus): String = when (status) {
    ModelStatus.Missing -> "Не импортирована"
    ModelStatus.PresentUnverified -> "Найдена приватная копия; ожидает проверки"
    is ModelStatus.Importing -> {
        val percent = if (status.expectedBytes > 0L) {
            (status.bytesCopied * 100L / status.expectedBytes).coerceIn(0L, 100L)
        } else {
            0L
        }
        "Копирование и SHA-256: $percent%"
    }

    is ModelStatus.Ready -> "Готова · SHA-256 проверен (${status.verifiedSizeBytes} байт)"
    is ModelStatus.Invalid -> when (status.details.reason) {
        com.faceswaplocal.app.inference.ModelValidationFailure.SIZE_MISMATCH -> "Отклонена: неверный размер"
        com.faceswaplocal.app.inference.ModelValidationFailure.CHECKSUM_MISMATCH -> "Отклонена: SHA-256 не совпадает"
    }

    is ModelStatus.Failed -> "Ошибка импорта: ${status.reason.name}"
}

@Composable
private fun MediaPickerCard(
    title: String,
    description: String,
    isSelected: Boolean,
    bitmap: Bitmap?,
    faces: List<DetectedFace>,
    onPick: () -> Unit,
) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceContainer),
    ) {
        Column(
            modifier = Modifier.padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(10.dp),
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Text(title, style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold)
                OutlinedButton(onClick = onPick) {
                    Text(if (isSelected) "Изменить" else "Выбрать")
                }
            }
            Text(
                description,
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )

            when {
                bitmap != null -> {
                    FacePreview(bitmap = bitmap, faces = faces)
                    Text(
                        text = "Найдено лиц: ${faces.size}",
                        style = MaterialTheme.typography.labelLarge,
                    )
                }

                isSelected -> Text(
                    text = "Фотография выбрана",
                    color = MaterialTheme.colorScheme.primary,
                    fontWeight = FontWeight.Medium,
                )
            }
        }
    }
}

@Composable
private fun FacePreview(bitmap: Bitmap, faces: List<DetectedFace>) {
    val imageBitmap = remember(bitmap) { bitmap.asImageBitmap() }
    val outlineColor = MaterialTheme.colorScheme.primary

    Box(
        modifier = Modifier
            .fillMaxWidth()
            .height(260.dp)
            .background(Color.Black, RoundedCornerShape(14.dp)),
    ) {
        Image(
            bitmap = imageBitmap,
            contentDescription = "Выбранная фотография",
            modifier = Modifier.matchParentSize(),
            contentScale = ContentScale.Fit,
        )
        Canvas(Modifier.matchParentSize()) {
            val scale = min(size.width / bitmap.width, size.height / bitmap.height)
            val renderedWidth = bitmap.width * scale
            val renderedHeight = bitmap.height * scale
            val offsetX = (size.width - renderedWidth) / 2f
            val offsetY = (size.height - renderedHeight) / 2f

            faces.forEachIndexed { index, face ->
                val left = offsetX + face.bounds.left * renderedWidth
                val top = offsetY + face.bounds.top * renderedHeight
                val width = face.bounds.width * renderedWidth
                val height = face.bounds.height * renderedHeight
                val badgeRadius = 13.dp.toPx()
                val badgeCenter = Offset(left + badgeRadius, top + badgeRadius)

                drawRoundRect(
                    color = outlineColor,
                    topLeft = Offset(left, top),
                    size = Size(width, height),
                    cornerRadius = CornerRadius(10.dp.toPx()),
                    style = Stroke(width = 3.dp.toPx()),
                )
                drawCircle(color = outlineColor, radius = badgeRadius, center = badgeCenter)

                val labelPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
                    color = android.graphics.Color.WHITE
                    textAlign = Paint.Align.CENTER
                    textSize = 14.sp.toPx()
                    typeface = android.graphics.Typeface.DEFAULT_BOLD
                }
                drawContext.canvas.nativeCanvas.drawText(
                    (index + 1).toString(),
                    badgeCenter.x,
                    badgeCenter.y - (labelPaint.ascent() + labelPaint.descent()) / 2f,
                    labelPaint,
                )
            }
        }
    }
}

@Composable
private fun AssignmentCard(
    sourceFaces: List<DetectedFace>,
    targetFaces: List<DetectedFace>,
    assignments: List<SwapAssignment>,
    onAssignSource: (FaceId, FaceId) -> Unit,
    onSetUnchanged: (FaceId) -> Unit,
    onApplySourceToAll: (FaceId) -> Unit,
    onRemoveSource: (FaceId) -> Unit,
) {
    var confirmSource by remember { mutableStateOf<FaceId?>(null) }
    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.primaryContainer),
    ) {
        Column(
            modifier = Modifier.padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(14.dp),
        ) {
            Text("3. Кто кого заменяет", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.Bold)
            Text(
                "Для каждого номера на целевом фото выберите номер лица с фотографии-источника.",
                style = MaterialTheme.typography.bodyMedium,
            )

            targetFaces.forEachIndexed { targetIndex, target ->
                val selectedSourceId = assignments
                    .firstOrNull { it.targetFaceId == target.id }
                    ?.sourceFaceId

                Column(
                    modifier = Modifier.testTag("target-$targetIndex"),
                    verticalArrangement = Arrangement.spacedBy(6.dp),
                ) {
                    Text("Целевое лицо ${targetIndex + 1}", fontWeight = FontWeight.SemiBold)
                    Row(
                        modifier = Modifier.horizontalScroll(rememberScrollState()),
                        horizontalArrangement = Arrangement.spacedBy(8.dp),
                    ) {
                        sourceFaces.forEachIndexed { sourceIndex, source ->
                            FilterChip(
                                modifier = Modifier.testTag("assign-$targetIndex-$sourceIndex"),
                                selected = selectedSourceId == source.id,
                                onClick = { onAssignSource(target.id, source.id) },
                                label = { Text("Источник ${sourceIndex + 1}") },
                            )
                        }
                        FilterChip(
                            modifier = Modifier.testTag("unchanged-$targetIndex"),
                            selected = selectedSourceId == null,
                            onClick = { onSetUnchanged(target.id) },
                            label = { Text("Не менять") },
                        )
                    }
                }
            }

            if (sourceFaces.isNotEmpty()) {
                Text("Применить источник ко всем", fontWeight = FontWeight.SemiBold)
                Row(modifier = Modifier.horizontalScroll(rememberScrollState()), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    sourceFaces.forEachIndexed { index, source ->
                        OutlinedButton(
                            modifier = Modifier.testTag("apply-all-$index"),
                            onClick = { confirmSource = source.id },
                        ) { Text("Источник ${index + 1}") }
                    }
                }
            }
            Text("Удалить источник до обработки", style = MaterialTheme.typography.labelMedium)
            Row(
                modifier = Modifier
                    .testTag("remove-sources-row")
                    .horizontalScroll(rememberScrollState()),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                sourceFaces.forEachIndexed { index, source ->
                    OutlinedButton(
                        modifier = Modifier.testTag("remove-source-$index"),
                        onClick = { onRemoveSource(source.id) },
                    ) { Text("Удалить ${index + 1}") }
                }
            }

            Surface(
                color = MaterialTheme.colorScheme.surface.copy(alpha = 0.62f),
                shape = RoundedCornerShape(12.dp),
            ) {
                Text(
                    text = "Нейропайплайн связывает выбранную рамку с независимым 5-точечным YOLOFace. Рамки ML Kit остаются только быстрым UI-превью.",
                    modifier = Modifier.padding(12.dp),
                    style = MaterialTheme.typography.bodyMedium,
                )
            }
        }
    }
    confirmSource?.let { sourceId ->
        AlertDialog(
            modifier = Modifier.testTag("apply-all-dialog"),
            onDismissRequest = { confirmSource = null },
            title = { Text("Применить ко всем?") },
            text = { Text("Выбранный источник заменит все целевые лица. Это действие можно изменить для каждого лица.") },
            confirmButton = {
                Button(
                    modifier = Modifier.testTag("confirm-apply-all"),
                    onClick = { onApplySourceToAll(sourceId); confirmSource = null },
                ) { Text("Применить") }
            },
            dismissButton = { OutlinedButton(onClick = { confirmSource = null }) { Text("Отмена") } },
        )
    }
}

@Composable
private fun PhotoSwapCard(
    state: FaceSwapUiState,
    onRestorationEnabledChange: (Boolean) -> Unit,
    onRestorationStrengthChange: (Float) -> Unit,
    onParserSwapMaskEnabledChange: (Boolean) -> Unit,
    onRunPhotoSwap: () -> Unit,
    onCancelPhotoSwap: () -> Unit,
    onPhotoResultDisposed: (MultiPhotoFaceSwapResult) -> Unit,
) {
    val controlsEnabled = !state.isProcessing
    val settings = state.qualitySettings
    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.secondaryContainer),
    ) {
        Column(
            modifier = Modifier.padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            Text(
                text = "4. Обработать фото",
                style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.Bold,
            )
            Text(
                text = "InSwapper создаёт новое лицо, BiSeNet при необходимости защищает границы волос и шеи, а отдельный второй проход GFPGAN восстанавливает детали.",
                style = MaterialTheme.typography.bodyMedium,
            )

            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Column(modifier = Modifier.weight(1f)) {
                    Text("Восстановление деталей", fontWeight = FontWeight.SemiBold)
                    Text(
                        "GFPGAN обрабатывает только назначенные лица",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
                Switch(
                    checked = settings.restorationEnabled,
                    onCheckedChange = onRestorationEnabledChange,
                    enabled = controlsEnabled,
                    modifier = Modifier.testTag("restoration-enabled"),
                )
            }
            Text(
                text = "Сила восстановления: ${(settings.restorationStrength * 100).toInt()}%",
                modifier = Modifier.testTag("restoration-strength-label"),
                style = MaterialTheme.typography.bodyMedium,
            )
            Slider(
                value = settings.restorationStrength,
                onValueChange = onRestorationStrengthChange,
                valueRange = 0f..1f,
                steps = 9,
                enabled = controlsEnabled && settings.restorationEnabled,
                modifier = Modifier
                    .fillMaxWidth()
                    .testTag("restoration-strength"),
            )
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Column(modifier = Modifier.weight(1f)) {
                    Text("BiSeNet-маска при свапе", fontWeight = FontWeight.SemiBold)
                    Text(
                        "Выключение возвращает аффинную box-mask; при GFPGAN BiSeNet всё равно защищает проход восстановления",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
                Switch(
                    checked = settings.parserSwapMaskEnabled,
                    onCheckedChange = onParserSwapMaskEnabledChange,
                    enabled = controlsEnabled,
                    modifier = Modifier.testTag("parser-swap-mask-enabled"),
                )
            }

            Button(
                onClick = onRunPhotoSwap,
                enabled = state.canRunPhotoSwap,
                modifier = Modifier
                    .fillMaxWidth()
                    .testTag("run-photo-swap"),
            ) {
                Text(
                    when (state.photoSwapPhase) {
                        PhotoSwapPhase.RUNNING -> "Локальная замена выполняется…"
                        PhotoSwapPhase.CANCELLING -> "Останавливаю на безопасной границе…"
                        PhotoSwapPhase.READY -> "Повторить обработку"
                        else -> "Обработать фото"
                    },
                )
            }

            if (!state.canRunPhotoSwap && !state.isProcessing) {
                Text(
                    text = "Для запуска нужны назначения и проверенные YOLOFace, ArcFace и InSwapper. BiSeNet и GFPGAN требуются только выбранным режимам качества.",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }

            if (state.isProcessing || state.exportPhase == ExportPhase.RUNNING) {
                ProcessingProgressView(
                    progress = state.processingProgress,
                    cancelling = state.photoSwapPhase == PhotoSwapPhase.CANCELLING,
                    // Saving is a short, atomic step: it is reported, but the cancel
                    // action stays bound to the long inference run.
                    showCancel = state.isProcessing,
                    canCancel = state.canCancelPhotoSwap,
                    onCancel = onCancelPhotoSwap,
                )
            }

            state.photoSwapError?.let { message ->
                Surface(
                    color = MaterialTheme.colorScheme.errorContainer,
                    contentColor = MaterialTheme.colorScheme.onErrorContainer,
                    shape = RoundedCornerShape(12.dp),
                ) {
                    Text(message, modifier = Modifier.padding(12.dp))
                }
            }

            state.photoSwapResult?.let { result ->
                val original = state.targetBitmap
                if (original != null) {
                    PhotoSwapResultView(
                        original = original,
                        result = result,
                        onResultDisposed = onPhotoResultDisposed,
                    )
                }
            }
        }
    }
}

/**
 * FR-PHOTO-07: stage, face number and total, plus a cancel action. No remaining-time
 * estimate is shown, as §9.4 forbids promising one without statistics.
 */
@Composable
private fun ProcessingProgressView(
    progress: ProcessingProgress?,
    cancelling: Boolean,
    showCancel: Boolean,
    canCancel: Boolean,
    onCancel: () -> Unit,
) {
    Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
        Row(
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(10.dp),
        ) {
            CircularProgressIndicator(modifier = Modifier.size(20.dp), strokeWidth = 3.dp)
            Text(
                text = if (cancelling) {
                    "Отмена: останавливаю на ближайшей безопасной границе…"
                } else {
                    progressStatusText(progress)
                },
                modifier = Modifier.testTag("processing-status"),
                style = MaterialTheme.typography.bodyMedium,
            )
        }
        LinearProgressIndicator(
            progress = { progress?.fraction ?: 0f },
            modifier = Modifier
                .fillMaxWidth()
                .testTag("processing-progress"),
        )
        if (showCancel) {
            OutlinedButton(
                onClick = onCancel,
                enabled = canCancel,
                modifier = Modifier.testTag("cancel-photo-swap"),
            ) {
                Text("Отменить")
            }
            Text(
                text = "Предыдущий результат остаётся доступен до готовности нового.",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
    }
}

private fun progressStatusText(progress: ProcessingProgress?): String {
    if (progress == null) return "Подготовка…"
    val stage = when (progress.stage) {
        ProcessingStage.PREPARING -> "Подготовка"
        ProcessingStage.DETECTING -> "Поиск лиц нейродетектором"
        ProcessingStage.SWAPPING -> "Замена лица"
        ProcessingStage.RESTORING -> "Восстановление деталей"
        ProcessingStage.EXPORTING -> "Сохранение в галерею"
        ProcessingStage.COMPLETED -> "Готово"
    }
    return if (progress.currentFace > 0) {
        "$stage · лицо ${progress.currentFace} из ${progress.totalFaces}"
    } else {
        stage
    }
}

/** FR-PHOTO-09 export controls and the saved-file confirmation. */
@Composable
private fun ExportCard(
    state: FaceSwapUiState,
    onExportFormatChange: (ExportFormat) -> Unit,
    onJpegQualityChange: (Int) -> Unit,
    onWatermarkEnabledChange: (Boolean) -> Unit,
    onExport: () -> Unit,
    onDismissExportError: () -> Unit,
) {
    val settings = state.exportSettings
    val controlsEnabled = state.exportPhase != ExportPhase.RUNNING &&
        state.exportDestinationRequest == null
    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.tertiaryContainer),
    ) {
        Column(
            modifier = Modifier.padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            Text(
                text = "5. Сохранить в галерею",
                style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.Bold,
            )
            Text(
                text = "Файл создаётся рядом с оригиналом под именем FaceSwapLocal_дата_время. Исходная фотография не изменяется, геолокация и другие EXIF-поля оригинала в новый файл не переносятся.",
                style = MaterialTheme.typography.bodyMedium,
            )

            if (state.exportIsDownscaled) {
                val source = state.targetSourceSize
                val decoded = state.targetBitmap
                Text(
                    text = "Свободной памяти не хватило на полный размер: оригинал " +
                        "${source?.width}×${source?.height} обработан как " +
                        "${decoded?.width}×${decoded?.height}, файл будет сохранён в этом размере.",
                    modifier = Modifier.testTag("export-downscale-notice"),
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.error,
                )
            }

            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                FilterChip(
                    selected = settings.format == ExportFormat.JPEG,
                    onClick = { onExportFormatChange(ExportFormat.JPEG) },
                    enabled = controlsEnabled,
                    label = { Text("JPEG") },
                    modifier = Modifier.testTag("export-format-jpeg"),
                )
                FilterChip(
                    selected = settings.format == ExportFormat.PNG,
                    onClick = { onExportFormatChange(ExportFormat.PNG) },
                    enabled = controlsEnabled,
                    label = { Text("PNG") },
                    modifier = Modifier.testTag("export-format-png"),
                )
            }

            Text(
                text = if (settings.format == ExportFormat.JPEG) {
                    "Качество JPEG: ${settings.jpegQuality}"
                } else {
                    "PNG сохраняется без потерь, ползунок качества не применяется"
                },
                modifier = Modifier.testTag("export-quality-label"),
                style = MaterialTheme.typography.bodyMedium,
            )
            Slider(
                value = settings.jpegQuality.toFloat(),
                onValueChange = { value -> onJpegQualityChange(value.toInt()) },
                valueRange = ExportSettings.MIN_JPEG_QUALITY.toFloat()..ExportSettings.MAX_JPEG_QUALITY.toFloat(),
                steps = ExportSettings.MAX_JPEG_QUALITY - ExportSettings.MIN_JPEG_QUALITY - 1,
                enabled = controlsEnabled && settings.format == ExportFormat.JPEG,
                modifier = Modifier
                    .fillMaxWidth()
                    .testTag("export-jpeg-quality"),
            )

            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Column(modifier = Modifier.weight(1f)) {
                    Text("Видимый водяной знак", fontWeight = FontWeight.SemiBold)
                    Text(
                        "Включён по умолчанию; выключается для личного использования",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
                Switch(
                    checked = settings.watermarkEnabled,
                    onCheckedChange = onWatermarkEnabledChange,
                    enabled = controlsEnabled,
                    modifier = Modifier.testTag("export-watermark"),
                )
            }

            Button(
                onClick = onExport,
                enabled = state.canExport,
                modifier = Modifier
                    .fillMaxWidth()
                    .testTag("export-save"),
            ) {
                Text(
                    when {
                        state.exportPhase == ExportPhase.RUNNING -> "Сохраняю…"
                        state.savedExport != null -> "Сохранить ещё раз"
                        else -> "Сохранить в галерею"
                    },
                )
            }

            if (state.exportDestinationRequest != null) {
                Text(
                    text = "Android 9 не поддерживает отложенную запись в галерею без полного доступа к хранилищу, поэтому выберите папку и имя файла в системном диалоге.",
                    modifier = Modifier.testTag("export-destination-hint"),
                    style = MaterialTheme.typography.bodySmall,
                )
            }

            state.exportError?.let { message ->
                ErrorCard(message = message, onDismiss = onDismissExportError)
            }

            state.savedExport?.let { saved ->
                SavedExportView(saved = saved, preview = state.photoSwapResult?.finalBitmap)
            }
        }
    }
}

@Composable
private fun SavedExportView(saved: SavedExport, preview: Bitmap?) {
    val context = LocalContext.current
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.spacedBy(12.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        if (preview != null && !preview.isRecycled) {
            Image(
                bitmap = remember(preview) { preview.asImageBitmap() },
                contentDescription = "Миниатюра сохранённого файла",
                modifier = Modifier
                    .size(64.dp)
                    .background(Color.Black, RoundedCornerShape(10.dp))
                    .testTag("export-thumbnail"),
                contentScale = ContentScale.Crop,
            )
        }
        Column(modifier = Modifier.weight(1f)) {
            Text("Сохранено", fontWeight = FontWeight.SemiBold)
            Text(
                text = saved.displayName,
                modifier = Modifier.testTag("export-file-name"),
                style = MaterialTheme.typography.bodySmall,
            )
            Text(
                text = buildString {
                    saved.album?.let { append("Альбом $it · ") }
                    append("${saved.width}×${saved.height}")
                },
                modifier = Modifier.testTag("export-location"),
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
        OutlinedButton(
            onClick = { openSavedImage(context, saved.uri) },
            modifier = Modifier.testTag("export-open"),
        ) {
            Text("Открыть")
        }
    }
}

/** Hands the saved file to the system viewer; no data leaves the device. */
private fun openSavedImage(context: android.content.Context, uri: Uri) {
    val intent = Intent(Intent.ACTION_VIEW).apply {
        setDataAndType(uri, "image/*")
        addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
    }
    try {
        context.startActivity(intent)
    } catch (_: ActivityNotFoundException) {
        // A device without a gallery viewer keeps the saved file; nothing else to do.
    }
}

@Composable
private fun PhotoSwapResultView(
    original: Bitmap,
    result: MultiPhotoFaceSwapResult,
    onResultDisposed: (MultiPhotoFaceSwapResult) -> Unit,
) {
    var showOriginal by remember(result.finalBitmap) { mutableStateOf(false) }
    val originalImage = remember(original) { original.asImageBitmap() }
    val resultImage = remember(result.finalBitmap) { result.finalBitmap.asImageBitmap() }
    DisposableEffect(result.finalBitmap) {
        onDispose { onResultDisposed(result) }
    }
    Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            FilterChip(
                selected = showOriginal,
                onClick = { showOriginal = true },
                label = { Text("До") },
                modifier = Modifier.testTag("compare-show-before"),
            )
            FilterChip(
                selected = !showOriginal,
                onClick = { showOriginal = false },
                label = { Text("После") },
                modifier = Modifier.testTag("compare-show-after"),
            )
        }
        Surface(
            modifier = Modifier.fillMaxWidth(),
            color = Color.Black,
            shape = RoundedCornerShape(14.dp),
        ) {
            Image(
                bitmap = if (showOriginal) originalImage else resultImage,
                contentDescription = if (showOriginal) {
                    "Оригинал до обработки"
                } else {
                    "Результат после обработки"
                },
                modifier = Modifier
                    .fillMaxWidth()
                    .height(420.dp)
                    .testTag(if (showOriginal) "comparison-before" else "comparison-after"),
                contentScale = ContentScale.Fit,
            )
        }
        Text(
            text = "InSwapper fp16 · итог ${result.finalBitmap.width}×${result.finalBitmap.height}",
            fontWeight = FontWeight.SemiBold,
        )
        Text(
            text = "Результат находится только в памяти. Сохранение выполняется отдельной кнопкой ниже.",
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
        Text(
            text = "Применено: восстановление ${if (result.restorationStrength > 0f) "${(result.restorationStrength * 100).toInt()}%" else "выключено"} · BiSeNet-маска свапа ${if (result.swapParserBackends.isNotEmpty()) "включена" else "выключена"}",
            modifier = Modifier.testTag("applied-quality-settings"),
            style = MaterialTheme.typography.bodySmall,
        )
        Text(
            text = "Backend: detector=${result.detectorBackend}, ArcFace=${result.recognizerBackend}, swapper=${result.swapperBackend}",
            style = MaterialTheme.typography.bodySmall,
        )
        Text(
            text = "Время: detector ${result.timings.detectorMs} мс · embedding ${result.timings.recognizerMs} мс · swapper ${result.timings.swapperMs} мс · blend ${result.timings.compositingMs} мс · всего ${result.timings.totalMs} мс",
            style = MaterialTheme.typography.bodySmall,
        )
    }
}

@Composable
private fun ErrorCard(message: String, onDismiss: () -> Unit) {
    Surface(
        color = MaterialTheme.colorScheme.errorContainer,
        contentColor = MaterialTheme.colorScheme.onErrorContainer,
        shape = RoundedCornerShape(14.dp),
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(14.dp),
            horizontalArrangement = Arrangement.spacedBy(12.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Text(message, modifier = Modifier.weight(1f), style = MaterialTheme.typography.bodyMedium)
            OutlinedButton(onClick = onDismiss) {
                Text("OK")
            }
        }
    }
}
