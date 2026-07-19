package com.faceswaplocal.app.ui

import android.graphics.Bitmap
import android.graphics.Paint
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts.PickVisualMedia
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.horizontalScroll
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
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.FilterChip
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
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
import com.faceswaplocal.app.domain.FaceId
import com.faceswaplocal.app.domain.SwapAssignment
import kotlin.math.min

@Composable
fun FaceSwapRoute(viewModel: FaceSwapViewModel = viewModel()) {
    val state by viewModel.state.collectAsStateWithLifecycle()
    val sourcePicker = rememberLauncherForActivityResult(PickVisualMedia()) { uri ->
        uri?.let(viewModel::selectSource)
    }
    val targetPicker = rememberLauncherForActivityResult(PickVisualMedia()) { uri ->
        uri?.let(viewModel::selectTarget)
    }

    FaceSwapScreen(
        state = state,
        onPickSource = {
            sourcePicker.launch(PickVisualMediaRequest(PickVisualMedia.ImageOnly))
        },
        onPickTarget = {
            targetPicker.launch(PickVisualMediaRequest(PickVisualMedia.ImageOnly))
        },
        onAnalyze = viewModel::analyze,
        onAssignSource = viewModel::assignSource,
        onDismissError = viewModel::dismissError,
    )
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun FaceSwapScreen(
    state: FaceSwapUiState,
    onPickSource: () -> Unit,
    onPickTarget: () -> Unit,
    onAnalyze: () -> Unit,
    onAssignSource: (FaceId, FaceId) -> Unit,
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
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(scaffoldPadding)
                .padding(horizontal = 16.dp, vertical = 12.dp)
                .verticalScroll(rememberScrollState()),
            verticalArrangement = Arrangement.spacedBy(14.dp),
        ) {
            PrivacyBanner()

            MediaPickerCard(
                title = "1. Лица-источники",
                description = "Выберите фото с одним или несколькими лицами, которые будем переносить.",
                isSelected = state.sourceUri != null,
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
                )
            }

            Spacer(Modifier.height(24.dp))
        }
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
) {
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

                Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
                    Text("Целевое лицо ${targetIndex + 1}", fontWeight = FontWeight.SemiBold)
                    Row(
                        modifier = Modifier.horizontalScroll(rememberScrollState()),
                        horizontalArrangement = Arrangement.spacedBy(8.dp),
                    ) {
                        sourceFaces.forEachIndexed { sourceIndex, source ->
                            FilterChip(
                                selected = selectedSourceId == source.id,
                                onClick = { onAssignSource(target.id, source.id) },
                                label = { Text("Источник ${sourceIndex + 1}") },
                            )
                        }
                    }
                }
            }

            Surface(
                color = MaterialTheme.colorScheme.surface.copy(alpha = 0.62f),
                shape = RoundedCornerShape(12.dp),
            ) {
                Text(
                    text = "Следующий этап проекта — подключение нейромодели и экспорт готового фото.",
                    modifier = Modifier.padding(12.dp),
                    style = MaterialTheme.typography.bodyMedium,
                )
            }
        }
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
