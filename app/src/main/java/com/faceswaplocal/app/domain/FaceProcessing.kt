package com.faceswaplocal.app.domain

import android.graphics.Bitmap
import java.io.Closeable

interface LocalFaceDetector : Closeable {
    suspend fun detect(bitmap: Bitmap, idPrefix: String): List<DetectedFace>
}

data class FaceSwapRequest(
    val sourceBitmap: Bitmap,
    val targetBitmap: Bitmap,
    val sourceFaces: List<DetectedFace>,
    val targetFaces: List<DetectedFace>,
    val assignments: List<SwapAssignment>,
)

/**
 * Domain boundary for the local neural runtime. The Stage C ONNX implementation lives
 * under `inference`; runtime-specific tensors and sessions must stay behind that layer.
 * The multi-assignment use case will bind this contract during Stage D.
 */
interface FaceSwapEngine : Closeable {
    suspend fun swap(request: FaceSwapRequest): Bitmap
}

