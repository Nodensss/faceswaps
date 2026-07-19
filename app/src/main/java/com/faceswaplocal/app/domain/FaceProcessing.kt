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
 * Boundary for the licensed neural runtime that will be added in the next checkpoint.
 * Runtime-specific tensors and sessions must stay behind this interface.
 */
interface FaceSwapEngine : Closeable {
    suspend fun swap(request: FaceSwapRequest): Bitmap
}

