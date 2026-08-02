package com.faceswaplocal.app.domain

import android.graphics.Bitmap
import java.io.Closeable

interface LocalFaceDetector : Closeable {
    suspend fun detect(bitmap: Bitmap, idPrefix: String): List<DetectedFace>
}

/** User-visible Stage-E quality controls, independent of a concrete inference runtime. */
data class FaceQualitySettings(
    val restorationEnabled: Boolean = true,
    val restorationStrength: Float = DEFAULT_RESTORATION_STRENGTH,
    val parserSwapMaskEnabled: Boolean = true,
) {
    init {
        require(restorationStrength.isFinite() && restorationStrength in 0f..1f) {
            "Restoration strength must be finite and within 0..1"
        }
    }

    /** A disabled restorer must not open GFPGAN even if its remembered slider is non-zero. */
    val effectiveRestorationStrength: Float
        get() = if (restorationEnabled) restorationStrength else 0f

    companion object {
        const val DEFAULT_RESTORATION_STRENGTH = 0.8f

        fun fromPersisted(
            restorationEnabled: Boolean?,
            restorationStrength: Float?,
            parserSwapMaskEnabled: Boolean?,
        ): FaceQualitySettings = FaceQualitySettings(
            restorationEnabled = restorationEnabled ?: true,
            restorationStrength = restorationStrength
                ?.takeIf { value -> value.isFinite() && value in 0f..1f }
                ?: DEFAULT_RESTORATION_STRENGTH,
            parserSwapMaskEnabled = parserSwapMaskEnabled ?: true,
        )
    }
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

