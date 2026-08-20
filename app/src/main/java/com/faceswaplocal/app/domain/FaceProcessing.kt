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

/**
 * The three configurations the photo pipeline actually supports, plus the state the user
 * lands in after touching a control by hand.
 *
 * Nothing here invents a quality level: each preset is a combination of the two switches
 * that already exist, and the ordering follows their measured cost. The live acceptance
 * run showed GFPGAN taking 46 of 55 minutes, so restoration is the one axis worth a
 * top-level choice; the parser swap mask is the cheaper refinement on top.
 *
 * [CUSTOM] carries no settings: it is what the UI reports once a manual change no longer
 * matches any preset, so a moved slider is never silently attributed to a named mode.
 */
enum class QualityPreset(
    val settings: FaceQualitySettings?,
    /**
     * Seconds per assigned face measured by `QualityPresetBenchmarkInstrumentedTest` on
     * AVD API 35, CPU backend. An emulator figure, quoted as such in the UI: it is useful
     * for choosing between the modes, not for predicting a phone.
     */
    val emulatorSecondsPerFace: Int?,
) {
    /** No GFPGAN pass at all - the swap and its blend, nothing more. Measured 34.8 s. */
    FAST(
        FaceQualitySettings(
            restorationEnabled = false,
            restorationStrength = FaceQualitySettings.DEFAULT_RESTORATION_STRENGTH,
            parserSwapMaskEnabled = false,
        ),
        emulatorSecondsPerFace = 35,
    ),

    /**
     * GFPGAN at the established default strength, affine box mask for the swap blend.
     * Measured 80.9 s: the restoration pass alone accounts for 37.9 s of that.
     */
    BALANCED(
        FaceQualitySettings(
            restorationEnabled = true,
            restorationStrength = FaceQualitySettings.DEFAULT_RESTORATION_STRENGTH,
            parserSwapMaskEnabled = false,
        ),
        emulatorSecondsPerFace = 80,
    ),

    /**
     * GFPGAN plus the BiSeNet parser region for the swap blend. Measured 81.3 s — the
     * parser pass costs about 3 s, so this is effectively free next to [BALANCED].
     */
    MAXIMUM(
        FaceQualitySettings(
            restorationEnabled = true,
            restorationStrength = FaceQualitySettings.DEFAULT_RESTORATION_STRENGTH,
            parserSwapMaskEnabled = true,
        ),
        emulatorSecondsPerFace = 80,
    ),

    /** Whatever the user assembled by hand; never selectable, only reported. */
    CUSTOM(null, emulatorSecondsPerFace = null),
    ;

    companion object {
        /** The preset [settings] describe exactly, or [CUSTOM] when none of them do. */
        fun of(settings: FaceQualitySettings): QualityPreset =
            entries.firstOrNull { preset -> preset.settings == settings } ?: CUSTOM

        /** Presets a user can pick, in increasing cost. */
        val selectable: List<QualityPreset> get() = entries.filter { it.settings != null }
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

