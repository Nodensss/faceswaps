package com.faceswaplocal.app.domain

/** Container formats the exporter can encode a result bitmap into (FR-PHOTO-09). */
enum class ExportFormat(val extension: String, val mimeType: String) {
    JPEG("jpg", "image/jpeg"),
    PNG("png", "image/png"),
}

/**
 * User-visible export controls. Runtime-specific encoder types stay in the data layer,
 * so this model can be unit-tested and persisted without Android dependencies.
 */
data class ExportSettings(
    val format: ExportFormat = ExportFormat.JPEG,
    val jpegQuality: Int = DEFAULT_JPEG_QUALITY,
    val watermarkEnabled: Boolean = true,
) {
    init {
        require(jpegQuality in MIN_JPEG_QUALITY..MAX_JPEG_QUALITY) {
            "JPEG quality must be within $MIN_JPEG_QUALITY..$MAX_JPEG_QUALITY"
        }
    }

    /** PNG is lossless: a remembered JPEG slider must never reach the encoder. */
    val effectiveQuality: Int
        get() = when (format) {
            ExportFormat.JPEG -> jpegQuality
            ExportFormat.PNG -> LOSSLESS_QUALITY
        }

    companion object {
        const val DEFAULT_JPEG_QUALITY = 95
        const val MIN_JPEG_QUALITY = 60
        const val MAX_JPEG_QUALITY = 100
        const val LOSSLESS_QUALITY = 100

        fun fromPersisted(
            formatName: String?,
            jpegQuality: Int?,
            watermarkEnabled: Boolean?,
        ): ExportSettings = ExportSettings(
            format = ExportFormat.entries.firstOrNull { it.name == formatName } ?: ExportFormat.JPEG,
            jpegQuality = jpegQuality
                ?.takeIf { value -> value in MIN_JPEG_QUALITY..MAX_JPEG_QUALITY }
                ?: DEFAULT_JPEG_QUALITY,
            watermarkEnabled = watermarkEnabled ?: true,
        )

        /** Slider values are clamped instead of rejected so the UI cannot construct an invalid state. */
        fun sanitizedQuality(value: Int): Int = value.coerceIn(MIN_JPEG_QUALITY, MAX_JPEG_QUALITY)
    }
}
