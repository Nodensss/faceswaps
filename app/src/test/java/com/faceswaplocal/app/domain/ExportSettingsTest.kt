package com.faceswaplocal.app.domain

import org.junit.Assert.assertEquals
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Test

class ExportSettingsTest {
    @Test
    fun `defaults are JPEG with a working quality and the watermark on`() {
        val defaults = ExportSettings()

        assertEquals(ExportFormat.JPEG, defaults.format)
        assertEquals(ExportSettings.DEFAULT_JPEG_QUALITY, defaults.jpegQuality)
        assertTrue("§5.3 requires the visible mark to be on by default", defaults.watermarkEnabled)
    }

    @Test
    fun `PNG ignores the remembered JPEG quality`() {
        val jpeg = ExportSettings(format = ExportFormat.JPEG, jpegQuality = 72)

        assertEquals(72, jpeg.effectiveQuality)
        assertEquals(
            ExportSettings.LOSSLESS_QUALITY,
            jpeg.copy(format = ExportFormat.PNG).effectiveQuality,
        )
        assertEquals(
            "switching back must restore the remembered slider",
            72,
            jpeg.copy(format = ExportFormat.PNG).copy(format = ExportFormat.JPEG).effectiveQuality,
        )
    }

    @Test
    fun `format carries the name extension and MIME type used by MediaStore`() {
        assertEquals("jpg", ExportFormat.JPEG.extension)
        assertEquals("image/jpeg", ExportFormat.JPEG.mimeType)
        assertEquals("png", ExportFormat.PNG.extension)
        assertEquals("image/png", ExportFormat.PNG.mimeType)
    }

    @Test
    fun `restored state falls back to defaults instead of failing`() {
        val fromEmptyProcess = ExportSettings.fromPersisted(null, null, null)
        assertEquals(ExportSettings(), fromEmptyProcess)

        val fromCorruptState = ExportSettings.fromPersisted("WEBP", 5, false)
        assertEquals(ExportFormat.JPEG, fromCorruptState.format)
        assertEquals(ExportSettings.DEFAULT_JPEG_QUALITY, fromCorruptState.jpegQuality)
        assertEquals(false, fromCorruptState.watermarkEnabled)
    }

    @Test
    fun `restored state keeps a valid saved selection`() {
        val restored = ExportSettings.fromPersisted("PNG", 70, false)

        assertEquals(ExportSettings(ExportFormat.PNG, 70, watermarkEnabled = false), restored)
    }

    @Test
    fun `slider values are clamped into the supported range`() {
        assertEquals(ExportSettings.MIN_JPEG_QUALITY, ExportSettings.sanitizedQuality(0))
        assertEquals(ExportSettings.MAX_JPEG_QUALITY, ExportSettings.sanitizedQuality(300))
        assertEquals(88, ExportSettings.sanitizedQuality(88))
    }

    @Test
    fun `an out-of-range quality cannot be constructed directly`() {
        assertThrows(IllegalArgumentException::class.java) { ExportSettings(jpegQuality = 10) }
        assertThrows(IllegalArgumentException::class.java) { ExportSettings(jpegQuality = 101) }
    }
}
