package com.faceswaplocal.app.ui

import androidx.lifecycle.SavedStateHandle
import com.faceswaplocal.app.domain.ExportFormat
import com.faceswaplocal.app.domain.ExportSettings
import org.junit.Assert.assertEquals
import org.junit.Test

class ExportSettingsSavedStateTest {
    @Test
    fun `export settings round trip through SavedStateHandle primitives`() {
        val handle = SavedStateHandle()
        val expected = ExportSettings(
            format = ExportFormat.PNG,
            jpegQuality = 71,
            watermarkEnabled = false,
        )

        ExportSettingsSavedState.write(handle, expected)

        assertEquals(expected, ExportSettingsSavedState.read(handle))
    }

    @Test
    fun `an empty handle yields the safe defaults instead of failing`() {
        assertEquals(ExportSettings(), ExportSettingsSavedState.read(SavedStateHandle()))
    }

    /** Only Parcelable-safe primitives may be written, or rotation would crash. */
    @Test
    fun `only primitives are stored`() {
        val handle = SavedStateHandle()

        ExportSettingsSavedState.write(handle, ExportSettings(ExportFormat.PNG, 80, false))

        assertEquals("PNG", handle.get<String>("photo_export_format"))
        assertEquals(80, handle.get<Int>("photo_export_jpeg_quality"))
        assertEquals(false, handle.get<Boolean>("photo_export_watermark_enabled"))
    }
}
