package com.faceswaplocal.app.data

import com.faceswaplocal.app.domain.ExportFormat
import java.time.Instant
import java.time.ZoneId
import java.util.Locale
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class ExportNamingTest {
    private val defaultLocale = Locale.getDefault()

    @After
    fun restoreLocale() {
        Locale.setDefault(defaultLocale)
    }

    @Test
    fun `file name follows the required FaceSwapLocal timestamp pattern`() {
        val millis = Instant.parse("2026-08-03T12:13:14Z").toEpochMilli()

        assertEquals(
            "FaceSwapLocal_20260803_121314.jpg",
            ExportNaming.fileName(millis, ZoneId.of("UTC"), ExportFormat.JPEG),
        )
        assertEquals(
            "FaceSwapLocal_20260803_121314.png",
            ExportNaming.fileName(millis, ZoneId.of("UTC"), ExportFormat.PNG),
        )
    }

    @Test
    fun `timestamp uses the device zone, not UTC`() {
        val millis = Instant.parse("2026-08-03T23:30:00Z").toEpochMilli()

        assertEquals(
            "FaceSwapLocal_20260804_023000.jpg",
            ExportNaming.fileName(millis, ZoneId.of("Europe/Moscow"), ExportFormat.JPEG),
        )
    }

    /** A Buddhist- or Hijri-calendar device locale must not change the produced digits. */
    @Test
    fun `name is independent of the default locale calendar`() {
        val millis = Instant.parse("2026-08-03T12:13:14Z").toEpochMilli()
        val expected = ExportNaming.fileName(millis, ZoneId.of("UTC"), ExportFormat.JPEG)

        Locale.setDefault(Locale.forLanguageTag("th-TH-u-ca-buddhist-nu-thai"))
        assertEquals(expected, ExportNaming.fileName(millis, ZoneId.of("UTC"), ExportFormat.JPEG))

        Locale.setDefault(Locale.forLanguageTag("ar-SA-u-ca-islamic-umalqura-nu-arab"))
        assertEquals(expected, ExportNaming.fileName(millis, ZoneId.of("UTC"), ExportFormat.JPEG))
    }

    @Test
    fun `every second of a day produces a name matching the specified pattern`() {
        val pattern = Regex("""FaceSwapLocal_\d{8}_\d{6}\.(jpg|png)""")
        val base = Instant.parse("2026-01-01T00:00:00Z").toEpochMilli()

        (0 until 24 * 60 * 60 step 137).forEach { second ->
            val name = ExportNaming.fileName(
                base + second * 1_000L,
                ZoneId.of("UTC"),
                ExportFormat.JPEG,
            )
            assertTrue(name, pattern.matches(name))
        }
    }
}
