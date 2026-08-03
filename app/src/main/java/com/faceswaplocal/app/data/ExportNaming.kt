package com.faceswaplocal.app.data

import com.faceswaplocal.app.domain.ExportFormat
import java.time.Instant
import java.time.ZoneId
import java.time.format.DateTimeFormatter
import java.util.Locale

/**
 * Builds the `FaceSwapLocal_yyyyMMdd_HHmmss.ext` name required by FR-PHOTO-09.
 *
 * The timestamp is formatted in the device zone with a fixed [Locale.ROOT] pattern, so a
 * non-Gregorian or non-ASCII device locale cannot change the produced digits.
 */
object ExportNaming {
    const val PREFIX = "FaceSwapLocal"
    const val ALBUM_NAME = "FaceSwapLocal"

    private val TIMESTAMP = DateTimeFormatter
        .ofPattern("yyyyMMdd_HHmmss", Locale.ROOT)
        .withChronology(java.time.chrono.IsoChronology.INSTANCE)

    fun baseName(epochMillis: Long, zone: ZoneId): String {
        val local = Instant.ofEpochMilli(epochMillis).atZone(zone)
        return "${PREFIX}_${TIMESTAMP.format(local)}"
    }

    fun fileName(epochMillis: Long, zone: ZoneId, format: ExportFormat): String =
        "${baseName(epochMillis, zone)}.${format.extension}"
}
