package com.faceswaplocal.app.data

import android.content.ContentResolver
import android.content.ContentValues
import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.media.ExifInterface
import android.net.Uri
import android.os.Bundle
import android.os.Environment
import android.provider.MediaStore
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import com.faceswaplocal.app.domain.ExportFormat
import com.faceswaplocal.app.domain.ExportSettings
import java.io.ByteArrayOutputStream
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.io.OutputStream
import java.security.MessageDigest
import java.time.Instant
import java.time.ZoneId
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith

/**
 * Stage E2 checkpoint 1 on a real API 35 device.
 *
 * Every case starts from a GPS-tagged JPEG standing in for the picked target photo, so
 * "the source file is untouched" and "source EXIF is not carried over" are measured, not
 * assumed.
 */
@RunWith(AndroidJUnit4::class)
class StageE2ExportInstrumentedTest {
    private val context = ApplicationProvider.getApplicationContext<Context>()
    private val resolver: ContentResolver get() = context.contentResolver
    private val zone = ZoneId.of("UTC")
    private val fixedMillis = Instant.parse("2026-08-03T09:41:07Z").toEpochMilli()

    private lateinit var workDirectory: File
    private lateinit var sourceFile: File
    private lateinit var sourceDigest: String
    private lateinit var resultBitmap: Bitmap
    private val createdRows = mutableListOf<Uri>()

    @Before
    fun createGpsTaggedSourcePhoto() {
        workDirectory = File(context.cacheDir, "stage-e2-export-test").apply {
            deleteRecursively()
            check(mkdirs()) { "Could not create $absolutePath" }
        }
        sourceFile = File(workDirectory, "target_with_gps.jpg")
        writeSourcePhoto(sourceFile)
        sourceDigest = sha256(sourceFile)

        val decodedTarget = BitmapFactory.decodeFile(sourceFile.absolutePath)
        resultBitmap = decodedTarget.copy(Bitmap.Config.ARGB_8888, true).also {
            decodedTarget.recycle()
            Canvas(it).drawCircle(
                it.width / 2f,
                it.height / 2f,
                it.width / 6f,
                Paint().apply { color = Color.MAGENTA },
            )
        }
    }

    @After
    fun cleanUp() {
        createdRows.forEach { uri -> runCatching { resolver.delete(uri, null, null) } }
        createdRows.clear()
        if (::resultBitmap.isInitialized && !resultBitmap.isRecycled) resultBitmap.recycle()
        workDirectory.deleteRecursively()
        runBlocking { exporter().sweepAbandonedData() }
    }

    @Test
    fun successfulExportKeepsRequiredNameAndTargetSizeWithoutSourceExif() = runBlocking {
        val exporter = exporter()
        val expectedName = ExportNaming.fileName(fixedMillis, zone, ExportFormat.JPEG)

        val outcome = exporter.export(
            bitmap = resultBitmap,
            settings = ExportSettings(format = ExportFormat.JPEG, jpegQuality = 95),
            nowMillis = fixedMillis,
            zone = zone,
        )

        val saved = outcome as? ExportOutcome.Saved
            ?: throw AssertionError("export failed: $outcome")
        createdRows += saved.uri
        assertEquals("FaceSwapLocal_20260803_094107.jpg", expectedName)
        assertEquals(expectedName, saved.displayName)
        assertEquals(ExportNaming.ALBUM_NAME, saved.album)

        val row = readRow(saved.uri)
        assertEquals(expectedName, row.getAsString(MediaStore.MediaColumns.DISPLAY_NAME))
        assertEquals("image/jpeg", row.getAsString(MediaStore.MediaColumns.MIME_TYPE))
        assertEquals(0, row.getAsInteger(MediaStore.MediaColumns.IS_PENDING))
        assertTrue(
            row.getAsString(MediaStore.MediaColumns.RELATIVE_PATH),
            row.getAsString(MediaStore.MediaColumns.RELATIVE_PATH)
                .contains("${Environment.DIRECTORY_PICTURES}/${ExportNaming.ALBUM_NAME}"),
        )
        assertEquals(resultBitmap.width, row.getAsInteger(MediaStore.MediaColumns.WIDTH))
        assertEquals(resultBitmap.height, row.getAsInteger(MediaStore.MediaColumns.HEIGHT))

        val exported = readExportedFile(saved.uri)
        val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        BitmapFactory.decodeFile(exported.absolutePath, bounds)
        assertEquals("exported width must equal the processed target", resultBitmap.width, bounds.outWidth)
        assertEquals("exported height must equal the processed target", resultBitmap.height, bounds.outHeight)

        val sourceExif = ExifInterface(sourceFile.absolutePath)
        assertTrue("the fixture must actually carry GPS", sourceExif.getLatLong(FloatArray(2)))
        val exportedExif = ExifInterface(exported.absolutePath)
        assertFalse(
            "geolocation must not reach the export",
            exportedExif.getLatLong(FloatArray(2)),
        )
        assertNull(exportedExif.getAttribute(ExifInterface.TAG_GPS_LATITUDE))
        assertNull(exportedExif.getAttribute(ExifInterface.TAG_GPS_LONGITUDE))
        assertNull(exportedExif.getAttribute(ExifInterface.TAG_MAKE))
        assertNull(exportedExif.getAttribute(ExifInterface.TAG_DATETIME_ORIGINAL))
        assertEquals(
            "a neutral edited note is required by §5.3",
            ResultExporter.EDIT_MARKER,
            exportedExif.getAttribute(ExifInterface.TAG_SOFTWARE),
        )

        assertEquals("the picked source file must not change", sourceDigest, sha256(sourceFile))
        assertTrue("no staging file may survive a success", exporter.stagingFiles().isEmpty())
    }

    @Test
    fun forcedWriteErrorLeavesNoPendingRecordAndNoStagingFile() = runBlocking {
        val exporter = exporter(openDestination = { FailingOutputStream() })
        val expectedName = ExportNaming.fileName(fixedMillis, zone, ExportFormat.JPEG)

        val outcome = exporter.export(
            bitmap = resultBitmap,
            settings = ExportSettings(),
            nowMillis = fixedMillis,
            zone = zone,
        )

        assertEquals(ExportOutcome.Failed(ExportFailure.WRITE_FAILED), outcome)
        assertEquals(
            "the pending MediaStore row must be deleted after a write failure",
            0,
            countRowsIncludingPending(expectedName),
        )
        assertTrue(exporter.stagingFiles().isEmpty())
        assertEquals(sourceDigest, sha256(sourceFile))
    }

    @Test
    fun cancellationMidExportLeavesNoPendingRecordAndNoStagingFile() {
        val cancelSignal = Job()
        val exporter = exporter(
            openDestination = {
                // The row already exists at this point; cancel before a single byte lands.
                cancelSignal.cancel()
                ByteArrayOutputStream()
            },
        )
        val expectedName = ExportNaming.fileName(fixedMillis, zone, ExportFormat.PNG)

        val running = CoroutineScope(Dispatchers.Default + cancelSignal).launch {
            exporter.export(
                bitmap = resultBitmap,
                settings = ExportSettings(format = ExportFormat.PNG),
                nowMillis = fixedMillis,
                zone = zone,
            )
        }
        runBlocking { running.join() }

        assertTrue("the export coroutine must end cancelled", running.isCancelled)
        assertEquals(
            "a cancelled export must not leave a pending MediaStore row",
            0,
            countRowsIncludingPending(expectedName),
        )
        assertTrue(
            "a cancelled export must not leave a temporary file",
            exporter.stagingFiles().isEmpty(),
        )
        assertEquals(sourceDigest, sha256(sourceFile))
    }

    @Test
    fun pngExportCarriesTheEditMarkerAndTargetSize() = runBlocking {
        val exporter = exporter()

        val outcome = exporter.export(
            bitmap = resultBitmap,
            settings = ExportSettings(format = ExportFormat.PNG),
            nowMillis = fixedMillis,
            zone = zone,
        )

        val saved = outcome as? ExportOutcome.Saved
            ?: throw AssertionError("export failed: $outcome")
        createdRows += saved.uri
        assertEquals(ExportNaming.fileName(fixedMillis, zone, ExportFormat.PNG), saved.displayName)

        val exported = readExportedFile(saved.uri)
        val text = String(exported.readBytes(), Charsets.ISO_8859_1)
        assertTrue("PNG must carry a tEXt chunk", text.contains("tEXt"))
        assertTrue(
            text.contains(
                ResultExporter.PNG_SOFTWARE_KEYWORD + 0.toChar() + ResultExporter.EDIT_MARKER,
            ),
        )

        val decoded = BitmapFactory.decodeFile(exported.absolutePath)
        try {
            assertEquals(resultBitmap.width, decoded.width)
            assertEquals(resultBitmap.height, decoded.height)
        } finally {
            decoded.recycle()
        }
        assertEquals(sourceDigest, sha256(sourceFile))
    }

    @Test
    fun watermarkIsAppliedToTheFileOnlyAndCanBeSwitchedOff() = runBlocking {
        val exporter = exporter()
        val pixelsBefore = resultBitmap.pixels()

        val marked = exporter.export(
            bitmap = resultBitmap,
            settings = ExportSettings(format = ExportFormat.PNG, watermarkEnabled = true),
            nowMillis = fixedMillis,
            zone = zone,
        ) as ExportOutcome.Saved
        createdRows += marked.uri
        val clean = exporter.export(
            bitmap = resultBitmap,
            settings = ExportSettings(format = ExportFormat.PNG, watermarkEnabled = false),
            nowMillis = fixedMillis + 1_000L,
            zone = zone,
        ) as ExportOutcome.Saved
        createdRows += clean.uri

        val markedBitmap = BitmapFactory.decodeFile(readExportedFile(marked.uri).absolutePath)
        val cleanBitmap = BitmapFactory.decodeFile(readExportedFile(clean.uri).absolutePath)
        try {
            assertEquals(resultBitmap.width, markedBitmap.width)
            assertFalse(
                "the watermark must change the exported pixels",
                markedBitmap.pixels().contentEquals(cleanBitmap.pixels()),
            )
            assertTrue(
                "an unmarked export must reproduce the result bitmap",
                cleanBitmap.pixels().contentEquals(pixelsBefore),
            )
        } finally {
            markedBitmap.recycle()
            cleanBitmap.recycle()
        }
        assertTrue(
            "the in-memory result must never be watermarked",
            resultBitmap.pixels().contentEquals(pixelsBefore),
        )
        assertEquals(sourceDigest, sha256(sourceFile))
    }

    /**
     * API 28 has no pending row, so the exporter asks for a SAF document. A supplied
     * destination is used as is and is deleted again when the write fails.
     */
    @Test
    fun api28RequestsASafDestinationAndDeletesItWhenTheWriteFails() = runBlocking {
        val legacyExporter = exporter(sdkInt = 28)

        val request = legacyExporter.export(
            bitmap = resultBitmap,
            settings = ExportSettings(),
            nowMillis = fixedMillis,
            zone = zone,
        )
        assertEquals(
            ExportOutcome.NeedsDestination(
                ExportNaming.fileName(fixedMillis, zone, ExportFormat.JPEG),
                "image/jpeg",
            ),
            request,
        )
        assertTrue(legacyExporter.stagingFiles().isEmpty())

        val destination = insertPlainRow("StageE2_saf_destination.jpg")
        createdRows += destination
        val failing = exporter(sdkInt = 28, openDestination = { FailingOutputStream() })

        val outcome = failing.export(
            bitmap = resultBitmap,
            settings = ExportSettings(),
            destination = destination,
            nowMillis = fixedMillis,
            zone = zone,
        )

        assertEquals(ExportOutcome.Failed(ExportFailure.WRITE_FAILED), outcome)
        assertEquals(
            "the destination created for this export must be removed",
            0,
            countRowsIncludingPending("StageE2_saf_destination.jpg"),
        )
        assertTrue(failing.stagingFiles().isEmpty())
        assertEquals(sourceDigest, sha256(sourceFile))
    }

    @Test
    fun sweepRemovesStagingFilesLeftByAKilledProcess() = runBlocking {
        val exporter = exporter()
        val cacheRoot = File(context.cacheDir, ResultExporter.CACHE_DIRECTORY_NAME)
        check(cacheRoot.isDirectory || cacheRoot.mkdirs())
        val abandoned = File(cacheRoot, "export_crashed${ResultExporter.TEMP_SUFFIX}")
        abandoned.writeBytes(ByteArray(1024))
        assertFalse(exporter.stagingFiles().isEmpty())

        exporter.sweepAbandonedData()

        assertFalse("the abandoned staging file must be deleted", abandoned.exists())
        assertTrue(exporter.stagingFiles().isEmpty())
    }

    private fun exporter(
        sdkInt: Int = android.os.Build.VERSION.SDK_INT,
        openDestination: ((Uri) -> OutputStream?)? = null,
    ): ResultExporter = if (openDestination == null) {
        ResultExporter(context, sdkInt = sdkInt)
    } else {
        ResultExporter(context, sdkInt = sdkInt, openDestination = openDestination)
    }

    private fun writeSourcePhoto(file: File) {
        val bitmap = Bitmap.createBitmap(320, 240, Bitmap.Config.ARGB_8888)
        Canvas(bitmap).apply {
            drawColor(Color.rgb(40, 90, 140))
            drawCircle(160f, 120f, 70f, Paint().apply { color = Color.rgb(230, 200, 160) })
        }
        FileOutputStream(file).use { output ->
            check(bitmap.compress(Bitmap.CompressFormat.JPEG, 92, output))
        }
        bitmap.recycle()

        ExifInterface(file.absolutePath).apply {
            setAttribute(ExifInterface.TAG_GPS_LATITUDE, "55/1,45/1,0/1")
            setAttribute(ExifInterface.TAG_GPS_LATITUDE_REF, "N")
            setAttribute(ExifInterface.TAG_GPS_LONGITUDE, "37/1,37/1,0/1")
            setAttribute(ExifInterface.TAG_GPS_LONGITUDE_REF, "E")
            setAttribute(ExifInterface.TAG_MAKE, "StageE2Fixture")
            setAttribute(ExifInterface.TAG_DATETIME_ORIGINAL, "2026:07:01 10:20:30")
            saveAttributes()
        }
    }

    private fun readRow(uri: Uri): ContentValues {
        val projection = arrayOf(
            MediaStore.MediaColumns.DISPLAY_NAME,
            MediaStore.MediaColumns.MIME_TYPE,
            MediaStore.MediaColumns.RELATIVE_PATH,
            MediaStore.MediaColumns.WIDTH,
            MediaStore.MediaColumns.HEIGHT,
            MediaStore.MediaColumns.IS_PENDING,
        )
        resolver.query(uri, projection, null, null, null)!!.use { cursor ->
            assertTrue("the saved row must exist", cursor.moveToFirst())
            return ContentValues().apply {
                put(MediaStore.MediaColumns.DISPLAY_NAME, cursor.getString(0))
                put(MediaStore.MediaColumns.MIME_TYPE, cursor.getString(1))
                put(MediaStore.MediaColumns.RELATIVE_PATH, cursor.getString(2))
                put(MediaStore.MediaColumns.WIDTH, cursor.getInt(3))
                put(MediaStore.MediaColumns.HEIGHT, cursor.getInt(4))
                put(MediaStore.MediaColumns.IS_PENDING, cursor.getInt(5))
            }
        }
    }

    /** Pending rows are matched explicitly so an unfinished record cannot hide. */
    private fun countRowsIncludingPending(displayName: String): Int {
        val args = Bundle().apply {
            putString(
                ContentResolver.QUERY_ARG_SQL_SELECTION,
                "${MediaStore.MediaColumns.DISPLAY_NAME} = ?",
            )
            putStringArray(ContentResolver.QUERY_ARG_SQL_SELECTION_ARGS, arrayOf(displayName))
            putInt(MediaStore.QUERY_ARG_MATCH_PENDING, MediaStore.MATCH_INCLUDE)
        }
        resolver.query(
            MediaStore.Images.Media.getContentUri(MediaStore.VOLUME_EXTERNAL_PRIMARY),
            arrayOf(MediaStore.MediaColumns._ID),
            args,
            null,
        )!!.use { cursor ->
            return cursor.count
        }
    }

    private fun insertPlainRow(displayName: String): Uri {
        val values = ContentValues().apply {
            put(MediaStore.MediaColumns.DISPLAY_NAME, displayName)
            put(MediaStore.MediaColumns.MIME_TYPE, "image/jpeg")
            put(
                MediaStore.MediaColumns.RELATIVE_PATH,
                "${Environment.DIRECTORY_PICTURES}/${ExportNaming.ALBUM_NAME}",
            )
        }
        return resolver.insert(
            MediaStore.Images.Media.getContentUri(MediaStore.VOLUME_EXTERNAL_PRIMARY),
            values,
        )!!
    }

    private fun readExportedFile(uri: Uri): File {
        val copy = File(workDirectory, "exported_${uri.lastPathSegment}")
        resolver.openInputStream(uri)!!.use { input ->
            FileOutputStream(copy).use { output -> input.copyTo(output) }
        }
        return copy
    }

    private fun sha256(file: File): String {
        val digest = MessageDigest.getInstance("SHA-256")
        file.inputStream().use { input ->
            val buffer = ByteArray(64 * 1024)
            while (true) {
                val read = input.read(buffer)
                if (read == -1) break
                digest.update(buffer, 0, read)
            }
        }
        return digest.digest().joinToString(separator = "") { byte -> "%02x".format(byte) }
    }

    private fun Bitmap.pixels(): IntArray = IntArray(width * height).also { pixels ->
        getPixels(pixels, 0, width, 0, 0, width, height)
    }

    private class FailingOutputStream : OutputStream() {
        override fun write(value: Int): Unit = throw IOException("forced write failure")

        override fun write(buffer: ByteArray, offset: Int, length: Int): Unit =
            throw IOException("forced write failure")
    }
}
