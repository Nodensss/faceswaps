package com.faceswaplocal.app.data

import android.content.ContentResolver
import android.content.ContentValues
import android.content.Context
import android.graphics.Bitmap
import android.media.ExifInterface
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.Environment
import android.provider.DocumentsContract
import android.provider.MediaStore
import androidx.annotation.RequiresApi
import com.faceswaplocal.app.domain.ExportFormat
import com.faceswaplocal.app.domain.ExportSettings
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.io.IOException
import java.io.InputStream
import java.io.OutputStream
import java.time.ZoneId
import kotlin.coroutines.coroutineContext
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.withContext

enum class ExportFailure {
    CACHE_UNAVAILABLE,
    ENCODE_FAILED,
    DESTINATION_UNAVAILABLE,
    WRITE_FAILED,
}

sealed interface ExportOutcome {
    /** No filesystem path is exposed; the URI is the only handle the UI receives. */
    data class Saved(
        val uri: Uri,
        val displayName: String,
        val album: String?,
        val width: Int,
        val height: Int,
    ) : ExportOutcome

    /**
     * API 28 has no pending MediaStore row and inserting into the images collection there
     * would require the forbidden full-storage permission (§5.1), so the user names the
     * file through the Storage Access Framework instead.
     */
    data class NeedsDestination(
        val suggestedName: String,
        val mimeType: String,
    ) : ExportOutcome

    data class Failed(val reason: ExportFailure) : ExportOutcome
}

/**
 * Saves a finished result without ever touching the picked source file.
 *
 * The bitmap is encoded into an app-private `cacheDir` staging file first, marked as
 * edited, and only then streamed into the destination. A destination is created as a
 * pending MediaStore row on API 29+; on API 28 the caller supplies a
 * `ACTION_CREATE_DOCUMENT` URI. Every failure path — including cancellation — deletes
 * both the staging file and the destination it created, so no half-written record is
 * left behind.
 */
class ResultExporter(
    context: Context,
    private val contentResolver: ContentResolver = context.contentResolver,
    private val cacheRoot: File = File(context.cacheDir, CACHE_DIRECTORY_NAME),
    private val packageName: String = context.packageName,
    private val ioDispatcher: CoroutineDispatcher = Dispatchers.IO,
    /** Forced below 29 by tests to exercise the API 28 SAF path on a modern emulator. */
    private val sdkInt: Int = Build.VERSION.SDK_INT,
    /** Replaced by tests to inject a write failure without corrupting a real volume. */
    private val openDestination: (Uri) -> OutputStream? = { uri ->
        contentResolver.openOutputStream(uri)
    },
) {
    suspend fun export(
        bitmap: Bitmap,
        settings: ExportSettings,
        destination: Uri? = null,
        nowMillis: Long = System.currentTimeMillis(),
        zone: ZoneId = ZoneId.systemDefault(),
    ): ExportOutcome = withContext(ioDispatcher) {
        val displayName = ExportNaming.fileName(nowMillis, zone, settings.format)
        if (destination == null && sdkInt < Build.VERSION_CODES.Q) {
            return@withContext ExportOutcome.NeedsDestination(displayName, settings.format.mimeType)
        }
        if (!ensureCacheDirectory()) {
            return@withContext ExportOutcome.Failed(ExportFailure.CACHE_UNAVAILABLE)
        }

        var staged: File? = null
        var ownedDestination: Uri? = null
        var published = false
        try {
            staged = try {
                File.createTempFile(TEMP_PREFIX, TEMP_SUFFIX, cacheRoot)
            } catch (_: IOException) {
                return@withContext ExportOutcome.Failed(ExportFailure.CACHE_UNAVAILABLE)
            }

            if (!encode(bitmap, settings, staged)) {
                return@withContext ExportOutcome.Failed(ExportFailure.ENCODE_FAILED)
            }
            coroutineContext.ensureActive()
            markAsEdited(staged, settings.format)

            val target = if (destination != null) {
                destination
            } else {
                if (Build.VERSION.SDK_INT < Build.VERSION_CODES.Q) {
                    return@withContext ExportOutcome.Failed(ExportFailure.DESTINATION_UNAVAILABLE)
                }
                insertPendingRow(displayName, settings.format, bitmap.width, bitmap.height)
                    ?.also { ownedDestination = it }
                    ?: return@withContext ExportOutcome.Failed(ExportFailure.DESTINATION_UNAVAILABLE)
            }

            coroutineContext.ensureActive()
            val written = try {
                writeStagedFile(staged, target)
            } catch (cancelled: CancellationException) {
                throw cancelled
            } catch (_: IOException) {
                false
            } catch (_: SecurityException) {
                false
            }
            if (!written) {
                return@withContext ExportOutcome.Failed(ExportFailure.WRITE_FAILED)
            }

            if (ownedDestination != null && Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                publishPendingRow(target)
            }
            published = true
            ExportOutcome.Saved(
                uri = target,
                displayName = displayName,
                album = if (destination == null) ExportNaming.ALBUM_NAME else null,
                width = bitmap.width,
                height = bitmap.height,
            )
        } finally {
            staged?.let(::deleteStagingFile)
            if (!published) {
                (ownedDestination ?: destination)?.let(::deleteIncompleteDestination)
            }
        }
    }

    /**
     * `ExifInterface.saveAttributes` rewrites through a `<name>.tmp` sibling, so a run
     * that dies inside it can leave one behind. Only this run's own files are touched,
     * which keeps two overlapping exports independent.
     */
    private fun deleteStagingFile(staged: File) {
        staged.delete()
        File("${staged.absolutePath}.tmp").delete()
    }

    /**
     * Removes data an earlier run could not clean up itself: staging files from a killed
     * process and MediaStore rows this app left pending (§5.1).
     */
    suspend fun sweepAbandonedData(): Int = withContext(ioDispatcher) {
        var removed = deleteStagingFiles()
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            removed += deleteOwnPendingRows()
        }
        removed
    }

    /**
     * Removes leftover staging files only. Used after every processing run, where an
     * in-flight export may legitimately own a pending MediaStore row.
     */
    suspend fun sweepStagingFiles(): Int = withContext(ioDispatcher) { deleteStagingFiles() }

    /**
     * The whole directory belongs to the exporter, so everything in it is temporary —
     * including the `.tmp` file `ExifInterface.saveAttributes` may leave behind if the
     * process dies mid-write. Nothing else may write here.
     */
    internal fun stagingFiles(): List<File> = cacheRoot
        .takeIf(File::isDirectory)
        ?.listFiles(File::isFile)
        ?.toList()
        .orEmpty()

    private fun deleteStagingFiles(): Int = stagingFiles().count(File::delete)

    @RequiresApi(Build.VERSION_CODES.Q)
    private fun deleteOwnPendingRows(): Int {
        val cursor = try {
            contentResolver.query(
                pendingQueryCollection(),
                arrayOf(MediaStore.MediaColumns._ID),
                pendingSelectionArgs(),
                null,
            )
        } catch (_: SecurityException) {
            null
        } catch (_: IllegalArgumentException) {
            null
        } ?: return 0

        var removed = 0
        cursor.use {
            while (it.moveToNext()) {
                val id = it.getLong(0)
                val uri = android.content.ContentUris.withAppendedId(imagesCollection(), id)
                if (deleteIncompleteDestination(uri)) removed++
            }
        }
        return removed
    }

    @RequiresApi(Build.VERSION_CODES.Q)
    @Suppress("DEPRECATION")
    private fun pendingQueryCollection(): Uri = if (Build.VERSION.SDK_INT < Build.VERSION_CODES.R) {
        MediaStore.setIncludePending(imagesCollection())
    } else {
        imagesCollection()
    }

    @RequiresApi(Build.VERSION_CODES.Q)
    private fun pendingSelectionArgs(): Bundle = Bundle().apply {
        putString(
            ContentResolver.QUERY_ARG_SQL_SELECTION,
            "${MediaStore.MediaColumns.IS_PENDING} = 1 AND " +
                "${MediaStore.MediaColumns.OWNER_PACKAGE_NAME} = ?",
        )
        putStringArray(ContentResolver.QUERY_ARG_SQL_SELECTION_ARGS, arrayOf(packageName))
    }

    @RequiresApi(Build.VERSION_CODES.Q)
    private fun imagesCollection(): Uri =
        MediaStore.Images.Media.getContentUri(MediaStore.VOLUME_EXTERNAL_PRIMARY)

    @RequiresApi(Build.VERSION_CODES.Q)
    private fun insertPendingRow(
        displayName: String,
        format: ExportFormat,
        width: Int,
        height: Int,
    ): Uri? {
        val values = ContentValues().apply {
            put(MediaStore.MediaColumns.DISPLAY_NAME, displayName)
            put(MediaStore.MediaColumns.MIME_TYPE, format.mimeType)
            put(
                MediaStore.MediaColumns.RELATIVE_PATH,
                "${Environment.DIRECTORY_PICTURES}/${ExportNaming.ALBUM_NAME}",
            )
            put(MediaStore.MediaColumns.WIDTH, width)
            put(MediaStore.MediaColumns.HEIGHT, height)
            put(MediaStore.MediaColumns.IS_PENDING, 1)
        }
        return try {
            contentResolver.insert(imagesCollection(), values)
        } catch (_: IllegalArgumentException) {
            null
        } catch (_: IllegalStateException) {
            null
        } catch (_: SecurityException) {
            null
        }
    }

    @RequiresApi(Build.VERSION_CODES.Q)
    private fun publishPendingRow(uri: Uri) {
        val values = ContentValues().apply { put(MediaStore.MediaColumns.IS_PENDING, 0) }
        contentResolver.update(uri, values, null, null)
    }

    private fun encode(bitmap: Bitmap, settings: ExportSettings, destination: File): Boolean {
        val watermarked = if (settings.watermarkEnabled) ResultWatermark.render(bitmap) else null
        return try {
            FileOutputStream(destination).use { output ->
                val encoded = (watermarked ?: bitmap).compress(
                    settings.format.toCompressFormat(),
                    settings.effectiveQuality,
                    output,
                )
                output.fd.sync()
                encoded
            }
        } catch (_: IOException) {
            false
        } finally {
            watermarked?.let { if (!it.isRecycled) it.recycle() }
        }
    }

    /**
     * Adds the neutral edited note (§5.3) and defensively clears location tags. Encoding
     * from a bitmap already drops every source EXIF field — `StageE2ExportInstrumentedTest`
     * proves that on a GPS-tagged input rather than assuming it.
     */
    private fun markAsEdited(file: File, format: ExportFormat) {
        when (format) {
            ExportFormat.JPEG -> try {
                ExifInterface(file.absolutePath).apply {
                    setAttribute(ExifInterface.TAG_SOFTWARE, EDIT_MARKER)
                    setAttribute(ExifInterface.TAG_IMAGE_DESCRIPTION, EDIT_NOTE)
                    LOCATION_TAGS.forEach { tag -> setAttribute(tag, null) }
                    saveAttributes()
                }
            } catch (_: IOException) {
                // An unmarked export is acceptable; a failed export because of a note is not.
            }

            ExportFormat.PNG -> PngEditMarker.writeTextChunk(file, PNG_SOFTWARE_KEYWORD, EDIT_MARKER)
        }
    }

    private suspend fun writeStagedFile(staged: File, destination: Uri): Boolean {
        val output = openDestination(destination) ?: return false
        output.use { sink ->
            FileInputStream(staged).use { source -> copy(source, sink) }
            if (sink is FileOutputStream) sink.fd.sync()
        }
        return true
    }

    private suspend fun copy(source: InputStream, sink: OutputStream) {
        val buffer = ByteArray(COPY_BUFFER_BYTES)
        while (true) {
            coroutineContext.ensureActive()
            val read = source.read(buffer)
            if (read == -1) break
            if (read == 0) continue
            sink.write(buffer, 0, read)
        }
        sink.flush()
    }

    /**
     * MediaStore rows are removed through the resolver; a SAF document rejects that call
     * and needs `DocumentsContract`, so both are attempted before giving up.
     */
    private fun deleteIncompleteDestination(uri: Uri): Boolean {
        val deletedRow = try {
            contentResolver.delete(uri, null, null) > 0
        } catch (_: SecurityException) {
            false
        } catch (_: UnsupportedOperationException) {
            false
        } catch (_: IllegalArgumentException) {
            false
        }
        return deletedRow || deleteDocument(uri)
    }

    private fun deleteDocument(uri: Uri): Boolean = try {
        DocumentsContract.deleteDocument(contentResolver, uri)
    } catch (_: SecurityException) {
        false
    } catch (_: IllegalArgumentException) {
        false
    } catch (_: java.io.FileNotFoundException) {
        false
    }

    private fun ensureCacheDirectory(): Boolean = cacheRoot.isDirectory || cacheRoot.mkdirs()

    private fun ExportFormat.toCompressFormat(): Bitmap.CompressFormat = when (this) {
        ExportFormat.JPEG -> Bitmap.CompressFormat.JPEG
        ExportFormat.PNG -> Bitmap.CompressFormat.PNG
    }

    companion object {
        internal const val CACHE_DIRECTORY_NAME = "export"
        internal const val TEMP_PREFIX = "export_"
        internal const val TEMP_SUFFIX = ".part"
        internal const val EDIT_MARKER = "FaceSwapLocal"
        internal const val EDIT_NOTE = "Edited image"
        internal const val PNG_SOFTWARE_KEYWORD = "Software"
        private const val COPY_BUFFER_BYTES = 256 * 1024

        private val LOCATION_TAGS = listOf(
            ExifInterface.TAG_GPS_LATITUDE,
            ExifInterface.TAG_GPS_LATITUDE_REF,
            ExifInterface.TAG_GPS_LONGITUDE,
            ExifInterface.TAG_GPS_LONGITUDE_REF,
            ExifInterface.TAG_GPS_ALTITUDE,
            ExifInterface.TAG_GPS_ALTITUDE_REF,
            ExifInterface.TAG_GPS_TIMESTAMP,
            ExifInterface.TAG_GPS_DATESTAMP,
            ExifInterface.TAG_GPS_PROCESSING_METHOD,
        )
    }
}
