package com.faceswaplocal.app.data

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.media.ExifInterface
import android.net.Uri
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import java.io.File
import java.io.FileOutputStream
import kotlinx.coroutines.runBlocking
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith

/**
 * Every fixture from stages A-E1 was an upright PNG, so nothing has ever exercised the
 * decode path with an EXIF orientation tag. Real camera files carry one: each 12 MP
 * JPEG in the sample set used for the live run is stored 4032x3024 with orientation 6
 * (rotate 90 CW), i.e. it must be shown as 3024x4032 portrait.
 *
 * This pins down what [BitmapLoader] actually does with such a file, because the answer
 * decides two user-visible things: whether faces reach the detector upright, and whether
 * [DecodedImage.isFullResolution] can tell the truth about a portrait photo.
 */
@RunWith(AndroidJUnit4::class)
class RealPhotoOrientationInstrumentedTest {
    private val context = ApplicationProvider.getApplicationContext<Context>()

    private lateinit var workDirectory: File
    private lateinit var rotatedFile: File

    @Before
    fun writeRotatedCameraFixture() {
        workDirectory = File(context.cacheDir, "real-photo-orientation").apply {
            deleteRecursively()
            check(mkdirs()) { "Could not create $absolutePath" }
        }
        rotatedFile = File(workDirectory, "camera_4032x3024_orientation6.jpg")
        writeMarkedJpeg(rotatedFile, STORED_WIDTH, STORED_HEIGHT)
        ExifInterface(rotatedFile.absolutePath).apply {
            setAttribute(ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_ROTATE_90.toString())
            saveAttributes()
        }
    }

    @After
    fun cleanUp() {
        workDirectory.deleteRecursively()
    }

    @Test
    fun aRotatedCameraTargetIsDecodedUprightAndReportsThatSize() = runBlocking {
        val loader = BitmapLoader(context.contentResolver, generousBudget())

        val decoded = loader.loadTarget(Uri.fromFile(rotatedFile))

        try {
            val markerTopLeft = decoded.bitmap.getPixel(PROBE_INSET, PROBE_INSET)
            val markerTopRight = decoded.bitmap.getPixel(decoded.bitmap.width - PROBE_INSET, PROBE_INSET)
            android.util.Log.i(
                ORIENTATION_TAG,
                "stored=${STORED_WIDTH}x$STORED_HEIGHT orientation=6 " +
                    "bitmap=${decoded.bitmap.width}x${decoded.bitmap.height} " +
                    "reportedSource=${decoded.sourceWidth}x${decoded.sourceHeight} " +
                    "isFullResolution=${decoded.isFullResolution} " +
                    "markerTopLeft=${Integer.toHexString(markerTopLeft)} " +
                    "markerTopRight=${Integer.toHexString(markerTopRight)}",
            )

            // The decoder is expected to honour the tag, so the upright image is portrait.
            assertEquals("decoded width", STORED_HEIGHT, decoded.bitmap.width)
            assertEquals("decoded height", STORED_WIDTH, decoded.bitmap.height)

            // And the reported original size must describe the same upright image, or the
            // export card compares a portrait bitmap against a landscape "original".
            assertEquals("reported source width", STORED_HEIGHT, decoded.sourceWidth)
            assertEquals("reported source height", STORED_WIDTH, decoded.sourceHeight)
            assertTrue(
                "a full-size decode of a rotated photo must not look downscaled",
                decoded.isFullResolution,
            )
        } finally {
            decoded.bitmap.recycle()
        }
    }

    private fun generousBudget() = ImageMemoryBudget(
        runtime = object : RuntimeMemory {
            override fun maxMemory(): Long = Long.MAX_VALUE / 4
            override fun totalMemory(): Long = 0
            override fun freeMemory(): Long = 0
        },
        systemMemory = {
            ImageMemoryBudget.SystemMemory(
                availableBytes = (
                    BUDGET_PIXELS.toLong() *
                        (ImageMemoryBudget.JAVA_BYTES_PER_PIXEL + ImageMemoryBudget.NATIVE_BYTES_PER_PIXEL) /
                        ImageMemoryBudget.SYSTEM_SAFETY_FRACTION
                    ).toLong(),
                lowMemory = false,
            )
        },
    )

    /** A red patch in the stored top-left corner; honouring orientation 6 moves it to the top-right. */
    private fun writeMarkedJpeg(file: File, width: Int, height: Int) {
        val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        try {
            Canvas(bitmap).apply {
                drawColor(Color.rgb(20, 20, 20))
                drawRect(
                    0f,
                    0f,
                    width * MARKER_FRACTION,
                    height * MARKER_FRACTION,
                    Paint().apply { color = Color.RED },
                )
            }
            FileOutputStream(file).use { output ->
                check(bitmap.compress(Bitmap.CompressFormat.JPEG, 95, output))
            }
        } finally {
            bitmap.recycle()
        }
    }

    companion object {
        const val STORED_WIDTH = 4_032
        const val STORED_HEIGHT = 3_024
        const val BUDGET_PIXELS = 16_000_000
        const val MARKER_FRACTION = 0.2f
        const val PROBE_INSET = 40
        const val ORIENTATION_TAG = "RealPhotoOrientation"
    }
}
