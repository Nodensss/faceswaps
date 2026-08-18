package com.faceswaplocal.app.data

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Rect
import android.net.Uri
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import java.io.File
import java.io.FileOutputStream
import kotlinx.coroutines.runBlocking
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith

/**
 * Stage E2 checkpoint 2: the decode limit follows the memory budget instead of the old
 * 2 560 px constant, and a 12 MP target survives decoding at its own size.
 */
@RunWith(AndroidJUnit4::class)
class StageE2FullResolutionInstrumentedTest {
    private val context = ApplicationProvider.getApplicationContext<Context>()
    private val assets = InstrumentationRegistry.getInstrumentation().context.assets

    private lateinit var workDirectory: File
    private lateinit var targetFile: File

    @Before
    fun createTwelveMegapixelFixture() {
        workDirectory = File(context.cacheDir, "stage-e2-fullres-test").apply {
            deleteRecursively()
            check(mkdirs()) { "Could not create $absolutePath" }
        }
        targetFile = File(workDirectory, "target_4000x3000.jpg")
        writeUpscaledFixture(targetFile, FIXTURE_WIDTH, FIXTURE_HEIGHT)
    }

    @After
    fun cleanUp() {
        workDirectory.deleteRecursively()
    }

    @Test
    fun aTwelveMegapixelTargetIsDecodedAtItsOwnSize() = runBlocking {
        val loader = BitmapLoader(context.contentResolver, budgetOf(maxPixels = 16_000_000))

        val decoded = loader.loadTarget(Uri.fromFile(targetFile))

        try {
            assertEquals(FIXTURE_WIDTH, decoded.bitmap.width)
            assertEquals(FIXTURE_HEIGHT, decoded.bitmap.height)
            assertEquals(FIXTURE_WIDTH, decoded.sourceWidth)
            assertEquals(FIXTURE_HEIGHT, decoded.sourceHeight)
            assertTrue(decoded.isFullResolution)
        } finally {
            decoded.bitmap.recycle()
        }
    }

    @Test
    fun aTightBudgetDownscalesAndStillReportsTheOriginalSize() = runBlocking {
        val maxPixels = 2_000_000
        val loader = BitmapLoader(context.contentResolver, budgetOf(maxPixels))

        val decoded = loader.loadTarget(Uri.fromFile(targetFile))

        try {
            val pixels = decoded.bitmap.width.toLong() * decoded.bitmap.height
            assertTrue("$pixels must fit the $maxPixels budget", pixels <= maxPixels)
            assertFalse(decoded.isFullResolution)
            assertEquals(FIXTURE_WIDTH, decoded.sourceWidth)
            assertEquals(FIXTURE_HEIGHT, decoded.sourceHeight)
            assertEquals(
                "the aspect ratio must survive downscaling",
                FIXTURE_WIDTH.toDouble() / FIXTURE_HEIGHT,
                decoded.bitmap.width.toDouble() / decoded.bitmap.height,
                0.01,
            )
        } finally {
            decoded.bitmap.recycle()
        }
    }

    @Test
    fun sourcesStayCappedAtTheStageBLongSide() = runBlocking {
        val loader = BitmapLoader(context.contentResolver, budgetOf(maxPixels = 64_000_000))

        val decoded = loader.loadSource(Uri.fromFile(targetFile))

        try {
            assertEquals(
                "a source only feeds an aligned crop, so the parity-era cap stays",
                ImageMemoryBudget.SOURCE_MAX_DIMENSION,
                maxOf(decoded.bitmap.width, decoded.bitmap.height),
            )
            assertEquals(FIXTURE_WIDTH, decoded.sourceWidth)
        } finally {
            decoded.bitmap.recycle()
        }
    }

    @Test
    fun theBudgetOnThisDeviceIsReportedForTheBenchmark() {
        val budget = ImageMemoryBudget.forContext(context)
        val maxPixels = budget.maxTargetPixels()
        val runtime = Runtime.getRuntime()

        android.util.Log.i(
            BENCHMARK_TAG,
            "maxTargetPixels=$maxPixels " +
                "maxHeap=${runtime.maxMemory()} " +
                "usedHeap=${runtime.totalMemory() - runtime.freeMemory()} " +
                "fitsTwelveMegapixels=${maxPixels >= FIXTURE_WIDTH * FIXTURE_HEIGHT}",
        )
        assertTrue("the budget must never collapse below the floor", maxPixels >= ImageMemoryBudget.MIN_TARGET_PIXELS)
    }

    private fun budgetOf(maxPixels: Int) = ImageMemoryBudget(
        runtime = object : RuntimeMemory {
            // Chosen so the heap term never binds; the system term is pinned below.
            override fun maxMemory(): Long = Long.MAX_VALUE / 4
            override fun totalMemory(): Long = 0
            override fun freeMemory(): Long = 0
        },
        systemMemory = {
            ImageMemoryBudget.SystemMemory(
                // The reserve is subtracted before the safety fraction, so a device that
                // can truly afford `maxPixels` of frame buffers must also carry the
                // sessions on top. Without this the helper under-provisions by ~599 MiB.
                availableBytes = (
                    maxPixels.toLong() *
                        (ImageMemoryBudget.JAVA_BYTES_PER_PIXEL + ImageMemoryBudget.NATIVE_BYTES_PER_PIXEL) /
                        ImageMemoryBudget.SYSTEM_SAFETY_FRACTION
                    ).toLong() + PipelinePass.peak().sessionReserveBytes,
                lowMemory = false,
            )
        },
    )

    /** Places the committed group fixture on a 12 MP canvas without distorting faces. */
    private fun writeUpscaledFixture(file: File, width: Int, height: Int) {
        val fixture = assets.open(FIXTURE_ASSET).use { input -> BitmapFactory.decodeStream(input) }
        val canvasBitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        try {
            val scale = minOf(
                width.toFloat() / fixture.width,
                height.toFloat() / fixture.height,
            )
            val scaledWidth = (fixture.width * scale).toInt()
            val scaledHeight = (fixture.height * scale).toInt()
            val left = (width - scaledWidth) / 2
            val top = (height - scaledHeight) / 2
            Canvas(canvasBitmap).apply {
                drawColor(Color.rgb(24, 28, 36))
                drawBitmap(
                    fixture,
                    null,
                    Rect(left, top, left + scaledWidth, top + scaledHeight),
                    Paint(Paint.FILTER_BITMAP_FLAG),
                )
            }
            FileOutputStream(file).use { output ->
                check(canvasBitmap.compress(Bitmap.CompressFormat.JPEG, 95, output))
            }
        } finally {
            fixture.recycle()
            canvasBitmap.recycle()
        }
    }

    companion object {
        const val FIXTURE_WIDTH = 4_000
        const val FIXTURE_HEIGHT = 3_000
        const val FIXTURE_ASSET = "inputs/stage_d_group_target.png"
        const val BENCHMARK_TAG = "StageE2Budget"
    }
}
