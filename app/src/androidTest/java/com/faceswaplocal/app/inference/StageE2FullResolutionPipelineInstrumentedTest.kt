package com.faceswaplocal.app.inference

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Rect
import android.net.Uri
import android.os.Debug
import android.util.Log
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.faceswaplocal.app.data.BitmapLoader
import com.faceswaplocal.app.data.ExportOutcome
import com.faceswaplocal.app.data.ImageMemoryBudget
import com.faceswaplocal.app.data.ResultExporter
import com.faceswaplocal.app.domain.ExportFormat
import com.faceswaplocal.app.domain.ExportSettings
import com.faceswaplocal.app.domain.FaceId
import java.io.File
import java.io.FileOutputStream
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.concurrent.thread
import kotlinx.coroutines.runBlocking
import org.json.JSONObject
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

/**
 * Stage E2 checkpoint 2: a 12 MP target is composited and exported at its own size, the
 * paste-ROI invariant still holds at that size, and the real peak memory is recorded.
 *
 * The comparison is streamed row by row so the test's own buffers do not distort the
 * memory figure it is measuring.
 */
@RunWith(AndroidJUnit4::class)
class StageE2FullResolutionPipelineInstrumentedTest {
    private val context = ApplicationProvider.getApplicationContext<Context>()
    private val assets = InstrumentationRegistry.getInstrumentation().context.assets
    private val resolver = context.contentResolver
    private val createdRows = mutableListOf<Uri>()

    @After
    fun cleanUp() {
        createdRows.forEach { uri -> runCatching { resolver.delete(uri, null, null) } }
        createdRows.clear()
        File(context.cacheDir, WORK_DIRECTORY).deleteRecursively()
    }

    @Test
    fun twelveMegapixelTargetKeepsItsSizeThroughCompositingAndExport() {
        val store = ModelStore(context)
        val statuses = runBlocking { store.refreshStatuses() }
        REQUIRED_MODELS.forEach { id ->
            assertTrue("${id.stableId} must be imported", statuses[id] is ModelStatus.Ready)
        }

        val workDirectory = File(context.cacheDir, WORK_DIRECTORY).apply {
            deleteRecursively()
            check(mkdirs()) { "Could not create $absolutePath" }
        }
        val targetFile = File(workDirectory, "target_${WIDTH}x$HEIGHT.jpg")
        writeUpscaledFixture(targetFile)

        val loader = BitmapLoader(resolver, ImageMemoryBudget.forContext(context))
        val budget = ImageMemoryBudget.forContext(context).maxTargetPixels()
        val decodedTarget = runBlocking { loader.loadTarget(Uri.fromFile(targetFile)) }
        val target = decodedTarget.bitmap
        Log.i(
            BENCHMARK_TAG,
            "budgetPixels=$budget decoded=${target.width}x${target.height} " +
                "source=${decodedTarget.sourceWidth}x${decodedTarget.sourceHeight} " +
                "fullResolution=${decodedTarget.isFullResolution}",
        )
        // The contract under test is "export equals what was composited, at whatever size
        // the budget allowed", so a device that cannot afford 12 MP is a recorded result
        // rather than a red test. Only the budget's own floor is non-negotiable.
        val fullResolution = decodedTarget.isFullResolution
        assertTrue(
            "the budget must never fall below its documented floor; budget=$budget",
            budget >= com.faceswaplocal.app.data.ImageMemoryBudget.MIN_TARGET_PIXELS,
        )
        assertEquals(
            "a decode within budget must not be downscaled",
            budget >= WIDTH * HEIGHT,
            fullResolution,
        )

        val sourceBitmap = runBlocking {
            loader.loadSource(Uri.fromFile(sourceFixture(workDirectory))).bitmap
        }
        val raw = OnnxRawFaceSwapPipeline(store)
        val photo = OnnxPhotoFaceSwapPipeline(store, rawPipeline = raw)
        val enhancer = OnnxFaceEnhancerPipeline(store)
        val parser = OnnxFaceParserPipeline(store)
        val coordinator = OnnxMultiPhotoFaceSwapPipeline(raw, photo, enhancer, parser)
        val sampler = MemorySampler().apply { start() }

        var result: MultiPhotoFaceSwapResult? = null
        try {
            val detected = runBlocking {
                raw.detectFaces(target, RequestedInferenceBackend.CPU_ONLY).first
            }
            assertTrue("the upscaled fixture must still yield faces", detected.isNotEmpty())
            val targetFace = detected.maxByOrNull(DetectedFace5::score)!!
            val sourceFace = runBlocking {
                raw.detectFaces(sourceBitmap, RequestedInferenceBackend.CPU_ONLY)
                    .first
                    .maxByOrNull(DetectedFace5::score)
            }!!

            val started = android.os.SystemClock.elapsedRealtime()
            result = runBlocking {
                coordinator.process(
                    target = target,
                    sources = listOf(
                        MultiPhotoSource(FaceId("source-0"), sourceBitmap, sourceFace.box),
                    ),
                    targetsInStableOrder = listOf(MultiPhotoTarget(FaceId("target-0"), targetFace.box)),
                    assignments = listOf(MultiPhotoAssignment(FaceId("target-0"), FaceId("source-0"))),
                    backend = RequestedInferenceBackend.CPU_ONLY,
                    restorationStrength = RESTORATION_STRENGTH,
                )
            }!!
            val coordinatorMs = android.os.SystemClock.elapsedRealtime() - started

            assertEquals("compositing must keep the target width", WIDTH, result.finalBitmap.width)
            assertEquals("compositing must keep the target height", HEIGHT, result.finalBitmap.height)

            val changedOutsideRoi = countChangedPixelsOutsideRoi(
                original = target,
                produced = result.finalBitmap,
                allowed = result.swapRois.map { it.bounds } + result.enhanceRois.map { it.bounds },
            )
            assertEquals("no pixel outside the paste ROI union may change", 0L, changedOutsideRoi)

            val exporter = ResultExporter(context)
            // Default settings on purpose: the watermark is on by default and copies the
            // whole result bitmap, so measuring without it would understate the real peak.
            val exportSettings = ExportSettings()
            check(exportSettings.format == ExportFormat.JPEG && exportSettings.watermarkEnabled)
            val outcome = runBlocking {
                exporter.export(bitmap = result.finalBitmap, settings = exportSettings)
            }
            val saved = outcome as? ExportOutcome.Saved
                ?: throw AssertionError("12 MP export failed: $outcome")
            createdRows += saved.uri
            assertEquals(WIDTH, saved.width)
            assertEquals(HEIGHT, saved.height)

            val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
            resolver.openInputStream(saved.uri)!!.use { input ->
                BitmapFactory.decodeStream(input, null, bounds)
            }
            assertEquals("the exported file must carry the target width", WIDTH, bounds.outWidth)
            assertEquals("the exported file must carry the target height", HEIGHT, bounds.outHeight)
            assertTrue(exporter.stagingFiles().isEmpty())

            sampler.stop()
            val report = JSONObject().apply {
                put("targetWidth", WIDTH)
                put("targetHeight", HEIGHT)
                put("budgetPixels", budget)
                put("decodedFullResolution", decodedTarget.isFullResolution)
                put("restorationStrength", RESTORATION_STRENGTH.toDouble())
                put("watermarkEnabled", exportSettings.watermarkEnabled)
                put("jpegQuality", exportSettings.jpegQuality)
                put("coordinatorMs", coordinatorMs)
                put("peakJavaHeapBytes", sampler.peakJavaHeapBytes)
                put("peakNativeHeapBytes", sampler.peakNativeHeapBytes)
                put("peakTotalPssBytes", sampler.peakTotalPssBytes)
                put("maxHeapBytes", Runtime.getRuntime().maxMemory())
                put("samples", sampler.sampleCount)
                put("changedPixelsOutsideRoi", changedOutsideRoi)
                put("exportedWidth", bounds.outWidth)
                put("exportedHeight", bounds.outHeight)
            }
            File(context.filesDir, OUTPUT_FILE).writeText(report.toString(2))
            Log.i(BENCHMARK_TAG, report.toString())
        } finally {
            sampler.stop()
            result?.finalBitmap?.recycleSafely()
            sourceBitmap.recycleSafely()
            target.recycleSafely()
        }
    }

    /**
     * Streams both images a row at a time. Only pixels outside every allowed rectangle
     * are compared, so an intersecting pair of ROIs cannot hide a stray write.
     */
    private fun countChangedPixelsOutsideRoi(
        original: Bitmap,
        produced: Bitmap,
        allowed: List<CompositeRoi>,
    ): Long {
        val width = original.width
        val originalRow = IntArray(width)
        val producedRow = IntArray(width)
        var changed = 0L

        for (y in 0 until original.height) {
            val spans = allowed.filter { roi -> y >= roi.top && y < roi.bottom }
            if (spans.size == 1 && spans[0].left <= 0 && spans[0].right >= width) continue

            original.getPixels(originalRow, 0, width, 0, y, width, 1)
            produced.getPixels(producedRow, 0, width, 0, y, width, 1)
            for (x in 0 until width) {
                if (spans.any { roi -> x >= roi.left && x < roi.right }) continue
                if (originalRow[x] != producedRow[x]) changed++
            }
        }
        return changed
    }

    private fun sourceFixture(workDirectory: File): File {
        val file = File(workDirectory, "source.png")
        assets.open(SOURCE_ASSET).use { input ->
            FileOutputStream(file).use { output -> input.copyTo(output) }
        }
        return file
    }

    private fun writeUpscaledFixture(file: File) {
        val fixture = assets.open(TARGET_ASSET).use { input -> BitmapFactory.decodeStream(input) }
        val canvasBitmap = Bitmap.createBitmap(WIDTH, HEIGHT, Bitmap.Config.ARGB_8888)
        try {
            val scale = minOf(
                WIDTH.toFloat() / fixture.width,
                HEIGHT.toFloat() / fixture.height,
            )
            val scaledWidth = (fixture.width * scale).toInt()
            val scaledHeight = (fixture.height * scale).toInt()
            val left = (WIDTH - scaledWidth) / 2
            val top = (HEIGHT - scaledHeight) / 2
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

    private fun Bitmap.recycleSafely() {
        if (!isRecycled) recycle()
    }

    /** Polls the three figures §12 asks for; PSS is the expensive one, so it is rarer. */
    private class MemorySampler {
        private val running = AtomicBoolean(false)
        private var worker: Thread? = null

        @Volatile
        var peakJavaHeapBytes: Long = 0
            private set

        @Volatile
        var peakNativeHeapBytes: Long = 0
            private set

        @Volatile
        var peakTotalPssBytes: Long = 0
            private set

        @Volatile
        var sampleCount: Int = 0
            private set

        fun start() {
            if (!running.compareAndSet(false, true)) return
            worker = thread(name = "stage-e2-memory-sampler", isDaemon = true) {
                var iteration = 0
                while (running.get()) {
                    val runtime = Runtime.getRuntime()
                    peakJavaHeapBytes = maxOf(
                        peakJavaHeapBytes,
                        runtime.totalMemory() - runtime.freeMemory(),
                    )
                    peakNativeHeapBytes =
                        maxOf(peakNativeHeapBytes, Debug.getNativeHeapAllocatedSize())
                    if (iteration % PSS_EVERY_N_SAMPLES == 0) {
                        val info = Debug.MemoryInfo()
                        Debug.getMemoryInfo(info)
                        peakTotalPssBytes = maxOf(peakTotalPssBytes, info.totalPss * 1024L)
                    }
                    sampleCount++
                    iteration++
                    Thread.sleep(SAMPLE_INTERVAL_MS)
                }
            }
        }

        fun stop() {
            if (!running.compareAndSet(true, false)) return
            worker?.join(2_000)
            worker = null
        }

        private companion object {
            const val SAMPLE_INTERVAL_MS = 250L
            const val PSS_EVERY_N_SAMPLES = 8
        }
    }

    private companion object {
        const val WIDTH = 4_000
        const val HEIGHT = 3_000
        const val RESTORATION_STRENGTH = 0.8f
        const val TARGET_ASSET = "inputs/stage_d_group_target.png"
        const val SOURCE_ASSET = "inputs/pair_01_source.png"
        const val WORK_DIRECTORY = "stage-e2-fullres-pipeline"
        const val OUTPUT_FILE = "stage_e2_full_resolution.json"
        const val BENCHMARK_TAG = "StageE2FullRes"
        val REQUIRED_MODELS = setOf(
            ModelId.YOLOFACE_8N,
            ModelId.ARCFACE_W600K_R50,
            ModelId.INSWAPPER_128_FP16,
            ModelId.GFPGAN_1_4,
            ModelId.BISENET_RESNET_34,
        )
    }
}
