package com.faceswaplocal.app.inference

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.faceswaplocal.app.domain.FaceId
import java.io.File
import java.io.FileOutputStream
import kotlin.math.abs
import kotlinx.coroutines.runBlocking
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class StageESwapParserMaskInstrumentedTest {
    private val context = ApplicationProvider.getApplicationContext<Context>()
    private val assets = InstrumentationRegistry.getInstrumentation().context.assets

    @Test
    fun pair02AndPair03UseParserMaskWithoutChangingColorMatch() = runBlocking {
        val listener = RecordingSessionLifecycle()
        val store = ModelStore(context)
        requireModels(store, RESTORATION_REQUIRED_MODELS - ModelId.GFPGAN_1_4)
        val rawPipeline = OnnxRawFaceSwapPipeline(store, sessionLifecycle = listener)
        val photoPipeline = OnnxPhotoFaceSwapPipeline(store, rawPipeline = rawPipeline)
        val parserPipeline = OnnxFaceParserPipeline(store, sessionLifecycle = listener)
        val outputDirectory = visualOutputDirectory()
        val metrics = JSONArray()

        parserPipeline.withSession(RequestedInferenceBackend.CPU_ONLY) { parserSession ->
            for (pair in listOf("pair_02", "pair_03")) {
                val source = bitmap("inputs/${pair}_source.png")
                val target = bitmap("inputs/${pair}_target.png")
                val committedBox = bitmap("android/api35-x86_64/${pair}_inswapper_final.png")
                var raw: RawFaceSwapResult? = null
                var production: PhotoFaceSwapResult? = null
                try {
                    raw = rawPipeline.process(
                        RawFaceSwapRequest(
                            source = source,
                            target = target,
                            swapper = SwapperModel.INSWAPPER_128_FP16,
                            backend = RequestedInferenceBackend.CPU_ONLY,
                        ),
                    )
                    val targetPixels = target.pixels()
                    val targetCropPixels = raw.alignedTarget.pixels()
                    val swappedCropPixels = raw.rawOutputBitmap.pixels()
                    val parsed = parserSession.createRegionMask(
                        cropPixels = swappedCropPixels,
                        cropWidth = raw.swapper.cropSize,
                        cropHeight = raw.swapper.cropSize,
                    )
                    val boxComposite = FaceCompositor.composite(
                        targetPixels = targetPixels,
                        targetWidth = target.width,
                        targetHeight = target.height,
                        targetCropPixels = targetCropPixels,
                        swappedCropPixels = swappedCropPixels,
                        cropWidth = raw.swapper.cropSize,
                        cropHeight = raw.swapper.cropSize,
                        targetToCrop = raw.targetToSwapperCrop,
                    )
                    val parserComposite = FaceCompositor.composite(
                        targetPixels = targetPixels,
                        targetWidth = target.width,
                        targetHeight = target.height,
                        targetCropPixels = targetCropPixels,
                        swappedCropPixels = swappedCropPixels,
                        cropWidth = raw.swapper.cropSize,
                        cropHeight = raw.swapper.cropSize,
                        targetToCrop = raw.targetToSwapperCrop,
                        blendConstraintMask = parsed.mask,
                    )
                    try {
                        val productionResult = photoPipeline.process(
                            PhotoFaceSwapRequest(
                                source = source,
                                target = target,
                                sourceFaceHint = raw.sourceFace.box,
                                targetFaceHint = raw.targetFace.box,
                                resolvedTargetFaces = listOf(raw.targetFace),
                                cachedSourceEmbedding = raw.sourceEmbedding,
                                swapBlendMaskMode = SwapBlendMaskMode.PARSER_REGION,
                                parserSession = parserSession,
                                backend = RequestedInferenceBackend.CPU_ONLY,
                            ),
                        )
                        production = productionResult
                        val productionPixels = productionResult.finalBitmap.pixels()
                        val committedPixels = committedBox.pixels()
                        try {
                            assertArrayEquals(
                                "$pair box-only regression must remain bit-identical",
                                committedPixels,
                                boxComposite.pixels,
                            )
                            assertArrayEquals(
                                "$pair production parser output must equal the isolated composite",
                                parserComposite.pixels,
                                productionPixels,
                            )
                            assertArrayEquals(
                                "$pair production parser mask must equal the isolated mask",
                                parserComposite.cropMask,
                                productionResult.cropMask,
                                0f,
                            )
                            assertEquals(InferenceBackend.CPU, productionResult.parserBackend)
                            assertTrue(productionResult.timings.parserMs > 0L)
                            assertEquals(boxComposite.colorAdjustment, parserComposite.colorAdjustment)
                            assertEquals(parserComposite.colorAdjustment, productionResult.colorAdjustment)
                            assertArrayEquals(
                                "$pair parser mask must not change color-matched crop",
                                boxComposite.colorMatchedCrop,
                                parserComposite.colorMatchedCrop,
                            )
                            assertTrue(
                                "$pair parser mask must only reduce box alpha",
                                parserComposite.cropMask.indices.all { index ->
                                    parserComposite.cropMask[index] <= boxComposite.cropMask[index] + 1e-6f
                                },
                            )
                        } finally {
                            productionPixels.fill(0)
                            committedPixels.fill(0)
                        }

                        val outsideBox = outsideRoiChanges(
                            targetPixels,
                            boxComposite.pixels,
                            target.width,
                            target.height,
                            boxComposite.roi,
                        )
                        val outsideParser = outsideRoiChanges(
                            targetPixels,
                            parserComposite.pixels,
                            target.width,
                            target.height,
                            parserComposite.roi,
                        )
                        assertEquals(0, outsideBox)
                        assertEquals(0, outsideParser)

                        val excluded = boxComposite.cropMask.indices.filter { index ->
                            boxComposite.cropMask[index] >= 0.5f &&
                                parserComposite.cropMask[index] <= 0.05f
                        }
                        assertTrue("$pair parser did not exclude a measurable hair/neck band", excluded.size > 100)
                        val beforeBandMae = cropBandMae(
                            boxComposite.colorMatchedCrop,
                            targetCropPixels,
                            excluded,
                        )
                        val afterBandMae = blendedCropBandMae(
                            boxComposite.colorMatchedCrop,
                            targetCropPixels,
                            parserComposite.cropMask,
                            excluded,
                        )
                        assertTrue(
                            "$pair protected band did not move toward target: $beforeBandMae -> $afterBandMae",
                            afterBandMae < beforeBandMae,
                        )

                        val boxBitmap = bitmapFromPixels(boxComposite.pixels, target.width, target.height)
                        try {
                            savePng(boxBitmap, File(outputDirectory, "${pair}_box_before.png"))
                            savePng(productionResult.finalBitmap, File(outputDirectory, "${pair}_parser_after.png"))
                            saveSideBySide(
                                boxBitmap,
                                productionResult.finalBitmap,
                                File(outputDirectory, "${pair}_box_vs_parser.png"),
                            )
                        } finally {
                            boxBitmap.recycleSafely()
                        }
                        saveMask(
                            parsed.mask,
                            raw.swapper.cropSize,
                            File(outputDirectory, "${pair}_region_mask_128.png"),
                        )
                        saveMask(
                            parserComposite.cropMask,
                            raw.swapper.cropSize,
                            File(outputDirectory, "${pair}_blend_mask_128.png"),
                        )

                        metrics.put(
                            JSONObject()
                                .put("pair", pair)
                                .put("box_before_matches_committed_pixels", true)
                                .put("production_parser_matches_isolated_pixels", true)
                                .put("parser_backend", parsed.backend.name)
                                .put("parser_inference_ms", parsed.inferenceMs)
                                .put("production_parser_inference_ms", productionResult.timings.parserMs)
                                .put("runtime_target_landmarks", landmarksJson(raw.targetFace.landmarks))
                                .put("target_to_swapper_crop", affineJson(raw.targetToSwapperCrop))
                                .put("paste_roi", roiJson(parserComposite.roi))
                                .put("outside_box_roi_changed_pixels", outsideBox)
                                .put("outside_parser_roi_changed_pixels", outsideParser)
                                .put("box_vs_parser_changed_pixels", changedPixels(boxComposite.pixels, parserComposite.pixels))
                                .put("excluded_band_crop_pixels", excluded.size)
                                .put("excluded_band_box_mae_to_target", beforeBandMae)
                                .put("excluded_band_parser_mae_to_target", afterBandMae)
                                .put("color_adjustment_unchanged", true),
                        )
                    } finally {
                        parsed.mask.fill(0f)
                        boxComposite.cropMask.fill(0f)
                        boxComposite.warpedMask?.fill(0f)
                        boxComposite.colorMatchedCrop.fill(0)
                        boxComposite.pixels.fill(0)
                        parserComposite.cropMask.fill(0f)
                        parserComposite.warpedMask?.fill(0f)
                        parserComposite.colorMatchedCrop.fill(0)
                        parserComposite.pixels.fill(0)
                        targetPixels.fill(0)
                        targetCropPixels.fill(0)
                        swappedCropPixels.fill(0)
                    }
                } finally {
                    production?.sourceEmbedding?.fill(0f)
                    production?.cropMask?.fill(0f)
                    production?.finalBitmap?.recycleSafely()
                    raw?.release()
                    source.recycleSafely()
                    target.recycleSafely()
                    committedBox.recycleSafely()
                }
            }
        }

        assertEquals(1, listener.openCount(BISENET_FILE))
        assertEquals(1, listener.closeCount(BISENET_FILE))
        assertEquals(4, listener.openCount(INSWAPPER_FILE))
        assertEquals(4, listener.closeCount(INSWAPPER_FILE))
        assertTrue(listener.peakSets.contains(setOf(BISENET_FILE, INSWAPPER_FILE)))
        assertEquals(2, listener.maxSimultaneousHeavySessions)
        assertTrue(listener.activeHeavySessions.isEmpty())
        File(outputDirectory, "checkpoint_2_results.json").writeText(
            JSONObject()
                .put("device_api", 35)
                .put("backend", "CPU")
                .put("pairs", metrics)
                .put("visual_gate", "manual checklist: hair/temple/beard boundary on pair_02 and pair_03")
                .put("session_events", JSONArray(listener.events))
                .put("observed_peak_pairs", peakSetsJson(listener.peakSets))
                .put("max_simultaneous_heavy_sessions", listener.maxSimultaneousHeavySessions)
                .toString(2),
        )
    }

    @Test
    fun productionPassesUseBisenetWithSwapperThenGfpganWithoutOverlap() = runBlocking {
        val outputDirectory = File(context.filesDir, OUTPUT_DIRECTORY).apply {
            if (!isDirectory) check(mkdirs()) { "Could not create $absolutePath" }
        }
        val lifecycleFile = File(outputDirectory, "session_lifecycle_results.json")
        if (lifecycleFile.exists()) check(lifecycleFile.delete()) {
            "Could not delete stale ${lifecycleFile.name}"
        }
        val listener = RecordingSessionLifecycle()
        val store = ModelStore(context)
        requireModels(store, RESTORATION_REQUIRED_MODELS)
        val raw = OnnxRawFaceSwapPipeline(store, sessionLifecycle = listener)
        val photo = OnnxPhotoFaceSwapPipeline(store, rawPipeline = raw)
        val enhancer = OnnxFaceEnhancerPipeline(store, sessionLifecycle = listener)
        val parser = OnnxFaceParserPipeline(store, sessionLifecycle = listener)
        val coordinator = OnnxMultiPhotoFaceSwapPipeline(raw, photo, enhancer, parser)
        val source = bitmap("inputs/pair_02_source.png")
        val target = bitmap("inputs/pair_02_target.png")
        var result: MultiPhotoFaceSwapResult? = null
        val started = System.nanoTime()
        try {
            result = requireNotNull(
                coordinator.process(
                    target = target,
                    sources = listOf(
                        MultiPhotoSource(
                            FaceId("source-1"),
                            source,
                            FaceBox(0.0, 0.0, source.width.toDouble(), source.height.toDouble()),
                        ),
                    ),
                    targetsInStableOrder = listOf(
                        MultiPhotoTarget(
                            FaceId("target-1"),
                            FaceBox(0.0, 0.0, target.width.toDouble(), target.height.toDouble()),
                        ),
                    ),
                    assignments = listOf(
                        MultiPhotoAssignment(FaceId("target-1"), FaceId("source-1")),
                    ),
                    backend = RequestedInferenceBackend.CPU_ONLY,
                    restorationStrength = 0.8f,
                    swapBlendMaskMode = SwapBlendMaskMode.PARSER_REGION,
                ),
            )
            val elapsedMs = (System.nanoTime() - started) / 1_000_000L
            assertEquals(1, listener.openCount(BISENET_FILE))
            assertEquals(1, listener.closeCount(BISENET_FILE))
            assertEquals(1, listener.openCount(INSWAPPER_FILE))
            assertEquals(1, listener.closeCount(INSWAPPER_FILE))
            assertEquals(1, listener.openCount(GFPGAN_FILE))
            assertEquals(1, listener.closeCount(GFPGAN_FILE))
            assertTrue(listener.peakSets.contains(setOf(BISENET_FILE, INSWAPPER_FILE)))
            assertTrue(listener.peakSets.contains(setOf(BISENET_FILE, GFPGAN_FILE)))
            assertFalse(listener.peakSets.any { INSWAPPER_FILE in it && GFPGAN_FILE in it })
            assertEquals(2, listener.maxSimultaneousHeavySessions)
            assertTrue(listener.activeHeavySessions.isEmpty())
            assertEquals(listOf(InferenceBackend.CPU), result.swapParserBackends)
            assertEquals(listOf(InferenceBackend.CPU), result.enhancementParserBackends)
            assertTrue(result.swapParserMs > 0L)
            val lastSwapperClose = listener.events.indexOfLast { it == "close:$INSWAPPER_FILE" }
            val firstGfpganOpen = listener.events.indexOfFirst { it == "open:$GFPGAN_FILE" }
            assertTrue(lastSwapperClose in 0 until firstGfpganOpen)

            lifecycleFile.writeText(
                JSONObject()
                    .put("pair", "pair_02")
                    .put("restoration_strength", 0.8)
                    .put("elapsed_ms", elapsedMs)
                    .put("swap_parser_ms", result.swapParserMs)
                    .put("enhancement_ms", result.enhancementMs)
                    .put("events", JSONArray(listener.events))
                    .put("observed_peak_pairs", peakSetsJson(listener.peakSets))
                    .put("max_simultaneous_heavy_sessions", listener.maxSimultaneousHeavySessions)
                    .toString(2),
            )
        } finally {
            result?.finalBitmap?.recycleSafely()
            source.recycleSafely()
            target.recycleSafely()
        }
    }

    private suspend fun requireModels(store: ModelStore, ids: Set<ModelId>) {
        val statuses = store.refreshStatuses()
        ids.forEach { id -> assertTrue("${id.stableId} must be imported", statuses[id] is ModelStatus.Ready) }
    }

    private fun visualOutputDirectory(): File = File(context.filesDir, OUTPUT_DIRECTORY).apply {
        if (!isDirectory) check(mkdirs()) { "Could not create $absolutePath" }
        listFiles()?.filter { file ->
            file.name == "checkpoint_2_results.json" || file.name.startsWith("pair_0")
        }?.forEach { file -> check(file.delete()) { "Could not delete stale ${file.name}" } }
    }

    private fun bitmap(path: String): Bitmap = assets.open(path).use { input ->
        requireNotNull(BitmapFactory.decodeStream(input)) { "Cannot decode $path" }
            .copy(Bitmap.Config.ARGB_8888, false)
    }

    private fun Bitmap.pixels(): IntArray = IntArray(width * height).also { destination ->
        getPixels(destination, 0, width, 0, 0, width, height)
    }

    private fun bitmapFromPixels(pixels: IntArray, width: Int, height: Int): Bitmap =
        Bitmap.createBitmap(pixels, width, height, Bitmap.Config.ARGB_8888)

    private fun savePng(bitmap: Bitmap, file: File) {
        FileOutputStream(file).use { output -> check(bitmap.compress(Bitmap.CompressFormat.PNG, 100, output)) }
    }

    private fun saveSideBySide(before: Bitmap, after: Bitmap, file: File) {
        require(before.width == after.width && before.height == after.height)
        val beforePixels = before.pixels()
        val afterPixels = after.pixels()
        val gap = 8
        val outputWidth = before.width * 2 + gap
        val output = IntArray(outputWidth * before.height) { OPAQUE_WHITE }
        try {
            for (y in 0 until before.height) {
                beforePixels.copyInto(output, y * outputWidth, y * before.width, (y + 1) * before.width)
                afterPixels.copyInto(
                    output,
                    y * outputWidth + before.width + gap,
                    y * after.width,
                    (y + 1) * after.width,
                )
            }
            val bitmap = Bitmap.createBitmap(output, outputWidth, before.height, Bitmap.Config.ARGB_8888)
            try {
                savePng(bitmap, file)
            } finally {
                bitmap.recycleSafely()
            }
        } finally {
            beforePixels.fill(0)
            afterPixels.fill(0)
            output.fill(0)
        }
    }

    private fun saveMask(mask: FloatArray, size: Int, file: File) {
        require(mask.size == size * size)
        val pixels = IntArray(mask.size) { index ->
            val value = (mask[index].coerceIn(0f, 1f) * 255f).toInt()
            (0xff shl 24) or (value shl 16) or (value shl 8) or value
        }
        val bitmap = Bitmap.createBitmap(pixels, size, size, Bitmap.Config.ARGB_8888)
        try {
            savePng(bitmap, file)
        } finally {
            bitmap.recycleSafely()
            pixels.fill(0)
        }
    }

    private fun outsideRoiChanges(
        original: IntArray,
        actual: IntArray,
        width: Int,
        height: Int,
        roi: CompositeRoi,
    ): Int = (0 until height).sumOf { y ->
        (0 until width).count { x ->
            (x !in roi.left until roi.right || y !in roi.top until roi.bottom) &&
                original[y * width + x] != actual[y * width + x]
        }
    }

    private fun changedPixels(first: IntArray, second: IntArray): Int =
        first.indices.count { index -> first[index] != second[index] }

    private fun cropBandMae(first: IntArray, target: IntArray, indices: List<Int>): Double =
        indices.sumOf { index -> rgbAbsoluteError(first[index], target[index]) } /
            (indices.size * 3.0)

    private fun blendedCropBandMae(
        swapped: IntArray,
        target: IntArray,
        mask: FloatArray,
        indices: List<Int>,
    ): Double = indices.sumOf { index ->
        val alpha = mask[index].toDouble()
        intArrayOf(16, 8, 0).sumOf { shift ->
            val targetValue = (target[index] ushr shift) and 0xff
            val swappedValue = (swapped[index] ushr shift) and 0xff
            abs((targetValue * (1.0 - alpha) + swappedValue * alpha) - targetValue)
        }
    } / (indices.size * 3.0)

    private fun rgbAbsoluteError(first: Int, second: Int): Int =
        intArrayOf(16, 8, 0).sumOf { shift ->
            abs(((first ushr shift) and 0xff) - ((second ushr shift) and 0xff))
        }

    private fun landmarksJson(points: List<Point2>): JSONArray = JSONArray().also { array ->
        points.forEach { point -> array.put(JSONArray().put(point.x).put(point.y)) }
    }

    private fun affineJson(matrix: AffineMatrix): JSONArray = JSONArray()
        .put(JSONArray().put(matrix.a).put(matrix.b).put(matrix.c))
        .put(JSONArray().put(matrix.d).put(matrix.e).put(matrix.f))

    private fun roiJson(roi: CompositeRoi): JSONObject = JSONObject()
        .put("left", roi.left)
        .put("top", roi.top)
        .put("right", roi.right)
        .put("bottom", roi.bottom)

    private fun peakSetsJson(sets: Set<Set<String>>): JSONArray = JSONArray().also { array ->
        sets.sortedBy { it.joinToString() }.forEach { set -> array.put(JSONArray(set.sorted())) }
    }

    private fun RawFaceSwapResult.release() {
        alignedSource112.recycleSafely()
        alignedTarget.recycleSafely()
        rawOutputBitmap.recycleSafely()
        rawOutput.fill(0f)
        rawMask?.fill(0f)
        sourceEmbedding.fill(0f)
    }

    private fun Bitmap.recycleSafely() {
        if (!isRecycled) recycle()
    }

    private class RecordingSessionLifecycle : InferenceSessionLifecycleListener {
        val events = mutableListOf<String>()
        val activeHeavySessions = mutableSetOf<String>()
        val peakSets = mutableSetOf<Set<String>>()
        var maxSimultaneousHeavySessions = 0
            private set

        @Synchronized
        override fun onSessionOpened(modelFileName: String) {
            events += "open:$modelFileName"
            if (modelFileName in HEAVY_MODEL_FILES) {
                check(activeHeavySessions.add(modelFileName)) { "$modelFileName opened twice" }
                peakSets += activeHeavySessions.toSet()
                maxSimultaneousHeavySessions = maxOf(
                    maxSimultaneousHeavySessions,
                    activeHeavySessions.size,
                )
            }
        }

        @Synchronized
        override fun onSessionClosed(modelFileName: String) {
            if (modelFileName in HEAVY_MODEL_FILES) {
                check(activeHeavySessions.remove(modelFileName)) { "$modelFileName close without open" }
            }
            events += "close:$modelFileName"
        }

        fun openCount(file: String): Int = events.count { it == "open:$file" }
        fun closeCount(file: String): Int = events.count { it == "close:$file" }
    }

    private companion object {
        const val OUTPUT_DIRECTORY = "stage-e-checkpoint-2"
        const val INSWAPPER_FILE = "inswapper_128_fp16.onnx"
        const val GFPGAN_FILE = "gfpgan_1.4.onnx"
        const val BISENET_FILE = "bisenet_resnet_34.onnx"
        const val OPAQUE_WHITE = -0x1
        val HEAVY_MODEL_FILES = setOf(INSWAPPER_FILE, GFPGAN_FILE, BISENET_FILE)
        val RESTORATION_REQUIRED_MODELS = setOf(
            ModelId.YOLOFACE_8N,
            ModelId.ARCFACE_W600K_R50,
            ModelId.INSWAPPER_128_FP16,
            ModelId.GFPGAN_1_4,
            ModelId.BISENET_RESNET_34,
        )
    }
}
