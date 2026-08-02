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
import kotlin.math.ceil
import kotlin.math.floor
import kotlin.math.sqrt
import kotlinx.coroutines.runBlocking
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

/** Real API-35 checkpoint for the assigned-only, two-pass Stage-E coordinator. */
@RunWith(AndroidJUnit4::class)
class StageETwoPassCoordinatorInstrumentedTest {
    private val context = ApplicationProvider.getApplicationContext<Context>()
    private val assets = InstrumentationRegistry.getInstrumentation().context.assets

    @Test
    fun twoPassRestorationPreservesAssignmentsIdentityAndWriteUnion() = runBlocking {
        val listener = RecordingSessionLifecycle()
        val store = ModelStore(context)
        val statuses = store.refreshStatuses()
        REQUIRED_MODELS.forEach { id ->
            assertTrue("${id.stableId} must be imported", statuses[id] is ModelStatus.Ready)
        }
        val raw = OnnxRawFaceSwapPipeline(store, sessionLifecycle = listener)
        val photo = OnnxPhotoFaceSwapPipeline(store, rawPipeline = raw)
        val enhancer = OnnxFaceEnhancerPipeline(store, sessionLifecycle = listener)
        val coordinator = OnnxMultiPhotoFaceSwapPipeline(raw, photo, enhancer)

        val target = bitmap("inputs/stage_d_group_target.png")
        val sourceBitmaps = (1..3).map { bitmap("inputs/pair_%02d_source.png".format(it)) }
        val sourceEmbeddings = mutableListOf<FloatArray>()
        val originalPixels = target.pixels()
        val outputDirectory = File(context.filesDir, OUTPUT_DIRECTORY).apply { mkdirs() }
        var baseline: MultiPhotoFaceSwapResult? = null
        var restored: MultiPhotoFaceSwapResult? = null
        try {
            val detected = raw.detectFaces(target, RequestedInferenceBackend.CPU_ONLY).first
            assertEquals("fixture must contain exactly four neural faces", 4, detected.size)
            val orderedTargets = orderFixtureFaces(detected)
            sourceBitmaps.forEach { source ->
                val face = raw.detectFaces(source, RequestedInferenceBackend.CPU_ONLY)
                    .first
                    .maxByOrNull(DetectedFace5::score)
                    ?: error("source face was not detected")
                sourceEmbeddings += raw.extractIdentityEmbedding(
                    source,
                    face,
                    RequestedInferenceBackend.CPU_ONLY,
                ).embedding
            }

            val targetIds = (1..4).map { FaceId("target-$it") }
            val sourceIds = (1..3).map { FaceId("source-$it") }
            val sources = sourceBitmaps.mapIndexed { index, bitmap ->
                MultiPhotoSource(
                    sourceIds[index],
                    bitmap,
                    FaceBox(0.0, 0.0, bitmap.width.toDouble(), bitmap.height.toDouble()),
                )
            }
            val targets = orderedTargets.mapIndexed { index, face ->
                MultiPhotoTarget(targetIds[index], face.box)
            }
            val assignments = (0..2).map { index ->
                MultiPhotoAssignment(targetIds[index], sourceIds[index])
            }

            val baselineEventStart = listener.size
            val baselineStarted = System.nanoTime()
            baseline = requireNotNull(
                coordinator.process(
                    target = target,
                    sources = sources,
                    targetsInStableOrder = targets,
                    assignments = assignments,
                    backend = RequestedInferenceBackend.CPU_ONLY,
                    restorationStrength = 0f,
                ),
            )
            val baselineMs = (System.nanoTime() - baselineStarted) / 1_000_000L
            val baselineEvents = listener.eventsFrom(baselineEventStart)
            assertBaselineSessionEvents(baselineEvents)
            val baselineMetrics = verifyRun(
                label = "strength_0",
                result = baseline,
                strength = 0f,
                originalPixels = originalPixels,
                targetWidth = target.width,
                targetHeight = target.height,
                orderedTargets = orderedTargets,
                targetIds = targetIds,
                sourceEmbeddings = sourceEmbeddings,
                raw = raw,
                elapsedMs = baselineMs,
            )
            savePng(baseline.finalBitmap, File(outputDirectory, "strength_0.png"))
            baseline.finalBitmap.recycleSafely()
            baseline = null

            val restoredEventStart = listener.size
            val restoredStarted = System.nanoTime()
            restored = requireNotNull(
                coordinator.process(
                    target = target,
                    sources = sources,
                    targetsInStableOrder = targets,
                    assignments = assignments,
                    backend = RequestedInferenceBackend.CPU_ONLY,
                    restorationStrength = RESTORATION_STRENGTH,
                ),
            )
            val restoredMs = (System.nanoTime() - restoredStarted) / 1_000_000L
            val restoredEvents = listener.eventsFrom(restoredEventStart)
            assertRestoredSessionEvents(restoredEvents)
            val restoredMetrics = verifyRun(
                label = "strength_0_8",
                result = restored,
                strength = RESTORATION_STRENGTH,
                originalPixels = originalPixels,
                targetWidth = target.width,
                targetHeight = target.height,
                orderedTargets = orderedTargets,
                targetIds = targetIds,
                sourceEmbeddings = sourceEmbeddings,
                raw = raw,
                elapsedMs = restoredMs,
            )
            savePng(restored.finalBitmap, File(outputDirectory, "strength_0_8.png"))

            for (targetIndex in 0..2) {
                val baselineIdentity = baselineMetrics.getJSONArray("identity").getJSONObject(targetIndex)
                val restoredIdentity = restoredMetrics.getJSONArray("identity").getJSONObject(targetIndex)
                assertEquals(
                    "restoration must not change nearest source for T${targetIndex + 1}",
                    baselineIdentity.getInt("nearest_source"),
                    restoredIdentity.getInt("nearest_source"),
                )
            }

            val metrics = JSONObject()
                .put("fixture", "inputs/stage_d_group_target.png")
                .put("device_api", 35)
                .put("backend", "CPU")
                .put("configuration_difference", "FaceFusion desktop defaults to enhancing all detected faces; Android checkpoint enhances assigned targets only")
                .put("baseline", baselineMetrics)
                .put("restored", restoredMetrics)
                .put("identity_deltas", identityDeltas(baselineMetrics, restoredMetrics))
                .put("baseline_session_events", JSONArray(baselineEvents))
                .put("restored_session_events", JSONArray(restoredEvents))
                .put("max_simultaneous_heavy_sessions", listener.maxSimultaneousHeavySessions)
            File(outputDirectory, "checkpoint_1_results.json").writeText(metrics.toString(2))
            assertEquals("at most one swapper/restorer/parser session may be open", 1, listener.maxSimultaneousHeavySessions)
        } finally {
            baseline?.finalBitmap?.recycleSafely()
            restored?.finalBitmap?.recycleSafely()
            sourceEmbeddings.forEach { it.fill(0f) }
            sourceBitmaps.forEach { bitmap -> bitmap.recycleSafely() }
            target.recycleSafely()
        }
    }

    private suspend fun verifyRun(
        label: String,
        result: MultiPhotoFaceSwapResult,
        strength: Float,
        originalPixels: IntArray,
        targetWidth: Int,
        targetHeight: Int,
        orderedTargets: List<DetectedFace5>,
        targetIds: List<FaceId>,
        sourceEmbeddings: List<FloatArray>,
        raw: OnnxRawFaceSwapPipeline,
        elapsedMs: Long,
    ): JSONObject {
        assertEquals(3, result.swapRois.size)
        assertEquals(targetIds.take(3), result.swapRois.map(AppliedFaceRoi::targetId))
        if (strength == 0f) {
            assertTrue("strength zero must skip enhancement", result.enhanceRois.isEmpty())
            assertTrue(result.enhancerBackends.isEmpty())
            assertTrue(result.parserBackends.isEmpty())
        } else {
            assertEquals(targetIds.take(3), result.enhanceRois.map(AppliedFaceRoi::targetId))
            assertFalse("T4 must never enter enhancement", result.enhanceRois.any { it.targetId == targetIds[3] })
        }

        val finalPixels = result.finalBitmap.pixels()
        val writeUnion = (result.swapRois + result.enhanceRois).map(AppliedFaceRoi::bounds)
        val outsideChanges = outsideUnionChanges(
            originalPixels,
            finalPixels,
            targetWidth,
            targetHeight,
            writeUnion,
        )
        assertEquals("$label changed pixels outside union(swap ROI, enhance ROI)", 0, outsideChanges)

        val t4Roi = ffhqRoi(orderedTargets[3], targetWidth, targetHeight)
        if (strength > 0f) {
            assertTrue(
                "T4 must be an explicit full-image no-write region",
                t4Roi in result.protectedUnassignedRois,
            )
        }
        val t4Changes = changedInRoi(originalPixels, finalPixels, targetWidth, t4Roi)
        assertEquals("$label T4 counterfactual FFHQ ROI must remain bit-identical", 0, t4Changes)

        val identities = JSONArray()
        for (targetIndex in 0..2) {
            val embedding = raw.extractIdentityEmbedding(
                result.finalBitmap,
                orderedTargets[targetIndex],
                RequestedInferenceBackend.CPU_ONLY,
            ).embedding
            try {
                val similarities = sourceEmbeddings.map { source -> cosine(embedding, source) }
                val nearest = similarities.indices.maxByOrNull(similarities::get)
                    ?: error("empty similarity set")
                val expected = similarities[targetIndex]
                val strongestOther = similarities.filterIndexed { index, _ -> index != targetIndex }.max()
                assertEquals(
                    "$label T${targetIndex + 1} must stay nearest to source ${targetIndex + 1}: $similarities",
                    targetIndex,
                    nearest,
                )
                identities.put(
                    JSONObject()
                        .put("target", targetIndex + 1)
                        .put("expected_source", targetIndex + 1)
                        .put("similarities", JSONArray(similarities))
                        .put("rank", JSONArray(similarities.indices.sortedByDescending(similarities::get).map { it + 1 }))
                        .put("nearest_source", nearest + 1)
                        .put("expected_margin", expected - strongestOther),
                )
            } finally {
                embedding.fill(0f)
            }
        }

        return JSONObject()
            .put("strength", strength.toDouble())
            .put("elapsed_ms", elapsedMs)
            .put("swap_rois", roisJson(result.swapRois))
            .put("enhance_rois", roisJson(result.enhanceRois))
            .put(
                "protected_unassigned_rois",
                JSONArray(result.protectedUnassignedRois.map(::roiJson)),
            )
            .put("outside_union_changed_pixels", outsideChanges)
            .put("t4_counterfactual_ffhq_roi", roiJson(t4Roi))
            .put("t4_changed_pixels", t4Changes)
            .put("identity", identities)
    }

    private fun assertBaselineSessionEvents(events: List<String>) {
        val heavy = events.filter(::isHeavyEvent)
        assertEquals(3, heavy.count { it == "open:$INSWAPPER_FILE" })
        assertEquals(3, heavy.count { it == "close:$INSWAPPER_FILE" })
        assertFalse(heavy.any { GFPGAN_FILE in it || BISENET_FILE in it })
    }

    private fun assertRestoredSessionEvents(events: List<String>) {
        val heavy = events.filter(::isHeavyEvent)
        assertEquals(3, heavy.count { it == "open:$INSWAPPER_FILE" })
        assertEquals(3, heavy.count { it == "close:$INSWAPPER_FILE" })
        assertEquals(3, heavy.count { it == "open:$GFPGAN_FILE" })
        assertEquals(3, heavy.count { it == "close:$GFPGAN_FILE" })
        assertEquals(3, heavy.count { it == "open:$BISENET_FILE" })
        assertEquals(3, heavy.count { it == "close:$BISENET_FILE" })
        val lastSwapperClose = heavy.indexOfLast { it == "close:$INSWAPPER_FILE" }
        val firstRestorationOpen = heavy.indexOfFirst {
            it == "open:$GFPGAN_FILE" || it == "open:$BISENET_FILE"
        }
        assertTrue("all swaps must close before restoration opens: $heavy", lastSwapperClose in 0 until firstRestorationOpen)
    }

    private fun identityDeltas(baseline: JSONObject, restored: JSONObject): JSONArray {
        val baselineIdentity = baseline.getJSONArray("identity")
        val restoredIdentity = restored.getJSONArray("identity")
        return JSONArray().also { output ->
            for (index in 0 until baselineIdentity.length()) {
                val before = baselineIdentity.getJSONObject(index)
                val after = restoredIdentity.getJSONObject(index)
                val expectedIndex = before.getInt("expected_source") - 1
                val beforeSimilarity = before.getJSONArray("similarities").getDouble(expectedIndex)
                val afterSimilarity = after.getJSONArray("similarities").getDouble(expectedIndex)
                output.put(
                    JSONObject()
                        .put("target", index + 1)
                        .put("expected_similarity_delta", afterSimilarity - beforeSimilarity)
                        .put("margin_delta", after.getDouble("expected_margin") - before.getDouble("expected_margin"))
                        .put("nearest_changed", before.getInt("nearest_source") != after.getInt("nearest_source")),
                )
            }
        }
    }

    private fun orderFixtureFaces(faces: List<DetectedFace5>): List<DetectedFace5> =
        faces.sortedBy { (it.box.left + it.box.right) / 2.0 }.let { byX ->
            val right = byX.takeLast(2).sortedBy { (it.box.top + it.box.bottom) / 2.0 }
            listOf(byX[0], byX[1], right[0], right[1])
        }

    private fun ffhqRoi(face: DetectedFace5, width: Int, height: Int): CompositeRoi {
        val imageToCrop = FaceGeometry.estimateSimilarity(
            source = face.landmarks,
            template = WarpTemplate.FFHQ_512,
            cropWidth = 512,
            cropHeight = 512,
        )
        val cropToImage = imageToCrop.inverse()
        val corners = listOf(
            cropToImage.map(Point2(0.0, 0.0)),
            cropToImage.map(Point2(512.0, 0.0)),
            cropToImage.map(Point2(512.0, 512.0)),
            cropToImage.map(Point2(0.0, 512.0)),
        )
        return CompositeRoi(
            floor(corners.minOf(Point2::x)).toInt().coerceIn(0, width),
            floor(corners.minOf(Point2::y)).toInt().coerceIn(0, height),
            ceil(corners.maxOf(Point2::x)).toInt().coerceIn(0, width),
            ceil(corners.maxOf(Point2::y)).toInt().coerceIn(0, height),
        )
    }

    private fun cosine(first: FloatArray, second: FloatArray): Double {
        require(first.size == second.size && first.isNotEmpty())
        var dot = 0.0
        var firstSquared = 0.0
        var secondSquared = 0.0
        for (index in first.indices) {
            val a = first[index].toDouble()
            val b = second[index].toDouble()
            require(a.isFinite() && b.isFinite())
            dot += a * b
            firstSquared += a * a
            secondSquared += b * b
        }
        val denominator = sqrt(firstSquared) * sqrt(secondSquared)
        require(denominator > 0.0 && denominator.isFinite())
        return dot / denominator
    }

    private fun outsideUnionChanges(
        original: IntArray,
        actual: IntArray,
        width: Int,
        height: Int,
        rois: List<CompositeRoi>,
    ): Int = (0 until height).sumOf { y ->
        (0 until width).count { x ->
            rois.none { roi -> x in roi.left until roi.right && y in roi.top until roi.bottom } &&
                original[y * width + x] != actual[y * width + x]
        }
    }

    private fun changedInRoi(
        original: IntArray,
        actual: IntArray,
        width: Int,
        roi: CompositeRoi,
    ): Int = (roi.top until roi.bottom).sumOf { y ->
        (roi.left until roi.right).count { x -> original[y * width + x] != actual[y * width + x] }
    }

    private fun roisJson(rois: List<AppliedFaceRoi>): JSONArray = JSONArray().also { array ->
        rois.forEach { applied ->
            array.put(
                roiJson(applied.bounds)
                    .put("target_id", applied.targetId.value)
                    .put("source_id", applied.sourceId.value),
            )
        }
    }

    private fun roiJson(roi: CompositeRoi): JSONObject = JSONObject()
        .put("left", roi.left)
        .put("top", roi.top)
        .put("right", roi.right)
        .put("bottom", roi.bottom)

    private fun savePng(bitmap: Bitmap, file: File) {
        FileOutputStream(file).use { output ->
            check(bitmap.compress(Bitmap.CompressFormat.PNG, 100, output))
        }
    }

    private fun bitmap(path: String): Bitmap = assets.open(path).use { stream ->
        requireNotNull(BitmapFactory.decodeStream(stream)) { "Cannot decode $path" }
            .copy(Bitmap.Config.ARGB_8888, false)
    }

    private fun Bitmap.pixels(): IntArray = IntArray(width * height).also { destination ->
        getPixels(destination, 0, width, 0, 0, width, height)
    }

    private fun Bitmap.recycleSafely() {
        if (!isRecycled) recycle()
    }

    private class RecordingSessionLifecycle : InferenceSessionLifecycleListener {
        private val events = mutableListOf<String>()
        private val openHeavy = mutableSetOf<String>()
        var maxSimultaneousHeavySessions: Int = 0
            private set

        val size: Int @Synchronized get() = events.size

        @Synchronized
        override fun onSessionOpened(modelFileName: String) {
            events += "open:$modelFileName"
            if (modelFileName in HEAVY_MODEL_FILES) {
                check(openHeavy.add(modelFileName)) { "$modelFileName was opened twice" }
                maxSimultaneousHeavySessions = maxOf(maxSimultaneousHeavySessions, openHeavy.size)
            }
        }

        @Synchronized
        override fun onSessionClosed(modelFileName: String) {
            if (modelFileName in HEAVY_MODEL_FILES) {
                check(openHeavy.remove(modelFileName)) { "$modelFileName close without open" }
            }
            events += "close:$modelFileName"
        }

        @Synchronized
        fun eventsFrom(index: Int): List<String> = events.drop(index)
    }

    private companion object {
        const val RESTORATION_STRENGTH = 0.8f
        const val OUTPUT_DIRECTORY = "stage-e-checkpoint-1"
        const val INSWAPPER_FILE = "inswapper_128_fp16.onnx"
        const val GFPGAN_FILE = "gfpgan_1.4.onnx"
        const val BISENET_FILE = "bisenet_resnet_34.onnx"
        val HEAVY_MODEL_FILES = setOf(INSWAPPER_FILE, GFPGAN_FILE, BISENET_FILE)
        val REQUIRED_MODELS = setOf(
            ModelId.YOLOFACE_8N,
            ModelId.ARCFACE_W600K_R50,
            ModelId.INSWAPPER_128_FP16,
            ModelId.GFPGAN_1_4,
            ModelId.BISENET_RESNET_34,
        )

        fun isHeavyEvent(event: String): Boolean = HEAVY_MODEL_FILES.any(event::contains)
    }
}
