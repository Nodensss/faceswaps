package com.faceswaplocal.app.inference

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.faceswaplocal.app.domain.FaceId
import com.faceswaplocal.app.domain.ProcessingStage
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

/**
 * Checkpoint 3, measurement 2: which sessions are open at the same time, by name.
 *
 * [StageETwoPassCoordinatorInstrumentedTest] already proves the InSwapper/GFPGAN barrier,
 * but it only tracks the three heavyweight files. §5.2 says to keep open only the sessions
 * the current step needs, so this widens the net to all five and pins the two questions
 * the memory work depends on: does the detector or the recognizer survive the preparation
 * phase, and what is the largest set of concurrently open sessions.
 */
@RunWith(AndroidJUnit4::class)
class SessionCoexistenceInstrumentedTest {
    private val context = ApplicationProvider.getApplicationContext<Context>()
    private val assets = InstrumentationRegistry.getInstrumentation().context.assets

    @Test
    fun detectorAndRecognizerAreClosedBeforeRestorationStarts() = runBlocking {
        val store = ModelStore(context)
        val statuses = store.refreshStatuses()
        REQUIRED_MODELS.forEach { id ->
            assertTrue("${id.stableId} must be imported", statuses[id] is ModelStatus.Ready)
        }

        val recorder = ConcurrencyRecorder()
        val raw = OnnxRawFaceSwapPipeline(store, sessionLifecycle = recorder)
        val photo = OnnxPhotoFaceSwapPipeline(store, rawPipeline = raw)
        val enhancer = OnnxFaceEnhancerPipeline(store, sessionLifecycle = recorder)
        val parser = OnnxFaceParserPipeline(store, sessionLifecycle = recorder)
        val coordinator = OnnxMultiPhotoFaceSwapPipeline(raw, photo, enhancer, parser)

        val target = bitmap("inputs/stage_e_dense_pair_target.png")
        val source = bitmap("inputs/pair_03_source.png")
        var result: MultiPhotoFaceSwapResult? = null
        try {
            val detected = raw.detectFaces(target, RequestedInferenceBackend.CPU_ONLY).first
            val ordered = detected.sortedBy { (it.box.left + it.box.right) / 2.0 }
            val sourceFace = raw.detectFaces(source, RequestedInferenceBackend.CPU_ONLY)
                .first
                .maxByOrNull(DetectedFace5::score)
                ?: error("source face was not detected")
            val assignedId = FaceId("target-a")
            val sourceId = FaceId("source-1")

            recorder.reset()
            result = coordinator.process(
                target = target,
                sources = listOf(MultiPhotoSource(sourceId, source, sourceFace.box)),
                targetsInStableOrder = listOf(
                    MultiPhotoTarget(assignedId, ordered[0].box),
                    MultiPhotoTarget(FaceId("target-b"), ordered[1].box),
                ),
                assignments = listOf(MultiPhotoAssignment(assignedId, sourceId)),
                backend = RequestedInferenceBackend.CPU_ONLY,
                restorationStrength = RESTORATION_STRENGTH,
                swapBlendMaskMode = SwapBlendMaskMode.PARSER_REGION,
                onProgress = { progress ->
                    if (progress.stage == ProcessingStage.RESTORING) recorder.markRestorationStarted()
                },
            )
            assertTrue("the run must produce a result", result != null)

            android.util.Log.i(TAG, "events: ${recorder.events.joinToString(" ")}")
            android.util.Log.i(
                TAG,
                "widest concurrent set = ${recorder.widestSet} " +
                    "(${recorder.widestSet.size} sessions, ~${recorder.widestSetKb()} kB)",
            )
            android.util.Log.i(TAG, "open when restoration began = ${recorder.openAtRestoration}")

            assertEquals(
                "the detector must not be open once restoration starts",
                false,
                recorder.openAtRestoration.contains(DETECTOR_FILE),
            )
            assertEquals(
                "ArcFace must not be open once restoration starts",
                false,
                recorder.openAtRestoration.contains(RECOGNIZER_FILE),
            )
            assertTrue(
                "no session may coexist with both swapper and enhancer",
                recorder.widestSet.none { it == INSWAPPER_FILE } ||
                    recorder.widestSet.none { it == GFPGAN_FILE },
            )
            assertTrue(
                "everything opened must also have been closed",
                recorder.stillOpen.isEmpty(),
            )
        } finally {
            result?.finalBitmap?.let { if (!it.isRecycled) it.recycle() }
            if (!source.isRecycled) source.recycle()
            if (!target.isRecycled) target.recycle()
        }
    }

    private fun bitmap(assetPath: String): Bitmap =
        assets.open(assetPath).use { input -> BitmapFactory.decodeStream(input) }

    private class ConcurrencyRecorder : InferenceSessionLifecycleListener {
        private val open = mutableSetOf<String>()
        val events = mutableListOf<String>()
        var widestSet: Set<String> = emptySet()
            private set
        var openAtRestoration: Set<String> = emptySet()
            private set
        val stillOpen: Set<String> @Synchronized get() = open.toSet()

        @Synchronized
        fun reset() {
            events.clear()
            widestSet = emptySet()
            openAtRestoration = emptySet()
        }

        @Synchronized
        fun markRestorationStarted() {
            if (openAtRestoration.isEmpty()) openAtRestoration = open.toSet()
        }

        @Synchronized
        fun widestSetKb(): Long = costOf(widestSet)

        private fun costOf(set: Set<String>): Long = set.sumOf { RESIDENT_KB[it] ?: 0L }

        @Synchronized
        override fun onSessionOpened(modelFileName: String) {
            open += modelFileName
            events += "+$modelFileName"
            // Ranked by resident cost, not by count: {parser, detector} and
            // {parser, swapper} are both two sessions but differ by 486 MB, and it is the
            // expensive one the budget has to be built on.
            if (costOf(open) > costOf(widestSet)) widestSet = open.toSet()
        }

        @Synchronized
        override fun onSessionClosed(modelFileName: String) {
            open -= modelFileName
            events += "-$modelFileName"
        }
    }

    private companion object {
        const val TAG = "SessionCoexistence"
        const val RESTORATION_STRENGTH = 0.8f
        const val DETECTOR_FILE = "yoloface_8n.onnx"
        const val RECOGNIZER_FILE = "arcface_w600k_r50.onnx"
        const val INSWAPPER_FILE = "inswapper_128_fp16.onnx"
        const val GFPGAN_FILE = "gfpgan_1.4.onnx"
        const val BISENET_FILE = "bisenet_resnet_34.onnx"

        /** Measured by SessionFootprintInstrumentedTest on AVD API 35. */
        val RESIDENT_KB = mapOf(
            DETECTOR_FILE to 32_696L,
            RECOGNIZER_FILE to 167_252L,
            INSWAPPER_FILE to 519_408L,
            GFPGAN_FILE to 335_376L,
            BISENET_FILE to 93_376L,
        )
        val REQUIRED_MODELS = setOf(
            ModelId.YOLOFACE_8N,
            ModelId.ARCFACE_W600K_R50,
            ModelId.INSWAPPER_128_FP16,
            ModelId.GFPGAN_1_4,
            ModelId.BISENET_RESNET_34,
        )
    }
}
