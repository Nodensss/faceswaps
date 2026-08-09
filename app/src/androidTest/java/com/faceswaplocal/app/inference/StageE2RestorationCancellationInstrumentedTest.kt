package com.faceswaplocal.app.inference

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.faceswaplocal.app.data.ResultExporter
import com.faceswaplocal.app.domain.FaceId
import com.faceswaplocal.app.domain.ProcessingProgress
import com.faceswaplocal.app.domain.ProcessingStage
import java.io.File
import java.util.Collections
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

/**
 * AUDIT_STAGE_E1-E2.md §2.4: `StageE2CancellationInstrumentedTest` only cancels inside
 * the swap pass and explicitly checks restoration never starts. The symmetric case —
 * cancelling after restoration has genuinely begun, with GFPGAN opened and at least one
 * face already restored — was covered structurally (`ensureActive` plus `finally` in
 * `OnnxFaceEnhancerPipeline.enhance`) but not by a real run. This closes that gap.
 */
@RunWith(AndroidJUnit4::class)
class StageE2RestorationCancellationInstrumentedTest {
    private val context = ApplicationProvider.getApplicationContext<Context>()
    private val assets = InstrumentationRegistry.getInstrumentation().context.assets

    @Test
    fun cancelDuringRestorationClosesGfpganAndLeavesNoTemporaryData() {
        val listener = RecordingSessionLifecycle()
        val store = ModelStore(context)
        val statuses = runBlocking { store.refreshStatuses() }
        REQUIRED_MODELS.forEach { id ->
            assertTrue("${id.stableId} must be imported", statuses[id] is ModelStatus.Ready)
        }
        val raw = OnnxRawFaceSwapPipeline(store, sessionLifecycle = listener)
        val photo = OnnxPhotoFaceSwapPipeline(store, rawPipeline = raw)
        val enhancer = OnnxFaceEnhancerPipeline(store, sessionLifecycle = listener)
        val parser = OnnxFaceParserPipeline(store, sessionLifecycle = listener)
        val coordinator = OnnxMultiPhotoFaceSwapPipeline(raw, photo, enhancer, parser)
        val exporter = ResultExporter(context)
        runBlocking { exporter.sweepAbandonedData() }

        val target = bitmap("inputs/stage_d_group_target.png")
        val sources = (1..3).map { bitmap("inputs/pair_%02d_source.png".format(it)) }
        val observed = Collections.synchronizedList(mutableListOf<ProcessingProgress>())
        val cancelSignal = Job()

        try {
            val detected = runBlocking {
                raw.detectFaces(target, RequestedInferenceBackend.CPU_ONLY).first
            }
            assertEquals("fixture must contain exactly four neural faces", 4, detected.size)
            val ordered = detected.sortedBy { (it.box.left + it.box.right) / 2.0 }
            val targets = ordered.mapIndexed { index, face ->
                MultiPhotoTarget(FaceId("target-$index"), face.box)
            }
            val multiSources = sources.mapIndexed { index, source ->
                val face = runBlocking {
                    raw.detectFaces(source, RequestedInferenceBackend.CPU_ONLY)
                        .first
                        .maxByOrNull(DetectedFace5::score)
                } ?: error("source face was not detected")
                MultiPhotoSource(FaceId("source-$index"), source, face.box)
            }
            val assignments = (0..2).map { index ->
                MultiPhotoAssignment(targets[index].id, multiSources[index].id)
            }

            val running = CoroutineScope(Dispatchers.Default + cancelSignal).launch {
                coordinator.process(
                    target = target,
                    sources = multiSources,
                    targetsInStableOrder = targets,
                    assignments = assignments,
                    backend = RequestedInferenceBackend.CPU_ONLY,
                    restorationStrength = RESTORATION_STRENGTH,
                    onProgress = { progress ->
                        observed += progress
                        // Stop once the first face has actually been restored, so GFPGAN
                        // is proven open and used before the cancel lands on the second.
                        if (progress.stage == ProcessingStage.RESTORING &&
                            progress.completedFaces == 1
                        ) {
                            cancelSignal.cancel()
                        }
                    },
                )
            }
            runBlocking { running.join() }

            assertTrue("the run must end cancelled", running.isCancelled)
            val stages = observed.map(ProcessingProgress::stage)
            assertEquals(ProcessingStage.PREPARING, stages.first())
            assertTrue("detection must be reported", stages.contains(ProcessingStage.DETECTING))
            assertTrue("swapping must be reported", stages.contains(ProcessingStage.SWAPPING))
            assertTrue("restoration must be reported", stages.contains(ProcessingStage.RESTORING))
            assertTrue(
                "the swap pass must fully complete before restoration is ever reported, " +
                    "per the coordinator's pass barrier",
                observed.filter { it.stage == ProcessingStage.SWAPPING }
                    .maxOf(ProcessingProgress::completedFaces) == 3,
            )
            observed.forEach { progress ->
                assertEquals("only assigned faces are counted", 3, progress.totalFaces)
                assertTrue(progress.fraction in 0f..1f)
            }

            assertTrue(
                "GFPGAN must have been opened, or this test would not exercise restoration " +
                    "at all",
                listener.events.contains("open:$GFPGAN_FILE"),
            )
            assertEquals(
                "every heavyweight session opened by the cancelled run must be closed",
                0,
                listener.openHeavySessionCount,
            )
            assertTrue(
                "no staging file may survive a cancelled run",
                exporter.stagingFiles().isEmpty(),
            )
            assertTrue(
                "the cache must not gain an unmanaged export directory entry",
                File(context.cacheDir, ResultExporter.CACHE_DIRECTORY_NAME)
                    .listFiles()
                    .orEmpty()
                    .isEmpty(),
            )
        } finally {
            target.recycle()
            sources.forEach(Bitmap::recycle)
        }
    }

    private fun bitmap(assetPath: String): Bitmap = assets.open(assetPath).use { input ->
        BitmapFactory.decodeStream(input).copy(Bitmap.Config.ARGB_8888, true)
    }

    private class RecordingSessionLifecycle : InferenceSessionLifecycleListener {
        val events = Collections.synchronizedList(mutableListOf<String>())
        private val openHeavy = mutableSetOf<String>()

        val openHeavySessionCount: Int @Synchronized get() = openHeavy.size

        @Synchronized
        override fun onSessionOpened(modelFileName: String) {
            events += "open:$modelFileName"
            if (modelFileName in HEAVY_MODEL_FILES) {
                check(openHeavy.add(modelFileName)) { "$modelFileName was opened twice" }
            }
        }

        @Synchronized
        override fun onSessionClosed(modelFileName: String) {
            events += "close:$modelFileName"
            if (modelFileName in HEAVY_MODEL_FILES) {
                check(openHeavy.remove(modelFileName)) { "$modelFileName close without open" }
            }
        }
    }

    private companion object {
        const val RESTORATION_STRENGTH = 0.8f
        const val GFPGAN_FILE = "gfpgan_1.4.onnx"
        val HEAVY_MODEL_FILES = setOf(
            "inswapper_128_fp16.onnx",
            GFPGAN_FILE,
            "bisenet_resnet_34.onnx",
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
