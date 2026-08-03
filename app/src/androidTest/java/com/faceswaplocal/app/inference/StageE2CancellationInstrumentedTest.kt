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
 * Stage E2 checkpoint 1: cancelling a real multi-face run stops at the coordinator's
 * safe boundary, closes every heavyweight session, leaves no staging file, and never
 * publishes a result bitmap. Progress is read from the same production callback the
 * ViewModel uses.
 */
@RunWith(AndroidJUnit4::class)
class StageE2CancellationInstrumentedTest {
    private val context = ApplicationProvider.getApplicationContext<Context>()
    private val assets = InstrumentationRegistry.getInstrumentation().context.assets

    @Test
    fun cancelDuringSwapReleasesSessionsAndLeavesNoTemporaryData() {
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
                        // Stop as soon as the second assigned face is about to be swapped.
                        if (progress.stage == ProcessingStage.SWAPPING &&
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
            assertFalse(
                "restoration must never start after a cancel inside the swap pass",
                stages.contains(ProcessingStage.RESTORING),
            )
            observed.forEach { progress ->
                assertEquals("only assigned faces are counted", 3, progress.totalFaces)
                assertTrue(progress.fraction in 0f..1f)
            }
            assertEquals(
                "the reported face number must stay one-based inside the total",
                listOf(1, 2),
                observed.filter { it.stage == ProcessingStage.SWAPPING }
                    .map(ProcessingProgress::currentFace)
                    .distinct(),
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
        private val openHeavy = mutableSetOf<String>()

        val openHeavySessionCount: Int @Synchronized get() = openHeavy.size

        @Synchronized
        override fun onSessionOpened(modelFileName: String) {
            if (modelFileName in HEAVY_MODEL_FILES) {
                check(openHeavy.add(modelFileName)) { "$modelFileName was opened twice" }
            }
        }

        @Synchronized
        override fun onSessionClosed(modelFileName: String) {
            if (modelFileName in HEAVY_MODEL_FILES) {
                check(openHeavy.remove(modelFileName)) { "$modelFileName close without open" }
            }
        }
    }

    private companion object {
        const val RESTORATION_STRENGTH = 0.8f
        val HEAVY_MODEL_FILES = setOf(
            "inswapper_128_fp16.onnx",
            "gfpgan_1.4.onnx",
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
