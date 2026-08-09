package com.faceswaplocal.app.inference

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.faceswaplocal.app.domain.FaceId
import com.faceswaplocal.app.domain.ProcessingProgress
import com.faceswaplocal.app.domain.ProcessingStage
import java.util.Collections
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

/**
 * AUDIT_STAGE_E1-E2.md §2.1: the coordinator's "every produced embedding is zeroed in
 * `finally`" claim was the only one of eight audited claims held up by reading the code
 * alone, with no assert on the actual array contents. `EmbeddingLifecycleListener`
 * captures the exact array references `OnnxMultiPhotoFaceSwapPipeline` will later zero,
 * so these tests check the arrays themselves after the run ends — on success and on a
 * mid-run cancellation.
 */
@RunWith(AndroidJUnit4::class)
class StageEEmbeddingHygieneInstrumentedTest {
    private val context = ApplicationProvider.getApplicationContext<Context>()
    private val assets = InstrumentationRegistry.getInstrumentation().context.assets

    @Test
    fun embeddingsAreZeroedAfterASuccessfulRun() {
        val listener = RecordingEmbeddingLifecycle()
        val store = ModelStore(context)
        val statuses = runBlocking { store.refreshStatuses() }
        REQUIRED_MODELS.forEach { id ->
            assertTrue("${id.stableId} must be imported", statuses[id] is ModelStatus.Ready)
        }
        val raw = OnnxRawFaceSwapPipeline(store)
        val photo = OnnxPhotoFaceSwapPipeline(store, rawPipeline = raw)
        val coordinator = OnnxMultiPhotoFaceSwapPipeline(raw, photo, embeddingLifecycle = listener)

        val target = bitmap("inputs/pair_01_target.png")
        val source = bitmap("inputs/pair_01_source.png")
        var result: MultiPhotoFaceSwapResult? = null
        try {
            val targetFace = runBlocking {
                raw.detectFaces(target, RequestedInferenceBackend.CPU_ONLY).first
                    .maxByOrNull(DetectedFace5::score)
            } ?: error("target face was not detected")
            val sourceFace = runBlocking {
                raw.detectFaces(source, RequestedInferenceBackend.CPU_ONLY).first
                    .maxByOrNull(DetectedFace5::score)
            } ?: error("source face was not detected")

            val targetId = FaceId("target-1")
            val sourceId = FaceId("source-1")
            result = runBlocking {
                coordinator.process(
                    target = target,
                    sources = listOf(MultiPhotoSource(sourceId, source, sourceFace.box)),
                    targetsInStableOrder = listOf(MultiPhotoTarget(targetId, targetFace.box)),
                    assignments = listOf(MultiPhotoAssignment(targetId, sourceId)),
                    backend = RequestedInferenceBackend.CPU_ONLY,
                    swapBlendMaskMode = SwapBlendMaskMode.AFFINE_BOX,
                )
            }
            requireNotNull(result) { "coordinator returned no result" }

            listener.assertEveryCapturedEmbeddingIsZeroed()
        } finally {
            result?.finalBitmap?.recycleSafely()
            source.recycleSafely()
            target.recycleSafely()
        }
    }

    @Test
    fun embeddingsAreZeroedAfterACancelledRun() {
        val listener = RecordingEmbeddingLifecycle()
        val store = ModelStore(context)
        val statuses = runBlocking { store.refreshStatuses() }
        REQUIRED_MODELS.forEach { id ->
            assertTrue("${id.stableId} must be imported", statuses[id] is ModelStatus.Ready)
        }
        val raw = OnnxRawFaceSwapPipeline(store)
        val photo = OnnxPhotoFaceSwapPipeline(store, rawPipeline = raw)
        val coordinator = OnnxMultiPhotoFaceSwapPipeline(raw, photo, embeddingLifecycle = listener)

        val target = bitmap("inputs/stage_d_group_target.png")
        val sources = (1..3).map { bitmap("inputs/pair_%02d_source.png".format(it)) }
        val cancelSignal = Job()
        try {
            val detected = runBlocking {
                raw.detectFaces(target, RequestedInferenceBackend.CPU_ONLY).first
            }
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
                    swapBlendMaskMode = SwapBlendMaskMode.AFFINE_BOX,
                    onProgress = { progress: ProcessingProgress ->
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
            listener.assertEveryCapturedEmbeddingIsZeroed()
        } finally {
            target.recycleSafely()
            sources.forEach(Bitmap::recycle)
        }
    }

    private fun bitmap(assetPath: String): Bitmap = assets.open(assetPath).use { input ->
        requireNotNull(BitmapFactory.decodeStream(input)) { "Cannot decode $assetPath" }
            .copy(Bitmap.Config.ARGB_8888, true)
    }

    private fun Bitmap.recycleSafely() {
        if (!isRecycled) recycle()
    }

    /**
     * Captures the exact array references the coordinator will zero in `finally`, plus
     * whether each one actually carried non-zero data at capture time — without that
     * check the zeroed-at-the-end assert would pass vacuously for an array that was
     * never populated in the first place.
     */
    private class RecordingEmbeddingLifecycle : EmbeddingLifecycleListener {
        private val captured = Collections.synchronizedList(mutableListOf<FloatArray>())
        private val wasNonZeroAtCapture = Collections.synchronizedList(mutableListOf<Boolean>())

        override fun onEmbeddingProduced(embedding: FloatArray) {
            captured += embedding
            wasNonZeroAtCapture += embedding.any { it != 0f }
        }

        fun assertEveryCapturedEmbeddingIsZeroed() {
            assertTrue("at least one embedding must have been produced", captured.isNotEmpty())
            assertTrue(
                "captured embeddings must have carried real data, or the zeroed-at-the-" +
                    "end check below would be vacuous",
                wasNonZeroAtCapture.all { it },
            )
            captured.forEachIndexed { index, embedding ->
                assertFalse(
                    "embedding #$index must be zeroed once the run ends",
                    embedding.any { it != 0f },
                )
            }
        }
    }

    private companion object {
        val REQUIRED_MODELS = setOf(
            ModelId.YOLOFACE_8N,
            ModelId.ARCFACE_W600K_R50,
            ModelId.INSWAPPER_128_FP16,
        )
    }
}
