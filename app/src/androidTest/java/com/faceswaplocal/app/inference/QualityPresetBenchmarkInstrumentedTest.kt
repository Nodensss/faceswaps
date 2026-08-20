package com.faceswaplocal.app.inference

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.SystemClock
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.faceswaplocal.app.domain.FaceId
import com.faceswaplocal.app.domain.QualityPreset
import java.io.File
import kotlinx.coroutines.runBlocking
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

/**
 * Checkpoint 7: what each quality preset actually costs, per assigned face.
 *
 * The presets exist because the live acceptance run showed GFPGAN eating 46 of 55 minutes,
 * so the UI has to quote a number rather than a vague promise. The figures it quotes come
 * from here and are written to `stage-e-checkpoint-7/quality_preset_benchmark.json`.
 *
 * One assigned face is timed, because that is the unit the UI multiplies by: the swapper,
 * the parser and GFPGAN all run on fixed-size crops, so per-face cost barely moves with
 * the photo's resolution — only the full-frame blend does, and it is the small term.
 */
@RunWith(AndroidJUnit4::class)
class QualityPresetBenchmarkInstrumentedTest {
    private val context = ApplicationProvider.getApplicationContext<Context>()
    private val assets = InstrumentationRegistry.getInstrumentation().context.assets

    @Test
    fun eachPresetIsTimedOnOneAssignedFace() = runBlocking {
        val store = ModelStore(context)
        val statuses = store.refreshStatuses()
        REQUIRED_MODELS.forEach { id ->
            assertTrue("${id.stableId} must be imported", statuses[id] is ModelStatus.Ready)
        }

        val raw = OnnxRawFaceSwapPipeline(store)
        val photo = OnnxPhotoFaceSwapPipeline(store, rawPipeline = raw)
        val enhancer = OnnxFaceEnhancerPipeline(store)
        val parser = OnnxFaceParserPipeline(store)
        val coordinator = OnnxMultiPhotoFaceSwapPipeline(raw, photo, enhancer, parser)

        val target = bitmap("inputs/stage_e_dense_pair_target.png")
        val source = bitmap("inputs/pair_03_source.png")
        val results = JSONArray()
        try {
            val detected = raw.detectFaces(target, RequestedInferenceBackend.CPU_ONLY).first
            val assigned = detected.sortedBy { (it.box.left + it.box.right) / 2.0 }.first()
            val sourceFace = raw.detectFaces(source, RequestedInferenceBackend.CPU_ONLY)
                .first
                .maxByOrNull(DetectedFace5::score)
                ?: error("source face was not detected")
            val targetId = FaceId("target-a")
            val sourceId = FaceId("source-1")

            for (preset in QualityPreset.entries) {
                val settings = preset.settings ?: continue
                var result: MultiPhotoFaceSwapResult? = null
                val started = SystemClock.elapsedRealtime()
                try {
                    result = coordinator.process(
                        target = target,
                        sources = listOf(MultiPhotoSource(sourceId, source, sourceFace.box)),
                        targetsInStableOrder = listOf(MultiPhotoTarget(targetId, assigned.box)),
                        assignments = listOf(MultiPhotoAssignment(targetId, sourceId)),
                        backend = RequestedInferenceBackend.CPU_ONLY,
                        restorationStrength = settings.effectiveRestorationStrength,
                        swapBlendMaskMode = if (settings.parserSwapMaskEnabled) {
                            SwapBlendMaskMode.PARSER_REGION
                        } else {
                            SwapBlendMaskMode.AFFINE_BOX
                        },
                    )
                } finally {
                    val elapsed = SystemClock.elapsedRealtime() - started
                    val timings = result?.timings
                    android.util.Log.i(
                        TAG,
                        "${preset.name} totalMs=$elapsed " +
                            "restoration=${settings.effectiveRestorationStrength} " +
                            "parserSwapMask=${settings.parserSwapMaskEnabled} " +
                            "swapperMs=${timings?.swapperMs} compositingMs=${timings?.compositingMs} " +
                            "enhancementMs=${result?.enhancementMs} swapParserMs=${result?.swapParserMs}",
                    )
                    results.put(
                        JSONObject()
                            .put("preset", preset.name)
                            .put("total_ms", elapsed)
                            .put("restoration_strength", settings.effectiveRestorationStrength.toDouble())
                            .put("parser_swap_mask", settings.parserSwapMaskEnabled)
                            .put("enhancement_ms", result?.enhancementMs ?: 0L)
                            .put("swap_parser_ms", result?.swapParserMs ?: 0L),
                    )
                    result?.finalBitmap?.let { if (!it.isRecycled) it.recycle() }
                }
            }

            val directory = File(context.filesDir, OUTPUT_DIRECTORY).apply { mkdirs() }
            File(directory, RESULTS_FILE).writeText(
                JSONObject()
                    .put("checkpoint", "E2 checkpoint 7: quality preset benchmark")
                    .put("device_api", 35)
                    .put("backend", "CPU")
                    .put("assigned_faces", 1)
                    .put("target", "stage_e_dense_pair_target.png")
                    .put("target_size", "${target.width}x${target.height}")
                    .put("presets", results)
                    .toString(2),
            )
            assertTrue("every preset must have been timed", results.length() == QualityPreset.entries.count { it.settings != null })
        } finally {
            if (!source.isRecycled) source.recycle()
            if (!target.isRecycled) target.recycle()
        }
    }

    private fun bitmap(assetPath: String): Bitmap =
        assets.open(assetPath).use { input -> BitmapFactory.decodeStream(input) }

    private companion object {
        const val TAG = "QualityPresetBenchmark"
        const val OUTPUT_DIRECTORY = "stage-e-checkpoint-7"
        const val RESULTS_FILE = "quality_preset_benchmark.json"
        val REQUIRED_MODELS = setOf(
            ModelId.YOLOFACE_8N,
            ModelId.ARCFACE_W600K_R50,
            ModelId.INSWAPPER_128_FP16,
            ModelId.GFPGAN_1_4,
            ModelId.BISENET_RESNET_34,
        )
    }
}
