package com.faceswaplocal.app.inference

import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import java.io.File
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

/**
 * Checkpoint 3, measurement 1: what each ONNX session actually costs while it is open.
 *
 * The 16 MP OOM report reasoned from the sum of all six weight files (857 MiB), which
 * overstates the peak because the pipeline never holds them all at once. The budget has
 * to be built on the resident cost of the sessions that are open *at the same time*, so
 * this measures one session at a time against the same metric the kernel used to pick
 * the OOM victim: `VmRSS` from `/proc/self/status`.
 *
 * It asserts almost nothing on purpose - it is a measurement whose output feeds
 * [ImageMemoryBudget.SESSION_RESERVE_BYTES]. The one assertion guards the method itself.
 */
@RunWith(AndroidJUnit4::class)
class SessionFootprintInstrumentedTest {
    private val context = ApplicationProvider.getApplicationContext<Context>()

    @Test
    fun eachSessionsResidentCostIsMeasuredIndividually() = runBlocking {
        val store = ModelStore(context)
        val statuses = store.refreshStatuses()
        val environment = OrtEnvironment.getEnvironment()

        val measured = linkedMapOf<ModelId, Long>()
        for (id in MEASURED_MODELS) {
            if (statuses[id] !is ModelStatus.Ready) {
                android.util.Log.w(TAG, "${id.stableId} is not imported, skipped")
                continue
            }
            val file = store.requireVerifiedModel(id)
            // Settle first: a previous iteration's arena is returned lazily.
            settle()
            val before = residentKb()
            var openDelta = 0L
            createOptions().use { options ->
                val session = environment.createSession(file.absolutePath, options)
                try {
                    openDelta = residentKb() - before
                } finally {
                    session.close()
                }
            }
            settle()
            val afterClose = residentKb() - before
            measured[id] = openDelta
            android.util.Log.i(
                TAG,
                "${id.stableId} fileBytes=${file.length()} " +
                    "openDeltaKb=$openDelta retainedAfterCloseKb=$afterClose",
            )
        }

        val total = measured.values.sum()
        android.util.Log.i(TAG, "sum of individually measured deltas = $total kB")
        // Pass 1 holds parser + swapper; pass 2 holds parser + enhancer. Detector and
        // recognizer are sequential and never coexist with a swapper (verified by
        // SessionCoexistenceInstrumentedTest), so these two are the peaks that matter.
        val parser = measured[ModelId.BISENET_RESNET_34] ?: 0L
        val swapPeak = parser + (measured[ModelId.INSWAPPER_128_FP16] ?: 0L)
        val restorePeak = parser + (measured[ModelId.GFPGAN_1_4] ?: 0L)
        android.util.Log.i(
            TAG,
            "peak pass1(parser+swapper)=$swapPeak kB peak pass2(parser+enhancer)=$restorePeak kB",
        )

        assertTrue(
            "at least one session must have been measured",
            measured.isNotEmpty(),
        )
    }

    private fun createOptions(): OrtSession.SessionOptions =
        OrtSession.SessionOptions().apply {
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
            setInterOpNumThreads(1)
            addConfigEntry("session.intra_op.allow_spinning", "0")
            setIntraOpNumThreads(1)
        }

    /** Anonymous+file resident set of this process, the metric the OOM killer ranks on. */
    private fun residentKb(): Long =
        File("/proc/self/status").readLines()
            .firstOrNull { it.startsWith("VmRSS:") }
            ?.filter(Char::isDigit)
            ?.toLongOrNull()
            ?: error("VmRSS is not readable")

    private fun settle() {
        System.gc()
        System.runFinalization()
        Thread.sleep(SETTLE_MS)
    }

    companion object {
        const val TAG = "SessionFootprint"
        const val SETTLE_MS = 700L
        val MEASURED_MODELS = listOf(
            ModelId.YOLOFACE_8N,
            ModelId.ARCFACE_W600K_R50,
            ModelId.INSWAPPER_128_FP16,
            ModelId.GFPGAN_1_4,
            ModelId.BISENET_RESNET_34,
        )
    }
}
