package com.faceswaplocal.app.data

import android.app.ActivityManager
import android.content.Context
import androidx.annotation.VisibleForTesting

/** `Runtime` is final, so the heap accessors are read through this seam in tests. */
interface RuntimeMemory {
    fun maxMemory(): Long

    fun totalMemory(): Long

    fun freeMemory(): Long
}

internal object PlatformRuntimeMemory : RuntimeMemory {
    override fun maxMemory(): Long = Runtime.getRuntime().maxMemory()

    override fun totalMemory(): Long = Runtime.getRuntime().totalMemory()

    override fun freeMemory(): Long = Runtime.getRuntime().freeMemory()
}

/**
 * The two halves of the photo pipeline, with the resident cost of the sessions each one
 * holds open. The coordinator's barrier guarantees InSwapper and GFPGAN never coexist,
 * while the BiSeNet parser session spans both, so each pass is `parser + its own model`.
 *
 * Figures are `VmRSS` deltas measured one session at a time by
 * `SessionFootprintInstrumentedTest` on AVD API 35, not file sizes. The gap matters:
 * InSwapper ships as 265 MiB of fp16 and occupies 507 MiB once open, because the CPU
 * execution provider materialises fp32 weights.
 */
enum class PipelinePass(val sessionReserveBytes: Long) {
    SWAP(BISENET_RESIDENT_BYTES + INSWAPPER_RESIDENT_BYTES),
    RESTORE(BISENET_RESIDENT_BYTES + GFPGAN_RESIDENT_BYTES),
    ;

    companion object {
        /**
         * A photo has to survive every pass it will be put through, so the decode is
         * sized against the most expensive one rather than the first.
         */
        fun peak(): PipelinePass = entries.maxBy(PipelinePass::sessionReserveBytes)
    }
}

private const val KIB = 1_024L
internal const val BISENET_RESIDENT_BYTES = 93_376L * KIB
internal const val INSWAPPER_RESIDENT_BYTES = 519_408L * KIB
internal const val GFPGAN_RESIDENT_BYTES = 335_376L * KIB

/**
 * Decides how many pixels of a target photo the device can still carry through the whole
 * photo pipeline, so the limit follows real memory instead of a hardcoded edge length
 * (§5.2, §12).
 *
 * The multipliers below are a census of the full-frame buffers that exist at the same
 * time during one assigned face, taken from the current pipeline:
 *
 * Java heap (`IntArray`, counted against `Runtime.maxMemory`)
 *  1. the accumulated composite the coordinator carries between faces;
 *  2. `FaceCompositor.pasteBack`'s `basePixels.copyOf()` result;
 *  3. the coordinator's `readPixels()` copy of the freshly produced bitmap.
 *
 * A fourth buffer used to sit here: `pasteBack`'s full-frame `warpedMask`. It was
 * allocated on every paste and never read outside tests, so it is now opt-in and the
 * Java multiplier dropped from four buffers to three.
 *
 * `BitmapSampling.warpAffine` and the detector input also read the whole frame into an
 * array, but both are released before the compositing buffers above coexist, so they
 * raise the transient floor rather than the peak.
 *
 * Native heap (bitmap pixels, since API 26 outside the Java heap)
 *  4. the decoded target itself, retained by the UI for the before/after view;
 *  5. the previous working bitmap, alive until the new one replaces it;
 *  6. the newly created result bitmap.
 *
 * Those per-pixel costs are only half the picture. The inference sessions do not scale
 * with the target, but they are large and they are resident while the full-frame buffers
 * above are alive, so they must be reserved rather than absorbed by a safety fraction —
 * that omission is what let a 16 MP photo pass the budget and then get OOM-killed twice
 * (see `docs/reports/STAGE_E2_MANUAL_ACCEPTANCE_REPORT.md`). [PipelinePass] carries the
 * measured resident cost of each pass.
 *
 * The safety fractions now cover only what is left unmodelled: the aligned crops, the
 * Compose texture and allocator slack, none of which scale with the target size.
 */
class ImageMemoryBudget(
    private val runtime: RuntimeMemory = PlatformRuntimeMemory,
    private val systemMemory: () -> SystemMemory,
) {
    data class SystemMemory(val availableBytes: Long, val lowMemory: Boolean)

    /** Free Java heap the process may still grow into, not merely what is unused now. */
    private fun freeHeapBytes(): Long =
        runtime.maxMemory() - (runtime.totalMemory() - runtime.freeMemory())

    /**
     * [pass] defaults to the most expensive pass because a decoded target must survive
     * all of them. Pass an explicit value to re-evaluate at a pass boundary, where the
     * resident set differs: the swap pass costs roughly 180 MiB more than restoration.
     */
    @JvmOverloads
    fun maxTargetPixels(pass: PipelinePass = PipelinePass.peak()): Int {
        val memory = systemMemory()
        return maxTargetPixels(
            freeHeapBytes = freeHeapBytes(),
            availableSystemBytes = memory.availableBytes,
            lowMemory = memory.lowMemory,
            sessionReserveBytes = pass.sessionReserveBytes,
        )
    }

    companion object {
        /**
         * Long-side cap for source photos. Only an aligned 112×112 crop is ever taken
         * from a source, up to eight of them can be decoded at once, and every parity
         * fixture of stages B–E1 was measured through this exact limit — so raising it
         * would cost memory without changing any result.
         */
        const val SOURCE_MAX_DIMENSION = 2_560

        @VisibleForTesting
        internal const val JAVA_BYTES_PER_PIXEL = 3 * 4

        @VisibleForTesting
        internal const val NATIVE_BYTES_PER_PIXEL = 3 * 4

        @VisibleForTesting
        internal const val HEAP_SAFETY_FRACTION = 0.60

        @VisibleForTesting
        internal const val SYSTEM_SAFETY_FRACTION = 0.50

        /**
         * Below this the result stops being a photo, so a device that cannot afford even
         * this decodes at it anyway and is allowed to fail loudly instead of silently
         * returning a thumbnail.
         */
        @VisibleForTesting
        internal const val MIN_TARGET_PIXELS = 1_280 * 960

        /** 64 MP is past any phone sensor; it only stops absurd inputs. */
        @VisibleForTesting
        internal const val MAX_TARGET_PIXELS = 64_000_000

        /** Applied on top of the computed budget while the system reports low memory. */
        @VisibleForTesting
        internal const val LOW_MEMORY_FRACTION = 0.5

        fun forContext(context: Context): ImageMemoryBudget {
            val activityManager =
                context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
            return ImageMemoryBudget(
                systemMemory = {
                    val info = ActivityManager.MemoryInfo()
                    activityManager.getMemoryInfo(info)
                    SystemMemory(
                        availableBytes = (info.availMem - info.threshold).coerceAtLeast(0L),
                        lowMemory = info.lowMemory,
                    )
                },
            )
        }

        @VisibleForTesting
        internal fun maxTargetPixels(
            freeHeapBytes: Long,
            availableSystemBytes: Long,
            lowMemory: Boolean,
            sessionReserveBytes: Long = PipelinePass.peak().sessionReserveBytes,
        ): Int {
            val heapPixels =
                (freeHeapBytes * HEAP_SAFETY_FRACTION / JAVA_BYTES_PER_PIXEL).toLong()
            // The reserve comes off the top: it is a known, already-quantified allocation,
            // so it must not be scaled by a fraction meant for unmodelled slack. ONNX
            // weights are native, so only the system term is charged for them.
            val systemBytesForPixels =
                (availableSystemBytes - sessionReserveBytes).coerceAtLeast(0L)
            val systemPixels = (
                systemBytesForPixels * SYSTEM_SAFETY_FRACTION /
                    (JAVA_BYTES_PER_PIXEL + NATIVE_BYTES_PER_PIXEL)
                ).toLong()
            val affordable = minOf(heapPixels, systemPixels)
                .let { pixels -> if (lowMemory) (pixels * LOW_MEMORY_FRACTION).toLong() else pixels }
            return affordable
                .coerceIn(MIN_TARGET_PIXELS.toLong(), MAX_TARGET_PIXELS.toLong())
                .toInt()
        }
    }
}
