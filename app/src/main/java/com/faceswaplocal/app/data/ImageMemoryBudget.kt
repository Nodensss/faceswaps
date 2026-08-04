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
 * Decides how many pixels of a target photo the device can still carry through the whole
 * photo pipeline, so the limit follows real memory instead of a hardcoded edge length
 * (§5.2, §12).
 *
 * The multipliers below are a census of the full-frame buffers that exist at the same
 * time during one assigned face, taken from the current pipeline:
 *
 * Java heap (`IntArray`/`FloatArray`, counted against `Runtime.maxMemory`)
 *  1. the accumulated composite the coordinator carries between faces;
 *  2. `FaceCompositor.pasteBack`'s `basePixels.copyOf()` result;
 *  3. its full-frame `warpedMask` (`FloatArray`, same 4 bytes per pixel);
 *  4. the coordinator's `readPixels()` copy of the freshly produced bitmap.
 *
 * `BitmapSampling.warpAffine` and the detector input also read the whole frame into an
 * array, but both are released before the compositing buffers above coexist, so they
 * raise the transient floor rather than the peak.
 *
 * Native heap (bitmap pixels, since API 26 outside the Java heap)
 *  5. the decoded target itself, retained by the UI for the before/after view;
 *  6. the previous working bitmap, alive until the new one replaces it;
 *  7. the newly created result bitmap.
 *
 * The safety fractions leave room for the ONNX Runtime arenas, the aligned crops and the
 * Compose texture, none of which scale with the target size.
 */
class ImageMemoryBudget(
    private val runtime: RuntimeMemory = PlatformRuntimeMemory,
    private val systemMemory: () -> SystemMemory,
) {
    data class SystemMemory(val availableBytes: Long, val lowMemory: Boolean)

    /** Free Java heap the process may still grow into, not merely what is unused now. */
    private fun freeHeapBytes(): Long =
        runtime.maxMemory() - (runtime.totalMemory() - runtime.freeMemory())

    fun maxTargetPixels(): Int {
        val memory = systemMemory()
        return maxTargetPixels(
            freeHeapBytes = freeHeapBytes(),
            availableSystemBytes = memory.availableBytes,
            lowMemory = memory.lowMemory,
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
        internal const val JAVA_BYTES_PER_PIXEL = 4 * 4

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
        ): Int {
            val heapPixels =
                (freeHeapBytes * HEAP_SAFETY_FRACTION / JAVA_BYTES_PER_PIXEL).toLong()
            val systemPixels = (
                availableSystemBytes * SYSTEM_SAFETY_FRACTION /
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
