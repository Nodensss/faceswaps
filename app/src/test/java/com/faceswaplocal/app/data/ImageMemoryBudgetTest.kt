package com.faceswaplocal.app.data

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class ImageMemoryBudgetTest {
    private val megabyte = 1024L * 1024L

    @Test
    fun `a roomy device is allowed a full 12 megapixel target`() {
        val pixels = ImageMemoryBudget.maxTargetPixels(
            freeHeapBytes = 512 * megabyte,
            availableSystemBytes = 3_000 * megabyte,
            lowMemory = false,
        )

        assertTrue(
            "4000x3000 must fit on a device with 512 MB of heap headroom, got $pixels",
            pixels >= 4_000 * 3_000,
        )
    }

    @Test
    fun `the smaller of heap and system headroom wins`() {
        val reserve = PipelinePass.peak().sessionReserveBytes
        val heapBound = ImageMemoryBudget.maxTargetPixels(
            freeHeapBytes = 64 * megabyte,
            availableSystemBytes = 4_000 * megabyte,
            lowMemory = false,
        )
        val systemBound = ImageMemoryBudget.maxTargetPixels(
            freeHeapBytes = 4_000 * megabyte,
            availableSystemBytes = 256 * megabyte + reserve,
            lowMemory = false,
        )

        assertEquals(
            (64 * megabyte * ImageMemoryBudget.HEAP_SAFETY_FRACTION /
                ImageMemoryBudget.JAVA_BYTES_PER_PIXEL).toInt(),
            heapBound,
        )
        assertEquals(
            (
                256 * megabyte * ImageMemoryBudget.SYSTEM_SAFETY_FRACTION /
                    (ImageMemoryBudget.JAVA_BYTES_PER_PIXEL + ImageMemoryBudget.NATIVE_BYTES_PER_PIXEL)
                ).toInt(),
            systemBound,
        )
    }

    @Test
    fun `budget scales with available memory instead of being a constant`() {
        val reserve = PipelinePass.peak().sessionReserveBytes
        val small = ImageMemoryBudget.maxTargetPixels(
            freeHeapBytes = 96 * megabyte,
            availableSystemBytes = 400 * megabyte + reserve,
            lowMemory = false,
        )
        val large = ImageMemoryBudget.maxTargetPixels(
            freeHeapBytes = 384 * megabyte,
            availableSystemBytes = 1_600 * megabyte + reserve,
            lowMemory = false,
        )

        assertTrue("a larger device must get a larger budget", large > small)
        assertEquals(
            "above the reserve the budget must stay linear in the headroom",
            4.0,
            large.toDouble() / small,
            0.01,
        )
    }

    @Test
    fun `the session reserve comes off the top before the safety fraction`() {
        val reserve = PipelinePass.SWAP.sessionReserveBytes
        val pixels = ImageMemoryBudget.maxTargetPixels(
            freeHeapBytes = 64L * 1024L * megabyte,
            availableSystemBytes = 1_000 * megabyte + reserve,
            lowMemory = false,
            sessionReserveBytes = reserve,
        )

        assertEquals(
            (
                1_000 * megabyte * ImageMemoryBudget.SYSTEM_SAFETY_FRACTION /
                    (ImageMemoryBudget.JAVA_BYTES_PER_PIXEL + ImageMemoryBudget.NATIVE_BYTES_PER_PIXEL)
                ).toInt(),
            pixels,
        )
    }

    @Test
    fun `a device that cannot even hold the sessions falls back to the minimum`() {
        val pixels = ImageMemoryBudget.maxTargetPixels(
            freeHeapBytes = 64L * 1024L * megabyte,
            availableSystemBytes = PipelinePass.SWAP.sessionReserveBytes / 2,
            lowMemory = false,
        )

        assertEquals(ImageMemoryBudget.MIN_TARGET_PIXELS, pixels)
    }

    @Test
    fun `the swap pass is the more expensive one and sets the decode budget`() {
        assertTrue(
            "InSwapper outweighs GFPGAN once resident, so SWAP must be the peak",
            PipelinePass.SWAP.sessionReserveBytes > PipelinePass.RESTORE.sessionReserveBytes,
        )
        assertEquals(PipelinePass.SWAP, PipelinePass.peak())

        val swapBudget = ImageMemoryBudget.maxTargetPixels(
            freeHeapBytes = 64L * 1024L * megabyte,
            availableSystemBytes = 3_000 * megabyte,
            lowMemory = false,
            sessionReserveBytes = PipelinePass.SWAP.sessionReserveBytes,
        )
        val restoreBudget = ImageMemoryBudget.maxTargetPixels(
            freeHeapBytes = 64L * 1024L * megabyte,
            availableSystemBytes = 3_000 * megabyte,
            lowMemory = false,
            sessionReserveBytes = PipelinePass.RESTORE.sessionReserveBytes,
        )

        assertTrue(
            "restoration frees room, so its budget must be the larger of the two",
            restoreBudget > swapBudget,
        )
    }

    @Test
    fun `low memory halves the budget`() {
        val normal = ImageMemoryBudget.maxTargetPixels(
            freeHeapBytes = 512 * megabyte,
            availableSystemBytes = 3_000 * megabyte,
            lowMemory = false,
        )
        val low = ImageMemoryBudget.maxTargetPixels(
            freeHeapBytes = 512 * megabyte,
            availableSystemBytes = 3_000 * megabyte,
            lowMemory = true,
        )

        assertEquals(normal / 2, low)
    }

    @Test
    fun `a starved device still gets the minimum instead of a thumbnail`() {
        val pixels = ImageMemoryBudget.maxTargetPixels(
            freeHeapBytes = 1 * megabyte,
            availableSystemBytes = 0,
            lowMemory = true,
        )

        assertEquals(ImageMemoryBudget.MIN_TARGET_PIXELS, pixels)
    }

    @Test
    fun `an absurd amount of memory is still capped`() {
        val pixels = ImageMemoryBudget.maxTargetPixels(
            freeHeapBytes = 64L * 1024L * megabyte,
            availableSystemBytes = 64L * 1024L * megabyte,
            lowMemory = false,
        )

        assertEquals(ImageMemoryBudget.MAX_TARGET_PIXELS, pixels)
    }

    @Test
    fun `free heap counts memory the process may still grow into`() {
        val runtime = FakeRuntime(maxMemory = 512 * megabyte, total = 64 * megabyte, free = 16 * megabyte)
        val budget = ImageMemoryBudget(
            runtime = runtime,
            systemMemory = { ImageMemoryBudget.SystemMemory(3_000 * megabyte, lowMemory = false) },
        )

        // 512 MB ceiling minus the 48 MB actually in use, not 16 MB of unused heap.
        assertEquals(
            ImageMemoryBudget.maxTargetPixels(
                freeHeapBytes = 464 * megabyte,
                availableSystemBytes = 3_000 * megabyte,
                lowMemory = false,
            ),
            budget.maxTargetPixels(),
        )
    }

    /**
     * `Runtime` is final, so the fake only has to answer the three accessors the budget
     * reads; anything else would fail loudly rather than silently return zero.
     */
    private class FakeRuntime(
        private val maxMemory: Long,
        private val total: Long,
        private val free: Long,
    ) : RuntimeMemory {
        override fun maxMemory(): Long = maxMemory
        override fun totalMemory(): Long = total
        override fun freeMemory(): Long = free
    }
}
