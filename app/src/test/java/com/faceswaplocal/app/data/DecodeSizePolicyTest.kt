package com.faceswaplocal.app.data

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class DecodeSizePolicyTest {
    @Test
    fun `a target within the budget is decoded untouched`() {
        assertNull(
            "12 MP under a 16 MP budget must keep the decoder's own size",
            DecodeSizePolicy.targetSize(4_000, 3_000, maxPixels = 16_000_000),
        )
        assertNull(DecodeSizePolicy.targetSize(4_000, 3_000, maxPixels = 12_000_000))
    }

    @Test
    fun `an oversized target is scaled to fit the budget and keeps its aspect ratio`() {
        val (width, height) = DecodeSizePolicy.targetSize(4_000, 3_000, maxPixels = 3_000_000)!!

        assertTrue("$width x $height must fit the budget", width.toLong() * height <= 3_000_000)
        assertEquals(4.0 / 3.0, width.toDouble() / height.toDouble(), 0.01)
        assertTrue(width < 4_000 && height < 3_000)
    }

    /** Rounding each side independently used to overshoot 2 000 000 by 425 pixels. */
    @Test
    fun `the pixel budget is a hard ceiling, not an approximation`() {
        val sizes = listOf(
            4_000 to 3_000,
            4_032 to 3_024,
            5_000 to 2_813,
            3_000 to 4_000,
            1_920 to 1_080,
        )
        val budgets = listOf(2_000_000, 3_000_000, 5_000_000, 8_000_000, 12_000_000)

        sizes.forEach { (width, height) ->
            budgets.forEach { budget ->
                val scaled = DecodeSizePolicy.targetSize(width, height, budget) ?: return@forEach
                val pixels = scaled.first.toLong() * scaled.second
                assertTrue(
                    "$width x $height into $budget produced $scaled = $pixels",
                    pixels <= budget,
                )
            }
        }
    }

    @Test
    fun `scaling never grows a side`() {
        listOf(1 to 1, 3 to 5_000, 4_001 to 3, 4_000 to 3_000).forEach { (width, height) ->
            val scaled = DecodeSizePolicy.targetSize(width, height, maxPixels = 1_000)
            if (scaled != null) {
                assertTrue("$scaled must not exceed $width x $height", scaled.first <= width)
                assertTrue("$scaled must not exceed $width x $height", scaled.second <= height)
                assertTrue(scaled.first >= 1 && scaled.second >= 1)
            }
        }
    }

    @Test
    fun `sources are limited by their long side, not by a pixel budget`() {
        assertNull(DecodeSizePolicy.longSideSize(2_560, 1_440, maxDimension = 2_560))

        val (width, height) = DecodeSizePolicy.longSideSize(4_000, 3_000, maxDimension = 2_560)!!
        assertEquals(2_560, width)
        assertEquals(1_920, height)

        val (portraitWidth, portraitHeight) =
            DecodeSizePolicy.longSideSize(3_000, 4_000, maxDimension = 2_560)!!
        assertEquals(1_920, portraitWidth)
        assertEquals(2_560, portraitHeight)
    }

    @Test
    fun `the removed 2560 constant no longer limits targets`() {
        assertNull(
            "a 4000 px target must survive when the budget allows it",
            DecodeSizePolicy.targetSize(4_000, 3_000, maxPixels = ImageMemoryBudget.MAX_TARGET_PIXELS),
        )
    }
}
