package com.faceswaplocal.app.inference

import kotlin.math.abs
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class FaceRegionMaskTest {
    @Test
    fun `parser tensor uses ImageNet RGB normalization after resize`() {
        val tensor = FaceRegionMask.rgbTensor(
            cropPixels = intArrayOf(argb(255, 64, 128, 192)),
            cropWidth = 1,
            cropHeight = 1,
        )
        val plane = 512 * 512

        assertEquals((64f / 255f - 0.485f) / 0.229f, tensor[0], 1e-6f)
        assertEquals((128f / 255f - 0.456f) / 0.224f, tensor[plane], 1e-6f)
        assertEquals((192f / 255f - 0.406f) / 0.225f, tensor[2 * plane], 1e-6f)
        assertEquals(tensor[0], tensor[plane - 1], 0f)
    }

    @Test
    fun `region mask keeps face classes and excludes hair and neck`() {
        val width = 64
        val classes = IntArray(width * width) { HAIR_CLASS }
        for (y in 4 until 32) {
            for (x in 12 until 52) classes[y * width + x] = SKIN_CLASS
        }
        for (y in 32 until 60) {
            for (x in 12 until 52) classes[y * width + x] = NECK_CLASS
        }

        val mask = FaceRegionMask.fromClasses(classes, width, width, width, width)

        assertTrue(mask.all { it.isFinite() && it in 0f..1f })
        assertEquals(0f, mask[index(2, 2, width)], 0f)
        assertTrue(mask[index(32, 16, width)] > 0.9f)
        assertEquals(0f, mask[index(32, 52, width)], 0f)
    }

    @Test
    fun `FaceFusion order resizes classes before applying sigma five blur`() {
        val sourceSize = 512
        val outputSize = 128
        val classes = IntArray(sourceSize * sourceSize)
        for (y in 128 until 384) {
            for (x in 128 until 384) classes[y * sourceSize + x] = SKIN_CLASS
        }

        val mask = FaceRegionMask.fromClasses(
            classes,
            sourceSize,
            sourceSize,
            outputSize,
            outputSize,
        )

        assertEquals(0f, mask[index(4, 64, outputSize)], 0f)
        assertTrue(mask[index(64, 64, outputSize)] > 0.99f)
        val firstSupported = (0 until outputSize).first { x ->
            mask[index(x, 64, outputSize)] > 0.01f
        }
        val firstNearOpaque = (0 until outputSize).first { x ->
            mask[index(x, 64, outputSize)] > 0.99f
        }
        // The 0.5 clip keeps support at the resized class boundary (x=32), while a
        // sigma-5 blur in the 128 crop needs roughly 11 further pixels to become opaque.
        // Blurring at 512 before downsampling would make this transition ~4x narrower.
        assertTrue("unexpected resized boundary: x=$firstSupported", abs(firstSupported - 32) <= 1)
        assertTrue("blur appears to run before resize: x=$firstNearOpaque", firstNearOpaque in 41..46)
    }

    private fun index(x: Int, y: Int, width: Int): Int = y * width + x

    private fun argb(alpha: Int, red: Int, green: Int, blue: Int): Int =
        (alpha shl 24) or (red shl 16) or (green shl 8) or blue

    private companion object {
        const val SKIN_CLASS = 1
        const val NECK_CLASS = 14
        const val HAIR_CLASS = 17
    }
}
