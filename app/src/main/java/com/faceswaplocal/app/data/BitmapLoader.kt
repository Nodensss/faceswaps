package com.faceswaplocal.app.data

import android.content.ContentResolver
import android.graphics.Bitmap
import android.graphics.ImageDecoder
import android.net.Uri
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt
import kotlin.math.sqrt

/**
 * A decoded photo together with the size it was stored at, so the UI can tell the user
 * when the export will not match the original file (FR-PHOTO-02).
 */
data class DecodedImage(
    val bitmap: Bitmap,
    val sourceWidth: Int,
    val sourceHeight: Int,
) {
    val isFullResolution: Boolean
        get() = bitmap.width == sourceWidth && bitmap.height == sourceHeight
}

class BitmapLoader(
    private val contentResolver: ContentResolver,
    private val budget: ImageMemoryBudget,
) {
    /**
     * Decodes the target at its original resolution whenever the current memory budget
     * allows it, because compositing and export must keep the target's own size (§5.2).
     * The neural models still run on a small aligned crop, so the budget only has to
     * cover the full-frame compositing buffers.
     */
    suspend fun loadTarget(uri: Uri): DecodedImage = withContext(Dispatchers.IO) {
        val maxPixels = budget.maxTargetPixels()
        decode(uri) { width, height ->
            DecodeSizePolicy.targetSize(width, height, maxPixels)
        }
    }

    /** Only an aligned crop is taken from a source; see [ImageMemoryBudget.SOURCE_MAX_DIMENSION]. */
    suspend fun loadSource(uri: Uri): DecodedImage = withContext(Dispatchers.IO) {
        decode(uri) { width, height ->
            DecodeSizePolicy.longSideSize(width, height, ImageMemoryBudget.SOURCE_MAX_DIMENSION)
        }
    }

    private fun decode(uri: Uri, targetSize: (Int, Int) -> Pair<Int, Int>?): DecodedImage {
        var sourceWidth = 0
        var sourceHeight = 0
        val source = ImageDecoder.createSource(contentResolver, uri)
        val bitmap = ImageDecoder.decodeBitmap(source) { decoder, info, _ ->
            decoder.allocator = ImageDecoder.ALLOCATOR_SOFTWARE

            sourceWidth = info.size.width
            sourceHeight = info.size.height
            targetSize(sourceWidth, sourceHeight)?.let { (width, height) ->
                decoder.setTargetSize(width, height)
            }
        }
        return DecodedImage(bitmap, sourceWidth, sourceHeight)
    }
}

/** Pure decode-size arithmetic, kept free of Android types so unit tests can cover it. */
internal object DecodeSizePolicy {
    /** Null keeps the decoder's own full-resolution output. */
    fun targetSize(width: Int, height: Int, maxPixels: Int): Pair<Int, Int>? {
        val pixels = width.toLong() * height.toLong()
        if (pixels <= maxPixels) return null
        // Both sides are floored, so their product cannot exceed width*height*scale²,
        // which is the budget itself. Rounding each side independently could not
        // promise that: 4000x3000 into 2 000 000 px rounds up to 2 000 425.
        val scale = sqrt(maxPixels.toDouble() / pixels.toDouble())
        return Pair(
            floorSide(width, scale),
            floorSide(height, scale),
        )
    }

    /** A source is bounded per side rather than by a pixel product, so the long side lands exactly. */
    fun longSideSize(width: Int, height: Int, maxDimension: Int): Pair<Int, Int>? {
        val longestSide = max(width, height)
        if (longestSide <= maxDimension) return null
        val scale = maxDimension.toDouble() / longestSide.toDouble()
        return Pair(
            if (width >= height) maxDimension else roundedSide(width, scale),
            if (height >= width) maxDimension else roundedSide(height, scale),
        )
    }

    private fun floorSide(side: Int, scale: Double): Int =
        min(side, (side * scale).toInt().coerceAtLeast(1))

    private fun roundedSide(side: Int, scale: Double): Int =
        min(side, (side * scale).roundToInt().coerceAtLeast(1))
}
