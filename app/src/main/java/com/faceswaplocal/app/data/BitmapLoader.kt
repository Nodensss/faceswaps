package com.faceswaplocal.app.data

import android.content.ContentResolver
import android.graphics.Bitmap
import android.graphics.ImageDecoder
import android.net.Uri
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlin.math.max
import kotlin.math.roundToInt

class BitmapLoader(
    private val contentResolver: ContentResolver,
    private val maxDimension: Int = 2_560,
) {
    suspend fun load(uri: Uri): Bitmap = withContext(Dispatchers.IO) {
        val source = ImageDecoder.createSource(contentResolver, uri)
        ImageDecoder.decodeBitmap(source) { decoder, info, _ ->
            decoder.allocator = ImageDecoder.ALLOCATOR_SOFTWARE

            val width = info.size.width
            val height = info.size.height
            val longestSide = max(width, height)
            if (longestSide > maxDimension) {
                val scale = maxDimension.toFloat() / longestSide.toFloat()
                decoder.setTargetSize(
                    (width * scale).roundToInt().coerceAtLeast(1),
                    (height * scale).roundToInt().coerceAtLeast(1),
                )
            }
        }
    }
}

