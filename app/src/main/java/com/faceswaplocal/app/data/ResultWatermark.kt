package com.faceswaplocal.app.data

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Typeface
import kotlin.math.max
import kotlin.math.min

/**
 * Visible edit mark required by §5.3. It is a separate setting, on by default, and is
 * drawn only into the exported copy — the in-memory result the user compares stays clean.
 */
internal object ResultWatermark {
    const val TEXT = "FaceSwapLocal"

    /** Text height scales with the image so a 12 MP export is marked like a preview. */
    internal fun textSizePx(width: Int, height: Int): Float =
        max(MIN_TEXT_PX, min(width, height) * TEXT_SIZE_RATIO)

    /**
     * Returns a new watermarked ARGB_8888 bitmap with the same dimensions; the caller
     * owns and must recycle it. The input bitmap is never mutated, which is why an
     * immutable decoded result can be exported safely.
     */
    fun render(source: Bitmap): Bitmap {
        val marked = source.copy(Bitmap.Config.ARGB_8888, true)
            ?: throw IllegalStateException("Watermark copy could not be allocated")
        try {
            draw(Canvas(marked), marked.width, marked.height)
        } catch (error: Throwable) {
            if (!marked.isRecycled) marked.recycle()
            throw error
        }
        return marked
    }

    internal fun draw(canvas: Canvas, width: Int, height: Int) {
        val textSize = textSizePx(width, height)
        val margin = textSize * MARGIN_RATIO
        val fill = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            this.textSize = textSize
            color = Color.WHITE
            alpha = FILL_ALPHA
            textAlign = Paint.Align.RIGHT
            typeface = Typeface.create(Typeface.DEFAULT, Typeface.BOLD)
        }
        val outline = Paint(fill).apply {
            style = Paint.Style.STROKE
            strokeWidth = max(1f, textSize * OUTLINE_RATIO)
            color = Color.BLACK
            alpha = OUTLINE_ALPHA
        }

        val x = width - margin
        val y = height - margin - fill.descent()
        canvas.drawText(TEXT, x, y, outline)
        canvas.drawText(TEXT, x, y, fill)
    }

    private const val TEXT_SIZE_RATIO = 0.028f
    private const val MIN_TEXT_PX = 14f
    private const val MARGIN_RATIO = 0.8f
    private const val OUTLINE_RATIO = 0.12f
    private const val FILL_ALPHA = 190
    private const val OUTLINE_ALPHA = 130
}
