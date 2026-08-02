package com.faceswaplocal.app.inference

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.faceswaplocal.app.domain.FaceId
import java.io.File
import java.io.FileOutputStream
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class StageDMultiFaceInstrumentedTest {
    private val context = ApplicationProvider.getApplicationContext<Context>()
    private val assets = InstrumentationRegistry.getInstrumentation().context.assets

    /**
     * Drives the production coordinator [OnnxMultiPhotoFaceSwapPipeline.process] end to
     * end (detection, stable target order, accumulation, per-source embedding cache) and
     * asserts its spatial guarantees on the real fixture: three different sources land on
     * T1/T2/T3, T4 is left unchanged, nothing outside the union of the reported paste ROIs
     * moves, and the second paste wins the T1/T2 overlap.
     */
    @Test
    fun coordinatorReplacesThreeSourcesLeavingT4AndOutsideUnionUntouched() {
        runBlocking {
        val store = ModelStore(context)
        store.refreshStatuses()
        val raw = OnnxRawFaceSwapPipeline(store)
        val photo = OnnxPhotoFaceSwapPipeline(store, rawPipeline = raw)
        val coordinator = OnnxMultiPhotoFaceSwapPipeline(raw, photo)

        val target = bitmap("inputs/stage_d_group_target.png")
        val sourceBitmaps = (1..3).map { bitmap("inputs/pair_%02d_source.png".format(it)) }
        val original = target.pixels()

        val detected = raw.detectFaces(target, RequestedInferenceBackend.CPU_ONLY).first
        assertEquals("fixture must contain exactly four neural faces", 4, detected.size)
        val ordered = detected.sortedBy { (it.box.left + it.box.right) / 2.0 }.let { byX ->
            val right = byX.takeLast(2).sortedBy { (it.box.top + it.box.bottom) / 2.0 }
            listOf(byX[0], byX[1], right[0], right[1])
        }

        val targetIds = (1..4).map { FaceId("target-$it") }
        val sourceIds = (1..3).map { FaceId("source-$it") }
        val sources = sourceBitmaps.mapIndexed { index, bmp ->
            MultiPhotoSource(sourceIds[index], bmp, FaceBox(0.0, 0.0, bmp.width.toDouble(), bmp.height.toDouble()))
        }
        val targets = ordered.mapIndexed { index, face -> MultiPhotoTarget(targetIds[index], face.box) }
        // T1<-source-1, T2<-source-2, T3<-source-3; T4 stays unassigned ("Не менять").
        val assignments = (0..2).map { MultiPhotoAssignment(targetIds[it], sourceIds[it]) }

        var result: MultiPhotoFaceSwapResult? = null
        try {
            result = coordinator.process(
                target = target,
                sources = sources,
                targetsInStableOrder = targets,
                assignments = assignments,
                backend = RequestedInferenceBackend.CPU_ONLY,
                swapBlendMaskMode = SwapBlendMaskMode.AFFINE_BOX,
            )
            val swap = requireNotNull(result) { "coordinator returned no result" }
            val finalPixels = swap.finalBitmap.pixels()
            val rois = swap.pasteRois
            assertEquals("three assigned targets must yield three paste ROIs", 3, rois.size)

            for (index in 0..2) {
                assertTrue("T${index + 1} must change", changedInBox(original, finalPixels, target.width, ordered[index].box) > 100)
            }
            assertEquals("T4 must remain bit-identical", 0, changedInBox(original, finalPixels, target.width, ordered[3].box))
            assertEquals("pixels outside union of paste ROIs", 0, outsideUnionChanges(original, finalPixels, target.width, target.height, rois))
            val overlap = intersect(rois[0], rois[1])
            assertTrue("T1/T2 paste ROIs must overlap", overlap.width > 0 && overlap.height > 0)
            assertTrue("overlap must contain the second paste, not original", changedInRoi(original, finalPixels, target.width, overlap) > 20)

            val output = File(context.filesDir, "stage-d-output").apply { mkdirs() }
            FileOutputStream(File(output, "STAGE_D_MULTI_FACE_RESULT.png")).use {
                swap.finalBitmap.compress(Bitmap.CompressFormat.PNG, 100, it)
            }
        } finally {
            result?.finalBitmap?.recycleSafely()
            sourceBitmaps.forEach(Bitmap::recycle)
            target.recycle()
        }
        }
    }

    private fun bitmap(path: String) = assets.open(path).use { requireNotNull(BitmapFactory.decodeStream(it)).copy(Bitmap.Config.ARGB_8888, false) }
    private fun Bitmap.pixels() = IntArray(width * height).also { getPixels(it, 0, width, 0, 0, width, height) }
    private fun Bitmap.recycleSafely() { if (!isRecycled) recycle() }
    private fun changedInBox(a: IntArray, b: IntArray, width: Int, box: FaceBox): Int = changedInRoi(a, b, width, CompositeRoi(box.left.toInt().coerceAtLeast(0), box.top.toInt().coerceAtLeast(0), box.right.toInt(), box.bottom.toInt()))
    private fun changedInRoi(a: IntArray, b: IntArray, width: Int, roi: CompositeRoi): Int = (roi.top until roi.bottom).sumOf { y -> (roi.left until roi.right).count { x -> a[y * width + x] != b[y * width + x] } }
    private fun outsideUnionChanges(a: IntArray, b: IntArray, width: Int, height: Int, rois: List<CompositeRoi>): Int = (0 until height).sumOf { y -> (0 until width).count { x -> rois.none { x in it.left until it.right && y in it.top until it.bottom } && a[y * width + x] != b[y * width + x] } }
    private fun intersect(a: CompositeRoi, b: CompositeRoi) = CompositeRoi(maxOf(a.left, b.left), maxOf(a.top, b.top), minOf(a.right, b.right), minOf(a.bottom, b.bottom))
}
