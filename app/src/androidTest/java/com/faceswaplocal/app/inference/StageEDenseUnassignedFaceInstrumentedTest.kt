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
import kotlin.math.ceil
import kotlin.math.floor
import kotlinx.coroutines.runBlocking
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

/**
 * "Do not change" must be structural, not a property of a roomy layout.
 *
 * `stage_e_dense_pair_target.png` places an unassigned face (B) right next to an
 * assigned one (A), reusing the Stage-D T1/T2 spacing whose paste ROIs are already
 * proven to overlap. Before unassigned regions were protected in the swap pass, the
 * strength-zero case here failed: face A pasted across face B.
 */
@RunWith(AndroidJUnit4::class)
class StageEDenseUnassignedFaceInstrumentedTest {
    private val context = ApplicationProvider.getApplicationContext<Context>()
    private val assets = InstrumentationRegistry.getInstrumentation().context.assets

    /** Isolates the swap pass: no restoration runs, so only pass 1 can touch face B. */
    @Test
    fun swapPassLeavesTheAdjacentUnassignedFaceBitIdentical() {
        runDenseCase(label = "swap_only", strength = 0f)
    }

    /** Both passes together, with the production parser blend mask. */
    @Test
    fun swapAndRestorationLeaveTheAdjacentUnassignedFaceBitIdentical() {
        runDenseCase(label = "swap_and_restore", strength = 0.8f)
    }

    private fun runDenseCase(label: String, strength: Float) = runBlocking {
        val store = ModelStore(context)
        val statuses = store.refreshStatuses()
        requiredModels(strength).forEach { id ->
            assertTrue("${id.stableId} must be imported", statuses[id] is ModelStatus.Ready)
        }
        val raw = OnnxRawFaceSwapPipeline(store)
        val photo = OnnxPhotoFaceSwapPipeline(store, rawPipeline = raw)
        val enhancer = OnnxFaceEnhancerPipeline(store)
        val parser = OnnxFaceParserPipeline(store)
        val coordinator = OnnxMultiPhotoFaceSwapPipeline(raw, photo, enhancer, parser)

        val target = bitmap("inputs/stage_e_dense_pair_target.png")
        val source = bitmap("inputs/pair_03_source.png")
        val originalPixels = target.pixels()
        var result: MultiPhotoFaceSwapResult? = null
        try {
            val detected = raw.detectFaces(target, RequestedInferenceBackend.CPU_ONLY).first
            assertEquals("dense fixture must contain exactly two neural faces", 2, detected.size)
            val ordered = detected.sortedBy { (it.box.left + it.box.right) / 2.0 }
            val assignedFace = ordered[0]
            val unassignedFace = ordered[1]

            val assignedId = FaceId("target-a")
            val unassignedId = FaceId("target-b")
            val sourceId = FaceId("source-1")
            val sourceFace = raw.detectFaces(source, RequestedInferenceBackend.CPU_ONLY)
                .first
                .maxByOrNull(DetectedFace5::score)
                ?: error("source face was not detected")

            result = coordinator.process(
                target = target,
                sources = listOf(MultiPhotoSource(sourceId, source, sourceFace.box)),
                targetsInStableOrder = listOf(
                    MultiPhotoTarget(assignedId, assignedFace.box),
                    MultiPhotoTarget(unassignedId, unassignedFace.box),
                ),
                assignments = listOf(MultiPhotoAssignment(assignedId, sourceId)),
                backend = RequestedInferenceBackend.CPU_ONLY,
                restorationStrength = strength,
                swapBlendMaskMode = SwapBlendMaskMode.PARSER_REGION,
            )
            val swap = requireNotNull(result) { "coordinator returned no result" }
            val finalPixels = swap.finalBitmap.pixels()

            assertEquals("only face A may be swapped", 1, swap.swapRois.size)
            assertEquals(assignedId, swap.swapRois.single().targetId)
            assertTrue(
                "face B must never enter restoration",
                swap.enhanceRois.none { it.targetId == unassignedId },
            )

            // The fixture is only meaningful while the passes really reach across face B.
            val protectedB = ffhqRoi(unassignedFace, target.width, target.height)
            val swapRoi = swap.swapRois.single().bounds
            val overlap = intersect(swapRoi, protectedB)
            assertTrue(
                "fixture is vacuous: face A's paste ROI does not reach face B ($swapRoi vs $protectedB)",
                overlap.width > 0 && overlap.height > 0,
            )
            assertTrue(
                "face A must actually change",
                changedInRoi(originalPixels, finalPixels, target.width, faceRoi(assignedFace, target)) > 100,
            )

            val faceBoxB = faceRoi(unassignedFace, target)
            val changedInB = changedInRoi(originalPixels, finalPixels, target.width, protectedB)
            val changedInFaceBoxB = changedInRoi(originalPixels, finalPixels, target.width, faceBoxB)
            val outsideUnion = outsideUnionChanges(
                originalPixels,
                finalPixels,
                target.width,
                target.height,
                (swap.swapRois + swap.enhanceRois).map(AppliedFaceRoi::bounds),
            )
            // A hard-edged no-write rectangle severs the neighbouring blend mid-alpha and
            // leaves a straight seam; a face-shaped region fades out instead. The jump is
            // measured on the change the paste introduced, not on image content.
            val seamCliff = maxProfileCliff(originalPixels, finalPixels, target.width, swapRoi)

            val directory = File(context.filesDir, OUTPUT_DIRECTORY).apply { mkdirs() }
            File(directory, "dense_$label.json").writeText(
                JSONObject()
                    .put("strength", strength.toDouble())
                    .put("swap_roi", roiJson(swapRoi))
                    .put("protected_b_roi", roiJson(protectedB))
                    .put("overlap", roiJson(overlap))
                    .put("overlap_pixels", overlap.width * overlap.height)
                    .put("face_box_b", roiJson(faceBoxB))
                    .put("changed_in_b_ffhq_roi", changedInB)
                    .put("changed_in_b_face_box", changedInFaceBoxB)
                    .put("outside_union_changed", outsideUnion)
                    .put("max_profile_cliff", seamCliff)
                    .toString(2),
            )
            savePng(swap.finalBitmap, File(directory, "dense_$label.png"))

            assertEquals(
                "$label: the unassigned face B must remain bit-identical",
                0,
                changedInFaceBoxB,
            )
            assertEquals(
                "$label: changed pixels outside union(swap ROI, enhance ROI)",
                0,
                outsideUnion,
            )
            assertTrue(
                "$label: the paste must fade around face B, not stop at an edge " +
                    "(profile cliff $seamCliff, hard rectangle measured 5.13)",
                seamCliff <= MAX_PROFILE_CLIFF,
            )
        } finally {
            result?.finalBitmap?.recycleSafely()
            source.recycleSafely()
            target.recycleSafely()
        }
    }

    private fun requiredModels(strength: Float): Set<ModelId> = buildSet {
        add(ModelId.YOLOFACE_8N)
        add(ModelId.ARCFACE_W600K_R50)
        add(ModelId.INSWAPPER_128_FP16)
        add(ModelId.BISENET_RESNET_34)
        if (strength > 0f) add(ModelId.GFPGAN_1_4)
    }

    private fun ffhqRoi(face: DetectedFace5, width: Int, height: Int): CompositeRoi {
        val imageToCrop = FaceGeometry.estimateSimilarity(
            source = face.landmarks,
            template = WarpTemplate.FFHQ_512,
            cropWidth = 512,
            cropHeight = 512,
        )
        val cropToImage = imageToCrop.inverse()
        val corners = listOf(
            cropToImage.map(Point2(0.0, 0.0)),
            cropToImage.map(Point2(512.0, 0.0)),
            cropToImage.map(Point2(512.0, 512.0)),
            cropToImage.map(Point2(0.0, 512.0)),
        )
        return CompositeRoi(
            floor(corners.minOf(Point2::x)).toInt().coerceIn(0, width),
            floor(corners.minOf(Point2::y)).toInt().coerceIn(0, height),
            ceil(corners.maxOf(Point2::x)).toInt().coerceIn(0, width),
            ceil(corners.maxOf(Point2::y)).toInt().coerceIn(0, height),
        )
    }

    private fun faceRoi(face: DetectedFace5, target: Bitmap): CompositeRoi = CompositeRoi(
        face.box.left.toInt().coerceIn(0, target.width),
        face.box.top.toInt().coerceIn(0, target.height),
        face.box.right.toInt().coerceIn(0, target.width),
        face.box.bottom.toInt().coerceIn(0, target.height),
    )

    private fun intersect(first: CompositeRoi, second: CompositeRoi) = CompositeRoi(
        maxOf(first.left, second.left),
        maxOf(first.top, second.top),
        minOf(first.right, second.right),
        minOf(first.bottom, second.bottom),
    )

    private fun changedInRoi(
        original: IntArray,
        actual: IntArray,
        width: Int,
        roi: CompositeRoi,
    ): Int = (roi.top until roi.bottom).sumOf { y ->
        (roi.left until roi.right).count { x -> original[y * width + x] != actual[y * width + x] }
    }

    private fun outsideUnionChanges(
        original: IntArray,
        actual: IntArray,
        width: Int,
        height: Int,
        rois: List<CompositeRoi>,
    ): Int = (0 until height).sumOf { y ->
        (0 until width).count { x ->
            rois.none { roi -> x in roi.left until roi.right && y in roi.top until roi.bottom } &&
                original[y * width + x] != actual[y * width + x]
        }
    }

    /**
     * Largest drop of the column-averaged paste delta into a column the paste did not
     * touch at all. Averaging over the column cancels face texture, which per-pixel
     * differencing cannot; what survives is exactly the "blend was cut dead here"
     * signature of a hard no-write edge.
     */
    private fun maxProfileCliff(
        original: IntArray,
        actual: IntArray,
        width: Int,
        roi: CompositeRoi,
    ): Double {
        val columnMeans = DoubleArray(roi.width) { column ->
            val x = roi.left + column
            var sum = 0L
            for (y in roi.top until roi.bottom) {
                val index = y * width + x
                val a = original[index]
                val b = actual[index]
                sum += kotlin.math.abs(((a ushr 16) and 0xff) - ((b ushr 16) and 0xff)) +
                    kotlin.math.abs(((a ushr 8) and 0xff) - ((b ushr 8) and 0xff)) +
                    kotlin.math.abs((a and 0xff) - (b and 0xff))
            }
            sum.toDouble() / roi.height
        }
        var worst = 0.0
        for (column in 0 until columnMeans.size - 1) {
            val next = columnMeans[column + 1]
            if (next == 0.0 && columnMeans[column] > worst) worst = columnMeans[column]
            val previous = columnMeans[column]
            if (previous == 0.0 && columnMeans[column + 1] > worst) worst = columnMeans[column + 1]
        }
        return worst
    }

    private fun roiJson(roi: CompositeRoi): JSONObject = JSONObject()
        .put("left", roi.left)
        .put("top", roi.top)
        .put("right", roi.right)
        .put("bottom", roi.bottom)

    private fun savePng(bitmap: Bitmap, file: File) {
        FileOutputStream(file).use { output ->
            check(bitmap.compress(Bitmap.CompressFormat.PNG, 100, output))
        }
    }

    private fun bitmap(path: String): Bitmap = assets.open(path).use { stream ->
        requireNotNull(BitmapFactory.decodeStream(stream)) { "Cannot decode $path" }
            .copy(Bitmap.Config.ARGB_8888, false)
    }

    private fun Bitmap.pixels(): IntArray = IntArray(width * height).also { destination ->
        getPixels(destination, 0, width, 0, 0, width, height)
    }

    private fun Bitmap.recycleSafely() {
        if (!isRecycled) recycle()
    }

    private companion object {
        const val OUTPUT_DIRECTORY = "stage-e-dense"

        /**
         * Measured on this fixture: the hard rectangular no-write region dropped a
         * column mean of 5.13 straight to zero at its edge. A blend that fades out on
         * its own reaches zero gradually, so anything approaching that step is a seam.
         */
        const val MAX_PROFILE_CLIFF = 2.0
    }
}
