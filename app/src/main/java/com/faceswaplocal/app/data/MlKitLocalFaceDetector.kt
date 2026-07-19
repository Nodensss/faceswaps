package com.faceswaplocal.app.data

import android.graphics.Bitmap
import com.faceswaplocal.app.domain.DetectedFace
import com.faceswaplocal.app.domain.FaceId
import com.faceswaplocal.app.domain.LocalFaceDetector
import com.faceswaplocal.app.domain.NormalizedRect
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.face.FaceDetection
import com.google.mlkit.vision.face.FaceDetector
import com.google.mlkit.vision.face.FaceDetectorOptions
import kotlinx.coroutines.tasks.await

class MlKitLocalFaceDetector : LocalFaceDetector {
    private val detector: FaceDetector = FaceDetection.getClient(
        FaceDetectorOptions.Builder()
            .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_ACCURATE)
            .setLandmarkMode(FaceDetectorOptions.LANDMARK_MODE_ALL)
            .setClassificationMode(FaceDetectorOptions.CLASSIFICATION_MODE_ALL)
            .setContourMode(FaceDetectorOptions.CONTOUR_MODE_NONE)
            .setMinFaceSize(0.08f)
            .build(),
    )

    override suspend fun detect(bitmap: Bitmap, idPrefix: String): List<DetectedFace> {
        val input = InputImage.fromBitmap(bitmap, 0)
        val faces = detector.process(input).await()
            .sortedWith(compareBy({ it.boundingBox.top }, { it.boundingBox.left }))

        return faces.mapIndexed { index, face ->
            val bounds = face.boundingBox
            DetectedFace(
                id = FaceId("$idPrefix-${index + 1}"),
                bounds = NormalizedRect.clamped(
                    left = bounds.left.toFloat() / bitmap.width,
                    top = bounds.top.toFloat() / bitmap.height,
                    right = bounds.right.toFloat() / bitmap.width,
                    bottom = bounds.bottom.toFloat() / bitmap.height,
                ),
                yawDegrees = face.headEulerAngleY,
                rollDegrees = face.headEulerAngleZ,
                smileProbability = face.smilingProbability?.takeIf { it >= 0f },
            )
        }
    }

    override fun close() {
        detector.close()
    }
}
