package com.faceswaplocal.app.inference

import kotlin.math.abs

/** A point in image pixel coordinates. */
data class Point2(
    val x: Double,
    val y: Double,
)

/**
 * Row-major 2 x 3 affine matrix:
 *
 *     x' = a * x + b * y + c
 *     y' = d * x + e * y + f
 */
data class AffineMatrix(
    val a: Double,
    val b: Double,
    val c: Double,
    val d: Double,
    val e: Double,
    val f: Double,
) {
    fun map(point: Point2): Point2 = Point2(
        x = a * point.x + b * point.y + c,
        y = d * point.x + e * point.y + f,
    )

    fun inverse(): AffineMatrix {
        val determinant = a * e - b * d
        require(determinant.isFinite() && abs(determinant) > MIN_DETERMINANT) {
            "Affine matrix is not invertible"
        }

        val inverseA = e / determinant
        val inverseB = -b / determinant
        val inverseD = -d / determinant
        val inverseE = a / determinant
        return AffineMatrix(
            a = inverseA,
            b = inverseB,
            c = -(inverseA * c + inverseB * f),
            d = inverseD,
            e = inverseE,
            f = -(inverseD * c + inverseE * f),
        )
    }

    fun inverseMap(point: Point2): Point2 = inverse().map(point)

    private companion object {
        const val MIN_DETERMINANT = 1e-15
    }
}

/** FaceFusion 3.7.1 five-point warp templates used by the Stage B models. */
enum class WarpTemplate(internal val normalizedPoints: List<Point2>) {
    ARCFACE_112_V2(
        listOf(
            Point2(0.34191607, 0.46157411),
            Point2(0.65653393, 0.45983393),
            Point2(0.50022500, 0.64050536),
            Point2(0.37097589, 0.82469196),
            Point2(0.63151696, 0.82325089),
        ),
    ),
    ARCFACE_128(
        listOf(
            Point2(0.36167656, 0.40387734),
            Point2(0.63696719, 0.40235469),
            Point2(0.50019687, 0.56044219),
            Point2(0.38710391, 0.72160547),
            Point2(0.61507734, 0.72034453),
        ),
    ),
}

object FaceGeometry {
    /**
     * Fits the orientation-preserving least-squares similarity transform from the
     * five detected landmarks to the selected FaceFusion template.
     *
     * The resulting matrix has the OpenCV/FaceFusion form `[a, b, c; -b, a, f]`.
     */
    fun estimateSimilarity(
        source: List<Point2>,
        template: WarpTemplate,
        cropWidth: Int,
        cropHeight: Int,
    ): AffineMatrix {
        require(source.size == LANDMARK_COUNT) {
            "Exactly $LANDMARK_COUNT source landmarks are required"
        }
        require(cropWidth > 0 && cropHeight > 0) {
            "Crop dimensions must be positive"
        }
        require(source.all { it.x.isFinite() && it.y.isFinite() }) {
            "Source landmarks must be finite"
        }

        val target = template.normalizedPoints.map { point ->
            Point2(
                x = point.x * cropWidth,
                y = point.y * cropHeight,
            )
        }

        val sourceMeanX = source.sumOf(Point2::x) / LANDMARK_COUNT
        val sourceMeanY = source.sumOf(Point2::y) / LANDMARK_COUNT
        val targetMeanX = target.sumOf(Point2::x) / LANDMARK_COUNT
        val targetMeanY = target.sumOf(Point2::y) / LANDMARK_COUNT

        var denominator = 0.0
        var numeratorA = 0.0
        var numeratorB = 0.0
        for (index in 0 until LANDMARK_COUNT) {
            val sourceX = source[index].x - sourceMeanX
            val sourceY = source[index].y - sourceMeanY
            val targetX = target[index].x - targetMeanX
            val targetY = target[index].y - targetMeanY

            denominator += sourceX * sourceX + sourceY * sourceY
            numeratorA += sourceX * targetX + sourceY * targetY
            numeratorB += sourceY * targetX - sourceX * targetY
        }
        require(denominator.isFinite() && denominator > MIN_LANDMARK_VARIANCE) {
            "Source landmarks do not define a similarity transform"
        }

        val a = numeratorA / denominator
        val b = numeratorB / denominator
        val c = targetMeanX - a * sourceMeanX - b * sourceMeanY
        val f = targetMeanY + b * sourceMeanX - a * sourceMeanY
        return AffineMatrix(
            a = a,
            b = b,
            c = c,
            d = -b,
            e = a,
            f = f,
        )
    }

    private const val LANDMARK_COUNT = 5
    private const val MIN_LANDMARK_VARIANCE = 1e-12
}
