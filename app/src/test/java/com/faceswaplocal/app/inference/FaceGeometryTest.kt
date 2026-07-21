package com.faceswaplocal.app.inference

import org.junit.Assert.assertEquals
import org.junit.Assert.assertThrows
import org.junit.Test

class FaceGeometryTest {
    @Test
    fun `arcface 112 v2 matrices match FaceFusion desktop references`() {
        assertReference(
            source = listOf(
                Point2(537.5440673828125, 540.87890625),
                Point2(759.3545532226562, 539.4194946289062),
                Point2(651.7015991210938, 686.5709838867188),
                Point2(554.3982543945312, 788.293212890625),
                Point2(741.6643676757812, 786.9005126953125),
            ),
            template = WarpTemplate.ARCFACE_112_V2,
            cropSize = 112,
            expected = AffineMatrix(
                a = 0.16094582069451432,
                b = 0.0006192408456529614,
                c = -48.83073351936581,
                d = -0.0006192408456529614,
                e = 0.16094582069451432,
                f = -35.27559121927607,
            ),
        )
        assertReference(
            source = listOf(
                Point2(510.39459228515625, 477.9002990722656),
                Point2(738.0992431640625, 473.98638916015625),
                Point2(623.2639770507812, 592.3964233398438),
                Point2(536.24609375, 729.9790649414062),
                Point2(722.9362182617188, 727.13232421875),
            ),
            template = WarpTemplate.ARCFACE_112_V2,
            cropSize = 112,
            expected = AffineMatrix(
                a = 0.1586024092705596,
                b = -0.0021816128969535616,
                c = -41.97919354758747,
                d = 0.0021816128969535616,
                e = 0.1586024092705596,
                f = -24.670998418558924,
            ),
        )
        assertReference(
            source = listOf(
                Point2(517.371337890625, 525.3013305664062),
                Point2(739.1956787109375, 515.0154418945312),
                Point2(627.4386596679688, 659.8773193359375),
                Point2(541.909423828125, 750.6635131835938),
                Point2(735.5901489257812, 741.7702026367188),
            ),
            template = WarpTemplate.ARCFACE_112_V2,
            cropSize = 112,
            expected = AffineMatrix(
                a = 0.16723402459810957,
                b = -0.006816990569199807,
                c = -45.363266917380585,
                d = 0.006816990569199807,
                e = 0.16723402459810957,
                f = -39.19280850574597,
            ),
        )
    }

    @Test
    fun `arcface 128 matrices match FaceFusion desktop references`() {
        assertReference(
            source = listOf(
                Point2(502.15142822265625, 527.9790649414062),
                Point2(748.8923950195312, 528.9376220703125),
                Point2(630.2299194335938, 677.1703491210938),
                Point2(518.0090942382812, 792.799072265625),
                Point2(726.3547973632812, 793.3845825195312),
            ),
            template = WarpTemplate.ARCFACE_128,
            cropSize = 256,
            expected = AffineMatrix(
                a = 0.2964897305193567,
                b = 0.004298180083124341,
                c = -60.14579691252937,
                d = -0.004298180083124341,
                e = 0.2964897305193567,
                f = -50.39675929497442,
            ),
        )
        assertReference(
            source = listOf(
                Point2(623.4712524414062, 507.28948974609375),
                Point2(817.2049560546875, 484.0359191894531),
                Point2(767.89208984375, 613.6771240234375),
                Point2(662.0415649414062, 738.5077514648438),
                Point2(821.240966796875, 719.4370727539062),
            ),
            template = WarpTemplate.ARCFACE_128,
            cropSize = 256,
            expected = AffineMatrix(
                a = 0.3468091499479692,
                b = -0.03332409008314546,
                c = -107.60722355677765,
                d = 0.03332409008314546,
                e = 0.3468091499479692,
                f = -93.25558528264953,
            ),
        )
        assertReference(
            source = listOf(
                Point2(568.8838500976562, 583.9884033203125),
                Point2(761.0355834960938, 544.8619995117188),
                Point2(704.29931640625, 673.9825439453125),
                Point2(625.9950561523438, 787.2018432617188),
                Point2(787.717041015625, 754.7252807617188),
            ),
            template = WarpTemplate.ARCFACE_128,
            cropSize = 256,
            expected = AffineMatrix(
                a = 0.3652797175415404,
                b = -0.07185972488070132,
                c = -75.76881415937862,
                d = 0.07185972488070132,
                e = 0.3652797175415404,
                f = -150.10651269830115,
            ),
        )
    }

    @Test
    fun `crop dimensions scale the FaceFusion template`() {
        val source = listOf(
            Point2(502.15142822265625, 527.9790649414062),
            Point2(748.8923950195312, 528.9376220703125),
            Point2(630.2299194335938, 677.1703491210938),
            Point2(518.0090942382812, 792.799072265625),
            Point2(726.3547973632812, 793.3845825195312),
        )

        val matrix = FaceGeometry.estimateSimilarity(
            source = source,
            template = WarpTemplate.ARCFACE_128,
            cropWidth = 128,
            cropHeight = 128,
        )

        assertMatrixEquals(
            expected = AffineMatrix(
                a = 0.14824486525967834,
                b = 0.0021490900415621706,
                c = -30.072898456264685,
                d = -0.0021490900415621706,
                e = 0.14824486525967834,
                f = -25.19837964748721,
            ),
            actual = matrix,
        )
    }

    @Test
    fun `inverse restores points after forward mapping`() {
        val matrix = AffineMatrix(
            a = 0.3652797188063376,
            b = -0.07185971497267118,
            c = -75.76881415937862,
            d = 0.07185971497267118,
            e = 0.3652797188063376,
            f = -150.1065089676849,
        )
        val source = Point2(509.4480285644531, 625.6807250976562)

        val mapped = matrix.map(source)
        val restored = matrix.inverse().map(mapped)
        val restoredDirectly = matrix.inverseMap(mapped)

        assertPointEquals(source, restored, ROUND_TRIP_TOLERANCE)
        assertPointEquals(source, restoredDirectly, ROUND_TRIP_TOLERANCE)
    }

    @Test
    fun `invalid or degenerate landmarks are rejected`() {
        val repeated = List(5) { Point2(1.0, 1.0) }

        assertThrows(IllegalArgumentException::class.java) {
            FaceGeometry.estimateSimilarity(
                source = repeated,
                template = WarpTemplate.ARCFACE_112_V2,
                cropWidth = 112,
                cropHeight = 112,
            )
        }
        assertThrows(IllegalArgumentException::class.java) {
            FaceGeometry.estimateSimilarity(
                source = repeated.take(4),
                template = WarpTemplate.ARCFACE_112_V2,
                cropWidth = 112,
                cropHeight = 112,
            )
        }
        assertThrows(IllegalArgumentException::class.java) {
            FaceGeometry.estimateSimilarity(
                source = repeated,
                template = WarpTemplate.ARCFACE_112_V2,
                cropWidth = 0,
                cropHeight = 112,
            )
        }
    }

    private fun assertReference(
        source: List<Point2>,
        template: WarpTemplate,
        cropSize: Int,
        expected: AffineMatrix,
    ) {
        val actual = FaceGeometry.estimateSimilarity(
            source = source,
            template = template,
            cropWidth = cropSize,
            cropHeight = cropSize,
        )

        assertMatrixEquals(expected, actual)
    }

    private fun assertMatrixEquals(expected: AffineMatrix, actual: AffineMatrix) {
        assertEquals(expected.a, actual.a, REFERENCE_TOLERANCE)
        assertEquals(expected.b, actual.b, REFERENCE_TOLERANCE)
        assertEquals(expected.c, actual.c, REFERENCE_TOLERANCE)
        assertEquals(expected.d, actual.d, REFERENCE_TOLERANCE)
        assertEquals(expected.e, actual.e, REFERENCE_TOLERANCE)
        assertEquals(expected.f, actual.f, REFERENCE_TOLERANCE)
    }

    private fun assertPointEquals(expected: Point2, actual: Point2, tolerance: Double) {
        assertEquals(expected.x, actual.x, tolerance)
        assertEquals(expected.y, actual.y, tolerance)
    }

    private companion object {
        const val REFERENCE_TOLERANCE = 1e-5
        const val ROUND_TRIP_TOLERANCE = 1e-9
    }
}
