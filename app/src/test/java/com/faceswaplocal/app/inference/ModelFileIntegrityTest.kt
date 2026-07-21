package com.faceswaplocal.app.inference

import java.io.ByteArrayInputStream
import java.io.ByteArrayOutputStream
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertSame
import org.junit.Test

class ModelFileIntegrityTest {
    private val payload = "FaceSwapLocal model fixture".encodeToByteArray()
    private val payloadSha256 = "ab153366d46f11857bd1810eceb477c7802c76a6e9225e654bdb878ebb5510b3"

    @Test
    fun `copy computes sha and size while preserving every byte`() {
        val destination = ByteArrayOutputStream()
        val progress = mutableListOf<Long>()

        val observation = ModelFileIntegrity.copyAndHash(
            input = ByteArrayInputStream(payload),
            output = destination,
            onChunk = progress::add,
        )

        assertArrayEquals(payload, destination.toByteArray())
        assertEquals(payload.size.toLong(), observation.sizeBytes)
        assertEquals(payloadSha256, observation.sha256)
        assertEquals(listOf(payload.size.toLong()), progress)
    }

    @Test
    fun `validation accepts an exact size and checksum match`() {
        val descriptor = fixtureDescriptor(
            expectedSizeBytes = payload.size.toLong(),
            expectedSha256 = payloadSha256,
        )

        val details = ModelFileIntegrity.validationDetails(
            descriptor = descriptor,
            observation = ModelFileObservation(payload.size.toLong(), payloadSha256),
            existingCopyRetained = false,
        )

        assertNull(details)
    }

    @Test
    fun `validation reports expected and actual values for size mismatch`() {
        val descriptor = fixtureDescriptor(
            expectedSizeBytes = payload.size.toLong() + 1L,
            expectedSha256 = payloadSha256,
        )

        val details = ModelFileIntegrity.validationDetails(
            descriptor = descriptor,
            observation = ModelFileObservation(payload.size.toLong(), payloadSha256),
            existingCopyRetained = true,
        )

        requireNotNull(details)
        assertEquals(ModelValidationFailure.SIZE_MISMATCH, details.reason)
        assertEquals(payload.size.toLong() + 1L, details.expectedSizeBytes)
        assertEquals(payload.size.toLong(), details.actualSizeBytes)
        assertEquals(payloadSha256, details.expectedSha256)
        assertEquals(payloadSha256, details.actualSha256)
        assertEquals(true, details.existingCopyRetained)
    }

    @Test
    fun `validation reports checksum mismatch after size matches`() {
        val wrongSha256 = "0".repeat(64)
        val descriptor = fixtureDescriptor(
            expectedSizeBytes = payload.size.toLong(),
            expectedSha256 = wrongSha256,
        )

        val details = ModelFileIntegrity.validationDetails(
            descriptor = descriptor,
            observation = ModelFileObservation(payload.size.toLong(), payloadSha256),
            existingCopyRetained = false,
        )

        requireNotNull(details)
        assertEquals(ModelValidationFailure.CHECKSUM_MISMATCH, details.reason)
        assertEquals(wrongSha256, details.expectedSha256)
        assertEquals(payloadSha256, details.actualSha256)
    }

    @Test
    fun `failed replacement keeps a revalidated installed model ready`() {
        val candidateFailure = ModelStatus.Failed(
            reason = ModelStoreFailure.SOURCE_UNAVAILABLE,
            existingCopyRetained = true,
        )
        val retainedReady = ModelStatus.Ready(verifiedSizeBytes = payload.size.toLong())

        val visibleStatus = replacementImportVisibleStatus(
            replacementFailure = candidateFailure,
            retainedCopyStatus = retainedReady,
        )

        assertSame(retainedReady, visibleStatus)
    }

    @Test
    fun `rejected replacement keeps a revalidated installed model ready`() {
        val candidateFailure = ModelStatus.Invalid(
            ModelValidationDetails(
                reason = ModelValidationFailure.SIZE_MISMATCH,
                expectedSizeBytes = payload.size.toLong(),
                actualSizeBytes = 1L,
                expectedSha256 = payloadSha256,
                actualSha256 = "0".repeat(64),
                existingCopyRetained = true,
            ),
        )
        val retainedReady = ModelStatus.Ready(verifiedSizeBytes = payload.size.toLong())

        val visibleStatus = replacementImportVisibleStatus(
            replacementFailure = candidateFailure,
            retainedCopyStatus = retainedReady,
        )

        assertSame(retainedReady, visibleStatus)
    }

    @Test
    fun `rejected replacement remains visible when no installed copy exists`() {
        val candidateFailure = ModelStatus.Invalid(
            ModelValidationDetails(
                reason = ModelValidationFailure.CHECKSUM_MISMATCH,
                expectedSizeBytes = payload.size.toLong(),
                actualSizeBytes = payload.size.toLong(),
                expectedSha256 = "0".repeat(64),
                actualSha256 = payloadSha256,
                existingCopyRetained = false,
            ),
        )

        val visibleStatus = replacementImportVisibleStatus(
            replacementFailure = candidateFailure,
            retainedCopyStatus = null,
        )

        assertSame(candidateFailure, visibleStatus)
    }

    private fun fixtureDescriptor(
        expectedSizeBytes: Long,
        expectedSha256: String,
    ) = ModelDescriptor(
        id = ModelId.YOLOFACE_8N,
        role = ModelRole.FACE_DETECTOR,
        fileName = "yoloface_8n.onnx",
        expectedSizeBytes = expectedSizeBytes,
        expectedSha256 = expectedSha256,
    )
}
