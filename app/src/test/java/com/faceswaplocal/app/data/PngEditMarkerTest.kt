package com.faceswaplocal.app.data

import java.io.File
import java.util.zip.CRC32
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder

class PngEditMarkerTest {
    @get:Rule
    val folder = TemporaryFolder()

    @Test
    fun `marker is inserted before IEND and the file stays a valid PNG`() {
        val file = folder.newFile("result.png")
        val original = minimalPng()
        file.writeBytes(original)

        assertTrue(PngEditMarker.writeTextChunk(file, "Software", "FaceSwapLocal"))

        val marked = file.readBytes()
        assertArrayEquals(
            "the 8-byte signature must survive",
            original.copyOfRange(0, 8),
            marked.copyOfRange(0, 8),
        )
        assertArrayEquals(
            "IEND must remain the last chunk",
            original.copyOfRange(original.size - 12, original.size),
            marked.copyOfRange(marked.size - 12, marked.size),
        )
        assertTrue(
            "keyword and text are separated by the mandatory null byte",
            String(marked, Charsets.ISO_8859_1)
                .contains("Software" + 0.toChar() + "FaceSwapLocal"),
        )
        assertChunksAreWellFormed(marked)
    }

    @Test
    fun `chunk declares the payload length and a correct CRC`() {
        val file = folder.newFile("result.png")
        file.writeBytes(minimalPng())
        PngEditMarker.writeTextChunk(file, "Software", "FaceSwapLocal")

        val marked = file.readBytes()
        val chunkStart = marked.size - 12 - ("Software".length + 1 + "FaceSwapLocal".length) - 12
        val declaredLength = readInt(marked, chunkStart)
        assertEquals("Software".length + 1 + "FaceSwapLocal".length, declaredLength)
        assertEquals("tEXt", String(marked, chunkStart + 4, 4, Charsets.US_ASCII))
        assertEquals(
            0.toByte(),
            marked[chunkStart + 8 + "Software".length],
        )

        val expectedCrc = CRC32().apply {
            update(marked, chunkStart + 4, 4 + declaredLength)
        }.value.toInt()
        assertEquals(expectedCrc, readInt(marked, chunkStart + 8 + declaredLength))
    }

    @Test
    fun `a file that is not a PNG is left untouched`() {
        val file = folder.newFile("result.jpg")
        val jpegLike = ByteArray(64) { index -> index.toByte() }
        file.writeBytes(jpegLike)

        assertFalse(PngEditMarker.writeTextChunk(file, "Software", "FaceSwapLocal"))
        assertArrayEquals(jpegLike, file.readBytes())
    }

    @Test
    fun `a truncated PNG is left untouched instead of being corrupted`() {
        val file = folder.newFile("truncated.png")
        val truncated = minimalPng().copyOfRange(0, 20)
        file.writeBytes(truncated)

        assertFalse(PngEditMarker.writeTextChunk(file, "Software", "FaceSwapLocal"))
        assertArrayEquals(truncated, file.readBytes())
    }

    @Test
    fun `writing the marker twice keeps the file well formed`() {
        val file = folder.newFile("result.png")
        file.writeBytes(minimalPng())

        assertTrue(PngEditMarker.writeTextChunk(file, "Software", "FaceSwapLocal"))
        assertTrue(PngEditMarker.writeTextChunk(file, "Software", "FaceSwapLocal"))

        assertChunksAreWellFormed(file.readBytes())
    }

    private fun assertChunksAreWellFormed(png: ByteArray) {
        var offset = 8
        val seen = mutableListOf<String>()
        while (offset < png.size) {
            val length = readInt(png, offset)
            val type = String(png, offset + 4, 4, Charsets.US_ASCII)
            val crc = readInt(png, offset + 8 + length)
            val expected = CRC32().apply { update(png, offset + 4, 4 + length) }.value.toInt()
            assertEquals("CRC of chunk $type", expected, crc)
            seen += type
            offset += 12 + length
        }
        assertEquals("chunks must exactly cover the file", png.size, offset)
        assertEquals("IHDR", seen.first())
        assertEquals("IEND", seen.last())
        assertTrue(seen.contains("tEXt"))
    }

    private fun readInt(bytes: ByteArray, offset: Int): Int =
        ((bytes[offset].toInt() and 0xFF) shl 24) or
            ((bytes[offset + 1].toInt() and 0xFF) shl 16) or
            ((bytes[offset + 2].toInt() and 0xFF) shl 8) or
            (bytes[offset + 3].toInt() and 0xFF)

    /** Signature + IHDR + IEND: the smallest structurally valid PNG for chunk editing. */
    private fun minimalPng(): ByteArray {
        val signature = byteArrayOf(0x89.toByte(), 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A)
        val ihdrPayload = byteArrayOf(
            0, 0, 0, 1, // width
            0, 0, 0, 1, // height
            8, 6, 0, 0, 0, // bit depth, colour type, compression, filter, interlace
        )
        return signature + chunk("IHDR", ihdrPayload) + chunk("IEND", ByteArray(0))
    }

    private fun chunk(type: String, payload: ByteArray): ByteArray {
        val typeBytes = type.toByteArray(Charsets.US_ASCII)
        val crc = CRC32().apply {
            update(typeBytes)
            update(payload)
        }.value.toInt()
        return intBytes(payload.size) + typeBytes + payload + intBytes(crc)
    }

    private fun intBytes(value: Int): ByteArray = byteArrayOf(
        (value ushr 24).toByte(),
        (value ushr 16).toByte(),
        (value ushr 8).toByte(),
        value.toByte(),
    )
}
