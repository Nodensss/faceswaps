package com.faceswaplocal.app.data

import java.io.File
import java.io.IOException
import java.io.RandomAccessFile
import java.util.zip.CRC32

/**
 * Writes the neutral "this image was edited" note into a PNG (§5.3).
 *
 * A `tEXt` chunk is valid anywhere between `IHDR` and `IEND`, so the marker is appended
 * in place: the trailing 12-byte `IEND` chunk is truncated, the new chunk is written and
 * `IEND` is restored. That keeps the operation O(1) instead of rewriting a full-size PNG
 * through a second temporary file.
 *
 * Pure `java.io`, so it is covered by JVM unit tests rather than only on a device.
 */
internal object PngEditMarker {
    private val SIGNATURE = byteArrayOf(
        0x89.toByte(), 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A,
    )
    private const val IEND_CHUNK_SIZE = 12L
    private const val TEXT_CHUNK_TYPE = "tEXt"

    /**
     * Returns true when the marker was written. A file that is not a PNG or has no
     * trailing `IEND` is left untouched: an unmarked export is acceptable, a corrupted
     * one is not.
     */
    fun writeTextChunk(file: File, keyword: String, text: String): Boolean {
        require(keyword.isNotEmpty() && keyword.length <= 79) { "PNG keyword must be 1..79 bytes" }
        val keywordBytes = keyword.toByteArray(Charsets.ISO_8859_1)
        val textBytes = text.toByteArray(Charsets.ISO_8859_1)

        return try {
            RandomAccessFile(file, "rw").use { raf ->
                if (raf.length() < SIGNATURE.size + IEND_CHUNK_SIZE) return false

                val signature = ByteArray(SIGNATURE.size)
                raf.seek(0L)
                raf.readFully(signature)
                if (!signature.contentEquals(SIGNATURE)) return false

                val iend = ByteArray(IEND_CHUNK_SIZE.toInt())
                raf.seek(raf.length() - IEND_CHUNK_SIZE)
                raf.readFully(iend)
                if (String(iend, 4, 4, Charsets.US_ASCII) != "IEND") return false

                raf.setLength(raf.length() - IEND_CHUNK_SIZE)
                raf.seek(raf.length())
                raf.write(textChunk(keywordBytes, textBytes))
                raf.write(iend)
                raf.fd.sync()
                true
            }
        } catch (_: IOException) {
            false
        }
    }

    private fun textChunk(keyword: ByteArray, text: ByteArray): ByteArray {
        val payload = ByteArray(keyword.size + 1 + text.size)
        keyword.copyInto(payload)
        payload[keyword.size] = 0
        text.copyInto(payload, keyword.size + 1)

        val type = TEXT_CHUNK_TYPE.toByteArray(Charsets.US_ASCII)
        val crc = CRC32().apply {
            update(type)
            update(payload)
        }.value

        val chunk = ByteArray(4 + type.size + payload.size + 4)
        writeInt(chunk, 0, payload.size)
        type.copyInto(chunk, 4)
        payload.copyInto(chunk, 4 + type.size)
        writeInt(chunk, 4 + type.size + payload.size, crc.toInt())
        return chunk
    }

    private fun writeInt(target: ByteArray, offset: Int, value: Int) {
        target[offset] = (value ushr 24).toByte()
        target[offset + 1] = (value ushr 16).toByte()
        target[offset + 2] = (value ushr 8).toByte()
        target[offset + 3] = value.toByte()
    }
}
