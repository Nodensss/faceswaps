package com.faceswaplocal.app.inference

import java.io.ByteArrayOutputStream
import java.io.File
import java.io.RandomAccessFile
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertThrows
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder

class OnnxInitializerReaderTest {
    @get:Rule
    val temporaryFolder = TemporaryFolder()

    @Test
    fun `reads exact raw data initializer even when it is not last`() {
        val model = model(
            tensor("emap.extra", longArrayOf(2, 2), rawValues = floatArrayOf(9f, 9f, 9f, 9f)),
            tensor("emap", longArrayOf(2, 2), rawValues = floatArrayOf(1.25f, -2f, 3.5f, 4f)),
            tensor("trailing", longArrayOf(1), rawValues = floatArrayOf(7f)),
        )
        val modelFile = writeModel(model)

        val result = OnnxInitializerReader.readFloatTensor(
            modelFile = modelFile,
            initializerName = "emap",
            expectedDimensions = longArrayOf(2, 2),
        )

        assertArrayEquals(floatArrayOf(1.25f, -2f, 3.5f, 4f), result, 0f)
    }

    @Test
    fun `reads packed float data when name follows tensor values`() {
        val target = tensor(
            name = "typed_values",
            dimensions = longArrayOf(3),
            floatValues = floatArrayOf(Float.NaN, Float.POSITIVE_INFINITY, -0.0f),
            nameAfterData = true,
            packedDimensions = false,
        )
        val modelFile = writeModel(model(target))

        val result = OnnxInitializerReader.readFloatTensor(
            modelFile = modelFile,
            initializerName = "typed_values",
            expectedDimensions = longArrayOf(3),
        )

        assertArrayEquals(
            floatArrayOf(Float.NaN, Float.POSITIVE_INFINITY, -0.0f),
            result,
            0f,
        )
    }

    @Test
    fun `rejects an initializer whose dimensions differ from expectation`() {
        val modelFile = writeModel(
            model(tensor("emap", longArrayOf(2, 3), rawValues = FloatArray(6))),
        )

        assertThrows(IllegalArgumentException::class.java) {
            OnnxInitializerReader.readFloatTensor(
                modelFile = modelFile,
                initializerName = "emap",
                expectedDimensions = longArrayOf(3, 2),
            )
        }
    }

    @Test
    fun `seeks over a large unrelated protobuf payload`() {
        val modelFile = temporaryFolder.newFile("large-model.onnx")
        val targetTensor = tensor(
            name = "emap",
            dimensions = longArrayOf(2),
            rawValues = floatArrayOf(5f, 6f),
        )
        val initializerField = lengthDelimitedField(5, targetTensor)
        val skippedPayloadBytes = 16L * 1024L * 1024L
        val skippedFieldHeader = tag(1, 2) + varint(skippedPayloadBytes)
        val graphLength = skippedFieldHeader.size + skippedPayloadBytes + initializerField.size

        RandomAccessFile(modelFile, "rw").use { file ->
            file.write(tag(7, 2))
            file.write(varint(graphLength))
            file.write(skippedFieldHeader)
            file.seek(file.filePointer + skippedPayloadBytes)
            file.write(initializerField)
        }

        val result = OnnxInitializerReader.readFloatTensor(
            modelFile = modelFile,
            initializerName = "emap",
            expectedDimensions = longArrayOf(2),
        )

        assertArrayEquals(floatArrayOf(5f, 6f), result, 0f)
    }

    private fun writeModel(bytes: ByteArray): File =
        temporaryFolder.newFile("model-${System.nanoTime()}.onnx").apply { writeBytes(bytes) }

    private fun model(vararg initializers: ByteArray): ByteArray {
        val graph = ByteArrayOutputStream().apply {
            write(lengthDelimitedField(2, "test-graph".toByteArray()))
            initializers.forEach { write(lengthDelimitedField(5, it)) }
        }.toByteArray()
        return lengthDelimitedField(7, graph)
    }

    private fun tensor(
        name: String,
        dimensions: LongArray,
        rawValues: FloatArray? = null,
        floatValues: FloatArray? = null,
        nameAfterData: Boolean = false,
        packedDimensions: Boolean = true,
    ): ByteArray {
        require((rawValues == null) != (floatValues == null))
        val output = ByteArrayOutputStream()
        if (packedDimensions) {
            val packed = ByteArrayOutputStream().apply {
                dimensions.forEach { write(varint(it)) }
            }.toByteArray()
            output.write(lengthDelimitedField(1, packed))
        } else {
            dimensions.forEach { dimension ->
                output.write(tag(1, 0))
                output.write(varint(dimension))
            }
        }
        output.write(tag(2, 0))
        output.write(varint(1)) // TensorProto.DataType.FLOAT
        if (!nameAfterData) output.write(lengthDelimitedField(8, name.toByteArray()))
        rawValues?.let { output.write(lengthDelimitedField(9, floatBytes(it))) }
        floatValues?.let { output.write(lengthDelimitedField(4, floatBytes(it))) }
        if (nameAfterData) output.write(lengthDelimitedField(8, name.toByteArray()))
        return output.toByteArray()
    }

    private fun floatBytes(values: FloatArray): ByteArray =
        ByteArray(values.size * 4).also { bytes ->
            values.forEachIndexed { index, value ->
                val bits = value.toRawBits()
                val offset = index * 4
                bytes[offset] = bits.toByte()
                bytes[offset + 1] = (bits ushr 8).toByte()
                bytes[offset + 2] = (bits ushr 16).toByte()
                bytes[offset + 3] = (bits ushr 24).toByte()
            }
        }

    private fun lengthDelimitedField(fieldNumber: Int, payload: ByteArray): ByteArray =
        tag(fieldNumber, 2) + varint(payload.size.toLong()) + payload

    private fun tag(fieldNumber: Int, wireType: Int): ByteArray =
        varint((fieldNumber.toLong() shl 3) or wireType.toLong())

    private fun varint(value: Long): ByteArray {
        require(value >= 0)
        var remaining = value
        return ByteArrayOutputStream().apply {
            do {
                var byte = (remaining and 0x7f).toInt()
                remaining = remaining ushr 7
                if (remaining != 0L) byte = byte or 0x80
                write(byte)
            } while (remaining != 0L)
        }.toByteArray()
    }
}
