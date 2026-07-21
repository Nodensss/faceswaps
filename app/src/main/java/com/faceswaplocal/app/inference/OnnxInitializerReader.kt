package com.faceswaplocal.app.inference

import java.io.File
import java.io.IOException
import java.io.RandomAccessFile

/**
 * Reads a single float initializer without materialising the ONNX model in memory.
 *
 * ONNX files use protobuf encoding. The reader walks only the ModelProto graph and
 * GraphProto initializer envelopes, seeks over unrelated fields, and allocates memory
 * only for the requested tensor. This is important for swap models that are hundreds
 * of megabytes while their embedding-conversion initializer is about one megabyte.
 */
object OnnxInitializerReader {
    // ONNX ModelProto.graph is field 7; field 8 is opset_import.
    private const val MODEL_GRAPH_FIELD = 7
    private const val GRAPH_INITIALIZER_FIELD = 5

    private const val TENSOR_DIMS_FIELD = 1
    private const val TENSOR_DATA_TYPE_FIELD = 2
    private const val TENSOR_FLOAT_DATA_FIELD = 4
    private const val TENSOR_NAME_FIELD = 8
    private const val TENSOR_RAW_DATA_FIELD = 9

    private const val ONNX_FLOAT_DATA_TYPE = 1
    private const val MAX_NAME_BYTES = 1024 * 1024
    private const val FLOAT_BYTES = 4
    private const val READ_BUFFER_BYTES = 64 * 1024

    /**
     * Returns the values of the initializer named [initializerName].
     *
     * Only ONNX FLOAT tensors backed by little-endian `raw_data` or protobuf
     * `float_data` are supported. [expectedDimensions] is mandatory so that a wrong
     * initializer or an incompatible model fails before its values reach inference.
     */
    @Throws(IOException::class)
    fun readFloatTensor(
        modelFile: File,
        initializerName: String,
        expectedDimensions: LongArray,
    ): FloatArray {
        require(initializerName.isNotEmpty()) { "Initializer name must not be empty" }
        val expectedElementCount = checkedElementCount(expectedDimensions)

        RandomAccessFile(modelFile, "r").use { file ->
            val modelEnd = file.length()
            while (file.filePointer < modelEnd) {
                val tag = readTag(file, modelEnd) ?: break
                if (tag.fieldNumber == MODEL_GRAPH_FIELD && tag.wireType == WireType.LENGTH_DELIMITED) {
                    val graphEnd = readDelimitedEnd(file, modelEnd, "ModelProto.graph")
                    val values = findInGraph(
                        file = file,
                        graphEnd = graphEnd,
                        initializerName = initializerName,
                        expectedDimensions = expectedDimensions,
                        expectedElementCount = expectedElementCount,
                    )
                    if (values != null) return values
                    file.seek(graphEnd)
                } else {
                    skipField(file, tag, modelEnd)
                }
            }
        }

        throw IllegalArgumentException(
            "ONNX initializer '$initializerName' was not found in ${modelFile.name}",
        )
    }

    private fun findInGraph(
        file: RandomAccessFile,
        graphEnd: Long,
        initializerName: String,
        expectedDimensions: LongArray,
        expectedElementCount: Int,
    ): FloatArray? {
        while (file.filePointer < graphEnd) {
            val tag = readTag(file, graphEnd) ?: break
            if (tag.fieldNumber == GRAPH_INITIALIZER_FIELD &&
                tag.wireType == WireType.LENGTH_DELIMITED
            ) {
                val tensorEnd = readDelimitedEnd(file, graphEnd, "GraphProto.initializer")
                val tensor = inspectTensor(file, tensorEnd)
                file.seek(tensorEnd)

                if (tensor.name == initializerName) {
                    validateTensor(
                        tensor = tensor,
                        initializerName = initializerName,
                        expectedDimensions = expectedDimensions,
                        expectedElementCount = expectedElementCount,
                    )
                    return readTensorValues(file, tensor, expectedElementCount)
                }
            } else {
                skipField(file, tag, graphEnd)
            }
        }
        return null
    }

    private fun inspectTensor(file: RandomAccessFile, tensorEnd: Long): TensorMetadata {
        val dimensions = mutableListOf<Long>()
        var dataType = 0
        var name: String? = null
        var rawData: DataSegment? = null
        val floatData = mutableListOf<DataSegment>()

        while (file.filePointer < tensorEnd) {
            val tag = readTag(file, tensorEnd) ?: break
            when (tag.fieldNumber) {
                TENSOR_DIMS_FIELD -> when (tag.wireType) {
                    WireType.VARINT -> dimensions += readVarint(file, tensorEnd, "TensorProto.dims")
                    WireType.LENGTH_DELIMITED -> {
                        val packedEnd = readDelimitedEnd(file, tensorEnd, "TensorProto.dims")
                        while (file.filePointer < packedEnd) {
                            dimensions += readVarint(file, packedEnd, "TensorProto.dims")
                        }
                    }
                    else -> skipField(file, tag, tensorEnd)
                }

                TENSOR_DATA_TYPE_FIELD -> {
                    if (tag.wireType == WireType.VARINT) {
                        val value = readVarint(file, tensorEnd, "TensorProto.data_type")
                        if (value !in Int.MIN_VALUE.toLong()..Int.MAX_VALUE.toLong()) {
                            throw OnnxFormatException("TensorProto.data_type is outside Int range")
                        }
                        dataType = value.toInt()
                    } else {
                        skipField(file, tag, tensorEnd)
                    }
                }

                TENSOR_FLOAT_DATA_FIELD -> when (tag.wireType) {
                    WireType.LENGTH_DELIMITED -> {
                        val dataEnd = readDelimitedEnd(file, tensorEnd, "TensorProto.float_data")
                        val start = file.filePointer
                        val byteCount = dataEnd - start
                        if (byteCount % FLOAT_BYTES != 0L) {
                            throw OnnxFormatException(
                                "Packed TensorProto.float_data length $byteCount is not divisible by $FLOAT_BYTES",
                            )
                        }
                        floatData += DataSegment(start, byteCount)
                        file.seek(dataEnd)
                    }
                    WireType.FIXED32 -> {
                        ensureAvailable(file, FLOAT_BYTES.toLong(), tensorEnd, "TensorProto.float_data")
                        floatData += DataSegment(file.filePointer, FLOAT_BYTES.toLong())
                        file.seek(file.filePointer + FLOAT_BYTES)
                    }
                    else -> skipField(file, tag, tensorEnd)
                }

                TENSOR_NAME_FIELD -> {
                    if (tag.wireType == WireType.LENGTH_DELIMITED) {
                        name = readString(file, tensorEnd, "TensorProto.name")
                    } else {
                        skipField(file, tag, tensorEnd)
                    }
                }

                TENSOR_RAW_DATA_FIELD -> {
                    if (tag.wireType == WireType.LENGTH_DELIMITED) {
                        val dataEnd = readDelimitedEnd(file, tensorEnd, "TensorProto.raw_data")
                        rawData = DataSegment(file.filePointer, dataEnd - file.filePointer)
                        file.seek(dataEnd)
                    } else {
                        skipField(file, tag, tensorEnd)
                    }
                }

                else -> skipField(file, tag, tensorEnd)
            }
        }

        if (file.filePointer != tensorEnd) {
            throw OnnxFormatException("TensorProto ended outside its declared boundary")
        }
        return TensorMetadata(
            name = name,
            dimensions = dimensions.toLongArray(),
            dataType = dataType,
            rawData = rawData,
            floatData = floatData,
        )
    }

    private fun validateTensor(
        tensor: TensorMetadata,
        initializerName: String,
        expectedDimensions: LongArray,
        expectedElementCount: Int,
    ) {
        if (!tensor.dimensions.contentEquals(expectedDimensions)) {
            throw IllegalArgumentException(
                "Initializer '$initializerName' dimensions ${tensor.dimensions.contentToString()} " +
                    "do not match expected ${expectedDimensions.contentToString()}",
            )
        }
        if (tensor.dataType != ONNX_FLOAT_DATA_TYPE) {
            throw IllegalArgumentException(
                "Initializer '$initializerName' has ONNX data_type ${tensor.dataType}; expected FLOAT (1)",
            )
        }
        if (tensor.rawData != null && tensor.floatData.isNotEmpty()) {
            throw OnnxFormatException(
                "Initializer '$initializerName' contains both raw_data and float_data",
            )
        }

        val expectedBytes = expectedElementCount.toLong() * FLOAT_BYTES
        val actualBytes = tensor.rawData?.byteCount ?: tensor.floatData.sumOf { it.byteCount }
        if (actualBytes != expectedBytes) {
            throw OnnxFormatException(
                "Initializer '$initializerName' contains $actualBytes data bytes; expected $expectedBytes",
            )
        }
    }

    private fun readTensorValues(
        file: RandomAccessFile,
        tensor: TensorMetadata,
        elementCount: Int,
    ): FloatArray {
        val segments = tensor.rawData?.let(::listOf) ?: tensor.floatData
        val values = FloatArray(elementCount)
        val buffer = ByteArray(READ_BUFFER_BYTES)
        var outputIndex = 0

        for (segment in segments) {
            file.seek(segment.offset)
            var bytesRemaining = segment.byteCount
            while (bytesRemaining > 0L) {
                val bytesToRead = minOf(bytesRemaining, buffer.size.toLong()).toInt()
                file.readFully(buffer, 0, bytesToRead)
                var offset = 0
                while (offset < bytesToRead) {
                    val bits =
                        (buffer[offset].toInt() and 0xff) or
                            ((buffer[offset + 1].toInt() and 0xff) shl 8) or
                            ((buffer[offset + 2].toInt() and 0xff) shl 16) or
                            ((buffer[offset + 3].toInt() and 0xff) shl 24)
                    values[outputIndex++] = Float.fromBits(bits)
                    offset += FLOAT_BYTES
                }
                bytesRemaining -= bytesToRead
            }
        }
        return values
    }

    private fun checkedElementCount(dimensions: LongArray): Int {
        var count = 1L
        for (dimension in dimensions) {
            require(dimension >= 0L) { "Expected dimensions must be non-negative" }
            if (dimension != 0L && count > Int.MAX_VALUE.toLong() / dimension) {
                throw IllegalArgumentException("Expected tensor is too large for a FloatArray")
            }
            count *= dimension
        }
        return count.toInt()
    }

    private fun readString(file: RandomAccessFile, limit: Long, context: String): String {
        val end = readDelimitedEnd(file, limit, context)
        val byteCount = end - file.filePointer
        if (byteCount > MAX_NAME_BYTES) {
            throw OnnxFormatException("$context is unexpectedly large: $byteCount bytes")
        }
        val bytes = ByteArray(byteCount.toInt())
        file.readFully(bytes)
        return bytes.toString(Charsets.UTF_8)
    }

    private fun readTag(file: RandomAccessFile, limit: Long): ProtoTag? {
        if (file.filePointer == limit) return null
        if (file.filePointer > limit) {
            throw OnnxFormatException("Protobuf cursor passed its enclosing boundary")
        }
        val rawTag = readVarint(file, limit, "protobuf tag")
        if (rawTag <= 0L || rawTag > Int.MAX_VALUE.toLong()) {
            throw OnnxFormatException("Invalid protobuf tag $rawTag")
        }
        val wireType = (rawTag and 0x7L).toInt()
        val fieldNumber = (rawTag ushr 3).toInt()
        if (fieldNumber == 0) throw OnnxFormatException("Protobuf field number must not be zero")
        return ProtoTag(fieldNumber, wireType)
    }

    private fun readDelimitedEnd(file: RandomAccessFile, limit: Long, context: String): Long {
        val byteCount = readVarint(file, limit, "$context length")
        ensureAvailable(file, byteCount, limit, context)
        return file.filePointer + byteCount
    }

    private fun readVarint(file: RandomAccessFile, limit: Long, context: String): Long {
        var result = 0L
        var shift = 0
        while (shift < 64) {
            ensureAvailable(file, 1L, limit, context)
            val byte = file.readUnsignedByte()
            if (shift == 63 && byte and 0xfe != 0) {
                throw OnnxFormatException("$context contains an overflowing varint")
            }
            result = result or ((byte and 0x7f).toLong() shl shift)
            if (byte and 0x80 == 0) return result
            shift += 7
        }
        throw OnnxFormatException("$context contains an unterminated varint")
    }

    private fun skipField(file: RandomAccessFile, tag: ProtoTag, limit: Long) {
        when (tag.wireType) {
            WireType.VARINT -> readVarint(file, limit, "field ${tag.fieldNumber}")
            WireType.FIXED64 -> skipBytes(file, 8L, limit, tag.fieldNumber)
            WireType.LENGTH_DELIMITED -> {
                val end = readDelimitedEnd(file, limit, "field ${tag.fieldNumber}")
                file.seek(end)
            }
            WireType.START_GROUP -> skipGroup(file, tag.fieldNumber, limit)
            WireType.END_GROUP -> throw OnnxFormatException("Unexpected protobuf end-group tag")
            WireType.FIXED32 -> skipBytes(file, 4L, limit, tag.fieldNumber)
            else -> throw OnnxFormatException("Unsupported protobuf wire type ${tag.wireType}")
        }
    }

    private fun skipGroup(file: RandomAccessFile, fieldNumber: Int, limit: Long) {
        while (true) {
            val nested = readTag(file, limit)
                ?: throw OnnxFormatException("Unterminated protobuf group $fieldNumber")
            if (nested.wireType == WireType.END_GROUP) {
                if (nested.fieldNumber != fieldNumber) {
                    throw OnnxFormatException(
                        "Protobuf group $fieldNumber ended by field ${nested.fieldNumber}",
                    )
                }
                return
            }
            skipField(file, nested, limit)
        }
    }

    private fun skipBytes(
        file: RandomAccessFile,
        byteCount: Long,
        limit: Long,
        fieldNumber: Int,
    ) {
        ensureAvailable(file, byteCount, limit, "field $fieldNumber")
        file.seek(file.filePointer + byteCount)
    }

    private fun ensureAvailable(
        file: RandomAccessFile,
        byteCount: Long,
        limit: Long,
        context: String,
    ) {
        val position = file.filePointer
        if (byteCount < 0L || position > limit || byteCount > limit - position) {
            throw OnnxFormatException("$context extends past its enclosing protobuf message")
        }
    }

    private data class ProtoTag(val fieldNumber: Int, val wireType: Int)

    private data class DataSegment(val offset: Long, val byteCount: Long)

    private data class TensorMetadata(
        val name: String?,
        val dimensions: LongArray,
        val dataType: Int,
        val rawData: DataSegment?,
        val floatData: List<DataSegment>,
    )

    private object WireType {
        const val VARINT = 0
        const val FIXED64 = 1
        const val LENGTH_DELIMITED = 2
        const val START_GROUP = 3
        const val END_GROUP = 4
        const val FIXED32 = 5
    }
}

class OnnxFormatException(message: String) : IOException(message)
