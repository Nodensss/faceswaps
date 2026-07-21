package com.faceswaplocal.app.inference

enum class ModelId(val stableId: String) {
    YOLOFACE_8N("yoloface_8n"),
    ARCFACE_W600K_R50("arcface_w600k_r50"),
    HYPERSWAP_1A_256("hyperswap_1a_256"),
    INSWAPPER_128_FP16("inswapper_128_fp16"),
}

enum class ModelRole {
    FACE_DETECTOR,
    FACE_RECOGNIZER,
    FACE_SWAPPER,
    FALLBACK_FACE_SWAPPER,
}

data class ModelDescriptor(
    val id: ModelId,
    val role: ModelRole,
    val fileName: String,
    val expectedSizeBytes: Long,
    val expectedSha256: String,
) {
    init {
        require(fileName == "${id.stableId}.onnx") {
            "The private model filename must be derived from its stable identifier"
        }
        require(expectedSizeBytes > 0L) { "Expected model size must be positive" }
        require(SHA_256_PATTERN.matches(expectedSha256)) {
            "Expected model checksum must be a lowercase SHA-256 value"
        }
    }

    private companion object {
        val SHA_256_PATTERN = Regex("[0-9a-f]{64}")
    }
}

/**
 * Immutable allowlist of model files accepted by the application.
 *
 * Filenames, sizes, and checksums come from the primary sources documented in
 * `docs/MODEL_CARD.md`. Imported display names are deliberately ignored.
 */
object ModelCatalog {
    val yoloface8n = ModelDescriptor(
        id = ModelId.YOLOFACE_8N,
        role = ModelRole.FACE_DETECTOR,
        fileName = "yoloface_8n.onnx",
        expectedSizeBytes = 12_659_761L,
        expectedSha256 = "821cdbb1e65fbbabdde7dd0933f754797a343e56fd962729c61ffcefcd135929",
    )

    val arcfaceW600kR50 = ModelDescriptor(
        id = ModelId.ARCFACE_W600K_R50,
        role = ModelRole.FACE_RECOGNIZER,
        fileName = "arcface_w600k_r50.onnx",
        expectedSizeBytes = 174_388_474L,
        expectedSha256 = "f1f79dc3b0b79a69f94799af1fffebff09fbd78fd96a275fd8f0cbbea23270d1",
    )

    val hyperswap1a256 = ModelDescriptor(
        id = ModelId.HYPERSWAP_1A_256,
        role = ModelRole.FACE_SWAPPER,
        fileName = "hyperswap_1a_256.onnx",
        expectedSizeBytes = 402_742_682L,
        expectedSha256 = "c0e98a8a03a238f461ed3d2570e426b49f46745ee400854a60dceeb70c246add",
    )

    val inswapper128Fp16 = ModelDescriptor(
        id = ModelId.INSWAPPER_128_FP16,
        role = ModelRole.FALLBACK_FACE_SWAPPER,
        fileName = "inswapper_128_fp16.onnx",
        expectedSizeBytes = 277_680_829L,
        expectedSha256 = "c4eccca86ad177586c85c28bf1a64a9d9ed237e283a15818d831f7facfd3f420",
    )

    val all: List<ModelDescriptor> = listOf(
        yoloface8n,
        arcfaceW600kR50,
        hyperswap1a256,
        inswapper128Fp16,
    )

    private val byId = all.associateBy(ModelDescriptor::id)

    fun descriptor(id: ModelId): ModelDescriptor = checkNotNull(byId[id]) {
        "No model descriptor registered for ${id.stableId}"
    }
}
