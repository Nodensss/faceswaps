package com.faceswaplocal.app.inference

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class ModelCatalogTest {
    @Test
    fun `catalog has one descriptor for every stable model id`() {
        assertEquals(ModelId.entries.size, ModelCatalog.all.size)
        assertEquals(ModelCatalog.all.size, ModelCatalog.all.map { it.id }.distinct().size)
        assertEquals(ModelCatalog.all.size, ModelCatalog.all.map { it.fileName }.distinct().size)

        ModelId.entries.forEach { id ->
            assertEquals(id, ModelCatalog.descriptor(id).id)
        }
    }

    @Test
    fun `catalog checksums and sizes match the documented allowlist`() {
        val expected = mapOf(
            ModelId.YOLOFACE_8N to Pair(
                12_659_761L,
                "821cdbb1e65fbbabdde7dd0933f754797a343e56fd962729c61ffcefcd135929",
            ),
            ModelId.ARCFACE_W600K_R50 to Pair(
                174_388_474L,
                "f1f79dc3b0b79a69f94799af1fffebff09fbd78fd96a275fd8f0cbbea23270d1",
            ),
            ModelId.HYPERSWAP_1A_256 to Pair(
                402_742_682L,
                "c0e98a8a03a238f461ed3d2570e426b49f46745ee400854a60dceeb70c246add",
            ),
            ModelId.INSWAPPER_128_FP16 to Pair(
                277_680_829L,
                "c4eccca86ad177586c85c28bf1a64a9d9ed237e283a15818d831f7facfd3f420",
            ),
        )

        expected.forEach { (id, expectedIntegrity) ->
            val descriptor = ModelCatalog.descriptor(id)
            assertEquals(expectedIntegrity.first, descriptor.expectedSizeBytes)
            assertEquals(expectedIntegrity.second, descriptor.expectedSha256)
            assertTrue(descriptor.expectedSha256.matches(Regex("[0-9a-f]{64}")))
        }
    }
}
