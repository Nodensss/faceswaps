package com.faceswaplocal.app.inference

import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class InferenceBackendPolicyTest {
    @Test
    fun `x86 emulator ABIs use CPU before native session creation`() {
        assertFalse(mayAttemptXnnpack(arrayOf("x86_64")))
        assertFalse(mayAttemptXnnpack(arrayOf("x86")))
    }

    @Test
    fun `arm ABI remains eligible for reference-device qualification`() {
        assertTrue(mayAttemptXnnpack(arrayOf("arm64-v8a", "armeabi-v7a")))
    }
}
