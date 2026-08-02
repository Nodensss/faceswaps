package com.faceswaplocal.app.inference

/**
 * Diagnostic hook for proving heavyweight ONNX session lifetime without exposing an
 * OrtSession outside the inference package. Production uses the no-op default; parity
 * instrumentation records the events and verifies the two-pass barrier from real runs.
 */
interface InferenceSessionLifecycleListener {
    fun onSessionOpened(modelFileName: String) = Unit

    fun onSessionClosed(modelFileName: String) = Unit
}

internal object NoOpInferenceSessionLifecycleListener : InferenceSessionLifecycleListener
