package com.faceswaplocal.app.inference

/**
 * Diagnostic hook for proving that every ArcFace embedding the coordinator produces is
 * zeroed once its run ends, without exposing the coordinator's internal cleanup list.
 * Production uses the no-op default; instrumentation captures the array reference and
 * re-checks its content after `process` returns or the run is cancelled.
 */
interface EmbeddingLifecycleListener {
    /**
     * Called once per embedding array the coordinator will later zero in `finally`,
     * including copies `putIfAbsent` rejected as duplicates — those are zeroed too, and
     * this hook is how a test proves it.
     */
    fun onEmbeddingProduced(embedding: FloatArray) = Unit
}

internal object NoOpEmbeddingLifecycleListener : EmbeddingLifecycleListener
