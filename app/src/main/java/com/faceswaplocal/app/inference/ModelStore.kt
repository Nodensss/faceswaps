package com.faceswaplocal.app.inference

import android.content.ContentResolver
import android.content.Context
import android.net.Uri
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.io.IOException
import java.io.InputStream
import java.io.OutputStream
import java.nio.file.AtomicMoveNotSupportedException
import java.nio.file.Files
import java.nio.file.StandardCopyOption
import java.security.MessageDigest
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext

enum class ModelValidationFailure {
    SIZE_MISMATCH,
    CHECKSUM_MISMATCH,
}

data class ModelValidationDetails(
    val reason: ModelValidationFailure,
    val expectedSizeBytes: Long,
    val actualSizeBytes: Long,
    val expectedSha256: String,
    val actualSha256: String,
    val existingCopyRetained: Boolean,
)

sealed interface ModelStatus {
    data object Missing : ModelStatus

    /** A private model copy exists but has not been verified during this process. */
    data object PresentUnverified : ModelStatus

    data class Importing(
        val bytesCopied: Long,
        val expectedBytes: Long,
    ) : ModelStatus

    data class Ready(
        val verifiedSizeBytes: Long,
    ) : ModelStatus

    data class Invalid(val details: ModelValidationDetails) : ModelStatus

    /** No source URI or private filesystem path is included in this UI-safe status. */
    data class Failed(
        val reason: ModelStoreFailure,
        val existingCopyRetained: Boolean,
    ) : ModelStatus
}

enum class ModelStoreFailure {
    SOURCE_UNAVAILABLE,
    READ_FAILED,
    PRIVATE_STORAGE_UNAVAILABLE,
    ATOMIC_INSTALL_UNAVAILABLE,
}

sealed interface ModelImportResult {
    data class Imported(
        val id: ModelId,
        val verifiedSizeBytes: Long,
    ) : ModelImportResult

    data class Rejected(
        val id: ModelId,
        val details: ModelValidationDetails,
    ) : ModelImportResult

    data class Failed(
        val id: ModelId,
        val reason: ModelStoreFailure,
        val existingCopyRetained: Boolean,
    ) : ModelImportResult
}

class ModelUnavailableException(
    val id: ModelId,
    val reason: ModelStatus,
) : IllegalStateException("Model ${id.stableId} is not available") {
    val validationDetails: ModelValidationDetails? =
        (reason as? ModelStatus.Invalid)?.details
}

/**
 * Imports allowlisted ONNX files into app-private storage.
 *
 * The picker URI is consumed as a one-shot stream and is never retained or persisted.
 * A candidate is copied to a `.part` file, checked by size and SHA-256, fsynced, and
 * atomically renamed over the installed copy only after successful validation.
 */
class ModelStore(
    context: Context,
    private val contentResolver: ContentResolver = context.contentResolver,
    private val ioDispatcher: CoroutineDispatcher = Dispatchers.IO,
) {
    private val modelDirectory = File(context.filesDir, MODEL_DIRECTORY_NAME)
    private val operationMutex = Mutex()
    private val mutableStatuses = MutableStateFlow(initialStatuses())

    val statuses: StateFlow<Map<ModelId, ModelStatus>> = mutableStatuses.asStateFlow()

    init {
        cleanupStalePartFiles()
    }

    suspend fun importModel(id: ModelId, source: Uri): ModelImportResult =
        withContext(ioDispatcher) {
            operationMutex.withLock {
                importLocked(ModelCatalog.descriptor(id), source)
            }
        }

    /** Revalidates every installed private model without exposing its filesystem path. */
    suspend fun refreshStatuses(): Map<ModelId, ModelStatus> = withContext(ioDispatcher) {
        operationMutex.withLock {
            ModelCatalog.all.forEach { descriptor -> verifyStatusLocked(descriptor) }
            statuses.value
        }
    }

    /** Deletes only interrupted import candidates; installed models are never removed. */
    suspend fun cleanupInterruptedImports(): Int = withContext(ioDispatcher) {
        operationMutex.withLock { cleanupStalePartFiles() }
    }

    /**
     * Returns a model file only after re-reading and validating the complete private copy.
     * Call this immediately before creating an inference session.
     */
    suspend fun requireVerifiedModel(id: ModelId): File = withContext(ioDispatcher) {
        operationMutex.withLock {
            val descriptor = ModelCatalog.descriptor(id)
            val destination = modelFile(descriptor)
            val status = verifyStatusLocked(descriptor)
            if (status !is ModelStatus.Ready) {
                throw ModelUnavailableException(id, status)
            }
            destination
        }
    }

    private suspend fun importLocked(
        descriptor: ModelDescriptor,
        source: Uri,
    ): ModelImportResult {
        if (!ensureModelDirectory()) {
            return failedResult(descriptor, ModelStoreFailure.PRIVATE_STORAGE_UNAVAILABLE)
        }

        val destination = modelFile(descriptor)
        val partFile = try {
            File.createTempFile("${descriptor.id.stableId}.", PART_FILE_SUFFIX, modelDirectory)
        } catch (_: IOException) {
            return failedResult(descriptor, ModelStoreFailure.PRIVATE_STORAGE_UNAVAILABLE)
        }

        return try {
            setStatus(
                descriptor.id,
                ModelStatus.Importing(bytesCopied = 0L, expectedBytes = descriptor.expectedSizeBytes),
            )
            val observation = copySourceToPart(source, partFile, descriptor)
                ?: return failedResult(descriptor, ModelStoreFailure.SOURCE_UNAVAILABLE)

            val validationDetails = ModelFileIntegrity.validationDetails(
                descriptor = descriptor,
                observation = observation,
                existingCopyRetained = destination.isFile,
            )
            if (validationDetails != null) {
                restoreRetainedStatusOr(
                    descriptor = descriptor,
                    replacementFailure = ModelStatus.Invalid(validationDetails),
                )
                return ModelImportResult.Rejected(
                    id = descriptor.id,
                    details = validationDetails,
                )
            }

            try {
                installAtomically(partFile, destination)
            } catch (_: AtomicMoveNotSupportedException) {
                return failedResult(descriptor, ModelStoreFailure.ATOMIC_INSTALL_UNAVAILABLE)
            } catch (_: IOException) {
                return failedResult(descriptor, ModelStoreFailure.PRIVATE_STORAGE_UNAVAILABLE)
            }

            setStatus(descriptor.id, ModelStatus.Ready(observation.sizeBytes))
            ModelImportResult.Imported(descriptor.id, observation.sizeBytes)
        } catch (cancelled: CancellationException) {
            setStatus(
                descriptor.id,
                if (destination.isFile) ModelStatus.PresentUnverified else ModelStatus.Missing,
            )
            throw cancelled
        } catch (_: SecurityException) {
            failedResult(descriptor, ModelStoreFailure.SOURCE_UNAVAILABLE)
        } catch (_: IOException) {
            failedResult(descriptor, ModelStoreFailure.READ_FAILED)
        } finally {
            if (partFile.exists()) {
                partFile.delete()
            }
        }
    }

    private suspend fun copySourceToPart(
        source: Uri,
        partFile: File,
        descriptor: ModelDescriptor,
    ): ModelFileObservation? {
        val input = contentResolver.openInputStream(source) ?: return null
        input.use {
            FileOutputStream(partFile).use { output ->
                val coroutineContext = currentCoroutineContext()
                val observation = ModelFileIntegrity.copyAndHash(
                    input = input,
                    output = output,
                    onChunk = { copiedBytes ->
                        coroutineContext.ensureActive()
                        setStatus(
                            descriptor.id,
                            ModelStatus.Importing(copiedBytes, descriptor.expectedSizeBytes),
                        )
                    },
                )
                output.fd.sync()
                return observation
            }
        }
    }

    private fun installAtomically(partFile: File, destination: File) {
        Files.move(
            partFile.toPath(),
            destination.toPath(),
            StandardCopyOption.ATOMIC_MOVE,
            StandardCopyOption.REPLACE_EXISTING,
        )
    }

    private fun failedResult(
        descriptor: ModelDescriptor,
        failure: ModelStoreFailure,
    ): ModelImportResult.Failed {
        val retained = modelFile(descriptor).isFile
        restoreRetainedStatusOr(
            descriptor = descriptor,
            replacementFailure = ModelStatus.Failed(failure, retained),
        )
        return ModelImportResult.Failed(descriptor.id, failure, retained)
    }

    /**
     * A failed candidate must not make a previously installed, valid model unavailable.
     * Re-read the retained copy because its last Ready state may belong to an earlier
     * process or may no longer describe the bytes currently on disk.
     */
    private fun restoreRetainedStatusOr(
        descriptor: ModelDescriptor,
        replacementFailure: ModelStatus,
    ) {
        val destination = modelFile(descriptor)
        val retainedStatus = if (destination.isFile) {
            verifyStatusLocked(descriptor)
        } else {
            null
        }
        setStatus(
            descriptor.id,
            replacementImportVisibleStatus(replacementFailure, retainedStatus),
        )
    }

    private fun verifyStatusLocked(descriptor: ModelDescriptor): ModelStatus {
        val destination = modelFile(descriptor)
        if (!destination.isFile) {
            return ModelStatus.Missing.also { setStatus(descriptor.id, it) }
        }

        val observation = try {
            ModelFileIntegrity.hash(destination)
        } catch (_: IOException) {
            return ModelStatus.Failed(
                reason = ModelStoreFailure.READ_FAILED,
                existingCopyRetained = destination.exists(),
            ).also { setStatus(descriptor.id, it) }
        } catch (_: SecurityException) {
            return ModelStatus.Failed(
                reason = ModelStoreFailure.READ_FAILED,
                existingCopyRetained = destination.exists(),
            ).also { setStatus(descriptor.id, it) }
        }
        val details = ModelFileIntegrity.validationDetails(
            descriptor = descriptor,
            observation = observation,
            existingCopyRetained = true,
        )
        return if (details == null) {
            ModelStatus.Ready(observation.sizeBytes)
        } else {
            ModelStatus.Invalid(details)
        }.also { setStatus(descriptor.id, it) }
    }

    private fun initialStatuses(): Map<ModelId, ModelStatus> = ModelCatalog.all.associate { descriptor ->
        descriptor.id to if (modelFile(descriptor).isFile) {
            ModelStatus.PresentUnverified
        } else {
            ModelStatus.Missing
        }
    }

    private fun setStatus(id: ModelId, status: ModelStatus) {
        mutableStatuses.update { current -> current + (id to status) }
    }

    private fun modelFile(descriptor: ModelDescriptor): File =
        File(modelDirectory, descriptor.fileName)

    private fun ensureModelDirectory(): Boolean =
        modelDirectory.isDirectory || modelDirectory.mkdirs()

    private fun cleanupStalePartFiles(): Int {
        val partFiles = modelDirectory
            .takeIf(File::isDirectory)
            ?.listFiles { file -> file.isFile && file.name.endsWith(PART_FILE_SUFFIX) }
            .orEmpty()
        return partFiles.count(File::delete)
    }

    private companion object {
        const val MODEL_DIRECTORY_NAME = "models"
        const val PART_FILE_SUFFIX = ".part"
    }
}

/**
 * Selects the status that controls inference after a replacement attempt. The import
 * result still reports the rejected/failed candidate, while a revalidated retained
 * copy remains independently usable.
 */
internal fun replacementImportVisibleStatus(
    replacementFailure: ModelStatus,
    retainedCopyStatus: ModelStatus?,
): ModelStatus = retainedCopyStatus ?: replacementFailure

internal data class ModelFileObservation(
    val sizeBytes: Long,
    val sha256: String,
)

internal object ModelFileIntegrity {
    private const val BUFFER_SIZE_BYTES = 1024 * 1024

    fun hash(file: File): ModelFileObservation = FileInputStream(file).use { input ->
        copyAndHash(input, NullOutputStream)
    }

    fun copyAndHash(
        input: InputStream,
        output: OutputStream,
        onChunk: (Long) -> Unit = {},
    ): ModelFileObservation {
        val digest = MessageDigest.getInstance("SHA-256")
        val buffer = ByteArray(BUFFER_SIZE_BYTES)
        var sizeBytes = 0L

        while (true) {
            val bytesRead = input.read(buffer)
            if (bytesRead == -1) break
            if (bytesRead == 0) continue

            output.write(buffer, 0, bytesRead)
            digest.update(buffer, 0, bytesRead)
            sizeBytes += bytesRead
            onChunk(sizeBytes)
        }

        return ModelFileObservation(
            sizeBytes = sizeBytes,
            sha256 = digest.digest().joinToString(separator = "") { byte -> "%02x".format(byte) },
        )
    }

    fun validationDetails(
        descriptor: ModelDescriptor,
        observation: ModelFileObservation,
        existingCopyRetained: Boolean,
    ): ModelValidationDetails? {
        val reason = when {
            observation.sizeBytes != descriptor.expectedSizeBytes -> ModelValidationFailure.SIZE_MISMATCH
            observation.sha256 != descriptor.expectedSha256 -> ModelValidationFailure.CHECKSUM_MISMATCH
            else -> return null
        }
        return ModelValidationDetails(
            reason = reason,
            expectedSizeBytes = descriptor.expectedSizeBytes,
            actualSizeBytes = observation.sizeBytes,
            expectedSha256 = descriptor.expectedSha256,
            actualSha256 = observation.sha256,
            existingCopyRetained = existingCopyRetained,
        )
    }

    private data object NullOutputStream : OutputStream() {
        override fun write(value: Int) = Unit

        override fun write(buffer: ByteArray, offset: Int, length: Int) = Unit
    }
}
