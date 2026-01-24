//! Backend error types.

use thiserror::Error;

/// Errors that can occur during backend execution.
#[derive(Debug, Error)]
pub enum BackendError {
    /// Node is not supported by this backend.
    #[error("node not supported by this backend")]
    Unsupported,

    /// Execution failed.
    #[error("execution failed: {0}")]
    ExecutionFailed(String),

    /// Data transfer failed.
    #[error("data transfer failed: {0}")]
    TransferFailed(String),

    /// No backend available for the requested operation.
    #[error("no backend available")]
    NoBackendAvailable,

    /// Named backend not found.
    #[error("backend not found: {0}")]
    BackendNotFound(String),

    /// Graph error from core.
    #[error("graph error: {0}")]
    GraphError(#[from] unshape_core::GraphError),
}
