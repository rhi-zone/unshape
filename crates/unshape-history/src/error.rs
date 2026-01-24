//! History error types.

use thiserror::Error;

/// Errors that can occur during history operations.
#[derive(Debug, Error)]
pub enum HistoryError {
    /// Serialization error from resin-serde.
    #[error("serialization error: {0}")]
    Serde(#[from] unshape_serde::SerdeError),

    /// Graph operation error.
    #[error("graph error: {0}")]
    Graph(#[from] unshape_core::GraphError),

    /// No more undo steps available.
    #[error("nothing to undo")]
    NothingToUndo,

    /// No more redo steps available.
    #[error("nothing to redo")]
    NothingToRedo,

    /// Node not found during event application.
    #[error("node not found: {0}")]
    NodeNotFound(u32),

    /// Wire not found during event application.
    #[error("wire not found")]
    WireNotFound,

    /// Invalid event (e.g., applying inverse without original).
    #[error("invalid event: {0}")]
    InvalidEvent(String),
}
