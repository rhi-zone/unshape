//! History error types.

use thiserror::Error;

/// Errors that can occur during history operations.
#[derive(Debug, Error)]
pub enum HistoryError {
    /// Serialization error from resin-serde.
    #[error("serialization error: {0}")]
    Serde(#[from] rhizome_resin_serde::SerdeError),

    /// Graph operation error.
    #[error("graph error: {0}")]
    Graph(#[from] rhizome_resin_core::GraphError),

    /// No more undo steps available.
    #[error("nothing to undo")]
    NothingToUndo,

    /// No more redo steps available.
    #[error("nothing to redo")]
    NothingToRedo,

    /// Node not found during event application.
    #[error("node not found: {0}")]
    NodeNotFound(u32),

    /// Edge not found during event application.
    #[error("edge not found")]
    EdgeNotFound,

    /// Invalid event (e.g., applying inverse without original).
    #[error("invalid event: {0}")]
    InvalidEvent(String),
}
