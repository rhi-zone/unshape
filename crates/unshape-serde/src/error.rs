//! Serialization error types.

use thiserror::Error;

/// Errors that can occur during graph serialization/deserialization.
#[derive(Debug, Error)]
pub enum SerdeError {
    /// Unknown node type encountered during deserialization.
    #[error("unknown node type: {0}")]
    UnknownNodeType(String),

    /// JSON serialization/deserialization error.
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),

    /// Bincode serialization/deserialization error.
    #[error("bincode error: {0}")]
    Bincode(#[from] bincode::error::DecodeError),

    /// Bincode encoding error.
    #[error("bincode encode error: {0}")]
    BincodeEncode(#[from] bincode::error::EncodeError),

    /// Graph error during reconstruction.
    #[error("graph error: {0}")]
    Graph(#[from] unshape_core::GraphError),

    /// Node does not implement SerializableNode.
    #[error("node type '{0}' does not support serialization")]
    NotSerializable(String),
}
