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

    /// Wire endpoint string has an invalid format.
    #[error("invalid wire format: {0}")]
    InvalidWireFormat(String),

    /// A graph saved under the removed feedback-edge recurrence model.
    ///
    /// The `feedback` wire flag and `connect_recurrence` were replaced by an
    /// explicit `Latch` node (`docs/design/recurrent-graphs.md`). Such graphs
    /// must be re-authored with a `Latch` rather than silently loaded.
    #[error(
        "graph uses the removed feedback-edge recurrence model (wire {0} has `feedback: true`); \
         migrate it to an explicit core::Latch node (see docs/design/recurrent-graphs.md)"
    )]
    LegacyFeedbackWire(String),
}
