//! Error types for resin-core.

use crate::value::ValueType;
use thiserror::Error;

/// Error when a value has the wrong type.
#[derive(Debug, Clone, Error)]
#[error("type error: expected {expected}, got {got}")]
pub struct TypeError {
    pub expected: ValueType,
    pub got: ValueType,
}

impl TypeError {
    pub fn expected(expected: ValueType, got: ValueType) -> Self {
        Self { expected, got }
    }
}

/// Errors that can occur during graph operations.
#[derive(Debug, Clone, Error)]
pub enum GraphError {
    #[error("node not found: {0}")]
    NodeNotFound(u32),

    #[error("port not found: node {node}, port {port}")]
    PortNotFound { node: u32, port: usize },

    #[error("type mismatch on edge: expected {expected}, got {got}")]
    TypeMismatch { expected: ValueType, got: ValueType },

    #[error("cycle detected in graph")]
    CycleDetected,

    #[error("unconnected input: node {node}, port {port}")]
    UnconnectedInput { node: u32, port: usize },

    #[error("execution error: {0}")]
    ExecutionError(String),
}
