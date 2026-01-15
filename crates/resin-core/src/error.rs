//! Error types for resin-core.

use crate::value::ValueType;
use thiserror::Error;

/// Error when a value has the wrong type.
#[derive(Debug, Clone, Error)]
#[error("type error: expected {expected}, got {got}")]
pub struct TypeError {
    /// The type that was expected.
    pub expected: ValueType,
    /// The type that was actually provided.
    pub got: ValueType,
}

impl TypeError {
    /// Create a new type error.
    pub fn expected(expected: ValueType, got: ValueType) -> Self {
        Self { expected, got }
    }
}

/// Errors that can occur during graph operations.
#[derive(Debug, Clone, Error)]
pub enum GraphError {
    /// Node with the given ID was not found.
    #[error("node not found: {0}")]
    NodeNotFound(u32),

    /// Port on a node was not found.
    #[error("port not found: node {node}, port {port}")]
    PortNotFound {
        /// Node ID.
        node: u32,
        /// Port index.
        port: usize,
    },

    /// Type mismatch when connecting ports.
    #[error("type mismatch on wire: expected {expected}, got {got}")]
    TypeMismatch {
        /// Expected type.
        expected: ValueType,
        /// Actual type.
        got: ValueType,
    },

    /// Graph contains a cycle.
    #[error("cycle detected in graph")]
    CycleDetected,

    /// Required input port is not connected.
    #[error("unconnected input: node {node}, port {port}")]
    UnconnectedInput {
        /// Node ID.
        node: u32,
        /// Port index.
        port: usize,
    },

    /// Error during node execution.
    #[error("execution error: {0}")]
    ExecutionError(String),

    /// Node with the given ID already exists.
    #[error("node already exists: {0}")]
    NodeAlreadyExists(u32),

    /// Wire was not found.
    #[error("wire not found")]
    WireNotFound,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_type_error_display() {
        let err = TypeError::expected(ValueType::F32, ValueType::Bool);
        let msg = err.to_string();
        assert!(msg.contains("f32"));
        assert!(msg.contains("bool"));
    }

    #[test]
    fn test_type_error_fields() {
        let err = TypeError::expected(ValueType::Vec3, ValueType::I32);
        assert_eq!(err.expected, ValueType::Vec3);
        assert_eq!(err.got, ValueType::I32);
    }

    #[test]
    fn test_graph_error_node_not_found() {
        let err = GraphError::NodeNotFound(42);
        assert!(err.to_string().contains("42"));
    }

    #[test]
    fn test_graph_error_port_not_found() {
        let err = GraphError::PortNotFound { node: 1, port: 2 };
        let msg = err.to_string();
        assert!(msg.contains("1"));
        assert!(msg.contains("2"));
    }

    #[test]
    fn test_graph_error_type_mismatch() {
        let err = GraphError::TypeMismatch {
            expected: ValueType::F32,
            got: ValueType::Vec3,
        };
        let msg = err.to_string();
        assert!(msg.contains("f32"));
        assert!(msg.contains("Vec3"));
    }

    #[test]
    fn test_graph_error_cycle_detected() {
        let err = GraphError::CycleDetected;
        assert!(err.to_string().contains("cycle"));
    }

    #[test]
    fn test_graph_error_unconnected_input() {
        let err = GraphError::UnconnectedInput { node: 5, port: 0 };
        let msg = err.to_string();
        assert!(msg.contains("5"));
        assert!(msg.contains("0"));
    }

    #[test]
    fn test_graph_error_execution_error() {
        let err = GraphError::ExecutionError("something went wrong".to_string());
        assert!(err.to_string().contains("something went wrong"));
    }
}
