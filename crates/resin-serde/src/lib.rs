//! Serialization for resin graphs.
//!
//! This crate provides serialization and deserialization for resin graphs,
//! supporting multiple formats (JSON, bincode) and registry-based node
//! reconstruction.
//!
//! # Overview
//!
//! Graphs contain trait objects (`Box<dyn DynNode>`) which cannot be directly
//! serialized. This crate solves this by:
//!
//! 1. Converting graphs to an intermediate `SerialGraph` format where nodes
//!    are represented as `(type_name, params_json)` pairs
//! 2. Using a `NodeRegistry` to map type names back to concrete deserializers
//!
//! # Example
//!
//! ```ignore
//! use resin_serde::{NodeRegistry, JsonFormat, serialize_graph, deserialize_graph};
//!
//! // Register node types
//! let mut registry = NodeRegistry::new();
//! registry.register_with_name::<MyNode>("my::Node");
//!
//! // Serialize
//! let format = JsonFormat::pretty();
//! let bytes = serialize_graph(&graph, &registry, &format)?;
//!
//! // Deserialize
//! let loaded = deserialize_graph(&bytes, &registry, &format)?;
//! ```

mod bincode;
mod error;
mod format;
mod json;
mod registry;
mod serial;

pub use crate::bincode::BincodeFormat;
pub use crate::error::SerdeError;
pub use crate::format::GraphFormat;
pub use crate::json::JsonFormat;
pub use crate::registry::{NodeRegistry, SerializableNode};
pub use crate::serial::{SerialGraph, SerialNode};

use rhizome_resin_core::Graph;

/// Converts a runtime `Graph` to a serializable `SerialGraph`.
///
/// Each node must implement `SerializableNode` for this to work.
/// The function iterates over all nodes and extracts their type name
/// and parameters.
///
/// # Errors
///
/// Returns `SerdeError::NotSerializable` if a node doesn't support
/// serialization (i.e., isn't in the registry or doesn't have a
/// `SerializableNode` implementation).
pub fn graph_to_serial<F>(graph: &Graph, extract_params: F) -> Result<SerialGraph, SerdeError>
where
    F: Fn(&dyn rhizome_resin_core::DynNode) -> Option<serde_json::Value>,
{
    let mut serial = SerialGraph {
        nodes: Vec::new(),
        wires: graph.wires().to_vec(),
        next_id: graph.next_id(),
    };

    for (id, node) in graph.nodes_iter() {
        let type_name = node.type_name().to_string();
        let params = extract_params(node.as_ref())
            .ok_or_else(|| SerdeError::NotSerializable(type_name.clone()))?;

        serial.nodes.push(SerialNode::new(id, type_name, params));
    }

    Ok(serial)
}

/// Converts a `SerialGraph` back to a runtime `Graph`.
///
/// Uses the registry to reconstruct nodes from their type names and parameters.
///
/// # Errors
///
/// Returns an error if:
/// - A node type is not registered
/// - Node deserialization fails
/// - Graph reconstruction fails (e.g., invalid wires)
pub fn serial_to_graph(serial: SerialGraph, registry: &NodeRegistry) -> Result<Graph, SerdeError> {
    let mut graph = Graph::with_next_id(serial.next_id);

    for serial_node in serial.nodes {
        let node = registry.deserialize(&serial_node.type_name, serial_node.params())?;
        graph.insert_node_with_id(serial_node.id, node)?;
    }

    for wire in serial.wires {
        graph.connect(wire.from_node, wire.from_port, wire.to_node, wire.to_port)?;
    }

    Ok(graph)
}

/// High-level function to serialize a graph to bytes.
///
/// Combines `graph_to_serial` and format serialization.
pub fn serialize_graph<F, Fmt>(
    graph: &Graph,
    extract_params: F,
    format: &Fmt,
) -> Result<Vec<u8>, SerdeError>
where
    F: Fn(&dyn rhizome_resin_core::DynNode) -> Option<serde_json::Value>,
    Fmt: GraphFormat,
{
    let serial = graph_to_serial(graph, extract_params)?;
    format.serialize(&serial)
}

/// High-level function to deserialize bytes to a graph.
///
/// Combines format deserialization and `serial_to_graph`.
pub fn deserialize_graph<Fmt>(
    bytes: &[u8],
    registry: &NodeRegistry,
    format: &Fmt,
) -> Result<Graph, SerdeError>
where
    Fmt: GraphFormat,
{
    let serial = format.deserialize(bytes)?;
    serial_to_graph(serial, registry)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rhizome_resin_core::{DynNode, GraphError, PortDescriptor, Value, ValueType};
    use serde::{Deserialize, Serialize};

    #[derive(Debug, Clone, Serialize, Deserialize)]
    struct ConstNode {
        value: f32,
    }

    impl DynNode for ConstNode {
        fn type_name(&self) -> &'static str {
            "test::Const"
        }

        fn inputs(&self) -> Vec<PortDescriptor> {
            vec![]
        }

        fn outputs(&self) -> Vec<PortDescriptor> {
            vec![PortDescriptor::new("value", ValueType::F32)]
        }

        fn execute(&self, _inputs: &[Value]) -> Result<Vec<Value>, GraphError> {
            Ok(vec![Value::F32(self.value)])
        }
    }

    impl SerializableNode for ConstNode {
        fn params(&self) -> serde_json::Value {
            serde_json::to_value(self).unwrap()
        }
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    struct AddNode;

    impl DynNode for AddNode {
        fn type_name(&self) -> &'static str {
            "test::Add"
        }

        fn inputs(&self) -> Vec<PortDescriptor> {
            vec![
                PortDescriptor::new("a", ValueType::F32),
                PortDescriptor::new("b", ValueType::F32),
            ]
        }

        fn outputs(&self) -> Vec<PortDescriptor> {
            vec![PortDescriptor::new("result", ValueType::F32)]
        }

        fn execute(&self, inputs: &[Value]) -> Result<Vec<Value>, GraphError> {
            let a = inputs[0]
                .as_f32()
                .map_err(|e| GraphError::ExecutionError(e.to_string()))?;
            let b = inputs[1]
                .as_f32()
                .map_err(|e| GraphError::ExecutionError(e.to_string()))?;
            Ok(vec![Value::F32(a + b)])
        }
    }

    impl SerializableNode for AddNode {
        fn params(&self) -> serde_json::Value {
            serde_json::json!({})
        }
    }

    fn make_extract_params() -> impl Fn(&dyn DynNode) -> Option<serde_json::Value> {
        |node: &dyn DynNode| -> Option<serde_json::Value> {
            match node.type_name() {
                "test::Const" => {
                    // We know it's a ConstNode, but can't downcast safely
                    // In real usage, this would use SerializableNode
                    Some(serde_json::json!({"value": 0.0}))
                }
                "test::Add" => Some(serde_json::json!({})),
                _ => None,
            }
        }
    }

    #[test]
    fn test_full_roundtrip_json() {
        // Build a simple graph
        let mut graph = Graph::new();
        let c1 = graph.add_node(ConstNode { value: 2.0 });
        let c2 = graph.add_node(ConstNode { value: 3.0 });
        let add = graph.add_node(AddNode);
        graph.connect(c1, 0, add, 0).unwrap();
        graph.connect(c2, 0, add, 1).unwrap();

        // Setup registry
        let mut registry = NodeRegistry::new();
        registry.register_factory("test::Const", |params| {
            let value = params["value"].as_f64().unwrap_or(0.0) as f32;
            Ok(Box::new(ConstNode { value }))
        });
        registry.register_factory("test::Add", |_| Ok(Box::new(AddNode)));

        // Custom extractor that properly reads the value
        let extract = |node: &dyn DynNode| -> Option<serde_json::Value> {
            match node.type_name() {
                "test::Const" => {
                    // Execute to get the value (hacky but works for test)
                    let outputs = node.execute(&[]).ok()?;
                    let value = outputs[0].as_f32().ok()?;
                    Some(serde_json::json!({"value": value}))
                }
                "test::Add" => Some(serde_json::json!({})),
                _ => None,
            }
        };

        // Serialize
        let format = JsonFormat::pretty();
        let bytes = serialize_graph(&graph, extract, &format).unwrap();

        // Check it looks reasonable
        let json_str = String::from_utf8_lossy(&bytes);
        assert!(json_str.contains("test::Const"));
        assert!(json_str.contains("test::Add"));

        // Deserialize
        let mut loaded = deserialize_graph(&bytes, &registry, &format).unwrap();

        // Verify structure
        assert_eq!(loaded.node_count(), 3);
        assert_eq!(loaded.wire_count(), 2);

        // Execute and verify result
        let result = loaded.execute(add).unwrap();
        assert_eq!(result[0].as_f32().unwrap(), 5.0);
    }

    #[test]
    fn test_full_roundtrip_bincode() {
        let mut graph = Graph::new();
        let c = graph.add_node(ConstNode { value: 42.0 });

        let mut registry = NodeRegistry::new();
        registry.register_factory("test::Const", |params| {
            let value = params["value"].as_f64().unwrap_or(0.0) as f32;
            Ok(Box::new(ConstNode { value }))
        });

        let extract = |node: &dyn DynNode| -> Option<serde_json::Value> {
            if node.type_name() == "test::Const" {
                let outputs = node.execute(&[]).ok()?;
                let value = outputs[0].as_f32().ok()?;
                Some(serde_json::json!({"value": value}))
            } else {
                None
            }
        };

        let format = BincodeFormat::new();
        let bytes = serialize_graph(&graph, extract, &format).unwrap();

        let mut loaded = deserialize_graph(&bytes, &registry, &format).unwrap();
        let result = loaded.execute(c).unwrap();
        assert_eq!(result[0].as_f32().unwrap(), 42.0);
    }
}
