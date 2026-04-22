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
mod typed_constants;

pub use crate::bincode::BincodeFormat;
pub use crate::error::SerdeError;
pub use crate::format::GraphFormat;
pub use crate::json::JsonFormat;
pub use crate::registry::{NodeRegistry, SerializableNode};
pub use crate::serial::{SerialGraph, SerialNode, SerialWire};
pub use crate::typed_constants::{ConstantImage, ConstantMesh, ImageValue, MeshValue};

use unshape_core::{ConstantNode, Graph, GraphInput, GraphOutput};

/// Register the built-in core nodes (`ConstantNode`, `GraphInput`, `GraphOutput`,
/// `ConstantMesh`, and `ConstantImage`) into a registry.
///
/// After calling this, the registry can deserialize nodes with type names:
/// - `"core::Constant"`
/// - `"core::GraphInput"`
/// - `"core::GraphOutput"`
/// - `"mesh::ConstantMesh"`
/// - `"image::ConstantImage"`
pub fn register_core_nodes(registry: &mut NodeRegistry) {
    registry.register_with_name::<ConstantNode>("core::Constant");
    registry.register_with_name::<GraphInput>("core::GraphInput");
    registry.register_with_name::<GraphOutput>("core::GraphOutput");
    registry.register_with_name::<ConstantMesh>("mesh::ConstantMesh");
    registry.register_with_name::<ConstantImage>("image::ConstantImage");
}

impl SerializableNode for ConstantNode {
    fn params(&self) -> serde_json::Value {
        serde_json::to_value(self)
            .unwrap_or_else(|e| serde_json::json!({ "__error": e.to_string() }))
    }
}

impl SerializableNode for GraphInput {
    fn params(&self) -> serde_json::Value {
        serde_json::to_value(self)
            .unwrap_or_else(|e| serde_json::json!({ "__error": e.to_string() }))
    }
}

impl SerializableNode for GraphOutput {
    fn params(&self) -> serde_json::Value {
        serde_json::to_value(self)
            .unwrap_or_else(|e| serde_json::json!({ "__error": e.to_string() }))
    }
}

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
    F: Fn(&dyn unshape_core::DynNode) -> Option<serde_json::Value>,
{
    let mut serial = SerialGraph {
        version: 1,
        nodes: Vec::new(),
        wires: graph
            .wires()
            .iter()
            .map(|w| {
                let from_node = graph.get_node(w.from_node).map(|n| n.as_ref());
                let to_node = graph.get_node(w.to_node).map(|n| n.as_ref());
                match (from_node, to_node) {
                    (Some(f), Some(t)) => SerialWire::from_wire(w, f, t),
                    _ => SerialWire {
                        from: format!("{}:out", w.from_node),
                        to: format!("{}:in", w.to_node),
                    },
                }
            })
            .collect(),
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

    for serial_wire in serial.wires {
        // Parse node IDs first to look up nodes for port name resolution.
        let from_node_id = SerialWire::node_id(&serial_wire.from)?;
        let to_node_id = SerialWire::node_id(&serial_wire.to)?;

        let from_node = graph.get_node(from_node_id).ok_or_else(|| {
            SerdeError::InvalidWireFormat(format!(
                "wire references unknown source node {}",
                from_node_id
            ))
        })?;
        let to_node = graph.get_node(to_node_id).ok_or_else(|| {
            SerdeError::InvalidWireFormat(format!(
                "wire references unknown destination node {}",
                to_node_id
            ))
        })?;

        let wire = serial_wire.to_wire(from_node.as_ref(), to_node.as_ref())?;
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
    F: Fn(&dyn unshape_core::DynNode) -> Option<serde_json::Value>,
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
    use serde::{Deserialize, Serialize};
    use unshape_core::{DynNode, EvalContext, GraphError, PortDescriptor, Value, ValueType};

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

        fn execute(&self, _inputs: &[Value], _ctx: &EvalContext) -> Result<Vec<Value>, GraphError> {
            Ok(vec![Value::F32(self.value)])
        }

        fn as_any(&self) -> &dyn std::any::Any {
            self
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

        fn execute(&self, inputs: &[Value], _ctx: &EvalContext) -> Result<Vec<Value>, GraphError> {
            let a = inputs[0]
                .as_f32()
                .map_err(|e| GraphError::ExecutionError(e.to_string()))?;
            let b = inputs[1]
                .as_f32()
                .map_err(|e| GraphError::ExecutionError(e.to_string()))?;
            Ok(vec![Value::F32(a + b)])
        }

        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
    }

    impl SerializableNode for AddNode {
        fn params(&self) -> serde_json::Value {
            serde_json::json!({})
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
                    let ctx = EvalContext::new();
                    let outputs = node.execute(&[], &ctx).ok()?;
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
                let ctx = EvalContext::new();
                let outputs = node.execute(&[], &ctx).ok()?;
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

    /// Round-trip a graph containing `ConstantNode` and `GraphInput` through JSON,
    /// then execute the deserialized graph with a named input.
    #[test]
    fn test_core_nodes_roundtrip_json() {
        use unshape_core::{ConstantNode, Graph, GraphInput, ValueType};

        // Build a graph with a ConstantNode (value=10.0) and a GraphInput
        // (name="bias", type=F32). Both are source nodes — no wires between them.
        let mut graph = Graph::new();
        let const_id = graph.add_node(ConstantNode::new(10.0f32));
        let input_id = graph.add_node(GraphInput::new("bias", ValueType::F32).with_default(5.0f32));

        // Register core nodes
        let mut registry = NodeRegistry::new();
        register_core_nodes(&mut registry);

        // Build the extract function using SerializableNode
        let extract = |node: &dyn DynNode| -> Option<serde_json::Value> {
            match node.type_name() {
                "core::Constant" => node
                    .as_any()
                    .downcast_ref::<ConstantNode>()
                    .map(|n| n.params()),
                "core::GraphInput" => node
                    .as_any()
                    .downcast_ref::<GraphInput>()
                    .map(|n| n.params()),
                _ => None,
            }
        };

        // Serialize to JSON
        let format = JsonFormat::pretty();
        let bytes = serialize_graph(&graph, extract, &format).unwrap();
        let json_str = String::from_utf8_lossy(&bytes);
        assert!(json_str.contains("core::Constant"));
        assert!(json_str.contains("core::GraphInput"));
        assert!(json_str.contains("bias"));

        // Deserialize
        let mut loaded = deserialize_graph(&bytes, &registry, &format).unwrap();
        assert_eq!(loaded.node_count(), 2);

        // Execute ConstantNode — should return 10.0
        let ctx = EvalContext::new().with_input("bias", 3.0f32);
        let const_result = loaded.execute_with_context(const_id, &ctx).unwrap();
        assert_eq!(const_result[0].as_f32().unwrap(), 10.0);

        // Execute GraphInput with provided value — should return 3.0
        let input_result = loaded.execute_with_context(input_id, &ctx).unwrap();
        assert_eq!(input_result[0].as_f32().unwrap(), 3.0);

        // Execute GraphInput without provided value — should use default 5.0
        let ctx_no_input = EvalContext::new();
        let default_result = loaded
            .execute_with_context(input_id, &ctx_no_input)
            .unwrap();
        assert_eq!(default_result[0].as_f32().unwrap(), 5.0);
    }

    /// Attempting to serialize a `ConstantNode` with an opaque value should
    /// produce a `__error` key in the params JSON (not silently succeed).
    #[test]
    fn test_constant_node_opaque_serialization_error() {
        use unshape_core::{ConstantNode, GraphValue, Value};

        #[derive(Debug)]
        struct Dummy;
        impl GraphValue for Dummy {
            fn as_any(&self) -> &dyn std::any::Any {
                self
            }
            fn type_name(&self) -> &'static str {
                "Dummy"
            }
        }

        let node = ConstantNode::new(Value::opaque(Dummy));
        let params = node.params();
        // Serializing an opaque value errors; params() encodes the error in the JSON
        assert!(
            params.get("__error").is_some(),
            "expected __error key in params for opaque ConstantNode, got: {params}"
        );
    }
}
