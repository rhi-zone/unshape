//! Serializable intermediate representations of graph structures.

use rhizome_resin_core::{NodeId, Wire};
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;

/// Serializable representation of a node.
///
/// Contains the node's type name and parameters as a JSON string.
/// The type name is used to look up the deserializer in the registry.
/// Parameters are stored as a JSON string for format compatibility
/// (bincode can't serialize arbitrary JSON values directly).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerialNode {
    /// Unique identifier for this node within the graph.
    pub id: NodeId,
    /// Fully qualified type name (e.g., "resin::mesh::Subdivide").
    pub type_name: String,
    /// Node parameters as a JSON string.
    params_json: String,
}

impl SerialNode {
    /// Creates a new SerialNode from a JSON value.
    pub fn new(id: NodeId, type_name: impl Into<String>, params: JsonValue) -> Self {
        Self {
            id,
            type_name: type_name.into(),
            params_json: serde_json::to_string(&params).unwrap_or_else(|_| "{}".to_string()),
        }
    }

    /// Creates a new SerialNode from a JSON string.
    pub fn from_json_string(id: NodeId, type_name: impl Into<String>, params_json: String) -> Self {
        Self {
            id,
            type_name: type_name.into(),
            params_json,
        }
    }

    /// Returns the parameters as a JSON value.
    pub fn params(&self) -> JsonValue {
        serde_json::from_str(&self.params_json).unwrap_or(JsonValue::Object(Default::default()))
    }

    /// Returns the raw JSON string of the parameters.
    pub fn params_json(&self) -> &str {
        &self.params_json
    }
}

/// Serializable representation of an entire graph.
///
/// This is the intermediate format used for serialization.
/// It can be converted to/from JSON, bincode, or other formats.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerialGraph {
    /// All nodes in the graph.
    pub nodes: Vec<SerialNode>,
    /// All wires connecting nodes.
    pub wires: Vec<Wire>,
    /// The next node ID that will be assigned.
    pub next_id: NodeId,
}

impl SerialGraph {
    /// Creates an empty SerialGraph.
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            wires: Vec::new(),
            next_id: 0,
        }
    }

    /// Returns the number of nodes.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Returns the number of wires.
    pub fn wire_count(&self) -> usize {
        self.wires.len()
    }
}

impl Default for SerialGraph {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_serial_node_new() {
        let node = SerialNode::new(0, "test::Node", serde_json::json!({"value": 42}));
        assert_eq!(node.id, 0);
        assert_eq!(node.type_name, "test::Node");
        assert_eq!(node.params()["value"], 42);
    }

    #[test]
    fn test_serial_node_params_json() {
        let node = SerialNode::new(0, "test::Node", serde_json::json!({"x": 1, "y": 2}));
        let json_str = node.params_json();
        assert!(json_str.contains("\"x\""));
        assert!(json_str.contains("\"y\""));
    }

    #[test]
    fn test_serial_graph_default() {
        let graph = SerialGraph::default();
        assert_eq!(graph.node_count(), 0);
        assert_eq!(graph.wire_count(), 0);
        assert_eq!(graph.next_id, 0);
    }

    #[test]
    fn test_serial_graph_roundtrip_json() {
        let mut graph = SerialGraph::new();
        graph.nodes.push(SerialNode::new(
            0,
            "test::Add",
            serde_json::json!({"a": 1.0, "b": 2.0}),
        ));
        graph.nodes.push(SerialNode::new(
            1,
            "test::Const",
            serde_json::json!({"value": 5.0}),
        ));
        graph.wires.push(Wire {
            from_node: 1,
            from_port: 0,
            to_node: 0,
            to_port: 0,
        });
        graph.next_id = 2;

        let json = serde_json::to_string(&graph).unwrap();
        let loaded: SerialGraph = serde_json::from_str(&json).unwrap();

        assert_eq!(loaded.node_count(), 2);
        assert_eq!(loaded.wire_count(), 1);
        assert_eq!(loaded.next_id, 2);
        assert_eq!(loaded.nodes[0].type_name, "test::Add");
    }
}
