//! Serializable intermediate representations of graph structures.

use crate::error::SerdeError;
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;
use unshape_core::{NodeId, Wire};

/// Serializable representation of a node.
///
/// Contains the node's type name and parameters as a structured JSON value.
/// The type name is used to look up the deserializer in the registry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerialNode {
    /// Unique identifier for this node within the graph.
    pub id: NodeId,
    /// Fully qualified type name (e.g., "resin::mesh::Subdivide").
    pub type_name: String,
    /// Node parameters as a JSON value.
    pub params: JsonValue,
}

impl SerialNode {
    /// Creates a new SerialNode from a JSON value.
    pub fn new(id: NodeId, type_name: impl Into<String>, params: JsonValue) -> Self {
        Self {
            id,
            type_name: type_name.into(),
            params,
        }
    }

    /// Returns the parameters as a JSON value.
    pub fn params(&self) -> JsonValue {
        self.params.clone()
    }
}

/// A wire in serialized form, using human-readable `"nodeId:portIndex"` strings.
///
/// Example: `{ "from": "42:0", "to": "7:2" }` connects output port 0 of node 42
/// to input port 2 of node 7.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerialWire {
    /// Source endpoint: `"nodeId:portIndex"`.
    pub from: String,
    /// Destination endpoint: `"nodeId:portIndex"`.
    pub to: String,
}

impl SerialWire {
    /// Converts a `Wire` to a `SerialWire`.
    pub fn from_wire(w: &Wire) -> Self {
        Self {
            from: format!("{}:{}", w.from_node, w.from_port),
            to: format!("{}:{}", w.to_node, w.to_port),
        }
    }

    /// Parses a `"nodeId:portIndex"` endpoint string.
    fn parse_endpoint(s: &str) -> Result<(NodeId, usize), SerdeError> {
        let (node_str, port_str) = s.split_once(':').ok_or_else(|| {
            SerdeError::InvalidWireFormat(format!("expected \"nodeId:portIndex\", got {:?}", s))
        })?;
        let node_id: NodeId = node_str.parse().map_err(|_| {
            SerdeError::InvalidWireFormat(format!("invalid node id {:?}", node_str))
        })?;
        let port: usize = port_str.parse().map_err(|_| {
            SerdeError::InvalidWireFormat(format!("invalid port index {:?}", port_str))
        })?;
        Ok((node_id, port))
    }

    /// Converts this `SerialWire` back to a `Wire`.
    pub fn to_wire(&self) -> Result<Wire, SerdeError> {
        let (from_node, from_port) = Self::parse_endpoint(&self.from)?;
        let (to_node, to_port) = Self::parse_endpoint(&self.to)?;
        Ok(Wire {
            from_node,
            from_port,
            to_node,
            to_port,
        })
    }
}

/// Serializable representation of an entire graph.
///
/// This is the intermediate format used for serialization.
/// It can be converted to/from JSON, bincode, or other formats.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerialGraph {
    /// Format version. Currently `1`. Old graphs without this field
    /// deserialize as version `0`.
    #[serde(default)]
    pub version: u32,
    /// All nodes in the graph.
    pub nodes: Vec<SerialNode>,
    /// All wires connecting nodes.
    pub wires: Vec<SerialWire>,
    /// The next node ID that will be assigned.
    pub next_id: NodeId,
}

impl SerialGraph {
    /// Creates an empty SerialGraph at the current format version.
    pub fn new() -> Self {
        Self {
            version: 1,
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
    fn test_serial_node_params_is_object() {
        let node = SerialNode::new(0, "test::Node", serde_json::json!({"x": 1, "y": 2}));
        // params should be a JSON object, not a double-encoded string
        assert!(node.params.is_object());
        assert_eq!(node.params["x"], 1);
        assert_eq!(node.params["y"], 2);
    }

    #[test]
    fn test_serial_wire_roundtrip() {
        let wire = Wire {
            from_node: 42,
            from_port: 0,
            to_node: 7,
            to_port: 2,
        };
        let serial = SerialWire::from_wire(&wire);
        assert_eq!(serial.from, "42:0");
        assert_eq!(serial.to, "7:2");

        let recovered = serial.to_wire().unwrap();
        assert_eq!(recovered.from_node, 42);
        assert_eq!(recovered.from_port, 0);
        assert_eq!(recovered.to_node, 7);
        assert_eq!(recovered.to_port, 2);
    }

    #[test]
    fn test_serial_wire_invalid_format() {
        let bad = SerialWire {
            from: "not-valid".to_string(),
            to: "0:0".to_string(),
        };
        assert!(bad.to_wire().is_err());

        let bad2 = SerialWire {
            from: "abc:0".to_string(),
            to: "0:0".to_string(),
        };
        assert!(bad2.to_wire().is_err());
    }

    #[test]
    fn test_serial_graph_default() {
        let graph = SerialGraph::default();
        assert_eq!(graph.version, 1);
        assert_eq!(graph.node_count(), 0);
        assert_eq!(graph.wire_count(), 0);
        assert_eq!(graph.next_id, 0);
    }

    #[test]
    fn test_serial_graph_version_field() {
        // New graphs have version 1
        let graph = SerialGraph::new();
        assert_eq!(graph.version, 1);

        // Old graphs without version field deserialize as version 0
        let old_json = r#"{"nodes":[],"wires":[],"next_id":0}"#;
        let loaded: SerialGraph = serde_json::from_str(old_json).unwrap();
        assert_eq!(loaded.version, 0);
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
        graph.wires.push(SerialWire::from_wire(&Wire {
            from_node: 1,
            from_port: 0,
            to_node: 0,
            to_port: 0,
        }));
        graph.next_id = 2;

        let json = serde_json::to_string_pretty(&graph).unwrap();

        // params must appear as an embedded object, not a string
        assert!(
            json.contains("\"params\": {"),
            "params should be a JSON object in output:\n{json}"
        );
        // wires must use the "from"/"to" string format
        assert!(
            json.contains("\"from\": \"1:0\""),
            "wire 'from' should be a string:\n{json}"
        );
        assert!(
            json.contains("\"to\": \"0:0\""),
            "wire 'to' should be a string:\n{json}"
        );
        // version field should be present
        assert!(
            json.contains("\"version\": 1"),
            "version field missing:\n{json}"
        );

        let loaded: SerialGraph = serde_json::from_str(&json).unwrap();

        assert_eq!(loaded.version, 1);
        assert_eq!(loaded.node_count(), 2);
        assert_eq!(loaded.wire_count(), 1);
        assert_eq!(loaded.next_id, 2);
        assert_eq!(loaded.nodes[0].type_name, "test::Add");

        let recovered_wire = loaded.wires[0].to_wire().unwrap();
        assert_eq!(recovered_wire.from_node, 1);
        assert_eq!(recovered_wire.to_node, 0);
    }
}
