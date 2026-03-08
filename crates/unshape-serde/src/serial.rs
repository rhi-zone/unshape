//! Serializable intermediate representations of graph structures.

use crate::error::SerdeError;
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;
use unshape_core::{DynNode, NodeId, Wire};

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

/// A wire in serialized form, using human-readable `"nodeId:portName"` strings.
///
/// Example: `{ "from": "42:out", "to": "7:value" }` connects the `out` output port
/// of node 42 to the `value` input port of node 7.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerialWire {
    /// Source endpoint: `"nodeId:portName"`.
    pub from: String,
    /// Destination endpoint: `"nodeId:portName"`.
    pub to: String,
}

impl SerialWire {
    /// Converts a `Wire` to a `SerialWire` using port names from the connected nodes.
    pub fn from_wire(w: &Wire, from_node: &dyn DynNode, to_node: &dyn DynNode) -> Self {
        let from_names = from_node.output_port_names();
        let from_name = from_names.get(w.from_port).copied().unwrap_or("out");
        let to_names = to_node.input_port_names();
        let to_name = to_names.get(w.to_port).copied().unwrap_or("in");
        Self {
            from: format!("{}:{}", w.from_node, from_name),
            to: format!("{}:{}", w.to_node, to_name),
        }
    }

    /// Parses a `"nodeId:portName"` endpoint string.
    fn parse_endpoint(s: &str) -> Result<(NodeId, &str), SerdeError> {
        let (node_str, port_name) = s.split_once(':').ok_or_else(|| {
            SerdeError::InvalidWireFormat(format!("expected \"nodeId:portName\", got {:?}", s))
        })?;
        let node_id: NodeId = node_str.parse().map_err(|_| {
            SerdeError::InvalidWireFormat(format!("invalid node id {:?}", node_str))
        })?;
        Ok((node_id, port_name))
    }

    /// Parses only the node ID from a `"nodeId:portName"` endpoint string.
    ///
    /// Used by deserialization to look up nodes before resolving port names.
    pub(crate) fn node_id(s: &str) -> Result<NodeId, SerdeError> {
        Self::parse_endpoint(s).map(|(id, _)| id)
    }

    /// Converts this `SerialWire` back to a `Wire` using port names from the connected nodes.
    pub fn to_wire(
        &self,
        from_node: &dyn DynNode,
        to_node: &dyn DynNode,
    ) -> Result<Wire, SerdeError> {
        let (from_node_id, from_name) = Self::parse_endpoint(&self.from)?;
        let (to_node_id, to_name) = Self::parse_endpoint(&self.to)?;

        let from_port = from_node
            .output_port_names()
            .into_iter()
            .position(|n| n == from_name)
            .ok_or_else(|| {
                SerdeError::InvalidWireFormat(format!(
                    "node {} has no output port named {:?}",
                    from_node_id, from_name
                ))
            })?;

        let to_port = to_node
            .input_port_names()
            .into_iter()
            .position(|n| n == to_name)
            .ok_or_else(|| {
                SerdeError::InvalidWireFormat(format!(
                    "node {} has no input port named {:?}",
                    to_node_id, to_name
                ))
            })?;

        Ok(Wire {
            from_node: from_node_id,
            from_port,
            to_node: to_node_id,
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
    use std::any::Any;
    use unshape_core::{EvalContext, GraphError, PortDescriptor, Value, ValueType};

    struct ConstNode;
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
        fn execute(&self, _: &[Value], _: &EvalContext) -> Result<Vec<Value>, GraphError> {
            Ok(vec![Value::F32(0.0)])
        }
        fn as_any(&self) -> &dyn Any {
            self
        }
    }

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
        fn execute(&self, _: &[Value], _: &EvalContext) -> Result<Vec<Value>, GraphError> {
            Ok(vec![Value::F32(0.0)])
        }
        fn as_any(&self) -> &dyn Any {
            self
        }
    }

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
        // ConstNode: output port 0 = "value"
        // AddNode: input port 0 = "a"
        let wire = Wire {
            from_node: 42,
            from_port: 0,
            to_node: 7,
            to_port: 0,
        };
        let serial = SerialWire::from_wire(&wire, &ConstNode, &AddNode);
        assert_eq!(serial.from, "42:value");
        assert_eq!(serial.to, "7:a");

        let recovered = serial.to_wire(&ConstNode, &AddNode).unwrap();
        assert_eq!(recovered.from_node, 42);
        assert_eq!(recovered.from_port, 0);
        assert_eq!(recovered.to_node, 7);
        assert_eq!(recovered.to_port, 0);
    }

    #[test]
    fn test_serial_wire_named_port_second_input() {
        // AddNode output port 0 = "result"; AddNode input port 1 = "b"
        let wire = Wire {
            from_node: 1,
            from_port: 0,
            to_node: 2,
            to_port: 1,
        };
        let serial = SerialWire::from_wire(&wire, &AddNode, &AddNode);
        assert_eq!(serial.from, "1:result");
        assert_eq!(serial.to, "2:b");

        let recovered = serial.to_wire(&AddNode, &AddNode).unwrap();
        assert_eq!(recovered.from_port, 0);
        assert_eq!(recovered.to_port, 1);
    }

    #[test]
    fn test_serial_wire_invalid_format() {
        let bad = SerialWire {
            from: "not-valid".to_string(),
            to: "0:a".to_string(),
        };
        assert!(bad.to_wire(&ConstNode, &AddNode).is_err());

        let bad_node_id = SerialWire {
            from: "abc:value".to_string(),
            to: "0:a".to_string(),
        };
        assert!(bad_node_id.to_wire(&ConstNode, &AddNode).is_err());

        let unknown_port = SerialWire {
            from: "0:nonexistent".to_string(),
            to: "1:a".to_string(),
        };
        assert!(unknown_port.to_wire(&ConstNode, &AddNode).is_err());
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
        // ConstNode output "value" -> AddNode input "a"
        graph.wires.push(SerialWire::from_wire(
            &Wire {
                from_node: 1,
                from_port: 0,
                to_node: 0,
                to_port: 0,
            },
            &ConstNode,
            &AddNode,
        ));
        graph.next_id = 2;

        let json = serde_json::to_string_pretty(&graph).unwrap();

        // params must appear as an embedded object, not a string
        assert!(
            json.contains("\"params\": {"),
            "params should be a JSON object in output:\n{json}"
        );
        // wires must use named port format
        assert!(
            json.contains("\"from\": \"1:value\""),
            "wire 'from' should use named port:\n{json}"
        );
        assert!(
            json.contains("\"to\": \"0:a\""),
            "wire 'to' should use named port:\n{json}"
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

        let recovered_wire = loaded.wires[0].to_wire(&ConstNode, &AddNode).unwrap();
        assert_eq!(recovered_wire.from_node, 1);
        assert_eq!(recovered_wire.to_node, 0);
        assert_eq!(recovered_wire.from_port, 0);
        assert_eq!(recovered_wire.to_port, 0);
    }
}
