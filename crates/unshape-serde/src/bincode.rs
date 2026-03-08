//! Bincode format implementation.

use crate::error::SerdeError;
use crate::format::GraphFormat;
use crate::serial::{SerialGraph, SerialNode, SerialWire};
use serde::{Deserialize, Serialize};
use unshape_core::NodeId;

/// Bincode serialization format.
///
/// Compact binary format, faster than JSON but not human-readable.
#[derive(Debug, Clone, Copy, Default)]
pub struct BincodeFormat;

impl BincodeFormat {
    /// Creates a new BincodeFormat.
    pub fn new() -> Self {
        Self
    }
}

// --- Bincode-specific wire representation (same as SerialWire; both are just strings) ---

/// Bincode-compatible node representation.
///
/// Unlike `SerialNode`, params are stored as a JSON string rather than a
/// `serde_json::Value`, because bincode's serde bridge does not support the
/// `any` data model required by `serde_json::Value`.
#[derive(Serialize, Deserialize)]
struct BincodeNode {
    id: NodeId,
    type_name: String,
    /// Node parameters as a JSON string.
    params_json: String,
}

/// Bincode-compatible graph representation.
#[derive(Serialize, Deserialize)]
struct BincodeGraph {
    version: u32,
    nodes: Vec<BincodeNode>,
    /// Wires use the same `"nodeId:portIndex"` string format as `SerialWire`.
    wires: Vec<SerialWire>,
    next_id: NodeId,
}

impl BincodeGraph {
    fn from_serial(graph: &SerialGraph) -> Result<Self, SerdeError> {
        let nodes = graph
            .nodes
            .iter()
            .map(|n| {
                Ok(BincodeNode {
                    id: n.id,
                    type_name: n.type_name.clone(),
                    params_json: serde_json::to_string(&n.params)?,
                })
            })
            .collect::<Result<Vec<_>, SerdeError>>()?;

        Ok(BincodeGraph {
            version: graph.version,
            nodes,
            wires: graph.wires.clone(),
            next_id: graph.next_id,
        })
    }

    fn into_serial(self) -> Result<SerialGraph, SerdeError> {
        let nodes = self
            .nodes
            .into_iter()
            .map(|n| {
                let params: serde_json::Value = serde_json::from_str(&n.params_json)?;
                Ok(SerialNode::new(n.id, n.type_name, params))
            })
            .collect::<Result<Vec<_>, SerdeError>>()?;

        Ok(SerialGraph {
            version: self.version,
            nodes,
            wires: self.wires,
            next_id: self.next_id,
        })
    }
}

impl GraphFormat for BincodeFormat {
    fn serialize(&self, graph: &SerialGraph) -> Result<Vec<u8>, SerdeError> {
        let bc = BincodeGraph::from_serial(graph)?;
        let bytes = bincode::serde::encode_to_vec(bc, bincode::config::standard())?;
        Ok(bytes)
    }

    fn deserialize(&self, bytes: &[u8]) -> Result<SerialGraph, SerdeError> {
        let (bc, _): (BincodeGraph, _) =
            bincode::serde::decode_from_slice(bytes, bincode::config::standard())?;
        bc.into_serial()
    }

    fn name(&self) -> &'static str {
        "bincode"
    }

    fn extension(&self) -> &'static str {
        "bin"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::serial::{SerialNode, SerialWire};
    use std::any::Any;
    use unshape_core::{DynNode, EvalContext, GraphError, PortDescriptor, Value, ValueType, Wire};

    struct SingleOutNode;
    impl DynNode for SingleOutNode {
        fn type_name(&self) -> &'static str {
            "test::SingleOut"
        }
        fn inputs(&self) -> Vec<PortDescriptor> {
            vec![]
        }
        fn outputs(&self) -> Vec<PortDescriptor> {
            vec![PortDescriptor::new("out", ValueType::F32)]
        }
        fn execute(&self, _: &[Value], _: &EvalContext) -> Result<Vec<Value>, GraphError> {
            Ok(vec![Value::F32(0.0)])
        }
        fn as_any(&self) -> &dyn Any {
            self
        }
    }

    struct SingleInNode;
    impl DynNode for SingleInNode {
        fn type_name(&self) -> &'static str {
            "test::SingleIn"
        }
        fn inputs(&self) -> Vec<PortDescriptor> {
            vec![PortDescriptor::new("in", ValueType::F32)]
        }
        fn outputs(&self) -> Vec<PortDescriptor> {
            vec![]
        }
        fn execute(&self, _: &[Value], _: &EvalContext) -> Result<Vec<Value>, GraphError> {
            Ok(vec![])
        }
        fn as_any(&self) -> &dyn Any {
            self
        }
    }

    #[test]
    fn test_bincode_roundtrip() {
        let mut graph = SerialGraph::new();
        graph.nodes.push(SerialNode::new(
            0,
            "test::Node",
            serde_json::json!({"value": 42}),
        ));
        graph.next_id = 1;

        let format = BincodeFormat::new();
        let bytes = format.serialize(&graph).unwrap();
        let loaded = format.deserialize(&bytes).unwrap();

        assert_eq!(loaded.node_count(), 1);
        assert_eq!(loaded.nodes[0].type_name, "test::Node");
        assert_eq!(loaded.nodes[0].params["value"], 42);
    }

    #[test]
    fn test_bincode_smaller_than_json() {
        let mut graph = SerialGraph::new();
        for i in 0..10 {
            graph.nodes.push(SerialNode::new(
                i,
                "test::SomeNodeType",
                serde_json::json!({"value": i, "name": "test"}),
            ));
        }
        graph.next_id = 10;

        let json = crate::json::JsonFormat::new();
        let bincode = BincodeFormat::new();

        let json_bytes = json.serialize(&graph).unwrap();
        let bincode_bytes = bincode.serialize(&graph).unwrap();

        // Bincode should be more compact
        assert!(bincode_bytes.len() < json_bytes.len());
    }

    #[test]
    fn test_bincode_with_wires() {
        let mut graph = SerialGraph::new();
        graph
            .nodes
            .push(SerialNode::new(0, "A", serde_json::json!({})));
        graph
            .nodes
            .push(SerialNode::new(1, "B", serde_json::json!({})));
        graph.wires.push(SerialWire::from_wire(
            &Wire {
                from_node: 0,
                from_port: 0,
                to_node: 1,
                to_port: 0,
            },
            &SingleOutNode,
            &SingleInNode,
        ));
        graph.next_id = 2;

        let format = BincodeFormat::new();
        let bytes = format.serialize(&graph).unwrap();
        let loaded = format.deserialize(&bytes).unwrap();

        assert_eq!(loaded.wire_count(), 1);
        let w = loaded.wires[0]
            .to_wire(&SingleOutNode, &SingleInNode)
            .unwrap();
        assert_eq!(w.from_node, 0);
        assert_eq!(w.to_node, 1);
    }

    #[test]
    fn test_bincode_preserves_version() {
        let graph = SerialGraph::new();
        assert_eq!(graph.version, 1);

        let format = BincodeFormat::new();
        let bytes = format.serialize(&graph).unwrap();
        let loaded = format.deserialize(&bytes).unwrap();

        assert_eq!(loaded.version, 1);
    }
}
