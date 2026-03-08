//! JSON format implementation.

use crate::error::SerdeError;
use crate::format::GraphFormat;
use crate::serial::SerialGraph;

/// JSON serialization format.
///
/// Human-readable, git-diffable, good for debugging.
#[derive(Debug, Clone, Default)]
pub struct JsonFormat {
    /// Whether to pretty-print with indentation.
    pub pretty: bool,
}

impl JsonFormat {
    /// Creates a new JsonFormat with default settings (compact).
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a new JsonFormat with pretty-printing enabled.
    pub fn pretty() -> Self {
        Self { pretty: true }
    }
}

impl GraphFormat for JsonFormat {
    fn serialize(&self, graph: &SerialGraph) -> Result<Vec<u8>, SerdeError> {
        let bytes = if self.pretty {
            serde_json::to_vec_pretty(graph)?
        } else {
            serde_json::to_vec(graph)?
        };
        Ok(bytes)
    }

    fn deserialize(&self, bytes: &[u8]) -> Result<SerialGraph, SerdeError> {
        Ok(serde_json::from_slice(bytes)?)
    }

    fn name(&self) -> &'static str {
        "JSON"
    }

    fn extension(&self) -> &'static str {
        "json"
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
    fn test_json_roundtrip() {
        let mut graph = SerialGraph::new();
        graph.nodes.push(SerialNode::new(
            0,
            "test::Node",
            serde_json::json!({"value": 42}),
        ));
        graph.next_id = 1;

        let format = JsonFormat::new();
        let bytes = format.serialize(&graph).unwrap();
        let loaded = format.deserialize(&bytes).unwrap();

        assert_eq!(loaded.node_count(), 1);
        assert_eq!(loaded.nodes[0].type_name, "test::Node");
    }

    #[test]
    fn test_json_pretty() {
        let mut graph = SerialGraph::new();
        graph
            .nodes
            .push(SerialNode::new(0, "test::Node", serde_json::json!({})));
        graph.next_id = 1;

        let compact = JsonFormat::new();
        let pretty = JsonFormat::pretty();

        let compact_bytes = compact.serialize(&graph).unwrap();
        let pretty_bytes = pretty.serialize(&graph).unwrap();

        // Pretty format should be larger due to whitespace
        assert!(pretty_bytes.len() > compact_bytes.len());

        // Both should deserialize correctly
        let _ = compact.deserialize(&compact_bytes).unwrap();
        let _ = pretty.deserialize(&pretty_bytes).unwrap();
    }

    #[test]
    fn test_json_with_wires() {
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

        let format = JsonFormat::new();
        let bytes = format.serialize(&graph).unwrap();
        let loaded = format.deserialize(&bytes).unwrap();

        assert_eq!(loaded.wire_count(), 1);
        let w = loaded.wires[0]
            .to_wire(&SingleOutNode, &SingleInNode)
            .unwrap();
        assert_eq!(w.from_node, 0);
        assert_eq!(w.to_node, 1);
    }
}
