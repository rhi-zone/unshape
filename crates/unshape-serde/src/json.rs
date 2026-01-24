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
    use crate::serial::SerialNode;
    use unshape_core::Wire;

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
        graph.wires.push(Wire {
            from_node: 0,
            from_port: 0,
            to_node: 1,
            to_port: 0,
        });
        graph.next_id = 2;

        let format = JsonFormat::new();
        let bytes = format.serialize(&graph).unwrap();
        let loaded = format.deserialize(&bytes).unwrap();

        assert_eq!(loaded.wire_count(), 1);
        assert_eq!(loaded.wires[0].from_node, 0);
        assert_eq!(loaded.wires[0].to_node, 1);
    }
}
