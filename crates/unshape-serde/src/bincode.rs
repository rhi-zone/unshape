//! Bincode format implementation.

use crate::error::SerdeError;
use crate::format::GraphFormat;
use crate::serial::SerialGraph;

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

impl GraphFormat for BincodeFormat {
    fn serialize(&self, graph: &SerialGraph) -> Result<Vec<u8>, SerdeError> {
        let bytes = bincode::serde::encode_to_vec(graph, bincode::config::standard())?;
        Ok(bytes)
    }

    fn deserialize(&self, bytes: &[u8]) -> Result<SerialGraph, SerdeError> {
        let (graph, _) = bincode::serde::decode_from_slice(bytes, bincode::config::standard())?;
        Ok(graph)
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
    use crate::serial::SerialNode;
    use unshape_core::Wire;

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
        graph.wires.push(Wire {
            from_node: 0,
            from_port: 0,
            to_node: 1,
            to_port: 0,
        });
        graph.next_id = 2;

        let format = BincodeFormat::new();
        let bytes = format.serialize(&graph).unwrap();
        let loaded = format.deserialize(&bytes).unwrap();

        assert_eq!(loaded.wire_count(), 1);
        assert_eq!(loaded.wires[0].from_node, 0);
        assert_eq!(loaded.wires[0].to_node, 1);
    }
}
