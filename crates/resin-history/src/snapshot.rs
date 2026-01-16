//! Snapshot-based history for undo/redo.
//!
//! Stores full graph snapshots at each history point.
//! Simple and reliable, trading storage space for simplicity.

use crate::error::HistoryError;
use rhizome_resin_core::{DynNode, Graph};
use rhizome_resin_serde::{NodeRegistry, SerialGraph, graph_to_serial, serial_to_graph};

/// Configuration for snapshot history.
#[derive(Debug, Clone)]
pub struct SnapshotConfig {
    /// Maximum number of snapshots to keep. 0 = unlimited.
    pub max_snapshots: usize,
}

impl Default for SnapshotConfig {
    fn default() -> Self {
        Self { max_snapshots: 100 }
    }
}

impl SnapshotConfig {
    /// Creates config with unlimited snapshots.
    pub fn unlimited() -> Self {
        Self { max_snapshots: 0 }
    }

    /// Creates config with a specific limit.
    pub fn with_limit(max: usize) -> Self {
        Self { max_snapshots: max }
    }
}

/// Snapshot-based history manager.
///
/// Stores full graph state at each checkpoint. Supports undo/redo
/// by restoring previous snapshots.
///
/// # Example
///
/// ```ignore
/// let mut history = SnapshotHistory::new(SnapshotConfig::default());
///
/// // Record initial state
/// history.record(&graph, &registry, extract_params)?;
///
/// // After user makes changes...
/// history.record(&graph, &registry, extract_params)?;
///
/// // Undo
/// if let Some(prev) = history.undo(&registry)? {
///     graph = prev;
/// }
/// ```
pub struct SnapshotHistory {
    /// All snapshots (oldest first).
    snapshots: Vec<SerialGraph>,
    /// Current position in snapshot list (1-indexed, 0 = before first snapshot).
    current: usize,
    /// Configuration.
    config: SnapshotConfig,
}

impl SnapshotHistory {
    /// Creates a new empty history.
    pub fn new(config: SnapshotConfig) -> Self {
        Self {
            snapshots: Vec::new(),
            current: 0,
            config,
        }
    }

    /// Records the current graph state as a new snapshot.
    ///
    /// This truncates any redo history (snapshots after current position).
    pub fn record<F>(&mut self, graph: &Graph, extract_params: F) -> Result<(), HistoryError>
    where
        F: Fn(&dyn DynNode) -> Option<serde_json::Value>,
    {
        // Truncate any redo history
        self.snapshots.truncate(self.current);

        // Capture snapshot
        let serial = graph_to_serial(graph, extract_params)?;
        self.snapshots.push(serial);
        self.current = self.snapshots.len();

        // Enforce limit by removing oldest snapshots
        if self.config.max_snapshots > 0 && self.snapshots.len() > self.config.max_snapshots {
            let remove_count = self.snapshots.len() - self.config.max_snapshots;
            self.snapshots.drain(0..remove_count);
            self.current = self.current.saturating_sub(remove_count);
        }

        Ok(())
    }

    /// Undoes to the previous snapshot.
    ///
    /// Returns the restored graph, or None if there's nothing to undo.
    pub fn undo(&mut self, registry: &NodeRegistry) -> Result<Option<Graph>, HistoryError> {
        if self.current <= 1 {
            return Ok(None);
        }

        self.current -= 1;
        let serial = self.snapshots[self.current - 1].clone();
        let graph = serial_to_graph(serial, registry)?;
        Ok(Some(graph))
    }

    /// Redoes to the next snapshot.
    ///
    /// Returns the restored graph, or None if there's nothing to redo.
    pub fn redo(&mut self, registry: &NodeRegistry) -> Result<Option<Graph>, HistoryError> {
        if self.current >= self.snapshots.len() {
            return Ok(None);
        }

        let serial = self.snapshots[self.current].clone();
        self.current += 1;
        let graph = serial_to_graph(serial, registry)?;
        Ok(Some(graph))
    }

    /// Returns true if undo is possible.
    pub fn can_undo(&self) -> bool {
        self.current > 1
    }

    /// Returns true if redo is possible.
    pub fn can_redo(&self) -> bool {
        self.current < self.snapshots.len()
    }

    /// Returns the number of stored snapshots.
    pub fn snapshot_count(&self) -> usize {
        self.snapshots.len()
    }

    /// Returns the current position (1-indexed).
    pub fn position(&self) -> usize {
        self.current
    }

    /// Returns how many undo steps are available.
    pub fn undo_count(&self) -> usize {
        self.current.saturating_sub(1)
    }

    /// Returns how many redo steps are available.
    pub fn redo_count(&self) -> usize {
        self.snapshots.len().saturating_sub(self.current)
    }

    /// Clears all history.
    pub fn clear(&mut self) {
        self.snapshots.clear();
        self.current = 0;
    }

    /// Returns a reference to all snapshots.
    pub fn snapshots(&self) -> &[SerialGraph] {
        &self.snapshots
    }
}

impl Default for SnapshotHistory {
    fn default() -> Self {
        Self::new(SnapshotConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rhizome_resin_core::{EvalContext, GraphError, PortDescriptor, Value, ValueType};
    use serde::{Deserialize, Serialize};

    #[derive(Debug, Clone, Serialize, Deserialize)]
    struct TestNode {
        value: f32,
    }

    impl DynNode for TestNode {
        fn type_name(&self) -> &'static str {
            "test::Node"
        }
        fn inputs(&self) -> Vec<PortDescriptor> {
            vec![]
        }
        fn outputs(&self) -> Vec<PortDescriptor> {
            vec![PortDescriptor::new("out", ValueType::F32)]
        }
        fn execute(&self, _: &[Value], _ctx: &EvalContext) -> Result<Vec<Value>, GraphError> {
            Ok(vec![Value::F32(self.value)])
        }
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
    }

    fn make_registry() -> NodeRegistry {
        let mut registry = NodeRegistry::new();
        registry.register_factory("test::Node", |params| {
            let value = params["value"].as_f64().unwrap_or(0.0) as f32;
            Ok(Box::new(TestNode { value }))
        });
        registry
    }

    fn extract_params(node: &dyn DynNode) -> Option<serde_json::Value> {
        if node.type_name() == "test::Node" {
            let ctx = EvalContext::new();
            let outputs = node.execute(&[], &ctx).ok()?;
            let value = outputs[0].as_f32().ok()?;
            Some(serde_json::json!({"value": value}))
        } else {
            None
        }
    }

    #[test]
    fn test_empty_history() {
        let history = SnapshotHistory::default();
        assert!(!history.can_undo());
        assert!(!history.can_redo());
        assert_eq!(history.snapshot_count(), 0);
    }

    #[test]
    fn test_single_record() {
        let mut history = SnapshotHistory::default();
        let mut graph = Graph::new();
        graph.add_node(TestNode { value: 1.0 });

        history.record(&graph, extract_params).unwrap();

        assert!(!history.can_undo()); // Need 2+ snapshots to undo
        assert!(!history.can_redo());
        assert_eq!(history.snapshot_count(), 1);
    }

    #[test]
    fn test_undo_redo() {
        let registry = make_registry();
        let mut history = SnapshotHistory::default();

        // Record state 1
        let mut graph = Graph::new();
        let n1 = graph.add_node(TestNode { value: 1.0 });
        history.record(&graph, extract_params).unwrap();

        // Record state 2
        graph.add_node(TestNode { value: 2.0 });
        history.record(&graph, extract_params).unwrap();

        assert!(history.can_undo());
        assert_eq!(history.undo_count(), 1);

        // Undo to state 1
        let mut restored = history.undo(&registry).unwrap().unwrap();
        assert_eq!(restored.node_count(), 1);

        // Execute to verify value
        let result = restored.execute(n1).unwrap();
        assert_eq!(result[0].as_f32().unwrap(), 1.0);

        // Redo to state 2
        assert!(history.can_redo());
        let restored = history.redo(&registry).unwrap().unwrap();
        assert_eq!(restored.node_count(), 2);
    }

    #[test]
    fn test_undo_then_record_truncates_redo() {
        let registry = make_registry();
        let mut history = SnapshotHistory::default();

        // Record states 1, 2, 3
        let mut graph = Graph::new();
        graph.add_node(TestNode { value: 1.0 });
        history.record(&graph, extract_params).unwrap();

        graph.add_node(TestNode { value: 2.0 });
        history.record(&graph, extract_params).unwrap();

        graph.add_node(TestNode { value: 3.0 });
        history.record(&graph, extract_params).unwrap();

        assert_eq!(history.snapshot_count(), 3);

        // Undo twice (to state 1)
        history.undo(&registry).unwrap();
        history.undo(&registry).unwrap();
        assert_eq!(history.redo_count(), 2);

        // Record new state (should truncate states 2 and 3)
        let mut graph = Graph::new();
        graph.add_node(TestNode { value: 10.0 });
        history.record(&graph, extract_params).unwrap();

        assert_eq!(history.snapshot_count(), 2); // State 1 and new state
        assert!(!history.can_redo()); // No more redo
    }

    #[test]
    fn test_max_snapshots() {
        let config = SnapshotConfig::with_limit(3);
        let mut history = SnapshotHistory::new(config);

        // Record 5 states
        for i in 1..=5 {
            let mut graph = Graph::new();
            graph.add_node(TestNode { value: i as f32 });
            history.record(&graph, extract_params).unwrap();
        }

        // Should only keep last 3
        assert_eq!(history.snapshot_count(), 3);

        // Oldest should be state 3 (values 3, 4, 5)
        let registry = make_registry();
        // Undo to state 4
        let mut restored = history.undo(&registry).unwrap().unwrap();
        let result = restored.execute(0).unwrap();
        assert_eq!(result[0].as_f32().unwrap(), 4.0);
    }

    #[test]
    fn test_clear() {
        let mut history = SnapshotHistory::default();
        let mut graph = Graph::new();
        graph.add_node(TestNode { value: 1.0 });
        history.record(&graph, extract_params).unwrap();
        history.record(&graph, extract_params).unwrap();

        history.clear();

        assert_eq!(history.snapshot_count(), 0);
        assert!(!history.can_undo());
        assert!(!history.can_redo());
    }
}
