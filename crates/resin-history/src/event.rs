//! Event sourcing for fine-grained history tracking.
//!
//! Records individual graph modifications as events. Enables:
//! - Detailed undo/redo
//! - Audit trails
//! - Collaborative editing (future)
//! - Replay and debugging

use crate::error::HistoryError;
use rhizome_resin_core::{Graph, NodeId, Wire};
use rhizome_resin_serde::NodeRegistry;
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;

/// Events that modify a graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GraphEvent {
    /// Add a new node.
    AddNode {
        /// Node ID.
        id: NodeId,
        /// Node type name for registry lookup.
        type_name: String,
        /// Node parameters as JSON.
        params: JsonValue,
    },

    /// Remove a node (and its wires).
    RemoveNode {
        /// Node ID to remove.
        id: NodeId,
        /// Captured type name (for inverse).
        type_name: String,
        /// Captured params (for inverse).
        params: JsonValue,
        /// Captured wires that were connected to this node (for inverse).
        wires: Vec<Wire>,
    },

    /// Update node parameters.
    UpdateParams {
        /// Node ID.
        id: NodeId,
        /// Old parameters (for inverse).
        old_params: JsonValue,
        /// New parameters.
        new_params: JsonValue,
    },

    /// Connect two nodes.
    Connect {
        /// Wire to add.
        wire: Wire,
    },

    /// Disconnect two nodes.
    Disconnect {
        /// Wire to remove.
        wire: Wire,
    },

    /// Multiple events as a single atomic operation.
    Batch {
        /// Events in this batch.
        events: Vec<GraphEvent>,
        /// Human-readable description of the batch.
        description: Option<String>,
    },
}

impl GraphEvent {
    /// Creates an AddNode event.
    pub fn add_node(id: NodeId, type_name: impl Into<String>, params: JsonValue) -> Self {
        Self::AddNode {
            id,
            type_name: type_name.into(),
            params,
        }
    }

    /// Creates a Connect event.
    pub fn connect(from_node: NodeId, from_port: usize, to_node: NodeId, to_port: usize) -> Self {
        Self::Connect {
            wire: Wire {
                from_node,
                from_port,
                to_node,
                to_port,
            },
        }
    }

    /// Creates a Disconnect event.
    pub fn disconnect(
        from_node: NodeId,
        from_port: usize,
        to_node: NodeId,
        to_port: usize,
    ) -> Self {
        Self::Disconnect {
            wire: Wire {
                from_node,
                from_port,
                to_node,
                to_port,
            },
        }
    }

    /// Creates a Batch event.
    pub fn batch(events: Vec<GraphEvent>, description: Option<String>) -> Self {
        Self::Batch {
            events,
            description,
        }
    }

    /// Computes the inverse of this event (for undo).
    pub fn inverse(&self) -> Self {
        match self {
            GraphEvent::AddNode {
                id,
                type_name,
                params,
            } => GraphEvent::RemoveNode {
                id: *id,
                type_name: type_name.clone(),
                params: params.clone(),
                wires: Vec::new(), // Wires are captured at apply time
            },

            GraphEvent::RemoveNode {
                id,
                type_name,
                params,
                wires,
            } => {
                // Inverse is: add the node back, then reconnect wires
                let mut events = vec![GraphEvent::AddNode {
                    id: *id,
                    type_name: type_name.clone(),
                    params: params.clone(),
                }];
                for wire in wires {
                    events.push(GraphEvent::Connect { wire: *wire });
                }
                if events.len() == 1 {
                    events.pop().unwrap()
                } else {
                    GraphEvent::Batch {
                        events,
                        description: Some("Restore node".to_string()),
                    }
                }
            }

            GraphEvent::UpdateParams {
                id,
                old_params,
                new_params,
            } => GraphEvent::UpdateParams {
                id: *id,
                old_params: new_params.clone(),
                new_params: old_params.clone(),
            },

            GraphEvent::Connect { wire } => GraphEvent::Disconnect { wire: *wire },

            GraphEvent::Disconnect { wire } => GraphEvent::Connect { wire: *wire },

            GraphEvent::Batch {
                events,
                description,
            } => {
                // Inverse of batch is reversed order of inverse events
                let inverse_events: Vec<_> = events.iter().rev().map(|e| e.inverse()).collect();
                GraphEvent::Batch {
                    events: inverse_events,
                    description: description.clone().map(|d| format!("Undo: {}", d)),
                }
            }
        }
    }

    /// Applies this event to a graph.
    pub fn apply(&self, graph: &mut Graph, registry: &NodeRegistry) -> Result<(), HistoryError> {
        match self {
            GraphEvent::AddNode {
                id,
                type_name,
                params,
            } => {
                let node = registry.deserialize(type_name, params.clone())?;
                graph.insert_node_with_id(*id, node)?;
            }

            GraphEvent::RemoveNode { id, .. } => {
                graph.remove_node(*id)?;
            }

            GraphEvent::UpdateParams { id, new_params, .. } => {
                let node = graph.get_node(*id).ok_or(HistoryError::NodeNotFound(*id))?;
                let type_name = node.type_name();
                let new_node = registry.deserialize(type_name, new_params.clone())?;
                graph.replace_node(*id, new_node)?;
            }

            GraphEvent::Connect { wire } => {
                graph.connect(wire.from_node, wire.from_port, wire.to_node, wire.to_port)?;
            }

            GraphEvent::Disconnect { wire } => {
                graph.disconnect(wire.from_node, wire.from_port, wire.to_node, wire.to_port)?;
            }

            GraphEvent::Batch { events, .. } => {
                for event in events {
                    event.apply(graph, registry)?;
                }
            }
        }
        Ok(())
    }

    /// Returns a human-readable description of this event.
    pub fn description(&self) -> String {
        match self {
            GraphEvent::AddNode { id, type_name, .. } => {
                format!("Add node {} ({})", id, type_name)
            }
            GraphEvent::RemoveNode { id, type_name, .. } => {
                format!("Remove node {} ({})", id, type_name)
            }
            GraphEvent::UpdateParams { id, .. } => {
                format!("Update params for node {}", id)
            }
            GraphEvent::Connect { wire } => {
                format!(
                    "Connect {}:{} -> {}:{}",
                    wire.from_node, wire.from_port, wire.to_node, wire.to_port
                )
            }
            GraphEvent::Disconnect { wire } => {
                format!(
                    "Disconnect {}:{} -> {}:{}",
                    wire.from_node, wire.from_port, wire.to_node, wire.to_port
                )
            }
            GraphEvent::Batch {
                description,
                events,
            } => description
                .clone()
                .unwrap_or_else(|| format!("Batch of {} events", events.len())),
        }
    }
}

/// Event with timestamp and metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StampedEvent {
    /// Monotonic sequence number.
    pub sequence: u64,
    /// The event.
    pub event: GraphEvent,
    /// Optional human-readable description.
    pub description: Option<String>,
}

/// Event-sourced history manager.
///
/// Records all graph modifications as events. Supports undo/redo
/// by applying inverse events.
pub struct EventHistory {
    /// All events (oldest first).
    events: Vec<StampedEvent>,
    /// Next sequence number.
    next_sequence: u64,
    /// Undo stack (inverse events).
    undo_stack: Vec<GraphEvent>,
    /// Redo stack (events to re-apply).
    redo_stack: Vec<GraphEvent>,
}

impl EventHistory {
    /// Creates a new empty event history.
    pub fn new() -> Self {
        Self {
            events: Vec::new(),
            next_sequence: 0,
            undo_stack: Vec::new(),
            redo_stack: Vec::new(),
        }
    }

    /// Records an event and applies it to the graph.
    ///
    /// The event is applied immediately. If you want to record without applying,
    /// use `record_only`.
    pub fn record(
        &mut self,
        event: GraphEvent,
        graph: &mut Graph,
        registry: &NodeRegistry,
    ) -> Result<(), HistoryError> {
        // Clear redo stack when new event is recorded
        self.redo_stack.clear();

        // Store inverse for undo before applying
        let inverse = event.inverse();

        // Apply the event
        event.apply(graph, registry)?;

        // Enhance RemoveNode with captured wires if needed
        let event = self.enhance_event(event, graph);

        // Store stamped event
        let stamped = StampedEvent {
            sequence: self.next_sequence,
            event,
            description: None,
        };
        self.next_sequence += 1;
        self.events.push(stamped);

        // Store inverse for undo
        self.undo_stack.push(inverse);

        Ok(())
    }

    /// Enhances an event with captured state (e.g., wires for RemoveNode).
    fn enhance_event(&self, event: GraphEvent, graph: &Graph) -> GraphEvent {
        match event {
            GraphEvent::RemoveNode {
                id,
                type_name,
                params,
                ..
            } => {
                // Capture wires that were connected to this node
                let wires: Vec<Wire> = graph
                    .wires()
                    .iter()
                    .filter(|e| e.from_node == id || e.to_node == id)
                    .copied()
                    .collect();
                GraphEvent::RemoveNode {
                    id,
                    type_name,
                    params,
                    wires,
                }
            }
            other => other,
        }
    }

    /// Records an event without applying it.
    ///
    /// Use this when replaying events from a log.
    pub fn record_only(&mut self, event: GraphEvent) {
        let stamped = StampedEvent {
            sequence: self.next_sequence,
            event,
            description: None,
        };
        self.next_sequence += 1;
        self.events.push(stamped);
    }

    /// Undoes the last event.
    pub fn undo(
        &mut self,
        graph: &mut Graph,
        registry: &NodeRegistry,
    ) -> Result<bool, HistoryError> {
        if let Some(inverse) = self.undo_stack.pop() {
            // Apply inverse
            inverse.apply(graph, registry)?;

            // Store the inverse's inverse for redo
            self.redo_stack.push(inverse.inverse());

            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Redoes the last undone event.
    pub fn redo(
        &mut self,
        graph: &mut Graph,
        registry: &NodeRegistry,
    ) -> Result<bool, HistoryError> {
        if let Some(event) = self.redo_stack.pop() {
            // Apply event
            event.apply(graph, registry)?;

            // Store inverse for undo
            self.undo_stack.push(event.inverse());

            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Returns true if undo is possible.
    pub fn can_undo(&self) -> bool {
        !self.undo_stack.is_empty()
    }

    /// Returns true if redo is possible.
    pub fn can_redo(&self) -> bool {
        !self.redo_stack.is_empty()
    }

    /// Returns the number of recorded events.
    pub fn event_count(&self) -> usize {
        self.events.len()
    }

    /// Returns a reference to all events.
    pub fn events(&self) -> &[StampedEvent] {
        &self.events
    }

    /// Replays all events on an empty graph.
    pub fn replay(&self, registry: &NodeRegistry) -> Result<Graph, HistoryError> {
        let mut graph = Graph::new();
        for stamped in &self.events {
            stamped.event.apply(&mut graph, registry)?;
        }
        Ok(graph)
    }

    /// Clears all history.
    pub fn clear(&mut self) {
        self.events.clear();
        self.undo_stack.clear();
        self.redo_stack.clear();
        self.next_sequence = 0;
    }
}

impl Default for EventHistory {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rhizome_resin_core::{DynNode, EvalContext, GraphError, PortDescriptor, Value, ValueType};
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
            vec![PortDescriptor::new("in", ValueType::F32)]
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

    #[test]
    fn test_add_node_event() {
        let registry = make_registry();
        let mut history = EventHistory::new();
        let mut graph = Graph::new();

        let event = GraphEvent::add_node(0, "test::Node", serde_json::json!({"value": 42.0}));
        history.record(event, &mut graph, &registry).unwrap();

        assert_eq!(graph.node_count(), 1);
        assert_eq!(history.event_count(), 1);
    }

    #[test]
    fn test_undo_add_node() {
        let registry = make_registry();
        let mut history = EventHistory::new();
        let mut graph = Graph::new();

        let event = GraphEvent::add_node(0, "test::Node", serde_json::json!({"value": 42.0}));
        history.record(event, &mut graph, &registry).unwrap();

        assert_eq!(graph.node_count(), 1);
        assert!(history.can_undo());

        history.undo(&mut graph, &registry).unwrap();
        assert_eq!(graph.node_count(), 0);

        assert!(history.can_redo());
        history.redo(&mut graph, &registry).unwrap();
        assert_eq!(graph.node_count(), 1);
    }

    #[test]
    fn test_connect_disconnect() {
        let registry = make_registry();
        let mut history = EventHistory::new();
        let mut graph = Graph::new();

        // Add two nodes
        history
            .record(
                GraphEvent::add_node(0, "test::Node", serde_json::json!({"value": 1.0})),
                &mut graph,
                &registry,
            )
            .unwrap();
        history
            .record(
                GraphEvent::add_node(1, "test::Node", serde_json::json!({"value": 2.0})),
                &mut graph,
                &registry,
            )
            .unwrap();

        // Connect them
        history
            .record(GraphEvent::connect(0, 0, 1, 0), &mut graph, &registry)
            .unwrap();
        assert_eq!(graph.wire_count(), 1);

        // Undo connect
        history.undo(&mut graph, &registry).unwrap();
        assert_eq!(graph.wire_count(), 0);

        // Redo connect
        history.redo(&mut graph, &registry).unwrap();
        assert_eq!(graph.wire_count(), 1);
    }

    #[test]
    fn test_batch_event() {
        let registry = make_registry();
        let mut history = EventHistory::new();
        let mut graph = Graph::new();

        let batch = GraphEvent::batch(
            vec![
                GraphEvent::add_node(0, "test::Node", serde_json::json!({"value": 1.0})),
                GraphEvent::add_node(1, "test::Node", serde_json::json!({"value": 2.0})),
                GraphEvent::connect(0, 0, 1, 0),
            ],
            Some("Create two connected nodes".to_string()),
        );

        history.record(batch, &mut graph, &registry).unwrap();

        assert_eq!(graph.node_count(), 2);
        assert_eq!(graph.wire_count(), 1);

        // Undo entire batch
        history.undo(&mut graph, &registry).unwrap();
        assert_eq!(graph.node_count(), 0);
        assert_eq!(graph.wire_count(), 0);
    }

    #[test]
    fn test_replay() {
        let registry = make_registry();
        let mut history = EventHistory::new();
        let mut graph = Graph::new();

        history
            .record(
                GraphEvent::add_node(0, "test::Node", serde_json::json!({"value": 1.0})),
                &mut graph,
                &registry,
            )
            .unwrap();
        history
            .record(
                GraphEvent::add_node(1, "test::Node", serde_json::json!({"value": 2.0})),
                &mut graph,
                &registry,
            )
            .unwrap();
        history
            .record(GraphEvent::connect(0, 0, 1, 0), &mut graph, &registry)
            .unwrap();

        // Replay from scratch
        let replayed = history.replay(&registry).unwrap();
        assert_eq!(replayed.node_count(), 2);
        assert_eq!(replayed.wire_count(), 1);
    }

    #[test]
    fn test_event_description() {
        let event = GraphEvent::add_node(0, "test::Node", serde_json::json!({}));
        assert!(event.description().contains("Add node"));
        assert!(event.description().contains("test::Node"));
    }
}
