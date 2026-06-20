//! Graph container and execution.
//!
//! This module provides a data flow graph where nodes process values
//! and wires connect output ports to input ports.
//!
//! # Example
//!
//! ```ignore
//! use unshape_core::{Graph, EvalContext};
//!
//! let mut graph = Graph::new();
//!
//! // Add nodes (must implement DynNode)
//! let a = graph.add_node(ConstNode { value: 2.0 });
//! let b = graph.add_node(ConstNode { value: 3.0 });
//! let add = graph.add_node(AddNode);
//!
//! // Connect: a.output[0] -> add.input[0], b.output[0] -> add.input[1]
//! graph.connect(a, 0, add, 0)?;
//! graph.connect(b, 0, add, 1)?;
//!
//! // Execute (eager: runs all nodes in topological order)
//! let result = graph.execute(add)?;
//! assert_eq!(result[0].as_f32()?, 5.0);
//!
//! // Or with custom context for time/cancellation
//! let ctx = EvalContext::new().with_time(1.0, 60, 1.0/60.0);
//! let result = graph.execute_with_context(add, &ctx)?;
//! ```
//!
//! For lazy evaluation with caching, see [`crate::LazyEvaluator`].
//!
//! # Terminology
//!
//! This module uses "Wire" for connections between node ports to distinguish
//! from geometric edges in mesh/vector domains (see `docs/conventions.md`).

use std::collections::HashMap;

use crate::error::GraphError;
use crate::eval::{EvalContext, FeedbackState, SeekBehavior, TickResult};
use crate::node::{BoxedNode, DynNode};
use crate::nodes::{GraphInput, GraphOutput};
use crate::value::{Value, ValueType};

/// Information about a [`GraphInput`] node in the graph.
#[derive(Debug, Clone)]
pub struct GraphInputInfo {
    /// The ID of the `GraphInput` node.
    pub node_id: NodeId,
    /// The host-facing name used to look up the value in [`EvalContext`].
    pub name: String,
    /// The declared value type for this input.
    pub value_type: ValueType,
}

/// Information about a [`GraphOutput`] node in the graph.
#[derive(Debug, Clone)]
pub struct GraphOutputInfo {
    /// The ID of the `GraphOutput` node.
    pub node_id: NodeId,
    /// The host-facing name identifying this output.
    pub name: String,
    /// The declared value type for this output.
    pub value_type: ValueType,
}

/// Unique identifier for a node in a graph.
pub type NodeId = u32;

/// A wire connecting an output port to an input port.
///
/// Wires carry data from one node's output to another node's input.
///
/// # Feedback wires
///
/// A *feedback* wire (`feedback == true`) is a back-edge in a recurrent graph:
/// instead of carrying a value computed within the current tick, it carries the
/// value its source produced on the *previous* tick. This makes cycles
/// well-defined — the graph is acyclic *per tick* because feedback wires are not
/// part of the within-tick dependency order. State lives on the feedback wire,
/// not inside the node, so nodes remain pure `&self` functions.
///
/// See `docs/design/recurrent-graphs.md` and [`Graph::tick`].
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Wire {
    /// Source node.
    pub from_node: NodeId,
    /// Output port index on source node.
    pub from_port: usize,
    /// Destination node.
    pub to_node: NodeId,
    /// Input port index on destination node.
    pub to_port: usize,
    /// Whether this wire is a feedback (back-)edge carrying the previous tick's value.
    ///
    /// Direct wires (`false`) participate in the within-tick topological order.
    /// Feedback wires (`true`) are excluded from it and instead seed their
    /// destination input from [`FeedbackState`](crate::FeedbackState).
    ///
    /// Kept as a plain `bool` so `Wire` stays `Copy`. Per-edge initial values for
    /// tick 0 are not stored on the wire (a `Value` is not `Copy`); pre-seed them
    /// into the [`FeedbackState`](crate::FeedbackState) before tick 0, otherwise
    /// the destination port's zero value is used (see [`Graph::tick`]).
    #[cfg_attr(feature = "serde", serde(default))]
    pub feedback: bool,
}

impl Wire {
    /// Creates a normal (direct) wire.
    pub fn direct(from_node: NodeId, from_port: usize, to_node: NodeId, to_port: usize) -> Self {
        Self {
            from_node,
            from_port,
            to_node,
            to_port,
            feedback: false,
        }
    }

    /// Creates a feedback (back-edge) wire.
    pub fn feedback(from_node: NodeId, from_port: usize, to_node: NodeId, to_port: usize) -> Self {
        Self {
            from_node,
            from_port,
            to_node,
            to_port,
            feedback: true,
        }
    }
}

/// A graph of nodes connected by wires.
#[derive(Default)]
pub struct Graph {
    nodes: HashMap<NodeId, BoxedNode>,
    wires: Vec<Wire>,
    next_id: NodeId,
    /// Cached topological order. Invalidated on structure change.
    topo_order: Option<Vec<NodeId>>,
}

impl Graph {
    /// Creates an empty graph.
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a node to the graph and returns its ID.
    pub fn add_node<N: DynNode + 'static>(&mut self, node: N) -> NodeId {
        let id = self.next_id;
        self.next_id += 1;
        self.nodes.insert(id, Box::new(node));
        self.topo_order = None; // Invalidate cache
        id
    }

    /// Connects an output port to an input port.
    ///
    /// # Arguments
    /// * `from_node` - Source node ID.
    /// * `from_port` - Output port index on source.
    /// * `to_node` - Destination node ID.
    /// * `to_port` - Input port index on destination.
    pub fn connect(
        &mut self,
        from_node: NodeId,
        from_port: usize,
        to_node: NodeId,
        to_port: usize,
    ) -> Result<(), GraphError> {
        // Validate nodes exist
        let from = self
            .nodes
            .get(&from_node)
            .ok_or(GraphError::NodeNotFound(from_node))?;
        let to = self
            .nodes
            .get(&to_node)
            .ok_or(GraphError::NodeNotFound(to_node))?;

        let from_outputs = from.outputs();
        let to_inputs = to.inputs();

        // Validate ports exist
        if from_port >= from_outputs.len() {
            return Err(GraphError::PortNotFound {
                node: from_node,
                port: from_port,
            });
        }
        if to_port >= to_inputs.len() {
            return Err(GraphError::PortNotFound {
                node: to_node,
                port: to_port,
            });
        }

        // Validate types match
        let from_type = from_outputs[from_port].value_type;
        let to_type = to_inputs[to_port].value_type;
        if from_type != to_type {
            return Err(GraphError::TypeMismatch {
                expected: to_type,
                got: from_type,
            });
        }

        self.wires
            .push(Wire::direct(from_node, from_port, to_node, to_port));
        self.topo_order = None; // Invalidate cache

        Ok(())
    }

    /// Connects an output port to an input port as a **feedback (back-)edge**.
    ///
    /// A feedback wire carries the source's value from the *previous* tick. It
    /// does not participate in the within-tick topological order, so it may form
    /// a cycle that a direct wire could not. The destination input is seeded from
    /// [`FeedbackState`](crate::FeedbackState) (or an initial value on tick 0) by
    /// the recurrent driver — see [`Graph::tick`].
    ///
    /// Type checking is identical to [`connect`](Self::connect).
    pub fn connect_feedback(
        &mut self,
        from_node: NodeId,
        from_port: usize,
        to_node: NodeId,
        to_port: usize,
    ) -> Result<(), GraphError> {
        let from = self
            .nodes
            .get(&from_node)
            .ok_or(GraphError::NodeNotFound(from_node))?;
        let to = self
            .nodes
            .get(&to_node)
            .ok_or(GraphError::NodeNotFound(to_node))?;

        let from_outputs = from.outputs();
        let to_inputs = to.inputs();

        if from_port >= from_outputs.len() {
            return Err(GraphError::PortNotFound {
                node: from_node,
                port: from_port,
            });
        }
        if to_port >= to_inputs.len() {
            return Err(GraphError::PortNotFound {
                node: to_node,
                port: to_port,
            });
        }

        let from_type = from_outputs[from_port].value_type;
        let to_type = to_inputs[to_port].value_type;
        if from_type != to_type {
            return Err(GraphError::TypeMismatch {
                expected: to_type,
                got: from_type,
            });
        }

        self.wires
            .push(Wire::feedback(from_node, from_port, to_node, to_port));
        self.topo_order = None;

        Ok(())
    }

    /// Returns the topological order of nodes, computing it if needed.
    fn topological_order(&mut self) -> Result<&[NodeId], GraphError> {
        if self.topo_order.is_none() {
            self.topo_order = Some(self.compute_topological_order()?);
        }
        Ok(self.topo_order.as_ref().unwrap())
    }

    /// Computes topological order using Kahn's algorithm.
    fn compute_topological_order(&self) -> Result<Vec<NodeId>, GraphError> {
        let mut in_degree: HashMap<NodeId, usize> = HashMap::new();
        let mut adj: HashMap<NodeId, Vec<NodeId>> = HashMap::new();

        // Initialize in-degrees
        for &id in self.nodes.keys() {
            in_degree.insert(id, 0);
            adj.insert(id, Vec::new());
        }

        // Build adjacency list and count in-degrees.
        // Feedback (back-)edges are excluded: they carry the *previous* tick's
        // value, so they do not constrain within-tick evaluation order. This is
        // what lets a recurrent graph remain acyclic per tick.
        for edge in &self.wires {
            if edge.feedback {
                continue;
            }
            adj.get_mut(&edge.from_node).unwrap().push(edge.to_node);
            *in_degree.get_mut(&edge.to_node).unwrap() += 1;
        }

        // Start with nodes that have no incoming edges
        let mut queue: Vec<NodeId> = in_degree
            .iter()
            .filter(|&(_, deg)| *deg == 0)
            .map(|(&id, _)| id)
            .collect();

        let mut result = Vec::with_capacity(self.nodes.len());

        while let Some(node) = queue.pop() {
            result.push(node);

            for &neighbor in &adj[&node] {
                let deg = in_degree.get_mut(&neighbor).unwrap();
                *deg -= 1;
                if *deg == 0 {
                    queue.push(neighbor);
                }
            }
        }

        if result.len() != self.nodes.len() {
            return Err(GraphError::CycleDetected);
        }

        Ok(result)
    }

    /// Executes the graph and returns outputs from the specified node.
    ///
    /// Uses a default `EvalContext`. For custom context (time, cancellation, etc.),
    /// use `execute_with_context`.
    ///
    /// # Arguments
    /// * `output_node` - The node whose outputs to return.
    pub fn execute(&mut self, output_node: NodeId) -> Result<Vec<Value>, GraphError> {
        self.execute_with_context(output_node, &EvalContext::new())
    }

    /// Executes the graph with a custom evaluation context.
    ///
    /// # Arguments
    /// * `output_node` - The node whose outputs to return.
    /// * `ctx` - Evaluation context (time, cancellation, quality hints, etc.)
    pub fn execute_with_context(
        &mut self,
        output_node: NodeId,
        ctx: &EvalContext,
    ) -> Result<Vec<Value>, GraphError> {
        let order = self.topological_order()?.to_vec();

        // Storage for computed values: (node_id, port_index) -> Value
        let mut values: HashMap<(NodeId, usize), Value> = HashMap::new();

        for node_id in order {
            // Check for cancellation between nodes
            if ctx.is_cancelled() {
                return Err(GraphError::Cancelled);
            }

            let node = self.nodes.get(&node_id).unwrap();
            let inputs_desc = node.inputs();
            let num_inputs = inputs_desc.len();

            // Gather inputs for this node
            let mut inputs = Vec::with_capacity(num_inputs);
            for port in 0..num_inputs {
                // Find the direct wire that feeds this input. Feedback wires are
                // ignored here: this path has no previous-tick state to seed
                // from. For recurrent evaluation use [`Graph::tick`].
                let wire = self
                    .wires
                    .iter()
                    .find(|w| !w.feedback && w.to_node == node_id && w.to_port == port);

                match wire {
                    Some(e) => {
                        let value = values
                            .get(&(e.from_node, e.from_port))
                            .cloned()
                            .ok_or_else(|| {
                                GraphError::ExecutionError(format!(
                                    "missing value for node {} port {}",
                                    e.from_node, e.from_port
                                ))
                            })?;
                        inputs.push(value);
                    }
                    None => {
                        return Err(GraphError::UnconnectedInput {
                            node: node_id,
                            port,
                        });
                    }
                }
            }

            // Execute node
            let outputs = node.execute(&inputs, ctx)?;

            // Store outputs
            for (port, value) in outputs.into_iter().enumerate() {
                values.insert((node_id, port), value);
            }
        }

        // Collect outputs from the requested node
        let node = self
            .nodes
            .get(&output_node)
            .ok_or(GraphError::NodeNotFound(output_node))?;

        let outputs_desc = node.outputs();
        let mut result = Vec::with_capacity(outputs_desc.len());
        for port in 0..outputs_desc.len() {
            let value = values.get(&(output_node, port)).cloned().ok_or_else(|| {
                GraphError::ExecutionError(format!(
                    "missing output for node {} port {}",
                    output_node, port
                ))
            })?;
            result.push(value);
        }

        Ok(result)
    }

    /// Returns the number of nodes in the graph.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Returns the number of wires in the graph.
    pub fn wire_count(&self) -> usize {
        self.wires.len()
    }

    /// Returns the next node ID that will be assigned.
    pub fn next_id(&self) -> NodeId {
        self.next_id
    }

    /// Returns a slice of all wires.
    pub fn wires(&self) -> &[Wire] {
        &self.wires
    }

    /// Iterates over all (NodeId, &BoxedNode) pairs.
    pub fn nodes_iter(&self) -> impl Iterator<Item = (NodeId, &BoxedNode)> {
        self.nodes.iter().map(|(&id, node)| (id, node))
    }

    /// Gets a node by ID.
    pub fn get_node(&self, id: NodeId) -> Option<&BoxedNode> {
        self.nodes.get(&id)
    }

    /// Creates a graph with a specific next_id (for deserialization).
    pub fn with_next_id(next_id: NodeId) -> Self {
        Self {
            next_id,
            ..Default::default()
        }
    }

    /// Inserts a node with a specific ID (for deserialization).
    ///
    /// Returns an error if a node with that ID already exists.
    pub fn insert_node_with_id(&mut self, id: NodeId, node: BoxedNode) -> Result<(), GraphError> {
        if self.nodes.contains_key(&id) {
            return Err(GraphError::NodeAlreadyExists(id));
        }
        self.nodes.insert(id, node);
        if id >= self.next_id {
            self.next_id = id + 1;
        }
        self.topo_order = None;
        Ok(())
    }

    /// Removes a node and all its connected wires.
    pub fn remove_node(&mut self, id: NodeId) -> Result<BoxedNode, GraphError> {
        let node = self.nodes.remove(&id).ok_or(GraphError::NodeNotFound(id))?;
        self.wires.retain(|w| w.from_node != id && w.to_node != id);
        self.topo_order = None;
        Ok(node)
    }

    /// Disconnects a specific wire.
    pub fn disconnect(
        &mut self,
        from_node: NodeId,
        from_port: usize,
        to_node: NodeId,
        to_port: usize,
    ) -> Result<(), GraphError> {
        let idx = self
            .wires
            .iter()
            .position(|w| {
                w.from_node == from_node
                    && w.from_port == from_port
                    && w.to_node == to_node
                    && w.to_port == to_port
            })
            .ok_or(GraphError::WireNotFound)?;
        self.wires.remove(idx);
        self.topo_order = None;
        Ok(())
    }

    /// Replaces a node with a new one, keeping the same ID.
    pub fn replace_node(&mut self, id: NodeId, node: BoxedNode) -> Result<BoxedNode, GraphError> {
        let old = self.nodes.remove(&id).ok_or(GraphError::NodeNotFound(id))?;
        self.nodes.insert(id, node);
        self.topo_order = None;
        Ok(old)
    }

    /// Returns information about all [`GraphInput`] nodes in the graph.
    ///
    /// Each entry carries the node's ID, its host-facing name, and declared value type.
    /// The order of the returned slice is unspecified (depends on `HashMap` iteration).
    pub fn input_nodes(&self) -> Vec<GraphInputInfo> {
        self.nodes
            .iter()
            .filter_map(|(&id, node)| {
                node.as_any()
                    .downcast_ref::<GraphInput>()
                    .map(|gi| GraphInputInfo {
                        node_id: id,
                        name: gi.name.clone(),
                        value_type: gi.value_type,
                    })
            })
            .collect()
    }

    /// Returns information about all [`GraphOutput`] nodes in the graph.
    ///
    /// Each entry carries the node's ID, its host-facing name, and declared value type.
    /// The order of the returned slice is unspecified (depends on `HashMap` iteration).
    pub fn output_nodes(&self) -> Vec<GraphOutputInfo> {
        self.nodes
            .iter()
            .filter_map(|(&id, node)| {
                node.as_any()
                    .downcast_ref::<GraphOutput>()
                    .map(|go| GraphOutputInfo {
                        node_id: id,
                        name: go.name.clone(),
                        value_type: go.value_type,
                    })
            })
            .collect()
    }

    /// Executes all [`GraphOutput`] nodes and returns a map of output name → value.
    ///
    /// For each `GraphOutput` node the graph finds the wire feeding its `"value"`
    /// input port, executes the upstream source node (using the normal topological
    /// execution path), and maps the output name to the resulting value.
    ///
    /// # Notes
    ///
    /// - If two `GraphOutput` nodes share the same name, the last one processed
    ///   wins (order is unspecified). Prefer unique names.
    /// - The supplied `EvalContext` is shared across all output nodes; host inputs
    ///   should be injected there.
    pub fn execute_named_outputs(
        &mut self,
        ctx: &EvalContext,
    ) -> Result<HashMap<String, Value>, GraphError> {
        let output_infos = self.output_nodes();
        let mut results = HashMap::with_capacity(output_infos.len());
        for info in output_infos {
            // Find the wire feeding input port 0 of this GraphOutput node.
            let wire = self
                .wires
                .iter()
                .find(|w| w.to_node == info.node_id && w.to_port == 0)
                .copied();

            if let Some(w) = wire {
                // Execute the upstream source node and pick the correct output port.
                let upstream_outputs = self.execute_with_context(w.from_node, ctx)?;
                if let Some(value) = upstream_outputs.into_iter().nth(w.from_port) {
                    results.insert(info.name, value);
                }
            }
        }
        Ok(results)
    }

    /// Returns `true` if any wire in the graph is a feedback (back-)edge.
    pub fn has_feedback(&self) -> bool {
        self.wires.iter().any(|w| w.feedback)
    }

    /// Evaluates one tick of a recurrent graph.
    ///
    /// This is the core of the "feedback wires ARE the state" model
    /// (`docs/design/recurrent-graphs.md`). Nodes stay pure (`&self`); all state
    /// that crosses tick boundaries lives in `state`.
    ///
    /// Per tick:
    /// 1. **Seed** each feedback wire's *destination* input from `state` (the
    ///    value its source produced last tick). On tick 0, or whenever `state`
    ///    has no entry for that source, the zero value of the destination port
    ///    type is used (see [`ValueType::zero_value`](crate::ValueType::zero_value)).
    /// 2. **Evaluate** the now-acyclic DAG in topological order. Direct wires
    ///    read from values computed this tick; feedback wires read from the seed.
    /// 3. **Write back** each feedback wire's *source* output into `state` for
    ///    the next tick.
    ///
    /// # Caching interaction
    ///
    /// This driver performs a full eager pass each tick and does **not** consult
    /// the content-addressed [`EvalCache`](crate::EvalCache). That cache keys on
    /// `(node_id, hash(inputs))`; for nodes downstream of a feedback edge the
    /// inputs change every tick, so caching them is unsound across ticks unless
    /// the tick index is folded into the key. Pure DAG sub-regions (no feedback
    /// upstream) are still safely cacheable via [`LazyEvaluator`] in the
    /// non-recurrent paths; mixing per-tick caching into this driver is left as
    /// future work to avoid a subtly wrong cache.
    ///
    /// `ctx` should carry the tick's time/frame/seed. Its feedback-state slot is
    /// ignored here — `state` is the source of truth.
    pub fn tick(
        &mut self,
        tick: u64,
        state: &mut FeedbackState,
        ctx: &EvalContext,
    ) -> Result<TickResult, GraphError> {
        let order = self.topological_order()?.to_vec();

        // Computed within-tick values: (node_id, output_port) -> Value
        let mut values: HashMap<(NodeId, usize), Value> = HashMap::new();

        for node_id in order {
            if ctx.is_cancelled() {
                return Err(GraphError::Cancelled);
            }

            let node = self.nodes.get(&node_id).unwrap();
            let inputs_desc = node.inputs();
            let num_inputs = inputs_desc.len();

            let mut inputs = Vec::with_capacity(num_inputs);
            for (port, desc) in inputs_desc.iter().enumerate() {
                // Prefer a direct wire; fall back to a feedback wire (seeded).
                let direct = self
                    .wires
                    .iter()
                    .find(|w| !w.feedback && w.to_node == node_id && w.to_port == port);

                if let Some(e) = direct {
                    let value = values
                        .get(&(e.from_node, e.from_port))
                        .cloned()
                        .ok_or_else(|| {
                            GraphError::ExecutionError(format!(
                                "missing value for node {} port {}",
                                e.from_node, e.from_port
                            ))
                        })?;
                    inputs.push(value);
                    continue;
                }

                let feedback = self
                    .wires
                    .iter()
                    .find(|w| w.feedback && w.to_node == node_id && w.to_port == port);

                match feedback {
                    Some(fb) => {
                        // Seed from previous-tick state, else the port's zero value.
                        let seeded = match state.get(fb.from_node, fb.from_port) {
                            Some(v) => v.clone(),
                            None => desc.value_type.zero_value().ok_or_else(|| {
                                GraphError::ExecutionError(format!(
                                    "feedback input node {} port {} has no initial value and \
                                     type {:?} has no zero value; seed it via FeedbackState",
                                    node_id, port, desc.value_type
                                ))
                            })?,
                        };
                        inputs.push(seeded);
                    }
                    None => {
                        return Err(GraphError::UnconnectedInput {
                            node: node_id,
                            port,
                        });
                    }
                }
            }

            let outputs = node.execute(&inputs, ctx)?;
            for (port, value) in outputs.into_iter().enumerate() {
                values.insert((node_id, port), value);
            }
        }

        // Write feedback sources back into state for the next tick.
        for w in self.wires.iter().filter(|w| w.feedback) {
            if let Some(v) = values.get(&(w.from_node, w.from_port)) {
                state.set(w.from_node, w.from_port, v.clone());
            }
        }

        Ok(TickResult {
            tick,
            outputs: values,
        })
    }

    /// Deterministically runs a recurrent graph from tick 0 up to `target_tick`
    /// (inclusive), returning the [`TickResult`] of the final tick.
    ///
    /// This implements [`SeekBehavior::Resimulate`]: it clears `state` and
    /// replays every tick `0..=target_tick`, so the result is identical to
    /// stepping [`tick`](Self::tick) that many times from a fresh state. This is
    /// the correct (if O(N)) way to seek a stateful graph.
    ///
    /// `make_ctx(tick)` builds the per-tick context (time/frame/seed). Keeping it
    /// a closure keeps determinism the caller's explicit choice: same closure +
    /// same graph + same `target_tick` ⇒ identical output.
    ///
    /// # SeekBehavior coverage
    ///
    /// - [`SeekBehavior::Resimulate`] — fully implemented here.
    /// - [`SeekBehavior::Discontinuity`] — see [`seek`](Self::seek), which applies
    ///   a single tick at the target using whatever state is supplied (hook).
    /// - [`SeekBehavior::Error`] — see [`seek`](Self::seek) (returns
    ///   [`GraphError::SeekUnsupported`]).
    pub fn run_to_tick(
        &mut self,
        target_tick: u64,
        state: &mut FeedbackState,
        make_ctx: impl Fn(u64) -> EvalContext,
    ) -> Result<TickResult, GraphError> {
        state.clear();
        let mut last: Option<TickResult> = None;
        for t in 0..=target_tick {
            let ctx = make_ctx(t);
            last = Some(self.tick(t, state, &ctx)?);
        }
        // `0..=target_tick` always runs at least tick 0, so `last` is Some.
        Ok(last.expect("run_to_tick evaluates at least tick 0"))
    }

    /// Seeks to `target_tick` according to `behavior`.
    ///
    /// - [`SeekBehavior::Resimulate`]: replays from tick 0 (clears `state`) via
    ///   [`run_to_tick`](Self::run_to_tick) — correct, O(N).
    /// - [`SeekBehavior::Discontinuity`]: applies a *single* tick at
    ///   `target_tick` using the `state` as-is (no replay). Fast, may glitch.
    ///   This is the minimal hook; per-domain fixups are the caller's job.
    /// - [`SeekBehavior::Error`]: returns [`GraphError::SeekUnsupported`] unless
    ///   `target_tick == current_tick` (a no-op seek), in which case it applies
    ///   that single tick.
    pub fn seek(
        &mut self,
        target_tick: u64,
        current_tick: u64,
        behavior: SeekBehavior,
        state: &mut FeedbackState,
        make_ctx: impl Fn(u64) -> EvalContext,
    ) -> Result<TickResult, GraphError> {
        match behavior {
            SeekBehavior::Resimulate => self.run_to_tick(target_tick, state, make_ctx),
            SeekBehavior::Discontinuity => {
                let ctx = make_ctx(target_tick);
                self.tick(target_tick, state, &ctx)
            }
            SeekBehavior::Error => {
                if target_tick == current_tick {
                    let ctx = make_ctx(target_tick);
                    self.tick(target_tick, state, &ctx)
                } else {
                    Err(GraphError::SeekUnsupported {
                        target: target_tick,
                        current: current_tick,
                    })
                }
            }
        }
    }
}

#[cfg(feature = "named-ports")]
impl Graph {
    /// Connects nodes by port name instead of index.
    ///
    /// Looks up `from_port` in the source node's output port names,
    /// and `to_port` in the destination node's input port names,
    /// then delegates to [`connect`].
    pub fn connect_named(
        &mut self,
        from_node: NodeId,
        from_port: &str,
        to_node: NodeId,
        to_port: &str,
    ) -> Result<(), GraphError> {
        let from_idx = {
            let node = self
                .nodes
                .get(&from_node)
                .ok_or(GraphError::NodeNotFound(from_node))?;
            node.output_port_names()
                .into_iter()
                .position(|n| n == from_port)
                .ok_or_else(|| GraphError::PortNameNotFound {
                    node: from_node,
                    name: from_port.to_string(),
                })?
        };
        let to_idx = {
            let node = self
                .nodes
                .get(&to_node)
                .ok_or(GraphError::NodeNotFound(to_node))?;
            node.input_port_names()
                .into_iter()
                .position(|n| n == to_port)
                .ok_or_else(|| GraphError::PortNameNotFound {
                    node: to_node,
                    name: to_port.to_string(),
                })?
        };
        self.connect(from_node, from_idx, to_node, to_idx)
    }

    /// Disconnects nodes by port name instead of index.
    pub fn disconnect_named(
        &mut self,
        from_node: NodeId,
        from_port: &str,
        to_node: NodeId,
        to_port: &str,
    ) -> Result<(), GraphError> {
        let from_idx = {
            let node = self
                .nodes
                .get(&from_node)
                .ok_or(GraphError::NodeNotFound(from_node))?;
            node.output_port_names()
                .into_iter()
                .position(|n| n == from_port)
                .ok_or_else(|| GraphError::PortNameNotFound {
                    node: from_node,
                    name: from_port.to_string(),
                })?
        };
        let to_idx = {
            let node = self
                .nodes
                .get(&to_node)
                .ok_or(GraphError::NodeNotFound(to_node))?;
            node.input_port_names()
                .into_iter()
                .position(|n| n == to_port)
                .ok_or_else(|| GraphError::PortNameNotFound {
                    node: to_node,
                    name: to_port.to_string(),
                })?
        };
        self.disconnect(from_node, from_idx, to_node, to_idx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::EvalContext;
    use crate::node::PortDescriptor;
    use crate::value::ValueType;
    use std::any::Any;

    // Test node: adds two f32 values
    struct AddNode;

    impl DynNode for AddNode {
        fn type_name(&self) -> &'static str {
            "Add"
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

        fn as_any(&self) -> &dyn Any {
            self
        }
    }

    // Test node: outputs a constant f32
    struct ConstNode(f32);

    impl DynNode for ConstNode {
        fn type_name(&self) -> &'static str {
            "Const"
        }

        fn inputs(&self) -> Vec<PortDescriptor> {
            vec![]
        }

        fn outputs(&self) -> Vec<PortDescriptor> {
            vec![PortDescriptor::new("value", ValueType::F32)]
        }

        fn execute(&self, _inputs: &[Value], _ctx: &EvalContext) -> Result<Vec<Value>, GraphError> {
            Ok(vec![Value::F32(self.0)])
        }

        fn as_any(&self) -> &dyn Any {
            self
        }
    }

    #[test]
    fn test_simple_graph() {
        let mut graph = Graph::new();

        let const_a = graph.add_node(ConstNode(2.0));
        let const_b = graph.add_node(ConstNode(3.0));
        let add = graph.add_node(AddNode);

        graph.connect(const_a, 0, add, 0).unwrap();
        graph.connect(const_b, 0, add, 1).unwrap();

        let outputs = graph.execute(add).unwrap();

        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].as_f32().unwrap(), 5.0);
    }

    #[test]
    fn test_type_mismatch() {
        struct BoolNode;

        impl DynNode for BoolNode {
            fn type_name(&self) -> &'static str {
                "Bool"
            }

            fn inputs(&self) -> Vec<PortDescriptor> {
                vec![]
            }

            fn outputs(&self) -> Vec<PortDescriptor> {
                vec![PortDescriptor::new("value", ValueType::Bool)]
            }

            fn execute(
                &self,
                _inputs: &[Value],
                _ctx: &EvalContext,
            ) -> Result<Vec<Value>, GraphError> {
                Ok(vec![Value::Bool(true)])
            }

            fn as_any(&self) -> &dyn Any {
                self
            }
        }

        let mut graph = Graph::new();
        let bool_node = graph.add_node(BoolNode);
        let add = graph.add_node(AddNode);

        // This should fail - bool output to f32 input
        let result = graph.connect(bool_node, 0, add, 0);
        assert!(matches!(result, Err(GraphError::TypeMismatch { .. })));
    }

    #[test]
    fn test_cycle_detection() {
        struct PassthroughNode;

        impl DynNode for PassthroughNode {
            fn type_name(&self) -> &'static str {
                "Passthrough"
            }

            fn inputs(&self) -> Vec<PortDescriptor> {
                vec![PortDescriptor::new("in", ValueType::F32)]
            }

            fn outputs(&self) -> Vec<PortDescriptor> {
                vec![PortDescriptor::new("out", ValueType::F32)]
            }

            fn execute(
                &self,
                inputs: &[Value],
                _ctx: &EvalContext,
            ) -> Result<Vec<Value>, GraphError> {
                Ok(vec![inputs[0].clone()])
            }

            fn as_any(&self) -> &dyn Any {
                self
            }
        }

        let mut graph = Graph::new();
        let a = graph.add_node(PassthroughNode);
        let b = graph.add_node(PassthroughNode);

        graph.connect(a, 0, b, 0).unwrap();
        graph.connect(b, 0, a, 0).unwrap(); // Creates cycle

        let result = graph.execute(a);
        assert!(matches!(result, Err(GraphError::CycleDetected)));
    }

    #[test]
    fn test_derive_macro() {
        use crate::DynNodeDerive;

        #[derive(DynNodeDerive, Clone, Default)]
        #[node(crate = "crate")]
        struct DerivedAdd {
            #[input]
            a: f32,
            #[input]
            b: f32,
            #[output]
            result: f32,
        }

        impl DerivedAdd {
            fn compute(&mut self) {
                self.result = self.a + self.b;
            }
        }

        // Test the derived implementation
        let node = DerivedAdd::default();
        assert_eq!(node.type_name(), "DerivedAdd");

        let inputs = node.inputs();
        assert_eq!(inputs.len(), 2);
        assert_eq!(inputs[0].name, "a");
        assert_eq!(inputs[1].name, "b");

        let outputs = node.outputs();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].name, "result");

        // Test execution
        let ctx = EvalContext::new();
        let result = node
            .execute(&[Value::F32(10.0), Value::F32(5.0)], &ctx)
            .unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].as_f32().unwrap(), 15.0);
    }

    #[test]
    fn test_derived_node_in_graph() {
        use crate::DynNodeDerive;

        #[derive(DynNodeDerive, Clone, Default)]
        #[node(crate = "crate")]
        struct Multiply {
            #[input]
            a: f32,
            #[input]
            b: f32,
            #[output]
            result: f32,
        }

        impl Multiply {
            fn compute(&mut self) {
                self.result = self.a * self.b;
            }
        }

        let mut graph = Graph::new();
        let c1 = graph.add_node(ConstNode(3.0));
        let c2 = graph.add_node(ConstNode(4.0));
        let mul = graph.add_node(Multiply::default());

        graph.connect(c1, 0, mul, 0).unwrap();
        graph.connect(c2, 0, mul, 1).unwrap();

        let outputs = graph.execute(mul).unwrap();
        assert_eq!(outputs[0].as_f32().unwrap(), 12.0);
    }

    #[test]
    fn test_lazy_evaluator() {
        use crate::eval::{Evaluator, LazyEvaluator};

        let mut graph = Graph::new();
        let const_a = graph.add_node(ConstNode(2.0));
        let const_b = graph.add_node(ConstNode(3.0));
        let add = graph.add_node(AddNode);

        graph.connect(const_a, 0, add, 0).unwrap();
        graph.connect(const_b, 0, add, 1).unwrap();

        let mut evaluator = LazyEvaluator::new();
        let ctx = EvalContext::new();

        // First evaluation - should compute all nodes
        let result = evaluator.evaluate(&graph, &[add], &ctx).unwrap();
        assert_eq!(result.outputs.len(), 1);
        assert_eq!(result.outputs[0][0].as_f32().unwrap(), 5.0);
        assert_eq!(result.computed_nodes.len(), 3); // const_a, const_b, add

        // Second evaluation - should use cache
        let result2 = evaluator.evaluate(&graph, &[add], &ctx).unwrap();
        assert_eq!(result2.outputs[0][0].as_f32().unwrap(), 5.0);
        // All nodes should be served from cache now
        assert_eq!(result2.computed_nodes.len(), 0);
        assert_eq!(result2.cached_nodes.len(), 3); // const_a, const_b, add
    }

    #[test]
    fn test_lazy_evaluator_partial() {
        use crate::eval::{Evaluator, LazyEvaluator};

        // Build a diamond graph:
        //   A
        //  / \
        // B   C
        //  \ /
        //   D
        let mut graph = Graph::new();
        let a = graph.add_node(ConstNode(10.0));
        let b = graph.add_node(AddNode);
        let c = graph.add_node(AddNode);
        let d = graph.add_node(AddNode);

        // B = A + 0 (we'll use a const for the second input)
        let zero1 = graph.add_node(ConstNode(0.0));
        graph.connect(a, 0, b, 0).unwrap();
        graph.connect(zero1, 0, b, 1).unwrap();

        // C = A + 1
        let one = graph.add_node(ConstNode(1.0));
        graph.connect(a, 0, c, 0).unwrap();
        graph.connect(one, 0, c, 1).unwrap();

        // D = B + C
        graph.connect(b, 0, d, 0).unwrap();
        graph.connect(c, 0, d, 1).unwrap();

        let mut evaluator = LazyEvaluator::new();
        let ctx = EvalContext::new();

        // Request only B - should only evaluate A, zero1, and B
        let result = evaluator.evaluate(&graph, &[b], &ctx).unwrap();
        assert_eq!(result.outputs[0][0].as_f32().unwrap(), 10.0);

        // Request D - should use cached A (via B's cache), compute C and D
        let result2 = evaluator.evaluate(&graph, &[d], &ctx).unwrap();
        // D = B + C = 10 + 11 = 21
        assert_eq!(result2.outputs[0][0].as_f32().unwrap(), 21.0);
    }

    #[test]
    fn test_lazy_evaluator_cancellation() {
        use crate::eval::{Evaluator, LazyEvaluator};

        let mut graph = Graph::new();
        let const_a = graph.add_node(ConstNode(2.0));
        let const_b = graph.add_node(ConstNode(3.0));
        let add = graph.add_node(AddNode);

        graph.connect(const_a, 0, add, 0).unwrap();
        graph.connect(const_b, 0, add, 1).unwrap();

        let mut evaluator = LazyEvaluator::new();
        let token = crate::CancellationToken::new();
        token.cancel(); // Cancel before evaluation

        let ctx = EvalContext::new().with_cancel(token);
        let result = evaluator.evaluate(&graph, &[add], &ctx);

        assert!(matches!(result, Err(GraphError::Cancelled)));
    }

    #[cfg(feature = "named-ports")]
    #[test]
    fn test_connect_named_success() {
        let mut graph = Graph::new();

        let const_a = graph.add_node(ConstNode(2.0));
        let const_b = graph.add_node(ConstNode(3.0));
        let add = graph.add_node(AddNode);

        graph.connect_named(const_a, "value", add, "a").unwrap();
        graph.connect_named(const_b, "value", add, "b").unwrap();

        let outputs = graph.execute(add).unwrap();
        assert_eq!(outputs[0].as_f32().unwrap(), 5.0);
    }

    #[cfg(feature = "named-ports")]
    #[test]
    fn test_connect_named_unknown_port() {
        let mut graph = Graph::new();

        let const_a = graph.add_node(ConstNode(1.0));
        let add = graph.add_node(AddNode);

        let result = graph.connect_named(const_a, "nonexistent", add, "a");
        assert!(matches!(result, Err(GraphError::PortNameNotFound { .. })));

        let result2 = graph.connect_named(const_a, "value", add, "nonexistent");
        assert!(matches!(result2, Err(GraphError::PortNameNotFound { .. })));
    }

    #[cfg(feature = "named-ports")]
    #[test]
    fn test_disconnect_named_success() {
        let mut graph = Graph::new();

        let const_a = graph.add_node(ConstNode(2.0));
        let add = graph.add_node(AddNode);

        graph.connect_named(const_a, "value", add, "a").unwrap();
        assert_eq!(graph.wire_count(), 1);

        graph.disconnect_named(const_a, "value", add, "a").unwrap();
        assert_eq!(graph.wire_count(), 0);
    }

    #[test]
    fn test_input_nodes_and_output_nodes() {
        use crate::nodes::{GraphInput, GraphOutput};

        let mut graph = Graph::new();
        let in_id = graph.add_node(GraphInput::new("x", ValueType::F32));
        let out_id = graph.add_node(GraphOutput::new("result", ValueType::F32));

        let inputs = graph.input_nodes();
        assert_eq!(inputs.len(), 1);
        assert_eq!(inputs[0].node_id, in_id);
        assert_eq!(inputs[0].name, "x");
        assert_eq!(inputs[0].value_type, ValueType::F32);

        let outputs = graph.output_nodes();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].node_id, out_id);
        assert_eq!(outputs[0].name, "result");
        assert_eq!(outputs[0].value_type, ValueType::F32);
    }

    #[test]
    fn test_execute_named_outputs_two_outputs() {
        use crate::nodes::{GraphInput, GraphOutput};

        // Graph:
        //   GraphInput("a", F32)  --> GraphOutput("out_a", F32)
        //   GraphInput("b", F32)  --> GraphOutput("out_b", F32)
        let mut graph = Graph::new();

        let in_a = graph.add_node(GraphInput::new("a", ValueType::F32));
        let in_b = graph.add_node(GraphInput::new("b", ValueType::F32));
        let out_a = graph.add_node(GraphOutput::new("out_a", ValueType::F32));
        let out_b = graph.add_node(GraphOutput::new("out_b", ValueType::F32));

        graph.connect(in_a, 0, out_a, 0).unwrap();
        graph.connect(in_b, 0, out_b, 0).unwrap();

        let ctx = EvalContext::new()
            .with_input("a", 1.0f32)
            .with_input("b", 2.0f32);

        let results = graph.execute_named_outputs(&ctx).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results["out_a"].as_f32().unwrap(), 1.0);
        assert_eq!(results["out_b"].as_f32().unwrap(), 2.0);
    }

    #[test]
    fn test_execute_named_outputs_empty() {
        let mut graph = Graph::new();
        let ctx = EvalContext::new();
        let results = graph.execute_named_outputs(&ctx).unwrap();
        assert!(results.is_empty());
    }

    // ========================================================================
    // Recurrent / feedback-edge tests
    // ========================================================================

    /// Stateless accumulator: `sum = x + prev`.
    ///
    /// Holds NO state — `prev` arrives on a feedback edge, so the running total
    /// lives entirely on the wire / in `FeedbackState`. Pure `&self`.
    struct AccumNode;

    impl DynNode for AccumNode {
        fn type_name(&self) -> &'static str {
            "Accum"
        }
        fn inputs(&self) -> Vec<PortDescriptor> {
            vec![
                PortDescriptor::new("x", ValueType::F32),
                PortDescriptor::new("prev", ValueType::F32),
            ]
        }
        fn outputs(&self) -> Vec<PortDescriptor> {
            vec![PortDescriptor::new("sum", ValueType::F32)]
        }
        fn execute(&self, inputs: &[Value], _ctx: &EvalContext) -> Result<Vec<Value>, GraphError> {
            let x = inputs[0]
                .as_f32()
                .map_err(|e| GraphError::ExecutionError(e.to_string()))?;
            let prev = inputs[1]
                .as_f32()
                .map_err(|e| GraphError::ExecutionError(e.to_string()))?;
            Ok(vec![Value::F32(x + prev)])
        }
        fn as_any(&self) -> &dyn Any {
            self
        }
    }

    /// Stateless one-pole IIR: `y = a*x + (1-a)*y_prev`.
    struct OnePoleNode {
        a: f32,
    }

    impl DynNode for OnePoleNode {
        fn type_name(&self) -> &'static str {
            "OnePole"
        }
        fn inputs(&self) -> Vec<PortDescriptor> {
            vec![
                PortDescriptor::new("x", ValueType::F32),
                PortDescriptor::new("y_prev", ValueType::F32),
            ]
        }
        fn outputs(&self) -> Vec<PortDescriptor> {
            vec![PortDescriptor::new("y", ValueType::F32)]
        }
        fn execute(&self, inputs: &[Value], _ctx: &EvalContext) -> Result<Vec<Value>, GraphError> {
            let x = inputs[0]
                .as_f32()
                .map_err(|e| GraphError::ExecutionError(e.to_string()))?;
            let y_prev = inputs[1]
                .as_f32()
                .map_err(|e| GraphError::ExecutionError(e.to_string()))?;
            Ok(vec![Value::F32(self.a * x + (1.0 - self.a) * y_prev)])
        }
        fn as_any(&self) -> &dyn Any {
            self
        }
    }

    /// A built accumulator graph and the id of its accumulator node.
    struct AccumGraph {
        graph: Graph,
        acc: NodeId,
    }

    /// Build `Const(step) -> Accum`, with `Accum.sum -> Accum.prev` as feedback.
    fn build_accumulator(step: f32) -> AccumGraph {
        let mut graph = Graph::new();
        let c = graph.add_node(ConstNode(step));
        let acc = graph.add_node(AccumNode);
        graph.connect(c, 0, acc, 0).unwrap(); // x = step
        graph.connect_feedback(acc, 0, acc, 1).unwrap(); // sum -> prev (back-edge)
        AccumGraph { graph, acc }
    }

    #[test]
    fn test_feedback_back_edge_is_acyclic_per_tick() {
        // A feedback self-loop must NOT be reported as a cycle: the recurrent
        // driver topo-sorts the graph (excluding back-edges) and evaluates it.
        let AccumGraph { mut graph, acc } = build_accumulator(1.0);
        assert!(graph.has_feedback());
        let mut state = crate::FeedbackState::new();
        let r = graph.tick(0, &mut state, &EvalContext::new());
        assert!(r.is_ok(), "feedback back-edge wrongly treated as a cycle");
        assert_eq!(r.unwrap().get(acc, 0).unwrap().as_f32().unwrap(), 1.0);

        // The plain DAG path (`execute`) does NOT handle feedback: the
        // feedback-fed input is reported as unconnected, by design.
        assert!(matches!(
            graph.execute(acc),
            Err(GraphError::UnconnectedInput { .. })
        ));
    }

    #[test]
    fn test_feedback_state_carries_across_ticks() {
        // (a) state carries across ticks via the feedback edge.
        let AccumGraph { mut graph, acc } = build_accumulator(1.0);
        let mut state = crate::FeedbackState::new();

        // tick 0: prev seeded to 0 -> sum = 1
        let r0 = graph.tick(0, &mut state, &EvalContext::new()).unwrap();
        assert_eq!(r0.get(acc, 0).unwrap().as_f32().unwrap(), 1.0);
        // tick 1: prev = 1 -> sum = 2
        let r1 = graph.tick(1, &mut state, &EvalContext::new()).unwrap();
        assert_eq!(r1.get(acc, 0).unwrap().as_f32().unwrap(), 2.0);
        // tick 2: prev = 2 -> sum = 3
        let r2 = graph.tick(2, &mut state, &EvalContext::new()).unwrap();
        assert_eq!(r2.get(acc, 0).unwrap().as_f32().unwrap(), 3.0);

        // (b) the node holds no state: a fresh FeedbackState restarts from 0.
        let mut fresh = crate::FeedbackState::new();
        let again = graph.tick(0, &mut fresh, &EvalContext::new()).unwrap();
        assert_eq!(again.get(acc, 0).unwrap().as_f32().unwrap(), 1.0);
    }

    #[test]
    fn test_feedback_determinism() {
        // (c) same graph + inputs + tick count => identical output.
        let run = || {
            let AccumGraph { mut graph, acc } = build_accumulator(2.0);
            let mut state = crate::FeedbackState::new();
            let mut last = 0.0;
            for t in 0..10 {
                let r = graph.tick(t, &mut state, &EvalContext::new()).unwrap();
                last = r.get(acc, 0).unwrap().as_f32().unwrap();
            }
            last
        };
        let a = run();
        let b = run();
        assert_eq!(a, b);
        assert_eq!(a, 20.0); // 10 ticks * step 2.0
    }

    #[test]
    fn test_seek_resimulate_matches_stepping() {
        // (d) seeking to tick N via Resimulate == stepping 0..=N.
        let AccumGraph { mut graph, acc } = build_accumulator(1.0);

        // Step manually to tick 5.
        let mut stepped_state = crate::FeedbackState::new();
        let mut stepped = 0.0;
        for t in 0..=5 {
            let r = graph
                .tick(t, &mut stepped_state, &EvalContext::new())
                .unwrap();
            stepped = r.get(acc, 0).unwrap().as_f32().unwrap();
        }

        // Resimulate to tick 5 from scratch.
        let mut seek_state = crate::FeedbackState::new();
        let seeked = graph
            .run_to_tick(5, &mut seek_state, |_t| EvalContext::new())
            .unwrap();
        assert_eq!(seeked.get(acc, 0).unwrap().as_f32().unwrap(), stepped);
        assert_eq!(stepped, 6.0); // ticks 0..=5 inclusive, step 1.0
    }

    #[test]
    fn test_seek_behaviors() {
        let AccumGraph { mut graph, acc } = build_accumulator(1.0);

        // Resimulate: correct replay.
        let mut s = crate::FeedbackState::new();
        let r = graph
            .seek(3, 0, crate::SeekBehavior::Resimulate, &mut s, |_t| {
                EvalContext::new()
            })
            .unwrap();
        assert_eq!(r.get(acc, 0).unwrap().as_f32().unwrap(), 4.0); // 0..=3

        // Discontinuity: single tick using empty state -> behaves like tick 0.
        let mut s2 = crate::FeedbackState::new();
        let rd = graph
            .seek(3, 0, crate::SeekBehavior::Discontinuity, &mut s2, |_t| {
                EvalContext::new()
            })
            .unwrap();
        assert_eq!(rd.get(acc, 0).unwrap().as_f32().unwrap(), 1.0);

        // Error: refuse to seek to a different tick.
        let mut s3 = crate::FeedbackState::new();
        let re = graph.seek(3, 0, crate::SeekBehavior::Error, &mut s3, |_t| {
            EvalContext::new()
        });
        assert!(matches!(
            re,
            Err(GraphError::SeekUnsupported {
                target: 3,
                current: 0
            })
        ));
    }

    #[test]
    fn test_one_pole_iir_converges() {
        // y = a*x + (1-a)*y_prev with constant x converges toward x.
        let mut graph = Graph::new();
        let c = graph.add_node(ConstNode(1.0));
        let filt = graph.add_node(OnePoleNode { a: 0.5 });
        graph.connect(c, 0, filt, 0).unwrap();
        graph.connect_feedback(filt, 0, filt, 1).unwrap();

        let mut state = crate::FeedbackState::new();
        let mut y = 0.0;
        for t in 0..20 {
            let r = graph.tick(t, &mut state, &EvalContext::new()).unwrap();
            y = r.get(filt, 0).unwrap().as_f32().unwrap();
        }
        // After 20 ticks, y is very close to the input 1.0.
        assert!((y - 1.0).abs() < 1e-3, "y = {y}");

        // First tick: y = 0.5*1 + 0.5*0 = 0.5
        let mut s0 = crate::FeedbackState::new();
        let r0 = graph.tick(0, &mut s0, &EvalContext::new()).unwrap();
        assert_eq!(r0.get(filt, 0).unwrap().as_f32().unwrap(), 0.5);
    }

    #[test]
    fn test_non_recurrent_graph_unaffected() {
        // The standard DAG path still works (no regression).
        let mut graph = Graph::new();
        let a = graph.add_node(ConstNode(2.0));
        let b = graph.add_node(ConstNode(3.0));
        let add = graph.add_node(AddNode);
        graph.connect(a, 0, add, 0).unwrap();
        graph.connect(b, 0, add, 1).unwrap();
        assert!(!graph.has_feedback());
        assert_eq!(graph.execute(add).unwrap()[0].as_f32().unwrap(), 5.0);
    }
}
