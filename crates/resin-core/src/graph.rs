//! Graph container and execution.
//!
//! This module provides a data flow graph where nodes process values
//! and wires connect output ports to input ports.
//!
//! # Example
//!
//! ```ignore
//! use rhizome_resin_core::{Graph, EvalContext};
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
use crate::eval::EvalContext;
use crate::node::{BoxedNode, DynNode};
use crate::value::Value;

/// Unique identifier for a node in a graph.
pub type NodeId = u32;

/// A wire connecting an output port to an input port.
///
/// Wires carry data from one node's output to another node's input.
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

        self.wires.push(Wire {
            from_node,
            from_port,
            to_node,
            to_port,
        });
        self.topo_order = None; // Invalidate cache

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

        // Build adjacency list and count in-degrees
        for edge in &self.wires {
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
                // Find wire that feeds this input
                let wire = self
                    .wires
                    .iter()
                    .find(|w| w.to_node == node_id && w.to_port == port);

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
        use crate::eval::{EvalResult, Evaluator, LazyEvaluator};

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
}
