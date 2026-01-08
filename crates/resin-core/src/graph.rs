//! Graph container and execution.

use std::collections::HashMap;

use crate::context::EvalContext;
use crate::error::GraphError;
use crate::node::{BoxedNode, DynNode};
use crate::value::Value;

/// Unique identifier for a node in a graph.
pub type NodeId = u32;

/// An edge connecting two nodes.
#[derive(Debug, Clone, Copy)]
pub struct Edge {
    /// Source node.
    pub from_node: NodeId,
    /// Output port index on source node.
    pub from_port: usize,
    /// Destination node.
    pub to_node: NodeId,
    /// Input port index on destination node.
    pub to_port: usize,
}

/// A graph of nodes connected by edges.
#[derive(Default)]
pub struct Graph {
    nodes: HashMap<NodeId, BoxedNode>,
    edges: Vec<Edge>,
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

        self.edges.push(Edge {
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
        for edge in &self.edges {
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
    /// # Arguments
    /// * `output_node` - The node whose outputs to return.
    /// * `ctx` - Evaluation context.
    pub fn execute(&mut self, output_node: NodeId, ctx: &EvalContext) -> Result<Vec<Value>, GraphError> {
        let order = self.topological_order()?.to_vec();

        // Storage for computed values: (node_id, port_index) -> Value
        let mut values: HashMap<(NodeId, usize), Value> = HashMap::new();

        for node_id in order {
            let node = self.nodes.get(&node_id).unwrap();
            let inputs_desc = node.inputs();
            let num_inputs = inputs_desc.len();

            // Gather inputs for this node
            let mut inputs = Vec::with_capacity(num_inputs);
            for port in 0..num_inputs {
                // Find edge that feeds this input
                let edge = self
                    .edges
                    .iter()
                    .find(|e| e.to_node == node_id && e.to_port == port);

                match edge {
                    Some(e) => {
                        let value = values
                            .get(&(e.from_node, e.from_port))
                            .cloned()
                            .ok_or_else(|| GraphError::ExecutionError(
                                format!("missing value for node {} port {}", e.from_node, e.from_port)
                            ))?;
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
            let value = values
                .get(&(output_node, port))
                .cloned()
                .ok_or_else(|| GraphError::ExecutionError(
                    format!("missing output for node {} port {}", output_node, port)
                ))?;
            result.push(value);
        }

        Ok(result)
    }

    /// Returns the number of nodes in the graph.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Returns the number of edges in the graph.
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::node::PortDescriptor;
    use crate::value::ValueType;

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
            let a = inputs[0].as_f32().map_err(|e| GraphError::ExecutionError(e.to_string()))?;
            let b = inputs[1].as_f32().map_err(|e| GraphError::ExecutionError(e.to_string()))?;
            Ok(vec![Value::F32(a + b)])
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
    }

    #[test]
    fn test_simple_graph() {
        let mut graph = Graph::new();

        let const_a = graph.add_node(ConstNode(2.0));
        let const_b = graph.add_node(ConstNode(3.0));
        let add = graph.add_node(AddNode);

        graph.connect(const_a, 0, add, 0).unwrap();
        graph.connect(const_b, 0, add, 1).unwrap();

        let ctx = EvalContext::new();
        let outputs = graph.execute(add, &ctx).unwrap();

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

            fn execute(&self, _inputs: &[Value], _ctx: &EvalContext) -> Result<Vec<Value>, GraphError> {
                Ok(vec![Value::Bool(true)])
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

            fn execute(&self, inputs: &[Value], _ctx: &EvalContext) -> Result<Vec<Value>, GraphError> {
                Ok(vec![inputs[0].clone()])
            }
        }

        let mut graph = Graph::new();
        let a = graph.add_node(PassthroughNode);
        let b = graph.add_node(PassthroughNode);

        graph.connect(a, 0, b, 0).unwrap();
        graph.connect(b, 0, a, 0).unwrap(); // Creates cycle

        let ctx = EvalContext::new();
        let result = graph.execute(a, &ctx);
        assert!(matches!(result, Err(GraphError::CycleDetected)));
    }
}
