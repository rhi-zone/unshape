//! Built-in source nodes for the unshape graph system.
//!
//! These nodes provide values into a graph without requiring upstream wires:
//!
//! - [`ConstantNode`] — an authored value embedded directly in the graph.
//! - [`GraphInput`] — reads a named value from [`EvalContext`](crate::EvalContext) at execution time.

pub mod constant;
pub mod graph_input;

pub use constant::ConstantNode;
pub use graph_input::GraphInput;
