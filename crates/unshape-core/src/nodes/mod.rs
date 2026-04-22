//! Built-in source and sink nodes for the unshape graph system.
//!
//! These nodes provide values into a graph or capture values out of it without
//! requiring external wires:
//!
//! - [`ConstantNode`] — an authored value embedded directly in the graph.
//! - [`GraphInput`] — reads a named value from [`EvalContext`](crate::EvalContext) at execution time.
//! - [`GraphOutput`] — a named terminal sink that captures a value from the graph.

pub mod constant;
pub mod graph_input;
pub mod graph_output;

pub use constant::ConstantNode;
pub use graph_input::GraphInput;
pub use graph_output::GraphOutput;
