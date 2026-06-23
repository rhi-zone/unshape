//! Built-in source and sink nodes for the unshape graph system.
//!
//! These nodes provide values into a graph or capture values out of it without
//! requiring external wires:
//!
//! - [`ConstantNode`] — an authored value embedded directly in the graph.
//! - [`GraphInput`] — reads a named value from [`EvalContext`](crate::EvalContext) at execution time.
//! - [`GraphOutput`] — a named terminal sink that captures a value from the graph.
//! - [`Latch`] — a seeded unit-delay (1-tick memory); the recurrence primitive.

pub mod constant;
pub mod graph_input;
pub mod graph_output;
pub mod latch;

pub use constant::ConstantNode;
pub use graph_input::GraphInput;
pub use graph_output::GraphOutput;
pub use latch::{LATCH_INIT_PORT, LATCH_OUT_PORT, LATCH_SIGNAL_PORT, Latch, Rate};
