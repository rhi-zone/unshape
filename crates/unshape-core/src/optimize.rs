//! Generic graph optimization infrastructure.
//!
//! This module provides traits and utilities for building optimization passes
//! that work with any graph type.
//!
//! # Overview
//!
//! Graph optimization typically involves pattern recognition and transformation:
//! 1. Find subgraphs that match certain patterns
//! 2. Replace them with equivalent but more efficient structures
//!
//! This module provides the infrastructure; domain-specific passes are implemented
//! in their respective crates (e.g., `unshape-audio::optimize`).
//!
//! # Example
//!
//! ```ignore
//! use unshape_core::optimize::{Optimizer, OptimizerPipeline};
//!
//! // Define a custom optimizer for your graph type
//! struct MyGraphOptimizer;
//!
//! impl Optimizer<MyGraph> for MyGraphOptimizer {
//!     fn apply(&self, graph: &mut MyGraph) -> usize {
//!         // ... optimization logic ...
//!         0
//!     }
//!     fn name(&self) -> &'static str { "MyGraphOptimizer" }
//! }
//!
//! // Compose into a pipeline
//! let mut pipeline = OptimizerPipeline::new();
//! pipeline.add(MyGraphOptimizer);
//! pipeline.add(AnotherOptimizer);
//!
//! // Run until fixpoint
//! let total_changes = pipeline.run(&mut graph);
//! ```

/// Trait for graph optimization passes.
///
/// Implement this trait to create custom optimization passes for any graph type.
/// Passes can be composed into pipelines that run until no more changes occur.
///
/// # Type Parameters
///
/// * `G` - The graph type this optimizer works with.
///
/// # Example
///
/// ```ignore
/// use unshape_core::optimize::Optimizer;
///
/// struct IdentityEliminator;
///
/// impl Optimizer<MyGraph> for IdentityEliminator {
///     fn apply(&self, graph: &mut MyGraph) -> usize {
///         let mut removed = 0;
///         // ... find and remove identity nodes ...
///         removed
///     }
///
///     fn name(&self) -> &'static str {
///         "IdentityEliminator"
///     }
/// }
/// ```
pub trait Optimizer<G> {
    /// Applies this optimization pass to the graph.
    ///
    /// Returns the number of nodes affected (removed, merged, or transformed).
    /// Returning 0 indicates the pass made no changes.
    fn apply(&self, graph: &mut G) -> usize;

    /// Returns the name of this optimization pass for debugging/logging.
    fn name(&self) -> &'static str;
}

/// A boxed optimizer for dynamic dispatch.
pub type BoxedOptimizer<G> = Box<dyn Optimizer<G>>;

/// Pipeline of optimizers that runs passes until fixpoint.
///
/// An optimizer pipeline composes multiple optimization passes and runs them
/// in sequence until no pass makes any changes (fixpoint reached).
///
/// # Example
///
/// ```ignore
/// use unshape_core::optimize::OptimizerPipeline;
///
/// let mut pipeline = OptimizerPipeline::new()
///     .add(ConstantFolder)
///     .add(IdentityEliminator)
///     .add(DeadCodeEliminator);
///
/// // Run all passes until no changes
/// let total = pipeline.run(&mut graph);
/// println!("Optimized {} nodes", total);
/// ```
pub struct OptimizerPipeline<G> {
    passes: Vec<BoxedOptimizer<G>>,
}

impl<G> Default for OptimizerPipeline<G> {
    fn default() -> Self {
        Self::new()
    }
}

impl<G> OptimizerPipeline<G> {
    /// Creates an empty pipeline.
    pub fn new() -> Self {
        Self { passes: Vec::new() }
    }

    /// Adds an optimizer to the pipeline.
    pub fn add<O: Optimizer<G> + 'static>(mut self, optimizer: O) -> Self {
        self.passes.push(Box::new(optimizer));
        self
    }

    /// Adds a boxed optimizer to the pipeline.
    pub fn add_boxed(mut self, optimizer: BoxedOptimizer<G>) -> Self {
        self.passes.push(optimizer);
        self
    }

    /// Returns the number of passes in this pipeline.
    pub fn len(&self) -> usize {
        self.passes.len()
    }

    /// Returns true if the pipeline has no passes.
    pub fn is_empty(&self) -> bool {
        self.passes.is_empty()
    }

    /// Runs all passes once and returns total changes.
    ///
    /// Unlike `run()`, this doesn't iterate to fixpoint.
    pub fn run_once(&self, graph: &mut G) -> usize {
        let mut total = 0;
        for pass in &self.passes {
            total += pass.apply(graph);
        }
        total
    }

    /// Runs all passes until no more changes occur (fixpoint).
    ///
    /// Returns the total number of nodes affected across all iterations.
    pub fn run(&self, graph: &mut G) -> usize {
        let mut total = 0;

        loop {
            let changed = self.run_once(graph);
            if changed == 0 {
                break;
            }
            total += changed;
        }

        total
    }

    /// Runs all passes with a maximum number of iterations.
    ///
    /// Returns the total number of nodes affected and whether fixpoint was reached.
    pub fn run_bounded(&self, graph: &mut G, max_iterations: usize) -> (usize, bool) {
        let mut total = 0;

        for _ in 0..max_iterations {
            let changed = self.run_once(graph);
            if changed == 0 {
                return (total, true);
            }
            total += changed;
        }

        (total, false)
    }

    /// Iterates over the optimizer names in the pipeline.
    pub fn pass_names(&self) -> impl Iterator<Item = &'static str> + '_ {
        self.passes.iter().map(|p| p.name())
    }
}

/// Statistics about optimization results.
#[derive(Debug, Clone, Default)]
pub struct OptimizeStats {
    /// Total nodes removed.
    pub nodes_removed: usize,
    /// Total nodes merged.
    pub nodes_merged: usize,
    /// Total nodes transformed.
    pub nodes_transformed: usize,
    /// Number of iterations until fixpoint.
    pub iterations: usize,
    /// Per-pass statistics.
    pub pass_stats: Vec<PassStats>,
}

/// Statistics for a single optimization pass.
#[derive(Debug, Clone)]
pub struct PassStats {
    /// Name of the pass.
    pub name: &'static str,
    /// Number of nodes affected by this pass.
    pub affected: usize,
}

/// Optimizer pipeline with statistics tracking.
///
/// Like `OptimizerPipeline` but collects detailed statistics about each pass.
pub struct TrackedPipeline<G> {
    passes: Vec<BoxedOptimizer<G>>,
}

impl<G> Default for TrackedPipeline<G> {
    fn default() -> Self {
        Self::new()
    }
}

impl<G> TrackedPipeline<G> {
    /// Creates an empty tracked pipeline.
    pub fn new() -> Self {
        Self { passes: Vec::new() }
    }

    /// Adds an optimizer to the pipeline.
    pub fn add<O: Optimizer<G> + 'static>(mut self, optimizer: O) -> Self {
        self.passes.push(Box::new(optimizer));
        self
    }

    /// Runs all passes until fixpoint, collecting statistics.
    pub fn run(&self, graph: &mut G) -> OptimizeStats {
        let mut stats = OptimizeStats::default();

        loop {
            let mut changed = 0;

            for pass in &self.passes {
                let affected = pass.apply(graph);
                if affected > 0 {
                    stats.pass_stats.push(PassStats {
                        name: pass.name(),
                        affected,
                    });
                    changed += affected;
                }
            }

            if changed == 0 {
                break;
            }

            stats.iterations += 1;
            stats.nodes_transformed += changed;
        }

        stats
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Simple test graph
    struct TestGraph {
        nodes: Vec<i32>,
    }

    impl TestGraph {
        fn new() -> Self {
            Self { nodes: Vec::new() }
        }

        fn add(&mut self, val: i32) {
            self.nodes.push(val);
        }
    }

    // Test optimizer that removes zeros
    struct RemoveZeros;

    impl Optimizer<TestGraph> for RemoveZeros {
        fn apply(&self, graph: &mut TestGraph) -> usize {
            let before = graph.nodes.len();
            graph.nodes.retain(|&x| x != 0);
            before - graph.nodes.len()
        }

        fn name(&self) -> &'static str {
            "RemoveZeros"
        }
    }

    // Test optimizer that doubles negatives
    struct DoubleNegatives;

    impl Optimizer<TestGraph> for DoubleNegatives {
        fn apply(&self, graph: &mut TestGraph) -> usize {
            let mut changed = 0;
            for val in &mut graph.nodes {
                if *val < 0 {
                    *val *= 2;
                    changed += 1;
                }
            }
            changed
        }

        fn name(&self) -> &'static str {
            "DoubleNegatives"
        }
    }

    #[test]
    fn test_single_optimizer() {
        let mut graph = TestGraph::new();
        graph.add(1);
        graph.add(0);
        graph.add(2);
        graph.add(0);
        graph.add(3);

        let opt = RemoveZeros;
        let removed = opt.apply(&mut graph);

        assert_eq!(removed, 2);
        assert_eq!(graph.nodes, vec![1, 2, 3]);
    }

    #[test]
    fn test_pipeline_run_once() {
        let mut graph = TestGraph::new();
        graph.add(1);
        graph.add(0);
        graph.add(-2);

        let pipeline = OptimizerPipeline::new()
            .add(RemoveZeros)
            .add(DoubleNegatives);

        let changed = pipeline.run_once(&mut graph);

        assert_eq!(changed, 2); // 1 zero removed + 1 negative doubled
        assert_eq!(graph.nodes, vec![1, -4]); // -2 doubled to -4
    }

    // Test optimizer that removes negative numbers (converges)
    struct RemoveNegatives;

    impl Optimizer<TestGraph> for RemoveNegatives {
        fn apply(&self, graph: &mut TestGraph) -> usize {
            let before = graph.nodes.len();
            graph.nodes.retain(|&x| x >= 0);
            before - graph.nodes.len()
        }

        fn name(&self) -> &'static str {
            "RemoveNegatives"
        }
    }

    #[test]
    fn test_pipeline_run_to_fixpoint() {
        let mut graph = TestGraph::new();
        graph.add(1);
        graph.add(0);
        graph.add(-2);
        graph.add(0);
        graph.add(-5);

        let pipeline = OptimizerPipeline::new()
            .add(RemoveZeros)
            .add(RemoveNegatives);

        let total = pipeline.run(&mut graph);

        // First iteration: remove 2 zeros and 2 negatives
        // Second iteration: no changes (converged)
        assert_eq!(total, 4);
        assert_eq!(graph.nodes, vec![1]);
    }

    #[test]
    fn test_pipeline_bounded() {
        let mut graph = TestGraph::new();
        graph.add(-1);

        let pipeline = OptimizerPipeline::new().add(DoubleNegatives);

        let (total, converged) = pipeline.run_bounded(&mut graph, 3);

        // Should run exactly 3 times without converging
        assert_eq!(total, 3);
        assert!(!converged);
        assert_eq!(graph.nodes, vec![-8]); // -1 -> -2 -> -4 -> -8
    }

    #[test]
    fn test_empty_pipeline() {
        let mut graph = TestGraph::new();
        graph.add(1);

        let pipeline: OptimizerPipeline<TestGraph> = OptimizerPipeline::new();

        let total = pipeline.run(&mut graph);
        assert_eq!(total, 0);
        assert!(pipeline.is_empty());
    }

    #[test]
    fn test_pipeline_pass_names() {
        let pipeline = OptimizerPipeline::new()
            .add(RemoveZeros)
            .add(DoubleNegatives);

        let names: Vec<_> = pipeline.pass_names().collect();
        assert_eq!(names, vec!["RemoveZeros", "DoubleNegatives"]);
    }

    #[test]
    fn test_tracked_pipeline() {
        let mut graph = TestGraph::new();
        graph.add(0);
        graph.add(0);
        graph.add(1);

        let pipeline = TrackedPipeline::new().add(RemoveZeros);

        let stats = pipeline.run(&mut graph);

        assert_eq!(stats.pass_stats.len(), 1);
        assert_eq!(stats.pass_stats[0].name, "RemoveZeros");
        assert_eq!(stats.pass_stats[0].affected, 2);
        assert_eq!(graph.nodes, vec![1]);
    }
}
