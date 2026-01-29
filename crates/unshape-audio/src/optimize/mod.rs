//! Graph pattern matching and optimization.
//!
//! Recognizes known subgraph patterns and replaces them with optimized implementations.
//!
//! # Overview
//!
//! Audio graphs built from primitives can be automatically optimized by recognizing
//! common patterns (tremolo, chorus, flanger, etc.) and replacing them with
//! monomorphized implementations that eliminate dynamic dispatch overhead.
//!
//! # Algorithm
//!
//! 1. **Fingerprint**: Count node types in graph (O(N) linear scan)
//! 2. **Filter**: Only check patterns whose required nodes are present
//! 3. **Match**: Structural matching on candidate patterns
//! 4. **Replace**: Substitute matched subgraph with optimized node
//!
//! # Example
//!
//! ```ignore
//! use unshape_audio::optimize::{optimize_graph, default_patterns};
//! use unshape_audio::graph::AudioGraph;
//!
//! let mut graph = AudioGraph::new();
//! // ... build graph with LFO modulating gain (tremolo pattern) ...
//!
//! // Optimize: replaces LFO+Gain subgraph with TremoloOptimized
//! optimize_graph(&mut graph, &default_patterns());
//! ```

mod effects;
mod engine;
mod passes;

pub use effects::*;
pub use engine::*;
pub use passes::*;

use crate::graph::AudioGraph;
use unshape_core::optimize::Optimizer;

/// Re-export the generic OptimizerPipeline with default audio passes.
///
/// For custom pipelines, use `unshape_core::optimize::OptimizerPipeline<AudioGraph>`.
pub use unshape_core::optimize::OptimizerPipeline as GenericPipeline;

/// Type alias for backwards compatibility.
///
/// The generic [`Optimizer`] trait from unshape-core provides the same interface.
/// Use `Optimizer<AudioGraph>` directly for new code.
pub trait GraphOptimizer: Optimizer<AudioGraph> {}

/// Blanket implementation: any Optimizer<AudioGraph> is a GraphOptimizer.
impl<T: Optimizer<AudioGraph>> GraphOptimizer for T {}

/// Fuses chains of affine (multiply-add) operations into single nodes.
///
/// Example: `Gain(0.5) → Offset(1.0) → Gain(2.0)` becomes `Affine(gain=1.0, offset=2.0)`
#[derive(Debug, Clone, Copy, Default)]
pub struct AffineChainFuser;

impl Optimizer<AudioGraph> for AffineChainFuser {
    fn apply(&self, graph: &mut AudioGraph) -> usize {
        fuse_affine_chains(graph)
    }

    fn name(&self) -> &'static str {
        "AffineChainFuser"
    }
}

/// Removes identity operations that have no effect.
///
/// Removes: `Gain(1.0)`, `Offset(0.0)`, `PassThrough`
#[derive(Debug, Clone, Copy, Default)]
pub struct IdentityEliminator;

impl Optimizer<AudioGraph> for IdentityEliminator {
    fn apply(&self, graph: &mut AudioGraph) -> usize {
        eliminate_identities(graph)
    }

    fn name(&self) -> &'static str {
        "IdentityEliminator"
    }
}

/// Removes nodes not connected to the output.
#[derive(Debug, Clone, Copy, Default)]
pub struct DeadNodeEliminator;

impl Optimizer<AudioGraph> for DeadNodeEliminator {
    fn apply(&self, graph: &mut AudioGraph) -> usize {
        eliminate_dead_nodes(graph)
    }

    fn name(&self) -> &'static str {
        "DeadNodeEliminator"
    }
}

/// Folds constant values through affine operations.
///
/// Example: `Constant(2.0) → Gain(3.0)` becomes `Constant(6.0)`
#[derive(Debug, Clone, Copy, Default)]
pub struct ConstantFolder;

impl Optimizer<AudioGraph> for ConstantFolder {
    fn apply(&self, graph: &mut AudioGraph) -> usize {
        fold_constants(graph)
    }

    fn name(&self) -> &'static str {
        "ConstantFolder"
    }
}

/// Merges consecutive delay nodes.
///
/// Example: `Delay(100) → Delay(50)` becomes `Delay(150)`
#[derive(Debug, Clone, Copy, Default)]
pub struct DelayMerger;

impl Optimizer<AudioGraph> for DelayMerger {
    fn apply(&self, graph: &mut AudioGraph) -> usize {
        merge_delays(graph)
    }

    fn name(&self) -> &'static str {
        "DelayMerger"
    }
}

/// Aggressively propagates constants through the graph.
///
/// Combines constant folding with affine fusion in a loop.
#[derive(Debug, Clone, Copy, Default)]
pub struct ConstantPropagator;

impl Optimizer<AudioGraph> for ConstantPropagator {
    fn apply(&self, graph: &mut AudioGraph) -> usize {
        propagate_constants(graph)
    }

    fn name(&self) -> &'static str {
        "ConstantPropagator"
    }
}

/// Audio-specific optimizer pipeline with default passes.
///
/// # Example
///
/// ```ignore
/// use unshape_audio::optimize::*;
///
/// let pipeline = OptimizerPipeline::default(); // Standard passes
/// pipeline.run(&mut graph);
/// ```
pub struct OptimizerPipeline {
    inner: GenericPipeline<AudioGraph>,
}

impl Default for OptimizerPipeline {
    fn default() -> Self {
        Self {
            inner: GenericPipeline::new()
                .add(AffineChainFuser)
                .add(ConstantFolder)
                .add(DelayMerger)
                .add(IdentityEliminator)
                .add(DeadNodeEliminator),
        }
    }
}

impl OptimizerPipeline {
    /// Creates an empty pipeline.
    pub fn new() -> Self {
        Self {
            inner: GenericPipeline::new(),
        }
    }

    /// Adds an optimizer to the pipeline.
    pub fn add<O: Optimizer<AudioGraph> + 'static>(mut self, optimizer: O) -> Self {
        self.inner = self.inner.add(optimizer);
        self
    }

    /// Runs all passes until no more changes occur.
    ///
    /// Returns the total number of nodes affected.
    pub fn run(&self, graph: &mut AudioGraph) -> usize {
        self.inner.run(graph)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::AudioContext;

    #[test]
    fn test_fingerprint_contains() {
        let mut graph_fp = GraphFingerprint::new();
        graph_fp.add(NodeType::Lfo);
        graph_fp.add(NodeType::Lfo);
        graph_fp.add(NodeType::Affine);
        graph_fp.add(NodeType::Delay);

        // Pattern needs 1 LFO and 1 Affine - should match
        let pattern_fp = fingerprint!(Lfo: 1, Affine: 1);
        assert!(graph_fp.contains(&pattern_fp));

        // Pattern needs 2 LFOs and 1 Affine - should match
        let pattern_fp2 = fingerprint!(Lfo: 2, Affine: 1);
        assert!(graph_fp.contains(&pattern_fp2));

        // Pattern needs 3 LFOs - should not match
        let pattern_fp3 = fingerprint!(Lfo: 3);
        assert!(!graph_fp.contains(&pattern_fp3));

        // Pattern needs an Allpass - should not match
        let pattern_fp4 = fingerprint!(Allpass: 1);
        assert!(!graph_fp.contains(&pattern_fp4));
    }

    #[test]
    fn test_fingerprint_macro() {
        let fp = fingerprint!(Lfo: 2, Affine: 1, Delay: 3);

        assert_eq!(fp.count(NodeType::Lfo), 2);
        assert_eq!(fp.count(NodeType::Affine), 1);
        assert_eq!(fp.count(NodeType::Delay), 3);
        assert_eq!(fp.count(NodeType::Allpass), 0);
    }

    #[test]
    fn test_tremolo_pattern_match() {
        use crate::graph::{AffineNode, AudioGraph};
        use crate::primitive::LfoNode;

        // Build a tremolo graph manually
        let mut graph = AudioGraph::new();
        let lfo = graph.add(LfoNode::with_freq(5.0, 44100.0));
        let gain = graph.add(AffineNode::gain(1.0));

        graph.connect_input(gain);
        graph.set_output(gain);
        graph.modulate(lfo, gain, AffineNode::PARAM_GAIN, 0.5, 0.5);

        // Verify fingerprint matches
        let fp = compute_fingerprint(&graph);
        let pattern = tremolo_pattern();
        assert!(fp.contains(&pattern.required));

        // Verify structural match
        let match_result = structural_match(&graph, &pattern);
        assert!(match_result.is_some());

        let m = match_result.unwrap();
        assert_eq!(m.pattern_name, "tremolo");
        assert_eq!(m.node_mapping.len(), 2);
    }

    #[test]
    fn test_optimize_tremolo_graph() {
        use crate::graph::{AffineNode, AudioGraph};
        use crate::primitive::LfoNode;

        // Build a tremolo graph
        let mut graph = AudioGraph::new();
        let lfo = graph.add(LfoNode::with_freq(5.0, 44100.0));
        let gain = graph.add(AffineNode::gain(1.0));

        graph.connect_input(gain);
        graph.set_output(gain);
        graph.modulate(lfo, gain, AffineNode::PARAM_GAIN, 0.5, 0.5);

        // Before optimization: 2 nodes
        assert_eq!(graph.node_count(), 2);

        // Optimize
        optimize_graph(&mut graph, &default_patterns());

        // After optimization: 1 node (the optimized tremolo)
        assert_eq!(graph.node_count(), 1);

        // Verify it still processes audio
        let ctx = AudioContext::new(44100.0);
        let output = graph.process(1.0, &ctx);
        assert!(output.abs() <= 1.0); // Should be in valid range
    }

    #[test]
    fn test_flanger_pattern_match() {
        use crate::graph::AudioGraph;
        use crate::primitive::{DelayNode, LfoNode};

        // Build a flanger graph
        let mut graph = AudioGraph::new();
        let lfo = graph.add(LfoNode::with_freq(0.3, 44100.0));
        let delay = graph.add(DelayNode::new(500));

        graph.connect_input(delay);
        graph.set_output(delay);
        graph.modulate(lfo, delay, DelayNode::PARAM_TIME, 220.0, 130.0);

        // Verify fingerprint matches
        let fp = compute_fingerprint(&graph);
        let pattern = flanger_pattern();
        assert!(fp.contains(&pattern.required));

        // Verify structural match
        let match_result = structural_match(&graph, &pattern);
        assert!(match_result.is_some());
        assert_eq!(match_result.unwrap().pattern_name, "flanger");
    }

    #[test]
    fn test_optimize_flanger_graph() {
        use crate::graph::AudioGraph;
        use crate::primitive::{DelayNode, LfoNode};

        let mut graph = AudioGraph::new();
        let lfo = graph.add(LfoNode::with_freq(0.3, 44100.0));
        let delay = graph.add(DelayNode::new(500));

        graph.connect_input(delay);
        graph.set_output(delay);
        graph.modulate(lfo, delay, DelayNode::PARAM_TIME, 220.0, 130.0);

        assert_eq!(graph.node_count(), 2);
        optimize_graph(&mut graph, &default_patterns());
        assert_eq!(graph.node_count(), 1);

        let ctx = AudioContext::new(44100.0);
        let output = graph.process(1.0, &ctx);
        assert!(output.abs() <= 2.0); // With feedback, might exceed 1.0 briefly
    }

    #[test]
    fn test_chorus_pattern_match() {
        use crate::graph::AudioGraph;
        use crate::primitive::{DelayNode, LfoNode, MixNode};

        // Build a chorus graph
        let mut graph = AudioGraph::new();
        let lfo = graph.add(LfoNode::with_freq(0.8, 44100.0));
        let delay = graph.add(DelayNode::new(2000));
        let mixer = graph.add(MixNode::new(0.5));

        graph.connect_input(delay);
        graph.connect(delay, mixer);
        graph.set_output(mixer);
        graph.modulate(lfo, delay, DelayNode::PARAM_TIME, 880.0, 220.0);

        // Verify fingerprint matches
        let fp = compute_fingerprint(&graph);
        let pattern = chorus_pattern();
        assert!(fp.contains(&pattern.required));

        // Verify structural match
        let match_result = structural_match(&graph, &pattern);
        assert!(match_result.is_some());
        assert_eq!(match_result.unwrap().pattern_name, "chorus");
    }

    #[test]
    fn test_optimize_chorus_graph() {
        use crate::graph::AudioGraph;
        use crate::primitive::{DelayNode, LfoNode, MixNode};

        let mut graph = AudioGraph::new();
        let lfo = graph.add(LfoNode::with_freq(0.8, 44100.0));
        let delay = graph.add(DelayNode::new(2000));
        let mixer = graph.add(MixNode::new(0.5));

        graph.connect_input(delay);
        graph.connect(delay, mixer);
        graph.set_output(mixer);
        graph.modulate(lfo, delay, DelayNode::PARAM_TIME, 880.0, 220.0);

        assert_eq!(graph.node_count(), 3);
        optimize_graph(&mut graph, &default_patterns());
        assert_eq!(graph.node_count(), 1);

        let ctx = AudioContext::new(44100.0);
        let output = graph.process(1.0, &ctx);
        assert!(output.abs() <= 1.5);
    }

    // ========================================================================
    // Graph Optimization Pass Tests
    // ========================================================================

    #[test]
    fn test_affine_composition() {
        // Gain(0.5) then Offset(1.0) then Gain(2.0)
        // Step 1: y = 0.5x
        // Step 2: y = 0.5x + 1.0
        // Step 3: y = 2.0 * (0.5x + 1.0) = 1.0x + 2.0
        let a = AffineNode::new(0.5, 0.0);
        let b = AffineNode::new(1.0, 1.0);
        let c = AffineNode::new(2.0, 0.0);

        let composed = a.then(b).then(c);
        assert!((composed.gain - 1.0).abs() < 1e-6);
        assert!((composed.offset - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_fuse_gain_offset_chain() {
        use crate::graph::{AffineNode, AudioGraph};

        // Build: Input -> Gain(0.5) -> Offset(1.0) -> Gain(2.0) -> Output
        let mut graph = AudioGraph::new();
        let g1 = graph.add(AffineNode::gain(0.5));
        let o1 = graph.add(AffineNode::offset(1.0));
        let g2 = graph.add(AffineNode::gain(2.0));

        graph.connect_input(g1);
        graph.connect(g1, o1);
        graph.connect(o1, g2);
        graph.set_output(g2);

        assert_eq!(graph.node_count(), 3);

        // Test original behavior
        let ctx = AudioContext::new(44100.0);
        let original_output = graph.process(2.0, &ctx);
        // (2.0 * 0.5 + 1.0) * 2.0 = (1.0 + 1.0) * 2.0 = 4.0
        assert!(
            (original_output - 4.0).abs() < 1e-6,
            "got {}",
            original_output
        );

        // Fuse the chain
        let fused = fuse_affine_chains(&mut graph);
        assert!(fused > 0, "expected some nodes to be fused");
        assert_eq!(graph.node_count(), 1, "chain should be fused to 1 node");

        // Verify same output
        let optimized_output = graph.process(2.0, &ctx);
        assert!(
            (optimized_output - original_output).abs() < 1e-6,
            "expected {}, got {}",
            original_output,
            optimized_output
        );
    }

    #[test]
    fn test_eliminate_identity_gain() {
        use crate::graph::{AffineNode, AudioGraph};

        // Build: Input -> Gain(1.0) -> Output (identity, should be removed)
        let mut graph = AudioGraph::new();
        let g = graph.add(AffineNode::gain(1.0));
        graph.connect_input(g);
        graph.set_output(g);

        assert_eq!(graph.node_count(), 1);

        let removed = eliminate_identities(&mut graph);
        assert_eq!(removed, 1);
        assert_eq!(graph.node_count(), 0);
    }

    #[test]
    fn test_eliminate_identity_offset() {
        use crate::graph::{AffineNode, AudioGraph};

        // Build: Input -> Offset(0.0) -> Output (identity, should be removed)
        let mut graph = AudioGraph::new();
        let o = graph.add(AffineNode::offset(0.0));
        graph.connect_input(o);
        graph.set_output(o);

        assert_eq!(graph.node_count(), 1);

        let removed = eliminate_identities(&mut graph);
        assert_eq!(removed, 1);
        assert_eq!(graph.node_count(), 0);
    }

    #[test]
    fn test_run_all_passes() {
        use crate::graph::{AffineNode, AudioGraph};

        // Build a complex chain with identities mixed in
        // Input -> PassThrough -> Gain(0.5) -> Offset(0.0) -> Gain(2.0) -> Output
        let mut graph = AudioGraph::new();
        let p = graph.add(AffineNode::identity());
        let g1 = graph.add(AffineNode::gain(0.5));
        let o = graph.add(AffineNode::offset(0.0)); // Identity
        let g2 = graph.add(AffineNode::gain(2.0));

        graph.connect_input(p);
        graph.connect(p, g1);
        graph.connect(g1, o);
        graph.connect(o, g2);
        graph.set_output(g2);

        assert_eq!(graph.node_count(), 4);

        // Test original behavior
        let ctx = AudioContext::new(44100.0);
        let original_output = graph.process(2.0, &ctx);
        // 2.0 * 0.5 * 2.0 = 2.0
        assert!(
            (original_output - 2.0).abs() < 1e-6,
            "got {}",
            original_output
        );

        // Run all optimization passes
        let total = run_optimization_passes(&mut graph);
        assert!(total > 0, "expected some optimizations");

        // Should reduce to 1 node (or possibly 0 if it becomes identity)
        assert!(
            graph.node_count() <= 2,
            "expected <= 2 nodes, got {}",
            graph.node_count()
        );

        // Verify same output (if graph is not empty)
        if graph.node_count() > 0 {
            let optimized_output = graph.process(2.0, &ctx);
            assert!(
                (optimized_output - original_output).abs() < 1e-6,
                "expected {}, got {}",
                original_output,
                optimized_output
            );
        }
    }

    #[test]
    fn test_eliminate_dead_simple() {
        use crate::graph::{AffineNode, AudioGraph};

        // Build: A -> Output, B (disconnected)
        let mut graph = AudioGraph::new();
        let a = graph.add(AffineNode::gain(2.0));
        let _b = graph.add(AffineNode::gain(3.0)); // Dead node

        graph.connect_input(a);
        graph.set_output(a);

        assert_eq!(graph.node_count(), 2);

        let removed = eliminate_dead_nodes(&mut graph);
        assert_eq!(removed, 1);
        assert_eq!(graph.node_count(), 1);

        // Verify output still works
        let ctx = AudioContext::new(44100.0);
        let out = graph.process(1.0, &ctx);
        assert!((out - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_eliminate_dead_chain() {
        use crate::graph::{AffineNode, AudioGraph};

        // Build: A -> B -> Output, C -> D (disconnected chain)
        let mut graph = AudioGraph::new();
        let a = graph.add(AffineNode::gain(2.0));
        let b = graph.add(AffineNode::gain(0.5));
        let c = graph.add(AffineNode::gain(10.0)); // Dead
        let d = graph.add(AffineNode::gain(10.0)); // Dead

        graph.connect_input(a);
        graph.connect(a, b);
        graph.connect(c, d); // Dead chain
        graph.set_output(b);

        assert_eq!(graph.node_count(), 4);

        let removed = eliminate_dead_nodes(&mut graph);
        assert_eq!(removed, 2);
        assert_eq!(graph.node_count(), 2);

        // Verify output: 1.0 * 2.0 * 0.5 = 1.0
        let ctx = AudioContext::new(44100.0);
        let out = graph.process(1.0, &ctx);
        assert!((out - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_eliminate_dead_keeps_modulators() {
        use crate::graph::{AffineNode, AudioGraph};
        use crate::primitive::LfoNode;

        // Build: LFO -> (modulates) Gain -> Output
        // The LFO doesn't have audio output to Gain, only param modulation
        let mut graph = AudioGraph::new();
        let lfo = graph.add(LfoNode::with_freq(5.0, 44100.0));
        let gain = graph.add(AffineNode::gain(1.0));

        graph.connect_input(gain);
        graph.set_output(gain);
        graph.modulate(lfo, gain, AffineNode::PARAM_GAIN, 0.5, 0.5);

        assert_eq!(graph.node_count(), 2);

        // LFO should NOT be removed - it modulates a live node
        let removed = eliminate_dead_nodes(&mut graph);
        assert_eq!(removed, 0);
        assert_eq!(graph.node_count(), 2);
    }

    #[test]
    fn test_eliminate_dead_no_output() {
        use crate::graph::{AffineNode, AudioGraph};

        // Graph with no output set - nothing should be removed
        let mut graph = AudioGraph::new();
        let _a = graph.add(AffineNode::gain(2.0));
        let _b = graph.add(AffineNode::gain(3.0));

        assert_eq!(graph.node_count(), 2);

        let removed = eliminate_dead_nodes(&mut graph);
        assert_eq!(removed, 0); // Can't determine liveness without output
    }

    // ========================================================================
    // Constant Folding Tests
    // ========================================================================

    #[test]
    fn test_fold_constant_through_gain() {
        use crate::graph::{AffineNode, AudioGraph, Constant};

        // Build: Constant(2.0) -> Gain(3.0) -> Output
        // Result should be Constant(6.0)
        let mut graph = AudioGraph::new();
        let c = graph.add(Constant(2.0));
        let g = graph.add(AffineNode::gain(3.0));

        graph.connect(c, g);
        graph.set_output(g);

        assert_eq!(graph.node_count(), 2);

        let folded = fold_constants(&mut graph);
        assert_eq!(folded, 1);
        assert_eq!(graph.node_count(), 1);

        // Verify output
        let ctx = AudioContext::new(44100.0);
        let out = graph.process(0.0, &ctx);
        assert!((out - 6.0).abs() < 1e-6, "expected 6.0, got {}", out);
    }

    #[test]
    fn test_fold_constant_through_offset() {
        use crate::graph::{AffineNode, AudioGraph, Constant};

        // Build: Constant(2.0) -> Offset(5.0) -> Output
        // Result should be Constant(7.0)
        let mut graph = AudioGraph::new();
        let c = graph.add(Constant(2.0));
        let o = graph.add(AffineNode::offset(5.0));

        graph.connect(c, o);
        graph.set_output(o);

        assert_eq!(graph.node_count(), 2);

        let folded = fold_constants(&mut graph);
        assert_eq!(folded, 1);
        assert_eq!(graph.node_count(), 1);

        // Verify output
        let ctx = AudioContext::new(44100.0);
        let out = graph.process(0.0, &ctx);
        assert!((out - 7.0).abs() < 1e-6, "expected 7.0, got {}", out);
    }

    #[test]
    fn test_fold_constant_through_affine() {
        use crate::graph::{AffineNode, AudioGraph, Constant};

        // Build: Constant(2.0) -> Affine(3.0, 1.0) -> Output
        // Result should be Constant(2.0 * 3.0 + 1.0 = 7.0)
        let mut graph = AudioGraph::new();
        let c = graph.add(Constant(2.0));
        let a = graph.add(AffineNode::new(3.0, 1.0));

        graph.connect(c, a);
        graph.set_output(a);

        assert_eq!(graph.node_count(), 2);

        let folded = fold_constants(&mut graph);
        assert_eq!(folded, 1);
        assert_eq!(graph.node_count(), 1);

        // Verify output
        let ctx = AudioContext::new(44100.0);
        let out = graph.process(0.0, &ctx);
        assert!((out - 7.0).abs() < 1e-6, "expected 7.0, got {}", out);
    }

    #[test]
    fn test_fold_constant_chain() {
        use crate::graph::{AffineNode, AudioGraph, Constant};

        // Build: Constant(1.0) -> Gain(2.0) -> Offset(3.0) -> Gain(4.0) -> Output
        // Result should be Constant(((1.0 * 2.0) + 3.0) * 4.0 = 20.0)
        let mut graph = AudioGraph::new();
        let c = graph.add(Constant(1.0));
        let g1 = graph.add(AffineNode::gain(2.0));
        let o = graph.add(AffineNode::offset(3.0));
        let g2 = graph.add(AffineNode::gain(4.0));

        graph.connect(c, g1);
        graph.connect(g1, o);
        graph.connect(o, g2);
        graph.set_output(g2);

        assert_eq!(graph.node_count(), 4);

        // Use propagate_constants which runs both fold and fuse
        let propagated = propagate_constants(&mut graph);
        assert!(propagated > 0);
        assert_eq!(graph.node_count(), 1);

        // Verify output
        let ctx = AudioContext::new(44100.0);
        let out = graph.process(0.0, &ctx);
        assert!((out - 20.0).abs() < 1e-6, "expected 20.0, got {}", out);
    }

    // ========================================================================
    // Delay Merging Tests
    // ========================================================================

    #[test]
    fn test_merge_simple_delays() {
        use crate::graph::AudioGraph;
        use crate::primitive::DelayNode;

        // Build: Input -> Delay(100) -> Delay(50) -> Output
        let mut graph = AudioGraph::new();

        let mut d1 = DelayNode::new(200);
        d1.set_time(100.0);
        let d1_idx = graph.add(d1);

        let mut d2 = DelayNode::new(100);
        d2.set_time(50.0);
        let d2_idx = graph.add(d2);

        graph.connect_input(d1_idx);
        graph.connect(d1_idx, d2_idx);
        graph.set_output(d2_idx);

        assert_eq!(graph.node_count(), 2);

        let merged = merge_delays(&mut graph);
        assert_eq!(merged, 1);
        assert_eq!(graph.node_count(), 1);

        // The merged delay should have time = 150
        let merged_time = graph.node_param_value(0, 0).unwrap_or(0.0);
        assert!(
            (merged_time - 150.0).abs() < 1e-6,
            "expected 150.0, got {}",
            merged_time
        );
    }

    #[test]
    fn test_no_merge_with_feedback() {
        use crate::graph::AudioGraph;
        use crate::primitive::DelayNode;

        // Build: Input -> Delay(100, feedback=0.5) -> Delay(50) -> Output
        let mut graph = AudioGraph::new();
        let mut d1 = DelayNode::new(200);
        d1.set_feedback(0.5); // Has feedback, shouldn't merge
        let d1_idx = graph.add(d1);
        let d2 = graph.add(DelayNode::new(100));

        graph.connect_input(d1_idx);
        graph.connect(d1_idx, d2);
        graph.set_output(d2);

        assert_eq!(graph.node_count(), 2);

        let merged = merge_delays(&mut graph);
        assert_eq!(merged, 0); // Should not merge due to feedback
        assert_eq!(graph.node_count(), 2);
    }

    #[test]
    fn test_merge_delay_chain() {
        use crate::graph::AudioGraph;
        use crate::primitive::DelayNode;

        // Build: Input -> Delay(50) -> Delay(30) -> Delay(20) -> Output
        let mut graph = AudioGraph::new();

        let mut d1 = DelayNode::new(100);
        d1.set_time(50.0);
        let d1_idx = graph.add(d1);

        let mut d2 = DelayNode::new(100);
        d2.set_time(30.0);
        let d2_idx = graph.add(d2);

        let mut d3 = DelayNode::new(100);
        d3.set_time(20.0);
        let d3_idx = graph.add(d3);

        graph.connect_input(d1_idx);
        graph.connect(d1_idx, d2_idx);
        graph.connect(d2_idx, d3_idx);
        graph.set_output(d3_idx);

        assert_eq!(graph.node_count(), 3);

        // Merge delays iteratively
        let mut total_merged = 0;
        loop {
            let merged = merge_delays(&mut graph);
            if merged == 0 {
                break;
            }
            total_merged += merged;
        }

        assert_eq!(total_merged, 2); // Two merge operations
        assert_eq!(graph.node_count(), 1);

        // Total delay should be 100
        let merged_time = graph.node_param_value(0, 0).unwrap_or(0.0);
        assert!(
            (merged_time - 100.0).abs() < 1e-6,
            "expected 100.0, got {}",
            merged_time
        );
    }

    #[test]
    fn test_optimizer_pipeline() {
        use crate::graph::{AffineNode, AudioGraph};

        // Build: Input -> PassThrough -> Gain(0.5) -> Offset(0.0) -> Gain(2.0) -> Output
        let mut graph = AudioGraph::new();
        let p = graph.add(AffineNode::identity());
        let g1 = graph.add(AffineNode::gain(0.5));
        let o = graph.add(AffineNode::offset(0.0));
        let g2 = graph.add(AffineNode::gain(2.0));

        graph.connect_input(p);
        graph.connect(p, g1);
        graph.connect(g1, o);
        graph.connect(o, g2);
        graph.set_output(g2);

        let ctx = AudioContext::new(44100.0);
        let original_output = graph.process(2.0, &ctx);

        // Use OptimizerPipeline instead of run_optimization_passes
        let pipeline = OptimizerPipeline::default();
        let total = pipeline.run(&mut graph);
        assert!(total > 0, "expected some optimizations");

        // Verify same output
        if graph.node_count() > 0 {
            let optimized_output = graph.process(2.0, &ctx);
            assert!(
                (optimized_output - original_output).abs() < 1e-6,
                "expected {}, got {}",
                original_output,
                optimized_output
            );
        }
    }

    #[test]
    fn test_optimizer_trait_individual() {
        use crate::graph::{AffineNode, AudioGraph};

        // Test individual optimizers via trait
        let mut graph = AudioGraph::new();
        let g1 = graph.add(AffineNode::gain(0.5));
        let g2 = graph.add(AffineNode::gain(2.0));

        graph.connect_input(g1);
        graph.connect(g1, g2);
        graph.set_output(g2);

        let fuser = AffineChainFuser;
        assert_eq!(fuser.name(), "AffineChainFuser");

        let fused = fuser.apply(&mut graph);
        assert_eq!(fused, 1);
        assert_eq!(graph.node_count(), 1);
    }
}
