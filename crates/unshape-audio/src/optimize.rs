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

use crate::graph::{AudioGraph, AudioNode, NodeIndex};
use unshape_core::optimize::Optimizer;
use std::any::TypeId;
use std::collections::HashMap;

// ============================================================================
// GraphOptimizer Trait
// ============================================================================

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

/// Re-export the generic OptimizerPipeline with default audio passes.
///
/// For custom pipelines, use `unshape_core::optimize::OptimizerPipeline<AudioGraph>`.
pub use unshape_core::optimize::OptimizerPipeline as GenericPipeline;

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

// ============================================================================
// Node Type Registry
// ============================================================================

/// Enumeration of known audio node types for fingerprinting.
///
/// This is used for fast pattern filtering - we count how many of each type
/// exist in a graph, then only check patterns that could possibly match.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum NodeType {
    // Primitives (from primitive.rs)
    Delay,
    Lfo,
    Envelope,
    Allpass,
    Mix,

    // Filters (from graph.rs)
    LowPass,
    HighPass,
    Biquad,
    FeedbackDelay,

    // Envelopes (from graph.rs)
    Adsr,
    Ar,

    // Basic (from graph.rs)
    Oscillator,
    Clip,
    SoftClip,
    Constant,
    Silence,

    // Linear transform (unified gain/offset/passthrough)
    Affine,

    // Unknown/custom nodes
    Unknown,
}

impl NodeType {
    /// Total number of known node types.
    pub const COUNT: usize = 18;

    /// Get node type from a type ID.
    pub fn from_type_id(id: TypeId) -> Self {
        NODE_TYPE_REGISTRY.with(|registry| registry.get(&id).copied().unwrap_or(NodeType::Unknown))
    }
}

// Build the node type registry - maps TypeId to NodeType for known nodes
fn build_node_type_registry() -> HashMap<TypeId, NodeType> {
    let mut map = HashMap::new();
    map.insert(TypeId::of::<crate::primitive::DelayNode>(), NodeType::Delay);
    map.insert(TypeId::of::<crate::primitive::LfoNode>(), NodeType::Lfo);
    map.insert(
        TypeId::of::<crate::primitive::EnvelopeNode>(),
        NodeType::Envelope,
    );
    map.insert(
        TypeId::of::<crate::primitive::AllpassNode>(),
        NodeType::Allpass,
    );
    map.insert(TypeId::of::<crate::primitive::MixNode>(), NodeType::Mix);
    map.insert(TypeId::of::<crate::graph::LowPassNode>(), NodeType::LowPass);
    map.insert(
        TypeId::of::<crate::graph::HighPassNode>(),
        NodeType::HighPass,
    );
    map.insert(TypeId::of::<crate::graph::BiquadNode>(), NodeType::Biquad);
    map.insert(
        TypeId::of::<crate::graph::FeedbackDelayNode>(),
        NodeType::FeedbackDelay,
    );
    map.insert(TypeId::of::<crate::graph::AdsrNode>(), NodeType::Adsr);
    map.insert(TypeId::of::<crate::graph::ArNode>(), NodeType::Ar);
    map.insert(
        TypeId::of::<crate::graph::Oscillator>(),
        NodeType::Oscillator,
    );
    map.insert(TypeId::of::<crate::graph::Clip>(), NodeType::Clip);
    map.insert(TypeId::of::<crate::graph::SoftClip>(), NodeType::SoftClip);
    map.insert(TypeId::of::<crate::graph::Constant>(), NodeType::Constant);
    map.insert(TypeId::of::<crate::graph::Silence>(), NodeType::Silence);
    map.insert(TypeId::of::<crate::graph::AffineNode>(), NodeType::Affine);
    map
}

thread_local! {
    static NODE_TYPE_REGISTRY: HashMap<TypeId, NodeType> = build_node_type_registry();
}

// ============================================================================
// Graph Fingerprint
// ============================================================================

/// Compact representation of node type counts in a graph.
///
/// Used for fast pattern filtering - if a graph doesn't have the required
/// node types, we skip detailed structural matching entirely.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct GraphFingerprint {
    /// Count of each node type (saturates at 255).
    counts: [u8; NodeType::COUNT],
}

impl GraphFingerprint {
    /// Create an empty fingerprint.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a fingerprint with specific counts.
    pub fn with_counts(counts: [u8; NodeType::COUNT]) -> Self {
        Self { counts }
    }

    /// Increment the count for a node type.
    pub fn add(&mut self, node_type: NodeType) {
        let idx = node_type as usize;
        if idx < NodeType::COUNT {
            self.counts[idx] = self.counts[idx].saturating_add(1);
        }
    }

    /// Get the count for a node type.
    pub fn count(&self, node_type: NodeType) -> u8 {
        let idx = node_type as usize;
        if idx < NodeType::COUNT {
            self.counts[idx]
        } else {
            0
        }
    }

    /// Check if this fingerprint contains at least the counts in `required`.
    ///
    /// Returns true if for every node type, `self.count >= required.count`.
    #[inline]
    pub fn contains(&self, required: &GraphFingerprint) -> bool {
        // This is SIMDable: element-wise >= comparison
        for i in 0..NodeType::COUNT {
            if self.counts[i] < required.counts[i] {
                return false;
            }
        }
        true
    }
}

/// Macro to create a fingerprint from node type counts.
#[macro_export]
macro_rules! fingerprint {
    ($($node_type:ident : $count:expr),* $(,)?) => {{
        let mut fp = $crate::optimize::GraphFingerprint::new();
        $(
            for _ in 0..$count {
                fp.add($crate::optimize::NodeType::$node_type);
            }
        )*
        fp
    }};
}

// ============================================================================
// Pattern Definition
// ============================================================================

/// A pattern that can be matched against a subgraph.
pub struct Pattern {
    /// Human-readable name for debugging.
    pub name: &'static str,

    /// Minimum node types required (for fingerprint filtering).
    pub required: GraphFingerprint,

    /// Internal pattern structure.
    pub structure: PatternStructure,

    /// Factory to create optimized replacement node.
    pub build: fn(&MatchResult) -> Box<dyn AudioNode>,

    /// Priority for overlapping matches (higher = preferred).
    pub priority: i32,
}

/// Structure of a pattern to match.
pub struct PatternStructure {
    /// Nodes in the pattern.
    pub nodes: Vec<PatternNode>,

    /// Required audio wires within the pattern (from_idx, to_idx).
    pub audio_wires: Vec<(usize, usize)>,

    /// Required param modulation wires (from_idx, to_idx, param_name).
    pub param_wires: Vec<(usize, usize, &'static str)>,

    /// Which pattern nodes receive external audio input.
    pub external_inputs: Vec<usize>,

    /// Which pattern nodes send audio to external nodes.
    pub external_outputs: Vec<usize>,
}

/// A node within a pattern.
pub struct PatternNode {
    /// Required node type.
    pub node_type: NodeType,

    /// Optional parameter constraints (not yet implemented).
    pub constraints: Vec<ParamConstraint>,
}

/// Constraint on a node's parameters (for future use).
pub enum ParamConstraint {
    /// Parameter must equal a specific value.
    Equals(&'static str, f32),
    /// Parameter must be in a range.
    Range(&'static str, f32, f32),
}

// ============================================================================
// Match Result
// ============================================================================

/// Result of successfully matching a pattern against a graph.
pub struct MatchResult {
    /// Pattern that was matched.
    pub pattern_name: &'static str,

    /// Mapping from pattern node indices to graph node indices.
    pub node_mapping: Vec<NodeIndex>,

    /// External audio inputs: (external_node, pattern_input_idx).
    pub external_audio_inputs: Vec<(NodeIndex, usize)>,

    /// External audio outputs: (pattern_output_idx, external_node).
    pub external_audio_outputs: Vec<(usize, NodeIndex)>,

    /// Extracted parameters from matched nodes.
    pub extracted_params: HashMap<(usize, &'static str), f32>,

    /// Modulation parameters: (from_pattern_idx, to_pattern_idx, base, scale).
    pub modulations: Vec<(usize, usize, f32, f32)>,
}

impl MatchResult {
    /// Get the number of nodes in the match.
    pub fn size(&self) -> usize {
        self.node_mapping.len()
    }

    /// Get an extracted parameter value.
    pub fn get_param(&self, pattern_node: usize, param_name: &str) -> Option<f32> {
        self.extracted_params
            .get(&(pattern_node, param_name))
            .copied()
    }

    /// Get modulation base/scale for a param wire.
    pub fn get_modulation(&self, from: usize, to: usize) -> Option<(f32, f32)> {
        self.modulations
            .iter()
            .find(|(f, t, _, _)| *f == from && *t == to)
            .map(|(_, _, base, scale)| (*base, *scale))
    }
}

// ============================================================================
// Structural Matching
// ============================================================================

/// Attempt to match a pattern against a graph.
///
/// Returns the first valid match found, or None if no match exists.
pub fn structural_match(graph: &AudioGraph, pattern: &Pattern) -> Option<MatchResult> {
    let structure = &pattern.structure;

    // Early exit if pattern has no nodes
    if structure.nodes.is_empty() {
        return None;
    }

    // Get graph info for matching
    let graph_info = GraphInfo::from_graph(graph);

    // Try to find an assignment of pattern nodes to graph nodes
    let mut assignment = vec![None; structure.nodes.len()];

    if find_assignment(graph, &graph_info, pattern, &mut assignment, 0) {
        // Build match result from assignment
        let node_mapping: Vec<NodeIndex> = assignment.into_iter().map(|o| o.unwrap()).collect();

        // Find external connections
        let external_audio_inputs = find_external_inputs(graph, &node_mapping, structure);
        let external_audio_outputs = find_external_outputs(graph, &node_mapping, structure);

        // Extract modulation parameters
        let modulations = extract_modulations(graph, &node_mapping, structure);

        Some(MatchResult {
            pattern_name: pattern.name,
            node_mapping,
            external_audio_inputs,
            external_audio_outputs,
            extracted_params: HashMap::new(), // TODO: extract from nodes
            modulations,
        })
    } else {
        None
    }
}

/// Information about graph structure for efficient matching.
struct GraphInfo {
    /// Nodes by type.
    nodes_by_type: HashMap<NodeType, Vec<NodeIndex>>,
    /// Audio wires as (from, to) pairs.
    audio_edges: Vec<(NodeIndex, NodeIndex)>,
    /// Param wires as (from, to, param_name).
    param_edges: Vec<(NodeIndex, NodeIndex, String)>,
}

impl GraphInfo {
    fn from_graph(graph: &AudioGraph) -> Self {
        let mut nodes_by_type: HashMap<NodeType, Vec<NodeIndex>> = HashMap::new();

        // Group nodes by type
        for i in 0..graph.node_count() {
            let node_type = graph.node_type(i).unwrap_or(NodeType::Unknown);
            nodes_by_type.entry(node_type).or_default().push(i);
        }

        // Collect edges
        let audio_edges = graph.audio_wires().iter().map(|w| (w.from, w.to)).collect();
        let param_edges = graph
            .param_wires()
            .iter()
            .map(|w| {
                let param_name = graph
                    .node_param_name(w.to, w.param)
                    .unwrap_or("unknown")
                    .to_string();
                (w.from, w.to, param_name)
            })
            .collect();

        Self {
            nodes_by_type,
            audio_edges,
            param_edges,
        }
    }
}

/// Recursive backtracking to find a valid assignment.
fn find_assignment(
    graph: &AudioGraph,
    info: &GraphInfo,
    pattern: &Pattern,
    assignment: &mut [Option<NodeIndex>],
    pattern_idx: usize,
) -> bool {
    // All nodes assigned successfully
    if pattern_idx >= pattern.structure.nodes.len() {
        return validate_wires(info, pattern, assignment);
    }

    let required_type = pattern.structure.nodes[pattern_idx].node_type;

    // Get candidate graph nodes of the required type
    let candidates = info
        .nodes_by_type
        .get(&required_type)
        .map(|v| v.as_slice())
        .unwrap_or(&[]);

    for &candidate in candidates {
        // Skip if already assigned
        if assignment.iter().any(|&a| a == Some(candidate)) {
            continue;
        }

        // Try this assignment
        assignment[pattern_idx] = Some(candidate);

        // Recurse to assign remaining nodes
        if find_assignment(graph, info, pattern, assignment, pattern_idx + 1) {
            return true;
        }

        // Backtrack
        assignment[pattern_idx] = None;
    }

    false
}

/// Validate that all required wires exist between assigned nodes.
fn validate_wires(info: &GraphInfo, pattern: &Pattern, assignment: &[Option<NodeIndex>]) -> bool {
    let structure = &pattern.structure;

    // Check audio wires
    for &(from_pat, to_pat) in &structure.audio_wires {
        let from_graph = assignment[from_pat].unwrap();
        let to_graph = assignment[to_pat].unwrap();

        if !info.audio_edges.contains(&(from_graph, to_graph)) {
            return false;
        }
    }

    // Check param wires
    for &(from_pat, to_pat, param_name) in &structure.param_wires {
        let from_graph = assignment[from_pat].unwrap();
        let to_graph = assignment[to_pat].unwrap();

        let found = info
            .param_edges
            .iter()
            .any(|(f, t, p)| *f == from_graph && *t == to_graph && p == param_name);

        if !found {
            return false;
        }
    }

    true
}

/// Find external audio inputs to the matched subgraph.
fn find_external_inputs(
    graph: &AudioGraph,
    node_mapping: &[NodeIndex],
    structure: &PatternStructure,
) -> Vec<(NodeIndex, usize)> {
    let mut inputs = Vec::new();

    for &wire in graph.audio_wires() {
        // Wire ends at a matched node
        if let Some(pattern_idx) = node_mapping.iter().position(|&n| n == wire.to) {
            // Wire starts from outside the match
            if !node_mapping.contains(&wire.from) {
                // And this pattern node is marked as external input
                if structure.external_inputs.contains(&pattern_idx) {
                    inputs.push((wire.from, pattern_idx));
                }
            }
        }
    }

    inputs
}

/// Find external audio outputs from the matched subgraph.
fn find_external_outputs(
    graph: &AudioGraph,
    node_mapping: &[NodeIndex],
    structure: &PatternStructure,
) -> Vec<(usize, NodeIndex)> {
    let mut outputs = Vec::new();

    for &wire in graph.audio_wires() {
        // Wire starts from a matched node
        if let Some(pattern_idx) = node_mapping.iter().position(|&n| n == wire.from) {
            // Wire ends outside the match
            if !node_mapping.contains(&wire.to) {
                // And this pattern node is marked as external output
                if structure.external_outputs.contains(&pattern_idx) {
                    outputs.push((pattern_idx, wire.to));
                }
            }
        }
    }

    outputs
}

/// Extract modulation parameters from param wires.
fn extract_modulations(
    graph: &AudioGraph,
    node_mapping: &[NodeIndex],
    structure: &PatternStructure,
) -> Vec<(usize, usize, f32, f32)> {
    let mut modulations = Vec::new();

    for &(from_pat, to_pat, _param_name) in &structure.param_wires {
        let from_graph = node_mapping[from_pat];
        let to_graph = node_mapping[to_pat];

        // Find the param wire in the graph
        for wire in graph.param_wires() {
            if wire.from == from_graph && wire.to == to_graph {
                modulations.push((from_pat, to_pat, wire.base, wire.scale));
                break;
            }
        }
    }

    modulations
}

// ============================================================================
// Graph Optimization
// ============================================================================

/// Optimize a graph by replacing recognized patterns with optimized implementations.
///
/// Applies patterns greedily, preferring larger matches and higher priority patterns.
/// Continues until no more patterns match.
pub fn optimize_graph(graph: &mut AudioGraph, patterns: &[Pattern]) {
    loop {
        let fingerprint = compute_fingerprint(graph);

        // Filter to patterns that could possibly match
        let candidates: Vec<&Pattern> = patterns
            .iter()
            .filter(|p| fingerprint.contains(&p.required))
            .collect();

        if candidates.is_empty() {
            break;
        }

        // Find best match across all candidate patterns
        let mut best: Option<(MatchResult, &Pattern)> = None;

        for pattern in candidates {
            if let Some(result) = structural_match(graph, pattern) {
                let dominated = best.as_ref().map_or(false, |(b, bp)| {
                    // Prefer: larger size, then higher priority
                    result.size() < b.size()
                        || (result.size() == b.size() && pattern.priority <= bp.priority)
                });

                if !dominated {
                    best = Some((result, pattern));
                }
            }
        }

        // Apply best match or terminate
        match best {
            Some((result, pattern)) => {
                let optimized = (pattern.build)(&result);
                replace_subgraph(graph, &result, optimized);
            }
            None => break,
        }
    }
}

/// Compute fingerprint for a graph.
fn compute_fingerprint(graph: &AudioGraph) -> GraphFingerprint {
    let mut fp = GraphFingerprint::new();

    for i in 0..graph.node_count() {
        if let Some(node_type) = graph.node_type(i) {
            fp.add(node_type);
        }
    }

    fp
}

/// Replace a matched subgraph with an optimized node.
fn replace_subgraph(
    graph: &mut AudioGraph,
    match_result: &MatchResult,
    replacement: Box<dyn AudioNode>,
) {
    // Add replacement node
    let new_id = graph.add_boxed(replacement);

    // Rewire external inputs to replacement
    for &(ext_node, _pattern_idx) in &match_result.external_audio_inputs {
        graph.reconnect_audio(ext_node, match_result.node_mapping[0], new_id);
    }

    // Rewire replacement output to external nodes
    for &(_pattern_idx, ext_node) in &match_result.external_audio_outputs {
        graph.reconnect_audio(match_result.node_mapping[0], ext_node, new_id);
    }

    // Handle input/output node references
    if let Some(input_node) = graph.input_node() {
        if match_result.node_mapping.contains(&input_node) {
            graph.connect_input(new_id);
        }
    }

    if let Some(output_node) = graph.output_node() {
        if match_result.node_mapping.contains(&output_node) {
            graph.set_output(new_id);
        }
    }

    // Remove matched nodes (in reverse order to preserve indices)
    let mut to_remove: Vec<NodeIndex> = match_result.node_mapping.clone();
    to_remove.sort_by(|a, b| b.cmp(a)); // Descending

    for node_id in to_remove {
        graph.remove_node(node_id);
    }
}

// ============================================================================
// Optimized Effect Implementations (Tier 2)
// ============================================================================

/// Optimized tremolo effect (LFO modulating gain).
///
/// This is the monomorphized version that eliminates dyn dispatch.
pub struct TremoloOptimized {
    lfo: crate::primitive::PhaseOsc,
    phase_inc: f32,
    base: f32,
    scale: f32,
}

impl TremoloOptimized {
    /// Create a new optimized tremolo.
    pub fn new(rate: f32, depth: f32, sample_rate: f32) -> Self {
        Self {
            lfo: crate::primitive::PhaseOsc::new(),
            phase_inc: rate / sample_rate,
            base: 1.0 - depth * 0.5,
            scale: depth * 0.5,
        }
    }

    /// Create from match result (used by pattern).
    pub fn from_match(m: &MatchResult, sample_rate: f32) -> Self {
        // Extract rate from LFO node (pattern idx 0)
        let rate = m.get_param(0, "rate").unwrap_or(5.0);
        // Extract modulation depth from param wire
        let (base, scale) = m.get_modulation(0, 1).unwrap_or((0.5, 0.5));

        Self {
            lfo: crate::primitive::PhaseOsc::new(),
            phase_inc: rate / sample_rate,
            base,
            scale,
        }
    }
}

impl crate::graph::AudioNode for TremoloOptimized {
    #[inline]
    fn process(&mut self, input: f32, _ctx: &crate::graph::AudioContext) -> f32 {
        let lfo_out = self.lfo.sine();
        self.lfo.advance(self.phase_inc);
        let gain = self.base + lfo_out * self.scale;
        input * gain
    }

    fn reset(&mut self) {
        self.lfo.reset();
    }
}

/// Optimized flanger effect (LFO modulating delay time).
///
/// Flanger = LFO → delay time, with feedback and dry/wet mix.
pub struct FlangerOptimized {
    lfo: crate::primitive::PhaseOsc,
    delay: crate::primitive::DelayLine<true>,
    phase_inc: f32,
    base_delay: f32,
    depth: f32,
    feedback: f32,
    // Pre-computed mix factors (constant folding)
    wet_mix: f32,
    dry_mix: f32,
}

impl FlangerOptimized {
    /// Create a new optimized flanger.
    pub fn new(
        rate: f32,
        base_delay_ms: f32,
        depth_ms: f32,
        feedback: f32,
        mix: f32,
        sample_rate: f32,
    ) -> Self {
        let base_delay = base_delay_ms * sample_rate / 1000.0;
        let depth = depth_ms * sample_rate / 1000.0;
        let max_delay = ((base_delay_ms + depth_ms * 2.0) * sample_rate / 1000.0) as usize + 1;

        Self {
            lfo: crate::primitive::PhaseOsc::new(),
            delay: crate::primitive::DelayLine::new(max_delay),
            phase_inc: rate / sample_rate,
            base_delay,
            depth,
            feedback,
            wet_mix: mix,
            dry_mix: 1.0 - mix,
        }
    }

    /// Create from match result.
    pub fn from_match(m: &MatchResult, sample_rate: f32) -> Self {
        let rate = m.get_param(0, "rate").unwrap_or(0.3);
        let (base_delay, depth) = m.get_modulation(0, 1).unwrap_or((220.0, 130.0)); // ~5ms, ~3ms at 44.1kHz
        let feedback = m.get_param(1, "feedback").unwrap_or(0.7);
        let mix = m.get_param(2, "mix").unwrap_or(0.5);

        let max_delay = (base_delay + depth * 2.0) as usize + 1;

        Self {
            lfo: crate::primitive::PhaseOsc::new(),
            delay: crate::primitive::DelayLine::new(max_delay),
            phase_inc: rate / sample_rate,
            base_delay,
            depth,
            feedback,
            wet_mix: mix,
            dry_mix: 1.0 - mix,
        }
    }
}

impl crate::graph::AudioNode for FlangerOptimized {
    #[inline]
    fn process(&mut self, input: f32, _ctx: &crate::graph::AudioContext) -> f32 {
        let lfo_out = self.lfo.sine();
        self.lfo.advance(self.phase_inc);

        let delay_time = self.base_delay + lfo_out * self.depth;
        let delayed = self.delay.read_interp(delay_time);

        // Write with feedback
        self.delay.write(input + delayed * self.feedback);

        // Mix dry and wet (pre-computed factors)
        input * self.dry_mix + delayed * self.wet_mix
    }

    fn reset(&mut self) {
        self.lfo.reset();
        self.delay.clear();
    }
}

/// Optimized chorus effect (LFO modulating delay time with mix).
///
/// Chorus = LFO → delay time, mixed with dry signal.
pub struct ChorusOptimized {
    lfo: crate::primitive::PhaseOsc,
    delay: crate::primitive::DelayLine<true>,
    phase_inc: f32,
    base_delay: f32,
    depth: f32,
    // Pre-computed mix factors (constant folding)
    wet_mix: f32,
    dry_mix: f32,
}

impl ChorusOptimized {
    /// Create a new optimized chorus.
    pub fn new(rate: f32, base_delay_ms: f32, depth_ms: f32, mix: f32, sample_rate: f32) -> Self {
        let base_delay = base_delay_ms * sample_rate / 1000.0;
        let depth = depth_ms * sample_rate / 1000.0;
        let max_delay = ((base_delay_ms + depth_ms * 2.0) * sample_rate / 1000.0) as usize + 1;

        Self {
            lfo: crate::primitive::PhaseOsc::new(),
            delay: crate::primitive::DelayLine::new(max_delay),
            phase_inc: rate / sample_rate,
            base_delay,
            depth,
            wet_mix: mix,
            dry_mix: 1.0 - mix,
        }
    }

    /// Create from match result.
    pub fn from_match(m: &MatchResult, sample_rate: f32) -> Self {
        let rate = m.get_param(0, "rate").unwrap_or(0.8);
        let (base_delay, depth) = m.get_modulation(0, 1).unwrap_or((880.0, 220.0)); // ~20ms, ~5ms at 44.1kHz
        let mix = m.get_param(2, "mix").unwrap_or(0.5);

        let max_delay = (base_delay + depth * 2.0) as usize + 1;

        Self {
            lfo: crate::primitive::PhaseOsc::new(),
            delay: crate::primitive::DelayLine::new(max_delay),
            phase_inc: rate / sample_rate,
            base_delay,
            depth,
            wet_mix: mix,
            dry_mix: 1.0 - mix,
        }
    }
}

impl crate::graph::AudioNode for ChorusOptimized {
    #[inline]
    fn process(&mut self, input: f32, _ctx: &crate::graph::AudioContext) -> f32 {
        let lfo_out = self.lfo.sine();
        self.lfo.advance(self.phase_inc);

        let delay_time = self.base_delay + lfo_out * self.depth;
        self.delay.write(input);
        let wet = self.delay.read_interp(delay_time);

        // Mix dry and wet (pre-computed factors)
        input * self.dry_mix + wet * self.wet_mix
    }

    fn reset(&mut self) {
        self.lfo.reset();
        self.delay.clear();
    }
}

// ============================================================================
// Default Patterns
// ============================================================================

/// Returns the default set of effect patterns.
pub fn default_patterns() -> Vec<Pattern> {
    vec![tremolo_pattern(), flanger_pattern(), chorus_pattern()]
}

/// Pattern for tremolo: LFO modulating an affine (gain) node.
fn tremolo_pattern() -> Pattern {
    Pattern {
        name: "tremolo",
        required: fingerprint!(Lfo: 1, Affine: 1),
        structure: PatternStructure {
            nodes: vec![
                PatternNode {
                    node_type: NodeType::Lfo,
                    constraints: vec![],
                },
                PatternNode {
                    node_type: NodeType::Affine,
                    constraints: vec![],
                },
            ],
            audio_wires: vec![],               // LFO doesn't send audio to Affine
            param_wires: vec![(0, 1, "gain")], // LFO modulates Affine's gain param
            external_inputs: vec![1],          // External audio enters at Affine
            external_outputs: vec![1],         // Audio leaves from Affine
        },
        build: |m| Box::new(TremoloOptimized::from_match(m, 44100.0)),
        priority: 0,
    }
}

/// Pattern for flanger: LFO modulating delay time (no mixer).
fn flanger_pattern() -> Pattern {
    Pattern {
        name: "flanger",
        required: fingerprint!(Lfo: 1, Delay: 1),
        structure: PatternStructure {
            nodes: vec![
                PatternNode {
                    node_type: NodeType::Lfo,
                    constraints: vec![],
                },
                PatternNode {
                    node_type: NodeType::Delay,
                    constraints: vec![],
                },
            ],
            audio_wires: vec![],               // LFO doesn't send audio to Delay
            param_wires: vec![(0, 1, "time")], // LFO modulates Delay's time param
            external_inputs: vec![1],          // External audio enters at Delay
            external_outputs: vec![1],         // Audio leaves from Delay
        },
        build: |m| Box::new(FlangerOptimized::from_match(m, 44100.0)),
        priority: 0,
    }
}

/// Pattern for chorus: LFO modulating delay time with mixer.
fn chorus_pattern() -> Pattern {
    Pattern {
        name: "chorus",
        required: fingerprint!(Lfo: 1, Delay: 1, Mix: 1),
        structure: PatternStructure {
            nodes: vec![
                PatternNode {
                    node_type: NodeType::Lfo,
                    constraints: vec![],
                },
                PatternNode {
                    node_type: NodeType::Delay,
                    constraints: vec![],
                },
                PatternNode {
                    node_type: NodeType::Mix,
                    constraints: vec![],
                },
            ],
            audio_wires: vec![(1, 2)],         // Delay → Mixer
            param_wires: vec![(0, 1, "time")], // LFO modulates Delay's time param
            external_inputs: vec![1],          // External audio enters at Delay
            external_outputs: vec![2],         // Audio leaves from Mixer
        },
        build: |m| Box::new(ChorusOptimized::from_match(m, 44100.0)),
        priority: 10, // Higher priority than flanger (more specific)
    }
}

// ============================================================================
// Graph Optimization Passes
// ============================================================================

// Re-export AffineNode from graph module
pub use crate::graph::AffineNode;

/// Represents an affine operation for chain detection.
#[derive(Debug, Clone, Copy)]
enum AffineOp {
    /// Affine transform: output = input * gain + offset
    Affine { gain: f32, offset: f32 },
}

impl AffineOp {
    /// Convert to an AffineNode.
    fn to_affine(self) -> AffineNode {
        match self {
            AffineOp::Affine { gain, offset } => AffineNode::new(gain, offset),
        }
    }

    /// Compose two affine operations.
    fn then(self, other: Self) -> Self {
        let AffineOp::Affine {
            gain: g1,
            offset: o1,
        } = self;
        let AffineOp::Affine {
            gain: g2,
            offset: o2,
        } = other;
        // (x * g1 + o1) * g2 + o2 = x * (g1*g2) + (o1*g2 + o2)
        AffineOp::Affine {
            gain: g1 * g2,
            offset: o1 * g2 + o2,
        }
    }
}

/// Try to interpret a node as an affine operation.
fn node_as_affine(graph: &AudioGraph, idx: NodeIndex) -> Option<AffineOp> {
    let node_type = graph.node_type(idx)?;
    match node_type {
        NodeType::Affine => {
            // AffineNode stores gain at param 0, offset at param 1
            let gain = graph.node_param_value(idx, 0).unwrap_or(1.0);
            let offset = graph.node_param_value(idx, 1).unwrap_or(0.0);
            Some(AffineOp::Affine { gain, offset })
        }
        _ => None,
    }
}

/// Fuse chains of affine operations (Gain, Offset, PassThrough) into single AffineNode.
///
/// This optimization pass finds linear chains of affine nodes and replaces them
/// with a single fused node, reducing node count and eliminating intermediate storage.
///
/// # Example
///
/// ```text
/// Input -> Gain(0.5) -> Offset(1.0) -> Gain(2.0) -> Output
/// ```
/// Becomes:
/// ```text
/// Input -> AffineNode { gain: 1.0, offset: 2.0 } -> Output
/// // Because: ((x * 0.5) + 1.0) * 2.0 = x * 1.0 + 2.0
/// ```
pub fn fuse_affine_chains(graph: &mut AudioGraph) -> usize {
    let mut fused_count = 0;

    loop {
        // Find a chain to fuse
        let chain = find_affine_chain(graph);
        if chain.is_empty() || chain.len() < 2 {
            break;
        }

        // Compute the fused affine transform
        let mut combined = AffineNode::identity();
        for &idx in &chain {
            if let Some(op) = node_as_affine(graph, idx) {
                combined = combined.then(op.to_affine());
            }
        }

        // Skip if the result is identity (will be handled by identity elimination)
        if combined.is_identity() && chain.len() == 1 {
            break;
        }

        // Replace chain with fused node
        replace_affine_chain(graph, &chain, combined);
        fused_count += chain.len() - 1; // We reduced N nodes to 1
    }

    fused_count
}

/// Find a chain of affine nodes with single in/out connections.
fn find_affine_chain(graph: &AudioGraph) -> Vec<NodeIndex> {
    let node_count = graph.node_count();

    // Build adjacency info
    let mut in_degree = vec![0usize; node_count];
    let mut out_degree = vec![0usize; node_count];
    let mut successor = vec![None; node_count];
    let mut predecessor = vec![None; node_count];

    for wire in graph.audio_wires() {
        if wire.from < node_count && wire.to < node_count {
            out_degree[wire.from] += 1;
            in_degree[wire.to] += 1;
            successor[wire.from] = Some(wire.to);
            predecessor[wire.to] = Some(wire.from);
        }
    }

    // Find start of a chain (affine node with in_degree <= 1, followed by another affine)
    for start in 0..node_count {
        if node_as_affine(graph, start).is_none() {
            continue;
        }
        if out_degree[start] != 1 {
            continue;
        }

        let next = match successor[start] {
            Some(n) => n,
            None => continue,
        };

        if node_as_affine(graph, next).is_none() {
            continue;
        }

        // Found potential chain start, extend it
        let mut chain = vec![start];
        let mut current = next;

        while node_as_affine(graph, current).is_some()
            && in_degree[current] == 1
            && out_degree[current] <= 1
        {
            chain.push(current);
            if out_degree[current] == 0 {
                break;
            }
            current = match successor[current] {
                Some(n) => n,
                None => break,
            };
        }

        if chain.len() >= 2 {
            return chain;
        }
    }

    Vec::new()
}

/// Replace a chain of nodes with a single AffineNode.
fn replace_affine_chain(graph: &mut AudioGraph, chain: &[NodeIndex], affine: AffineNode) {
    if chain.is_empty() {
        return;
    }

    let first = chain[0];
    let last = chain[chain.len() - 1];

    // Add the new fused node
    let new_node = graph.add(affine);

    // Rewire inputs: anything that fed the first node should feed the new node
    let wires: Vec<_> = graph.audio_wires().to_vec();
    for wire in &wires {
        if wire.to == first && !chain.contains(&wire.from) {
            graph.connect(wire.from, new_node);
        }
    }

    // Rewire outputs: anything the last node fed should be fed by the new node
    for wire in &wires {
        if wire.from == last && !chain.contains(&wire.to) {
            graph.connect(new_node, wire.to);
        }
    }

    // Update input/output node references
    if graph.input_node() == Some(first) {
        graph.connect_input(new_node);
    }
    if graph.output_node() == Some(last) {
        graph.set_output(new_node);
    }

    // Remove chain nodes (in reverse order to preserve indices)
    let mut to_remove: Vec<NodeIndex> = chain.to_vec();
    to_remove.sort_by(|a, b| b.cmp(a));
    for idx in to_remove {
        graph.remove_node(idx);
    }
}

/// Remove identity nodes (Gain(1.0), Offset(0.0), PassThrough) from the graph.
///
/// These nodes don't change the signal and can be safely removed by rewiring
/// their inputs directly to their outputs.
pub fn eliminate_identities(graph: &mut AudioGraph) -> usize {
    let mut removed = 0;

    loop {
        let identity = find_identity_node(graph);
        if identity.is_none() {
            break;
        }
        let idx = identity.unwrap();

        // Rewire: connect predecessors directly to successors
        let wires: Vec<_> = graph.audio_wires().to_vec();
        let predecessors: Vec<NodeIndex> = wires
            .iter()
            .filter(|w| w.to == idx)
            .map(|w| w.from)
            .collect();
        let successors: Vec<NodeIndex> = wires
            .iter()
            .filter(|w| w.from == idx)
            .map(|w| w.to)
            .collect();

        // Connect each predecessor to each successor
        for &pred in &predecessors {
            for &succ in &successors {
                graph.connect(pred, succ);
            }
        }

        // Update input/output references
        if graph.input_node() == Some(idx) {
            if let Some(&succ) = successors.first() {
                graph.connect_input(succ);
            }
        }
        if graph.output_node() == Some(idx) {
            if let Some(&pred) = predecessors.first() {
                graph.set_output(pred);
            }
        }

        graph.remove_node(idx);
        removed += 1;
    }

    removed
}

/// Find a node that is an identity operation.
fn find_identity_node(graph: &AudioGraph) -> Option<NodeIndex> {
    for idx in 0..graph.node_count() {
        let node_type = graph.node_type(idx);
        let is_identity = match node_type {
            Some(NodeType::Affine) => {
                // Check if AffineNode is identity (gain ~= 1, offset ~= 0)
                let gain = graph.node_param_value(idx, 0).unwrap_or(1.0);
                let offset = graph.node_param_value(idx, 1).unwrap_or(0.0);
                (gain - 1.0).abs() < 1e-10 && offset.abs() < 1e-10
            }
            _ => false,
        };

        if is_identity {
            return Some(idx);
        }
    }
    None
}

/// Remove nodes that are not connected to the output.
///
/// A node is "live" if it's reachable from the output via audio wires,
/// or if it modulates a parameter of a live node.
///
/// # Example
///
/// ```text
/// Before:
///   A -> B -> Output
///   C -> D (disconnected)
///
/// After:
///   A -> B -> Output
///   (C and D removed)
/// ```
pub fn eliminate_dead_nodes(graph: &mut AudioGraph) -> usize {
    let output = match graph.output_node() {
        Some(out) => out,
        None => return 0, // No output, nothing to do
    };

    // Find all live nodes by walking backwards from output
    let mut live = vec![false; graph.node_count()];
    let mut worklist = vec![output];

    // Also mark input node as live if it exists
    if let Some(input) = graph.input_node() {
        worklist.push(input);
    }

    let audio_wires: Vec<_> = graph.audio_wires().to_vec();
    let param_wires: Vec<_> = graph.param_wires().to_vec();

    while let Some(idx) = worklist.pop() {
        if idx >= live.len() || live[idx] {
            continue;
        }
        live[idx] = true;

        // Add predecessors (nodes that feed into this one)
        for wire in &audio_wires {
            if wire.to == idx && !live[wire.from] {
                worklist.push(wire.from);
            }
        }

        // Add modulators (nodes that modulate this node's params)
        for wire in &param_wires {
            if wire.to == idx && !live[wire.from] {
                worklist.push(wire.from);
            }
        }
    }

    // Collect dead nodes (in reverse order to preserve indices during removal)
    let mut dead: Vec<NodeIndex> = (0..graph.node_count()).filter(|&i| !live[i]).collect();
    dead.sort_by(|a, b| b.cmp(a)); // Reverse order

    let removed = dead.len();
    for idx in dead {
        graph.remove_node(idx);
    }

    removed
}

/// Fold constant values through affine operations.
///
/// When a `Constant(a)` feeds into an `AffineNode { gain, offset }`,
/// the result is a known constant `a * gain + offset`, so we can replace
/// both nodes with a single `Constant`.
///
/// # Example
///
/// ```text
/// Before:
///   Constant(2.0) -> Gain(3.0) -> Offset(1.0) -> Output
///
/// After:
///   Constant(7.0) -> Output
///   // Because: 2.0 * 3.0 + 1.0 = 7.0
/// ```
pub fn fold_constants(graph: &mut AudioGraph) -> usize {
    let mut folded = 0;

    loop {
        let fold = find_constant_affine_pair(graph);
        if fold.is_none() {
            break;
        }
        let (const_idx, affine_idx, result_value) = fold.unwrap();

        // Replace the affine node with the folded constant
        let new_const = graph.add(crate::graph::Constant(result_value));

        // Rewire: anything the affine fed should now be fed by new constant
        let wires: Vec<_> = graph.audio_wires().to_vec();
        for wire in &wires {
            if wire.from == affine_idx {
                graph.connect(new_const, wire.to);
            }
        }

        // Update output if needed
        if graph.output_node() == Some(affine_idx) {
            graph.set_output(new_const);
        }

        // Remove both old nodes (affine first since it has higher index typically)
        let mut to_remove = vec![const_idx, affine_idx];
        to_remove.sort_by(|a, b| b.cmp(a)); // Reverse order
        for idx in to_remove {
            graph.remove_node(idx);
        }

        folded += 1;
    }

    folded
}

/// Find a Constant -> Affine pair that can be folded.
fn find_constant_affine_pair(graph: &AudioGraph) -> Option<(NodeIndex, NodeIndex, f32)> {
    let wires = graph.audio_wires();

    for wire in wires {
        // Check if source is a Constant
        let src_type = graph.node_type(wire.from);
        if src_type != Some(NodeType::Constant) {
            continue;
        }

        // Check if dest is an Affine
        let dst_type = graph.node_type(wire.to);
        if dst_type != Some(NodeType::Affine) {
            continue;
        }

        // Make sure the affine has only this one input (no other audio inputs)
        let input_count = wires.iter().filter(|w| w.to == wire.to).count();
        if input_count != 1 {
            continue;
        }

        // Make sure the constant has only this one output (not used elsewhere)
        let output_count = wires.iter().filter(|w| w.from == wire.from).count();
        if output_count != 1 {
            continue;
        }

        // Get the constant value and affine params
        let const_value = graph.node_param_value(wire.from, 0).unwrap_or(0.0);
        let gain = graph.node_param_value(wire.to, 0).unwrap_or(1.0);
        let offset = graph.node_param_value(wire.to, 1).unwrap_or(0.0);

        // Compute folded value
        let result = const_value * gain + offset;

        return Some((wire.from, wire.to, result));
    }

    None
}

/// Propagate constant values through chains of affine operations.
///
/// This is more aggressive than `fold_constants` - it tracks which nodes
/// have known constant outputs and propagates through longer chains.
///
/// # Example
///
/// ```text
/// Before:
///   Constant(1.0) -> Gain(2.0) -> Offset(3.0) -> Gain(4.0) -> Output
///
/// After:
///   Constant(20.0) -> Output
///   // Because: ((1.0 * 2.0) + 3.0) * 4.0 = 20.0
/// ```
pub fn propagate_constants(graph: &mut AudioGraph) -> usize {
    // This pass combines constant folding with affine chain fusion
    // Run both passes in a loop until no more changes
    let mut propagated = 0;

    loop {
        let folded = fold_constants(graph);
        let fused = fuse_affine_chains(graph);

        if folded == 0 && fused == 0 {
            break;
        }
        propagated += folded + fused;
    }

    propagated
}

/// Merge consecutive delay nodes with zero feedback.
///
/// When two `DelayNode` instances with `feedback == 0` are connected in series,
/// they can be merged into a single delay with combined time.
///
/// # Example
///
/// ```text
/// Before:
///   Input -> Delay(100 samples) -> Delay(50 samples) -> Output
///
/// After:
///   Input -> Delay(150 samples) -> Output
/// ```
///
/// # Limitations
///
/// - Only merges delays with zero feedback (feedback creates recurrence)
/// - Uses a conservative max buffer size (sum of both delay times + margin)
pub fn merge_delays(graph: &mut AudioGraph) -> usize {
    let mut merged = 0;

    loop {
        let pair = find_delay_pair(graph);
        if pair.is_none() {
            break;
        }
        let (first_idx, second_idx, combined_time) = pair.unwrap();

        // Create merged delay with buffer large enough for combined time
        let max_samples = (combined_time * 1.5) as usize + 100; // Add margin
        let mut new_delay = crate::primitive::DelayNode::new(max_samples);
        new_delay.set_time(combined_time);

        let new_node = graph.add(new_delay);

        // Rewire: inputs to first -> new node
        let wires: Vec<_> = graph.audio_wires().to_vec();
        for wire in &wires {
            if wire.to == first_idx && wire.from != second_idx {
                graph.connect(wire.from, new_node);
            }
        }

        // Rewire: outputs from second -> new node
        for wire in &wires {
            if wire.from == second_idx && wire.to != first_idx {
                graph.connect(new_node, wire.to);
            }
        }

        // Update input/output references
        if graph.input_node() == Some(first_idx) {
            graph.connect_input(new_node);
        }
        if graph.output_node() == Some(second_idx) {
            graph.set_output(new_node);
        }

        // Remove old nodes (in reverse order)
        let mut to_remove = vec![first_idx, second_idx];
        to_remove.sort_by(|a, b| b.cmp(a));
        for idx in to_remove {
            graph.remove_node(idx);
        }

        merged += 1;
    }

    merged
}

/// Find a pair of consecutive delays with zero feedback.
fn find_delay_pair(graph: &AudioGraph) -> Option<(NodeIndex, NodeIndex, f32)> {
    let wires = graph.audio_wires();

    for wire in wires {
        // Check if both nodes are delays
        if graph.node_type(wire.from) != Some(NodeType::Delay) {
            continue;
        }
        if graph.node_type(wire.to) != Some(NodeType::Delay) {
            continue;
        }

        // Check that first delay only outputs to second (single out-edge)
        let out_count = wires.iter().filter(|w| w.from == wire.from).count();
        if out_count != 1 {
            continue;
        }

        // Check that second delay only receives from first (single in-edge)
        let in_count = wires.iter().filter(|w| w.to == wire.to).count();
        if in_count != 1 {
            continue;
        }

        // Check feedback is zero for both
        // PARAM_FEEDBACK = 1 for DelayNode
        let feedback1 = graph.node_param_value(wire.from, 1).unwrap_or(0.0);
        let feedback2 = graph.node_param_value(wire.to, 1).unwrap_or(0.0);

        if feedback1.abs() > 1e-6 || feedback2.abs() > 1e-6 {
            continue;
        }

        // Get delay times (PARAM_TIME = 0)
        let time1 = graph.node_param_value(wire.from, 0).unwrap_or(0.0);
        let time2 = graph.node_param_value(wire.to, 0).unwrap_or(0.0);

        return Some((wire.from, wire.to, time1 + time2));
    }

    None
}

/// Run all graph optimization passes.
///
/// Returns the total number of nodes removed/fused.
pub fn run_optimization_passes(graph: &mut AudioGraph) -> usize {
    let mut total = 0;

    // Run passes until no more changes
    loop {
        let fused = fuse_affine_chains(graph);
        let folded = fold_constants(graph);
        let delays = merge_delays(graph);
        let identities = eliminate_identities(graph);
        let dead = eliminate_dead_nodes(graph);

        if fused == 0 && folded == 0 && delays == 0 && identities == 0 && dead == 0 {
            break;
        }
        total += fused + folded + delays + identities + dead;
    }

    total
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
