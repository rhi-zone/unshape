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
//! use rhizome_resin_audio::optimize::{optimize_graph, default_patterns};
//! use rhizome_resin_audio::graph::AudioGraph;
//!
//! let mut graph = AudioGraph::new();
//! // ... build graph with LFO modulating gain (tremolo pattern) ...
//!
//! // Optimize: replaces LFO+Gain subgraph with TremoloOptimized
//! optimize_graph(&mut graph, &default_patterns());
//! ```

use crate::graph::{AudioGraph, AudioNode, NodeIndex};
use std::any::TypeId;
use std::collections::HashMap;

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
    Gain,
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
    Offset,
    Constant,
    PassThrough,
    Silence,

    // Unknown/custom nodes
    Unknown,
}

impl NodeType {
    /// Total number of known node types.
    pub const COUNT: usize = 20;

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
    map.insert(TypeId::of::<crate::primitive::GainNode>(), NodeType::Gain);
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
    map.insert(TypeId::of::<crate::graph::Offset>(), NodeType::Offset);
    map.insert(TypeId::of::<crate::graph::Constant>(), NodeType::Constant);
    map.insert(
        TypeId::of::<crate::graph::PassThrough>(),
        NodeType::PassThrough,
    );
    map.insert(TypeId::of::<crate::graph::Silence>(), NodeType::Silence);
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
    mix: f32,
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
            mix,
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
            mix,
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

        // Mix dry and wet (matches ModulatedDelay behavior)
        input * (1.0 - self.mix) + delayed * self.mix
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
    mix: f32,
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
            mix,
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
            mix,
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

        // Mix dry and wet
        input * (1.0 - self.mix) + wet * self.mix
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

/// Pattern for tremolo: LFO modulating a gain node.
fn tremolo_pattern() -> Pattern {
    Pattern {
        name: "tremolo",
        required: fingerprint!(Lfo: 1, Gain: 1),
        structure: PatternStructure {
            nodes: vec![
                PatternNode {
                    node_type: NodeType::Lfo,
                    constraints: vec![],
                },
                PatternNode {
                    node_type: NodeType::Gain,
                    constraints: vec![],
                },
            ],
            audio_wires: vec![],               // LFO doesn't send audio to Gain
            param_wires: vec![(0, 1, "gain")], // LFO modulates Gain's gain param
            external_inputs: vec![1],          // External audio enters at Gain
            external_outputs: vec![1],         // Audio leaves from Gain
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fingerprint_contains() {
        let mut graph_fp = GraphFingerprint::new();
        graph_fp.add(NodeType::Lfo);
        graph_fp.add(NodeType::Lfo);
        graph_fp.add(NodeType::Gain);
        graph_fp.add(NodeType::Delay);

        // Pattern needs 1 LFO and 1 Gain - should match
        let pattern_fp = fingerprint!(Lfo: 1, Gain: 1);
        assert!(graph_fp.contains(&pattern_fp));

        // Pattern needs 2 LFOs and 1 Gain - should match
        let pattern_fp2 = fingerprint!(Lfo: 2, Gain: 1);
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
        let fp = fingerprint!(Lfo: 2, Gain: 1, Delay: 3);

        assert_eq!(fp.count(NodeType::Lfo), 2);
        assert_eq!(fp.count(NodeType::Gain), 1);
        assert_eq!(fp.count(NodeType::Delay), 3);
        assert_eq!(fp.count(NodeType::Allpass), 0);
    }

    #[test]
    fn test_tremolo_pattern_match() {
        use crate::graph::{AudioContext, AudioGraph};
        use crate::primitive::{GainNode, LfoNode};

        // Build a tremolo graph manually
        let mut graph = AudioGraph::new();
        let lfo = graph.add(LfoNode::with_freq(5.0, 44100.0));
        let gain = graph.add(GainNode::new(1.0));

        graph.connect_input(gain);
        graph.set_output(gain);
        graph.modulate(lfo, gain, GainNode::PARAM_GAIN, 0.5, 0.5);

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
        use crate::graph::{AudioContext, AudioGraph};
        use crate::primitive::{GainNode, LfoNode};

        // Build a tremolo graph
        let mut graph = AudioGraph::new();
        let lfo = graph.add(LfoNode::with_freq(5.0, 44100.0));
        let gain = graph.add(GainNode::new(1.0));

        graph.connect_input(gain);
        graph.set_output(gain);
        graph.modulate(lfo, gain, GainNode::PARAM_GAIN, 0.5, 0.5);

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
        use crate::graph::{AudioContext, AudioGraph};
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
        use crate::graph::{AudioContext, AudioGraph};
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
}
