//! Code generation for static audio graphs.
//!
//! This crate generates optimized Rust code from serialized graph descriptions.
//! Use this in your `build.rs` to compile audio graphs into efficient static structs
//! that eliminate dynamic dispatch and wire iteration overhead.
//!
//! # Usage
//!
//! In your `build.rs`:
//!
//! ```ignore
//! use unshape_audio_codegen::{SerialAudioGraph, SerialAudioNode, SerialParamWire, generate_effect};
//!
//! fn main() {
//!     let graph = SerialAudioGraph {
//!         nodes: vec![
//!             SerialAudioNode::Lfo { rate: 5.0 },
//!             SerialAudioNode::Gain { gain: 1.0 },
//!         ],
//!         audio_wires: vec![],
//!         param_wires: vec![
//!             SerialParamWire { from: 0, to: 1, param: 0, base: 0.5, scale: 0.5 },
//!         ],
//!         input_node: Some(1),
//!         output_node: Some(1),
//!     };
//!
//!     let code = generate_effect(&graph, "MyTremolo");
//!
//!     let out_dir = std::env::var("OUT_DIR").unwrap();
//!     std::fs::write(format!("{out_dir}/my_tremolo.rs"), code).unwrap();
//! }
//! ```
//!
//! Then include in your crate:
//!
//! ```ignore
//! mod generated {
//!     include!(concat!(env!("OUT_DIR"), "/my_tremolo.rs"));
//! }
//! use generated::MyTremolo;
//! ```

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use std::collections::{HashMap, HashSet, VecDeque};

// ============================================================================
// Serializable Graph Types
// ============================================================================

/// Serializable representation of an audio graph for codegen.
///
/// This is the input format for code generation. Build this struct
/// in your `build.rs` and pass it to [`generate_effect`].
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct SerialAudioGraph {
    /// Nodes in the graph (index = node ID).
    pub nodes: Vec<SerialAudioNode>,
    /// Audio signal routing: (from_node, to_node).
    pub audio_wires: Vec<(usize, usize)>,
    /// Parameter modulation routing.
    pub param_wires: Vec<SerialParamWire>,
    /// Which node receives external input (if any).
    pub input_node: Option<usize>,
    /// Which node provides output.
    pub output_node: Option<usize>,
}

/// Parameter modulation wire.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct SerialParamWire {
    /// Source node (modulator).
    pub from: usize,
    /// Target node.
    pub to: usize,
    /// Parameter index on target.
    pub param: usize,
    /// Base value (added to modulation).
    pub base: f32,
    /// Scale factor for modulation.
    pub scale: f32,
}

/// Serializable audio node types.
///
/// Each variant contains the parameters needed to construct that node type.
/// The codegen uses these to generate field declarations and initialization code.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(tag = "type"))]
pub enum SerialAudioNode {
    /// LFO (low-frequency oscillator) for modulation.
    Lfo {
        /// Frequency in Hz.
        rate: f32,
    },
    /// Gain/amplitude control.
    Gain {
        /// Initial gain value.
        gain: f32,
    },
    /// Delay line with interpolation.
    Delay {
        /// Maximum delay in samples.
        max_samples: usize,
        /// Feedback amount (0.0 = no feedback).
        feedback: f32,
        /// Wet/dry mix (0.0 = dry only, 1.0 = wet only).
        mix: f32,
    },
    /// Wet/dry mixer.
    Mix {
        /// Mix amount (0 = dry, 1 = wet).
        mix: f32,
    },
    /// Envelope follower.
    Envelope {
        /// Attack time in seconds.
        attack: f32,
        /// Release time in seconds.
        release: f32,
    },
    /// First-order allpass filter.
    Allpass {
        /// Filter coefficient.
        coefficient: f32,
    },
    /// Passthrough (for input/output markers).
    Passthrough,
}

impl SerialAudioNode {
    /// Returns the Rust type for this node's state.
    fn field_type(&self) -> &'static str {
        match self {
            SerialAudioNode::Lfo { .. } => "unshape_audio::primitive::PhaseOsc",
            SerialAudioNode::Gain { .. } => "f32",
            SerialAudioNode::Delay { .. } => "unshape_audio::primitive::DelayLine<true>",
            SerialAudioNode::Mix { .. } => "f32",
            SerialAudioNode::Envelope { .. } => "unshape_audio::primitive::EnvelopeFollower",
            SerialAudioNode::Allpass { .. } => "unshape_audio::primitive::Allpass1",
            SerialAudioNode::Passthrough => "()",
        }
    }

    /// Returns whether this node needs a field in the generated struct.
    fn needs_field(&self) -> bool {
        !matches!(self, SerialAudioNode::Passthrough)
    }

    /// Returns the initialization expression for this node.
    fn init_code(&self, sample_rate: &str) -> String {
        match self {
            SerialAudioNode::Lfo { .. } => {
                "unshape_audio::primitive::PhaseOsc::new()".to_string()
            }
            SerialAudioNode::Gain { gain } => format!("{gain}_f32"),
            SerialAudioNode::Delay { max_samples, .. } => {
                format!("unshape_audio::primitive::DelayLine::new({max_samples})")
            }
            SerialAudioNode::Mix { mix } => format!("{mix}_f32"),
            SerialAudioNode::Envelope { attack, release } => {
                format!(
                    "unshape_audio::primitive::EnvelopeFollower::new({sample_rate}, {attack}, {release})"
                )
            }
            SerialAudioNode::Allpass { coefficient } => {
                format!("unshape_audio::primitive::Allpass1::new({coefficient})")
            }
            SerialAudioNode::Passthrough => "()".to_string(),
        }
    }
}

// ============================================================================
// Code Generation
// ============================================================================

/// Returns the common header that should be included once at the top of generated files.
pub fn generate_header() -> String {
    let mut code = String::new();
    code.push_str("// Generated by unshape-audio-codegen\n");
    code.push_str("// Do not edit manually\n\n");
    code.push_str("use unshape_audio::graph::{AudioContext, AudioNode};\n\n");
    code
}

/// Generates optimized Rust code for an audio effect.
///
/// The generated code includes:
/// - A struct with fields for each node's state
/// - A `new(sample_rate: f32)` constructor
/// - An `AudioNode` implementation with inlined processing
///
/// Note: This does NOT include the header imports. Call [`generate_header`] once
/// at the top of your file if generating multiple effects.
pub fn generate_effect(graph: &SerialAudioGraph, name: &str) -> String {
    let mut code = String::new();

    // Analyze graph
    let analysis = analyze_graph(graph);

    // Generate struct
    code.push_str(&generate_struct(graph, name, &analysis));
    code.push('\n');

    // Generate impl
    code.push_str(&generate_impl(graph, name, &analysis));
    code.push('\n');

    // Generate AudioNode impl
    code.push_str(&generate_audio_node_impl(graph, name, &analysis));

    code
}

/// Generates a complete, standalone effect with header.
pub fn generate_effect_standalone(graph: &SerialAudioGraph, name: &str) -> String {
    let mut code = generate_header();
    code.push_str(&generate_effect(graph, name));
    code
}

// ============================================================================
// Graph Analysis
// ============================================================================

/// Analysis results for code generation.
struct GraphAnalysis {
    /// Processing order (topological sort based on dependencies).
    process_order: Vec<usize>,
    /// For each node, which nodes provide audio input to it.
    audio_inputs: HashMap<usize, Vec<usize>>,
    /// For each node, param modulations: (param_idx, source_node, base, scale).
    param_mods: HashMap<usize, Vec<(usize, usize, f32, f32)>>,
    /// LFO nodes with their phase increment field names.
    lfo_phase_incs: HashMap<usize, String>,
}

fn analyze_graph(graph: &SerialAudioGraph) -> GraphAnalysis {
    // Build adjacency info
    let mut audio_inputs: HashMap<usize, Vec<usize>> = HashMap::new();
    for &(from, to) in &graph.audio_wires {
        audio_inputs.entry(to).or_default().push(from);
    }

    // Build param modulation info
    let mut param_mods: HashMap<usize, Vec<(usize, usize, f32, f32)>> = HashMap::new();
    for wire in &graph.param_wires {
        param_mods
            .entry(wire.to)
            .or_default()
            .push((wire.param, wire.from, wire.base, wire.scale));
    }

    // Topological sort
    let process_order = topological_sort(graph);

    // Find LFO nodes
    let mut lfo_phase_incs = HashMap::new();
    for (i, node) in graph.nodes.iter().enumerate() {
        if matches!(node, SerialAudioNode::Lfo { .. }) {
            lfo_phase_incs.insert(i, format!("node_{i}_phase_inc"));
        }
    }

    GraphAnalysis {
        process_order,
        audio_inputs,
        param_mods,
        lfo_phase_incs,
    }
}

/// Topological sort of nodes based on audio wires and param wires.
fn topological_sort(graph: &SerialAudioGraph) -> Vec<usize> {
    let n = graph.nodes.len();
    if n == 0 {
        return vec![];
    }

    // Build dependency graph (node -> nodes it depends on)
    let mut dependencies: HashMap<usize, HashSet<usize>> = HashMap::new();
    for i in 0..n {
        dependencies.insert(i, HashSet::new());
    }

    // Audio wires: to depends on from
    for &(from, to) in &graph.audio_wires {
        dependencies.get_mut(&to).unwrap().insert(from);
    }

    // Param wires: to depends on from (modulator must run first)
    for wire in &graph.param_wires {
        dependencies.get_mut(&wire.to).unwrap().insert(wire.from);
    }

    // Kahn's algorithm
    let mut in_degree: HashMap<usize, usize> = HashMap::new();
    for i in 0..n {
        in_degree.insert(i, dependencies[&i].len());
    }

    let mut queue: VecDeque<usize> = VecDeque::new();
    for i in 0..n {
        if in_degree[&i] == 0 {
            queue.push_back(i);
        }
    }

    let mut result = Vec::with_capacity(n);
    while let Some(node) = queue.pop_front() {
        result.push(node);

        // Find nodes that depend on this one
        for i in 0..n {
            if dependencies[&i].contains(&node) {
                let deg = in_degree.get_mut(&i).unwrap();
                *deg -= 1;
                if *deg == 0 {
                    queue.push_back(i);
                }
            }
        }
    }

    // If we didn't get all nodes, there's a cycle - fall back to index order
    if result.len() != n {
        (0..n).collect()
    } else {
        result
    }
}

// ============================================================================
// Struct Generation
// ============================================================================

fn generate_struct(graph: &SerialAudioGraph, name: &str, analysis: &GraphAnalysis) -> String {
    let mut code = format!("pub struct {name} {{\n");

    for (i, node) in graph.nodes.iter().enumerate() {
        if node.needs_field() {
            let field_type = node.field_type();
            code.push_str(&format!("    node_{i}: {field_type},\n"));

            // Delay nodes get additional fields for feedback/mix
            if let SerialAudioNode::Delay { feedback, .. } = node {
                if *feedback != 0.0 {
                    code.push_str(&format!("    node_{i}_feedback: f32,\n"));
                }
                code.push_str(&format!("    node_{i}_wet_mix: f32,\n"));
                code.push_str(&format!("    node_{i}_dry_mix: f32,\n"));
            }
        }
    }

    // Phase increment fields for LFOs
    for (idx, field_name) in &analysis.lfo_phase_incs {
        let _ = idx;
        code.push_str(&format!("    {field_name}: f32,\n"));
    }

    // Modulation constants
    for (i, wire) in graph.param_wires.iter().enumerate() {
        let _ = wire;
        code.push_str(&format!("    mod_{i}_base: f32,\n"));
        code.push_str(&format!("    mod_{i}_scale: f32,\n"));
    }

    code.push_str("}\n");
    code
}

// ============================================================================
// Impl Generation
// ============================================================================

fn generate_impl(graph: &SerialAudioGraph, name: &str, analysis: &GraphAnalysis) -> String {
    let mut code = format!("impl {name} {{\n");
    code.push_str("    pub fn new(sample_rate: f32) -> Self {\n");
    code.push_str("        let _ = sample_rate;\n");
    code.push_str("        Self {\n");

    for (i, node) in graph.nodes.iter().enumerate() {
        if node.needs_field() {
            let init = node.init_code("sample_rate");
            code.push_str(&format!("            node_{i}: {init},\n"));

            if let SerialAudioNode::Delay { feedback, mix, .. } = node {
                if *feedback != 0.0 {
                    code.push_str(&format!("            node_{i}_feedback: {feedback}_f32,\n"));
                }
                code.push_str(&format!("            node_{i}_wet_mix: {mix}_f32,\n"));
                let dry_mix = 1.0 - mix;
                code.push_str(&format!("            node_{i}_dry_mix: {dry_mix}_f32,\n"));
            }
        }
    }

    // Initialize phase increments
    for (idx, field_name) in &analysis.lfo_phase_incs {
        if let SerialAudioNode::Lfo { rate } = &graph.nodes[*idx] {
            code.push_str(&format!(
                "            {field_name}: {rate}_f32 / sample_rate,\n"
            ));
        }
    }

    // Initialize modulation constants
    for (i, wire) in graph.param_wires.iter().enumerate() {
        code.push_str(&format!("            mod_{i}_base: {}_f32,\n", wire.base));
        code.push_str(&format!("            mod_{i}_scale: {}_f32,\n", wire.scale));
    }

    code.push_str("        }\n");
    code.push_str("    }\n");
    code.push_str("}\n");
    code
}

// ============================================================================
// AudioNode Impl Generation (Generic Graph Compilation)
// ============================================================================

fn generate_audio_node_impl(
    graph: &SerialAudioGraph,
    name: &str,
    analysis: &GraphAnalysis,
) -> String {
    let mut code = format!("impl AudioNode for {name} {{\n");
    code.push_str("    #[inline]\n");
    code.push_str("    fn process(&mut self, input: f32, _ctx: &AudioContext) -> f32 {\n");

    // Generate processing code for each node in topological order
    for &idx in &analysis.process_order {
        code.push_str(&generate_node_process(graph, idx, analysis));
    }

    // Return output
    if let Some(output_idx) = graph.output_node {
        code.push_str(&format!("        node_{output_idx}_out\n"));
    } else {
        code.push_str("        input\n");
    }

    code.push_str("    }\n\n");

    // Reset method
    code.push_str("    fn reset(&mut self) {\n");
    for (i, node) in graph.nodes.iter().enumerate() {
        if node.needs_field() {
            match node {
                SerialAudioNode::Lfo { .. } => {
                    code.push_str(&format!("        self.node_{i}.reset();\n"));
                }
                SerialAudioNode::Delay { .. } => {
                    code.push_str(&format!("        self.node_{i}.clear();\n"));
                }
                SerialAudioNode::Envelope { .. } => {
                    code.push_str(&format!("        self.node_{i}.reset();\n"));
                }
                SerialAudioNode::Allpass { .. } => {
                    code.push_str(&format!("        self.node_{i}.reset();\n"));
                }
                _ => {}
            }
        }
    }
    code.push_str("    }\n");

    code.push_str("}\n");
    code
}

/// Generate processing code for a single node.
fn generate_node_process(graph: &SerialAudioGraph, idx: usize, analysis: &GraphAnalysis) -> String {
    let node = &graph.nodes[idx];
    let mut code = String::new();

    // Determine the audio input for this node
    let audio_input = if Some(idx) == graph.input_node {
        "input".to_string()
    } else if let Some(inputs) = analysis.audio_inputs.get(&idx) {
        if inputs.len() == 1 {
            format!("node_{}_out", inputs[0])
        } else if inputs.len() > 1 {
            // Multiple inputs - sum them
            let sum: Vec<String> = inputs.iter().map(|i| format!("node_{i}_out")).collect();
            sum.join(" + ")
        } else {
            "0.0_f32".to_string()
        }
    } else {
        "0.0_f32".to_string()
    };

    // Check for param modulations on this node
    let mods = analysis.param_mods.get(&idx);

    match node {
        SerialAudioNode::Lfo { .. } => {
            // LFO produces modulation signal
            code.push_str(&format!(
                "        let node_{idx}_out = self.node_{idx}.sine();\n"
            ));
            if let Some(phase_inc) = analysis.lfo_phase_incs.get(&idx) {
                code.push_str(&format!(
                    "        self.node_{idx}.advance(self.{phase_inc});\n"
                ));
            }
        }

        SerialAudioNode::Gain { .. } => {
            // Check if gain is modulated
            if let Some(mod_list) = mods {
                // Find the modulation for param 0 (gain)
                if let Some(&(_, src, _, _)) = mod_list.iter().find(|(p, _, _, _)| *p == 0) {
                    // Find which mod index this is
                    let mod_idx = graph
                        .param_wires
                        .iter()
                        .position(|w| w.from == src && w.to == idx)
                        .unwrap_or(0);
                    code.push_str(&format!(
                        "        let node_{idx}_gain = self.mod_{mod_idx}_base + node_{src}_out * self.mod_{mod_idx}_scale;\n"
                    ));
                    code.push_str(&format!(
                        "        let node_{idx}_out = {audio_input} * node_{idx}_gain;\n"
                    ));
                } else {
                    code.push_str(&format!(
                        "        let node_{idx}_out = {audio_input} * self.node_{idx};\n"
                    ));
                }
            } else {
                code.push_str(&format!(
                    "        let node_{idx}_out = {audio_input} * self.node_{idx};\n"
                ));
            }
        }

        SerialAudioNode::Delay { feedback, .. } => {
            // Check if delay time is modulated
            let delay_time = if let Some(mod_list) = mods {
                if let Some(&(_, src, _, _)) = mod_list.iter().find(|(p, _, _, _)| *p == 0) {
                    let mod_idx = graph
                        .param_wires
                        .iter()
                        .position(|w| w.from == src && w.to == idx)
                        .unwrap_or(0);
                    format!("self.mod_{mod_idx}_base + node_{src}_out * self.mod_{mod_idx}_scale")
                } else {
                    format!("self.node_{idx}.len() as f32 / 2.0")
                }
            } else {
                format!("self.node_{idx}.len() as f32 / 2.0")
            };

            code.push_str(&format!(
                "        let node_{idx}_delayed = self.node_{idx}.read_interp({delay_time});\n"
            ));

            // Write with optional feedback
            if *feedback != 0.0 {
                code.push_str(&format!(
                    "        self.node_{idx}.write({audio_input} + node_{idx}_delayed * self.node_{idx}_feedback);\n"
                ));
            } else {
                code.push_str(&format!("        self.node_{idx}.write({audio_input});\n"));
            }

            // Mix dry and wet
            code.push_str(&format!(
                "        let node_{idx}_out = {audio_input} * self.node_{idx}_dry_mix + node_{idx}_delayed * self.node_{idx}_wet_mix;\n"
            ));
        }

        SerialAudioNode::Mix { .. } => {
            // Mix node blends input with... what? Need to think about this.
            // For now, just pass through with mix applied
            code.push_str(&format!(
                "        let node_{idx}_out = {audio_input} * self.node_{idx};\n"
            ));
        }

        SerialAudioNode::Envelope { .. } => {
            code.push_str(&format!(
                "        let node_{idx}_out = self.node_{idx}.process({audio_input});\n"
            ));
        }

        SerialAudioNode::Allpass { .. } => {
            code.push_str(&format!(
                "        let node_{idx}_out = self.node_{idx}.process({audio_input});\n"
            ));
        }

        SerialAudioNode::Passthrough => {
            code.push_str(&format!("        let node_{idx}_out = {audio_input};\n"));
        }
    }

    code
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_topological_sort() {
        // LFO (0) -> modulates Gain (1), audio flows through Gain
        let graph = SerialAudioGraph {
            nodes: vec![
                SerialAudioNode::Lfo { rate: 5.0 },
                SerialAudioNode::Gain { gain: 1.0 },
            ],
            audio_wires: vec![],
            param_wires: vec![SerialParamWire {
                from: 0,
                to: 1,
                param: 0,
                base: 0.5,
                scale: 0.5,
            }],
            input_node: Some(1),
            output_node: Some(1),
        };

        let order = topological_sort(&graph);
        // LFO must come before Gain (because Gain depends on LFO for modulation)
        let lfo_pos = order.iter().position(|&x| x == 0).unwrap();
        let gain_pos = order.iter().position(|&x| x == 1).unwrap();
        assert!(lfo_pos < gain_pos);
    }

    #[test]
    fn test_tremolo_codegen() {
        let graph = SerialAudioGraph {
            nodes: vec![
                SerialAudioNode::Lfo { rate: 5.0 },
                SerialAudioNode::Gain { gain: 1.0 },
            ],
            audio_wires: vec![],
            param_wires: vec![SerialParamWire {
                from: 0,
                to: 1,
                param: 0,
                base: 0.5,
                scale: 0.5,
            }],
            input_node: Some(1),
            output_node: Some(1),
        };

        let code = generate_effect(&graph, "TestTremolo");

        assert!(code.contains("pub struct TestTremolo"));
        assert!(code.contains("impl AudioNode for TestTremolo"));
        assert!(code.contains("fn process"));
        assert!(code.contains("node_0_out")); // LFO output
        assert!(code.contains("node_1_out")); // Gain output
        assert!(code.contains("mod_0_base"));
        assert!(code.contains("mod_0_scale"));
    }

    #[test]
    fn test_chorus_codegen() {
        let graph = SerialAudioGraph {
            nodes: vec![
                SerialAudioNode::Lfo { rate: 0.5 },
                SerialAudioNode::Delay {
                    max_samples: 4096,
                    feedback: 0.0,
                    mix: 0.5,
                },
            ],
            audio_wires: vec![],
            param_wires: vec![SerialParamWire {
                from: 0,
                to: 1,
                param: 0,
                base: 882.0,
                scale: 220.0,
            }],
            input_node: Some(1),
            output_node: Some(1),
        };

        let code = generate_effect(&graph, "TestChorus");

        assert!(code.contains("pub struct TestChorus"));
        assert!(code.contains("read_interp"));
        assert!(code.contains("node_1_wet_mix"));
        assert!(code.contains("node_1_dry_mix"));
    }

    #[test]
    fn test_flanger_with_feedback() {
        let graph = SerialAudioGraph {
            nodes: vec![
                SerialAudioNode::Lfo { rate: 0.3 },
                SerialAudioNode::Delay {
                    max_samples: 512,
                    feedback: 0.7,
                    mix: 0.5,
                },
            ],
            audio_wires: vec![],
            param_wires: vec![SerialParamWire {
                from: 0,
                to: 1,
                param: 0,
                base: 132.0,
                scale: 88.0,
            }],
            input_node: Some(1),
            output_node: Some(1),
        };

        let code = generate_effect(&graph, "TestFlanger");

        assert!(code.contains("node_1_feedback"));
        assert!(code.contains("node_1_delayed * self.node_1_feedback"));
    }

    #[test]
    fn test_chain_of_nodes() {
        // Test a chain: Input -> Gain -> Allpass -> Output
        let graph = SerialAudioGraph {
            nodes: vec![
                SerialAudioNode::Gain { gain: 0.8 },
                SerialAudioNode::Allpass { coefficient: 0.5 },
            ],
            audio_wires: vec![(0, 1)], // Gain -> Allpass
            param_wires: vec![],
            input_node: Some(0),
            output_node: Some(1),
        };

        let code = generate_effect(&graph, "TestChain");

        // Verify the chain is processed in order
        assert!(code.contains("node_0_out")); // Gain output
        assert!(code.contains("node_1_out")); // Allpass output
        // Allpass should receive Gain's output
        assert!(code.contains("self.node_1.process(node_0_out)"));
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_serial_roundtrip() {
        let graph = SerialAudioGraph {
            nodes: vec![
                SerialAudioNode::Lfo { rate: 5.0 },
                SerialAudioNode::Gain { gain: 1.0 },
            ],
            audio_wires: vec![],
            param_wires: vec![SerialParamWire {
                from: 0,
                to: 1,
                param: 0,
                base: 0.5,
                scale: 0.5,
            }],
            input_node: Some(1),
            output_node: Some(1),
        };

        let json = serde_json::to_string(&graph).unwrap();
        let loaded: SerialAudioGraph = serde_json::from_str(&json).unwrap();

        assert_eq!(loaded.nodes.len(), 2);
        assert_eq!(loaded.param_wires.len(), 1);
    }
}
