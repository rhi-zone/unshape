//! Code generation for static audio graphs.
//!
//! This module generates optimized Rust code from serialized graph descriptions.
//! Use this in your `build.rs` to compile audio graphs into efficient static structs
//! that eliminate dynamic dispatch and wire iteration overhead.
//!
//! # Usage
//!
//! In your `build.rs`:
//!
//! ```ignore
//! use rhizome_resin_audio::codegen::{SerialAudioGraph, SerialAudioNode, SerialParamWire, generate_effect};
//!
//! fn main() {
//!     let graph = SerialAudioGraph {
//!         nodes: vec![
//!             SerialAudioNode::Lfo { rate: 5.0 },
//!             SerialAudioNode::Gain { gain: 1.0 },
//!         ],
//!         audio_wires: vec![(0, 1)],  // LFO doesn't carry audio, but shows connectivity
//!         param_wires: vec![
//!             SerialParamWire { from: 0, to: 1, param: 0, base: 0.5, scale: 0.5 },
//!         ],
//!         input_node: Some(1),   // Audio enters at Gain
//!         output_node: Some(1),  // Audio exits from Gain
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

#[cfg(feature = "codegen")]
use serde::{Deserialize, Serialize};

// ============================================================================
// Serializable Graph Types
// ============================================================================

/// Serializable representation of an audio graph for codegen.
///
/// This is the input format for code generation. Build this struct
/// in your `build.rs` and pass it to [`generate_effect`].
#[derive(Debug, Clone)]
#[cfg_attr(feature = "codegen", derive(Serialize, Deserialize))]
pub struct SerialAudioGraph {
    /// Nodes in processing order (index = node ID).
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
#[cfg_attr(feature = "codegen", derive(Serialize, Deserialize))]
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
#[cfg_attr(feature = "codegen", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "codegen", serde(tag = "type"))]
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
            SerialAudioNode::Lfo { .. } => "crate::primitive::PhaseOsc",
            SerialAudioNode::Gain { .. } => "f32", // Just store the gain value
            SerialAudioNode::Delay { .. } => "crate::primitive::DelayLine<true>",
            SerialAudioNode::Mix { .. } => "f32", // Just store the mix value
            SerialAudioNode::Envelope { .. } => "crate::primitive::EnvelopeFollower",
            SerialAudioNode::Allpass { .. } => "crate::primitive::Allpass1",
            SerialAudioNode::Passthrough => "()", // No state
        }
    }

    /// Returns whether this node needs a field in the generated struct.
    fn needs_field(&self) -> bool {
        !matches!(self, SerialAudioNode::Passthrough)
    }

    /// Returns the initialization expression for this node.
    fn init_code(&self, sample_rate: &str) -> String {
        match self {
            SerialAudioNode::Lfo { .. } => "crate::primitive::PhaseOsc::new()".to_string(),
            SerialAudioNode::Gain { gain } => format!("{gain}_f32"),
            SerialAudioNode::Delay { max_samples } => {
                format!("crate::primitive::DelayLine::new({max_samples})")
            }
            SerialAudioNode::Mix { mix } => format!("{mix}_f32"),
            SerialAudioNode::Envelope { attack, release } => {
                format!(
                    "crate::primitive::EnvelopeFollower::new({sample_rate}, {attack}, {release})"
                )
            }
            SerialAudioNode::Allpass { coefficient } => {
                format!("crate::primitive::Allpass1::new({coefficient})")
            }
            SerialAudioNode::Passthrough => "()".to_string(),
        }
    }

    /// Returns the output expression for this node given an input variable.
    fn process_code(&self, field: &str, input: &str) -> String {
        match self {
            SerialAudioNode::Lfo { .. } => format!("{field}.sine()"),
            SerialAudioNode::Gain { .. } => format!("{input} * {field}"),
            SerialAudioNode::Delay { .. } => {
                // Delay needs special handling - read then write
                format!("{field}.read_interp({field}.len() as f32 / 2.0)")
            }
            SerialAudioNode::Mix { .. } => {
                // Mix needs two inputs - handled specially in graph codegen
                format!("{input}") // Placeholder
            }
            SerialAudioNode::Envelope { .. } => format!("{field}.process({input})"),
            SerialAudioNode::Allpass { .. } => format!("{field}.process({input})"),
            SerialAudioNode::Passthrough => input.to_string(),
        }
    }

    /// Returns extra code to run after processing (e.g., advancing LFO phase).
    fn post_process_code(&self, field: &str, _phase_inc_field: Option<&str>) -> Option<String> {
        match self {
            SerialAudioNode::Lfo { .. } => {
                // Phase advancement - caller should provide the phase_inc field name
                None // Handled by caller with param wire info
            }
            SerialAudioNode::Delay { .. } => {
                // Write input to delay after reading
                Some(format!("// {field}.write(input) called by caller"))
            }
            _ => None,
        }
    }
}

// ============================================================================
// Code Generation
// ============================================================================

/// Generates optimized Rust code for an audio effect.
///
/// The generated code includes:
/// - A struct with fields for each node's state
/// - A `new(sample_rate: f32)` constructor
/// - An `AudioNode` implementation with inlined processing
///
/// # Arguments
///
/// * `graph` - The serialized graph description
/// * `name` - Name for the generated struct (e.g., "MyTremolo")
///
/// # Returns
///
/// A string containing valid Rust code that can be written to a file
/// and included with `include!()`.
pub fn generate_effect(graph: &SerialAudioGraph, name: &str) -> String {
    let mut code = String::new();

    // Header
    code.push_str("// Generated by rhizome-resin-audio codegen\n");
    code.push_str("// Do not edit manually\n\n");
    code.push_str("use rhizome_resin_audio::graph::{AudioContext, AudioNode};\n\n");

    // Analyze graph for optimizations
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

/// Analysis results for optimization.
struct GraphAnalysis {
    /// For each LFO node, the phase increment field name.
    lfo_phase_incs: Vec<(usize, String)>,
    /// Pre-computed modulation constants: (node_idx, base, scale).
    mod_constants: Vec<(usize, f32, f32)>,
    /// Processing order (topological sort).
    process_order: Vec<usize>,
}

fn analyze_graph(graph: &SerialAudioGraph) -> GraphAnalysis {
    // Find LFO nodes and their rates
    let mut lfo_phase_incs = Vec::new();
    for (i, node) in graph.nodes.iter().enumerate() {
        if let SerialAudioNode::Lfo { rate } = node {
            lfo_phase_incs.push((i, format!("node_{i}_phase_inc")));
            // Store rate for later use in init
            let _ = rate; // Used in init_code
        }
    }

    // Extract modulation constants from param wires
    let mut mod_constants = Vec::new();
    for wire in &graph.param_wires {
        mod_constants.push((wire.to, wire.base, wire.scale));
    }

    // Simple processing order: just use node indices
    // A more sophisticated version would do topological sort
    let process_order: Vec<usize> = (0..graph.nodes.len()).collect();

    GraphAnalysis {
        lfo_phase_incs,
        mod_constants,
        process_order,
    }
}

fn generate_struct(graph: &SerialAudioGraph, name: &str, analysis: &GraphAnalysis) -> String {
    let mut code = format!("pub struct {name} {{\n");

    // Add fields for each node that needs state
    for (i, node) in graph.nodes.iter().enumerate() {
        if node.needs_field() {
            let field_type = node.field_type();
            code.push_str(&format!("    node_{i}: {field_type},\n"));
        }
    }

    // Add phase increment fields for LFOs
    for (idx, field_name) in &analysis.lfo_phase_incs {
        let _ = idx; // Used for documentation
        code.push_str(&format!("    {field_name}: f32,\n"));
    }

    // Add pre-computed modulation constants
    for (i, (node_idx, _base, _scale)) in analysis.mod_constants.iter().enumerate() {
        let _ = node_idx;
        code.push_str(&format!("    mod_{i}_base: f32,\n"));
        code.push_str(&format!("    mod_{i}_scale: f32,\n"));
    }

    code.push_str("}\n");
    code
}

fn generate_impl(graph: &SerialAudioGraph, name: &str, analysis: &GraphAnalysis) -> String {
    let mut code = format!("impl {name} {{\n");
    code.push_str("    pub fn new(sample_rate: f32) -> Self {\n");
    code.push_str("        Self {\n");

    // Initialize each node
    for (i, node) in graph.nodes.iter().enumerate() {
        if node.needs_field() {
            let init = node.init_code("sample_rate");
            code.push_str(&format!("            node_{i}: {init},\n"));
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
    for (i, (_node_idx, base, scale)) in analysis.mod_constants.iter().enumerate() {
        code.push_str(&format!("            mod_{i}_base: {base}_f32,\n"));
        code.push_str(&format!("            mod_{i}_scale: {scale}_f32,\n"));
    }

    code.push_str("        }\n");
    code.push_str("    }\n");
    code.push_str("}\n");
    code
}

fn generate_audio_node_impl(
    graph: &SerialAudioGraph,
    name: &str,
    analysis: &GraphAnalysis,
) -> String {
    let mut code = format!("impl AudioNode for {name} {{\n");
    code.push_str("    fn process(&mut self, input: f32, _ctx: &AudioContext) -> f32 {\n");

    // Generate processing code based on graph structure
    // This is a simplified version - a full implementation would:
    // 1. Topologically sort nodes
    // 2. Generate code for each node in order
    // 3. Handle audio wiring between nodes

    // For now, generate code for common patterns
    if is_tremolo_pattern(graph) {
        code.push_str(&generate_tremolo_process(analysis));
    } else if is_chorus_pattern(graph) {
        code.push_str(&generate_chorus_process(graph, analysis));
    } else {
        // Fallback: generate generic processing
        code.push_str(&generate_generic_process(graph, analysis));
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

// ============================================================================
// Pattern Detection and Specialized Codegen
// ============================================================================

fn is_tremolo_pattern(graph: &SerialAudioGraph) -> bool {
    // Tremolo: LFO modulating Gain
    let has_lfo = graph
        .nodes
        .iter()
        .any(|n| matches!(n, SerialAudioNode::Lfo { .. }));
    let has_gain = graph
        .nodes
        .iter()
        .any(|n| matches!(n, SerialAudioNode::Gain { .. }));
    let has_param_wire = !graph.param_wires.is_empty();

    has_lfo && has_gain && has_param_wire && graph.nodes.len() <= 3
}

fn is_chorus_pattern(graph: &SerialAudioGraph) -> bool {
    // Chorus: LFO modulating Delay time, with Mix
    let has_lfo = graph
        .nodes
        .iter()
        .any(|n| matches!(n, SerialAudioNode::Lfo { .. }));
    let has_delay = graph
        .nodes
        .iter()
        .any(|n| matches!(n, SerialAudioNode::Delay { .. }));
    let has_mix = graph
        .nodes
        .iter()
        .any(|n| matches!(n, SerialAudioNode::Mix { .. }));

    has_lfo && has_delay && has_mix
}

fn generate_tremolo_process(analysis: &GraphAnalysis) -> String {
    let mut code = String::new();

    // Optimized tremolo: LFO → gain multiplication
    code.push_str("        // Optimized tremolo processing\n");
    code.push_str("        let lfo_out = self.node_0.sine();\n");

    if let Some((_, phase_inc_field)) = analysis.lfo_phase_incs.first() {
        code.push_str(&format!(
            "        self.node_0.advance(self.{phase_inc_field});\n"
        ));
    }

    code.push_str("        let gain = self.mod_0_base + lfo_out * self.mod_0_scale;\n");
    code.push_str("        input * gain\n");

    code
}

fn generate_chorus_process(graph: &SerialAudioGraph, analysis: &GraphAnalysis) -> String {
    let mut code = String::new();

    // Find the delay node index
    let delay_idx = graph
        .nodes
        .iter()
        .position(|n| matches!(n, SerialAudioNode::Delay { .. }))
        .unwrap_or(1);

    code.push_str("        // Optimized chorus processing\n");
    code.push_str("        let lfo_out = self.node_0.sine();\n");

    if let Some((_, phase_inc_field)) = analysis.lfo_phase_incs.first() {
        code.push_str(&format!(
            "        self.node_0.advance(self.{phase_inc_field});\n"
        ));
    }

    code.push_str("        let delay_time = self.mod_0_base + lfo_out * self.mod_0_scale;\n");
    code.push_str(&format!(
        "        let delayed = self.node_{delay_idx}.read_interp(delay_time);\n"
    ));
    code.push_str(&format!("        self.node_{delay_idx}.write(input);\n"));

    // Find mix value
    let mix_idx = graph
        .nodes
        .iter()
        .position(|n| matches!(n, SerialAudioNode::Mix { .. }));
    if let Some(idx) = mix_idx {
        code.push_str(&format!(
            "        input * (1.0 - self.node_{idx}) + delayed * self.node_{idx}\n"
        ));
    } else {
        code.push_str("        input * 0.5 + delayed * 0.5\n");
    }

    code
}

fn generate_generic_process(graph: &SerialAudioGraph, analysis: &GraphAnalysis) -> String {
    let mut code = String::new();

    code.push_str("        // Generic graph processing\n");
    code.push_str("        let mut signal = input;\n");

    for idx in &analysis.process_order {
        let node = &graph.nodes[*idx];
        if node.needs_field() {
            let field = format!("self.node_{idx}");
            match node {
                SerialAudioNode::Lfo { .. } => {
                    code.push_str(&format!("        let node_{idx}_out = {field}.sine();\n"));
                    if let Some((_, phase_inc)) =
                        analysis.lfo_phase_incs.iter().find(|(i, _)| i == idx)
                    {
                        code.push_str(&format!("        {field}.advance(self.{phase_inc});\n"));
                    }
                }
                SerialAudioNode::Gain { .. } => {
                    code.push_str(&format!("        signal = signal * {field};\n"));
                }
                SerialAudioNode::Delay { .. } => {
                    code.push_str(&format!(
                        "        let node_{idx}_out = {field}.read_interp(signal);\n"
                    ));
                    code.push_str(&format!("        {field}.write(signal);\n"));
                    code.push_str(&format!("        signal = node_{idx}_out;\n"));
                }
                _ => {}
            }
        }
    }

    code.push_str("        signal\n");
    code
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

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
        assert!(code.contains("node_0: crate::primitive::PhaseOsc"));
        assert!(code.contains("mod_0_base"));
        assert!(code.contains("mod_0_scale"));
    }

    #[test]
    fn test_chorus_codegen() {
        let graph = SerialAudioGraph {
            nodes: vec![
                SerialAudioNode::Lfo { rate: 0.5 },
                SerialAudioNode::Delay { max_samples: 4096 },
                SerialAudioNode::Mix { mix: 0.5 },
            ],
            audio_wires: vec![(1, 2)],
            param_wires: vec![SerialParamWire {
                from: 0,
                to: 1,
                param: 0,
                base: 882.0,  // ~20ms at 44.1kHz
                scale: 220.0, // ~5ms depth
            }],
            input_node: Some(1),
            output_node: Some(2),
        };

        let code = generate_effect(&graph, "TestChorus");

        assert!(code.contains("pub struct TestChorus"));
        assert!(code.contains("node_1: crate::primitive::DelayLine<true>"));
        assert!(code.contains("read_interp"));
    }

    #[cfg(feature = "codegen")]
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
