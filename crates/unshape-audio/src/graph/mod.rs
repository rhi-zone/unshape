//! Audio processing graph and signal chain.
//!
//! Provides a flexible way to connect audio processors into chains and graphs.

mod chain;
mod mixer;
mod nodes;
mod params;
mod swappable;
mod traits;

pub use chain::*;
pub use mixer::*;
pub use nodes::*;
pub use params::*;
pub use swappable::*;
pub use traits::*;

// ============================================================================
// AudioGraph with Parameter Modulation
// ============================================================================

/// Index for a node in the audio graph.
pub type NodeIndex = usize;

/// Audio wire connecting one node's output to another's input.
#[derive(Debug, Clone, Copy)]
pub struct AudioWire {
    /// Source node index.
    pub from: NodeIndex,
    /// Destination node index.
    pub to: NodeIndex,
}

/// Parameter modulation wire.
///
/// Connects a node's output to another node's parameter with scaling.
/// Final param value = base + (source_output * scale)
#[derive(Debug, Clone, Copy)]
pub struct ParamWire {
    /// Source node index (provides modulation signal).
    pub from: NodeIndex,
    /// Destination node index.
    pub to: NodeIndex,
    /// Parameter index on destination node.
    pub param: usize,
    /// Base value for the parameter.
    pub base: f32,
    /// Scale factor for modulation.
    pub scale: f32,
}

/// Pre-computed execution info for a single node.
///
/// Built once when graph structure changes, used on every sample.
#[derive(Debug, Clone, Default)]
struct NodeExecInfo {
    /// Indices of nodes that feed audio into this node.
    audio_inputs: Vec<NodeIndex>,
    /// Parameter modulations: (source_node, param_index, base, scale).
    param_mods: Vec<(NodeIndex, usize, f32, f32)>,
    /// Whether this node receives external input.
    receives_input: bool,
}

/// Audio graph with parameter modulation support.
///
/// Unlike [`Chain`] which is linear, `AudioGraph` supports arbitrary routing
/// and parameter modulation (connecting one node's output to another's parameter).
///
/// # Example
///
/// ```
/// use unshape_audio::graph::{AffineNode, AudioGraph, AudioContext};
/// use unshape_audio::primitive::LfoNode;
///
/// let mut graph = AudioGraph::new();
///
/// // Build a simple tremolo: LFO modulates gain
/// let lfo = graph.add(LfoNode::with_freq(5.0, 44100.0));
/// let gain = graph.add(AffineNode::gain(1.0));
///
/// // Audio path: input → gain → output
/// graph.connect_input(gain);
/// graph.set_output(gain);
///
/// // Modulation: LFO → gain parameter (base=0.5, scale=0.5 means 0-1 range)
/// graph.modulate(lfo, gain, AffineNode::PARAM_GAIN, 0.5, 0.5);
///
/// // Process
/// let ctx = AudioContext::new(44100.0);
/// let output = graph.process(1.0, &ctx);
/// ```
pub struct AudioGraph {
    nodes: Vec<Box<dyn AudioNode>>,
    /// Type IDs for each node (for pattern matching).
    node_type_ids: Vec<std::any::TypeId>,
    /// Audio signal routing (node → node).
    audio_wires: Vec<AudioWire>,
    /// Parameter modulation routing (node → param).
    param_wires: Vec<ParamWire>,
    /// Which node receives external input.
    input_node: Option<NodeIndex>,
    /// Which node provides output.
    output_node: Option<NodeIndex>,
    /// Cached node outputs (reused each process call).
    outputs: Vec<f32>,
    /// Pre-computed per-node execution info. None means needs rebuild.
    exec_info: Option<Vec<NodeExecInfo>>,
    /// Sample counter for control-rate updates.
    sample_count: u32,
    /// Control rate divisor (update params every N samples). 0 = every sample.
    control_rate: u32,
}

impl AudioGraph {
    /// Creates an empty audio graph.
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            node_type_ids: Vec::new(),
            audio_wires: Vec::new(),
            param_wires: Vec::new(),
            input_node: None,
            output_node: None,
            outputs: Vec::new(),
            exec_info: None,
            sample_count: 0,
            control_rate: 0, // 0 = audio rate (every sample)
        }
    }

    /// Sets control rate divisor for parameter updates.
    ///
    /// Parameters are updated every `rate` samples. Set to 0 for audio-rate
    /// updates (every sample). Typical values: 32, 64, 128.
    ///
    /// Control-rate updates reduce CPU but add latency to modulation.
    /// For LFO modulation, 64 samples (~1.5ms at 44.1kHz) is usually fine.
    pub fn set_control_rate(&mut self, rate: u32) {
        self.control_rate = rate;
    }

    /// Adds a node to the graph and returns its index.
    pub fn add<N: AudioNode + 'static>(&mut self, node: N) -> NodeIndex {
        let index = self.nodes.len();
        self.nodes.push(Box::new(node));
        self.node_type_ids.push(std::any::TypeId::of::<N>());
        self.outputs.push(0.0);
        self.exec_info = None; // Invalidate cache
        index
    }

    /// Connects audio output of one node to input of another.
    pub fn connect(&mut self, from: NodeIndex, to: NodeIndex) {
        self.audio_wires.push(AudioWire { from, to });
        self.exec_info = None; // Invalidate cache
    }

    /// Connects external input to a node.
    pub fn connect_input(&mut self, to: NodeIndex) {
        self.input_node = Some(to);
        self.exec_info = None; // Invalidate cache
    }

    /// Sets which node provides the graph output.
    pub fn set_output(&mut self, node: NodeIndex) {
        self.output_node = Some(node);
    }

    /// Adds parameter modulation.
    ///
    /// The source node's output modulates the destination's parameter:
    /// `param_value = base + source_output * scale`
    ///
    /// # Arguments
    /// * `from` - Source node (modulator)
    /// * `to` - Destination node
    /// * `param` - Parameter index on destination
    /// * `base` - Base value when modulator is 0
    /// * `scale` - How much modulation affects the parameter
    pub fn modulate(
        &mut self,
        from: NodeIndex,
        to: NodeIndex,
        param: usize,
        base: f32,
        scale: f32,
    ) {
        self.param_wires.push(ParamWire {
            from,
            to,
            param,
            base,
            scale,
        });
        self.exec_info = None; // Invalidate cache
    }

    /// Modulates by parameter name (convenience wrapper).
    ///
    /// Looks up the parameter index by name. Panics if not found.
    pub fn modulate_named(
        &mut self,
        from: NodeIndex,
        to: NodeIndex,
        param_name: &str,
        base: f32,
        scale: f32,
    ) {
        let params = self.nodes[to].params();
        let param_idx = params
            .iter()
            .position(|p| p.name == param_name)
            .unwrap_or_else(|| panic!("parameter '{}' not found on node {}", param_name, to));
        self.modulate(from, to, param_idx, base, scale);
    }

    /// Returns the number of nodes.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Builds per-node execution info from wires.
    fn build_exec_info(&mut self) {
        let mut info: Vec<NodeExecInfo> = (0..self.nodes.len())
            .map(|_| NodeExecInfo::default())
            .collect();

        // Build audio input lists
        for wire in &self.audio_wires {
            info[wire.to].audio_inputs.push(wire.from);
        }

        // Build param modulation lists
        for wire in &self.param_wires {
            info[wire.to]
                .param_mods
                .push((wire.from, wire.param, wire.base, wire.scale));
        }

        // Mark input node
        if let Some(input_idx) = self.input_node {
            info[input_idx].receives_input = true;
        }

        self.exec_info = Some(info);
    }

    /// Ensures exec_info is built.
    #[inline]
    fn ensure_compiled(&mut self) {
        if self.exec_info.is_none() {
            self.build_exec_info();
        }
    }

    /// Processes one sample through the graph.
    ///
    /// Evaluates nodes in index order. Nodes receive the sum of all connected
    /// inputs. Parameters are modulated before each node processes.
    pub fn process(&mut self, input: f32, ctx: &AudioContext) -> f32 {
        self.ensure_compiled();

        // Check if we should update params this sample
        let update_params = self.control_rate == 0 || self.sample_count == 0;
        self.sample_count = if self.control_rate == 0 {
            0
        } else {
            (self.sample_count + 1) % self.control_rate
        };

        // Clear outputs
        for out in &mut self.outputs {
            *out = 0.0;
        }

        // Safety: we just ensured exec_info is Some
        let exec_info = self.exec_info.as_ref().unwrap();

        // Process each node in order
        for i in 0..self.nodes.len() {
            let info = &exec_info[i];

            // Apply parameter modulation only at control rate
            if update_params {
                for &(from, param, base, scale) in &info.param_mods {
                    let mod_value = self.outputs[from];
                    let param_value = base + mod_value * scale;
                    self.nodes[i].set_param(param, param_value);
                }
            }

            // Gather audio input
            let mut node_input = 0.0;

            // External input
            if info.receives_input {
                node_input += input;
            }

            // Inputs from other nodes (iterate small per-node vec)
            for &from in &info.audio_inputs {
                node_input += self.outputs[from];
            }

            // Process and store output
            self.outputs[i] = self.nodes[i].process(node_input, ctx);
        }

        // Return output node's value
        self.output_node.map(|i| self.outputs[i]).unwrap_or(0.0)
    }

    /// Processes a buffer of samples.
    pub fn process_buffer(&mut self, buffer: &mut [f32], ctx: &mut AudioContext) {
        for sample in buffer {
            *sample = self.process(*sample, ctx);
            ctx.advance();
        }
    }

    /// Generates samples (no input).
    pub fn generate(&mut self, buffer: &mut [f32], ctx: &mut AudioContext) {
        for sample in buffer {
            *sample = self.process(0.0, ctx);
            ctx.advance();
        }
    }

    /// Resets all nodes.
    pub fn reset(&mut self) {
        for node in &mut self.nodes {
            node.reset();
        }
    }

    // ========================================================================
    // Methods for graph optimization / pattern matching
    // ========================================================================

    /// Returns the audio wires.
    pub fn audio_wires(&self) -> &[AudioWire] {
        &self.audio_wires
    }

    /// Returns the param wires.
    pub fn param_wires(&self) -> &[ParamWire] {
        &self.param_wires
    }

    /// Returns the input node index if set.
    pub fn input_node(&self) -> Option<NodeIndex> {
        self.input_node
    }

    /// Returns the output node index if set.
    pub fn output_node(&self) -> Option<NodeIndex> {
        self.output_node
    }

    /// Returns the type ID of a node.
    pub fn node_type_id(&self, index: NodeIndex) -> Option<std::any::TypeId> {
        self.node_type_ids.get(index).copied()
    }

    /// Returns the NodeType enum for a node (for pattern matching).
    #[cfg(feature = "optimize")]
    pub fn node_type(&self, index: NodeIndex) -> Option<crate::optimize::NodeType> {
        self.node_type_id(index)
            .map(crate::optimize::NodeType::from_type_id)
    }

    /// Returns the name of a parameter on a node.
    pub fn node_param_name(&self, node: NodeIndex, param_idx: usize) -> Option<&'static str> {
        self.nodes
            .get(node)
            .and_then(|n| n.params().get(param_idx))
            .map(|p| p.name)
    }

    /// Returns the current value of a parameter on a node.
    ///
    /// Used by JIT compilation to extract parameter values at compile time.
    pub fn node_param_value(&self, node: NodeIndex, param_idx: usize) -> Option<f32> {
        self.nodes.get(node).and_then(|n| n.get_param(param_idx))
    }

    /// Takes a node out of the graph, replacing it with a passthrough.
    ///
    /// Used by JIT compilation to move stateful nodes into the compiled graph.
    /// Returns `None` if the index is out of bounds.
    pub fn take_node(&mut self, node: NodeIndex) -> Option<Box<dyn AudioNode>> {
        if node < self.nodes.len() {
            Some(std::mem::replace(
                &mut self.nodes[node],
                Box::new(AffineNode::identity()),
            ))
        } else {
            None
        }
    }

    /// Adds a boxed node to the graph with a known type ID.
    ///
    /// Use this when the concrete type is known. For unknown types, use
    /// `add_boxed_unknown` (the node won't be matchable in pattern optimization).
    pub fn add_boxed_typed(
        &mut self,
        node: Box<dyn AudioNode>,
        type_id: std::any::TypeId,
    ) -> NodeIndex {
        let index = self.nodes.len();
        self.nodes.push(node);
        self.node_type_ids.push(type_id);
        self.outputs.push(0.0);
        self.exec_info = None;
        index
    }

    /// Adds a boxed node with unknown type.
    ///
    /// The node won't be matchable in pattern optimization.
    pub fn add_boxed(&mut self, node: Box<dyn AudioNode>) -> NodeIndex {
        // Use unit type as a placeholder - won't match any pattern
        self.add_boxed_typed(node, std::any::TypeId::of::<()>())
    }

    /// Reconnects an audio wire from one destination to another.
    pub fn reconnect_audio(&mut self, from: NodeIndex, _old_to: NodeIndex, new_to: NodeIndex) {
        for wire in &mut self.audio_wires {
            if wire.from == from {
                wire.to = new_to;
            }
        }
        self.exec_info = None;
    }

    /// Removes a node from the graph.
    ///
    /// Warning: This invalidates indices! Nodes after the removed index shift down.
    pub fn remove_node(&mut self, index: NodeIndex) {
        if index >= self.nodes.len() {
            return;
        }

        // Remove the node and associated data
        self.nodes.remove(index);
        self.node_type_ids.remove(index);
        self.outputs.remove(index);

        // Update wire indices
        self.audio_wires.retain_mut(|w| {
            if w.from == index || w.to == index {
                return false; // Remove wires to/from deleted node
            }
            // Adjust indices for nodes that shifted
            if w.from > index {
                w.from -= 1;
            }
            if w.to > index {
                w.to -= 1;
            }
            true
        });

        self.param_wires.retain_mut(|w| {
            if w.from == index || w.to == index {
                return false;
            }
            if w.from > index {
                w.from -= 1;
            }
            if w.to > index {
                w.to -= 1;
            }
            true
        });

        // Update input/output references
        if let Some(ref mut input) = self.input_node {
            if *input == index {
                self.input_node = None;
            } else if *input > index {
                *input -= 1;
            }
        }

        if let Some(ref mut output) = self.output_node {
            if *output == index {
                self.output_node = None;
            } else if *output > index {
                *output -= 1;
            }
        }

        self.exec_info = None;
    }
}

impl Default for AudioGraph {
    fn default() -> Self {
        Self::new()
    }
}

impl AudioNode for AudioGraph {
    fn process(&mut self, input: f32, ctx: &AudioContext) -> f32 {
        AudioGraph::process(self, input, ctx)
    }

    fn reset(&mut self) {
        AudioGraph::reset(self);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_audio_context() {
        let mut ctx = AudioContext::new(44100.0);
        assert_eq!(ctx.sample_index, 0);
        assert!((ctx.time - 0.0).abs() < 0.0001);

        ctx.advance();
        assert_eq!(ctx.sample_index, 1);
        assert!((ctx.time - ctx.dt).abs() < 0.0001);
    }

    #[test]
    fn test_oscillator_sine() {
        let mut osc = Oscillator::sine(440.0);
        let ctx = AudioContext::new(44100.0);

        let sample = osc.process(0.0, &ctx);
        assert!(sample >= -1.0 && sample <= 1.0);
    }

    #[test]
    fn test_chain_empty() {
        let mut chain = Chain::new();
        let ctx = AudioContext::new(44100.0);

        let output = chain.process(1.0, &ctx);
        assert!((output - 1.0).abs() < 0.0001);
    }

    #[test]
    fn test_chain_gain() {
        let mut chain = Chain::new().with(AffineNode::gain(0.5));
        let ctx = AudioContext::new(44100.0);

        let output = chain.process(1.0, &ctx);
        assert!((output - 0.5).abs() < 0.0001);
    }

    #[test]
    fn test_chain_multiple_nodes() {
        let mut chain = Chain::new()
            .with(AffineNode::gain(2.0))
            .with(AffineNode::offset(1.0))
            .with(Clip::symmetric(2.0));

        let ctx = AudioContext::new(44100.0);

        // 1.0 * 2.0 = 2.0, + 1.0 = 3.0, clip to 2.0
        let output = chain.process(1.0, &ctx);
        assert!((output - 2.0).abs() < 0.0001);
    }

    #[test]
    fn test_chain_generate() {
        let sample_rate = 44100.0;
        let mut chain = Chain::new().with(Oscillator::sine(440.0));

        let mut buffer = vec![0.0; 100];
        let mut ctx = AudioContext::new(sample_rate);

        chain.generate(&mut buffer, &mut ctx);

        // Should have generated non-zero samples
        assert!(buffer.iter().any(|&s| s.abs() > 0.01));
    }

    #[test]
    fn test_mixer() {
        let mut mixer = Mixer::new()
            .with(Constant(1.0), 0.5)
            .with(Constant(2.0), 0.25);

        let ctx = AudioContext::new(44100.0);
        let output = mixer.process(0.0, &ctx);

        // 1.0 * 0.5 + 2.0 * 0.25 = 0.5 + 0.5 = 1.0
        assert!((output - 1.0).abs() < 0.0001);
    }

    #[test]
    fn test_soft_clip() {
        let mut clip = SoftClip::new(1.0);
        let ctx = AudioContext::new(44100.0);

        // Moderate input should be slightly compressed
        let out1 = clip.process(0.5, &ctx);
        assert!(out1 > 0.4 && out1 < 0.5);

        // Large input should be heavily compressed toward 1.0
        let out2 = clip.process(10.0, &ctx);
        assert!(out2 > 0.999); // tanh(10) ≈ 0.9999999958
    }

    #[test]
    fn test_lowpass_node() {
        let mut filter = LowPassNode::new(1000.0, 44100.0);
        let ctx = AudioContext::new(44100.0);

        // Process a few samples
        for _ in 0..100 {
            filter.process(1.0, &ctx);
        }

        let output = filter.process(1.0, &ctx);
        assert!(output > 0.9); // Should approach 1.0
    }

    #[test]
    fn test_adsr_node() {
        let mut env = AdsrNode::new(0.01, 0.01, 0.5, 0.01);
        env.trigger();

        let mut ctx = AudioContext::new(44100.0);

        // Process through attack
        let mut max_output = 0.0f32;
        for _ in 0..500 {
            let out = env.process(1.0, &ctx);
            max_output = max_output.max(out);
            ctx.advance();
        }

        assert!(max_output > 0.9);
    }

    #[test]
    fn test_ring_mod() {
        let carrier = Oscillator::sine(440.0);
        let modulator = Oscillator::sine(110.0);

        let mut ring = Chain::new().with(carrier).with(RingMod::new(modulator));

        let ctx = AudioContext::new(44100.0);
        let output = ring.process(0.0, &ctx);

        // Should produce some output
        assert!(output.abs() <= 1.0);
    }

    // ========================================================================
    // AudioGraph tests
    // ========================================================================

    #[test]
    fn test_audio_graph_simple_passthrough() {
        let mut graph = AudioGraph::new();
        let gain = graph.add(AffineNode::gain(1.0));
        graph.connect_input(gain);
        graph.set_output(gain);

        let ctx = AudioContext::new(44100.0);
        let out = graph.process(0.5, &ctx);
        assert!((out - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_audio_graph_gain() {
        let mut graph = AudioGraph::new();
        let gain = graph.add(AffineNode::gain(2.0));
        graph.connect_input(gain);
        graph.set_output(gain);

        let ctx = AudioContext::new(44100.0);
        let out = graph.process(0.5, &ctx);
        assert!((out - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_audio_graph_chain() {
        let mut graph = AudioGraph::new();
        let gain1 = graph.add(AffineNode::gain(2.0));
        let gain2 = graph.add(AffineNode::gain(0.5));

        graph.connect_input(gain1);
        graph.connect(gain1, gain2);
        graph.set_output(gain2);

        let ctx = AudioContext::new(44100.0);
        let out = graph.process(1.0, &ctx);
        // 1.0 * 2.0 * 0.5 = 1.0
        assert!((out - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_audio_graph_parameter_modulation() {
        use crate::primitive::LfoNode;

        let mut graph = AudioGraph::new();

        // LFO modulates gain
        let lfo = graph.add(LfoNode::with_freq(10.0, 44100.0));
        let gain = graph.add(AffineNode::gain(1.0));

        graph.connect_input(gain);
        graph.set_output(gain);

        // Modulate gain: base=0.5, scale=0.5 (so gain varies 0-1)
        graph.modulate(lfo, gain, AffineNode::PARAM_GAIN, 0.5, 0.5);

        let ctx = AudioContext::new(44100.0);

        // Collect outputs over one LFO cycle
        let mut outputs = Vec::new();
        for _ in 0..4410 {
            // ~1/10th second = one 10Hz cycle
            outputs.push(graph.process(1.0, &ctx));
        }

        // Should have variation due to modulation
        let min = outputs.iter().cloned().fold(f32::MAX, f32::min);
        let max = outputs.iter().cloned().fold(f32::MIN, f32::max);

        // With base=0.5, scale=0.5, and LFO going -1 to 1,
        // gain should vary from 0 to 1
        assert!(min < 0.1, "min was {}", min);
        assert!(max > 0.9, "max was {}", max);
    }

    #[test]
    fn test_audio_graph_modulate_named() {
        let mut graph = AudioGraph::new();
        let lfo = graph.add(Oscillator::sine(5.0)); // Use oscillator as modulator
        let gain = graph.add(AffineNode::gain(1.0));

        graph.connect_input(gain);
        graph.set_output(gain);

        // Modulate by name
        graph.modulate_named(lfo, gain, "gain", 0.5, 0.5);

        assert_eq!(graph.param_wires.len(), 1);
        assert_eq!(graph.param_wires[0].param, AffineNode::PARAM_GAIN);
    }

    #[test]
    fn test_audio_graph_as_audio_node() {
        // AudioGraph implements AudioNode, so can be nested in Chain
        let mut inner = AudioGraph::new();
        let gain = inner.add(AffineNode::gain(2.0));
        inner.connect_input(gain);
        inner.set_output(gain);

        let mut chain = Chain::new().with(inner).with(AffineNode::gain(0.5));

        let ctx = AudioContext::new(44100.0);
        let out = chain.process(1.0, &ctx);
        // 1.0 * 2.0 * 0.5 = 1.0
        assert!((out - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_atomic_f32() {
        let param = AtomicF32::new(0.5);
        assert!((param.get() - 0.5).abs() < 1e-6);

        param.set(0.75);
        assert!((param.get() - 0.75).abs() < 1e-6);

        // Test special values
        param.set(0.0);
        assert!((param.get() - 0.0).abs() < 1e-6);

        param.set(-1.0);
        assert!((param.get() - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn test_atomic_params() {
        let mut params = AtomicParams::new();
        params.add("cutoff", 1000.0);
        params.add("resonance", 0.5);
        params.add("gain", 1.0);

        assert_eq!(params.len(), 3);

        // Get by name
        assert!((params.get("cutoff").unwrap() - 1000.0).abs() < 1e-6);
        assert!((params.get("resonance").unwrap() - 0.5).abs() < 1e-6);
        assert!(params.get("nonexistent").is_none());

        // Set by name
        assert!(params.set("cutoff", 2000.0));
        assert!((params.get("cutoff").unwrap() - 2000.0).abs() < 1e-6);
        assert!(!params.set("nonexistent", 0.0));

        // Get/set by index
        assert!((params.get_index(0).unwrap() - 2000.0).abs() < 1e-6);
        assert!(params.set_index(1, 0.8));
        assert!((params.get_index(1).unwrap() - 0.8).abs() < 1e-6);
        assert!(params.get_index(99).is_none());
    }

    // =========================================================================
    // SwappableGraph tests
    // =========================================================================

    #[test]
    fn test_swappable_graph_basic() {
        // Create a simple graph that outputs 1.0
        let mut graph1 = AudioGraph::new();
        let const1 = graph1.add(Constant(1.0));
        graph1.set_output(const1);

        let mut swappable = SwappableGraph::new(graph1);
        let ctx = AudioContext::new(44100.0);

        // Process should output 1.0
        assert!((swappable.process(0.0, &ctx) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_swappable_graph_swap_no_crossfade() {
        // Create first graph that outputs 1.0
        let mut graph1 = AudioGraph::new();
        let const1 = graph1.add(Constant(1.0));
        graph1.set_output(const1);

        let mut swappable = SwappableGraph::new(graph1);
        let ctx = AudioContext::new(44100.0);

        assert!((swappable.process(0.0, &ctx) - 1.0).abs() < 1e-6);

        // Create second graph that outputs 2.0
        let mut graph2 = AudioGraph::new();
        let const2 = graph2.add(Constant(2.0));
        graph2.set_output(const2);

        // Swap with 0 crossfade (instant)
        swappable.swap(graph2, 0);

        // Should immediately output 2.0
        assert!((swappable.process(0.0, &ctx) - 2.0).abs() < 1e-6);
        assert!(!swappable.is_crossfading());
    }

    #[test]
    fn test_swappable_graph_crossfade() {
        // Create first graph that outputs 0.0
        let mut graph1 = AudioGraph::new();
        let const1 = graph1.add(Constant(0.0));
        graph1.set_output(const1);

        let mut swappable = SwappableGraph::new(graph1);
        let ctx = AudioContext::new(44100.0);

        // Create second graph that outputs 1.0
        let mut graph2 = AudioGraph::new();
        let const2 = graph2.add(Constant(1.0));
        graph2.set_output(const2);

        // Swap with 10 sample crossfade
        swappable.swap(graph2, 10);
        assert!(swappable.is_crossfading());

        // Process through crossfade
        let mut outputs = Vec::new();
        for _ in 0..15 {
            outputs.push(swappable.process(0.0, &ctx));
        }

        // First sample should be closer to 0.0 (old graph)
        assert!(outputs[0] < 0.5);

        // Middle sample should be around 0.5-ish (equal power crossfade)
        // At t=0.5, equal power gives sin(0.5 * π/2) ≈ 0.707 for new
        // so output ≈ 0 * 0.707 + 1 * 0.707 ≈ 0.707 (but actually 0 * old_gain + 1 * new_gain)

        // Last samples should be 1.0 (new graph only)
        assert!((outputs[14] - 1.0).abs() < 1e-6);
        assert!(!swappable.is_crossfading());
    }

    #[test]
    fn test_swappable_graph_cancel_swap() {
        let mut graph1 = AudioGraph::new();
        let const1 = graph1.add(Constant(1.0));
        graph1.set_output(const1);

        let mut swappable = SwappableGraph::new(graph1);
        let ctx = AudioContext::new(44100.0);

        let mut graph2 = AudioGraph::new();
        let const2 = graph2.add(Constant(2.0));
        graph2.set_output(const2);

        // Start a swap
        swappable.swap(graph2, 100);
        assert!(swappable.is_crossfading());

        // Process a few samples
        for _ in 0..10 {
            swappable.process(0.0, &ctx);
        }

        // Cancel the swap
        swappable.cancel_swap();
        assert!(!swappable.is_crossfading());

        // Should be back to outputting 1.0
        assert!((swappable.process(0.0, &ctx) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_swappable_graph_process_buffer() {
        let mut graph1 = AudioGraph::new();
        let const1 = graph1.add(Constant(0.5));
        graph1.set_output(const1);

        let mut swappable = SwappableGraph::new(graph1);
        let mut ctx = AudioContext::new(44100.0);

        let mut buffer = vec![0.0; 64];
        swappable.process_buffer(&mut buffer, &mut ctx);

        for sample in &buffer {
            assert!((sample - 0.5).abs() < 1e-6);
        }
    }

    #[test]
    fn test_swappable_graph_generate() {
        let mut graph1 = AudioGraph::new();
        let const1 = graph1.add(Constant(0.75));
        graph1.set_output(const1);

        let mut swappable = SwappableGraph::new(graph1);
        let mut ctx = AudioContext::new(44100.0);

        let mut buffer = vec![0.0; 32];
        swappable.generate(&mut buffer, &mut ctx);

        for sample in &buffer {
            assert!((sample - 0.75).abs() < 1e-6);
        }
    }

    #[test]
    fn test_swappable_graph_crossfade_progress() {
        let graph1 = AudioGraph::new();
        let mut swappable = SwappableGraph::new(graph1);
        let ctx = AudioContext::new(44100.0);

        // No crossfade - progress is 1.0
        assert!((swappable.crossfade_progress() - 1.0).abs() < 1e-6);

        // Start crossfade
        swappable.swap(AudioGraph::new(), 100);
        assert!((swappable.crossfade_progress() - 0.0).abs() < 1e-6);

        // Process halfway
        for _ in 0..50 {
            swappable.process(0.0, &ctx);
        }
        assert!((swappable.crossfade_progress() - 0.5).abs() < 0.02);

        // Process to end
        for _ in 0..50 {
            swappable.process(0.0, &ctx);
        }
        assert!((swappable.crossfade_progress() - 1.0).abs() < 1e-6);
    }
}
