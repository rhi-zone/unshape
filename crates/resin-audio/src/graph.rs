//! Audio processing graph and signal chain.
//!
//! Provides a flexible way to connect audio processors into chains and graphs.

use crate::envelope::{Adsr, Ar, Lfo};
use crate::filter::{Biquad, Delay, FeedbackDelay, HighPass, LowPass};
use crate::osc;
use std::sync::atomic::{AtomicU32, Ordering};

// ============================================================================
// Lock-free parameter updates
// ============================================================================

/// An atomic f32 for lock-free parameter updates.
///
/// Use this for parameters that need to be updated from a UI thread
/// while audio is processing. Updates are wait-free and won't cause
/// audio glitches.
///
/// # Example
///
/// ```
/// use rhizome_resin_audio::graph::AtomicF32;
///
/// let param = AtomicF32::new(0.5);
///
/// // Audio thread reads
/// let value = param.get();
///
/// // UI thread writes (lock-free)
/// param.set(0.75);
/// ```
#[derive(Debug)]
pub struct AtomicF32(AtomicU32);

impl AtomicF32 {
    /// Creates a new atomic f32 with the given initial value.
    pub const fn new(value: f32) -> Self {
        Self(AtomicU32::new(value.to_bits()))
    }

    /// Gets the current value (relaxed ordering, suitable for audio).
    #[inline]
    pub fn get(&self) -> f32 {
        f32::from_bits(self.0.load(Ordering::Relaxed))
    }

    /// Sets the value (relaxed ordering, suitable for UI updates).
    #[inline]
    pub fn set(&self, value: f32) {
        self.0.store(value.to_bits(), Ordering::Relaxed);
    }

    /// Gets the current value with acquire ordering.
    #[inline]
    pub fn get_acquire(&self) -> f32 {
        f32::from_bits(self.0.load(Ordering::Acquire))
    }

    /// Sets the value with release ordering.
    #[inline]
    pub fn set_release(&self, value: f32) {
        self.0.store(value.to_bits(), Ordering::Release);
    }
}

impl Default for AtomicF32 {
    fn default() -> Self {
        Self::new(0.0)
    }
}

impl Clone for AtomicF32 {
    fn clone(&self) -> Self {
        Self::new(self.get())
    }
}

/// A set of named atomic parameters for lock-free audio control.
///
/// This provides a simple way to expose multiple parameters that can be
/// safely updated from any thread while audio is processing.
///
/// # Example
///
/// ```
/// use rhizome_resin_audio::graph::AtomicParams;
///
/// let mut params = AtomicParams::new();
/// params.add("cutoff", 1000.0);
/// params.add("resonance", 0.5);
///
/// // Audio thread reads
/// let cutoff = params.get("cutoff").unwrap();
///
/// // UI thread writes (lock-free)
/// params.set("cutoff", 2000.0);
/// ```
#[derive(Debug, Clone, Default)]
pub struct AtomicParams {
    names: Vec<&'static str>,
    values: Vec<AtomicF32>,
}

impl AtomicParams {
    /// Creates a new empty parameter set.
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a parameter with the given name and initial value.
    pub fn add(&mut self, name: &'static str, value: f32) {
        self.names.push(name);
        self.values.push(AtomicF32::new(value));
    }

    /// Gets a parameter value by name.
    pub fn get(&self, name: &str) -> Option<f32> {
        self.names
            .iter()
            .position(|&n| n == name)
            .map(|i| self.values[i].get())
    }

    /// Sets a parameter value by name.
    pub fn set(&self, name: &str, value: f32) -> bool {
        if let Some(i) = self.names.iter().position(|&n| n == name) {
            self.values[i].set(value);
            true
        } else {
            false
        }
    }

    /// Gets a parameter value by index.
    pub fn get_index(&self, index: usize) -> Option<f32> {
        self.values.get(index).map(|v| v.get())
    }

    /// Sets a parameter value by index.
    pub fn set_index(&self, index: usize, value: f32) -> bool {
        if let Some(v) = self.values.get(index) {
            v.set(value);
            true
        } else {
            false
        }
    }

    /// Returns the number of parameters.
    pub fn len(&self) -> usize {
        self.names.len()
    }

    /// Returns true if there are no parameters.
    pub fn is_empty(&self) -> bool {
        self.names.is_empty()
    }

    /// Returns an iterator over parameter names.
    pub fn names(&self) -> impl Iterator<Item = &'static str> + '_ {
        self.names.iter().copied()
    }
}

/// Audio processing context passed to nodes.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct AudioContext {
    /// Sample rate in Hz.
    pub sample_rate: f32,
    /// Current time in seconds.
    pub time: f32,
    /// Time delta (1 / sample_rate).
    pub dt: f32,
    /// Current sample index.
    pub sample_index: u64,
}

impl AudioContext {
    /// Creates a new audio context.
    pub fn new(sample_rate: f32) -> Self {
        let dt = 1.0 / sample_rate;
        Self {
            sample_rate,
            time: 0.0,
            dt,
            sample_index: 0,
        }
    }

    /// Advances the context by one sample.
    pub fn advance(&mut self) {
        self.sample_index += 1;
        self.time = self.sample_index as f32 * self.dt;
    }

    /// Resets the context to time zero.
    pub fn reset(&mut self) {
        self.sample_index = 0;
        self.time = 0.0;
    }
}

/// Describes a modulatable parameter on an audio node.
#[derive(Debug, Clone, Copy)]
pub struct ParamDescriptor {
    /// Parameter name.
    pub name: &'static str,
    /// Default value.
    pub default: f32,
    /// Minimum value.
    pub min: f32,
    /// Maximum value.
    pub max: f32,
}

impl ParamDescriptor {
    /// Creates a new parameter descriptor.
    pub const fn new(name: &'static str, default: f32, min: f32, max: f32) -> Self {
        Self {
            name,
            default,
            min,
            max,
        }
    }
}

/// Trait for audio processing nodes.
pub trait AudioNode: Send {
    /// Processes a single sample.
    fn process(&mut self, input: f32, ctx: &AudioContext) -> f32;

    /// Resets the node's internal state.
    fn reset(&mut self) {}

    /// Returns descriptors for modulatable parameters.
    ///
    /// Override this to expose parameters that can be modulated by other nodes
    /// in a graph. Parameters are set via [`set_param`] before each [`process`] call.
    fn params(&self) -> &'static [ParamDescriptor] {
        &[]
    }

    /// Sets a parameter value by index.
    ///
    /// Called by the graph executor to apply modulation before processing.
    /// Index corresponds to the parameter's position in [`params()`].
    fn set_param(&mut self, _index: usize, _value: f32) {}

    /// Gets a parameter's current value by index.
    ///
    /// Returns `None` if the index is out of bounds.
    /// Used by JIT compilation to extract parameter values at compile time.
    fn get_param(&self, _index: usize) -> Option<f32> {
        None
    }
}

/// Trait for block-based audio processing.
///
/// This is the unified interface for all audio processing tiers:
/// - Tier 1/2/4: Default impl loops over `AudioNode::process()` (already efficient)
/// - Tier 3 JIT: Native block impl (amortizes function call overhead)
///
/// Use this trait when you want code that works with any tier.
///
/// # Example
///
/// ```
/// use rhizome_resin_audio::graph::{BlockProcessor, AudioContext, Chain, AffineNode};
///
/// fn apply_effect<P: BlockProcessor>(effect: &mut P, audio: &mut [f32], sample_rate: f32) {
///     let mut output = vec![0.0; audio.len()];
///     let mut ctx = AudioContext::new(sample_rate);
///     effect.process_block(audio, &mut output, &mut ctx);
///     audio.copy_from_slice(&output);
/// }
/// ```
pub trait BlockProcessor: Send {
    /// Processes a block of samples.
    ///
    /// # Arguments
    /// * `input` - Input sample buffer
    /// * `output` - Output sample buffer (must be same length as input)
    /// * `ctx` - Audio context (will be advanced for each sample)
    fn process_block(&mut self, input: &[f32], output: &mut [f32], ctx: &mut AudioContext);

    /// Resets the processor's internal state.
    fn reset(&mut self);
}

/// Blanket implementation of BlockProcessor for all AudioNode types.
///
/// This provides efficient per-sample processing for Tier 1/2/4 where
/// the compiler can inline the process() calls.
impl<T: AudioNode> BlockProcessor for T {
    fn process_block(&mut self, input: &[f32], output: &mut [f32], ctx: &mut AudioContext) {
        debug_assert_eq!(
            input.len(),
            output.len(),
            "input and output buffers must be same length"
        );
        for (inp, out) in input.iter().zip(output.iter_mut()) {
            *out = self.process(*inp, ctx);
            ctx.advance();
        }
    }

    fn reset(&mut self) {
        AudioNode::reset(self);
    }
}

// ============================================================================
// Signal Chain
// ============================================================================

/// A linear chain of audio processors.
#[derive(Default)]
pub struct Chain {
    nodes: Vec<Box<dyn AudioNode>>,
}

impl Chain {
    /// Creates an empty chain.
    pub fn new() -> Self {
        Self { nodes: Vec::new() }
    }

    /// Adds a node to the end of the chain.
    pub fn push<N: AudioNode + 'static>(&mut self, node: N) {
        self.nodes.push(Box::new(node));
    }

    /// Adds a node and returns self (for builder pattern).
    pub fn with<N: AudioNode + 'static>(mut self, node: N) -> Self {
        self.push(node);
        self
    }

    /// Returns the number of nodes.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Returns true if the chain is empty.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Processes a single sample through the chain.
    pub fn process(&mut self, input: f32, ctx: &AudioContext) -> f32 {
        let mut signal = input;
        for node in &mut self.nodes {
            signal = node.process(signal, ctx);
        }
        signal
    }

    /// Processes a buffer of samples.
    pub fn process_buffer(&mut self, buffer: &mut [f32], ctx: &mut AudioContext) {
        for sample in buffer {
            *sample = self.process(*sample, ctx);
            ctx.advance();
        }
    }

    /// Generates samples into a buffer (no input).
    pub fn generate(&mut self, buffer: &mut [f32], ctx: &mut AudioContext) {
        for sample in buffer {
            *sample = self.process(0.0, ctx);
            ctx.advance();
        }
    }

    /// Resets all nodes in the chain.
    pub fn reset(&mut self) {
        for node in &mut self.nodes {
            node.reset();
        }
    }
}

impl BlockProcessor for Chain {
    fn process_block(&mut self, input: &[f32], output: &mut [f32], ctx: &mut AudioContext) {
        debug_assert_eq!(
            input.len(),
            output.len(),
            "input and output buffers must be same length"
        );
        for (inp, out) in input.iter().zip(output.iter_mut()) {
            *out = self.process(*inp, ctx);
            ctx.advance();
        }
    }

    fn reset(&mut self) {
        Chain::reset(self);
    }
}

// ============================================================================
// Mixer
// ============================================================================

/// Mixes multiple audio sources together.
pub struct Mixer {
    sources: Vec<Box<dyn AudioNode>>,
    gains: Vec<f32>,
}

impl Mixer {
    /// Creates an empty mixer.
    pub fn new() -> Self {
        Self {
            sources: Vec::new(),
            gains: Vec::new(),
        }
    }

    /// Adds a source with the given gain.
    pub fn add<N: AudioNode + 'static>(&mut self, node: N, gain: f32) {
        self.sources.push(Box::new(node));
        self.gains.push(gain);
    }

    /// Adds a source and returns self.
    pub fn with<N: AudioNode + 'static>(mut self, node: N, gain: f32) -> Self {
        self.add(node, gain);
        self
    }

    /// Sets the gain for a source by index.
    pub fn set_gain(&mut self, index: usize, gain: f32) {
        if index < self.gains.len() {
            self.gains[index] = gain;
        }
    }

    /// Processes and mixes all sources.
    pub fn process(&mut self, input: f32, ctx: &AudioContext) -> f32 {
        let mut output = 0.0;
        for (source, &gain) in self.sources.iter_mut().zip(self.gains.iter()) {
            output += source.process(input, ctx) * gain;
        }
        output
    }

    /// Resets all sources.
    pub fn reset(&mut self) {
        for source in &mut self.sources {
            source.reset();
        }
    }
}

impl Default for Mixer {
    fn default() -> Self {
        Self::new()
    }
}

impl AudioNode for Mixer {
    fn process(&mut self, input: f32, ctx: &AudioContext) -> f32 {
        Mixer::process(self, input, ctx)
    }

    fn reset(&mut self) {
        Mixer::reset(self);
    }
}

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
/// use rhizome_resin_audio::graph::{AffineNode, AudioGraph, AudioContext};
/// use rhizome_resin_audio::primitive::LfoNode;
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

// ============================================================================
// Built-in Audio Nodes
// ============================================================================

/// Oscillator node that generates waveforms.
#[derive(Debug, Clone)]
pub struct Oscillator {
    /// Frequency in Hz.
    pub frequency: f32,
    /// Amplitude (0-1).
    pub amplitude: f32,
    /// Waveform type.
    pub waveform: Waveform,
    /// Phase offset (0-1).
    pub phase_offset: f32,
}

/// Waveform types for oscillators.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Waveform {
    /// Sine wave.
    #[default]
    Sine,
    /// Square wave.
    Square,
    /// Sawtooth wave.
    Saw,
    /// Triangle wave.
    Triangle,
    /// Pulse wave with duty cycle (0-100%).
    Pulse(u8),
}

impl Default for Oscillator {
    fn default() -> Self {
        Self {
            frequency: 440.0,
            amplitude: 1.0,
            waveform: Waveform::Sine,
            phase_offset: 0.0,
        }
    }
}

impl Oscillator {
    /// Creates a new sine oscillator.
    pub fn sine(frequency: f32) -> Self {
        Self {
            frequency,
            ..Default::default()
        }
    }

    /// Creates a new square oscillator.
    pub fn square(frequency: f32) -> Self {
        Self {
            frequency,
            waveform: Waveform::Square,
            ..Default::default()
        }
    }

    /// Creates a new sawtooth oscillator.
    pub fn saw(frequency: f32) -> Self {
        Self {
            frequency,
            waveform: Waveform::Saw,
            ..Default::default()
        }
    }

    /// Creates a new triangle oscillator.
    pub fn triangle(frequency: f32) -> Self {
        Self {
            frequency,
            waveform: Waveform::Triangle,
            ..Default::default()
        }
    }
}

impl AudioNode for Oscillator {
    fn process(&mut self, _input: f32, ctx: &AudioContext) -> f32 {
        let phase = osc::freq_to_phase(self.frequency, ctx.time) + self.phase_offset;

        let raw = match self.waveform {
            Waveform::Sine => osc::sine(phase),
            Waveform::Square => osc::square(phase),
            Waveform::Saw => osc::saw(phase),
            Waveform::Triangle => osc::triangle(phase),
            Waveform::Pulse(duty) => osc::pulse(phase, duty as f32 / 100.0),
        };

        raw * self.amplitude
    }
}

/// Affine transform node: output = input * gain + offset.
///
/// This is the canonical linear transform node. Use the constructors for common cases:
/// - `AffineNode::gain(g)` - multiply by g (equivalent to old `Gain`)
/// - `AffineNode::offset(o)` - add o (equivalent to old `Offset`)
/// - `AffineNode::identity()` - pass through unchanged (equivalent to old `PassThrough`)
///
/// Affine nodes compose naturally via `then()`:
/// ```
/// # use rhizome_resin_audio::graph::AffineNode;
/// let a = AffineNode::gain(2.0);      // y = 2x
/// let b = AffineNode::offset(1.0);    // z = y + 1
/// let c = a.then(b);                  // z = 2x + 1
/// assert_eq!(c.gain, 2.0);
/// assert_eq!(c.offset, 1.0);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct AffineNode {
    /// Multiplicative gain factor.
    pub gain: f32,
    /// Additive offset.
    pub offset: f32,
}

impl AffineNode {
    /// Parameter index for gain.
    pub const PARAM_GAIN: usize = 0;
    /// Parameter index for offset.
    pub const PARAM_OFFSET: usize = 1;

    const PARAMS: &'static [ParamDescriptor] = &[
        ParamDescriptor::new("gain", 1.0, 0.0, 10.0),
        ParamDescriptor::new("offset", 0.0, -10.0, 10.0),
    ];

    /// Create an affine node with explicit gain and offset.
    pub fn new(gain: f32, offset: f32) -> Self {
        Self { gain, offset }
    }

    /// Create a pure gain (multiply) node: output = input * value.
    pub fn gain(value: f32) -> Self {
        Self {
            gain: value,
            offset: 0.0,
        }
    }

    /// Create a pure offset (add) node: output = input + value.
    pub fn offset(value: f32) -> Self {
        Self {
            gain: 1.0,
            offset: value,
        }
    }

    /// Create an identity transform (pass through): output = input.
    pub fn identity() -> Self {
        Self {
            gain: 1.0,
            offset: 0.0,
        }
    }

    /// Returns true if this is effectively an identity (no-op).
    pub fn is_identity(&self) -> bool {
        (self.gain - 1.0).abs() < 1e-10 && self.offset.abs() < 1e-10
    }

    /// Returns true if this is a pure gain (no offset).
    pub fn is_pure_gain(&self) -> bool {
        self.offset.abs() < 1e-10
    }

    /// Returns true if this is a pure offset (gain = 1).
    pub fn is_pure_offset(&self) -> bool {
        (self.gain - 1.0).abs() < 1e-10
    }

    /// Compose two affine transforms: self followed by other.
    ///
    /// If self is `y = ax + b` and other is `z = cy + d`, then
    /// the composed transform is `z = c(ax + b) + d = (ca)x + (cb + d)`.
    pub fn then(self, other: Self) -> Self {
        Self {
            gain: other.gain * self.gain,
            offset: other.gain * self.offset + other.offset,
        }
    }
}

impl Default for AffineNode {
    fn default() -> Self {
        Self::identity()
    }
}

impl AudioNode for AffineNode {
    fn process(&mut self, input: f32, _ctx: &AudioContext) -> f32 {
        input.mul_add(self.gain, self.offset)
    }

    fn params(&self) -> &'static [ParamDescriptor] {
        Self::PARAMS
    }

    fn set_param(&mut self, index: usize, value: f32) {
        match index {
            Self::PARAM_GAIN => self.gain = value,
            Self::PARAM_OFFSET => self.offset = value,
            _ => {}
        }
    }

    fn get_param(&self, index: usize) -> Option<f32> {
        match index {
            Self::PARAM_GAIN => Some(self.gain),
            Self::PARAM_OFFSET => Some(self.offset),
            _ => None,
        }
    }
}

/// Clipping/saturation node.
#[derive(Debug, Clone, Copy)]
pub struct Clip {
    /// Minimum output value.
    pub min: f32,
    /// Maximum output value.
    pub max: f32,
}

impl Clip {
    /// Create a new clip node with min/max bounds.
    pub fn new(min: f32, max: f32) -> Self {
        Self { min, max }
    }

    /// Create a symmetric clip node (-threshold to +threshold).
    pub fn symmetric(threshold: f32) -> Self {
        Self::new(-threshold, threshold)
    }
}

impl Default for Clip {
    fn default() -> Self {
        Self::symmetric(1.0)
    }
}

impl AudioNode for Clip {
    fn process(&mut self, input: f32, _ctx: &AudioContext) -> f32 {
        input.clamp(self.min, self.max)
    }

    fn get_param(&self, index: usize) -> Option<f32> {
        match index {
            0 => Some(self.min),
            1 => Some(self.max),
            _ => None,
        }
    }
}

/// Soft clipping (tanh saturation).
#[derive(Debug, Clone, Copy)]
pub struct SoftClip {
    /// Drive amount (higher = more saturation).
    pub drive: f32,
}

impl SoftClip {
    /// Create a new soft clip node.
    pub fn new(drive: f32) -> Self {
        Self { drive }
    }
}

impl Default for SoftClip {
    fn default() -> Self {
        Self { drive: 1.0 }
    }
}

impl AudioNode for SoftClip {
    fn process(&mut self, input: f32, _ctx: &AudioContext) -> f32 {
        (input * self.drive).tanh()
    }
}

// ============================================================================
// Wrapper nodes for existing filter types
// ============================================================================

/// Wrapper for LowPass filter.
pub struct LowPassNode(pub LowPass);

impl LowPassNode {
    /// Create a new low-pass filter node.
    pub fn new(cutoff: f32, sample_rate: f32) -> Self {
        Self(LowPass::new(cutoff, sample_rate))
    }
}

impl AudioNode for LowPassNode {
    fn process(&mut self, input: f32, _ctx: &AudioContext) -> f32 {
        self.0.process(input)
    }

    fn reset(&mut self) {
        self.0.reset();
    }
}

/// Wrapper for HighPass filter.
pub struct HighPassNode(pub HighPass);

impl HighPassNode {
    /// Create a new high-pass filter node.
    pub fn new(cutoff: f32, sample_rate: f32) -> Self {
        Self(HighPass::new(cutoff, sample_rate))
    }
}

impl AudioNode for HighPassNode {
    fn process(&mut self, input: f32, _ctx: &AudioContext) -> f32 {
        self.0.process(input)
    }

    fn reset(&mut self) {
        self.0.reset();
    }
}

/// Wrapper for Biquad filter.
pub struct BiquadNode(pub Biquad);

impl BiquadNode {
    /// Create a low-pass biquad filter node.
    pub fn lowpass(cutoff: f32, q: f32, sample_rate: f32) -> Self {
        Self(Biquad::lowpass(cutoff, q, sample_rate))
    }

    /// Create a high-pass biquad filter node.
    pub fn highpass(cutoff: f32, q: f32, sample_rate: f32) -> Self {
        Self(Biquad::highpass(cutoff, q, sample_rate))
    }

    /// Create a band-pass biquad filter node.
    pub fn bandpass(center: f32, q: f32, sample_rate: f32) -> Self {
        Self(Biquad::bandpass(center, q, sample_rate))
    }

    /// Create a notch biquad filter node.
    pub fn notch(center: f32, q: f32, sample_rate: f32) -> Self {
        Self(Biquad::notch(center, q, sample_rate))
    }
}

impl AudioNode for BiquadNode {
    fn process(&mut self, input: f32, _ctx: &AudioContext) -> f32 {
        self.0.process(input)
    }

    fn reset(&mut self) {
        self.0.reset();
    }
}

/// Wrapper for Delay.
pub struct DelayNode(pub Delay);

impl DelayNode {
    /// Create a new delay node with sample counts.
    pub fn new(max_samples: usize, delay_samples: usize) -> Self {
        Self(Delay::new(max_samples, delay_samples))
    }

    /// Create a new delay node with time values.
    pub fn from_time(max_seconds: f32, delay_seconds: f32, sample_rate: f32) -> Self {
        Self(Delay::from_time(max_seconds, delay_seconds, sample_rate))
    }
}

impl AudioNode for DelayNode {
    fn process(&mut self, input: f32, _ctx: &AudioContext) -> f32 {
        self.0.process(input)
    }

    fn reset(&mut self) {
        self.0.clear();
    }
}

/// Wrapper for FeedbackDelay.
pub struct FeedbackDelayNode(pub FeedbackDelay);

impl FeedbackDelayNode {
    /// Create a new feedback delay node with sample counts.
    pub fn new(max_samples: usize, delay_samples: usize, feedback: f32) -> Self {
        Self(FeedbackDelay::new(max_samples, delay_samples, feedback))
    }

    /// Create a new feedback delay node with time values.
    pub fn from_time(
        max_seconds: f32,
        delay_seconds: f32,
        feedback: f32,
        sample_rate: f32,
    ) -> Self {
        Self(FeedbackDelay::from_time(
            max_seconds,
            delay_seconds,
            feedback,
            sample_rate,
        ))
    }
}

impl AudioNode for FeedbackDelayNode {
    fn process(&mut self, input: f32, _ctx: &AudioContext) -> f32 {
        self.0.process(input)
    }

    fn reset(&mut self) {
        self.0.clear();
    }
}

// ============================================================================
// Envelope nodes
// ============================================================================

/// ADSR envelope as an amplitude modulator.
pub struct AdsrNode {
    env: Adsr,
}

impl AdsrNode {
    /// Create a new ADSR envelope node.
    pub fn new(attack: f32, decay: f32, sustain: f32, release: f32) -> Self {
        Self {
            env: Adsr::with_params(attack, decay, sustain, release),
        }
    }

    /// Trigger the envelope (note on).
    pub fn trigger(&mut self) {
        self.env.trigger();
    }

    /// Release the envelope (note off).
    pub fn release(&mut self) {
        self.env.release();
    }

    /// Returns true if the envelope is still active.
    pub fn is_active(&self) -> bool {
        self.env.is_active()
    }
}

impl AudioNode for AdsrNode {
    fn process(&mut self, input: f32, ctx: &AudioContext) -> f32 {
        let env_value = self.env.process(ctx.dt);
        input * env_value
    }

    fn reset(&mut self) {
        self.env.reset();
    }
}

/// AR envelope as an amplitude modulator.
pub struct ArNode {
    env: Ar,
}

impl ArNode {
    /// Create a new AR envelope node.
    pub fn new(attack: f32, release: f32) -> Self {
        Self {
            env: Ar::new(attack, release),
        }
    }

    /// Trigger the envelope.
    pub fn trigger(&mut self) {
        self.env.trigger();
    }
}

impl AudioNode for ArNode {
    fn process(&mut self, input: f32, ctx: &AudioContext) -> f32 {
        let env_value = self.env.process(ctx.dt);
        input * env_value
    }

    fn reset(&mut self) {
        self.env.reset();
    }
}

/// LFO as a modulation source.
pub struct LfoNode {
    lfo: Lfo,
}

impl LfoNode {
    /// Create a new LFO node with the given frequency.
    pub fn new(frequency: f32) -> Self {
        Self {
            lfo: Lfo::with_frequency(frequency),
        }
    }
}

impl AudioNode for LfoNode {
    fn process(&mut self, input: f32, ctx: &AudioContext) -> f32 {
        let mod_value = self.lfo.process(ctx.dt);
        input * mod_value
    }

    fn reset(&mut self) {
        self.lfo.reset();
    }
}

// ============================================================================
// Utility nodes
// ============================================================================

/// Ring modulation (multiply two signals).
pub struct RingMod {
    modulator: Box<dyn AudioNode>,
}

impl RingMod {
    /// Create a new ring modulator with the given modulator node.
    pub fn new<N: AudioNode + 'static>(modulator: N) -> Self {
        Self {
            modulator: Box::new(modulator),
        }
    }
}

impl AudioNode for RingMod {
    fn process(&mut self, input: f32, ctx: &AudioContext) -> f32 {
        let mod_signal = self.modulator.process(0.0, ctx);
        input * mod_signal
    }

    fn reset(&mut self) {
        self.modulator.reset();
    }
}

/// Outputs silence.
#[derive(Debug, Clone, Copy, Default)]
pub struct Silence;

impl AudioNode for Silence {
    fn process(&mut self, _input: f32, _ctx: &AudioContext) -> f32 {
        0.0
    }
}

/// Constant value output.
#[derive(Debug, Clone, Copy)]
pub struct Constant(pub f32);

impl AudioNode for Constant {
    fn process(&mut self, _input: f32, _ctx: &AudioContext) -> f32 {
        self.0
    }

    fn get_param(&self, index: usize) -> Option<f32> {
        if index == 0 { Some(self.0) } else { None }
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
}
