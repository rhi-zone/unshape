//! Audio processing graph and signal chain.
//!
//! Provides a flexible way to connect audio processors into chains and graphs.

use crate::envelope::{Adsr, Ar, Lfo};
use crate::filter::{Biquad, Delay, FeedbackDelay, HighPass, LowPass};
use crate::osc;

/// Audio processing context passed to nodes.
#[derive(Debug, Clone, Copy)]
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

/// Audio graph with parameter modulation support.
///
/// Unlike [`Chain`] which is linear, `AudioGraph` supports arbitrary routing
/// and parameter modulation (connecting one node's output to another's parameter).
///
/// # Example
///
/// ```
/// use rhizome_resin_audio::graph::{AudioGraph, AudioContext};
/// use rhizome_resin_audio::primitive::{LfoNode, DelayNode, GainNode};
///
/// let mut graph = AudioGraph::new();
///
/// // Build a simple tremolo: LFO modulates gain
/// let lfo = graph.add(LfoNode::with_freq(5.0, 44100.0));
/// let gain = graph.add(GainNode::new(1.0));
///
/// // Audio path: input → gain → output
/// graph.connect_input(gain);
/// graph.set_output(gain);
///
/// // Modulation: LFO → gain parameter (base=0.5, scale=0.5 means 0-1 range)
/// graph.modulate(lfo, gain, GainNode::PARAM_GAIN, 0.5, 0.5);
///
/// // Process
/// let ctx = AudioContext::new(44100.0);
/// let output = graph.process(1.0, &ctx);
/// ```
pub struct AudioGraph {
    nodes: Vec<Box<dyn AudioNode>>,
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
}

impl AudioGraph {
    /// Creates an empty audio graph.
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            audio_wires: Vec::new(),
            param_wires: Vec::new(),
            input_node: None,
            output_node: None,
            outputs: Vec::new(),
        }
    }

    /// Adds a node to the graph and returns its index.
    pub fn add<N: AudioNode + 'static>(&mut self, node: N) -> NodeIndex {
        let index = self.nodes.len();
        self.nodes.push(Box::new(node));
        self.outputs.push(0.0);
        index
    }

    /// Connects audio output of one node to input of another.
    pub fn connect(&mut self, from: NodeIndex, to: NodeIndex) {
        self.audio_wires.push(AudioWire { from, to });
    }

    /// Connects external input to a node.
    pub fn connect_input(&mut self, to: NodeIndex) {
        self.input_node = Some(to);
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

    /// Processes one sample through the graph.
    ///
    /// Evaluates nodes in index order. Nodes receive the sum of all connected
    /// inputs. Parameters are modulated before each node processes.
    pub fn process(&mut self, input: f32, ctx: &AudioContext) -> f32 {
        // Clear outputs
        for out in &mut self.outputs {
            *out = 0.0;
        }

        // Process each node in order
        for i in 0..self.nodes.len() {
            // Apply parameter modulation
            for wire in &self.param_wires {
                if wire.to == i {
                    let mod_value = self.outputs[wire.from];
                    let param_value = wire.base + mod_value * wire.scale;
                    self.nodes[i].set_param(wire.param, param_value);
                }
            }

            // Gather audio input
            let mut node_input = 0.0;

            // External input
            if self.input_node == Some(i) {
                node_input += input;
            }

            // Inputs from other nodes
            for wire in &self.audio_wires {
                if wire.to == i {
                    node_input += self.outputs[wire.from];
                }
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

/// Gain node that scales the signal.
#[derive(Debug, Clone, Copy)]
pub struct Gain {
    /// Gain multiplier.
    pub value: f32,
}

impl Gain {
    /// Create a new gain node.
    pub fn new(value: f32) -> Self {
        Self { value }
    }
}

impl AudioNode for Gain {
    fn process(&mut self, input: f32, _ctx: &AudioContext) -> f32 {
        input * self.value
    }
}

/// DC offset node.
#[derive(Debug, Clone, Copy)]
pub struct Offset {
    /// Offset value to add.
    pub value: f32,
}

impl Offset {
    /// Create a new offset node.
    pub fn new(value: f32) -> Self {
        Self { value }
    }
}

impl AudioNode for Offset {
    fn process(&mut self, input: f32, _ctx: &AudioContext) -> f32 {
        input + self.value
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

/// Passes through input unchanged (useful as placeholder).
#[derive(Debug, Clone, Copy, Default)]
pub struct PassThrough;

impl AudioNode for PassThrough {
    fn process(&mut self, input: f32, _ctx: &AudioContext) -> f32 {
        input
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
        let mut chain = Chain::new().with(Gain::new(0.5));
        let ctx = AudioContext::new(44100.0);

        let output = chain.process(1.0, &ctx);
        assert!((output - 0.5).abs() < 0.0001);
    }

    #[test]
    fn test_chain_multiple_nodes() {
        let mut chain = Chain::new()
            .with(Gain::new(2.0))
            .with(Offset::new(1.0))
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
        use crate::primitive::GainNode;

        let mut graph = AudioGraph::new();
        let gain = graph.add(GainNode::new(1.0));
        graph.connect_input(gain);
        graph.set_output(gain);

        let ctx = AudioContext::new(44100.0);
        let out = graph.process(0.5, &ctx);
        assert!((out - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_audio_graph_gain() {
        use crate::primitive::GainNode;

        let mut graph = AudioGraph::new();
        let gain = graph.add(GainNode::new(2.0));
        graph.connect_input(gain);
        graph.set_output(gain);

        let ctx = AudioContext::new(44100.0);
        let out = graph.process(0.5, &ctx);
        assert!((out - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_audio_graph_chain() {
        use crate::primitive::GainNode;

        let mut graph = AudioGraph::new();
        let gain1 = graph.add(GainNode::new(2.0));
        let gain2 = graph.add(GainNode::new(0.5));

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
        use crate::primitive::{GainNode, LfoNode};

        let mut graph = AudioGraph::new();

        // LFO modulates gain
        let lfo = graph.add(LfoNode::with_freq(10.0, 44100.0));
        let gain = graph.add(GainNode::new(1.0));

        graph.connect_input(gain);
        graph.set_output(gain);

        // Modulate gain: base=0.5, scale=0.5 (so gain varies 0-1)
        graph.modulate(lfo, gain, GainNode::PARAM_GAIN, 0.5, 0.5);

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
        use crate::primitive::GainNode;

        let mut graph = AudioGraph::new();
        let lfo = graph.add(Oscillator::sine(5.0)); // Use oscillator as modulator
        let gain = graph.add(GainNode::new(1.0));

        graph.connect_input(gain);
        graph.set_output(gain);

        // Modulate by name
        graph.modulate_named(lfo, gain, "gain", 0.5, 0.5);

        assert_eq!(graph.param_wires.len(), 1);
        assert_eq!(graph.param_wires[0].param, GainNode::PARAM_GAIN);
    }

    #[test]
    fn test_audio_graph_as_audio_node() {
        use crate::primitive::GainNode;

        // AudioGraph implements AudioNode, so can be nested in Chain
        let mut inner = AudioGraph::new();
        let gain = inner.add(GainNode::new(2.0));
        inner.connect_input(gain);
        inner.set_output(gain);

        let mut chain = Chain::new().with(inner).with(GainNode::new(0.5));

        let ctx = AudioContext::new(44100.0);
        let out = chain.process(1.0, &ctx);
        // 1.0 * 2.0 * 0.5 = 1.0
        assert!((out - 1.0).abs() < 0.001);
    }
}
