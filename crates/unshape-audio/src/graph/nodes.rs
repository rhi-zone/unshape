use super::{AudioContext, AudioNode, ParamDescriptor};
use crate::envelope::{Adsr, Ar, Lfo};
use crate::filter::{Biquad, Delay, FeedbackDelay, HighPass, LowPass};
use crate::osc;

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
/// # use unshape_audio::graph::AffineNode;
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
