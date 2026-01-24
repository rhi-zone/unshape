//! Low-level audio primitives for building effects.
//!
//! These are building blocks, not standalone effects. They are concrete
//! generic types designed for composition within effect structs.
//!
//! # Primitives
//!
//! - [`DelayLine`] - Circular buffer delay with optional interpolation
//! - [`PhaseOsc`] - Phase accumulator with waveform generation
//! - [`EnvelopeFollower`] - Attack/release envelope tracking
//! - [`Allpass1`] - First-order allpass filter (for phasers)
//! - [`Smoother`] - One-pole parameter smoothing

use std::f32::consts::TAU;

// ============================================================================
// DelayLine
// ============================================================================

/// Circular buffer delay line with optional interpolation.
///
/// Use `DelayLine<false>` for fixed integer delays (reverb, simple echo).
/// Use `DelayLine<true>` for fractional/modulated delays (chorus, flanger).
///
/// # Example
///
/// ```
/// use unshape_audio::primitive::DelayLine;
///
/// // Simple delay (no interpolation)
/// let mut delay: DelayLine<false> = DelayLine::new(1000);
/// delay.write(1.0);
/// let output = delay.read(500); // Read 500 samples behind
///
/// // Interpolating delay (for modulation)
/// let mut mod_delay: DelayLine<true> = DelayLine::new(1000);
/// mod_delay.write(1.0);
/// let output = mod_delay.read_interp(500.5); // Fractional delay
/// ```
pub struct DelayLine<const INTERP: bool = false> {
    buffer: Vec<f32>,
    write_pos: usize,
}

impl<const INTERP: bool> DelayLine<INTERP> {
    /// Create a new delay line with given maximum size in samples.
    pub fn new(max_samples: usize) -> Self {
        Self {
            buffer: vec![0.0; max_samples.max(1)],
            write_pos: 0,
        }
    }

    /// Create a delay line sized for a maximum delay time.
    pub fn from_time(max_seconds: f32, sample_rate: f32) -> Self {
        let samples = (max_seconds * sample_rate).ceil() as usize;
        Self::new(samples)
    }

    /// Write a sample to the delay line and advance the write head.
    #[inline]
    pub fn write(&mut self, sample: f32) {
        self.buffer[self.write_pos] = sample;
        self.write_pos = (self.write_pos + 1) % self.buffer.len();
    }

    /// Clear the buffer to silence.
    pub fn clear(&mut self) {
        self.buffer.fill(0.0);
    }

    /// Get the buffer length (maximum delay in samples).
    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    /// Check if buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }
}

impl DelayLine<false> {
    /// Read at integer delay (samples behind write head).
    #[inline]
    pub fn read(&self, delay_samples: usize) -> f32 {
        let read_pos = (self.write_pos + self.buffer.len() - delay_samples) % self.buffer.len();
        self.buffer[read_pos]
    }

    /// Read and write in one operation.
    #[inline]
    pub fn tap(&mut self, input: f32, delay_samples: usize) -> f32 {
        let output = self.read(delay_samples);
        self.write(input);
        output
    }
}

impl DelayLine<true> {
    /// Read at fractional delay with linear interpolation.
    #[inline]
    pub fn read_interp(&self, delay_samples: f32) -> f32 {
        let read_pos = self.write_pos as f32 - delay_samples;
        let read_pos = if read_pos < 0.0 {
            read_pos + self.buffer.len() as f32
        } else {
            read_pos
        };

        let idx0 = read_pos.floor() as usize % self.buffer.len();
        let idx1 = (idx0 + 1) % self.buffer.len();
        let frac = read_pos.fract();

        self.buffer[idx0] * (1.0 - frac) + self.buffer[idx1] * frac
    }

    /// Read and write in one operation with interpolation.
    #[inline]
    pub fn tap_interp(&mut self, input: f32, delay_samples: f32) -> f32 {
        let output = self.read_interp(delay_samples);
        self.write(input);
        output
    }
}

// ============================================================================
// PhaseOsc
// ============================================================================

/// Phase accumulator with waveform generation.
///
/// Stores only the phase (0.0 to 1.0). Caller provides the phase increment
/// per sample, typically `rate / sample_rate`.
///
/// # Example
///
/// ```
/// use unshape_audio::primitive::PhaseOsc;
///
/// let mut osc = PhaseOsc::new();
/// let phase_inc = 5.0 / 44100.0; // 5 Hz at 44.1kHz
///
/// for _ in 0..44100 {
///     let lfo_value = osc.sine();
///     osc.advance(phase_inc);
/// }
/// ```
pub struct PhaseOsc {
    phase: f32,
}

impl PhaseOsc {
    /// Create a new oscillator at phase 0.
    pub fn new() -> Self {
        Self { phase: 0.0 }
    }

    /// Create with initial phase (0.0 to 1.0).
    pub fn with_phase(phase: f32) -> Self {
        Self { phase }
    }

    /// Advance phase by increment (typically rate / sample_rate).
    #[inline]
    pub fn advance(&mut self, increment: f32) {
        self.phase += increment;
        if self.phase >= 1.0 {
            self.phase -= 1.0;
        } else if self.phase < 0.0 {
            self.phase += 1.0;
        }
    }

    /// Get current phase (0.0 to 1.0).
    #[inline]
    pub fn phase(&self) -> f32 {
        self.phase
    }

    /// Reset phase to 0.
    pub fn reset(&mut self) {
        self.phase = 0.0;
    }

    /// Sine wave (-1.0 to 1.0).
    #[inline]
    pub fn sine(&self) -> f32 {
        (self.phase * TAU).sin()
    }

    /// Unipolar sine (0.0 to 1.0).
    #[inline]
    pub fn sine_uni(&self) -> f32 {
        self.sine() * 0.5 + 0.5
    }

    /// Triangle wave (-1.0 to 1.0).
    #[inline]
    pub fn triangle(&self) -> f32 {
        let t = self.phase * 4.0;
        if t < 1.0 {
            t
        } else if t < 3.0 {
            2.0 - t
        } else {
            t - 4.0
        }
    }

    /// Sawtooth wave (-1.0 to 1.0).
    #[inline]
    pub fn saw(&self) -> f32 {
        self.phase * 2.0 - 1.0
    }

    /// Square wave (-1.0 to 1.0).
    #[inline]
    pub fn square(&self) -> f32 {
        if self.phase < 0.5 { 1.0 } else { -1.0 }
    }
}

impl Default for PhaseOsc {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// EnvelopeFollower
// ============================================================================

/// Attack/release envelope follower for dynamics processing.
///
/// Tracks the amplitude of an input signal with configurable attack
/// and release times.
///
/// # Example
///
/// ```
/// use unshape_audio::primitive::EnvelopeFollower;
///
/// let mut env = EnvelopeFollower::new(0.01, 0.1, 44100.0);
///
/// let level = env.process(0.8); // Track input amplitude
/// ```
pub struct EnvelopeFollower {
    level: f32,
    attack_coeff: f32,
    release_coeff: f32,
}

impl EnvelopeFollower {
    /// Create with attack/release times in seconds.
    pub fn new(attack: f32, release: f32, sample_rate: f32) -> Self {
        Self {
            level: 0.0,
            attack_coeff: Self::time_to_coeff(attack, sample_rate),
            release_coeff: Self::time_to_coeff(release, sample_rate),
        }
    }

    /// Convert time constant to filter coefficient.
    fn time_to_coeff(time: f32, sample_rate: f32) -> f32 {
        if time <= 0.0 {
            0.0
        } else {
            (-1.0 / (time * sample_rate)).exp()
        }
    }

    /// Process input and return smoothed envelope level.
    #[inline]
    pub fn process(&mut self, input: f32) -> f32 {
        let input_level = input.abs();
        let coeff = if input_level > self.level {
            self.attack_coeff
        } else {
            self.release_coeff
        };
        self.level = coeff * self.level + (1.0 - coeff) * input_level;
        self.level
    }

    /// Process with target level (for sidechain or external detection).
    #[inline]
    pub fn process_target(&mut self, target: f32) -> f32 {
        let coeff = if target > self.level {
            self.attack_coeff
        } else {
            self.release_coeff
        };
        self.level = coeff * self.level + (1.0 - coeff) * target;
        self.level
    }

    /// Get current envelope level.
    pub fn level(&self) -> f32 {
        self.level
    }

    /// Reset envelope to zero.
    pub fn reset(&mut self) {
        self.level = 0.0;
    }

    /// Update attack time.
    pub fn set_attack(&mut self, attack: f32, sample_rate: f32) {
        self.attack_coeff = Self::time_to_coeff(attack, sample_rate);
    }

    /// Update release time.
    pub fn set_release(&mut self, release: f32, sample_rate: f32) {
        self.release_coeff = Self::time_to_coeff(release, sample_rate);
    }
}

// ============================================================================
// Allpass1
// ============================================================================

/// First-order allpass filter for phaser effects.
///
/// Passes all frequencies with equal gain but shifts the phase.
/// Multiple stages create the characteristic phaser notches.
///
/// # Example
///
/// ```
/// use unshape_audio::primitive::Allpass1;
///
/// let mut ap = Allpass1::new();
/// let coeff = 0.5; // Controls notch frequency
/// let output = ap.process(1.0, coeff);
/// ```
pub struct Allpass1 {
    prev_input: f32,
    prev_output: f32,
}

impl Allpass1 {
    /// Create a new allpass filter.
    pub fn new() -> Self {
        Self {
            prev_input: 0.0,
            prev_output: 0.0,
        }
    }

    /// Process a sample with given coefficient.
    ///
    /// Coefficient controls the notch frequency:
    /// `coeff = (tan(PI * freq / sample_rate) - 1) / (tan(...) + 1)`
    #[inline]
    pub fn process(&mut self, input: f32, coeff: f32) -> f32 {
        let output = coeff * input + self.prev_input - coeff * self.prev_output;
        self.prev_input = input;
        self.prev_output = output;
        output
    }

    /// Reset filter state.
    pub fn clear(&mut self) {
        self.prev_input = 0.0;
        self.prev_output = 0.0;
    }
}

impl Default for Allpass1 {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Smoother
// ============================================================================

/// One-pole smoothing filter for parameter interpolation.
///
/// Use this to smooth parameter changes and avoid clicks/zippers.
///
/// # Example
///
/// ```
/// use unshape_audio::primitive::Smoother;
///
/// let mut smooth = Smoother::new(0.01, 44100.0); // 10ms smoothing
/// smooth.set_target(1.0);
///
/// for _ in 0..1000 {
///     let value = smooth.next();
///     // Use smoothed value...
/// }
/// ```
pub struct Smoother {
    value: f32,
    target: f32,
    coeff: f32,
}

impl Smoother {
    /// Create with smoothing time in seconds.
    pub fn new(time: f32, sample_rate: f32) -> Self {
        Self {
            value: 0.0,
            target: 0.0,
            coeff: Self::time_to_coeff(time, sample_rate),
        }
    }

    /// Convert time constant to filter coefficient.
    fn time_to_coeff(time: f32, sample_rate: f32) -> f32 {
        if time <= 0.0 {
            0.0
        } else {
            (-1.0 / (time * sample_rate)).exp()
        }
    }

    /// Set the target value to smooth towards.
    pub fn set_target(&mut self, target: f32) {
        self.target = target;
    }

    /// Set value immediately (no smoothing).
    pub fn set_immediate(&mut self, value: f32) {
        self.value = value;
        self.target = value;
    }

    /// Get the next smoothed value.
    #[inline]
    pub fn next(&mut self) -> f32 {
        self.value = self.coeff * self.value + (1.0 - self.coeff) * self.target;
        self.value
    }

    /// Get current value without advancing.
    pub fn value(&self) -> f32 {
        self.value
    }

    /// Check if smoothing is complete (value ≈ target).
    pub fn is_settled(&self) -> bool {
        (self.value - self.target).abs() < 1e-6
    }

    /// Reset to zero.
    pub fn reset(&mut self) {
        self.value = 0.0;
        self.target = 0.0;
    }
}

// ============================================================================
// Mix helpers
// ============================================================================

/// Dry/wet mixing utilities.
pub struct Mix;

impl Mix {
    /// Linear dry/wet blend. mix=0 is dry, mix=1 is wet.
    #[inline]
    pub fn blend(dry: f32, wet: f32, mix: f32) -> f32 {
        dry * (1.0 - mix) + wet * mix
    }

    /// Equal-power crossfade (smoother for audio).
    #[inline]
    pub fn crossfade(dry: f32, wet: f32, mix: f32) -> f32 {
        let angle = mix * std::f32::consts::FRAC_PI_2;
        dry * angle.cos() + wet * angle.sin()
    }
}

// ============================================================================
// Tests
// ============================================================================

// ============================================================================
// AudioNode implementations
// ============================================================================

use crate::graph::{AudioContext, AudioNode, ParamDescriptor};

/// Delay node wrapping DelayLine<true> for use in graphs.
///
/// Parameters:
/// - `time`: Delay time in samples (modulatable)
pub struct DelayNode {
    delay: DelayLine<true>,
    time: f32,
    feedback: f32,
}

impl DelayNode {
    /// Parameter index for delay time.
    pub const PARAM_TIME: usize = 0;
    /// Parameter index for feedback.
    pub const PARAM_FEEDBACK: usize = 1;

    const PARAMS: &'static [ParamDescriptor] = &[
        ParamDescriptor::new("time", 100.0, 0.0, 44100.0),
        ParamDescriptor::new("feedback", 0.0, -0.99, 0.99),
    ];

    /// Create a delay node with given max buffer size.
    pub fn new(max_samples: usize) -> Self {
        Self {
            delay: DelayLine::new(max_samples),
            time: 100.0,
            feedback: 0.0,
        }
    }

    /// Create from max time in seconds.
    pub fn from_time(max_seconds: f32, sample_rate: f32) -> Self {
        Self {
            delay: DelayLine::from_time(max_seconds, sample_rate),
            time: 100.0,
            feedback: 0.0,
        }
    }

    /// Set delay time in samples.
    pub fn set_time(&mut self, samples: f32) {
        self.time = samples.max(0.0);
    }

    /// Set feedback amount (-0.99 to 0.99).
    pub fn set_feedback(&mut self, feedback: f32) {
        self.feedback = feedback.clamp(-0.99, 0.99);
    }
}

impl AudioNode for DelayNode {
    fn process(&mut self, input: f32, _ctx: &AudioContext) -> f32 {
        let output = self.delay.read_interp(self.time);
        self.delay.write(input + output * self.feedback);
        output
    }

    fn reset(&mut self) {
        self.delay.clear();
    }

    fn params(&self) -> &'static [ParamDescriptor] {
        Self::PARAMS
    }

    fn get_param(&self, index: usize) -> Option<f32> {
        match index {
            Self::PARAM_TIME => Some(self.time),
            Self::PARAM_FEEDBACK => Some(self.feedback),
            _ => None,
        }
    }

    fn set_param(&mut self, index: usize, value: f32) {
        match index {
            Self::PARAM_TIME => self.time = value.max(0.0),
            Self::PARAM_FEEDBACK => self.feedback = value.clamp(-0.99, 0.99),
            _ => {}
        }
    }
}

/// LFO node wrapping PhaseOsc for use in graphs.
///
/// Outputs a modulation signal (-1 to 1 by default).
///
/// Parameters:
/// - `rate`: Phase increment per sample (freq / sample_rate)
pub struct LfoNode {
    osc: PhaseOsc,
    rate: f32,
}

impl LfoNode {
    /// Parameter index for rate.
    pub const PARAM_RATE: usize = 0;

    const PARAMS: &'static [ParamDescriptor] = &[ParamDescriptor::new("rate", 0.0001, 0.0, 0.1)];

    /// Create an LFO node.
    pub fn new() -> Self {
        Self {
            osc: PhaseOsc::new(),
            rate: 0.0001, // ~4.4 Hz at 44.1kHz
        }
    }

    /// Create with frequency and sample rate.
    pub fn with_freq(freq: f32, sample_rate: f32) -> Self {
        Self {
            osc: PhaseOsc::new(),
            rate: freq / sample_rate,
        }
    }

    /// Set rate directly (phase increment per sample).
    pub fn set_rate(&mut self, rate: f32) {
        self.rate = rate;
    }

    /// Set frequency given sample rate.
    pub fn set_freq(&mut self, freq: f32, sample_rate: f32) {
        self.rate = freq / sample_rate;
    }
}

impl Default for LfoNode {
    fn default() -> Self {
        Self::new()
    }
}

impl AudioNode for LfoNode {
    fn process(&mut self, _input: f32, _ctx: &AudioContext) -> f32 {
        let out = self.osc.sine();
        self.osc.advance(self.rate);
        out
    }

    fn reset(&mut self) {
        self.osc.reset();
    }

    fn params(&self) -> &'static [ParamDescriptor] {
        Self::PARAMS
    }

    fn set_param(&mut self, index: usize, value: f32) {
        if index == Self::PARAM_RATE {
            self.rate = value;
        }
    }
}

/// Envelope follower node for dynamics processing.
///
/// Outputs the envelope level of the input signal.
pub struct EnvelopeNode {
    follower: EnvelopeFollower,
}

impl EnvelopeNode {
    /// Create an envelope follower with given attack/release times.
    pub fn new(attack: f32, release: f32, sample_rate: f32) -> Self {
        Self {
            follower: EnvelopeFollower::new(attack, release, sample_rate),
        }
    }
}

impl AudioNode for EnvelopeNode {
    fn process(&mut self, input: f32, _ctx: &AudioContext) -> f32 {
        self.follower.process(input)
    }

    fn reset(&mut self) {
        self.follower.reset();
    }
}

/// First-order allpass node for phaser effects.
///
/// Parameters:
/// - `coeff`: Filter coefficient (controls notch frequency)
pub struct AllpassNode {
    filter: Allpass1,
    coeff: f32,
}

impl AllpassNode {
    /// Parameter index for coefficient.
    pub const PARAM_COEFF: usize = 0;

    const PARAMS: &'static [ParamDescriptor] = &[ParamDescriptor::new("coeff", 0.5, -1.0, 1.0)];

    /// Create an allpass node.
    pub fn new() -> Self {
        Self {
            filter: Allpass1::new(),
            coeff: 0.5,
        }
    }

    /// Set the filter coefficient.
    pub fn set_coeff(&mut self, coeff: f32) {
        self.coeff = coeff;
    }
}

impl Default for AllpassNode {
    fn default() -> Self {
        Self::new()
    }
}

impl AudioNode for AllpassNode {
    fn process(&mut self, input: f32, _ctx: &AudioContext) -> f32 {
        self.filter.process(input, self.coeff)
    }

    fn reset(&mut self) {
        self.filter.clear();
    }

    fn params(&self) -> &'static [ParamDescriptor] {
        Self::PARAMS
    }

    fn set_param(&mut self, index: usize, value: f32) {
        if index == Self::PARAM_COEFF {
            self.coeff = value;
        }
    }
}
/// Mix node for dry/wet blending.
///
/// Takes the input as "wet" signal, stores "dry" internally, outputs blend.
/// Note: For graph use, typically connect dry→input of mix node,
/// and modulate the mix parameter.
///
/// Parameters:
/// - `mix`: Blend amount (0 = dry, 1 = wet)
pub struct MixNode {
    mix: f32,
    dry: f32,
}

impl MixNode {
    /// Parameter index for mix amount.
    pub const PARAM_MIX: usize = 0;
    /// Parameter index for dry signal.
    pub const PARAM_DRY: usize = 1;

    const PARAMS: &'static [ParamDescriptor] = &[
        ParamDescriptor::new("mix", 0.5, 0.0, 1.0),
        ParamDescriptor::new("dry", 0.0, -10.0, 10.0),
    ];

    /// Create a mix node.
    pub fn new(mix: f32) -> Self {
        Self { mix, dry: 0.0 }
    }

    /// Set the mix amount (0 = dry, 1 = wet).
    pub fn set_mix(&mut self, mix: f32) {
        self.mix = mix.clamp(0.0, 1.0);
    }

    /// Set the dry signal (for when dry isn't routed through graph).
    pub fn set_dry(&mut self, dry: f32) {
        self.dry = dry;
    }
}

impl Default for MixNode {
    fn default() -> Self {
        Self::new(0.5)
    }
}

impl AudioNode for MixNode {
    fn process(&mut self, wet: f32, _ctx: &AudioContext) -> f32 {
        Mix::blend(self.dry, wet, self.mix)
    }

    fn params(&self) -> &'static [ParamDescriptor] {
        Self::PARAMS
    }

    fn set_param(&mut self, index: usize, value: f32) {
        match index {
            Self::PARAM_MIX => self.mix = value.clamp(0.0, 1.0),
            Self::PARAM_DRY => self.dry = value,
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_delay_line_simple() {
        let mut delay: DelayLine<false> = DelayLine::new(10);

        // Write some samples
        for i in 0..5 {
            delay.write(i as f32);
        }

        // Read at various delays
        assert_eq!(delay.read(1), 4.0); // 1 sample ago
        assert_eq!(delay.read(2), 3.0); // 2 samples ago
        assert_eq!(delay.read(5), 0.0); // 5 samples ago (first write)
    }

    #[test]
    fn test_delay_line_interp() {
        let mut delay: DelayLine<true> = DelayLine::new(10);

        delay.write(0.0);
        delay.write(1.0);

        // Exact sample
        assert!((delay.read_interp(1.0) - 1.0).abs() < 0.001);

        // Interpolated
        let interp = delay.read_interp(1.5);
        assert!((interp - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_phase_osc_advance() {
        let mut osc = PhaseOsc::new();

        osc.advance(0.25);
        assert!((osc.phase() - 0.25).abs() < 0.001);

        osc.advance(0.25);
        assert!((osc.phase() - 0.5).abs() < 0.001);

        // Wrap around
        osc.advance(0.75);
        assert!((osc.phase() - 0.25).abs() < 0.001);
    }

    #[test]
    fn test_phase_osc_waveforms() {
        let mut osc = PhaseOsc::new();

        // Sine at phase 0 should be 0
        assert!(osc.sine().abs() < 0.001);

        // Sine at phase 0.25 should be 1
        osc = PhaseOsc::with_phase(0.25);
        assert!((osc.sine() - 1.0).abs() < 0.001);

        // Triangle at phase 0 should be 0
        osc = PhaseOsc::new();
        assert!(osc.triangle().abs() < 0.001);

        // Triangle at phase 0.25 should be 1
        osc = PhaseOsc::with_phase(0.25);
        assert!((osc.triangle() - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_envelope_follower() {
        let mut env = EnvelopeFollower::new(0.001, 0.01, 44100.0);

        // Attack
        for _ in 0..100 {
            env.process(1.0);
        }
        assert!(env.level() > 0.5);

        // Release
        for _ in 0..1000 {
            env.process(0.0);
        }
        assert!(env.level() < 0.1);
    }

    #[test]
    fn test_allpass1() {
        let mut ap = Allpass1::new();

        // Process some samples
        let mut output = 0.0;
        for _ in 0..100 {
            output = ap.process(1.0, 0.5);
        }

        // Should converge to input for constant input
        assert!((output - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_smoother() {
        let mut smooth = Smoother::new(0.001, 44100.0); // 1ms smoothing
        smooth.set_target(1.0);

        // Should approach target (1ms = 44 samples, need ~5x time constant)
        for _ in 0..500 {
            smooth.next();
        }
        assert!(smooth.value() > 0.9, "value was {}", smooth.value());

        // Should get very close to target
        for _ in 0..5000 {
            smooth.next();
        }
        assert!(
            (smooth.value() - 1.0).abs() < 0.0001,
            "value was {}",
            smooth.value()
        );
    }

    #[test]
    fn test_mix_blend() {
        assert!((Mix::blend(1.0, 0.0, 0.0) - 1.0).abs() < 0.001); // Full dry
        assert!((Mix::blend(1.0, 0.0, 1.0) - 0.0).abs() < 0.001); // Full wet
        assert!((Mix::blend(1.0, 0.0, 0.5) - 0.5).abs() < 0.001); // 50/50
    }

    // ========================================================================
    // AudioNode implementation tests
    // ========================================================================

    #[test]
    fn test_delay_node_audio_node() {
        let ctx = AudioContext::new(44100.0);
        let mut delay = DelayNode::new(1000);
        delay.set_time(100.0);

        // Write impulse
        let out1 = delay.process(1.0, &ctx);
        assert!(out1.abs() < 0.001); // No output yet

        // Process silence until delay time
        for _ in 0..99 {
            delay.process(0.0, &ctx);
        }

        // Should get impulse back
        let out2 = delay.process(0.0, &ctx);
        assert!((out2 - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_delay_node_set_param() {
        let mut delay = DelayNode::new(1000);

        // Set via index
        delay.set_param(DelayNode::PARAM_TIME, 200.0);
        delay.set_param(DelayNode::PARAM_FEEDBACK, 0.5);

        assert!((delay.time - 200.0).abs() < 0.001);
        assert!((delay.feedback - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_lfo_node_audio_node() {
        let ctx = AudioContext::new(44100.0);
        let mut lfo = LfoNode::with_freq(1.0, 44100.0); // 1 Hz

        // Collect one cycle worth of samples
        let mut samples = Vec::new();
        for _ in 0..44100 {
            samples.push(lfo.process(0.0, &ctx));
        }

        // Should have completed ~1 cycle, find min and max
        let min = samples.iter().cloned().fold(f32::MAX, f32::min);
        let max = samples.iter().cloned().fold(f32::MIN, f32::max);

        assert!(min < -0.9);
        assert!(max > 0.9);
    }

    #[test]
    fn test_lfo_node_params() {
        let lfo = LfoNode::new();
        let params = lfo.params();

        assert_eq!(params.len(), 1);
        assert_eq!(params[0].name, "rate");
    }

    #[test]
    fn test_allpass_node_audio_node() {
        let ctx = AudioContext::new(44100.0);
        let mut ap = AllpassNode::new();

        // Process constant input - should converge
        let mut output = 0.0;
        for _ in 0..100 {
            output = ap.process(1.0, &ctx);
        }

        assert!((output - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_allpass_node_set_param() {
        let mut ap = AllpassNode::new();
        ap.set_param(AllpassNode::PARAM_COEFF, 0.7);
        assert!((ap.coeff - 0.7).abs() < 0.001);
    }

    #[test]
    fn test_affine_node_audio_node() {
        use crate::graph::AffineNode;

        let ctx = AudioContext::new(44100.0);
        let mut gain = AffineNode::gain(2.0);

        let out = gain.process(0.5, &ctx);
        assert!((out - 1.0).abs() < 0.001);

        // Modulate gain
        gain.set_param(AffineNode::PARAM_GAIN, 0.5);
        let out2 = gain.process(1.0, &ctx);
        assert!((out2 - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_mix_node_audio_node() {
        let ctx = AudioContext::new(44100.0);
        let mut mix = MixNode::new(0.5);
        mix.set_dry(1.0);

        // 50% mix: (1.0 * 0.5) + (wet * 0.5)
        let out = mix.process(0.0, &ctx); // wet = 0
        assert!((out - 0.5).abs() < 0.001);

        let out2 = mix.process(1.0, &ctx); // wet = 1
        assert!((out2 - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_envelope_node_audio_node() {
        let ctx = AudioContext::new(44100.0);
        let mut env = EnvelopeNode::new(0.001, 0.01, 44100.0);

        // Attack
        for _ in 0..100 {
            env.process(1.0, &ctx);
        }
        assert!(env.follower.level() > 0.5);

        // Release
        for _ in 0..1000 {
            env.process(0.0, &ctx);
        }
        assert!(env.follower.level() < 0.1);
    }
}
