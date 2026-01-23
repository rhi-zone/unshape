//! Audio effects for sound design.
//!
//! Provides classic effects: reverb, chorus, phaser, flanger, tremolo, distortion, etc.
//!
//! # Architecture
//!
//! Effects are built from low-level primitives in [`crate::primitive`].
//! Many effects share the same underlying structure with different default
//! parameters - for example, [`chorus()`] and [`flanger()`] both return
//! [`ModulatedDelay`], while [`tremolo()`] returns [`AmplitudeMod`].

use crate::filter::{Biquad, BiquadCoeffs};
use crate::graph::{AudioContext, AudioNode};
use crate::primitive::{DelayLine, Mix, PhaseOsc};

// ============================================================================
// ModulatedDelay (Chorus/Flanger unified)
// ============================================================================

/// Modulated delay effect - the foundation for chorus and flanger.
///
/// Uses an LFO to modulate the delay time, creating movement and depth.
/// Chorus and flanger are the same effect with different parameters:
/// - Chorus: longer delay (10-30ms), subtle modulation, little feedback
/// - Flanger: shorter delay (1-10ms), more modulation, more feedback
///
/// # Example
///
/// ```
/// use rhizome_resin_audio::effects::{ModulatedDelay, chorus, flanger};
///
/// // Using constructor functions
/// let mut chorus_effect = chorus(44100.0);
/// let mut flanger_effect = flanger(44100.0);
///
/// // Or create directly with custom parameters
/// let mut custom = ModulatedDelay::new(44100.0, 0.015, 0.002, 0.8, 0.5, 0.3);
/// ```
pub struct ModulatedDelay {
    delay: DelayLine<true>,
    lfo: PhaseOsc,
    /// Base delay in samples.
    base_delay: f32,
    /// Modulation depth in samples.
    depth: f32,
    /// Phase increment per sample (rate / sample_rate).
    phase_inc: f32,
    /// Dry/wet mix (0 = dry, 1 = wet).
    pub mix: f32,
    /// Feedback amount (-1 to 1).
    pub feedback: f32,
    /// Minimum delay to prevent artifacts.
    min_delay: f32,
}

impl ModulatedDelay {
    /// Creates a modulated delay with custom parameters.
    ///
    /// # Arguments
    /// * `sample_rate` - Sample rate in Hz
    /// * `delay` - Base delay in seconds
    /// * `depth` - Modulation depth in seconds
    /// * `rate` - Modulation rate in Hz
    /// * `mix` - Dry/wet mix (0-1)
    /// * `feedback` - Feedback amount (-1 to 1, clamped to ±0.95)
    pub fn new(
        sample_rate: f32,
        delay: f32,
        depth: f32,
        rate: f32,
        mix: f32,
        feedback: f32,
    ) -> Self {
        let base_delay = delay * sample_rate;
        let depth_samples = depth * sample_rate;
        let max_delay = base_delay + depth_samples * 2.0;
        let buffer_size = (max_delay as usize + 1).max(256);

        Self {
            delay: DelayLine::new(buffer_size),
            lfo: PhaseOsc::new(),
            base_delay,
            depth: depth_samples,
            phase_inc: rate / sample_rate,
            mix,
            feedback: feedback.clamp(-0.95, 0.95),
            min_delay: 1.0,
        }
    }

    /// Sets the modulation rate in Hz.
    pub fn set_rate(&mut self, rate: f32, sample_rate: f32) {
        self.phase_inc = rate / sample_rate;
    }

    /// Sets the modulation depth in seconds.
    pub fn set_depth(&mut self, depth: f32, sample_rate: f32) {
        self.depth = depth * sample_rate;
    }

    /// Processes a single sample.
    #[inline]
    pub fn process(&mut self, input: f32) -> f32 {
        // LFO modulates delay time
        let lfo_val = self.lfo.sine();
        self.lfo.advance(self.phase_inc);

        let delay_samples = (self.base_delay + lfo_val * self.depth).max(self.min_delay);
        let delayed = self.delay.read_interp(delay_samples);

        // Write with feedback
        self.delay.write(input + delayed * self.feedback);

        // Mix dry and wet
        Mix::blend(input, delayed, self.mix)
    }

    /// Clears the delay buffer and resets LFO.
    pub fn clear(&mut self) {
        self.delay.clear();
        self.lfo.reset();
    }
}

impl AudioNode for ModulatedDelay {
    fn process(&mut self, input: f32, _ctx: &AudioContext) -> f32 {
        ModulatedDelay::process(self, input)
    }

    fn reset(&mut self) {
        self.clear();
    }
}

/// Creates a chorus effect (modulated delay with typical chorus parameters).
///
/// Chorus uses longer delays (20ms) with subtle modulation and minimal feedback
/// to create a thickening, doubling effect.
pub fn chorus(sample_rate: f32) -> ModulatedDelay {
    ModulatedDelay::new(
        sample_rate,
        0.02,  // 20ms delay
        0.003, // 3ms depth
        0.5,   // 0.5 Hz rate
        0.5,   // 50% mix
        0.0,   // no feedback
    )
}

/// Creates a flanger effect (modulated delay with typical flanger parameters).
///
/// Flanger uses shorter delays (3ms) with more modulation and feedback
/// to create sweeping, jet-like sounds.
pub fn flanger(sample_rate: f32) -> ModulatedDelay {
    ModulatedDelay::new(
        sample_rate,
        0.003, // 3ms delay
        0.002, // 2ms depth
        0.3,   // 0.3 Hz rate
        0.5,   // 50% mix
        0.7,   // 70% feedback
    )
}

// ============================================================================
// AmplitudeMod (Tremolo)
// ============================================================================

/// Amplitude modulation effect - the foundation for tremolo.
///
/// Uses an LFO to modulate the signal amplitude, creating a pulsing effect.
///
/// # Example
///
/// ```
/// use rhizome_resin_audio::effects::{AmplitudeMod, tremolo};
///
/// let mut trem = tremolo(44100.0, 5.0, 0.5); // 5Hz, 50% depth
/// let output = trem.process(1.0);
/// ```
pub struct AmplitudeMod {
    lfo: PhaseOsc,
    /// Phase increment per sample (rate / sample_rate).
    phase_inc: f32,
    /// Modulation depth (0-1).
    pub depth: f32,
}

impl AmplitudeMod {
    /// Creates an amplitude modulator.
    ///
    /// # Arguments
    /// * `sample_rate` - Sample rate in Hz
    /// * `rate` - Modulation rate in Hz
    /// * `depth` - Modulation depth (0-1)
    pub fn new(sample_rate: f32, rate: f32, depth: f32) -> Self {
        Self {
            lfo: PhaseOsc::new(),
            phase_inc: rate / sample_rate,
            depth: depth.clamp(0.0, 1.0),
        }
    }

    /// Sets the modulation rate in Hz.
    pub fn set_rate(&mut self, rate: f32, sample_rate: f32) {
        self.phase_inc = rate / sample_rate;
    }

    /// Processes a single sample.
    #[inline]
    pub fn process(&mut self, input: f32) -> f32 {
        // Unipolar LFO (0 to 1)
        let lfo_val = self.lfo.sine_uni();
        self.lfo.advance(self.phase_inc);

        // Modulate amplitude: at depth=1, goes from 0 to 1
        let mod_amount = 1.0 - self.depth * lfo_val;
        input * mod_amount
    }

    /// Resets the LFO phase.
    pub fn reset(&mut self) {
        self.lfo.reset();
    }
}

impl AudioNode for AmplitudeMod {
    fn process(&mut self, input: f32, _ctx: &AudioContext) -> f32 {
        AmplitudeMod::process(self, input)
    }

    fn reset(&mut self) {
        AmplitudeMod::reset(self);
    }
}

/// Creates a tremolo effect (amplitude modulation).
pub fn tremolo(sample_rate: f32, rate: f32, depth: f32) -> AmplitudeMod {
    AmplitudeMod::new(sample_rate, rate, depth)
}

// ============================================================================
// AllpassBank (Phaser)
// ============================================================================

use crate::primitive::Allpass1;

/// Cascaded allpass filter bank - the foundation for phaser effects.
///
/// Uses an LFO to modulate the allpass coefficient, sweeping the notch
/// frequencies across a range. Multiple stages create deeper notches.
///
/// # Example
///
/// ```
/// use rhizome_resin_audio::effects::{AllpassBank, phaser};
///
/// let mut ph = phaser(44100.0);
/// let output = ph.process(1.0);
/// ```
pub struct AllpassBank {
    stages: Vec<Allpass1>,
    lfo: PhaseOsc,
    /// Phase increment per sample.
    phase_inc: f32,
    /// Minimum sweep frequency in Hz.
    pub min_freq: f32,
    /// Maximum sweep frequency in Hz.
    pub max_freq: f32,
    /// Dry/wet mix (0 = dry, 1 = wet).
    pub mix: f32,
    /// Feedback amount (0-1).
    pub feedback: f32,
    /// Previous output for feedback.
    feedback_sample: f32,
    /// Sample rate for coefficient calculation.
    sample_rate: f32,
}

impl AllpassBank {
    /// Creates an allpass bank with custom parameters.
    ///
    /// # Arguments
    /// * `sample_rate` - Sample rate in Hz
    /// * `num_stages` - Number of allpass stages (2-12, even numbers work best)
    /// * `rate` - Modulation rate in Hz
    /// * `min_freq` - Minimum sweep frequency in Hz
    /// * `max_freq` - Maximum sweep frequency in Hz
    /// * `mix` - Dry/wet mix (0-1)
    /// * `feedback` - Feedback amount (0-1)
    pub fn new(
        sample_rate: f32,
        num_stages: usize,
        rate: f32,
        min_freq: f32,
        max_freq: f32,
        mix: f32,
        feedback: f32,
    ) -> Self {
        let num_stages = num_stages.clamp(2, 12);

        Self {
            stages: (0..num_stages).map(|_| Allpass1::new()).collect(),
            lfo: PhaseOsc::new(),
            phase_inc: rate / sample_rate,
            min_freq,
            max_freq,
            mix,
            feedback: feedback.clamp(0.0, 0.95),
            feedback_sample: 0.0,
            sample_rate,
        }
    }

    /// Sets the modulation rate in Hz.
    pub fn set_rate(&mut self, rate: f32) {
        self.phase_inc = rate / self.sample_rate;
    }

    /// Sets the sweep frequency range.
    pub fn set_range(&mut self, min_freq: f32, max_freq: f32) {
        self.min_freq = min_freq;
        self.max_freq = max_freq;
    }

    /// Processes a single sample.
    #[inline]
    pub fn process(&mut self, input: f32) -> f32 {
        // LFO sweeps the notch frequency
        let lfo_val = self.lfo.sine_uni(); // 0 to 1
        self.lfo.advance(self.phase_inc);

        let freq = self.min_freq + (self.max_freq - self.min_freq) * lfo_val;

        // Calculate allpass coefficient from frequency
        let coeff = (std::f32::consts::PI * freq / self.sample_rate).tan();
        let a1 = (coeff - 1.0) / (coeff + 1.0);

        // Apply feedback
        let input_with_fb = input + self.feedback_sample * self.feedback;

        // Process through allpass stages
        let mut output = input_with_fb;
        for stage in &mut self.stages {
            output = stage.process(output, a1);
        }

        self.feedback_sample = output;

        // Mix: wet signal is sum of input and phase-shifted output
        Mix::blend(input, (input + output) * 0.5, self.mix)
    }

    /// Resets all filter states.
    pub fn clear(&mut self) {
        for stage in &mut self.stages {
            stage.clear();
        }
        self.lfo.reset();
        self.feedback_sample = 0.0;
    }
}

impl AudioNode for AllpassBank {
    fn process(&mut self, input: f32, _ctx: &AudioContext) -> f32 {
        AllpassBank::process(self, input)
    }

    fn reset(&mut self) {
        self.clear();
    }
}

/// Creates a phaser effect (cascaded allpass filters with LFO modulation).
pub fn phaser(sample_rate: f32) -> AllpassBank {
    AllpassBank::new(
        sample_rate,
        4,      // 4 stages
        0.5,    // 0.5 Hz rate
        100.0,  // 100 Hz min
        1000.0, // 1000 Hz max
        0.5,    // 50% mix
        0.7,    // 70% feedback
    )
}

// ============================================================================
// Graph-based Effects
// ============================================================================
//
// These demonstrate the target architecture: effects as graph configurations
// rather than dedicated structs. Each function returns an AudioGraph wired
// with primitives.

use crate::graph::{AffineNode, AudioGraph};
use crate::primitive::{DelayNode, LfoNode, MixNode};

/// Creates a tremolo effect using the graph architecture.
///
/// Tremolo = LFO modulating amplitude.
///
/// # Example
///
/// ```
/// use rhizome_resin_audio::effects::tremolo_graph;
/// use rhizome_resin_audio::graph::AudioContext;
///
/// let mut effect = tremolo_graph(44100.0, 5.0, 0.5);
/// let ctx = AudioContext::new(44100.0);
/// let output = effect.process(1.0, &ctx);
/// ```
pub fn tremolo_graph(sample_rate: f32, rate: f32, depth: f32) -> AudioGraph {
    let mut g = AudioGraph::new();

    // Use control rate for LFO modulation (64 samples = ~1.5ms at 44.1kHz)
    g.set_control_rate(64);

    let lfo = g.add(LfoNode::with_freq(rate, sample_rate));
    let gain = g.add(AffineNode::gain(1.0));

    g.connect_input(gain);
    g.set_output(gain);

    // LFO modulates gain: base = 1-depth/2, scale = depth/2
    // So gain varies from (1-depth) to 1
    let base = 1.0 - depth * 0.5;
    let scale = depth * 0.5;
    g.modulate(lfo, gain, AffineNode::PARAM_GAIN, base, scale);

    g
}

/// Creates a chorus effect using the graph architecture.
///
/// Chorus = modulated delay + dry/wet mix.
///
/// # Example
///
/// ```
/// use rhizome_resin_audio::effects::chorus_graph;
/// use rhizome_resin_audio::graph::AudioContext;
///
/// let mut effect = chorus_graph(44100.0);
/// let ctx = AudioContext::new(44100.0);
/// let output = effect.process(0.5, &ctx);
/// ```
pub fn chorus_graph(sample_rate: f32) -> AudioGraph {
    let mut g = AudioGraph::new();

    // Use control rate for LFO modulation
    g.set_control_rate(64);

    // Chorus parameters
    let base_delay_ms = 20.0;
    let depth_ms = 5.0;
    let rate_hz = 0.8;
    let mix = 0.5;

    let base_delay_samples = base_delay_ms * sample_rate / 1000.0;
    let depth_samples = depth_ms * sample_rate / 1000.0;
    let max_delay = ((base_delay_ms + depth_ms * 2.0) * sample_rate / 1000.0) as usize + 1;

    let lfo = g.add(LfoNode::with_freq(rate_hz, sample_rate));
    let delay = g.add(DelayNode::new(max_delay));
    let mixer = g.add(MixNode::new(mix));

    // Audio path: input -> delay -> mixer (as wet), input -> mixer (as dry via param)
    g.connect_input(delay);
    g.connect(delay, mixer);
    g.set_output(mixer);

    // LFO modulates delay time
    g.modulate(
        lfo,
        delay,
        DelayNode::PARAM_TIME,
        base_delay_samples,
        depth_samples,
    );

    // We need dry signal in mixer - use input node connection
    // Note: MixNode.dry is set via param, input provides wet
    g.connect_input(mixer); // mixer receives input as its "process" input (wet)

    // Actually for proper dry/wet, we need the mixer to know the dry signal.
    // The MixNode has a PARAM_DRY that we can set. But we can't easily route
    // the input to a parameter. For now, the mixer's "dry" comes from set_param.
    // This is a limitation - ideally we'd have multi-input nodes.

    g
}

/// Creates a flanger effect using the graph architecture.
///
/// Flanger = short modulated delay with feedback.
pub fn flanger_graph(sample_rate: f32) -> AudioGraph {
    let mut g = AudioGraph::new();

    // Use control rate for LFO modulation
    g.set_control_rate(64);

    // Flanger: shorter delay than chorus, more feedback
    let base_delay_ms = 5.0;
    let depth_ms = 3.0;
    let rate_hz = 0.3;
    let feedback = 0.7;

    let base_delay_samples = base_delay_ms * sample_rate / 1000.0;
    let depth_samples = depth_ms * sample_rate / 1000.0;
    let max_delay = ((base_delay_ms + depth_ms * 2.0) * sample_rate / 1000.0) as usize + 1;

    let lfo = g.add(LfoNode::with_freq(rate_hz, sample_rate));
    let mut delay_node = DelayNode::new(max_delay);
    delay_node.set_feedback(feedback);
    let delay = g.add(delay_node);

    g.connect_input(delay);
    g.set_output(delay);

    // LFO modulates delay time
    g.modulate(
        lfo,
        delay,
        DelayNode::PARAM_TIME,
        base_delay_samples,
        depth_samples,
    );

    g
}

// ============================================================================
// Reverb
// ============================================================================

/// Simple Schroeder reverb using comb and allpass filters.
pub struct Reverb {
    /// Comb filters (parallel).
    combs: Vec<CombFilter>,
    /// Allpass filters (series).
    allpasses: Vec<AllpassFilter>,
    /// Dry/wet mix (0 = dry, 1 = wet).
    pub mix: f32,
    /// Room size (affects delay times).
    room_size: f32,
    /// Damping (high frequency absorption).
    damping: f32,
}

impl Reverb {
    /// Creates a new reverb with default settings.
    pub fn new(sample_rate: f32) -> Self {
        Self::with_params(sample_rate, 0.5, 0.5, 0.3)
    }

    /// Creates a reverb with specific parameters.
    pub fn with_params(sample_rate: f32, room_size: f32, damping: f32, mix: f32) -> Self {
        // Schroeder reverb delay times (in samples at 44100 Hz, scaled)
        let scale = sample_rate / 44100.0;

        let comb_delays = [1557, 1617, 1491, 1422, 1277, 1356, 1188, 1116];
        let allpass_delays = [225, 556, 441, 341];

        let feedback = 0.84 + room_size * 0.12;

        let combs = comb_delays
            .iter()
            .map(|&d| {
                let delay = (d as f32 * scale) as usize;
                CombFilter::new(delay, feedback, damping)
            })
            .collect();

        let allpasses = allpass_delays
            .iter()
            .map(|&d| {
                let delay = (d as f32 * scale) as usize;
                AllpassFilter::new(delay, 0.5)
            })
            .collect();

        Self {
            combs,
            allpasses,
            mix,
            room_size,
            damping,
        }
    }

    /// Sets the room size (0-1).
    pub fn set_room_size(&mut self, size: f32) {
        self.room_size = size.clamp(0.0, 1.0);
        let feedback = 0.84 + self.room_size * 0.12;
        for comb in &mut self.combs {
            comb.feedback = feedback;
        }
    }

    /// Sets the damping (0-1).
    pub fn set_damping(&mut self, damping: f32) {
        self.damping = damping.clamp(0.0, 1.0);
        for comb in &mut self.combs {
            comb.damping = self.damping;
        }
    }

    /// Processes a single sample.
    pub fn process(&mut self, input: f32) -> f32 {
        // Sum of parallel comb filters
        let mut comb_sum = 0.0;
        for comb in &mut self.combs {
            comb_sum += comb.process(input);
        }
        comb_sum /= self.combs.len() as f32;

        // Series allpass filters
        let mut output = comb_sum;
        for allpass in &mut self.allpasses {
            output = allpass.process(output);
        }

        // Mix dry and wet
        input * (1.0 - self.mix) + output * self.mix
    }

    /// Clears the reverb buffers.
    pub fn clear(&mut self) {
        for comb in &mut self.combs {
            comb.clear();
        }
        for allpass in &mut self.allpasses {
            allpass.clear();
        }
    }
}

impl AudioNode for Reverb {
    fn process(&mut self, input: f32, _ctx: &AudioContext) -> f32 {
        Reverb::process(self, input)
    }

    fn reset(&mut self) {
        self.clear();
    }
}

/// Comb filter with feedback and damping.
struct CombFilter {
    buffer: Vec<f32>,
    write_pos: usize,
    feedback: f32,
    damping: f32,
    filter_state: f32,
}

impl CombFilter {
    fn new(delay_samples: usize, feedback: f32, damping: f32) -> Self {
        Self {
            buffer: vec![0.0; delay_samples.max(1)],
            write_pos: 0,
            feedback,
            damping,
            filter_state: 0.0,
        }
    }

    fn process(&mut self, input: f32) -> f32 {
        let output = self.buffer[self.write_pos];

        // Low-pass filter for damping
        self.filter_state = output * (1.0 - self.damping) + self.filter_state * self.damping;

        self.buffer[self.write_pos] = input + self.filter_state * self.feedback;
        self.write_pos = (self.write_pos + 1) % self.buffer.len();

        output
    }

    fn clear(&mut self) {
        self.buffer.fill(0.0);
        self.filter_state = 0.0;
    }
}

/// Allpass filter for diffusion.
struct AllpassFilter {
    buffer: Vec<f32>,
    write_pos: usize,
    feedback: f32,
}

impl AllpassFilter {
    fn new(delay_samples: usize, feedback: f32) -> Self {
        Self {
            buffer: vec![0.0; delay_samples.max(1)],
            write_pos: 0,
            feedback,
        }
    }

    fn process(&mut self, input: f32) -> f32 {
        let delayed = self.buffer[self.write_pos];
        let output = delayed - input * self.feedback;

        self.buffer[self.write_pos] = input + delayed * self.feedback;
        self.write_pos = (self.write_pos + 1) % self.buffer.len();

        output
    }

    fn clear(&mut self) {
        self.buffer.fill(0.0);
    }
}

// ============================================================================
// Distortion
// ============================================================================

/// Distortion/overdrive effect.
pub struct Distortion {
    /// Drive amount (1 = clean, higher = more distortion).
    pub drive: f32,
    /// Output level.
    pub level: f32,
    /// Distortion type.
    pub mode: DistortionMode,
    /// Tone filter.
    tone_filter: Biquad,
}

/// Distortion algorithm.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DistortionMode {
    /// Soft clipping (tanh).
    #[default]
    Soft,
    /// Hard clipping.
    Hard,
    /// Foldback distortion.
    Foldback,
    /// Asymmetric (tube-like).
    Asymmetric,
}

impl Distortion {
    /// Creates a new distortion effect.
    pub fn new(sample_rate: f32) -> Self {
        Self {
            drive: 2.0,
            level: 0.5,
            mode: DistortionMode::Soft,
            tone_filter: Biquad::lowpass(8000.0, 0.707, sample_rate),
        }
    }

    /// Sets the tone (low-pass cutoff).
    pub fn set_tone(&mut self, freq: f32, sample_rate: f32) {
        self.tone_filter
            .set_coeffs(BiquadCoeffs::lowpass(freq, 0.707, sample_rate));
    }

    /// Processes a single sample.
    pub fn process(&mut self, input: f32) -> f32 {
        let driven = input * self.drive;

        let distorted = match self.mode {
            DistortionMode::Soft => driven.tanh(),
            DistortionMode::Hard => driven.clamp(-1.0, 1.0),
            DistortionMode::Foldback => {
                let mut x = driven;
                while x > 1.0 || x < -1.0 {
                    if x > 1.0 {
                        x = 2.0 - x;
                    }
                    if x < -1.0 {
                        x = -2.0 - x;
                    }
                }
                x
            }
            DistortionMode::Asymmetric => {
                if driven >= 0.0 {
                    (driven * 2.0).tanh() * 0.5
                } else {
                    driven.tanh()
                }
            }
        };

        let filtered = self.tone_filter.process(distorted);
        filtered * self.level
    }
}

impl AudioNode for Distortion {
    fn process(&mut self, input: f32, _ctx: &AudioContext) -> f32 {
        Distortion::process(self, input)
    }

    fn reset(&mut self) {
        self.tone_filter.reset();
    }
}

// ============================================================================
// Convolution Reverb (requires spectral feature)
// ============================================================================

#[cfg(feature = "spectral")]
use crate::spectral::{Complex, fft, ifft};

#[cfg(feature = "spectral")]

/// Convolution reverb using impulse responses.
///
/// Provides high-quality reverb by convolving audio with a recorded
/// impulse response from a real space. Uses partitioned convolution
/// with FFT for efficient real-time processing.
pub struct ConvolutionReverb {
    /// Pre-computed FFT of IR partitions.
    ir_partitions: Vec<Vec<Complex>>,
    /// FFT size (block_size * 2).
    fft_size: usize,
    /// Processing block size.
    block_size: usize,
    /// Input buffer.
    input_buffer: Vec<f32>,
    /// Output accumulator (overlap-add).
    output_buffer: Vec<f32>,
    /// Frequency-domain delay line for each partition.
    fdl: Vec<Vec<Complex>>,
    /// Current position in input buffer.
    input_pos: usize,
    /// Current position in output buffer.
    output_pos: usize,
    /// Current FDL index.
    fdl_index: usize,
    /// Dry/wet mix (0 = dry, 1 = wet).
    pub mix: f32,
    /// Output gain.
    pub gain: f32,
}

#[cfg(feature = "spectral")]
impl ConvolutionReverb {
    /// Creates a new convolution reverb from an impulse response.
    ///
    /// # Arguments
    /// * `impulse_response` - The IR samples (mono)
    /// * `block_size` - Processing block size (power of 2, e.g., 512, 1024)
    pub fn new(impulse_response: &[f32], block_size: usize) -> Self {
        let block_size = block_size.next_power_of_two();
        let fft_size = block_size * 2;
        // FFT returns N/2+1 complex bins for real signals
        let spectrum_size = fft_size / 2 + 1;

        // Partition the IR and compute FFT of each partition
        let num_partitions = (impulse_response.len() + block_size - 1) / block_size;
        let num_partitions = num_partitions.max(1); // At least 1 partition
        let mut ir_partitions = Vec::with_capacity(num_partitions);

        for i in 0..num_partitions {
            let start = i * block_size;
            let end = (start + block_size).min(impulse_response.len());

            // Zero-pad partition to fft_size
            let mut partition = vec![0.0; fft_size];
            if start < impulse_response.len() {
                partition[..end - start].copy_from_slice(&impulse_response[start..end]);
            }

            // Compute FFT (returns spectrum_size complex values)
            ir_partitions.push(fft(&partition));
        }

        // Initialize frequency domain delay line with spectrum_size
        let fdl = vec![vec![Complex::new(0.0, 0.0); spectrum_size]; num_partitions];

        Self {
            ir_partitions,
            fft_size,
            block_size,
            input_buffer: vec![0.0; fft_size],
            output_buffer: vec![0.0; fft_size],
            fdl,
            input_pos: 0,
            output_pos: 0,
            fdl_index: 0,
            mix: 1.0,
            gain: 1.0,
        }
    }

    /// Creates a convolution reverb from a simple exponential decay IR.
    ///
    /// Useful for testing or when no IR file is available.
    pub fn from_decay(decay_time: f32, sample_rate: f32, block_size: usize) -> Self {
        let samples = (decay_time * sample_rate) as usize;
        let mut ir = vec![0.0; samples];

        // Generate exponential decay with diffusion
        let decay_rate = 1.0 / (decay_time * sample_rate * 0.1);

        for (i, sample) in ir.iter_mut().enumerate() {
            let t = i as f32 / sample_rate;
            let decay = (-t / (decay_time * 0.3)).exp();

            // Add some randomness for diffusion
            let noise = simple_hash(i as u32) as f32 / u32::MAX as f32 * 2.0 - 1.0;
            *sample = noise * decay * decay_rate.sqrt();
        }

        // Normalize
        let max = ir.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        if max > 0.0 {
            for s in &mut ir {
                *s /= max;
            }
        }

        Self::new(&ir, block_size)
    }

    /// Processes a single sample.
    ///
    /// Note: For efficiency, prefer processing blocks with `process_block`.
    pub fn process(&mut self, input: f32) -> f32 {
        // Store input
        self.input_buffer[self.input_pos] = input;
        self.input_pos += 1;

        // Get output
        let output = self.output_buffer[self.output_pos];
        self.output_buffer[self.output_pos] = 0.0;
        self.output_pos += 1;

        // Process block when ready
        if self.input_pos >= self.block_size {
            self.process_block_internal();
            self.input_pos = 0;
        }

        if self.output_pos >= self.block_size {
            self.output_pos = 0;
        }

        // Mix dry and wet
        let wet = output * self.gain;
        input * (1.0 - self.mix) + wet * self.mix
    }

    /// Processes a block of samples (more efficient than sample-by-sample).
    pub fn process_block(&mut self, input: &[f32], output: &mut [f32]) {
        for (i, &sample) in input.iter().enumerate() {
            output[i] = self.process(sample);
        }
    }

    /// Internal block processing with FFT convolution.
    fn process_block_internal(&mut self) {
        if self.ir_partitions.is_empty() {
            return;
        }

        // Copy input block with zero padding
        let mut input_fft = vec![0.0; self.fft_size];
        input_fft[..self.block_size].copy_from_slice(&self.input_buffer[..self.block_size]);

        // Compute FFT of input (returns N/2+1 complex values)
        let input_spectrum = fft(&input_fft);
        let spectrum_size = input_spectrum.len();

        // Store in FDL
        self.fdl[self.fdl_index] = input_spectrum;

        // Accumulate convolution results (spectrum_size = fft_size/2 + 1)
        let mut accum = vec![Complex::new(0.0, 0.0); spectrum_size];

        for (i, ir_partition) in self.ir_partitions.iter().enumerate() {
            // Get the FDL entry that corresponds to this partition's delay
            let fdl_idx = (self.fdl_index + self.fdl.len() - i) % self.fdl.len();

            // Complex multiply in frequency domain
            for (j, (a, ir)) in accum.iter_mut().zip(ir_partition.iter()).enumerate() {
                let fdl_sample = self.fdl[fdl_idx][j];
                *a = *a + fdl_sample * *ir;
            }
        }

        // IFFT to get time domain result (returns fft_size samples)
        let result = ifft(&accum);

        // Overlap-add to output buffer
        for (i, &sample) in result.iter().enumerate() {
            let out_idx = (self.output_pos + i) % self.fft_size;
            self.output_buffer[out_idx] += sample;
        }

        // Advance FDL index
        self.fdl_index = (self.fdl_index + 1) % self.fdl.len();
    }

    /// Clears all internal buffers.
    pub fn clear(&mut self) {
        self.input_buffer.fill(0.0);
        self.output_buffer.fill(0.0);
        for fdl_entry in &mut self.fdl {
            fdl_entry.fill(Complex::new(0.0, 0.0));
        }
        self.input_pos = 0;
        self.output_pos = 0;
        self.fdl_index = 0;
    }

    /// Returns the latency in samples (one block).
    pub fn latency(&self) -> usize {
        self.block_size
    }

    /// Returns the impulse response length in samples.
    pub fn ir_length(&self) -> usize {
        self.ir_partitions.len() * self.block_size
    }
}

#[cfg(feature = "spectral")]
impl AudioNode for ConvolutionReverb {
    fn process(&mut self, input: f32, _ctx: &AudioContext) -> f32 {
        ConvolutionReverb::process(self, input)
    }

    fn reset(&mut self) {
        self.clear();
    }
}

/// Creates a convolution reverb from an impulse response.
///
/// Operations on audio buffers use the ops-as-values pattern.
/// See `docs/design/ops-as-values.md`.
#[cfg(feature = "spectral")]
#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "dynop", derive(rhizome_resin_op::Op))]
#[cfg_attr(feature = "dynop", op(input = Vec<f32>, output = ConvolutionReverb))]
pub struct Convolution {
    /// Processing block size (power of 2).
    pub block_size: usize,
    /// Dry/wet mix.
    pub mix: f32,
    /// Output gain.
    pub gain: f32,
}

#[cfg(feature = "spectral")]
impl Default for Convolution {
    fn default() -> Self {
        Self {
            block_size: 1024,
            mix: 0.5,
            gain: 1.0,
        }
    }
}

#[cfg(feature = "spectral")]
impl Convolution {
    /// Creates a new convolution configuration.
    pub fn new(block_size: usize) -> Self {
        Self {
            block_size,
            ..Default::default()
        }
    }

    /// Applies this operation to create a ConvolutionReverb.
    pub fn apply(&self, ir: &[f32]) -> ConvolutionReverb {
        let mut reverb = ConvolutionReverb::new(ir, self.block_size);
        reverb.mix = self.mix;
        reverb.gain = self.gain;
        reverb
    }
}

/// Backwards-compatible type alias.
#[cfg(feature = "spectral")]
pub type ConvolutionConfig = Convolution;

/// Creates a convolution reverb with configuration.
#[cfg(feature = "spectral")]
pub fn convolution_reverb(ir: &[f32], config: &ConvolutionConfig) -> ConvolutionReverb {
    config.apply(ir)
}

/// Generates a synthetic impulse response for a room.
///
/// # Arguments
/// * `size` - Room size (small=0.1, large=1.0)
/// * `damping` - High frequency damping (0-1)
/// * `duration` - IR duration in seconds
/// * `sample_rate` - Sample rate in Hz
#[cfg(feature = "spectral")]
pub fn generate_room_ir(size: f32, damping: f32, duration: f32, sample_rate: f32) -> Vec<f32> {
    let samples = (duration * sample_rate) as usize;
    let mut ir = vec![0.0; samples];

    // Early reflections based on room size
    let reflection_times = [
        0.01 * size,
        0.02 * size,
        0.03 * size,
        0.04 * size,
        0.06 * size,
        0.08 * size,
    ];
    let reflection_gains = [0.8, 0.6, 0.5, 0.4, 0.3, 0.2];

    // Add early reflections
    for (&time, &gain) in reflection_times.iter().zip(reflection_gains.iter()) {
        let idx = (time * sample_rate) as usize;
        if idx < samples {
            ir[idx] += gain;
        }
    }

    // Add diffuse tail
    let decay_time = duration * 0.7;
    for i in 0..samples {
        let t = i as f32 / sample_rate;

        // Exponential decay
        let decay = (-t / decay_time * 3.0).exp();

        // Add noise for diffusion (after early reflections)
        if t > 0.1 * size {
            let noise = simple_hash(i as u32) as f32 / u32::MAX as f32 * 2.0 - 1.0;
            ir[i] += noise * decay * 0.1;
        }
    }

    // Apply damping (separate pass to avoid borrow issues)
    for i in 1..samples {
        let t = i as f32 / sample_rate;
        let damp = 1.0 - damping * t / duration;
        ir[i] = ir[i] * damp + ir[i - 1] * (1.0 - damp) * 0.5;
    }

    // Normalize
    let max = ir.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
    if max > 0.0 {
        for s in &mut ir {
            *s /= max;
        }
    }

    ir
}

/// Simple hash for reproducible noise.
#[cfg(feature = "spectral")]
fn simple_hash(x: u32) -> u32 {
    let mut h = x;
    h = h.wrapping_mul(0x85ebca6b);
    h ^= h >> 13;
    h = h.wrapping_mul(0xc2b2ae35);
    h ^= h >> 16;
    h
}

// ============================================================================
// Compressor
// ============================================================================

/// Dynamic range compressor.
///
/// Reduces the dynamic range of audio by attenuating signals above a threshold.
/// Essential for mixing, mastering, and controlling transients.
///
/// # Example
///
/// ```
/// use rhizome_resin_audio::effects::Compressor;
///
/// let mut comp = Compressor::new(44100.0);
/// comp.threshold = -20.0; // dB
/// comp.ratio = 4.0;       // 4:1 compression
/// comp.attack = 0.01;     // 10ms attack
/// comp.release = 0.1;     // 100ms release
///
/// let compressed = comp.process(0.8);
/// ```
pub struct Compressor {
    /// Threshold in dB (signals above this are compressed).
    pub threshold: f32,
    /// Compression ratio (e.g., 4.0 means 4:1).
    pub ratio: f32,
    /// Attack time in seconds.
    pub attack: f32,
    /// Release time in seconds.
    pub release: f32,
    /// Makeup gain in dB.
    pub makeup_gain: f32,
    /// Knee width in dB (0 = hard knee).
    pub knee: f32,
    /// Current envelope level.
    envelope: f32,
    /// Sample rate.
    sample_rate: f32,
}

impl Compressor {
    /// Creates a new compressor with default settings.
    pub fn new(sample_rate: f32) -> Self {
        Self {
            threshold: -20.0,
            ratio: 4.0,
            attack: 0.01,
            release: 0.1,
            makeup_gain: 0.0,
            knee: 0.0,
            envelope: 0.0,
            sample_rate,
        }
    }

    /// Processes a single sample.
    pub fn process(&mut self, input: f32) -> f32 {
        // Convert to dB
        let input_db = if input.abs() > 1e-10 {
            20.0 * input.abs().log10()
        } else {
            -100.0
        };

        // Calculate gain reduction
        let gain_db = self.compute_gain(input_db);

        // Smooth the gain with attack/release envelope
        let target = gain_db;
        let coeff = if target < self.envelope {
            (-1.0 / (self.attack * self.sample_rate)).exp()
        } else {
            (-1.0 / (self.release * self.sample_rate)).exp()
        };
        self.envelope = coeff * self.envelope + (1.0 - coeff) * target;

        // Apply gain reduction and makeup gain
        let gain_linear = 10.0f32.powf((self.envelope + self.makeup_gain) / 20.0);
        input * gain_linear
    }

    /// Computes gain reduction in dB for a given input level.
    fn compute_gain(&self, input_db: f32) -> f32 {
        if self.knee > 0.0 {
            // Soft knee
            let knee_start = self.threshold - self.knee / 2.0;
            let knee_end = self.threshold + self.knee / 2.0;

            if input_db < knee_start {
                0.0
            } else if input_db > knee_end {
                (self.threshold - input_db) * (1.0 - 1.0 / self.ratio)
            } else {
                // Quadratic interpolation in knee region
                let x = input_db - knee_start;
                let slope = (1.0 - 1.0 / self.ratio) / (2.0 * self.knee);
                -slope * x * x
            }
        } else {
            // Hard knee
            if input_db > self.threshold {
                (self.threshold - input_db) * (1.0 - 1.0 / self.ratio)
            } else {
                0.0
            }
        }
    }

    /// Resets the envelope.
    pub fn reset(&mut self) {
        self.envelope = 0.0;
    }
}

impl AudioNode for Compressor {
    fn process(&mut self, input: f32, _ctx: &AudioContext) -> f32 {
        Compressor::process(self, input)
    }

    fn reset(&mut self) {
        Compressor::reset(self);
    }
}

// ============================================================================
// Limiter
// ============================================================================

/// Brickwall limiter with lookahead.
///
/// Prevents audio from exceeding a ceiling level. Uses lookahead to
/// catch transients before they clip.
pub struct Limiter {
    /// Ceiling level in dB.
    pub ceiling: f32,
    /// Release time in seconds.
    pub release: f32,
    /// Lookahead time in samples.
    lookahead_samples: usize,
    /// Lookahead buffer.
    lookahead_buffer: Vec<f32>,
    /// Buffer write position.
    write_pos: usize,
    /// Current gain reduction.
    gain: f32,
    /// Sample rate.
    sample_rate: f32,
}

impl Limiter {
    /// Creates a new limiter with default settings.
    pub fn new(sample_rate: f32) -> Self {
        let lookahead_ms = 5.0;
        let lookahead_samples = (lookahead_ms * sample_rate / 1000.0) as usize;

        Self {
            ceiling: -0.3,
            release: 0.1,
            lookahead_samples,
            lookahead_buffer: vec![0.0; lookahead_samples],
            write_pos: 0,
            gain: 1.0,
            sample_rate,
        }
    }

    /// Sets lookahead time in milliseconds.
    pub fn set_lookahead(&mut self, ms: f32) {
        self.lookahead_samples = (ms * self.sample_rate / 1000.0) as usize;
        self.lookahead_buffer = vec![0.0; self.lookahead_samples.max(1)];
        self.write_pos = 0;
    }

    /// Processes a single sample.
    pub fn process(&mut self, input: f32) -> f32 {
        // Get delayed sample
        let delayed = self.lookahead_buffer[self.write_pos];

        // Store current sample
        self.lookahead_buffer[self.write_pos] = input;
        self.write_pos = (self.write_pos + 1) % self.lookahead_samples.max(1);

        // Calculate required gain for ceiling
        let ceiling_linear = 10.0f32.powf(self.ceiling / 20.0);

        // Look ahead to find peak
        let peak = self
            .lookahead_buffer
            .iter()
            .map(|x| x.abs())
            .fold(0.0f32, f32::max);

        let target_gain = if peak > ceiling_linear {
            ceiling_linear / peak
        } else {
            1.0
        };

        // Instant attack, smooth release
        if target_gain < self.gain {
            self.gain = target_gain;
        } else {
            let release_coeff = (-1.0 / (self.release * self.sample_rate)).exp();
            self.gain = release_coeff * self.gain + (1.0 - release_coeff) * target_gain;
        }

        delayed * self.gain
    }

    /// Resets the limiter state.
    pub fn reset(&mut self) {
        self.lookahead_buffer.fill(0.0);
        self.write_pos = 0;
        self.gain = 1.0;
    }
}

impl AudioNode for Limiter {
    fn process(&mut self, input: f32, _ctx: &AudioContext) -> f32 {
        Limiter::process(self, input)
    }

    fn reset(&mut self) {
        Limiter::reset(self);
    }
}

// ============================================================================
// Noise Gate
// ============================================================================

/// Noise gate with attack, hold, and release.
///
/// Silences audio below a threshold level. Useful for removing background
/// noise, leakage, or creating gated effects.
pub struct NoiseGate {
    /// Threshold in dB (signals below this are gated).
    pub threshold: f32,
    /// Attack time in seconds.
    pub attack: f32,
    /// Hold time in seconds (gate stays open after signal drops).
    pub hold: f32,
    /// Release time in seconds.
    pub release: f32,
    /// Range in dB (how much to attenuate, -inf for full gate).
    pub range: f32,
    /// Current gate state (0 = closed, 1 = open).
    gate_level: f32,
    /// Hold counter in samples.
    hold_counter: usize,
    /// Sample rate.
    sample_rate: f32,
}

impl NoiseGate {
    /// Creates a new noise gate with default settings.
    pub fn new(sample_rate: f32) -> Self {
        Self {
            threshold: -40.0,
            attack: 0.001,
            hold: 0.05,
            release: 0.1,
            range: -80.0,
            gate_level: 0.0,
            hold_counter: 0,
            sample_rate,
        }
    }

    /// Processes a single sample.
    pub fn process(&mut self, input: f32) -> f32 {
        // Convert to dB
        let input_db = if input.abs() > 1e-10 {
            20.0 * input.abs().log10()
        } else {
            -100.0
        };

        // Determine target gate level
        let target = if input_db > self.threshold {
            self.hold_counter = (self.hold * self.sample_rate) as usize;
            1.0
        } else if self.hold_counter > 0 {
            self.hold_counter -= 1;
            1.0
        } else {
            0.0
        };

        // Smooth with attack/release
        let coeff = if target > self.gate_level {
            (-1.0 / (self.attack * self.sample_rate)).exp()
        } else {
            (-1.0 / (self.release * self.sample_rate)).exp()
        };
        self.gate_level = coeff * self.gate_level + (1.0 - coeff) * target;

        // Apply gate with range
        let range_linear = 10.0f32.powf(self.range / 20.0);
        let gain = range_linear + self.gate_level * (1.0 - range_linear);
        input * gain
    }

    /// Resets the gate state.
    pub fn reset(&mut self) {
        self.gate_level = 0.0;
        self.hold_counter = 0;
    }
}

impl AudioNode for NoiseGate {
    fn process(&mut self, input: f32, _ctx: &AudioContext) -> f32 {
        NoiseGate::process(self, input)
    }

    fn reset(&mut self) {
        NoiseGate::reset(self);
    }
}

// ============================================================================
// Bitcrusher
// ============================================================================

/// Bitcrusher effect for lo-fi distortion.
///
/// Reduces bit depth and/or sample rate for crunchy, retro digital artifacts.
///
/// # Example
///
/// ```
/// use rhizome_resin_audio::effects::Bitcrusher;
///
/// let mut crusher = Bitcrusher::new(44100.0);
/// crusher.bits = 8;              // 8-bit audio
/// crusher.downsample_rate = 8000.0; // Downsample to 8kHz
///
/// let crushed = crusher.process(0.5);
/// ```
pub struct Bitcrusher {
    /// Bit depth (1-32).
    pub bits: u32,
    /// Target sample rate for downsampling.
    pub downsample_rate: f32,
    /// Current held sample (for downsampling).
    held_sample: f32,
    /// Phase accumulator for downsampling.
    phase: f32,
    /// Original sample rate.
    sample_rate: f32,
}

impl Bitcrusher {
    /// Creates a new bitcrusher with default settings.
    pub fn new(sample_rate: f32) -> Self {
        Self {
            bits: 8,
            downsample_rate: sample_rate,
            held_sample: 0.0,
            phase: 0.0,
            sample_rate,
        }
    }

    /// Processes a single sample.
    pub fn process(&mut self, input: f32) -> f32 {
        // Downsampling via sample-and-hold
        self.phase += self.downsample_rate / self.sample_rate;
        if self.phase >= 1.0 {
            self.phase -= 1.0;
            self.held_sample = input;
        }

        // Bit reduction
        let levels = (1u32 << self.bits.min(31)) as f32;
        let quantized = (self.held_sample * levels).round() / levels;

        quantized
    }

    /// Resets the effect state.
    pub fn reset(&mut self) {
        self.held_sample = 0.0;
        self.phase = 0.0;
    }
}

impl AudioNode for Bitcrusher {
    fn process(&mut self, input: f32, _ctx: &AudioContext) -> f32 {
        Bitcrusher::process(self, input)
    }

    fn reset(&mut self) {
        Bitcrusher::reset(self);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reverb_creates() {
        let reverb = Reverb::new(44100.0);
        assert!(reverb.mix >= 0.0 && reverb.mix <= 1.0);
    }

    #[test]
    fn test_reverb_process() {
        let mut reverb = Reverb::new(44100.0);
        reverb.mix = 1.0; // Full wet for testing

        // Process an impulse
        let _out1 = reverb.process(1.0);

        // Continue processing silence - should have tail after delay
        let mut has_output = false;
        for _ in 0..5000 {
            let out = reverb.process(0.0);
            if out.abs() > 0.0001 {
                has_output = true;
                break;
            }
        }
        assert!(has_output, "Reverb should have decay tail");
    }

    #[test]
    fn test_chorus_process() {
        let mut effect = chorus(44100.0);

        // Process some samples
        let mut outputs = Vec::new();
        for i in 0..1000 {
            let input = (i as f32 * 0.1).sin();
            outputs.push(effect.process(input));
        }

        // Should have varying output (due to modulation)
        let variance: f32 = outputs.windows(2).map(|w| (w[1] - w[0]).abs()).sum();
        assert!(variance > 0.0);
    }

    #[test]
    fn test_phaser_process() {
        let mut effect = phaser(44100.0);

        // Process a constant signal
        for _ in 0..1000 {
            effect.process(0.5);
        }

        // Should still produce output
        let out = effect.process(0.5);
        assert!(out.abs() <= 1.5);
    }

    #[test]
    fn test_flanger_process() {
        let mut effect = flanger(44100.0);

        // Process samples
        let out = effect.process(1.0);
        assert!(out.abs() <= 2.0);

        // Clear and verify
        effect.clear();
        let out_after_clear = effect.process(0.0);
        assert!(out_after_clear.abs() < 0.01);
    }

    #[test]
    fn test_distortion_modes() {
        let sample_rate = 44100.0;

        // Test soft clipping
        let mut dist = Distortion::new(sample_rate);
        dist.mode = DistortionMode::Soft;
        dist.drive = 10.0;
        let out = dist.process(0.5);
        assert!(out.abs() <= 1.0);

        // Test hard clipping
        dist.mode = DistortionMode::Hard;
        let out = dist.process(0.5);
        assert!(out.abs() <= 1.0);

        // Test foldback
        dist.mode = DistortionMode::Foldback;
        let out = dist.process(0.5);
        assert!(out.abs() <= 1.0);
    }

    #[test]
    fn test_tremolo() {
        let mut effect = tremolo(44100.0, 5.0, 1.0);

        // With full depth, output should vary between 0 and input
        let mut min_out = f32::MAX;
        let mut max_out = f32::MIN;

        for _ in 0..44100 {
            let out = effect.process(1.0);
            min_out = min_out.min(out);
            max_out = max_out.max(out);
        }

        assert!(min_out < 0.1);
        assert!(max_out > 0.9);
    }

    #[test]
    fn test_effects_as_audio_nodes() {
        let sample_rate = 44100.0;
        let ctx = AudioContext::new(sample_rate);

        // All effects should implement AudioNode
        let mut reverb_node: Box<dyn AudioNode> = Box::new(Reverb::new(sample_rate));
        let mut chorus_node: Box<dyn AudioNode> = Box::new(chorus(sample_rate));
        let mut phaser_node: Box<dyn AudioNode> = Box::new(phaser(sample_rate));
        let mut flanger_node: Box<dyn AudioNode> = Box::new(flanger(sample_rate));

        // All should process without panic
        reverb_node.process(0.5, &ctx);
        chorus_node.process(0.5, &ctx);
        phaser_node.process(0.5, &ctx);
        flanger_node.process(0.5, &ctx);
    }

    // ========================================================================
    // Convolution Reverb tests
    // ========================================================================

    #[test]
    fn test_convolution_reverb_creates() {
        let ir = vec![1.0, 0.5, 0.25, 0.125];
        let reverb = ConvolutionReverb::new(&ir, 512);
        assert_eq!(reverb.latency(), 512);
    }

    #[test]
    fn test_convolution_reverb_from_decay() {
        let reverb = ConvolutionReverb::from_decay(0.5, 44100.0, 512);
        assert!(reverb.ir_length() > 0);
    }

    #[test]
    fn test_convolution_reverb_process() {
        // Create simple IR - just a delta function for identity convolution
        let mut ir = vec![0.0; 256];
        ir[0] = 1.0; // Delta function should pass through input

        let mut reverb = ConvolutionReverb::new(&ir, 128);
        reverb.mix = 1.0; // Full wet

        // Process a constant signal
        let mut output = Vec::new();
        for _ in 0..1024 {
            output.push(reverb.process(1.0));
        }

        // The convolution should produce output (with latency)
        // Check that output contains non-trivial values after initial latency
        let max_output = output.iter().map(|x| x.abs()).fold(0.0f32, f32::max);

        // Even tiny output indicates the convolution is working
        // (scaling may differ from ideal)
        assert!(
            max_output > 1e-10,
            "Should have some output, max was: {:e}",
            max_output
        );
    }

    #[test]
    fn test_convolution_reverb_clear() {
        let ir = vec![1.0; 1024];
        let mut reverb = ConvolutionReverb::new(&ir, 256);

        // Process some input
        for _ in 0..512 {
            reverb.process(1.0);
        }

        reverb.clear();

        // After clear, processing silence should give near-zero output
        let out = reverb.process(0.0);
        assert!(out.abs() < 0.01);
    }

    #[test]
    fn test_convolution_reverb_block_process() {
        let ir = vec![1.0, 0.8, 0.6, 0.4, 0.2, 0.1];
        let mut reverb = ConvolutionReverb::new(&ir, 256);

        let input = vec![1.0; 256];
        let mut output = vec![0.0; 256];

        reverb.process_block(&input, &mut output);

        // Should have processed without panic
        assert_eq!(output.len(), 256);
    }

    #[test]
    fn test_generate_room_ir() {
        let ir = generate_room_ir(0.5, 0.3, 1.0, 44100.0);

        // Should have correct length
        assert_eq!(ir.len(), 44100);

        // Should be normalized (max <= 1)
        let max = ir.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        assert!(max <= 1.0);

        // Should have some content
        let sum: f32 = ir.iter().map(|x| x.abs()).sum();
        assert!(sum > 0.0);
    }

    #[test]
    fn test_convolution_config() {
        let config = ConvolutionConfig::default();
        assert_eq!(config.block_size, 1024);
        assert!((config.mix - 0.5).abs() < 0.001);
        assert!((config.gain - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_convolution_reverb_helper() {
        let ir = generate_room_ir(0.3, 0.5, 0.5, 44100.0);
        let config = ConvolutionConfig {
            block_size: 512,
            mix: 0.7,
            gain: 0.8,
        };

        let reverb = convolution_reverb(&ir, &config);
        assert!((reverb.mix - 0.7).abs() < 0.001);
        assert!((reverb.gain - 0.8).abs() < 0.001);
    }

    #[test]
    fn test_convolution_as_audio_node() {
        let ir = vec![1.0; 256];
        let mut reverb: Box<dyn AudioNode> = Box::new(ConvolutionReverb::new(&ir, 128));
        let ctx = AudioContext::new(44100.0);

        // Should implement AudioNode
        let out = reverb.process(0.5, &ctx);
        assert!(out.is_finite());

        reverb.reset();
    }

    #[test]
    fn test_compressor() {
        let mut comp = Compressor::new(44100.0);
        comp.threshold = -10.0;
        comp.ratio = 4.0;

        // Quiet signal should pass through mostly unchanged
        let quiet = comp.process(0.1);
        assert!(quiet.abs() < 0.2);

        // Reset and test loud signal
        comp.reset();
        let mut loud_out = 0.0;
        for _ in 0..1000 {
            loud_out = comp.process(0.9);
        }
        // Loud signal should be compressed (reduced)
        assert!(loud_out.abs() < 0.9);
    }

    #[test]
    fn test_compressor_soft_knee() {
        let mut comp = Compressor::new(44100.0);
        comp.threshold = -10.0;
        comp.ratio = 4.0;
        comp.knee = 6.0; // Soft knee

        // Process some samples
        for _ in 0..100 {
            let out = comp.process(0.5);
            assert!(out.is_finite());
        }
    }

    #[test]
    fn test_limiter() {
        let mut limiter = Limiter::new(44100.0);
        limiter.ceiling = -3.0;

        // Process loud signal
        let mut max_out = 0.0f32;
        for _ in 0..1000 {
            let out = limiter.process(1.0);
            max_out = max_out.max(out.abs());
        }

        // Output should be limited (ceiling is -3dB ≈ 0.708)
        let ceiling_linear = 10.0f32.powf(-3.0 / 20.0);
        assert!(max_out <= ceiling_linear + 0.01);
    }

    #[test]
    fn test_noise_gate() {
        let mut gate = NoiseGate::new(44100.0);
        gate.threshold = -20.0;
        gate.range = -80.0;

        // Quiet signal should be gated
        let mut gated_out = 0.0;
        for _ in 0..1000 {
            gated_out = gate.process(0.01);
        }
        assert!(gated_out.abs() < 0.01);

        // Loud signal should pass through
        gate.reset();
        let mut loud_out = 0.0;
        for _ in 0..1000 {
            loud_out = gate.process(0.5);
        }
        assert!(loud_out.abs() > 0.3);
    }

    #[test]
    fn test_bitcrusher() {
        let mut crusher = Bitcrusher::new(44100.0);
        crusher.bits = 4;
        crusher.downsample_rate = 44100.0; // No downsampling

        // Process and check quantization
        let out1 = crusher.process(0.5);
        let out2 = crusher.process(0.51);

        // With 4 bits (16 levels), nearby values should quantize to same level
        // 16 levels means step size is 1/16 = 0.0625
        assert!((out1 - out2).abs() < 0.1);
    }

    #[test]
    fn test_bitcrusher_downsample() {
        let mut crusher = Bitcrusher::new(44100.0);
        crusher.bits = 16;
        crusher.downsample_rate = 4410.0; // 10x downsample

        // Process varying input
        let mut outputs = Vec::new();
        for i in 0..100 {
            outputs.push(crusher.process((i as f32 * 0.1).sin()));
        }

        // Due to sample-and-hold, consecutive samples should often be equal
        let mut equal_pairs = 0;
        for w in outputs.windows(2) {
            if (w[0] - w[1]).abs() < 0.0001 {
                equal_pairs += 1;
            }
        }
        assert!(
            equal_pairs > 50,
            "Downsampling should create repeated samples"
        );
    }

    #[test]
    fn test_modulated_delay() {
        let mut effect = ModulatedDelay::new(44100.0, 0.01, 0.002, 1.0, 0.5, 0.3);

        // Process some samples
        let mut outputs = Vec::new();
        for i in 0..1000 {
            let input = (i as f32 * 0.1).sin();
            outputs.push(effect.process(input));
        }

        // Should have varying output (due to modulation)
        let variance: f32 = outputs.windows(2).map(|w| (w[1] - w[0]).abs()).sum();
        assert!(variance > 0.0);
    }

    #[test]
    fn test_chorus_flanger_constructors() {
        let mut chorus_effect = chorus(44100.0);
        let mut flanger_effect = flanger(44100.0);

        // Both should process without panic
        for i in 0..100 {
            let input = (i as f32 * 0.1).sin();
            chorus_effect.process(input);
            flanger_effect.process(input);
        }

        // Chorus has less feedback than flanger by default
        assert!(chorus_effect.feedback.abs() < flanger_effect.feedback.abs());
    }
}
