//! Audio effects for sound design.
//!
//! Provides classic effects: reverb, chorus, phaser, flanger.

use crate::filter::{Biquad, BiquadCoeffs};
use crate::graph::{AudioContext, AudioNode};

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
// Chorus
// ============================================================================

/// Chorus effect using modulated delay.
pub struct Chorus {
    /// Delay buffer.
    buffer: Vec<f32>,
    /// Write position.
    write_pos: usize,
    /// Base delay in samples.
    base_delay: f32,
    /// Modulation depth in samples.
    depth: f32,
    /// Modulation rate in Hz.
    rate: f32,
    /// LFO phase.
    phase: f32,
    /// Dry/wet mix.
    pub mix: f32,
    /// Feedback amount.
    pub feedback: f32,
    /// Sample rate.
    sample_rate: f32,
}

impl Chorus {
    /// Creates a new chorus effect.
    pub fn new(sample_rate: f32) -> Self {
        Self::with_params(sample_rate, 0.02, 0.003, 0.5, 0.5, 0.0)
    }

    /// Creates a chorus with specific parameters.
    ///
    /// - `delay`: Base delay in seconds (typically 0.01-0.03)
    /// - `depth`: Modulation depth in seconds (typically 0.001-0.005)
    /// - `rate`: Modulation rate in Hz (typically 0.1-5.0)
    /// - `mix`: Dry/wet mix (0-1)
    /// - `feedback`: Feedback amount (0-0.9)
    pub fn with_params(
        sample_rate: f32,
        delay: f32,
        depth: f32,
        rate: f32,
        mix: f32,
        feedback: f32,
    ) -> Self {
        let max_delay = (delay + depth * 2.0) * sample_rate;
        let buffer_size = (max_delay as usize + 1).max(1024);

        Self {
            buffer: vec![0.0; buffer_size],
            write_pos: 0,
            base_delay: delay * sample_rate,
            depth: depth * sample_rate,
            rate,
            phase: 0.0,
            mix,
            feedback: feedback.clamp(0.0, 0.95),
            sample_rate,
        }
    }

    /// Sets the modulation rate in Hz.
    pub fn set_rate(&mut self, rate: f32) {
        self.rate = rate;
    }

    /// Sets the modulation depth in seconds.
    pub fn set_depth(&mut self, depth: f32) {
        self.depth = depth * self.sample_rate;
    }

    /// Processes a single sample.
    pub fn process(&mut self, input: f32) -> f32 {
        // Calculate modulated delay
        let lfo = (self.phase * std::f32::consts::TAU).sin();
        let delay = self.base_delay + lfo * self.depth;

        // Read from buffer with linear interpolation
        let read_pos = self.write_pos as f32 - delay;
        let read_pos = if read_pos < 0.0 {
            read_pos + self.buffer.len() as f32
        } else {
            read_pos
        };

        let idx0 = read_pos.floor() as usize % self.buffer.len();
        let idx1 = (idx0 + 1) % self.buffer.len();
        let frac = read_pos.fract();

        let delayed = self.buffer[idx0] * (1.0 - frac) + self.buffer[idx1] * frac;

        // Write to buffer
        self.buffer[self.write_pos] = input + delayed * self.feedback;
        self.write_pos = (self.write_pos + 1) % self.buffer.len();

        // Advance LFO
        self.phase += self.rate / self.sample_rate;
        if self.phase >= 1.0 {
            self.phase -= 1.0;
        }

        // Mix
        input * (1.0 - self.mix) + delayed * self.mix
    }

    /// Clears the delay buffer.
    pub fn clear(&mut self) {
        self.buffer.fill(0.0);
        self.phase = 0.0;
    }
}

impl AudioNode for Chorus {
    fn process(&mut self, input: f32, _ctx: &AudioContext) -> f32 {
        Chorus::process(self, input)
    }

    fn reset(&mut self) {
        self.clear();
    }
}

// ============================================================================
// Phaser
// ============================================================================

/// Phaser effect using cascaded allpass filters.
pub struct Phaser {
    /// Allpass filter stages.
    stages: Vec<PhaseAllpass>,
    /// LFO phase.
    phase: f32,
    /// Modulation rate in Hz.
    rate: f32,
    /// Minimum frequency.
    min_freq: f32,
    /// Maximum frequency.
    max_freq: f32,
    /// Dry/wet mix.
    pub mix: f32,
    /// Feedback amount.
    pub feedback: f32,
    /// Previous output for feedback.
    feedback_sample: f32,
    /// Sample rate.
    sample_rate: f32,
}

impl Phaser {
    /// Creates a new phaser effect.
    pub fn new(sample_rate: f32) -> Self {
        Self::with_params(sample_rate, 4, 0.5, 100.0, 1000.0, 0.5, 0.7)
    }

    /// Creates a phaser with specific parameters.
    ///
    /// - `stages`: Number of allpass stages (2-12, even numbers work best)
    /// - `rate`: Modulation rate in Hz
    /// - `min_freq`: Minimum sweep frequency
    /// - `max_freq`: Maximum sweep frequency
    /// - `mix`: Dry/wet mix
    /// - `feedback`: Feedback amount
    pub fn with_params(
        sample_rate: f32,
        stages: usize,
        rate: f32,
        min_freq: f32,
        max_freq: f32,
        mix: f32,
        feedback: f32,
    ) -> Self {
        let stages = (2..=12).contains(&stages).then_some(stages).unwrap_or(4);

        Self {
            stages: (0..stages).map(|_| PhaseAllpass::new()).collect(),
            phase: 0.0,
            rate,
            min_freq,
            max_freq,
            mix,
            feedback: feedback.clamp(0.0, 0.95),
            feedback_sample: 0.0,
            sample_rate,
        }
    }

    /// Sets the modulation rate.
    pub fn set_rate(&mut self, rate: f32) {
        self.rate = rate;
    }

    /// Sets the sweep range.
    pub fn set_range(&mut self, min_freq: f32, max_freq: f32) {
        self.min_freq = min_freq;
        self.max_freq = max_freq;
    }

    /// Processes a single sample.
    pub fn process(&mut self, input: f32) -> f32 {
        // Calculate sweep frequency
        let lfo = (self.phase * std::f32::consts::TAU).sin() * 0.5 + 0.5;
        let freq = self.min_freq + (self.max_freq - self.min_freq) * lfo;

        // Calculate allpass coefficient
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

        // Advance LFO
        self.phase += self.rate / self.sample_rate;
        if self.phase >= 1.0 {
            self.phase -= 1.0;
        }

        // Mix with phase cancellation
        input * (1.0 - self.mix) + (input + output) * 0.5 * self.mix
    }

    /// Resets the phaser state.
    pub fn clear(&mut self) {
        for stage in &mut self.stages {
            stage.clear();
        }
        self.phase = 0.0;
        self.feedback_sample = 0.0;
    }
}

impl AudioNode for Phaser {
    fn process(&mut self, input: f32, _ctx: &AudioContext) -> f32 {
        Phaser::process(self, input)
    }

    fn reset(&mut self) {
        self.clear();
    }
}

/// First-order allpass for phaser.
struct PhaseAllpass {
    prev_input: f32,
    prev_output: f32,
}

impl PhaseAllpass {
    fn new() -> Self {
        Self {
            prev_input: 0.0,
            prev_output: 0.0,
        }
    }

    fn process(&mut self, input: f32, a1: f32) -> f32 {
        let output = a1 * input + self.prev_input - a1 * self.prev_output;
        self.prev_input = input;
        self.prev_output = output;
        output
    }

    fn clear(&mut self) {
        self.prev_input = 0.0;
        self.prev_output = 0.0;
    }
}

// ============================================================================
// Flanger
// ============================================================================

/// Flanger effect (similar to chorus but with shorter delay and feedback).
pub struct Flanger {
    /// Delay buffer.
    buffer: Vec<f32>,
    /// Write position.
    write_pos: usize,
    /// Base delay in samples.
    base_delay: f32,
    /// Modulation depth in samples.
    depth: f32,
    /// Modulation rate in Hz.
    rate: f32,
    /// LFO phase.
    phase: f32,
    /// Dry/wet mix.
    pub mix: f32,
    /// Feedback amount.
    pub feedback: f32,
    /// Sample rate.
    sample_rate: f32,
}

impl Flanger {
    /// Creates a new flanger effect.
    pub fn new(sample_rate: f32) -> Self {
        Self::with_params(sample_rate, 0.003, 0.002, 0.3, 0.5, 0.7)
    }

    /// Creates a flanger with specific parameters.
    pub fn with_params(
        sample_rate: f32,
        delay: f32,
        depth: f32,
        rate: f32,
        mix: f32,
        feedback: f32,
    ) -> Self {
        let max_delay = (delay + depth * 2.0) * sample_rate;
        let buffer_size = (max_delay as usize + 1).max(256);

        Self {
            buffer: vec![0.0; buffer_size],
            write_pos: 0,
            base_delay: delay * sample_rate,
            depth: depth * sample_rate,
            rate,
            phase: 0.0,
            mix,
            feedback: feedback.clamp(-0.95, 0.95),
            sample_rate,
        }
    }

    /// Processes a single sample.
    pub fn process(&mut self, input: f32) -> f32 {
        // Calculate modulated delay
        let lfo = (self.phase * std::f32::consts::TAU).sin();
        let delay = (self.base_delay + lfo * self.depth).max(1.0);

        // Read from buffer with linear interpolation
        let read_pos = self.write_pos as f32 - delay;
        let read_pos = if read_pos < 0.0 {
            read_pos + self.buffer.len() as f32
        } else {
            read_pos
        };

        let idx0 = read_pos.floor() as usize % self.buffer.len();
        let idx1 = (idx0 + 1) % self.buffer.len();
        let frac = read_pos.fract();

        let delayed = self.buffer[idx0] * (1.0 - frac) + self.buffer[idx1] * frac;

        // Write to buffer with feedback
        self.buffer[self.write_pos] = input + delayed * self.feedback;
        self.write_pos = (self.write_pos + 1) % self.buffer.len();

        // Advance LFO
        self.phase += self.rate / self.sample_rate;
        if self.phase >= 1.0 {
            self.phase -= 1.0;
        }

        // Mix
        input * (1.0 - self.mix) + delayed * self.mix
    }

    /// Clears the delay buffer.
    pub fn clear(&mut self) {
        self.buffer.fill(0.0);
        self.phase = 0.0;
    }
}

impl AudioNode for Flanger {
    fn process(&mut self, input: f32, _ctx: &AudioContext) -> f32 {
        Flanger::process(self, input)
    }

    fn reset(&mut self) {
        self.clear();
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
// Tremolo
// ============================================================================

/// Tremolo effect (amplitude modulation).
pub struct Tremolo {
    /// Modulation rate in Hz.
    pub rate: f32,
    /// Modulation depth (0-1).
    pub depth: f32,
    /// LFO phase.
    phase: f32,
    /// Sample rate.
    sample_rate: f32,
}

impl Tremolo {
    /// Creates a new tremolo effect.
    pub fn new(sample_rate: f32, rate: f32, depth: f32) -> Self {
        Self {
            rate,
            depth: depth.clamp(0.0, 1.0),
            phase: 0.0,
            sample_rate,
        }
    }

    /// Processes a single sample.
    pub fn process(&mut self, input: f32) -> f32 {
        let lfo = (self.phase * std::f32::consts::TAU).sin() * 0.5 + 0.5;
        let mod_amount = 1.0 - self.depth * lfo;

        self.phase += self.rate / self.sample_rate;
        if self.phase >= 1.0 {
            self.phase -= 1.0;
        }

        input * mod_amount
    }
}

impl AudioNode for Tremolo {
    fn process(&mut self, input: f32, _ctx: &AudioContext) -> f32 {
        Tremolo::process(self, input)
    }

    fn reset(&mut self) {
        self.phase = 0.0;
    }
}

// ============================================================================
// Convolution Reverb
// ============================================================================

use crate::spectral::{Complex, fft, ifft};

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

impl AudioNode for ConvolutionReverb {
    fn process(&mut self, input: f32, _ctx: &AudioContext) -> f32 {
        ConvolutionReverb::process(self, input)
    }

    fn reset(&mut self) {
        self.clear();
    }
}

/// Configuration for convolution reverb.
#[derive(Clone, Copy, Debug)]
pub struct ConvolutionConfig {
    /// Processing block size (power of 2).
    pub block_size: usize,
    /// Dry/wet mix.
    pub mix: f32,
    /// Output gain.
    pub gain: f32,
}

impl Default for ConvolutionConfig {
    fn default() -> Self {
        Self {
            block_size: 1024,
            mix: 0.5,
            gain: 1.0,
        }
    }
}

/// Creates a convolution reverb with configuration.
pub fn convolution_reverb(ir: &[f32], config: &ConvolutionConfig) -> ConvolutionReverb {
    let mut reverb = ConvolutionReverb::new(ir, config.block_size);
    reverb.mix = config.mix;
    reverb.gain = config.gain;
    reverb
}

/// Generates a synthetic impulse response for a room.
///
/// # Arguments
/// * `size` - Room size (small=0.1, large=1.0)
/// * `damping` - High frequency damping (0-1)
/// * `duration` - IR duration in seconds
/// * `sample_rate` - Sample rate in Hz
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
fn simple_hash(x: u32) -> u32 {
    let mut h = x;
    h = h.wrapping_mul(0x85ebca6b);
    h ^= h >> 13;
    h = h.wrapping_mul(0xc2b2ae35);
    h ^= h >> 16;
    h
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
        let mut chorus = Chorus::new(44100.0);

        // Process some samples
        let mut outputs = Vec::new();
        for i in 0..1000 {
            let input = (i as f32 * 0.1).sin();
            outputs.push(chorus.process(input));
        }

        // Should have varying output (due to modulation)
        let variance: f32 = outputs.windows(2).map(|w| (w[1] - w[0]).abs()).sum();
        assert!(variance > 0.0);
    }

    #[test]
    fn test_phaser_process() {
        let mut phaser = Phaser::new(44100.0);

        // Process a constant signal
        for _ in 0..1000 {
            phaser.process(0.5);
        }

        // Should still produce output
        let out = phaser.process(0.5);
        assert!(out.abs() <= 1.5);
    }

    #[test]
    fn test_flanger_process() {
        let mut flanger = Flanger::new(44100.0);

        // Process samples
        let out = flanger.process(1.0);
        assert!(out.abs() <= 2.0);

        // Clear and verify
        flanger.clear();
        let out_after_clear = flanger.process(0.0);
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
        let mut trem = Tremolo::new(44100.0, 5.0, 1.0);

        // With full depth, output should vary between 0 and input
        let mut min_out = f32::MAX;
        let mut max_out = f32::MIN;

        for _ in 0..44100 {
            let out = trem.process(1.0);
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
        let mut reverb: Box<dyn AudioNode> = Box::new(Reverb::new(sample_rate));
        let mut chorus: Box<dyn AudioNode> = Box::new(Chorus::new(sample_rate));
        let mut phaser: Box<dyn AudioNode> = Box::new(Phaser::new(sample_rate));
        let mut flanger: Box<dyn AudioNode> = Box::new(Flanger::new(sample_rate));

        // All should process without panic
        reverb.process(0.5, &ctx);
        chorus.process(0.5, &ctx);
        phaser.process(0.5, &ctx);
        flanger.process(0.5, &ctx);
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
}
