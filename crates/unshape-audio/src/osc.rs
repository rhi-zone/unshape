//! Audio oscillators.
//!
//! Pure functions of phase - no internal state.
//! Phase is in [0, 1] representing one cycle.

use std::f32::consts::TAU;

/// Sine wave oscillator.
///
/// # Arguments
/// * `phase` - Phase in [0, 1], wraps automatically.
///
/// # Returns
/// Value in [-1, 1].
#[inline]
pub fn sine(phase: f32) -> f32 {
    (phase * TAU).sin()
}

/// Square wave oscillator.
///
/// # Arguments
/// * `phase` - Phase in [0, 1].
///
/// # Returns
/// Value in [-1, 1].
#[inline]
pub fn square(phase: f32) -> f32 {
    if phase.fract() < 0.5 { 1.0 } else { -1.0 }
}

/// Pulse wave oscillator with variable duty cycle.
///
/// # Arguments
/// * `phase` - Phase in [0, 1].
/// * `duty` - Duty cycle in [0, 1]. 0.5 = square wave.
///
/// # Returns
/// Value in [-1, 1].
#[inline]
pub fn pulse(phase: f32, duty: f32) -> f32 {
    if phase.fract() < duty { 1.0 } else { -1.0 }
}

/// Sawtooth wave oscillator (rising).
///
/// # Arguments
/// * `phase` - Phase in [0, 1].
///
/// # Returns
/// Value in [-1, 1].
#[inline]
pub fn saw(phase: f32) -> f32 {
    2.0 * phase.fract() - 1.0
}

/// Reverse sawtooth wave oscillator (falling).
///
/// # Arguments
/// * `phase` - Phase in [0, 1].
///
/// # Returns
/// Value in [-1, 1].
#[inline]
pub fn saw_rev(phase: f32) -> f32 {
    1.0 - 2.0 * phase.fract()
}

/// Triangle wave oscillator.
///
/// # Arguments
/// * `phase` - Phase in [0, 1].
///
/// # Returns
/// Value in [-1, 1].
#[inline]
pub fn triangle(phase: f32) -> f32 {
    let p = phase.fract();
    if p < 0.5 {
        4.0 * p - 1.0
    } else {
        3.0 - 4.0 * p
    }
}

/// Converts frequency (Hz) and time (seconds) to phase.
///
/// # Example
/// ```
/// use unshape_audio::osc::{freq_to_phase, sine};
///
/// let frequency = 440.0; // A4
/// let time = 0.001; // 1ms
/// let phase = freq_to_phase(frequency, time);
/// let sample = sine(phase);
/// ```
#[inline]
pub fn freq_to_phase(frequency: f32, time: f32) -> f32 {
    frequency * time
}

/// Generates a phase value from sample index and sample rate.
///
/// # Arguments
/// * `frequency` - Oscillator frequency in Hz.
/// * `sample_index` - Current sample number.
/// * `sample_rate` - Samples per second.
#[inline]
pub fn sample_to_phase(frequency: f32, sample_index: u64, sample_rate: f32) -> f32 {
    frequency * (sample_index as f32 / sample_rate)
}

/// Polyblep anti-aliasing correction for naive waveforms.
///
/// Reduces aliasing artifacts at discontinuities.
#[inline]
fn poly_blep(t: f32, dt: f32) -> f32 {
    if t < dt {
        let t = t / dt;
        2.0 * t - t * t - 1.0
    } else if t > 1.0 - dt {
        let t = (t - 1.0) / dt;
        t * t + 2.0 * t + 1.0
    } else {
        0.0
    }
}

/// Band-limited square wave using polyblep.
///
/// Reduces aliasing compared to naive square.
#[inline]
pub fn square_blep(phase: f32, dt: f32) -> f32 {
    let p = phase.fract();
    let mut value = if p < 0.5 { 1.0 } else { -1.0 };
    value += poly_blep(p, dt);
    value -= poly_blep((p + 0.5).fract(), dt);
    value
}

/// Band-limited sawtooth wave using polyblep.
#[inline]
pub fn saw_blep(phase: f32, dt: f32) -> f32 {
    let p = phase.fract();
    let mut value = 2.0 * p - 1.0;
    value -= poly_blep(p, dt);
    value
}

/// A wavetable for wavetable synthesis.
///
/// Contains a single cycle waveform that can be sampled with linear interpolation.
#[derive(Debug, Clone)]
pub struct Wavetable {
    /// The waveform samples (one complete cycle).
    pub samples: Vec<f32>,
}

impl Wavetable {
    /// Creates a wavetable from raw samples.
    pub fn from_samples(samples: Vec<f32>) -> Self {
        Self { samples }
    }

    /// Creates a wavetable from a function sampled at `size` points.
    pub fn from_fn<F>(f: F, size: usize) -> Self
    where
        F: Fn(f32) -> f32,
    {
        let samples: Vec<f32> = (0..size).map(|i| f(i as f32 / size as f32)).collect();
        Self { samples }
    }

    /// Creates a sine wavetable.
    pub fn sine(size: usize) -> Self {
        Self::from_fn(sine, size)
    }

    /// Creates a saw wavetable.
    pub fn saw(size: usize) -> Self {
        Self::from_fn(saw, size)
    }

    /// Creates a square wavetable.
    pub fn square(size: usize) -> Self {
        Self::from_fn(square, size)
    }

    /// Creates a triangle wavetable.
    pub fn triangle(size: usize) -> Self {
        Self::from_fn(triangle, size)
    }

    /// Returns the wavetable size.
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    /// Returns true if the wavetable is empty.
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// Samples the wavetable at the given phase with linear interpolation.
    ///
    /// # Arguments
    /// * `phase` - Phase in [0, 1], wraps automatically.
    ///
    /// # Returns
    /// Interpolated value, typically in [-1, 1].
    #[inline]
    pub fn sample(&self, phase: f32) -> f32 {
        if self.samples.is_empty() {
            return 0.0;
        }

        let len = self.samples.len() as f32;
        let pos = phase.fract() * len;
        let idx0 = pos.floor() as usize % self.samples.len();
        let idx1 = (idx0 + 1) % self.samples.len();
        let frac = pos.fract();

        self.samples[idx0] * (1.0 - frac) + self.samples[idx1] * frac
    }

    /// Samples without interpolation (nearest neighbor).
    #[inline]
    pub fn sample_nearest(&self, phase: f32) -> f32 {
        if self.samples.is_empty() {
            return 0.0;
        }

        let len = self.samples.len() as f32;
        let idx = (phase.fract() * len) as usize % self.samples.len();
        self.samples[idx]
    }

    /// Normalizes the wavetable to [-1, 1] range.
    pub fn normalize(&mut self) {
        if self.samples.is_empty() {
            return;
        }

        let max = self
            .samples
            .iter()
            .map(|s| s.abs())
            .fold(0.0f32, |a, b| a.max(b));

        if max > 0.0 {
            for s in &mut self.samples {
                *s /= max;
            }
        }
    }
}

/// A wavetable bank for morphing between multiple waveforms.
#[derive(Debug, Clone)]
pub struct WavetableBank {
    /// The wavetables to morph between.
    tables: Vec<Wavetable>,
}

impl WavetableBank {
    /// Creates a wavetable bank from a list of wavetables.
    pub fn new(tables: Vec<Wavetable>) -> Self {
        Self { tables }
    }

    /// Creates a standard bank with sine, triangle, saw, and square.
    pub fn standard(size: usize) -> Self {
        Self::new(vec![
            Wavetable::sine(size),
            Wavetable::triangle(size),
            Wavetable::saw(size),
            Wavetable::square(size),
        ])
    }

    /// Returns the number of wavetables.
    pub fn len(&self) -> usize {
        self.tables.len()
    }

    /// Returns true if the bank is empty.
    pub fn is_empty(&self) -> bool {
        self.tables.is_empty()
    }

    /// Samples the wavetable bank with morphing between tables.
    ///
    /// # Arguments
    /// * `phase` - Oscillator phase in [0, 1].
    /// * `morph` - Morph position in [0, 1], where 0 = first table, 1 = last table.
    #[inline]
    pub fn sample(&self, phase: f32, morph: f32) -> f32 {
        if self.tables.is_empty() {
            return 0.0;
        }

        if self.tables.len() == 1 {
            return self.tables[0].sample(phase);
        }

        // Map morph [0,1] to table indices
        let morph = morph.clamp(0.0, 1.0);
        let table_pos = morph * (self.tables.len() - 1) as f32;
        let table_idx = table_pos.floor() as usize;
        let table_frac = table_pos.fract();

        if table_idx >= self.tables.len() - 1 {
            return self.tables[self.tables.len() - 1].sample(phase);
        }

        // Crossfade between adjacent tables
        let sample0 = self.tables[table_idx].sample(phase);
        let sample1 = self.tables[table_idx + 1].sample(phase);

        sample0 * (1.0 - table_frac) + sample1 * table_frac
    }
}

/// A wavetable oscillator with state for continuous playback.
#[derive(Debug)]
pub struct WavetableOsc {
    /// The wavetable to sample.
    pub wavetable: Wavetable,
    /// Current phase.
    pub phase: f32,
    /// Phase increment per sample.
    pub phase_inc: f32,
}

impl WavetableOsc {
    /// Creates a wavetable oscillator.
    pub fn new(wavetable: Wavetable, _sample_rate: f32) -> Self {
        Self {
            wavetable,
            phase: 0.0,
            phase_inc: 0.0,
        }
    }

    /// Sets the oscillator frequency.
    pub fn set_frequency(&mut self, frequency: f32, sample_rate: f32) {
        self.phase_inc = frequency / sample_rate;
    }

    /// Generates the next sample.
    #[inline]
    pub fn next_sample(&mut self) -> f32 {
        let sample = self.wavetable.sample(self.phase);
        self.phase = (self.phase + self.phase_inc).fract();
        sample
    }

    /// Generates a buffer of samples.
    pub fn generate(&mut self, buffer: &mut [f32]) {
        for sample in buffer {
            *sample = self.next_sample();
        }
    }

    /// Resets the phase to zero.
    pub fn reset(&mut self) {
        self.phase = 0.0;
    }
}

/// Creates an additive synthesis wavetable from harmonics.
///
/// # Arguments
/// * `harmonics` - Pairs of (harmonic number, amplitude).
/// * `size` - Wavetable size.
pub fn additive_wavetable(harmonics: &[(u32, f32)], size: usize) -> Wavetable {
    let mut samples = vec![0.0f32; size];

    for (harmonic, amplitude) in harmonics {
        for i in 0..size {
            let phase = (i as f32 / size as f32) * (*harmonic as f32);
            samples[i] += amplitude * (phase * TAU).sin();
        }
    }

    let mut table = Wavetable::from_samples(samples);
    table.normalize();
    table
}

/// Creates a wavetable from supersaw (detuned saws).
///
/// # Arguments
/// * `voices` - Number of saw voices.
/// * `detune` - Detune amount in semitones.
/// * `size` - Wavetable size.
pub fn supersaw_wavetable(voices: usize, detune: f32, size: usize) -> Wavetable {
    let mut samples = vec![0.0f32; size];
    let detune_ratio = 2.0f32.powf(detune / 12.0);

    for voice in 0..voices {
        let ratio = if voices == 1 {
            1.0
        } else {
            let t = voice as f32 / (voices - 1) as f32;
            1.0 / detune_ratio + t * (detune_ratio - 1.0 / detune_ratio)
        };

        for i in 0..size {
            let phase = (i as f32 / size as f32) * ratio;
            samples[i] += saw(phase);
        }
    }

    let mut table = Wavetable::from_samples(samples);
    table.normalize();
    table
}

// ===========================================================================
// FM Synthesis
// ===========================================================================

/// An FM operator (oscillator with ratio and modulation capabilities).
#[derive(Debug, Clone)]
pub struct FmOperator {
    /// Phase accumulator.
    pub phase: f32,
    /// Frequency ratio relative to the fundamental.
    pub ratio: f32,
    /// Output level (0.0 to 1.0).
    pub level: f32,
    /// Feedback amount (for self-modulation).
    pub feedback: f32,
    /// Previous output for feedback.
    prev_output: f32,
}

impl Default for FmOperator {
    fn default() -> Self {
        Self {
            phase: 0.0,
            ratio: 1.0,
            level: 1.0,
            feedback: 0.0,
            prev_output: 0.0,
        }
    }
}

impl FmOperator {
    /// Creates a new FM operator with the given ratio.
    pub fn new(ratio: f32) -> Self {
        Self {
            ratio,
            ..Default::default()
        }
    }

    /// Generates a sample with phase modulation input.
    ///
    /// # Arguments
    /// * `base_freq` - Base frequency in Hz.
    /// * `sample_rate` - Sample rate in Hz.
    /// * `phase_mod` - Phase modulation input (in radians).
    #[inline]
    pub fn tick(&mut self, base_freq: f32, sample_rate: f32, phase_mod: f32) -> f32 {
        let freq = base_freq * self.ratio;
        let phase_inc = freq / sample_rate;

        // Add feedback from previous output
        let feedback_mod = self.prev_output * self.feedback * TAU;

        // Calculate output
        let output = ((self.phase * TAU) + phase_mod + feedback_mod).sin();
        self.prev_output = output;

        // Advance phase
        self.phase = (self.phase + phase_inc).fract();

        output * self.level
    }

    /// Resets the operator phase.
    pub fn reset(&mut self) {
        self.phase = 0.0;
        self.prev_output = 0.0;
    }
}

/// FM synthesis algorithm defining operator connections.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FmAlgorithm {
    /// 4 -> 3 -> 2 -> 1 (serial stack)
    Serial,
    /// (2+3+4) -> 1 (parallel into carrier)
    Parallel,
    /// 3 -> 2 -> 1, 4 separate output
    TwoCarrier,
    /// 4 -> 3, 2 -> 1, both output
    DualSerial,
    /// Custom algorithm (implement via FmSynth::tick_custom)
    Custom,
}

/// A 4-operator FM synthesizer.
#[derive(Debug, Clone)]
pub struct FmSynth {
    /// The four operators.
    pub ops: [FmOperator; 4],
    /// Modulation index (global multiplier for modulation depth).
    pub mod_index: f32,
    /// Algorithm for operator routing.
    pub algorithm: FmAlgorithm,
}

impl Default for FmSynth {
    fn default() -> Self {
        Self {
            ops: [
                FmOperator::new(1.0), // Carrier
                FmOperator::new(1.0), // Modulator
                FmOperator::new(1.0),
                FmOperator::new(1.0),
            ],
            mod_index: 1.0,
            algorithm: FmAlgorithm::Serial,
        }
    }
}

impl FmSynth {
    /// Creates a new FM synth with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets operator ratios.
    pub fn with_ratios(mut self, ratios: [f32; 4]) -> Self {
        for (op, ratio) in self.ops.iter_mut().zip(ratios.iter()) {
            op.ratio = *ratio;
        }
        self
    }

    /// Sets operator levels.
    pub fn with_levels(mut self, levels: [f32; 4]) -> Self {
        for (op, level) in self.ops.iter_mut().zip(levels.iter()) {
            op.level = *level;
        }
        self
    }

    /// Generates a sample.
    #[inline]
    pub fn tick(&mut self, freq: f32, sample_rate: f32) -> f32 {
        let mod_scale = self.mod_index * TAU;

        match self.algorithm {
            FmAlgorithm::Serial => {
                // 4 -> 3 -> 2 -> 1
                let op4 = self.ops[3].tick(freq, sample_rate, 0.0);
                let op3 = self.ops[2].tick(freq, sample_rate, op4 * mod_scale);
                let op2 = self.ops[1].tick(freq, sample_rate, op3 * mod_scale);
                self.ops[0].tick(freq, sample_rate, op2 * mod_scale)
            }
            FmAlgorithm::Parallel => {
                // (2+3+4) -> 1
                let op2 = self.ops[1].tick(freq, sample_rate, 0.0);
                let op3 = self.ops[2].tick(freq, sample_rate, 0.0);
                let op4 = self.ops[3].tick(freq, sample_rate, 0.0);
                let mod_sum = (op2 + op3 + op4) * mod_scale;
                self.ops[0].tick(freq, sample_rate, mod_sum)
            }
            FmAlgorithm::TwoCarrier => {
                // 3 -> 2 -> 1 (output), 4 (output)
                let op3 = self.ops[2].tick(freq, sample_rate, 0.0);
                let op2 = self.ops[1].tick(freq, sample_rate, op3 * mod_scale);
                let op1 = self.ops[0].tick(freq, sample_rate, op2 * mod_scale);
                let op4 = self.ops[3].tick(freq, sample_rate, 0.0);
                (op1 + op4) * 0.5
            }
            FmAlgorithm::DualSerial => {
                // 4 -> 3 (output), 2 -> 1 (output)
                let op4 = self.ops[3].tick(freq, sample_rate, 0.0);
                let op3 = self.ops[2].tick(freq, sample_rate, op4 * mod_scale);
                let op2 = self.ops[1].tick(freq, sample_rate, 0.0);
                let op1 = self.ops[0].tick(freq, sample_rate, op2 * mod_scale);
                (op1 + op3) * 0.5
            }
            FmAlgorithm::Custom => 0.0, // Use tick_custom instead
        }
    }

    /// Generates a buffer of samples.
    pub fn generate(&mut self, freq: f32, sample_rate: f32, buffer: &mut [f32]) {
        for sample in buffer {
            *sample = self.tick(freq, sample_rate);
        }
    }

    /// Resets all operators.
    pub fn reset(&mut self) {
        for op in &mut self.ops {
            op.reset();
        }
    }
}

/// Simple 2-operator FM for basic use cases.
#[derive(Debug, Clone)]
pub struct FmOsc {
    /// Carrier phase.
    pub carrier_phase: f32,
    /// Modulator phase.
    pub mod_phase: f32,
    /// Modulator frequency ratio.
    pub mod_ratio: f32,
    /// Modulation index.
    pub mod_index: f32,
}

impl Default for FmOsc {
    fn default() -> Self {
        Self {
            carrier_phase: 0.0,
            mod_phase: 0.0,
            mod_ratio: 1.0,
            mod_index: 1.0,
        }
    }
}

impl FmOsc {
    /// Creates a new 2-operator FM oscillator.
    pub fn new(mod_ratio: f32, mod_index: f32) -> Self {
        Self {
            mod_ratio,
            mod_index,
            ..Default::default()
        }
    }

    /// Generates a sample.
    #[inline]
    pub fn tick(&mut self, freq: f32, sample_rate: f32) -> f32 {
        let carrier_inc = freq / sample_rate;
        let mod_inc = freq * self.mod_ratio / sample_rate;

        // Calculate modulator output
        let modulator = (self.mod_phase * TAU).sin();

        // Calculate carrier with phase modulation
        let carrier = ((self.carrier_phase * TAU) + modulator * self.mod_index * TAU).sin();

        // Advance phases
        self.carrier_phase = (self.carrier_phase + carrier_inc).fract();
        self.mod_phase = (self.mod_phase + mod_inc).fract();

        carrier
    }

    /// Generates a buffer of samples.
    pub fn generate(&mut self, freq: f32, sample_rate: f32, buffer: &mut [f32]) {
        for sample in buffer {
            *sample = self.tick(freq, sample_rate);
        }
    }

    /// Resets the oscillator.
    pub fn reset(&mut self) {
        self.carrier_phase = 0.0;
        self.mod_phase = 0.0;
    }
}

/// Creates classic FM timbres.
pub mod fm_presets {
    use super::{FmAlgorithm, FmSynth};

    /// Electric piano-like timbre.
    pub fn electric_piano() -> FmSynth {
        let mut synth = FmSynth {
            algorithm: FmAlgorithm::Serial,
            mod_index: 1.5,
            ..Default::default()
        };
        synth = synth.with_ratios([1.0, 1.0, 3.0, 1.0]);
        synth.with_levels([1.0, 0.7, 0.3, 0.0])
    }

    /// Bell-like timbre.
    pub fn bell() -> FmSynth {
        let mut synth = FmSynth {
            algorithm: FmAlgorithm::Serial,
            mod_index: 3.0,
            ..Default::default()
        };
        synth = synth.with_ratios([1.0, 3.5, 1.0, 7.0]);
        synth.with_levels([1.0, 0.8, 0.0, 0.5])
    }

    /// Brass-like timbre.
    pub fn brass() -> FmSynth {
        let mut synth = FmSynth {
            algorithm: FmAlgorithm::Serial,
            mod_index: 2.0,
            ..Default::default()
        };
        synth = synth.with_ratios([1.0, 1.0, 1.0, 1.0]);
        synth.with_levels([1.0, 0.9, 0.7, 0.0])
    }

    /// Bass-like timbre.
    pub fn bass() -> FmSynth {
        let mut synth = FmSynth {
            algorithm: FmAlgorithm::Serial,
            mod_index: 1.0,
            ..Default::default()
        };
        synth = synth.with_ratios([1.0, 2.0, 1.0, 1.0]);
        synth.with_levels([1.0, 0.5, 0.0, 0.0])
    }

    /// Organ-like timbre.
    pub fn organ() -> FmSynth {
        let mut synth = FmSynth {
            algorithm: FmAlgorithm::Parallel,
            mod_index: 0.5,
            ..Default::default()
        };
        synth = synth.with_ratios([1.0, 2.0, 3.0, 4.0]);
        synth.with_levels([1.0, 0.5, 0.3, 0.2])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sine_range() {
        for i in 0..100 {
            let phase = i as f32 / 100.0;
            let v = sine(phase);
            assert!(
                (-1.0..=1.0).contains(&v),
                "sine({}) = {} out of range",
                phase,
                v
            );
        }
    }

    #[test]
    fn test_sine_zero_crossing() {
        assert!((sine(0.0)).abs() < 0.001);
        assert!((sine(0.5)).abs() < 0.001);
    }

    #[test]
    fn test_sine_peaks() {
        assert!((sine(0.25) - 1.0).abs() < 0.001);
        assert!((sine(0.75) + 1.0).abs() < 0.001);
    }

    #[test]
    fn test_square_range() {
        for i in 0..100 {
            let phase = i as f32 / 100.0;
            let v = square(phase);
            assert!(v == 1.0 || v == -1.0);
        }
    }

    #[test]
    fn test_square_duty() {
        assert_eq!(square(0.0), 1.0);
        assert_eq!(square(0.49), 1.0);
        assert_eq!(square(0.51), -1.0);
        assert_eq!(square(0.99), -1.0);
    }

    #[test]
    fn test_saw_range() {
        for i in 0..100 {
            let phase = i as f32 / 100.0;
            let v = saw(phase);
            assert!(
                (-1.0..=1.0).contains(&v),
                "saw({}) = {} out of range",
                phase,
                v
            );
        }
    }

    #[test]
    fn test_triangle_range() {
        for i in 0..100 {
            let phase = i as f32 / 100.0;
            let v = triangle(phase);
            assert!(
                (-1.0..=1.0).contains(&v),
                "triangle({}) = {} out of range",
                phase,
                v
            );
        }
    }

    #[test]
    fn test_triangle_peaks() {
        assert!((triangle(0.0) + 1.0).abs() < 0.001);
        assert!((triangle(0.25)).abs() < 0.001);
        assert!((triangle(0.5) - 1.0).abs() < 0.001);
        assert!((triangle(0.75)).abs() < 0.001);
    }

    #[test]
    fn test_pulse_duty() {
        // 25% duty cycle
        assert_eq!(pulse(0.0, 0.25), 1.0);
        assert_eq!(pulse(0.2, 0.25), 1.0);
        assert_eq!(pulse(0.3, 0.25), -1.0);
        assert_eq!(pulse(0.9, 0.25), -1.0);
    }

    #[test]
    fn test_freq_to_phase() {
        // 1 Hz at t=1 should give phase 1.0 (one full cycle)
        assert!((freq_to_phase(1.0, 1.0) - 1.0).abs() < 0.001);
        // 440 Hz at t=1/440 should give phase 1.0
        assert!((freq_to_phase(440.0, 1.0 / 440.0) - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_wavetable_from_fn() {
        let table = Wavetable::from_fn(sine, 256);
        assert_eq!(table.len(), 256);

        // Sample at 0.25 should be ~1.0 (sine peak)
        let sample = table.sample(0.25);
        assert!((sample - 1.0).abs() < 0.05, "sample = {}", sample);
    }

    #[test]
    fn test_wavetable_interpolation() {
        let table = Wavetable::sine(256);

        // Sampling between samples should interpolate
        let s0 = table.sample(0.0);
        let s1 = table.sample(0.5 / 256.0);
        let s2 = table.sample(1.0 / 256.0);

        // s1 should be between s0 and s2
        assert!(s1 >= s0.min(s2) && s1 <= s0.max(s2));
    }

    #[test]
    fn test_wavetable_wrapping() {
        let table = Wavetable::sine(256);

        // Phase wrapping should work
        let s1 = table.sample(0.0);
        let s2 = table.sample(1.0);
        let s3 = table.sample(2.0);

        assert!((s1 - s2).abs() < 0.001);
        assert!((s1 - s3).abs() < 0.001);
    }

    #[test]
    fn test_wavetable_bank_morph() {
        let bank = WavetableBank::standard(256);

        // morph=0 should be sine
        let sine_sample = bank.sample(0.25, 0.0);
        assert!((sine_sample - 1.0).abs() < 0.1);

        // morph=1 should be square
        let square_sample = bank.sample(0.25, 1.0);
        assert!((square_sample - 1.0).abs() < 0.1);

        // morph=0.5 blends triangle (idx 1) and saw (idx 2)
        // triangle at 0.25 ≈ 1.0, saw at 0.25 ≈ -0.5
        // blended result should be in valid [-1, 1] range
        let mid_sample = bank.sample(0.25, 0.5);
        assert!(mid_sample >= -1.0 && mid_sample <= 1.0);
    }

    #[test]
    fn test_wavetable_osc() {
        let table = Wavetable::sine(256);
        let mut osc = WavetableOsc::new(table, 44100.0);
        osc.set_frequency(440.0, 44100.0);

        // Generate some samples
        let mut buffer = vec![0.0; 100];
        osc.generate(&mut buffer);

        // All samples should be in valid range
        for s in &buffer {
            assert!((-1.0..=1.0).contains(s), "sample = {}", s);
        }
    }

    #[test]
    fn test_additive_wavetable() {
        // First 4 odd harmonics (square-ish)
        let harmonics = vec![(1, 1.0), (3, 0.33), (5, 0.2), (7, 0.14)];
        let table = additive_wavetable(&harmonics, 512);

        assert_eq!(table.len(), 512);

        // Should be normalized
        let max: f32 = table
            .samples
            .iter()
            .map(|s| s.abs())
            .fold(0.0, |a, b| a.max(b));
        assert!((max - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_supersaw_wavetable() {
        let table = supersaw_wavetable(7, 0.5, 512);

        assert_eq!(table.len(), 512);

        // Should be normalized
        let max: f32 = table
            .samples
            .iter()
            .map(|s| s.abs())
            .fold(0.0, |a, b| a.max(b));
        assert!((max - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_fm_operator() {
        let mut op = FmOperator::new(1.0);
        let sample_rate = 44100.0;
        let freq = 440.0;

        // Generate samples and check range
        for _ in 0..1000 {
            let s = op.tick(freq, sample_rate, 0.0);
            assert!((-1.0..=1.0).contains(&s));
        }
    }

    #[test]
    fn test_fm_operator_ratio() {
        let mut op = FmOperator::new(2.0);

        // With ratio=2, phase should advance twice as fast
        let s1 = op.tick(440.0, 44100.0, 0.0);
        assert!((-1.0..=1.0).contains(&s1));
    }

    #[test]
    fn test_fm_osc() {
        let mut fm = FmOsc::new(2.0, 1.0);

        // Generate 100 samples
        let mut buffer = vec![0.0; 100];
        fm.generate(440.0, 44100.0, &mut buffer);

        // All samples should be in valid range
        for s in &buffer {
            assert!((-1.0..=1.0).contains(s), "sample = {}", s);
        }
    }

    #[test]
    fn test_fm_osc_modulation() {
        // With high mod_index, output should have more harmonics
        let mut fm_low = FmOsc::new(1.0, 0.1);
        let mut fm_high = FmOsc::new(1.0, 5.0);

        let mut buf_low = vec![0.0; 1000];
        let mut buf_high = vec![0.0; 1000];

        fm_low.generate(440.0, 44100.0, &mut buf_low);
        fm_high.generate(440.0, 44100.0, &mut buf_high);

        // Higher mod index should produce more varied samples
        // (rougher waveform with more zero crossings)
        let crossings_low = buf_low.windows(2).filter(|w| w[0] * w[1] < 0.0).count();
        let crossings_high = buf_high.windows(2).filter(|w| w[0] * w[1] < 0.0).count();

        assert!(crossings_high > crossings_low);
    }

    #[test]
    fn test_fm_synth_serial() {
        let mut synth = FmSynth {
            algorithm: FmAlgorithm::Serial,
            mod_index: 1.0,
            ..Default::default()
        };

        let mut buffer = vec![0.0; 100];
        synth.generate(440.0, 44100.0, &mut buffer);

        for s in &buffer {
            assert!((-2.0..=2.0).contains(s), "sample = {}", s);
        }
    }

    #[test]
    fn test_fm_synth_parallel() {
        let mut synth = FmSynth {
            algorithm: FmAlgorithm::Parallel,
            mod_index: 0.5,
            ..Default::default()
        };

        let mut buffer = vec![0.0; 100];
        synth.generate(440.0, 44100.0, &mut buffer);

        for s in &buffer {
            assert!((-2.0..=2.0).contains(s), "sample = {}", s);
        }
    }

    #[test]
    fn test_fm_presets() {
        use fm_presets::*;

        // Just verify presets create valid synths
        let _ep = electric_piano();
        let _bell = bell();
        let _brass = brass();
        let _bass = bass();
        let _organ = organ();
    }

    #[test]
    fn test_fm_synth_reset() {
        let mut synth = FmSynth::new();

        // Generate some samples
        for _ in 0..100 {
            synth.tick(440.0, 44100.0);
        }

        // Reset
        synth.reset();

        // All phases should be 0
        for op in &synth.ops {
            assert_eq!(op.phase, 0.0);
        }
    }
}

// ============================================================================
// Invariant tests - oscillator properties and frequency accuracy
// ============================================================================

#[cfg(all(test, feature = "invariant-tests"))]
mod invariant_tests {
    use super::*;
    use rustfft::{FftPlanner, num_complex::Complex};

    const SAMPLE_RATE: f32 = 44100.0;
    const FFT_SIZE: usize = 4096;

    /// Generate samples from an oscillator at a given frequency
    fn generate_signal<F: Fn(f32) -> f32>(
        osc_fn: F,
        frequency: f32,
        num_samples: usize,
    ) -> Vec<f32> {
        (0..num_samples)
            .map(|i| {
                let phase = sample_to_phase(frequency, i as u64, SAMPLE_RATE);
                osc_fn(phase)
            })
            .collect()
    }

    /// Find the bin with maximum magnitude (ignoring DC)
    fn find_peak_bin(signal: &[f32]) -> (usize, f32) {
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(signal.len());

        let mut buffer: Vec<Complex<f32>> = signal.iter().map(|&x| Complex::new(x, 0.0)).collect();
        fft.process(&mut buffer);

        let magnitudes: Vec<f32> = buffer[1..signal.len() / 2]
            .iter()
            .map(|c| c.norm())
            .collect();

        let (max_idx, &max_mag) = magnitudes
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap();

        (max_idx + 1, max_mag) // +1 because we skipped bin 0
    }

    /// Convert bin index to frequency
    fn bin_to_freq(bin: usize, fft_size: usize) -> f32 {
        bin as f32 * SAMPLE_RATE / fft_size as f32
    }

    #[test]
    fn test_oscillator_output_range() {
        // All oscillators should output values in [-1, 1]
        let oscillators: Vec<(&str, fn(f32) -> f32)> = vec![
            ("sine", sine),
            ("square", square),
            ("saw", saw),
            ("saw_rev", saw_rev),
            ("triangle", triangle),
        ];

        for (name, osc_fn) in oscillators {
            for i in 0..1000 {
                let phase = i as f32 / 1000.0;
                let value = osc_fn(phase);
                assert!(
                    value >= -1.0 && value <= 1.0,
                    "{} out of range at phase {}: {}",
                    name,
                    phase,
                    value
                );
            }
        }
    }

    #[test]
    fn test_sine_frequency_accuracy() {
        let test_freq = 440.0;
        let signal = generate_signal(sine, test_freq, FFT_SIZE);
        let (peak_bin, _) = find_peak_bin(&signal);
        let detected_freq = bin_to_freq(peak_bin, FFT_SIZE);

        // Allow one bin tolerance
        let bin_width = SAMPLE_RATE / FFT_SIZE as f32;
        assert!(
            (detected_freq - test_freq).abs() < bin_width * 1.5,
            "Sine frequency mismatch: expected {}, got {}",
            test_freq,
            detected_freq
        );
    }

    #[test]
    fn test_square_fundamental_frequency() {
        let test_freq = 440.0;
        let signal = generate_signal(square, test_freq, FFT_SIZE);
        let (peak_bin, _) = find_peak_bin(&signal);
        let detected_freq = bin_to_freq(peak_bin, FFT_SIZE);

        let bin_width = SAMPLE_RATE / FFT_SIZE as f32;
        assert!(
            (detected_freq - test_freq).abs() < bin_width * 1.5,
            "Square fundamental mismatch: expected {}, got {}",
            test_freq,
            detected_freq
        );
    }

    #[test]
    fn test_saw_fundamental_frequency() {
        let test_freq = 440.0;
        let signal = generate_signal(saw, test_freq, FFT_SIZE);
        let (peak_bin, _) = find_peak_bin(&signal);
        let detected_freq = bin_to_freq(peak_bin, FFT_SIZE);

        let bin_width = SAMPLE_RATE / FFT_SIZE as f32;
        assert!(
            (detected_freq - test_freq).abs() < bin_width * 1.5,
            "Saw fundamental mismatch: expected {}, got {}",
            test_freq,
            detected_freq
        );
    }

    #[test]
    fn test_triangle_fundamental_frequency() {
        let test_freq = 440.0;
        let signal = generate_signal(triangle, test_freq, FFT_SIZE);
        let (peak_bin, _) = find_peak_bin(&signal);
        let detected_freq = bin_to_freq(peak_bin, FFT_SIZE);

        let bin_width = SAMPLE_RATE / FFT_SIZE as f32;
        assert!(
            (detected_freq - test_freq).abs() < bin_width * 1.5,
            "Triangle fundamental mismatch: expected {}, got {}",
            test_freq,
            detected_freq
        );
    }

    #[test]
    fn test_phase_continuity() {
        // Phase should wrap smoothly
        let mut prev_value = sine(0.0);
        for i in 1..1000 {
            let phase = i as f32 / 100.0; // Goes beyond 1.0
            let value = sine(phase);
            let delta = (value - prev_value).abs();
            // Maximum rate of change for sine at 1% phase step
            assert!(
                delta < 0.1,
                "Discontinuity at phase {}: prev={}, curr={}, delta={}",
                phase,
                prev_value,
                value,
                delta
            );
            prev_value = value;
        }
    }

    #[test]
    fn test_sine_is_zero_at_origin() {
        let value = sine(0.0);
        assert!(
            value.abs() < 1e-6,
            "Sine should be 0 at phase 0, got {}",
            value
        );
    }

    #[test]
    fn test_sine_symmetry() {
        // sin(x) = -sin(-x) = -sin(1-x) in our phase space
        for i in 1..100 {
            let phase = i as f32 / 100.0;
            let positive = sine(phase);
            let negative = sine(1.0 - phase);
            assert!(
                (positive + negative).abs() < 1e-5,
                "Sine asymmetry at phase {}: {} vs {}",
                phase,
                positive,
                negative
            );
        }
    }

    #[test]
    fn test_square_duty_cycle() {
        // Square wave should spend 50% of time at +1 and 50% at -1
        let mut positive_count = 0;
        let mut negative_count = 0;

        for i in 0..1000 {
            let phase = i as f32 / 1000.0;
            let value = square(phase);
            if value > 0.0 {
                positive_count += 1;
            } else {
                negative_count += 1;
            }
        }

        let ratio = positive_count as f32 / negative_count as f32;
        assert!(
            (ratio - 1.0).abs() < 0.01,
            "Square wave should have 50/50 duty cycle, got {}/{}",
            positive_count,
            negative_count
        );
    }

    #[test]
    fn test_pulse_duty_cycle() {
        // Pulse with 0.25 duty should be high 25% of the time
        let duty = 0.25;
        let mut positive_count = 0;

        for i in 0..1000 {
            let phase = i as f32 / 1000.0;
            if pulse(phase, duty) > 0.0 {
                positive_count += 1;
            }
        }

        let actual_duty = positive_count as f32 / 1000.0;
        assert!(
            (actual_duty - duty).abs() < 0.02,
            "Pulse duty cycle mismatch: expected {}, got {}",
            duty,
            actual_duty
        );
    }

    #[test]
    fn test_wavetable_interpolation_smoothness() {
        // Wavetable interpolation should be smooth
        let sine_table = Wavetable::sine(256);

        let mut prev = sine_table.sample(0.0);
        for i in 1..1000 {
            let phase = i as f32 / 1000.0;
            let curr = sine_table.sample(phase);
            let delta = (curr - prev).abs();
            // Should be smooth
            assert!(
                delta < 0.05,
                "Wavetable discontinuity at phase {}: delta={}",
                phase,
                delta
            );
            prev = curr;
        }
    }
}
