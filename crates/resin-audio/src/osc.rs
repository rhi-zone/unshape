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
/// use resin_audio::osc::{freq_to_phase, sine};
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
}
