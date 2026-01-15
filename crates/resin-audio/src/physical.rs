//! Physical modeling synthesis.
//!
//! Implements physically-inspired sound synthesis algorithms.
//!
//! # Example
//!
//! ```
//! use rhizome_resin_audio::physical::{KarplusStrong, PluckSynth};
//!
//! let mut ks = KarplusStrong::new(44100.0);
//! ks.pluck(440.0, PluckSynth::default());
//!
//! let mut buffer = vec![0.0; 1000];
//! ks.generate(&mut buffer);
//! ```

use std::collections::VecDeque;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Input for pluck synthesis operation.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct PluckInput {
    /// Frequency in Hz.
    pub frequency: f32,
    /// Sample rate in Hz.
    pub sample_rate: f32,
    /// Number of samples to generate.
    pub num_samples: usize,
}

/// Configuration for plucking a string.
///
/// Operations on audio buffers use the ops-as-values pattern.
/// See `docs/design/ops-as-values.md`.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dynop", derive(rhizome_resin_op::Op))]
#[cfg_attr(feature = "dynop", op(input = PluckInput, output = Vec<f32>))]
pub struct PluckSynth {
    /// Initial amplitude (0.0 to 1.0).
    pub amplitude: f32,
    /// Damping factor (0.0 to 1.0). Higher = more damping = faster decay.
    pub damping: f32,
    /// Brightness (0.0 to 1.0). Lower = duller sound.
    pub brightness: f32,
    /// Blend between noise (0.0) and sawtooth (1.0) for initial excitation.
    pub noise_blend: f32,
}

impl Default for PluckSynth {
    fn default() -> Self {
        Self {
            amplitude: 0.8,
            damping: 0.5,
            brightness: 0.5,
            noise_blend: 0.0, // Pure noise for classic pluck
        }
    }
}

impl PluckSynth {
    /// Applies this pluck configuration to generate samples.
    ///
    /// Takes pluck input and returns audio samples.
    pub fn apply(&self, input: &PluckInput) -> Vec<f32> {
        let mut ks = KarplusStrong::new(input.sample_rate);
        ks.pluck(input.frequency, self.clone());
        let mut output = vec![0.0; input.num_samples];
        ks.generate(&mut output);
        output
    }
}

/// Backwards-compatible type alias.
pub type PluckConfig = PluckSynth;

/// Karplus-Strong string synthesis.
///
/// A simple but effective physical modeling algorithm for plucked strings.
/// Uses a delay line with low-pass filtering to simulate string vibration.
#[derive(Debug, Clone)]
pub struct KarplusStrong {
    /// Delay line buffer.
    buffer: VecDeque<f32>,
    /// Sample rate.
    sample_rate: f32,
    /// Current position in buffer.
    position: usize,
    /// Low-pass filter state.
    prev_sample: f32,
    /// Damping factor.
    damping: f32,
    /// Brightness (filter coefficient).
    brightness: f32,
    /// Simple RNG state.
    rng_state: u32,
}

impl KarplusStrong {
    /// Creates a new Karplus-Strong synthesizer.
    pub fn new(sample_rate: f32) -> Self {
        Self {
            buffer: VecDeque::new(),
            sample_rate,
            position: 0,
            prev_sample: 0.0,
            damping: 0.5,
            brightness: 0.5,
            rng_state: 12345,
        }
    }

    /// Plucks the string at the given frequency.
    pub fn pluck(&mut self, frequency: f32, config: PluckSynth) {
        // Calculate delay line length from frequency
        let period = (self.sample_rate / frequency).round() as usize;
        let period = period.max(2); // Minimum 2 samples

        // Initialize buffer with excitation signal
        self.buffer.clear();
        self.buffer.reserve(period);

        for i in 0..period {
            // Generate noise
            let noise = self.next_random() * 2.0 - 1.0;

            // Generate sawtooth
            let saw = (i as f32 / period as f32) * 2.0 - 1.0;

            // Blend between noise and sawtooth
            let sample = noise * (1.0 - config.noise_blend) + saw * config.noise_blend;

            self.buffer.push_back(sample * config.amplitude);
        }

        self.position = 0;
        self.prev_sample = 0.0;
        self.damping = config.damping;
        self.brightness = config.brightness;
    }

    /// Generates the next sample.
    #[inline]
    pub fn tick(&mut self) -> f32 {
        if self.buffer.is_empty() {
            return 0.0;
        }

        // Get the current sample
        let current = self.buffer.pop_front().unwrap_or(0.0);

        // Low-pass filter: weighted average of current and previous
        // brightness controls the blend (1.0 = no filtering, 0.0 = heavy filtering)
        let filtered = current * self.brightness + self.prev_sample * (1.0 - self.brightness);

        // Apply damping (controls decay rate)
        let damped = filtered * (1.0 - self.damping * 0.01);

        // Store for next iteration
        self.prev_sample = current;

        // Put filtered sample back into buffer
        self.buffer.push_back(damped);

        current
    }

    /// Generates a buffer of samples.
    pub fn generate(&mut self, buffer: &mut [f32]) {
        for sample in buffer {
            *sample = self.tick();
        }
    }

    /// Returns true if the string is still vibrating.
    pub fn is_active(&self) -> bool {
        if self.buffer.is_empty() {
            return false;
        }

        // Check if there's still significant energy
        let energy: f32 = self.buffer.iter().map(|s| s.abs()).sum();
        energy > 0.001
    }

    /// Mutes the string.
    pub fn mute(&mut self) {
        self.buffer.clear();
    }

    /// Simple random number generator (returns 0.0 to 1.0).
    fn next_random(&mut self) -> f32 {
        self.rng_state = self.rng_state.wrapping_mul(1103515245).wrapping_add(12345);
        ((self.rng_state >> 16) & 0x7FFF) as f32 / 32767.0
    }
}

/// Extended Karplus-Strong with additional features.
#[derive(Debug, Clone)]
pub struct ExtendedKarplusStrong {
    /// Base synthesizer.
    ks: KarplusStrong,
    /// Stretch factor for inharmonicity.
    stretch: f32,
    /// Pick position (0.0 to 1.0).
    pick_position: f32,
    /// Dynamics filter state.
    dynamics_filter: f32,
}

impl ExtendedKarplusStrong {
    /// Creates a new extended Karplus-Strong synthesizer.
    pub fn new(sample_rate: f32) -> Self {
        Self {
            ks: KarplusStrong::new(sample_rate),
            stretch: 1.0,
            pick_position: 0.5,
            dynamics_filter: 0.0,
        }
    }

    /// Sets the stretch factor for inharmonicity (piano-like).
    pub fn set_stretch(&mut self, stretch: f32) {
        self.stretch = stretch;
    }

    /// Sets the pick position (affects timbre).
    pub fn set_pick_position(&mut self, position: f32) {
        self.pick_position = position.clamp(0.01, 0.99);
    }

    /// Plucks the string.
    pub fn pluck(&mut self, frequency: f32, config: PluckSynth) {
        self.ks.pluck(frequency, config);

        // Apply pick position filtering to initial buffer
        if !self.ks.buffer.is_empty() {
            let len = self.ks.buffer.len();
            let pick_sample = (self.pick_position * len as f32) as usize;

            // Create a comb filter effect based on pick position
            for i in 0..len {
                if i >= pick_sample {
                    let delay_idx = i - pick_sample;
                    if delay_idx < len {
                        let delayed = self.ks.buffer[delay_idx];
                        self.ks.buffer[i] -= delayed * 0.5;
                    }
                }
            }
        }
    }

    /// Generates the next sample.
    #[inline]
    pub fn tick(&mut self) -> f32 {
        let sample = self.ks.tick();

        // Apply dynamics filter for body resonance
        self.dynamics_filter = self.dynamics_filter * 0.99 + sample * 0.01;

        sample + self.dynamics_filter * 0.1
    }

    /// Generates a buffer of samples.
    pub fn generate(&mut self, buffer: &mut [f32]) {
        for sample in buffer {
            *sample = self.tick();
        }
    }

    /// Returns true if the string is still vibrating.
    pub fn is_active(&self) -> bool {
        self.ks.is_active()
    }

    /// Mutes the string.
    pub fn mute(&mut self) {
        self.ks.mute();
    }
}

/// A polyphonic string synthesizer.
#[derive(Debug)]
pub struct PolyStrings {
    /// Individual string voices.
    voices: Vec<KarplusStrong>,
    /// Sample rate.
    sample_rate: f32,
    /// Maximum number of voices.
    max_voices: usize,
}

impl PolyStrings {
    /// Creates a new polyphonic string synthesizer.
    pub fn new(sample_rate: f32, max_voices: usize) -> Self {
        Self {
            voices: Vec::with_capacity(max_voices),
            sample_rate,
            max_voices,
        }
    }

    /// Plucks a new string.
    pub fn pluck(&mut self, frequency: f32, config: PluckSynth) {
        // Find an inactive voice or steal the oldest
        let voice_idx = self
            .voices
            .iter()
            .position(|v| !v.is_active())
            .unwrap_or_else(|| {
                if self.voices.len() < self.max_voices {
                    self.voices.push(KarplusStrong::new(self.sample_rate));
                    self.voices.len() - 1
                } else {
                    0 // Steal oldest voice
                }
            });

        if voice_idx < self.voices.len() {
            self.voices[voice_idx].pluck(frequency, config);
        }
    }

    /// Generates the next sample (sum of all voices).
    #[inline]
    pub fn tick(&mut self) -> f32 {
        let mut sum = 0.0;
        for voice in &mut self.voices {
            sum += voice.tick();
        }
        // Normalize by voice count to prevent clipping
        if !self.voices.is_empty() {
            sum / self.voices.len() as f32
        } else {
            0.0
        }
    }

    /// Generates a buffer of samples.
    pub fn generate(&mut self, buffer: &mut [f32]) {
        for sample in buffer {
            *sample = self.tick();
        }
    }

    /// Mutes all strings.
    pub fn mute_all(&mut self) {
        for voice in &mut self.voices {
            voice.mute();
        }
    }

    /// Returns the number of active voices.
    pub fn active_voices(&self) -> usize {
        self.voices.iter().filter(|v| v.is_active()).count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_karplus_strong_basic() {
        let mut ks = KarplusStrong::new(44100.0);
        ks.pluck(440.0, PluckSynth::default());

        // Generate some samples
        let mut buffer = vec![0.0; 1000];
        ks.generate(&mut buffer);

        // Should produce non-zero output
        let energy: f32 = buffer.iter().map(|s| s.abs()).sum();
        assert!(energy > 0.0);

        // All samples should be in valid range
        for s in &buffer {
            assert!(s.abs() <= 1.0, "sample {} out of range", s);
        }
    }

    #[test]
    fn test_karplus_strong_decay() {
        let mut ks = KarplusStrong::new(44100.0);
        ks.pluck(440.0, PluckSynth::default());

        // Generate two chunks
        let mut buffer1 = vec![0.0; 1000];
        let mut buffer2 = vec![0.0; 1000];

        ks.generate(&mut buffer1);
        ks.generate(&mut buffer2);

        // Second buffer should have less energy (decay)
        let energy1: f32 = buffer1.iter().map(|s| s.abs()).sum();
        let energy2: f32 = buffer2.iter().map(|s| s.abs()).sum();

        assert!(energy2 < energy1, "sound should decay over time");
    }

    #[test]
    fn test_karplus_strong_frequency() {
        let mut ks = KarplusStrong::new(44100.0);

        // Higher frequency should have shorter period
        ks.pluck(880.0, PluckSynth::default());
        let len_high = ks.buffer.len();

        ks.pluck(440.0, PluckSynth::default());
        let len_low = ks.buffer.len();

        assert!(len_high < len_low, "higher freq should have shorter buffer");
        assert_eq!(len_low, len_high * 2, "octave should double buffer length");
    }

    #[test]
    fn test_karplus_strong_damping() {
        let mut ks1 = KarplusStrong::new(44100.0);
        let mut ks2 = KarplusStrong::new(44100.0);

        ks1.pluck(
            440.0,
            PluckSynth {
                damping: 0.1,
                ..Default::default()
            },
        );
        ks2.pluck(
            440.0,
            PluckSynth {
                damping: 0.9,
                ..Default::default()
            },
        );

        // Generate samples
        let mut buf1 = vec![0.0; 5000];
        let mut buf2 = vec![0.0; 5000];

        ks1.generate(&mut buf1);
        ks2.generate(&mut buf2);

        // Higher damping should decay faster
        let energy1: f32 = buf1[4000..].iter().map(|s| s.abs()).sum();
        let energy2: f32 = buf2[4000..].iter().map(|s| s.abs()).sum();

        assert!(energy2 < energy1, "higher damping should decay faster");
    }

    #[test]
    fn test_karplus_strong_is_active() {
        let mut ks = KarplusStrong::new(44100.0);

        assert!(!ks.is_active(), "should be inactive before pluck");

        ks.pluck(440.0, PluckSynth::default());
        assert!(ks.is_active(), "should be active after pluck");

        ks.mute();
        assert!(!ks.is_active(), "should be inactive after mute");
    }

    #[test]
    fn test_extended_karplus_strong() {
        let mut eks = ExtendedKarplusStrong::new(44100.0);
        eks.set_pick_position(0.25);
        eks.pluck(440.0, PluckSynth::default());

        let mut buffer = vec![0.0; 1000];
        eks.generate(&mut buffer);

        let energy: f32 = buffer.iter().map(|s| s.abs()).sum();
        assert!(energy > 0.0);
    }

    #[test]
    fn test_poly_strings() {
        let mut poly = PolyStrings::new(44100.0, 4);

        // Pluck multiple strings
        poly.pluck(440.0, PluckSynth::default());
        poly.pluck(554.37, PluckSynth::default()); // C#
        poly.pluck(659.25, PluckSynth::default()); // E

        assert_eq!(poly.active_voices(), 3);

        let mut buffer = vec![0.0; 1000];
        poly.generate(&mut buffer);

        let energy: f32 = buffer.iter().map(|s| s.abs()).sum();
        assert!(energy > 0.0);

        poly.mute_all();
        assert_eq!(poly.active_voices(), 0);
    }

    #[test]
    fn test_pluck_config_struct() {
        let config = PluckSynth {
            amplitude: 0.5,
            damping: 0.3,
            brightness: 0.8,
            noise_blend: 0.2,
        };

        assert_eq!(config.amplitude, 0.5);
        assert_eq!(config.damping, 0.3);
        assert_eq!(config.brightness, 0.8);
        assert_eq!(config.noise_blend, 0.2);
    }
}
