//! Physical modeling synthesis for percussion instruments.
//!
//! Implements modal synthesis for:
//! - Membranes (drums) - circular and rectangular
//! - Bars (xylophone, marimba, vibraphone)
//! - Plates (cymbals, bells)
//!
//! # Example
//!
//! ```
//! use unshape_audio::percussion::{Membrane, MembraneSynth};
//!
//! let config = MembraneSynth::snare();
//! let mut drum = Membrane::new(config, 44100.0);
//!
//! // Trigger a hit
//! drum.strike(0.8);
//!
//! // Generate samples
//! let samples: Vec<f32> = (0..44100).map(|_| drum.process()).collect();
//! ```

use std::f32::consts::PI;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Input for percussion synthesis operations.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct PercussionInput {
    /// Hit velocity (0.0 to 1.0).
    pub velocity: f32,
    /// Sample rate in Hz.
    pub sample_rate: f32,
    /// Number of samples to generate.
    pub num_samples: usize,
}

// ============================================================================
// Modal synthesis core
// ============================================================================

/// A single resonant mode.
#[derive(Debug, Clone)]
struct Mode {
    /// Amplitude (relative).
    amplitude: f32,
    /// Current phase.
    phase: f32,
    /// Current amplitude envelope.
    envelope: f32,
    /// Phase increment per sample.
    phase_inc: f32,
    /// Decay coefficient per sample.
    decay_coeff: f32,
}

impl Mode {
    fn new(freq: f32, amplitude: f32, decay: f32, sample_rate: f32) -> Self {
        let phase_inc = 2.0 * PI * freq / sample_rate;
        let decay_coeff = (-1.0 / (decay * sample_rate)).exp();

        Self {
            amplitude,
            phase: 0.0,
            envelope: 0.0,
            phase_inc,
            decay_coeff,
        }
    }

    fn trigger(&mut self, velocity: f32) {
        self.envelope = velocity * self.amplitude;
        // Randomize phase slightly for natural sound
        self.phase = 0.0;
    }

    fn process(&mut self) -> f32 {
        let output = self.envelope * self.phase.sin();
        self.phase += self.phase_inc;
        if self.phase > 2.0 * PI {
            self.phase -= 2.0 * PI;
        }
        self.envelope *= self.decay_coeff;
        output
    }
}

// ============================================================================
// Membrane (Drums)
// ============================================================================

/// Configuration for membrane synthesis.
///
/// Operations on audio buffers use the ops-as-values pattern.
/// See `docs/design/ops-as-values.md`.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dynop", derive(unshape_op::Op))]
#[cfg_attr(feature = "dynop", op(input = PercussionInput, output = Vec<f32>))]
pub struct MembraneSynth {
    /// Fundamental frequency in Hz.
    pub fundamental: f32,
    /// Number of modes to simulate.
    pub num_modes: usize,
    /// Tension parameter (affects frequency ratios).
    pub tension: f32,
    /// Damping factor (higher = faster decay).
    pub damping: f32,
    /// Decay time for fundamental in seconds.
    pub decay_time: f32,
}

impl Default for MembraneSynth {
    fn default() -> Self {
        Self {
            fundamental: 150.0,
            num_modes: 12,
            tension: 1.0,
            damping: 1.0,
            decay_time: 0.5,
        }
    }
}

impl MembraneSynth {
    /// Creates a snare drum configuration.
    pub fn snare() -> Self {
        Self {
            fundamental: 180.0,
            num_modes: 16,
            tension: 1.2,
            damping: 1.5,
            decay_time: 0.3,
        }
    }

    /// Creates a kick drum configuration.
    pub fn kick() -> Self {
        Self {
            fundamental: 60.0,
            num_modes: 8,
            tension: 0.8,
            damping: 2.0,
            decay_time: 0.4,
        }
    }

    /// Creates a tom drum configuration.
    pub fn tom(pitch: f32) -> Self {
        Self {
            fundamental: pitch,
            num_modes: 12,
            tension: 1.0,
            damping: 1.2,
            decay_time: 0.5,
        }
    }

    /// Creates a timpani configuration.
    pub fn timpani(pitch: f32) -> Self {
        Self {
            fundamental: pitch,
            num_modes: 16,
            tension: 1.1,
            damping: 0.5,
            decay_time: 2.0,
        }
    }

    /// Applies this membrane configuration to generate samples.
    ///
    /// Takes percussion input and returns audio samples.
    pub fn apply(&self, input: &PercussionInput) -> Vec<f32> {
        let mut membrane = Membrane::new(self.clone(), input.sample_rate);
        membrane.generate(input.velocity, input.num_samples)
    }
}

/// Backwards-compatible type alias.
pub type MembraneConfig = MembraneSynth;

/// Circular membrane physical model (drums).
///
/// Uses Bessel function zeros for mode frequencies.
pub struct Membrane {
    modes: Vec<Mode>,
}

impl Membrane {
    /// Creates a new membrane with the given configuration.
    pub fn new(config: MembraneSynth, sample_rate: f32) -> Self {
        // Bessel function zeros for circular membrane (J_mn)
        // These are the frequency ratios relative to fundamental
        let bessel_ratios = [
            1.000, // (0,1)
            1.594, // (1,1)
            2.136, // (2,1)
            2.296, // (0,2)
            2.653, // (3,1)
            2.918, // (1,2)
            3.156, // (4,1)
            3.501, // (2,2)
            3.600, // (0,3)
            3.652, // (5,1)
            4.059, // (3,2)
            4.154, // (6,1)
            4.241, // (1,3)
            4.601, // (4,2)
            4.610, // (2,3)
            4.654, // (7,1)
        ];

        let num_modes = config.num_modes.min(bessel_ratios.len());
        let mut modes = Vec::with_capacity(num_modes);

        for i in 0..num_modes {
            let ratio = bessel_ratios[i] * config.tension;
            let freq = config.fundamental * ratio;

            // Higher modes have lower amplitude and faster decay
            let amplitude = 1.0 / (1.0 + i as f32 * 0.3);
            let decay = config.decay_time / (1.0 + i as f32 * 0.2 * config.damping);

            modes.push(Mode::new(freq, amplitude, decay, sample_rate));
        }

        Self { modes }
    }

    /// Triggers the membrane with the given velocity (0-1).
    pub fn strike(&mut self, velocity: f32) {
        let v = velocity.clamp(0.0, 1.0);
        for mode in &mut self.modes {
            mode.trigger(v);
        }
    }

    /// Processes one sample.
    pub fn process(&mut self) -> f32 {
        let mut output = 0.0;
        for mode in &mut self.modes {
            output += mode.process();
        }
        output * 0.3 // Normalize
    }

    /// Generates a buffer of samples after a strike.
    pub fn generate(&mut self, velocity: f32, num_samples: usize) -> Vec<f32> {
        self.strike(velocity);
        (0..num_samples).map(|_| self.process()).collect()
    }

    /// Resets all modes.
    pub fn reset(&mut self) {
        for mode in &mut self.modes {
            mode.envelope = 0.0;
            mode.phase = 0.0;
        }
    }
}

// ============================================================================
// Bar (Xylophone, Marimba, Vibraphone)
// ============================================================================

/// Configuration for bar synthesis.
///
/// Operations on audio buffers use the ops-as-values pattern.
/// See `docs/design/ops-as-values.md`.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dynop", derive(unshape_op::Op))]
#[cfg_attr(feature = "dynop", op(input = PercussionInput, output = Vec<f32>))]
pub struct BarSynth {
    /// Fundamental frequency in Hz.
    pub fundamental: f32,
    /// Number of modes to simulate.
    pub num_modes: usize,
    /// Material stiffness (affects inharmonicity).
    pub stiffness: f32,
    /// Decay time for fundamental in seconds.
    pub decay_time: f32,
    /// Brightness (high frequency emphasis).
    pub brightness: f32,
}

impl Default for BarSynth {
    fn default() -> Self {
        Self {
            fundamental: 440.0,
            num_modes: 8,
            stiffness: 1.0,
            decay_time: 2.0,
            brightness: 1.0,
        }
    }
}

impl BarSynth {
    /// Creates a xylophone bar configuration.
    pub fn xylophone(note_freq: f32) -> Self {
        Self {
            fundamental: note_freq,
            num_modes: 6,
            stiffness: 1.5,
            decay_time: 1.0,
            brightness: 1.5,
        }
    }

    /// Creates a marimba bar configuration.
    pub fn marimba(note_freq: f32) -> Self {
        Self {
            fundamental: note_freq,
            num_modes: 8,
            stiffness: 0.8,
            decay_time: 1.5,
            brightness: 0.8,
        }
    }

    /// Creates a vibraphone bar configuration.
    pub fn vibraphone(note_freq: f32) -> Self {
        Self {
            fundamental: note_freq,
            num_modes: 10,
            stiffness: 0.6,
            decay_time: 4.0,
            brightness: 0.6,
        }
    }

    /// Creates a glockenspiel configuration.
    pub fn glockenspiel(note_freq: f32) -> Self {
        Self {
            fundamental: note_freq,
            num_modes: 6,
            stiffness: 2.0,
            decay_time: 3.0,
            brightness: 2.0,
        }
    }

    /// Applies this bar configuration to generate samples.
    ///
    /// Takes percussion input and returns audio samples.
    pub fn apply(&self, input: &PercussionInput) -> Vec<f32> {
        let mut bar = Bar::new(self.clone(), input.sample_rate);
        bar.generate(input.velocity, input.num_samples)
    }
}

/// Backwards-compatible type alias.
pub type BarConfig = BarSynth;

/// Bar physical model (xylophone, marimba, vibraphone).
///
/// Uses the bar equation frequency ratios (proportional to n^2).
pub struct Bar {
    modes: Vec<Mode>,
    sample_rate: f32,
    /// Optional vibrato for vibraphone effect.
    vibrato_phase: f32,
    vibrato_rate: f32,
    vibrato_depth: f32,
}

impl Bar {
    /// Creates a new bar with the given configuration.
    pub fn new(config: BarSynth, sample_rate: f32) -> Self {
        // For a free-free bar, frequency ratios are proportional to (n + 0.5)^2
        // First few ratios: 1.0, 2.76, 5.40, 8.93, 13.34...
        let bar_ratios: [f32; 8] = [
            1.000,  // n=1
            2.756,  // n=2
            5.404,  // n=3
            8.933,  // n=4
            13.344, // n=5
            18.636, // n=6
            24.810, // n=7
            31.865, // n=8
        ];

        let num_modes = config.num_modes.min(bar_ratios.len());
        let mut modes = Vec::with_capacity(num_modes);

        for i in 0..num_modes {
            // Apply stiffness to affect inharmonicity
            let ratio = bar_ratios[i].powf(config.stiffness.sqrt());
            let freq = config.fundamental * ratio;

            // Amplitude falls off with mode number, affected by brightness
            let amplitude = config.brightness / (1.0 + i as f32 * 0.5);
            let decay = config.decay_time / (1.0 + i as f32 * 0.3);

            modes.push(Mode::new(freq, amplitude, decay, sample_rate));
        }

        Self {
            modes,
            sample_rate,
            vibrato_phase: 0.0,
            vibrato_rate: 0.0,
            vibrato_depth: 0.0,
        }
    }

    /// Enables vibrato (for vibraphone effect).
    pub fn set_vibrato(&mut self, rate: f32, depth: f32) {
        self.vibrato_rate = rate;
        self.vibrato_depth = depth;
    }

    /// Triggers the bar with the given velocity (0-1).
    pub fn strike(&mut self, velocity: f32) {
        let v = velocity.clamp(0.0, 1.0);
        for mode in &mut self.modes {
            mode.trigger(v);
        }
    }

    /// Processes one sample.
    pub fn process(&mut self) -> f32 {
        // Apply vibrato modulation to amplitude
        let vibrato_mod = if self.vibrato_rate > 0.0 {
            self.vibrato_phase += 2.0 * PI * self.vibrato_rate / self.sample_rate;
            if self.vibrato_phase > 2.0 * PI {
                self.vibrato_phase -= 2.0 * PI;
            }
            1.0 + self.vibrato_depth * self.vibrato_phase.sin()
        } else {
            1.0
        };

        let mut output = 0.0;
        for mode in &mut self.modes {
            output += mode.process();
        }
        output * 0.3 * vibrato_mod
    }

    /// Generates a buffer of samples after a strike.
    pub fn generate(&mut self, velocity: f32, num_samples: usize) -> Vec<f32> {
        self.strike(velocity);
        (0..num_samples).map(|_| self.process()).collect()
    }

    /// Resets all modes.
    pub fn reset(&mut self) {
        for mode in &mut self.modes {
            mode.envelope = 0.0;
            mode.phase = 0.0;
        }
        self.vibrato_phase = 0.0;
    }
}

// ============================================================================
// Plate (Cymbals, Bells)
// ============================================================================

/// Configuration for plate synthesis.
///
/// Operations on audio buffers use the ops-as-values pattern.
/// See `docs/design/ops-as-values.md`.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dynop", derive(unshape_op::Op))]
#[cfg_attr(feature = "dynop", op(input = PercussionInput, output = Vec<f32>))]
pub struct PlateSynth {
    /// Fundamental frequency in Hz.
    pub fundamental: f32,
    /// Number of modes to simulate.
    pub num_modes: usize,
    /// Thickness parameter (affects frequency spread).
    pub thickness: f32,
    /// Decay time for fundamental in seconds.
    pub decay_time: f32,
    /// High frequency density.
    pub density: f32,
}

impl Default for PlateSynth {
    fn default() -> Self {
        Self {
            fundamental: 300.0,
            num_modes: 24,
            thickness: 1.0,
            decay_time: 3.0,
            density: 1.0,
        }
    }
}

impl PlateSynth {
    /// Creates a hi-hat configuration.
    pub fn hihat() -> Self {
        Self {
            fundamental: 400.0,
            num_modes: 32,
            thickness: 0.5,
            decay_time: 0.3,
            density: 2.0,
        }
    }

    /// Creates a crash cymbal configuration.
    pub fn crash() -> Self {
        Self {
            fundamental: 300.0,
            num_modes: 48,
            thickness: 0.8,
            decay_time: 2.0,
            density: 1.5,
        }
    }

    /// Creates a ride cymbal configuration.
    pub fn ride() -> Self {
        Self {
            fundamental: 350.0,
            num_modes: 40,
            thickness: 1.0,
            decay_time: 4.0,
            density: 1.2,
        }
    }

    /// Creates a bell configuration.
    pub fn bell(pitch: f32) -> Self {
        Self {
            fundamental: pitch,
            num_modes: 16,
            thickness: 1.5,
            decay_time: 5.0,
            density: 0.8,
        }
    }

    /// Creates a gong configuration.
    pub fn gong(pitch: f32) -> Self {
        Self {
            fundamental: pitch,
            num_modes: 32,
            thickness: 0.6,
            decay_time: 8.0,
            density: 1.0,
        }
    }

    /// Applies this plate configuration to generate samples.
    ///
    /// Takes percussion input and returns audio samples.
    pub fn apply(&self, input: &PercussionInput) -> Vec<f32> {
        let mut plate = Plate::new(self.clone(), input.sample_rate);
        plate.generate(input.velocity, input.num_samples)
    }
}

/// Backwards-compatible type alias.
pub type PlateConfig = PlateSynth;

/// Plate physical model (cymbals, bells, gongs).
///
/// Uses a dense, inharmonic spectrum characteristic of plates.
pub struct Plate {
    modes: Vec<Mode>,
    noise_state: u32,
}

impl Plate {
    /// Creates a new plate with the given configuration.
    pub fn new(config: PlateSynth, sample_rate: f32) -> Self {
        let mut modes = Vec::with_capacity(config.num_modes);

        // Plates have a complex, inharmonic spectrum
        // Frequency ratios roughly follow: f_mn ~ (m^2 + n^2) but with irregularities
        let mut rng_state = 12345u32;

        for i in 0..config.num_modes {
            // Base frequency ratio with some randomization for complexity
            let base_ratio =
                1.0 + (i as f32 * 0.7 * config.density).powf(1.0 + config.thickness * 0.2);

            // Add some randomness for natural sound
            rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
            let random = (rng_state as f32 / u32::MAX as f32 - 0.5) * 0.1;
            let ratio = base_ratio * (1.0 + random);

            let freq = config.fundamental * ratio;

            // Amplitude with random variation
            rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
            let amp_random = 0.5 + (rng_state as f32 / u32::MAX as f32) * 0.5;
            let amplitude = amp_random / (1.0 + i as f32 * 0.2);

            // Higher modes decay faster
            let decay = config.decay_time / (1.0 + i as f32 * 0.15);

            modes.push(Mode::new(freq, amplitude, decay, sample_rate));
        }

        Self {
            modes,
            noise_state: 54321,
        }
    }

    /// Triggers the plate with the given velocity (0-1).
    pub fn strike(&mut self, velocity: f32) {
        let v = velocity.clamp(0.0, 1.0);

        // Add some randomization to each mode's phase for natural sound
        for mode in &mut self.modes {
            self.noise_state = self
                .noise_state
                .wrapping_mul(1103515245)
                .wrapping_add(12345);
            let phase_offset = (self.noise_state as f32 / u32::MAX as f32) * 2.0 * PI;
            mode.phase = phase_offset;
            mode.envelope = v * mode.amplitude;
        }
    }

    /// Processes one sample.
    pub fn process(&mut self) -> f32 {
        let mut output = 0.0;
        for mode in &mut self.modes {
            output += mode.process();
        }

        // Add a tiny bit of noise for attack transient
        self.noise_state = self
            .noise_state
            .wrapping_mul(1103515245)
            .wrapping_add(12345);

        output * 0.15 // Normalize (plates have many modes)
    }

    /// Generates a buffer of samples after a strike.
    pub fn generate(&mut self, velocity: f32, num_samples: usize) -> Vec<f32> {
        self.strike(velocity);
        (0..num_samples).map(|_| self.process()).collect()
    }

    /// Resets all modes.
    pub fn reset(&mut self) {
        for mode in &mut self.modes {
            mode.envelope = 0.0;
            mode.phase = 0.0;
        }
    }
}

// ============================================================================
// Utility: Simple noise burst for attack transients
// ============================================================================

/// Generates a noise burst for percussive attacks.
pub fn noise_burst(length: usize, decay: f32, sample_rate: f32) -> Vec<f32> {
    let decay_coeff = (-1.0 / (decay * sample_rate)).exp();
    let mut envelope = 1.0f32;
    let mut state = 12345u32;

    (0..length)
        .map(|_| {
            state = state.wrapping_mul(1103515245).wrapping_add(12345);
            let noise = (state as f32 / u32::MAX as f32) * 2.0 - 1.0;
            let sample = noise * envelope;
            envelope *= decay_coeff;
            sample
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_membrane_snare() {
        let config = MembraneSynth::snare();
        let mut drum = Membrane::new(config, 44100.0);

        drum.strike(0.8);
        let samples: Vec<f32> = (0..4410).map(|_| drum.process()).collect();

        // Should have some output
        let energy: f32 = samples.iter().map(|x| x * x).sum();
        assert!(energy > 0.001);

        // Should decay
        let first_energy: f32 = samples[..441].iter().map(|x| x * x).sum();
        let last_energy: f32 = samples[3969..].iter().map(|x| x * x).sum();
        assert!(first_energy > last_energy);
    }

    #[test]
    fn test_membrane_kick() {
        let config = MembraneSynth::kick();
        let mut drum = Membrane::new(config, 44100.0);

        let samples = drum.generate(1.0, 22050);
        assert_eq!(samples.len(), 22050);

        // Kick should have energy
        let energy: f32 = samples.iter().map(|x| x * x).sum();
        assert!(energy > 0.001);
    }

    #[test]
    fn test_bar_xylophone() {
        let config = BarSynth::xylophone(440.0);
        let mut bar = Bar::new(config, 44100.0);

        bar.strike(0.7);
        let samples: Vec<f32> = (0..44100).map(|_| bar.process()).collect();

        let energy: f32 = samples.iter().map(|x| x * x).sum();
        assert!(energy > 0.001);
    }

    #[test]
    fn test_bar_vibrato() {
        let config = BarSynth::vibraphone(440.0);
        let mut bar = Bar::new(config, 44100.0);
        bar.set_vibrato(5.0, 0.3); // 5Hz vibrato

        bar.strike(0.7);
        let samples: Vec<f32> = (0..44100).map(|_| bar.process()).collect();

        let energy: f32 = samples.iter().map(|x| x * x).sum();
        assert!(energy > 0.001);
    }

    #[test]
    fn test_plate_hihat() {
        let config = PlateSynth::hihat();
        let mut hihat = Plate::new(config, 44100.0);

        hihat.strike(0.6);
        let samples: Vec<f32> = (0..22050).map(|_| hihat.process()).collect();

        let energy: f32 = samples.iter().map(|x| x * x).sum();
        assert!(energy > 0.001);
    }

    #[test]
    fn test_plate_crash() {
        let config = PlateSynth::crash();
        let mut cymbal = Plate::new(config, 44100.0);

        let samples = cymbal.generate(0.9, 44100);
        assert_eq!(samples.len(), 44100);

        // Crash should have sustained energy
        let mid_energy: f32 = samples[22050..33075].iter().map(|x| x * x).sum();
        assert!(mid_energy > 0.0001);
    }

    #[test]
    fn test_plate_bell() {
        let config = PlateSynth::bell(523.25); // C5
        let mut bell = Plate::new(config, 44100.0);

        bell.strike(0.8);
        let samples: Vec<f32> = (0..88200).map(|_| bell.process()).collect();

        // Bell should ring for a while
        let late_energy: f32 = samples[44100..66150].iter().map(|x| x * x).sum();
        assert!(late_energy > 0.00001);
    }

    #[test]
    fn test_noise_burst() {
        let burst = noise_burst(4410, 0.05, 44100.0);
        assert_eq!(burst.len(), 4410);

        // Should start loud and decay
        let first_energy: f32 = burst[..441].iter().map(|x| x * x).sum();
        let last_energy: f32 = burst[3969..].iter().map(|x| x * x).sum();
        assert!(first_energy > last_energy * 10.0);
    }

    #[test]
    fn test_membrane_reset() {
        let config = MembraneSynth::snare();
        let mut drum = Membrane::new(config, 44100.0);

        drum.strike(1.0);
        let _ = drum.process();

        drum.reset();
        let sample = drum.process();
        assert!(sample.abs() < 0.0001);
    }

    #[test]
    fn test_configs() {
        // Test that all preset configs can be created
        let _ = MembraneSynth::snare();
        let _ = MembraneSynth::kick();
        let _ = MembraneSynth::tom(200.0);
        let _ = MembraneSynth::timpani(100.0);

        let _ = BarSynth::xylophone(440.0);
        let _ = BarSynth::marimba(220.0);
        let _ = BarSynth::vibraphone(330.0);
        let _ = BarSynth::glockenspiel(880.0);

        let _ = PlateSynth::hihat();
        let _ = PlateSynth::crash();
        let _ = PlateSynth::ride();
        let _ = PlateSynth::bell(440.0);
        let _ = PlateSynth::gong(100.0);
    }
}
