//! Granular synthesis for audio generation.
//!
//! Granular synthesis creates sound by combining many small audio "grains",
//! typically 10-100ms in duration. Each grain has its own parameters like
//! position, pitch, and amplitude.
//!
//! # Example
//!
//! ```
//! use rhizome_resin_audio::{GrainCloud, GrainConfig};
//!
//! // Create a grain cloud with a sample buffer
//! let buffer = vec![0.0f32; 44100]; // 1 second of silence
//! let mut cloud = GrainCloud::new(buffer, 44100.0);
//!
//! // Configure grain parameters
//! cloud.set_grain_size(50.0); // 50ms grains
//! cloud.set_density(10.0); // 10 grains per second
//!
//! // Generate samples
//! let output = cloud.tick();
//! ```

use std::collections::VecDeque;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Input for granular synthesis operation.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct GranularInput {
    /// Source audio buffer.
    pub buffer: Vec<f32>,
    /// Sample rate in Hz.
    pub sample_rate: f32,
    /// Number of samples to generate.
    pub num_samples: usize,
}

/// Configuration for grain generation.
///
/// Operations on audio buffers use the ops-as-values pattern.
/// See `docs/design/ops-as-values.md`.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dynop", derive(rhizome_resin_op::Op))]
#[cfg_attr(feature = "dynop", op(input = GranularInput, output = Vec<f32>))]
pub struct GranularSynth {
    /// Grain size in milliseconds.
    pub size_ms: f32,
    /// Randomization of grain size (0.0 to 1.0).
    pub size_jitter: f32,
    /// Position in the source buffer (0.0 to 1.0).
    pub position: f32,
    /// Randomization of position.
    pub position_jitter: f32,
    /// Pitch multiplier (1.0 = original pitch).
    pub pitch: f32,
    /// Randomization of pitch.
    pub pitch_jitter: f32,
    /// Grain density (grains per second).
    pub density: f32,
    /// Pan position (-1.0 to 1.0).
    pub pan: f32,
    /// Randomization of pan.
    pub pan_jitter: f32,
}

impl Default for GranularSynth {
    fn default() -> Self {
        Self {
            size_ms: 50.0,
            size_jitter: 0.0,
            position: 0.0,
            position_jitter: 0.0,
            pitch: 1.0,
            pitch_jitter: 0.0,
            density: 10.0,
            pan: 0.0,
            pan_jitter: 0.0,
        }
    }
}

impl GranularSynth {
    /// Creates a new grain configuration with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Applies this granular synthesis configuration to generate samples.
    ///
    /// Takes a source buffer and sample rate, returns synthesized audio.
    pub fn apply(&self, input: &GranularInput) -> Vec<f32> {
        let mut cloud = GrainCloud::new(input.buffer.clone(), input.sample_rate);
        cloud.set_config(self.clone());
        cloud.generate(input.num_samples)
    }
}

/// Backwards-compatible type alias.
pub type GrainConfig = GranularSynth;

/// A single grain of audio.
#[derive(Debug, Clone)]
struct Grain {
    /// Position in the source buffer (in samples).
    position: f32,
    /// Playback rate (pitch multiplier).
    rate: f32,
    /// Current playhead position within the grain.
    phase: f32,
    /// Total duration of the grain in samples.
    duration: f32,
    /// Pan position (-1.0 to 1.0).
    pan: f32,
    /// Amplitude.
    amplitude: f32,
}

impl Grain {
    /// Creates a new grain.
    fn new(position: f32, rate: f32, duration: f32, pan: f32) -> Self {
        Self {
            position,
            rate,
            phase: 0.0,
            duration,
            pan,
            amplitude: 1.0,
        }
    }

    /// Returns true if the grain has finished playing.
    fn is_finished(&self) -> bool {
        self.phase >= self.duration
    }

    /// Returns the envelope value at the current phase (Hann window).
    fn envelope(&self) -> f32 {
        let t = self.phase / self.duration;
        // Hann window: 0.5 * (1 - cos(2πt))
        0.5 * (1.0 - (std::f32::consts::TAU * t).cos())
    }

    /// Advances the grain and returns the output sample.
    fn tick(&mut self, buffer: &[f32]) -> f32 {
        if self.is_finished() {
            return 0.0;
        }

        // Get sample position in the buffer
        let sample_pos = self.position + self.phase * self.rate;
        let sample = interpolate_buffer(buffer, sample_pos);

        // Apply envelope
        let output = sample * self.envelope() * self.amplitude;

        // Advance phase
        self.phase += 1.0;

        output
    }

    /// Returns stereo output (left, right) based on pan.
    fn tick_stereo(&mut self, buffer: &[f32]) -> (f32, f32) {
        let mono = self.tick(buffer);

        // Equal power panning
        let pan_normalized = (self.pan + 1.0) / 2.0; // 0 to 1
        let left_gain = (1.0 - pan_normalized).sqrt();
        let right_gain = pan_normalized.sqrt();

        (mono * left_gain, mono * right_gain)
    }
}

/// Interpolates a sample from a buffer using linear interpolation.
fn interpolate_buffer(buffer: &[f32], position: f32) -> f32 {
    if buffer.is_empty() {
        return 0.0;
    }

    let len = buffer.len() as f32;
    let pos = position.rem_euclid(len);

    let index = pos as usize;
    let frac = pos - index as f32;

    let sample1 = buffer[index % buffer.len()];
    let sample2 = buffer[(index + 1) % buffer.len()];

    sample1 + (sample2 - sample1) * frac
}

/// A cloud of grains for granular synthesis.
#[derive(Debug)]
pub struct GrainCloud {
    /// Source audio buffer.
    buffer: Vec<f32>,
    /// Sample rate.
    sample_rate: f32,
    /// Active grains.
    grains: VecDeque<Grain>,
    /// Maximum number of simultaneous grains.
    max_grains: usize,
    /// Grain configuration.
    config: GranularSynth,
    /// Time until next grain spawn.
    next_grain_time: f32,
    /// Random number generator state.
    rng_state: u64,
    /// Master volume.
    volume: f32,
}

impl GrainCloud {
    /// Creates a new grain cloud with the given source buffer.
    pub fn new(buffer: Vec<f32>, sample_rate: f32) -> Self {
        Self {
            buffer,
            sample_rate,
            grains: VecDeque::new(),
            max_grains: 64,
            config: GranularSynth::default(),
            next_grain_time: 0.0,
            rng_state: 12345,
            volume: 1.0,
        }
    }

    /// Sets the grain configuration.
    pub fn set_config(&mut self, config: GranularSynth) {
        self.config = config;
    }

    /// Returns a mutable reference to the configuration.
    pub fn config_mut(&mut self) -> &mut GranularSynth {
        &mut self.config
    }

    /// Sets the grain size in milliseconds.
    pub fn set_grain_size(&mut self, size_ms: f32) {
        self.config.size_ms = size_ms;
    }

    /// Sets the grain density (grains per second).
    pub fn set_density(&mut self, density: f32) {
        self.config.density = density;
    }

    /// Sets the playback position (0.0 to 1.0).
    pub fn set_position(&mut self, position: f32) {
        self.config.position = position.clamp(0.0, 1.0);
    }

    /// Sets the pitch multiplier.
    pub fn set_pitch(&mut self, pitch: f32) {
        self.config.pitch = pitch;
    }

    /// Sets the maximum number of simultaneous grains.
    pub fn set_max_grains(&mut self, max: usize) {
        self.max_grains = max;
    }

    /// Sets the master volume.
    pub fn set_volume(&mut self, volume: f32) {
        self.volume = volume;
    }

    /// Replaces the source buffer.
    pub fn set_buffer(&mut self, buffer: Vec<f32>) {
        self.buffer = buffer;
        self.grains.clear();
    }

    /// Returns the current number of active grains.
    pub fn active_grain_count(&self) -> usize {
        self.grains.len()
    }

    /// Generates a random float in [0, 1].
    fn random(&mut self) -> f32 {
        self.rng_state = self
            .rng_state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.rng_state >> 33) as f32) / (u32::MAX as f32)
    }

    /// Generates a random float in [-1, 1].
    fn random_bipolar(&mut self) -> f32 {
        self.random() * 2.0 - 1.0
    }

    /// Spawns a new grain with the current configuration.
    fn spawn_grain(&mut self) {
        if self.grains.len() >= self.max_grains || self.buffer.is_empty() {
            return;
        }

        // Calculate grain parameters with jitter
        let size_ms = self.config.size_ms
            + self.config.size_ms * self.config.size_jitter * self.random_bipolar();
        let size_samples = (size_ms / 1000.0 * self.sample_rate).max(1.0);

        let position = self.config.position + self.config.position_jitter * self.random_bipolar();
        let position_samples = position.clamp(0.0, 1.0) * self.buffer.len() as f32;

        let pitch = self.config.pitch
            + self.config.pitch * self.config.pitch_jitter * self.random_bipolar();

        let pan =
            (self.config.pan + self.config.pan_jitter * self.random_bipolar()).clamp(-1.0, 1.0);

        let grain = Grain::new(position_samples, pitch.max(0.01), size_samples, pan);
        self.grains.push_back(grain);
    }

    /// Processes one sample and returns mono output.
    pub fn tick(&mut self) -> f32 {
        // Spawn new grains based on density
        self.next_grain_time -= 1.0;
        if self.next_grain_time <= 0.0 {
            self.spawn_grain();
            // Time between grains in samples
            let interval = self.sample_rate / self.config.density.max(0.1);
            self.next_grain_time = interval;
        }

        // Process all active grains
        let mut output = 0.0;
        for grain in &mut self.grains {
            output += grain.tick(&self.buffer);
        }

        // Remove finished grains
        self.grains.retain(|g| !g.is_finished());

        // Apply volume and normalize by grain count to prevent clipping
        let normalize = (self.grains.len() as f32 + 1.0).sqrt();
        output * self.volume / normalize
    }

    /// Processes one sample and returns stereo output (left, right).
    pub fn tick_stereo(&mut self) -> (f32, f32) {
        // Spawn new grains based on density
        self.next_grain_time -= 1.0;
        if self.next_grain_time <= 0.0 {
            self.spawn_grain();
            let interval = self.sample_rate / self.config.density.max(0.1);
            self.next_grain_time = interval;
        }

        // Process all active grains
        let mut left = 0.0;
        let mut right = 0.0;
        for grain in &mut self.grains {
            let (l, r) = grain.tick_stereo(&self.buffer);
            left += l;
            right += r;
        }

        // Remove finished grains
        self.grains.retain(|g| !g.is_finished());

        // Normalize
        let normalize = (self.grains.len() as f32 + 1.0).sqrt();
        (
            left * self.volume / normalize,
            right * self.volume / normalize,
        )
    }

    /// Generates a buffer of samples.
    pub fn generate(&mut self, num_samples: usize) -> Vec<f32> {
        (0..num_samples).map(|_| self.tick()).collect()
    }

    /// Generates a stereo buffer of samples.
    pub fn generate_stereo(&mut self, num_samples: usize) -> Vec<(f32, f32)> {
        (0..num_samples).map(|_| self.tick_stereo()).collect()
    }
}

/// Generates a simple sine grain buffer.
pub fn sine_grain_buffer(frequency: f32, sample_rate: f32, duration_ms: f32) -> Vec<f32> {
    let num_samples = (duration_ms / 1000.0 * sample_rate) as usize;
    let phase_inc = std::f32::consts::TAU * frequency / sample_rate;

    (0..num_samples)
        .map(|i| (phase_inc * i as f32).sin())
        .collect()
}

/// Generates a noise grain buffer.
pub fn noise_grain_buffer(sample_rate: f32, duration_ms: f32, seed: u64) -> Vec<f32> {
    let num_samples = (duration_ms / 1000.0 * sample_rate) as usize;
    let mut rng = seed;

    (0..num_samples)
        .map(|_| {
            rng = rng
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((rng >> 33) as f32 / (u32::MAX as f32)) * 2.0 - 1.0
        })
        .collect()
}

/// A simple grain player that plays grains at specific times.
#[derive(Debug)]
pub struct GrainScheduler {
    /// Source buffer.
    buffer: Vec<f32>,
    /// Sample rate.
    sample_rate: f32,
    /// Scheduled grains (start_sample, grain).
    scheduled: Vec<(usize, Grain)>,
    /// Current sample position.
    current_sample: usize,
}

impl GrainScheduler {
    /// Creates a new grain scheduler.
    pub fn new(buffer: Vec<f32>, sample_rate: f32) -> Self {
        Self {
            buffer,
            sample_rate,
            scheduled: Vec::new(),
            current_sample: 0,
        }
    }

    /// Schedules a grain to play at a specific time.
    pub fn schedule(&mut self, start_ms: f32, position: f32, duration_ms: f32, pitch: f32) {
        let start_sample = (start_ms / 1000.0 * self.sample_rate) as usize;
        let position_samples = position.clamp(0.0, 1.0) * self.buffer.len() as f32;
        let duration_samples = (duration_ms / 1000.0 * self.sample_rate).max(1.0);

        let grain = Grain::new(position_samples, pitch, duration_samples, 0.0);
        self.scheduled.push((start_sample, grain));
    }

    /// Resets the scheduler to the beginning.
    pub fn reset(&mut self) {
        self.current_sample = 0;
        // Reset all grain phases
        for (_, grain) in &mut self.scheduled {
            grain.phase = 0.0;
        }
    }

    /// Processes one sample.
    pub fn tick(&mut self) -> f32 {
        let mut output = 0.0;

        for (start_sample, grain) in &mut self.scheduled {
            if self.current_sample >= *start_sample && !grain.is_finished() {
                output += grain.tick(&self.buffer);
            }
        }

        self.current_sample += 1;
        output
    }

    /// Generates a buffer of samples.
    pub fn generate(&mut self, num_samples: usize) -> Vec<f32> {
        (0..num_samples).map(|_| self.tick()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grain_config_struct() {
        let config = GranularSynth {
            size_ms: 100.0,
            density: 20.0,
            pitch: 1.5,
            position: 0.5,
            ..Default::default()
        };

        assert_eq!(config.size_ms, 100.0);
        assert_eq!(config.density, 20.0);
        assert_eq!(config.pitch, 1.5);
        assert_eq!(config.position, 0.5);
    }

    #[test]
    fn test_grain_cloud_creation() {
        let buffer = vec![0.0; 44100];
        let cloud = GrainCloud::new(buffer, 44100.0);
        assert_eq!(cloud.active_grain_count(), 0);
    }

    #[test]
    fn test_grain_cloud_generates_samples() {
        let buffer = sine_grain_buffer(440.0, 44100.0, 1000.0);
        let mut cloud = GrainCloud::new(buffer, 44100.0);

        cloud.set_density(50.0);
        cloud.set_grain_size(50.0);

        let samples = cloud.generate(1000);
        assert_eq!(samples.len(), 1000);
    }

    #[test]
    fn test_grain_cloud_stereo() {
        let buffer = sine_grain_buffer(440.0, 44100.0, 500.0);
        let mut cloud = GrainCloud::new(buffer, 44100.0);

        cloud.set_density(20.0);
        cloud.config_mut().pan_jitter = 1.0; // Full stereo spread

        let samples = cloud.generate_stereo(1000);
        assert_eq!(samples.len(), 1000);
    }

    #[test]
    fn test_grain_envelope() {
        let mut grain = Grain::new(0.0, 1.0, 100.0, 0.0);

        // At start, envelope should be near 0
        assert!(grain.envelope() < 0.1);

        // Advance to middle
        grain.phase = 50.0;
        // At middle, envelope should be near 1
        assert!(grain.envelope() > 0.9);

        // Advance to end
        grain.phase = 99.0;
        // Near end, envelope should be near 0
        assert!(grain.envelope() < 0.1);
    }

    #[test]
    fn test_interpolate_buffer() {
        let buffer = vec![0.0, 1.0, 0.0];

        // Exact positions
        assert_eq!(interpolate_buffer(&buffer, 0.0), 0.0);
        assert_eq!(interpolate_buffer(&buffer, 1.0), 1.0);
        assert_eq!(interpolate_buffer(&buffer, 2.0), 0.0);

        // Interpolated position
        let val = interpolate_buffer(&buffer, 0.5);
        assert!((val - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_sine_grain_buffer() {
        let buffer = sine_grain_buffer(440.0, 44100.0, 100.0);

        // Should have correct number of samples
        let expected = (100.0 / 1000.0 * 44100.0) as usize;
        assert_eq!(buffer.len(), expected);

        // Values should be in [-1, 1]
        for sample in &buffer {
            assert!(*sample >= -1.0 && *sample <= 1.0);
        }
    }

    #[test]
    fn test_noise_grain_buffer() {
        let buffer = noise_grain_buffer(44100.0, 100.0, 12345);

        let expected = (100.0 / 1000.0 * 44100.0) as usize;
        assert_eq!(buffer.len(), expected);

        // Values should be in [-1, 1]
        for sample in &buffer {
            assert!(*sample >= -1.0 && *sample <= 1.0);
        }
    }

    #[test]
    fn test_grain_scheduler() {
        let buffer = sine_grain_buffer(440.0, 44100.0, 500.0);
        let mut scheduler = GrainScheduler::new(buffer, 44100.0);

        // Schedule a grain at 100ms
        scheduler.schedule(100.0, 0.0, 50.0, 1.0);

        // Generate samples
        let samples = scheduler.generate(10000);
        assert_eq!(samples.len(), 10000);

        // Early samples should be silent
        assert!(samples[0].abs() < 0.01);
        assert!(samples[100].abs() < 0.01);
    }

    #[test]
    fn test_grain_cloud_position_jitter() {
        let buffer = vec![1.0; 44100];
        let mut cloud = GrainCloud::new(buffer, 44100.0);

        cloud.set_density(100.0);
        cloud.config_mut().position_jitter = 0.5;

        // Should not crash with jitter
        let samples = cloud.generate(1000);
        assert_eq!(samples.len(), 1000);
    }
}
