//! 3D spatial audio and HRTF processing.
//!
//! Provides binaural audio spatialization for 3D sound positioning:
//! - HRTF-based binaural rendering
//! - Distance attenuation models
//! - Simple panning algorithms
//!
//! # Example
//!
//! ```
//! use rhizome_resin_audio::spatial::{SpatialSource, SpatialListener, Spatializer, Vec3};
//!
//! let listener = SpatialListener::default();
//! let source = SpatialSource::at(Vec3::new(1.0, 0.0, 0.0)); // Right of listener
//!
//! let mono_audio: Vec<f32> = (0..1024).map(|i| (i as f32 * 0.1).sin()).collect();
//!
//! let spatializer = Spatializer::new(44100);
//! let (left, right) = spatializer.process_mono(&mono_audio, &source, &listener);
//! ```

use std::f32::consts::PI;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// A 3D vector for spatial audio.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Vec3 {
    /// Creates a new vector.
    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    /// The zero vector.
    pub const ZERO: Self = Self {
        x: 0.0,
        y: 0.0,
        z: 0.0,
    };

    /// The forward direction (negative Z).
    pub const FORWARD: Self = Self {
        x: 0.0,
        y: 0.0,
        z: -1.0,
    };

    /// The up direction (positive Y).
    pub const UP: Self = Self {
        x: 0.0,
        y: 1.0,
        z: 0.0,
    };

    /// Returns the length of the vector.
    pub fn length(&self) -> f32 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }

    /// Returns a normalized version of the vector.
    pub fn normalize(&self) -> Self {
        let len = self.length();
        if len > 1e-8 {
            Self {
                x: self.x / len,
                y: self.y / len,
                z: self.z / len,
            }
        } else {
            Self::ZERO
        }
    }

    /// Computes the dot product.
    pub fn dot(&self, other: Self) -> f32 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    /// Computes the cross product.
    pub fn cross(&self, other: Self) -> Self {
        Self {
            x: self.y * other.z - self.z * other.y,
            y: self.z * other.x - self.x * other.z,
            z: self.x * other.y - self.y * other.x,
        }
    }
}

impl std::ops::Sub for Vec3 {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
            z: self.z - rhs.z,
        }
    }
}

impl std::ops::Add for Vec3 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
            z: self.z + rhs.z,
        }
    }
}

impl std::ops::Mul<f32> for Vec3 {
    type Output = Self;
    fn mul(self, rhs: f32) -> Self {
        Self {
            x: self.x * rhs,
            y: self.y * rhs,
            z: self.z * rhs,
        }
    }
}

/// A spatial audio source.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct SpatialSource {
    /// Position in world space.
    pub position: Vec3,
    /// Velocity for Doppler effect (optional).
    pub velocity: Vec3,
    /// Source directivity (1.0 = omnidirectional, 0.0 = fully directional).
    pub directivity: f32,
    /// Direction the source is facing (for directional sources).
    pub direction: Vec3,
    /// Inner cone angle in radians (full volume).
    pub inner_cone: f32,
    /// Outer cone angle in radians (attenuated volume).
    pub outer_cone: f32,
    /// Outer cone gain (0.0 - 1.0).
    pub outer_gain: f32,
}

impl Default for SpatialSource {
    fn default() -> Self {
        Self {
            position: Vec3::ZERO,
            velocity: Vec3::ZERO,
            directivity: 1.0,
            direction: Vec3::FORWARD,
            inner_cone: PI,
            outer_cone: 2.0 * PI,
            outer_gain: 0.0,
        }
    }
}

impl SpatialSource {
    /// Creates a source at the given position.
    pub fn at(position: Vec3) -> Self {
        Self {
            position,
            ..Default::default()
        }
    }

    /// Makes the source directional.
    pub fn with_cone(mut self, direction: Vec3, inner: f32, outer: f32, outer_gain: f32) -> Self {
        self.direction = direction.normalize();
        self.inner_cone = inner;
        self.outer_cone = outer;
        self.outer_gain = outer_gain;
        self.directivity = 0.0;
        self
    }
}

/// A spatial audio listener (typically the camera/player).
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct SpatialListener {
    /// Position in world space.
    pub position: Vec3,
    /// Forward direction (where the listener is looking).
    pub forward: Vec3,
    /// Up direction.
    pub up: Vec3,
    /// Velocity for Doppler effect.
    pub velocity: Vec3,
}

impl Default for SpatialListener {
    fn default() -> Self {
        Self {
            position: Vec3::ZERO,
            forward: Vec3::FORWARD,
            up: Vec3::UP,
            velocity: Vec3::ZERO,
        }
    }
}

impl SpatialListener {
    /// Creates a listener at the given position.
    pub fn at(position: Vec3) -> Self {
        Self {
            position,
            ..Default::default()
        }
    }

    /// Sets the listener orientation.
    pub fn with_orientation(mut self, forward: Vec3, up: Vec3) -> Self {
        self.forward = forward.normalize();
        self.up = up.normalize();
        self
    }

    /// Computes the right vector from forward and up.
    pub fn right(&self) -> Vec3 {
        self.forward.cross(self.up).normalize()
    }
}

/// Distance attenuation model.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum DistanceModel {
    /// No distance attenuation.
    None,
    /// Linear falloff from ref_dist to max_dist.
    Linear { ref_dist: f32, max_dist: f32 },
    /// Inverse distance (1/distance).
    Inverse { ref_dist: f32, rolloff: f32 },
    /// Exponential falloff.
    Exponential { ref_dist: f32, rolloff: f32 },
}

impl Default for DistanceModel {
    fn default() -> Self {
        Self::Inverse {
            ref_dist: 1.0,
            rolloff: 1.0,
        }
    }
}

impl DistanceModel {
    /// Computes the gain for a given distance.
    pub fn gain(&self, distance: f32) -> f32 {
        match *self {
            DistanceModel::None => 1.0,
            DistanceModel::Linear { ref_dist, max_dist } => {
                if distance <= ref_dist {
                    1.0
                } else if distance >= max_dist {
                    0.0
                } else {
                    1.0 - (distance - ref_dist) / (max_dist - ref_dist)
                }
            }
            DistanceModel::Inverse { ref_dist, rolloff } => {
                if distance <= ref_dist {
                    1.0
                } else {
                    ref_dist / (ref_dist + rolloff * (distance - ref_dist))
                }
            }
            DistanceModel::Exponential { ref_dist, rolloff } => {
                if distance <= ref_dist {
                    1.0
                } else {
                    (distance / ref_dist).powf(-rolloff)
                }
            }
        }
    }
}

/// Input for spatialize operation.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct SpatializeInput {
    /// Mono audio samples.
    pub audio: Vec<f32>,
    /// Sound source position.
    pub source: SpatialSource,
    /// Listener position.
    pub listener: SpatialListener,
    /// Sample rate in Hz.
    pub sample_rate: u32,
}

/// Output from spatialize operation.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct SpatializeOutput {
    /// Left channel audio.
    pub left: Vec<f32>,
    /// Right channel audio.
    pub right: Vec<f32>,
}

/// Configuration for the spatializer.
///
/// Operations on audio buffers use the ops-as-values pattern.
/// See `docs/design/ops-as-values.md`.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dynop", derive(rhizome_resin_op::Op))]
#[cfg_attr(feature = "dynop", op(input = SpatializeInput, output = SpatializeOutput))]
pub struct Spatialize {
    /// Distance attenuation model.
    pub distance_model: DistanceModel,
    /// Whether to enable Doppler effect.
    pub enable_doppler: bool,
    /// Speed of sound in units per second (default: 343 m/s).
    pub speed_of_sound: f32,
    /// HRTF mode.
    pub hrtf_mode: HrtfMode,
    /// Maximum delay for ITD in samples.
    pub max_itd_samples: usize,
}

impl Default for Spatialize {
    fn default() -> Self {
        Self {
            distance_model: DistanceModel::default(),
            enable_doppler: true,
            speed_of_sound: 343.0,
            hrtf_mode: HrtfMode::Simple,
            max_itd_samples: 44, // ~1ms at 44.1kHz
        }
    }
}

impl Spatialize {
    /// Applies this spatializer configuration to process mono audio.
    ///
    /// Returns stereo output (left, right).
    pub fn apply(&self, input: &SpatializeInput) -> SpatializeOutput {
        let spatializer = Spatializer::with_config(input.sample_rate, self.clone());
        let (left, right) = spatializer.process_mono(&input.audio, &input.source, &input.listener);
        SpatializeOutput { left, right }
    }
}

/// Backwards-compatible type alias.
pub type SpatializerConfig = Spatialize;

/// HRTF processing mode.
#[derive(Debug, Clone, Copy, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum HrtfMode {
    /// Simple panning (no HRTF).
    #[default]
    Simple,
    /// Basic HRTF with ITD and ILD.
    Basic,
    /// Enhanced HRTF with frequency-dependent filtering.
    Enhanced,
}

/// Spatial audio processor.
#[derive(Debug, Clone)]
pub struct Spatializer {
    /// Sample rate.
    pub sample_rate: u32,
    /// Configuration.
    pub config: Spatialize,
    /// Delay buffer for ITD.
    delay_buffer_l: Vec<f32>,
    delay_buffer_r: Vec<f32>,
    /// Write position in delay buffers.
    delay_pos: usize,
    /// Previous low-pass state for head shadow.
    lp_state_l: f32,
    lp_state_r: f32,
}

impl Spatializer {
    /// Creates a new spatializer.
    pub fn new(sample_rate: u32) -> Self {
        Self::with_config(sample_rate, Spatialize::default())
    }

    /// Creates a spatializer with custom configuration.
    pub fn with_config(sample_rate: u32, config: Spatialize) -> Self {
        let buffer_size = config.max_itd_samples * 2;
        Self {
            sample_rate,
            config,
            delay_buffer_l: vec![0.0; buffer_size],
            delay_buffer_r: vec![0.0; buffer_size],
            delay_pos: 0,
            lp_state_l: 0.0,
            lp_state_r: 0.0,
        }
    }

    /// Processes a mono signal and returns stereo (left, right).
    pub fn process_mono(
        &self,
        input: &[f32],
        source: &SpatialSource,
        listener: &SpatialListener,
    ) -> (Vec<f32>, Vec<f32>) {
        // Compute relative position
        let relative = source.position - listener.position;
        let distance = relative.length();

        // Transform to listener space
        let right = listener.right();
        let forward = listener.forward;

        let local_x = relative.dot(right); // Left (-) / Right (+)
        let local_z = relative.dot(forward); // Behind (-) / Front (+)
        let local_y = relative.dot(listener.up); // Below (-) / Above (+)

        // Calculate azimuth angle (-PI to PI, 0 = front)
        // atan2(x, z) gives 0 when x=0 and z>0 (front)
        let azimuth = local_x.atan2(local_z);

        // Calculate elevation (-PI/2 to PI/2)
        let elevation = (local_y / distance.max(0.001)).clamp(-1.0, 1.0).asin();

        // Distance attenuation
        let distance_gain = self.config.distance_model.gain(distance);

        // Source directivity
        let directivity_gain = if source.directivity < 1.0 {
            let to_listener = (listener.position - source.position).normalize();
            let angle = source.direction.dot(to_listener).clamp(-1.0, 1.0).acos();

            if angle <= source.inner_cone {
                1.0
            } else if angle >= source.outer_cone {
                source.outer_gain
            } else {
                let t = (angle - source.inner_cone) / (source.outer_cone - source.inner_cone);
                1.0 - t * (1.0 - source.outer_gain)
            }
        } else {
            1.0
        };

        let gain = distance_gain * directivity_gain;

        // Compute panning/HRTF based on mode
        match self.config.hrtf_mode {
            HrtfMode::Simple => self.process_simple_pan(input, azimuth, gain),
            HrtfMode::Basic => self.process_basic_hrtf(input, azimuth, elevation, distance, gain),
            HrtfMode::Enhanced => {
                self.process_enhanced_hrtf(input, azimuth, elevation, distance, gain)
            }
        }
    }

    /// Simple stereo panning.
    fn process_simple_pan(&self, input: &[f32], azimuth: f32, gain: f32) -> (Vec<f32>, Vec<f32>) {
        // Convert azimuth to pan position (-1 = left, +1 = right)
        let pan = (azimuth / PI).clamp(-1.0, 1.0);

        // Constant power panning
        let angle = (pan + 1.0) * PI / 4.0; // 0 to PI/2
        let gain_l = angle.cos() * gain;
        let gain_r = angle.sin() * gain;

        let left: Vec<f32> = input.iter().map(|&s| s * gain_l).collect();
        let right: Vec<f32> = input.iter().map(|&s| s * gain_r).collect();

        (left, right)
    }

    /// Basic HRTF with ITD and ILD.
    fn process_basic_hrtf(
        &self,
        input: &[f32],
        azimuth: f32,
        _elevation: f32,
        distance: f32,
        gain: f32,
    ) -> (Vec<f32>, Vec<f32>) {
        let mut left = vec![0.0; input.len()];
        let mut right = vec![0.0; input.len()];

        // Head radius approximation (average human head ~8.5cm radius)
        let head_radius = 0.085;

        // ITD (Interaural Time Difference)
        // Woodworth model: ITD = r/c * (sin(θ) + θ)
        let itd_seconds =
            head_radius / self.config.speed_of_sound * (azimuth.sin().abs() + azimuth.abs());
        let itd_samples = (itd_seconds * self.sample_rate as f32) as usize;
        let itd_samples = itd_samples.min(self.config.max_itd_samples);

        // ILD (Interaural Level Difference)
        // Simple model: more attenuation on shadow side
        let ild_factor = 0.3; // dB reduction per radian
        let ild_db = ild_factor * azimuth.abs() * (180.0 / PI) / 90.0 * 6.0;
        let ild_linear = 10.0f32.powf(-ild_db / 20.0);

        // Determine which ear is closer
        let (near_gain, far_gain, near_delay, far_delay) = if azimuth >= 0.0 {
            // Sound is on the right
            (gain, gain * ild_linear, 0, itd_samples)
        } else {
            // Sound is on the left
            (gain * ild_linear, gain, itd_samples, 0)
        };

        // Apply delays and gains
        for (i, &sample) in input.iter().enumerate() {
            // Left ear
            let left_idx = i.saturating_sub(near_delay);
            if left_idx < input.len() && azimuth < 0.0 {
                left[i] = input[left_idx] * near_gain;
            } else {
                let far_idx = i.saturating_sub(far_delay);
                if far_idx < input.len() {
                    left[i] = input[far_idx] * far_gain;
                }
            }

            // Right ear
            let right_idx = i.saturating_sub(near_delay);
            if right_idx < input.len() && azimuth >= 0.0 {
                right[i] = input[right_idx] * near_gain;
            } else {
                let far_idx = i.saturating_sub(far_delay);
                if far_idx < input.len() {
                    right[i] = input[far_idx] * far_gain;
                }
            }
        }

        // Simple distance cue: slightly more reverb for far sources (not implemented here)
        let _ = distance;

        (left, right)
    }

    /// Enhanced HRTF with frequency-dependent head shadow.
    fn process_enhanced_hrtf(
        &self,
        input: &[f32],
        azimuth: f32,
        elevation: f32,
        distance: f32,
        gain: f32,
    ) -> (Vec<f32>, Vec<f32>) {
        // Start with basic HRTF
        let (mut left, mut right) =
            self.process_basic_hrtf(input, azimuth, elevation, distance, gain);

        // Add frequency-dependent head shadow (low-pass on far ear)
        // More low-pass for sources behind the head
        let shadow_amount = (1.0 - azimuth.cos()).max(0.0) * 0.5;

        // Simple one-pole low-pass
        let cutoff_factor = 1.0 - shadow_amount * 0.8;
        let alpha = cutoff_factor.clamp(0.1, 1.0);

        // Apply head shadow to far ear
        if azimuth >= 0.0 {
            // Right is near, apply shadow to left
            apply_lowpass(&mut left, alpha);
        } else {
            // Left is near, apply shadow to right
            apply_lowpass(&mut right, alpha);
        }

        // Pinna cues for elevation (simplified)
        // Real HRTF would use measured/modeled filters
        let elevation_effect = elevation.abs() / (PI / 2.0);
        let high_shelf = 1.0 - elevation_effect * 0.2;

        for sample in left.iter_mut() {
            *sample *= high_shelf;
        }
        for sample in right.iter_mut() {
            *sample *= high_shelf;
        }

        (left, right)
    }

    /// Resets internal state.
    pub fn reset(&mut self) {
        self.delay_buffer_l.fill(0.0);
        self.delay_buffer_r.fill(0.0);
        self.delay_pos = 0;
        self.lp_state_l = 0.0;
        self.lp_state_r = 0.0;
    }
}

/// Computes Doppler pitch shift factor.
pub fn doppler_factor(
    source_pos: Vec3,
    source_vel: Vec3,
    listener_pos: Vec3,
    listener_vel: Vec3,
    speed_of_sound: f32,
) -> f32 {
    let direction = (source_pos - listener_pos).normalize();

    // Velocity components along the line between source and listener
    let v_source = source_vel.dot(direction);
    let v_listener = listener_vel.dot(direction);

    // Doppler formula: f' = f * (c + v_listener) / (c + v_source)
    let c = speed_of_sound;
    ((c + v_listener) / (c + v_source)).clamp(0.5, 2.0)
}

/// Applies a simple one-pole low-pass filter in place.
fn apply_lowpass(signal: &mut [f32], alpha: f32) {
    let mut state = signal.first().copied().unwrap_or(0.0);
    for sample in signal.iter_mut() {
        state = alpha * *sample + (1.0 - alpha) * state;
        *sample = state;
    }
}

/// Stereo mix utilities.
pub struct StereoMix;

impl StereoMix {
    /// Mixes multiple stereo sources.
    pub fn mix(sources: &[(Vec<f32>, Vec<f32>)]) -> (Vec<f32>, Vec<f32>) {
        if sources.is_empty() {
            return (vec![], vec![]);
        }

        let len = sources.iter().map(|(l, _)| l.len()).max().unwrap_or(0);
        let mut left = vec![0.0; len];
        let mut right = vec![0.0; len];

        for (src_left, src_right) in sources {
            for (i, &s) in src_left.iter().enumerate() {
                left[i] += s;
            }
            for (i, &s) in src_right.iter().enumerate() {
                right[i] += s;
            }
        }

        (left, right)
    }

    /// Interleaves left and right channels.
    pub fn interleave(left: &[f32], right: &[f32]) -> Vec<f32> {
        let len = left.len().min(right.len());
        let mut output = Vec::with_capacity(len * 2);

        for i in 0..len {
            output.push(left[i]);
            output.push(right[i]);
        }

        output
    }

    /// De-interleaves stereo to left and right channels.
    pub fn deinterleave(stereo: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let len = stereo.len() / 2;
        let mut left = Vec::with_capacity(len);
        let mut right = Vec::with_capacity(len);

        for i in 0..len {
            left.push(stereo[i * 2]);
            right.push(stereo[i * 2 + 1]);
        }

        (left, right)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vec3_operations() {
        let a = Vec3::new(1.0, 0.0, 0.0);
        let b = Vec3::new(0.0, 1.0, 0.0);

        assert!((a.dot(b)).abs() < 0.001);
        assert!((a.length() - 1.0).abs() < 0.001);

        let c = a.cross(b);
        assert!((c.z - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_distance_model_inverse() {
        let model = DistanceModel::Inverse {
            ref_dist: 1.0,
            rolloff: 1.0,
        };

        assert!((model.gain(0.5) - 1.0).abs() < 0.001);
        assert!((model.gain(1.0) - 1.0).abs() < 0.001);
        assert!((model.gain(2.0) - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_distance_model_linear() {
        let model = DistanceModel::Linear {
            ref_dist: 1.0,
            max_dist: 10.0,
        };

        assert!((model.gain(0.5) - 1.0).abs() < 0.001);
        assert!((model.gain(1.0) - 1.0).abs() < 0.001);
        assert!((model.gain(10.0) - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_spatializer_simple_pan() {
        let spatializer = Spatializer::new(44100);

        let input: Vec<f32> = (0..1024).map(|i| (i as f32 * 0.1).sin()).collect();

        // Source on the right
        let source = SpatialSource::at(Vec3::new(1.0, 0.0, 0.0));
        let listener = SpatialListener::default();

        let (left, right) = spatializer.process_mono(&input, &source, &listener);

        // Right should be louder than left
        let left_energy: f32 = left.iter().map(|s| s * s).sum();
        let right_energy: f32 = right.iter().map(|s| s * s).sum();

        assert!(right_energy > left_energy);
    }

    #[test]
    fn test_spatializer_center() {
        let spatializer = Spatializer::new(44100);

        let input: Vec<f32> = (0..1024).map(|i| (i as f32 * 0.1).sin()).collect();

        // Source in front
        let source = SpatialSource::at(Vec3::new(0.0, 0.0, -1.0));
        let listener = SpatialListener::default();

        let (left, right) = spatializer.process_mono(&input, &source, &listener);

        // Should be roughly equal
        let left_energy: f32 = left.iter().map(|s| s * s).sum();
        let right_energy: f32 = right.iter().map(|s| s * s).sum();

        let ratio = left_energy / right_energy.max(0.001);
        assert!(
            (ratio - 1.0).abs() < 0.1,
            "Center source should have equal L/R"
        );
    }

    #[test]
    fn test_doppler_factor() {
        // Approaching source
        let factor = doppler_factor(
            Vec3::new(10.0, 0.0, 0.0),
            Vec3::new(-10.0, 0.0, 0.0), // Moving toward listener
            Vec3::ZERO,
            Vec3::ZERO,
            343.0,
        );
        assert!(factor > 1.0, "Approaching source should have higher pitch");

        // Receding source
        let factor = doppler_factor(
            Vec3::new(10.0, 0.0, 0.0),
            Vec3::new(10.0, 0.0, 0.0), // Moving away from listener
            Vec3::ZERO,
            Vec3::ZERO,
            343.0,
        );
        assert!(factor < 1.0, "Receding source should have lower pitch");
    }

    #[test]
    fn test_stereo_interleave() {
        let left = vec![1.0, 2.0, 3.0];
        let right = vec![4.0, 5.0, 6.0];

        let interleaved = StereoMix::interleave(&left, &right);
        assert_eq!(interleaved, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);

        let (l, r) = StereoMix::deinterleave(&interleaved);
        assert_eq!(l, left);
        assert_eq!(r, right);
    }

    #[test]
    fn test_stereo_mix() {
        let source1 = (vec![1.0, 2.0], vec![0.5, 1.0]);
        let source2 = (vec![0.5, 1.0], vec![1.0, 2.0]);

        let (left, right) = StereoMix::mix(&[source1, source2]);

        assert!((left[0] - 1.5).abs() < 0.001);
        assert!((right[0] - 1.5).abs() < 0.001);
    }

    #[test]
    fn test_hrtf_basic() {
        let mut config = Spatialize::default();
        config.hrtf_mode = HrtfMode::Basic;
        let spatializer = Spatializer::with_config(44100, config);

        let input: Vec<f32> = (0..1024).map(|i| (i as f32 * 0.1).sin()).collect();

        let source = SpatialSource::at(Vec3::new(1.0, 0.0, 0.0));
        let listener = SpatialListener::default();

        let (left, right) = spatializer.process_mono(&input, &source, &listener);

        assert_eq!(left.len(), input.len());
        assert_eq!(right.len(), input.len());
    }

    #[test]
    fn test_directional_source() {
        let spatializer = Spatializer::new(44100);

        let input: Vec<f32> = vec![1.0; 100];

        // Directional source pointing away from listener
        let source = SpatialSource::at(Vec3::new(0.0, 0.0, -2.0)).with_cone(
            Vec3::FORWARD,
            PI / 4.0,
            PI / 2.0,
            0.1,
        );
        let listener = SpatialListener::default();

        let (left, _right) = spatializer.process_mono(&input, &source, &listener);

        // Should be attenuated because listener is behind the cone
        let energy: f32 = left.iter().map(|s| s * s).sum();
        assert!(energy < 100.0 * 0.5, "Directional source should attenuate");
    }
}
