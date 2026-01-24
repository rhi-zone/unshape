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
//! use unshape_audio::spatial::{SpatialSource, SpatialListener, Spatializer, Vec3};
//!
//! let listener = SpatialListener::default();
//! let source = SpatialSource::at(Vec3::new(1.0, 0.0, 0.0)); // Right of listener
//!
//! let mono_audio: Vec<f32> = (0..1024).map(|i| (i as f32 * 0.1).sin()).collect();
//!
//! let mut spatializer = Spatializer::new(44100);
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

/// Pre-computed spatial parameters for sample-by-sample processing.
///
/// Compute once per frame with [`Spatializer::compute_params`], then use
/// for processing all samples in that frame with [`Spatializer::process_sample`].
#[derive(Debug, Clone, Copy)]
pub struct SpatialParams {
    /// Azimuth angle in radians (-PI to PI, 0 = front).
    pub azimuth: f32,
    /// Elevation angle in radians (-PI/2 to PI/2).
    pub elevation: f32,
    /// Distance from listener to source.
    pub distance: f32,
    /// Combined gain (distance attenuation * directivity).
    pub gain: f32,
    /// ITD delay in samples.
    pub itd_samples: usize,
    /// ILD linear gain factor.
    pub ild_linear: f32,
    /// Left channel pan gain (for Simple mode).
    pub pan_gain_l: f32,
    /// Right channel pan gain (for Simple mode).
    pub pan_gain_r: f32,
}

/// Configuration for the spatializer.
///
/// Operations on audio buffers use the ops-as-values pattern.
/// See `docs/design/ops-as-values.md`.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dynop", derive(unshape_op::Op))]
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
        let mut spatializer = Spatializer::with_config(input.sample_rate, self.clone());
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
    /// Fractional read position for Doppler resampling.
    resample_pos: f64,
    /// Previous sample for linear interpolation across buffer boundaries.
    prev_sample: f32,
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
            resample_pos: 0.0,
            prev_sample: 0.0,
        }
    }

    /// Processes a mono signal and returns stereo (left, right).
    ///
    /// This method maintains internal state (delay buffers, filter state) for
    /// proper streaming across buffer boundaries. For stateless processing,
    /// use [`process_mono_stateless`].
    pub fn process_mono(
        &mut self,
        input: &[f32],
        source: &SpatialSource,
        listener: &SpatialListener,
    ) -> (Vec<f32>, Vec<f32>) {
        // Apply Doppler effect if enabled
        let processed_input: Vec<f32>;
        let audio = if self.config.enable_doppler {
            let factor = doppler_factor(
                source.position,
                source.velocity,
                listener.position,
                listener.velocity,
                self.config.speed_of_sound,
            );
            processed_input = self.resample_doppler(input, factor as f64);
            &processed_input[..]
        } else {
            input
        };

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
            HrtfMode::Simple => self.process_simple_pan(audio, azimuth, gain),
            HrtfMode::Basic => self.process_basic_hrtf(audio, azimuth, gain),
            HrtfMode::Enhanced => self.process_enhanced_hrtf(audio, azimuth, elevation, gain),
        }
    }

    /// Resamples input by Doppler factor using linear interpolation.
    ///
    /// Factor > 1 means higher pitch (approaching source), < 1 means lower pitch (receding).
    fn resample_doppler(&mut self, input: &[f32], factor: f64) -> Vec<f32> {
        if input.is_empty() {
            return Vec::new();
        }

        // Output length: same as input (real-time constraint)
        // We read from input at a rate determined by the Doppler factor
        let mut output = Vec::with_capacity(input.len());

        for _ in 0..input.len() {
            // Integer and fractional parts of read position
            let pos_int = self.resample_pos as usize;
            let pos_frac = self.resample_pos - pos_int as f64;

            // Get samples for interpolation
            let s0 = if pos_int == 0 {
                self.prev_sample
            } else {
                input.get(pos_int - 1).copied().unwrap_or(0.0)
            };
            let s1 = input.get(pos_int).copied().unwrap_or(0.0);

            // Linear interpolation
            let sample = s0 + (s1 - s0) * pos_frac as f32;
            output.push(sample);

            // Advance read position by Doppler factor
            self.resample_pos += factor;
        }

        // Normalize position relative to input length and save state
        if !input.is_empty() {
            // Save last sample for next buffer's interpolation
            self.prev_sample = *input.last().unwrap();
            // Keep fractional overshoot for next buffer
            self.resample_pos -= input.len() as f64;
            // Clamp to valid range
            self.resample_pos = self.resample_pos.max(0.0);
        }

        output
    }

    /// Simple stereo panning (stateless, no delay buffers needed).
    fn process_simple_pan(
        &mut self,
        input: &[f32],
        azimuth: f32,
        gain: f32,
    ) -> (Vec<f32>, Vec<f32>) {
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
    ///
    /// Uses circular delay buffers for proper ITD across buffer boundaries.
    fn process_basic_hrtf(
        &mut self,
        input: &[f32],
        azimuth: f32,
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

        // Determine which ear is closer and compute gains/delays
        let (gain_l, gain_r, delay_l, delay_r) = if azimuth >= 0.0 {
            // Sound is on the right: right ear is near (no delay), left ear is far (delayed)
            (gain * ild_linear, gain, itd_samples, 0)
        } else {
            // Sound is on the left: left ear is near (no delay), right ear is far (delayed)
            (gain, gain * ild_linear, 0, itd_samples)
        };

        let buffer_len = self.delay_buffer_l.len();

        // Process each sample using circular delay buffers
        for (i, &sample) in input.iter().enumerate() {
            // Write input to both delay buffers at current position
            self.delay_buffer_l[self.delay_pos] = sample;
            self.delay_buffer_r[self.delay_pos] = sample;

            // Read from delay buffers with appropriate delays
            let read_pos_l = (self.delay_pos + buffer_len - delay_l) % buffer_len;
            let read_pos_r = (self.delay_pos + buffer_len - delay_r) % buffer_len;

            left[i] = self.delay_buffer_l[read_pos_l] * gain_l;
            right[i] = self.delay_buffer_r[read_pos_r] * gain_r;

            // Advance write position
            self.delay_pos = (self.delay_pos + 1) % buffer_len;
        }

        (left, right)
    }

    /// Enhanced HRTF with frequency-dependent head shadow.
    ///
    /// Uses persistent lowpass filter state for proper streaming.
    fn process_enhanced_hrtf(
        &mut self,
        input: &[f32],
        azimuth: f32,
        elevation: f32,
        gain: f32,
    ) -> (Vec<f32>, Vec<f32>) {
        // Start with basic HRTF
        let (mut left, mut right) = self.process_basic_hrtf(input, azimuth, gain);

        // Add frequency-dependent head shadow (low-pass on far ear)
        // More low-pass for sources behind the head
        let shadow_amount = (1.0 - azimuth.cos()).max(0.0) * 0.5;

        // Simple one-pole low-pass coefficient
        let cutoff_factor = 1.0 - shadow_amount * 0.8;
        let alpha = cutoff_factor.clamp(0.1, 1.0);

        // Apply head shadow to far ear using persistent state
        if azimuth >= 0.0 {
            // Right is near, apply shadow to left
            for sample in left.iter_mut() {
                self.lp_state_l = alpha * *sample + (1.0 - alpha) * self.lp_state_l;
                *sample = self.lp_state_l;
            }
        } else {
            // Left is near, apply shadow to right
            for sample in right.iter_mut() {
                self.lp_state_r = alpha * *sample + (1.0 - alpha) * self.lp_state_r;
                *sample = self.lp_state_r;
            }
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
        self.resample_pos = 0.0;
        self.prev_sample = 0.0;
    }

    /// Computes spatial parameters for a source/listener pair.
    ///
    /// Use this to pre-compute parameters once per frame, then process
    /// multiple samples with [`process_sample`].
    pub fn compute_params(
        &self,
        source: &SpatialSource,
        listener: &SpatialListener,
    ) -> SpatialParams {
        let relative = source.position - listener.position;
        let distance = relative.length();

        let right = listener.right();
        let forward = listener.forward;

        let local_x = relative.dot(right);
        let local_z = relative.dot(forward);
        let local_y = relative.dot(listener.up);

        let azimuth = local_x.atan2(local_z);
        let elevation = (local_y / distance.max(0.001)).clamp(-1.0, 1.0).asin();

        let distance_gain = self.config.distance_model.gain(distance);

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

        // Compute ITD/ILD parameters for Basic/Enhanced HRTF
        let head_radius = 0.085;
        let itd_seconds =
            head_radius / self.config.speed_of_sound * (azimuth.sin().abs() + azimuth.abs());
        let itd_samples =
            ((itd_seconds * self.sample_rate as f32) as usize).min(self.config.max_itd_samples);

        let ild_factor = 0.3;
        let ild_db = ild_factor * azimuth.abs() * (180.0 / PI) / 90.0 * 6.0;
        let ild_linear = 10.0f32.powf(-ild_db / 20.0);

        // Pan gains for simple mode
        let pan = (azimuth / PI).clamp(-1.0, 1.0);
        let pan_angle = (pan + 1.0) * PI / 4.0;

        SpatialParams {
            azimuth,
            elevation,
            distance,
            gain,
            itd_samples,
            ild_linear,
            pan_gain_l: pan_angle.cos(),
            pan_gain_r: pan_angle.sin(),
        }
    }

    /// Processes a single sample and returns stereo output (left, right).
    ///
    /// Use [`compute_params`] to pre-compute spatial parameters, then call
    /// this method for each sample. This avoids recomputing parameters for
    /// every sample when source/listener positions don't change within a buffer.
    ///
    /// Note: Doppler effect is not applied in sample-by-sample mode. Use
    /// [`process_mono`] for Doppler support.
    pub fn process_sample(&mut self, sample: f32, params: &SpatialParams) -> (f32, f32) {
        match self.config.hrtf_mode {
            HrtfMode::Simple => {
                let left = sample * params.pan_gain_l * params.gain;
                let right = sample * params.pan_gain_r * params.gain;
                (left, right)
            }
            HrtfMode::Basic => {
                let (gain_l, gain_r, delay_l, delay_r) = if params.azimuth >= 0.0 {
                    (
                        params.gain * params.ild_linear,
                        params.gain,
                        params.itd_samples,
                        0,
                    )
                } else {
                    (
                        params.gain,
                        params.gain * params.ild_linear,
                        0,
                        params.itd_samples,
                    )
                };

                let buffer_len = self.delay_buffer_l.len();

                // Write to delay buffers
                self.delay_buffer_l[self.delay_pos] = sample;
                self.delay_buffer_r[self.delay_pos] = sample;

                // Read with delays
                let read_pos_l = (self.delay_pos + buffer_len - delay_l) % buffer_len;
                let read_pos_r = (self.delay_pos + buffer_len - delay_r) % buffer_len;

                let left = self.delay_buffer_l[read_pos_l] * gain_l;
                let right = self.delay_buffer_r[read_pos_r] * gain_r;

                self.delay_pos = (self.delay_pos + 1) % buffer_len;

                (left, right)
            }
            HrtfMode::Enhanced => {
                // Start with basic processing
                let (mut left, mut right) = {
                    let (gain_l, gain_r, delay_l, delay_r) = if params.azimuth >= 0.0 {
                        (
                            params.gain * params.ild_linear,
                            params.gain,
                            params.itd_samples,
                            0,
                        )
                    } else {
                        (
                            params.gain,
                            params.gain * params.ild_linear,
                            0,
                            params.itd_samples,
                        )
                    };

                    let buffer_len = self.delay_buffer_l.len();

                    self.delay_buffer_l[self.delay_pos] = sample;
                    self.delay_buffer_r[self.delay_pos] = sample;

                    let read_pos_l = (self.delay_pos + buffer_len - delay_l) % buffer_len;
                    let read_pos_r = (self.delay_pos + buffer_len - delay_r) % buffer_len;

                    let l = self.delay_buffer_l[read_pos_l] * gain_l;
                    let r = self.delay_buffer_r[read_pos_r] * gain_r;

                    self.delay_pos = (self.delay_pos + 1) % buffer_len;

                    (l, r)
                };

                // Head shadow filtering
                let shadow_amount = (1.0 - params.azimuth.cos()).max(0.0) * 0.5;
                let cutoff_factor = 1.0 - shadow_amount * 0.8;
                let alpha = cutoff_factor.clamp(0.1, 1.0);

                if params.azimuth >= 0.0 {
                    self.lp_state_l = alpha * left + (1.0 - alpha) * self.lp_state_l;
                    left = self.lp_state_l;
                } else {
                    self.lp_state_r = alpha * right + (1.0 - alpha) * self.lp_state_r;
                    right = self.lp_state_r;
                }

                // Elevation effect
                let elevation_effect = params.elevation.abs() / (PI / 2.0);
                let high_shelf = 1.0 - elevation_effect * 0.2;
                left *= high_shelf;
                right *= high_shelf;

                (left, right)
            }
        }
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
        let mut spatializer = Spatializer::new(44100);

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
        let mut spatializer = Spatializer::new(44100);

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
        let mut spatializer = Spatializer::with_config(44100, config);

        let input: Vec<f32> = (0..1024).map(|i| (i as f32 * 0.1).sin()).collect();

        let source = SpatialSource::at(Vec3::new(1.0, 0.0, 0.0));
        let listener = SpatialListener::default();

        let (left, right) = spatializer.process_mono(&input, &source, &listener);

        assert_eq!(left.len(), input.len());
        assert_eq!(right.len(), input.len());
    }

    #[test]
    fn test_directional_source() {
        let mut spatializer = Spatializer::new(44100);

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

    #[test]
    fn test_streaming_api_matches_batch() {
        // Test that sample-by-sample processing matches batch processing
        // (when Doppler is disabled)
        let mut config = Spatialize::default();
        config.enable_doppler = false;
        config.hrtf_mode = HrtfMode::Simple;

        let input: Vec<f32> = (0..256).map(|i| (i as f32 * 0.1).sin()).collect();
        let source = SpatialSource::at(Vec3::new(1.0, 0.0, 0.0));
        let listener = SpatialListener::default();

        // Batch processing
        let mut batch_spatializer = Spatializer::with_config(44100, config.clone());
        let (batch_left, batch_right) = batch_spatializer.process_mono(&input, &source, &listener);

        // Sample-by-sample processing
        let mut stream_spatializer = Spatializer::with_config(44100, config);
        let params = stream_spatializer.compute_params(&source, &listener);

        let mut stream_left = Vec::with_capacity(input.len());
        let mut stream_right = Vec::with_capacity(input.len());

        for &sample in &input {
            let (l, r) = stream_spatializer.process_sample(sample, &params);
            stream_left.push(l);
            stream_right.push(r);
        }

        // Compare outputs
        for i in 0..input.len() {
            assert!(
                (batch_left[i] - stream_left[i]).abs() < 0.001,
                "Left mismatch at {}: {} vs {}",
                i,
                batch_left[i],
                stream_left[i]
            );
            assert!(
                (batch_right[i] - stream_right[i]).abs() < 0.001,
                "Right mismatch at {}: {} vs {}",
                i,
                batch_right[i],
                stream_right[i]
            );
        }
    }

    #[test]
    fn test_streaming_api_hrtf_basic() {
        let mut config = Spatialize::default();
        config.enable_doppler = false;
        config.hrtf_mode = HrtfMode::Basic;

        let input: Vec<f32> = (0..256).map(|i| (i as f32 * 0.1).sin()).collect();
        let source = SpatialSource::at(Vec3::new(1.0, 0.0, 0.0));
        let listener = SpatialListener::default();

        // Process using streaming API
        let mut spatializer = Spatializer::with_config(44100, config);
        let params = spatializer.compute_params(&source, &listener);

        let mut left = Vec::with_capacity(input.len());
        let mut right = Vec::with_capacity(input.len());

        for &sample in &input {
            let (l, r) = spatializer.process_sample(sample, &params);
            left.push(l);
            right.push(r);
        }

        // Right should be louder (source is on the right)
        let left_energy: f32 = left.iter().map(|s| s * s).sum();
        let right_energy: f32 = right.iter().map(|s| s * s).sum();
        assert!(right_energy > left_energy);
    }
}
