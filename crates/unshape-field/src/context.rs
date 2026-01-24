//! Evaluation context for field sampling.

use glam::UVec2;

/// Context provided during field evaluation.
///
/// Contains runtime parameters that fields may query.
/// Follows the Shadertoy pattern for time/resolution.
#[derive(Debug, Clone)]
pub struct EvalContext {
    /// Absolute time in seconds (like Shadertoy's iTime).
    pub time: f32,

    /// Delta time since last evaluation (like iTimeDelta).
    pub dt: f32,

    /// Frame number (like iFrame).
    pub frame: u64,

    /// Output resolution when materializing (like iResolution).
    pub resolution: UVec2,

    /// Sample rate in Hz (for audio).
    pub sample_rate: f32,
}

impl Default for EvalContext {
    fn default() -> Self {
        Self {
            time: 0.0,
            dt: 1.0 / 60.0,
            frame: 0,
            resolution: UVec2::new(1024, 1024),
            sample_rate: 48000.0,
        }
    }
}

impl EvalContext {
    /// Creates a new context with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the time.
    pub fn with_time(mut self, time: f32) -> Self {
        self.time = time;
        self
    }

    /// Sets the delta time.
    pub fn with_dt(mut self, dt: f32) -> Self {
        self.dt = dt;
        self
    }

    /// Sets the frame number.
    pub fn with_frame(mut self, frame: u64) -> Self {
        self.frame = frame;
        self
    }

    /// Sets the resolution.
    pub fn with_resolution(mut self, width: u32, height: u32) -> Self {
        self.resolution = UVec2::new(width, height);
        self
    }

    /// Sets the sample rate.
    pub fn with_sample_rate(mut self, sample_rate: f32) -> Self {
        self.sample_rate = sample_rate;
        self
    }

    /// Advances time by dt and increments frame.
    pub fn advance(&mut self) {
        self.time += self.dt;
        self.frame += 1;
    }
}
