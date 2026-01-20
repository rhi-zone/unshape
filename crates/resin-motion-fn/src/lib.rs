//! Motion functions for animation.
//!
//! Provides time-based value generators for smooth animations:
//! - [`Constant`] - constant value
//! - [`Lerp`] - linear interpolation
//! - [`Eased`] - eased interpolation with any easing function
//! - [`Spring`] - critically damped spring motion
//! - [`Oscillate`] - sine wave oscillation
//! - [`Wiggle`] - noise-based random motion
//!
//! All motion functions implement `Field<f32, T>` where the input is time
//! and the output is the animated value.
//!
//! # Example
//!
//! ```
//! use rhizome_resin_motion_fn::{Spring, Oscillate, Motion};
//!
//! // Spring animation from 0 to 100
//! let spring = Spring::new(0.0, 100.0, 300.0, 15.0);
//! let value = spring.at(0.5); // Value at t=0.5
//!
//! // Oscillating value between -1 and 1
//! let osc = Oscillate::new(0.0, 1.0, 2.0, 0.0); // center, amplitude, frequency, phase
//! let value = osc.at(0.25);
//! ```

use glam::{Vec2, Vec3};
use rhizome_resin_field::{EvalContext, Field};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

// ============================================================================
// Motion trait
// ============================================================================

/// A motion function that produces values over time.
///
/// This is a convenience trait for `Field<f32, T>` with a simpler API.
pub trait Motion<T>: Clone {
    /// Get the value at time `t`.
    fn at(&self, t: f32) -> T;

    /// Get the value with normalized time (0..1 maps to 0..duration).
    fn at_normalized(&self, t: f32, duration: f32) -> T {
        self.at(t / duration)
    }
}

/// Macro to implement Field<f32, T> for a Motion type.
macro_rules! impl_field_for_motion {
    ($motion:ty, $output:ty) => {
        impl Field<f32, $output> for $motion {
            fn sample(&self, t: f32, _ctx: &EvalContext) -> $output {
                Motion::at(self, t)
            }
        }
    };
}

// ============================================================================
// Constant
// ============================================================================

/// A constant value that doesn't change over time.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Constant<T> {
    /// The constant value.
    pub value: T,
}

impl<T: Clone> Constant<T> {
    /// Creates a new constant motion.
    pub fn new(value: T) -> Self {
        Self { value }
    }
}

impl<T: Clone> Motion<T> for Constant<T> {
    fn at(&self, _t: f32) -> T {
        self.value.clone()
    }
}

impl_field_for_motion!(Constant<f32>, f32);
impl_field_for_motion!(Constant<Vec2>, Vec2);
impl_field_for_motion!(Constant<Vec3>, Vec3);

// ============================================================================
// Lerp (linear interpolation)
// ============================================================================

/// Linear interpolation from one value to another.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Lerp<T> {
    /// Starting value at t=0.
    pub from: T,
    /// Ending value at t=duration.
    pub to: T,
    /// Duration of the interpolation.
    pub duration: f32,
}

impl<T: Clone> Lerp<T> {
    /// Creates a new linear interpolation.
    pub fn new(from: T, to: T, duration: f32) -> Self {
        Self { from, to, duration }
    }
}

macro_rules! impl_lerp_motion {
    ($t:ty, $lerp:expr) => {
        impl Motion<$t> for Lerp<$t> {
            fn at(&self, t: f32) -> $t {
                let progress = (t / self.duration).clamp(0.0, 1.0);
                $lerp(self.from, self.to, progress)
            }
        }
    };
}

impl_lerp_motion!(f32, |a: f32, b: f32, t: f32| a + (b - a) * t);
impl_lerp_motion!(Vec2, Vec2::lerp);
impl_lerp_motion!(Vec3, Vec3::lerp);

impl_field_for_motion!(Lerp<f32>, f32);
impl_field_for_motion!(Lerp<Vec2>, Vec2);
impl_field_for_motion!(Lerp<Vec3>, Vec3);

// ============================================================================
// Eased (interpolation with easing)
// ============================================================================

/// Interpolation with an easing function.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Eased<T> {
    /// Starting value at t=0.
    pub from: T,
    /// Ending value at t=duration.
    pub to: T,
    /// Duration of the interpolation.
    pub duration: f32,
    /// Easing function to apply.
    pub easing: EasingType,
}

/// Available easing types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum EasingType {
    #[default]
    Linear,
    QuadIn,
    QuadOut,
    QuadInOut,
    CubicIn,
    CubicOut,
    CubicInOut,
    QuartIn,
    QuartOut,
    QuartInOut,
    QuintIn,
    QuintOut,
    QuintInOut,
    SineIn,
    SineOut,
    SineInOut,
    ExpoIn,
    ExpoOut,
    ExpoInOut,
    CircIn,
    CircOut,
    CircInOut,
    ElasticIn,
    ElasticOut,
    ElasticInOut,
    BackIn,
    BackOut,
    BackInOut,
    BounceIn,
    BounceOut,
    BounceInOut,
}

impl EasingType {
    /// Apply the easing function to a value in 0..1 range.
    pub fn apply(self, t: f32) -> f32 {
        use rhizome_resin_easing::*;
        match self {
            Self::Linear => t,
            Self::QuadIn => quad_in(t),
            Self::QuadOut => quad_out(t),
            Self::QuadInOut => quad_in_out(t),
            Self::CubicIn => cubic_in(t),
            Self::CubicOut => cubic_out(t),
            Self::CubicInOut => cubic_in_out(t),
            Self::QuartIn => quart_in(t),
            Self::QuartOut => quart_out(t),
            Self::QuartInOut => quart_in_out(t),
            Self::QuintIn => quint_in(t),
            Self::QuintOut => quint_out(t),
            Self::QuintInOut => quint_in_out(t),
            Self::SineIn => sine_in(t),
            Self::SineOut => sine_out(t),
            Self::SineInOut => sine_in_out(t),
            Self::ExpoIn => expo_in(t),
            Self::ExpoOut => expo_out(t),
            Self::ExpoInOut => expo_in_out(t),
            Self::CircIn => circ_in(t),
            Self::CircOut => circ_out(t),
            Self::CircInOut => circ_in_out(t),
            Self::ElasticIn => elastic_in(t),
            Self::ElasticOut => elastic_out(t),
            Self::ElasticInOut => elastic_in_out(t),
            Self::BackIn => back_in(t),
            Self::BackOut => back_out(t),
            Self::BackInOut => back_in_out(t),
            Self::BounceIn => bounce_in(t),
            Self::BounceOut => bounce_out(t),
            Self::BounceInOut => bounce_in_out(t),
        }
    }
}

impl<T: Clone> Eased<T> {
    /// Creates a new eased interpolation.
    pub fn new(from: T, to: T, duration: f32, easing: EasingType) -> Self {
        Self {
            from,
            to,
            duration,
            easing,
        }
    }
}

macro_rules! impl_eased_motion {
    ($t:ty, $lerp:expr) => {
        impl Motion<$t> for Eased<$t> {
            fn at(&self, t: f32) -> $t {
                let linear = (t / self.duration).clamp(0.0, 1.0);
                let eased = self.easing.apply(linear);
                $lerp(self.from, self.to, eased)
            }
        }
    };
}

impl_eased_motion!(f32, |a: f32, b: f32, t: f32| a + (b - a) * t);
impl_eased_motion!(Vec2, Vec2::lerp);
impl_eased_motion!(Vec3, Vec3::lerp);

impl_field_for_motion!(Eased<f32>, f32);
impl_field_for_motion!(Eased<Vec2>, Vec2);
impl_field_for_motion!(Eased<Vec3>, Vec3);

// ============================================================================
// Spring (critically damped spring motion)
// ============================================================================

/// Critically damped spring motion toward a target.
///
/// This produces smooth, organic motion with optional overshoot.
/// The spring equation is: x'' + 2ζω*x' + ω²*x = ω²*target
///
/// Where:
/// - ω = sqrt(stiffness) is the natural frequency
/// - ζ = damping / (2 * sqrt(stiffness)) is the damping ratio
///
/// For critically damped motion (no oscillation), use damping = 2 * sqrt(stiffness).
/// For underdamped motion (with overshoot), use lower damping.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Spring<T> {
    /// Starting value.
    pub from: T,
    /// Target value.
    pub to: T,
    /// Spring stiffness (higher = faster).
    pub stiffness: f32,
    /// Damping coefficient (higher = less oscillation).
    pub damping: f32,
}

impl<T: Clone> Spring<T> {
    /// Creates a new spring motion.
    ///
    /// For smooth motion without overshoot, use `Spring::critical(from, to, stiffness)`.
    pub fn new(from: T, to: T, stiffness: f32, damping: f32) -> Self {
        Self {
            from,
            to,
            stiffness,
            damping,
        }
    }

    /// Creates a critically damped spring (no overshoot).
    pub fn critical(from: T, to: T, stiffness: f32) -> Self {
        let damping = 2.0 * stiffness.sqrt();
        Self::new(from, to, stiffness, damping)
    }

    /// Creates an underdamped spring (with overshoot).
    pub fn bouncy(from: T, to: T, stiffness: f32) -> Self {
        let damping = stiffness.sqrt(); // Half of critical damping
        Self::new(from, to, stiffness, damping)
    }
}

/// Compute spring position at time t using analytical solution.
fn spring_value(from: f32, to: f32, stiffness: f32, damping: f32, t: f32) -> f32 {
    if t <= 0.0 {
        return from;
    }

    let omega = stiffness.sqrt(); // Natural frequency
    let zeta = damping / (2.0 * omega); // Damping ratio
    let delta = to - from;

    if zeta >= 1.0 {
        // Critically damped or overdamped
        let decay = (-omega * zeta * t).exp();
        if (zeta - 1.0).abs() < 0.001 {
            // Critically damped: x(t) = target - (A + Bt)e^(-ωt)
            to - delta * (1.0 + omega * t) * decay
        } else {
            // Overdamped: two real roots
            let s = (zeta * zeta - 1.0).sqrt();
            let r1 = -omega * (zeta - s);
            let r2 = -omega * (zeta + s);
            let c2 = delta * r1 / (r1 - r2);
            let c1 = delta - c2;
            to - c1 * (r1 * t).exp() - c2 * (r2 * t).exp()
        }
    } else {
        // Underdamped: oscillating
        let omega_d = omega * (1.0 - zeta * zeta).sqrt();
        let decay = (-omega * zeta * t).exp();
        let cos_term = (omega_d * t).cos();
        let sin_term = (omega_d * t).sin();
        to - delta * decay * (cos_term + (zeta * omega / omega_d) * sin_term)
    }
}

impl Motion<f32> for Spring<f32> {
    fn at(&self, t: f32) -> f32 {
        spring_value(self.from, self.to, self.stiffness, self.damping, t)
    }
}

impl Motion<Vec2> for Spring<Vec2> {
    fn at(&self, t: f32) -> Vec2 {
        Vec2::new(
            spring_value(self.from.x, self.to.x, self.stiffness, self.damping, t),
            spring_value(self.from.y, self.to.y, self.stiffness, self.damping, t),
        )
    }
}

impl Motion<Vec3> for Spring<Vec3> {
    fn at(&self, t: f32) -> Vec3 {
        Vec3::new(
            spring_value(self.from.x, self.to.x, self.stiffness, self.damping, t),
            spring_value(self.from.y, self.to.y, self.stiffness, self.damping, t),
            spring_value(self.from.z, self.to.z, self.stiffness, self.damping, t),
        )
    }
}

impl_field_for_motion!(Spring<f32>, f32);
impl_field_for_motion!(Spring<Vec2>, Vec2);
impl_field_for_motion!(Spring<Vec3>, Vec3);

// ============================================================================
// Oscillate (sine wave)
// ============================================================================

/// Sine wave oscillation.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Oscillate<T> {
    /// Center value.
    pub center: T,
    /// Amplitude (half the range).
    pub amplitude: T,
    /// Frequency in Hz.
    pub frequency: f32,
    /// Phase offset in radians.
    pub phase: f32,
}

impl<T: Clone> Oscillate<T> {
    /// Creates a new oscillation.
    pub fn new(center: T, amplitude: T, frequency: f32, phase: f32) -> Self {
        Self {
            center,
            amplitude,
            frequency,
            phase,
        }
    }
}

impl Motion<f32> for Oscillate<f32> {
    fn at(&self, t: f32) -> f32 {
        let angle = std::f32::consts::TAU * self.frequency * t + self.phase;
        self.center + self.amplitude * angle.sin()
    }
}

impl Motion<Vec2> for Oscillate<Vec2> {
    fn at(&self, t: f32) -> Vec2 {
        let angle = std::f32::consts::TAU * self.frequency * t + self.phase;
        let sin = angle.sin();
        self.center + self.amplitude * sin
    }
}

impl Motion<Vec3> for Oscillate<Vec3> {
    fn at(&self, t: f32) -> Vec3 {
        let angle = std::f32::consts::TAU * self.frequency * t + self.phase;
        let sin = angle.sin();
        self.center + self.amplitude * sin
    }
}

impl_field_for_motion!(Oscillate<f32>, f32);
impl_field_for_motion!(Oscillate<Vec2>, Vec2);
impl_field_for_motion!(Oscillate<Vec3>, Vec3);

// ============================================================================
// Wiggle (noise-based)
// ============================================================================

/// Noise-based random motion.
///
/// Produces smooth random values using Perlin noise.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Wiggle {
    /// Base value around which to wiggle.
    pub center: f32,
    /// Maximum deviation from center.
    pub amplitude: f32,
    /// Speed of the wiggle (frequency).
    pub frequency: f32,
    /// Seed for noise generation.
    pub seed: f32,
    /// Number of noise octaves for more detail.
    pub octaves: u32,
}

impl Wiggle {
    /// Creates a new wiggle motion.
    pub fn new(center: f32, amplitude: f32, frequency: f32, seed: f32) -> Self {
        Self {
            center,
            amplitude,
            frequency,
            seed,
            octaves: 1,
        }
    }

    /// Creates a wiggle with multiple octaves for more detail.
    pub fn with_octaves(mut self, octaves: u32) -> Self {
        self.octaves = octaves;
        self
    }
}

impl Motion<f32> for Wiggle {
    fn at(&self, t: f32) -> f32 {
        use rhizome_resin_noise::Noise2D;
        let noise = if self.octaves > 1 {
            rhizome_resin_noise::Fbm::new(rhizome_resin_noise::Perlin2D::with_seed(
                self.seed as i32,
            ))
            .octaves(self.octaves)
            .sample(t * self.frequency, 0.0)
        } else {
            rhizome_resin_noise::Perlin2D::with_seed(self.seed as i32)
                .sample(t * self.frequency, 0.0)
        };
        // Perlin noise returns 0..1, convert to -1..1 then scale
        let normalized = noise * 2.0 - 1.0;
        self.center + self.amplitude * normalized
    }
}

/// 2D noise-based random motion.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Wiggle2D {
    /// Base value around which to wiggle.
    pub center: Vec2,
    /// Maximum deviation from center.
    pub amplitude: Vec2,
    /// Speed of the wiggle (frequency).
    pub frequency: f32,
    /// Seed for noise generation.
    pub seed: f32,
}

impl Wiggle2D {
    /// Creates a new 2D wiggle motion.
    pub fn new(center: Vec2, amplitude: Vec2, frequency: f32, seed: f32) -> Self {
        Self {
            center,
            amplitude,
            frequency,
            seed,
        }
    }
}

impl Motion<Vec2> for Wiggle2D {
    fn at(&self, t: f32) -> Vec2 {
        use rhizome_resin_noise::Noise2D;
        let nx = rhizome_resin_noise::Perlin2D::with_seed(self.seed as i32)
            .sample(t * self.frequency, 0.0);
        let ny = rhizome_resin_noise::Perlin2D::with_seed((self.seed + 100.0) as i32)
            .sample(t * self.frequency, 0.0);
        let normalized_x = nx * 2.0 - 1.0;
        let normalized_y = ny * 2.0 - 1.0;
        Vec2::new(
            self.center.x + self.amplitude.x * normalized_x,
            self.center.y + self.amplitude.y * normalized_y,
        )
    }
}

impl_field_for_motion!(Wiggle, f32);
impl_field_for_motion!(Wiggle2D, Vec2);

// ============================================================================
// Delay
// ============================================================================

/// Delays a motion by a given amount.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Delay<M> {
    /// The inner motion.
    pub motion: M,
    /// Delay amount in seconds.
    pub delay: f32,
}

impl<M> Delay<M> {
    /// Creates a delayed motion.
    pub fn new(motion: M, delay: f32) -> Self {
        Self { motion, delay }
    }
}

impl<T, M: Motion<T>> Motion<T> for Delay<M> {
    fn at(&self, t: f32) -> T {
        self.motion.at((t - self.delay).max(0.0))
    }
}

// ============================================================================
// TimeScale
// ============================================================================

/// Scales the time of a motion.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct TimeScale<M> {
    /// The inner motion.
    pub motion: M,
    /// Time scale factor (>1 = faster, <1 = slower).
    pub scale: f32,
}

impl<M> TimeScale<M> {
    /// Creates a time-scaled motion.
    pub fn new(motion: M, scale: f32) -> Self {
        Self { motion, scale }
    }
}

impl<T, M: Motion<T>> Motion<T> for TimeScale<M> {
    fn at(&self, t: f32) -> T {
        self.motion.at(t * self.scale)
    }
}

// ============================================================================
// Loop
// ============================================================================

/// Loops a motion over a given duration.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Loop<M> {
    /// The inner motion.
    pub motion: M,
    /// Duration of one loop cycle.
    pub duration: f32,
}

impl<M> Loop<M> {
    /// Creates a looping motion.
    pub fn new(motion: M, duration: f32) -> Self {
        Self { motion, duration }
    }
}

impl<T, M: Motion<T>> Motion<T> for Loop<M> {
    fn at(&self, t: f32) -> T {
        let looped_t = if self.duration > 0.0 {
            t % self.duration
        } else {
            0.0
        };
        self.motion.at(looped_t)
    }
}

// ============================================================================
// PingPong
// ============================================================================

/// Ping-pongs a motion (forward then backward).
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct PingPong<M> {
    /// The inner motion.
    pub motion: M,
    /// Duration of one direction (full cycle = 2x this).
    pub duration: f32,
}

impl<M> PingPong<M> {
    /// Creates a ping-pong motion.
    pub fn new(motion: M, duration: f32) -> Self {
        Self { motion, duration }
    }
}

impl<T, M: Motion<T>> Motion<T> for PingPong<M> {
    fn at(&self, t: f32) -> T {
        let cycle = self.duration * 2.0;
        let t_in_cycle = if cycle > 0.0 { t % cycle } else { 0.0 };
        let actual_t = if t_in_cycle < self.duration {
            t_in_cycle
        } else {
            cycle - t_in_cycle
        };
        self.motion.at(actual_t)
    }
}

// ============================================================================
// Dew expression integration
// ============================================================================

/// Dew expression functions for motion.
///
/// Enable with the `dew` feature, then register with `register_motion_functions`.
///
/// # Available functions
///
/// - `spring(from, to, stiffness, damping, t)` - critically damped spring
/// - `spring_critical(from, to, stiffness, t)` - critically damped (auto damping)
/// - `oscillate(center, amplitude, frequency, phase, t)` - sine wave
/// - `wiggle(center, amplitude, frequency, seed, t)` - noise-based
/// - `lerp_motion(from, to, duration, t)` - linear interpolation
#[cfg(feature = "dew")]
pub mod dew_functions {
    use rhizome_dew_scalar::ScalarFn;

    /// spring(from, to, stiffness, damping, t)
    pub struct SpringFn;
    impl ScalarFn<f32> for SpringFn {
        fn name(&self) -> &str {
            "spring"
        }
        fn arg_count(&self) -> usize {
            5
        }
        fn call(&self, args: &[f32]) -> f32 {
            let [from, to, stiffness, damping, t] = args else {
                return 0.0;
            };
            super::spring_value(*from, *to, *stiffness, *damping, *t)
        }
    }

    /// spring_critical(from, to, stiffness, t)
    pub struct SpringCriticalFn;
    impl ScalarFn<f32> for SpringCriticalFn {
        fn name(&self) -> &str {
            "spring_critical"
        }
        fn arg_count(&self) -> usize {
            4
        }
        fn call(&self, args: &[f32]) -> f32 {
            let [from, to, stiffness, t] = args else {
                return 0.0;
            };
            let damping = 2.0 * stiffness.sqrt();
            super::spring_value(*from, *to, *stiffness, damping, *t)
        }
    }

    /// oscillate(center, amplitude, frequency, phase, t)
    pub struct OscillateFn;
    impl ScalarFn<f32> for OscillateFn {
        fn name(&self) -> &str {
            "oscillate"
        }
        fn arg_count(&self) -> usize {
            5
        }
        fn call(&self, args: &[f32]) -> f32 {
            let [center, amplitude, frequency, phase, t] = args else {
                return 0.0;
            };
            let angle = std::f32::consts::TAU * frequency * t + phase;
            center + amplitude * angle.sin()
        }
    }

    /// wiggle(center, amplitude, frequency, seed, t)
    pub struct WiggleFn;
    impl ScalarFn<f32> for WiggleFn {
        fn name(&self) -> &str {
            "wiggle"
        }
        fn arg_count(&self) -> usize {
            5
        }
        fn call(&self, args: &[f32]) -> f32 {
            let [center, amplitude, frequency, seed, t] = args else {
                return 0.0;
            };
            let noise = rhizome_resin_noise::perlin2(t * frequency, *seed);
            let normalized = noise * 2.0 - 1.0;
            center + amplitude * normalized
        }
    }

    /// lerp_motion(from, to, duration, t) - clamped to duration
    pub struct LerpMotionFn;
    impl ScalarFn<f32> for LerpMotionFn {
        fn name(&self) -> &str {
            "lerp_motion"
        }
        fn arg_count(&self) -> usize {
            4
        }
        fn call(&self, args: &[f32]) -> f32 {
            let [from, to, duration, t] = args else {
                return 0.0;
            };
            let progress = (t / duration).clamp(0.0, 1.0);
            from + (to - from) * progress
        }
    }

    /// Registers all motion functions with a dew FunctionRegistry.
    pub fn register_motion_functions(registry: &mut rhizome_dew_scalar::FunctionRegistry<f32>) {
        registry.register(SpringFn);
        registry.register(SpringCriticalFn);
        registry.register(OscillateFn);
        registry.register(WiggleFn);
        registry.register(LerpMotionFn);
    }
}

#[cfg(feature = "dew")]
pub use dew_functions::register_motion_functions;

// ============================================================================
// MotionExpr - Typed AST for motion expressions
// ============================================================================

use std::collections::HashMap;

/// A typed expression AST for motion (time → value).
///
/// Unlike raw dew AST, this has typed variants for each motion function,
/// enabling UI introspection, JSON serialization, and GPU compilation.
///
/// # Example
///
/// ```
/// use rhizome_resin_motion_fn::MotionExpr;
///
/// // Build expression: spring(0, 100, 300, 15) + wiggle(0, 5, 2, 42)
/// let expr = MotionExpr::Add(
///     Box::new(MotionExpr::Spring {
///         from: Box::new(MotionExpr::Constant(0.0)),
///         to: Box::new(MotionExpr::Constant(100.0)),
///         stiffness: 300.0,
///         damping: 15.0,
///     }),
///     Box::new(MotionExpr::Wiggle {
///         center: 0.0,
///         amplitude: 5.0,
///         frequency: 2.0,
///         seed: 42.0,
///     }),
/// );
///
/// // Evaluate at t=0.5
/// let value = expr.eval(0.5, &Default::default());
/// ```
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum MotionExpr {
    // === Literals ===
    /// Constant value.
    Constant(f32),

    /// Variable reference (user-defined, not `t`).
    Var(String),

    // === Binary operations ===
    /// Addition.
    Add(Box<MotionExpr>, Box<MotionExpr>),

    /// Subtraction.
    Sub(Box<MotionExpr>, Box<MotionExpr>),

    /// Multiplication.
    Mul(Box<MotionExpr>, Box<MotionExpr>),

    /// Division.
    Div(Box<MotionExpr>, Box<MotionExpr>),

    /// Modulo.
    Mod(Box<MotionExpr>, Box<MotionExpr>),

    /// Power.
    Pow(Box<MotionExpr>, Box<MotionExpr>),

    // === Unary operations ===
    /// Negation.
    Neg(Box<MotionExpr>),

    // === Motion functions ===
    /// Spring motion toward target.
    Spring {
        from: Box<MotionExpr>,
        to: Box<MotionExpr>,
        stiffness: f32,
        damping: f32,
    },

    /// Critically damped spring (auto-calculated damping).
    SpringCritical {
        from: Box<MotionExpr>,
        to: Box<MotionExpr>,
        stiffness: f32,
    },

    /// Sine wave oscillation.
    Oscillate {
        center: Box<MotionExpr>,
        amplitude: Box<MotionExpr>,
        frequency: f32,
        phase: f32,
    },

    /// Noise-based random motion.
    Wiggle {
        center: f32,
        amplitude: f32,
        frequency: f32,
        seed: f32,
    },

    /// Linear interpolation (clamped to duration).
    LerpMotion {
        from: Box<MotionExpr>,
        to: Box<MotionExpr>,
        duration: f32,
    },

    /// Eased interpolation.
    Eased {
        from: Box<MotionExpr>,
        to: Box<MotionExpr>,
        duration: f32,
        easing: EasingType,
    },

    // === Math functions ===
    /// Sine.
    Sin(Box<MotionExpr>),

    /// Cosine.
    Cos(Box<MotionExpr>),

    /// Absolute value.
    Abs(Box<MotionExpr>),

    /// Floor.
    Floor(Box<MotionExpr>),

    /// Ceiling.
    Ceil(Box<MotionExpr>),

    /// Square root.
    Sqrt(Box<MotionExpr>),

    /// Minimum of two values.
    Min(Box<MotionExpr>, Box<MotionExpr>),

    /// Maximum of two values.
    Max(Box<MotionExpr>, Box<MotionExpr>),

    /// Clamp value to range.
    Clamp {
        value: Box<MotionExpr>,
        min: Box<MotionExpr>,
        max: Box<MotionExpr>,
    },

    /// Linear interpolation (unclamped).
    Lerp {
        a: Box<MotionExpr>,
        b: Box<MotionExpr>,
        t_expr: Box<MotionExpr>,
    },

    // === Conditionals ===
    /// If-then-else.
    IfThenElse {
        condition: Box<MotionExpr>,
        then_expr: Box<MotionExpr>,
        else_expr: Box<MotionExpr>,
    },

    /// Greater than (returns 1.0 or 0.0).
    Gt(Box<MotionExpr>, Box<MotionExpr>),

    /// Less than (returns 1.0 or 0.0).
    Lt(Box<MotionExpr>, Box<MotionExpr>),
}

impl MotionExpr {
    /// Evaluate the expression at time `t` with variable bindings.
    pub fn eval(&self, t: f32, vars: &HashMap<String, f32>) -> f32 {
        match self {
            // Literals
            Self::Constant(v) => *v,
            Self::Var(name) => *vars.get(name).unwrap_or(&0.0),

            // Binary ops
            Self::Add(a, b) => a.eval(t, vars) + b.eval(t, vars),
            Self::Sub(a, b) => a.eval(t, vars) - b.eval(t, vars),
            Self::Mul(a, b) => a.eval(t, vars) * b.eval(t, vars),
            Self::Div(a, b) => a.eval(t, vars) / b.eval(t, vars),
            Self::Mod(a, b) => a.eval(t, vars) % b.eval(t, vars),
            Self::Pow(a, b) => a.eval(t, vars).powf(b.eval(t, vars)),

            // Unary ops
            Self::Neg(a) => -a.eval(t, vars),

            // Motion functions
            Self::Spring {
                from,
                to,
                stiffness,
                damping,
            } => spring_value(
                from.eval(t, vars),
                to.eval(t, vars),
                *stiffness,
                *damping,
                t,
            ),

            Self::SpringCritical {
                from,
                to,
                stiffness,
            } => {
                let damping = 2.0 * stiffness.sqrt();
                spring_value(from.eval(t, vars), to.eval(t, vars), *stiffness, damping, t)
            }

            Self::Oscillate {
                center,
                amplitude,
                frequency,
                phase,
            } => {
                let angle = std::f32::consts::TAU * frequency * t + phase;
                center.eval(t, vars) + amplitude.eval(t, vars) * angle.sin()
            }

            Self::Wiggle {
                center,
                amplitude,
                frequency,
                seed,
            } => {
                use rhizome_resin_noise::Noise2D;
                let noise = rhizome_resin_noise::Perlin2D::with_seed(*seed as i32)
                    .sample(t * frequency, 0.0);
                let normalized = noise * 2.0 - 1.0;
                center + amplitude * normalized
            }

            Self::LerpMotion { from, to, duration } => {
                let progress = (t / duration).clamp(0.0, 1.0);
                let a = from.eval(t, vars);
                let b = to.eval(t, vars);
                a + (b - a) * progress
            }

            Self::Eased {
                from,
                to,
                duration,
                easing,
            } => {
                let linear = (t / duration).clamp(0.0, 1.0);
                let eased = easing.apply(linear);
                let a = from.eval(t, vars);
                let b = to.eval(t, vars);
                a + (b - a) * eased
            }

            // Math functions
            Self::Sin(a) => a.eval(t, vars).sin(),
            Self::Cos(a) => a.eval(t, vars).cos(),
            Self::Abs(a) => a.eval(t, vars).abs(),
            Self::Floor(a) => a.eval(t, vars).floor(),
            Self::Ceil(a) => a.eval(t, vars).ceil(),
            Self::Sqrt(a) => a.eval(t, vars).sqrt(),
            Self::Min(a, b) => a.eval(t, vars).min(b.eval(t, vars)),
            Self::Max(a, b) => a.eval(t, vars).max(b.eval(t, vars)),
            Self::Clamp { value, min, max } => value
                .eval(t, vars)
                .clamp(min.eval(t, vars), max.eval(t, vars)),
            Self::Lerp { a, b, t_expr } => {
                let a_val = a.eval(t, vars);
                let b_val = b.eval(t, vars);
                let t_val = t_expr.eval(t, vars);
                a_val + (b_val - a_val) * t_val
            }

            // Conditionals
            Self::IfThenElse {
                condition,
                then_expr,
                else_expr,
            } => {
                if condition.eval(t, vars) > 0.5 {
                    then_expr.eval(t, vars)
                } else {
                    else_expr.eval(t, vars)
                }
            }
            Self::Gt(a, b) => {
                if a.eval(t, vars) > b.eval(t, vars) {
                    1.0
                } else {
                    0.0
                }
            }
            Self::Lt(a, b) => {
                if a.eval(t, vars) < b.eval(t, vars) {
                    1.0
                } else {
                    0.0
                }
            }
        }
    }

    /// Returns the free variables in this expression (excluding `t`).
    pub fn free_vars(&self) -> std::collections::HashSet<String> {
        let mut vars = std::collections::HashSet::new();
        self.collect_vars(&mut vars);
        vars
    }

    fn collect_vars(&self, vars: &mut std::collections::HashSet<String>) {
        match self {
            Self::Var(name) => {
                vars.insert(name.clone());
            }
            Self::Constant(_) | Self::Wiggle { .. } => {}

            Self::Add(a, b)
            | Self::Sub(a, b)
            | Self::Mul(a, b)
            | Self::Div(a, b)
            | Self::Mod(a, b)
            | Self::Pow(a, b)
            | Self::Min(a, b)
            | Self::Max(a, b)
            | Self::Gt(a, b)
            | Self::Lt(a, b) => {
                a.collect_vars(vars);
                b.collect_vars(vars);
            }

            Self::Neg(a)
            | Self::Sin(a)
            | Self::Cos(a)
            | Self::Abs(a)
            | Self::Floor(a)
            | Self::Ceil(a)
            | Self::Sqrt(a) => {
                a.collect_vars(vars);
            }

            Self::Spring { from, to, .. } | Self::SpringCritical { from, to, .. } => {
                from.collect_vars(vars);
                to.collect_vars(vars);
            }

            Self::Oscillate {
                center, amplitude, ..
            } => {
                center.collect_vars(vars);
                amplitude.collect_vars(vars);
            }

            Self::LerpMotion { from, to, .. } | Self::Eased { from, to, .. } => {
                from.collect_vars(vars);
                to.collect_vars(vars);
            }

            Self::Clamp { value, min, max } => {
                value.collect_vars(vars);
                min.collect_vars(vars);
                max.collect_vars(vars);
            }

            Self::Lerp { a, b, t_expr } => {
                a.collect_vars(vars);
                b.collect_vars(vars);
                t_expr.collect_vars(vars);
            }

            Self::IfThenElse {
                condition,
                then_expr,
                else_expr,
            } => {
                condition.collect_vars(vars);
                then_expr.collect_vars(vars);
                else_expr.collect_vars(vars);
            }
        }
    }
}

// Conversion to/from dew AST
#[cfg(feature = "dew")]
impl MotionExpr {
    /// Convert to dew AST for compilation to WGSL/Cranelift.
    pub fn to_dew_ast(&self) -> rhizome_dew_core::Ast {
        use rhizome_dew_core::{Ast, BinOp, UnaryOp};

        match self {
            Self::Constant(v) => Ast::Num(*v),
            Self::Var(name) => Ast::Var(name.clone()),

            Self::Add(a, b) => Ast::BinOp(
                BinOp::Add,
                Box::new(a.to_dew_ast()),
                Box::new(b.to_dew_ast()),
            ),
            Self::Sub(a, b) => Ast::BinOp(
                BinOp::Sub,
                Box::new(a.to_dew_ast()),
                Box::new(b.to_dew_ast()),
            ),
            Self::Mul(a, b) => Ast::BinOp(
                BinOp::Mul,
                Box::new(a.to_dew_ast()),
                Box::new(b.to_dew_ast()),
            ),
            Self::Div(a, b) => Ast::BinOp(
                BinOp::Div,
                Box::new(a.to_dew_ast()),
                Box::new(b.to_dew_ast()),
            ),
            Self::Mod(a, b) => {
                // Modulo as function call since dew doesn't have Mod binary op
                Ast::Call("mod".into(), vec![a.to_dew_ast(), b.to_dew_ast()])
            }
            Self::Pow(a, b) => Ast::BinOp(
                BinOp::Pow,
                Box::new(a.to_dew_ast()),
                Box::new(b.to_dew_ast()),
            ),

            Self::Neg(a) => Ast::UnaryOp(UnaryOp::Neg, Box::new(a.to_dew_ast())),

            Self::Spring {
                from,
                to,
                stiffness,
                damping,
            } => Ast::Call(
                "spring".into(),
                vec![
                    from.to_dew_ast(),
                    to.to_dew_ast(),
                    Ast::Num(*stiffness),
                    Ast::Num(*damping),
                    Ast::Var("t".into()),
                ],
            ),

            Self::SpringCritical {
                from,
                to,
                stiffness,
            } => Ast::Call(
                "spring_critical".into(),
                vec![
                    from.to_dew_ast(),
                    to.to_dew_ast(),
                    Ast::Num(*stiffness),
                    Ast::Var("t".into()),
                ],
            ),

            Self::Oscillate {
                center,
                amplitude,
                frequency,
                phase,
            } => Ast::Call(
                "oscillate".into(),
                vec![
                    center.to_dew_ast(),
                    amplitude.to_dew_ast(),
                    Ast::Num(*frequency),
                    Ast::Num(*phase),
                    Ast::Var("t".into()),
                ],
            ),

            Self::Wiggle {
                center,
                amplitude,
                frequency,
                seed,
            } => Ast::Call(
                "wiggle".into(),
                vec![
                    Ast::Num(*center),
                    Ast::Num(*amplitude),
                    Ast::Num(*frequency),
                    Ast::Num(*seed),
                    Ast::Var("t".into()),
                ],
            ),

            Self::LerpMotion { from, to, duration } => Ast::Call(
                "lerp_motion".into(),
                vec![
                    from.to_dew_ast(),
                    to.to_dew_ast(),
                    Ast::Num(*duration),
                    Ast::Var("t".into()),
                ],
            ),

            Self::Eased {
                from,
                to,
                duration,
                easing,
            } => {
                // For now, convert easing to a string - backend needs to handle
                Ast::Call(
                    "eased".into(),
                    vec![
                        from.to_dew_ast(),
                        to.to_dew_ast(),
                        Ast::Num(*duration),
                        Ast::Var(format!("{:?}", easing)),
                        Ast::Var("t".into()),
                    ],
                )
            }

            Self::Sin(a) => Ast::Call("sin".into(), vec![a.to_dew_ast()]),
            Self::Cos(a) => Ast::Call("cos".into(), vec![a.to_dew_ast()]),
            Self::Abs(a) => Ast::Call("abs".into(), vec![a.to_dew_ast()]),
            Self::Floor(a) => Ast::Call("floor".into(), vec![a.to_dew_ast()]),
            Self::Ceil(a) => Ast::Call("ceil".into(), vec![a.to_dew_ast()]),
            Self::Sqrt(a) => Ast::Call("sqrt".into(), vec![a.to_dew_ast()]),

            Self::Min(a, b) => Ast::Call("min".into(), vec![a.to_dew_ast(), b.to_dew_ast()]),
            Self::Max(a, b) => Ast::Call("max".into(), vec![a.to_dew_ast(), b.to_dew_ast()]),

            Self::Clamp { value, min, max } => Ast::Call(
                "clamp".into(),
                vec![value.to_dew_ast(), min.to_dew_ast(), max.to_dew_ast()],
            ),

            Self::Lerp { a, b, t_expr } => Ast::Call(
                "lerp".into(),
                vec![a.to_dew_ast(), b.to_dew_ast(), t_expr.to_dew_ast()],
            ),

            Self::IfThenElse {
                condition,
                then_expr,
                else_expr,
            } => Ast::If(
                Box::new(condition.to_dew_ast()),
                Box::new(then_expr.to_dew_ast()),
                Box::new(else_expr.to_dew_ast()),
            ),

            Self::Gt(a, b) => Ast::Compare(
                rhizome_dew_core::CompareOp::Gt,
                Box::new(a.to_dew_ast()),
                Box::new(b.to_dew_ast()),
            ),
            Self::Lt(a, b) => Ast::Compare(
                rhizome_dew_core::CompareOp::Lt,
                Box::new(a.to_dew_ast()),
                Box::new(b.to_dew_ast()),
            ),
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constant() {
        let motion = Constant::new(42.0);
        assert_eq!(motion.at(0.0), 42.0);
        assert_eq!(motion.at(1.0), 42.0);
        assert_eq!(motion.at(100.0), 42.0);
    }

    #[test]
    fn test_lerp() {
        let motion = Lerp::new(0.0, 100.0, 1.0);
        assert_eq!(motion.at(0.0), 0.0);
        assert_eq!(motion.at(0.5), 50.0);
        assert_eq!(motion.at(1.0), 100.0);
        // Clamped
        assert_eq!(motion.at(2.0), 100.0);
    }

    #[test]
    fn test_eased() {
        let motion = Eased::new(0.0, 100.0, 1.0, EasingType::QuadInOut);
        assert_eq!(motion.at(0.0), 0.0);
        assert_eq!(motion.at(1.0), 100.0);
        // Mid-point should be close to 50 for quad_in_out
        let mid = motion.at(0.5);
        assert!((mid - 50.0).abs() < 1.0);
    }

    #[test]
    fn test_spring_critical() {
        let motion = Spring::critical(0.0, 100.0, 300.0);
        assert_eq!(motion.at(0.0), 0.0);
        // Should approach target over time
        let late = motion.at(1.0);
        assert!(late > 99.0, "Spring should nearly reach target: {}", late);
    }

    #[test]
    fn test_spring_bouncy() {
        let motion = Spring::bouncy(0.0, 100.0, 300.0);
        // Bouncy spring might overshoot
        let values: Vec<f32> = (0..20).map(|i| motion.at(i as f32 * 0.05)).collect();
        // Should eventually settle near target
        let last = *values.last().unwrap();
        assert!(
            (last - 100.0).abs() < 10.0,
            "Spring should settle near target: {}",
            last
        );
    }

    #[test]
    fn test_oscillate() {
        let motion = Oscillate::new(0.0, 1.0, 1.0, 0.0);
        // At t=0, sin(0) = 0
        assert!((motion.at(0.0) - 0.0).abs() < 0.01);
        // At t=0.25 (1/4 period), sin(π/2) = 1
        assert!((motion.at(0.25) - 1.0).abs() < 0.01);
        // At t=0.5 (1/2 period), sin(π) = 0
        assert!((motion.at(0.5) - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_wiggle() {
        let motion = Wiggle::new(0.0, 1.0, 1.0, 42.0);
        // Wiggle should produce values in range
        for i in 0..10 {
            let v = motion.at(i as f32 * 0.1);
            assert!(v >= -1.5 && v <= 1.5, "Wiggle out of expected range: {}", v);
        }
    }

    #[test]
    fn test_delay() {
        let motion = Delay::new(Lerp::new(0.0, 100.0, 1.0), 0.5);
        assert_eq!(motion.at(0.0), 0.0); // Before delay
        assert_eq!(motion.at(0.5), 0.0); // At delay start
        assert_eq!(motion.at(1.0), 50.0); // 0.5s into motion
        assert_eq!(motion.at(1.5), 100.0); // Motion complete
    }

    #[test]
    fn test_loop() {
        let motion = Loop::new(Lerp::new(0.0, 100.0, 1.0), 1.0);
        assert_eq!(motion.at(0.0), 0.0);
        assert_eq!(motion.at(0.5), 50.0);
        assert_eq!(motion.at(1.0), 0.0); // Loops back
        assert_eq!(motion.at(1.5), 50.0);
    }

    #[test]
    fn test_ping_pong() {
        let motion = PingPong::new(Lerp::new(0.0, 100.0, 1.0), 1.0);
        assert_eq!(motion.at(0.0), 0.0);
        assert_eq!(motion.at(0.5), 50.0);
        assert_eq!(motion.at(1.0), 100.0);
        assert_eq!(motion.at(1.5), 50.0); // Going back
        assert_eq!(motion.at(2.0), 0.0); // Back to start
    }

    #[test]
    fn test_time_scale() {
        let motion = TimeScale::new(Lerp::new(0.0, 100.0, 1.0), 2.0);
        // 2x speed means 0.5s of real time = 1.0s of motion time
        assert_eq!(motion.at(0.0), 0.0);
        assert_eq!(motion.at(0.5), 100.0);
    }

    #[test]
    fn test_vec2_lerp() {
        let motion = Lerp::new(Vec2::ZERO, Vec2::new(100.0, 200.0), 1.0);
        let mid = motion.at(0.5);
        assert_eq!(mid, Vec2::new(50.0, 100.0));
    }

    #[test]
    fn test_vec3_spring() {
        let motion = Spring::critical(Vec3::ZERO, Vec3::new(100.0, 100.0, 100.0), 300.0);
        let late = motion.at(1.0);
        assert!(late.x > 99.0);
        assert!(late.y > 99.0);
        assert!(late.z > 99.0);
    }

    // MotionExpr tests

    #[test]
    fn test_motion_expr_constant() {
        let expr = MotionExpr::Constant(42.0);
        assert_eq!(expr.eval(0.0, &Default::default()), 42.0);
        assert_eq!(expr.eval(1.0, &Default::default()), 42.0);
    }

    #[test]
    fn test_motion_expr_var() {
        let expr = MotionExpr::Var("x".into());
        let mut vars = HashMap::new();
        vars.insert("x".into(), 100.0);
        assert_eq!(expr.eval(0.0, &vars), 100.0);
        // Missing var returns 0
        assert_eq!(expr.eval(0.0, &Default::default()), 0.0);
    }

    #[test]
    fn test_motion_expr_arithmetic() {
        let expr = MotionExpr::Add(
            Box::new(MotionExpr::Constant(10.0)),
            Box::new(MotionExpr::Mul(
                Box::new(MotionExpr::Constant(5.0)),
                Box::new(MotionExpr::Constant(3.0)),
            )),
        );
        // 10 + (5 * 3) = 25
        assert_eq!(expr.eval(0.0, &Default::default()), 25.0);
    }

    #[test]
    fn test_motion_expr_spring() {
        let expr = MotionExpr::Spring {
            from: Box::new(MotionExpr::Constant(0.0)),
            to: Box::new(MotionExpr::Constant(100.0)),
            stiffness: 300.0,
            damping: 2.0 * 300.0_f32.sqrt(), // Critical damping
        };
        assert_eq!(expr.eval(0.0, &Default::default()), 0.0);
        let late = expr.eval(1.0, &Default::default());
        assert!(late > 99.0, "Spring should reach near target: {}", late);
    }

    #[test]
    fn test_motion_expr_oscillate() {
        let expr = MotionExpr::Oscillate {
            center: Box::new(MotionExpr::Constant(0.0)),
            amplitude: Box::new(MotionExpr::Constant(1.0)),
            frequency: 1.0,
            phase: 0.0,
        };
        // At t=0, sin(0) = 0
        assert!((expr.eval(0.0, &Default::default()) - 0.0).abs() < 0.01);
        // At t=0.25 (1/4 period), sin(π/2) = 1
        assert!((expr.eval(0.25, &Default::default()) - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_motion_expr_free_vars() {
        let expr = MotionExpr::Add(
            Box::new(MotionExpr::Var("x".into())),
            Box::new(MotionExpr::Mul(
                Box::new(MotionExpr::Var("y".into())),
                Box::new(MotionExpr::Constant(2.0)),
            )),
        );
        let vars = expr.free_vars();
        assert!(vars.contains("x"));
        assert!(vars.contains("y"));
        assert_eq!(vars.len(), 2);
    }

    #[test]
    fn test_motion_expr_conditionals() {
        // If t > 0.5 then 100 else 0
        let expr = MotionExpr::IfThenElse {
            condition: Box::new(MotionExpr::Gt(
                Box::new(MotionExpr::Var("t_val".into())),
                Box::new(MotionExpr::Constant(0.5)),
            )),
            then_expr: Box::new(MotionExpr::Constant(100.0)),
            else_expr: Box::new(MotionExpr::Constant(0.0)),
        };
        let mut vars = HashMap::new();
        vars.insert("t_val".into(), 0.3);
        assert_eq!(expr.eval(0.0, &vars), 0.0);
        vars.insert("t_val".into(), 0.7);
        assert_eq!(expr.eval(0.0, &vars), 100.0);
    }

    #[test]
    fn test_motion_expr_math_functions() {
        let sin_expr = MotionExpr::Sin(Box::new(MotionExpr::Constant(0.0)));
        assert!((sin_expr.eval(0.0, &Default::default()) - 0.0).abs() < 0.01);

        let cos_expr = MotionExpr::Cos(Box::new(MotionExpr::Constant(0.0)));
        assert!((cos_expr.eval(0.0, &Default::default()) - 1.0).abs() < 0.01);

        let abs_expr = MotionExpr::Abs(Box::new(MotionExpr::Constant(-5.0)));
        assert_eq!(abs_expr.eval(0.0, &Default::default()), 5.0);

        let clamp_expr = MotionExpr::Clamp {
            value: Box::new(MotionExpr::Constant(150.0)),
            min: Box::new(MotionExpr::Constant(0.0)),
            max: Box::new(MotionExpr::Constant(100.0)),
        };
        assert_eq!(clamp_expr.eval(0.0, &Default::default()), 100.0);
    }

    #[cfg(feature = "dew")]
    #[test]
    fn test_motion_expr_to_dew_ast() {
        let expr = MotionExpr::Add(
            Box::new(MotionExpr::Constant(1.0)),
            Box::new(MotionExpr::Constant(2.0)),
        );
        let ast = expr.to_dew_ast();
        // Just verify it doesn't panic and produces something
        assert!(matches!(ast, rhizome_dew_core::Ast::BinOp(..)));
    }

    #[cfg(feature = "dew")]
    #[test]
    fn test_motion_expr_spring_to_dew() {
        let expr = MotionExpr::Spring {
            from: Box::new(MotionExpr::Constant(0.0)),
            to: Box::new(MotionExpr::Constant(100.0)),
            stiffness: 300.0,
            damping: 15.0,
        };
        let ast = expr.to_dew_ast();
        // Should be Call("spring", [...])
        if let rhizome_dew_core::Ast::Call(name, args) = ast {
            assert_eq!(name, "spring");
            assert_eq!(args.len(), 5); // from, to, stiffness, damping, t
        } else {
            panic!("Expected Call AST");
        }
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_motion_expr_serde_roundtrip() {
        let expr = MotionExpr::Spring {
            from: Box::new(MotionExpr::Constant(0.0)),
            to: Box::new(MotionExpr::Var("target".into())),
            stiffness: 300.0,
            damping: 15.0,
        };
        let json = serde_json::to_string(&expr).unwrap();
        let parsed: MotionExpr = serde_json::from_str(&json).unwrap();
        assert_eq!(expr, parsed);
    }
}
