//! Easing functions for animation and interpolation.
//!
//! Provides standard easing curves (Robert Penner style) and utilities
//! for smooth animation transitions.
//!
//! # Lerp Trait
//!
//! The [`Lerp`] trait provides a unified interface for linear interpolation
//! across all resin crates. Implementations are provided for common types.
//!
//! ```
//! use rhizome_resin_easing::Lerp;
//! use glam::Vec3;
//!
//! let a = Vec3::ZERO;
//! let b = Vec3::ONE;
//! let mid = a.lerp_to(&b, 0.5);
//! assert!((mid - Vec3::splat(0.5)).length() < 0.001);
//! ```

use std::f32::consts::PI;

use glam::{Quat, Vec2, Vec3, Vec4};

// ============================================================================
// Lerp Trait
// ============================================================================

/// Trait for types that support linear interpolation.
///
/// This is the canonical interpolation trait for resin. Implement this
/// for custom types to enable animation, blending, and easing support.
///
/// # Example
///
/// ```
/// use rhizome_resin_easing::Lerp;
///
/// struct MyColor { r: f32, g: f32, b: f32 }
///
/// impl Lerp for MyColor {
///     fn lerp_to(&self, other: &Self, t: f32) -> Self {
///         MyColor {
///             r: self.r + (other.r - self.r) * t,
///             g: self.g + (other.g - self.g) * t,
///             b: self.b + (other.b - self.b) * t,
///         }
///     }
/// }
/// ```
pub trait Lerp {
    /// Linearly interpolates from `self` to `other` by factor `t`.
    ///
    /// - `t = 0.0` returns `self`
    /// - `t = 1.0` returns `other`
    /// - Values outside `[0, 1]` extrapolate
    fn lerp_to(&self, other: &Self, t: f32) -> Self;
}

impl Lerp for f32 {
    #[inline]
    fn lerp_to(&self, other: &Self, t: f32) -> Self {
        self + (other - self) * t
    }
}

impl Lerp for f64 {
    #[inline]
    fn lerp_to(&self, other: &Self, t: f32) -> Self {
        self + (other - self) * t as f64
    }
}

impl Lerp for Vec2 {
    #[inline]
    fn lerp_to(&self, other: &Self, t: f32) -> Self {
        Vec2::lerp(*self, *other, t)
    }
}

impl Lerp for Vec3 {
    #[inline]
    fn lerp_to(&self, other: &Self, t: f32) -> Self {
        Vec3::lerp(*self, *other, t)
    }
}

impl Lerp for Vec4 {
    #[inline]
    fn lerp_to(&self, other: &Self, t: f32) -> Self {
        Vec4::lerp(*self, *other, t)
    }
}

impl Lerp for Quat {
    /// Uses spherical linear interpolation (slerp) for quaternions.
    #[inline]
    fn lerp_to(&self, other: &Self, t: f32) -> Self {
        self.slerp(*other, t)
    }
}

impl<T: Lerp, const N: usize> Lerp for [T; N] {
    fn lerp_to(&self, other: &Self, t: f32) -> Self {
        std::array::from_fn(|i| self[i].lerp_to(&other[i], t))
    }
}

/// Easing function type.
pub type EaseFn = fn(f32) -> f32;

/// Standard easing function types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Easing {
    /// Linear interpolation (no easing).
    #[default]
    Linear,
    /// Quadratic ease in.
    QuadIn,
    /// Quadratic ease out.
    QuadOut,
    /// Quadratic ease in-out.
    QuadInOut,
    /// Cubic ease in.
    CubicIn,
    /// Cubic ease out.
    CubicOut,
    /// Cubic ease in-out.
    CubicInOut,
    /// Quartic ease in.
    QuartIn,
    /// Quartic ease out.
    QuartOut,
    /// Quartic ease in-out.
    QuartInOut,
    /// Quintic ease in.
    QuintIn,
    /// Quintic ease out.
    QuintOut,
    /// Quintic ease in-out.
    QuintInOut,
    /// Sine ease in.
    SineIn,
    /// Sine ease out.
    SineOut,
    /// Sine ease in-out.
    SineInOut,
    /// Exponential ease in.
    ExpoIn,
    /// Exponential ease out.
    ExpoOut,
    /// Exponential ease in-out.
    ExpoInOut,
    /// Circular ease in.
    CircIn,
    /// Circular ease out.
    CircOut,
    /// Circular ease in-out.
    CircInOut,
    /// Back ease in (overshoots start).
    BackIn,
    /// Back ease out (overshoots end).
    BackOut,
    /// Back ease in-out.
    BackInOut,
    /// Elastic ease in.
    ElasticIn,
    /// Elastic ease out.
    ElasticOut,
    /// Elastic ease in-out.
    ElasticInOut,
    /// Bounce ease in.
    BounceIn,
    /// Bounce ease out.
    BounceOut,
    /// Bounce ease in-out.
    BounceInOut,
}

impl Easing {
    /// Evaluates the easing function at t (0-1).
    pub fn ease(self, t: f32) -> f32 {
        match self {
            Easing::Linear => linear(t),
            Easing::QuadIn => quad_in(t),
            Easing::QuadOut => quad_out(t),
            Easing::QuadInOut => quad_in_out(t),
            Easing::CubicIn => cubic_in(t),
            Easing::CubicOut => cubic_out(t),
            Easing::CubicInOut => cubic_in_out(t),
            Easing::QuartIn => quart_in(t),
            Easing::QuartOut => quart_out(t),
            Easing::QuartInOut => quart_in_out(t),
            Easing::QuintIn => quint_in(t),
            Easing::QuintOut => quint_out(t),
            Easing::QuintInOut => quint_in_out(t),
            Easing::SineIn => sine_in(t),
            Easing::SineOut => sine_out(t),
            Easing::SineInOut => sine_in_out(t),
            Easing::ExpoIn => expo_in(t),
            Easing::ExpoOut => expo_out(t),
            Easing::ExpoInOut => expo_in_out(t),
            Easing::CircIn => circ_in(t),
            Easing::CircOut => circ_out(t),
            Easing::CircInOut => circ_in_out(t),
            Easing::BackIn => back_in(t),
            Easing::BackOut => back_out(t),
            Easing::BackInOut => back_in_out(t),
            Easing::ElasticIn => elastic_in(t),
            Easing::ElasticOut => elastic_out(t),
            Easing::ElasticInOut => elastic_in_out(t),
            Easing::BounceIn => bounce_in(t),
            Easing::BounceOut => bounce_out(t),
            Easing::BounceInOut => bounce_in_out(t),
        }
    }

    /// Returns the corresponding function pointer.
    pub fn as_fn(self) -> EaseFn {
        match self {
            Easing::Linear => linear,
            Easing::QuadIn => quad_in,
            Easing::QuadOut => quad_out,
            Easing::QuadInOut => quad_in_out,
            Easing::CubicIn => cubic_in,
            Easing::CubicOut => cubic_out,
            Easing::CubicInOut => cubic_in_out,
            Easing::QuartIn => quart_in,
            Easing::QuartOut => quart_out,
            Easing::QuartInOut => quart_in_out,
            Easing::QuintIn => quint_in,
            Easing::QuintOut => quint_out,
            Easing::QuintInOut => quint_in_out,
            Easing::SineIn => sine_in,
            Easing::SineOut => sine_out,
            Easing::SineInOut => sine_in_out,
            Easing::ExpoIn => expo_in,
            Easing::ExpoOut => expo_out,
            Easing::ExpoInOut => expo_in_out,
            Easing::CircIn => circ_in,
            Easing::CircOut => circ_out,
            Easing::CircInOut => circ_in_out,
            Easing::BackIn => back_in,
            Easing::BackOut => back_out,
            Easing::BackInOut => back_in_out,
            Easing::ElasticIn => elastic_in,
            Easing::ElasticOut => elastic_out,
            Easing::ElasticInOut => elastic_in_out,
            Easing::BounceIn => bounce_in,
            Easing::BounceOut => bounce_out,
            Easing::BounceInOut => bounce_in_out,
        }
    }
}

// ============================================================================
// Linear
// ============================================================================

/// Linear interpolation (no easing).
#[inline]
pub fn linear(t: f32) -> f32 {
    t
}

// ============================================================================
// Quadratic
// ============================================================================

/// Quadratic ease in.
#[inline]
pub fn quad_in(t: f32) -> f32 {
    t * t
}

/// Quadratic ease out.
#[inline]
pub fn quad_out(t: f32) -> f32 {
    1.0 - (1.0 - t) * (1.0 - t)
}

/// Quadratic ease in-out.
#[inline]
pub fn quad_in_out(t: f32) -> f32 {
    if t < 0.5 {
        2.0 * t * t
    } else {
        1.0 - (-2.0 * t + 2.0).powi(2) / 2.0
    }
}

// ============================================================================
// Cubic
// ============================================================================

/// Cubic ease in.
#[inline]
pub fn cubic_in(t: f32) -> f32 {
    t * t * t
}

/// Cubic ease out.
#[inline]
pub fn cubic_out(t: f32) -> f32 {
    1.0 - (1.0 - t).powi(3)
}

/// Cubic ease in-out.
#[inline]
pub fn cubic_in_out(t: f32) -> f32 {
    if t < 0.5 {
        4.0 * t * t * t
    } else {
        1.0 - (-2.0 * t + 2.0).powi(3) / 2.0
    }
}

// ============================================================================
// Quartic
// ============================================================================

/// Quartic ease in.
#[inline]
pub fn quart_in(t: f32) -> f32 {
    t * t * t * t
}

/// Quartic ease out.
#[inline]
pub fn quart_out(t: f32) -> f32 {
    1.0 - (1.0 - t).powi(4)
}

/// Quartic ease in-out.
#[inline]
pub fn quart_in_out(t: f32) -> f32 {
    if t < 0.5 {
        8.0 * t * t * t * t
    } else {
        1.0 - (-2.0 * t + 2.0).powi(4) / 2.0
    }
}

// ============================================================================
// Quintic
// ============================================================================

/// Quintic ease in.
#[inline]
pub fn quint_in(t: f32) -> f32 {
    t * t * t * t * t
}

/// Quintic ease out.
#[inline]
pub fn quint_out(t: f32) -> f32 {
    1.0 - (1.0 - t).powi(5)
}

/// Quintic ease in-out.
#[inline]
pub fn quint_in_out(t: f32) -> f32 {
    if t < 0.5 {
        16.0 * t * t * t * t * t
    } else {
        1.0 - (-2.0 * t + 2.0).powi(5) / 2.0
    }
}

// ============================================================================
// Sine
// ============================================================================

/// Sine ease in.
#[inline]
pub fn sine_in(t: f32) -> f32 {
    1.0 - (t * PI / 2.0).cos()
}

/// Sine ease out.
#[inline]
pub fn sine_out(t: f32) -> f32 {
    (t * PI / 2.0).sin()
}

/// Sine ease in-out.
#[inline]
pub fn sine_in_out(t: f32) -> f32 {
    -(t * PI).cos() / 2.0 + 0.5
}

// ============================================================================
// Exponential
// ============================================================================

/// Exponential ease in.
#[inline]
pub fn expo_in(t: f32) -> f32 {
    if t == 0.0 {
        0.0
    } else {
        2.0f32.powf(10.0 * t - 10.0)
    }
}

/// Exponential ease out.
#[inline]
pub fn expo_out(t: f32) -> f32 {
    if t == 1.0 {
        1.0
    } else {
        1.0 - 2.0f32.powf(-10.0 * t)
    }
}

/// Exponential ease in-out.
#[inline]
pub fn expo_in_out(t: f32) -> f32 {
    if t == 0.0 {
        0.0
    } else if t == 1.0 {
        1.0
    } else if t < 0.5 {
        2.0f32.powf(20.0 * t - 10.0) / 2.0
    } else {
        (2.0 - 2.0f32.powf(-20.0 * t + 10.0)) / 2.0
    }
}

// ============================================================================
// Circular
// ============================================================================

/// Circular ease in.
#[inline]
pub fn circ_in(t: f32) -> f32 {
    1.0 - (1.0 - t * t).sqrt()
}

/// Circular ease out.
#[inline]
pub fn circ_out(t: f32) -> f32 {
    (1.0 - (t - 1.0).powi(2)).sqrt()
}

/// Circular ease in-out.
#[inline]
pub fn circ_in_out(t: f32) -> f32 {
    if t < 0.5 {
        (1.0 - (1.0 - (2.0 * t).powi(2)).sqrt()) / 2.0
    } else {
        ((1.0 - (-2.0 * t + 2.0).powi(2)).sqrt() + 1.0) / 2.0
    }
}

// ============================================================================
// Back (overshoot)
// ============================================================================

const BACK_C1: f32 = 1.70158;
const BACK_C2: f32 = BACK_C1 * 1.525;
const BACK_C3: f32 = BACK_C1 + 1.0;

/// Back ease in (overshoots backwards at start).
#[inline]
pub fn back_in(t: f32) -> f32 {
    BACK_C3 * t * t * t - BACK_C1 * t * t
}

/// Back ease out (overshoots at end).
#[inline]
pub fn back_out(t: f32) -> f32 {
    1.0 + BACK_C3 * (t - 1.0).powi(3) + BACK_C1 * (t - 1.0).powi(2)
}

/// Back ease in-out.
#[inline]
pub fn back_in_out(t: f32) -> f32 {
    if t < 0.5 {
        ((2.0 * t).powi(2) * ((BACK_C2 + 1.0) * 2.0 * t - BACK_C2)) / 2.0
    } else {
        ((2.0 * t - 2.0).powi(2) * ((BACK_C2 + 1.0) * (t * 2.0 - 2.0) + BACK_C2) + 2.0) / 2.0
    }
}

// ============================================================================
// Elastic
// ============================================================================

const ELASTIC_C4: f32 = (2.0 * PI) / 3.0;
const ELASTIC_C5: f32 = (2.0 * PI) / 4.5;

/// Elastic ease in.
#[inline]
pub fn elastic_in(t: f32) -> f32 {
    if t == 0.0 {
        0.0
    } else if t == 1.0 {
        1.0
    } else {
        -2.0f32.powf(10.0 * t - 10.0) * ((t * 10.0 - 10.75) * ELASTIC_C4).sin()
    }
}

/// Elastic ease out.
#[inline]
pub fn elastic_out(t: f32) -> f32 {
    if t == 0.0 {
        0.0
    } else if t == 1.0 {
        1.0
    } else {
        2.0f32.powf(-10.0 * t) * ((t * 10.0 - 0.75) * ELASTIC_C4).sin() + 1.0
    }
}

/// Elastic ease in-out.
#[inline]
pub fn elastic_in_out(t: f32) -> f32 {
    if t == 0.0 {
        0.0
    } else if t == 1.0 {
        1.0
    } else if t < 0.5 {
        -(2.0f32.powf(20.0 * t - 10.0) * ((20.0 * t - 11.125) * ELASTIC_C5).sin()) / 2.0
    } else {
        (2.0f32.powf(-20.0 * t + 10.0) * ((20.0 * t - 11.125) * ELASTIC_C5).sin()) / 2.0 + 1.0
    }
}

// ============================================================================
// Bounce
// ============================================================================

const BOUNCE_N1: f32 = 7.5625;
const BOUNCE_D1: f32 = 2.75;

/// Bounce ease out.
#[inline]
pub fn bounce_out(t: f32) -> f32 {
    if t < 1.0 / BOUNCE_D1 {
        BOUNCE_N1 * t * t
    } else if t < 2.0 / BOUNCE_D1 {
        let t = t - 1.5 / BOUNCE_D1;
        BOUNCE_N1 * t * t + 0.75
    } else if t < 2.5 / BOUNCE_D1 {
        let t = t - 2.25 / BOUNCE_D1;
        BOUNCE_N1 * t * t + 0.9375
    } else {
        let t = t - 2.625 / BOUNCE_D1;
        BOUNCE_N1 * t * t + 0.984375
    }
}

/// Bounce ease in.
#[inline]
pub fn bounce_in(t: f32) -> f32 {
    1.0 - bounce_out(1.0 - t)
}

/// Bounce ease in-out.
#[inline]
pub fn bounce_in_out(t: f32) -> f32 {
    if t < 0.5 {
        (1.0 - bounce_out(1.0 - 2.0 * t)) / 2.0
    } else {
        (1.0 + bounce_out(2.0 * t - 1.0)) / 2.0
    }
}

// ============================================================================
// Custom/Utility
// ============================================================================

/// Smoothstep interpolation (cubic Hermite).
#[inline]
pub fn smoothstep(t: f32) -> f32 {
    t * t * (3.0 - 2.0 * t)
}

/// Smootherstep interpolation (quintic Hermite, by Ken Perlin).
#[inline]
pub fn smootherstep(t: f32) -> f32 {
    t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
}

/// Steps between 0 and 1 at the given threshold.
#[inline]
pub fn step(t: f32, threshold: f32) -> f32 {
    if t < threshold { 0.0 } else { 1.0 }
}

/// Creates a stepped easing with n discrete levels.
#[inline]
pub fn stepped(t: f32, steps: u32) -> f32 {
    let steps = steps.max(1) as f32;
    (t * steps).floor() / (steps - 1.0).max(1.0)
}

/// Applies an easing function with reversed direction.
pub fn reverse(t: f32, ease_fn: EaseFn) -> f32 {
    1.0 - ease_fn(1.0 - t)
}

/// Applies an easing function mirrored around the midpoint.
pub fn mirror(t: f32, ease_fn: EaseFn) -> f32 {
    if t < 0.5 {
        ease_fn(t * 2.0) / 2.0
    } else {
        1.0 - ease_fn((1.0 - t) * 2.0) / 2.0
    }
}

/// Applies an easing to a value range.
pub fn ease_value<T>(start: T, end: T, t: f32, ease_fn: EaseFn) -> T
where
    T: std::ops::Add<Output = T>
        + std::ops::Sub<Output = T>
        + std::ops::Mul<f32, Output = T>
        + Copy,
{
    let eased = ease_fn(t.clamp(0.0, 1.0));
    start + (end - start) * eased
}

/// Applies an easing enum to a value range.
pub fn ease_between<T>(start: T, end: T, t: f32, easing: Easing) -> T
where
    T: std::ops::Add<Output = T>
        + std::ops::Sub<Output = T>
        + std::ops::Mul<f32, Output = T>
        + Copy,
{
    ease_value(start, end, t, easing.as_fn())
}

/// Applies an easing function to a [`Lerp`] type.
///
/// This is the preferred way to ease between values that implement `Lerp`.
///
/// # Example
///
/// ```
/// use rhizome_resin_easing::{ease_lerp, Easing, Lerp};
/// use glam::Vec3;
///
/// let start = Vec3::ZERO;
/// let end = Vec3::ONE;
/// let result = ease_lerp(&start, &end, 0.5, Easing::QuadIn);
/// // QuadIn(0.5) = 0.25, so result is Vec3(0.25, 0.25, 0.25)
/// ```
pub fn ease_lerp<T: Lerp>(start: &T, end: &T, t: f32, easing: Easing) -> T {
    let eased = easing.ease(t.clamp(0.0, 1.0));
    start.lerp_to(end, eased)
}

/// Applies an easing function pointer to a [`Lerp`] type.
pub fn ease_lerp_fn<T: Lerp>(start: &T, end: &T, t: f32, ease_fn: EaseFn) -> T {
    let eased = ease_fn(t.clamp(0.0, 1.0));
    start.lerp_to(end, eased)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Tests that an easing function starts at 0 and ends at 1.
    fn test_bounds(ease: EaseFn, name: &str) {
        let start = ease(0.0);
        let end = ease(1.0);
        assert!(
            (start - 0.0).abs() < 0.001,
            "{} at t=0: expected 0, got {}",
            name,
            start
        );
        assert!(
            (end - 1.0).abs() < 0.001,
            "{} at t=1: expected 1, got {}",
            name,
            end
        );
    }

    #[test]
    fn test_all_easings_bounds() {
        let easings = [
            (linear as EaseFn, "linear"),
            (quad_in, "quad_in"),
            (quad_out, "quad_out"),
            (quad_in_out, "quad_in_out"),
            (cubic_in, "cubic_in"),
            (cubic_out, "cubic_out"),
            (cubic_in_out, "cubic_in_out"),
            (quart_in, "quart_in"),
            (quart_out, "quart_out"),
            (quart_in_out, "quart_in_out"),
            (quint_in, "quint_in"),
            (quint_out, "quint_out"),
            (quint_in_out, "quint_in_out"),
            (sine_in, "sine_in"),
            (sine_out, "sine_out"),
            (sine_in_out, "sine_in_out"),
            (expo_in, "expo_in"),
            (expo_out, "expo_out"),
            (expo_in_out, "expo_in_out"),
            (circ_in, "circ_in"),
            (circ_out, "circ_out"),
            (circ_in_out, "circ_in_out"),
            (back_in, "back_in"),
            (back_out, "back_out"),
            (back_in_out, "back_in_out"),
            (elastic_in, "elastic_in"),
            (elastic_out, "elastic_out"),
            (elastic_in_out, "elastic_in_out"),
            (bounce_in, "bounce_in"),
            (bounce_out, "bounce_out"),
            (bounce_in_out, "bounce_in_out"),
        ];

        for (ease_fn, name) in easings {
            test_bounds(ease_fn, name);
        }
    }

    #[test]
    fn test_easing_enum() {
        // Test that enum dispatch matches direct function calls
        assert_eq!(Easing::Linear.ease(0.5), linear(0.5));
        assert_eq!(Easing::QuadIn.ease(0.5), quad_in(0.5));
        assert_eq!(Easing::BounceOut.ease(0.5), bounce_out(0.5));
    }

    #[test]
    fn test_smoothstep() {
        assert!((smoothstep(0.0) - 0.0).abs() < 0.001);
        assert!((smoothstep(1.0) - 1.0).abs() < 0.001);
        assert!((smoothstep(0.5) - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_smootherstep() {
        assert!((smootherstep(0.0) - 0.0).abs() < 0.001);
        assert!((smootherstep(1.0) - 1.0).abs() < 0.001);
        assert!((smootherstep(0.5) - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_stepped() {
        assert_eq!(stepped(0.0, 4), 0.0);
        assert_eq!(stepped(0.3, 4), 1.0 / 3.0);
        assert_eq!(stepped(0.6, 4), 2.0 / 3.0);
    }

    #[test]
    fn test_ease_value() {
        let result = ease_value(0.0, 100.0, 0.5, quad_in);
        assert!((result - 25.0).abs() < 0.001); // quad_in(0.5) = 0.25
    }

    #[test]
    fn test_ease_between() {
        let result = ease_between(0.0, 100.0, 0.5, Easing::Linear);
        assert!((result - 50.0).abs() < 0.001);
    }

    #[test]
    fn test_in_out_symmetry() {
        // In-out functions should be symmetric around 0.5
        let easings = [
            quad_in_out as EaseFn,
            cubic_in_out,
            sine_in_out,
            expo_in_out,
            circ_in_out,
        ];

        for ease in easings {
            let at_mid = ease(0.5);
            assert!(
                (at_mid - 0.5).abs() < 0.001,
                "In-out function not symmetric at midpoint"
            );
        }
    }

    #[test]
    fn test_reverse() {
        // reverse(quad_in) should behave like quad_out
        let rev = reverse(0.75, quad_in);
        let out = quad_out(0.75);
        assert!((rev - out).abs() < 0.001);
    }

    #[test]
    fn test_back_overshoots() {
        // Back easing should go below 0 or above 1
        let back_start = back_in(0.2);
        assert!(back_start < 0.0, "back_in should overshoot below 0");

        let back_end = back_out(0.8);
        assert!(back_end > 1.0, "back_out should overshoot above 1");
    }

    #[test]
    fn test_elastic_oscillates() {
        // Elastic should oscillate
        let samples: Vec<f32> = (0..100).map(|i| elastic_out(i as f32 / 100.0)).collect();

        let mut crossings = 0;
        for i in 1..samples.len() {
            if (samples[i] - 1.0).signum() != (samples[i - 1] - 1.0).signum() {
                crossings += 1;
            }
        }

        assert!(crossings >= 2, "elastic_out should oscillate around target");
    }

    #[test]
    fn test_bounce_multiple_bounces() {
        // Bounce should have multiple peaks
        let samples: Vec<f32> = (0..100).map(|i| bounce_out(i as f32 / 100.0)).collect();

        let mut local_maxima = 0;
        for i in 1..samples.len() - 1 {
            if samples[i] > samples[i - 1] && samples[i] > samples[i + 1] {
                local_maxima += 1;
            }
        }

        assert!(local_maxima >= 3, "bounce should have multiple bounces");
    }

    // ========================================================================
    // Lerp trait tests
    // ========================================================================

    #[test]
    fn test_lerp_f32() {
        assert!((0.0f32.lerp_to(&1.0, 0.0) - 0.0).abs() < 0.001);
        assert!((0.0f32.lerp_to(&1.0, 1.0) - 1.0).abs() < 0.001);
        assert!((0.0f32.lerp_to(&1.0, 0.5) - 0.5).abs() < 0.001);
        // Extrapolation
        assert!((0.0f32.lerp_to(&1.0, 2.0) - 2.0).abs() < 0.001);
    }

    #[test]
    fn test_lerp_vec3() {
        let a = Vec3::ZERO;
        let b = Vec3::ONE;
        let mid = a.lerp_to(&b, 0.5);
        assert!((mid - Vec3::splat(0.5)).length() < 0.001);
    }

    #[test]
    fn test_lerp_quat() {
        let a = Quat::IDENTITY;
        let b = Quat::from_rotation_z(std::f32::consts::FRAC_PI_2); // 90 degrees
        let mid = a.lerp_to(&b, 0.5);
        // Should be roughly 45 degrees
        let angle = mid.to_euler(glam::EulerRot::ZYX).0;
        assert!(
            (angle - std::f32::consts::FRAC_PI_4).abs() < 0.01,
            "Expected ~45 deg, got {} rad",
            angle
        );
    }

    #[test]
    fn test_lerp_array() {
        let a = [0.0f32, 0.0, 0.0];
        let b = [1.0f32, 2.0, 3.0];
        let mid = a.lerp_to(&b, 0.5);
        assert!((mid[0] - 0.5).abs() < 0.001);
        assert!((mid[1] - 1.0).abs() < 0.001);
        assert!((mid[2] - 1.5).abs() < 0.001);
    }

    #[test]
    fn test_ease_lerp() {
        let start = Vec3::ZERO;
        let end = Vec3::ONE;
        let result = ease_lerp(&start, &end, 0.5, Easing::QuadIn);
        // QuadIn(0.5) = 0.25
        assert!((result - Vec3::splat(0.25)).length() < 0.001);
    }
}
