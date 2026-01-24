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
//! use unshape_easing::Lerp;
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
/// use unshape_easing::Lerp;
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
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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
/// use unshape_easing::{ease_lerp, Easing, Lerp};
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

// ============================================================================
// Invariant tests
// ============================================================================

/// Invariant tests for easing functions.
///
/// These tests verify mathematical properties that should hold for all
/// easing functions. Run with:
///
/// ```sh
/// cargo test -p unshape-easing --features invariant-tests
/// ```
#[cfg(all(test, feature = "invariant-tests"))]
mod invariant_tests {
    use super::*;

    // All easing functions to test
    const ALL_EASINGS: &[(Easing, &str)] = &[
        (Easing::Linear, "Linear"),
        (Easing::QuadIn, "QuadIn"),
        (Easing::QuadOut, "QuadOut"),
        (Easing::QuadInOut, "QuadInOut"),
        (Easing::CubicIn, "CubicIn"),
        (Easing::CubicOut, "CubicOut"),
        (Easing::CubicInOut, "CubicInOut"),
        (Easing::QuartIn, "QuartIn"),
        (Easing::QuartOut, "QuartOut"),
        (Easing::QuartInOut, "QuartInOut"),
        (Easing::QuintIn, "QuintIn"),
        (Easing::QuintOut, "QuintOut"),
        (Easing::QuintInOut, "QuintInOut"),
        (Easing::SineIn, "SineIn"),
        (Easing::SineOut, "SineOut"),
        (Easing::SineInOut, "SineInOut"),
        (Easing::ExpoIn, "ExpoIn"),
        (Easing::ExpoOut, "ExpoOut"),
        (Easing::ExpoInOut, "ExpoInOut"),
        (Easing::CircIn, "CircIn"),
        (Easing::CircOut, "CircOut"),
        (Easing::CircInOut, "CircInOut"),
        (Easing::BackIn, "BackIn"),
        (Easing::BackOut, "BackOut"),
        (Easing::BackInOut, "BackInOut"),
        (Easing::ElasticIn, "ElasticIn"),
        (Easing::ElasticOut, "ElasticOut"),
        (Easing::ElasticInOut, "ElasticInOut"),
        (Easing::BounceIn, "BounceIn"),
        (Easing::BounceOut, "BounceOut"),
        (Easing::BounceInOut, "BounceInOut"),
    ];

    // ========================================================================
    // Boundary conditions
    // ========================================================================

    /// All easing functions must pass through (0, 0).
    #[test]
    fn test_all_easings_start_at_zero() {
        for (easing, name) in ALL_EASINGS {
            let value = easing.ease(0.0);
            assert!(
                value.abs() < 0.001,
                "{name}: ease(0) should be 0, got {value}"
            );
        }
    }

    /// All easing functions must pass through (1, 1).
    #[test]
    fn test_all_easings_end_at_one() {
        for (easing, name) in ALL_EASINGS {
            let value = easing.ease(1.0);
            assert!(
                (value - 1.0).abs() < 0.001,
                "{name}: ease(1) should be 1, got {value}"
            );
        }
    }

    // ========================================================================
    // In/Out duality
    // ========================================================================

    /// For polynomial easings: ease_out(t) = 1 - ease_in(1 - t).
    #[test]
    fn test_in_out_duality() {
        let pairs = [
            (Easing::QuadIn, Easing::QuadOut),
            (Easing::CubicIn, Easing::CubicOut),
            (Easing::QuartIn, Easing::QuartOut),
            (Easing::QuintIn, Easing::QuintOut),
            (Easing::SineIn, Easing::SineOut),
            (Easing::ExpoIn, Easing::ExpoOut),
            (Easing::CircIn, Easing::CircOut),
            (Easing::BackIn, Easing::BackOut),
            (Easing::BounceIn, Easing::BounceOut),
        ];

        for (ease_in, ease_out) in pairs {
            for i in 0..=20 {
                let t = i as f32 / 20.0;
                let out_value = ease_out.ease(t);
                let dual_value = 1.0 - ease_in.ease(1.0 - t);

                assert!(
                    (out_value - dual_value).abs() < 0.001,
                    "{:?} out(t) should equal 1 - in(1-t) at t={t}",
                    ease_in
                );
            }
        }
    }

    // ========================================================================
    // In-out symmetry
    // ========================================================================

    /// In-out functions should be point-symmetric around (0.5, 0.5).
    /// This means: ease(0.5 - d) + ease(0.5 + d) = 1 for any d in [0, 0.5].
    #[test]
    fn test_in_out_point_symmetry() {
        let in_outs = [
            Easing::QuadInOut,
            Easing::CubicInOut,
            Easing::QuartInOut,
            Easing::QuintInOut,
            Easing::SineInOut,
            Easing::ExpoInOut,
            Easing::CircInOut,
            Easing::BackInOut,
            Easing::ElasticInOut,
            Easing::BounceInOut,
        ];

        for easing in in_outs {
            for i in 0..=10 {
                let d = i as f32 / 20.0; // d from 0 to 0.5
                let low = easing.ease(0.5 - d);
                let high = easing.ease(0.5 + d);
                let sum = low + high;

                assert!(
                    (sum - 1.0).abs() < 0.01,
                    "{:?}: ease(0.5-{d}) + ease(0.5+{d}) = {sum}, expected 1.0",
                    easing
                );
            }
        }
    }

    /// All in-out functions should pass through (0.5, 0.5).
    #[test]
    fn test_in_out_midpoint() {
        let in_outs = [
            Easing::QuadInOut,
            Easing::CubicInOut,
            Easing::QuartInOut,
            Easing::QuintInOut,
            Easing::SineInOut,
            Easing::ExpoInOut,
            Easing::CircInOut,
            Easing::BackInOut,
            Easing::BounceInOut,
        ];

        for easing in in_outs {
            let value = easing.ease(0.5);
            assert!(
                (value - 0.5).abs() < 0.01,
                "{:?}: ease(0.5) should be 0.5, got {value}",
                easing
            );
        }
    }

    // ========================================================================
    // Monotonicity
    // ========================================================================

    /// Simple polynomial ease-in functions should be monotonically increasing.
    #[test]
    fn test_monotonic_ease_in() {
        let monotonic_ins = [
            Easing::Linear,
            Easing::QuadIn,
            Easing::CubicIn,
            Easing::QuartIn,
            Easing::QuintIn,
            Easing::SineIn,
            Easing::ExpoIn,
            Easing::CircIn,
        ];

        for easing in monotonic_ins {
            let mut prev = easing.ease(0.0);
            for i in 1..=100 {
                let t = i as f32 / 100.0;
                let value = easing.ease(t);
                assert!(
                    value >= prev - 0.0001,
                    "{:?} should be monotonic: ease({}) = {} < ease({}) = {}",
                    easing,
                    t,
                    value,
                    (i - 1) as f32 / 100.0,
                    prev
                );
                prev = value;
            }
        }
    }

    /// Simple polynomial ease-out functions should be monotonically increasing.
    #[test]
    fn test_monotonic_ease_out() {
        let monotonic_outs = [
            Easing::QuadOut,
            Easing::CubicOut,
            Easing::QuartOut,
            Easing::QuintOut,
            Easing::SineOut,
            Easing::ExpoOut,
            Easing::CircOut,
        ];

        for easing in monotonic_outs {
            let mut prev = easing.ease(0.0);
            for i in 1..=100 {
                let t = i as f32 / 100.0;
                let value = easing.ease(t);
                assert!(value >= prev - 0.0001, "{:?} should be monotonic", easing);
                prev = value;
            }
        }
    }

    // ========================================================================
    // Smoothstep properties
    // ========================================================================

    /// Smoothstep should have zero derivative at endpoints.
    #[test]
    fn test_smoothstep_zero_derivative_at_endpoints() {
        let eps = 0.0001;

        // At t=0
        let d0 = (smoothstep(eps) - smoothstep(0.0)) / eps;
        assert!(
            d0.abs() < 0.01,
            "smoothstep derivative at t=0 should be ~0, got {d0}"
        );

        // At t=1
        let d1 = (smoothstep(1.0) - smoothstep(1.0 - eps)) / eps;
        assert!(
            d1.abs() < 0.01,
            "smoothstep derivative at t=1 should be ~0, got {d1}"
        );
    }

    /// Smootherstep should have zero first and second derivative at endpoints.
    #[test]
    fn test_smootherstep_zero_derivatives_at_endpoints() {
        let eps = 0.0001;

        // First derivative at t=0
        let d0 = (smootherstep(eps) - smootherstep(0.0)) / eps;
        assert!(
            d0.abs() < 0.01,
            "smootherstep derivative at t=0 should be ~0, got {d0}"
        );

        // First derivative at t=1
        let d1 = (smootherstep(1.0) - smootherstep(1.0 - eps)) / eps;
        assert!(
            d1.abs() < 0.01,
            "smootherstep derivative at t=1 should be ~0, got {d1}"
        );
    }

    /// Smoothstep and smootherstep should pass through (0.5, 0.5).
    #[test]
    fn test_smooth_midpoint() {
        assert!(
            (smoothstep(0.5) - 0.5).abs() < 0.001,
            "smoothstep(0.5) should be 0.5"
        );
        assert!(
            (smootherstep(0.5) - 0.5).abs() < 0.001,
            "smootherstep(0.5) should be 0.5"
        );
    }

    // ========================================================================
    // Lerp invariants
    // ========================================================================

    /// Lerp at t=0 returns start value.
    #[test]
    fn test_lerp_at_zero_returns_start() {
        assert!((0.0f32.lerp_to(&100.0, 0.0) - 0.0).abs() < 0.001);
        assert!((Vec3::ZERO.lerp_to(&Vec3::ONE, 0.0) - Vec3::ZERO).length() < 0.001);
        assert!((Vec2::ZERO.lerp_to(&Vec2::ONE, 0.0) - Vec2::ZERO).length() < 0.001);
    }

    /// Lerp at t=1 returns end value.
    #[test]
    fn test_lerp_at_one_returns_end() {
        assert!((0.0f32.lerp_to(&100.0, 1.0) - 100.0).abs() < 0.001);
        assert!((Vec3::ZERO.lerp_to(&Vec3::ONE, 1.0) - Vec3::ONE).length() < 0.001);
        assert!((Vec2::ZERO.lerp_to(&Vec2::ONE, 1.0) - Vec2::ONE).length() < 0.001);
    }

    /// Lerp is linear: lerp(a, b, t) = a + (b - a) * t.
    #[test]
    fn test_lerp_linearity() {
        for i in 0..=20 {
            let t = i as f32 / 20.0;
            let a = 10.0f32;
            let b = 50.0f32;

            let lerped = a.lerp_to(&b, t);
            let expected = a + (b - a) * t;

            assert!(
                (lerped - expected).abs() < 0.001,
                "Lerp should be linear at t={t}"
            );
        }
    }

    /// Lerp should extrapolate beyond [0, 1].
    #[test]
    fn test_lerp_extrapolation() {
        // t = -0.5 should go below start
        let below = 0.0f32.lerp_to(&10.0, -0.5);
        assert!((below - (-5.0)).abs() < 0.001);

        // t = 1.5 should go above end
        let above = 0.0f32.lerp_to(&10.0, 1.5);
        assert!((above - 15.0).abs() < 0.001);
    }

    /// Lerp from x to x should always return x.
    #[test]
    fn test_lerp_same_value() {
        let value = 42.0f32;
        for i in 0..=10 {
            let t = i as f32 / 10.0;
            let result = value.lerp_to(&value, t);
            assert!(
                (result - value).abs() < 0.001,
                "Lerp from x to x should be x at any t"
            );
        }
    }

    // ========================================================================
    // Reverse function
    // ========================================================================

    /// reverse preserves boundary conditions.
    #[test]
    fn test_reverse_boundary_conditions() {
        // reverse(0, f) should be 1 - f(1) = 1 - 1 = 0
        // reverse(1, f) should be 1 - f(0) = 1 - 0 = 1
        for (easing, name) in ALL_EASINGS {
            let ease_fn = easing.as_fn();

            let at_zero = reverse(0.0, ease_fn);
            let at_one = reverse(1.0, ease_fn);

            assert!(
                at_zero.abs() < 0.001,
                "{name}: reverse(0) should be 0, got {at_zero}"
            );
            assert!(
                (at_one - 1.0).abs() < 0.001,
                "{name}: reverse(1) should be 1, got {at_one}"
            );
        }
    }

    /// reverse(t, ease_in) should behave like ease_out.
    #[test]
    fn test_reverse_in_gives_out() {
        for i in 0..=20 {
            let t = i as f32 / 20.0;

            // reverse(quad_in) should equal quad_out
            let reversed = reverse(t, quad_in);
            let direct = quad_out(t);
            assert!(
                (reversed - direct).abs() < 0.001,
                "reverse(quad_in) should equal quad_out at t={t}"
            );
        }
    }

    // ========================================================================
    // Stepped function
    // ========================================================================

    /// Stepped should produce discrete, quantized values within [0, 1).
    #[test]
    fn test_stepped_produces_discrete_values() {
        let steps = 5u32;
        let mut seen_values = std::collections::HashSet::new();

        // Sample within [0, 1) to avoid edge case at t=1.0
        for i in 0..100 {
            let t = i as f32 / 100.0;
            let value = stepped(t, steps);
            // Round to avoid float comparison issues
            let rounded = (value * 1000.0).round() as i32;
            seen_values.insert(rounded);
        }

        // Within [0, 1), should have `steps` distinct values
        assert!(
            seen_values.len() <= steps as usize,
            "stepped({steps}) within [0,1) should produce at most {steps} values, got {}",
            seen_values.len()
        );
    }

    /// Stepped at t=0 should produce 0.
    #[test]
    fn test_stepped_at_zero() {
        assert!((stepped(0.0, 4) - 0.0).abs() < 0.001);
        assert!((stepped(0.0, 10) - 0.0).abs() < 0.001);
    }

    // ========================================================================
    // Mirror function
    // ========================================================================

    /// Mirror should start and end at same values as original.
    #[test]
    fn test_mirror_boundary_conditions() {
        for (easing, name) in ALL_EASINGS {
            let ease_fn = easing.as_fn();
            let start = mirror(0.0, ease_fn);
            let end = mirror(1.0, ease_fn);

            assert!(
                start.abs() < 0.001,
                "{name}: mirror(0) should be 0, got {start}"
            );
            assert!(
                (end - 1.0).abs() < 0.001,
                "{name}: mirror(1) should be 1, got {end}"
            );
        }
    }

    // ========================================================================
    // Ease value function
    // ========================================================================

    /// ease_value should properly scale to value range.
    #[test]
    fn test_ease_value_scaling() {
        let start = 100.0f32;
        let end = 200.0f32;

        for (easing, _) in ALL_EASINGS {
            let ease_fn = easing.as_fn();

            // At t=0, should be start
            let at_zero = ease_value(start, end, 0.0, ease_fn);
            assert!(
                (at_zero - start).abs() < 0.01,
                "{:?}: ease_value at t=0 should be start",
                easing
            );

            // At t=1, should be end
            let at_one = ease_value(start, end, 1.0, ease_fn);
            assert!(
                (at_one - end).abs() < 0.01,
                "{:?}: ease_value at t=1 should be end",
                easing
            );
        }
    }
}
