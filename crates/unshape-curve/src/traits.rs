//! Core traits for curves.

use crate::{Aabb, CubicBezier, Interpolatable, Rect};
use glam::{Vec2, Vec3};

/// Vector operations needed for arc length calculation.
///
/// Extends [`Interpolatable`] with length and normalization.
pub trait VectorSpace: Interpolatable {
    /// Returns the length (magnitude) of this vector.
    fn length(&self) -> f32;

    /// Returns a normalized (unit length) version of this vector.
    ///
    /// Returns zero vector if length is zero.
    fn normalize(&self) -> Self;

    /// Returns the dot product with another vector.
    fn dot(&self, other: Self) -> f32;
}

impl VectorSpace for Vec2 {
    #[inline]
    fn length(&self) -> f32 {
        Vec2::length(*self)
    }

    #[inline]
    fn normalize(&self) -> Self {
        Vec2::normalize_or_zero(*self)
    }

    #[inline]
    fn dot(&self, other: Self) -> f32 {
        Vec2::dot(*self, other)
    }
}

impl VectorSpace for Vec3 {
    #[inline]
    fn length(&self) -> f32 {
        Vec3::length(*self)
    }

    #[inline]
    fn normalize(&self) -> Self {
        Vec3::normalize_or_zero(*self)
    }

    #[inline]
    fn dot(&self, other: Self) -> f32 {
        Vec3::dot(*self, other)
    }
}

/// Unified curve interface for any dimension.
///
/// All curve types implement this trait, enabling generic algorithms
/// that work with any curve representation.
pub trait Curve: Clone {
    /// The point type (Vec2 or Vec3).
    type Point: VectorSpace;

    /// Returns the point at parameter t ∈ [0, 1].
    fn position_at(&self, t: f32) -> Self::Point;

    /// Returns the tangent vector at parameter t (not normalized).
    fn tangent_at(&self, t: f32) -> Self::Point;

    /// Splits the curve at parameter t, returning (before, after).
    fn split(&self, t: f32) -> (Self, Self)
    where
        Self: Sized;

    /// Converts to cubic Bézier approximation(s).
    ///
    /// Some curves (arcs, NURBS) may produce multiple cubics.
    fn to_cubics(&self) -> Vec<CubicBezier<Self::Point>>;

    /// Returns the start point (t = 0).
    #[inline]
    fn start(&self) -> Self::Point {
        self.position_at(0.0)
    }

    /// Returns the end point (t = 1).
    #[inline]
    fn end(&self) -> Self::Point {
        self.position_at(1.0)
    }

    /// Approximate arc length using Gaussian quadrature.
    fn length(&self) -> f32 {
        // 5-point Gaussian quadrature for [0, 1]
        // Weights are scaled by 0.5 (standard weights sum to 2.0 for [-1,1])
        const WEIGHTS: [f32; 5] = [
            0.2369269 * 0.5,
            0.4786287 * 0.5,
            0.5688889 * 0.5,
            0.4786287 * 0.5,
            0.2369269 * 0.5,
        ];
        const POINTS: [f32; 5] = [0.0469101, 0.2307653, 0.5, 0.7692347, 0.9530899];

        let mut sum = 0.0;
        for (w, t) in WEIGHTS.iter().zip(POINTS.iter()) {
            sum += w * self.tangent_at(*t).length();
        }
        sum
    }

    /// Sample points for rendering using adaptive subdivision.
    fn flatten(&self, tolerance: f32) -> Vec<Self::Point> {
        let mut points = vec![self.start()];
        self.flatten_recursive(0.0, 1.0, tolerance, &mut points);
        points
    }

    /// Helper for adaptive flattening.
    fn flatten_recursive(&self, t0: f32, t1: f32, tolerance: f32, points: &mut Vec<Self::Point>) {
        let mid_t = (t0 + t1) / 2.0;
        let p0 = self.position_at(t0);
        let p1 = self.position_at(t1);
        let mid = self.position_at(mid_t);

        // Check if the midpoint is close enough to the line p0-p1
        let chord = p1 - p0;
        let chord_len = chord.length();

        if chord_len < 1e-10 {
            points.push(p1);
            return;
        }

        // Distance from mid to line p0-p1
        let to_mid = mid - p0;
        let proj_len = to_mid.dot(chord) / chord_len;
        let proj = chord * (proj_len / chord_len);
        let perp = to_mid - proj;
        let dist = perp.length();

        if dist <= tolerance {
            points.push(p1);
        } else {
            self.flatten_recursive(t0, mid_t, tolerance, points);
            self.flatten_recursive(mid_t, t1, tolerance, points);
        }
    }
}

/// Extension trait for 2D curves.
pub trait Curve2DExt: Curve<Point = Vec2> {
    /// Returns the axis-aligned bounding box.
    fn bounding_box(&self) -> Rect;
}

/// Extension trait for 3D curves.
pub trait Curve3DExt: Curve<Point = Vec3> {
    /// Returns the axis-aligned bounding box.
    fn bounding_box(&self) -> Aabb;
}
