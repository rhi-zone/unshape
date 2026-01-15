//! Unified curve trait and types for 2D/3D paths.
//!
//! This crate provides:
//! - [`VectorSpace`] - trait for types that support vector operations
//! - [`Curve`] - unified interface for all curve types
//! - Concrete curve types: [`Line`], [`QuadBezier`], [`CubicBezier`], [`Arc`]
//! - Segment enums: [`Segment2D`], [`Segment3D`]
//! - Path types: [`Path`], [`ArcLengthPath`]

use glam::{Vec2, Vec3};
use std::ops::{Add, Mul, Sub};

mod arc;
mod bezier;
mod line;
mod path;
mod segment;
mod traits;

pub use arc::Arc;
pub use bezier::{CubicBezier, QuadBezier};
pub use line::Line;
pub use path::{ArcLengthPath, Path};
pub use segment::{Segment2D, Segment3D};
pub use traits::{Curve, Curve2DExt, Curve3DExt, VectorSpace};

/// Trait for types that can be interpolated.
///
/// This is the base requirement for curve control points.
pub trait Interpolatable:
    Clone + Copy + Add<Output = Self> + Sub<Output = Self> + Mul<f32, Output = Self>
{
}

impl Interpolatable for f32 {}
impl Interpolatable for Vec2 {}
impl Interpolatable for Vec3 {}

/// Linear interpolation between two values.
#[inline]
pub fn lerp<T: Interpolatable>(a: T, b: T, t: f32) -> T {
    a * (1.0 - t) + b * t
}

/// Axis-aligned bounding box (3D).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Aabb {
    pub min: Vec3,
    pub max: Vec3,
}

impl Aabb {
    pub fn new(min: Vec3, max: Vec3) -> Self {
        Self { min, max }
    }

    pub fn from_points(points: impl IntoIterator<Item = Vec3>) -> Option<Self> {
        let mut iter = points.into_iter();
        let first = iter.next()?;
        let mut min = first;
        let mut max = first;
        for p in iter {
            min = min.min(p);
            max = max.max(p);
        }
        Some(Self { min, max })
    }
}

/// Axis-aligned bounding rectangle (2D).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rect {
    pub min: Vec2,
    pub max: Vec2,
}

impl Rect {
    pub fn new(min: Vec2, max: Vec2) -> Self {
        Self { min, max }
    }

    pub fn from_points(points: impl IntoIterator<Item = Vec2>) -> Option<Self> {
        let mut iter = points.into_iter();
        let first = iter.next()?;
        let mut min = first;
        let mut max = first;
        for p in iter {
            min = min.min(p);
            max = max.max(p);
        }
        Some(Self { min, max })
    }

    pub fn width(&self) -> f32 {
        self.max.x - self.min.x
    }

    pub fn height(&self) -> f32 {
        self.max.y - self.min.y
    }
}
