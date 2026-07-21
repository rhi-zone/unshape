//! Vector warp/deform operations.
//!
//! These ops displace points (and, via [`Path::transform`], whole paths) using
//! spatial deformation techniques: lattice-based free-form deformation, cage
//! (mean value coordinates) deformation, brush-style pushes, twists, and tapers.

use crate::{Path, Rect};
use glam::Vec2;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Numerical epsilon used for degenerate-case checks (coincident points, zero-length
/// vectors) across the warp ops.
const EPS: f32 = 1e-6;

// ============================================================================
// Falloff
// ============================================================================

/// Distance-based attenuation curve shared by brush-like warp ops.
///
/// `evaluate(t)` takes a normalized distance `t` in `[0, 1]` (0 = at the effect
/// center, 1 = at the edge of its radius) and returns a weight in `[0, 1]`
/// (1 = full effect, 0 = no effect). Values of `t` outside `[0, 1]` are clamped
/// at the boundary weight (1 for `t <= 0`, 0 for `t >= 1`).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum Falloff {
    /// Linear falloff: weight decreases proportionally with distance.
    #[default]
    Linear,
    /// Smoothstep falloff: eases in/out at the center and edge.
    Smooth,
    /// Gaussian falloff: sharp peak at the center, quick decay towards the edge.
    Gaussian,
}

impl Falloff {
    /// Evaluates the falloff weight for a normalized distance `t`.
    pub fn evaluate(&self, t: f32) -> f32 {
        if t >= 1.0 {
            return 0.0;
        }
        if t <= 0.0 {
            return 1.0;
        }
        match self {
            Falloff::Linear => 1.0 - t,
            Falloff::Smooth => {
                let s = 1.0 - t;
                s * s * (3.0 - 2.0 * s)
            }
            Falloff::Gaussian => (-4.5 * t * t).exp(),
        }
    }
}

// ============================================================================
// LatticeDeform
// ============================================================================

/// Free-form deformation via a rectangular grid of control points.
///
/// A `rows` x `cols` grid of nodes is laid out evenly across `bounds`. Each
/// node's undisplaced position is implied by its position in the grid;
/// `control_points` holds each node's *displaced* position (row-major:
/// `control_points[row * cols + col]`). Points are deformed by bilinearly
/// interpolating the surrounding four nodes' displacements.
///
/// If `rows < 2`, `cols < 2`, or `control_points.len() != rows * cols`, the
/// configuration is invalid and points pass through unchanged.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dynop", derive(unshape_op::Op))]
#[cfg_attr(feature = "dynop", op(input = Vec<Vec2>, output = Vec<Vec2>))]
pub struct LatticeDeform {
    /// Number of grid rows (must be >= 2 for the lattice to have any effect).
    pub rows: u32,
    /// Number of grid columns (must be >= 2 for the lattice to have any effect).
    pub cols: u32,
    /// Displaced position of each grid node, row-major (`row * cols + col`).
    pub control_points: Vec<Vec2>,
    /// The region of the plane the undisplaced lattice covers.
    pub bounds: Rect,
}

impl LatticeDeform {
    /// Creates an identity lattice (no displacement) covering `bounds` with the
    /// given resolution. Mutate `control_points` to introduce deformation.
    pub fn new(rows: u32, cols: u32, bounds: Rect) -> Self {
        let control_points = Self::identity_grid(rows, cols, bounds);
        Self {
            rows,
            cols,
            control_points,
            bounds,
        }
    }

    /// Generates the undisplaced (identity) grid node positions for a given
    /// resolution and bounds, in the same row-major order expected by
    /// `control_points`.
    pub fn identity_grid(rows: u32, cols: u32, bounds: Rect) -> Vec<Vec2> {
        if rows < 2 || cols < 2 {
            return Vec::new();
        }
        let size = bounds.max - bounds.min;
        let mut grid = Vec::with_capacity((rows * cols) as usize);
        for j in 0..rows {
            let v = j as f32 / (rows - 1) as f32;
            for i in 0..cols {
                let u = i as f32 / (cols - 1) as f32;
                grid.push(bounds.min + size * Vec2::new(u, v));
            }
        }
        grid
    }

    fn is_valid(&self) -> bool {
        self.rows >= 2
            && self.cols >= 2
            && self.control_points.len() == (self.rows * self.cols) as usize
    }

    fn deform_point(&self, p: Vec2) -> Vec2 {
        if !self.is_valid() {
            return p;
        }

        let size = self.bounds.max - self.bounds.min;
        let u = if size.x.abs() < EPS {
            0.0
        } else {
            (p.x - self.bounds.min.x) / size.x
        };
        let v = if size.y.abs() < EPS {
            0.0
        } else {
            (p.y - self.bounds.min.y) / size.y
        };

        let cols = self.cols as usize;
        let rows = self.rows as usize;

        let fx = u * (cols - 1) as f32;
        let fy = v * (rows - 1) as f32;

        let i0 = (fx.floor() as isize).clamp(0, cols as isize - 2) as usize;
        let j0 = (fy.floor() as isize).clamp(0, rows as isize - 2) as usize;
        let i1 = i0 + 1;
        let j1 = j0 + 1;

        let tx = fx - i0 as f32;
        let ty = fy - j0 as f32;

        let c00 = self.control_points[j0 * cols + i0];
        let c10 = self.control_points[j0 * cols + i1];
        let c01 = self.control_points[j1 * cols + i0];
        let c11 = self.control_points[j1 * cols + i1];

        let top = c00.lerp(c10, tx);
        let bottom = c01.lerp(c11, tx);
        top.lerp(bottom, ty)
    }

    /// Applies the lattice deformation to a set of points.
    pub fn apply(&self, points: &[Vec2]) -> Vec<Vec2> {
        points.iter().map(|&p| self.deform_point(p)).collect()
    }

    /// Applies the lattice deformation to every point of a path.
    pub fn apply_to_path(&self, path: &Path) -> Path {
        let mut result = path.clone();
        result.transform(|p| self.deform_point(p));
        result
    }
}

// ============================================================================
// CageDeform
// ============================================================================

/// Cage-based deformation using mean value coordinates.
///
/// A polygon "cage" (`source_cage`) surrounds the geometry to deform. Each
/// point's mean value coordinates with respect to `source_cage` are computed
/// once, then used as weights to blend the corresponding vertices of
/// `target_cage`, producing a smooth deformation as the cage is reshaped.
///
/// `source_cage` and `target_cage` must have the same length (>= 3); otherwise
/// points pass through unchanged.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dynop", derive(unshape_op::Op))]
#[cfg_attr(feature = "dynop", op(input = Vec<Vec2>, output = Vec<Vec2>))]
pub struct CageDeform {
    /// Cage vertices surrounding the source geometry, in order.
    pub source_cage: Vec<Vec2>,
    /// Displaced cage vertices; must be the same length as `source_cage`.
    pub target_cage: Vec<Vec2>,
}

impl CageDeform {
    /// Creates a new cage deform from matching source/target cage vertex lists.
    pub fn new(source_cage: Vec<Vec2>, target_cage: Vec<Vec2>) -> Self {
        Self {
            source_cage,
            target_cage,
        }
    }

    fn is_valid(&self) -> bool {
        self.source_cage.len() >= 3 && self.source_cage.len() == self.target_cage.len()
    }

    fn deform_point(&self, p: Vec2) -> Vec2 {
        if !self.is_valid() {
            return p;
        }

        let weights = mean_value_weights(&self.source_cage, p);
        weights
            .iter()
            .zip(self.target_cage.iter())
            .fold(Vec2::ZERO, |acc, (&w, &target)| acc + target * w)
    }

    /// Applies the cage deformation to a set of points.
    pub fn apply(&self, points: &[Vec2]) -> Vec<Vec2> {
        points.iter().map(|&p| self.deform_point(p)).collect()
    }

    /// Applies the cage deformation to every point of a path.
    pub fn apply_to_path(&self, path: &Path) -> Path {
        let mut result = path.clone();
        result.transform(|p| self.deform_point(p));
        result
    }
}

/// Computes 2D mean value coordinates of `p` with respect to the polygon `cage`.
///
/// Returns one weight per cage vertex; the weights sum to 1 (barycentric-like)
/// and reproduce `p` exactly when dotted with `cage` itself. Based on Hormann &
/// Floater, "Mean Value Coordinates for Arbitrary Planar Polygons" (2006).
fn mean_value_weights(cage: &[Vec2], p: Vec2) -> Vec<f32> {
    let n = cage.len();
    let s: Vec<Vec2> = cage.iter().map(|&v| v - p).collect();
    let r: Vec<f32> = s.iter().map(|v| v.length()).collect();

    // Point coincides with a cage vertex: full weight there.
    for i in 0..n {
        if r[i] < EPS {
            let mut w = vec![0.0; n];
            w[i] = 1.0;
            return w;
        }
    }

    // Point lies on a cage edge: linear interpolation of that edge's endpoints.
    for i in 0..n {
        let j = (i + 1) % n;
        let cross = s[i].x * s[j].y - s[i].y * s[j].x;
        let dot = s[i].dot(s[j]);
        if cross.abs() < EPS && dot < 0.0 {
            let total = r[i] + r[j];
            let mut w = vec![0.0; n];
            if total > EPS {
                w[i] = r[j] / total;
                w[j] = r[i] / total;
            }
            return w;
        }
    }

    // tan_half[i] = tan(angle(s_i, s_{i+1}) / 2), for edge i -> i+1.
    let mut tan_half = vec![0.0f32; n];
    for i in 0..n {
        let j = (i + 1) % n;
        let cross = s[i].x * s[j].y - s[i].y * s[j].x;
        let dot = s[i].dot(s[j]);
        let denom = r[i] * r[j] + dot;
        tan_half[i] = if denom.abs() > EPS {
            cross / denom
        } else {
            0.0
        };
    }

    let mut w = vec![0.0f32; n];
    for i in 0..n {
        let prev = (i + n - 1) % n;
        w[i] = (tan_half[prev] + tan_half[i]) / r[i];
    }

    let sum: f32 = w.iter().sum();
    if sum.abs() > EPS {
        for wi in w.iter_mut() {
            *wi /= sum;
        }
    }
    w
}

// ============================================================================
// PointPush
// ============================================================================

/// Brush-like displacement: points within `radius` of `center` are pushed by
/// `direction`, attenuated by `falloff` based on distance from the center.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dynop", derive(unshape_op::Op))]
#[cfg_attr(feature = "dynop", op(input = Vec<Vec2>, output = Vec<Vec2>))]
pub struct PointPush {
    /// Center of the push brush.
    pub center: Vec2,
    /// Displacement applied at the center (magnitude scales with distance from
    /// the encoded direction vector's own length).
    pub direction: Vec2,
    /// Radius of effect; points beyond this distance are unaffected.
    pub radius: f32,
    /// Attenuation curve from center (full effect) to radius (no effect).
    pub falloff: Falloff,
}

impl PointPush {
    /// Creates a new point push.
    pub fn new(center: Vec2, direction: Vec2, radius: f32, falloff: Falloff) -> Self {
        Self {
            center,
            direction,
            radius,
            falloff,
        }
    }

    fn deform_point(&self, p: Vec2) -> Vec2 {
        if self.radius <= 0.0 {
            return p;
        }
        let dist = (p - self.center).length();
        if dist >= self.radius {
            return p;
        }
        let t = dist / self.radius;
        let w = self.falloff.evaluate(t);
        p + self.direction * w
    }

    /// Applies the push to a set of points.
    pub fn apply(&self, points: &[Vec2]) -> Vec<Vec2> {
        points.iter().map(|&p| self.deform_point(p)).collect()
    }

    /// Applies the push to every point of a path.
    pub fn apply_to_path(&self, path: &Path) -> Path {
        let mut result = path.clone();
        result.transform(|p| self.deform_point(p));
        result
    }
}

// ============================================================================
// Twist
// ============================================================================

/// Rotates points around `center` by an angle proportional to `falloff` of
/// their distance from the center, up to `radius`.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dynop", derive(unshape_op::Op))]
#[cfg_attr(feature = "dynop", op(input = Vec<Vec2>, output = Vec<Vec2>))]
pub struct Twist {
    /// Center of rotation.
    pub center: Vec2,
    /// Maximum rotation angle (radians), applied in full at the center.
    pub angle: f32,
    /// Radius of effect; points beyond this distance are unaffected.
    pub radius: f32,
    /// Attenuation curve from center (full angle) to radius (no rotation).
    pub falloff: Falloff,
}

impl Twist {
    /// Creates a new twist.
    pub fn new(center: Vec2, angle: f32, radius: f32, falloff: Falloff) -> Self {
        Self {
            center,
            angle,
            radius,
            falloff,
        }
    }

    fn deform_point(&self, p: Vec2) -> Vec2 {
        if self.radius <= 0.0 {
            return p;
        }
        let offset = p - self.center;
        let dist = offset.length();
        if dist >= self.radius {
            return p;
        }
        let t = dist / self.radius;
        let applied_angle = self.angle * self.falloff.evaluate(t);
        let cos = applied_angle.cos();
        let sin = applied_angle.sin();
        let rotated = Vec2::new(
            offset.x * cos - offset.y * sin,
            offset.x * sin + offset.y * cos,
        );
        self.center + rotated
    }

    /// Applies the twist to a set of points.
    pub fn apply(&self, points: &[Vec2]) -> Vec<Vec2> {
        points.iter().map(|&p| self.deform_point(p)).collect()
    }

    /// Applies the twist to every point of a path.
    pub fn apply_to_path(&self, path: &Path) -> Path {
        let mut result = path.clone();
        result.transform(|p| self.deform_point(p));
        result
    }
}

// ============================================================================
// Taper
// ============================================================================

/// Scales points perpendicular to `axis`, interpolating from `start_scale` at
/// `origin` to `end_scale` at `length` along the axis. Points beyond `[0,
/// length]` clamp to the nearest end's scale.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dynop", derive(unshape_op::Op))]
#[cfg_attr(feature = "dynop", op(input = Vec<Vec2>, output = Vec<Vec2>))]
pub struct Taper {
    /// Direction along which the scale interpolates.
    pub axis: Vec2,
    /// Origin of the taper (`start_scale` applies here).
    pub origin: Vec2,
    /// Scale applied perpendicular to the axis at `origin`.
    pub start_scale: f32,
    /// Scale applied perpendicular to the axis at `origin + axis.normalize() * length`.
    pub end_scale: f32,
    /// Distance along the axis over which the scale interpolates.
    pub length: f32,
}

impl Taper {
    /// Creates a new taper.
    pub fn new(axis: Vec2, origin: Vec2, start_scale: f32, end_scale: f32, length: f32) -> Self {
        Self {
            axis,
            origin,
            start_scale,
            end_scale,
            length,
        }
    }

    fn deform_point(&self, p: Vec2) -> Vec2 {
        let axis_len = self.axis.length();
        if axis_len < EPS || self.length <= 0.0 {
            return p;
        }
        let axis_n = self.axis / axis_len;
        let rel = p - self.origin;
        let along = rel.dot(axis_n);
        let perp = rel - axis_n * along;
        let t = (along / self.length).clamp(0.0, 1.0);
        let scale = self.start_scale + (self.end_scale - self.start_scale) * t;
        self.origin + axis_n * along + perp * scale
    }

    /// Applies the taper to a set of points.
    pub fn apply(&self, points: &[Vec2]) -> Vec<Vec2> {
        points.iter().map(|&p| self.deform_point(p)).collect()
    }

    /// Applies the taper to every point of a path.
    pub fn apply_to_path(&self, path: &Path) -> Path {
        let mut result = path.clone();
        result.transform(|p| self.deform_point(p));
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Rect, line, rect};

    const TOL: f32 = 1e-3;

    fn approx(a: Vec2, b: Vec2, tol: f32) -> bool {
        a.distance(b) < tol
    }

    // =========================================================================
    // Falloff tests
    // =========================================================================

    #[test]
    fn falloff_boundaries() {
        for f in [Falloff::Linear, Falloff::Smooth, Falloff::Gaussian] {
            assert!((f.evaluate(0.0) - 1.0).abs() < TOL, "{f:?} at t=0");
            assert!(f.evaluate(1.0).abs() < TOL, "{f:?} at t=1");
            assert_eq!(f.evaluate(1.5), 0.0, "{f:?} beyond radius");
            assert_eq!(f.evaluate(-0.5), 1.0, "{f:?} before center");
        }
    }

    #[test]
    fn falloff_monotonic_decreasing() {
        for f in [Falloff::Linear, Falloff::Smooth, Falloff::Gaussian] {
            let mut prev = f.evaluate(0.0);
            for i in 1..=10 {
                let t = i as f32 / 10.0;
                let w = f.evaluate(t);
                assert!(w <= prev + 1e-6, "{f:?} should be non-increasing");
                prev = w;
            }
        }
    }

    // =========================================================================
    // LatticeDeform tests
    // =========================================================================

    #[test]
    fn lattice_identity_leaves_points_unchanged() {
        let bounds = Rect::new(Vec2::ZERO, Vec2::new(10.0, 10.0));
        let lattice = LatticeDeform::new(3, 3, bounds);

        let points = [
            Vec2::new(0.0, 0.0),
            Vec2::new(5.0, 5.0),
            Vec2::new(2.5, 7.5),
            Vec2::new(10.0, 10.0),
        ];
        let deformed = lattice.apply(&points);

        for (p, d) in points.iter().zip(deformed.iter()) {
            assert!(approx(*p, *d, TOL), "expected {p:?}, got {d:?}");
        }
    }

    #[test]
    fn lattice_displacing_corner_moves_nearby_points() {
        let bounds = Rect::new(Vec2::ZERO, Vec2::new(10.0, 10.0));
        let mut lattice = LatticeDeform::new(2, 2, bounds);
        // Move the (max, max) corner node (index 3 in a 2x2 row-major grid).
        lattice.control_points[3] += Vec2::new(4.0, 0.0);

        // Corner itself should be fully displaced.
        let corner = lattice.apply(&[Vec2::new(10.0, 10.0)])[0];
        assert!(approx(corner, Vec2::new(14.0, 10.0), TOL));

        // Opposite corner should be unaffected.
        let opposite = lattice.apply(&[Vec2::new(0.0, 0.0)])[0];
        assert!(approx(opposite, Vec2::new(0.0, 0.0), TOL));

        // Center should be partially displaced (bilinear midpoint).
        let center = lattice.apply(&[Vec2::new(5.0, 5.0)])[0];
        assert!(approx(center, Vec2::new(6.0, 5.0), TOL));
    }

    #[test]
    fn lattice_invalid_config_is_identity() {
        let bounds = Rect::new(Vec2::ZERO, Vec2::new(10.0, 10.0));
        let lattice = LatticeDeform {
            rows: 3,
            cols: 3,
            control_points: vec![Vec2::ZERO; 4], // mismatched length
            bounds,
        };
        let p = Vec2::new(3.0, 4.0);
        assert!(approx(lattice.apply(&[p])[0], p, TOL));
    }

    #[test]
    fn lattice_apply_to_path() {
        let bounds = Rect::new(Vec2::ZERO, Vec2::new(10.0, 10.0));
        let lattice = LatticeDeform::new(2, 2, bounds);
        let path = rect(Vec2::new(2.0, 2.0), Vec2::new(8.0, 8.0));
        let warped = lattice.apply_to_path(&path);
        assert_eq!(warped.commands().len(), path.commands().len());
    }

    // =========================================================================
    // CageDeform tests
    // =========================================================================

    #[test]
    fn cage_identity_leaves_points_unchanged() {
        let square = vec![
            Vec2::new(0.0, 0.0),
            Vec2::new(1.0, 0.0),
            Vec2::new(1.0, 1.0),
            Vec2::new(0.0, 1.0),
        ];
        let cage = CageDeform::new(square.clone(), square);

        let p = Vec2::new(0.3, 0.7);
        let deformed = cage.apply(&[p])[0];
        assert!(approx(deformed, p, TOL));
    }

    #[test]
    fn cage_scaling_square_scales_center() {
        let source = vec![
            Vec2::new(0.0, 0.0),
            Vec2::new(1.0, 0.0),
            Vec2::new(1.0, 1.0),
            Vec2::new(0.0, 1.0),
        ];
        let target = vec![
            Vec2::new(0.0, 0.0),
            Vec2::new(2.0, 0.0),
            Vec2::new(2.0, 2.0),
            Vec2::new(0.0, 2.0),
        ];
        let cage = CageDeform::new(source, target);

        // The center of a square has mean value coordinates (0.25, 0.25, 0.25, 0.25)
        // by symmetry, so it maps to the average of the target cage vertices.
        let center = cage.apply(&[Vec2::new(0.5, 0.5)])[0];
        assert!(approx(center, Vec2::new(1.0, 1.0), 0.01), "got {center:?}");
    }

    #[test]
    fn cage_vertex_maps_to_target_vertex() {
        let source = vec![
            Vec2::new(0.0, 0.0),
            Vec2::new(1.0, 0.0),
            Vec2::new(1.0, 1.0),
            Vec2::new(0.0, 1.0),
        ];
        let target = vec![
            Vec2::new(-1.0, -1.0),
            Vec2::new(3.0, 0.0),
            Vec2::new(2.0, 3.0),
            Vec2::new(-2.0, 2.0),
        ];
        let cage = CageDeform::new(source.clone(), target.clone());

        for (s, t) in source.iter().zip(target.iter()) {
            let deformed = cage.apply(&[*s])[0];
            assert!(
                approx(deformed, *t, TOL),
                "vertex {s:?} -> expected {t:?}, got {deformed:?}"
            );
        }
    }

    #[test]
    fn cage_invalid_config_is_identity() {
        let cage = CageDeform::new(
            vec![Vec2::ZERO, Vec2::X],
            vec![Vec2::ZERO, Vec2::X, Vec2::Y],
        );
        let p = Vec2::new(1.0, 2.0);
        assert!(approx(cage.apply(&[p])[0], p, TOL));
    }

    // =========================================================================
    // PointPush tests
    // =========================================================================

    #[test]
    fn point_push_full_effect_at_center() {
        let push = PointPush::new(Vec2::ZERO, Vec2::new(10.0, 0.0), 5.0, Falloff::Linear);
        let deformed = push.apply(&[Vec2::ZERO])[0];
        assert!(approx(deformed, Vec2::new(10.0, 0.0), TOL));
    }

    #[test]
    fn point_push_no_effect_outside_radius() {
        let push = PointPush::new(Vec2::ZERO, Vec2::new(10.0, 0.0), 5.0, Falloff::Linear);
        let p = Vec2::new(10.0, 0.0);
        let deformed = push.apply(&[p])[0];
        assert!(approx(deformed, p, TOL));
    }

    #[test]
    fn point_push_partial_effect_at_half_radius() {
        let push = PointPush::new(Vec2::ZERO, Vec2::new(10.0, 0.0), 10.0, Falloff::Linear);
        let p = Vec2::new(5.0, 0.0);
        let deformed = push.apply(&[p])[0];
        // t = 0.5, linear falloff weight = 0.5, displacement = (5, 0)
        assert!(approx(deformed, Vec2::new(10.0, 0.0), TOL));
    }

    // =========================================================================
    // Twist tests
    // =========================================================================

    #[test]
    fn twist_no_rotation_at_zero_angle() {
        let twist = Twist::new(Vec2::ZERO, 0.0, 10.0, Falloff::Linear);
        let p = Vec2::new(5.0, 0.0);
        assert!(approx(twist.apply(&[p])[0], p, TOL));
    }

    #[test]
    fn twist_no_effect_outside_radius() {
        let twist = Twist::new(
            Vec2::ZERO,
            std::f32::consts::FRAC_PI_2,
            5.0,
            Falloff::Linear,
        );
        let p = Vec2::new(10.0, 0.0);
        assert!(approx(twist.apply(&[p])[0], p, TOL));
    }

    #[test]
    fn twist_rotates_by_falloff_scaled_angle() {
        let twist = Twist::new(
            Vec2::ZERO,
            std::f32::consts::FRAC_PI_2,
            10.0,
            Falloff::Linear,
        );
        let p = Vec2::new(5.0, 0.0);
        // t = 0.5, linear falloff weight = 0.5, applied_angle = PI/4
        let deformed = twist.apply(&[p])[0];
        let expected = Vec2::new(
            5.0 * std::f32::consts::FRAC_PI_4.cos(),
            5.0 * std::f32::consts::FRAC_PI_4.sin(),
        );
        assert!(approx(deformed, expected, TOL), "got {deformed:?}");
    }

    // =========================================================================
    // Taper tests
    // =========================================================================

    #[test]
    fn taper_no_scale_at_start() {
        let taper = Taper::new(Vec2::X, Vec2::ZERO, 1.0, 2.0, 10.0);
        let p = Vec2::new(0.0, 3.0);
        assert!(approx(taper.apply(&[p])[0], p, TOL));
    }

    #[test]
    fn taper_full_scale_at_end() {
        let taper = Taper::new(Vec2::X, Vec2::ZERO, 1.0, 2.0, 10.0);
        let p = Vec2::new(10.0, 3.0);
        assert!(approx(taper.apply(&[p])[0], Vec2::new(10.0, 6.0), TOL));
    }

    #[test]
    fn taper_midpoint_interpolates() {
        let taper = Taper::new(Vec2::X, Vec2::ZERO, 1.0, 2.0, 10.0);
        let p = Vec2::new(5.0, 4.0);
        // t = 0.5, scale = 1.5
        assert!(approx(taper.apply(&[p])[0], Vec2::new(5.0, 6.0), TOL));
    }

    #[test]
    fn taper_clamps_beyond_length() {
        let taper = Taper::new(Vec2::X, Vec2::ZERO, 1.0, 2.0, 10.0);
        let p = Vec2::new(20.0, 3.0);
        assert!(approx(taper.apply(&[p])[0], Vec2::new(20.0, 6.0), TOL));
    }

    #[test]
    fn taper_apply_to_path() {
        let taper = Taper::new(Vec2::X, Vec2::ZERO, 1.0, 2.0, 10.0);
        let path = line(Vec2::new(0.0, 1.0), Vec2::new(10.0, 1.0));
        let warped = taper.apply_to_path(&path);
        assert_eq!(warped.commands().len(), path.commands().len());
    }
}
