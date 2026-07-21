//! Vector warp/deform operations.
//!
//! These ops displace points (and, via [`Path::transform`], whole paths) using
//! spatial deformation techniques: lattice-based free-form deformation, cage
//! (mean value coordinates) deformation, brush-style pushes, twists, tapers,
//! arc bends, radial bulges, sinusoidal waves, envelope fitting,
//! spine-following path deform, lens-style spherize/pincushion distortion,
//! deterministic roughening, centroid-relative pucker/bloat, and noise-based
//! field displacement.
//!
//! Every warp op maps an input position to an output position without
//! depending on neighbors, global state, or topology, so each one is a
//! `Field<Vec2, Vec2>` (`unshape-field-ops`, behind the `field` feature) —
//! the op struct itself is a field constructor, not a standalone
//! abstraction. `apply`/`apply_to_path` are always available (no feature
//! required) and are sugar that calls the same per-point logic as `sample`.
//! [`PuckerBloat`] and [`Roughen`] are noted exceptions: see their docs.
//!
//! [`Twist`], [`Wave`], [`Bend`], [`Bulge`], [`PointPush`], [`Taper`],
//! [`Spherize`], and [`PuckerBloat`] are closed-form formulas over a few
//! parameters, so they're also expression-decomposable: each has a
//! `to_dew_ast()` (behind the `wick` feature) that compiles the same
//! formula their `Field` impl evaluates into a `wick_core::Ast`, for
//! GPU/JIT execution. [`LatticeDeform`], [`CageDeform`], [`EnvelopeDeform`],
//! and [`PathDeform`] depend on variable-length authored data (control
//! grids, cages, boundary polylines, spine curves) and so are genuine
//! primitives — they implement `Field` but have no `to_dew_ast()`.

use crate::{Path, Rect};
use glam::Vec2;

#[cfg(feature = "field")]
use unshape_field_ops::{EvalContext, Field};

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

/// Dew AST construction helpers shared by the warp ops' `to_dew_ast`
/// methods.
///
/// The convention (matching `unshape-image`'s `UvExpr`/`ColorExpr` and
/// `unshape-expr-field`'s `FieldExpr`) is that the generated AST expects a
/// single free variable, `p` (a `Vec2`, the point being deformed), and
/// evaluates to the deformed `Vec2` position. Op parameters (center, radius,
/// angle, ...) are plain `f32`/`Vec2` fields on the struct, not
/// sub-expressions, so they're folded directly into `Ast::Num`/`vec2(...)`
/// literals rather than threaded through as additional variables.
#[cfg(feature = "wick")]
mod dew_ast {
    use super::Falloff;
    use glam::Vec2;
    use wick_core::{Ast, BinOp, CompareOp};

    pub(crate) fn var_p() -> Ast {
        Ast::Var("p".into())
    }

    pub(crate) fn num(v: f32) -> Ast {
        Ast::Num(v as f64)
    }

    pub(crate) fn vec2_const(v: Vec2) -> Ast {
        Ast::Call("vec2".into(), vec![num(v.x), num(v.y)])
    }

    pub(crate) fn call1(name: &str, a: Ast) -> Ast {
        Ast::Call(name.into(), vec![a])
    }

    pub(crate) fn call2(name: &str, a: Ast, b: Ast) -> Ast {
        Ast::Call(name.into(), vec![a, b])
    }

    pub(crate) fn call3(name: &str, a: Ast, b: Ast, c: Ast) -> Ast {
        Ast::Call(name.into(), vec![a, b, c])
    }

    pub(crate) fn add(a: Ast, b: Ast) -> Ast {
        Ast::BinOp(BinOp::Add, Box::new(a), Box::new(b))
    }

    pub(crate) fn sub(a: Ast, b: Ast) -> Ast {
        Ast::BinOp(BinOp::Sub, Box::new(a), Box::new(b))
    }

    pub(crate) fn mul(a: Ast, b: Ast) -> Ast {
        Ast::BinOp(BinOp::Mul, Box::new(a), Box::new(b))
    }

    pub(crate) fn div(a: Ast, b: Ast) -> Ast {
        Ast::BinOp(BinOp::Div, Box::new(a), Box::new(b))
    }

    pub(crate) fn ge(a: Ast, b: Ast) -> Ast {
        Ast::Compare(CompareOp::Ge, Box::new(a), Box::new(b))
    }

    pub(crate) fn lt(a: Ast, b: Ast) -> Ast {
        Ast::Compare(CompareOp::Lt, Box::new(a), Box::new(b))
    }

    pub(crate) fn or(a: Ast, b: Ast) -> Ast {
        Ast::Or(Box::new(a), Box::new(b))
    }

    pub(crate) fn if_then_else(cond: Ast, then_branch: Ast, else_branch: Ast) -> Ast {
        Ast::If(Box::new(cond), Box::new(then_branch), Box::new(else_branch))
    }

    /// Translates [`Falloff::evaluate`] into an AST fragment.
    ///
    /// `t` must be an AST that evaluates within `[0, 1)` — callers guard the
    /// surrounding formula with a runtime `dist < radius` check (or fold the
    /// identity case away entirely when `radius <= 0.0` is known at
    /// `to_dew_ast` build time), so the `t >= 1` / `t <= 0` clamped branches
    /// of `evaluate` never apply here. All three variants' raw formulas
    /// already equal 1 at `t == 0`, matching `evaluate`'s `t <= 0` case, so
    /// no extra clamping is needed.
    pub(crate) fn falloff_ast(falloff: Falloff, t: Ast) -> Ast {
        match falloff {
            Falloff::Linear => sub(num(1.0), t),
            Falloff::Smooth => {
                let s = sub(num(1.0), t);
                let s_sq = mul(s.clone(), s.clone());
                mul(s_sq, sub(num(3.0), mul(num(2.0), s)))
            }
            Falloff::Gaussian => call1("exp", mul(num(-4.5), mul(t.clone(), t))),
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

#[cfg(feature = "field")]
impl Field<Vec2, Vec2> for LatticeDeform {
    fn sample(&self, input: Vec2, _ctx: &EvalContext) -> Vec2 {
        self.deform_point(input)
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

#[cfg(feature = "field")]
impl Field<Vec2, Vec2> for CageDeform {
    fn sample(&self, input: Vec2, _ctx: &EvalContext) -> Vec2 {
        self.deform_point(input)
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

    /// Converts this push to a Dew AST for JIT/WGSL compilation.
    ///
    /// The resulting AST expects a `p` Vec2 variable (the point being
    /// deformed) and returns the deformed Vec2 position.
    #[cfg(feature = "wick")]
    pub fn to_dew_ast(&self) -> wick_core::Ast {
        use dew_ast::*;

        if self.radius <= 0.0 {
            return var_p();
        }

        let p = var_p();
        let offset = sub(p.clone(), vec2_const(self.center));
        let dist = call1("length", offset);
        let t = div(dist.clone(), num(self.radius));
        let weight = falloff_ast(self.falloff, t);
        let result = add(p.clone(), mul(vec2_const(self.direction), weight));

        if_then_else(ge(dist, num(self.radius)), p, result)
    }
}

#[cfg(feature = "field")]
impl Field<Vec2, Vec2> for PointPush {
    fn sample(&self, input: Vec2, _ctx: &EvalContext) -> Vec2 {
        self.deform_point(input)
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

    /// Converts this twist to a Dew AST for JIT/WGSL compilation.
    ///
    /// The resulting AST expects a `p` Vec2 variable (the point being
    /// deformed) and returns the deformed Vec2 position.
    #[cfg(feature = "wick")]
    pub fn to_dew_ast(&self) -> wick_core::Ast {
        use dew_ast::*;

        if self.radius <= 0.0 {
            return var_p();
        }

        let p = var_p();
        let center = vec2_const(self.center);
        let offset = sub(p.clone(), center.clone());
        let dist = call1("length", offset.clone());
        let t = div(dist.clone(), num(self.radius));
        let weight = falloff_ast(self.falloff, t);
        let applied_angle = mul(num(self.angle), weight);
        let rotated = call2("rotate2d", offset, applied_angle);
        let result = add(center, rotated);

        if_then_else(ge(dist, num(self.radius)), p, result)
    }
}

#[cfg(feature = "field")]
impl Field<Vec2, Vec2> for Twist {
    fn sample(&self, input: Vec2, _ctx: &EvalContext) -> Vec2 {
        self.deform_point(input)
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

    /// Converts this taper to a Dew AST for JIT/WGSL compilation.
    ///
    /// The resulting AST expects a `p` Vec2 variable (the point being
    /// deformed) and returns the deformed Vec2 position. `axis` is
    /// normalized at build time (it's a constant field, not a runtime
    /// value), so the AST embeds the unit axis directly.
    #[cfg(feature = "wick")]
    pub fn to_dew_ast(&self) -> wick_core::Ast {
        use dew_ast::*;

        let axis_len = self.axis.length();
        if axis_len < EPS || self.length <= 0.0 {
            return var_p();
        }
        let axis_n = self.axis / axis_len;

        let p = var_p();
        let origin = vec2_const(self.origin);
        let rel = sub(p, origin.clone());
        let along = call2("dot", rel.clone(), vec2_const(axis_n));
        let perp = sub(rel, mul(vec2_const(axis_n), along.clone()));

        let t = call3(
            "clamp",
            div(along.clone(), num(self.length)),
            num(0.0),
            num(1.0),
        );
        let scale = add(
            num(self.start_scale),
            mul(num(self.end_scale - self.start_scale), t),
        );

        add(
            add(origin, mul(vec2_const(axis_n), along)),
            mul(perp, scale),
        )
    }
}

#[cfg(feature = "field")]
impl Field<Vec2, Vec2> for Taper {
    fn sample(&self, input: Vec2, _ctx: &EvalContext) -> Vec2 {
        self.deform_point(input)
    }
}

// ============================================================================
// Bend
// ============================================================================

/// Curves geometry along an arc: points along `axis` (from `origin`) are
/// rotated proportionally to their distance from `origin`, up to `angle` at
/// `length`. Points beyond `length` clamp to the full angle.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dynop", derive(unshape_op::Op))]
#[cfg_attr(feature = "dynop", op(input = Vec<Vec2>, output = Vec<Vec2>))]
pub struct Bend {
    /// Origin of the bend (the pivot at zero rotation).
    pub origin: Vec2,
    /// Direction along which the bend angle accumulates.
    pub axis: Vec2,
    /// Total bend angle (radians), reached at `length` along the axis.
    pub angle: f32,
    /// Distance along the axis over which the full angle accumulates.
    pub length: f32,
}

impl Bend {
    /// Creates a new bend.
    pub fn new(origin: Vec2, axis: Vec2, angle: f32, length: f32) -> Self {
        Self {
            origin,
            axis,
            angle,
            length,
        }
    }

    fn deform_point(&self, p: Vec2) -> Vec2 {
        let axis_len = self.axis.length();
        if axis_len < EPS || self.length <= 0.0 {
            return p;
        }
        let axis_n = self.axis / axis_len;
        let perp_n = Vec2::new(-axis_n.y, axis_n.x);
        let rel = p - self.origin;
        let along = rel.dot(axis_n);
        let perp = rel.dot(perp_n);

        let t = (along / self.length).clamp(0.0, 1.0);
        let applied_angle = self.angle * t;

        // Bend around a pivot at distance `radius = length / angle` from the
        // axis, so points at `along = length` land at the full angle with
        // their perpendicular offset preserved as radial distance.
        if self.angle.abs() < EPS {
            return p;
        }
        let radius = self.length / self.angle;
        let arc_radius = radius - perp;
        let cos = applied_angle.cos();
        let sin = applied_angle.sin();
        self.origin + axis_n * (arc_radius * sin) + perp_n * (radius - arc_radius * cos)
    }

    /// Applies the bend to a set of points.
    pub fn apply(&self, points: &[Vec2]) -> Vec<Vec2> {
        points.iter().map(|&p| self.deform_point(p)).collect()
    }

    /// Applies the bend to every point of a path.
    pub fn apply_to_path(&self, path: &Path) -> Path {
        let mut result = path.clone();
        result.transform(|p| self.deform_point(p));
        result
    }

    /// Converts this bend to a Dew AST for JIT/WGSL compilation.
    ///
    /// The resulting AST expects a `p` Vec2 variable (the point being
    /// deformed) and returns the deformed Vec2 position. `axis` and the
    /// pivot `radius = length / angle` are constant fields, so they're
    /// normalized/computed at build time and embedded as literals.
    #[cfg(feature = "wick")]
    pub fn to_dew_ast(&self) -> wick_core::Ast {
        use dew_ast::*;

        let axis_len = self.axis.length();
        if axis_len < EPS || self.length <= 0.0 || self.angle.abs() < EPS {
            return var_p();
        }
        let axis_n = self.axis / axis_len;
        let perp_n = Vec2::new(-axis_n.y, axis_n.x);
        let radius = self.length / self.angle;

        let p = var_p();
        let origin = vec2_const(self.origin);
        let rel = sub(p, origin.clone());
        let along = call2("dot", rel.clone(), vec2_const(axis_n));
        let perp = call2("dot", rel, vec2_const(perp_n));

        let t = call3("clamp", div(along, num(self.length)), num(0.0), num(1.0));
        let applied_angle = mul(num(self.angle), t);
        let arc_radius = sub(num(radius), perp);
        let cos = call1("cos", applied_angle.clone());
        let sin = call1("sin", applied_angle);

        let axis_term = mul(vec2_const(axis_n), mul(arc_radius.clone(), sin));
        let perp_term = mul(vec2_const(perp_n), sub(num(radius), mul(arc_radius, cos)));

        add(add(origin, axis_term), perp_term)
    }
}

#[cfg(feature = "field")]
impl Field<Vec2, Vec2> for Bend {
    fn sample(&self, input: Vec2, _ctx: &EvalContext) -> Vec2 {
        self.deform_point(input)
    }
}

// ============================================================================
// Bulge
// ============================================================================

/// Radially scales points towards/away from `center`: points within `radius`
/// are displaced radially by `strength * falloff_weight` (positive inflates,
/// negative pinches).
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dynop", derive(unshape_op::Op))]
#[cfg_attr(feature = "dynop", op(input = Vec<Vec2>, output = Vec<Vec2>))]
pub struct Bulge {
    /// Center of the bulge.
    pub center: Vec2,
    /// Radius of effect; points beyond this distance are unaffected.
    pub radius: f32,
    /// Radial displacement magnitude at the center of effect (positive =
    /// inflate outward, negative = pinch inward).
    pub strength: f32,
    /// Attenuation curve from center (full effect) to radius (no effect).
    pub falloff: Falloff,
}

impl Bulge {
    /// Creates a new bulge.
    pub fn new(center: Vec2, radius: f32, strength: f32, falloff: Falloff) -> Self {
        Self {
            center,
            radius,
            strength,
            falloff,
        }
    }

    fn deform_point(&self, p: Vec2) -> Vec2 {
        if self.radius <= 0.0 {
            return p;
        }
        let offset = p - self.center;
        let dist = offset.length();
        if dist >= self.radius || dist < EPS {
            return p;
        }
        let t = dist / self.radius;
        let w = self.falloff.evaluate(t);
        let dir = offset / dist;
        p + dir * (self.strength * w)
    }

    /// Applies the bulge to a set of points.
    pub fn apply(&self, points: &[Vec2]) -> Vec<Vec2> {
        points.iter().map(|&p| self.deform_point(p)).collect()
    }

    /// Applies the bulge to every point of a path.
    pub fn apply_to_path(&self, path: &Path) -> Path {
        let mut result = path.clone();
        result.transform(|p| self.deform_point(p));
        result
    }

    /// Converts this bulge to a Dew AST for JIT/WGSL compilation.
    ///
    /// The resulting AST expects a `p` Vec2 variable (the point being
    /// deformed) and returns the deformed Vec2 position.
    #[cfg(feature = "wick")]
    pub fn to_dew_ast(&self) -> wick_core::Ast {
        use dew_ast::*;

        if self.radius <= 0.0 {
            return var_p();
        }

        let p = var_p();
        let offset = sub(p.clone(), vec2_const(self.center));
        let dist = call1("length", offset.clone());
        let t = div(dist.clone(), num(self.radius));
        let weight = falloff_ast(self.falloff, t);
        let dir = div(offset, dist.clone());
        let result = add(p.clone(), mul(dir, mul(num(self.strength), weight)));

        let cond = or(ge(dist.clone(), num(self.radius)), lt(dist, num(EPS)));
        if_then_else(cond, p, result)
    }
}

#[cfg(feature = "field")]
impl Field<Vec2, Vec2> for Bulge {
    fn sample(&self, input: Vec2, _ctx: &EvalContext) -> Vec2 {
        self.deform_point(input)
    }
}

// ============================================================================
// Wave
// ============================================================================

/// Sinusoidally displaces points perpendicular to `axis`, by
/// `amplitude * sin(frequency * distance_along_axis + phase)`.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dynop", derive(unshape_op::Op))]
#[cfg_attr(feature = "dynop", op(input = Vec<Vec2>, output = Vec<Vec2>))]
pub struct Wave {
    /// Direction of wave propagation.
    pub axis: Vec2,
    /// Peak displacement perpendicular to the axis.
    pub amplitude: f32,
    /// Spatial frequency (radians per unit distance along the axis).
    pub frequency: f32,
    /// Phase offset (radians).
    pub phase: f32,
}

impl Wave {
    /// Creates a new wave.
    pub fn new(axis: Vec2, amplitude: f32, frequency: f32, phase: f32) -> Self {
        Self {
            axis,
            amplitude,
            frequency,
            phase,
        }
    }

    fn deform_point(&self, p: Vec2) -> Vec2 {
        let axis_len = self.axis.length();
        if axis_len < EPS {
            return p;
        }
        let axis_n = self.axis / axis_len;
        let perp_n = Vec2::new(-axis_n.y, axis_n.x);
        let along = p.dot(axis_n);
        let displacement = self.amplitude * (self.frequency * along + self.phase).sin();
        p + perp_n * displacement
    }

    /// Applies the wave to a set of points.
    pub fn apply(&self, points: &[Vec2]) -> Vec<Vec2> {
        points.iter().map(|&p| self.deform_point(p)).collect()
    }

    /// Applies the wave to every point of a path.
    pub fn apply_to_path(&self, path: &Path) -> Path {
        let mut result = path.clone();
        result.transform(|p| self.deform_point(p));
        result
    }

    /// Converts this wave to a Dew AST for JIT/WGSL compilation.
    ///
    /// The resulting AST expects a `p` Vec2 variable (the point being
    /// deformed) and returns the deformed Vec2 position. `axis` is
    /// normalized at build time (it's a constant field, not a runtime
    /// value), so the AST embeds the unit axis and its perpendicular
    /// directly.
    #[cfg(feature = "wick")]
    pub fn to_dew_ast(&self) -> wick_core::Ast {
        use dew_ast::*;

        let axis_len = self.axis.length();
        if axis_len < EPS {
            return var_p();
        }
        let axis_n = self.axis / axis_len;
        let perp_n = Vec2::new(-axis_n.y, axis_n.x);

        let p = var_p();
        let along = call2("dot", p.clone(), vec2_const(axis_n));
        let inner = add(mul(num(self.frequency), along), num(self.phase));
        let displacement = mul(num(self.amplitude), call1("sin", inner));

        add(p, mul(vec2_const(perp_n), displacement))
    }
}

#[cfg(feature = "field")]
impl Field<Vec2, Vec2> for Wave {
    fn sample(&self, input: Vec2, _ctx: &EvalContext) -> Vec2 {
        self.deform_point(input)
    }
}

// ============================================================================
// EnvelopeDeform
// ============================================================================

/// Fits geometry between two boundary polylines: `source_top`/`source_bottom`
/// describe the source envelope, `target_top`/`target_bottom` the target.
/// Each point is expressed in normalized envelope coordinates `(u, v)` — `u`
/// the fraction along the envelope's length, `v` the fraction from bottom
/// (0) to top (1) — with respect to the source envelope, then re-projected
/// into the target envelope at the same `(u, v)`.
///
/// `source_top`, `source_bottom`, `target_top`, and `target_bottom` must each
/// have at least 2 points; otherwise points pass through unchanged.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dynop", derive(unshape_op::Op))]
#[cfg_attr(feature = "dynop", op(input = Vec<Vec2>, output = Vec<Vec2>))]
pub struct EnvelopeDeform {
    /// Top boundary of the source envelope, in order of increasing `u`.
    pub source_top: Vec<Vec2>,
    /// Bottom boundary of the source envelope, in order of increasing `u`.
    pub source_bottom: Vec<Vec2>,
    /// Top boundary of the target envelope, in order of increasing `u`.
    pub target_top: Vec<Vec2>,
    /// Bottom boundary of the target envelope, in order of increasing `u`.
    pub target_bottom: Vec<Vec2>,
}

impl EnvelopeDeform {
    /// Creates a new envelope deform from matching source/target boundaries.
    pub fn new(
        source_top: Vec<Vec2>,
        source_bottom: Vec<Vec2>,
        target_top: Vec<Vec2>,
        target_bottom: Vec<Vec2>,
    ) -> Self {
        Self {
            source_top,
            source_bottom,
            target_top,
            target_bottom,
        }
    }

    fn is_valid(&self) -> bool {
        self.source_top.len() >= 2
            && self.source_bottom.len() >= 2
            && self.target_top.len() >= 2
            && self.target_bottom.len() >= 2
    }

    /// Finds the bounding box of `source_top` and `source_bottom` combined,
    /// used to establish the source envelope's `(u, v)` parameterization.
    fn source_bounds(&self) -> Option<Rect> {
        let mut points = self.source_top.iter().chain(self.source_bottom.iter());
        let first = *points.next()?;
        let mut min = first;
        let mut max = first;
        for &p in points {
            min = min.min(p);
            max = max.max(p);
        }
        Some(Rect::new(min, max))
    }

    fn deform_point(&self, p: Vec2) -> Vec2 {
        if !self.is_valid() {
            return p;
        }
        let Some(bounds) = self.source_bounds() else {
            return p;
        };
        let size = bounds.max - bounds.min;
        let u = if size.x.abs() < EPS {
            0.0
        } else {
            ((p.x - bounds.min.x) / size.x).clamp(0.0, 1.0)
        };
        let v = if size.y.abs() < EPS {
            0.0
        } else {
            ((p.y - bounds.min.y) / size.y).clamp(0.0, 1.0)
        };

        let top = sample_polyline(&self.target_top, u);
        let bottom = sample_polyline(&self.target_bottom, u);
        bottom.lerp(top, v)
    }

    /// Applies the envelope deformation to a set of points.
    pub fn apply(&self, points: &[Vec2]) -> Vec<Vec2> {
        points.iter().map(|&p| self.deform_point(p)).collect()
    }

    /// Applies the envelope deformation to every point of a path.
    pub fn apply_to_path(&self, path: &Path) -> Path {
        let mut result = path.clone();
        result.transform(|p| self.deform_point(p));
        result
    }
}

#[cfg(feature = "field")]
impl Field<Vec2, Vec2> for EnvelopeDeform {
    fn sample(&self, input: Vec2, _ctx: &EvalContext) -> Vec2 {
        self.deform_point(input)
    }
}

// ============================================================================
// PathDeform
// ============================================================================

/// Wraps geometry along a spine curve: a point's signed distance along
/// `original_axis` (from `origin`) maps to arc-length distance along
/// `spine`, and its perpendicular offset from the axis is preserved relative
/// to the spine's local normal at that point.
///
/// `spine` must have at least 2 points and `original_axis` must be non-zero;
/// otherwise points pass through unchanged.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dynop", derive(unshape_op::Op))]
#[cfg_attr(feature = "dynop", op(input = Vec<Vec2>, output = Vec<Vec2>))]
pub struct PathDeform {
    /// The spine curve geometry is wrapped onto, as a polyline.
    pub spine: Vec<Vec2>,
    /// The straight axis (in source space) that maps onto the spine.
    pub original_axis: Vec2,
    /// Origin of the source axis (maps to the start of the spine).
    pub origin: Vec2,
}

impl PathDeform {
    /// Creates a new path deform.
    pub fn new(spine: Vec<Vec2>, original_axis: Vec2, origin: Vec2) -> Self {
        Self {
            spine,
            original_axis,
            origin,
        }
    }

    fn is_valid(&self) -> bool {
        self.spine.len() >= 2 && self.original_axis.length() >= EPS
    }

    /// Samples the spine at arc-length `dist` from its start, returning the
    /// point and unit tangent there. `dist` is clamped to the spine's
    /// extent.
    fn sample_spine(&self, dist: f32) -> (Vec2, Vec2) {
        let mut remaining = dist.max(0.0);
        for w in self.spine.windows(2) {
            let (a, b) = (w[0], w[1]);
            let seg = b - a;
            let seg_len = seg.length();
            if seg_len < EPS {
                continue;
            }
            if remaining <= seg_len {
                let t = remaining / seg_len;
                return (a.lerp(b, t), seg / seg_len);
            }
            remaining -= seg_len;
        }
        // Beyond the end: extrapolate along the final segment's tangent.
        let last = self.spine[self.spine.len() - 1];
        for w in self.spine.windows(2).rev() {
            let (a, b) = (w[0], w[1]);
            let seg = b - a;
            let seg_len = seg.length();
            if seg_len >= EPS {
                let tangent = seg / seg_len;
                return (last + tangent * remaining, tangent);
            }
        }
        (last, Vec2::X)
    }

    fn deform_point(&self, p: Vec2) -> Vec2 {
        if !self.is_valid() {
            return p;
        }
        let axis_len = self.original_axis.length();
        let axis_n = self.original_axis / axis_len;
        let perp_n = Vec2::new(-axis_n.y, axis_n.x);
        let rel = p - self.origin;
        let along = rel.dot(axis_n);
        let perp = rel.dot(perp_n);

        let (spine_point, tangent) = self.sample_spine(along);
        let normal = Vec2::new(-tangent.y, tangent.x);
        spine_point + normal * perp
    }

    /// Applies the path deform to a set of points.
    pub fn apply(&self, points: &[Vec2]) -> Vec<Vec2> {
        points.iter().map(|&p| self.deform_point(p)).collect()
    }

    /// Applies the path deform to every point of a path.
    pub fn apply_to_path(&self, path: &Path) -> Path {
        let mut result = path.clone();
        result.transform(|p| self.deform_point(p));
        result
    }
}

#[cfg(feature = "field")]
impl Field<Vec2, Vec2> for PathDeform {
    fn sample(&self, input: Vec2, _ctx: &EvalContext) -> Vec2 {
        self.deform_point(input)
    }
}

/// Samples a polyline at normalized parameter `u` in `[0, 1]` (0 = first
/// point, 1 = last point), linearly interpolating by point index (not
/// arc-length). `u` is clamped to `[0, 1]`. The polyline must have at least
/// 2 points.
fn sample_polyline(polyline: &[Vec2], u: f32) -> Vec2 {
    let n = polyline.len();
    let t = u.clamp(0.0, 1.0) * (n - 1) as f32;
    let i0 = (t.floor() as usize).min(n - 2);
    let i1 = i0 + 1;
    let local_t = t - i0 as f32;
    polyline[i0].lerp(polyline[i1], local_t)
}

/// Collects the anchor (on-curve endpoint) positions of a path's commands,
/// in order: `MoveTo`/`LineTo` points and the `to` endpoint of curve
/// commands. Control points and `Close` are excluded. Used to compute an
/// input-agnostic centroid for path-level warps like [`PuckerBloat`].
fn path_anchor_points(path: &Path) -> Vec<Vec2> {
    path.commands()
        .iter()
        .filter_map(|cmd| match *cmd {
            crate::PathCommand::MoveTo(p) => Some(p),
            crate::PathCommand::LineTo(p) => Some(p),
            crate::PathCommand::QuadTo { to, .. } => Some(to),
            crate::PathCommand::CubicTo { to, .. } => Some(to),
            crate::PathCommand::Close => None,
        })
        .collect()
}

// ============================================================================
// Spherize
// ============================================================================

/// Barrel/pincushion lens distortion: points within `radius` of `center` are
/// radially remapped through a nonlinear projection, `displaced_dist = dist *
/// (1 + strength * normalized_dist^2)` where `normalized_dist = dist /
/// radius`. Positive `strength` bulges outward like a barrel lens (spherize);
/// negative `strength` pulls inward like a pincushion lens.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dynop", derive(unshape_op::Op))]
#[cfg_attr(feature = "dynop", op(input = Vec<Vec2>, output = Vec<Vec2>))]
pub struct Spherize {
    /// Center of the lens effect.
    pub center: Vec2,
    /// Radius of effect; points beyond this distance are unaffected.
    pub radius: f32,
    /// Distortion strength (positive = barrel/spherize, negative = pincushion).
    pub strength: f32,
}

impl Spherize {
    /// Creates a new spherize.
    pub fn new(center: Vec2, radius: f32, strength: f32) -> Self {
        Self {
            center,
            radius,
            strength,
        }
    }

    fn deform_point(&self, p: Vec2) -> Vec2 {
        if self.radius <= 0.0 {
            return p;
        }
        let offset = p - self.center;
        let dist = offset.length();
        if dist >= self.radius || dist < EPS {
            return p;
        }
        let normalized = dist / self.radius;
        let displaced_dist = dist * (1.0 + self.strength * normalized * normalized);
        let dir = offset / dist;
        self.center + dir * displaced_dist
    }

    /// Applies the spherize to a set of points.
    pub fn apply(&self, points: &[Vec2]) -> Vec<Vec2> {
        points.iter().map(|&p| self.deform_point(p)).collect()
    }

    /// Applies the spherize to every point of a path.
    pub fn apply_to_path(&self, path: &Path) -> Path {
        let mut result = path.clone();
        result.transform(|p| self.deform_point(p));
        result
    }

    /// Converts this spherize to a Dew AST for JIT/WGSL compilation.
    ///
    /// The resulting AST expects a `p` Vec2 variable (the point being
    /// deformed) and returns the deformed Vec2 position.
    #[cfg(feature = "wick")]
    pub fn to_dew_ast(&self) -> wick_core::Ast {
        use dew_ast::*;

        if self.radius <= 0.0 {
            return var_p();
        }

        let p = var_p();
        let offset = sub(p.clone(), vec2_const(self.center));
        let dist = call1("length", offset.clone());
        let normalized = div(dist.clone(), num(self.radius));
        let displaced_dist = mul(
            dist.clone(),
            add(
                num(1.0),
                mul(num(self.strength), mul(normalized.clone(), normalized)),
            ),
        );
        let dir = div(offset, dist.clone());
        let result = add(vec2_const(self.center), mul(dir, displaced_dist));

        let cond = or(ge(dist.clone(), num(self.radius)), lt(dist, num(EPS)));
        if_then_else(cond, p, result)
    }
}

#[cfg(feature = "field")]
impl Field<Vec2, Vec2> for Spherize {
    fn sample(&self, input: Vec2, _ctx: &EvalContext) -> Vec2 {
        self.deform_point(input)
    }
}

// ============================================================================
// Roughen
// ============================================================================

/// Displaces each point by a deterministic pseudo-random offset, up to
/// `amplitude` in magnitude, derived from the point's index and `seed`. Given
/// the same seed and point count, the result is stable across runs.
///
/// `apply`/`apply_to_path` key the pseudo-random offset by point *index*, so
/// each point in the sequence gets an independent, order-stable value. Its
/// `Field<Vec2, Vec2>` impl can't see an index (a field is sampled at
/// arbitrary, unordered positions), so it hashes the input *position*
/// instead — this makes the field a pure function of location (two equal
/// points always displace identically) rather than of point order, which is
/// a different, still-deterministic notion of "random" from the indexed one
/// used by `apply`.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dynop", derive(unshape_op::Op))]
#[cfg_attr(feature = "dynop", op(input = Vec<Vec2>, output = Vec<Vec2>))]
pub struct Roughen {
    /// Maximum displacement magnitude.
    pub amplitude: f32,
    /// Seed for the deterministic pseudo-random displacement.
    pub seed: u32,
}

impl Roughen {
    /// Creates a new roughen.
    pub fn new(amplitude: f32, seed: u32) -> Self {
        Self { amplitude, seed }
    }

    fn deform_point(&self, index: usize, p: Vec2) -> Vec2 {
        if self.amplitude <= 0.0 {
            return p;
        }
        let h_angle = hash_to_unit(self.seed, index as u32, 0);
        let h_mag = hash_to_unit(self.seed, index as u32, 1);
        let angle = h_angle * std::f32::consts::TAU;
        let mag = h_mag * self.amplitude;
        p + Vec2::new(angle.cos(), angle.sin()) * mag
    }

    /// Position-hashed variant of [`Self::deform_point`] used by the
    /// `Field<Vec2, Vec2>` impl, which has no notion of point index. Hashes
    /// the input position's bit patterns instead, so the same position
    /// always displaces the same way.
    #[cfg(feature = "field")]
    fn deform_point_at_position(&self, p: Vec2) -> Vec2 {
        if self.amplitude <= 0.0 {
            return p;
        }
        let hx = p.x.to_bits();
        let hy = p.y.to_bits();
        let h_angle = hash_to_unit(self.seed, hx, hy);
        let h_mag = hash_to_unit(hx, hy, self.seed);
        let angle = h_angle * std::f32::consts::TAU;
        let mag = h_mag * self.amplitude;
        p + Vec2::new(angle.cos(), angle.sin()) * mag
    }

    /// Applies the roughening to a set of points.
    pub fn apply(&self, points: &[Vec2]) -> Vec<Vec2> {
        points
            .iter()
            .enumerate()
            .map(|(i, &p)| self.deform_point(i, p))
            .collect()
    }

    /// Applies the roughening to every point of a path.
    pub fn apply_to_path(&self, path: &Path) -> Path {
        let mut result = path.clone();
        let index = std::cell::Cell::new(0usize);
        result.transform(|p| {
            let i = index.get();
            index.set(i + 1);
            self.deform_point(i, p)
        });
        result
    }
}

#[cfg(feature = "field")]
impl Field<Vec2, Vec2> for Roughen {
    /// Uses position-based hashing (see [`Roughen::deform_point_at_position`]),
    /// not the index-based hashing `apply` uses — a field has no index, only
    /// a position to sample at.
    fn sample(&self, input: Vec2, _ctx: &EvalContext) -> Vec2 {
        self.deform_point_at_position(input)
    }
}

// ============================================================================
// PuckerBloat
// ============================================================================

/// Moves each point towards or away from the centroid of the input point set.
/// `apply`/`apply_to_path` compute the centroid fresh from whatever points
/// (or path anchors) are passed in — this makes `PuckerBloat` genuinely
/// *not* a field in that mode: the output at a point depends on every other
/// point in the set (global dependence), not just its own position.
/// Positive `strength` bloats points outward from the centroid; negative
/// `strength` puckers them inward.
///
/// The `Field<Vec2, Vec2>` impl can't recover a global centroid from a
/// single sampled position, so it uses `centroid` as a fixed parameter
/// instead of deriving it from input — turning the global dependence into
/// an explicit, stored value. `apply`/`apply_to_path` ignore this field and
/// keep computing their own centroid, so existing behavior is unchanged.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dynop", derive(unshape_op::Op))]
#[cfg_attr(feature = "dynop", op(input = Vec<Vec2>, output = Vec<Vec2>))]
pub struct PuckerBloat {
    /// Displacement factor along the centroid-to-point vector (negative =
    /// pucker inward, positive = bloat outward).
    pub strength: f32,
    /// Centroid used by the `Field<Vec2, Vec2>` impl only; `apply` and
    /// `apply_to_path` compute their own centroid from the input and ignore
    /// this field. Defaults to the origin.
    pub centroid: Vec2,
}

impl PuckerBloat {
    /// Creates a new pucker/bloat. `centroid` (used only by the
    /// `Field<Vec2, Vec2>` impl) defaults to the origin; set it with
    /// [`Self::with_centroid`].
    pub fn new(strength: f32) -> Self {
        Self {
            strength,
            centroid: Vec2::ZERO,
        }
    }

    /// Sets the fixed centroid used by the `Field<Vec2, Vec2>` impl.
    pub fn with_centroid(mut self, centroid: Vec2) -> Self {
        self.centroid = centroid;
        self
    }

    fn deform_point(&self, centroid: Vec2, p: Vec2) -> Vec2 {
        let offset = p - centroid;
        p + offset * self.strength
    }

    /// Applies the pucker/bloat to a set of points, using their own centroid.
    pub fn apply(&self, points: &[Vec2]) -> Vec<Vec2> {
        if points.is_empty() {
            return Vec::new();
        }
        let centroid = points.iter().fold(Vec2::ZERO, |acc, &p| acc + p) / points.len() as f32;
        points
            .iter()
            .map(|&p| self.deform_point(centroid, p))
            .collect()
    }

    /// Applies the pucker/bloat to every point of a path, using the centroid
    /// of the path's anchor points.
    pub fn apply_to_path(&self, path: &Path) -> Path {
        let anchors = path_anchor_points(path);
        if anchors.is_empty() {
            return path.clone();
        }
        let centroid = anchors.iter().fold(Vec2::ZERO, |acc, &p| acc + p) / anchors.len() as f32;
        let mut result = path.clone();
        result.transform(|p| self.deform_point(centroid, p));
        result
    }

    /// Converts this pucker/bloat to a Dew AST for JIT/WGSL compilation.
    ///
    /// The resulting AST expects a `p` Vec2 variable (the point being
    /// deformed) and returns the deformed Vec2 position, using `self.centroid`
    /// as a fixed parameter — matching the `Field<Vec2, Vec2>` impl, not
    /// `apply`/`apply_to_path` (which compute their own centroid from the
    /// input set; see the struct docs for why that mode isn't a field, and
    /// so can't be expression-decomposed here).
    #[cfg(feature = "wick")]
    pub fn to_dew_ast(&self) -> wick_core::Ast {
        use dew_ast::*;

        let p = var_p();
        let offset = sub(p.clone(), vec2_const(self.centroid));
        add(p, mul(offset, num(self.strength)))
    }
}

#[cfg(feature = "field")]
impl Field<Vec2, Vec2> for PuckerBloat {
    /// Uses `self.centroid` as a fixed parameter — see the struct docs for
    /// why this can't be the same global-centroid computation `apply` does.
    fn sample(&self, input: Vec2, _ctx: &EvalContext) -> Vec2 {
        self.deform_point(self.centroid, input)
    }
}

// ============================================================================
// FieldDisplace
// ============================================================================

/// Displaces points using deterministic value noise evaluated at
/// `position * frequency`, scaled independently per axis by `scale`. Self-
/// contained noise; composing with `unshape-field` for richer fields is a
/// future extension.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dynop", derive(unshape_op::Op))]
#[cfg_attr(feature = "dynop", op(input = Vec<Vec2>, output = Vec<Vec2>))]
pub struct FieldDisplace {
    /// Per-axis displacement magnitude.
    pub scale: Vec2,
    /// Spatial frequency of the noise field.
    pub frequency: f32,
    /// Seed for the deterministic noise field.
    pub seed: u32,
}

impl FieldDisplace {
    /// Creates a new field displace.
    pub fn new(scale: Vec2, frequency: f32, seed: u32) -> Self {
        Self {
            scale,
            frequency,
            seed,
        }
    }

    fn deform_point(&self, p: Vec2) -> Vec2 {
        let sample = p * self.frequency;
        let nx = value_noise_2d(sample, self.seed, 0) * 2.0 - 1.0;
        let ny = value_noise_2d(sample, self.seed, 1) * 2.0 - 1.0;
        p + Vec2::new(nx * self.scale.x, ny * self.scale.y)
    }

    /// Applies the field displacement to a set of points.
    pub fn apply(&self, points: &[Vec2]) -> Vec<Vec2> {
        points.iter().map(|&p| self.deform_point(p)).collect()
    }

    /// Applies the field displacement to every point of a path.
    pub fn apply_to_path(&self, path: &Path) -> Path {
        let mut result = path.clone();
        result.transform(|p| self.deform_point(p));
        result
    }
}

#[cfg(feature = "field")]
impl Field<Vec2, Vec2> for FieldDisplace {
    fn sample(&self, input: Vec2, _ctx: &EvalContext) -> Vec2 {
        self.deform_point(input)
    }
}

/// Hashes three `u32`s into a pseudo-random value in `[0, 1)`. Used for
/// deterministic per-index randomness (e.g. [`Roughen`]).
fn hash_to_unit(a: u32, b: u32, c: u32) -> f32 {
    let mut h = a
        .wrapping_mul(0x9e3779b1)
        .wrapping_add(b.wrapping_mul(0x85ebca6b))
        .wrapping_add(c.wrapping_mul(0xc2b2ae35));
    h ^= h >> 16;
    h = h.wrapping_mul(0x7feb352d);
    h ^= h >> 15;
    h = h.wrapping_mul(0x846ca68b);
    h ^= h >> 16;
    (h as f64 / u32::MAX as f64) as f32
}

/// Hashes an integer lattice coordinate `(x, y)` plus `seed` and `channel`
/// into a pseudo-random value in `[0, 1)`. Used as the value-noise lattice
/// generator for [`FieldDisplace`].
fn hash_lattice(x: i32, y: i32, seed: u32, channel: u32) -> f32 {
    hash_to_unit(
        (x as u32).wrapping_mul(0x27d4eb2d) ^ seed,
        (y as u32).wrapping_mul(0x165667b1),
        channel.wrapping_mul(0x9e3779b9),
    )
}

/// Smoothstep-interpolated 2D value noise at `p`, seeded by `seed` with an
/// independent `channel` (so multiple channels can be sampled at the same
/// position without correlating).
fn value_noise_2d(p: Vec2, seed: u32, channel: u32) -> f32 {
    let x0 = p.x.floor() as i32;
    let y0 = p.y.floor() as i32;
    let x1 = x0 + 1;
    let y1 = y0 + 1;

    let fx = p.x - x0 as f32;
    let fy = p.y - y0 as f32;
    let sx = fx * fx * (3.0 - 2.0 * fx);
    let sy = fy * fy * (3.0 - 2.0 * fy);

    let n00 = hash_lattice(x0, y0, seed, channel);
    let n10 = hash_lattice(x1, y0, seed, channel);
    let n01 = hash_lattice(x0, y1, seed, channel);
    let n11 = hash_lattice(x1, y1, seed, channel);

    let top = n00 + (n10 - n00) * sx;
    let bottom = n01 + (n11 - n01) * sx;
    top + (bottom - top) * sy
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

    // =========================================================================
    // Bend tests
    // =========================================================================

    #[test]
    fn bend_zero_angle_is_identity() {
        let bend = Bend::new(Vec2::ZERO, Vec2::X, 0.0, 10.0);
        let p = Vec2::new(4.0, 3.0);
        assert!(approx(bend.apply(&[p])[0], p, TOL));
    }

    #[test]
    fn bend_on_axis_start_is_identity() {
        let bend = Bend::new(Vec2::ZERO, Vec2::X, std::f32::consts::FRAC_PI_2, 10.0);
        let p = Vec2::new(0.0, 3.0);
        assert!(approx(bend.apply(&[p])[0], p, TOL));
    }

    #[test]
    fn bend_full_arc_matches_known_endpoint() {
        let angle = std::f32::consts::FRAC_PI_2;
        let bend = Bend::new(Vec2::ZERO, Vec2::X, angle, 10.0);
        let radius = 10.0 / angle;
        let deformed = bend.apply(&[Vec2::new(10.0, 0.0)])[0];
        assert!(
            approx(deformed, Vec2::new(radius, radius), TOL),
            "got {deformed:?}, expected ({radius}, {radius})"
        );
    }

    #[test]
    fn bend_apply_to_path() {
        let bend = Bend::new(Vec2::ZERO, Vec2::X, std::f32::consts::FRAC_PI_2, 10.0);
        let path = line(Vec2::new(0.0, 0.0), Vec2::new(10.0, 0.0));
        let warped = bend.apply_to_path(&path);
        assert_eq!(warped.commands().len(), path.commands().len());
    }

    // =========================================================================
    // Bulge tests
    // =========================================================================

    #[test]
    fn bulge_center_point_unaffected() {
        let bulge = Bulge::new(Vec2::ZERO, 10.0, 5.0, Falloff::Linear);
        let deformed = bulge.apply(&[Vec2::ZERO])[0];
        assert!(approx(deformed, Vec2::ZERO, TOL));
    }

    #[test]
    fn bulge_inflates_radially() {
        let bulge = Bulge::new(Vec2::ZERO, 10.0, 5.0, Falloff::Linear);
        let p = Vec2::new(4.0, 0.0);
        // t = 0.4, linear falloff weight = 0.6, displacement = 3.0 outward
        let deformed = bulge.apply(&[p])[0];
        assert!(
            approx(deformed, Vec2::new(7.0, 0.0), TOL),
            "got {deformed:?}"
        );
    }

    #[test]
    fn bulge_negative_strength_pinches() {
        let bulge = Bulge::new(Vec2::ZERO, 10.0, -5.0, Falloff::Linear);
        let p = Vec2::new(4.0, 0.0);
        let deformed = bulge.apply(&[p])[0];
        assert!(
            approx(deformed, Vec2::new(1.0, 0.0), TOL),
            "got {deformed:?}"
        );
    }

    #[test]
    fn bulge_no_effect_outside_radius() {
        let bulge = Bulge::new(Vec2::ZERO, 5.0, 10.0, Falloff::Linear);
        let p = Vec2::new(10.0, 0.0);
        assert!(approx(bulge.apply(&[p])[0], p, TOL));
    }

    #[test]
    fn bulge_apply_to_path() {
        let bulge = Bulge::new(Vec2::ZERO, 10.0, 5.0, Falloff::Linear);
        let path = rect(Vec2::new(-2.0, -2.0), Vec2::new(2.0, 2.0));
        let warped = bulge.apply_to_path(&path);
        assert_eq!(warped.commands().len(), path.commands().len());
    }

    // =========================================================================
    // Wave tests
    // =========================================================================

    #[test]
    fn wave_zero_along_axis_is_identity() {
        let wave = Wave::new(Vec2::X, 2.0, 1.0, 0.0);
        let p = Vec2::ZERO;
        assert!(approx(wave.apply(&[p])[0], p, TOL));
    }

    #[test]
    fn wave_displaces_perpendicular_at_peak() {
        let freq = 1.0f32;
        let wave = Wave::new(Vec2::X, 2.0, freq, 0.0);
        let along = std::f32::consts::FRAC_PI_2 / freq;
        let p = Vec2::new(along, 0.0);
        let deformed = wave.apply(&[p])[0];
        assert!(
            approx(deformed, Vec2::new(along, 2.0), TOL),
            "got {deformed:?}"
        );
    }

    #[test]
    fn wave_degenerate_axis_is_identity() {
        let wave = Wave::new(Vec2::ZERO, 2.0, 1.0, 0.0);
        let p = Vec2::new(3.0, 4.0);
        assert!(approx(wave.apply(&[p])[0], p, TOL));
    }

    #[test]
    fn wave_apply_to_path() {
        let wave = Wave::new(Vec2::X, 2.0, 1.0, 0.0);
        let path = line(Vec2::new(0.0, 0.0), Vec2::new(10.0, 0.0));
        let warped = wave.apply_to_path(&path);
        assert_eq!(warped.commands().len(), path.commands().len());
    }

    // =========================================================================
    // EnvelopeDeform tests
    // =========================================================================

    #[test]
    fn envelope_identity_leaves_points_unchanged() {
        let top = vec![Vec2::new(0.0, 10.0), Vec2::new(10.0, 10.0)];
        let bottom = vec![Vec2::new(0.0, 0.0), Vec2::new(10.0, 0.0)];
        let envelope = EnvelopeDeform::new(top.clone(), bottom.clone(), top, bottom);
        let p = Vec2::new(5.0, 5.0);
        assert!(approx(envelope.apply(&[p])[0], p, TOL));
    }

    #[test]
    fn envelope_stretches_into_target() {
        let source_top = vec![Vec2::new(0.0, 10.0), Vec2::new(10.0, 10.0)];
        let source_bottom = vec![Vec2::new(0.0, 0.0), Vec2::new(10.0, 0.0)];
        let target_top = vec![Vec2::new(0.0, 20.0), Vec2::new(10.0, 20.0)];
        let target_bottom = vec![Vec2::new(0.0, 0.0), Vec2::new(10.0, 0.0)];
        let envelope = EnvelopeDeform::new(source_top, source_bottom, target_top, target_bottom);
        let p = Vec2::new(5.0, 5.0);
        let deformed = envelope.apply(&[p])[0];
        assert!(
            approx(deformed, Vec2::new(5.0, 10.0), TOL),
            "got {deformed:?}"
        );
    }

    #[test]
    fn envelope_invalid_config_is_identity() {
        let envelope = EnvelopeDeform::new(
            vec![Vec2::ZERO], // too short
            vec![Vec2::ZERO, Vec2::X],
            vec![Vec2::ZERO, Vec2::X],
            vec![Vec2::ZERO, Vec2::X],
        );
        let p = Vec2::new(3.0, 4.0);
        assert!(approx(envelope.apply(&[p])[0], p, TOL));
    }

    #[test]
    fn envelope_apply_to_path() {
        let top = vec![Vec2::new(0.0, 10.0), Vec2::new(10.0, 10.0)];
        let bottom = vec![Vec2::new(0.0, 0.0), Vec2::new(10.0, 0.0)];
        let envelope = EnvelopeDeform::new(top.clone(), bottom.clone(), top, bottom);
        let path = rect(Vec2::new(2.0, 2.0), Vec2::new(8.0, 8.0));
        let warped = envelope.apply_to_path(&path);
        assert_eq!(warped.commands().len(), path.commands().len());
    }

    // =========================================================================
    // PathDeform tests
    // =========================================================================

    #[test]
    fn path_deform_straight_spine_is_identity() {
        let spine = vec![Vec2::new(0.0, 0.0), Vec2::new(10.0, 0.0)];
        let deform = PathDeform::new(spine, Vec2::X, Vec2::ZERO);
        let p = Vec2::new(5.0, 3.0);
        assert!(approx(deform.apply(&[p])[0], p, TOL));
    }

    #[test]
    fn path_deform_follows_bent_spine() {
        let spine = vec![
            Vec2::new(0.0, 0.0),
            Vec2::new(5.0, 0.0),
            Vec2::new(5.0, 5.0),
        ];
        let deform = PathDeform::new(spine, Vec2::X, Vec2::ZERO);

        let on_axis = deform.apply(&[Vec2::new(7.5, 0.0)])[0];
        assert!(approx(on_axis, Vec2::new(5.0, 2.5), TOL), "got {on_axis:?}");

        let offset = deform.apply(&[Vec2::new(7.5, 1.0)])[0];
        assert!(approx(offset, Vec2::new(4.0, 2.5), TOL), "got {offset:?}");
    }

    #[test]
    fn path_deform_invalid_config_is_identity() {
        let deform = PathDeform::new(vec![Vec2::ZERO], Vec2::X, Vec2::ZERO);
        let p = Vec2::new(3.0, 4.0);
        assert!(approx(deform.apply(&[p])[0], p, TOL));
    }

    #[test]
    fn path_deform_apply_to_path() {
        let spine = vec![Vec2::new(0.0, 0.0), Vec2::new(10.0, 0.0)];
        let deform = PathDeform::new(spine, Vec2::X, Vec2::ZERO);
        let path = line(Vec2::new(0.0, 0.0), Vec2::new(10.0, 0.0));
        let warped = deform.apply_to_path(&path);
        assert_eq!(warped.commands().len(), path.commands().len());
    }

    // =========================================================================
    // Spherize tests
    // =========================================================================

    #[test]
    fn spherize_center_point_unaffected() {
        let spherize = Spherize::new(Vec2::ZERO, 10.0, 1.0);
        let deformed = spherize.apply(&[Vec2::ZERO])[0];
        assert!(approx(deformed, Vec2::ZERO, TOL));
    }

    #[test]
    fn spherize_positive_strength_bulges_outward() {
        let spherize = Spherize::new(Vec2::ZERO, 10.0, 1.0);
        let p = Vec2::new(5.0, 0.0);
        // normalized = 0.5, displaced = 5 * (1 + 1*0.25) = 6.25
        let deformed = spherize.apply(&[p])[0];
        assert!(
            approx(deformed, Vec2::new(6.25, 0.0), TOL),
            "got {deformed:?}"
        );
    }

    #[test]
    fn spherize_negative_strength_pulls_inward() {
        let spherize = Spherize::new(Vec2::ZERO, 10.0, -1.0);
        let p = Vec2::new(5.0, 0.0);
        // normalized = 0.5, displaced = 5 * (1 - 1*0.25) = 3.75
        let deformed = spherize.apply(&[p])[0];
        assert!(
            approx(deformed, Vec2::new(3.75, 0.0), TOL),
            "got {deformed:?}"
        );
    }

    #[test]
    fn spherize_no_effect_outside_radius() {
        let spherize = Spherize::new(Vec2::ZERO, 5.0, 1.0);
        let p = Vec2::new(10.0, 0.0);
        assert!(approx(spherize.apply(&[p])[0], p, TOL));
    }

    #[test]
    fn spherize_apply_to_path() {
        let spherize = Spherize::new(Vec2::ZERO, 10.0, 1.0);
        let path = rect(Vec2::new(-2.0, -2.0), Vec2::new(2.0, 2.0));
        let warped = spherize.apply_to_path(&path);
        assert_eq!(warped.commands().len(), path.commands().len());
    }

    // =========================================================================
    // Roughen tests
    // =========================================================================

    #[test]
    fn roughen_zero_amplitude_is_identity() {
        let roughen = Roughen::new(0.0, 42);
        let p = Vec2::new(3.0, 4.0);
        assert!(approx(roughen.apply(&[p])[0], p, TOL));
    }

    #[test]
    fn roughen_bounded_by_amplitude() {
        let roughen = Roughen::new(2.0, 42);
        let points: Vec<Vec2> = (0..20).map(|i| Vec2::new(i as f32, 0.0)).collect();
        let deformed = roughen.apply(&points);
        for (p, d) in points.iter().zip(deformed.iter()) {
            assert!(
                p.distance(*d) <= 2.0 + TOL,
                "displacement exceeded amplitude"
            );
        }
    }

    #[test]
    fn roughen_deterministic_for_same_seed() {
        let roughen = Roughen::new(3.0, 7);
        let points: Vec<Vec2> = (0..10).map(|i| Vec2::new(i as f32, i as f32)).collect();
        let a = roughen.apply(&points);
        let b = roughen.apply(&points);
        assert_eq!(a, b);
    }

    #[test]
    fn roughen_different_seeds_differ() {
        let a = Roughen::new(3.0, 1).apply(&[Vec2::new(1.0, 1.0)])[0];
        let b = Roughen::new(3.0, 2).apply(&[Vec2::new(1.0, 1.0)])[0];
        assert!(a.distance(b) > TOL);
    }

    #[test]
    fn roughen_apply_to_path() {
        let roughen = Roughen::new(1.0, 5);
        let path = rect(Vec2::new(0.0, 0.0), Vec2::new(4.0, 4.0));
        let warped = roughen.apply_to_path(&path);
        assert_eq!(warped.commands().len(), path.commands().len());
    }

    // =========================================================================
    // PuckerBloat tests
    // =========================================================================

    #[test]
    fn pucker_bloat_zero_strength_is_identity() {
        let pucker = PuckerBloat::new(0.0);
        let points = vec![
            Vec2::new(0.0, 0.0),
            Vec2::new(4.0, 0.0),
            Vec2::new(2.0, 4.0),
        ];
        let deformed = pucker.apply(&points);
        for (p, d) in points.iter().zip(deformed.iter()) {
            assert!(approx(*p, *d, TOL));
        }
    }

    #[test]
    fn pucker_bloat_positive_moves_away_from_centroid() {
        let pucker = PuckerBloat::new(1.0);
        let points = vec![Vec2::new(0.0, 0.0), Vec2::new(10.0, 0.0)];
        // centroid = (5, 0); point (10, 0) offset = (5, 0); displaced by
        // strength * offset => (10,0) + (5,0) = (15, 0)
        let deformed = pucker.apply(&points);
        assert!(approx(deformed[1], Vec2::new(15.0, 0.0), TOL));
        assert!(approx(deformed[0], Vec2::new(-5.0, 0.0), TOL));
    }

    #[test]
    fn pucker_bloat_negative_moves_toward_centroid() {
        let pucker = PuckerBloat::new(-0.5);
        let points = vec![Vec2::new(0.0, 0.0), Vec2::new(10.0, 0.0)];
        // centroid = (5, 0); point (10, 0) offset = (5, 0); displaced by
        // -0.5 * offset => (10, 0) + (-2.5, 0) = (7.5, 0)
        let deformed = pucker.apply(&points);
        assert!(approx(deformed[1], Vec2::new(7.5, 0.0), TOL));
    }

    #[test]
    fn pucker_bloat_empty_input_is_empty() {
        let pucker = PuckerBloat::new(1.0);
        let deformed = pucker.apply(&[]);
        assert!(deformed.is_empty());
    }

    #[test]
    fn pucker_bloat_apply_to_path() {
        let pucker = PuckerBloat::new(0.5);
        let path = rect(Vec2::new(0.0, 0.0), Vec2::new(4.0, 4.0));
        let warped = pucker.apply_to_path(&path);
        assert_eq!(warped.commands().len(), path.commands().len());
    }

    // =========================================================================
    // FieldDisplace tests
    // =========================================================================

    #[test]
    fn field_displace_zero_scale_is_identity() {
        let field = FieldDisplace::new(Vec2::ZERO, 1.0, 1);
        let p = Vec2::new(3.0, 4.0);
        assert!(approx(field.apply(&[p])[0], p, TOL));
    }

    #[test]
    fn field_displace_bounded_by_scale() {
        let scale = Vec2::new(2.0, 3.0);
        let field = FieldDisplace::new(scale, 0.3, 9);
        let points: Vec<Vec2> = (0..30).map(|i| Vec2::new(i as f32, -i as f32)).collect();
        let deformed = field.apply(&points);
        for (p, d) in points.iter().zip(deformed.iter()) {
            let disp = *d - *p;
            assert!(disp.x.abs() <= scale.x + TOL);
            assert!(disp.y.abs() <= scale.y + TOL);
        }
    }

    #[test]
    fn field_displace_deterministic_for_same_seed() {
        let field = FieldDisplace::new(Vec2::new(1.0, 1.0), 0.5, 3);
        let points: Vec<Vec2> = (0..10)
            .map(|i| Vec2::new(i as f32, i as f32 * 0.5))
            .collect();
        let a = field.apply(&points);
        let b = field.apply(&points);
        assert_eq!(a, b);
    }

    #[test]
    fn field_displace_different_seeds_differ() {
        let p = Vec2::new(1.0, 1.0);
        let a = FieldDisplace::new(Vec2::new(1.0, 1.0), 0.5, 1).apply(&[p])[0];
        let b = FieldDisplace::new(Vec2::new(1.0, 1.0), 0.5, 2).apply(&[p])[0];
        assert!(a.distance(b) > TOL);
    }

    #[test]
    fn field_displace_apply_to_path() {
        let field = FieldDisplace::new(Vec2::new(0.5, 0.5), 0.4, 11);
        let path = rect(Vec2::new(0.0, 0.0), Vec2::new(4.0, 4.0));
        let warped = field.apply_to_path(&path);
        assert_eq!(warped.commands().len(), path.commands().len());
    }

    // =========================================================================
    // to_dew_ast tests
    //
    // wick-linalg (the Vec2-capable dew evaluator) isn't a dependency of this
    // crate, so these check AST structure rather than numeric evaluation:
    // degenerate configurations (recognized at `to_dew_ast` build time, the
    // same way `apply`/`sample` short-circuit them) should compile straight
    // to the identity `Var("p")`; non-degenerate configurations should not.
    // =========================================================================

    #[cfg(feature = "wick")]
    mod dew_ast_tests {
        use super::*;
        use wick_core::{Ast, BinOp};

        fn is_identity(ast: &Ast) -> bool {
            matches!(ast, Ast::Var(name) if name == "p")
        }

        #[test]
        fn point_push_identity_at_zero_radius() {
            let push = PointPush::new(Vec2::ZERO, Vec2::new(10.0, 0.0), 0.0, Falloff::Linear);
            assert!(is_identity(&push.to_dew_ast()));
        }

        #[test]
        fn point_push_active_case_is_conditional() {
            let push = PointPush::new(Vec2::ZERO, Vec2::new(10.0, 0.0), 5.0, Falloff::Linear);
            let ast = push.to_dew_ast();
            assert!(!is_identity(&ast));
            assert!(matches!(ast, Ast::If(..)));
        }

        #[test]
        fn twist_identity_at_zero_radius() {
            let twist = Twist::new(Vec2::ZERO, 1.0, 0.0, Falloff::Linear);
            assert!(is_identity(&twist.to_dew_ast()));
        }

        #[test]
        fn twist_active_case_is_conditional() {
            let twist = Twist::new(
                Vec2::ZERO,
                std::f32::consts::FRAC_PI_2,
                10.0,
                Falloff::Smooth,
            );
            let ast = twist.to_dew_ast();
            assert!(!is_identity(&ast));
            assert!(matches!(ast, Ast::If(..)));
        }

        #[test]
        fn taper_identity_when_axis_degenerate() {
            let taper = Taper::new(Vec2::ZERO, Vec2::ZERO, 1.0, 2.0, 10.0);
            assert!(is_identity(&taper.to_dew_ast()));
        }

        #[test]
        fn taper_active_case_is_not_identity() {
            let taper = Taper::new(Vec2::X, Vec2::ZERO, 1.0, 2.0, 10.0);
            let ast = taper.to_dew_ast();
            assert!(!is_identity(&ast));
            assert!(matches!(ast, Ast::BinOp(BinOp::Add, ..)));
        }

        #[test]
        fn bend_identity_when_angle_zero() {
            let bend = Bend::new(Vec2::ZERO, Vec2::X, 0.0, 10.0);
            assert!(is_identity(&bend.to_dew_ast()));
        }

        #[test]
        fn bend_active_case_is_not_identity() {
            let bend = Bend::new(Vec2::ZERO, Vec2::X, std::f32::consts::FRAC_PI_2, 10.0);
            let ast = bend.to_dew_ast();
            assert!(!is_identity(&ast));
            assert!(matches!(ast, Ast::BinOp(BinOp::Add, ..)));
        }

        #[test]
        fn bulge_identity_at_zero_radius() {
            let bulge = Bulge::new(Vec2::ZERO, 0.0, 5.0, Falloff::Linear);
            assert!(is_identity(&bulge.to_dew_ast()));
        }

        #[test]
        fn bulge_active_case_is_conditional() {
            let bulge = Bulge::new(Vec2::ZERO, 10.0, 5.0, Falloff::Gaussian);
            let ast = bulge.to_dew_ast();
            assert!(!is_identity(&ast));
            assert!(matches!(ast, Ast::If(..)));
        }

        #[test]
        fn wave_identity_when_axis_degenerate() {
            let wave = Wave::new(Vec2::ZERO, 2.0, 1.0, 0.0);
            assert!(is_identity(&wave.to_dew_ast()));
        }

        #[test]
        fn wave_active_case_is_not_identity() {
            let wave = Wave::new(Vec2::X, 2.0, 1.0, 0.0);
            let ast = wave.to_dew_ast();
            assert!(!is_identity(&ast));
            assert!(matches!(ast, Ast::BinOp(BinOp::Add, ..)));
        }

        #[test]
        fn spherize_identity_at_zero_radius() {
            let spherize = Spherize::new(Vec2::ZERO, 0.0, 1.0);
            assert!(is_identity(&spherize.to_dew_ast()));
        }

        #[test]
        fn spherize_active_case_is_conditional() {
            let spherize = Spherize::new(Vec2::ZERO, 10.0, 1.0);
            let ast = spherize.to_dew_ast();
            assert!(!is_identity(&ast));
            assert!(matches!(ast, Ast::If(..)));
        }

        #[test]
        fn pucker_bloat_is_unconditional_linear_formula() {
            // PuckerBloat has no radius/falloff guard, so its AST is always a
            // direct `p + (p - centroid) * strength` — never Var("p") alone
            // (even at strength 0.0, the formula is still emitted rather than
            // folded, matching how the other ops fold only their *degenerate*
            // configurations, not merely-inactive ones).
            let pucker = PuckerBloat::new(0.5).with_centroid(Vec2::new(1.0, 1.0));
            let ast = pucker.to_dew_ast();
            assert!(!is_identity(&ast));
            assert!(matches!(ast, Ast::BinOp(BinOp::Add, ..)));
        }
    }
}
