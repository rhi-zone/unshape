use glam::{Vec2, Vec3};

use crate::{EvalContext, Field};

/// Distance field - returns distance from a point.
#[derive(Debug, Clone, Copy)]
pub struct DistancePoint {
    pub point: Vec2,
}

impl DistancePoint {
    pub fn new(point: Vec2) -> Self {
        Self { point }
    }
}

impl Field<Vec2, f32> for DistancePoint {
    fn sample(&self, input: Vec2, _ctx: &EvalContext) -> f32 {
        (input - self.point).length()
    }
}

/// Distance field - returns distance from a line segment.
#[derive(Debug, Clone, Copy)]
pub struct DistanceLine {
    pub a: Vec2,
    pub b: Vec2,
}

impl DistanceLine {
    pub fn new(a: Vec2, b: Vec2) -> Self {
        Self { a, b }
    }
}

impl Field<Vec2, f32> for DistanceLine {
    fn sample(&self, input: Vec2, _ctx: &EvalContext) -> f32 {
        let pa = input - self.a;
        let ba = self.b - self.a;
        let t = (pa.dot(ba) / ba.dot(ba)).clamp(0.0, 1.0);
        (pa - ba * t).length()
    }
}

/// Distance field - returns distance from a circle.
#[derive(Debug, Clone, Copy)]
pub struct DistanceCircle {
    pub center: Vec2,
    pub radius: f32,
}

impl DistanceCircle {
    pub fn new(center: Vec2, radius: f32) -> Self {
        Self { center, radius }
    }
}

impl Field<Vec2, f32> for DistanceCircle {
    fn sample(&self, input: Vec2, _ctx: &EvalContext) -> f32 {
        (input - self.center).length() - self.radius
    }
}

/// Distance field - returns distance from a box.
#[derive(Debug, Clone, Copy)]
pub struct DistanceBox {
    pub center: Vec2,
    pub half_size: Vec2,
}

impl DistanceBox {
    pub fn new(center: Vec2, half_size: Vec2) -> Self {
        Self { center, half_size }
    }
}

impl Field<Vec2, f32> for DistanceBox {
    fn sample(&self, input: Vec2, _ctx: &EvalContext) -> f32 {
        let p = (input - self.center).abs();
        let d = p - self.half_size;
        d.max(Vec2::ZERO).length() + d.x.max(d.y).min(0.0)
    }
}

/// Distance field - returns distance from a rounded box.
#[derive(Debug, Clone, Copy)]
pub struct DistanceRoundedBox {
    /// Center of the box.
    pub center: Vec2,
    /// Half-size of the box (before rounding).
    pub half_size: Vec2,
    /// Corner radius.
    pub radius: f32,
}

impl DistanceRoundedBox {
    /// Creates a new rounded box SDF.
    pub fn new(center: Vec2, half_size: Vec2, radius: f32) -> Self {
        Self {
            center,
            half_size,
            radius,
        }
    }
}

impl Field<Vec2, f32> for DistanceRoundedBox {
    fn sample(&self, input: Vec2, _ctx: &EvalContext) -> f32 {
        let p = (input - self.center).abs();
        let q = p - self.half_size + Vec2::splat(self.radius);
        q.max(Vec2::ZERO).length() + q.x.max(q.y).min(0.0) - self.radius
    }
}

/// Distance field - returns distance from an ellipse.
#[derive(Debug, Clone, Copy)]
pub struct DistanceEllipse {
    /// Center of the ellipse.
    pub center: Vec2,
    /// Radii (half-width, half-height).
    pub radii: Vec2,
}

impl DistanceEllipse {
    /// Creates a new ellipse SDF.
    pub fn new(center: Vec2, radii: Vec2) -> Self {
        Self { center, radii }
    }
}

impl Field<Vec2, f32> for DistanceEllipse {
    fn sample(&self, input: Vec2, _ctx: &EvalContext) -> f32 {
        // Approximate ellipse SDF using normalized space
        let p = (input - self.center).abs();
        let ab = self.radii;

        if ab.x == ab.y {
            // Circle case
            return p.length() - ab.x;
        }

        // Iterative Newton-Raphson for accurate ellipse SDF
        let mut t = 0.25 * std::f32::consts::PI;
        for _ in 0..4 {
            let c = t.cos();
            let s = t.sin();
            let e = Vec2::new(ab.x * c, ab.y * s);
            let d = p - e;
            let g = Vec2::new(-ab.x * s, ab.y * c);
            let dt = d.dot(g) / g.dot(g);
            t = (t + dt).clamp(0.0, std::f32::consts::FRAC_PI_2);
        }

        let closest = Vec2::new(ab.x * t.cos(), ab.y * t.sin());
        let dist = (p - closest).length();

        // Determine sign (inside or outside)
        let normalized = p / ab;
        if normalized.length_squared() < 1.0 {
            -dist
        } else {
            dist
        }
    }
}

/// Distance field - returns distance from a capsule (stadium shape).
#[derive(Debug, Clone, Copy)]
pub struct DistanceCapsule {
    /// First endpoint of the capsule centerline.
    pub a: Vec2,
    /// Second endpoint of the capsule centerline.
    pub b: Vec2,
    /// Capsule radius.
    pub radius: f32,
}

impl DistanceCapsule {
    /// Creates a new capsule SDF.
    pub fn new(a: Vec2, b: Vec2, radius: f32) -> Self {
        Self { a, b, radius }
    }
}

impl Field<Vec2, f32> for DistanceCapsule {
    fn sample(&self, input: Vec2, _ctx: &EvalContext) -> f32 {
        let pa = input - self.a;
        let ba = self.b - self.a;
        let t = (pa.dot(ba) / ba.dot(ba)).clamp(0.0, 1.0);
        (pa - ba * t).length() - self.radius
    }
}

/// Distance field - returns distance from a triangle.
#[derive(Debug, Clone, Copy)]
pub struct DistanceTriangle {
    /// First vertex.
    pub a: Vec2,
    /// Second vertex.
    pub b: Vec2,
    /// Third vertex.
    pub c: Vec2,
}

impl DistanceTriangle {
    /// Creates a new triangle SDF.
    pub fn new(a: Vec2, b: Vec2, c: Vec2) -> Self {
        Self { a, b, c }
    }
}

impl Field<Vec2, f32> for DistanceTriangle {
    fn sample(&self, input: Vec2, _ctx: &EvalContext) -> f32 {
        let p = input;

        // Edge vectors and point-to-vertex vectors
        let e0 = self.b - self.a;
        let e1 = self.c - self.b;
        let e2 = self.a - self.c;
        let v0 = p - self.a;
        let v1 = p - self.b;
        let v2 = p - self.c;

        // Perpendicular vectors
        let pq0 = v0 - e0 * (v0.dot(e0) / e0.dot(e0)).clamp(0.0, 1.0);
        let pq1 = v1 - e1 * (v1.dot(e1) / e1.dot(e1)).clamp(0.0, 1.0);
        let pq2 = v2 - e2 * (v2.dot(e2) / e2.dot(e2)).clamp(0.0, 1.0);

        // Signed distance
        let s = (e0.x * e2.y - e0.y * e2.x).signum();
        let d = (pq0.dot(pq0).min(pq1.dot(pq1)).min(pq2.dot(pq2))).sqrt();

        // Determine if inside or outside
        let c0 = s * (v0.x * e0.y - v0.y * e0.x);
        let c1 = s * (v1.x * e1.y - v1.y * e1.x);
        let c2 = s * (v2.x * e2.y - v2.y * e2.x);

        if c0 >= 0.0 && c1 >= 0.0 && c2 >= 0.0 {
            -d
        } else {
            d
        }
    }
}

/// Distance field - returns distance from a regular polygon.
#[derive(Debug, Clone, Copy)]
pub struct DistanceRegularPolygon {
    /// Center of the polygon.
    pub center: Vec2,
    /// Circumradius (distance from center to vertex).
    pub radius: f32,
    /// Number of sides (3 = triangle, 4 = square, 6 = hexagon, etc.).
    pub sides: u32,
}

impl DistanceRegularPolygon {
    /// Creates a new regular polygon SDF.
    pub fn new(center: Vec2, radius: f32, sides: u32) -> Self {
        Self {
            center,
            radius,
            sides: sides.max(3),
        }
    }
}

impl Field<Vec2, f32> for DistanceRegularPolygon {
    fn sample(&self, input: Vec2, _ctx: &EvalContext) -> f32 {
        use std::f32::consts::PI;

        let p = input - self.center;
        let n = self.sides as f32;

        // Angle between vertices
        let an = PI / n;

        // Convert to polar, reduce to one sector
        let angle = p.y.atan2(p.x);
        let sector_angle = ((angle + PI) / (2.0 * an)).floor() * 2.0 * an - PI + an;

        // Rotate point into canonical sector
        let cs = sector_angle.cos();
        let sn = sector_angle.sin();
        let q = Vec2::new(p.x * cs + p.y * sn, -p.x * sn + p.y * cs).abs();

        // Distance to edge
        let edge_dist = q.x - self.radius * an.cos();
        let corner_dist = (q - Vec2::new(self.radius * an.cos(), self.radius * an.sin())).length();

        if q.y > self.radius * an.sin() {
            corner_dist
        } else {
            edge_dist
        }
    }
}

/// Distance field - returns distance from a pie/arc shape.
#[derive(Debug, Clone, Copy)]
pub struct DistanceArc {
    /// Center of the arc.
    pub center: Vec2,
    /// Radius of the arc.
    pub radius: f32,
    /// Half-angle of the arc in radians.
    pub half_angle: f32,
    /// Thickness of the arc (0 for just the arc curve).
    pub thickness: f32,
}

impl DistanceArc {
    /// Creates a new arc/pie SDF.
    pub fn new(center: Vec2, radius: f32, half_angle: f32, thickness: f32) -> Self {
        Self {
            center,
            radius,
            half_angle,
            thickness,
        }
    }

    /// Creates a pie shape (filled arc).
    pub fn pie(center: Vec2, radius: f32, half_angle: f32) -> Self {
        Self {
            center,
            radius,
            half_angle,
            thickness: radius, // Fill to center
        }
    }
}

impl Field<Vec2, f32> for DistanceArc {
    fn sample(&self, input: Vec2, _ctx: &EvalContext) -> f32 {
        let p = input - self.center;

        // Use symmetry around x-axis
        let p = Vec2::new(p.x, p.y.abs());
        let r = p.length();

        // Angle of point from positive x-axis
        let angle = p.y.atan2(p.x);

        // For a pie/sector shape (thickness >= radius means filled to center):
        // - Inside: angle <= half_angle AND r <= radius
        // - Distance is to the nearest boundary

        if angle <= self.half_angle {
            // Inside angular range
            if self.thickness >= self.radius {
                // Pie mode: distance to arc (outer boundary)
                r - self.radius
            } else {
                // Arc mode: distance to thick arc boundary
                let radial_dist = (r - self.radius).abs();
                radial_dist - self.thickness
            }
        } else {
            // Outside angular range - distance to edge line
            // Edge direction from center
            let edge_dir = Vec2::new(self.half_angle.cos(), self.half_angle.sin());

            // Project p onto edge direction (line from origin along edge_dir)
            let proj_length = p.dot(edge_dir).max(0.0).min(self.radius);
            let proj_point = edge_dir * proj_length;
            let dist_to_edge = (p - proj_point).length();

            if self.thickness >= self.radius {
                // Pie mode: just distance to edge line
                dist_to_edge
            } else {
                dist_to_edge - self.thickness
            }
        }
    }
}

// ============================================================================
// SDF operations
// ============================================================================

/// Union of two SDFs (min).
pub struct SdfUnion<A, B> {
    pub a: A,
    pub b: B,
}

impl<A, B> SdfUnion<A, B> {
    pub fn new(a: A, b: B) -> Self {
        Self { a, b }
    }
}

impl<I: Clone, A, B> Field<I, f32> for SdfUnion<A, B>
where
    A: Field<I, f32>,
    B: Field<I, f32>,
{
    fn sample(&self, input: I, ctx: &EvalContext) -> f32 {
        let a = self.a.sample(input.clone(), ctx);
        let b = self.b.sample(input, ctx);
        a.min(b)
    }
}

/// Intersection of two SDFs (max).
pub struct SdfIntersection<A, B> {
    pub a: A,
    pub b: B,
}

impl<A, B> SdfIntersection<A, B> {
    pub fn new(a: A, b: B) -> Self {
        Self { a, b }
    }
}

impl<I: Clone, A, B> Field<I, f32> for SdfIntersection<A, B>
where
    A: Field<I, f32>,
    B: Field<I, f32>,
{
    fn sample(&self, input: I, ctx: &EvalContext) -> f32 {
        let a = self.a.sample(input.clone(), ctx);
        let b = self.b.sample(input, ctx);
        a.max(b)
    }
}

/// Subtraction of SDF B from A.
pub struct SdfSubtraction<A, B> {
    pub a: A,
    pub b: B,
}

impl<A, B> SdfSubtraction<A, B> {
    pub fn new(a: A, b: B) -> Self {
        Self { a, b }
    }
}

impl<I: Clone, A, B> Field<I, f32> for SdfSubtraction<A, B>
where
    A: Field<I, f32>,
    B: Field<I, f32>,
{
    fn sample(&self, input: I, ctx: &EvalContext) -> f32 {
        let a = self.a.sample(input.clone(), ctx);
        let b = self.b.sample(input, ctx);
        a.max(-b)
    }
}

/// Smooth union of two SDFs using polynomial smooth min.
pub struct SdfSmoothUnion<A, B> {
    pub a: A,
    pub b: B,
    pub k: f32,
}

impl<A, B> SdfSmoothUnion<A, B> {
    pub fn new(a: A, b: B, k: f32) -> Self {
        Self { a, b, k }
    }
}

impl<I: Clone, A, B> Field<I, f32> for SdfSmoothUnion<A, B>
where
    A: Field<I, f32>,
    B: Field<I, f32>,
{
    fn sample(&self, input: I, ctx: &EvalContext) -> f32 {
        let a = self.a.sample(input.clone(), ctx);
        let b = self.b.sample(input, ctx);
        smooth_min(a, b, self.k)
    }
}

/// Smooth intersection of two SDFs.
pub struct SdfSmoothIntersection<A, B> {
    pub a: A,
    pub b: B,
    pub k: f32,
}

impl<A, B> SdfSmoothIntersection<A, B> {
    pub fn new(a: A, b: B, k: f32) -> Self {
        Self { a, b, k }
    }
}

impl<I: Clone, A, B> Field<I, f32> for SdfSmoothIntersection<A, B>
where
    A: Field<I, f32>,
    B: Field<I, f32>,
{
    fn sample(&self, input: I, ctx: &EvalContext) -> f32 {
        let a = self.a.sample(input.clone(), ctx);
        let b = self.b.sample(input, ctx);
        smooth_max(a, b, self.k)
    }
}

/// Smooth subtraction of SDF B from A.
pub struct SdfSmoothSubtraction<A, B> {
    pub a: A,
    pub b: B,
    pub k: f32,
}

impl<A, B> SdfSmoothSubtraction<A, B> {
    pub fn new(a: A, b: B, k: f32) -> Self {
        Self { a, b, k }
    }
}

impl<I: Clone, A, B> Field<I, f32> for SdfSmoothSubtraction<A, B>
where
    A: Field<I, f32>,
    B: Field<I, f32>,
{
    fn sample(&self, input: I, ctx: &EvalContext) -> f32 {
        let a = self.a.sample(input.clone(), ctx);
        let b = self.b.sample(input, ctx);
        smooth_max(a, -b, self.k)
    }
}

/// Polynomial smooth minimum.
fn smooth_min(a: f32, b: f32, k: f32) -> f32 {
    if k <= 0.0 {
        return a.min(b);
    }
    let h = (k - (a - b).abs()).max(0.0) / k;
    a.min(b) - h * h * k * 0.25
}

/// Polynomial smooth maximum.
fn smooth_max(a: f32, b: f32, k: f32) -> f32 {
    -smooth_min(-a, -b, k)
}

/// Rounds/expands an SDF by a radius.
pub struct SdfRound<F> {
    pub field: F,
    pub radius: f32,
}

impl<F> SdfRound<F> {
    pub fn new(field: F, radius: f32) -> Self {
        Self { field, radius }
    }
}

impl<I, F: Field<I, f32>> Field<I, f32> for SdfRound<F> {
    fn sample(&self, input: I, ctx: &EvalContext) -> f32 {
        self.field.sample(input, ctx) - self.radius
    }
}

/// Annular (ring/shell) version of an SDF.
pub struct SdfAnnular<F> {
    pub field: F,
    pub thickness: f32,
}

impl<F> SdfAnnular<F> {
    pub fn new(field: F, thickness: f32) -> Self {
        Self { field, thickness }
    }
}

impl<I, F: Field<I, f32>> Field<I, f32> for SdfAnnular<F> {
    fn sample(&self, input: I, ctx: &EvalContext) -> f32 {
        self.field.sample(input, ctx).abs() - self.thickness
    }
}

// ============================================================================
// Domain modifiers
// ============================================================================

/// Twists space around the Y axis (for 3D fields).
pub struct Twist<F> {
    pub field: F,
    pub amount: f32,
}

impl<F> Twist<F> {
    pub fn new(field: F, amount: f32) -> Self {
        Self { field, amount }
    }
}

impl<O, F: Field<Vec3, O>> Field<Vec3, O> for Twist<F> {
    fn sample(&self, input: Vec3, ctx: &EvalContext) -> O {
        let angle = input.y * self.amount;
        let cos = angle.cos();
        let sin = angle.sin();
        let twisted = Vec3::new(
            input.x * cos - input.z * sin,
            input.y,
            input.x * sin + input.z * cos,
        );
        self.field.sample(twisted, ctx)
    }
}

/// Bends space around the Y axis (for 3D fields).
pub struct Bend<F> {
    pub field: F,
    pub amount: f32,
}

impl<F> Bend<F> {
    pub fn new(field: F, amount: f32) -> Self {
        Self { field, amount }
    }
}

impl<O, F: Field<Vec3, O>> Field<Vec3, O> for Bend<F> {
    fn sample(&self, input: Vec3, ctx: &EvalContext) -> O {
        let angle = input.x * self.amount;
        let cos = angle.cos();
        let sin = angle.sin();
        let bent = Vec3::new(
            cos * input.x - sin * input.y,
            sin * input.x + cos * input.y,
            input.z,
        );
        self.field.sample(bent, ctx)
    }
}

/// Repeats space infinitely.
pub struct Repeat<F> {
    pub field: F,
    pub period: Vec2,
}

impl<F> Repeat<F> {
    pub fn new(field: F, period: Vec2) -> Self {
        Self { field, period }
    }
}

impl<O, F: Field<Vec2, O>> Field<Vec2, O> for Repeat<F> {
    fn sample(&self, input: Vec2, ctx: &EvalContext) -> O {
        let repeated = Vec2::new(
            ((input.x % self.period.x) + self.period.x) % self.period.x - self.period.x * 0.5,
            ((input.y % self.period.y) + self.period.y) % self.period.y - self.period.y * 0.5,
        );
        self.field.sample(repeated, ctx)
    }
}

/// Repeats space infinitely (3D).
pub struct Repeat3D<F> {
    pub field: F,
    pub period: Vec3,
}

impl<F> Repeat3D<F> {
    pub fn new(field: F, period: Vec3) -> Self {
        Self { field, period }
    }
}

impl<O, F: Field<Vec3, O>> Field<Vec3, O> for Repeat3D<F> {
    fn sample(&self, input: Vec3, ctx: &EvalContext) -> O {
        let repeated = Vec3::new(
            ((input.x % self.period.x) + self.period.x) % self.period.x - self.period.x * 0.5,
            ((input.y % self.period.y) + self.period.y) % self.period.y - self.period.y * 0.5,
            ((input.z % self.period.z) + self.period.z) % self.period.z - self.period.z * 0.5,
        );
        self.field.sample(repeated, ctx)
    }
}

/// Rotates 2D input coordinates.
pub struct Rotate2D<F> {
    pub field: F,
    pub angle: f32,
}

impl<F> Rotate2D<F> {
    pub fn new(field: F, angle: f32) -> Self {
        Self { field, angle }
    }
}

impl<O, F: Field<Vec2, O>> Field<Vec2, O> for Rotate2D<F> {
    fn sample(&self, input: Vec2, ctx: &EvalContext) -> O {
        let cos = self.angle.cos();
        let sin = self.angle.sin();
        let rotated = Vec2::new(input.x * cos - input.y * sin, input.x * sin + input.y * cos);
        self.field.sample(rotated, ctx)
    }
}

/// Mirrors space across an axis.
pub struct Mirror<F> {
    pub field: F,
    pub axis: Vec2,
}

impl<F> Mirror<F> {
    pub fn new(field: F, axis: Vec2) -> Self {
        Self {
            field,
            axis: axis.normalize(),
        }
    }

    pub fn x(field: F) -> Self {
        Self::new(field, Vec2::X)
    }

    pub fn y(field: F) -> Self {
        Self::new(field, Vec2::Y)
    }
}

impl<O, F: Field<Vec2, O>> Field<Vec2, O> for Mirror<F> {
    fn sample(&self, input: Vec2, ctx: &EvalContext) -> O {
        let d = input.dot(self.axis);
        let mirrored = if d < 0.0 {
            input - 2.0 * d * self.axis
        } else {
            input
        };
        self.field.sample(mirrored, ctx)
    }
}

/// Domain warping - distorts input coordinates using another field.
pub struct Warp<F, D> {
    pub field: F,
    pub displacement: D,
    pub amount: f32,
}

impl<F, D> Warp<F, D> {
    pub fn new(field: F, displacement: D, amount: f32) -> Self {
        Self {
            field,
            displacement,
            amount,
        }
    }
}

impl<F, D> Field<Vec2, f32> for Warp<F, D>
where
    F: Field<Vec2, f32>,
    D: Field<Vec2, Vec2>,
{
    fn sample(&self, input: Vec2, ctx: &EvalContext) -> f32 {
        let offset = self.displacement.sample(input, ctx);
        let warped = input + offset * self.amount;
        self.field.sample(warped, ctx)
    }
}

/// Creates displacement from two scalar fields (for x and y).
pub struct Displacement<X, Y> {
    pub x: X,
    pub y: Y,
}

impl<X, Y> Displacement<X, Y> {
    pub fn new(x: X, y: Y) -> Self {
        Self { x, y }
    }
}

impl<X, Y> Field<Vec2, Vec2> for Displacement<X, Y>
where
    X: Field<Vec2, f32>,
    Y: Field<Vec2, f32>,
{
    fn sample(&self, input: Vec2, ctx: &EvalContext) -> Vec2 {
        Vec2::new(self.x.sample(input, ctx), self.y.sample(input, ctx))
    }
}
