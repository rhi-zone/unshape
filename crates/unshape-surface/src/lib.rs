//! NURBS surfaces for 3D modeling.
//!
//! Provides NURBS (Non-Uniform Rational B-Spline) surfaces, which are the
//! tensor product extension of NURBS curves. NURBS surfaces can exactly
//! represent quadric surfaces (spheres, cylinders, cones, tori).
//!
//! # Example
//!
//! ```ignore
//! use unshape_core::surface::{NurbsSurface, nurbs_sphere};
//!
//! // Create a unit sphere
//! let sphere = nurbs_sphere(glam::Vec3::ZERO, 1.0);
//!
//! // Evaluate at a parameter point
//! let point = sphere.evaluate(0.5, 0.5);
//!
//! // Tessellate to mesh
//! let mesh = sphere.tessellate(32, 16);
//! ```

use glam::Vec3;

/// A weighted control point for NURBS surfaces.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SurfacePoint {
    /// The control point position.
    pub point: Vec3,
    /// The weight (typically > 0).
    pub weight: f32,
}

impl SurfacePoint {
    /// Creates a new weighted surface point.
    pub fn new(point: Vec3, weight: f32) -> Self {
        Self { point, weight }
    }

    /// Creates a surface point with weight 1.0.
    pub fn unweighted(point: Vec3) -> Self {
        Self { point, weight: 1.0 }
    }
}

/// Parameter domain for a NURBS surface.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SurfaceDomain {
    /// Minimum and maximum u parameter values.
    pub u: ParameterRange,
    /// Minimum and maximum v parameter values.
    pub v: ParameterRange,
}

/// A parameter range (min, max).
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ParameterRange {
    /// Minimum parameter value.
    pub min: f32,
    /// Maximum parameter value.
    pub max: f32,
}

/// A NURBS surface (tensor product of NURBS curves).
///
/// Control points are arranged in a 2D grid with dimensions (num_u, num_v).
/// The surface is parameterized by (u, v) coordinates.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct NurbsSurface {
    /// Control points in row-major order (v varies fastest).
    /// Index = u * num_v + v
    points: Vec<SurfacePoint>,
    /// Number of control points in U direction.
    num_u: usize,
    /// Number of control points in V direction.
    num_v: usize,
    /// Degree in U direction.
    degree_u: usize,
    /// Degree in V direction.
    degree_v: usize,
    /// Knot vector in U direction.
    knots_u: Vec<f32>,
    /// Knot vector in V direction.
    knots_v: Vec<f32>,
}

impl NurbsSurface {
    /// Creates a new NURBS surface with uniform knots.
    pub fn new(
        points: Vec<SurfacePoint>,
        num_u: usize,
        num_v: usize,
        degree_u: usize,
        degree_v: usize,
    ) -> Self {
        assert_eq!(
            points.len(),
            num_u * num_v,
            "Control point count must match grid dimensions"
        );

        let knots_u = uniform_clamped_knots(num_u, degree_u);
        let knots_v = uniform_clamped_knots(num_v, degree_v);

        Self {
            points,
            num_u,
            num_v,
            degree_u,
            degree_v,
            knots_u,
            knots_v,
        }
    }

    /// Creates a new NURBS surface with custom knots.
    pub fn with_knots(
        points: Vec<SurfacePoint>,
        num_u: usize,
        num_v: usize,
        degree_u: usize,
        degree_v: usize,
        knots_u: Vec<f32>,
        knots_v: Vec<f32>,
    ) -> Self {
        assert_eq!(points.len(), num_u * num_v);
        Self {
            points,
            num_u,
            num_v,
            degree_u,
            degree_v,
            knots_u,
            knots_v,
        }
    }

    /// Creates a bicubic NURBS surface (degree 3 in both directions).
    pub fn bicubic(points: Vec<SurfacePoint>, num_u: usize, num_v: usize) -> Self {
        Self::new(points, num_u, num_v, 3, 3)
    }

    /// Creates a biquadratic NURBS surface (degree 2 in both directions).
    pub fn biquadratic(points: Vec<SurfacePoint>, num_u: usize, num_v: usize) -> Self {
        Self::new(points, num_u, num_v, 2, 2)
    }

    /// Creates from unweighted points.
    pub fn from_points(
        points: Vec<Vec3>,
        num_u: usize,
        num_v: usize,
        degree_u: usize,
        degree_v: usize,
    ) -> Self {
        let weighted = points.into_iter().map(SurfacePoint::unweighted).collect();
        Self::new(weighted, num_u, num_v, degree_u, degree_v)
    }

    /// Returns the control point at grid position (u_idx, v_idx).
    pub fn control_point(&self, u_idx: usize, v_idx: usize) -> &SurfacePoint {
        &self.points[u_idx * self.num_v + v_idx]
    }

    /// Returns the parameter domain.
    pub fn domain(&self) -> SurfaceDomain {
        SurfaceDomain {
            u: ParameterRange {
                min: self.knots_u[self.degree_u],
                max: self.knots_u[self.num_u],
            },
            v: ParameterRange {
                min: self.knots_v[self.degree_v],
                max: self.knots_v[self.num_v],
            },
        }
    }

    /// Evaluates the surface at parameter (u, v).
    pub fn evaluate(&self, u: f32, v: f32) -> Vec3 {
        let domain = self.domain();
        let u = u.clamp(domain.u.min, domain.u.max - 0.0001);
        let v = v.clamp(domain.v.min, domain.v.max - 0.0001);

        // Find knot spans
        let span_u = find_span(u, &self.knots_u, self.num_u, self.degree_u);
        let span_v = find_span(v, &self.knots_v, self.num_v, self.degree_v);

        // Compute basis functions
        let basis_u = basis_functions(u, span_u, &self.knots_u, self.degree_u);
        let basis_v = basis_functions(v, span_v, &self.knots_v, self.degree_v);

        // Evaluate using tensor product
        let mut sum = Vec3::ZERO;
        let mut weight_sum = 0.0;

        for i in 0..=self.degree_u {
            let u_idx = span_u - self.degree_u + i;
            for j in 0..=self.degree_v {
                let v_idx = span_v - self.degree_v + j;
                let cp = self.control_point(u_idx, v_idx);
                let basis = basis_u[i] * basis_v[j] * cp.weight;
                sum += cp.point * basis;
                weight_sum += basis;
            }
        }

        if weight_sum.abs() < 1e-10 {
            sum
        } else {
            sum / weight_sum
        }
    }

    /// Evaluates the partial derivative with respect to u.
    pub fn derivative_u(&self, u: f32, v: f32) -> Vec3 {
        let h = 0.001;
        let domain = self.domain();
        let u_lo = (u - h).max(domain.u.min);
        let u_hi = (u + h).min(domain.u.max - 0.0001);
        (self.evaluate(u_hi, v) - self.evaluate(u_lo, v)) / (u_hi - u_lo)
    }

    /// Evaluates the partial derivative with respect to v.
    pub fn derivative_v(&self, u: f32, v: f32) -> Vec3 {
        let h = 0.001;
        let domain = self.domain();
        let v_lo = (v - h).max(domain.v.min);
        let v_hi = (v + h).min(domain.v.max - 0.0001);
        (self.evaluate(u, v_hi) - self.evaluate(u, v_lo)) / (v_hi - v_lo)
    }

    /// Evaluates the surface normal at parameter (u, v).
    pub fn normal(&self, u: f32, v: f32) -> Vec3 {
        let du = self.derivative_u(u, v);
        let dv = self.derivative_v(u, v);
        du.cross(dv).normalize_or_zero()
    }

    /// Tessellates the surface into a triangle mesh.
    ///
    /// Returns (positions, normals, uvs, indices).
    pub fn tessellate(&self, divisions_u: usize, divisions_v: usize) -> TessellatedSurface {
        let domain = self.domain();

        let num_verts = (divisions_u + 1) * (divisions_v + 1);
        let mut positions = Vec::with_capacity(num_verts);
        let mut normals = Vec::with_capacity(num_verts);
        let mut uvs = Vec::with_capacity(num_verts);

        // Generate vertices
        for i in 0..=divisions_u {
            let u = domain.u.min + (i as f32 / divisions_u as f32) * (domain.u.max - domain.u.min);
            for j in 0..=divisions_v {
                let v =
                    domain.v.min + (j as f32 / divisions_v as f32) * (domain.v.max - domain.v.min);

                positions.push(self.evaluate(u, v));
                normals.push(self.normal(u, v));
                uvs.push(glam::Vec2::new(
                    i as f32 / divisions_u as f32,
                    j as f32 / divisions_v as f32,
                ));
            }
        }

        // Generate indices (two triangles per quad)
        let mut indices = Vec::with_capacity(divisions_u * divisions_v * 6);
        for i in 0..divisions_u {
            for j in 0..divisions_v {
                let idx00 = (i * (divisions_v + 1) + j) as u32;
                let idx10 = ((i + 1) * (divisions_v + 1) + j) as u32;
                let idx01 = (i * (divisions_v + 1) + j + 1) as u32;
                let idx11 = ((i + 1) * (divisions_v + 1) + j + 1) as u32;

                // First triangle
                indices.push(idx00);
                indices.push(idx10);
                indices.push(idx11);

                // Second triangle
                indices.push(idx00);
                indices.push(idx11);
                indices.push(idx01);
            }
        }

        TessellatedSurface {
            positions,
            normals,
            uvs,
            indices,
        }
    }
}

/// Result of tessellating a NURBS surface.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TessellatedSurface {
    /// Vertex positions.
    pub positions: Vec<Vec3>,
    /// Vertex normals.
    pub normals: Vec<Vec3>,
    /// Texture coordinates.
    pub uvs: Vec<glam::Vec2>,
    /// Triangle indices.
    pub indices: Vec<u32>,
}

// ============================================================================
// NURBS Surface Primitives
// ============================================================================

/// Creates a NURBS sphere.
pub fn nurbs_sphere(center: Vec3, radius: f32) -> NurbsSurface {
    // A NURBS sphere uses 9x5 control points with degree 2
    // It's composed of 4 quarter-circle arcs in each direction

    let w = std::f32::consts::FRAC_1_SQRT_2; // 1/sqrt(2)

    // Circle control point positions (at bounding box corners for NURBS)
    let circle_x = [1.0, 1.0, 0.0, -1.0, -1.0, -1.0, 0.0, 1.0, 1.0];
    let circle_z = [0.0, 1.0, 1.0, 1.0, 0.0, -1.0, -1.0, -1.0, 0.0];
    let circle_weights = [1.0, w, 1.0, w, 1.0, w, 1.0, w, 1.0];

    // Latitude control points for NURBS (at bounding box corners, not on curve)
    // Two stacked quarter circles from north pole to south pole
    let latitudes: [(f32, f32); 5] = [
        (1.0, 0.0),  // North pole (on curve)
        (1.0, 1.0),  // 45° N control point (at corner)
        (0.0, 1.0),  // Equator (on curve)
        (-1.0, 1.0), // 45° S control point (at corner)
        (-1.0, 0.0), // South pole (on curve)
    ];
    let lat_weights = [1.0, w, 1.0, w, 1.0];

    let mut points = Vec::with_capacity(9 * 5);

    // Points stored as u * num_v + v, where U is around circle, V is latitude
    for lon_idx in 0..9 {
        for (lat_idx, &(y_scale, r_scale)) in latitudes.iter().enumerate() {
            let pos = center
                + Vec3::new(
                    circle_x[lon_idx] * r_scale * radius,
                    y_scale * radius,
                    circle_z[lon_idx] * r_scale * radius,
                );
            let weight = circle_weights[lon_idx] * lat_weights[lat_idx];
            points.push(SurfacePoint::new(pos, weight));
        }
    }

    // Knot vectors for periodic curves
    let knots_u = vec![0.0, 0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 4.0];
    let knots_v = vec![0.0, 0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 2.0];

    NurbsSurface::with_knots(points, 9, 5, 2, 2, knots_u, knots_v)
}

/// Creates a NURBS cylinder.
pub fn nurbs_cylinder(center: Vec3, radius: f32, height: f32) -> NurbsSurface {
    let w = std::f32::consts::FRAC_1_SQRT_2;

    // Circle control point positions
    let circle_x = [1.0, 1.0, 0.0, -1.0, -1.0, -1.0, 0.0, 1.0, 1.0];
    let circle_z = [0.0, 1.0, 1.0, 1.0, 0.0, -1.0, -1.0, -1.0, 0.0];
    let circle_weights = [1.0, w, 1.0, w, 1.0, w, 1.0, w, 1.0];

    let half_height = height / 2.0;
    let mut points = Vec::with_capacity(9 * 2);

    // Points stored as u * num_v + v, where U is around circle, V is height
    for i in 0..9 {
        // Bottom
        let pos = center + Vec3::new(circle_x[i] * radius, -half_height, circle_z[i] * radius);
        points.push(SurfacePoint::new(pos, circle_weights[i]));
        // Top
        let pos = center + Vec3::new(circle_x[i] * radius, half_height, circle_z[i] * radius);
        points.push(SurfacePoint::new(pos, circle_weights[i]));
    }

    let knots_u = vec![0.0, 0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 4.0];
    let knots_v = vec![0.0, 0.0, 1.0, 1.0];

    NurbsSurface::with_knots(points, 9, 2, 2, 1, knots_u, knots_v)
}

/// Creates a NURBS torus.
pub fn nurbs_torus(center: Vec3, major_radius: f32, minor_radius: f32) -> NurbsSurface {
    let w = std::f32::consts::FRAC_1_SQRT_2;

    // Circle control point positions (NURBS circle representation)
    let circle_x = [1.0, 1.0, 0.0, -1.0, -1.0, -1.0, 0.0, 1.0, 1.0];
    let circle_z = [0.0, 1.0, 1.0, 1.0, 0.0, -1.0, -1.0, -1.0, 0.0];

    // Minor circle offsets (radial outward, vertical)
    let minor_r = [1.0, 1.0, 0.0, -1.0, -1.0, -1.0, 0.0, 1.0, 1.0]; // radial
    let minor_y = [0.0, 1.0, 1.0, 1.0, 0.0, -1.0, -1.0, -1.0, 0.0]; // vertical

    let circle_weights = [1.0, w, 1.0, w, 1.0, w, 1.0, w, 1.0];

    let mut points = Vec::with_capacity(9 * 9);

    // Points stored as u * num_v + v, where U is around major circle, V is around minor circle
    for i in 0..9 {
        for j in 0..9 {
            let r = major_radius + minor_r[j] * minor_radius;
            let y = minor_y[j] * minor_radius;

            let pos = center + Vec3::new(circle_x[i] * r, y, circle_z[i] * r);
            let weight = circle_weights[i] * circle_weights[j];
            points.push(SurfacePoint::new(pos, weight));
        }
    }

    let knots = vec![0.0, 0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 4.0];

    NurbsSurface::with_knots(points, 9, 9, 2, 2, knots.clone(), knots)
}

/// Creates a NURBS cone.
pub fn nurbs_cone(apex: Vec3, base_center: Vec3, radius: f32) -> NurbsSurface {
    let w = std::f32::consts::FRAC_1_SQRT_2;

    let axis = (apex - base_center).normalize_or_zero();

    // Find perpendicular vectors for the base circle
    let up = if axis.y.abs() < 0.9 { Vec3::Y } else { Vec3::X };
    let tangent1 = axis.cross(up).normalize();
    let tangent2 = axis.cross(tangent1).normalize();

    // Circle control point offsets (NURBS circle representation)
    let circle_t1 = [1.0, 1.0, 0.0, -1.0, -1.0, -1.0, 0.0, 1.0, 1.0];
    let circle_t2 = [0.0, 1.0, 1.0, 1.0, 0.0, -1.0, -1.0, -1.0, 0.0];
    let circle_weights = [1.0, w, 1.0, w, 1.0, w, 1.0, w, 1.0];

    let mut points = Vec::with_capacity(9 * 2);

    // Points stored as u * num_v + v, where U is around circle, V is from base to apex
    for i in 0..9 {
        // Base
        let dir = tangent1 * circle_t1[i] + tangent2 * circle_t2[i];
        let pos = base_center + dir * radius;
        points.push(SurfacePoint::new(pos, circle_weights[i]));
        // Apex
        points.push(SurfacePoint::new(apex, circle_weights[i]));
    }

    let knots_u = vec![0.0, 0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 4.0];
    let knots_v = vec![0.0, 0.0, 1.0, 1.0];

    NurbsSurface::with_knots(points, 9, 2, 2, 1, knots_u, knots_v)
}

/// Creates a bilinear NURBS patch (degree 1 in both directions).
pub fn nurbs_bilinear_patch(p00: Vec3, p10: Vec3, p01: Vec3, p11: Vec3) -> NurbsSurface {
    let points = vec![
        SurfacePoint::unweighted(p00),
        SurfacePoint::unweighted(p01),
        SurfacePoint::unweighted(p10),
        SurfacePoint::unweighted(p11),
    ];
    NurbsSurface::new(points, 2, 2, 1, 1)
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Generates uniform clamped knots for n control points and degree k.
fn uniform_clamped_knots(n: usize, k: usize) -> Vec<f32> {
    let num_knots = n + k + 1;
    let mut knots = Vec::with_capacity(num_knots);

    for i in 0..num_knots {
        if i <= k {
            knots.push(0.0);
        } else if i >= n {
            knots.push((n - k) as f32);
        } else {
            knots.push((i - k) as f32);
        }
    }

    knots
}

/// Finds the knot span index for parameter t.
fn find_span(t: f32, knots: &[f32], n: usize, degree: usize) -> usize {
    let mut span = degree;
    for i in degree..n {
        if t < knots[i + 1] {
            span = i;
            break;
        }
    }
    span
}

/// Computes the non-zero basis functions at parameter t.
fn basis_functions(t: f32, span: usize, knots: &[f32], degree: usize) -> Vec<f32> {
    let mut basis = vec![0.0; degree + 1];
    let mut left = vec![0.0; degree + 1];
    let mut right = vec![0.0; degree + 1];

    basis[0] = 1.0;

    for j in 1..=degree {
        left[j] = t - knots[span + 1 - j];
        right[j] = knots[span + j] - t;
        let mut saved = 0.0;

        for r in 0..j {
            let temp = basis[r] / (right[r + 1] + left[j - r]);
            basis[r] = saved + right[r + 1] * temp;
            saved = left[j - r] * temp;
        }

        basis[j] = saved;
    }

    basis
}

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // SurfacePoint tests
    // =========================================================================

    #[test]
    fn test_surface_point_new() {
        let pt = SurfacePoint::new(Vec3::new(1.0, 2.0, 3.0), 0.5);
        assert_eq!(pt.point, Vec3::new(1.0, 2.0, 3.0));
        assert_eq!(pt.weight, 0.5);
    }

    #[test]
    fn test_surface_point_unweighted() {
        let pt = SurfacePoint::unweighted(Vec3::new(4.0, 5.0, 6.0));
        assert_eq!(pt.point, Vec3::new(4.0, 5.0, 6.0));
        assert_eq!(pt.weight, 1.0);
    }

    // =========================================================================
    // NurbsSurface constructor tests
    // =========================================================================

    #[test]
    fn test_nurbs_surface_new() {
        let points: Vec<SurfacePoint> = (0..9)
            .map(|i| SurfacePoint::unweighted(Vec3::splat(i as f32)))
            .collect();
        let surface = NurbsSurface::new(points, 3, 3, 2, 2);
        assert_eq!(surface.num_u, 3);
        assert_eq!(surface.num_v, 3);
    }

    #[test]
    fn test_nurbs_bicubic() {
        let points: Vec<SurfacePoint> = (0..25)
            .map(|i| SurfacePoint::unweighted(Vec3::splat(i as f32)))
            .collect();
        let surface = NurbsSurface::bicubic(points, 5, 5);
        assert_eq!(surface.degree_u, 3);
        assert_eq!(surface.degree_v, 3);
    }

    #[test]
    fn test_nurbs_biquadratic() {
        let points: Vec<SurfacePoint> = (0..9)
            .map(|i| SurfacePoint::unweighted(Vec3::splat(i as f32)))
            .collect();
        let surface = NurbsSurface::biquadratic(points, 3, 3);
        assert_eq!(surface.degree_u, 2);
        assert_eq!(surface.degree_v, 2);
    }

    #[test]
    fn test_nurbs_from_points() {
        let points: Vec<Vec3> = (0..9).map(|i| Vec3::splat(i as f32)).collect();
        let surface = NurbsSurface::from_points(points, 3, 3, 2, 2);
        // All weights should be 1.0
        for i in 0..3 {
            for j in 0..3 {
                assert_eq!(surface.control_point(i, j).weight, 1.0);
            }
        }
    }

    // =========================================================================
    // Domain and control point tests
    // =========================================================================

    #[test]
    fn test_domain() {
        let patch = nurbs_bilinear_patch(Vec3::ZERO, Vec3::X, Vec3::Y, Vec3::new(1.0, 1.0, 0.0));
        let domain = patch.domain();
        assert!(domain.u.min < domain.u.max);
        assert!(domain.v.min < domain.v.max);
    }

    #[test]
    fn test_control_point_access() {
        let patch = nurbs_bilinear_patch(
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
        );
        // For bilinear patch with 2x2 control points
        let cp00 = patch.control_point(0, 0);
        let cp10 = patch.control_point(1, 0);
        assert!((cp00.point - Vec3::new(0.0, 0.0, 0.0)).length() < 0.01);
        assert!((cp10.point - Vec3::new(1.0, 0.0, 0.0)).length() < 0.01);
    }

    // =========================================================================
    // Derivative tests
    // =========================================================================

    #[test]
    fn test_derivative_u() {
        let patch = nurbs_bilinear_patch(
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
        );
        let du = patch.derivative_u(0.5, 0.5);
        // Bilinear patch: derivative in u should point roughly in +X
        assert!(du.x > 0.0);
        assert!(du.y.abs() < 0.1);
    }

    #[test]
    fn test_derivative_v() {
        let patch = nurbs_bilinear_patch(
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
        );
        let dv = patch.derivative_v(0.5, 0.5);
        // Bilinear patch: derivative in v should point roughly in +Y
        assert!(dv.y > 0.0);
        assert!(dv.x.abs() < 0.1);
    }

    // =========================================================================
    // Helper function tests
    // =========================================================================

    #[test]
    fn test_uniform_clamped_knots() {
        let knots = uniform_clamped_knots(4, 2);
        // For n=4, k=2: num_knots = 4+2+1 = 7
        assert_eq!(knots.len(), 7);
        // First k+1 should be 0
        assert_eq!(knots[0], 0.0);
        assert_eq!(knots[1], 0.0);
        assert_eq!(knots[2], 0.0);
        // Last k+1 should be n-k
        assert_eq!(knots[4], 2.0);
        assert_eq!(knots[5], 2.0);
        assert_eq!(knots[6], 2.0);
    }

    #[test]
    fn test_find_span() {
        let knots = vec![0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 3.0, 3.0];
        let span = find_span(1.5, &knots, 5, 2);
        assert_eq!(span, 3); // Between knots[3]=1.0 and knots[4]=2.0
    }

    #[test]
    fn test_basis_functions() {
        let knots = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let basis = basis_functions(0.5, 2, &knots, 2);
        // Should have degree+1 = 3 basis functions
        assert_eq!(basis.len(), 3);
        // Sum should be 1.0 (partition of unity)
        let sum: f32 = basis.iter().sum();
        assert!((sum - 1.0).abs() < 0.001);
    }

    // =========================================================================
    // Primitive tests
    // =========================================================================

    #[test]
    fn test_torus() {
        let torus = nurbs_torus(Vec3::ZERO, 2.0, 0.5);
        let domain = torus.domain();

        // Sample a point on the outer edge (major + minor radius)
        let outer = torus.evaluate(domain.u.min, domain.v.min);
        let outer_dist = outer.length();
        // Should be at major_radius + minor_radius = 2.5
        assert!(
            (outer_dist - 2.5).abs() < 0.2,
            "Outer distance {} should be ~2.5",
            outer_dist
        );

        // Sample a point on inner edge
        let inner_v = (domain.v.min + domain.v.max) / 2.0;
        let inner = torus.evaluate(domain.u.min, inner_v);
        // At v=0.5, should be at major_radius - minor_radius = 1.5 (roughly)
        let inner_dist = (inner.x * inner.x + inner.z * inner.z).sqrt();
        assert!(
            inner_dist > 1.0 && inner_dist < 2.5,
            "Inner distance {} should be between 1.0 and 2.5",
            inner_dist
        );
    }

    #[test]
    fn test_sphere_poles() {
        let sphere = nurbs_sphere(Vec3::ZERO, 1.0);
        let domain = sphere.domain();

        // North pole (v_min)
        let north = sphere.evaluate(domain.u.min, domain.v.min);
        assert!(
            (north - Vec3::new(0.0, 1.0, 0.0)).length() < 0.15,
            "North pole {:?} should be at (0, 1, 0)",
            north
        );

        // South pole (v_max)
        let south = sphere.evaluate(domain.u.min, domain.v.max - 0.001);
        assert!(
            (south - Vec3::new(0.0, -1.0, 0.0)).length() < 0.15,
            "South pole {:?} should be at (0, -1, 0)",
            south
        );
    }

    #[test]
    fn test_cylinder_height() {
        let cylinder = nurbs_cylinder(Vec3::ZERO, 1.0, 4.0);
        let domain = cylinder.domain();

        // Bottom
        let bottom = cylinder.evaluate(domain.u.min, domain.v.min);
        assert!(
            (bottom.y - (-2.0)).abs() < 0.1,
            "Bottom y {} should be -2.0",
            bottom.y
        );

        // Top
        let top = cylinder.evaluate(domain.u.min, domain.v.max - 0.001);
        assert!((top.y - 2.0).abs() < 0.1, "Top y {} should be 2.0", top.y);
    }

    // =========================================================================
    // TessellatedSurface tests
    // =========================================================================

    #[test]
    fn test_tessellation_indices_valid() {
        let sphere = nurbs_sphere(Vec3::ZERO, 1.0);
        let mesh = sphere.tessellate(8, 8);

        let max_vertex = mesh.positions.len() as u32;
        for &idx in &mesh.indices {
            assert!(
                idx < max_vertex,
                "Index {} out of bounds (max {})",
                idx,
                max_vertex - 1
            );
        }
    }

    #[test]
    fn test_tessellation_normals_normalized() {
        let patch = nurbs_bilinear_patch(Vec3::ZERO, Vec3::X, Vec3::Y, Vec3::new(1.0, 1.0, 0.0));
        let mesh = patch.tessellate(4, 4);

        for normal in &mesh.normals {
            let len = normal.length();
            assert!(
                (len - 1.0).abs() < 0.01 || len < 0.01, // normalized or zero
                "Normal {:?} has length {}",
                normal,
                len
            );
        }
    }

    #[test]
    fn test_tessellation_uvs_range() {
        let patch = nurbs_bilinear_patch(Vec3::ZERO, Vec3::X, Vec3::Y, Vec3::new(1.0, 1.0, 0.0));
        let mesh = patch.tessellate(4, 4);

        for uv in &mesh.uvs {
            assert!(uv.x >= 0.0 && uv.x <= 1.0, "UV.x {} out of [0,1]", uv.x);
            assert!(uv.y >= 0.0 && uv.y <= 1.0, "UV.y {} out of [0,1]", uv.y);
        }
    }

    // =========================================================================
    // Original tests
    // =========================================================================

    #[test]
    fn test_bilinear_patch() {
        let patch = nurbs_bilinear_patch(
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
        );

        // Corners should match control points
        let domain = patch.domain();

        let p00 = patch.evaluate(domain.u.min, domain.v.min);
        assert!((p00 - Vec3::new(0.0, 0.0, 0.0)).length() < 0.01);

        let p10 = patch.evaluate(domain.u.max - 0.001, domain.v.min);
        assert!((p10 - Vec3::new(1.0, 0.0, 0.0)).length() < 0.01);

        let p01 = patch.evaluate(domain.u.min, domain.v.max - 0.001);
        assert!((p01 - Vec3::new(0.0, 1.0, 0.0)).length() < 0.01);

        // Center should be average
        let center = patch.evaluate(0.5, 0.5);
        assert!((center - Vec3::new(0.5, 0.5, 0.0)).length() < 0.01);
    }

    #[test]
    fn test_nurbs_sphere() {
        let sphere = nurbs_sphere(Vec3::ZERO, 1.0);
        let domain = sphere.domain();

        // Sample points should be on the sphere surface
        for i in 0..10 {
            for j in 0..10 {
                let u = domain.u.min + (i as f32 / 9.0) * (domain.u.max - domain.u.min - 0.001);
                let v = domain.v.min + (j as f32 / 9.0) * (domain.v.max - domain.v.min - 0.001);

                let point = sphere.evaluate(u, v);
                let dist = point.length();

                assert!(
                    (dist - 1.0).abs() < 0.1,
                    "Point at ({}, {}) = {:?} has distance {} from center",
                    u,
                    v,
                    point,
                    dist
                );
            }
        }
    }

    #[test]
    fn test_nurbs_cylinder() {
        let cylinder = nurbs_cylinder(Vec3::ZERO, 1.0, 2.0);
        let domain = cylinder.domain();

        // Sample points should have correct radius
        for i in 0..8 {
            let u = domain.u.min + (i as f32 / 7.0) * (domain.u.max - domain.u.min - 0.001);
            let point = cylinder.evaluate(u, (domain.v.min + domain.v.max) / 2.0);

            let horizontal_dist = (point.x * point.x + point.z * point.z).sqrt();
            assert!(
                (horizontal_dist - 1.0).abs() < 0.1,
                "Horizontal distance {} should be ~1.0",
                horizontal_dist
            );
        }
    }

    #[test]
    fn test_tessellation() {
        let patch = nurbs_bilinear_patch(
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
        );

        let mesh = patch.tessellate(4, 4);

        assert_eq!(mesh.positions.len(), 5 * 5); // (4+1) * (4+1)
        assert_eq!(mesh.normals.len(), 5 * 5);
        assert_eq!(mesh.uvs.len(), 5 * 5);
        assert_eq!(mesh.indices.len(), 4 * 4 * 6); // 4*4 quads * 2 tris * 3 indices
    }

    #[test]
    fn test_surface_normal() {
        let patch = nurbs_bilinear_patch(
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
        );

        let normal = patch.normal(0.5, 0.5);

        // Flat patch in XY plane should have Z normal
        assert!((normal - Vec3::Z).length() < 0.1 || (normal - Vec3::NEG_Z).length() < 0.1);
    }

    #[test]
    fn test_cone() {
        let cone = nurbs_cone(Vec3::new(0.0, 2.0, 0.0), Vec3::new(0.0, 0.0, 0.0), 1.0);
        let domain = cone.domain();

        // Base should have radius 1
        let base_point = cone.evaluate((domain.u.min + domain.u.max) / 2.0, domain.v.min);
        let base_radius = (base_point.x * base_point.x + base_point.z * base_point.z).sqrt();
        assert!((base_radius - 1.0).abs() < 0.2);

        // Apex should be at (0, 2, 0)
        let apex = cone.evaluate((domain.u.min + domain.u.max) / 2.0, domain.v.max - 0.001);
        assert!((apex - Vec3::new(0.0, 2.0, 0.0)).length() < 0.2);
    }
}
