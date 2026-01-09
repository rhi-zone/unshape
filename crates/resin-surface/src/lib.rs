//! NURBS surfaces for 3D modeling.
//!
//! Provides NURBS (Non-Uniform Rational B-Spline) surfaces, which are the
//! tensor product extension of NURBS curves. NURBS surfaces can exactly
//! represent quadric surfaces (spheres, cylinders, cones, tori).
//!
//! # Example
//!
//! ```ignore
//! use resin_core::surface::{NurbsSurface, nurbs_sphere};
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

/// A NURBS surface (tensor product of NURBS curves).
///
/// Control points are arranged in a 2D grid with dimensions (num_u, num_v).
/// The surface is parameterized by (u, v) coordinates.
#[derive(Debug, Clone)]
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

    /// Returns the parameter domain [(u_min, u_max), (v_min, v_max)].
    pub fn domain(&self) -> ((f32, f32), (f32, f32)) {
        let u_min = self.knots_u[self.degree_u];
        let u_max = self.knots_u[self.num_u];
        let v_min = self.knots_v[self.degree_v];
        let v_max = self.knots_v[self.num_v];
        ((u_min, u_max), (v_min, v_max))
    }

    /// Evaluates the surface at parameter (u, v).
    pub fn evaluate(&self, u: f32, v: f32) -> Vec3 {
        let ((u_min, u_max), (v_min, v_max)) = self.domain();
        let u = u.clamp(u_min, u_max - 0.0001);
        let v = v.clamp(v_min, v_max - 0.0001);

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
        let ((u_min, u_max), _) = self.domain();
        let u_lo = (u - h).max(u_min);
        let u_hi = (u + h).min(u_max - 0.0001);
        (self.evaluate(u_hi, v) - self.evaluate(u_lo, v)) / (u_hi - u_lo)
    }

    /// Evaluates the partial derivative with respect to v.
    pub fn derivative_v(&self, u: f32, v: f32) -> Vec3 {
        let h = 0.001;
        let (_, (v_min, v_max)) = self.domain();
        let v_lo = (v - h).max(v_min);
        let v_hi = (v + h).min(v_max - 0.0001);
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
        let ((u_min, u_max), (v_min, v_max)) = self.domain();

        let num_verts = (divisions_u + 1) * (divisions_v + 1);
        let mut positions = Vec::with_capacity(num_verts);
        let mut normals = Vec::with_capacity(num_verts);
        let mut uvs = Vec::with_capacity(num_verts);

        // Generate vertices
        for i in 0..=divisions_u {
            let u = u_min + (i as f32 / divisions_u as f32) * (u_max - u_min);
            for j in 0..=divisions_v {
                let v = v_min + (j as f32 / divisions_v as f32) * (v_max - v_min);

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

    #[test]
    fn test_bilinear_patch() {
        let patch = nurbs_bilinear_patch(
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
        );

        // Corners should match control points
        let ((u_min, u_max), (v_min, v_max)) = patch.domain();

        let p00 = patch.evaluate(u_min, v_min);
        assert!((p00 - Vec3::new(0.0, 0.0, 0.0)).length() < 0.01);

        let p10 = patch.evaluate(u_max - 0.001, v_min);
        assert!((p10 - Vec3::new(1.0, 0.0, 0.0)).length() < 0.01);

        let p01 = patch.evaluate(u_min, v_max - 0.001);
        assert!((p01 - Vec3::new(0.0, 1.0, 0.0)).length() < 0.01);

        // Center should be average
        let center = patch.evaluate(0.5, 0.5);
        assert!((center - Vec3::new(0.5, 0.5, 0.0)).length() < 0.01);
    }

    #[test]
    fn test_nurbs_sphere() {
        let sphere = nurbs_sphere(Vec3::ZERO, 1.0);

        // Sample points should be on the sphere surface
        for i in 0..10 {
            for j in 0..10 {
                let ((u_min, u_max), (v_min, v_max)) = sphere.domain();
                let u = u_min + (i as f32 / 9.0) * (u_max - u_min - 0.001);
                let v = v_min + (j as f32 / 9.0) * (v_max - v_min - 0.001);

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

        // Sample points should have correct radius
        let ((u_min, u_max), (v_min, v_max)) = cylinder.domain();

        for i in 0..8 {
            let u = u_min + (i as f32 / 7.0) * (u_max - u_min - 0.001);
            let point = cylinder.evaluate(u, (v_min + v_max) / 2.0);

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

        let ((u_min, u_max), (v_min, v_max)) = cone.domain();

        // Base should have radius 1
        let base_point = cone.evaluate((u_min + u_max) / 2.0, v_min);
        let base_radius = (base_point.x * base_point.x + base_point.z * base_point.z).sqrt();
        assert!((base_radius - 1.0).abs() < 0.2);

        // Apex should be at (0, 2, 0)
        let apex = cone.evaluate((u_min + u_max) / 2.0, v_max - 0.001);
        assert!((apex - Vec3::new(0.0, 2.0, 0.0)).length() < 0.2);
    }
}
