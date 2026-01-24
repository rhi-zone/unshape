//! Mesh curvature calculations.
//!
//! Computes discrete curvature values at mesh vertices:
//! - Gaussian curvature (K): product of principal curvatures
//! - Mean curvature (H): average of principal curvatures
//! - Principal curvatures (k1, k2): maximum and minimum curvatures
//!
//! # Example
//!
//! ```
//! use unshape_mesh::{UvSphere, gaussian_curvature, mean_curvature};
//!
//! let mesh = UvSphere::default().apply();
//! let k_values = gaussian_curvature(&mesh);
//! let h_values = mean_curvature(&mesh);
//!
//! // For a unit sphere, K ≈ 1 and H ≈ 1 at all vertices
//! ```

use crate::Mesh;
use glam::Vec3;
use std::collections::HashMap;
use std::f32::consts::PI;

/// Result of curvature analysis at each vertex.
#[derive(Debug, Clone)]
pub struct CurvatureResult {
    /// Gaussian curvature (K) per vertex.
    pub gaussian: Vec<f32>,
    /// Mean curvature (H) per vertex.
    pub mean: Vec<f32>,
    /// Maximum principal curvature (k1) per vertex.
    pub k1: Vec<f32>,
    /// Minimum principal curvature (k2) per vertex.
    pub k2: Vec<f32>,
}

/// Computes Gaussian curvature at each vertex using the angle deficit method.
///
/// For a vertex v, the discrete Gaussian curvature is:
/// K(v) = (2π - Σθ) / A
///
/// Where Σθ is the sum of angles at incident faces and A is the mixed Voronoi area.
///
/// # Returns
/// A vector of Gaussian curvature values, one per vertex.
pub fn gaussian_curvature(mesh: &Mesh) -> Vec<f32> {
    let n = mesh.positions.len();
    if n == 0 || mesh.indices.is_empty() {
        return vec![];
    }

    let mut angle_sum = vec![0.0f32; n];
    let mut area = vec![0.0f32; n];

    // Process each triangle
    for tri in mesh.indices.chunks(3) {
        if tri.len() < 3 {
            continue;
        }
        let i0 = tri[0] as usize;
        let i1 = tri[1] as usize;
        let i2 = tri[2] as usize;

        if i0 >= n || i1 >= n || i2 >= n {
            continue;
        }

        let p0 = mesh.positions[i0];
        let p1 = mesh.positions[i1];
        let p2 = mesh.positions[i2];

        // Edge vectors
        let e01 = p1 - p0;
        let e02 = p2 - p0;
        let e12 = p2 - p1;

        // Triangle area
        let tri_area = e01.cross(e02).length() * 0.5;
        if tri_area < 1e-10 {
            continue;
        }

        // Angles at each vertex
        let angle0 = angle_between(e01, e02);
        let angle1 = angle_between(-e01, e12);
        let angle2 = angle_between(-e02, -e12);

        angle_sum[i0] += angle0;
        angle_sum[i1] += angle1;
        angle_sum[i2] += angle2;

        // Mixed Voronoi area contribution (simplified: use barycentric area = tri_area / 3)
        let area_contrib = tri_area / 3.0;
        area[i0] += area_contrib;
        area[i1] += area_contrib;
        area[i2] += area_contrib;
    }

    // Compute Gaussian curvature: K = (2π - angle_sum) / area
    (0..n)
        .map(|i| {
            if area[i] > 1e-10 {
                (2.0 * PI - angle_sum[i]) / area[i]
            } else {
                0.0
            }
        })
        .collect()
}

/// Computes mean curvature at each vertex using the cotangent Laplacian.
///
/// The mean curvature normal is: H·n = (1/2A) Σ (cot α + cot β) (vj - vi)
///
/// Where the sum is over edges, and α, β are the opposite angles.
///
/// # Returns
/// A vector of mean curvature values (signed), one per vertex.
pub fn mean_curvature(mesh: &Mesh) -> Vec<f32> {
    let n = mesh.positions.len();
    if n == 0 || mesh.indices.is_empty() {
        return vec![];
    }

    // Build edge to opposite angles map
    let mut edge_cotans: HashMap<(usize, usize), Vec<f32>> = HashMap::new();

    for tri in mesh.indices.chunks(3) {
        if tri.len() < 3 {
            continue;
        }
        let indices = [tri[0] as usize, tri[1] as usize, tri[2] as usize];

        for k in 0..3 {
            let i = indices[k];
            let j = indices[(k + 1) % 3];
            let o = indices[(k + 2) % 3]; // opposite vertex

            if i >= n || j >= n || o >= n {
                continue;
            }

            let pi = mesh.positions[i];
            let pj = mesh.positions[j];
            let po = mesh.positions[o];

            // Angle at opposite vertex
            let e1 = pi - po;
            let e2 = pj - po;
            let cos_angle = e1.dot(e2) / (e1.length() * e2.length() + 1e-10);
            let sin_angle = e1.cross(e2).length() / (e1.length() * e2.length() + 1e-10);
            let cot = cos_angle / (sin_angle + 1e-10);

            let edge = if i < j { (i, j) } else { (j, i) };
            edge_cotans.entry(edge).or_default().push(cot);
        }
    }

    // Compute Laplacian and area for each vertex
    let mut laplacian = vec![Vec3::ZERO; n];
    let mut area = vec![0.0f32; n];

    for tri in mesh.indices.chunks(3) {
        if tri.len() < 3 {
            continue;
        }
        let i0 = tri[0] as usize;
        let i1 = tri[1] as usize;
        let i2 = tri[2] as usize;

        if i0 >= n || i1 >= n || i2 >= n {
            continue;
        }

        let p0 = mesh.positions[i0];
        let p1 = mesh.positions[i1];
        let p2 = mesh.positions[i2];

        let tri_area = (p1 - p0).cross(p2 - p0).length() * 0.5;
        let area_contrib = tri_area / 3.0;
        area[i0] += area_contrib;
        area[i1] += area_contrib;
        area[i2] += area_contrib;
    }

    // Sum cotangent-weighted edge differences
    for (&(i, j), cotans) in &edge_cotans {
        let cotan_sum: f32 = cotans.iter().sum();
        let pi = mesh.positions[i];
        let pj = mesh.positions[j];
        let edge_vec = pj - pi;

        laplacian[i] += cotan_sum * edge_vec;
        laplacian[j] -= cotan_sum * edge_vec;
    }

    // Compute mean curvature: H = |Laplacian| / (4 * area)
    // The sign is determined by comparing with vertex normal
    let normals = compute_vertex_normals(mesh);

    (0..n)
        .map(|i| {
            if area[i] > 1e-10 {
                let h_vec = laplacian[i] / (4.0 * area[i]);
                let h_mag = h_vec.length();
                // Sign based on normal direction
                if h_vec.dot(normals[i]) < 0.0 {
                    -h_mag
                } else {
                    h_mag
                }
            } else {
                0.0
            }
        })
        .collect()
}

/// Computes principal curvatures (k1, k2) from Gaussian and mean curvature.
///
/// Given H (mean) and K (Gaussian):
/// - k1 = H + sqrt(H² - K)
/// - k2 = H - sqrt(H² - K)
///
/// # Returns
/// A tuple of (k1, k2) vectors, where k1 >= k2.
pub fn principal_curvatures(mesh: &Mesh) -> (Vec<f32>, Vec<f32>) {
    let gaussian = gaussian_curvature(mesh);
    let mean = mean_curvature(mesh);

    let n = gaussian.len();
    let mut k1 = vec![0.0f32; n];
    let mut k2 = vec![0.0f32; n];

    for i in 0..n {
        let h = mean[i];
        let k = gaussian[i];
        let discriminant = h * h - k;

        if discriminant >= 0.0 {
            let sqrt_disc = discriminant.sqrt();
            k1[i] = h + sqrt_disc;
            k2[i] = h - sqrt_disc;
        } else {
            // Complex roots - use mean curvature as approximation
            k1[i] = h;
            k2[i] = h;
        }
    }

    (k1, k2)
}

/// Computes all curvature values for a mesh.
///
/// This is more efficient than calling individual functions separately
/// when you need multiple curvature types.
pub fn compute_curvature(mesh: &Mesh) -> CurvatureResult {
    let gaussian = gaussian_curvature(mesh);
    let mean = mean_curvature(mesh);

    let n = gaussian.len();
    let mut k1 = vec![0.0f32; n];
    let mut k2 = vec![0.0f32; n];

    for i in 0..n {
        let h = mean[i];
        let k = gaussian[i];
        let discriminant = h * h - k;

        if discriminant >= 0.0 {
            let sqrt_disc = discriminant.sqrt();
            k1[i] = h + sqrt_disc;
            k2[i] = h - sqrt_disc;
        } else {
            k1[i] = h;
            k2[i] = h;
        }
    }

    CurvatureResult {
        gaussian,
        mean,
        k1,
        k2,
    }
}

/// Computes the angle between two vectors in radians.
fn angle_between(a: Vec3, b: Vec3) -> f32 {
    let cos_angle = a.dot(b) / (a.length() * b.length() + 1e-10);
    cos_angle.clamp(-1.0, 1.0).acos()
}

/// Computes vertex normals by averaging face normals.
fn compute_vertex_normals(mesh: &Mesh) -> Vec<Vec3> {
    let n = mesh.positions.len();
    let mut normals = vec![Vec3::ZERO; n];

    for tri in mesh.indices.chunks(3) {
        if tri.len() < 3 {
            continue;
        }
        let i0 = tri[0] as usize;
        let i1 = tri[1] as usize;
        let i2 = tri[2] as usize;

        if i0 >= n || i1 >= n || i2 >= n {
            continue;
        }

        let p0 = mesh.positions[i0];
        let p1 = mesh.positions[i1];
        let p2 = mesh.positions[i2];

        let face_normal = (p1 - p0).cross(p2 - p0);
        normals[i0] += face_normal;
        normals[i1] += face_normal;
        normals[i2] += face_normal;
    }

    for normal in &mut normals {
        let len = normal.length();
        if len > 1e-10 {
            *normal /= len;
        }
    }

    normals
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Cuboid, UvSphere};

    #[test]
    fn test_gaussian_curvature_sphere() {
        // A sphere should have positive Gaussian curvature everywhere
        let mesh = UvSphere::new(1.0, 16, 16).apply();
        let k = gaussian_curvature(&mesh);

        assert!(!k.is_empty());
        // Most vertices should have positive curvature (except poles may be irregular)
        let positive_count = k.iter().filter(|&&v| v > 0.0).count();
        assert!(positive_count > k.len() / 2);
    }

    #[test]
    fn test_mean_curvature_sphere() {
        // A sphere should have positive mean curvature everywhere
        let mesh = UvSphere::new(1.0, 16, 16).apply();
        let h = mean_curvature(&mesh);

        assert!(!h.is_empty());
        // Most vertices should have positive mean curvature
        let positive_count = h.iter().filter(|&&v| v > 0.0).count();
        assert!(positive_count > h.len() / 2);
    }

    #[test]
    fn test_principal_curvatures() {
        let mesh = UvSphere::default().apply();
        let (k1, k2) = principal_curvatures(&mesh);

        assert_eq!(k1.len(), mesh.positions.len());
        assert_eq!(k2.len(), mesh.positions.len());

        // k1 should be >= k2
        for i in 0..k1.len() {
            assert!(k1[i] >= k2[i] - 1e-5, "k1 should be >= k2");
        }
    }

    #[test]
    fn test_compute_curvature() {
        let mesh = UvSphere::default().apply();
        let result = compute_curvature(&mesh);

        assert_eq!(result.gaussian.len(), mesh.positions.len());
        assert_eq!(result.mean.len(), mesh.positions.len());
        assert_eq!(result.k1.len(), mesh.positions.len());
        assert_eq!(result.k2.len(), mesh.positions.len());
    }

    #[test]
    fn test_box_curvature() {
        // A box has zero Gaussian curvature on faces, undefined at edges/corners
        let mesh = Cuboid::default().apply();
        let k = gaussian_curvature(&mesh);
        let h = mean_curvature(&mesh);

        assert_eq!(k.len(), mesh.positions.len());
        assert_eq!(h.len(), mesh.positions.len());
    }

    #[test]
    fn test_empty_mesh() {
        let mesh = Mesh::default();
        let k = gaussian_curvature(&mesh);
        let h = mean_curvature(&mesh);

        assert!(k.is_empty());
        assert!(h.is_empty());
    }

    #[test]
    fn test_angle_between() {
        let a = Vec3::X;
        let b = Vec3::Y;
        let angle = angle_between(a, b);
        assert!((angle - PI / 2.0).abs() < 1e-5);

        let c = Vec3::X;
        let d = Vec3::X;
        let angle2 = angle_between(c, d);
        assert!(angle2.abs() < 1e-5);
    }
}
