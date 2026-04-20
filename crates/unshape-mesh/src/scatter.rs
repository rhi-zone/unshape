//! Scatter points on a mesh surface.
//!
//! Distributes points uniformly over a triangulated mesh surface, optionally
//! weighted by face area. Uses a seeded hash-based RNG for reproducibility.

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use glam::Vec3;

use crate::Mesh;

// ============================================================================
// Op struct
// ============================================================================

/// Scatters points uniformly on a mesh surface.
///
/// Points are placed randomly on triangulated faces. When `weight_by_area` is
/// true, each face contributes points proportional to its area (uniform surface
/// density). When false, each face gets an equal share (biased toward small faces).
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Scatter {
    /// Number of points to scatter.
    pub count: u32,
    /// Seed for the random number generator. Same seed → same result.
    pub seed: u64,
    /// If true, weight point density by face area (uniform surface coverage).
    /// If false, each face receives an equal number of points.
    pub weight_by_area: bool,
}

impl Default for Scatter {
    fn default() -> Self {
        Self {
            count: 100,
            seed: 0,
            weight_by_area: true,
        }
    }
}

impl Scatter {
    /// Creates a new scatter operation with the given count and seed.
    pub fn new(count: u32, seed: u64) -> Self {
        Self {
            count,
            seed,
            weight_by_area: true,
        }
    }

    /// Applies this operation to a mesh, returning scattered point data.
    pub fn apply(&self, mesh: &Mesh) -> ScatterResult {
        scatter(mesh, self)
    }
}

/// Result of a [`Scatter`] operation.
pub struct ScatterResult {
    /// World-space positions of the scattered points.
    pub positions: Vec<Vec3>,
    /// Surface normals at each scattered point (interpolated from face normal).
    pub normals: Vec<Vec3>,
    /// Index of the face each point landed on (into `mesh.indices / 3`).
    pub face_indices: Vec<usize>,
}

// ============================================================================
// Implementation
// ============================================================================

/// Simple splitmix64 hash for a u64 → u64 bijection.
fn splitmix64(x: u64) -> u64 {
    let x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    let x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

/// Minimal seeded RNG state.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(splitmix64(seed.wrapping_add(1)))
    }

    /// Returns the next pseudo-random u64.
    fn next_u64(&mut self) -> u64 {
        self.0 = splitmix64(self.0);
        self.0
    }

    /// Returns a pseudo-random f32 in [0, 1).
    fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 11) as f32 / (1u64 << 53) as f32
    }
}

/// Computes the area of a triangle from three vertices.
fn triangle_area(a: Vec3, b: Vec3, c: Vec3) -> f32 {
    (b - a).cross(c - a).length() * 0.5
}

/// Applies a [`Scatter`] op to a mesh.
pub fn scatter(mesh: &Mesh, op: &Scatter) -> ScatterResult {
    if mesh.indices.is_empty() || op.count == 0 {
        return ScatterResult {
            positions: Vec::new(),
            normals: Vec::new(),
            face_indices: Vec::new(),
        };
    }

    let face_count = mesh.indices.len() / 3;
    let mut rng = Rng::new(op.seed);

    // Build face normals (used for output normals and area computation).
    let face_normals: Vec<Vec3> = (0..face_count)
        .map(|fi| {
            let i0 = mesh.indices[fi * 3] as usize;
            let i1 = mesh.indices[fi * 3 + 1] as usize;
            let i2 = mesh.indices[fi * 3 + 2] as usize;
            let a = mesh.positions[i0];
            let b = mesh.positions[i1];
            let c = mesh.positions[i2];
            (b - a).cross(c - a).normalize_or_zero()
        })
        .collect();

    // Build face selection weights (cumulative distribution).
    let face_areas: Vec<f32> = (0..face_count)
        .map(|fi| {
            if op.weight_by_area {
                let i0 = mesh.indices[fi * 3] as usize;
                let i1 = mesh.indices[fi * 3 + 1] as usize;
                let i2 = mesh.indices[fi * 3 + 2] as usize;
                triangle_area(mesh.positions[i0], mesh.positions[i1], mesh.positions[i2])
            } else {
                1.0
            }
        })
        .collect();

    let total_weight: f32 = face_areas.iter().sum();

    // Build cumulative distribution for O(log n) face sampling.
    let mut cdf: Vec<f32> = Vec::with_capacity(face_count);
    let mut running = 0.0_f32;
    for &w in &face_areas {
        running += w / total_weight;
        cdf.push(running);
    }

    let mut positions = Vec::with_capacity(op.count as usize);
    let mut normals = Vec::with_capacity(op.count as usize);
    let mut face_indices_out = Vec::with_capacity(op.count as usize);

    for _ in 0..op.count {
        // Sample a face via binary search on the CDF.
        let u = rng.next_f32();
        let fi = cdf.partition_point(|&v| v < u).min(face_count - 1);

        let i0 = mesh.indices[fi * 3] as usize;
        let i1 = mesh.indices[fi * 3 + 1] as usize;
        let i2 = mesh.indices[fi * 3 + 2] as usize;
        let a = mesh.positions[i0];
        let b = mesh.positions[i1];
        let c = mesh.positions[i2];

        // Uniform barycentric sampling: square-root method.
        let r1 = rng.next_f32().sqrt();
        let r2 = rng.next_f32();
        let u_bary = 1.0 - r1;
        let v_bary = r1 * (1.0 - r2);
        let w_bary = r1 * r2;

        let point = a * u_bary + b * v_bary + c * w_bary;
        let normal = face_normals[fi];

        positions.push(point);
        normals.push(normal);
        face_indices_out.push(fi);
    }

    ScatterResult {
        positions,
        normals,
        face_indices: face_indices_out,
    }
}

// ============================================================================
// Method sugar on Mesh
// ============================================================================

impl Mesh {
    /// Scatters `count` points on this mesh surface.
    ///
    /// Sugar for `Scatter::new(count, seed).apply(self)`.
    pub fn scatter(&self, count: u32, seed: u64) -> ScatterResult {
        Scatter::new(count, seed).apply(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Cuboid;

    #[test]
    fn test_scatter_count() {
        let mesh = Cuboid::default().apply();
        let result = Scatter::new(50, 42).apply(&mesh);
        assert_eq!(result.positions.len(), 50);
        assert_eq!(result.normals.len(), 50);
        assert_eq!(result.face_indices.len(), 50);
    }

    #[test]
    fn test_scatter_reproducible() {
        let mesh = Cuboid::default().apply();
        let r1 = Scatter::new(20, 7).apply(&mesh);
        let r2 = Scatter::new(20, 7).apply(&mesh);
        for (a, b) in r1.positions.iter().zip(r2.positions.iter()) {
            assert!(a.distance(*b) < 1e-6, "same seed must give same result");
        }
    }

    #[test]
    fn test_scatter_different_seeds() {
        let mesh = Cuboid::default().apply();
        let r1 = Scatter::new(20, 0).apply(&mesh);
        let r2 = Scatter::new(20, 1).apply(&mesh);
        // Very unlikely all points are identical with different seeds.
        let identical = r1
            .positions
            .iter()
            .zip(r2.positions.iter())
            .all(|(a, b)| a.distance(*b) < 1e-6);
        assert!(!identical, "different seeds should give different results");
    }

    #[test]
    fn test_scatter_face_indices_in_bounds() {
        let mesh = Cuboid::default().apply();
        let face_count = mesh.indices.len() / 3;
        let result = Scatter::new(100, 0).apply(&mesh);
        for fi in &result.face_indices {
            assert!(*fi < face_count);
        }
    }

    #[test]
    fn test_scatter_empty_mesh() {
        let mesh = Mesh::new();
        let result = Scatter::new(10, 0).apply(&mesh);
        assert!(result.positions.is_empty());
    }

    #[test]
    fn test_scatter_normals_unit_length() {
        let mesh = Cuboid::default().apply();
        let result = Scatter::new(50, 0).apply(&mesh);
        for (i, n) in result.normals.iter().enumerate() {
            let len = n.length();
            assert!(
                (len - 1.0).abs() < 1e-4 || len < 1e-6,
                "normal {i} should be unit length, got {len}"
            );
        }
    }
}
