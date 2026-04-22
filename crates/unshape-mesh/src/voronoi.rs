//! Voronoi fracture op for mesh decomposition.
//!
//! Decomposes a mesh into cell-shaped shards based on Voronoi diagram
//! computed from seed points scattered on the mesh surface.
//!
//! # Algorithm
//!
//! 1. Generate seed points on the mesh surface using [`Scatter`].
//! 2. Assign each triangle to its nearest seed (nearest centroid).
//! 3. Clip triangles that straddle cell boundaries using Sutherland-Hodgman
//!    clipping against bisector half-planes.
//! 4. Apply `interior_offset` to shrink each shard inward.
//!
//! Note: boundary clipping is approximate — triangles wholly on one side of a
//! bisector plane are kept; those that straddle it are split. Cap faces are
//! not generated for the interior cuts, so shards may be open meshes along
//! shared boundaries.

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use glam::Vec3;

use crate::{Mesh, Scatter};

// ============================================================================
// Op struct
// ============================================================================

/// Fractures a mesh into Voronoi cell shards.
///
/// Seed points are scattered on the mesh surface (area-weighted), then each
/// triangle is assigned to — or clipped against — its nearest cell.
///
/// # Example
///
/// ```
/// use unshape_mesh::{Cuboid, VoronoiFracture};
///
/// let cube = Cuboid::default().apply();
/// let result = VoronoiFracture { cell_count: 5, seed: 42, interior_offset: 0.0 }.apply(&cube);
/// assert_eq!(result.shards.len(), 5);
/// assert_eq!(result.cell_centers.len(), 5);
/// ```
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct VoronoiFracture {
    /// Number of Voronoi cells (shards) to produce.
    pub cell_count: usize,
    /// Seed for reproducible fracture patterns.
    pub seed: u64,
    /// Distance to shrink each shard inward along centroid-to-vertex direction.
    /// `0.0` = no gap between shards.
    pub interior_offset: f32,
}

impl Default for VoronoiFracture {
    fn default() -> Self {
        Self {
            cell_count: 10,
            seed: 0,
            interior_offset: 0.0,
        }
    }
}

impl VoronoiFracture {
    /// Creates a new fracture op with the given cell count and seed.
    pub fn new(cell_count: usize, seed: u64) -> Self {
        Self {
            cell_count,
            seed,
            interior_offset: 0.0,
        }
    }

    /// Applies this operation to a mesh, returning all shards and their seed centers.
    pub fn apply(&self, mesh: &Mesh) -> FractureResult {
        voronoi_fracture(mesh, self)
    }
}

/// Result of a [`VoronoiFracture`] operation.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct FractureResult {
    /// One mesh per Voronoi cell. Each shard contains only triangles (and
    /// clipped triangle fragments) that belong to that cell.
    pub shards: Vec<Mesh>,
    /// The seed point for each cell, on the surface of the original mesh.
    pub cell_centers: Vec<Vec3>,
}

// ============================================================================
// Implementation
// ============================================================================

/// Applies a [`VoronoiFracture`] op to a mesh.
pub fn voronoi_fracture(mesh: &Mesh, op: &VoronoiFracture) -> FractureResult {
    let cell_count = op.cell_count;

    if mesh.indices.is_empty() || cell_count == 0 {
        return FractureResult {
            shards: Vec::new(),
            cell_centers: Vec::new(),
        };
    }

    // --- Step 1: generate seed points (cell centers) on the mesh surface ----
    let scatter = Scatter {
        count: cell_count as u32,
        seed: op.seed,
        weight_by_area: true,
    };
    let scattered = scatter.apply(mesh);

    // If the mesh has no area (degenerate), scatter may return fewer points.
    let actual_count = scattered.positions.len();
    if actual_count == 0 {
        return FractureResult {
            shards: Vec::new(),
            cell_centers: Vec::new(),
        };
    }
    let cell_centers = scattered.positions.clone();

    // --- Step 2 & 3: for each triangle, assign to nearest cell, with clipping
    // We build per-cell lists of (polygon vertices) to accumulate.
    let mut cell_polys: Vec<Vec<[Vec3; 3]>> = vec![Vec::new(); actual_count];

    let face_count = mesh.indices.len() / 3;
    for fi in 0..face_count {
        let i0 = mesh.indices[fi * 3] as usize;
        let i1 = mesh.indices[fi * 3 + 1] as usize;
        let i2 = mesh.indices[fi * 3 + 2] as usize;
        let v0 = mesh.positions[i0];
        let v1 = mesh.positions[i1];
        let v2 = mesh.positions[i2];

        // Find the nearest cell for the triangle centroid.
        let centroid = (v0 + v1 + v2) / 3.0;
        let primary_cell = nearest_cell(&cell_centers, centroid);

        // Clip the triangle polygon against bisector half-planes for every
        // other cell that any vertex is closer to.  We only clip against cells
        // that are "relevant" to this triangle (candidates = cells nearest to
        // each vertex).
        let candidate_cells: Vec<usize> = {
            let mut c = vec![primary_cell];
            for &v in &[v0, v1, v2] {
                let nc = nearest_cell(&cell_centers, v);
                if !c.contains(&nc) {
                    c.push(nc);
                }
            }
            c
        };

        if candidate_cells.len() == 1 {
            // All vertices in the same cell — keep whole triangle.
            cell_polys[primary_cell].push([v0, v1, v2]);
        } else {
            // Clip the triangle successively against each bisector plane,
            // keeping the fragment on the primary_cell side.
            let mut poly: Vec<Vec3> = vec![v0, v1, v2];

            for &other in &candidate_cells {
                if other == primary_cell {
                    continue;
                }
                let (plane_pt, plane_normal) =
                    bisector_plane(cell_centers[primary_cell], cell_centers[other]);
                poly = clip_polygon_by_plane(&poly, plane_pt, plane_normal);
                if poly.is_empty() {
                    break;
                }
            }

            // Triangulate the clipped polygon (fan triangulation).
            if poly.len() >= 3 {
                for i in 1..poly.len() - 1 {
                    cell_polys[primary_cell].push([poly[0], poly[i], poly[i + 1]]);
                }
            }
        }
    }

    // --- Step 4: build a Mesh per cell, apply interior_offset ---------------
    let shards: Vec<Mesh> = cell_polys
        .into_iter()
        .enumerate()
        .map(|(ci, tris)| build_shard_mesh(&tris, cell_centers[ci], op.interior_offset))
        .collect();

    FractureResult {
        shards,
        cell_centers,
    }
}

/// Returns the index of the cell center nearest to `point`.
fn nearest_cell(centers: &[Vec3], point: Vec3) -> usize {
    centers
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| {
            a.distance_squared(point)
                .partial_cmp(&b.distance_squared(point))
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(i, _)| i)
        .unwrap_or(0)
}

/// Returns the midpoint and outward normal of the bisector plane between two
/// cell centers.  The normal points from `a` toward `b` (i.e., the half-space
/// containing `a` is on the negative side of this plane).
fn bisector_plane(a: Vec3, b: Vec3) -> (Vec3, Vec3) {
    let mid = (a + b) * 0.5;
    let normal = (b - a).normalize_or_zero();
    (mid, normal)
}

/// Sutherland-Hodgman polygon clipping against a single half-plane.
///
/// Keeps the half-space on the *negative* side of the plane (i.e., points
/// where `dot(p - plane_pt, plane_normal) <= 0`).
fn clip_polygon_by_plane(poly: &[Vec3], plane_pt: Vec3, plane_normal: Vec3) -> Vec<Vec3> {
    if poly.is_empty() {
        return Vec::new();
    }

    let signed_dist = |p: Vec3| (p - plane_pt).dot(plane_normal);

    let mut output: Vec<Vec3> = Vec::new();
    let n = poly.len();

    for i in 0..n {
        let curr = poly[i];
        let next = poly[(i + 1) % n];
        let dc = signed_dist(curr);
        let dn = signed_dist(next);

        if dc <= 0.0 {
            output.push(curr);
        }
        // Edge crosses the plane — add intersection point.
        if (dc < 0.0 && dn > 0.0) || (dc > 0.0 && dn < 0.0) {
            let t = dc / (dc - dn);
            output.push(curr + t * (next - curr));
        }
    }

    output
}

/// Builds a [`Mesh`] from a list of triangles, applying the interior offset.
fn build_shard_mesh(tris: &[[Vec3; 3]], cell_center: Vec3, interior_offset: f32) -> Mesh {
    if tris.is_empty() {
        return Mesh::new();
    }

    // Compute the centroid of all triangle vertices for offset direction.
    let vertex_centroid: Vec3 = {
        let sum: Vec3 = tris.iter().flat_map(|t| t.iter().copied()).sum();
        let count = (tris.len() * 3) as f32;
        sum / count
    };

    // Reference point for shrink direction — prefer the scatter seed if available
    // (passed as cell_center), fall back to vertex centroid.
    let origin = if cell_center.length_squared() > 0.0 {
        cell_center
    } else {
        vertex_centroid
    };

    let mut mesh = Mesh::with_capacity(tris.len() * 3, tris.len());

    for tri in tris {
        let base_idx = mesh.positions.len() as u32;

        for &v in tri {
            let offset_v = if interior_offset != 0.0 {
                let dir = (v - origin).normalize_or_zero();
                v - dir * interior_offset
            } else {
                v
            };
            mesh.positions.push(offset_v);
        }

        // Compute flat normal for the triangle.
        let a = tri[0];
        let b = tri[1];
        let c = tri[2];
        let normal = (b - a).cross(c - a).normalize_or_zero();
        mesh.normals.push(normal);
        mesh.normals.push(normal);
        mesh.normals.push(normal);

        mesh.indices.push(base_idx);
        mesh.indices.push(base_idx + 1);
        mesh.indices.push(base_idx + 2);
    }

    mesh
}

// ============================================================================
// Method sugar on Mesh
// ============================================================================

impl Mesh {
    /// Fractures this mesh into Voronoi cell shards.
    ///
    /// Sugar for `VoronoiFracture::new(cell_count, seed).apply(self)`.
    pub fn voronoi_fracture(&self, cell_count: usize, seed: u64) -> FractureResult {
        VoronoiFracture::new(cell_count, seed).apply(self)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Cuboid;

    #[test]
    fn test_fracture_shard_count() {
        let mesh = Cuboid::default().apply();
        let result = VoronoiFracture::new(5, 42).apply(&mesh);
        assert_eq!(result.shards.len(), 5, "should produce exactly 5 shards");
        assert_eq!(
            result.cell_centers.len(),
            5,
            "should produce exactly 5 cell centers"
        );
    }

    #[test]
    fn test_fracture_shards_nonempty() {
        let mesh = Cuboid::default().apply();
        let result = VoronoiFracture::new(5, 42).apply(&mesh);
        for (i, shard) in result.shards.iter().enumerate() {
            assert!(
                shard.triangle_count() > 0,
                "shard {i} should have at least one triangle"
            );
        }
    }

    #[test]
    fn test_fracture_combined_vertices_cover_original() {
        let mesh = Cuboid::default().apply();
        let result = VoronoiFracture::new(5, 42).apply(&mesh);

        // Every position in every shard should be within the bounding box of
        // the original mesh (with small tolerance).
        let bb_min = mesh
            .positions
            .iter()
            .fold(Vec3::splat(f32::MAX), |acc, &p| acc.min(p));
        let bb_max = mesh
            .positions
            .iter()
            .fold(Vec3::splat(f32::MIN), |acc, &p| acc.max(p));
        let tol = 1e-4;

        for (si, shard) in result.shards.iter().enumerate() {
            for (vi, &pos) in shard.positions.iter().enumerate() {
                assert!(
                    pos.x >= bb_min.x - tol
                        && pos.x <= bb_max.x + tol
                        && pos.y >= bb_min.y - tol
                        && pos.y <= bb_max.y + tol
                        && pos.z >= bb_min.z - tol
                        && pos.z <= bb_max.z + tol,
                    "shard {si} vertex {vi} at {pos:?} is outside original bounding box"
                );
            }
        }
    }

    #[test]
    fn test_fracture_different_seeds() {
        let mesh = Cuboid::default().apply();
        let r1 = VoronoiFracture::new(5, 0).apply(&mesh);
        let r2 = VoronoiFracture::new(5, 99).apply(&mesh);

        // Cell centers should differ with different seeds.
        let all_same = r1
            .cell_centers
            .iter()
            .zip(r2.cell_centers.iter())
            .all(|(a, b)| a.distance(*b) < 1e-4);
        assert!(
            !all_same,
            "different seeds should produce different patterns"
        );
    }

    #[test]
    fn test_fracture_cell_centers_count() {
        let mesh = Cuboid::default().apply();
        for count in [1, 3, 8] {
            let result = VoronoiFracture::new(count, 7).apply(&mesh);
            assert_eq!(
                result.cell_centers.len(),
                count,
                "cell_centers.len() should equal cell_count={count}"
            );
        }
    }

    #[test]
    fn test_fracture_interior_offset() {
        let mesh = Cuboid::default().apply();
        let no_gap = VoronoiFracture {
            cell_count: 3,
            seed: 1,
            interior_offset: 0.0,
        }
        .apply(&mesh);
        let with_gap = VoronoiFracture {
            cell_count: 3,
            seed: 1,
            interior_offset: 0.05,
        }
        .apply(&mesh);

        // With a gap, vertex positions should differ from no-gap.
        let any_different = no_gap
            .shards
            .iter()
            .zip(with_gap.shards.iter())
            .flat_map(|(s1, s2)| s1.positions.iter().zip(s2.positions.iter()))
            .any(|(a, b)| a.distance(*b) > 1e-6);
        assert!(
            any_different,
            "interior_offset > 0 should move vertices inward"
        );
    }

    #[test]
    fn test_fracture_method_sugar() {
        let mesh = Cuboid::default().apply();
        let result = mesh.voronoi_fracture(4, 123);
        assert_eq!(result.shards.len(), 4);
        assert_eq!(result.cell_centers.len(), 4);
    }
}
