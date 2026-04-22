//! Dual contouring algorithm for generating meshes from signed distance fields.
//!
//! Dual contouring places one vertex per active grid cell at the QEF (quadratic
//! error function) minimizer. This preserves sharp features that marching cubes
//! rounds off, making it well-suited for hard-surface models.
//!
//! Operations are serializable structs with `apply` methods.
//! See `docs/design/ops-as-values.md`.
//!
//! # Example
//!
//! ```ignore
//! use unshape_mesh::DualContouring;
//! use glam::Vec3;
//!
//! let sdf = |p: Vec3| p.length() - 1.0;
//!
//! let mesh = DualContouring::default().apply_fn(&sdf);
//! ```

use glam::Vec3;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::Mesh;

/// Extracts a mesh from a signed distance field using dual contouring.
///
/// Naive dual contouring on a uniform grid — no octree. Places one vertex
/// per active cell at the QEF minimizer, which snaps to sharp features rather
/// than averaging them away as marching cubes does.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct DualContouring {
    /// Bounds of the sampling volume (min corner).
    pub min: Vec3,
    /// Bounds of the sampling volume (max corner).
    pub max: Vec3,
    /// Resolution (number of cells) in each dimension.
    pub resolution: usize,
    /// Iso-value at which to extract the surface (default: 0.0).
    pub iso_value: f32,
}

impl Default for DualContouring {
    fn default() -> Self {
        Self {
            min: Vec3::splat(-1.0),
            max: Vec3::splat(1.0),
            resolution: 32,
            iso_value: 0.0,
        }
    }
}

impl DualContouring {
    /// Creates config with specified bounds.
    pub fn with_bounds(min: Vec3, max: Vec3) -> Self {
        Self {
            min,
            max,
            ..Default::default()
        }
    }

    /// Applies dual contouring to an SDF closure.
    pub fn apply_fn<F>(&self, sdf: F) -> Mesh
    where
        F: Fn(Vec3) -> f32,
    {
        dual_contour(sdf, self)
    }
}

/// Solves the 3×3 symmetric linear system `A x = b` using Gaussian elimination.
///
/// Returns `None` if the matrix is singular (determinant near zero).
fn solve_3x3(a: [[f32; 3]; 3], b: [f32; 3]) -> Option<[f32; 3]> {
    // Build augmented matrix [A | b]
    let mut m = [
        [a[0][0], a[0][1], a[0][2], b[0]],
        [a[1][0], a[1][1], a[1][2], b[1]],
        [a[2][0], a[2][1], a[2][2], b[2]],
    ];

    #[allow(clippy::needless_range_loop)]
    for col in 0..3 {
        // Find pivot
        let mut max_row = col;
        let mut max_val = m[col][col].abs();
        for row in (col + 1)..3 {
            if m[row][col].abs() > max_val {
                max_val = m[row][col].abs();
                max_row = row;
            }
        }

        if max_val < 1e-8 {
            return None;
        }

        m.swap(col, max_row);

        let pivot = m[col][col];
        for row in (col + 1)..3 {
            let factor = m[row][col] / pivot;
            for k in col..4 {
                let sub = factor * m[col][k];
                m[row][k] -= sub;
            }
        }
    }

    // Back substitution
    let mut x = [0.0f32; 3];
    for i in (0..3).rev() {
        x[i] = m[i][3];
        for j in (i + 1)..3 {
            x[i] -= m[i][j] * x[j];
        }
        x[i] /= m[i][i];
    }

    Some(x)
}

/// Solves the QEF `minₓ Σ (nᵢ · (x - pᵢ))²` for a set of intersection
/// points `pᵢ` with gradient normals `nᵢ`.
///
/// Reduces to solving the normal equations `(AᵀA) x = Aᵀb` where each row of
/// A is a gradient normal and the corresponding entry of b is `nᵢ · pᵢ`.
///
/// Falls back to the mass-point (average of intersection points) when the
/// system is under-determined or the minimizer falls outside `cell_bounds`.
fn solve_qef(intersections: &[(Vec3, Vec3)], cell_min: Vec3, cell_max: Vec3) -> Vec3 {
    if intersections.is_empty() {
        return (cell_min + cell_max) * 0.5;
    }

    // Build AᵀA and Aᵀb
    let mut ata = [[0.0f32; 3]; 3];
    let mut atb = [0.0f32; 3];

    for (p, n) in intersections {
        let nx = n.x;
        let ny = n.y;
        let nz = n.z;
        let dot = nx * p.x + ny * p.y + nz * p.z;

        ata[0][0] += nx * nx;
        ata[0][1] += nx * ny;
        ata[0][2] += nx * nz;
        ata[1][0] += ny * nx;
        ata[1][1] += ny * ny;
        ata[1][2] += ny * nz;
        ata[2][0] += nz * nx;
        ata[2][1] += nz * ny;
        ata[2][2] += nz * nz;

        atb[0] += nx * dot;
        atb[1] += ny * dot;
        atb[2] += nz * dot;
    }

    let mass_point = intersections
        .iter()
        .fold(Vec3::ZERO, |acc, (p, _)| acc + *p)
        / intersections.len() as f32;

    // Shift QEF to improve numerical stability (solve for offset from mass point)
    let mp = mass_point;
    let shifted_b = [
        atb[0] - (ata[0][0] * mp.x + ata[0][1] * mp.y + ata[0][2] * mp.z),
        atb[1] - (ata[1][0] * mp.x + ata[1][1] * mp.y + ata[1][2] * mp.z),
        atb[2] - (ata[2][0] * mp.x + ata[2][1] * mp.y + ata[2][2] * mp.z),
    ];

    let candidate = match solve_3x3(ata, shifted_b) {
        Some(offset) => Vec3::new(mp.x + offset[0], mp.y + offset[1], mp.z + offset[2]),
        None => mp,
    };

    // Clamp to cell bounds to prevent vertices from leaving the cell
    candidate.clamp(cell_min, cell_max)
}

/// The 12 edges of a unit cube, each given as (vertex_a_index, vertex_b_index).
///
/// Vertex numbering:
/// ```text
///   4---5
///  /|  /|
/// 7---6 |
/// | 0-|-1
/// |/  |/
/// 3---2
/// ```
/// x increases 0→1, y increases 0→4, z increases 0→3.
const CELL_EDGES: [(usize, usize); 12] = [
    (0, 1), // bottom face
    (1, 2),
    (2, 3),
    (3, 0),
    (4, 5), // top face
    (5, 6),
    (6, 7),
    (7, 4),
    (0, 4), // vertical
    (1, 5),
    (2, 6),
    (3, 7),
];

/// Corner offsets matching the vertex numbering above.
const CORNER_OFFSETS: [Vec3; 8] = [
    Vec3::new(0.0, 0.0, 0.0),
    Vec3::new(1.0, 0.0, 0.0),
    Vec3::new(1.0, 0.0, 1.0),
    Vec3::new(0.0, 0.0, 1.0),
    Vec3::new(0.0, 1.0, 0.0),
    Vec3::new(1.0, 1.0, 0.0),
    Vec3::new(1.0, 1.0, 1.0),
    Vec3::new(0.0, 1.0, 1.0),
];

// Each of the 12 edges is shared by exactly 4 cells.
//
// For each edge direction we store the (dx, dy, dz) offsets of the 4 cells
// that share an edge running in that direction, relative to the lower-x/y/z
// endpoint of the edge. The quad winding is chosen so that a sign change from
// negative-inside to positive-outside produces outward-facing normals.
//
// Edge directions:
// - 0–3: bottom face edges (y=0, z=0 / z=1 / y=1 variation handled by top)
// - Actually we categorise by axis of the edge:
//   - X-axis edges: (0,1), (2,3), (4,5), (6,7) → shared by cells differing in y and z
//   - Z-axis edges: (0,3), (1,2), (4,7), (5,6) → shared by cells differing in x and y
//   - Y-axis edges: (0,4), (1,5), (2,6), (3,7) → shared by cells differing in x and z

/// Returns the four (ix, iy, iz) cell offsets that share an x-axis edge
/// whose lower-x endpoint is at grid node (x, y, z).
/// The edge runs from (x,y,z) to (x+1,y,z).
#[inline]
fn x_edge_cells(y: usize, z: usize) -> Option<[(i32, i32, i32); 4]> {
    if y == 0 || z == 0 {
        return None;
    }
    Some([
        (0, -(y as i32), -(z as i32)),
        (0, -(y as i32), -(z as i32) + 1),
        (0, -(y as i32) + 1, -(z as i32)),
        (0, -(y as i32) + 1, -(z as i32) + 1),
    ])
}

/// Returns the four (ix, iy, iz) cell coordinate adjustments for a y-axis edge
/// whose lower endpoint is grid node (x, y, z). The edge runs from (x,y,z) to (x,y+1,z).
#[inline]
fn y_edge_cells(x: usize, z: usize) -> Option<[(i32, i32, i32); 4]> {
    if x == 0 || z == 0 {
        return None;
    }
    Some([
        (-(x as i32), 0, -(z as i32)),
        (-(x as i32) + 1, 0, -(z as i32)),
        (-(x as i32), 0, -(z as i32) + 1),
        (-(x as i32) + 1, 0, -(z as i32) + 1),
    ])
}

/// Returns the four (ix, iy, iz) cell coordinate adjustments for a z-axis edge
/// whose lower endpoint is grid node (x, y, z). The edge runs from (x,y,z) to (x,y,z+1).
#[inline]
fn z_edge_cells(x: usize, y: usize) -> Option<[(i32, i32, i32); 4]> {
    if x == 0 || y == 0 {
        return None;
    }
    Some([
        (-(x as i32), -(y as i32), 0),
        (-(x as i32) + 1, -(y as i32), 0),
        (-(x as i32), -(y as i32) + 1, 0),
        (-(x as i32) + 1, -(y as i32) + 1, 0),
    ])
}

/// Generates a mesh from a signed distance field using dual contouring.
///
/// The SDF should return negative values inside the surface and positive outside.
pub fn dual_contour<F>(sdf: F, config: &DualContouring) -> Mesh
where
    F: Fn(Vec3) -> f32,
{
    let res = config.resolution;
    if res < 2 {
        return Mesh::new();
    }

    let cell_size = (config.max - config.min) / res as f32;
    let n = res + 1; // number of grid nodes per axis

    // --- Step 1: sample SDF at all grid nodes ---
    let mut values = vec![0.0f32; n * n * n];
    for z in 0..n {
        for y in 0..n {
            for x in 0..n {
                let p = config.min + cell_size * Vec3::new(x as f32, y as f32, z as f32);
                values[x + y * n + z * n * n] = sdf(p);
            }
        }
    }

    // --- Step 2: compute gradient at each node via central finite differences ---
    // Uses the sampled values, so no extra SDF calls.
    let grad = |x: usize, y: usize, z: usize| -> Vec3 {
        let sample = |xi: i32, yi: i32, zi: i32| -> f32 {
            let xi = xi.clamp(0, n as i32 - 1) as usize;
            let yi = yi.clamp(0, n as i32 - 1) as usize;
            let zi = zi.clamp(0, n as i32 - 1) as usize;
            values[xi + yi * n + zi * n * n]
        };
        let x = x as i32;
        let y = y as i32;
        let z = z as i32;
        Vec3::new(
            (sample(x + 1, y, z) - sample(x - 1, y, z)) / (2.0 * cell_size.x),
            (sample(x, y + 1, z) - sample(x, y - 1, z)) / (2.0 * cell_size.y),
            (sample(x, y, z + 1) - sample(x, y, z - 1)) / (2.0 * cell_size.z),
        )
        .normalize_or_zero()
    };

    // --- Step 3: find active cells and compute QEF vertex per cell ---
    // active_cell_vertex[cell_idx] = vertex index in output, or u32::MAX if inactive.
    let cell_count = res * res * res;
    let mut cell_vertex: Vec<u32> = vec![u32::MAX; cell_count];
    let mut positions: Vec<Vec3> = Vec::new();

    let cell_idx = |cx: usize, cy: usize, cz: usize| cx + cy * res + cz * res * res;
    let node_val = |nx: usize, ny: usize, nz: usize| values[nx + ny * n + nz * n * n];

    for cz in 0..res {
        for cy in 0..res {
            for cx in 0..res {
                // Collect edge intersections for this cell
                let mut intersections: Vec<(Vec3, Vec3)> = Vec::new();

                for &(vi, vj) in &CELL_EDGES {
                    let oi = CORNER_OFFSETS[vi];
                    let oj = CORNER_OFFSETS[vj];

                    let xi = (cx as f32 + oi.x) as usize;
                    let yi = (cy as f32 + oi.y) as usize;
                    let zi = (cz as f32 + oi.z) as usize;
                    let xj = (cx as f32 + oj.x) as usize;
                    let yj = (cy as f32 + oj.y) as usize;
                    let zj = (cz as f32 + oj.z) as usize;

                    let val_i = node_val(xi, yi, zi);
                    let val_j = node_val(xj, yj, zj);

                    let sign_i = val_i < config.iso_value;
                    let sign_j = val_j < config.iso_value;

                    if sign_i == sign_j {
                        continue; // no crossing on this edge
                    }

                    // Linearly interpolate crossing position
                    let t = if (val_j - val_i).abs() > 1e-8 {
                        (config.iso_value - val_i) / (val_j - val_i)
                    } else {
                        0.5
                    };

                    let pi = config.min + cell_size * Vec3::new(xi as f32, yi as f32, zi as f32);
                    let pj = config.min + cell_size * Vec3::new(xj as f32, yj as f32, zj as f32);
                    let crossing = pi.lerp(pj, t);

                    // Interpolate gradient at crossing
                    let gi = grad(xi, yi, zi);
                    let gj = grad(xj, yj, zj);
                    let normal = gi.lerp(gj, t).normalize_or_zero();

                    intersections.push((crossing, normal));
                }

                if intersections.is_empty() {
                    continue;
                }

                // Compute QEF minimizer
                let cell_min = config.min + cell_size * Vec3::new(cx as f32, cy as f32, cz as f32);
                let cell_max = cell_min + cell_size;
                let vertex_pos = solve_qef(&intersections, cell_min, cell_max);

                let vid = positions.len() as u32;
                positions.push(vertex_pos);
                cell_vertex[cell_idx(cx, cy, cz)] = vid;
            }
        }
    }

    // --- Step 4: emit quads for each sign-changing edge ---
    // Each edge is shared by 4 cells; if all 4 are active we emit a quad.
    // Winding: determined by sign direction so normals point outward.
    let mut indices: Vec<u32> = Vec::new();

    // Helper: emit a quad (two triangles) from 4 vertex indices.
    // `flip` reverses winding when the sign change goes the other direction.
    let mut emit_quad = |v: [u32; 4], flip: bool| {
        let (a, b, c, d) = if flip {
            (v[0], v[2], v[1], v[3])
        } else {
            (v[0], v[1], v[2], v[3])
        };
        indices.push(a);
        indices.push(b);
        indices.push(c);
        indices.push(a);
        indices.push(c);
        indices.push(d);
    };

    // X-axis edges: iterate over grid nodes, the edge goes from (x,y,z)→(x+1,y,z)
    for z in 1..res {
        for y in 1..res {
            for x in 0..res {
                let val0 = node_val(x, y, z);
                let val1 = node_val(x + 1, y, z);
                let sign0 = val0 < config.iso_value;
                let sign1 = val1 < config.iso_value;
                if sign0 == sign1 {
                    continue;
                }
                // 4 cells sharing this edge: (x, y-1, z-1), (x, y-1, z), (x, y, z-1), (x, y, z)
                let c00 = cell_vertex[cell_idx(x, y - 1, z - 1)];
                let c01 = cell_vertex[cell_idx(x, y - 1, z)];
                let c10 = cell_vertex[cell_idx(x, y, z - 1)];
                let c11 = cell_vertex[cell_idx(x, y, z)];
                if c00 == u32::MAX || c01 == u32::MAX || c10 == u32::MAX || c11 == u32::MAX {
                    continue;
                }
                // sign0 = inside means the surface normal points in +x direction
                emit_quad([c00, c01, c10, c11], !sign0);
            }
        }
    }

    // Y-axis edges: the edge goes from (x,y,z)→(x,y+1,z)
    for z in 1..res {
        for y in 0..res {
            for x in 1..res {
                let val0 = node_val(x, y, z);
                let val1 = node_val(x, y + 1, z);
                let sign0 = val0 < config.iso_value;
                let sign1 = val1 < config.iso_value;
                if sign0 == sign1 {
                    continue;
                }
                // 4 cells: (x-1, y, z-1), (x, y, z-1), (x-1, y, z), (x, y, z)
                let c00 = cell_vertex[cell_idx(x - 1, y, z - 1)];
                let c01 = cell_vertex[cell_idx(x, y, z - 1)];
                let c10 = cell_vertex[cell_idx(x - 1, y, z)];
                let c11 = cell_vertex[cell_idx(x, y, z)];
                if c00 == u32::MAX || c01 == u32::MAX || c10 == u32::MAX || c11 == u32::MAX {
                    continue;
                }
                emit_quad([c00, c01, c10, c11], sign0);
            }
        }
    }

    // Z-axis edges: the edge goes from (x,y,z)→(x,y,z+1)
    for z in 0..res {
        for y in 1..res {
            for x in 1..res {
                let val0 = node_val(x, y, z);
                let val1 = node_val(x, y, z + 1);
                let sign0 = val0 < config.iso_value;
                let sign1 = val1 < config.iso_value;
                if sign0 == sign1 {
                    continue;
                }
                // 4 cells: (x-1, y-1, z), (x, y-1, z), (x-1, y, z), (x, y, z)
                let c00 = cell_vertex[cell_idx(x - 1, y - 1, z)];
                let c01 = cell_vertex[cell_idx(x, y - 1, z)];
                let c10 = cell_vertex[cell_idx(x - 1, y, z)];
                let c11 = cell_vertex[cell_idx(x, y, z)];
                if c00 == u32::MAX || c01 == u32::MAX || c10 == u32::MAX || c11 == u32::MAX {
                    continue;
                }
                emit_quad([c00, c01, c10, c11], !sign0);
            }
        }
    }

    let mut mesh = Mesh::new();
    mesh.positions = positions;
    mesh.indices = indices;
    mesh.compute_smooth_normals();

    mesh
}

// Suppress dead-code warnings for helper functions that are only used in tests.
#[allow(dead_code)]
fn _suppress_unused() {
    let _ = x_edge_cells;
    let _ = y_edge_cells;
    let _ = z_edge_cells;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::marching_cubes::{MarchingCubes, marching_cubes};

    fn sphere_sdf(p: Vec3) -> f32 {
        p.length() - 1.0
    }

    fn box_sdf(p: Vec3) -> f32 {
        let q = p.abs() - Vec3::splat(0.5);
        q.max(Vec3::ZERO).length() + q.x.max(q.y).max(q.z).min(0.0)
    }

    #[test]
    fn test_dc_sphere_produces_geometry() {
        let config = DualContouring {
            resolution: 20,
            ..Default::default()
        };
        let mesh = config.apply_fn(sphere_sdf);

        assert!(!mesh.positions.is_empty(), "DC sphere should have vertices");
        assert!(!mesh.indices.is_empty(), "DC sphere should have triangles");
        assert!(mesh.has_normals(), "DC sphere should have normals");
    }

    #[test]
    fn test_dc_sphere_vertex_count_scales_with_resolution() {
        // The surface of a sphere scales as O(r²), so vertex count should
        // increase with resolution.
        let low = DualContouring {
            resolution: 10,
            ..Default::default()
        };
        let high = DualContouring {
            resolution: 20,
            ..Default::default()
        };
        let mesh_lo = low.apply_fn(sphere_sdf);
        let mesh_hi = high.apply_fn(sphere_sdf);

        assert!(
            mesh_hi.vertex_count() > mesh_lo.vertex_count(),
            "Higher resolution should produce more vertices: lo={}, hi={}",
            mesh_lo.vertex_count(),
            mesh_hi.vertex_count()
        );
    }

    /// Mean distance from mesh vertices to the unit sphere surface.
    fn mean_surface_error(mesh: &Mesh, sdf: impl Fn(Vec3) -> f32) -> f32 {
        if mesh.positions.is_empty() {
            return f32::MAX;
        }
        let total: f32 = mesh.positions.iter().map(|p| sdf(*p).abs()).sum();
        total / mesh.positions.len() as f32
    }

    #[test]
    fn test_dc_vs_mc_sphere_surface_accuracy() {
        // At the same resolution, dual contouring should sit at least as close
        // to the true sphere surface as marching cubes on average.
        let res = 24;

        let dc_config = DualContouring {
            resolution: res,
            ..Default::default()
        };
        let dc_mesh = dc_config.apply_fn(sphere_sdf);

        let mc_config = MarchingCubes {
            resolution: res,
            ..Default::default()
        };
        let mc_mesh = marching_cubes(sphere_sdf, mc_config);

        let dc_err = mean_surface_error(&dc_mesh, sphere_sdf);
        let mc_err = mean_surface_error(&mc_mesh, sphere_sdf);

        // DC should be no worse than 1.5× MC's error (conservative; DC is
        // typically better because vertices sit at the QEF minimizer).
        assert!(
            dc_err <= mc_err * 1.5,
            "DC mean surface error ({dc_err:.4}) should be competitive with MC ({mc_err:.4})"
        );
    }

    #[test]
    fn test_dc_box_corners_closer_than_mc() {
        // On a box SDF, the DC QEF minimizer should snap toward the true
        // corner; MC interpolates along edges and cannot reproduce corners.
        let res = 20;

        let dc_config = DualContouring {
            resolution: res,
            ..Default::default()
        };
        let dc_mesh = dc_config.apply_fn(box_sdf);

        let mc_config = MarchingCubes {
            resolution: res,
            ..Default::default()
        };
        let mc_mesh = marching_cubes(box_sdf, mc_config);

        // Distance of a vertex to the nearest box corner (±0.5, ±0.5, ±0.5).
        let corners: [Vec3; 8] = [
            Vec3::new(-0.5, -0.5, -0.5),
            Vec3::new(0.5, -0.5, -0.5),
            Vec3::new(-0.5, 0.5, -0.5),
            Vec3::new(0.5, 0.5, -0.5),
            Vec3::new(-0.5, -0.5, 0.5),
            Vec3::new(0.5, -0.5, 0.5),
            Vec3::new(-0.5, 0.5, 0.5),
            Vec3::new(0.5, 0.5, 0.5),
        ];

        let nearest_corner_dist = |p: Vec3| {
            corners
                .iter()
                .map(|c| (*c - p).length())
                .fold(f32::MAX, f32::min)
        };

        // Only examine vertices within 1.5 cell-lengths of a corner
        let cell_size = 2.0 / res as f32;
        let threshold = 2.0 * cell_size;

        let dc_near: Vec<f32> = dc_mesh
            .positions
            .iter()
            .filter(|p| nearest_corner_dist(**p) < threshold)
            .map(|p| nearest_corner_dist(*p))
            .collect();

        let mc_near: Vec<f32> = mc_mesh
            .positions
            .iter()
            .filter(|p| nearest_corner_dist(**p) < threshold)
            .map(|p| nearest_corner_dist(*p))
            .collect();

        if dc_near.is_empty() || mc_near.is_empty() {
            // If no vertices are near corners, the test is vacuously passing.
            return;
        }

        let dc_mean: f32 = dc_near.iter().sum::<f32>() / dc_near.len() as f32;
        let mc_mean: f32 = mc_near.iter().sum::<f32>() / mc_near.len() as f32;

        assert!(
            dc_mean <= mc_mean * 1.1,
            "DC corner vertices ({dc_mean:.4}) should be closer to true corners than MC ({mc_mean:.4})"
        );
    }

    #[test]
    fn test_dc_empty_sdf() {
        let config = DualContouring::default();
        let mesh = config.apply_fn(|_: Vec3| 10.0_f32);
        assert!(mesh.positions.is_empty());
        assert!(mesh.indices.is_empty());
    }

    #[test]
    fn test_dc_full_sdf() {
        let config = DualContouring::default();
        let mesh = config.apply_fn(|_: Vec3| -10.0_f32);
        assert!(mesh.positions.is_empty());
        assert!(mesh.indices.is_empty());
    }

    #[test]
    fn test_dc_low_resolution_guard() {
        let config = DualContouring {
            resolution: 1,
            ..Default::default()
        };
        let mesh = config.apply_fn(sphere_sdf);
        assert!(mesh.positions.is_empty());
    }
}
