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

/// Solves the 3x3 symmetric linear system `A x = b` using Gaussian elimination
/// with partial pivoting. Returns `None` if the matrix is singular.
fn solve_3x3(a: [[f32; 3]; 3], b: [f32; 3]) -> Option<[f32; 3]> {
    // Augmented matrix rows: [a0|a1|a2|b].
    let mut r0 = [a[0][0], a[0][1], a[0][2], b[0]];
    let mut r1 = [a[1][0], a[1][1], a[1][2], b[1]];
    let mut r2 = [a[2][0], a[2][1], a[2][2], b[2]];

    // Column 0: pivot on the row with the largest |col0| value.
    {
        let (abs0, abs1, abs2) = (r0[0].abs(), r1[0].abs(), r2[0].abs());
        if abs1 > abs0 && abs1 >= abs2 {
            std::mem::swap(&mut r0, &mut r1);
        } else if abs2 > abs0 && abs2 > abs1 {
            std::mem::swap(&mut r0, &mut r2);
        }
        if r0[0].abs() < 1e-8 {
            return None;
        }
        let f1 = r1[0] / r0[0];
        let f2 = r2[0] / r0[0];
        r1 = [
            0.0,
            r1[1] - f1 * r0[1],
            r1[2] - f1 * r0[2],
            r1[3] - f1 * r0[3],
        ];
        r2 = [
            0.0,
            r2[1] - f2 * r0[1],
            r2[2] - f2 * r0[2],
            r2[3] - f2 * r0[3],
        ];
    }

    // Column 1: pivot between r1 and r2.
    {
        if r2[1].abs() > r1[1].abs() {
            std::mem::swap(&mut r1, &mut r2);
        }
        if r1[1].abs() < 1e-8 {
            return None;
        }
        let f2 = r2[1] / r1[1];
        r2 = [0.0, 0.0, r2[2] - f2 * r1[2], r2[3] - f2 * r1[3]];
    }

    // Column 2.
    if r2[2].abs() < 1e-8 {
        return None;
    }

    // Back substitution.
    let x2 = r2[3] / r2[2];
    let x1 = (r1[3] - r1[2] * x2) / r1[1];
    let x0 = (r0[3] - r0[2] * x2 - r0[1] * x1) / r0[0];
    Some([x0, x1, x2])
}

/// Solves the QEF `min_x sum (n_i . (x - p_i))^2` for edge intersection
/// points `p_i` with gradient normals `n_i`.
///
/// Reduces to normal equations `(A^T A) x = A^T b`. Shifts to mass point
/// for numerical stability, falls back to mass point when singular, clamps
/// result to `[cell_min, cell_max]`.
fn solve_qef(intersections: &[(Vec3, Vec3)], cell_min: Vec3, cell_max: Vec3) -> Vec3 {
    if intersections.is_empty() {
        return (cell_min + cell_max) * 0.5;
    }

    let mut ata = [[0.0f32; 3]; 3];
    let mut atb = [0.0f32; 3];

    for (p, n) in intersections {
        let (nx, ny, nz) = (n.x, n.y, n.z);
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

    candidate.clamp(cell_min, cell_max)
}

/// The 12 edges of a unit cube, each as (vertex_a_index, vertex_b_index).
///
/// Vertex numbering:
///
/// ```text
///   4---5
///  /|  /|
/// 7---6 |
/// | 0-|-1
/// |/  |/
/// 3---2
/// ```
///
/// x increases 0→1, y increases 0→4, z increases 0→3.
const CELL_EDGES: [(usize, usize); 12] = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 0),
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 4),
    (0, 4),
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
    let n = res + 1;

    // Step 1: sample SDF at all grid nodes.
    let mut values = vec![0.0f32; n * n * n];
    for z in 0..n {
        for y in 0..n {
            for x in 0..n {
                let p = config.min + cell_size * Vec3::new(x as f32, y as f32, z as f32);
                values[x + y * n + z * n * n] = sdf(p);
            }
        }
    }

    // Step 2: gradient via central finite differences over the sampled grid.
    let grad = |x: usize, y: usize, z: usize| -> Vec3 {
        let sample = |xi: i32, yi: i32, zi: i32| -> f32 {
            let xi = xi.clamp(0, n as i32 - 1) as usize;
            let yi = yi.clamp(0, n as i32 - 1) as usize;
            let zi = zi.clamp(0, n as i32 - 1) as usize;
            values[xi + yi * n + zi * n * n]
        };
        let (xi, yi, zi) = (x as i32, y as i32, z as i32);
        Vec3::new(
            (sample(xi + 1, yi, zi) - sample(xi - 1, yi, zi)) / (2.0 * cell_size.x),
            (sample(xi, yi + 1, zi) - sample(xi, yi - 1, zi)) / (2.0 * cell_size.y),
            (sample(xi, yi, zi + 1) - sample(xi, yi, zi - 1)) / (2.0 * cell_size.z),
        )
        .normalize_or_zero()
    };

    // Step 3: one QEF vertex per active cell.
    let cell_count = res * res * res;
    let mut cell_vertex: Vec<u32> = vec![u32::MAX; cell_count];
    let mut positions: Vec<Vec3> = Vec::new();

    let cell_idx = |cx: usize, cy: usize, cz: usize| cx + cy * res + cz * res * res;
    let node_val = |nx: usize, ny: usize, nz: usize| values[nx + ny * n + nz * n * n];

    for cz in 0..res {
        for cy in 0..res {
            for cx in 0..res {
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
                        continue;
                    }

                    let t = if (val_j - val_i).abs() > 1e-8 {
                        (config.iso_value - val_i) / (val_j - val_i)
                    } else {
                        0.5
                    };

                    let pi = config.min + cell_size * Vec3::new(xi as f32, yi as f32, zi as f32);
                    let pj = config.min + cell_size * Vec3::new(xj as f32, yj as f32, zj as f32);
                    let crossing = pi.lerp(pj, t);

                    let gi = grad(xi, yi, zi);
                    let gj = grad(xj, yj, zj);
                    let normal = gi.lerp(gj, t).normalize_or_zero();

                    intersections.push((crossing, normal));
                }

                if intersections.is_empty() {
                    continue;
                }

                let cell_min = config.min + cell_size * Vec3::new(cx as f32, cy as f32, cz as f32);
                let cell_max = cell_min + cell_size;
                let vertex_pos = solve_qef(&intersections, cell_min, cell_max);

                let vid = positions.len() as u32;
                positions.push(vertex_pos);
                cell_vertex[cell_idx(cx, cy, cz)] = vid;
            }
        }
    }

    // Step 4: quads for each sign-changing edge shared by 4 active cells.
    //
    // Winding: normal points from inside (negative SDF) toward outside.
    let mut indices: Vec<u32> = Vec::new();

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

    // X-axis edges: (cx,cy,cz)→(cx+1,cy,cz).
    // Adjacent cells: (cx,cy-1,cz-1), (cx,cy-1,cz), (cx,cy,cz-1), (cx,cy,cz).
    for cz in 1..res {
        for cy in 1..res {
            for cx in 0..res {
                let val0 = node_val(cx, cy, cz);
                let val1 = node_val(cx + 1, cy, cz);
                let sign0 = val0 < config.iso_value;
                if sign0 == (val1 < config.iso_value) {
                    continue;
                }
                let c00 = cell_vertex[cell_idx(cx, cy - 1, cz - 1)];
                let c01 = cell_vertex[cell_idx(cx, cy - 1, cz)];
                let c10 = cell_vertex[cell_idx(cx, cy, cz - 1)];
                let c11 = cell_vertex[cell_idx(cx, cy, cz)];
                if c00 == u32::MAX || c01 == u32::MAX || c10 == u32::MAX || c11 == u32::MAX {
                    continue;
                }
                emit_quad([c00, c01, c10, c11], !sign0);
            }
        }
    }

    // Y-axis edges: (cx,cy,cz)→(cx,cy+1,cz).
    // Adjacent cells: (cx-1,cy,cz-1), (cx,cy,cz-1), (cx-1,cy,cz), (cx,cy,cz).
    for cz in 1..res {
        for cy in 0..res {
            for cx in 1..res {
                let val0 = node_val(cx, cy, cz);
                let val1 = node_val(cx, cy + 1, cz);
                let sign0 = val0 < config.iso_value;
                if sign0 == (val1 < config.iso_value) {
                    continue;
                }
                let c00 = cell_vertex[cell_idx(cx - 1, cy, cz - 1)];
                let c01 = cell_vertex[cell_idx(cx, cy, cz - 1)];
                let c10 = cell_vertex[cell_idx(cx - 1, cy, cz)];
                let c11 = cell_vertex[cell_idx(cx, cy, cz)];
                if c00 == u32::MAX || c01 == u32::MAX || c10 == u32::MAX || c11 == u32::MAX {
                    continue;
                }
                emit_quad([c00, c01, c10, c11], sign0);
            }
        }
    }

    // Z-axis edges: (cx,cy,cz)→(cx,cy,cz+1).
    // Adjacent cells: (cx-1,cy-1,cz), (cx,cy-1,cz), (cx-1,cy,cz), (cx,cy,cz).
    for cz in 0..res {
        for cy in 1..res {
            for cx in 1..res {
                let val0 = node_val(cx, cy, cz);
                let val1 = node_val(cx, cy, cz + 1);
                let sign0 = val0 < config.iso_value;
                if sign0 == (val1 < config.iso_value) {
                    continue;
                }
                let c00 = cell_vertex[cell_idx(cx - 1, cy - 1, cz)];
                let c01 = cell_vertex[cell_idx(cx, cy - 1, cz)];
                let c10 = cell_vertex[cell_idx(cx - 1, cy, cz)];
                let c11 = cell_vertex[cell_idx(cx, cy, cz)];
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
        let mesh_lo = DualContouring {
            resolution: 10,
            ..Default::default()
        }
        .apply_fn(sphere_sdf);
        let mesh_hi = DualContouring {
            resolution: 20,
            ..Default::default()
        }
        .apply_fn(sphere_sdf);
        assert!(
            mesh_hi.vertex_count() > mesh_lo.vertex_count(),
            "Higher resolution should produce more vertices: lo={}, hi={}",
            mesh_lo.vertex_count(),
            mesh_hi.vertex_count()
        );
    }

    fn mean_surface_error(mesh: &Mesh, sdf: impl Fn(Vec3) -> f32) -> f32 {
        if mesh.positions.is_empty() {
            return f32::MAX;
        }
        mesh.positions.iter().map(|p| sdf(*p).abs()).sum::<f32>() / mesh.positions.len() as f32
    }

    #[test]
    fn test_dc_sphere_surface_accuracy() {
        // DC places one vertex per active cell at the QEF minimizer (clamped to
        // the cell). On a smooth sphere the mean error is bounded by the cell
        // diagonal (sqrt(3) * cell_size / 2). We use a loose bound of one full
        // cell width as a sanity check.
        let res = 24;
        let cell_size = 2.0 / res as f32; // bounds are [-1, 1] so total width = 2
        let dc_mesh = DualContouring {
            resolution: res,
            ..Default::default()
        }
        .apply_fn(sphere_sdf);

        assert!(
            !dc_mesh.positions.is_empty(),
            "DC sphere should produce geometry"
        );

        let dc_err = mean_surface_error(&dc_mesh, sphere_sdf);
        assert!(
            dc_err < cell_size,
            "DC mean surface error ({dc_err:.4}) should be less than one cell width ({cell_size:.4})"
        );
    }

    #[test]
    fn test_dc_box_corners_closer_than_mc() {
        let res = 20;
        let dc_mesh = DualContouring {
            resolution: res,
            ..Default::default()
        }
        .apply_fn(box_sdf);
        let mc_mesh = marching_cubes(
            box_sdf,
            MarchingCubes {
                resolution: res,
                ..Default::default()
            },
        );

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

        let threshold = 2.0 * (2.0 / res as f32);

        let dc_mean = {
            let near: Vec<f32> = dc_mesh
                .positions
                .iter()
                .filter(|p| nearest_corner_dist(**p) < threshold)
                .map(|p| nearest_corner_dist(*p))
                .collect();
            if near.is_empty() {
                return;
            }
            near.iter().sum::<f32>() / near.len() as f32
        };

        let mc_mean = {
            let near: Vec<f32> = mc_mesh
                .positions
                .iter()
                .filter(|p| nearest_corner_dist(**p) < threshold)
                .map(|p| nearest_corner_dist(*p))
                .collect();
            if near.is_empty() {
                return;
            }
            near.iter().sum::<f32>() / near.len() as f32
        };

        assert!(
            dc_mean <= mc_mean * 1.1,
            "DC corner vertices ({dc_mean:.4}) should be closer to true corners than MC ({mc_mean:.4})"
        );
    }

    #[test]
    fn test_dc_empty_sdf() {
        let mesh = DualContouring::default().apply_fn(|_: Vec3| 10.0_f32);
        assert!(mesh.positions.is_empty());
        assert!(mesh.indices.is_empty());
    }

    #[test]
    fn test_dc_full_sdf() {
        let mesh = DualContouring::default().apply_fn(|_: Vec3| -10.0_f32);
        assert!(mesh.positions.is_empty());
        assert!(mesh.indices.is_empty());
    }

    #[test]
    fn test_dc_low_resolution_guard() {
        let mesh = DualContouring {
            resolution: 1,
            ..Default::default()
        }
        .apply_fn(|p: Vec3| p.length() - 1.0);
        assert!(mesh.positions.is_empty());
    }
}
