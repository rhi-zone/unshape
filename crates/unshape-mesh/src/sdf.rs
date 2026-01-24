//! Signed distance field generation from meshes.
//!
//! Converts triangle meshes to signed distance fields for use in
//! Boolean operations, collision detection, and procedural generation.
//!
//! Operations are serializable structs with `apply` methods.
//! See `docs/design/ops-as-values.md`.

use crate::Mesh;
use glam::Vec3;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Generates a signed distance field from a mesh.
///
/// Converts a triangle mesh to a 3D grid of signed distance values.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dynop", derive(unshape_op::Op))]
#[cfg_attr(feature = "dynop", op(input = Mesh, output = SdfGrid))]
pub struct GenerateSdf {
    /// Grid resolution in each dimension.
    pub resolution: (usize, usize, usize),
    /// Bounding box padding (multiplier, 1.1 = 10% padding).
    pub padding: f32,
    /// Whether to compute exact distances (slower) or approximate.
    pub exact: bool,
}

impl Default for GenerateSdf {
    fn default() -> Self {
        Self {
            resolution: (32, 32, 32),
            padding: 1.1,
            exact: true,
        }
    }
}

impl GenerateSdf {
    /// Applies this operation to generate an SDF from a mesh.
    pub fn apply(&self, mesh: &Mesh) -> SdfGrid {
        mesh_to_sdf(mesh, self)
    }
}

/// Backwards-compatible type alias.
pub type SdfConfig = GenerateSdf;

/// A 3D signed distance field stored as a regular grid.
#[derive(Debug, Clone)]
pub struct SdfGrid {
    /// Distance values at grid points.
    pub data: Vec<f32>,
    /// Grid dimensions (x, y, z).
    pub dimensions: (usize, usize, usize),
    /// World-space bounds minimum.
    pub bounds_min: Vec3,
    /// World-space bounds maximum.
    pub bounds_max: Vec3,
}

impl SdfGrid {
    /// Creates a new SDF grid.
    pub fn new(dimensions: (usize, usize, usize), bounds_min: Vec3, bounds_max: Vec3) -> Self {
        let size = dimensions.0 * dimensions.1 * dimensions.2;
        Self {
            data: vec![f32::MAX; size],
            dimensions,
            bounds_min,
            bounds_max,
        }
    }

    /// Gets the value at grid coordinates.
    pub fn get(&self, x: usize, y: usize, z: usize) -> f32 {
        if x < self.dimensions.0 && y < self.dimensions.1 && z < self.dimensions.2 {
            self.data[z * self.dimensions.0 * self.dimensions.1 + y * self.dimensions.0 + x]
        } else {
            f32::MAX
        }
    }

    /// Sets the value at grid coordinates.
    pub fn set(&mut self, x: usize, y: usize, z: usize, value: f32) {
        if x < self.dimensions.0 && y < self.dimensions.1 && z < self.dimensions.2 {
            self.data[z * self.dimensions.0 * self.dimensions.1 + y * self.dimensions.0 + x] =
                value;
        }
    }

    /// Returns the cell size in world units.
    pub fn cell_size(&self) -> Vec3 {
        let extent = self.bounds_max - self.bounds_min;
        Vec3::new(
            extent.x / (self.dimensions.0 - 1).max(1) as f32,
            extent.y / (self.dimensions.1 - 1).max(1) as f32,
            extent.z / (self.dimensions.2 - 1).max(1) as f32,
        )
    }

    /// Converts grid coordinates to world position.
    pub fn grid_to_world(&self, x: usize, y: usize, z: usize) -> Vec3 {
        let cell = self.cell_size();
        self.bounds_min + Vec3::new(x as f32 * cell.x, y as f32 * cell.y, z as f32 * cell.z)
    }

    /// Converts world position to grid coordinates.
    pub fn world_to_grid(&self, pos: Vec3) -> (usize, usize, usize) {
        let cell = self.cell_size();
        let local = pos - self.bounds_min;
        (
            (local.x / cell.x).round() as usize,
            (local.y / cell.y).round() as usize,
            (local.z / cell.z).round() as usize,
        )
    }

    /// Samples the SDF at a world position using trilinear interpolation.
    pub fn sample(&self, pos: Vec3) -> f32 {
        let cell = self.cell_size();
        let local = pos - self.bounds_min;

        let fx = local.x / cell.x;
        let fy = local.y / cell.y;
        let fz = local.z / cell.z;

        let x0 = fx.floor() as i32;
        let y0 = fy.floor() as i32;
        let z0 = fz.floor() as i32;

        let tx = fx - x0 as f32;
        let ty = fy - y0 as f32;
        let tz = fz - z0 as f32;

        // Clamp to valid range
        let get_clamped = |x: i32, y: i32, z: i32| -> f32 {
            let x = x.clamp(0, self.dimensions.0 as i32 - 1) as usize;
            let y = y.clamp(0, self.dimensions.1 as i32 - 1) as usize;
            let z = z.clamp(0, self.dimensions.2 as i32 - 1) as usize;
            self.get(x, y, z)
        };

        // Trilinear interpolation
        let c000 = get_clamped(x0, y0, z0);
        let c100 = get_clamped(x0 + 1, y0, z0);
        let c010 = get_clamped(x0, y0 + 1, z0);
        let c110 = get_clamped(x0 + 1, y0 + 1, z0);
        let c001 = get_clamped(x0, y0, z0 + 1);
        let c101 = get_clamped(x0 + 1, y0, z0 + 1);
        let c011 = get_clamped(x0, y0 + 1, z0 + 1);
        let c111 = get_clamped(x0 + 1, y0 + 1, z0 + 1);

        let c00 = c000 * (1.0 - tx) + c100 * tx;
        let c10 = c010 * (1.0 - tx) + c110 * tx;
        let c01 = c001 * (1.0 - tx) + c101 * tx;
        let c11 = c011 * (1.0 - tx) + c111 * tx;

        let c0 = c00 * (1.0 - ty) + c10 * ty;
        let c1 = c01 * (1.0 - ty) + c11 * ty;

        c0 * (1.0 - tz) + c1 * tz
    }

    /// Computes the gradient at a world position.
    pub fn gradient(&self, pos: Vec3) -> Vec3 {
        let h = self.cell_size().x * 0.5;
        let dx = self.sample(pos + Vec3::X * h) - self.sample(pos - Vec3::X * h);
        let dy = self.sample(pos + Vec3::Y * h) - self.sample(pos - Vec3::Y * h);
        let dz = self.sample(pos + Vec3::Z * h) - self.sample(pos - Vec3::Z * h);
        Vec3::new(dx, dy, dz).normalize_or_zero()
    }
}

/// Generates an SDF from a mesh.
pub fn mesh_to_sdf(mesh: &Mesh, config: &GenerateSdf) -> SdfGrid {
    // Compute bounds
    let (bounds_min, bounds_max) = compute_bounds(mesh, config.padding);

    // Create grid
    let mut grid = SdfGrid::new(config.resolution, bounds_min, bounds_max);

    // Collect triangles
    let triangles = collect_triangles(mesh);

    // For each grid point, compute distance
    let (nx, ny, nz) = config.resolution;
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                let pos = grid.grid_to_world(x, y, z);
                let (dist, sign) = compute_signed_distance(&pos, &triangles, config.exact);
                grid.set(x, y, z, dist * sign);
            }
        }
    }

    grid
}

/// A triangle for distance computation.
struct Triangle {
    v0: Vec3,
    v1: Vec3,
    v2: Vec3,
    normal: Vec3,
}

/// Computes mesh bounds with padding.
fn compute_bounds(mesh: &Mesh, padding: f32) -> (Vec3, Vec3) {
    if mesh.positions.is_empty() {
        return (Vec3::ZERO, Vec3::ONE);
    }

    let mut min = Vec3::splat(f32::MAX);
    let mut max = Vec3::splat(f32::MIN);

    for pos in &mesh.positions {
        min = min.min(*pos);
        max = max.max(*pos);
    }

    // Apply padding
    let center = (min + max) * 0.5;
    let extent = (max - min) * padding * 0.5;

    (center - extent, center + extent)
}

/// Collects triangles from a mesh.
fn collect_triangles(mesh: &Mesh) -> Vec<Triangle> {
    let mut triangles = Vec::with_capacity(mesh.triangle_count());

    for i in (0..mesh.indices.len()).step_by(3) {
        let i0 = mesh.indices[i] as usize;
        let i1 = mesh.indices[i + 1] as usize;
        let i2 = mesh.indices[i + 2] as usize;

        let v0 = mesh.positions[i0];
        let v1 = mesh.positions[i1];
        let v2 = mesh.positions[i2];

        let edge1 = v1 - v0;
        let edge2 = v2 - v0;
        let normal = edge1.cross(edge2).normalize_or_zero();

        triangles.push(Triangle { v0, v1, v2, normal });
    }

    triangles
}

/// Computes signed distance from a point to a set of triangles.
fn compute_signed_distance(point: &Vec3, triangles: &[Triangle], exact: bool) -> (f32, f32) {
    let mut min_dist = f32::MAX;
    let mut closest_tri = 0;

    for (i, tri) in triangles.iter().enumerate() {
        let dist = if exact {
            point_triangle_distance(*point, tri.v0, tri.v1, tri.v2)
        } else {
            // Approximate: use distance to triangle plane
            ((*point - tri.v0).dot(tri.normal)).abs()
        };

        if dist < min_dist {
            min_dist = dist;
            closest_tri = i;
        }
    }

    // Compute sign using angle-weighted pseudo-normal
    let sign = compute_sign(*point, triangles, closest_tri);

    (min_dist, sign)
}

/// Computes the sign (inside/outside) using pseudo-normal.
fn compute_sign(point: Vec3, triangles: &[Triangle], closest_idx: usize) -> f32 {
    // Simple approach: use normal of closest triangle
    let tri = &triangles[closest_idx];
    let to_point = point - tri.v0;

    if to_point.dot(tri.normal) < 0.0 {
        -1.0 // Inside
    } else {
        1.0 // Outside
    }
}

/// Computes the distance from a point to a triangle.
fn point_triangle_distance(p: Vec3, v0: Vec3, v1: Vec3, v2: Vec3) -> f32 {
    // Based on "Real-Time Collision Detection" by Christer Ericson
    let edge0 = v1 - v0;
    let edge1 = v2 - v0;
    let v0_to_p = p - v0;

    let a = edge0.dot(edge0);
    let b = edge0.dot(edge1);
    let c = edge1.dot(edge1);
    let d = edge0.dot(v0_to_p);
    let e = edge1.dot(v0_to_p);

    let det = a * c - b * b;
    let mut s = b * e - c * d;
    let mut t = b * d - a * e;

    if s + t <= det {
        if s < 0.0 {
            if t < 0.0 {
                // Region 4
                if d < 0.0 {
                    t = 0.0;
                    s = (-d).min(a).max(0.0);
                } else {
                    s = 0.0;
                    t = (-e).min(c).max(0.0);
                }
            } else {
                // Region 3
                s = 0.0;
                t = (-e).min(c).max(0.0);
            }
        } else if t < 0.0 {
            // Region 5
            t = 0.0;
            s = (-d).min(a).max(0.0);
        } else {
            // Region 0
            let inv_det = 1.0 / det;
            s *= inv_det;
            t *= inv_det;
        }
    } else {
        if s < 0.0 {
            // Region 2
            let tmp0 = b + d;
            let tmp1 = c + e;
            if tmp1 > tmp0 {
                let numer = tmp1 - tmp0;
                let denom = a - 2.0 * b + c;
                s = (numer / denom).min(1.0).max(0.0);
                t = 1.0 - s;
            } else {
                s = 0.0;
                t = (-e).min(c).max(0.0);
            }
        } else if t < 0.0 {
            // Region 6
            let tmp0 = b + e;
            let tmp1 = a + d;
            if tmp1 > tmp0 {
                let numer = tmp1 - tmp0;
                let denom = a - 2.0 * b + c;
                t = (numer / denom).min(1.0).max(0.0);
                s = 1.0 - t;
            } else {
                t = 0.0;
                s = (-d).min(a).max(0.0);
            }
        } else {
            // Region 1
            let numer = (c + e) - (b + d);
            if numer <= 0.0 {
                s = 0.0;
            } else {
                let denom = a - 2.0 * b + c;
                s = (numer / denom).min(1.0);
            }
            t = 1.0 - s;
        }
    }

    // Ensure s and t are clamped
    s = s.max(0.0);
    t = t.max(0.0);

    let closest = v0 + edge0 * s + edge1 * t;
    (p - closest).length()
}

/// Computes an SDF from a mesh using fast marching (approximate).
pub fn mesh_to_sdf_fast(mesh: &Mesh, config: &GenerateSdf) -> SdfGrid {
    let mut config = config.clone();
    config.exact = false;
    mesh_to_sdf(mesh, &config)
}

/// Ray-marches the SDF and returns the hit point and distance.
pub fn raymarch(
    sdf: &SdfGrid,
    origin: Vec3,
    direction: Vec3,
    max_steps: u32,
    epsilon: f32,
) -> Option<(Vec3, f32)> {
    let mut t = 0.0;
    let dir = direction.normalize();

    for _ in 0..max_steps {
        let pos = origin + dir * t;

        // Check bounds
        if pos.x < sdf.bounds_min.x
            || pos.x > sdf.bounds_max.x
            || pos.y < sdf.bounds_min.y
            || pos.y > sdf.bounds_max.y
            || pos.z < sdf.bounds_min.z
            || pos.z > sdf.bounds_max.z
        {
            return None;
        }

        let d = sdf.sample(pos);

        if d.abs() < epsilon {
            return Some((pos, t));
        }

        t += d.abs().max(epsilon);
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Cuboid;

    #[test]
    fn test_sdf_grid_creation() {
        let grid = SdfGrid::new((8, 8, 8), Vec3::ZERO, Vec3::ONE);

        assert_eq!(grid.dimensions, (8, 8, 8));
        assert_eq!(grid.data.len(), 512);
    }

    #[test]
    fn test_sdf_grid_get_set() {
        let mut grid = SdfGrid::new((4, 4, 4), Vec3::ZERO, Vec3::ONE);

        grid.set(1, 2, 3, 0.5);
        assert_eq!(grid.get(1, 2, 3), 0.5);
    }

    #[test]
    fn test_sdf_grid_world_conversion() {
        let grid = SdfGrid::new((11, 11, 11), Vec3::ZERO, Vec3::splat(10.0));

        // Grid (5, 5, 5) should map to world (5, 5, 5)
        let world = grid.grid_to_world(5, 5, 5);
        assert!((world - Vec3::splat(5.0)).length() < 0.01);

        // World (5, 5, 5) should map back
        let (x, y, z) = grid.world_to_grid(Vec3::splat(5.0));
        assert_eq!((x, y, z), (5, 5, 5));
    }

    #[test]
    fn test_sdf_config_default() {
        let config = SdfConfig::default();
        assert_eq!(config.resolution, (32, 32, 32));
        assert!(config.exact);
    }

    #[test]
    fn test_mesh_to_sdf_box() {
        let mesh = Cuboid::default().apply();
        let config = SdfConfig {
            resolution: (16, 16, 16),
            padding: 1.5,
            exact: true,
        };

        let sdf = mesh_to_sdf(&mesh, &config);

        // Center should be inside (negative)
        let center_dist = sdf.sample(Vec3::ZERO);
        assert!(center_dist < 0.0);

        // Far outside should be positive
        let outside_dist = sdf.sample(Vec3::splat(2.0));
        assert!(outside_dist > 0.0);
    }

    #[test]
    fn test_mesh_to_sdf_fast() {
        let mesh = Cuboid::default().apply();
        let config = SdfConfig {
            resolution: (8, 8, 8),
            ..Default::default()
        };

        let sdf = mesh_to_sdf_fast(&mesh, &config);

        // Should still produce valid distances
        assert!(sdf.data.iter().any(|&d| d < 0.0));
        assert!(sdf.data.iter().any(|&d| d > 0.0));
    }

    #[test]
    fn test_sdf_gradient() {
        let mesh = Cuboid::default().apply();
        let config = SdfConfig {
            resolution: (16, 16, 16),
            padding: 1.5,
            exact: true,
        };

        let sdf = mesh_to_sdf(&mesh, &config);

        // Gradient at surface should point outward
        let surface_point = Vec3::new(0.5, 0.0, 0.0);
        let grad = sdf.gradient(surface_point);

        // Should roughly point in +X direction
        assert!(grad.x > 0.5);
    }

    #[test]
    fn test_sdf_trilinear() {
        let mut grid = SdfGrid::new((3, 3, 3), Vec3::ZERO, Vec3::splat(2.0));

        // Set all to 1.0 except center to 0.0
        for z in 0..3 {
            for y in 0..3 {
                for x in 0..3 {
                    grid.set(x, y, z, 1.0);
                }
            }
        }
        grid.set(1, 1, 1, 0.0);

        // Interpolation at center should be 0
        let center = grid.sample(Vec3::ONE);
        assert!(center.abs() < 0.01);
    }

    #[test]
    fn test_raymarch() {
        // Create a simple SDF grid with known values
        let mut grid = SdfGrid::new((11, 11, 11), Vec3::splat(-1.0), Vec3::splat(1.0));

        // Fill with spherical SDF (distance from origin - radius)
        let radius = 0.5;
        for z in 0..11 {
            for y in 0..11 {
                for x in 0..11 {
                    let pos = grid.grid_to_world(x, y, z);
                    let dist = pos.length() - radius;
                    grid.set(x, y, z, dist);
                }
            }
        }

        // Ray from outside toward center
        let origin = Vec3::new(0.9, 0.0, 0.0);
        let direction = Vec3::new(-1.0, 0.0, 0.0);

        let result = raymarch(&grid, origin, direction, 100, 0.01);
        assert!(result.is_some());

        if let Some((hit, _t)) = result {
            // Should hit near x = 0.5 (sphere surface)
            assert!((hit.x - radius).abs() < 0.1);
        }
    }

    #[test]
    fn test_point_triangle_distance() {
        let v0 = Vec3::ZERO;
        let v1 = Vec3::X;
        let v2 = Vec3::Z;

        // Point directly above inside the triangle (triangle in XZ plane)
        let p = Vec3::new(0.2, 1.0, 0.2);
        let dist = point_triangle_distance(p, v0, v1, v2);
        assert!((dist - 1.0).abs() < 0.1);

        // Point at triangle vertex
        let p = Vec3::ZERO;
        let dist = point_triangle_distance(p, v0, v1, v2);
        assert!(dist < 0.001);
    }
}
