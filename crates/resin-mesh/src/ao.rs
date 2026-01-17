//! Ambient occlusion baking.
//!
//! Generates per-vertex or texture-space ambient occlusion data
//! by raycasting against mesh geometry.
//!
//! Operations are serializable structs with `apply` methods.
//! See `docs/design/ops-as-values.md`.

use crate::Mesh;
use glam::Vec3;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Bakes ambient occlusion to per-vertex values.
///
/// Uses raycasting against mesh geometry to compute per-vertex
/// ambient occlusion values.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dynop", derive(rhizome_resin_op::Op))]
#[cfg_attr(feature = "dynop", op(input = Mesh, output = Vec<f32>))]
pub struct BakeAo {
    /// Number of rays to cast per sample.
    pub ray_count: u32,
    /// Maximum ray distance.
    pub max_distance: f32,
    /// Bias offset along normal to avoid self-intersection.
    pub bias: f32,
    /// Whether to use cosine-weighted hemisphere sampling.
    pub cosine_weighted: bool,
    /// Power for AO falloff (higher = softer shadows).
    pub falloff_power: f32,
}

impl Default for BakeAo {
    fn default() -> Self {
        Self {
            ray_count: 64,
            max_distance: 10.0,
            bias: 0.001,
            cosine_weighted: true,
            falloff_power: 1.0,
        }
    }
}

impl BakeAo {
    /// Low quality preset (fast).
    pub fn low() -> Self {
        Self {
            ray_count: 16,
            ..Default::default()
        }
    }

    /// Medium quality preset.
    pub fn medium() -> Self {
        Self {
            ray_count: 64,
            ..Default::default()
        }
    }

    /// High quality preset (slow).
    pub fn high() -> Self {
        Self {
            ray_count: 256,
            ..Default::default()
        }
    }

    /// Applies this operation to bake AO values for a mesh.
    pub fn apply(&self, mesh: &Mesh) -> Vec<f32> {
        bake_ao_vertices(mesh, self)
    }
}

/// Backwards-compatible type alias.
pub type AoBakeConfig = BakeAo;

/// A simple BVH node for ray intersection.
#[derive(Debug, Clone)]
struct BvhNode {
    min: Vec3,
    max: Vec3,
    left: Option<Box<BvhNode>>,
    right: Option<Box<BvhNode>>,
    triangles: Vec<usize>, // Triangle indices (leaf only)
}

impl BvhNode {
    /// Checks if a ray intersects this AABB.
    fn intersects_ray(&self, origin: Vec3, dir_inv: Vec3, t_max: f32) -> bool {
        let t1 = (self.min - origin) * dir_inv;
        let t2 = (self.max - origin) * dir_inv;

        let t_near = t1.min(t2);
        let t_far = t1.max(t2);

        let t_enter = t_near.x.max(t_near.y).max(t_near.z);
        let t_exit = t_far.x.min(t_far.y).min(t_far.z);

        t_enter <= t_exit && t_exit >= 0.0 && t_enter <= t_max
    }
}

/// A simple acceleration structure for ray-mesh intersection.
#[derive(Debug)]
pub struct AoAccelerator {
    triangles: Vec<[Vec3; 3]>,
    bvh: BvhNode,
}

impl AoAccelerator {
    /// Builds an accelerator from a mesh.
    pub fn build(mesh: &Mesh) -> Self {
        let triangle_count = mesh.indices.len() / 3;
        let mut triangles = Vec::with_capacity(triangle_count);

        for i in 0..triangle_count {
            let i0 = mesh.indices[i * 3] as usize;
            let i1 = mesh.indices[i * 3 + 1] as usize;
            let i2 = mesh.indices[i * 3 + 2] as usize;

            triangles.push([mesh.positions[i0], mesh.positions[i1], mesh.positions[i2]]);
        }

        let indices: Vec<usize> = (0..triangle_count).collect();
        let bvh = Self::build_bvh(&triangles, indices, 0);

        Self { triangles, bvh }
    }

    /// Builds BVH recursively.
    fn build_bvh(triangles: &[[Vec3; 3]], indices: Vec<usize>, depth: usize) -> BvhNode {
        if indices.is_empty() {
            return BvhNode {
                min: Vec3::ZERO,
                max: Vec3::ZERO,
                left: None,
                right: None,
                triangles: Vec::new(),
            };
        }

        // Compute bounds
        let mut min = Vec3::splat(f32::MAX);
        let mut max = Vec3::splat(f32::MIN);

        for &idx in &indices {
            let tri = &triangles[idx];
            for v in tri {
                min = min.min(*v);
                max = max.max(*v);
            }
        }

        // Leaf node if small enough or too deep
        if indices.len() <= 4 || depth >= 20 {
            return BvhNode {
                min,
                max,
                left: None,
                right: None,
                triangles: indices,
            };
        }

        // Split along longest axis
        let extent = max - min;
        let axis = if extent.x >= extent.y && extent.x >= extent.z {
            0
        } else if extent.y >= extent.z {
            1
        } else {
            2
        };

        let mid = (min[axis] + max[axis]) * 0.5;

        let (left_indices, right_indices): (Vec<usize>, Vec<usize>) =
            indices.into_iter().partition(|&idx| {
                let tri = &triangles[idx];
                let centroid = (tri[0] + tri[1] + tri[2]) / 3.0;
                centroid[axis] < mid
            });

        // Avoid empty partitions
        if left_indices.is_empty() || right_indices.is_empty() {
            return BvhNode {
                min,
                max,
                left: None,
                right: None,
                triangles: left_indices.into_iter().chain(right_indices).collect(),
            };
        }

        BvhNode {
            min,
            max,
            left: Some(Box::new(Self::build_bvh(
                triangles,
                left_indices,
                depth + 1,
            ))),
            right: Some(Box::new(Self::build_bvh(
                triangles,
                right_indices,
                depth + 1,
            ))),
            triangles: Vec::new(),
        }
    }

    /// Tests if a ray hits any triangle within max_distance.
    pub fn occluded(&self, origin: Vec3, direction: Vec3, max_distance: f32) -> bool {
        let dir_inv = Vec3::new(
            if direction.x.abs() > 0.0001 {
                1.0 / direction.x
            } else {
                f32::MAX
            },
            if direction.y.abs() > 0.0001 {
                1.0 / direction.y
            } else {
                f32::MAX
            },
            if direction.z.abs() > 0.0001 {
                1.0 / direction.z
            } else {
                f32::MAX
            },
        );

        self.occluded_bvh(&self.bvh, origin, direction, dir_inv, max_distance)
    }

    fn occluded_bvh(
        &self,
        node: &BvhNode,
        origin: Vec3,
        direction: Vec3,
        dir_inv: Vec3,
        max_distance: f32,
    ) -> bool {
        if !node.intersects_ray(origin, dir_inv, max_distance) {
            return false;
        }

        // Check triangles in leaf
        for &idx in &node.triangles {
            if self.ray_triangle_intersect(origin, direction, idx, max_distance) {
                return true;
            }
        }

        // Check children
        if let Some(left) = &node.left {
            if self.occluded_bvh(left, origin, direction, dir_inv, max_distance) {
                return true;
            }
        }
        if let Some(right) = &node.right {
            if self.occluded_bvh(right, origin, direction, dir_inv, max_distance) {
                return true;
            }
        }

        false
    }

    /// Möller–Trumbore ray-triangle intersection.
    fn ray_triangle_intersect(
        &self,
        origin: Vec3,
        direction: Vec3,
        tri_idx: usize,
        max_distance: f32,
    ) -> bool {
        let tri = &self.triangles[tri_idx];
        let v0 = tri[0];
        let v1 = tri[1];
        let v2 = tri[2];

        let edge1 = v1 - v0;
        let edge2 = v2 - v0;

        let h = direction.cross(edge2);
        let a = edge1.dot(h);

        if a.abs() < 0.00001 {
            return false;
        }

        let f = 1.0 / a;
        let s = origin - v0;
        let u = f * s.dot(h);

        if !(0.0..=1.0).contains(&u) {
            return false;
        }

        let q = s.cross(edge1);
        let v = f * direction.dot(q);

        if v < 0.0 || u + v > 1.0 {
            return false;
        }

        let t = f * edge2.dot(q);

        t > 0.0 && t < max_distance
    }
}

/// Bakes ambient occlusion to per-vertex values.
pub fn bake_ao_vertices(mesh: &Mesh, config: &BakeAo) -> Vec<f32> {
    let accel = AoAccelerator::build(mesh);
    let normals = compute_vertex_normals(mesh);

    let mut ao_values = Vec::with_capacity(mesh.positions.len());

    for (i, &pos) in mesh.positions.iter().enumerate() {
        let normal = normals[i];
        let ao = compute_ao_at_point(&accel, pos, normal, config);
        ao_values.push(ao);
    }

    ao_values
}

/// Computes AO at a single point.
fn compute_ao_at_point(
    accel: &AoAccelerator,
    position: Vec3,
    normal: Vec3,
    config: &BakeAo,
) -> f32 {
    let origin = position + normal * config.bias;

    let mut occlusion = 0.0;
    let mut total_weight = 0.0;

    // Generate hemisphere samples
    let (tangent, bitangent) = compute_tangent_frame(normal);

    for i in 0..config.ray_count {
        let (local_dir, weight) = hemisphere_sample(i, config.ray_count, config.cosine_weighted);

        // Transform to world space
        let dir = tangent * local_dir.x + normal * local_dir.y + bitangent * local_dir.z;

        let weight = if config.cosine_weighted {
            1.0 // Already weighted by cosine in sampling
        } else {
            weight * dir.dot(normal).max(0.0)
        };

        if accel.occluded(origin, dir, config.max_distance) {
            occlusion += weight;
        }

        total_weight += weight;
    }

    if total_weight > 0.0 {
        1.0 - (occlusion / total_weight).powf(config.falloff_power)
    } else {
        1.0
    }
}

/// Generates a hemisphere sample direction.
fn hemisphere_sample(index: u32, total: u32, cosine_weighted: bool) -> (Vec3, f32) {
    // Use stratified sampling with golden ratio
    let i = index as f32;
    let n = total as f32;

    let phi = 2.0 * std::f32::consts::PI * (i * 0.618033988749895); // Golden ratio
    let cos_theta = if cosine_weighted {
        (1.0 - (i + 0.5) / n).sqrt()
    } else {
        1.0 - (i + 0.5) / n
    };
    let sin_theta = (1.0 - cos_theta * cos_theta).sqrt();

    let dir = Vec3::new(sin_theta * phi.cos(), cos_theta, sin_theta * phi.sin());

    let weight = if cosine_weighted { 1.0 } else { cos_theta };

    (dir, weight)
}

/// Computes tangent and bitangent from a normal.
fn compute_tangent_frame(normal: Vec3) -> (Vec3, Vec3) {
    let up = if normal.y.abs() < 0.999 {
        Vec3::Y
    } else {
        Vec3::X
    };

    let tangent = up.cross(normal).normalize();
    let bitangent = normal.cross(tangent);

    (tangent, bitangent)
}

/// Computes per-vertex normals.
fn compute_vertex_normals(mesh: &Mesh) -> Vec<Vec3> {
    let mut normals = vec![Vec3::ZERO; mesh.positions.len()];

    for i in (0..mesh.indices.len()).step_by(3) {
        let i0 = mesh.indices[i] as usize;
        let i1 = mesh.indices[i + 1] as usize;
        let i2 = mesh.indices[i + 2] as usize;

        let v0 = mesh.positions[i0];
        let v1 = mesh.positions[i1];
        let v2 = mesh.positions[i2];

        let face_normal = (v1 - v0).cross(v2 - v0);

        normals[i0] += face_normal;
        normals[i1] += face_normal;
        normals[i2] += face_normal;
    }

    for normal in &mut normals {
        *normal = normal.normalize_or_zero();
        if *normal == Vec3::ZERO {
            *normal = Vec3::Y;
        }
    }

    normals
}

/// Result of AO baking to a texture.
#[derive(Debug, Clone)]
pub struct AoTexture {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// AO values (0-255, row-major).
    pub data: Vec<u8>,
}

impl AoTexture {
    /// Creates a new texture filled with white (no occlusion).
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            data: vec![255; (width * height) as usize],
        }
    }

    /// Gets the AO value at a UV coordinate.
    pub fn sample(&self, u: f32, v: f32) -> f32 {
        let x = ((u * self.width as f32) as u32).min(self.width - 1);
        let y = ((v * self.height as f32) as u32).min(self.height - 1);
        self.data[(y * self.width + x) as usize] as f32 / 255.0
    }

    /// Sets the AO value at a pixel.
    pub fn set_pixel(&mut self, x: u32, y: u32, value: f32) {
        if x < self.width && y < self.height {
            self.data[(y * self.width + x) as usize] = (value.clamp(0.0, 1.0) * 255.0) as u8;
        }
    }
}

/// Bakes AO to a texture using UV coordinates.
pub fn bake_ao_texture(mesh: &Mesh, config: &BakeAo, width: u32, height: u32) -> Option<AoTexture> {
    if mesh.uvs.is_empty() {
        return None;
    }

    let accel = AoAccelerator::build(mesh);
    let uvs = &mesh.uvs;
    let mut texture = AoTexture::new(width, height);

    // Rasterize triangles to texture
    for i in (0..mesh.indices.len()).step_by(3) {
        let i0 = mesh.indices[i] as usize;
        let i1 = mesh.indices[i + 1] as usize;
        let i2 = mesh.indices[i + 2] as usize;

        if i0 >= uvs.len() || i1 >= uvs.len() || i2 >= uvs.len() {
            continue;
        }

        let uv0 = uvs[i0];
        let uv1 = uvs[i1];
        let uv2 = uvs[i2];

        let p0 = mesh.positions[i0];
        let p1 = mesh.positions[i1];
        let p2 = mesh.positions[i2];

        let face_normal = (p1 - p0).cross(p2 - p0).normalize_or_zero();

        // Compute UV bounding box
        let min_u = uv0.x.min(uv1.x).min(uv2.x);
        let max_u = uv0.x.max(uv1.x).max(uv2.x);
        let min_v = uv0.y.min(uv1.y).min(uv2.y);
        let max_v = uv0.y.max(uv1.y).max(uv2.y);

        let start_x = (min_u * width as f32).floor() as i32;
        let end_x = (max_u * width as f32).ceil() as i32;
        let start_y = (min_v * height as f32).floor() as i32;
        let end_y = (max_v * height as f32).ceil() as i32;

        for py in start_y..=end_y {
            for px in start_x..=end_x {
                if px < 0 || py < 0 || px >= width as i32 || py >= height as i32 {
                    continue;
                }

                let u = (px as f32 + 0.5) / width as f32;
                let v = (py as f32 + 0.5) / height as f32;

                // Check if point is in triangle
                if let Some((w0, w1, w2)) = barycentric_2d(glam::Vec2::new(u, v), uv0, uv1, uv2) {
                    if w0 >= 0.0 && w1 >= 0.0 && w2 >= 0.0 {
                        // Interpolate position
                        let pos = p0 * w0 + p1 * w1 + p2 * w2;
                        let ao = compute_ao_at_point(&accel, pos, face_normal, config);
                        texture.set_pixel(px as u32, py as u32, ao);
                    }
                }
            }
        }
    }

    Some(texture)
}

/// Computes barycentric coordinates in 2D.
fn barycentric_2d(
    p: glam::Vec2,
    a: glam::Vec2,
    b: glam::Vec2,
    c: glam::Vec2,
) -> Option<(f32, f32, f32)> {
    let v0 = c - a;
    let v1 = b - a;
    let v2 = p - a;

    let dot00 = v0.dot(v0);
    let dot01 = v0.dot(v1);
    let dot02 = v0.dot(v2);
    let dot11 = v1.dot(v1);
    let dot12 = v1.dot(v2);

    let denom = dot00 * dot11 - dot01 * dot01;
    if denom.abs() < 0.00001 {
        return None;
    }

    let inv_denom = 1.0 / denom;
    let u = (dot11 * dot02 - dot01 * dot12) * inv_denom;
    let v = (dot00 * dot12 - dot01 * dot02) * inv_denom;

    Some((1.0 - u - v, v, u))
}

/// Blurs an AO texture to reduce noise.
pub fn blur_ao_texture(texture: &AoTexture, radius: u32) -> AoTexture {
    let mut result = AoTexture::new(texture.width, texture.height);
    let r = radius as i32;

    for y in 0..texture.height {
        for x in 0..texture.width {
            let mut sum = 0.0;
            let mut count = 0.0;

            for dy in -r..=r {
                for dx in -r..=r {
                    let nx = x as i32 + dx;
                    let ny = y as i32 + dy;

                    if nx >= 0 && nx < texture.width as i32 && ny >= 0 && ny < texture.height as i32
                    {
                        let idx = (ny as u32 * texture.width + nx as u32) as usize;
                        sum += texture.data[idx] as f32;
                        count += 1.0;
                    }
                }
            }

            result.data[(y * texture.width + x) as usize] = (sum / count) as u8;
        }
    }

    result
}

/// Converts AO values to vertex colors (RGBA).
pub fn ao_to_vertex_colors(ao_values: &[f32]) -> Vec<[f32; 4]> {
    ao_values
        .iter()
        .map(|&ao| {
            let c = ao.clamp(0.0, 1.0);
            [c, c, c, 1.0]
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Cuboid;

    #[test]
    fn test_ao_config_default() {
        let config = BakeAo::default();
        assert!(config.ray_count > 0);
        assert!(config.max_distance > 0.0);
    }

    #[test]
    fn test_ao_config_presets() {
        let low = BakeAo::low();
        let high = BakeAo::high();

        assert!(low.ray_count < high.ray_count);
    }

    #[test]
    fn test_ao_accelerator_build() {
        let mesh = Cuboid::default().apply();
        let accel = AoAccelerator::build(&mesh);

        assert!(!accel.triangles.is_empty());
    }

    #[test]
    fn test_ao_accelerator_occluded() {
        let mesh = Cuboid::default().apply();
        let accel = AoAccelerator::build(&mesh);

        // Ray from outside pointing at mesh should hit
        let hit = accel.occluded(Vec3::new(5.0, 0.0, 0.0), Vec3::NEG_X, 10.0);
        assert!(hit);

        // Ray pointing away should not hit
        let miss = accel.occluded(Vec3::new(5.0, 0.0, 0.0), Vec3::X, 10.0);
        assert!(!miss);
    }

    #[test]
    fn test_bake_ao_vertices() {
        let mesh = Cuboid::default().apply();
        let config = BakeAo::low();

        let ao_values = bake_ao_vertices(&mesh, &config);

        assert_eq!(ao_values.len(), mesh.positions.len());
        for &ao in &ao_values {
            assert!(ao >= 0.0 && ao <= 1.0);
        }
    }

    #[test]
    fn test_hemisphere_sample() {
        for i in 0..10 {
            let (dir, weight) = hemisphere_sample(i, 10, false);

            // Y should be positive (upper hemisphere)
            assert!(dir.y >= 0.0);
            assert!(weight >= 0.0);
        }
    }

    #[test]
    fn test_compute_tangent_frame() {
        let normal = Vec3::Y;
        let (tangent, bitangent) = compute_tangent_frame(normal);

        // Should be orthonormal
        assert!((tangent.dot(normal)).abs() < 0.001);
        assert!((bitangent.dot(normal)).abs() < 0.001);
        assert!((tangent.dot(bitangent)).abs() < 0.001);
    }

    #[test]
    fn test_ao_texture_new() {
        let texture = AoTexture::new(64, 64);

        assert_eq!(texture.width, 64);
        assert_eq!(texture.height, 64);
        assert_eq!(texture.data.len(), 64 * 64);
    }

    #[test]
    fn test_ao_texture_set_pixel() {
        let mut texture = AoTexture::new(10, 10);

        texture.set_pixel(5, 5, 0.5);
        let value = texture.data[(5 * 10 + 5) as usize];

        assert!((value as f32 / 255.0 - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_ao_texture_sample() {
        let mut texture = AoTexture::new(10, 10);
        texture.set_pixel(5, 5, 0.5);

        let sampled = texture.sample(0.55, 0.55);
        assert!((sampled - 0.5).abs() < 0.1);
    }

    #[test]
    fn test_blur_ao_texture() {
        let mut texture = AoTexture::new(10, 10);
        texture.set_pixel(5, 5, 0.0); // Dark spot

        let blurred = blur_ao_texture(&texture, 1);

        // Spot should be lighter after blur
        let original = texture.data[(5 * 10 + 5) as usize];
        let after_blur = blurred.data[(5 * 10 + 5) as usize];

        assert!(after_blur > original);
    }

    #[test]
    fn test_barycentric_2d() {
        let a = glam::Vec2::new(0.0, 0.0);
        let b = glam::Vec2::new(1.0, 0.0);
        let c = glam::Vec2::new(0.0, 1.0);

        // Center should have roughly equal weights
        let center = glam::Vec2::new(0.33, 0.33);
        let result = barycentric_2d(center, a, b, c);

        assert!(result.is_some());
        let (w0, w1, w2) = result.unwrap();
        assert!(w0 > 0.0 && w1 > 0.0 && w2 > 0.0);
    }

    #[test]
    fn test_ao_to_vertex_colors() {
        let ao_values = vec![0.5; 10];

        let colors = ao_to_vertex_colors(&ao_values);

        assert_eq!(colors.len(), 10);
        for color in &colors {
            assert!((color[0] - 0.5).abs() < 0.01);
            assert_eq!(color[3], 1.0); // Alpha
        }
    }

    #[test]
    fn test_compute_vertex_normals() {
        let mesh = Cuboid::default().apply();
        let normals = compute_vertex_normals(&mesh);

        assert_eq!(normals.len(), mesh.positions.len());
        for normal in &normals {
            assert!((normal.length() - 1.0).abs() < 0.01);
        }
    }
}
