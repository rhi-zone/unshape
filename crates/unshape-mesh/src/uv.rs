//! UV projection and mapping operations.
//!
//! Provides various methods for generating texture coordinates on meshes,
//! including UV atlas packing for combining multiple UV charts efficiently.

use glam::{Mat4, Vec2, Vec3};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

use crate::Mesh;

/// Projects UVs onto a mesh using planar projection.
///
/// Projects vertices onto the XY plane (by default) and uses the result as UVs.
/// The projection can be transformed using the provided matrix.
pub fn project_planar(mesh: &mut Mesh, transform: Mat4, scale: Vec2) {
    mesh.uvs.clear();
    mesh.uvs.reserve(mesh.positions.len());

    for pos in &mesh.positions {
        // Transform position
        let p = transform.transform_point3(*pos);

        // Project onto XY plane
        let uv = Vec2::new(p.x * scale.x, p.y * scale.y);
        mesh.uvs.push(uv);
    }
}

/// Projects UVs using planar projection along a specific axis.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum ProjectionAxis {
    /// Project along X axis (use YZ plane).
    X,
    /// Project along Y axis (use XZ plane).
    #[default]
    Y,
    /// Project along Z axis (use XY plane).
    Z,
}

/// Projects UVs along the specified axis.
pub fn project_planar_axis(mesh: &mut Mesh, axis: ProjectionAxis, scale: Vec2) {
    mesh.uvs.clear();
    mesh.uvs.reserve(mesh.positions.len());

    for pos in &mesh.positions {
        let uv = match axis {
            ProjectionAxis::X => Vec2::new(pos.y * scale.x, pos.z * scale.y),
            ProjectionAxis::Y => Vec2::new(pos.x * scale.x, pos.z * scale.y),
            ProjectionAxis::Z => Vec2::new(pos.x * scale.x, pos.y * scale.y),
        };
        mesh.uvs.push(uv);
    }
}

/// Configuration for cylindrical UV projection.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct CylindricalConfig {
    /// Center of the cylinder.
    pub center: Vec3,
    /// Axis of the cylinder (default Y).
    pub axis: Vec3,
    /// UV scale.
    pub scale: Vec2,
    /// Whether to use the mesh bounds for V coordinate.
    pub use_bounds: bool,
}

impl Default for CylindricalConfig {
    fn default() -> Self {
        Self {
            center: Vec3::ZERO,
            axis: Vec3::Y,
            scale: Vec2::ONE,
            use_bounds: true,
        }
    }
}

/// Projects UVs using cylindrical projection.
///
/// Wraps UV coordinates around a cylinder aligned with the specified axis.
/// U = angle around axis, V = distance along axis.
pub fn project_cylindrical(mesh: &mut Mesh, config: &CylindricalConfig) {
    mesh.uvs.clear();
    mesh.uvs.reserve(mesh.positions.len());

    let axis = config.axis.normalize();

    // Compute bounds along axis if needed
    let (min_v, max_v) = if config.use_bounds {
        let mut min = f32::MAX;
        let mut max = f32::MIN;
        for pos in &mesh.positions {
            let v = (*pos - config.center).dot(axis);
            min = min.min(v);
            max = max.max(v);
        }
        (min, max)
    } else {
        (0.0, 1.0)
    };

    let v_range = (max_v - min_v).max(0.001);

    // Build orthonormal basis
    let (tangent, bitangent) = orthonormal_basis(axis);

    for pos in &mesh.positions {
        let local = *pos - config.center;

        // Project onto plane perpendicular to axis
        let x = local.dot(tangent);
        let y = local.dot(bitangent);

        // Angle around axis (0 to 1)
        let u = (y.atan2(x) / std::f32::consts::TAU + 0.5).fract();

        // Distance along axis (normalized to bounds)
        let v = if config.use_bounds {
            (local.dot(axis) - min_v) / v_range
        } else {
            local.dot(axis)
        };

        mesh.uvs
            .push(Vec2::new(u * config.scale.x, v * config.scale.y));
    }
}

/// Configuration for spherical UV projection.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct SphericalConfig {
    /// Center of the sphere.
    pub center: Vec3,
    /// Up axis for the sphere (default Y).
    pub up: Vec3,
    /// UV scale.
    pub scale: Vec2,
}

impl Default for SphericalConfig {
    fn default() -> Self {
        Self {
            center: Vec3::ZERO,
            up: Vec3::Y,
            scale: Vec2::ONE,
        }
    }
}

/// Projects UVs using spherical projection.
///
/// Maps positions to UV coordinates on a sphere.
/// U = longitude (angle around up axis), V = latitude (angle from up axis).
pub fn project_spherical(mesh: &mut Mesh, config: &SphericalConfig) {
    mesh.uvs.clear();
    mesh.uvs.reserve(mesh.positions.len());

    let up = config.up.normalize();
    let (tangent, bitangent) = orthonormal_basis(up);

    for pos in &mesh.positions {
        let dir = (*pos - config.center).normalize_or_zero();

        // Longitude (U): angle around up axis
        let x = dir.dot(tangent);
        let y = dir.dot(bitangent);
        let u = (y.atan2(x) / std::f32::consts::TAU + 0.5).fract();

        // Latitude (V): angle from up axis (0 at bottom, 1 at top)
        let v = (dir.dot(up) * 0.5 + 0.5).clamp(0.0, 1.0);

        mesh.uvs
            .push(Vec2::new(u * config.scale.x, v * config.scale.y));
    }
}

/// Configuration for box/triplanar UV projection.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct BoxConfig {
    /// Center of the projection.
    pub center: Vec3,
    /// UV scale.
    pub scale: Vec2,
    /// Blend sharpness for triplanar (higher = sharper transitions).
    pub blend_sharpness: f32,
}

impl Default for BoxConfig {
    fn default() -> Self {
        Self {
            center: Vec3::ZERO,
            scale: Vec2::ONE,
            blend_sharpness: 1.0,
        }
    }
}

/// Projects UVs using box projection (per-face dominant axis).
///
/// Each triangle is projected along its dominant normal axis.
/// This creates clean projections for box-like or architectural geometry.
pub fn project_box(mesh: &mut Mesh, config: &BoxConfig) {
    // We need to split faces for proper box projection since
    // vertices shared between faces with different dominant axes
    // would have conflicting UVs.

    // For now, project based on vertex normal (approximate)
    // For proper box projection, use project_box_per_face

    mesh.uvs.clear();
    mesh.uvs.reserve(mesh.positions.len());

    // Ensure we have normals
    let normals: Vec<Vec3> = if mesh.normals.len() == mesh.positions.len() {
        mesh.normals.clone()
    } else {
        compute_vertex_normals(mesh)
    };

    for (pos, normal) in mesh.positions.iter().zip(normals.iter()) {
        let local = *pos - config.center;
        let abs_normal = normal.abs();

        // Choose projection based on dominant normal axis
        let uv = if abs_normal.x >= abs_normal.y && abs_normal.x >= abs_normal.z {
            // Project along X (use YZ)
            Vec2::new(local.y, local.z)
        } else if abs_normal.y >= abs_normal.z {
            // Project along Y (use XZ)
            Vec2::new(local.x, local.z)
        } else {
            // Project along Z (use XY)
            Vec2::new(local.x, local.y)
        };

        mesh.uvs.push(uv * config.scale);
    }
}

/// Projects UVs using box projection on a per-face basis.
///
/// This creates a new mesh where each face has its own vertices
/// with UVs projected along the face's dominant axis.
pub fn project_box_per_face(mesh: &Mesh, config: &BoxConfig) -> Mesh {
    let triangle_count = mesh.triangle_count();
    let mut result = Mesh::with_capacity(triangle_count * 3, triangle_count);

    for tri in mesh.indices.chunks(3) {
        let [i0, i1, i2] = [tri[0] as usize, tri[1] as usize, tri[2] as usize];

        let p0 = mesh.positions[i0];
        let p1 = mesh.positions[i1];
        let p2 = mesh.positions[i2];

        // Compute face normal
        let edge1 = p1 - p0;
        let edge2 = p2 - p0;
        let face_normal = edge1.cross(edge2).normalize_or_zero();
        let abs_normal = face_normal.abs();

        // Determine dominant axis
        let (u_axis, v_axis) = if abs_normal.x >= abs_normal.y && abs_normal.x >= abs_normal.z {
            (1, 2) // YZ plane
        } else if abs_normal.y >= abs_normal.z {
            (0, 2) // XZ plane
        } else {
            (0, 1) // XY plane
        };

        let base = result.positions.len() as u32;

        for pos in [p0, p1, p2] {
            let local = pos - config.center;
            let uv = Vec2::new(
                [local.x, local.y, local.z][u_axis],
                [local.x, local.y, local.z][v_axis],
            ) * config.scale;

            result.positions.push(pos);
            result.normals.push(face_normal);
            result.uvs.push(uv);
        }

        result.indices.push(base);
        result.indices.push(base + 1);
        result.indices.push(base + 2);
    }

    result
}

/// Scales existing UVs.
pub fn scale_uvs(mesh: &mut Mesh, scale: Vec2) {
    for uv in &mut mesh.uvs {
        *uv *= scale;
    }
}

/// Translates existing UVs.
pub fn translate_uvs(mesh: &mut Mesh, offset: Vec2) {
    for uv in &mut mesh.uvs {
        *uv += offset;
    }
}

/// Rotates existing UVs around a pivot point.
pub fn rotate_uvs(mesh: &mut Mesh, angle: f32, pivot: Vec2) {
    let cos_a = angle.cos();
    let sin_a = angle.sin();

    for uv in &mut mesh.uvs {
        let local = *uv - pivot;
        let rotated = Vec2::new(
            local.x * cos_a - local.y * sin_a,
            local.x * sin_a + local.y * cos_a,
        );
        *uv = rotated + pivot;
    }
}

/// Transforms UVs by a 2D matrix.
pub fn transform_uvs(mesh: &mut Mesh, matrix: glam::Mat3) {
    for uv in &mut mesh.uvs {
        let v3 = matrix * glam::Vec3::new(uv.x, uv.y, 1.0);
        *uv = Vec2::new(v3.x, v3.y);
    }
}

/// Normalizes UVs to fit within [0, 1] range based on current bounds.
pub fn normalize_uvs(mesh: &mut Mesh) {
    if mesh.uvs.is_empty() {
        return;
    }

    let mut min = Vec2::splat(f32::MAX);
    let mut max = Vec2::splat(f32::MIN);

    for uv in &mesh.uvs {
        min = min.min(*uv);
        max = max.max(*uv);
    }

    let range = max - min;
    let scale = Vec2::new(
        if range.x > 0.001 { 1.0 / range.x } else { 1.0 },
        if range.y > 0.001 { 1.0 / range.y } else { 1.0 },
    );

    for uv in &mut mesh.uvs {
        *uv = (*uv - min) * scale;
    }
}

/// Flips UVs along the U axis.
pub fn flip_u(mesh: &mut Mesh) {
    for uv in &mut mesh.uvs {
        uv.x = 1.0 - uv.x;
    }
}

/// Flips UVs along the V axis.
pub fn flip_v(mesh: &mut Mesh) {
    for uv in &mut mesh.uvs {
        uv.y = 1.0 - uv.y;
    }
}

// ============================================================================
// UV Atlas Packing
// ============================================================================

/// A UV chart (island) that can be packed into an atlas.
#[derive(Debug, Clone)]
pub struct UvChart {
    /// Indices of vertices in this chart.
    pub vertex_indices: Vec<usize>,
    /// Original UVs for vertices in this chart.
    pub uvs: Vec<Vec2>,
    /// Bounding box minimum in UV space.
    pub min: Vec2,
    /// Bounding box maximum in UV space.
    pub max: Vec2,
}

impl UvChart {
    /// Creates a chart from vertex indices and their UVs.
    pub fn new(vertex_indices: Vec<usize>, uvs: Vec<Vec2>) -> Self {
        let (min, max) = compute_bounds(&uvs);
        Self {
            vertex_indices,
            uvs,
            min,
            max,
        }
    }

    /// Returns the width of this chart.
    pub fn width(&self) -> f32 {
        self.max.x - self.min.x
    }

    /// Returns the height of this chart.
    pub fn height(&self) -> f32 {
        self.max.y - self.min.y
    }

    /// Returns the area of this chart's bounding box.
    pub fn area(&self) -> f32 {
        self.width() * self.height()
    }
}

/// Configuration for UV atlas packing.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct AtlasPackConfig {
    /// Padding between charts in UV space (0.0 to 1.0).
    pub padding: f32,
    /// Whether to allow chart rotation for better packing.
    pub allow_rotation: bool,
    /// Target atlas aspect ratio (width / height).
    pub target_aspect: f32,
}

impl Default for AtlasPackConfig {
    fn default() -> Self {
        Self {
            padding: 0.01,
            allow_rotation: true,
            target_aspect: 1.0,
        }
    }
}

/// Result of UV atlas packing.
#[derive(Debug, Clone)]
pub struct AtlasPackResult {
    /// Packed charts with their new positions.
    pub charts: Vec<PackedChart>,
    /// Total atlas width (normalized, may exceed 1.0).
    pub width: f32,
    /// Total atlas height (normalized, may exceed 1.0).
    pub height: f32,
    /// Packing efficiency (chart area / atlas area).
    pub efficiency: f32,
}

/// A chart that has been placed in the atlas.
#[derive(Debug, Clone)]
pub struct PackedChart {
    /// Original chart index.
    pub chart_index: usize,
    /// New position (bottom-left corner).
    pub position: Vec2,
    /// Whether the chart was rotated 90 degrees.
    pub rotated: bool,
}

/// Finds UV islands (connected components) in a mesh.
///
/// Returns a list of charts, where each chart contains the vertex indices
/// and UV coordinates for one connected UV region.
pub fn find_uv_islands(mesh: &Mesh) -> Vec<UvChart> {
    if mesh.uvs.is_empty() || mesh.indices.is_empty() {
        return vec![];
    }

    let n = mesh.positions.len();

    // Build adjacency from triangles
    let mut adjacency: HashMap<usize, HashSet<usize>> = HashMap::new();

    for tri in mesh.indices.chunks(3) {
        if tri.len() < 3 {
            continue;
        }
        let [i0, i1, i2] = [tri[0] as usize, tri[1] as usize, tri[2] as usize];

        if i0 >= n || i1 >= n || i2 >= n {
            continue;
        }

        // Connect vertices that share a triangle
        adjacency.entry(i0).or_default().insert(i1);
        adjacency.entry(i0).or_default().insert(i2);
        adjacency.entry(i1).or_default().insert(i0);
        adjacency.entry(i1).or_default().insert(i2);
        adjacency.entry(i2).or_default().insert(i0);
        adjacency.entry(i2).or_default().insert(i1);
    }

    // Find connected components using flood fill
    let mut visited = vec![false; n];
    let mut charts = Vec::new();

    for start in 0..n {
        if visited[start] || !adjacency.contains_key(&start) {
            continue;
        }

        // Flood fill from this vertex
        let mut component = Vec::new();
        let mut stack = vec![start];

        while let Some(v) = stack.pop() {
            if visited[v] {
                continue;
            }
            visited[v] = true;
            component.push(v);

            if let Some(neighbors) = adjacency.get(&v) {
                for &neighbor in neighbors {
                    if !visited[neighbor] {
                        stack.push(neighbor);
                    }
                }
            }
        }

        if !component.is_empty() {
            let uvs: Vec<Vec2> = component
                .iter()
                .map(|&i| {
                    if i < mesh.uvs.len() {
                        mesh.uvs[i]
                    } else {
                        Vec2::ZERO
                    }
                })
                .collect();

            charts.push(UvChart::new(component, uvs));
        }
    }

    charts
}

/// Packs UV charts into an atlas using the maxrects algorithm.
///
/// Returns packing result with chart positions and atlas dimensions.
pub fn pack_uv_charts(charts: &[UvChart], config: &AtlasPackConfig) -> AtlasPackResult {
    if charts.is_empty() {
        return AtlasPackResult {
            charts: vec![],
            width: 0.0,
            height: 0.0,
            efficiency: 0.0,
        };
    }

    // Sort charts by area (largest first) for better packing
    let mut sorted_indices: Vec<usize> = (0..charts.len()).collect();
    sorted_indices.sort_by(|&a, &b| {
        let area_a = charts[a].area();
        let area_b = charts[b].area();
        area_b
            .partial_cmp(&area_a)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Calculate total area and estimate atlas size
    let total_chart_area: f32 = charts.iter().map(|c| c.area()).sum();
    let padding = config.padding;

    // Start with an estimated size
    let estimated_side = (total_chart_area * 1.5).sqrt();
    let atlas_width = estimated_side * config.target_aspect.sqrt();
    let mut atlas_height = estimated_side / config.target_aspect.sqrt();

    // Use maxrects bin packing
    let mut packed = Vec::with_capacity(charts.len());
    let mut free_rects = vec![Rect {
        x: 0.0,
        y: 0.0,
        w: atlas_width,
        h: atlas_height,
    }];

    for &chart_idx in &sorted_indices {
        let chart = &charts[chart_idx];
        let w = chart.width() + padding * 2.0;
        let h = chart.height() + padding * 2.0;

        // Try to find best placement
        let (best_rect_idx, best_pos, rotated) =
            find_best_placement(&free_rects, w, h, config.allow_rotation);

        if let Some(rect_idx) = best_rect_idx {
            let (actual_w, actual_h) = if rotated { (h, w) } else { (w, h) };

            packed.push(PackedChart {
                chart_index: chart_idx,
                position: Vec2::new(best_pos.x + padding, best_pos.y + padding),
                rotated,
            });

            // Split the free rect
            let free_rect = free_rects[rect_idx];
            free_rects.remove(rect_idx);

            // Add new free rects (guillotine split)
            let right = Rect {
                x: free_rect.x + actual_w,
                y: free_rect.y,
                w: free_rect.w - actual_w,
                h: actual_h,
            };
            let top = Rect {
                x: free_rect.x,
                y: free_rect.y + actual_h,
                w: free_rect.w,
                h: free_rect.h - actual_h,
            };

            if right.w > 0.001 && right.h > 0.001 {
                free_rects.push(right);
            }
            if top.w > 0.001 && top.h > 0.001 {
                free_rects.push(top);
            }
        } else {
            // Expand atlas and retry
            atlas_height += h;
            free_rects.push(Rect {
                x: 0.0,
                y: atlas_height - h,
                w: atlas_width,
                h,
            });

            packed.push(PackedChart {
                chart_index: chart_idx,
                position: Vec2::new(padding, atlas_height - h + padding),
                rotated: false,
            });
        }
    }

    // Calculate actual bounds
    let mut max_x = 0.0f32;
    let mut max_y = 0.0f32;

    for pc in &packed {
        let chart = &charts[pc.chart_index];
        let (w, h) = if pc.rotated {
            (chart.height(), chart.width())
        } else {
            (chart.width(), chart.height())
        };
        max_x = max_x.max(pc.position.x + w + padding);
        max_y = max_y.max(pc.position.y + h + padding);
    }

    let atlas_area = max_x * max_y;
    let efficiency = if atlas_area > 0.0 {
        total_chart_area / atlas_area
    } else {
        0.0
    };

    AtlasPackResult {
        charts: packed,
        width: max_x,
        height: max_y,
        efficiency,
    }
}

/// Applies atlas packing result to a mesh's UVs.
///
/// This modifies the mesh's UV coordinates according to the packing result,
/// normalizing them to fit within [0, 1] range.
pub fn apply_atlas_pack(mesh: &mut Mesh, islands: &[UvChart], result: &AtlasPackResult) {
    if result.width < 0.001 || result.height < 0.001 {
        return;
    }

    // Create a map from original vertex index to packed position
    for packed in &result.charts {
        let island = &islands[packed.chart_index];

        for (local_idx, &global_idx) in island.vertex_indices.iter().enumerate() {
            if global_idx >= mesh.uvs.len() {
                continue;
            }

            // Get original UV relative to island bounds
            let original_uv = island.uvs[local_idx];
            let local_uv = original_uv - island.min;

            // Apply rotation if needed
            let (local_u, local_v) = if packed.rotated {
                (local_uv.y, island.width() - local_uv.x)
            } else {
                (local_uv.x, local_uv.y)
            };

            // Apply packed position and normalize
            let packed_uv = Vec2::new(
                (packed.position.x + local_u) / result.width,
                (packed.position.y + local_v) / result.height,
            );

            mesh.uvs[global_idx] = packed_uv;
        }
    }
}

/// Packs a mesh's UV islands into an atlas.
///
/// This is a convenience function that finds islands, packs them, and applies
/// the result in one step.
pub fn pack_mesh_uvs(mesh: &mut Mesh, config: &AtlasPackConfig) {
    let islands = find_uv_islands(mesh);
    if islands.is_empty() {
        return;
    }

    let result = pack_uv_charts(&islands, config);
    apply_atlas_pack(mesh, &islands, &result);
}

/// Packs UV charts from multiple meshes into a single atlas.
///
/// Returns the packing result. The caller is responsible for applying
/// the result to individual meshes using `apply_atlas_pack_to_mesh`.
pub fn pack_multi_mesh_uvs(
    meshes: &[&Mesh],
    config: &AtlasPackConfig,
) -> (Vec<Vec<UvChart>>, AtlasPackResult) {
    let mut all_charts = Vec::new();
    let mut charts_per_mesh = Vec::new();

    for mesh in meshes {
        let islands = find_uv_islands(mesh);
        charts_per_mesh.push(islands.clone());
        all_charts.extend(islands);
    }

    let result = pack_uv_charts(&all_charts, config);
    (charts_per_mesh, result)
}

// Internal rectangle for packing
#[derive(Debug, Clone, Copy)]
struct Rect {
    x: f32,
    y: f32,
    w: f32,
    h: f32,
}

/// Finds the best placement for a rectangle in the free rects.
fn find_best_placement(
    free_rects: &[Rect],
    w: f32,
    h: f32,
    allow_rotation: bool,
) -> (Option<usize>, Vec2, bool) {
    let mut best_idx = None;
    let mut best_pos = Vec2::ZERO;
    let mut best_rotated = false;
    let mut best_score = f32::MAX;

    for (i, rect) in free_rects.iter().enumerate() {
        // Try normal orientation
        if w <= rect.w && h <= rect.h {
            // Best short side fit
            let score = (rect.w - w).min(rect.h - h);
            if score < best_score {
                best_score = score;
                best_idx = Some(i);
                best_pos = Vec2::new(rect.x, rect.y);
                best_rotated = false;
            }
        }

        // Try rotated
        if allow_rotation && h <= rect.w && w <= rect.h {
            let score = (rect.w - h).min(rect.h - w);
            if score < best_score {
                best_score = score;
                best_idx = Some(i);
                best_pos = Vec2::new(rect.x, rect.y);
                best_rotated = true;
            }
        }
    }

    (best_idx, best_pos, best_rotated)
}

/// Computes min/max bounds of UV coordinates.
fn compute_bounds(uvs: &[Vec2]) -> (Vec2, Vec2) {
    if uvs.is_empty() {
        return (Vec2::ZERO, Vec2::ZERO);
    }

    let mut min = Vec2::splat(f32::MAX);
    let mut max = Vec2::splat(f32::MIN);

    for uv in uvs {
        min = min.min(*uv);
        max = max.max(*uv);
    }

    (min, max)
}

// ============================================================================
// Helper functions
// ============================================================================

/// Computes smooth vertex normals.
fn compute_vertex_normals(mesh: &Mesh) -> Vec<Vec3> {
    let mut normals = vec![Vec3::ZERO; mesh.positions.len()];

    for tri in mesh.indices.chunks(3) {
        let [i0, i1, i2] = [tri[0] as usize, tri[1] as usize, tri[2] as usize];
        let v0 = mesh.positions[i0];
        let v1 = mesh.positions[i1];
        let v2 = mesh.positions[i2];

        let normal = (v1 - v0).cross(v2 - v0);
        normals[i0] += normal;
        normals[i1] += normal;
        normals[i2] += normal;
    }

    for normal in &mut normals {
        *normal = normal.normalize_or_zero();
    }

    normals
}

/// Builds an orthonormal basis from a single vector.
fn orthonormal_basis(n: Vec3) -> (Vec3, Vec3) {
    let sign = if n.z >= 0.0 { 1.0 } else { -1.0 };
    let a = -1.0 / (sign + n.z);
    let b = n.x * n.y * a;

    let tangent = Vec3::new(1.0 + sign * n.x * n.x * a, sign * b, -sign * n.x);
    let bitangent = Vec3::new(b, sign + n.y * n.y * a, -n.y);

    (tangent, bitangent)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Cuboid;

    #[test]
    fn test_planar_projection() {
        let mut mesh = Cuboid::default().apply();
        project_planar(&mut mesh, Mat4::IDENTITY, Vec2::ONE);

        assert_eq!(mesh.uvs.len(), mesh.positions.len());
    }

    #[test]
    fn test_planar_axis_projection() {
        let mut mesh = Cuboid::default().apply();
        project_planar_axis(&mut mesh, ProjectionAxis::Z, Vec2::ONE);

        assert_eq!(mesh.uvs.len(), mesh.positions.len());
    }

    #[test]
    fn test_cylindrical_projection() {
        let mut mesh = Cuboid::default().apply();
        project_cylindrical(&mut mesh, &CylindricalConfig::default());

        assert_eq!(mesh.uvs.len(), mesh.positions.len());

        // All UVs should be in valid range
        for uv in &mesh.uvs {
            assert!(uv.x >= 0.0 && uv.x <= 1.0);
            assert!(uv.y >= 0.0 && uv.y <= 1.0);
        }
    }

    #[test]
    fn test_spherical_projection() {
        let mut mesh = Cuboid::default().apply();
        project_spherical(&mut mesh, &SphericalConfig::default());

        assert_eq!(mesh.uvs.len(), mesh.positions.len());

        // All UVs should be in valid range
        for uv in &mesh.uvs {
            assert!(uv.x >= 0.0 && uv.x <= 1.0);
            assert!(uv.y >= 0.0 && uv.y <= 1.0);
        }
    }

    #[test]
    fn test_box_projection() {
        let mut mesh = Cuboid::default().apply();
        project_box(&mut mesh, &BoxConfig::default());

        assert_eq!(mesh.uvs.len(), mesh.positions.len());
    }

    #[test]
    fn test_box_per_face_projection() {
        let mesh = Cuboid::default().apply();
        let projected = project_box_per_face(&mesh, &BoxConfig::default());

        // Each triangle gets 3 unique vertices
        assert_eq!(projected.vertex_count(), mesh.triangle_count() * 3);
        assert_eq!(projected.uvs.len(), projected.positions.len());
    }

    #[test]
    fn test_scale_uvs() {
        let mut mesh = Cuboid::default().apply();
        project_planar(&mut mesh, Mat4::IDENTITY, Vec2::ONE);

        let original_uvs = mesh.uvs.clone();
        scale_uvs(&mut mesh, Vec2::new(2.0, 0.5));

        for (orig, scaled) in original_uvs.iter().zip(mesh.uvs.iter()) {
            assert!((scaled.x - orig.x * 2.0).abs() < 0.001);
            assert!((scaled.y - orig.y * 0.5).abs() < 0.001);
        }
    }

    #[test]
    fn test_translate_uvs() {
        let mut mesh = Cuboid::default().apply();
        project_planar(&mut mesh, Mat4::IDENTITY, Vec2::ONE);

        let original_uvs = mesh.uvs.clone();
        translate_uvs(&mut mesh, Vec2::new(0.5, -0.25));

        for (orig, translated) in original_uvs.iter().zip(mesh.uvs.iter()) {
            assert!((translated.x - (orig.x + 0.5)).abs() < 0.001);
            assert!((translated.y - (orig.y - 0.25)).abs() < 0.001);
        }
    }

    #[test]
    fn test_normalize_uvs() {
        let mut mesh = Cuboid::default().apply();
        project_planar(&mut mesh, Mat4::IDENTITY, Vec2::ONE);

        normalize_uvs(&mut mesh);

        // All UVs should be in [0, 1] range
        for uv in &mesh.uvs {
            assert!(uv.x >= -0.001 && uv.x <= 1.001);
            assert!(uv.y >= -0.001 && uv.y <= 1.001);
        }
    }

    #[test]
    fn test_rotate_uvs() {
        let mut mesh = Cuboid::default().apply();
        project_planar(&mut mesh, Mat4::IDENTITY, Vec2::ONE);
        normalize_uvs(&mut mesh);

        // Rotate 90 degrees around center
        rotate_uvs(&mut mesh, std::f32::consts::FRAC_PI_2, Vec2::splat(0.5));

        // UVs should still exist
        assert_eq!(mesh.uvs.len(), mesh.positions.len());
    }

    #[test]
    fn test_find_uv_islands() {
        let mut mesh = Cuboid::default().apply();
        project_box(&mut mesh, &BoxConfig::default());

        let islands = find_uv_islands(&mesh);

        // Box mesh should have at least one island
        assert!(!islands.is_empty());

        // Total vertices in islands should match mesh
        let total_vertices: usize = islands.iter().map(|i| i.vertex_indices.len()).sum();
        assert_eq!(total_vertices, mesh.positions.len());
    }

    #[test]
    fn test_uv_chart() {
        let uvs = vec![
            Vec2::new(0.0, 0.0),
            Vec2::new(1.0, 0.5),
            Vec2::new(0.5, 1.0),
        ];
        let chart = UvChart::new(vec![0, 1, 2], uvs);

        assert_eq!(chart.min, Vec2::new(0.0, 0.0));
        assert_eq!(chart.max, Vec2::new(1.0, 1.0));
        assert!((chart.width() - 1.0).abs() < 0.001);
        assert!((chart.height() - 1.0).abs() < 0.001);
        assert!((chart.area() - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_pack_uv_charts() {
        // Create some test charts
        let charts = vec![
            UvChart::new(
                vec![0, 1, 2],
                vec![Vec2::ZERO, Vec2::new(0.3, 0.0), Vec2::new(0.15, 0.2)],
            ),
            UvChart::new(
                vec![3, 4, 5],
                vec![Vec2::ZERO, Vec2::new(0.2, 0.0), Vec2::new(0.1, 0.25)],
            ),
            UvChart::new(
                vec![6, 7, 8],
                vec![Vec2::ZERO, Vec2::new(0.4, 0.0), Vec2::new(0.2, 0.3)],
            ),
        ];

        let config = AtlasPackConfig::default();
        let result = pack_uv_charts(&charts, &config);

        // All charts should be packed
        assert_eq!(result.charts.len(), 3);

        // Atlas should have positive dimensions
        assert!(result.width > 0.0);
        assert!(result.height > 0.0);

        // Efficiency should be reasonable (> 0)
        assert!(result.efficiency > 0.0);
    }

    #[test]
    fn test_pack_mesh_uvs() {
        let mut mesh = Cuboid::default().apply();
        project_box(&mut mesh, &BoxConfig::default());

        let config = AtlasPackConfig::default();
        pack_mesh_uvs(&mut mesh, &config);

        // UVs should still exist
        assert_eq!(mesh.uvs.len(), mesh.positions.len());

        // All UVs should be in [0, 1] range after packing
        for uv in &mesh.uvs {
            assert!(uv.x >= -0.01 && uv.x <= 1.01, "UV x out of range: {}", uv.x);
            assert!(uv.y >= -0.01 && uv.y <= 1.01, "UV y out of range: {}", uv.y);
        }
    }

    #[test]
    fn test_atlas_pack_config() {
        let config = AtlasPackConfig {
            padding: 0.02,
            allow_rotation: false,
            target_aspect: 2.0,
        };

        let charts = vec![UvChart::new(
            vec![0, 1, 2],
            vec![Vec2::ZERO, Vec2::new(0.5, 0.0), Vec2::new(0.25, 0.5)],
        )];

        let result = pack_uv_charts(&charts, &config);
        assert_eq!(result.charts.len(), 1);
        // With no rotation allowed, the chart should not be rotated
        assert!(!result.charts[0].rotated);
    }

    #[test]
    fn test_empty_atlas_pack() {
        let charts: Vec<UvChart> = vec![];
        let result = pack_uv_charts(&charts, &AtlasPackConfig::default());

        assert!(result.charts.is_empty());
        assert_eq!(result.width, 0.0);
        assert_eq!(result.height, 0.0);
    }
}
