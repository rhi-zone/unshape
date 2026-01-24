//! Procedural architecture generation.
//!
//! Provides tools for generating building geometry procedurally,
//! including walls, floors, roofs, and windows.
//!
//! Operations are serializable structs with `apply` methods.
//! See `docs/design/ops-as-values.md`.

use crate::Mesh;
use glam::Vec3;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Configuration for wall generation.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct WallConfig {
    /// Wall height.
    pub height: f32,
    /// Wall thickness.
    pub thickness: f32,
    /// Whether to generate inner faces.
    pub double_sided: bool,
}

impl Default for WallConfig {
    fn default() -> Self {
        Self {
            height: 3.0,
            thickness: 0.2,
            double_sided: true,
        }
    }
}

/// Configuration for window placement.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct WindowConfig {
    /// Window width.
    pub width: f32,
    /// Window height.
    pub height: f32,
    /// Distance from floor to window bottom.
    pub sill_height: f32,
    /// Frame thickness.
    pub frame_width: f32,
}

impl Default for WindowConfig {
    fn default() -> Self {
        Self {
            width: 1.2,
            height: 1.5,
            sill_height: 0.9,
            frame_width: 0.1,
        }
    }
}

/// Configuration for door placement.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct DoorConfig {
    /// Door width.
    pub width: f32,
    /// Door height.
    pub height: f32,
    /// Frame thickness.
    pub frame_width: f32,
}

impl Default for DoorConfig {
    fn default() -> Self {
        Self {
            width: 0.9,
            height: 2.1,
            frame_width: 0.1,
        }
    }
}

/// Configuration for roof generation.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct RoofConfig {
    /// Roof style.
    pub style: RoofStyle,
    /// Overhang distance.
    pub overhang: f32,
    /// Roof thickness.
    pub thickness: f32,
    /// Pitch angle in degrees (for gabled/hipped roofs).
    pub pitch: f32,
}

impl Default for RoofConfig {
    fn default() -> Self {
        Self {
            style: RoofStyle::Gabled,
            overhang: 0.3,
            thickness: 0.15,
            pitch: 30.0,
        }
    }
}

/// Roof style variants.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum RoofStyle {
    /// Flat roof.
    Flat,
    /// Gabled (two sloping sides).
    Gabled,
    /// Hipped (four sloping sides).
    Hipped,
    /// Shed (single slope).
    Shed,
}

/// A floor plan point.
#[derive(Debug, Clone, Copy)]
pub struct FloorPoint {
    /// X coordinate.
    pub x: f32,
    /// Z coordinate (depth).
    pub z: f32,
}

impl FloorPoint {
    /// Creates a new floor point.
    pub fn new(x: f32, z: f32) -> Self {
        Self { x, z }
    }
}

/// A building definition.
#[derive(Debug, Clone)]
pub struct Building {
    /// Floor plan polygon (clockwise, XZ plane).
    pub floor_plan: Vec<FloorPoint>,
    /// Number of floors.
    pub floor_count: u32,
    /// Height per floor.
    pub floor_height: f32,
    /// Wall configuration.
    pub wall_config: WallConfig,
    /// Roof configuration.
    pub roof_config: RoofConfig,
}

impl Building {
    /// Creates a new building with the given floor plan.
    pub fn new(floor_plan: Vec<FloorPoint>) -> Self {
        Self {
            floor_plan,
            floor_count: 1,
            floor_height: 3.0,
            wall_config: WallConfig::default(),
            roof_config: RoofConfig::default(),
        }
    }

    /// Creates a rectangular building.
    pub fn rectangle(width: f32, depth: f32) -> Self {
        let hw = width / 2.0;
        let hd = depth / 2.0;
        Self::new(vec![
            FloorPoint::new(-hw, -hd),
            FloorPoint::new(hw, -hd),
            FloorPoint::new(hw, hd),
            FloorPoint::new(-hw, hd),
        ])
    }

    /// Creates an L-shaped building.
    pub fn l_shape(width: f32, depth: f32, cut_width: f32, cut_depth: f32) -> Self {
        let hw = width / 2.0;
        let hd = depth / 2.0;
        Self::new(vec![
            FloorPoint::new(-hw, -hd),
            FloorPoint::new(hw, -hd),
            FloorPoint::new(hw, hd - cut_depth),
            FloorPoint::new(hw - cut_width, hd - cut_depth),
            FloorPoint::new(hw - cut_width, hd),
            FloorPoint::new(-hw, hd),
        ])
    }
}

/// Generates a building mesh.
pub fn generate_building(building: &Building) -> Mesh {
    let mut mesh = Mesh::new();

    let total_height = building.floor_count as f32 * building.floor_height;

    // Generate walls
    let wall_mesh = generate_walls(&building.floor_plan, total_height, &building.wall_config);
    merge_mesh(&mut mesh, &wall_mesh);

    // Generate floor slabs
    for floor in 0..=building.floor_count {
        let y = floor as f32 * building.floor_height;
        let floor_mesh =
            generate_floor_slab(&building.floor_plan, y, building.wall_config.thickness);
        merge_mesh(&mut mesh, &floor_mesh);
    }

    // Generate roof
    let roof_mesh = generate_roof(&building.floor_plan, total_height, &building.roof_config);
    merge_mesh(&mut mesh, &roof_mesh);

    mesh
}

/// Generates walls from a floor plan.
pub fn generate_walls(floor_plan: &[FloorPoint], height: f32, config: &WallConfig) -> Mesh {
    let mut mesh = Mesh::new();

    let n = floor_plan.len();
    if n < 3 {
        return mesh;
    }

    for i in 0..n {
        let p0 = floor_plan[i];
        let p1 = floor_plan[(i + 1) % n];

        let v0 = Vec3::new(p0.x, 0.0, p0.z);
        let v1 = Vec3::new(p1.x, 0.0, p1.z);
        let v2 = Vec3::new(p1.x, height, p1.z);
        let v3 = Vec3::new(p0.x, height, p0.z);

        // Outer wall face
        let base = mesh.positions.len() as u32;
        mesh.positions.extend_from_slice(&[v0, v1, v2, v3]);
        mesh.indices.extend_from_slice(&[base, base + 1, base + 2]);
        mesh.indices.extend_from_slice(&[base, base + 2, base + 3]);

        if config.double_sided {
            // Inner wall face (offset inward)
            let dir = (v1 - v0).normalize();
            let normal = Vec3::new(-dir.z, 0.0, dir.x);
            let offset = normal * config.thickness;

            let iv0 = v0 + offset;
            let iv1 = v1 + offset;
            let iv2 = v2 + offset;
            let iv3 = v3 + offset;

            let base = mesh.positions.len() as u32;
            mesh.positions.extend_from_slice(&[iv0, iv1, iv2, iv3]);
            mesh.indices.extend_from_slice(&[base + 2, base + 1, base]);
            mesh.indices.extend_from_slice(&[base + 3, base + 2, base]);
        }
    }

    mesh
}

/// Generates a floor slab.
fn generate_floor_slab(floor_plan: &[FloorPoint], y: f32, thickness: f32) -> Mesh {
    let mut mesh = Mesh::new();

    let n = floor_plan.len();
    if n < 3 {
        return mesh;
    }

    // Simple fan triangulation for convex polygons
    let base = mesh.positions.len() as u32;

    // Top surface
    for p in floor_plan {
        mesh.positions.push(Vec3::new(p.x, y, p.z));
    }

    for i in 1..n - 1 {
        mesh.indices
            .extend_from_slice(&[base, base + i as u32, base + (i + 1) as u32]);
    }

    // Bottom surface
    let base2 = mesh.positions.len() as u32;
    for p in floor_plan {
        mesh.positions.push(Vec3::new(p.x, y - thickness, p.z));
    }

    for i in 1..n - 1 {
        mesh.indices
            .extend_from_slice(&[base2 + (i + 1) as u32, base2 + i as u32, base2]);
    }

    mesh
}

/// Generates a roof.
fn generate_roof(floor_plan: &[FloorPoint], base_height: f32, config: &RoofConfig) -> Mesh {
    match config.style {
        RoofStyle::Flat => generate_flat_roof(floor_plan, base_height, config),
        RoofStyle::Gabled => generate_gabled_roof(floor_plan, base_height, config),
        RoofStyle::Hipped => generate_hipped_roof(floor_plan, base_height, config),
        RoofStyle::Shed => generate_shed_roof(floor_plan, base_height, config),
    }
}

/// Generates a flat roof.
fn generate_flat_roof(floor_plan: &[FloorPoint], base_height: f32, config: &RoofConfig) -> Mesh {
    let mut mesh = Mesh::new();

    let n = floor_plan.len();
    if n < 3 {
        return mesh;
    }

    // Expand floor plan by overhang
    let expanded = expand_polygon(floor_plan, config.overhang);

    let base = mesh.positions.len() as u32;
    let y = base_height + config.thickness;

    for p in &expanded {
        mesh.positions.push(Vec3::new(p.x, y, p.z));
    }

    for i in 1..n - 1 {
        mesh.indices
            .extend_from_slice(&[base, base + i as u32, base + (i + 1) as u32]);
    }

    mesh
}

/// Generates a gabled roof.
fn generate_gabled_roof(floor_plan: &[FloorPoint], base_height: f32, config: &RoofConfig) -> Mesh {
    let mut mesh = Mesh::new();

    // Get bounding box
    let (min_x, max_x, min_z, max_z) = bounding_box(floor_plan);

    let width = max_x - min_x;
    let center_x = (min_x + max_x) / 2.0;

    // Ridge height based on pitch
    let pitch_rad = config.pitch.to_radians();
    let ridge_height = (width / 2.0) * pitch_rad.tan();

    let y_base = base_height;
    let y_ridge = base_height + ridge_height;

    let overhang = config.overhang;

    // Front slope
    let base = mesh.positions.len() as u32;
    mesh.positions.extend_from_slice(&[
        Vec3::new(min_x - overhang, y_base, min_z - overhang),
        Vec3::new(max_x + overhang, y_base, min_z - overhang),
        Vec3::new(center_x, y_ridge, min_z - overhang),
        Vec3::new(center_x, y_ridge, max_z + overhang),
        Vec3::new(min_x - overhang, y_base, max_z + overhang),
        Vec3::new(max_x + overhang, y_base, max_z + overhang),
    ]);

    // Left slope
    mesh.indices.extend_from_slice(&[base, base + 2, base + 3]);
    mesh.indices.extend_from_slice(&[base, base + 3, base + 4]);

    // Right slope
    mesh.indices
        .extend_from_slice(&[base + 1, base + 5, base + 3]);
    mesh.indices
        .extend_from_slice(&[base + 1, base + 3, base + 2]);

    // Front gable
    mesh.indices.extend_from_slice(&[base, base + 1, base + 2]);

    // Back gable
    mesh.indices
        .extend_from_slice(&[base + 5, base + 4, base + 3]);

    mesh
}

/// Generates a hipped roof.
fn generate_hipped_roof(floor_plan: &[FloorPoint], base_height: f32, config: &RoofConfig) -> Mesh {
    let mut mesh = Mesh::new();

    let (min_x, max_x, min_z, max_z) = bounding_box(floor_plan);

    let width = max_x - min_x;
    let depth = max_z - min_z;
    let center_x = (min_x + max_x) / 2.0;
    let center_z = (min_z + max_z) / 2.0;

    let pitch_rad = config.pitch.to_radians();
    let ridge_height = (width.min(depth) / 2.0) * pitch_rad.tan();

    let y_base = base_height;
    let y_peak = base_height + ridge_height;

    let overhang = config.overhang;

    // Four corners and center
    let base = mesh.positions.len() as u32;
    mesh.positions.extend_from_slice(&[
        Vec3::new(min_x - overhang, y_base, min_z - overhang), // 0: front-left
        Vec3::new(max_x + overhang, y_base, min_z - overhang), // 1: front-right
        Vec3::new(max_x + overhang, y_base, max_z + overhang), // 2: back-right
        Vec3::new(min_x - overhang, y_base, max_z + overhang), // 3: back-left
        Vec3::new(center_x, y_peak, center_z),                 // 4: peak
    ]);

    // Four triangular faces
    mesh.indices.extend_from_slice(&[base, base + 1, base + 4]); // front
    mesh.indices
        .extend_from_slice(&[base + 1, base + 2, base + 4]); // right
    mesh.indices
        .extend_from_slice(&[base + 2, base + 3, base + 4]); // back
    mesh.indices.extend_from_slice(&[base + 3, base, base + 4]); // left

    mesh
}

/// Generates a shed roof.
fn generate_shed_roof(floor_plan: &[FloorPoint], base_height: f32, config: &RoofConfig) -> Mesh {
    let mut mesh = Mesh::new();

    let (min_x, max_x, min_z, max_z) = bounding_box(floor_plan);

    let width = max_x - min_x;
    let pitch_rad = config.pitch.to_radians();
    let rise = width * pitch_rad.tan();

    let y_low = base_height;
    let y_high = base_height + rise;

    let overhang = config.overhang;

    let base = mesh.positions.len() as u32;
    mesh.positions.extend_from_slice(&[
        Vec3::new(min_x - overhang, y_low, min_z - overhang),
        Vec3::new(max_x + overhang, y_high, min_z - overhang),
        Vec3::new(max_x + overhang, y_high, max_z + overhang),
        Vec3::new(min_x - overhang, y_low, max_z + overhang),
    ]);

    mesh.indices.extend_from_slice(&[base, base + 1, base + 2]);
    mesh.indices.extend_from_slice(&[base, base + 2, base + 3]);

    mesh
}

/// Generates a wall with a window opening.
pub fn generate_wall_with_window(
    start: Vec3,
    end: Vec3,
    height: f32,
    window: &WindowConfig,
    window_pos: f32, // 0-1 along wall
) -> Mesh {
    let mut mesh = Mesh::new();

    let dir = end - start;
    let wall_length = dir.length();
    let dir_norm = dir / wall_length;

    // Window position along wall
    let window_center = wall_length * window_pos;
    let half_w = window.width / 2.0;

    let left = window_center - half_w;
    let right = window_center + half_w;
    let bottom = window.sill_height;
    let top = window.sill_height + window.height;

    // Wall sections (4 quads around window)
    let sections = [
        // Below window
        (0.0, left, 0.0, bottom),
        (left, right, 0.0, bottom),
        (right, wall_length, 0.0, bottom),
        // Left of window
        (0.0, left, bottom, top),
        // Right of window
        (right, wall_length, bottom, top),
        // Above window
        (0.0, left, top, height),
        (left, right, top, height),
        (right, wall_length, top, height),
    ];

    for (x0, x1, y0, y1) in sections {
        if x1 <= x0 || y1 <= y0 {
            continue;
        }

        let base = mesh.positions.len() as u32;
        mesh.positions.extend_from_slice(&[
            start + dir_norm * x0 + Vec3::Y * y0,
            start + dir_norm * x1 + Vec3::Y * y0,
            start + dir_norm * x1 + Vec3::Y * y1,
            start + dir_norm * x0 + Vec3::Y * y1,
        ]);
        mesh.indices.extend_from_slice(&[base, base + 1, base + 2]);
        mesh.indices.extend_from_slice(&[base, base + 2, base + 3]);
    }

    mesh
}

/// Generates a wall with a door opening.
pub fn generate_wall_with_door(
    start: Vec3,
    end: Vec3,
    height: f32,
    door: &DoorConfig,
    door_pos: f32, // 0-1 along wall
) -> Mesh {
    let mut mesh = Mesh::new();

    let dir = end - start;
    let wall_length = dir.length();
    let dir_norm = dir / wall_length;

    let door_center = wall_length * door_pos;
    let half_w = door.width / 2.0;

    let left = door_center - half_w;
    let right = door_center + half_w;
    let top = door.height;

    // Wall sections (3 quads around door)
    let sections = [
        // Left of door
        (0.0, left, 0.0, height),
        // Above door
        (left, right, top, height),
        // Right of door
        (right, wall_length, 0.0, height),
    ];

    for (x0, x1, y0, y1) in sections {
        if x1 <= x0 || y1 <= y0 {
            continue;
        }

        let base = mesh.positions.len() as u32;
        mesh.positions.extend_from_slice(&[
            start + dir_norm * x0 + Vec3::Y * y0,
            start + dir_norm * x1 + Vec3::Y * y0,
            start + dir_norm * x1 + Vec3::Y * y1,
            start + dir_norm * x0 + Vec3::Y * y1,
        ]);
        mesh.indices.extend_from_slice(&[base, base + 1, base + 2]);
        mesh.indices.extend_from_slice(&[base, base + 2, base + 3]);
    }

    mesh
}

/// Generates stairs.
pub fn generate_stairs(
    start: Vec3,
    direction: Vec3,
    total_rise: f32,
    total_run: f32,
    width: f32,
    step_count: u32,
) -> Mesh {
    let mut mesh = Mesh::new();

    if step_count == 0 {
        return mesh;
    }

    let dir_norm = direction.normalize();
    let right = dir_norm.cross(Vec3::Y).normalize() * (width / 2.0);

    let rise_per_step = total_rise / step_count as f32;
    let run_per_step = total_run / step_count as f32;

    for i in 0..step_count {
        let y = i as f32 * rise_per_step;
        let z = i as f32 * run_per_step;
        let next_y = (i + 1) as f32 * rise_per_step;

        let base = mesh.positions.len() as u32;

        // Tread (horizontal)
        let tread_front = start + dir_norm * z + Vec3::Y * next_y;
        let tread_back = start + dir_norm * (z + run_per_step) + Vec3::Y * next_y;

        mesh.positions.extend_from_slice(&[
            tread_front - right,
            tread_front + right,
            tread_back + right,
            tread_back - right,
        ]);
        mesh.indices.extend_from_slice(&[base, base + 1, base + 2]);
        mesh.indices.extend_from_slice(&[base, base + 2, base + 3]);

        // Riser (vertical)
        let base = mesh.positions.len() as u32;
        let riser_bottom = start + dir_norm * z + Vec3::Y * y;
        let riser_top = start + dir_norm * z + Vec3::Y * next_y;

        mesh.positions.extend_from_slice(&[
            riser_bottom - right,
            riser_bottom + right,
            riser_top + right,
            riser_top - right,
        ]);
        mesh.indices.extend_from_slice(&[base, base + 1, base + 2]);
        mesh.indices.extend_from_slice(&[base, base + 2, base + 3]);
    }

    mesh
}

/// Merges a source mesh into a destination mesh.
fn merge_mesh(dest: &mut Mesh, src: &Mesh) {
    let offset = dest.positions.len() as u32;
    dest.positions.extend_from_slice(&src.positions);
    dest.indices.extend(src.indices.iter().map(|i| i + offset));
}

/// Expands a polygon by a distance.
fn expand_polygon(points: &[FloorPoint], distance: f32) -> Vec<FloorPoint> {
    let n = points.len();
    if n < 3 {
        return points.to_vec();
    }

    let mut result = Vec::with_capacity(n);

    for i in 0..n {
        let prev = &points[(i + n - 1) % n];
        let curr = &points[i];
        let next = &points[(i + 1) % n];

        // Edge directions
        let d1 = Vec3::new(curr.x - prev.x, 0.0, curr.z - prev.z).normalize();
        let d2 = Vec3::new(next.x - curr.x, 0.0, next.z - curr.z).normalize();

        // Outward normals
        let n1 = Vec3::new(-d1.z, 0.0, d1.x);
        let n2 = Vec3::new(-d2.z, 0.0, d2.x);

        // Average normal
        let avg = (n1 + n2).normalize_or_zero();
        let offset = if avg.length_squared() > 0.001 {
            avg * distance
        } else {
            n1 * distance
        };

        result.push(FloorPoint::new(curr.x + offset.x, curr.z + offset.z));
    }

    result
}

/// Computes bounding box of floor plan.
fn bounding_box(points: &[FloorPoint]) -> (f32, f32, f32, f32) {
    if points.is_empty() {
        return (0.0, 0.0, 0.0, 0.0);
    }

    let mut min_x = f32::MAX;
    let mut max_x = f32::MIN;
    let mut min_z = f32::MAX;
    let mut max_z = f32::MIN;

    for p in points {
        min_x = min_x.min(p.x);
        max_x = max_x.max(p.x);
        min_z = min_z.min(p.z);
        max_z = max_z.max(p.z);
    }

    (min_x, max_x, min_z, max_z)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wall_config_default() {
        let config = WallConfig::default();
        assert!(config.height > 0.0);
        assert!(config.thickness > 0.0);
    }

    #[test]
    fn test_window_config_default() {
        let config = WindowConfig::default();
        assert!(config.width > 0.0);
        assert!(config.height > 0.0);
    }

    #[test]
    fn test_door_config_default() {
        let config = DoorConfig::default();
        assert!(config.width > 0.0);
        assert!(config.height > 0.0);
    }

    #[test]
    fn test_roof_config_default() {
        let config = RoofConfig::default();
        assert!(config.pitch > 0.0);
        assert!(config.overhang >= 0.0);
    }

    #[test]
    fn test_floor_point() {
        let p = FloorPoint::new(1.0, 2.0);
        assert_eq!(p.x, 1.0);
        assert_eq!(p.z, 2.0);
    }

    #[test]
    fn test_building_rectangle() {
        let building = Building::rectangle(10.0, 8.0);
        assert_eq!(building.floor_plan.len(), 4);
    }

    #[test]
    fn test_building_l_shape() {
        let building = Building::l_shape(10.0, 8.0, 4.0, 3.0);
        assert_eq!(building.floor_plan.len(), 6);
    }

    #[test]
    fn test_building_with_floors() {
        let mut building = Building::rectangle(10.0, 8.0);
        building.floor_count = 3;
        assert_eq!(building.floor_count, 3);
    }

    #[test]
    fn test_generate_building() {
        let mut building = Building::rectangle(10.0, 8.0);
        building.floor_count = 2;
        let mesh = generate_building(&building);

        assert!(!mesh.positions.is_empty());
        assert!(!mesh.indices.is_empty());
        assert_eq!(mesh.indices.len() % 3, 0);
    }

    #[test]
    fn test_generate_walls() {
        let floor_plan = vec![
            FloorPoint::new(0.0, 0.0),
            FloorPoint::new(5.0, 0.0),
            FloorPoint::new(5.0, 5.0),
            FloorPoint::new(0.0, 5.0),
        ];
        let config = WallConfig::default();
        let mesh = generate_walls(&floor_plan, 3.0, &config);

        assert!(!mesh.positions.is_empty());
        assert!(!mesh.indices.is_empty());
    }

    #[test]
    fn test_generate_flat_roof() {
        let floor_plan = vec![
            FloorPoint::new(0.0, 0.0),
            FloorPoint::new(5.0, 0.0),
            FloorPoint::new(5.0, 5.0),
            FloorPoint::new(0.0, 5.0),
        ];
        let config = RoofConfig {
            style: RoofStyle::Flat,
            ..Default::default()
        };
        let mesh = generate_flat_roof(&floor_plan, 3.0, &config);

        assert!(!mesh.positions.is_empty());
    }

    #[test]
    fn test_generate_gabled_roof() {
        let floor_plan = vec![
            FloorPoint::new(0.0, 0.0),
            FloorPoint::new(5.0, 0.0),
            FloorPoint::new(5.0, 5.0),
            FloorPoint::new(0.0, 5.0),
        ];
        let config = RoofConfig {
            style: RoofStyle::Gabled,
            ..Default::default()
        };
        let mesh = generate_gabled_roof(&floor_plan, 3.0, &config);

        assert!(!mesh.positions.is_empty());
    }

    #[test]
    fn test_generate_hipped_roof() {
        let floor_plan = vec![
            FloorPoint::new(0.0, 0.0),
            FloorPoint::new(5.0, 0.0),
            FloorPoint::new(5.0, 5.0),
            FloorPoint::new(0.0, 5.0),
        ];
        let config = RoofConfig {
            style: RoofStyle::Hipped,
            ..Default::default()
        };
        let mesh = generate_hipped_roof(&floor_plan, 3.0, &config);

        assert!(!mesh.positions.is_empty());
    }

    #[test]
    fn test_generate_shed_roof() {
        let floor_plan = vec![
            FloorPoint::new(0.0, 0.0),
            FloorPoint::new(5.0, 0.0),
            FloorPoint::new(5.0, 5.0),
            FloorPoint::new(0.0, 5.0),
        ];
        let config = RoofConfig {
            style: RoofStyle::Shed,
            ..Default::default()
        };
        let mesh = generate_shed_roof(&floor_plan, 3.0, &config);

        assert!(!mesh.positions.is_empty());
    }

    #[test]
    fn test_generate_wall_with_window() {
        let start = Vec3::ZERO;
        let end = Vec3::new(5.0, 0.0, 0.0);
        let window = WindowConfig::default();

        let mesh = generate_wall_with_window(start, end, 3.0, &window, 0.5);

        assert!(!mesh.positions.is_empty());
        assert!(!mesh.indices.is_empty());
    }

    #[test]
    fn test_generate_wall_with_door() {
        let start = Vec3::ZERO;
        let end = Vec3::new(5.0, 0.0, 0.0);
        let door = DoorConfig::default();

        let mesh = generate_wall_with_door(start, end, 3.0, &door, 0.5);

        assert!(!mesh.positions.is_empty());
        assert!(!mesh.indices.is_empty());
    }

    #[test]
    fn test_generate_stairs() {
        let mesh = generate_stairs(
            Vec3::ZERO,
            Vec3::Z,
            2.5, // total rise
            4.0, // total run
            1.0, // width
            10,  // steps
        );

        assert!(!mesh.positions.is_empty());
        assert!(!mesh.indices.is_empty());
    }

    #[test]
    fn test_bounding_box() {
        let points = vec![
            FloorPoint::new(-2.0, -3.0),
            FloorPoint::new(4.0, 1.0),
            FloorPoint::new(1.0, 5.0),
        ];
        let (min_x, max_x, min_z, max_z) = bounding_box(&points);

        assert_eq!(min_x, -2.0);
        assert_eq!(max_x, 4.0);
        assert_eq!(min_z, -3.0);
        assert_eq!(max_z, 5.0);
    }

    #[test]
    fn test_expand_polygon() {
        // Clockwise winding (standard for building floor plans)
        let points = vec![
            FloorPoint::new(0.0, 0.0),
            FloorPoint::new(0.0, 1.0),
            FloorPoint::new(1.0, 1.0),
            FloorPoint::new(1.0, 0.0),
        ];
        let expanded = expand_polygon(&points, 0.5);

        assert_eq!(expanded.len(), 4);
        // Corners should be pushed outward
        assert!(expanded[0].x < 0.0);
        assert!(expanded[0].z < 0.0);
    }
}
