//! Lattice deformation (Free-Form Deformation).
//!
//! Implements FFD (Free-Form Deformation) using a control lattice.
//! Points are deformed by manipulating control points in a 3D grid.

use glam::Vec3;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::Mesh;

/// A 3D control lattice for free-form deformation.
///
/// The lattice defines a grid of control points that can be manipulated
/// to deform geometry. Points inside the lattice bounds are deformed
/// using trilinear interpolation.
#[derive(Debug, Clone)]
pub struct Lattice {
    /// Control point positions.
    /// Indexed as [z * (res_y * res_x) + y * res_x + x].
    pub control_points: Vec<Vec3>,
    /// Resolution in X direction.
    pub res_x: usize,
    /// Resolution in Y direction.
    pub res_y: usize,
    /// Resolution in Z direction.
    pub res_z: usize,
    /// Original bounds minimum.
    pub bounds_min: Vec3,
    /// Original bounds maximum.
    pub bounds_max: Vec3,
}

impl Lattice {
    /// Creates a new lattice with uniform resolution.
    ///
    /// # Arguments
    /// * `bounds_min` - Minimum corner of the lattice bounds
    /// * `bounds_max` - Maximum corner of the lattice bounds
    /// * `resolution` - Number of control points in each axis (must be >= 2)
    pub fn new(bounds_min: Vec3, bounds_max: Vec3, resolution: usize) -> Self {
        Self::with_resolution(bounds_min, bounds_max, resolution, resolution, resolution)
    }

    /// Creates a new lattice with per-axis resolution.
    ///
    /// # Arguments
    /// * `bounds_min` - Minimum corner of the lattice bounds
    /// * `bounds_max` - Maximum corner of the lattice bounds
    /// * `res_x`, `res_y`, `res_z` - Resolution in each axis (must be >= 2)
    pub fn with_resolution(
        bounds_min: Vec3,
        bounds_max: Vec3,
        res_x: usize,
        res_y: usize,
        res_z: usize,
    ) -> Self {
        let res_x = res_x.max(2);
        let res_y = res_y.max(2);
        let res_z = res_z.max(2);

        let mut control_points = Vec::with_capacity(res_x * res_y * res_z);

        for iz in 0..res_z {
            for iy in 0..res_y {
                for ix in 0..res_x {
                    let t = Vec3::new(
                        ix as f32 / (res_x - 1) as f32,
                        iy as f32 / (res_y - 1) as f32,
                        iz as f32 / (res_z - 1) as f32,
                    );
                    let pos = bounds_min + (bounds_max - bounds_min) * t;
                    control_points.push(pos);
                }
            }
        }

        Self {
            control_points,
            res_x,
            res_y,
            res_z,
            bounds_min,
            bounds_max,
        }
    }

    /// Creates a lattice from a mesh's bounding box.
    pub fn from_mesh(mesh: &Mesh, resolution: usize) -> Self {
        let (min, max) = mesh_bounds(mesh);
        // Add small padding to avoid edge issues
        let padding = (max - min) * 0.01;
        Self::new(min - padding, max + padding, resolution)
    }

    /// Gets the index into control_points for grid coordinates.
    fn index(&self, ix: usize, iy: usize, iz: usize) -> usize {
        iz * (self.res_y * self.res_x) + iy * self.res_x + ix
    }

    /// Gets a control point by grid coordinates.
    pub fn get(&self, ix: usize, iy: usize, iz: usize) -> Vec3 {
        self.control_points[self.index(ix, iy, iz)]
    }

    /// Sets a control point by grid coordinates.
    pub fn set(&mut self, ix: usize, iy: usize, iz: usize, pos: Vec3) {
        let idx = self.index(ix, iy, iz);
        self.control_points[idx] = pos;
    }

    /// Moves a control point by an offset.
    pub fn translate(&mut self, ix: usize, iy: usize, iz: usize, offset: Vec3) {
        let idx = self.index(ix, iy, iz);
        self.control_points[idx] += offset;
    }

    /// Converts a world position to normalized lattice coordinates (0-1).
    pub fn world_to_lattice(&self, pos: Vec3) -> Vec3 {
        let size = self.bounds_max - self.bounds_min;
        let t = (pos - self.bounds_min) / size;
        // Clamp to valid range
        Vec3::new(
            t.x.clamp(0.0, 1.0),
            t.y.clamp(0.0, 1.0),
            t.z.clamp(0.0, 1.0),
        )
    }

    /// Evaluates the deformed position using trilinear interpolation.
    ///
    /// Points outside the lattice bounds are clamped to the boundary.
    pub fn deform_point(&self, pos: Vec3) -> Vec3 {
        let t = self.world_to_lattice(pos);

        // Convert to cell coordinates
        let fx = t.x * (self.res_x - 1) as f32;
        let fy = t.y * (self.res_y - 1) as f32;
        let fz = t.z * (self.res_z - 1) as f32;

        // Get cell indices
        let ix = (fx as usize).min(self.res_x - 2);
        let iy = (fy as usize).min(self.res_y - 2);
        let iz = (fz as usize).min(self.res_z - 2);

        // Fractional position within cell
        let u = fx - ix as f32;
        let v = fy - iy as f32;
        let w = fz - iz as f32;

        // Get the 8 corners of the cell
        let c000 = self.get(ix, iy, iz);
        let c100 = self.get(ix + 1, iy, iz);
        let c010 = self.get(ix, iy + 1, iz);
        let c110 = self.get(ix + 1, iy + 1, iz);
        let c001 = self.get(ix, iy, iz + 1);
        let c101 = self.get(ix + 1, iy, iz + 1);
        let c011 = self.get(ix, iy + 1, iz + 1);
        let c111 = self.get(ix + 1, iy + 1, iz + 1);

        // Trilinear interpolation
        let c00 = c000.lerp(c100, u);
        let c10 = c010.lerp(c110, u);
        let c01 = c001.lerp(c101, u);
        let c11 = c011.lerp(c111, u);

        let c0 = c00.lerp(c10, v);
        let c1 = c01.lerp(c11, v);

        c0.lerp(c1, w)
    }

    /// Resets all control points to their original positions.
    pub fn reset(&mut self) {
        for iz in 0..self.res_z {
            for iy in 0..self.res_y {
                for ix in 0..self.res_x {
                    let t = Vec3::new(
                        ix as f32 / (self.res_x - 1) as f32,
                        iy as f32 / (self.res_y - 1) as f32,
                        iz as f32 / (self.res_z - 1) as f32,
                    );
                    let pos = self.bounds_min + (self.bounds_max - self.bounds_min) * t;
                    self.set(ix, iy, iz, pos);
                }
            }
        }
    }

    /// Returns the total number of control points.
    pub fn control_point_count(&self) -> usize {
        self.res_x * self.res_y * self.res_z
    }
}

/// Configuration for lattice deformation.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct LatticeDeformConfig {
    /// Whether to update normals after deformation.
    pub update_normals: bool,
}

impl Default for LatticeDeformConfig {
    fn default() -> Self {
        Self {
            update_normals: true,
        }
    }
}

/// Applies lattice deformation to a mesh.
///
/// Each vertex position is deformed according to the lattice control points.
/// Normals are optionally recalculated after deformation.
pub fn lattice_deform(mesh: &mut Mesh, lattice: &Lattice) {
    lattice_deform_with_config(mesh, lattice, &LatticeDeformConfig::default());
}

/// Applies lattice deformation with configuration.
pub fn lattice_deform_with_config(
    mesh: &mut Mesh,
    lattice: &Lattice,
    config: &LatticeDeformConfig,
) {
    for pos in &mut mesh.positions {
        *pos = lattice.deform_point(*pos);
    }

    if config.update_normals && mesh.has_normals() {
        mesh.compute_smooth_normals();
    }
}

/// Deforms a single point using a lattice.
pub fn lattice_deform_point(point: Vec3, lattice: &Lattice) -> Vec3 {
    lattice.deform_point(point)
}

/// Deforms a set of points using a lattice.
pub fn lattice_deform_points(points: &mut [Vec3], lattice: &Lattice) {
    for p in points {
        *p = lattice.deform_point(*p);
    }
}

// Lattice manipulation helpers

/// Bends the lattice around an axis.
///
/// # Arguments
/// * `lattice` - The lattice to bend
/// * `axis` - The axis to bend around (0=X, 1=Y, 2=Z)
/// * `angle` - Bend angle in radians
/// * `center` - Center of the bend (0-1 in axis direction)
pub fn bend_lattice(lattice: &mut Lattice, axis: usize, angle: f32, center: f32) {
    let size = lattice.bounds_max - lattice.bounds_min;

    for iz in 0..lattice.res_z {
        for iy in 0..lattice.res_y {
            for ix in 0..lattice.res_x {
                let pos = lattice.get(ix, iy, iz);

                // Get position relative to bounds (0-1)
                let t = (pos - lattice.bounds_min) / size;

                // Calculate bend based on axis
                let (bend_t, radius_axis, out_axis) = match axis {
                    0 => (t.x, 1, 2), // Bend around X, Y is radius, Z is out
                    1 => (t.y, 0, 2), // Bend around Y, X is radius, Z is out
                    _ => (t.z, 0, 1), // Bend around Z, X is radius, Y is out
                };

                let bend_amount = (bend_t - center) * angle;

                // Apply rotation
                let radius_t = [t.x, t.y, t.z][radius_axis];
                let out_t = [t.x, t.y, t.z][out_axis];

                let radius = (radius_t - 0.5) * size[radius_axis as usize];
                let new_radius = radius * bend_amount.cos()
                    - out_t * size[out_axis as usize] * bend_amount.sin();
                let new_out = radius * bend_amount.sin()
                    + out_t * size[out_axis as usize] * bend_amount.cos();

                let mut new_pos = pos;
                new_pos[radius_axis as usize] = lattice.bounds_min[radius_axis as usize]
                    + size[radius_axis as usize] * 0.5
                    + new_radius;
                new_pos[out_axis as usize] = lattice.bounds_min[out_axis as usize] + new_out;

                lattice.set(ix, iy, iz, new_pos);
            }
        }
    }
}

/// Twists the lattice around an axis.
///
/// # Arguments
/// * `lattice` - The lattice to twist
/// * `axis` - The axis to twist around (0=X, 1=Y, 2=Z)
/// * `angle` - Total twist angle in radians
pub fn twist_lattice(lattice: &mut Lattice, axis: usize, angle: f32) {
    let center = (lattice.bounds_min + lattice.bounds_max) * 0.5;

    for iz in 0..lattice.res_z {
        for iy in 0..lattice.res_y {
            for ix in 0..lattice.res_x {
                let pos = lattice.get(ix, iy, iz);

                // Get position relative to bounds (0-1)
                let t = match axis {
                    0 => {
                        (pos.x - lattice.bounds_min.x)
                            / (lattice.bounds_max.x - lattice.bounds_min.x)
                    }
                    1 => {
                        (pos.y - lattice.bounds_min.y)
                            / (lattice.bounds_max.y - lattice.bounds_min.y)
                    }
                    _ => {
                        (pos.z - lattice.bounds_min.z)
                            / (lattice.bounds_max.z - lattice.bounds_min.z)
                    }
                };

                let twist_angle = t * angle;
                let cos_a = twist_angle.cos();
                let sin_a = twist_angle.sin();

                // Rotate around axis
                let new_pos = match axis {
                    0 => {
                        // Twist around X
                        let y = pos.y - center.y;
                        let z = pos.z - center.z;
                        Vec3::new(
                            pos.x,
                            center.y + y * cos_a - z * sin_a,
                            center.z + y * sin_a + z * cos_a,
                        )
                    }
                    1 => {
                        // Twist around Y
                        let x = pos.x - center.x;
                        let z = pos.z - center.z;
                        Vec3::new(
                            center.x + x * cos_a + z * sin_a,
                            pos.y,
                            center.z - x * sin_a + z * cos_a,
                        )
                    }
                    _ => {
                        // Twist around Z
                        let x = pos.x - center.x;
                        let y = pos.y - center.y;
                        Vec3::new(
                            center.x + x * cos_a - y * sin_a,
                            center.y + x * sin_a + y * cos_a,
                            pos.z,
                        )
                    }
                };

                lattice.set(ix, iy, iz, new_pos);
            }
        }
    }
}

/// Tapers the lattice along an axis.
///
/// # Arguments
/// * `lattice` - The lattice to taper
/// * `axis` - The axis to taper along (0=X, 1=Y, 2=Z)
/// * `amount` - Taper amount (1.0 = no taper, 0.0 = point at end)
pub fn taper_lattice(lattice: &mut Lattice, axis: usize, amount: f32) {
    let center = (lattice.bounds_min + lattice.bounds_max) * 0.5;

    for iz in 0..lattice.res_z {
        for iy in 0..lattice.res_y {
            for ix in 0..lattice.res_x {
                let pos = lattice.get(ix, iy, iz);

                // Get position along axis (0-1)
                let t = match axis {
                    0 => {
                        (pos.x - lattice.bounds_min.x)
                            / (lattice.bounds_max.x - lattice.bounds_min.x)
                    }
                    1 => {
                        (pos.y - lattice.bounds_min.y)
                            / (lattice.bounds_max.y - lattice.bounds_min.y)
                    }
                    _ => {
                        (pos.z - lattice.bounds_min.z)
                            / (lattice.bounds_max.z - lattice.bounds_min.z)
                    }
                };

                // Calculate scale (linear interpolation from 1 to amount)
                let scale = 1.0 + (amount - 1.0) * t;

                // Scale perpendicular to axis
                let new_pos = match axis {
                    0 => Vec3::new(
                        pos.x,
                        center.y + (pos.y - center.y) * scale,
                        center.z + (pos.z - center.z) * scale,
                    ),
                    1 => Vec3::new(
                        center.x + (pos.x - center.x) * scale,
                        pos.y,
                        center.z + (pos.z - center.z) * scale,
                    ),
                    _ => Vec3::new(
                        center.x + (pos.x - center.x) * scale,
                        center.y + (pos.y - center.y) * scale,
                        pos.z,
                    ),
                };

                lattice.set(ix, iy, iz, new_pos);
            }
        }
    }
}

/// Scales the lattice non-uniformly.
pub fn scale_lattice(lattice: &mut Lattice, scale: Vec3) {
    let center = (lattice.bounds_min + lattice.bounds_max) * 0.5;

    for point in &mut lattice.control_points {
        *point = center + (*point - center) * scale;
    }
}

/// Computes the bounding box of a mesh.
fn mesh_bounds(mesh: &Mesh) -> (Vec3, Vec3) {
    if mesh.positions.is_empty() {
        return (Vec3::ZERO, Vec3::ZERO);
    }

    let mut min = mesh.positions[0];
    let mut max = mesh.positions[0];

    for pos in &mesh.positions {
        min = min.min(*pos);
        max = max.max(*pos);
    }

    (min, max)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lattice_new() {
        let lattice = Lattice::new(Vec3::ZERO, Vec3::ONE, 3);
        assert_eq!(lattice.res_x, 3);
        assert_eq!(lattice.res_y, 3);
        assert_eq!(lattice.res_z, 3);
        assert_eq!(lattice.control_point_count(), 27);
    }

    #[test]
    fn test_lattice_with_resolution() {
        let lattice = Lattice::with_resolution(Vec3::ZERO, Vec3::ONE, 2, 3, 4);
        assert_eq!(lattice.res_x, 2);
        assert_eq!(lattice.res_y, 3);
        assert_eq!(lattice.res_z, 4);
        assert_eq!(lattice.control_point_count(), 24);
    }

    #[test]
    fn test_lattice_minimum_resolution() {
        let lattice = Lattice::with_resolution(Vec3::ZERO, Vec3::ONE, 1, 1, 1);
        // Minimum resolution is 2
        assert_eq!(lattice.res_x, 2);
        assert_eq!(lattice.res_y, 2);
        assert_eq!(lattice.res_z, 2);
    }

    #[test]
    fn test_lattice_control_points_initial() {
        let lattice = Lattice::new(Vec3::ZERO, Vec3::ONE, 2);

        // Corner points should be at bounds
        assert!((lattice.get(0, 0, 0) - Vec3::ZERO).length() < 0.001);
        assert!((lattice.get(1, 1, 1) - Vec3::ONE).length() < 0.001);
    }

    #[test]
    fn test_lattice_world_to_lattice() {
        let lattice = Lattice::new(Vec3::ZERO, Vec3::new(10.0, 10.0, 10.0), 2);

        let t = lattice.world_to_lattice(Vec3::new(5.0, 5.0, 5.0));
        assert!((t - Vec3::splat(0.5)).length() < 0.001);

        let t = lattice.world_to_lattice(Vec3::ZERO);
        assert!((t - Vec3::ZERO).length() < 0.001);

        let t = lattice.world_to_lattice(Vec3::new(10.0, 10.0, 10.0));
        assert!((t - Vec3::ONE).length() < 0.001);
    }

    #[test]
    fn test_lattice_deform_identity() {
        // With unmodified control points, deformation should be identity
        let lattice = Lattice::new(Vec3::ZERO, Vec3::ONE, 3);

        let test_points = [
            Vec3::new(0.25, 0.25, 0.25),
            Vec3::new(0.5, 0.5, 0.5),
            Vec3::new(0.75, 0.75, 0.75),
        ];

        for point in &test_points {
            let deformed = lattice.deform_point(*point);
            assert!(
                (*point - deformed).length() < 0.001,
                "Point {:?} deformed to {:?}",
                point,
                deformed
            );
        }
    }

    #[test]
    fn test_lattice_deform_translation() {
        let mut lattice = Lattice::new(Vec3::ZERO, Vec3::ONE, 2);

        // Move all control points up
        let offset = Vec3::new(0.0, 1.0, 0.0);
        for point in &mut lattice.control_points {
            *point += offset;
        }

        let point = Vec3::new(0.5, 0.5, 0.5);
        let deformed = lattice.deform_point(point);

        assert!((deformed - (point + offset)).length() < 0.001);
    }

    #[test]
    fn test_lattice_deform_scale() {
        let mut lattice = Lattice::new(Vec3::ZERO, Vec3::ONE, 2);

        // Scale by 2x
        let center = Vec3::splat(0.5);
        for point in &mut lattice.control_points {
            *point = center + (*point - center) * 2.0;
        }

        let point = Vec3::new(0.75, 0.75, 0.75);
        let deformed = lattice.deform_point(point);

        let expected = center + (point - center) * 2.0;
        assert!((deformed - expected).length() < 0.001);
    }

    #[test]
    fn test_lattice_get_set() {
        let mut lattice = Lattice::new(Vec3::ZERO, Vec3::ONE, 3);

        let new_pos = Vec3::new(5.0, 5.0, 5.0);
        lattice.set(1, 1, 1, new_pos);

        assert_eq!(lattice.get(1, 1, 1), new_pos);
    }

    #[test]
    fn test_lattice_translate() {
        let mut lattice = Lattice::new(Vec3::ZERO, Vec3::ONE, 3);

        let original = lattice.get(1, 1, 1);
        let offset = Vec3::new(0.5, 0.5, 0.5);
        lattice.translate(1, 1, 1, offset);

        assert!((lattice.get(1, 1, 1) - (original + offset)).length() < 0.001);
    }

    #[test]
    fn test_lattice_reset() {
        let mut lattice = Lattice::new(Vec3::ZERO, Vec3::ONE, 3);

        // Modify some control points
        lattice.set(1, 1, 1, Vec3::new(10.0, 10.0, 10.0));

        lattice.reset();

        // Should be back to original
        let expected = Vec3::splat(0.5);
        assert!((lattice.get(1, 1, 1) - expected).length() < 0.001);
    }

    #[test]
    fn test_lattice_deform_mesh() {
        let mut mesh = crate::Cuboid::default().apply();

        let mut lattice = Lattice::from_mesh(&mesh, 3);

        // Move top control points up
        for ix in 0..lattice.res_x {
            for iz in 0..lattice.res_z {
                lattice.translate(ix, lattice.res_y - 1, iz, Vec3::new(0.0, 1.0, 0.0));
            }
        }

        let original_positions = mesh.positions.clone();

        lattice_deform(&mut mesh, &lattice);

        // Mesh should be modified
        assert_ne!(mesh.positions, original_positions);

        // Top vertices should be higher
        for (orig, new) in original_positions.iter().zip(mesh.positions.iter()) {
            if orig.y > 0.4 {
                // Top vertices
                assert!(new.y > orig.y);
            }
        }
    }

    #[test]
    fn test_twist_lattice() {
        let mut lattice = Lattice::new(Vec3::new(-1.0, 0.0, -1.0), Vec3::new(1.0, 2.0, 1.0), 3);

        let original_top = lattice.get(2, 2, 2);

        twist_lattice(&mut lattice, 1, std::f32::consts::FRAC_PI_2); // 90 degree twist

        let twisted_top = lattice.get(2, 2, 2);

        // Top corner should be rotated around Y axis
        assert!((original_top - twisted_top).length() > 0.5);
    }

    #[test]
    fn test_taper_lattice() {
        let mut lattice = Lattice::new(Vec3::new(-1.0, 0.0, -1.0), Vec3::new(1.0, 2.0, 1.0), 3);

        let original_top = lattice.get(2, 2, 2);

        taper_lattice(&mut lattice, 1, 0.5); // Taper to 50% at top

        let tapered_top = lattice.get(2, 2, 2);

        // Top corner should be closer to center (tapered)
        let center = Vec3::new(0.0, original_top.y, 0.0);
        let orig_dist = (original_top - center).length();
        let new_dist = (tapered_top - center).length();

        assert!(new_dist < orig_dist);
    }

    #[test]
    fn test_scale_lattice() {
        let mut lattice = Lattice::new(Vec3::new(-1.0, -1.0, -1.0), Vec3::new(1.0, 1.0, 1.0), 2);

        scale_lattice(&mut lattice, Vec3::new(2.0, 1.0, 1.0));

        // X extent should be doubled
        let min = lattice.get(0, 0, 0);
        let max = lattice.get(1, 1, 1);

        assert!((max.x - min.x - 4.0).abs() < 0.001); // Was 2, now 4
        assert!((max.y - min.y - 2.0).abs() < 0.001); // Unchanged
    }

    #[test]
    fn test_lattice_deform_points() {
        let lattice = Lattice::new(Vec3::ZERO, Vec3::ONE, 2);

        let mut points = vec![
            Vec3::new(0.25, 0.25, 0.25),
            Vec3::new(0.5, 0.5, 0.5),
            Vec3::new(0.75, 0.75, 0.75),
        ];

        let original = points.clone();

        lattice_deform_points(&mut points, &lattice);

        // With identity lattice, points should be unchanged
        for (orig, new) in original.iter().zip(points.iter()) {
            assert!((*orig - *new).length() < 0.001);
        }
    }
}
