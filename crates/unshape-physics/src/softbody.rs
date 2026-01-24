//! Soft body simulation using Finite Element Method.
//!
//! Implements FEM-based soft body simulation with tetrahedral elements
//! for volumetric deformation.

use glam::{Mat3, Vec3};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Configuration for soft body simulation.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dynop", derive(unshape_op::Op))]
#[cfg_attr(feature = "dynop", op(input = (), output = SoftBodyConfig))]
pub struct SoftBodyConfig {
    /// Young's modulus (stiffness).
    pub youngs_modulus: f32,
    /// Poisson's ratio (0-0.5, incompressibility).
    pub poisson_ratio: f32,
    /// Mass density.
    pub density: f32,
    /// Global damping factor.
    pub damping: f32,
    /// Gravity vector.
    pub gravity: Vec3,
    /// Number of solver iterations.
    pub iterations: u32,
}

impl Default for SoftBodyConfig {
    fn default() -> Self {
        Self {
            youngs_modulus: 1000.0,
            poisson_ratio: 0.3,
            density: 1000.0,
            damping: 0.1,
            gravity: Vec3::new(0.0, -9.81, 0.0),
            iterations: 10,
        }
    }
}

impl SoftBodyConfig {
    /// Preset for soft jelly-like material.
    pub fn jelly() -> Self {
        Self {
            youngs_modulus: 100.0,
            poisson_ratio: 0.45,
            density: 500.0,
            damping: 0.2,
            ..Default::default()
        }
    }

    /// Preset for rubber-like material.
    pub fn rubber() -> Self {
        Self {
            youngs_modulus: 5000.0,
            poisson_ratio: 0.49,
            density: 1100.0,
            damping: 0.1,
            ..Default::default()
        }
    }

    /// Preset for stiff material.
    pub fn stiff() -> Self {
        Self {
            youngs_modulus: 50000.0,
            poisson_ratio: 0.3,
            density: 2000.0,
            damping: 0.05,
            ..Default::default()
        }
    }

    /// Applies this configuration (returns self as a generator op).
    pub fn apply(&self) -> SoftBodyConfig {
        self.clone()
    }

    /// Computes Lamé parameters from Young's modulus and Poisson's ratio.
    pub fn lame_parameters(&self) -> LameParameters {
        let e = self.youngs_modulus;
        let nu = self.poisson_ratio;

        let mu = e / (2.0 * (1.0 + nu));
        let lambda = (e * nu) / ((1.0 + nu) * (1.0 - 2.0 * nu));

        LameParameters { mu, lambda }
    }
}

/// Lamé parameters for elastic materials.
#[derive(Debug, Clone, Copy)]
pub struct LameParameters {
    /// Shear modulus (μ).
    pub mu: f32,
    /// First Lamé parameter (λ).
    pub lambda: f32,
}

/// A vertex in the soft body.
#[derive(Debug, Clone)]
pub struct SoftVertex {
    /// Current position.
    pub position: Vec3,
    /// Rest position.
    pub rest_position: Vec3,
    /// Velocity.
    pub velocity: Vec3,
    /// Mass.
    pub mass: f32,
    /// Inverse mass (0 = fixed).
    pub inv_mass: f32,
    /// Whether this vertex is fixed.
    pub fixed: bool,
}

impl SoftVertex {
    /// Creates a new soft vertex.
    pub fn new(position: Vec3, mass: f32) -> Self {
        Self {
            position,
            rest_position: position,
            velocity: Vec3::ZERO,
            mass,
            inv_mass: if mass > 0.0 { 1.0 / mass } else { 0.0 },
            fixed: false,
        }
    }

    /// Creates a fixed (immovable) vertex.
    pub fn fixed(position: Vec3) -> Self {
        Self {
            position,
            rest_position: position,
            velocity: Vec3::ZERO,
            mass: 0.0,
            inv_mass: 0.0,
            fixed: true,
        }
    }
}

/// A tetrahedral element.
#[derive(Debug, Clone)]
pub struct Tetrahedron {
    /// Vertex indices (4 vertices).
    pub vertices: [usize; 4],
    /// Rest volume.
    pub rest_volume: f32,
    /// Inverse rest matrix (Dm^-1).
    pub inv_rest_matrix: Mat3,
}

impl Tetrahedron {
    /// Creates a new tetrahedron from vertex indices and positions.
    pub fn new(vertices: [usize; 4], positions: &[Vec3]) -> Self {
        let p0 = positions[vertices[0]];
        let p1 = positions[vertices[1]];
        let p2 = positions[vertices[2]];
        let p3 = positions[vertices[3]];

        // Compute rest matrix Dm = [p1-p0, p2-p0, p3-p0]
        let dm = Mat3::from_cols(p1 - p0, p2 - p0, p3 - p0);

        // Compute volume
        let rest_volume = dm.determinant().abs() / 6.0;

        // Compute inverse rest matrix
        let inv_rest_matrix = dm.inverse();

        Self {
            vertices,
            rest_volume,
            inv_rest_matrix,
        }
    }

    /// Computes the deformation gradient F.
    pub fn deformation_gradient(&self, positions: &[Vec3]) -> Mat3 {
        let p0 = positions[self.vertices[0]];
        let p1 = positions[self.vertices[1]];
        let p2 = positions[self.vertices[2]];
        let p3 = positions[self.vertices[3]];

        // Current deformed matrix Ds
        let ds = Mat3::from_cols(p1 - p0, p2 - p0, p3 - p0);

        // Deformation gradient F = Ds * Dm^-1
        ds * self.inv_rest_matrix
    }

    /// Computes current volume.
    pub fn current_volume(&self, positions: &[Vec3]) -> f32 {
        let p0 = positions[self.vertices[0]];
        let p1 = positions[self.vertices[1]];
        let p2 = positions[self.vertices[2]];
        let p3 = positions[self.vertices[3]];

        let ds = Mat3::from_cols(p1 - p0, p2 - p0, p3 - p0);
        ds.determinant().abs() / 6.0
    }
}

/// A soft body composed of tetrahedral elements.
#[derive(Debug, Clone)]
pub struct SoftBody {
    /// Vertices.
    pub vertices: Vec<SoftVertex>,
    /// Tetrahedral elements.
    pub tetrahedra: Vec<Tetrahedron>,
    /// Configuration.
    pub config: SoftBodyConfig,
    /// Lamé parameters (cached).
    lame: LameParameters,
}

impl SoftBody {
    /// Creates a soft body from vertices and tetrahedra.
    pub fn new(
        positions: &[Vec3],
        tetrahedra_indices: &[[usize; 4]],
        config: SoftBodyConfig,
    ) -> Self {
        // Compute total volume for mass distribution
        let temp_tets: Vec<Tetrahedron> = tetrahedra_indices
            .iter()
            .map(|&indices| Tetrahedron::new(indices, positions))
            .collect();

        let total_volume: f32 = temp_tets.iter().map(|t| t.rest_volume).sum();

        // Compute per-vertex mass based on adjacent tetrahedra
        let mut vertex_volumes = vec![0.0f32; positions.len()];
        for tet in &temp_tets {
            let tet_volume = tet.rest_volume / 4.0;
            for &vi in &tet.vertices {
                vertex_volumes[vi] += tet_volume;
            }
        }

        // Create vertices with distributed mass
        let total_mass = total_volume * config.density;
        let vertices: Vec<SoftVertex> = positions
            .iter()
            .enumerate()
            .map(|(i, &pos)| {
                let mass = (vertex_volumes[i] / total_volume) * total_mass;
                SoftVertex::new(pos, mass.max(0.001))
            })
            .collect();

        let lame = config.lame_parameters();

        Self {
            vertices,
            tetrahedra: temp_tets,
            config,
            lame,
        }
    }

    /// Creates a cubic soft body.
    pub fn cube(center: Vec3, size: f32, subdivisions: usize, config: SoftBodyConfig) -> Self {
        let (positions, tetrahedra) = generate_cube_mesh(center, size, subdivisions);
        Self::new(&positions, &tetrahedra, config)
    }

    /// Sets the configuration.
    pub fn with_config(mut self, config: SoftBodyConfig) -> Self {
        self.lame = config.lame_parameters();
        self.config = config;
        self
    }

    /// Fixes vertices at specific indices.
    pub fn fix_vertices(&mut self, indices: &[usize]) {
        for &i in indices {
            if i < self.vertices.len() {
                self.vertices[i].fixed = true;
                self.vertices[i].inv_mass = 0.0;
            }
        }
    }

    /// Fixes vertices above a Y threshold.
    pub fn fix_above_y(&mut self, y_threshold: f32) {
        for vertex in &mut self.vertices {
            if vertex.rest_position.y > y_threshold {
                vertex.fixed = true;
                vertex.inv_mass = 0.0;
            }
        }
    }

    /// Steps the simulation.
    pub fn step(&mut self, dt: f32) {
        let gravity = self.config.gravity;
        let damping = self.config.damping;

        // Apply external forces
        for vertex in &mut self.vertices {
            if !vertex.fixed {
                vertex.velocity += gravity * dt;
                vertex.velocity *= 1.0 - damping * dt;
            }
        }

        // Predict positions
        let mut predicted: Vec<Vec3> = self
            .vertices
            .iter()
            .map(|v| {
                if v.fixed {
                    v.position
                } else {
                    v.position + v.velocity * dt
                }
            })
            .collect();

        // Solve constraints
        for _ in 0..self.config.iterations {
            self.solve_constraints(&mut predicted);
        }

        // Update velocities and positions
        for (i, vertex) in self.vertices.iter_mut().enumerate() {
            if !vertex.fixed {
                vertex.velocity = (predicted[i] - vertex.position) / dt;
                vertex.position = predicted[i];
            }
        }
    }

    /// Solves elastic constraints.
    fn solve_constraints(&self, positions: &mut [Vec3]) {
        let mu = self.lame.mu;
        let lambda = self.lame.lambda;

        // Temporary force accumulator
        let mut forces = vec![Vec3::ZERO; positions.len()];

        for tet in &self.tetrahedra {
            // Compute deformation gradient
            let f = tet_deformation_gradient(tet, positions);

            // Compute Green strain: E = 0.5 * (F^T * F - I)
            let ftf = f.transpose() * f;
            let e = (ftf - Mat3::IDENTITY) * 0.5;

            // Compute stress using St. Venant-Kirchhoff model
            // S = lambda * tr(E) * I + 2 * mu * E
            let trace_e = e.col(0).x + e.col(1).y + e.col(2).z;
            let s = Mat3::IDENTITY * (lambda * trace_e) + e * (2.0 * mu);

            // First Piola-Kirchhoff stress: P = F * S
            let p = f * s;

            // Compute force on each vertex
            // Force = -V0 * P * Dm^-T
            let h = p * tet.inv_rest_matrix.transpose() * (-tet.rest_volume);

            // Distribute forces to vertices
            let f1 = h.col(0);
            let f2 = h.col(1);
            let f3 = h.col(2);
            let f0 = -(f1 + f2 + f3);

            forces[tet.vertices[0]] += f0;
            forces[tet.vertices[1]] += f1;
            forces[tet.vertices[2]] += f2;
            forces[tet.vertices[3]] += f3;
        }

        // Apply forces as position corrections
        let dt_sq = 1.0 / (self.config.iterations as f32);
        for (i, vertex) in self.vertices.iter().enumerate() {
            if !vertex.fixed && vertex.inv_mass > 0.0 {
                positions[i] += forces[i] * vertex.inv_mass * dt_sq * 0.01;
            }
        }

        // Volume preservation constraint
        for tet in &self.tetrahedra {
            self.solve_volume_constraint(tet, positions);
        }
    }

    /// Solves volume preservation constraint.
    fn solve_volume_constraint(&self, tet: &Tetrahedron, positions: &mut [Vec3]) {
        let current_vol = tet_volume(tet, positions);
        let rest_vol = tet.rest_volume;

        let error = current_vol - rest_vol;
        if error.abs() < 0.0001 * rest_vol {
            return;
        }

        // Compute gradient of volume w.r.t. each vertex
        let p0 = positions[tet.vertices[0]];
        let p1 = positions[tet.vertices[1]];
        let p2 = positions[tet.vertices[2]];
        let p3 = positions[tet.vertices[3]];

        // Volume gradient for each vertex
        let g1 = (p2 - p0).cross(p3 - p0) / 6.0;
        let g2 = (p3 - p0).cross(p1 - p0) / 6.0;
        let g3 = (p1 - p0).cross(p2 - p0) / 6.0;
        let g0 = -(g1 + g2 + g3);

        let gradients = [g0, g1, g2, g3];

        // Compute denominator
        let mut denom = 0.0;
        for (i, g) in gradients.iter().enumerate() {
            let inv_mass = self.vertices[tet.vertices[i]].inv_mass;
            denom += inv_mass * g.length_squared();
        }

        if denom < 0.0001 {
            return;
        }

        let correction = error / denom * 0.5; // Softness factor

        // Apply corrections
        for (i, g) in gradients.iter().enumerate() {
            let vi = tet.vertices[i];
            if !self.vertices[vi].fixed {
                positions[vi] -= *g * correction * self.vertices[vi].inv_mass;
            }
        }
    }

    /// Gets all positions.
    pub fn positions(&self) -> Vec<Vec3> {
        self.vertices.iter().map(|v| v.position).collect()
    }

    /// Applies an impulse at a point.
    pub fn apply_impulse(&mut self, vertex_idx: usize, impulse: Vec3) {
        if vertex_idx < self.vertices.len() && !self.vertices[vertex_idx].fixed {
            let inv_mass = self.vertices[vertex_idx].inv_mass;
            self.vertices[vertex_idx].velocity += impulse * inv_mass;
        }
    }

    /// Resets to rest state.
    pub fn reset(&mut self) {
        for vertex in &mut self.vertices {
            vertex.position = vertex.rest_position;
            vertex.velocity = Vec3::ZERO;
        }
    }

    /// Computes surface triangles for rendering.
    pub fn surface_triangles(&self) -> Vec<[usize; 3]> {
        // Extract surface faces from tetrahedra
        use std::collections::HashMap;

        let mut face_count: HashMap<[usize; 3], usize> = HashMap::new();

        for tet in &self.tetrahedra {
            let v = tet.vertices;
            // Four faces of tetrahedron
            let faces = [
                [v[0], v[2], v[1]],
                [v[0], v[1], v[3]],
                [v[0], v[3], v[2]],
                [v[1], v[2], v[3]],
            ];

            for face in &faces {
                // Sort to create canonical key
                let mut key = *face;
                key.sort();
                *face_count.entry(key).or_insert(0) += 1;
            }
        }

        // Surface faces appear only once (not shared)
        let mut surface = Vec::new();
        for tet in &self.tetrahedra {
            let v = tet.vertices;
            let faces = [
                [v[0], v[2], v[1]],
                [v[0], v[1], v[3]],
                [v[0], v[3], v[2]],
                [v[1], v[2], v[3]],
            ];

            for face in faces {
                let mut key = face;
                key.sort();
                if face_count.get(&key) == Some(&1) {
                    surface.push(face);
                }
            }
        }

        surface
    }
}

/// Computes deformation gradient for a tetrahedron.
fn tet_deformation_gradient(tet: &Tetrahedron, positions: &[Vec3]) -> Mat3 {
    let p0 = positions[tet.vertices[0]];
    let p1 = positions[tet.vertices[1]];
    let p2 = positions[tet.vertices[2]];
    let p3 = positions[tet.vertices[3]];

    let ds = Mat3::from_cols(p1 - p0, p2 - p0, p3 - p0);
    ds * tet.inv_rest_matrix
}

/// Computes volume of a tetrahedron.
fn tet_volume(tet: &Tetrahedron, positions: &[Vec3]) -> f32 {
    let p0 = positions[tet.vertices[0]];
    let p1 = positions[tet.vertices[1]];
    let p2 = positions[tet.vertices[2]];
    let p3 = positions[tet.vertices[3]];

    let ds = Mat3::from_cols(p1 - p0, p2 - p0, p3 - p0);
    ds.determinant().abs() / 6.0
}

/// Generates a subdivided cube mesh with tetrahedra.
pub fn generate_cube_mesh(
    center: Vec3,
    size: f32,
    subdivisions: usize,
) -> (Vec<Vec3>, Vec<[usize; 4]>) {
    let n = subdivisions.max(1);
    let step = size / n as f32;
    let half = size / 2.0;

    // Generate vertices
    let mut positions = Vec::new();
    for z in 0..=n {
        for y in 0..=n {
            for x in 0..=n {
                let pos = center
                    + Vec3::new(
                        x as f32 * step - half,
                        y as f32 * step - half,
                        z as f32 * step - half,
                    );
                positions.push(pos);
            }
        }
    }

    // Generate tetrahedra for each cube cell
    let mut tetrahedra = Vec::new();
    let stride = n + 1;

    for z in 0..n {
        for y in 0..n {
            for x in 0..n {
                // 8 corners of the cube cell
                let v000 = z * stride * stride + y * stride + x;
                let v100 = v000 + 1;
                let v010 = v000 + stride;
                let v110 = v010 + 1;
                let v001 = v000 + stride * stride;
                let v101 = v001 + 1;
                let v011 = v001 + stride;
                let v111 = v011 + 1;

                // Split cube into 5 tetrahedra
                tetrahedra.push([v000, v100, v010, v001]);
                tetrahedra.push([v100, v110, v010, v111]);
                tetrahedra.push([v001, v010, v011, v111]);
                tetrahedra.push([v100, v001, v101, v111]);
                tetrahedra.push([v100, v010, v001, v111]);
            }
        }
    }

    (positions, tetrahedra)
}

/// Generates a tetrahedral mesh from surface triangles.
pub fn tetrahedralize_surface(
    positions: &[Vec3],
    triangles: &[[usize; 3]],
) -> (Vec<Vec3>, Vec<[usize; 4]>) {
    // Simple approach: add center point and connect to each triangle
    let mut new_positions = positions.to_vec();

    // Compute centroid
    let centroid: Vec3 = positions.iter().copied().sum::<Vec3>() / positions.len().max(1) as f32;
    let center_idx = new_positions.len();
    new_positions.push(centroid);

    // Create tetrahedra from triangles
    let tetrahedra: Vec<[usize; 4]> = triangles
        .iter()
        .map(|tri| [tri[0], tri[1], tri[2], center_idx])
        .collect();

    (new_positions, tetrahedra)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_soft_body_config_default() {
        let config = SoftBodyConfig::default();
        assert!(config.youngs_modulus > 0.0);
        assert!(config.poisson_ratio > 0.0 && config.poisson_ratio < 0.5);
    }

    #[test]
    fn test_soft_body_config_presets() {
        let jelly = SoftBodyConfig::jelly();
        let stiff = SoftBodyConfig::stiff();

        assert!(jelly.youngs_modulus < stiff.youngs_modulus);
    }

    #[test]
    fn test_lame_parameters() {
        let config = SoftBodyConfig::default();
        let lame = config.lame_parameters();

        assert!(lame.mu > 0.0);
        assert!(lame.lambda > 0.0);
    }

    #[test]
    fn test_soft_vertex_new() {
        let vertex = SoftVertex::new(Vec3::new(1.0, 2.0, 3.0), 1.0);

        assert_eq!(vertex.position, Vec3::new(1.0, 2.0, 3.0));
        assert!(!vertex.fixed);
        assert!(vertex.inv_mass > 0.0);
    }

    #[test]
    fn test_soft_vertex_fixed() {
        let vertex = SoftVertex::fixed(Vec3::ZERO);

        assert!(vertex.fixed);
        assert_eq!(vertex.inv_mass, 0.0);
    }

    #[test]
    fn test_tetrahedron_new() {
        let positions = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
        ];

        let tet = Tetrahedron::new([0, 1, 2, 3], &positions);

        assert!(tet.rest_volume > 0.0);
    }

    #[test]
    fn test_tetrahedron_volume() {
        let positions = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
        ];

        let tet = Tetrahedron::new([0, 1, 2, 3], &positions);
        let vol = tet.current_volume(&positions);

        // Volume of unit corner tetrahedron = 1/6
        assert!((vol - 1.0 / 6.0).abs() < 0.001);
    }

    #[test]
    fn test_generate_cube_mesh() {
        let (positions, tetrahedra) = generate_cube_mesh(Vec3::ZERO, 1.0, 2);

        // 2 subdivisions = 3x3x3 = 27 vertices
        assert_eq!(positions.len(), 27);

        // 2^3 = 8 cubes, 5 tetrahedra each = 40
        assert_eq!(tetrahedra.len(), 40);
    }

    #[test]
    fn test_soft_body_cube() {
        let config = SoftBodyConfig::default();
        let body = SoftBody::cube(Vec3::ZERO, 1.0, 2, config);

        assert_eq!(body.vertices.len(), 27);
        assert_eq!(body.tetrahedra.len(), 40);
    }

    #[test]
    fn test_soft_body_fix_vertices() {
        let config = SoftBodyConfig::default();
        let mut body = SoftBody::cube(Vec3::ZERO, 1.0, 2, config);

        body.fix_vertices(&[0, 1, 2]);

        assert!(body.vertices[0].fixed);
        assert!(body.vertices[1].fixed);
        assert!(body.vertices[2].fixed);
        assert!(!body.vertices[3].fixed);
    }

    #[test]
    fn test_soft_body_step() {
        let mut config = SoftBodyConfig::default();
        config.gravity = Vec3::new(0.0, -9.81, 0.0);

        let mut body = SoftBody::cube(Vec3::Y * 2.0, 1.0, 1, config);
        body.fix_above_y(2.4);

        let initial_y: f32 = body.vertices.iter().map(|v| v.position.y).sum();

        // Step simulation
        for _ in 0..10 {
            body.step(1.0 / 60.0);
        }

        let final_y: f32 = body.vertices.iter().map(|v| v.position.y).sum();

        // Non-fixed vertices should have fallen
        assert!(final_y < initial_y);
    }

    #[test]
    fn test_soft_body_positions() {
        let body = SoftBody::cube(Vec3::ZERO, 1.0, 1, SoftBodyConfig::default());
        let positions = body.positions();

        assert_eq!(positions.len(), body.vertices.len());
    }

    #[test]
    fn test_soft_body_reset() {
        let mut body = SoftBody::cube(Vec3::ZERO, 1.0, 1, SoftBodyConfig::default());

        // Modify positions
        body.vertices[0].position = Vec3::ONE * 10.0;
        body.vertices[0].velocity = Vec3::ONE;

        body.reset();

        assert_eq!(body.vertices[0].position, body.vertices[0].rest_position);
        assert_eq!(body.vertices[0].velocity, Vec3::ZERO);
    }

    #[test]
    fn test_soft_body_surface_triangles() {
        let body = SoftBody::cube(Vec3::ZERO, 1.0, 1, SoftBodyConfig::default());
        let surface = body.surface_triangles();

        // 1 subdivision cube = 6 faces * 2 triangles = 12 surface triangles
        // But with tetrahedralization this may differ
        assert!(!surface.is_empty());
    }

    #[test]
    fn test_tetrahedralize_surface() {
        let positions = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        ];
        let triangles = vec![[0, 1, 2]];

        let (new_pos, tets) = tetrahedralize_surface(&positions, &triangles);

        assert_eq!(new_pos.len(), 4); // 3 original + 1 center
        assert_eq!(tets.len(), 1);
    }

    #[test]
    fn test_soft_body_apply_impulse() {
        let mut body = SoftBody::cube(Vec3::ZERO, 1.0, 1, SoftBodyConfig::default());

        let initial_vel = body.vertices[0].velocity;
        body.apply_impulse(0, Vec3::X * 10.0);

        assert_ne!(body.vertices[0].velocity, initial_vel);
    }

    #[test]
    fn test_deformation_gradient_identity() {
        // At rest, deformation gradient should be identity
        let positions = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
        ];

        let tet = Tetrahedron::new([0, 1, 2, 3], &positions);
        let f = tet.deformation_gradient(&positions);

        // Should be close to identity
        let diff = f - Mat3::IDENTITY;
        let error = diff.col(0).length() + diff.col(1).length() + diff.col(2).length();
        assert!(error < 0.001);
    }
}
