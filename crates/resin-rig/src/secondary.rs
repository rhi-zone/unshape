//! Secondary motion effects for procedural animation.
//!
//! Provides physics-based secondary motion:
//! - Jiggle (spring dynamics for soft body parts like hair, clothing, flesh)
//! - Follow-through (delayed reaction to parent motion)
//! - Overlapping action (different parts move at different rates)
//! - Drag (resistance to motion)
//!
//! # Example
//!
//! ```
//! use rhizome_resin_rig::secondary::{JiggleBone, SecondaryMotion, SecondaryConfig};
//! use glam::Vec3;
//!
//! // Create a jiggle bone for hair
//! let config = SecondaryConfig {
//!     stiffness: 50.0,
//!     damping: 5.0,
//!     gravity: Vec3::new(0.0, -9.81, 0.0),
//!     ..SecondaryConfig::default()
//! };
//! let mut jiggle = JiggleBone::from_config(config);
//!
//! // Update each frame
//! let parent_position = Vec3::ZERO;
//! let parent_rotation = glam::Quat::IDENTITY;
//! jiggle.update(parent_position, parent_rotation, 0.016);
//!
//! // Get the resulting offset
//! let offset = jiggle.offset();
//! ```

use glam::{Quat, Vec3};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Configuration for secondary motion effects.
///
/// This struct configures jiggle physics parameters for secondary motion
/// like hair, clothing, and soft body deformation.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dynop", derive(rhizome_resin_op::Op))]
#[cfg_attr(feature = "dynop", op(input = (), output = Secondary))]
pub struct Secondary {
    /// Spring stiffness (higher = snappier return to rest).
    pub stiffness: f32,
    /// Damping coefficient (higher = less oscillation).
    pub damping: f32,
    /// Mass of the simulated point.
    pub mass: f32,
    /// Gravity vector.
    pub gravity: Vec3,
    /// Maximum displacement from rest position.
    pub max_displacement: f32,
    /// Whether to enable collision with parent bone.
    pub enable_collision: bool,
}

/// Backwards-compatible type alias.
pub type SecondaryConfig = Secondary;

impl Secondary {
    /// Applies this generator, returning the configuration.
    pub fn apply(&self) -> Secondary {
        self.clone()
    }
}

impl Default for Secondary {
    fn default() -> Self {
        Self {
            stiffness: 100.0,
            damping: 10.0,
            mass: 1.0,
            gravity: Vec3::new(0.0, -9.81, 0.0),
            max_displacement: 1.0,
            enable_collision: false,
        }
    }
}

impl Secondary {
    /// Creates a config for soft/bouncy motion (low stiffness, low damping).
    pub fn soft() -> Self {
        Self {
            stiffness: 30.0,
            damping: 3.0,
            mass: 1.0,
            gravity: Vec3::new(0.0, -9.81, 0.0),
            max_displacement: 2.0,
            enable_collision: false,
        }
    }

    /// Creates a config for stiff motion (high stiffness, high damping).
    pub fn stiff() -> Self {
        Self {
            stiffness: 300.0,
            damping: 30.0,
            mass: 1.0,
            gravity: Vec3::new(0.0, -9.81, 0.0),
            max_displacement: 0.3,
            enable_collision: false,
        }
    }

    /// Creates a config for hair-like motion.
    pub fn hair() -> Self {
        Self {
            stiffness: 50.0,
            damping: 5.0,
            mass: 0.5,
            gravity: Vec3::new(0.0, -9.81, 0.0),
            max_displacement: 1.5,
            enable_collision: true,
        }
    }

    /// Creates a config for tail-like motion.
    pub fn tail() -> Self {
        Self {
            stiffness: 80.0,
            damping: 8.0,
            mass: 1.5,
            gravity: Vec3::new(0.0, -9.81, 0.0),
            max_displacement: 2.0,
            enable_collision: false,
        }
    }

    /// Creates a config for cloth-like motion.
    pub fn cloth() -> Self {
        Self {
            stiffness: 40.0,
            damping: 6.0,
            mass: 0.3,
            gravity: Vec3::new(0.0, -9.81, 0.0),
            max_displacement: 3.0,
            enable_collision: true,
        }
    }
}

/// A single jiggle bone with spring physics.
#[derive(Debug, Clone)]
pub struct JiggleBone {
    /// Current position offset from rest.
    position: Vec3,
    /// Current velocity.
    velocity: Vec3,
    /// Rest position in local space.
    rest_position: Vec3,
    /// Previous parent position (for inertia calculation).
    prev_parent_position: Vec3,
    /// Configuration.
    config: SecondaryConfig,
}

impl Default for JiggleBone {
    fn default() -> Self {
        Self::new()
    }
}

impl JiggleBone {
    /// Creates a new jiggle bone.
    pub fn new() -> Self {
        Self {
            position: Vec3::ZERO,
            velocity: Vec3::ZERO,
            rest_position: Vec3::ZERO,
            prev_parent_position: Vec3::ZERO,
            config: SecondaryConfig::default(),
        }
    }

    /// Creates from configuration.
    pub fn from_config(config: SecondaryConfig) -> Self {
        Self {
            position: Vec3::ZERO,
            velocity: Vec3::ZERO,
            rest_position: Vec3::ZERO,
            prev_parent_position: Vec3::ZERO,
            config,
        }
    }

    /// Sets the rest position in local space.
    pub fn with_rest_position(mut self, rest: Vec3) -> Self {
        self.rest_position = rest;
        self.position = rest;
        self
    }

    /// Updates the jiggle bone simulation.
    ///
    /// # Arguments
    /// * `parent_position` - Parent bone's world position
    /// * `parent_rotation` - Parent bone's world rotation
    /// * `dt` - Time step in seconds
    pub fn update(&mut self, parent_position: Vec3, parent_rotation: Quat, dt: f32) {
        if dt <= 0.0 {
            return;
        }

        // Compute target position in world space
        let target_world = parent_position + parent_rotation * self.rest_position;

        // Inertia: react to parent acceleration
        let parent_velocity = (parent_position - self.prev_parent_position) / dt;
        self.prev_parent_position = parent_position;

        // Current world position
        let current_world = parent_position + parent_rotation * self.position;

        // Spring force toward target
        let displacement = current_world - target_world;
        let spring_force = -self.config.stiffness * displacement;

        // Damping force
        let damping_force = -self.config.damping * self.velocity;

        // Gravity
        let gravity_force = self.config.gravity * self.config.mass;

        // Inertia force (react to parent motion)
        let inertia_force = -parent_velocity * self.config.mass * 0.5;

        // Total acceleration
        let total_force = spring_force + damping_force + gravity_force + inertia_force;
        let acceleration = total_force / self.config.mass;

        // Integrate velocity and position (semi-implicit Euler)
        self.velocity += acceleration * dt;
        let new_world = current_world + self.velocity * dt;

        // Convert back to local space
        let inv_rotation = parent_rotation.inverse();
        self.position = inv_rotation * (new_world - parent_position);

        // Clamp displacement
        let local_displacement = self.position - self.rest_position;
        if local_displacement.length() > self.config.max_displacement {
            self.position =
                self.rest_position + local_displacement.normalize() * self.config.max_displacement;
            // Reduce velocity when hitting limit
            self.velocity *= 0.5;
        }
    }

    /// Returns the current position offset from rest.
    pub fn offset(&self) -> Vec3 {
        self.position - self.rest_position
    }

    /// Returns the current position in local space.
    pub fn position(&self) -> Vec3 {
        self.position
    }

    /// Returns the current velocity.
    pub fn velocity(&self) -> Vec3 {
        self.velocity
    }

    /// Resets to rest state.
    pub fn reset(&mut self) {
        self.position = self.rest_position;
        self.velocity = Vec3::ZERO;
    }
}

/// A chain of jiggle bones for hair, tails, etc.
#[derive(Debug, Clone)]
pub struct JiggleChain {
    /// Bones in the chain.
    pub bones: Vec<JiggleBone>,
    /// Segment lengths between bones.
    lengths: Vec<f32>,
    /// Root position.
    root_position: Vec3,
    /// Root rotation.
    root_rotation: Quat,
}

impl JiggleChain {
    /// Creates a new jiggle chain.
    pub fn new(bone_count: usize, segment_length: f32, config: SecondaryConfig) -> Self {
        let mut bones = Vec::with_capacity(bone_count);
        let lengths = vec![segment_length; bone_count.saturating_sub(1)];

        for i in 0..bone_count {
            let rest_pos = Vec3::new(0.0, -(i as f32) * segment_length, 0.0);
            bones.push(JiggleBone::from_config(config.clone()).with_rest_position(rest_pos));
        }

        Self {
            bones,
            lengths,
            root_position: Vec3::ZERO,
            root_rotation: Quat::IDENTITY,
        }
    }

    /// Updates the chain simulation.
    pub fn update(&mut self, root_position: Vec3, root_rotation: Quat, dt: f32) {
        self.root_position = root_position;
        self.root_rotation = root_rotation;

        if self.bones.is_empty() {
            return;
        }

        // Update first bone from root
        self.bones[0].update(root_position, root_rotation, dt);

        // Update subsequent bones from previous bone
        for i in 1..self.bones.len() {
            let prev_world = root_position + root_rotation * self.bones[i - 1].position();

            // Compute direction from previous bone
            let curr_world = root_position + root_rotation * self.bones[i].position();
            let dir = (curr_world - prev_world).normalize_or_zero();

            // Create rotation that looks along the chain
            let look_rot = if dir.length_squared() > 0.01 {
                Quat::from_rotation_arc(Vec3::NEG_Y, dir)
            } else {
                Quat::IDENTITY
            };

            self.bones[i].update(prev_world, look_rot * root_rotation, dt);
        }

        // Apply distance constraints
        self.apply_distance_constraints();
    }

    fn apply_distance_constraints(&mut self) {
        for i in 1..self.bones.len() {
            let target_length = self.lengths[i - 1];
            let prev_pos = self.bones[i - 1].position;
            let curr_pos = self.bones[i].position;

            let delta = curr_pos - prev_pos;
            let current_length = delta.length();

            if current_length > 0.001 {
                let correction = delta.normalize() * (target_length - current_length);
                self.bones[i].position = curr_pos + correction * 0.5;
            }
        }
    }

    /// Returns positions of all bones in world space.
    pub fn world_positions(&self) -> Vec<Vec3> {
        self.bones
            .iter()
            .map(|b| self.root_position + self.root_rotation * b.position())
            .collect()
    }

    /// Returns the number of bones.
    pub fn bone_count(&self) -> usize {
        self.bones.len()
    }

    /// Resets all bones to rest state.
    pub fn reset(&mut self) {
        for bone in &mut self.bones {
            bone.reset();
        }
    }
}

/// Follow-through motion that delays response to parent changes.
#[derive(Debug, Clone)]
pub struct FollowThrough {
    /// Current value.
    value: Vec3,
    /// Target value.
    target: Vec3,
    /// Smoothing factor (0-1, lower = more lag).
    smoothing: f32,
}

impl Default for FollowThrough {
    fn default() -> Self {
        Self::new(0.1)
    }
}

impl FollowThrough {
    /// Creates a new follow-through with given smoothing.
    pub fn new(smoothing: f32) -> Self {
        Self {
            value: Vec3::ZERO,
            target: Vec3::ZERO,
            smoothing: smoothing.clamp(0.01, 1.0),
        }
    }

    /// Sets the smoothing factor.
    pub fn with_smoothing(mut self, smoothing: f32) -> Self {
        self.smoothing = smoothing.clamp(0.01, 1.0);
        self
    }

    /// Updates with a new target.
    pub fn update(&mut self, target: Vec3, dt: f32) {
        self.target = target;
        // Exponential smoothing
        let factor = 1.0 - (-self.smoothing * dt * 60.0).exp();
        self.value = self.value + (self.target - self.value) * factor;
    }

    /// Returns the current smoothed value.
    pub fn value(&self) -> Vec3 {
        self.value
    }

    /// Returns the lag (difference from target).
    pub fn lag(&self) -> Vec3 {
        self.target - self.value
    }

    /// Resets to target.
    pub fn reset(&mut self, value: Vec3) {
        self.value = value;
        self.target = value;
    }
}

/// Rotation follow-through for quaternions.
#[derive(Debug, Clone)]
pub struct RotationFollowThrough {
    /// Current rotation.
    rotation: Quat,
    /// Target rotation.
    target: Quat,
    /// Smoothing factor.
    smoothing: f32,
}

impl Default for RotationFollowThrough {
    fn default() -> Self {
        Self::new(0.1)
    }
}

impl RotationFollowThrough {
    /// Creates a new rotation follow-through.
    pub fn new(smoothing: f32) -> Self {
        Self {
            rotation: Quat::IDENTITY,
            target: Quat::IDENTITY,
            smoothing: smoothing.clamp(0.01, 1.0),
        }
    }

    /// Updates with a new target.
    pub fn update(&mut self, target: Quat, dt: f32) {
        self.target = target;
        let factor = 1.0 - (-self.smoothing * dt * 60.0).exp();
        self.rotation = self.rotation.slerp(self.target, factor);
    }

    /// Returns the current rotation.
    pub fn rotation(&self) -> Quat {
        self.rotation
    }

    /// Resets to target.
    pub fn reset(&mut self, rotation: Quat) {
        self.rotation = rotation;
        self.target = rotation;
    }
}

/// Overlapping action with multiple time delays.
#[derive(Debug, Clone)]
pub struct OverlappingAction {
    /// Follow-through instances with different delays.
    layers: Vec<FollowThrough>,
    /// Weights for each layer.
    weights: Vec<f32>,
}

impl OverlappingAction {
    /// Creates overlapping action with given delay values.
    ///
    /// Each delay creates a layer that follows at that rate.
    pub fn new(delays: &[f32]) -> Self {
        let layers: Vec<_> = delays.iter().map(|&d| FollowThrough::new(d)).collect();
        let weights = vec![1.0 / layers.len() as f32; layers.len()];
        Self { layers, weights }
    }

    /// Creates with custom weights.
    pub fn with_weights(mut self, weights: &[f32]) -> Self {
        let total: f32 = weights.iter().sum();
        self.weights = weights.iter().map(|&w| w / total).collect();
        if self.weights.len() < self.layers.len() {
            let last = *self.weights.last().unwrap_or(&0.0);
            self.weights.resize(self.layers.len(), last);
        }
        self
    }

    /// Updates all layers.
    pub fn update(&mut self, target: Vec3, dt: f32) {
        for layer in &mut self.layers {
            layer.update(target, dt);
        }
    }

    /// Returns the weighted average value.
    pub fn value(&self) -> Vec3 {
        let mut result = Vec3::ZERO;
        for (layer, &weight) in self.layers.iter().zip(&self.weights) {
            result += layer.value() * weight;
        }
        result
    }

    /// Returns the value at a specific layer.
    pub fn layer_value(&self, index: usize) -> Option<Vec3> {
        self.layers.get(index).map(|l| l.value())
    }

    /// Resets all layers.
    pub fn reset(&mut self, value: Vec3) {
        for layer in &mut self.layers {
            layer.reset(value);
        }
    }
}

/// Drag effect that resists velocity.
#[derive(Debug, Clone)]
pub struct Drag {
    /// Current velocity.
    velocity: Vec3,
    /// Drag coefficient.
    coefficient: f32,
}

impl Default for Drag {
    fn default() -> Self {
        Self::new(0.5)
    }
}

impl Drag {
    /// Creates a new drag effect.
    pub fn new(coefficient: f32) -> Self {
        Self {
            velocity: Vec3::ZERO,
            coefficient: coefficient.max(0.0),
        }
    }

    /// Updates with current position and computes drag force.
    pub fn update(&mut self, position: Vec3, prev_position: Vec3, dt: f32) -> Vec3 {
        if dt <= 0.0 {
            return Vec3::ZERO;
        }

        self.velocity = (position - prev_position) / dt;
        -self.velocity * self.coefficient
    }

    /// Returns the current velocity.
    pub fn velocity(&self) -> Vec3 {
        self.velocity
    }
}

/// Combined secondary motion controller.
#[derive(Debug, Clone)]
pub struct SecondaryMotion {
    /// Jiggle bones by name.
    jiggle_bones: Vec<(String, JiggleBone)>,
    /// Jiggle chains by name.
    jiggle_chains: Vec<(String, JiggleChain)>,
    /// Follow-through effects by name.
    follow_throughs: Vec<(String, FollowThrough)>,
    /// Rotation follow-throughs by name.
    rotation_follow_throughs: Vec<(String, RotationFollowThrough)>,
}

impl Default for SecondaryMotion {
    fn default() -> Self {
        Self::new()
    }
}

impl SecondaryMotion {
    /// Creates a new secondary motion controller.
    pub fn new() -> Self {
        Self {
            jiggle_bones: Vec::new(),
            jiggle_chains: Vec::new(),
            follow_throughs: Vec::new(),
            rotation_follow_throughs: Vec::new(),
        }
    }

    /// Adds a jiggle bone.
    pub fn add_jiggle_bone(&mut self, name: &str, bone: JiggleBone) {
        self.jiggle_bones.push((name.to_string(), bone));
    }

    /// Adds a jiggle chain.
    pub fn add_jiggle_chain(&mut self, name: &str, chain: JiggleChain) {
        self.jiggle_chains.push((name.to_string(), chain));
    }

    /// Adds a follow-through.
    pub fn add_follow_through(&mut self, name: &str, ft: FollowThrough) {
        self.follow_throughs.push((name.to_string(), ft));
    }

    /// Adds a rotation follow-through.
    pub fn add_rotation_follow_through(&mut self, name: &str, ft: RotationFollowThrough) {
        self.rotation_follow_throughs.push((name.to_string(), ft));
    }

    /// Updates a jiggle bone by name.
    pub fn update_jiggle_bone(
        &mut self,
        name: &str,
        parent_position: Vec3,
        parent_rotation: Quat,
        dt: f32,
    ) {
        if let Some((_, bone)) = self.jiggle_bones.iter_mut().find(|(n, _)| n == name) {
            bone.update(parent_position, parent_rotation, dt);
        }
    }

    /// Updates a jiggle chain by name.
    pub fn update_jiggle_chain(
        &mut self,
        name: &str,
        root_position: Vec3,
        root_rotation: Quat,
        dt: f32,
    ) {
        if let Some((_, chain)) = self.jiggle_chains.iter_mut().find(|(n, _)| n == name) {
            chain.update(root_position, root_rotation, dt);
        }
    }

    /// Updates a follow-through by name.
    pub fn update_follow_through(&mut self, name: &str, target: Vec3, dt: f32) {
        if let Some((_, ft)) = self.follow_throughs.iter_mut().find(|(n, _)| n == name) {
            ft.update(target, dt);
        }
    }

    /// Updates a rotation follow-through by name.
    pub fn update_rotation_follow_through(&mut self, name: &str, target: Quat, dt: f32) {
        if let Some((_, ft)) = self
            .rotation_follow_throughs
            .iter_mut()
            .find(|(n, _)| n == name)
        {
            ft.update(target, dt);
        }
    }

    /// Gets a jiggle bone offset by name.
    pub fn get_jiggle_offset(&self, name: &str) -> Option<Vec3> {
        self.jiggle_bones
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, b)| b.offset())
    }

    /// Gets jiggle chain positions by name.
    pub fn get_chain_positions(&self, name: &str) -> Option<Vec<Vec3>> {
        self.jiggle_chains
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, c)| c.world_positions())
    }

    /// Gets a follow-through value by name.
    pub fn get_follow_through(&self, name: &str) -> Option<Vec3> {
        self.follow_throughs
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, ft)| ft.value())
    }

    /// Gets a rotation follow-through by name.
    pub fn get_rotation_follow_through(&self, name: &str) -> Option<Quat> {
        self.rotation_follow_throughs
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, ft)| ft.rotation())
    }

    /// Resets all secondary motion.
    pub fn reset(&mut self) {
        for (_, bone) in &mut self.jiggle_bones {
            bone.reset();
        }
        for (_, chain) in &mut self.jiggle_chains {
            chain.reset();
        }
        for (_, ft) in &mut self.follow_throughs {
            ft.reset(Vec3::ZERO);
        }
        for (_, ft) in &mut self.rotation_follow_throughs {
            ft.reset(Quat::IDENTITY);
        }
    }
}

// ============================================================================
// Mesh-Based Jiggle
// ============================================================================

/// A mesh-based jiggle system for soft body secondary motion.
///
/// Applies jiggle physics to mesh vertices, useful for flesh, cloth, or other
/// deformable surfaces attached to a skeleton.
#[derive(Debug, Clone)]
pub struct JiggleMesh {
    /// Jiggle bones for each vertex.
    bones: Vec<JiggleBone>,
    /// Original vertex positions (rest pose).
    rest_positions: Vec<Vec3>,
    /// Anchor mask (true = fixed, false = can jiggle).
    anchored: Vec<bool>,
}

impl JiggleMesh {
    /// Creates a jiggle mesh from vertex positions.
    pub fn new(positions: &[Vec3], config: SecondaryConfig) -> Self {
        let bones: Vec<JiggleBone> = positions
            .iter()
            .map(|&pos| JiggleBone::from_config(config.clone()).with_rest_position(pos))
            .collect();

        Self {
            bones,
            rest_positions: positions.to_vec(),
            anchored: vec![false; positions.len()],
        }
    }

    /// Sets vertices as anchored (won't jiggle).
    pub fn anchor_vertices(&mut self, indices: &[usize]) {
        for &idx in indices {
            if idx < self.anchored.len() {
                self.anchored[idx] = true;
            }
        }
    }

    /// Anchors vertices above a certain Y threshold.
    pub fn anchor_above_y(&mut self, y_threshold: f32) {
        for (i, pos) in self.rest_positions.iter().enumerate() {
            if pos.y > y_threshold {
                self.anchored[i] = true;
            }
        }
    }

    /// Updates the jiggle mesh with new target positions.
    pub fn update(&mut self, targets: &[Vec3], dt: f32) {
        for (i, (bone, &target)) in self.bones.iter_mut().zip(targets.iter()).enumerate() {
            if self.anchored[i] {
                // Anchored vertices snap to target without physics
                bone.position = target;
                bone.velocity = Vec3::ZERO;
            } else {
                bone.update(target, Quat::IDENTITY, dt);
            }
        }
    }

    /// Updates with a transform applied to rest positions.
    pub fn update_with_transform(&mut self, translation: Vec3, rotation: Quat, dt: f32) {
        for (i, (bone, &rest)) in self
            .bones
            .iter_mut()
            .zip(self.rest_positions.iter())
            .enumerate()
        {
            let target = translation + rotation * rest;
            if self.anchored[i] {
                // Anchored vertices snap to target without physics
                bone.position = target;
                bone.velocity = Vec3::ZERO;
            } else {
                bone.update(target, rotation, dt);
            }
        }
    }

    /// Resets all points to their rest positions.
    pub fn reset(&mut self) {
        for (bone, &rest) in self.bones.iter_mut().zip(self.rest_positions.iter()) {
            bone.position = rest;
            bone.velocity = Vec3::ZERO;
        }
    }

    /// Gets current positions.
    pub fn positions(&self) -> Vec<Vec3> {
        self.bones.iter().map(|b| b.position()).collect()
    }

    /// Gets offset from rest for each vertex.
    pub fn offsets(&self) -> Vec<Vec3> {
        self.bones.iter().map(|b| b.offset()).collect()
    }

    /// Returns the number of vertices.
    pub fn vertex_count(&self) -> usize {
        self.bones.len()
    }
}

// ============================================================================
// Wind Force
// ============================================================================

/// Simulates wind force on jiggle systems.
///
/// Provides a wind vector with optional turbulence for more natural motion.
#[derive(Debug, Clone, Copy)]
pub struct WindForce {
    /// Base wind direction and strength.
    pub direction: Vec3,
    /// Turbulence strength (0 = steady wind).
    pub turbulence: f32,
    /// Turbulence frequency (higher = faster variation).
    pub frequency: f32,
    /// Current time for noise sampling.
    time: f32,
}

impl Default for WindForce {
    fn default() -> Self {
        Self {
            direction: Vec3::new(1.0, 0.0, 0.0),
            turbulence: 0.3,
            frequency: 2.0,
            time: 0.0,
        }
    }
}

impl WindForce {
    /// Creates a new wind force.
    pub fn new(direction: Vec3, turbulence: f32) -> Self {
        Self {
            direction,
            turbulence,
            frequency: 2.0,
            time: 0.0,
        }
    }

    /// Advances time and returns the current wind vector at a position.
    ///
    /// The wind varies based on position and time to simulate turbulence.
    pub fn sample(&mut self, dt: f32, position: Vec3) -> Vec3 {
        self.time += dt;

        if self.turbulence <= 0.0 {
            return self.direction;
        }

        // Simple turbulence using position-based variation
        let noise_x = ((position.x + self.time * self.frequency) * 3.0).sin();
        let noise_y = ((position.y + self.time * self.frequency * 1.3) * 2.5).sin();
        let noise_z = ((position.z + self.time * self.frequency * 0.7) * 2.8).sin();

        let turbulence_vec = Vec3::new(noise_x, noise_y, noise_z) * self.turbulence;

        self.direction + turbulence_vec
    }

    /// Gets the current wind direction without advancing time.
    pub fn current(&self, position: Vec3) -> Vec3 {
        if self.turbulence <= 0.0 {
            return self.direction;
        }

        let noise_x = ((position.x + self.time * self.frequency) * 3.0).sin();
        let noise_y = ((position.y + self.time * self.frequency * 1.3) * 2.5).sin();
        let noise_z = ((position.z + self.time * self.frequency * 0.7) * 2.8).sin();

        self.direction + Vec3::new(noise_x, noise_y, noise_z) * self.turbulence
    }

    /// Resets the time accumulator.
    pub fn reset(&mut self) {
        self.time = 0.0;
    }
}

/// Applies wind force to a jiggle bone.
pub fn apply_wind_to_bone(bone: &mut JiggleBone, wind: &mut WindForce, dt: f32) {
    let wind_force = wind.sample(dt, bone.position());
    // Wind acts as an additional velocity impulse
    bone.velocity += wind_force * dt;
}

/// Applies wind force to a jiggle chain.
pub fn apply_wind_to_chain(chain: &mut JiggleChain, wind: &mut WindForce, dt: f32) {
    let positions = chain.world_positions();
    for (i, pos) in positions.iter().enumerate() {
        if i == 0 {
            continue; // Skip root
        }
        let wind_force = wind.sample(dt, *pos);
        chain.bones[i].velocity += wind_force * dt;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_secondary_config_presets() {
        let soft = SecondaryConfig::soft();
        let stiff = SecondaryConfig::stiff();

        assert!(soft.stiffness < stiff.stiffness);
        assert!(soft.max_displacement > stiff.max_displacement);
    }

    #[test]
    fn test_jiggle_bone_creation() {
        let config = SecondaryConfig {
            stiffness: 50.0,
            damping: 5.0,
            ..SecondaryConfig::default()
        };
        let bone = JiggleBone::from_config(config).with_rest_position(Vec3::new(0.0, 1.0, 0.0));

        assert_eq!(bone.position(), Vec3::new(0.0, 1.0, 0.0));
        assert_eq!(bone.velocity(), Vec3::ZERO);
    }

    #[test]
    fn test_jiggle_bone_update() {
        let config = SecondaryConfig {
            stiffness: 100.0,
            damping: 10.0,
            gravity: Vec3::ZERO,
            ..SecondaryConfig::default()
        };
        let mut bone = JiggleBone::from_config(config);

        // Move parent
        bone.update(Vec3::ZERO, Quat::IDENTITY, 0.016);
        bone.update(Vec3::new(1.0, 0.0, 0.0), Quat::IDENTITY, 0.016);

        // Bone should lag behind
        let offset = bone.offset();
        assert!(offset.x < 1.0);
    }

    #[test]
    fn test_jiggle_bone_reset() {
        let mut bone = JiggleBone::new().with_rest_position(Vec3::X);

        bone.update(Vec3::Y, Quat::IDENTITY, 0.1);
        bone.reset();

        assert_eq!(bone.position(), Vec3::X);
        assert_eq!(bone.velocity(), Vec3::ZERO);
    }

    #[test]
    fn test_jiggle_chain() {
        let config = SecondaryConfig::tail();
        let mut chain = JiggleChain::new(5, 0.2, config);

        assert_eq!(chain.bone_count(), 5);

        chain.update(Vec3::ZERO, Quat::IDENTITY, 0.016);
        let positions = chain.world_positions();
        assert_eq!(positions.len(), 5);
    }

    #[test]
    fn test_follow_through() {
        let mut ft = FollowThrough::new(0.1);

        ft.update(Vec3::X, 0.016);
        assert!(ft.value().x > 0.0 && ft.value().x < 1.0);

        // After many updates, should approach target
        for _ in 0..100 {
            ft.update(Vec3::X, 0.016);
        }
        assert!((ft.value() - Vec3::X).length() < 0.01);
    }

    #[test]
    fn test_follow_through_lag() {
        let mut ft = FollowThrough::new(0.05);
        ft.reset(Vec3::ZERO);

        ft.update(Vec3::X, 0.016);
        let lag = ft.lag();

        assert!(lag.x > 0.0); // Target is ahead
    }

    #[test]
    fn test_rotation_follow_through() {
        let mut ft = RotationFollowThrough::new(0.1);

        let target = Quat::from_rotation_y(std::f32::consts::FRAC_PI_2);
        ft.update(target, 0.016);

        assert!(ft.rotation() != Quat::IDENTITY);
        assert!(ft.rotation() != target);
    }

    #[test]
    fn test_overlapping_action() {
        let mut overlap = OverlappingAction::new(&[0.5, 0.1, 0.05]);

        // After multiple updates, faster layers should be closer to target
        for _ in 0..10 {
            overlap.update(Vec3::X, 0.016);
        }

        // Each layer should have different values
        let v0 = overlap.layer_value(0).unwrap();
        let v1 = overlap.layer_value(1).unwrap();
        let v2 = overlap.layer_value(2).unwrap();

        // Faster layer (higher smoothing) should be closer to target
        assert!(v0.x >= v1.x, "v0={} should be >= v1={}", v0.x, v1.x);
        assert!(v1.x >= v2.x, "v1={} should be >= v2={}", v1.x, v2.x);
    }

    #[test]
    fn test_overlapping_weights() {
        let overlap = OverlappingAction::new(&[0.1, 0.2]).with_weights(&[0.8, 0.2]);

        assert_eq!(overlap.weights.len(), 2);
    }

    #[test]
    fn test_drag() {
        let mut drag = Drag::new(0.5);

        let force = drag.update(Vec3::X, Vec3::ZERO, 0.016);

        // Drag force should oppose velocity
        assert!(force.x < 0.0);
        assert!(drag.velocity().x > 0.0);
    }

    #[test]
    fn test_secondary_motion_controller() {
        let mut sm = SecondaryMotion::new();

        sm.add_jiggle_bone("hair", JiggleBone::new());
        sm.add_jiggle_chain("tail", JiggleChain::new(3, 0.5, SecondaryConfig::tail()));
        sm.add_follow_through("hip_sway", FollowThrough::new(0.1));

        sm.update_jiggle_bone("hair", Vec3::ZERO, Quat::IDENTITY, 0.016);
        sm.update_jiggle_chain("tail", Vec3::ZERO, Quat::IDENTITY, 0.016);
        sm.update_follow_through("hip_sway", Vec3::X, 0.016);

        assert!(sm.get_jiggle_offset("hair").is_some());
        assert!(sm.get_chain_positions("tail").is_some());
        assert!(sm.get_follow_through("hip_sway").is_some());
    }

    #[test]
    fn test_secondary_motion_reset() {
        let mut sm = SecondaryMotion::new();
        sm.add_jiggle_bone("test", JiggleBone::new());

        sm.update_jiggle_bone("test", Vec3::X, Quat::IDENTITY, 0.1);
        sm.reset();

        assert_eq!(sm.get_jiggle_offset("test"), Some(Vec3::ZERO));
    }

    #[test]
    fn test_max_displacement() {
        let config = SecondaryConfig {
            stiffness: 1.0,
            damping: 0.1,
            max_displacement: 0.5,
            gravity: Vec3::new(0.0, -100.0, 0.0),
            ..SecondaryConfig::default()
        };
        let mut bone = JiggleBone::from_config(config);

        // Apply strong gravity for many frames
        for _ in 0..100 {
            bone.update(Vec3::ZERO, Quat::IDENTITY, 0.016);
        }

        // Should be clamped to max displacement
        assert!(bone.offset().length() <= 0.5 + 0.01);
    }

    // ========================================================================
    // JiggleMesh Tests
    // ========================================================================

    #[test]
    fn test_jiggle_mesh_creation() {
        let positions = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        ];

        let mesh = JiggleMesh::new(&positions, SecondaryConfig::soft());
        assert_eq!(mesh.vertex_count(), 3);
    }

    #[test]
    fn test_jiggle_mesh_anchor() {
        let positions = vec![Vec3::ZERO, Vec3::X, Vec3::Y];
        let mut mesh = JiggleMesh::new(&positions, SecondaryConfig::default());

        mesh.anchor_vertices(&[0]);

        // Anchored vertex should stay fixed after update
        let targets = vec![Vec3::new(10.0, 0.0, 0.0), Vec3::X, Vec3::Y];
        mesh.update(&targets, 0.016);

        let pos = mesh.positions();
        assert_eq!(pos[0], Vec3::new(10.0, 0.0, 0.0)); // Anchored, moved to target
    }

    #[test]
    fn test_jiggle_mesh_anchor_above_y() {
        let positions = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.5, 0.0),
            Vec3::new(0.0, 1.5, 0.0),
        ];

        let mut mesh = JiggleMesh::new(&positions, SecondaryConfig::default());
        mesh.anchor_above_y(1.0);

        // Only the point at y=1.5 should be anchored
        assert!(!mesh.anchored[0]);
        assert!(!mesh.anchored[1]);
        assert!(mesh.anchored[2]);
    }

    #[test]
    fn test_jiggle_mesh_update_with_transform() {
        let positions = vec![Vec3::ZERO, Vec3::X];
        // Use a critically damped config without gravity for predictable test
        let config = SecondaryConfig {
            stiffness: 300.0,
            damping: 35.0, // Critically damped
            mass: 1.0,
            gravity: Vec3::ZERO,
            max_displacement: 2.0,
            enable_collision: false,
        };
        let mut mesh = JiggleMesh::new(&positions, config);

        // Run enough updates for spring to converge toward target
        for _ in 0..100 {
            mesh.update_with_transform(Vec3::Y, Quat::IDENTITY, 0.016);
        }

        // With critically damped spring, should approach target
        let pos = mesh.positions();
        // Target for first vertex is ZERO + Y = (0, 1, 0)
        // Position is in local space (relative to parent which is at Y)
        // So local position should be close to ZERO (which is rest_position)
        assert!(
            pos[0].length() < 0.5,
            "position {} should be close to rest",
            pos[0]
        );
    }

    #[test]
    fn test_jiggle_mesh_reset() {
        let positions = vec![Vec3::ZERO, Vec3::X];
        let mut mesh = JiggleMesh::new(&positions, SecondaryConfig::default());

        mesh.update(&[Vec3::Y, Vec3::Y], 0.1);
        mesh.reset();

        let pos = mesh.positions();
        assert_eq!(pos[0], Vec3::ZERO);
        assert_eq!(pos[1], Vec3::X);
    }

    // ========================================================================
    // WindForce Tests
    // ========================================================================

    #[test]
    fn test_wind_force_creation() {
        let wind = WindForce::new(Vec3::X, 0.5);
        assert_eq!(wind.direction, Vec3::X);
        assert_eq!(wind.turbulence, 0.5);
    }

    #[test]
    fn test_wind_force_no_turbulence() {
        let mut wind = WindForce::new(Vec3::X, 0.0);
        let sample = wind.sample(0.016, Vec3::ZERO);
        assert_eq!(sample, Vec3::X);
    }

    #[test]
    fn test_wind_force_with_turbulence() {
        let mut wind = WindForce::new(Vec3::X, 0.5);

        let s1 = wind.sample(0.016, Vec3::ZERO);
        let s2 = wind.sample(0.016, Vec3::new(10.0, 5.0, 3.0));

        // With turbulence, samples at different positions/times should differ
        assert!(s1 != s2 || wind.turbulence == 0.0);
    }

    #[test]
    fn test_wind_force_reset() {
        let mut wind = WindForce::new(Vec3::X, 0.3);
        wind.sample(1.0, Vec3::ZERO);
        wind.reset();
        assert_eq!(wind.time, 0.0);
    }

    #[test]
    fn test_apply_wind_to_bone() {
        let mut bone = JiggleBone::new();
        let mut wind = WindForce::new(Vec3::X * 10.0, 0.0);

        let initial_vel = bone.velocity();
        apply_wind_to_bone(&mut bone, &mut wind, 0.1);

        assert!(bone.velocity().x > initial_vel.x);
    }

    #[test]
    fn test_apply_wind_to_chain() {
        let config = SecondaryConfig::tail();
        let mut chain = JiggleChain::new(3, 0.5, config);
        chain.update(Vec3::ZERO, Quat::IDENTITY, 0.016);

        let mut wind = WindForce::new(Vec3::X * 10.0, 0.0);
        apply_wind_to_chain(&mut chain, &mut wind, 0.1);

        // Non-root bones should have velocity from wind
        assert!(chain.bones[1].velocity().x > 0.0);
    }
}
