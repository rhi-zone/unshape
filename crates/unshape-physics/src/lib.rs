//! Rigid body physics simulation for resin.
//!
//! Provides basic rigid body dynamics with collision detection and response:
//! - `RigidBody` - dynamic or static rigid body with mass and inertia
//! - `Collider` - collision shapes (sphere, box, plane)
//! - `PhysicsWorld` - simulation container with gravity and constraints

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

pub mod cloth;
pub mod softbody;

pub use cloth::{
    Cloth, ClothCollider, ClothConfig, ClothParticle, CollisionResult,
    DistanceConstraint as ClothDistanceConstraint, SelfCollisionGrid, query_collision,
    solve_self_collision,
};
pub use softbody::{
    LameParameters, SoftBody, SoftBodyConfig, SoftVertex, Tetrahedron, generate_cube_mesh,
    tetrahedralize_surface,
};

/// Registers all physics operations with an [`OpRegistry`].
///
/// Call this to enable deserialization of physics ops from saved pipelines.
#[cfg(feature = "dynop")]
pub fn register_ops(registry: &mut unshape_op::OpRegistry) {
    registry.register_type::<ClothConfig>("resin::ClothConfig");
    registry.register_type::<SoftBodyConfig>("resin::SoftBodyConfig");
    registry.register_type::<Physics>("resin::Physics");
}

use glam::{Mat3, Quat, Vec3};

// ============================================================================
// Collider Shapes
// ============================================================================

/// Collision shape for rigid bodies.
#[derive(Clone, Debug)]
pub enum Collider {
    /// Sphere collider.
    Sphere {
        /// Radius of the sphere.
        radius: f32,
    },
    /// Axis-aligned box collider.
    Box {
        /// Half-extents along each axis.
        half_extents: Vec3,
    },
    /// Infinite plane defined by normal and distance from origin.
    Plane {
        /// Unit normal pointing away from the solid side.
        normal: Vec3,
        /// Distance from origin along the normal.
        distance: f32,
    },
}

impl Collider {
    /// Create a sphere collider.
    pub fn sphere(radius: f32) -> Self {
        Collider::Sphere { radius }
    }

    /// Create a box collider.
    pub fn box_shape(half_extents: Vec3) -> Self {
        Collider::Box { half_extents }
    }

    /// Create a plane collider (infinite ground plane).
    pub fn plane(normal: Vec3, distance: f32) -> Self {
        Collider::Plane {
            normal: normal.normalize(),
            distance,
        }
    }

    /// Create a ground plane at y=0.
    pub fn ground() -> Self {
        Collider::plane(Vec3::Y, 0.0)
    }
}

// ============================================================================
// Rigid Body
// ============================================================================

/// A rigid body in the physics simulation.
#[derive(Clone, Debug)]
pub struct RigidBody {
    /// Position in world space.
    pub position: Vec3,
    /// Orientation as quaternion.
    pub orientation: Quat,
    /// Linear velocity.
    pub velocity: Vec3,
    /// Angular velocity.
    pub angular_velocity: Vec3,
    /// Mass (0 = infinite/static).
    pub mass: f32,
    /// Inverse mass (cached).
    pub inv_mass: f32,
    /// Inertia tensor (diagonal approximation).
    pub inertia: Vec3,
    /// Inverse inertia tensor.
    pub inv_inertia: Vec3,
    /// Restitution (bounciness) 0-1.
    pub restitution: f32,
    /// Friction coefficient.
    pub friction: f32,
    /// Linear damping.
    pub linear_damping: f32,
    /// Angular damping.
    pub angular_damping: f32,
    /// Collision shape.
    pub collider: Collider,
    /// Whether this body is static (infinite mass).
    pub is_static: bool,
    /// Accumulated force for this frame.
    force: Vec3,
    /// Accumulated torque for this frame.
    torque: Vec3,
}

impl RigidBody {
    /// Create a new dynamic rigid body.
    pub fn new(position: Vec3, collider: Collider, mass: f32) -> Self {
        let inv_mass = if mass > 0.0 { 1.0 / mass } else { 0.0 };
        let inertia = compute_inertia(&collider, mass);
        let inv_inertia = if mass > 0.0 {
            Vec3::new(1.0 / inertia.x, 1.0 / inertia.y, 1.0 / inertia.z)
        } else {
            Vec3::ZERO
        };

        Self {
            position,
            orientation: Quat::IDENTITY,
            velocity: Vec3::ZERO,
            angular_velocity: Vec3::ZERO,
            mass,
            inv_mass,
            inertia,
            inv_inertia,
            restitution: 0.3,
            friction: 0.5,
            linear_damping: 0.01,
            angular_damping: 0.01,
            collider,
            is_static: mass == 0.0,
            force: Vec3::ZERO,
            torque: Vec3::ZERO,
        }
    }

    /// Create a static (immovable) rigid body.
    pub fn new_static(position: Vec3, collider: Collider) -> Self {
        Self {
            position,
            orientation: Quat::IDENTITY,
            velocity: Vec3::ZERO,
            angular_velocity: Vec3::ZERO,
            mass: 0.0,
            inv_mass: 0.0,
            inertia: Vec3::ZERO,
            inv_inertia: Vec3::ZERO,
            restitution: 0.3,
            friction: 0.5,
            linear_damping: 0.0,
            angular_damping: 0.0,
            collider,
            is_static: true,
            force: Vec3::ZERO,
            torque: Vec3::ZERO,
        }
    }

    /// Apply a force at the center of mass.
    pub fn apply_force(&mut self, force: Vec3) {
        if !self.is_static {
            self.force += force;
        }
    }

    /// Apply a force at a world-space point (generates torque).
    pub fn apply_force_at_point(&mut self, force: Vec3, point: Vec3) {
        if !self.is_static {
            self.force += force;
            let r = point - self.position;
            self.torque += r.cross(force);
        }
    }

    /// Apply an impulse at the center of mass.
    pub fn apply_impulse(&mut self, impulse: Vec3) {
        if !self.is_static {
            self.velocity += impulse * self.inv_mass;
        }
    }

    /// Apply an impulse at a world-space point.
    pub fn apply_impulse_at_point(&mut self, impulse: Vec3, point: Vec3) {
        if !self.is_static {
            self.velocity += impulse * self.inv_mass;
            let r = point - self.position;
            self.angular_velocity += self.inv_inertia * r.cross(impulse);
        }
    }

    /// Apply torque.
    pub fn apply_torque(&mut self, torque: Vec3) {
        if !self.is_static {
            self.torque += torque;
        }
    }

    /// Get the velocity at a world-space point on the body.
    pub fn velocity_at_point(&self, point: Vec3) -> Vec3 {
        let r = point - self.position;
        self.velocity + self.angular_velocity.cross(r)
    }

    /// Get the rotation matrix.
    pub fn rotation_matrix(&self) -> Mat3 {
        Mat3::from_quat(self.orientation)
    }

    /// Clear accumulated forces.
    fn clear_forces(&mut self) {
        self.force = Vec3::ZERO;
        self.torque = Vec3::ZERO;
    }
}

/// Compute inertia tensor for a shape.
fn compute_inertia(collider: &Collider, mass: f32) -> Vec3 {
    if mass == 0.0 {
        return Vec3::ZERO;
    }

    match collider {
        Collider::Sphere { radius } => {
            let i = 0.4 * mass * radius * radius;
            Vec3::splat(i)
        }
        Collider::Box { half_extents } => {
            let e = *half_extents * 2.0; // full extents
            let factor = mass / 12.0;
            Vec3::new(
                factor * (e.y * e.y + e.z * e.z),
                factor * (e.x * e.x + e.z * e.z),
                factor * (e.x * e.x + e.y * e.y),
            )
        }
        Collider::Plane { .. } => Vec3::ZERO, // Planes are always static
    }
}

// ============================================================================
// Contact / Collision
// ============================================================================

/// A contact point between two bodies.
#[derive(Clone, Debug)]
pub struct Contact {
    /// Index of first body.
    pub body_a: usize,
    /// Index of second body.
    pub body_b: usize,
    /// Contact point in world space.
    pub point: Vec3,
    /// Contact normal (from A to B).
    pub normal: Vec3,
    /// Penetration depth.
    pub depth: f32,
}

impl Contact {
    /// Flip a contact to swap bodies and invert normal.
    ///
    /// Used to handle symmetric collision pairs (e.g., sphere-plane vs plane-sphere).
    #[inline]
    fn flip(mut self) -> Self {
        self.normal = -self.normal;
        std::mem::swap(&mut self.body_a, &mut self.body_b);
        self
    }
}

// ============================================================================
// Constraints / Joints
// ============================================================================

/// A constraint between bodies.
#[derive(Clone, Debug)]
pub enum Constraint {
    /// Distance constraint - maintains fixed distance between two points.
    Distance(DistanceConstraint),
    /// Point constraint - anchors a body point to world space or another body.
    Point(PointConstraint),
    /// Hinge constraint - rotation around a single axis.
    Hinge(HingeConstraint),
    /// Spring constraint - elastic connection between points.
    Spring(SpringConstraint),
}

/// Distance constraint parameters.
#[derive(Clone, Debug)]
pub struct DistanceConstraint {
    /// First body index.
    pub body_a: usize,
    /// Second body index.
    pub body_b: usize,
    /// Local anchor point on body A.
    pub local_anchor_a: Vec3,
    /// Local anchor point on body B.
    pub local_anchor_b: Vec3,
    /// Target distance between anchors.
    pub distance: f32,
    /// Constraint stiffness (0-1, 1 = rigid).
    pub stiffness: f32,
}

impl DistanceConstraint {
    /// Create a new distance constraint.
    pub fn new(body_a: usize, body_b: usize, distance: f32) -> Self {
        Self {
            body_a,
            body_b,
            local_anchor_a: Vec3::ZERO,
            local_anchor_b: Vec3::ZERO,
            distance,
            stiffness: 1.0,
        }
    }

    /// Set anchor points in local body space.
    pub fn with_anchors(mut self, anchor_a: Vec3, anchor_b: Vec3) -> Self {
        self.local_anchor_a = anchor_a;
        self.local_anchor_b = anchor_b;
        self
    }

    /// Set constraint stiffness.
    pub fn with_stiffness(mut self, stiffness: f32) -> Self {
        self.stiffness = stiffness.clamp(0.0, 1.0);
        self
    }
}

/// Point constraint - anchors a body to a world point or another body.
#[derive(Clone, Debug)]
pub struct PointConstraint {
    /// Body index.
    pub body: usize,
    /// Local anchor point on the body.
    pub local_anchor: Vec3,
    /// Target position (world space) or second body info.
    pub target: PointConstraintTarget,
    /// Constraint stiffness (0-1, 1 = rigid).
    pub stiffness: f32,
}

/// Target for a point constraint.
#[derive(Clone, Debug)]
pub enum PointConstraintTarget {
    /// Fixed world position.
    World(Vec3),
    /// Point on another body.
    Body {
        /// Other body index.
        body: usize,
        /// Local anchor on other body.
        local_anchor: Vec3,
    },
}

impl PointConstraint {
    /// Create a point constraint anchoring body to a world position.
    pub fn to_world(body: usize, local_anchor: Vec3, world_pos: Vec3) -> Self {
        Self {
            body,
            local_anchor,
            target: PointConstraintTarget::World(world_pos),
            stiffness: 1.0,
        }
    }

    /// Create a point constraint connecting two bodies (ball joint).
    pub fn between_bodies(
        body_a: usize,
        local_anchor_a: Vec3,
        body_b: usize,
        local_anchor_b: Vec3,
    ) -> Self {
        Self {
            body: body_a,
            local_anchor: local_anchor_a,
            target: PointConstraintTarget::Body {
                body: body_b,
                local_anchor: local_anchor_b,
            },
            stiffness: 1.0,
        }
    }

    /// Set constraint stiffness.
    pub fn with_stiffness(mut self, stiffness: f32) -> Self {
        self.stiffness = stiffness.clamp(0.0, 1.0);
        self
    }
}

/// Hinge constraint - rotation around a single axis.
#[derive(Clone, Debug)]
pub struct HingeConstraint {
    /// First body index.
    pub body_a: usize,
    /// Second body index.
    pub body_b: usize,
    /// Anchor point in body A's local space.
    pub local_anchor_a: Vec3,
    /// Anchor point in body B's local space.
    pub local_anchor_b: Vec3,
    /// Hinge axis in body A's local space.
    pub local_axis_a: Vec3,
    /// Hinge axis in body B's local space.
    pub local_axis_b: Vec3,
    /// Optional angle limits (min, max) in radians.
    pub limits: Option<(f32, f32)>,
    /// Constraint stiffness.
    pub stiffness: f32,
}

impl HingeConstraint {
    /// Create a hinge constraint with axis along Y.
    pub fn new(body_a: usize, body_b: usize) -> Self {
        Self {
            body_a,
            body_b,
            local_anchor_a: Vec3::ZERO,
            local_anchor_b: Vec3::ZERO,
            local_axis_a: Vec3::Y,
            local_axis_b: Vec3::Y,
            limits: None,
            stiffness: 1.0,
        }
    }

    /// Set anchor points.
    pub fn with_anchors(mut self, anchor_a: Vec3, anchor_b: Vec3) -> Self {
        self.local_anchor_a = anchor_a;
        self.local_anchor_b = anchor_b;
        self
    }

    /// Set hinge axes (should be unit vectors).
    pub fn with_axes(mut self, axis_a: Vec3, axis_b: Vec3) -> Self {
        self.local_axis_a = axis_a.normalize();
        self.local_axis_b = axis_b.normalize();
        self
    }

    /// Set constraint stiffness.
    pub fn with_stiffness(mut self, stiffness: f32) -> Self {
        self.stiffness = stiffness.clamp(0.0, 1.0);
        self
    }
}

/// Spring constraint - elastic connection between points.
#[derive(Clone, Debug)]
pub struct SpringConstraint {
    /// First body index.
    pub body_a: usize,
    /// Second body index.
    pub body_b: usize,
    /// Local anchor point on body A.
    pub local_anchor_a: Vec3,
    /// Local anchor point on body B.
    pub local_anchor_b: Vec3,
    /// Rest length of the spring.
    pub rest_length: f32,
    /// Spring stiffness (Hooke's law k).
    pub stiffness: f32,
    /// Damping coefficient.
    pub damping: f32,
}

impl SpringConstraint {
    /// Create a new spring constraint.
    pub fn new(body_a: usize, body_b: usize, rest_length: f32) -> Self {
        Self {
            body_a,
            body_b,
            local_anchor_a: Vec3::ZERO,
            local_anchor_b: Vec3::ZERO,
            rest_length,
            stiffness: 100.0,
            damping: 1.0,
        }
    }

    /// Set anchor points in local body space.
    pub fn with_anchors(mut self, anchor_a: Vec3, anchor_b: Vec3) -> Self {
        self.local_anchor_a = anchor_a;
        self.local_anchor_b = anchor_b;
        self
    }

    /// Set spring stiffness.
    pub fn with_stiffness(mut self, stiffness: f32) -> Self {
        self.stiffness = stiffness.max(0.0);
        self
    }

    /// Set damping coefficient.
    pub fn with_damping(mut self, damping: f32) -> Self {
        self.damping = damping.max(0.0);
        self
    }
}

// ============================================================================
// Physics World
// ============================================================================

/// Configuration for physics simulation.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dynop", derive(unshape_op::Op))]
#[cfg_attr(feature = "dynop", op(input = (), output = Physics))]
pub struct Physics {
    /// Gravity acceleration.
    pub gravity: Vec3,
    /// Number of constraint solver iterations.
    pub solver_iterations: u32,
    /// Time step.
    pub dt: f32,
}

impl Default for Physics {
    fn default() -> Self {
        Self {
            gravity: Vec3::new(0.0, -9.81, 0.0),
            solver_iterations: 10,
            dt: 1.0 / 60.0,
        }
    }
}

impl Physics {
    /// Applies this configuration (returns self as a generator op).
    pub fn apply(&self) -> Physics {
        self.clone()
    }
}

/// Backwards-compatible type alias.
pub type PhysicsConfig = Physics;

/// The physics simulation world.
pub struct PhysicsWorld {
    /// All rigid bodies.
    pub bodies: Vec<RigidBody>,
    /// All constraints.
    pub constraints: Vec<Constraint>,
    /// Configuration.
    pub config: Physics,
}

impl PhysicsWorld {
    /// Create a new physics world.
    pub fn new(config: PhysicsConfig) -> Self {
        Self {
            bodies: Vec::new(),
            constraints: Vec::new(),
            config,
        }
    }

    /// Add a rigid body and return its index.
    pub fn add_body(&mut self, body: RigidBody) -> usize {
        let index = self.bodies.len();
        self.bodies.push(body);
        index
    }

    /// Get a body by index.
    pub fn body(&self, index: usize) -> Option<&RigidBody> {
        self.bodies.get(index)
    }

    /// Get a mutable body by index.
    pub fn body_mut(&mut self, index: usize) -> Option<&mut RigidBody> {
        self.bodies.get_mut(index)
    }

    /// Add a constraint and return its index.
    pub fn add_constraint(&mut self, constraint: Constraint) -> usize {
        let index = self.constraints.len();
        self.constraints.push(constraint);
        index
    }

    /// Add a distance constraint between two bodies.
    pub fn add_distance_constraint(
        &mut self,
        body_a: usize,
        body_b: usize,
        distance: f32,
    ) -> usize {
        self.add_constraint(Constraint::Distance(DistanceConstraint::new(
            body_a, body_b, distance,
        )))
    }

    /// Add a point constraint anchoring a body to world position.
    pub fn add_point_constraint_to_world(
        &mut self,
        body: usize,
        local_anchor: Vec3,
        world_pos: Vec3,
    ) -> usize {
        self.add_constraint(Constraint::Point(PointConstraint::to_world(
            body,
            local_anchor,
            world_pos,
        )))
    }

    /// Add a ball joint between two bodies.
    pub fn add_ball_joint(
        &mut self,
        body_a: usize,
        local_anchor_a: Vec3,
        body_b: usize,
        local_anchor_b: Vec3,
    ) -> usize {
        self.add_constraint(Constraint::Point(PointConstraint::between_bodies(
            body_a,
            local_anchor_a,
            body_b,
            local_anchor_b,
        )))
    }

    /// Add a hinge joint between two bodies.
    pub fn add_hinge_joint(&mut self, body_a: usize, body_b: usize) -> usize {
        self.add_constraint(Constraint::Hinge(HingeConstraint::new(body_a, body_b)))
    }

    /// Add a spring between two bodies.
    pub fn add_spring(
        &mut self,
        body_a: usize,
        body_b: usize,
        rest_length: f32,
        stiffness: f32,
        damping: f32,
    ) -> usize {
        self.add_constraint(Constraint::Spring(
            SpringConstraint::new(body_a, body_b, rest_length)
                .with_stiffness(stiffness)
                .with_damping(damping),
        ))
    }

    /// Get a constraint by index.
    pub fn constraint(&self, index: usize) -> Option<&Constraint> {
        self.constraints.get(index)
    }

    /// Get a mutable constraint by index.
    pub fn constraint_mut(&mut self, index: usize) -> Option<&mut Constraint> {
        self.constraints.get_mut(index)
    }

    /// Remove a constraint by index.
    pub fn remove_constraint(&mut self, index: usize) -> Option<Constraint> {
        if index < self.constraints.len() {
            Some(self.constraints.remove(index))
        } else {
            None
        }
    }

    /// Step the simulation forward.
    pub fn step(&mut self) {
        let dt = self.config.dt;

        self.apply_forces(dt);
        self.integrate_velocities(dt);

        let contacts = self.detect_collisions();
        self.solve_contacts_and_constraints(&contacts);

        self.integrate_positions(dt);
    }

    /// Apply external forces (gravity) and spring forces to all bodies.
    fn apply_forces(&mut self, dt: f32) {
        let gravity = self.config.gravity;

        for body in &mut self.bodies {
            if !body.is_static {
                body.apply_force(gravity * body.mass);
            }
        }

        self.apply_spring_forces(dt);
    }

    /// Integrate velocities from accumulated forces.
    fn integrate_velocities(&mut self, dt: f32) {
        for body in &mut self.bodies {
            if !body.is_static {
                // Linear
                body.velocity += body.force * body.inv_mass * dt;
                body.velocity *= 1.0 - body.linear_damping;

                // Angular
                body.angular_velocity += body.inv_inertia * body.torque * dt;
                body.angular_velocity *= 1.0 - body.angular_damping;
            }
            body.clear_forces();
        }
    }

    /// Solve collision contacts and positional constraints iteratively.
    fn solve_contacts_and_constraints(&mut self, contacts: &[Contact]) {
        for _ in 0..self.config.solver_iterations {
            for contact in contacts {
                self.resolve_contact(contact);
            }
            self.solve_constraints();
        }
    }

    /// Integrate positions from velocities.
    fn integrate_positions(&mut self, dt: f32) {
        for body in &mut self.bodies {
            if !body.is_static {
                body.position += body.velocity * dt;

                // Orientation (using quaternion integration)
                let w = body.angular_velocity;
                let q = body.orientation;
                let dq = Quat::from_xyzw(w.x, w.y, w.z, 0.0) * q * 0.5 * dt;
                body.orientation = (q + dq).normalize();
            }
        }
    }

    /// Apply spring forces to bodies.
    fn apply_spring_forces(&mut self, _dt: f32) {
        // Collect spring data first to avoid borrow issues
        let spring_data: Vec<_> = self
            .constraints
            .iter()
            .filter_map(|c| {
                if let Constraint::Spring(spring) = c {
                    Some((
                        spring.body_a,
                        spring.body_b,
                        spring.local_anchor_a,
                        spring.local_anchor_b,
                        spring.rest_length,
                        spring.stiffness,
                        spring.damping,
                    ))
                } else {
                    None
                }
            })
            .collect();

        for (body_a, body_b, local_a, local_b, rest_length, stiffness, damping) in spring_data {
            if body_a >= self.bodies.len() || body_b >= self.bodies.len() {
                continue;
            }

            // Get world space anchor positions
            let pos_a = self.bodies[body_a].position + self.bodies[body_a].orientation * local_a;
            let pos_b = self.bodies[body_b].position + self.bodies[body_b].orientation * local_b;

            let delta = pos_b - pos_a;
            let distance = delta.length();

            if distance < 0.0001 {
                continue;
            }

            let direction = delta / distance;
            let stretch = distance - rest_length;

            // Hooke's law: F = -kx
            let spring_force = direction * stretch * stiffness;

            // Damping: F = -cv
            let vel_a = self.bodies[body_a].velocity_at_point(pos_a);
            let vel_b = self.bodies[body_b].velocity_at_point(pos_b);
            let relative_vel = vel_b - vel_a;
            let damping_force = direction * relative_vel.dot(direction) * damping;

            let total_force = spring_force + damping_force;

            // Apply forces to bodies (split borrow)
            if body_a < body_b {
                let (left, right) = self.bodies.split_at_mut(body_b);
                left[body_a].apply_force_at_point(total_force, pos_a);
                right[0].apply_force_at_point(-total_force, pos_b);
            } else {
                let (left, right) = self.bodies.split_at_mut(body_a);
                right[0].apply_force_at_point(total_force, pos_a);
                left[body_b].apply_force_at_point(-total_force, pos_b);
            }
        }
    }

    /// Solve positional constraints.
    fn solve_constraints(&mut self) {
        // Clone constraints to avoid borrow issues
        let constraints: Vec<_> = self.constraints.clone();

        for constraint in &constraints {
            match constraint {
                Constraint::Distance(c) => self.solve_distance_constraint(c),
                Constraint::Point(c) => self.solve_point_constraint(c),
                Constraint::Hinge(c) => self.solve_hinge_constraint(c),
                Constraint::Spring(_) => {} // Springs handled in force phase
            }
        }
    }

    /// Solve a distance constraint.
    fn solve_distance_constraint(&mut self, constraint: &DistanceConstraint) {
        let body_a_idx = constraint.body_a;
        let body_b_idx = constraint.body_b;

        if body_a_idx >= self.bodies.len() || body_b_idx >= self.bodies.len() {
            return;
        }

        // Get world space anchor positions
        let pos_a = self.bodies[body_a_idx].position
            + self.bodies[body_a_idx].orientation * constraint.local_anchor_a;
        let pos_b = self.bodies[body_b_idx].position
            + self.bodies[body_b_idx].orientation * constraint.local_anchor_b;

        let delta = pos_b - pos_a;
        let current_distance = delta.length();

        if current_distance < 0.0001 {
            return;
        }

        let direction = delta / current_distance;
        let error = current_distance - constraint.distance;

        // Skip if error is negligible
        if error.abs() < 0.0001 {
            return;
        }

        // Compute correction based on inverse masses
        let inv_mass_a = self.bodies[body_a_idx].inv_mass;
        let inv_mass_b = self.bodies[body_b_idx].inv_mass;
        let total_inv_mass = inv_mass_a + inv_mass_b;

        if total_inv_mass < 0.0001 {
            return;
        }

        let correction = direction * error * constraint.stiffness / total_inv_mass;

        // Apply position corrections
        if !self.bodies[body_a_idx].is_static {
            self.bodies[body_a_idx].position += correction * inv_mass_a;
        }
        if !self.bodies[body_b_idx].is_static {
            self.bodies[body_b_idx].position -= correction * inv_mass_b;
        }
    }

    /// Solve a point constraint.
    fn solve_point_constraint(&mut self, constraint: &PointConstraint) {
        let body_idx = constraint.body;
        if body_idx >= self.bodies.len() {
            return;
        }

        // Get world space anchor position on body
        let anchor_world = self.bodies[body_idx].position
            + self.bodies[body_idx].orientation * constraint.local_anchor;

        // Get target position
        let target_pos = match &constraint.target {
            PointConstraintTarget::World(pos) => *pos,
            PointConstraintTarget::Body { body, local_anchor } => {
                if *body >= self.bodies.len() {
                    return;
                }
                self.bodies[*body].position + self.bodies[*body].orientation * *local_anchor
            }
        };

        let delta = target_pos - anchor_world;
        let error = delta.length();

        if error < 0.0001 {
            return;
        }

        match &constraint.target {
            PointConstraintTarget::World(_) => {
                // Anchor to world - body moves toward target
                if !self.bodies[body_idx].is_static {
                    self.bodies[body_idx].position += delta * constraint.stiffness;
                }
            }
            PointConstraintTarget::Body {
                body: other_idx, ..
            } => {
                // Ball joint between two bodies
                let inv_mass_a = self.bodies[body_idx].inv_mass;
                let inv_mass_b = self.bodies[*other_idx].inv_mass;
                let total_inv_mass = inv_mass_a + inv_mass_b;

                if total_inv_mass < 0.0001 {
                    return;
                }

                let correction = delta * constraint.stiffness / total_inv_mass;

                if !self.bodies[body_idx].is_static {
                    self.bodies[body_idx].position += correction * inv_mass_a;
                }
                if !self.bodies[*other_idx].is_static {
                    self.bodies[*other_idx].position -= correction * inv_mass_b;
                }
            }
        }
    }

    /// Solve a hinge constraint.
    fn solve_hinge_constraint(&mut self, constraint: &HingeConstraint) {
        let body_a_idx = constraint.body_a;
        let body_b_idx = constraint.body_b;

        if body_a_idx >= self.bodies.len() || body_b_idx >= self.bodies.len() {
            return;
        }

        // First, solve the point constraint to keep anchors together
        let pos_a = self.bodies[body_a_idx].position
            + self.bodies[body_a_idx].orientation * constraint.local_anchor_a;
        let pos_b = self.bodies[body_b_idx].position
            + self.bodies[body_b_idx].orientation * constraint.local_anchor_b;

        let delta = pos_b - pos_a;

        if delta.length() > 0.0001 {
            let inv_mass_a = self.bodies[body_a_idx].inv_mass;
            let inv_mass_b = self.bodies[body_b_idx].inv_mass;
            let total_inv_mass = inv_mass_a + inv_mass_b;

            if total_inv_mass > 0.0001 {
                let correction = delta * constraint.stiffness / total_inv_mass;

                if !self.bodies[body_a_idx].is_static {
                    self.bodies[body_a_idx].position += correction * inv_mass_a;
                }
                if !self.bodies[body_b_idx].is_static {
                    self.bodies[body_b_idx].position -= correction * inv_mass_b;
                }
            }
        }

        // Now solve the angular constraint to align axes
        let world_axis_a = self.bodies[body_a_idx].orientation * constraint.local_axis_a;
        let world_axis_b = self.bodies[body_b_idx].orientation * constraint.local_axis_b;

        // Cross product gives rotation axis, dot product gives alignment
        let axis_error = world_axis_a.cross(world_axis_b);

        if axis_error.length() > 0.0001 {
            // Apply angular correction to align axes
            let correction = axis_error * constraint.stiffness * 0.5;

            if !self.bodies[body_a_idx].is_static {
                let q = self.bodies[body_a_idx].orientation;
                let dq = Quat::from_xyzw(correction.x, correction.y, correction.z, 0.0) * q * 0.5;
                self.bodies[body_a_idx].orientation = (q + dq).normalize();
            }
            if !self.bodies[body_b_idx].is_static {
                let q = self.bodies[body_b_idx].orientation;
                let dq =
                    Quat::from_xyzw(-correction.x, -correction.y, -correction.z, 0.0) * q * 0.5;
                self.bodies[body_b_idx].orientation = (q + dq).normalize();
            }
        }

        // Apply angle limits if set
        if let Some((min_angle, max_angle)) = constraint.limits {
            // Compute relative rotation angle around hinge axis
            let rel_quat =
                self.bodies[body_b_idx].orientation * self.bodies[body_a_idx].orientation.inverse();

            // Project to rotation around hinge axis
            let (axis, angle) = rel_quat.to_axis_angle();
            let projected_angle = angle * axis.dot(world_axis_a);

            if projected_angle < min_angle {
                let correction_angle = min_angle - projected_angle;
                let correction_quat = Quat::from_axis_angle(world_axis_a, correction_angle * 0.5);
                if !self.bodies[body_b_idx].is_static {
                    self.bodies[body_b_idx].orientation =
                        correction_quat * self.bodies[body_b_idx].orientation;
                }
            } else if projected_angle > max_angle {
                let correction_angle = max_angle - projected_angle;
                let correction_quat = Quat::from_axis_angle(world_axis_a, correction_angle * 0.5);
                if !self.bodies[body_b_idx].is_static {
                    self.bodies[body_b_idx].orientation =
                        correction_quat * self.bodies[body_b_idx].orientation;
                }
            }
        }
    }

    /// Detect all collisions between bodies.
    fn detect_collisions(&self) -> Vec<Contact> {
        let mut contacts = Vec::new();

        for i in 0..self.bodies.len() {
            for j in (i + 1)..self.bodies.len() {
                if self.bodies[i].is_static && self.bodies[j].is_static {
                    continue;
                }

                if let Some(contact) = self.test_collision(i, j) {
                    contacts.push(contact);
                }
            }
        }

        contacts
    }

    /// Test collision between two bodies.
    fn test_collision(&self, a: usize, b: usize) -> Option<Contact> {
        let body_a = &self.bodies[a];
        let body_b = &self.bodies[b];

        match (&body_a.collider, &body_b.collider) {
            (Collider::Sphere { radius: r1 }, Collider::Sphere { radius: r2 }) => {
                sphere_sphere(a, b, body_a.position, *r1, body_b.position, *r2)
            }
            (Collider::Sphere { radius }, Collider::Plane { normal, distance }) => {
                sphere_plane(a, b, body_a.position, *radius, *normal, *distance)
            }
            (Collider::Plane { normal, distance }, Collider::Sphere { radius }) => {
                sphere_plane(b, a, body_b.position, *radius, *normal, *distance).map(Contact::flip)
            }
            (Collider::Box { half_extents }, Collider::Plane { normal, distance }) => box_plane(
                a,
                b,
                body_a.position,
                body_a.orientation,
                *half_extents,
                *normal,
                *distance,
            ),
            (Collider::Plane { normal, distance }, Collider::Box { half_extents }) => box_plane(
                b,
                a,
                body_b.position,
                body_b.orientation,
                *half_extents,
                *normal,
                *distance,
            )
            .map(Contact::flip),
            (Collider::Sphere { radius }, Collider::Box { half_extents }) => sphere_box(
                a,
                b,
                body_a.position,
                *radius,
                body_b.position,
                body_b.orientation,
                *half_extents,
            ),
            (Collider::Box { half_extents }, Collider::Sphere { radius }) => sphere_box(
                b,
                a,
                body_b.position,
                *radius,
                body_a.position,
                body_a.orientation,
                *half_extents,
            )
            .map(Contact::flip),
            (Collider::Box { half_extents: he1 }, Collider::Box { half_extents: he2 }) => {
                box_box_aabb(a, b, body_a.position, *he1, body_b.position, *he2)
            }
            _ => None,
        }
    }

    /// Resolve a contact constraint using impulse.
    fn resolve_contact(&mut self, contact: &Contact) {
        let (body_a, body_b) = {
            let (left, right) = self.bodies.split_at_mut(contact.body_b);
            (&mut left[contact.body_a], &mut right[0])
        };

        // Relative velocity at contact point
        let r_a = contact.point - body_a.position;
        let r_b = contact.point - body_b.position;
        let v_a = body_a.velocity_at_point(contact.point);
        let v_b = body_b.velocity_at_point(contact.point);
        let relative_vel = v_a - v_b;

        // Relative velocity along normal
        let vn = relative_vel.dot(contact.normal);

        // Don't resolve if separating
        if vn > 0.0 {
            return;
        }

        // Combined restitution
        let e = (body_a.restitution + body_b.restitution) * 0.5;

        // Compute impulse denominator
        let rn_a = r_a.cross(contact.normal);
        let rn_b = r_b.cross(contact.normal);
        let k = body_a.inv_mass
            + body_b.inv_mass
            + rn_a.dot(body_a.inv_inertia * rn_a)
            + rn_b.dot(body_b.inv_inertia * rn_b);

        // Compute normal impulse
        let j = -(1.0 + e) * vn / k;
        let impulse = contact.normal * j;

        // Apply impulse
        body_a.apply_impulse_at_point(impulse, contact.point);
        body_b.apply_impulse_at_point(-impulse, contact.point);

        // Positional correction (prevent sinking)
        let slop = 0.01;
        let percent = 0.8;
        let correction = contact.normal * (contact.depth - slop).max(0.0) * percent
            / (body_a.inv_mass + body_b.inv_mass);

        if !body_a.is_static {
            body_a.position += correction * body_a.inv_mass;
        }
        if !body_b.is_static {
            body_b.position -= correction * body_b.inv_mass;
        }

        // Friction (simplified)
        let tangent = (relative_vel - contact.normal * vn).normalize_or_zero();
        if tangent.length_squared() > 0.0 {
            let vt = relative_vel.dot(tangent);
            let mu = (body_a.friction + body_b.friction) * 0.5;
            let jt = (-vt / k).clamp(-j * mu, j * mu);
            let friction_impulse = tangent * jt;

            body_a.apply_impulse_at_point(friction_impulse, contact.point);
            body_b.apply_impulse_at_point(-friction_impulse, contact.point);
        }
    }

    /// Get positions of all bodies.
    pub fn positions(&self) -> Vec<Vec3> {
        self.bodies.iter().map(|b| b.position).collect()
    }

    /// Get orientations of all bodies.
    pub fn orientations(&self) -> Vec<Quat> {
        self.bodies.iter().map(|b| b.orientation).collect()
    }
}

// ============================================================================
// Collision Detection Functions
// ============================================================================

fn sphere_sphere(
    a: usize,
    b: usize,
    pos_a: Vec3,
    radius_a: f32,
    pos_b: Vec3,
    radius_b: f32,
) -> Option<Contact> {
    let d = pos_b - pos_a;
    let dist_sq = d.length_squared();
    let radius_sum = radius_a + radius_b;

    if dist_sq < radius_sum * radius_sum {
        let dist = dist_sq.sqrt();
        // Normal points from A toward B
        let normal = if dist > 0.0 { d / dist } else { Vec3::Y };
        let depth = radius_sum - dist;
        let point = pos_a + normal * radius_a;

        Some(Contact {
            body_a: a,
            body_b: b,
            point,
            normal,
            depth,
        })
    } else {
        None
    }
}

fn sphere_plane(
    sphere_idx: usize,
    plane_idx: usize,
    sphere_pos: Vec3,
    radius: f32,
    plane_normal: Vec3,
    plane_dist: f32,
) -> Option<Contact> {
    let dist = sphere_pos.dot(plane_normal) - plane_dist;

    if dist < radius {
        let depth = radius - dist;
        let point = sphere_pos - plane_normal * dist;

        Some(Contact {
            body_a: sphere_idx,
            body_b: plane_idx,
            point,
            normal: plane_normal,
            depth,
        })
    } else {
        None
    }
}

fn box_plane(
    box_idx: usize,
    plane_idx: usize,
    box_pos: Vec3,
    box_rot: Quat,
    half_extents: Vec3,
    plane_normal: Vec3,
    plane_dist: f32,
) -> Option<Contact> {
    // Get box axes
    let rot = Mat3::from_quat(box_rot);
    let axes = [
        rot.col(0) * half_extents.x,
        rot.col(1) * half_extents.y,
        rot.col(2) * half_extents.z,
    ];

    // Find the vertex furthest in the direction of -normal
    let mut min_dist = f32::MAX;
    let mut min_point = Vec3::ZERO;

    for sx in [-1.0_f32, 1.0] {
        for sy in [-1.0_f32, 1.0] {
            for sz in [-1.0_f32, 1.0] {
                let vertex = box_pos + axes[0] * sx + axes[1] * sy + axes[2] * sz;
                let dist = vertex.dot(plane_normal) - plane_dist;
                if dist < min_dist {
                    min_dist = dist;
                    min_point = vertex;
                }
            }
        }
    }

    if min_dist < 0.0 {
        Some(Contact {
            body_a: box_idx,
            body_b: plane_idx,
            point: min_point,
            normal: plane_normal,
            depth: -min_dist,
        })
    } else {
        None
    }
}

fn sphere_box(
    sphere_idx: usize,
    box_idx: usize,
    sphere_pos: Vec3,
    radius: f32,
    box_pos: Vec3,
    box_rot: Quat,
    half_extents: Vec3,
) -> Option<Contact> {
    // Transform sphere to box local space
    let inv_rot = box_rot.inverse();
    let local_pos = inv_rot * (sphere_pos - box_pos);

    // Find closest point on box
    let clamped = Vec3::new(
        local_pos.x.clamp(-half_extents.x, half_extents.x),
        local_pos.y.clamp(-half_extents.y, half_extents.y),
        local_pos.z.clamp(-half_extents.z, half_extents.z),
    );

    let diff = local_pos - clamped;
    let dist_sq = diff.length_squared();

    if dist_sq < radius * radius {
        let dist = dist_sq.sqrt();
        let local_normal = if dist > 0.0 {
            diff / dist
        } else {
            // Sphere center inside box - push out along shortest axis
            let penetrations = half_extents - local_pos.abs();
            if penetrations.x < penetrations.y && penetrations.x < penetrations.z {
                Vec3::X * local_pos.x.signum()
            } else if penetrations.y < penetrations.z {
                Vec3::Y * local_pos.y.signum()
            } else {
                Vec3::Z * local_pos.z.signum()
            }
        };

        let normal = box_rot * local_normal;
        let point = box_pos + box_rot * clamped;
        let depth = radius - dist;

        Some(Contact {
            body_a: sphere_idx,
            body_b: box_idx,
            point,
            normal,
            depth,
        })
    } else {
        None
    }
}

fn box_box_aabb(
    a: usize,
    b: usize,
    pos_a: Vec3,
    he_a: Vec3,
    pos_b: Vec3,
    he_b: Vec3,
) -> Option<Contact> {
    // Simple AABB vs AABB (ignores rotation)
    let overlap = Vec3::new(
        (he_a.x + he_b.x) - (pos_b.x - pos_a.x).abs(),
        (he_a.y + he_b.y) - (pos_b.y - pos_a.y).abs(),
        (he_a.z + he_b.z) - (pos_b.z - pos_a.z).abs(),
    );

    if overlap.x > 0.0 && overlap.y > 0.0 && overlap.z > 0.0 {
        // Find axis of minimum penetration
        let min_overlap = overlap.x.min(overlap.y).min(overlap.z);
        let normal;
        let depth;

        if min_overlap == overlap.x {
            normal = Vec3::X * (pos_b.x - pos_a.x).signum();
            depth = overlap.x;
        } else if min_overlap == overlap.y {
            normal = Vec3::Y * (pos_b.y - pos_a.y).signum();
            depth = overlap.y;
        } else {
            normal = Vec3::Z * (pos_b.z - pos_a.z).signum();
            depth = overlap.z;
        }

        let point = (pos_a + pos_b) * 0.5;

        Some(Contact {
            body_a: a,
            body_b: b,
            point,
            normal,
            depth,
        })
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rigid_body_creation() {
        let body = RigidBody::new(Vec3::ZERO, Collider::sphere(1.0), 1.0);
        assert_eq!(body.mass, 1.0);
        assert!(!body.is_static);
    }

    #[test]
    fn test_static_body() {
        let body = RigidBody::new_static(Vec3::ZERO, Collider::ground());
        assert!(body.is_static);
        assert_eq!(body.inv_mass, 0.0);
    }

    #[test]
    fn test_apply_force() {
        let mut body = RigidBody::new(Vec3::ZERO, Collider::sphere(1.0), 1.0);
        body.apply_force(Vec3::new(10.0, 0.0, 0.0));
        assert_eq!(body.force, Vec3::new(10.0, 0.0, 0.0));
    }

    #[test]
    fn test_apply_impulse() {
        let mut body = RigidBody::new(Vec3::ZERO, Collider::sphere(1.0), 1.0);
        body.apply_impulse(Vec3::new(10.0, 0.0, 0.0));
        assert_eq!(body.velocity, Vec3::new(10.0, 0.0, 0.0));
    }

    #[test]
    fn test_physics_world_creation() {
        let world = PhysicsWorld::new(PhysicsConfig::default());
        assert_eq!(world.bodies.len(), 0);
    }

    #[test]
    fn test_add_body() {
        let mut world = PhysicsWorld::new(PhysicsConfig::default());
        let idx = world.add_body(RigidBody::new(Vec3::Y * 5.0, Collider::sphere(1.0), 1.0));
        assert_eq!(idx, 0);
        assert_eq!(world.bodies.len(), 1);
    }

    #[test]
    fn test_gravity() {
        let mut world = PhysicsWorld::new(PhysicsConfig::default());
        world.add_body(RigidBody::new(Vec3::Y * 5.0, Collider::sphere(1.0), 1.0));

        let initial_y = world.bodies[0].position.y;
        world.step();
        let final_y = world.bodies[0].position.y;

        // Should have fallen
        assert!(final_y < initial_y);
    }

    #[test]
    fn test_sphere_ground_collision() {
        let mut world = PhysicsWorld::new(PhysicsConfig::default());

        // Add ground
        world.add_body(RigidBody::new_static(Vec3::ZERO, Collider::ground()));

        // Add falling sphere
        world.add_body(RigidBody::new(Vec3::Y * 2.0, Collider::sphere(1.0), 1.0));

        // Run simulation
        for _ in 0..100 {
            world.step();
        }

        // Sphere should be resting on ground (radius = 1, so y should be around 1)
        let y = world.bodies[1].position.y;
        assert!(y >= 0.9 && y <= 1.5, "y = {}", y);
    }

    #[test]
    fn test_sphere_sphere_collision() {
        // Test that overlapping spheres get separated by positional correction
        let mut world = PhysicsWorld::new(PhysicsConfig {
            gravity: Vec3::ZERO,
            dt: 1.0 / 60.0,
            solver_iterations: 10,
        });

        // Two overlapping spheres (radius 1 each, centers 1.5 apart = 0.5 overlap)
        let body1 = RigidBody::new(Vec3::ZERO, Collider::sphere(1.0), 1.0);
        world.add_body(body1);

        let body2 = RigidBody::new(Vec3::X * 1.5, Collider::sphere(1.0), 1.0);
        world.add_body(body2);

        let initial_dist = (world.bodies[1].position - world.bodies[0].position).length();

        // Run several steps to let collision push them apart
        for _ in 0..10 {
            world.step();
        }

        let final_dist = (world.bodies[1].position - world.bodies[0].position).length();

        // Spheres should be pushed apart
        assert!(
            final_dist > initial_dist,
            "final_dist = {} should be > initial_dist = {}",
            final_dist,
            initial_dist
        );
    }

    #[test]
    fn test_box_creation() {
        let body = RigidBody::new(Vec3::ZERO, Collider::box_shape(Vec3::ONE), 1.0);
        assert!(!body.is_static);
    }

    #[test]
    fn test_box_ground_collision() {
        let mut world = PhysicsWorld::new(PhysicsConfig::default());

        // Add ground
        world.add_body(RigidBody::new_static(Vec3::ZERO, Collider::ground()));

        // Add falling box
        world.add_body(RigidBody::new(
            Vec3::Y * 3.0,
            Collider::box_shape(Vec3::splat(0.5)),
            1.0,
        ));

        // Run simulation
        for _ in 0..100 {
            world.step();
        }

        // Box should be resting on ground
        let y = world.bodies[1].position.y;
        assert!(y >= 0.4 && y <= 1.0, "y = {}", y);
    }

    #[test]
    fn test_stacking() {
        let mut world = PhysicsWorld::new(PhysicsConfig::default());

        // Add ground
        world.add_body(RigidBody::new_static(Vec3::ZERO, Collider::ground()));

        // Add stacked boxes (more stable than spheres)
        let mut b1 = RigidBody::new(Vec3::Y * 0.5, Collider::box_shape(Vec3::splat(0.5)), 1.0);
        b1.restitution = 0.0;
        world.add_body(b1);

        let mut b2 = RigidBody::new(Vec3::Y * 1.5, Collider::box_shape(Vec3::splat(0.5)), 1.0);
        b2.restitution = 0.0;
        world.add_body(b2);

        let mut b3 = RigidBody::new(Vec3::Y * 2.5, Collider::box_shape(Vec3::splat(0.5)), 1.0);
        b3.restitution = 0.0;
        world.add_body(b3);

        // Run simulation until settled
        for _ in 0..200 {
            world.step();
        }

        // All boxes should be above ground
        let y1 = world.bodies[1].position.y;
        let y2 = world.bodies[2].position.y;
        let y3 = world.bodies[3].position.y;

        // Basic check: all should be above ground
        assert!(y1 > 0.3, "y1 = {} should be above ground", y1);
        assert!(y2 > 0.3, "y2 = {} should be above ground", y2);
        assert!(y3 > 0.3, "y3 = {} should be above ground", y3);
    }

    #[test]
    fn test_friction() {
        let mut world = PhysicsWorld::new(PhysicsConfig {
            gravity: Vec3::new(0.0, -9.81, 0.0),
            ..Default::default()
        });

        // Add tilted ground (simulated by giving sphere sideways velocity on flat ground)
        world.add_body(RigidBody::new_static(Vec3::ZERO, Collider::ground()));

        // Add sphere with horizontal velocity
        let mut body = RigidBody::new(Vec3::Y * 1.0, Collider::sphere(1.0), 1.0);
        body.velocity = Vec3::X * 5.0;
        body.friction = 0.5;
        world.add_body(body);

        // Let it settle and slide
        for _ in 0..100 {
            world.step();
        }

        // Should have slowed down due to friction
        let vx = world.bodies[1].velocity.x;
        assert!(vx < 4.0, "vx = {} should have slowed", vx);
    }

    // ========================================================================
    // Constraint Tests
    // ========================================================================

    #[test]
    fn test_distance_constraint_creation() {
        let constraint = DistanceConstraint::new(0, 1, 2.0)
            .with_anchors(Vec3::X, Vec3::NEG_X)
            .with_stiffness(0.5);

        assert_eq!(constraint.body_a, 0);
        assert_eq!(constraint.body_b, 1);
        assert_eq!(constraint.distance, 2.0);
        assert_eq!(constraint.stiffness, 0.5);
    }

    #[test]
    fn test_distance_constraint() {
        let mut world = PhysicsWorld::new(PhysicsConfig {
            gravity: Vec3::ZERO,
            dt: 1.0 / 60.0,
            solver_iterations: 20,
        });

        // Two spheres at distance 5, constrained to distance 3
        let a = world.add_body(RigidBody::new(Vec3::ZERO, Collider::sphere(0.5), 1.0));
        let b = world.add_body(RigidBody::new(Vec3::X * 5.0, Collider::sphere(0.5), 1.0));

        world.add_distance_constraint(a, b, 3.0);

        // Run simulation
        for _ in 0..100 {
            world.step();
        }

        // Bodies should be pulled closer to distance 3
        let dist = (world.bodies[b].position - world.bodies[a].position).length();
        assert!(
            (dist - 3.0).abs() < 0.5,
            "dist = {} should be close to 3.0",
            dist
        );
    }

    #[test]
    fn test_point_constraint_to_world() {
        let mut world = PhysicsWorld::new(PhysicsConfig {
            gravity: Vec3::ZERO,
            dt: 1.0 / 60.0,
            solver_iterations: 20,
        });

        // Sphere at (5, 0, 0) anchored to origin
        let a = world.add_body(RigidBody::new(Vec3::X * 5.0, Collider::sphere(0.5), 1.0));

        world.add_point_constraint_to_world(a, Vec3::ZERO, Vec3::ZERO);

        // Run simulation
        for _ in 0..100 {
            world.step();
        }

        // Body should be pulled to origin
        let dist = world.bodies[a].position.length();
        assert!(dist < 0.5, "dist = {} should be close to 0", dist);
    }

    #[test]
    fn test_ball_joint() {
        let mut world = PhysicsWorld::new(PhysicsConfig {
            gravity: Vec3::new(0.0, -9.81, 0.0),
            dt: 1.0 / 60.0,
            solver_iterations: 20,
        });

        // Fixed sphere at origin
        let a = world.add_body(RigidBody::new_static(Vec3::ZERO, Collider::sphere(0.5)));

        // Dynamic sphere below, connected by ball joint
        let b = world.add_body(RigidBody::new(
            Vec3::new(0.0, -2.0, 0.0),
            Collider::sphere(0.5),
            1.0,
        ));

        // Connect at bottom of A to top of B
        world.add_ball_joint(a, Vec3::Y * -0.5, b, Vec3::Y * 0.5);

        // Run simulation
        for _ in 0..200 {
            world.step();
        }

        // The anchor points should be close together
        let anchor_a = world.bodies[a].position + Vec3::Y * -0.5;
        let anchor_b = world.bodies[b].position + Vec3::Y * 0.5;
        let anchor_dist = (anchor_b - anchor_a).length();

        assert!(
            anchor_dist < 0.3,
            "anchor distance = {} should be small",
            anchor_dist
        );
    }

    #[test]
    fn test_spring_constraint() {
        let mut world = PhysicsWorld::new(PhysicsConfig {
            gravity: Vec3::ZERO,
            dt: 1.0 / 60.0,
            solver_iterations: 10,
        });

        // Two spheres: one fixed, one connected by spring
        let a = world.add_body(RigidBody::new_static(Vec3::ZERO, Collider::sphere(0.5)));
        let b = world.add_body(RigidBody::new(Vec3::X * 5.0, Collider::sphere(0.5), 1.0));

        // Spring with rest length 2, should pull body B toward A
        world.add_spring(a, b, 2.0, 50.0, 5.0);

        let initial_dist = world.bodies[b].position.length();

        // Run simulation
        for _ in 0..200 {
            world.step();
        }

        // Body B should have moved closer (spring pulls it toward rest length)
        let final_dist = world.bodies[b].position.length();
        assert!(
            final_dist < initial_dist,
            "Body should move closer: initial={}, final={}",
            initial_dist,
            final_dist
        );
    }

    #[test]
    fn test_spring_oscillation() {
        let mut world = PhysicsWorld::new(PhysicsConfig {
            gravity: Vec3::ZERO,
            dt: 1.0 / 60.0,
            solver_iterations: 10,
        });

        // Fixed anchor
        let a = world.add_body(RigidBody::new_static(Vec3::ZERO, Collider::sphere(0.1)));

        // Mass on spring - start stretched
        let b = world.add_body(RigidBody::new(Vec3::X * 3.0, Collider::sphere(0.1), 1.0));

        // Low damping spring - should oscillate
        world.add_spring(a, b, 1.0, 100.0, 0.5);

        // Track positions over time
        let mut crossed_rest = false;
        let mut min_x = 3.0_f32;

        for _ in 0..300 {
            world.step();
            let x = world.bodies[b].position.x;
            min_x = min_x.min(x);
            if x < 1.0 {
                crossed_rest = true;
            }
        }

        // Should have oscillated past rest length
        assert!(
            crossed_rest,
            "Spring should oscillate past rest length, min_x = {}",
            min_x
        );
    }

    #[test]
    fn test_hinge_constraint() {
        let mut world = PhysicsWorld::new(PhysicsConfig {
            gravity: Vec3::ZERO,
            dt: 1.0 / 60.0,
            solver_iterations: 20,
        });

        // Two boxes connected by hinge
        let a = world.add_body(RigidBody::new_static(
            Vec3::ZERO,
            Collider::box_shape(Vec3::ONE),
        ));
        let b = world.add_body(RigidBody::new(
            Vec3::X * 3.0,
            Collider::box_shape(Vec3::ONE),
            1.0,
        ));

        // Hinge at right side of A, left side of B
        world.add_constraint(Constraint::Hinge(
            HingeConstraint::new(a, b)
                .with_anchors(Vec3::X * 1.0, Vec3::X * -1.0)
                .with_axes(Vec3::Y, Vec3::Y),
        ));

        // Run simulation
        for _ in 0..100 {
            world.step();
        }

        // Anchors should be close together
        let anchor_a = world.bodies[a].position + world.bodies[a].orientation * Vec3::X;
        let anchor_b = world.bodies[b].position + world.bodies[b].orientation * Vec3::X * -1.0;
        let dist = (anchor_b - anchor_a).length();

        assert!(
            dist < 0.5,
            "Hinge anchor distance = {} should be small",
            dist
        );
    }

    #[test]
    fn test_pendulum_chain() {
        let mut world = PhysicsWorld::new(PhysicsConfig {
            gravity: Vec3::new(0.0, -9.81, 0.0),
            dt: 1.0 / 60.0,
            solver_iterations: 20,
        });

        // Fixed anchor point
        let anchor = world.add_body(RigidBody::new_static(Vec3::Y * 5.0, Collider::sphere(0.1)));

        // Chain of 3 spheres connected by distance constraints
        let mut prev = anchor;
        let mut bodies = vec![anchor];

        for i in 0..3 {
            let pos = Vec3::new((i + 1) as f32, 5.0, 0.0);
            let body = world.add_body(RigidBody::new(pos, Collider::sphere(0.3), 1.0));
            world.add_distance_constraint(prev, body, 1.0);
            bodies.push(body);
            prev = body;
        }

        // Run simulation - chain should fall and swing
        for _ in 0..200 {
            world.step();
        }

        // All bodies should be below the anchor
        for &idx in &bodies[1..] {
            let y = world.bodies[idx].position.y;
            assert!(y < 5.0, "Body {} should be below anchor, y = {}", idx, y);
        }
    }

    #[test]
    fn test_constraint_removal() {
        let mut world = PhysicsWorld::new(PhysicsConfig::default());

        let a = world.add_body(RigidBody::new(Vec3::ZERO, Collider::sphere(0.5), 1.0));
        let b = world.add_body(RigidBody::new(Vec3::X * 2.0, Collider::sphere(0.5), 1.0));

        let c_idx = world.add_distance_constraint(a, b, 1.0);
        assert_eq!(world.constraints.len(), 1);

        world.remove_constraint(c_idx);
        assert_eq!(world.constraints.len(), 0);
    }

    #[test]
    fn test_multiple_constraints() {
        let mut world = PhysicsWorld::new(PhysicsConfig {
            gravity: Vec3::ZERO,
            dt: 1.0 / 60.0,
            solver_iterations: 20,
        });

        // Triangle of 3 bodies with distance constraints
        let a = world.add_body(RigidBody::new(Vec3::ZERO, Collider::sphere(0.2), 1.0));
        let b = world.add_body(RigidBody::new(Vec3::X * 3.0, Collider::sphere(0.2), 1.0));
        let c = world.add_body(RigidBody::new(Vec3::Y * 3.0, Collider::sphere(0.2), 1.0));

        // Target: equilateral-ish triangle with side length 2
        world.add_distance_constraint(a, b, 2.0);
        world.add_distance_constraint(b, c, 2.0);
        world.add_distance_constraint(c, a, 2.0);

        // Run simulation
        for _ in 0..200 {
            world.step();
        }

        // Check all distances are close to 2
        let d_ab = (world.bodies[b].position - world.bodies[a].position).length();
        let d_bc = (world.bodies[c].position - world.bodies[b].position).length();
        let d_ca = (world.bodies[a].position - world.bodies[c].position).length();

        assert!((d_ab - 2.0).abs() < 0.3, "d_ab = {} should be ~2.0", d_ab);
        assert!((d_bc - 2.0).abs() < 0.3, "d_bc = {} should be ~2.0", d_bc);
        assert!((d_ca - 2.0).abs() < 0.3, "d_ca = {} should be ~2.0", d_ca);
    }
}

/// Invariant tests for physics simulation.
///
/// Run with: cargo test -p unshape-physics --features invariant-tests
#[cfg(all(test, feature = "invariant-tests"))]
mod invariant_tests {
    use super::*;

    /// Calculate kinetic energy of a body: KE = 0.5 * m * v^2 + 0.5 * I * w^2
    fn kinetic_energy(body: &RigidBody) -> f32 {
        if body.is_static {
            return 0.0;
        }
        let linear_ke = 0.5 * body.mass * body.velocity.length_squared();
        let angular_ke = 0.5
            * body
                .inertia
                .dot(body.angular_velocity * body.angular_velocity);
        linear_ke + angular_ke
    }

    /// Calculate potential energy relative to y=0: PE = m * g * h
    fn potential_energy(body: &RigidBody, gravity_magnitude: f32) -> f32 {
        if body.is_static {
            return 0.0;
        }
        body.mass * gravity_magnitude * body.position.y
    }

    /// Calculate total mechanical energy of the system.
    fn total_energy(world: &PhysicsWorld) -> f32 {
        let gravity_mag = world.config.gravity.length();
        world
            .bodies
            .iter()
            .map(|b| kinetic_energy(b) + potential_energy(b, gravity_mag))
            .sum()
    }

    /// Calculate total momentum of the system.
    fn total_momentum(world: &PhysicsWorld) -> Vec3 {
        world
            .bodies
            .iter()
            .filter(|b| !b.is_static)
            .map(|b| b.velocity * b.mass)
            .sum()
    }

    // ========================================================================
    // Energy Conservation Tests
    // ========================================================================

    #[test]
    fn invariant_energy_conservation_free_fall() {
        // In free fall (no collisions, no damping), mechanical energy should be conserved.
        let mut world = PhysicsWorld::new(PhysicsConfig {
            gravity: Vec3::new(0.0, -9.81, 0.0),
            dt: 1.0 / 120.0, // Small timestep for accuracy
            solver_iterations: 10,
        });

        // Single falling sphere with zero damping
        let mut body = RigidBody::new(Vec3::Y * 10.0, Collider::sphere(1.0), 1.0);
        body.linear_damping = 0.0;
        body.angular_damping = 0.0;
        world.add_body(body);

        let initial_energy = total_energy(&world);

        // Simulate for a short time (before hitting anything)
        for _ in 0..60 {
            world.step();
        }

        let final_energy = total_energy(&world);

        // Energy should be conserved within numerical tolerance
        let energy_error = (final_energy - initial_energy).abs() / initial_energy.abs();
        assert!(
            energy_error < 0.01,
            "Energy conservation violated: initial={}, final={}, error={}%",
            initial_energy,
            final_energy,
            energy_error * 100.0
        );
    }

    #[test]
    fn invariant_energy_conservation_zero_gravity() {
        // With no gravity and no damping, kinetic energy should be conserved.
        let mut world = PhysicsWorld::new(PhysicsConfig {
            gravity: Vec3::ZERO,
            dt: 1.0 / 60.0,
            solver_iterations: 10,
        });

        // Moving sphere with initial velocity
        let mut body = RigidBody::new(Vec3::ZERO, Collider::sphere(1.0), 1.0);
        body.velocity = Vec3::new(5.0, 3.0, -2.0);
        body.angular_velocity = Vec3::new(1.0, 0.5, 0.0);
        body.linear_damping = 0.0;
        body.angular_damping = 0.0;
        world.add_body(body);

        let initial_ke = kinetic_energy(&world.bodies[0]);

        // Simulate
        for _ in 0..100 {
            world.step();
        }

        let final_ke = kinetic_energy(&world.bodies[0]);

        // Kinetic energy should be conserved
        let ke_error = (final_ke - initial_ke).abs();
        assert!(
            ke_error < 0.001,
            "Kinetic energy should be conserved: initial={}, final={}, error={}",
            initial_ke,
            final_ke,
            ke_error
        );
    }

    // ========================================================================
    // Momentum Conservation Tests
    // ========================================================================

    #[test]
    fn invariant_momentum_conservation_isolated_system() {
        // In an isolated system (no external forces), total momentum is conserved.
        let mut world = PhysicsWorld::new(PhysicsConfig {
            gravity: Vec3::ZERO,
            dt: 1.0 / 60.0,
            solver_iterations: 10,
        });

        // Two spheres moving toward each other
        let mut body1 = RigidBody::new(Vec3::new(-5.0, 0.0, 0.0), Collider::sphere(1.0), 2.0);
        body1.velocity = Vec3::new(3.0, 0.0, 0.0);
        body1.linear_damping = 0.0;
        world.add_body(body1);

        let mut body2 = RigidBody::new(Vec3::new(5.0, 0.0, 0.0), Collider::sphere(1.0), 1.0);
        body2.velocity = Vec3::new(-2.0, 0.0, 0.0);
        body2.linear_damping = 0.0;
        world.add_body(body2);

        let initial_momentum = total_momentum(&world);

        // Simulate (they may or may not collide)
        for _ in 0..100 {
            world.step();
        }

        let final_momentum = total_momentum(&world);

        // Momentum should be conserved
        let momentum_error = (final_momentum - initial_momentum).length();
        assert!(
            momentum_error < 0.01,
            "Momentum conservation violated: initial={:?}, final={:?}, error={}",
            initial_momentum,
            final_momentum,
            momentum_error
        );
    }

    #[test]
    fn invariant_momentum_conservation_collision() {
        // Test momentum conservation through a collision.
        let mut world = PhysicsWorld::new(PhysicsConfig {
            gravity: Vec3::ZERO,
            dt: 1.0 / 60.0,
            solver_iterations: 20,
        });

        // Two spheres that will collide
        let mut body1 = RigidBody::new(Vec3::new(-2.0, 0.0, 0.0), Collider::sphere(1.0), 1.0);
        body1.velocity = Vec3::new(5.0, 0.0, 0.0);
        body1.linear_damping = 0.0;
        body1.restitution = 1.0; // Elastic collision
        world.add_body(body1);

        let mut body2 = RigidBody::new(Vec3::new(2.0, 0.0, 0.0), Collider::sphere(1.0), 1.0);
        body2.velocity = Vec3::new(-5.0, 0.0, 0.0);
        body2.linear_damping = 0.0;
        body2.restitution = 1.0;
        world.add_body(body2);

        let initial_momentum = total_momentum(&world);

        // Simulate through collision
        for _ in 0..100 {
            world.step();
        }

        let final_momentum = total_momentum(&world);

        let momentum_error = (final_momentum - initial_momentum).length();
        assert!(
            momentum_error < 0.1,
            "Momentum not conserved through collision: initial={:?}, final={:?}",
            initial_momentum,
            final_momentum
        );
    }

    // ========================================================================
    // Collision Detection Tests
    // ========================================================================

    #[test]
    fn invariant_overlapping_spheres_detected() {
        // Overlapping spheres should always be detected as colliding.
        let world = {
            let mut w = PhysicsWorld::new(PhysicsConfig::default());

            // Two overlapping spheres (radius 1 each, centers 1.0 apart)
            w.add_body(RigidBody::new(Vec3::ZERO, Collider::sphere(1.0), 1.0));
            w.add_body(RigidBody::new(Vec3::X * 1.0, Collider::sphere(1.0), 1.0));
            w
        };

        let contacts = world.detect_collisions();

        assert!(
            !contacts.is_empty(),
            "Overlapping spheres should be detected"
        );
        assert_eq!(contacts.len(), 1, "Should have exactly one contact");

        let contact = &contacts[0];
        assert!(contact.depth > 0.0, "Penetration depth should be positive");
        assert!(
            (contact.depth - 1.0).abs() < 0.01,
            "Penetration depth should be ~1.0, got {}",
            contact.depth
        );
    }

    #[test]
    fn invariant_separated_spheres_not_colliding() {
        // Non-overlapping spheres should not be detected as colliding.
        let world = {
            let mut w = PhysicsWorld::new(PhysicsConfig::default());

            // Two separated spheres (radius 1 each, centers 3.0 apart)
            w.add_body(RigidBody::new(Vec3::ZERO, Collider::sphere(1.0), 1.0));
            w.add_body(RigidBody::new(Vec3::X * 3.0, Collider::sphere(1.0), 1.0));
            w
        };

        let contacts = world.detect_collisions();
        assert!(contacts.is_empty(), "Separated spheres should not collide");
    }

    #[test]
    fn invariant_sphere_plane_detection() {
        // Sphere below plane should be detected as colliding.
        let world = {
            let mut w = PhysicsWorld::new(PhysicsConfig::default());

            // Ground plane at y=0
            w.add_body(RigidBody::new_static(Vec3::ZERO, Collider::ground()));

            // Sphere penetrating ground (center at y=0.5, radius=1.0)
            w.add_body(RigidBody::new(Vec3::Y * 0.5, Collider::sphere(1.0), 1.0));
            w
        };

        let contacts = world.detect_collisions();

        assert!(
            !contacts.is_empty(),
            "Sphere penetrating ground should be detected"
        );

        let contact = &contacts[0];
        assert!(
            (contact.depth - 0.5).abs() < 0.01,
            "Depth should be ~0.5, got {}",
            contact.depth
        );
    }

    #[test]
    fn invariant_box_plane_detection() {
        // Box below plane should be detected as colliding.
        let world = {
            let mut w = PhysicsWorld::new(PhysicsConfig::default());

            // Ground plane at y=0
            w.add_body(RigidBody::new_static(Vec3::ZERO, Collider::ground()));

            // Box penetrating ground (center at y=0.25, half-extent=0.5)
            w.add_body(RigidBody::new(
                Vec3::Y * 0.25,
                Collider::box_shape(Vec3::splat(0.5)),
                1.0,
            ));
            w
        };

        let contacts = world.detect_collisions();

        assert!(
            !contacts.is_empty(),
            "Box penetrating ground should be detected"
        );
    }

    // ========================================================================
    // Integration / Trajectory Tests
    // ========================================================================

    #[test]
    fn invariant_free_fall_trajectory() {
        // Under gravity, a body should follow the expected kinematic equation:
        // y(t) = y0 + v0*t - 0.5*g*t^2
        let mut world = PhysicsWorld::new(PhysicsConfig {
            gravity: Vec3::new(0.0, -10.0, 0.0), // Nice round number
            dt: 1.0 / 100.0,
            solver_iterations: 10,
        });

        let y0 = 100.0;
        let mut body = RigidBody::new(Vec3::Y * y0, Collider::sphere(0.5), 1.0);
        body.linear_damping = 0.0;
        body.angular_damping = 0.0;
        world.add_body(body);

        let g = 10.0;
        let dt = world.config.dt;

        // Simulate for 1 second
        for step in 1..=100 {
            world.step();

            let t = step as f32 * dt;
            let expected_y = y0 - 0.5 * g * t * t;
            let actual_y = world.bodies[0].position.y;

            // Allow small numerical error that grows with time
            let tolerance = 0.1 + t * 0.1;
            assert!(
                (actual_y - expected_y).abs() < tolerance,
                "At t={}: expected y={}, got y={}, error={}",
                t,
                expected_y,
                actual_y,
                (actual_y - expected_y).abs()
            );
        }
    }

    #[test]
    fn invariant_uniform_motion_no_forces() {
        // With no forces, a body should move in a straight line at constant velocity.
        let mut world = PhysicsWorld::new(PhysicsConfig {
            gravity: Vec3::ZERO,
            dt: 1.0 / 60.0,
            solver_iterations: 10,
        });

        let initial_pos = Vec3::new(1.0, 2.0, 3.0);
        let initial_vel = Vec3::new(5.0, -3.0, 2.0);

        let mut body = RigidBody::new(initial_pos, Collider::sphere(0.5), 1.0);
        body.velocity = initial_vel;
        body.linear_damping = 0.0;
        body.angular_damping = 0.0;
        world.add_body(body);

        let dt = world.config.dt;

        for step in 1..=60 {
            world.step();

            let t = step as f32 * dt;
            let expected_pos = initial_pos + initial_vel * t;
            let actual_pos = world.bodies[0].position;

            let error = (actual_pos - expected_pos).length();
            assert!(error < 0.001, "At t={}: position error = {}", t, error);

            // Velocity should remain constant
            let vel_error = (world.bodies[0].velocity - initial_vel).length();
            assert!(
                vel_error < 0.001,
                "Velocity should be constant, error = {}",
                vel_error
            );
        }
    }

    #[test]
    fn invariant_projectile_motion() {
        // Test that horizontal and vertical motion are independent (projectile motion).
        let mut world = PhysicsWorld::new(PhysicsConfig {
            gravity: Vec3::new(0.0, -10.0, 0.0),
            dt: 1.0 / 100.0,
            solver_iterations: 10,
        });

        let initial_pos = Vec3::ZERO;
        let initial_vel = Vec3::new(10.0, 20.0, 0.0); // 45 degree launch

        let mut body = RigidBody::new(initial_pos, Collider::sphere(0.5), 1.0);
        body.velocity = initial_vel;
        body.linear_damping = 0.0;
        body.angular_damping = 0.0;
        world.add_body(body);

        let g = 10.0;
        let dt = world.config.dt;

        // Simulate for 2 seconds
        for step in 1..=200 {
            world.step();

            let t = step as f32 * dt;

            // x(t) = v0x * t (constant horizontal velocity)
            let expected_x = initial_vel.x * t;
            // y(t) = v0y * t - 0.5 * g * t^2
            let expected_y = initial_vel.y * t - 0.5 * g * t * t;

            let actual_pos = world.bodies[0].position;

            let x_error = (actual_pos.x - expected_x).abs();
            let y_error = (actual_pos.y - expected_y).abs();

            let tolerance = 0.1 + t * 0.05;
            assert!(
                x_error < tolerance,
                "At t={}: x error = {} (expected {}, got {})",
                t,
                x_error,
                expected_x,
                actual_pos.x
            );
            assert!(
                y_error < tolerance,
                "At t={}: y error = {} (expected {}, got {})",
                t,
                y_error,
                expected_y,
                actual_pos.y
            );
        }
    }

    // ========================================================================
    // Additional Physics Invariants
    // ========================================================================

    #[test]
    fn invariant_static_bodies_immovable() {
        // Static bodies should never move regardless of forces or collisions.
        let mut world = PhysicsWorld::new(PhysicsConfig {
            gravity: Vec3::new(0.0, -9.81, 0.0),
            dt: 1.0 / 60.0,
            solver_iterations: 10,
        });

        let static_pos = Vec3::new(5.0, 3.0, 2.0);
        world.add_body(RigidBody::new_static(static_pos, Collider::sphere(1.0)));

        // Add a dynamic body that will collide with it
        let mut dynamic = RigidBody::new(static_pos + Vec3::X * 1.5, Collider::sphere(1.0), 10.0);
        dynamic.velocity = Vec3::X * -10.0; // Moving toward static body
        world.add_body(dynamic);

        // Simulate with collisions
        for _ in 0..100 {
            world.step();
        }

        // Static body should not have moved
        let final_pos = world.bodies[0].position;
        assert_eq!(
            final_pos, static_pos,
            "Static body moved from {:?} to {:?}",
            static_pos, final_pos
        );
    }

    #[test]
    fn invariant_inverse_mass_relationship() {
        // Verify that inv_mass is correctly computed as 1/mass.
        let body = RigidBody::new(Vec3::ZERO, Collider::sphere(1.0), 5.0);
        assert!(
            (body.inv_mass - 0.2).abs() < 0.0001,
            "inv_mass should be 1/mass = 0.2, got {}",
            body.inv_mass
        );

        let static_body = RigidBody::new_static(Vec3::ZERO, Collider::sphere(1.0));
        assert_eq!(
            static_body.inv_mass, 0.0,
            "Static body should have inv_mass = 0"
        );
    }

    #[test]
    fn invariant_contact_normal_direction() {
        // Contact normal should point from body A toward body B.
        let world = {
            let mut w = PhysicsWorld::new(PhysicsConfig::default());
            w.add_body(RigidBody::new(Vec3::ZERO, Collider::sphere(1.0), 1.0));
            w.add_body(RigidBody::new(Vec3::X * 1.5, Collider::sphere(1.0), 1.0));
            w
        };

        let contacts = world.detect_collisions();
        assert!(!contacts.is_empty());

        let contact = &contacts[0];
        let body_a_pos = world.bodies[contact.body_a].position;
        let body_b_pos = world.bodies[contact.body_b].position;

        // Normal should roughly point from A to B
        let a_to_b = (body_b_pos - body_a_pos).normalize();
        let dot = contact.normal.dot(a_to_b);

        assert!(
            dot > 0.9,
            "Contact normal should point from A to B, dot product = {}",
            dot
        );
    }

    #[test]
    fn invariant_symmetry_equal_mass_collision() {
        // With symmetric initial conditions, the center of mass should remain stationary.
        let mut world = PhysicsWorld::new(PhysicsConfig {
            gravity: Vec3::ZERO,
            dt: 1.0 / 60.0,
            solver_iterations: 20,
        });

        // Two equal-mass bodies with equal and opposite velocities
        let mut body1 = RigidBody::new(Vec3::new(-2.0, 0.0, 0.0), Collider::sphere(1.0), 1.0);
        body1.velocity = Vec3::X * 5.0;
        body1.linear_damping = 0.0;
        body1.restitution = 0.5;
        world.add_body(body1);

        let mut body2 = RigidBody::new(Vec3::new(2.0, 0.0, 0.0), Collider::sphere(1.0), 1.0);
        body2.velocity = Vec3::X * -5.0;
        body2.linear_damping = 0.0;
        body2.restitution = 0.5;
        world.add_body(body2);

        // Initial center of mass should be at origin
        let initial_com = (world.bodies[0].position + world.bodies[1].position) / 2.0;
        assert!(
            initial_com.length() < 0.01,
            "Initial COM should be at origin"
        );

        // Initial total momentum should be zero
        let initial_momentum = total_momentum(&world);
        assert!(
            initial_momentum.length() < 0.01,
            "Initial momentum should be zero"
        );

        // Simulate through collision
        for _ in 0..100 {
            world.step();
        }

        // Center of mass should remain at origin (within tolerance)
        let final_com = (world.bodies[0].position + world.bodies[1].position) / 2.0;
        assert!(
            final_com.length() < 0.5,
            "Center of mass should stay near origin: {:?}",
            final_com
        );

        // Total momentum should remain zero
        let final_momentum = total_momentum(&world);
        assert!(
            final_momentum.length() < 0.1,
            "Momentum should remain zero: {:?}",
            final_momentum
        );
    }

    #[test]
    fn invariant_damping_reduces_energy() {
        // With damping, kinetic energy should decrease over time.
        let mut world = PhysicsWorld::new(PhysicsConfig {
            gravity: Vec3::ZERO,
            dt: 1.0 / 60.0,
            solver_iterations: 10,
        });

        let mut body = RigidBody::new(Vec3::ZERO, Collider::sphere(1.0), 1.0);
        body.velocity = Vec3::new(10.0, 5.0, 3.0);
        body.angular_velocity = Vec3::new(2.0, 1.0, 0.5);
        body.linear_damping = 0.05;
        body.angular_damping = 0.05;
        world.add_body(body);

        let initial_ke = kinetic_energy(&world.bodies[0]);

        for _ in 0..100 {
            world.step();
        }

        let final_ke = kinetic_energy(&world.bodies[0]);

        assert!(
            final_ke < initial_ke,
            "Damping should reduce energy: initial={}, final={}",
            initial_ke,
            final_ke
        );
    }
}
