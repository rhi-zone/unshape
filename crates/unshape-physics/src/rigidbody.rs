//! Rigid body dynamics.
//!
//! Provides the core `RigidBody` type with mass, inertia, forces, and impulses.

use glam::{Mat3, Quat, Vec3};

use crate::Collider;

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
    pub(crate) force: Vec3,
    /// Accumulated torque for this frame.
    pub(crate) torque: Vec3,
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
    pub(crate) fn clear_forces(&mut self) {
        self.force = Vec3::ZERO;
        self.torque = Vec3::ZERO;
    }
}

/// Compute inertia tensor for a shape.
pub fn compute_inertia(collider: &Collider, mass: f32) -> Vec3 {
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
