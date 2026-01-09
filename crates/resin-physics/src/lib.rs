//! Rigid body physics simulation for resin.
//!
//! Provides basic rigid body dynamics with collision detection and response:
//! - `RigidBody` - dynamic or static rigid body with mass and inertia
//! - `Collider` - collision shapes (sphere, box, plane)
//! - `PhysicsWorld` - simulation container with gravity and constraints

use glam::{Mat3, Quat, Vec3};

// ============================================================================
// Collider Shapes
// ============================================================================

/// Collision shape for rigid bodies.
#[derive(Clone, Debug)]
pub enum Collider {
    /// Sphere with radius.
    Sphere { radius: f32 },
    /// Axis-aligned box with half-extents.
    Box { half_extents: Vec3 },
    /// Infinite plane defined by normal and distance from origin.
    Plane { normal: Vec3, distance: f32 },
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

// ============================================================================
// Physics World
// ============================================================================

/// Configuration for physics simulation.
#[derive(Clone, Debug)]
pub struct PhysicsConfig {
    /// Gravity acceleration.
    pub gravity: Vec3,
    /// Number of constraint solver iterations.
    pub solver_iterations: u32,
    /// Time step.
    pub dt: f32,
}

impl Default for PhysicsConfig {
    fn default() -> Self {
        Self {
            gravity: Vec3::new(0.0, -9.81, 0.0),
            solver_iterations: 10,
            dt: 1.0 / 60.0,
        }
    }
}

/// The physics simulation world.
pub struct PhysicsWorld {
    /// All rigid bodies.
    pub bodies: Vec<RigidBody>,
    /// Configuration.
    pub config: PhysicsConfig,
}

impl PhysicsWorld {
    /// Create a new physics world.
    pub fn new(config: PhysicsConfig) -> Self {
        Self {
            bodies: Vec::new(),
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

    /// Step the simulation forward.
    pub fn step(&mut self) {
        let dt = self.config.dt;
        let gravity = self.config.gravity;

        // Apply gravity
        for body in &mut self.bodies {
            if !body.is_static {
                body.apply_force(gravity * body.mass);
            }
        }

        // Integrate velocities
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

        // Detect collisions
        let contacts = self.detect_collisions();

        // Resolve collisions
        for _ in 0..self.config.solver_iterations {
            for contact in &contacts {
                self.resolve_contact(contact);
            }
        }

        // Integrate positions
        for body in &mut self.bodies {
            if !body.is_static {
                // Position
                body.position += body.velocity * dt;

                // Orientation (using quaternion integration)
                let w = body.angular_velocity;
                let q = body.orientation;
                let dq = Quat::from_xyzw(w.x, w.y, w.z, 0.0) * q * 0.5 * dt;
                body.orientation = (q + dq).normalize();
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
                sphere_plane(b, a, body_b.position, *radius, *normal, *distance).map(|mut c| {
                    c.normal = -c.normal;
                    std::mem::swap(&mut c.body_a, &mut c.body_b);
                    c
                })
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
            .map(|mut c| {
                c.normal = -c.normal;
                std::mem::swap(&mut c.body_a, &mut c.body_b);
                c
            }),
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
            .map(|mut c| {
                c.normal = -c.normal;
                std::mem::swap(&mut c.body_a, &mut c.body_b);
                c
            }),
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
}
