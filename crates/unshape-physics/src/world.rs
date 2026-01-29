//! Physics simulation world.
//!
//! Contains the `PhysicsWorld` container that drives rigid body simulation
//! including force integration, collision detection, and constraint solving.

use glam::{Quat, Vec3};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::collision;
use crate::{
    Collider, Constraint, Contact, DistanceConstraint, HingeConstraint, PointConstraint,
    PointConstraintTarget, RigidBody, SpringConstraint,
};

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
    pub(crate) fn detect_collisions(&self) -> Vec<Contact> {
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
                collision::sphere_sphere(a, b, body_a.position, *r1, body_b.position, *r2)
            }
            (Collider::Sphere { radius }, Collider::Plane { normal, distance }) => {
                collision::sphere_plane(a, b, body_a.position, *radius, *normal, *distance)
            }
            (Collider::Plane { normal, distance }, Collider::Sphere { radius }) => {
                collision::sphere_plane(b, a, body_b.position, *radius, *normal, *distance)
                    .map(Contact::flip)
            }
            (Collider::Box { half_extents }, Collider::Plane { normal, distance }) => {
                collision::box_plane(
                    a,
                    b,
                    body_a.position,
                    body_a.orientation,
                    *half_extents,
                    *normal,
                    *distance,
                )
            }
            (Collider::Plane { normal, distance }, Collider::Box { half_extents }) => {
                collision::box_plane(
                    b,
                    a,
                    body_b.position,
                    body_b.orientation,
                    *half_extents,
                    *normal,
                    *distance,
                )
                .map(Contact::flip)
            }
            (Collider::Sphere { radius }, Collider::Box { half_extents }) => collision::sphere_box(
                a,
                b,
                body_a.position,
                *radius,
                body_b.position,
                body_b.orientation,
                *half_extents,
            ),
            (Collider::Box { half_extents }, Collider::Sphere { radius }) => collision::sphere_box(
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
                collision::box_box_aabb(a, b, body_a.position, *he1, body_b.position, *he2)
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
