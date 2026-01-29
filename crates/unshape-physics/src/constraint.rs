//! Constraints and joints for rigid body simulation.
//!
//! Provides distance, point, hinge, and spring constraints with a unified
//! `ConstraintSolver` trait for computing and correcting constraint errors.

use glam::{Quat, Vec3};

use crate::RigidBody;

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

// ============================================================================
// Unified Constraint Solver Pattern
// ============================================================================

/// Computed error for a constraint.
///
/// Contains the positional/angular error that needs to be corrected.
#[derive(Debug, Clone)]
pub struct ConstraintError {
    /// Position error vector (world space).
    pub position_error: Vec3,
    /// Angular error vector (axis-angle representation).
    pub angular_error: Vec3,
    /// Bodies involved (indices).
    pub body_a: usize,
    pub body_b: Option<usize>,
}

impl ConstraintError {
    /// Creates a position-only error.
    pub fn position(body_a: usize, body_b: Option<usize>, error: Vec3) -> Self {
        Self {
            position_error: error,
            angular_error: Vec3::ZERO,
            body_a,
            body_b,
        }
    }

    /// Creates a combined position and angular error.
    pub fn full(body_a: usize, body_b: usize, pos_error: Vec3, ang_error: Vec3) -> Self {
        Self {
            position_error: pos_error,
            angular_error: ang_error,
            body_a,
            body_b: Some(body_b),
        }
    }

    /// Returns true if the error is negligible.
    pub fn is_negligible(&self) -> bool {
        self.position_error.length_squared() < 0.00001
            && self.angular_error.length_squared() < 0.00001
    }
}

/// Trait for constraint solvers.
///
/// All constraints follow the same pattern:
/// 1. Compute error (position/angular deviation from constraint)
/// 2. Apply correction (adjust bodies to reduce error)
///
/// This trait makes the pattern explicit and allows custom constraints.
pub trait ConstraintSolver {
    /// Compute the constraint error given current body states.
    ///
    /// Returns `None` if bodies are invalid or constraint doesn't apply.
    fn compute_error(&self, bodies: &[RigidBody]) -> Option<ConstraintError>;

    /// Apply corrections to bodies to satisfy the constraint.
    ///
    /// The stiffness parameter controls how aggressively to correct (0-1).
    fn apply_correction(&self, bodies: &mut [RigidBody], stiffness: f32);

    /// Returns the constraint stiffness.
    fn stiffness(&self) -> f32;
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
// ConstraintSolver Implementations
// ============================================================================

impl ConstraintSolver for DistanceConstraint {
    fn compute_error(&self, bodies: &[RigidBody]) -> Option<ConstraintError> {
        if self.body_a >= bodies.len() || self.body_b >= bodies.len() {
            return None;
        }

        let pos_a =
            bodies[self.body_a].position + bodies[self.body_a].orientation * self.local_anchor_a;
        let pos_b =
            bodies[self.body_b].position + bodies[self.body_b].orientation * self.local_anchor_b;

        let delta = pos_b - pos_a;
        let current_distance = delta.length();

        if current_distance < 0.0001 {
            return None;
        }

        let direction = delta / current_distance;
        let error = direction * (current_distance - self.distance);

        Some(ConstraintError::position(
            self.body_a,
            Some(self.body_b),
            error,
        ))
    }

    fn apply_correction(&self, bodies: &mut [RigidBody], stiffness: f32) {
        if self.body_a >= bodies.len() || self.body_b >= bodies.len() {
            return;
        }

        let pos_a =
            bodies[self.body_a].position + bodies[self.body_a].orientation * self.local_anchor_a;
        let pos_b =
            bodies[self.body_b].position + bodies[self.body_b].orientation * self.local_anchor_b;

        let delta = pos_b - pos_a;
        let current_distance = delta.length();

        if current_distance < 0.0001 {
            return;
        }

        let direction = delta / current_distance;
        let error = current_distance - self.distance;

        if error.abs() < 0.0001 {
            return;
        }

        let inv_mass_a = bodies[self.body_a].inv_mass;
        let inv_mass_b = bodies[self.body_b].inv_mass;
        let total_inv_mass = inv_mass_a + inv_mass_b;

        if total_inv_mass < 0.0001 {
            return;
        }

        let correction = direction * error * stiffness / total_inv_mass;

        if !bodies[self.body_a].is_static {
            bodies[self.body_a].position += correction * inv_mass_a;
        }
        if !bodies[self.body_b].is_static {
            bodies[self.body_b].position -= correction * inv_mass_b;
        }
    }

    fn stiffness(&self) -> f32 {
        self.stiffness
    }
}

impl ConstraintSolver for PointConstraint {
    fn compute_error(&self, bodies: &[RigidBody]) -> Option<ConstraintError> {
        if self.body >= bodies.len() {
            return None;
        }

        let anchor_world =
            bodies[self.body].position + bodies[self.body].orientation * self.local_anchor;

        let (target_pos, other_body) = match &self.target {
            PointConstraintTarget::World(pos) => (*pos, None),
            PointConstraintTarget::Body { body, local_anchor } => {
                if *body >= bodies.len() {
                    return None;
                }
                let pos = bodies[*body].position + bodies[*body].orientation * *local_anchor;
                (pos, Some(*body))
            }
        };

        let error = target_pos - anchor_world;
        Some(ConstraintError::position(self.body, other_body, error))
    }

    fn apply_correction(&self, bodies: &mut [RigidBody], stiffness: f32) {
        if self.body >= bodies.len() {
            return;
        }

        let anchor_world =
            bodies[self.body].position + bodies[self.body].orientation * self.local_anchor;

        let target_pos = match &self.target {
            PointConstraintTarget::World(pos) => *pos,
            PointConstraintTarget::Body { body, local_anchor } => {
                if *body >= bodies.len() {
                    return;
                }
                bodies[*body].position + bodies[*body].orientation * *local_anchor
            }
        };

        let delta = target_pos - anchor_world;

        if delta.length() < 0.0001 {
            return;
        }

        match &self.target {
            PointConstraintTarget::World(_) => {
                if !bodies[self.body].is_static {
                    bodies[self.body].position += delta * stiffness;
                }
            }
            PointConstraintTarget::Body {
                body: other_idx, ..
            } => {
                let inv_mass_a = bodies[self.body].inv_mass;
                let inv_mass_b = bodies[*other_idx].inv_mass;
                let total_inv_mass = inv_mass_a + inv_mass_b;

                if total_inv_mass < 0.0001 {
                    return;
                }

                let correction = delta * stiffness / total_inv_mass;

                if !bodies[self.body].is_static {
                    bodies[self.body].position += correction * inv_mass_a;
                }
                if !bodies[*other_idx].is_static {
                    bodies[*other_idx].position -= correction * inv_mass_b;
                }
            }
        }
    }

    fn stiffness(&self) -> f32 {
        self.stiffness
    }
}

impl ConstraintSolver for HingeConstraint {
    fn compute_error(&self, bodies: &[RigidBody]) -> Option<ConstraintError> {
        if self.body_a >= bodies.len() || self.body_b >= bodies.len() {
            return None;
        }

        // Position error (anchor alignment)
        let pos_a =
            bodies[self.body_a].position + bodies[self.body_a].orientation * self.local_anchor_a;
        let pos_b =
            bodies[self.body_b].position + bodies[self.body_b].orientation * self.local_anchor_b;
        let pos_error = pos_b - pos_a;

        // Angular error (axis alignment)
        let world_axis_a = bodies[self.body_a].orientation * self.local_axis_a;
        let world_axis_b = bodies[self.body_b].orientation * self.local_axis_b;
        let ang_error = world_axis_a.cross(world_axis_b);

        Some(ConstraintError::full(
            self.body_a,
            self.body_b,
            pos_error,
            ang_error,
        ))
    }

    fn apply_correction(&self, bodies: &mut [RigidBody], stiffness: f32) {
        if self.body_a >= bodies.len() || self.body_b >= bodies.len() {
            return;
        }

        // Position correction
        let pos_a =
            bodies[self.body_a].position + bodies[self.body_a].orientation * self.local_anchor_a;
        let pos_b =
            bodies[self.body_b].position + bodies[self.body_b].orientation * self.local_anchor_b;
        let delta = pos_b - pos_a;

        if delta.length() > 0.0001 {
            let inv_mass_a = bodies[self.body_a].inv_mass;
            let inv_mass_b = bodies[self.body_b].inv_mass;
            let total_inv_mass = inv_mass_a + inv_mass_b;

            if total_inv_mass > 0.0001 {
                let correction = delta * stiffness / total_inv_mass;

                if !bodies[self.body_a].is_static {
                    bodies[self.body_a].position += correction * inv_mass_a;
                }
                if !bodies[self.body_b].is_static {
                    bodies[self.body_b].position -= correction * inv_mass_b;
                }
            }
        }

        // Angular correction
        let world_axis_a = bodies[self.body_a].orientation * self.local_axis_a;
        let world_axis_b = bodies[self.body_b].orientation * self.local_axis_b;
        let axis_error = world_axis_a.cross(world_axis_b);

        if axis_error.length() > 0.0001 {
            let correction = axis_error * stiffness * 0.5;

            if !bodies[self.body_a].is_static {
                let q = bodies[self.body_a].orientation;
                let dq = Quat::from_xyzw(correction.x, correction.y, correction.z, 0.0) * q * 0.5;
                bodies[self.body_a].orientation = (q + dq).normalize();
            }
            if !bodies[self.body_b].is_static {
                let q = bodies[self.body_b].orientation;
                let dq =
                    Quat::from_xyzw(-correction.x, -correction.y, -correction.z, 0.0) * q * 0.5;
                bodies[self.body_b].orientation = (q + dq).normalize();
            }
        }

        // Note: Angle limits are not implemented in this simplified version
    }

    fn stiffness(&self) -> f32 {
        self.stiffness
    }
}

/// Implementation for the Constraint enum (delegates to inner type).
impl ConstraintSolver for Constraint {
    fn compute_error(&self, bodies: &[RigidBody]) -> Option<ConstraintError> {
        match self {
            Constraint::Distance(c) => c.compute_error(bodies),
            Constraint::Point(c) => c.compute_error(bodies),
            Constraint::Hinge(c) => c.compute_error(bodies),
            Constraint::Spring(_) => None, // Springs handled as forces, not position constraints
        }
    }

    fn apply_correction(&self, bodies: &mut [RigidBody], stiffness: f32) {
        match self {
            Constraint::Distance(c) => c.apply_correction(bodies, stiffness),
            Constraint::Point(c) => c.apply_correction(bodies, stiffness),
            Constraint::Hinge(c) => c.apply_correction(bodies, stiffness),
            Constraint::Spring(_) => {} // Springs handled as forces
        }
    }

    fn stiffness(&self) -> f32 {
        match self {
            Constraint::Distance(c) => c.stiffness(),
            Constraint::Point(c) => c.stiffness(),
            Constraint::Hinge(c) => c.stiffness(),
            Constraint::Spring(c) => c.stiffness / 100.0, // Normalize spring stiffness
        }
    }
}
