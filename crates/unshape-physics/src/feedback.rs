//! Recurrent-graph integration for the rigid-body physics simulation.
//!
//! Ports the stateful [`PhysicsWorld`] onto the feedback-edge mechanism in
//! `unshape-core` (`docs/design/recurrent-graphs.md`). The entire mutable state
//! (bodies + constraints + config) is carried on a feedback wire as an opaque
//! [`Value`], so the step node ([`PhysicsStep`]) is a pure `&self` [`DynNode`]:
//! it clones the previous world, advances one step via the existing native
//! [`PhysicsWorld::step`] (apply forces → integrate → collide → solve
//! constraints), and returns the new world. No state lives in the node.
//!
//! The simulation is fully deterministic and holds no RNG, so the carried
//! [`PhysicsWorld`] is the entire simulation state — nothing else needs to travel
//! on the feedback edge.
//!
//! # Tick-0 state
//!
//! The initial [`PhysicsWorld`] is produced by an in-graph [`PhysicsInit`] source
//! node (a pure `&self` `DynNode` taking no inputs) describing the simulation
//! config ([`Physics`]) plus a set of pure-data body and distance-constraint
//! descriptors. Deterministic — the same config always yields the same world.

use std::any::Any;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use unshape_core::{
    DataLocation, DynNode, EvalContext, GraphError, GraphValue, PortDescriptor, Value, ValueType,
};

use glam::Vec3;

use crate::{Collider, Physics, PhysicsWorld, RigidBody};

/// The opaque value type name for [`PhysicsWorld`] state on a wire.
pub const PHYSICS_STATE_NAME: &str = "PhysicsWorld";

impl GraphValue for PhysicsWorld {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn type_name(&self) -> &'static str {
        PHYSICS_STATE_NAME
    }

    fn location(&self) -> DataLocation {
        DataLocation::Cpu
    }
}

/// Returns the [`ValueType`] used for [`PhysicsWorld`] state on a wire.
pub fn physics_state_type() -> ValueType {
    ValueType::of::<PhysicsWorld>(PHYSICS_STATE_NAME)
}

/// Pure-data collision-shape descriptor.
///
/// Mirrors [`Collider`] as serializable data so a [`PhysicsInit`] node can carry
/// body shapes without depending on `Collider` itself being serializable.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum ColliderDesc {
    /// Sphere of the given radius.
    Sphere {
        /// Radius.
        radius: f32,
    },
    /// Axis-aligned box with the given half-extents.
    Box {
        /// Half-extents along each axis.
        half_extents: Vec3,
    },
    /// Infinite plane (normal + distance from origin).
    Plane {
        /// Unit normal pointing away from the solid side.
        normal: Vec3,
        /// Distance from origin along the normal.
        distance: f32,
    },
}

impl ColliderDesc {
    fn build(&self) -> Collider {
        match *self {
            ColliderDesc::Sphere { radius } => Collider::sphere(radius),
            ColliderDesc::Box { half_extents } => Collider::box_shape(half_extents),
            ColliderDesc::Plane { normal, distance } => Collider::plane(normal, distance),
        }
    }
}

/// Pure-data rigid-body descriptor for [`PhysicsInit`].
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct BodyDesc {
    /// Initial position.
    pub position: Vec3,
    /// Collision shape.
    pub collider: ColliderDesc,
    /// Mass (0 = static/infinite).
    pub mass: f32,
    /// Initial linear velocity.
    pub velocity: Vec3,
}

impl BodyDesc {
    /// Creates a dynamic body descriptor.
    pub fn new(position: Vec3, collider: ColliderDesc, mass: f32) -> Self {
        Self {
            position,
            collider,
            mass,
            velocity: Vec3::ZERO,
        }
    }

    /// Sets the initial linear velocity.
    pub fn with_velocity(mut self, velocity: Vec3) -> Self {
        self.velocity = velocity;
        self
    }

    fn build(&self) -> RigidBody {
        let mut body = RigidBody::new(self.position, self.collider.build(), self.mass);
        body.velocity = self.velocity;
        body
    }
}

/// Pure-data distance-constraint descriptor for [`PhysicsInit`].
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct DistanceConstraintDesc {
    /// Index of the first body.
    pub body_a: usize,
    /// Index of the second body.
    pub body_b: usize,
    /// Target distance between the bodies.
    pub distance: f32,
}

/// Pure in-graph **source** node producing the initial [`PhysicsWorld`].
///
/// Takes no inputs; outputs a world built from the simulation [`Physics`] config,
/// a set of [`BodyDesc`] bodies, and a set of [`DistanceConstraintDesc`]
/// constraints. Deterministic — the same config yields the same world.
///
/// # Ports
/// - Output `0` `"state"`: `Custom(PhysicsWorld)` — initial world.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct PhysicsInit {
    /// Simulation configuration (gravity, dt, solver iterations).
    pub config: Physics,
    /// Bodies to add, in order (indices match insertion order).
    pub bodies: Vec<BodyDesc>,
    /// Distance constraints to add, referencing body indices.
    pub constraints: Vec<DistanceConstraintDesc>,
}

impl PhysicsInit {
    /// Creates an init node with the given config and no bodies.
    pub fn new(config: Physics) -> Self {
        Self {
            config,
            bodies: Vec::new(),
            constraints: Vec::new(),
        }
    }

    /// Adds a body descriptor.
    pub fn with_body(mut self, body: BodyDesc) -> Self {
        self.bodies.push(body);
        self
    }

    /// Adds a distance constraint.
    pub fn with_distance_constraint(mut self, constraint: DistanceConstraintDesc) -> Self {
        self.constraints.push(constraint);
        self
    }

    /// Builds the initial [`PhysicsWorld`] from this config (pure).
    pub fn build(&self) -> PhysicsWorld {
        let mut world = PhysicsWorld::new(self.config.clone());
        for body in &self.bodies {
            world.add_body(body.build());
        }
        for c in &self.constraints {
            world.add_distance_constraint(c.body_a, c.body_b, c.distance);
        }
        world
    }
}

impl DynNode for PhysicsInit {
    fn type_name(&self) -> &'static str {
        "physics::feedback::PhysicsInit"
    }

    fn inputs(&self) -> Vec<PortDescriptor> {
        vec![]
    }

    fn outputs(&self) -> Vec<PortDescriptor> {
        vec![PortDescriptor::new("state", physics_state_type())]
    }

    fn execute(&self, _inputs: &[Value], _ctx: &EvalContext) -> Result<Vec<Value>, GraphError> {
        Ok(vec![Value::opaque(self.build())])
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Pure per-tick step node for the rigid-body physics simulation.
///
/// State (bodies + constraints + config) lives on the feedback edge, not in the
/// node. `execute` clones the previous [`PhysicsWorld`] and advances it one step
/// via the native [`PhysicsWorld::step`] (which uses `config.dt`).
///
/// # Ports
/// - Input `0` `"state"`: `Custom(PhysicsWorld)` — previous-tick state.
/// - Output `0` `"state"`: `Custom(PhysicsWorld)` — advanced state.
#[derive(Debug, Clone, Copy, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct PhysicsStep;

impl DynNode for PhysicsStep {
    fn type_name(&self) -> &'static str {
        "physics::feedback::PhysicsStep"
    }

    fn inputs(&self) -> Vec<PortDescriptor> {
        vec![PortDescriptor::new("state", physics_state_type())]
    }

    fn outputs(&self) -> Vec<PortDescriptor> {
        vec![PortDescriptor::new("state", physics_state_type())]
    }

    fn execute(&self, inputs: &[Value], _ctx: &EvalContext) -> Result<Vec<Value>, GraphError> {
        let prev = inputs[0].downcast_ref::<PhysicsWorld>().ok_or_else(|| {
            GraphError::ExecutionError(
                "physics::feedback::PhysicsStep expects a PhysicsWorld state input".to_string(),
            )
        })?;
        let mut next = prev.clone();
        next.step();
        Ok(vec![Value::opaque(next)])
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use unshape_core::{FeedbackState, Graph};

    fn init_node() -> PhysicsInit {
        PhysicsInit::new(Physics::default())
            // A falling sphere above a static ground plane.
            .with_body(BodyDesc::new(
                Vec3::new(0.0, 5.0, 0.0),
                ColliderDesc::Sphere { radius: 0.5 },
                1.0,
            ))
            .with_body(BodyDesc::new(
                Vec3::ZERO,
                ColliderDesc::Plane {
                    normal: Vec3::Y,
                    distance: 0.0,
                },
                0.0,
            ))
    }

    fn positions(world: &PhysicsWorld) -> Vec<Vec3> {
        world.bodies.iter().map(|b| b.position).collect()
    }

    #[test]
    fn evolves_like_mut_step_loop() {
        let n = 40u64;

        let mut reference = init_node().build();
        for _ in 0..n {
            reference.step();
        }

        let mut graph = Graph::new();
        let init = graph.add_node(init_node());
        let step = graph.add_node(PhysicsStep);
        graph.connect(init, 0, step, 0).unwrap();
        graph.connect_recurrence(step, 0, step, 0).unwrap();

        let mut state = FeedbackState::new();
        let mut last = None;
        for t in 0..n {
            let r = graph.tick(t, &mut state, &EvalContext::new()).unwrap();
            last = Some(
                r.get(step, 0)
                    .unwrap()
                    .downcast_ref::<PhysicsWorld>()
                    .unwrap()
                    .clone(),
            );
        }
        let evolved = last.unwrap();

        let ref_pos = positions(&reference);
        let evo_pos = positions(&evolved);
        assert_eq!(ref_pos.len(), evo_pos.len());
        for (a, b) in evo_pos.iter().zip(&ref_pos) {
            assert!((*a - *b).length() < 1e-4, "{a:?} != {b:?}");
        }
    }

    #[test]
    fn deterministic() {
        let run = || {
            let mut graph = Graph::new();
            let init = graph.add_node(init_node());
            let step = graph.add_node(PhysicsStep);
            graph.connect(init, 0, step, 0).unwrap();
            graph.connect_recurrence(step, 0, step, 0).unwrap();
            let mut state = FeedbackState::new();
            let mut last = Vec::new();
            for t in 0..25 {
                let r = graph.tick(t, &mut state, &EvalContext::new()).unwrap();
                last = positions(
                    r.get(step, 0)
                        .unwrap()
                        .downcast_ref::<PhysicsWorld>()
                        .unwrap(),
                );
            }
            last
        };
        assert_eq!(run(), run());
    }

    #[test]
    fn run_to_tick_matches_manual_stepping() {
        let target = 30u64;

        let manual = {
            let mut world = init_node().build();
            for _ in 0..=target {
                world.step();
            }
            positions(&world)
        };

        let mut graph = Graph::new();
        let init = graph.add_node(init_node());
        let step = graph.add_node(PhysicsStep);
        graph.connect(init, 0, step, 0).unwrap();
        graph.connect_recurrence(step, 0, step, 0).unwrap();
        let mut state = FeedbackState::new();
        let r = graph
            .run_to_tick(target, &mut state, |_t| EvalContext::new())
            .unwrap();
        let resimulated = positions(
            r.get(step, 0)
                .unwrap()
                .downcast_ref::<PhysicsWorld>()
                .unwrap(),
        );

        assert_eq!(resimulated.len(), manual.len());
        for (a, b) in resimulated.iter().zip(&manual) {
            assert!((*a - *b).length() < 1e-4);
        }
    }
}
