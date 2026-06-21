//! Recurrent-graph integration for the spring soft-body simulation.
//!
//! Ports the stateful [`SpringSystem`] onto the feedback-edge mechanism in
//! `unshape-core` (`docs/design/recurrent-graphs.md`). The entire mutable state
//! (particles + Verlet history) is carried on a feedback wire as an opaque
//! [`Value`], so the step node ([`SpringStep`]) is a pure `&self` [`DynNode`]: it
//! clones the previous state, advances one Verlet step via the existing native
//! [`SpringSystem::step`], and returns the new state. No state lives in the node.
//!
//! The Verlet integration is fully deterministic and holds no RNG, so the carried
//! [`SpringSystem`] is the entire simulation state — nothing else needs to travel
//! on the feedback edge.
//!
//! # `dt` as config
//!
//! `dt` is a field on [`SpringStep`] (config), not a per-tick external input,
//! mirroring how the imperative loop calls `system.step(dt)` with a fixed step.
//!
//! # Tick-0 state
//!
//! The initial [`SpringSystem`] is produced by an in-graph [`SpringInit`] source
//! node (a pure `&self` `DynNode` taking no inputs) describing the body to build
//! from one of the constructor helpers ([`create_rope`], [`create_cloth`],
//! [`create_soft_sphere`]) plus solver settings (gravity, damping, constraint
//! iterations). Deterministic — the same config always yields the same body.

use std::any::Any;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use unshape_core::{
    DataLocation, DynNode, EvalContext, GraphError, GraphValue, PortDescriptor, Value, ValueType,
};

use glam::Vec3;

use crate::{SpringConfig, SpringSystem, create_cloth, create_rope, create_soft_sphere};

/// The opaque value type name for [`SpringSystem`] state on a wire.
pub const SPRING_STATE_NAME: &str = "SpringSystem";

impl GraphValue for SpringSystem {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn type_name(&self) -> &'static str {
        SPRING_STATE_NAME
    }

    fn location(&self) -> DataLocation {
        DataLocation::Cpu
    }
}

/// Returns the [`ValueType`] used for [`SpringSystem`] state on a wire.
pub fn spring_state_type() -> ValueType {
    ValueType::of::<SpringSystem>(SPRING_STATE_NAME)
}

/// Pure-data description of the soft body to build for a [`SpringInit`].
///
/// Each variant maps directly onto one of the crate's constructor helpers, so a
/// node can hold the body shape as data rather than as a baked [`SpringSystem`].
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum SpringBody {
    /// A linear rope (see [`create_rope`]).
    Rope {
        /// Start endpoint.
        start: Vec3,
        /// End endpoint.
        end: Vec3,
        /// Number of segments.
        segments: usize,
        /// Spring config for each segment.
        config: SpringConfig,
    },
    /// A 2D cloth grid (see [`create_cloth`]).
    Cloth {
        /// Grid origin (top-left corner).
        origin: Vec3,
        /// Total width.
        width: f32,
        /// Total height.
        height: f32,
        /// Column count.
        cols: usize,
        /// Row count.
        rows: usize,
        /// Spring config for each spring.
        config: SpringConfig,
    },
    /// A soft-body sphere (see [`create_soft_sphere`]).
    SoftSphere {
        /// Center position.
        center: Vec3,
        /// Radius.
        radius: f32,
        /// Icosphere subdivisions.
        subdivisions: usize,
    },
}

impl SpringBody {
    fn build(&self) -> SpringSystem {
        match *self {
            SpringBody::Rope {
                start,
                end,
                segments,
                config,
            } => create_rope(start, end, segments, config),
            SpringBody::Cloth {
                origin,
                width,
                height,
                cols,
                rows,
                config,
            } => create_cloth(origin, width, height, cols, rows, config),
            SpringBody::SoftSphere {
                center,
                radius,
                subdivisions,
            } => create_soft_sphere(center, radius, subdivisions),
        }
    }
}

/// Pure in-graph **source** node producing the initial [`SpringSystem`].
///
/// Takes no inputs; outputs a freshly built soft body ([`SpringBody`]) with the
/// configured solver settings applied (gravity, damping, constraint iterations).
/// Deterministic — the same config yields the same body.
///
/// # Ports
/// - Output `0` `"state"`: `Custom(SpringSystem)` — initial system.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct SpringInit {
    /// The body to build.
    pub body: SpringBody,
    /// Gravity applied each step.
    pub gravity: Vec3,
    /// Velocity damping (per step).
    pub damping: f32,
    /// Number of constraint-relaxation iterations per step.
    pub constraint_iterations: usize,
    /// Index of a particle to pin (immovable), if any.
    pub pin: Option<usize>,
}

impl SpringInit {
    /// Creates an init node for the given body with default solver settings.
    pub fn new(body: SpringBody) -> Self {
        Self {
            body,
            gravity: Vec3::new(0.0, -9.8, 0.0),
            damping: 0.99,
            constraint_iterations: 5,
            pin: None,
        }
    }

    /// Sets the gravity vector.
    pub fn with_gravity(mut self, gravity: Vec3) -> Self {
        self.gravity = gravity;
        self
    }

    /// Sets the velocity damping.
    pub fn with_damping(mut self, damping: f32) -> Self {
        self.damping = damping;
        self
    }

    /// Sets the constraint iteration count.
    pub fn with_constraint_iterations(mut self, iterations: usize) -> Self {
        self.constraint_iterations = iterations;
        self
    }

    /// Pins the given particle index (immovable).
    pub fn with_pin(mut self, id: usize) -> Self {
        self.pin = Some(id);
        self
    }

    /// Builds the initial [`SpringSystem`] from this config (pure).
    pub fn build(&self) -> SpringSystem {
        let mut system = self.body.build();
        system.set_gravity(self.gravity);
        system.set_damping(self.damping);
        system.set_constraint_iterations(self.constraint_iterations);
        if let Some(id) = self.pin {
            system.pin_particle(id);
        }
        system
    }
}

impl DynNode for SpringInit {
    fn type_name(&self) -> &'static str {
        "spring::feedback::SpringInit"
    }

    fn inputs(&self) -> Vec<PortDescriptor> {
        vec![]
    }

    fn outputs(&self) -> Vec<PortDescriptor> {
        vec![PortDescriptor::new("state", spring_state_type())]
    }

    fn execute(&self, _inputs: &[Value], _ctx: &EvalContext) -> Result<Vec<Value>, GraphError> {
        Ok(vec![Value::opaque(self.build())])
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Pure per-tick step node for the spring soft-body simulation.
///
/// State (particles + Verlet history) lives on the feedback edge, not in the
/// node. `execute` clones the previous [`SpringSystem`] and advances it one
/// Verlet step via the native [`SpringSystem::step`].
///
/// # Ports
/// - Input `0` `"state"`: `Custom(SpringSystem)` — previous-tick state.
/// - Output `0` `"state"`: `Custom(SpringSystem)` — advanced state.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct SpringStep {
    /// Time step (seconds) per tick.
    pub dt: f32,
}

impl SpringStep {
    /// Creates a step node with the given time step.
    pub fn new(dt: f32) -> Self {
        Self { dt }
    }
}

impl Default for SpringStep {
    fn default() -> Self {
        Self { dt: 0.016 }
    }
}

impl DynNode for SpringStep {
    fn type_name(&self) -> &'static str {
        "spring::feedback::SpringStep"
    }

    fn inputs(&self) -> Vec<PortDescriptor> {
        vec![PortDescriptor::new("state", spring_state_type())]
    }

    fn outputs(&self) -> Vec<PortDescriptor> {
        vec![PortDescriptor::new("state", spring_state_type())]
    }

    fn execute(&self, inputs: &[Value], _ctx: &EvalContext) -> Result<Vec<Value>, GraphError> {
        let prev = inputs[0].downcast_ref::<SpringSystem>().ok_or_else(|| {
            GraphError::ExecutionError(
                "spring::feedback::SpringStep expects a SpringSystem state input".to_string(),
            )
        })?;
        let mut next = prev.clone();
        next.step(self.dt);
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

    fn init_node() -> SpringInit {
        SpringInit::new(SpringBody::Cloth {
            origin: Vec3::ZERO,
            width: 4.0,
            height: 4.0,
            cols: 6,
            rows: 6,
            config: SpringConfig::default(),
        })
        .with_pin(0)
    }

    fn positions(sys: &SpringSystem) -> Vec<Vec3> {
        sys.positions()
    }

    #[test]
    fn evolves_like_mut_step_loop() {
        let n = 30u64;
        let dt = 0.016f32;

        let mut reference = init_node().build();
        reference.steps(dt, n as usize);

        let mut graph = Graph::new();
        let init = graph.add_node(init_node());
        let step = graph.add_node(SpringStep::new(dt));
        graph.connect(init, 0, step, 0).unwrap();
        graph.connect_recurrence(step, 0, step, 0).unwrap();

        let mut state = FeedbackState::new();
        let mut last = None;
        for t in 0..n {
            let r = graph.tick(t, &mut state, &EvalContext::new()).unwrap();
            last = Some(
                r.get(step, 0)
                    .unwrap()
                    .downcast_ref::<SpringSystem>()
                    .unwrap()
                    .clone(),
            );
        }
        let evolved = last.unwrap();

        let ref_pos = positions(&reference);
        let evo_pos = positions(&evolved);
        assert_eq!(ref_pos.len(), evo_pos.len());
        for (a, b) in evo_pos.iter().zip(&ref_pos) {
            assert!((*a - *b).length() < 1e-5, "{a:?} != {b:?}");
        }
    }

    #[test]
    fn deterministic() {
        let dt = 0.016f32;
        let run = || {
            let mut graph = Graph::new();
            let init = graph.add_node(init_node());
            let step = graph.add_node(SpringStep::new(dt));
            graph.connect(init, 0, step, 0).unwrap();
            graph.connect_recurrence(step, 0, step, 0).unwrap();
            let mut state = FeedbackState::new();
            let mut last = Vec::new();
            for t in 0..20 {
                let r = graph.tick(t, &mut state, &EvalContext::new()).unwrap();
                last = positions(
                    r.get(step, 0)
                        .unwrap()
                        .downcast_ref::<SpringSystem>()
                        .unwrap(),
                );
            }
            last
        };
        assert_eq!(run(), run());
    }

    #[test]
    fn run_to_tick_matches_manual_stepping() {
        let target = 25u64;
        let dt = 0.016f32;

        let manual = {
            let mut sys = init_node().build();
            sys.steps(dt, (target + 1) as usize);
            positions(&sys)
        };

        let mut graph = Graph::new();
        let init = graph.add_node(init_node());
        let step = graph.add_node(SpringStep::new(dt));
        graph.connect(init, 0, step, 0).unwrap();
        graph.connect_recurrence(step, 0, step, 0).unwrap();
        let mut state = FeedbackState::new();
        let r = graph
            .run_to_tick(target, &mut state, |_t| EvalContext::new())
            .unwrap();
        let resimulated = positions(
            r.get(step, 0)
                .unwrap()
                .downcast_ref::<SpringSystem>()
                .unwrap(),
        );

        assert_eq!(resimulated.len(), manual.len());
        for (a, b) in resimulated.iter().zip(&manual) {
            assert!((*a - *b).length() < 1e-5);
        }
    }
}
