//! Recurrent-graph integration for the grid-based fluid simulation.
//!
//! Ports the stateful [`FluidGrid2D`] (Jos Stam stable fluids) onto the
//! feedback-edge mechanism in `unshape-core` (`docs/design/recurrent-graphs.md`).
//! The entire mutable grid state (velocity, density, and scratch fields, plus
//! the embedded config) is carried on a feedback wire as an opaque [`Value`], so
//! the step node ([`Step`]) is a pure `&self` [`DynNode`]: it clones the previous
//! grid, advances one Navier-Stokes step, and returns the new grid. No state
//! lives in the node.
//!
//! # Tick-0 state
//!
//! Opaque types have no zero value, so the initial [`FluidGrid2D`] (already
//! seeded with density/velocity sources) must be pre-seeded into
//! [`FeedbackState`](unshape_core::FeedbackState) before the first tick.

use std::any::Any;

use unshape_core::{
    DataLocation, DynNode, EvalContext, GraphError, GraphValue, PortDescriptor, Value, ValueType,
};

use crate::FluidGrid2D;

/// The opaque value type name for [`FluidGrid2D`] state on a wire.
pub const FLUID_GRID_2D_NAME: &str = "FluidGrid2D";

impl GraphValue for FluidGrid2D {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn type_name(&self) -> &'static str {
        FLUID_GRID_2D_NAME
    }

    fn location(&self) -> DataLocation {
        DataLocation::Cpu
    }
}

/// Returns the [`ValueType`] used for [`FluidGrid2D`] state on a wire.
pub fn fluid_grid_2d_type() -> ValueType {
    ValueType::of::<FluidGrid2D>(FLUID_GRID_2D_NAME)
}

/// Pure per-tick step node for the 2D fluid simulation.
///
/// State lives on the feedback edge, not in the node. `execute` clones the
/// previous [`FluidGrid2D`], advances it one step (clone-and-advance, mirroring
/// [`FluidGrid2D::step`]), and returns the new grid. Simulation parameters
/// (diffusion, iterations, dt) live in the grid's embedded config.
///
/// # Ports
/// - Input `0` `"state"`: `Custom(FluidGrid2D)` — previous-tick grid.
/// - Output `0` `"state"`: `Custom(FluidGrid2D)` — advanced grid.
#[derive(Debug, Clone, Copy, Default)]
pub struct Step;

impl DynNode for Step {
    fn type_name(&self) -> &'static str {
        "fluid::feedback::Step"
    }

    fn inputs(&self) -> Vec<PortDescriptor> {
        vec![PortDescriptor::new("state", fluid_grid_2d_type())]
    }

    fn outputs(&self) -> Vec<PortDescriptor> {
        vec![PortDescriptor::new("state", fluid_grid_2d_type())]
    }

    fn execute(&self, inputs: &[Value], _ctx: &EvalContext) -> Result<Vec<Value>, GraphError> {
        let prev = inputs[0].downcast_ref::<FluidGrid2D>().ok_or_else(|| {
            GraphError::ExecutionError(
                "fluid::feedback::Step expects a FluidGrid2D state input".to_string(),
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
    use crate::FluidConfig;
    use unshape_core::{FeedbackState, Graph};

    fn seeded_grid() -> FluidGrid2D {
        let mut g = FluidGrid2D::new(32, 32, FluidConfig::default());
        g.add_density(16, 16, 100.0);
        g.add_velocity(16, 16, 5.0, 2.0);
        g
    }

    fn build() -> (Graph, u32, FeedbackState) {
        let mut graph = Graph::new();
        let step = graph.add_node(Step);
        graph.connect_feedback(step, 0, step, 0).unwrap();
        let mut state = FeedbackState::new();
        state.set(step, 0, Value::opaque(seeded_grid()));
        (graph, step, state)
    }

    fn density_sum(g: &FluidGrid2D) -> f64 {
        g.density_field().iter().map(|&x| x as f64).sum()
    }

    #[test]
    fn evolves_like_mut_step_loop() {
        // (a) feedback stepping N times matches the &mut step loop N times.
        let n = 15u64;

        let mut reference = seeded_grid();
        for _ in 0..n {
            reference.step();
        }

        let (mut graph, step, mut state) = build();
        let mut last = None;
        for t in 0..n {
            let r = graph.tick(t, &mut state, &EvalContext::new()).unwrap();
            last = Some(
                r.get(step, 0)
                    .unwrap()
                    .downcast_ref::<FluidGrid2D>()
                    .unwrap()
                    .clone(),
            );
        }
        let evolved = last.unwrap();

        for (a, b) in evolved
            .density_field()
            .iter()
            .zip(reference.density_field())
        {
            assert_eq!(a, b);
        }
        let (evx, evy) = evolved.velocity_field();
        let (rvx, rvy) = reference.velocity_field();
        for (a, b) in evx.iter().zip(rvx) {
            assert_eq!(a, b);
        }
        for (a, b) in evy.iter().zip(rvy) {
            assert_eq!(a, b);
        }
    }

    #[test]
    fn node_is_pure_fresh_state_restarts() {
        // (b) the node holds no state: a fresh seed restarts from one step.
        let (mut graph, step, mut state) = build();
        for t in 0..5 {
            graph.tick(t, &mut state, &EvalContext::new()).unwrap();
        }

        let mut fresh = FeedbackState::new();
        fresh.set(step, 0, Value::opaque(seeded_grid()));
        let r = graph.tick(0, &mut fresh, &EvalContext::new()).unwrap();
        let one = r
            .get(step, 0)
            .unwrap()
            .downcast_ref::<FluidGrid2D>()
            .unwrap()
            .clone();

        let mut reference = seeded_grid();
        reference.step();
        for (a, b) in one.density_field().iter().zip(reference.density_field()) {
            assert_eq!(a, b);
        }
    }

    #[test]
    fn deterministic() {
        // (c) same seed + inputs + N -> identical output.
        let run = || {
            let (mut graph, step, mut state) = build();
            let mut last = 0.0;
            for t in 0..12 {
                let r = graph.tick(t, &mut state, &EvalContext::new()).unwrap();
                last = density_sum(
                    r.get(step, 0)
                        .unwrap()
                        .downcast_ref::<FluidGrid2D>()
                        .unwrap(),
                );
            }
            last
        };
        assert_eq!(run(), run());
    }

    #[test]
    fn run_to_tick_errors_on_opaque_seed() {
        // Documents the opaque-state + run_to_tick limitation (see rd/particle).
        let (mut graph, _step, mut state) = build();
        let r = graph.run_to_tick(3, &mut state, |_t| EvalContext::new());
        assert!(matches!(r, Err(GraphError::ExecutionError(_))));
    }
}
