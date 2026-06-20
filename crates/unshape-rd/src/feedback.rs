//! Recurrent-graph integration for the Gray-Scott simulation.
//!
//! This module ports the stateful reaction-diffusion simulator onto the
//! feedback-edge mechanism in `unshape-core` (`docs/design/recurrent-graphs.md`).
//! The entire mutable simulation state ([`ReactionDiffusion`]) is carried on a
//! feedback wire as an opaque [`Value`], so the step node ([`Step`]) is a pure
//! `&self` [`DynNode`]: it reads the previous state, advances one step, and
//! returns the new state. No state lives inside the node.
//!
//! # Usage
//!
//! ```
//! use unshape_rd::ReactionDiffusion;
//! use unshape_rd::feedback::Step;
//! use unshape_core::{Graph, FeedbackState, EvalContext, Value};
//!
//! let mut graph = Graph::new();
//! let step = graph.add_node(Step);
//! // self-wire: state output -> state input (back-edge)
//! graph.connect_feedback(step, 0, step, 0).unwrap();
//!
//! // Pre-seed the initial state (opaque types have no zero value).
//! let mut rd = ReactionDiffusion::new(32, 32);
//! rd.add_seed_circle(16, 16, 4);
//! let mut state = FeedbackState::new();
//! state.set(step, 0, Value::opaque(rd));
//!
//! let r = graph.tick(0, &mut state, &EvalContext::new()).unwrap();
//! assert!(r.get(step, 0).is_some());
//! ```

use std::any::Any;

use unshape_core::{
    DataLocation, DynNode, EvalContext, GraphError, GraphValue, PortDescriptor, Value, ValueType,
};

use crate::ReactionDiffusion;

/// The opaque value type name for [`ReactionDiffusion`] state on a wire.
pub const RD_STATE_NAME: &str = "ReactionDiffusion";

impl GraphValue for ReactionDiffusion {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn type_name(&self) -> &'static str {
        RD_STATE_NAME
    }

    fn location(&self) -> DataLocation {
        DataLocation::Cpu
    }
}

/// Returns the [`ValueType`] used for [`ReactionDiffusion`] state on a wire.
pub fn rd_state_type() -> ValueType {
    ValueType::of::<ReactionDiffusion>(RD_STATE_NAME)
}

/// Pure step node for the Gray-Scott simulation.
///
/// State lives on the feedback edge, not in the node. `execute` reads the
/// previous [`ReactionDiffusion`] state, advances it by `count` steps
/// (`crate::Step::apply`, which clones-and-advances), and returns the new state.
///
/// # Ports
/// - Input `0` `"state"`: `Custom(ReactionDiffusion)` — previous-tick state.
/// - Output `0` `"state"`: `Custom(ReactionDiffusion)` — advanced state.
#[derive(Debug, Clone, Copy, Default)]
pub struct Step;

impl DynNode for Step {
    fn type_name(&self) -> &'static str {
        "rd::feedback::Step"
    }

    fn inputs(&self) -> Vec<PortDescriptor> {
        vec![PortDescriptor::new("state", rd_state_type())]
    }

    fn outputs(&self) -> Vec<PortDescriptor> {
        vec![PortDescriptor::new("state", rd_state_type())]
    }

    fn execute(&self, inputs: &[Value], _ctx: &EvalContext) -> Result<Vec<Value>, GraphError> {
        let prev = inputs[0]
            .downcast_ref::<ReactionDiffusion>()
            .ok_or_else(|| {
                GraphError::ExecutionError(
                    "rd::feedback::Step expects a ReactionDiffusion state input".to_string(),
                )
            })?;
        // Pure clone-and-advance via the existing op.
        let next = crate::Step { count: 1 }.apply(prev);
        Ok(vec![Value::opaque(next)])
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::GrayScottPreset;
    use unshape_core::{FeedbackState, Graph};

    fn seeded_rd() -> ReactionDiffusion {
        let mut rd = ReactionDiffusion::new(32, 32);
        rd.set_preset(GrayScottPreset::Coral);
        rd.add_seed_circle(16, 16, 4);
        rd
    }

    /// Build the self-wired step graph and seed initial state into `FeedbackState`.
    fn build() -> (Graph, u32, FeedbackState) {
        let mut graph = Graph::new();
        let step = graph.add_node(Step);
        graph.connect_feedback(step, 0, step, 0).unwrap();
        let mut state = FeedbackState::new();
        state.set(step, 0, Value::opaque(seeded_rd()));
        (graph, step, state)
    }

    fn v_sum(rd: &ReactionDiffusion) -> f64 {
        rd.v_buffer().iter().map(|&x| x as f64).sum()
    }

    #[test]
    fn evolves_like_mut_step_loop() {
        // (a) feedback stepping N times matches the old &mut step loop N times.
        let n = 20u64;

        // Reference: imperative loop.
        let mut reference = seeded_rd();
        reference.steps(n as usize);

        // Feedback driver: tick 0..n (n+1 ticks would over-step; we want N steps).
        // tick 0 applies step #1, ..., tick N-1 applies step #N.
        let (mut graph, step, mut state) = build();
        let mut last = None;
        for t in 0..n {
            let r = graph.tick(t, &mut state, &EvalContext::new()).unwrap();
            last = Some(
                r.get(step, 0)
                    .unwrap()
                    .downcast_ref::<ReactionDiffusion>()
                    .unwrap()
                    .clone(),
            );
        }
        let evolved = last.unwrap();

        // Same grid contents.
        assert_eq!(evolved.width(), reference.width());
        for (a, b) in evolved.v_buffer().iter().zip(reference.v_buffer()) {
            assert_eq!(a, b);
        }
        for (a, b) in evolved.u_buffer().iter().zip(reference.u_buffer()) {
            assert_eq!(a, b);
        }
    }

    #[test]
    fn node_is_pure_fresh_state_restarts() {
        // (b) the node holds no state: a fresh FeedbackState restarts from seed.
        let (mut graph, step, mut state) = build();
        for t in 0..5 {
            graph.tick(t, &mut state, &EvalContext::new()).unwrap();
        }

        // Fresh seeded state, single tick -> equals one step from the seed.
        let mut fresh = FeedbackState::new();
        fresh.set(step, 0, Value::opaque(seeded_rd()));
        let r = graph.tick(0, &mut fresh, &EvalContext::new()).unwrap();
        let one = r
            .get(step, 0)
            .unwrap()
            .downcast_ref::<ReactionDiffusion>()
            .unwrap()
            .clone();

        let mut reference = seeded_rd();
        reference.step();
        for (a, b) in one.v_buffer().iter().zip(reference.v_buffer()) {
            assert_eq!(a, b);
        }
    }

    #[test]
    fn deterministic() {
        // (c) same seed + inputs + N -> identical output.
        let run = || {
            let (mut graph, step, mut state) = build();
            let mut last = 0.0;
            for t in 0..15 {
                let r = graph.tick(t, &mut state, &EvalContext::new()).unwrap();
                last = v_sum(
                    r.get(step, 0)
                        .unwrap()
                        .downcast_ref::<ReactionDiffusion>()
                        .unwrap(),
                );
            }
            last
        };
        assert_eq!(run(), run());
    }

    #[test]
    fn stepping_0_to_n_is_deterministic() {
        // (d) stepping 0..=N is reproducible (the per-tick logic the resimulate
        // driver replays). NOTE: `Graph::run_to_tick` cannot be used directly
        // here because it calls `state.clear()` before tick 0, and an opaque
        // state type has no `zero_value()` to re-seed from — see
        // `run_to_tick_errors_on_opaque_seed` below. Resimulating an
        // opaque-state sim requires a seed source (e.g. a const/seed node) rather
        // than a pre-seeded FeedbackState; that is a follow-on, not a defect of
        // this port.
        let target = 8u64;

        let step_manually = || {
            let (mut graph, step, mut state) = build();
            let mut last = None;
            for t in 0..=target {
                let r = graph.tick(t, &mut state, &EvalContext::new()).unwrap();
                last = Some(
                    r.get(step, 0)
                        .unwrap()
                        .downcast_ref::<ReactionDiffusion>()
                        .unwrap()
                        .clone(),
                );
            }
            last.unwrap()
        };
        let a = step_manually();
        let b = step_manually();
        for (x, y) in a.v_buffer().iter().zip(b.v_buffer()) {
            assert_eq!(x, y);
        }
    }

    #[test]
    fn run_to_tick_errors_on_opaque_seed() {
        // Documents the opaque-state + run_to_tick limitation: run_to_tick clears
        // state, then tick 0 finds no seed and no zero_value for the Custom type.
        let (mut graph, step, mut state) = build();
        let _ = step;
        let r = graph.run_to_tick(3, &mut state, |_t| EvalContext::new());
        assert!(matches!(r, Err(GraphError::ExecutionError(_))));
    }
}
