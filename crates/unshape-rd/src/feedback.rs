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
//! The initial state is produced by an in-graph [`GrayScottInit`] source node
//! (a pure `&self` `DynNode` taking no state input), wired into a [`Latch`]'s
//! `init` (seed) port. The latch's `out` feeds the [`Step`]'s state input, and
//! the `Step`'s output feeds the latch's `signal`. On tick 0 the latch emits the
//! seed; on later ticks it emits the previously-captured step output. This makes
//! the graph fully rewindable via
//! [`Graph::run_to_tick_latched`](unshape_core::Graph::run_to_tick_latched) with
//! no manual seed.
//!
//! ```
//! use unshape_rd::feedback::{GrayScottInit, Step, rd_state_type};
//! use unshape_core::{Graph, Latch, LatchSnapshot, EvalContext};
//!
//! let mut graph = Graph::new();
//! let init = graph.add_node(GrayScottInit::circle(32, 32, 16, 16, 4));
//! let latch = graph.add_node(Latch::new(rd_state_type()));
//! let step = graph.add_node(Step);
//! graph.connect(init, 0, latch, 0).unwrap();  // Init -> latch.init (seed)
//! graph.connect(latch, 0, step, 0).unwrap();  // latch.out -> step.state
//! graph.connect(step, 0, latch, 1).unwrap();  // step.state -> latch.signal
//!
//! let mut state = LatchSnapshot::new();
//! // Resimulate to tick 5 — no manual pre-seed; tick 0 resolves via Init.
//! let r = graph.run_to_tick_latched(5, &mut state, |_t| EvalContext::new()).unwrap();
//! assert!(r.get(step, 0).is_some());
//! ```

use std::any::Any;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use unshape_core::{
    DataLocation, DynNode, EvalContext, GraphError, GraphValue, PortDescriptor, Value, ValueType,
};

use crate::{GrayScottPreset, ReactionDiffusion};

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

/// The initial seed pattern for a [`GrayScottInit`] source node.
///
/// Pure-data description of how to seed chemical V into a fresh grid; mirrors the
/// imperative seeding helpers on [`ReactionDiffusion`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum SeedPattern {
    /// A single circular seed of V at `(cx, cy)` with the given radius.
    Circle {
        /// Center x.
        cx: usize,
        /// Center y.
        cy: usize,
        /// Radius.
        radius: usize,
    },
    /// `count` random circular seeds of `radius`, placed with the given seed.
    Random {
        /// Number of seeds.
        count: usize,
        /// Radius of each seed.
        radius: usize,
        /// RNG seed (deterministic).
        seed: u64,
    },
}

/// Pure in-graph **source** node producing the initial Gray-Scott state.
///
/// Takes no inputs; outputs a freshly seeded [`ReactionDiffusion`] grid from its
/// config (grid size, preset, and [`SeedPattern`]). Deterministic — the random
/// seed pattern is fully seeded, so the same config always yields the same grid.
///
/// Wire this into a [`Step`]'s state port with a *direct* edge (the tick-0 seed),
/// alongside the [`Step`]'s feedback self-loop.
///
/// # Ports
/// - Output `0` `"state"`: `Custom(ReactionDiffusion)` — initial state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct GrayScottInit {
    /// Grid width.
    pub width: usize,
    /// Grid height.
    pub height: usize,
    /// Gray-Scott parameter preset.
    pub preset: GrayScottPreset,
    /// Initial seed pattern for chemical V.
    pub seed: SeedPattern,
}

impl GrayScottInit {
    /// Creates an init node with a single circular seed.
    pub fn circle(width: usize, height: usize, cx: usize, cy: usize, radius: usize) -> Self {
        Self {
            width,
            height,
            preset: GrayScottPreset::Coral,
            seed: SeedPattern::Circle { cx, cy, radius },
        }
    }

    /// Creates an init node with random seeds.
    pub fn random(width: usize, height: usize, count: usize, radius: usize, seed: u64) -> Self {
        Self {
            width,
            height,
            preset: GrayScottPreset::Coral,
            seed: SeedPattern::Random {
                count,
                radius,
                seed,
            },
        }
    }

    /// Sets the Gray-Scott parameter preset.
    pub fn with_preset(mut self, preset: GrayScottPreset) -> Self {
        self.preset = preset;
        self
    }

    /// Builds the initial [`ReactionDiffusion`] grid from this config (pure).
    pub fn build(&self) -> ReactionDiffusion {
        let mut rd = ReactionDiffusion::new(self.width, self.height);
        rd.set_preset(self.preset);
        match self.seed {
            SeedPattern::Circle { cx, cy, radius } => rd.add_seed_circle(cx, cy, radius),
            SeedPattern::Random {
                count,
                radius,
                seed,
            } => rd.add_random_seeds(count, radius, seed),
        }
        rd
    }
}

impl DynNode for GrayScottInit {
    fn type_name(&self) -> &'static str {
        "rd::feedback::GrayScottInit"
    }

    fn inputs(&self) -> Vec<PortDescriptor> {
        vec![]
    }

    fn outputs(&self) -> Vec<PortDescriptor> {
        vec![PortDescriptor::new("state", rd_state_type())]
    }

    fn execute(&self, _inputs: &[Value], _ctx: &EvalContext) -> Result<Vec<Value>, GraphError> {
        Ok(vec![Value::opaque(self.build())])
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
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
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
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
    use unshape_core::{Graph, Latch, LatchSnapshot};

    fn init_node() -> GrayScottInit {
        GrayScottInit::circle(32, 32, 16, 16, 4).with_preset(GrayScottPreset::Coral)
    }

    fn seeded_rd() -> ReactionDiffusion {
        init_node().build()
    }

    /// A built feedback graph: `Init --direct--> Step.state`, `Step --feedback--> Step.state`.
    struct Built {
        graph: Graph,
        step: u32,
    }

    /// Build the Init-seeded, self-wired step graph (no manual FeedbackState seed).
    fn build() -> Built {
        let mut graph = Graph::new();
        let init = graph.add_node(init_node());
        let latch = graph.add_node(Latch::new(rd_state_type()));
        let step = graph.add_node(Step);
        graph.connect(init, 0, latch, 0).unwrap(); // Init -> latch.init (seed)
        graph.connect(latch, 0, step, 0).unwrap(); // latch.out -> step.state
        graph.connect(step, 0, latch, 1).unwrap(); // step.state -> latch.signal
        Built { graph, step }
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
        let Built { mut graph, step } = build();
        let mut state = LatchSnapshot::new();
        let mut last = None;
        for t in 0..n {
            let r = graph
                .tick_latched(t, &mut state, &EvalContext::new())
                .unwrap();
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
        // (b) the node holds no state: a fresh FeedbackState restarts from seed
        // (the Init source re-seeds tick 0).
        let Built { mut graph, step } = build();
        let mut state = LatchSnapshot::new();
        for t in 0..5 {
            graph
                .tick_latched(t, &mut state, &EvalContext::new())
                .unwrap();
        }

        // Fresh feedback state, single tick -> equals one step from the seed.
        let mut fresh = LatchSnapshot::new();
        let r = graph
            .tick_latched(0, &mut fresh, &EvalContext::new())
            .unwrap();
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
            let Built { mut graph, step } = build();
            let mut state = LatchSnapshot::new();
            let mut last = 0.0;
            for t in 0..15 {
                let r = graph
                    .tick_latched(t, &mut state, &EvalContext::new())
                    .unwrap();
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
    fn run_to_tick_matches_manual_stepping() {
        // (d) run_to_tick now SUCCEEDS for the opaque-state sim: the in-graph
        // GrayScottInit source re-seeds tick 0 after run_to_tick clears state.
        // The resimulated result equals manually stepping 0..=N from the seed.
        let target = 8u64;

        // Manual stepping (the old &mut-style loop, via the feedback driver).
        let manual = {
            let Built { mut graph, step } = build();
            let mut state = LatchSnapshot::new();
            let mut last = None;
            for t in 0..=target {
                let r = graph
                    .tick_latched(t, &mut state, &EvalContext::new())
                    .unwrap();
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

        // run_to_tick / Resimulate — no manual pre-seed.
        let Built { mut graph, step } = build();
        let mut state = LatchSnapshot::new();
        let r = graph
            .run_to_tick_latched(target, &mut state, |_t| EvalContext::new())
            .unwrap();
        let resimulated = r
            .get(step, 0)
            .unwrap()
            .downcast_ref::<ReactionDiffusion>()
            .unwrap();

        assert_eq!(resimulated.width(), manual.width());
        for (x, y) in resimulated.v_buffer().iter().zip(manual.v_buffer()) {
            assert_eq!(x, y);
        }
        for (x, y) in resimulated.u_buffer().iter().zip(manual.u_buffer()) {
            assert_eq!(x, y);
        }
    }

    #[test]
    fn seek_resimulate_is_deterministic_and_reproducible() {
        // (e) seek(Resimulate) works; same config + N -> identical; fresh run
        // reproduces.
        let seek_to = |target: u64| {
            let Built { mut graph, step } = build();
            let mut state = LatchSnapshot::new();
            let r = graph
                .seek_latched(
                    target,
                    0,
                    unshape_core::SeekBehavior::Resimulate,
                    &mut state,
                    |_t| EvalContext::new(),
                )
                .unwrap();
            v_sum(
                r.get(step, 0)
                    .unwrap()
                    .downcast_ref::<ReactionDiffusion>()
                    .unwrap(),
            )
        };
        assert_eq!(seek_to(6), seek_to(6));
    }
}
