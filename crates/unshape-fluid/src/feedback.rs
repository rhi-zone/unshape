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
//! The initial [`FluidGrid2D`] is produced by an in-graph [`FluidInit`] source
//! node (a pure `&self` `DynNode` taking no inputs) wired into the [`Step`]'s
//! state port with a *direct* edge, alongside the feedback self-loop. On tick 0
//! the `Init` node seeds the grid; later ticks use the carried feedback value.
//! This makes the sim rewindable via
//! [`Graph::run_to_tick`](unshape_core::Graph::run_to_tick) with no manual seed.

use std::any::Any;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use unshape_core::{
    DataLocation, DynNode, EvalContext, GraphError, GraphValue, PortDescriptor, Value, ValueType,
};

use crate::{FluidConfig, FluidGrid2D};

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

/// A pure-data source applied to a fresh grid by [`FluidInit`].
///
/// Mirrors the imperative `add_density` / `add_velocity` helpers on
/// [`FluidGrid2D`].
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum FluidSource {
    /// Inject density at a cell.
    Density {
        /// Cell x.
        x: usize,
        /// Cell y.
        y: usize,
        /// Amount of density to add.
        amount: f32,
    },
    /// Inject velocity at a cell.
    Velocity {
        /// Cell x.
        x: usize,
        /// Cell y.
        y: usize,
        /// X velocity.
        vx: f32,
        /// Y velocity.
        vy: f32,
    },
}

/// Pure in-graph **source** node producing the initial [`FluidGrid2D`].
///
/// Takes no inputs; outputs a fresh grid of `width`×`height` with the given
/// [`FluidConfig`], after applying each [`FluidSource`] in order. Deterministic.
///
/// Wire this into a [`Step`]'s state port with a *direct* edge (tick-0 seed),
/// alongside the [`Step`]'s feedback self-loop.
///
/// # Ports
/// - Output `0` `"state"`: `Custom(FluidGrid2D)` — initial grid.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct FluidInit {
    /// Grid width.
    pub width: usize,
    /// Grid height.
    pub height: usize,
    /// Simulation configuration (diffusion, iterations, dt).
    pub config: FluidConfig,
    /// Sources applied to the fresh grid, in order.
    pub sources: Vec<FluidSource>,
}

impl FluidInit {
    /// Creates an init node with the given size and config and no sources.
    pub fn new(width: usize, height: usize, config: FluidConfig) -> Self {
        Self {
            width,
            height,
            config,
            sources: Vec::new(),
        }
    }

    /// Adds a density source.
    pub fn with_density(mut self, x: usize, y: usize, amount: f32) -> Self {
        self.sources.push(FluidSource::Density { x, y, amount });
        self
    }

    /// Adds a velocity source.
    pub fn with_velocity(mut self, x: usize, y: usize, vx: f32, vy: f32) -> Self {
        self.sources.push(FluidSource::Velocity { x, y, vx, vy });
        self
    }

    /// Builds the initial [`FluidGrid2D`] from this config (pure).
    pub fn build(&self) -> FluidGrid2D {
        let mut g = FluidGrid2D::new(self.width, self.height, self.config.clone());
        for source in &self.sources {
            match *source {
                FluidSource::Density { x, y, amount } => g.add_density(x, y, amount),
                FluidSource::Velocity { x, y, vx, vy } => g.add_velocity(x, y, vx, vy),
            }
        }
        g
    }
}

impl DynNode for FluidInit {
    fn type_name(&self) -> &'static str {
        "fluid::feedback::FluidInit"
    }

    fn inputs(&self) -> Vec<PortDescriptor> {
        vec![]
    }

    fn outputs(&self) -> Vec<PortDescriptor> {
        vec![PortDescriptor::new("state", fluid_grid_2d_type())]
    }

    fn execute(&self, _inputs: &[Value], _ctx: &EvalContext) -> Result<Vec<Value>, GraphError> {
        Ok(vec![Value::opaque(self.build())])
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
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

    fn init_node() -> FluidInit {
        FluidInit::new(32, 32, FluidConfig::default())
            .with_density(16, 16, 100.0)
            .with_velocity(16, 16, 5.0, 2.0)
    }

    fn seeded_grid() -> FluidGrid2D {
        init_node().build()
    }

    /// A built feedback graph: `Init --direct--> Step.state`, `Step --feedback--> Step.state`.
    struct Built {
        graph: Graph,
        step: u32,
    }

    fn build() -> Built {
        let mut graph = Graph::new();
        let init = graph.add_node(init_node());
        let step = graph.add_node(Step);
        graph.connect(init, 0, step, 0).unwrap();
        graph.connect_recurrence(step, 0, step, 0).unwrap();
        Built { graph, step }
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

        let Built { mut graph, step } = build();
        let mut state = FeedbackState::new();
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
        // (b) the node holds no state: a fresh seed restarts from one step (the
        // Init source re-seeds tick 0).
        let Built { mut graph, step } = build();
        let mut state = FeedbackState::new();
        for t in 0..5 {
            graph.tick(t, &mut state, &EvalContext::new()).unwrap();
        }

        let mut fresh = FeedbackState::new();
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
            let Built { mut graph, step } = build();
            let mut state = FeedbackState::new();
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
    fn run_to_tick_matches_manual_stepping() {
        // (d) run_to_tick now SUCCEEDS: the in-graph FluidInit source re-seeds
        // tick 0 after run_to_tick clears state. Resimulated == manual stepping.
        let target = 10u64;

        let manual = {
            let Built { mut graph, step } = build();
            let mut state = FeedbackState::new();
            let mut last = None;
            for t in 0..=target {
                let r = graph.tick(t, &mut state, &EvalContext::new()).unwrap();
                last = Some(
                    r.get(step, 0)
                        .unwrap()
                        .downcast_ref::<FluidGrid2D>()
                        .unwrap()
                        .clone(),
                );
            }
            last.unwrap()
        };

        let Built { mut graph, step } = build();
        let mut state = FeedbackState::new();
        let r = graph
            .run_to_tick(target, &mut state, |_t| EvalContext::new())
            .unwrap();
        let resimulated = r
            .get(step, 0)
            .unwrap()
            .downcast_ref::<FluidGrid2D>()
            .unwrap();

        for (a, b) in resimulated
            .density_field()
            .iter()
            .zip(manual.density_field())
        {
            assert_eq!(a, b);
        }
        let (rvx, rvy) = resimulated.velocity_field();
        let (mvx, mvy) = manual.velocity_field();
        for (a, b) in rvx.iter().zip(mvx) {
            assert_eq!(a, b);
        }
        for (a, b) in rvy.iter().zip(mvy) {
            assert_eq!(a, b);
        }
    }

    #[test]
    fn seek_resimulate_is_deterministic() {
        // (e) seek(Resimulate) works and reproduces.
        let seek_to = |target: u64| {
            let Built { mut graph, step } = build();
            let mut state = FeedbackState::new();
            let r = graph
                .seek(
                    target,
                    0,
                    unshape_core::SeekBehavior::Resimulate,
                    &mut state,
                    |_t| EvalContext::new(),
                )
                .unwrap();
            density_sum(
                r.get(step, 0)
                    .unwrap()
                    .downcast_ref::<FluidGrid2D>()
                    .unwrap(),
            )
        };
        assert_eq!(seek_to(7), seek_to(7));
    }
}
