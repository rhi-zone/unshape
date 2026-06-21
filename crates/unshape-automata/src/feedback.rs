//! Recurrent-graph integration for cellular automata.
//!
//! Ports the stateful automata simulators onto the feedback-edge mechanism in
//! `unshape-core` (`docs/design/recurrent-graphs.md`). Each automaton's mutable
//! state (its grid/board) is carried on a feedback wire as an opaque [`Value`],
//! so the step nodes are pure `&self` [`DynNode`]s: they read the previous
//! state, advance one step (cloning and calling the existing native
//! `step(&mut self)`), and return the new state. No state lives in the node.
//!
//! There is one `*Init` source node and one `*Step` node per distinct automaton
//! type covered here:
//!
//! - [`ElementaryInit`] / [`ElementaryStep`] — 1D Wolfram elementary CA
//!   ([`ElementaryCA`]).
//! - [`LifeInit`] / [`LifeStep`] — 2D life-like CA ([`CellularAutomaton2D`]).
//! - [`SmoothLifeInit`] / [`SmoothLifeStep`] — continuous-state SmoothLife
//!   ([`SmoothLife`]).
//!
//! Seeding (`set_center` / `randomize`) is deterministic: `randomize` is fully
//! seeded via the engine's `SimpleRng`, so the same config always yields the
//! same initial grid. The step rules are deterministic and hold no RNG, so the
//! carried state is the entire simulation state — nothing else needs to travel
//! on the feedback edge.
//!
//! # Usage
//!
//! ```
//! use unshape_automata::feedback::{ElementaryInit, ElementaryStep};
//! use unshape_core::{Graph, FeedbackState, EvalContext};
//!
//! let mut graph = Graph::new();
//! let init = graph.add_node(ElementaryInit::center(64, 30));
//! let step = graph.add_node(ElementaryStep);
//! graph.connect(init, 0, step, 0).unwrap();            // Init -> state (direct)
//! graph.connect_recurrence(step, 0, step, 0).unwrap(); // evolve: state -> state
//!
//! let mut state = FeedbackState::new();
//! let r = graph.run_to_tick(5, &mut state, |_t| EvalContext::new()).unwrap();
//! assert!(r.get(step, 0).is_some());
//! ```

use std::any::Any;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use unshape_core::{
    DataLocation, DynNode, EvalContext, GraphError, GraphValue, PortDescriptor, Value, ValueType,
};

use crate::{CellularAutomaton2D, ElementaryCA, SmoothLife, SmoothLifeConfig};

// ===========================================================================
// Shared GraphValue impls for the carried state types
// ===========================================================================

/// The opaque value type name for [`ElementaryCA`] state on a wire.
pub const ELEMENTARY_STATE_NAME: &str = "ElementaryCA";
/// The opaque value type name for [`CellularAutomaton2D`] state on a wire.
pub const LIFE_STATE_NAME: &str = "CellularAutomaton2D";
/// The opaque value type name for [`SmoothLife`] state on a wire.
pub const SMOOTH_LIFE_STATE_NAME: &str = "SmoothLife";

impl GraphValue for ElementaryCA {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn type_name(&self) -> &'static str {
        ELEMENTARY_STATE_NAME
    }
    fn location(&self) -> DataLocation {
        DataLocation::Cpu
    }
}

impl GraphValue for CellularAutomaton2D {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn type_name(&self) -> &'static str {
        LIFE_STATE_NAME
    }
    fn location(&self) -> DataLocation {
        DataLocation::Cpu
    }
}

impl GraphValue for SmoothLife {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn type_name(&self) -> &'static str {
        SMOOTH_LIFE_STATE_NAME
    }
    fn location(&self) -> DataLocation {
        DataLocation::Cpu
    }
}

/// Returns the [`ValueType`] used for [`ElementaryCA`] state on a wire.
pub fn elementary_state_type() -> ValueType {
    ValueType::of::<ElementaryCA>(ELEMENTARY_STATE_NAME)
}

/// Returns the [`ValueType`] used for [`CellularAutomaton2D`] state on a wire.
pub fn life_state_type() -> ValueType {
    ValueType::of::<CellularAutomaton2D>(LIFE_STATE_NAME)
}

/// Returns the [`ValueType`] used for [`SmoothLife`] state on a wire.
pub fn smooth_life_state_type() -> ValueType {
    ValueType::of::<SmoothLife>(SMOOTH_LIFE_STATE_NAME)
}

// ===========================================================================
// Elementary (1D)
// ===========================================================================

/// How an [`ElementaryInit`] seeds the initial 1D grid (pure data).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum ElementarySeed {
    /// Single live cell in the center (common Wolfram starting condition).
    Center,
    /// Fully random row with the given deterministic seed.
    Random {
        /// RNG seed (deterministic).
        seed: u64,
    },
}

/// Pure in-graph **source** node producing the initial [`ElementaryCA`] state.
///
/// Takes no inputs; outputs a freshly seeded 1D CA from its config (width, rule,
/// and [`ElementarySeed`]). Deterministic.
///
/// # Ports
/// - Output `0` `"state"`: `Custom(ElementaryCA)` — initial state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ElementaryInit {
    /// Grid width.
    pub width: usize,
    /// Wolfram rule number (0-255).
    pub rule: u8,
    /// Initial seed.
    pub seed: ElementarySeed,
}

impl ElementaryInit {
    /// Creates an init node seeding a single center cell.
    pub fn center(width: usize, rule: u8) -> Self {
        Self {
            width,
            rule,
            seed: ElementarySeed::Center,
        }
    }

    /// Creates an init node seeding a deterministic random row.
    pub fn random(width: usize, rule: u8, seed: u64) -> Self {
        Self {
            width,
            rule,
            seed: ElementarySeed::Random { seed },
        }
    }

    /// Builds the initial [`ElementaryCA`] from this config (pure).
    pub fn build(&self) -> ElementaryCA {
        let mut ca = ElementaryCA::new(self.width, self.rule);
        match self.seed {
            ElementarySeed::Center => ca.set_center(),
            ElementarySeed::Random { seed } => ca.randomize(seed),
        }
        ca
    }
}

impl DynNode for ElementaryInit {
    fn type_name(&self) -> &'static str {
        "automata::feedback::ElementaryInit"
    }
    fn inputs(&self) -> Vec<PortDescriptor> {
        vec![]
    }
    fn outputs(&self) -> Vec<PortDescriptor> {
        vec![PortDescriptor::new("state", elementary_state_type())]
    }
    fn execute(&self, _inputs: &[Value], _ctx: &EvalContext) -> Result<Vec<Value>, GraphError> {
        Ok(vec![Value::opaque(self.build())])
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Pure step node for the 1D elementary CA.
///
/// State lives on the feedback edge. `execute` clones the previous
/// [`ElementaryCA`], advances it by one native `step(&mut self)`, and returns the
/// new state.
///
/// # Ports
/// - Input `0` `"state"`: `Custom(ElementaryCA)` — previous-tick state.
/// - Output `0` `"state"`: `Custom(ElementaryCA)` — advanced state.
#[derive(Debug, Clone, Copy, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ElementaryStep;

impl DynNode for ElementaryStep {
    fn type_name(&self) -> &'static str {
        "automata::feedback::ElementaryStep"
    }
    fn inputs(&self) -> Vec<PortDescriptor> {
        vec![PortDescriptor::new("state", elementary_state_type())]
    }
    fn outputs(&self) -> Vec<PortDescriptor> {
        vec![PortDescriptor::new("state", elementary_state_type())]
    }
    fn execute(&self, inputs: &[Value], _ctx: &EvalContext) -> Result<Vec<Value>, GraphError> {
        let prev = inputs[0].downcast_ref::<ElementaryCA>().ok_or_else(|| {
            GraphError::ExecutionError(
                "automata::feedback::ElementaryStep expects an ElementaryCA state input"
                    .to_string(),
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

// ===========================================================================
// Life (2D)
// ===========================================================================

/// Pure in-graph **source** node producing the initial [`CellularAutomaton2D`].
///
/// Takes no inputs; outputs a freshly seeded 2D life-like CA from its config
/// (size, B/S rules, and a deterministic randomized density). Deterministic.
///
/// # Ports
/// - Output `0` `"state"`: `Custom(CellularAutomaton2D)` — initial state.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct LifeInit {
    /// Grid width.
    pub width: usize,
    /// Grid height.
    pub height: usize,
    /// Birth rule (neighbor counts that cause birth).
    pub birth: Vec<u8>,
    /// Survival rule (neighbor counts that allow survival).
    pub survive: Vec<u8>,
    /// RNG seed for the deterministic initial randomization.
    pub seed: u64,
    /// Initial alive density (0.0 to 1.0).
    pub density: f32,
}

impl LifeInit {
    /// Creates a Game-of-Life (B3/S23) init with a randomized grid.
    pub fn life(width: usize, height: usize, seed: u64, density: f32) -> Self {
        Self {
            width,
            height,
            birth: vec![3],
            survive: vec![2, 3],
            seed,
            density,
        }
    }

    /// Creates a life-like init with custom birth/survival rules.
    pub fn new(
        width: usize,
        height: usize,
        birth: &[u8],
        survive: &[u8],
        seed: u64,
        density: f32,
    ) -> Self {
        Self {
            width,
            height,
            birth: birth.to_vec(),
            survive: survive.to_vec(),
            seed,
            density,
        }
    }

    /// Builds the initial [`CellularAutomaton2D`] from this config (pure).
    pub fn build(&self) -> CellularAutomaton2D {
        let mut ca = CellularAutomaton2D::new(self.width, self.height, &self.birth, &self.survive);
        ca.randomize(self.seed, self.density);
        ca
    }
}

impl DynNode for LifeInit {
    fn type_name(&self) -> &'static str {
        "automata::feedback::LifeInit"
    }
    fn inputs(&self) -> Vec<PortDescriptor> {
        vec![]
    }
    fn outputs(&self) -> Vec<PortDescriptor> {
        vec![PortDescriptor::new("state", life_state_type())]
    }
    fn execute(&self, _inputs: &[Value], _ctx: &EvalContext) -> Result<Vec<Value>, GraphError> {
        Ok(vec![Value::opaque(self.build())])
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Pure step node for the 2D life-like CA.
///
/// # Ports
/// - Input `0` `"state"`: `Custom(CellularAutomaton2D)` — previous-tick state.
/// - Output `0` `"state"`: `Custom(CellularAutomaton2D)` — advanced state.
#[derive(Debug, Clone, Copy, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct LifeStep;

impl DynNode for LifeStep {
    fn type_name(&self) -> &'static str {
        "automata::feedback::LifeStep"
    }
    fn inputs(&self) -> Vec<PortDescriptor> {
        vec![PortDescriptor::new("state", life_state_type())]
    }
    fn outputs(&self) -> Vec<PortDescriptor> {
        vec![PortDescriptor::new("state", life_state_type())]
    }
    fn execute(&self, inputs: &[Value], _ctx: &EvalContext) -> Result<Vec<Value>, GraphError> {
        let prev = inputs[0]
            .downcast_ref::<CellularAutomaton2D>()
            .ok_or_else(|| {
                GraphError::ExecutionError(
                    "automata::feedback::LifeStep expects a CellularAutomaton2D state input"
                        .to_string(),
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

// ===========================================================================
// SmoothLife (continuous-state)
// ===========================================================================

/// Pure in-graph **source** node producing the initial [`SmoothLife`] state.
///
/// Takes no inputs; outputs a freshly seeded SmoothLife grid from its config
/// (size, [`SmoothLifeConfig`], and deterministic randomized density).
///
/// # Ports
/// - Output `0` `"state"`: `Custom(SmoothLife)` — initial state.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct SmoothLifeInit {
    /// Grid width.
    pub width: usize,
    /// Grid height.
    pub height: usize,
    /// SmoothLife configuration.
    pub config: SmoothLifeConfig,
    /// RNG seed for the deterministic initial randomization.
    pub seed: u64,
    /// Initial density (0.0 to 1.0).
    pub density: f32,
}

impl SmoothLifeInit {
    /// Creates a SmoothLife init with the given config and randomized grid.
    pub fn new(
        width: usize,
        height: usize,
        config: SmoothLifeConfig,
        seed: u64,
        density: f32,
    ) -> Self {
        Self {
            width,
            height,
            config,
            seed,
            density,
        }
    }

    /// Builds the initial [`SmoothLife`] from this config (pure).
    pub fn build(&self) -> SmoothLife {
        let mut sl = SmoothLife::new(self.width, self.height, self.config);
        sl.randomize(self.seed, self.density);
        sl
    }
}

impl DynNode for SmoothLifeInit {
    fn type_name(&self) -> &'static str {
        "automata::feedback::SmoothLifeInit"
    }
    fn inputs(&self) -> Vec<PortDescriptor> {
        vec![]
    }
    fn outputs(&self) -> Vec<PortDescriptor> {
        vec![PortDescriptor::new("state", smooth_life_state_type())]
    }
    fn execute(&self, _inputs: &[Value], _ctx: &EvalContext) -> Result<Vec<Value>, GraphError> {
        Ok(vec![Value::opaque(self.build())])
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Pure step node for the continuous-state SmoothLife.
///
/// `dt` is config on the node (a field), not a per-tick external input, since the
/// native `SmoothLife::step(dt)` takes the time step as a parameter.
///
/// # Ports
/// - Input `0` `"state"`: `Custom(SmoothLife)` — previous-tick state.
/// - Output `0` `"state"`: `Custom(SmoothLife)` — advanced state.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct SmoothLifeStep {
    /// Time step per tick.
    pub dt: f32,
}

impl SmoothLifeStep {
    /// Creates a step node with the given time step.
    pub fn new(dt: f32) -> Self {
        Self { dt }
    }
}

impl Default for SmoothLifeStep {
    fn default() -> Self {
        Self { dt: 0.1 }
    }
}

impl DynNode for SmoothLifeStep {
    fn type_name(&self) -> &'static str {
        "automata::feedback::SmoothLifeStep"
    }
    fn inputs(&self) -> Vec<PortDescriptor> {
        vec![PortDescriptor::new("state", smooth_life_state_type())]
    }
    fn outputs(&self) -> Vec<PortDescriptor> {
        vec![PortDescriptor::new("state", smooth_life_state_type())]
    }
    fn execute(&self, inputs: &[Value], _ctx: &EvalContext) -> Result<Vec<Value>, GraphError> {
        let prev = inputs[0].downcast_ref::<SmoothLife>().ok_or_else(|| {
            GraphError::ExecutionError(
                "automata::feedback::SmoothLifeStep expects a SmoothLife state input".to_string(),
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
    use unshape_core::{EvalContext, FeedbackState, Graph};

    fn run_feedback<I, S, T, F>(init: I, step: S, n: u64, extract: F) -> T
    where
        I: DynNode + 'static,
        S: DynNode + Clone + 'static,
        F: Fn(&Value) -> T,
    {
        let mut graph = Graph::new();
        let i = graph.add_node(init);
        let s = graph.add_node(step);
        graph.connect(i, 0, s, 0).unwrap();
        graph.connect_recurrence(s, 0, s, 0).unwrap();
        let mut state = FeedbackState::new();
        let mut last = None;
        for t in 0..n {
            let r = graph.tick(t, &mut state, &EvalContext::new()).unwrap();
            last = Some(extract(r.get(s, 0).unwrap()));
        }
        last.unwrap()
    }

    #[test]
    fn elementary_evolves_like_mut_step_loop() {
        let n = 25u64;
        let init = ElementaryInit::random(64, 30, 999);

        let mut reference = init.build();
        reference.steps(n as usize);

        let evolved = run_feedback(init, ElementaryStep, n, |v| {
            v.downcast_ref::<ElementaryCA>().unwrap().cells().to_vec()
        });
        assert_eq!(evolved, reference.cells());
    }

    #[test]
    fn life_evolves_like_mut_step_loop() {
        let n = 15u64;
        let init = LifeInit::life(48, 48, 12345, 0.3);

        let mut reference = init.build();
        reference.steps(n as usize);

        let evolved = run_feedback(init, LifeStep, n, |v| {
            v.downcast_ref::<CellularAutomaton2D>()
                .unwrap()
                .cells()
                .clone()
        });
        assert_eq!(&evolved, reference.cells());
    }

    #[test]
    fn smooth_life_evolves_like_mut_step_loop() {
        let n = 8u64;
        let dt = 0.1f32;
        let init = SmoothLifeInit::new(40, 40, SmoothLifeConfig::default(), 7, 0.3);

        let mut reference = init.build();
        reference.steps(n as usize, dt);

        let evolved = run_feedback(init, SmoothLifeStep::new(dt), n, |v| {
            v.downcast_ref::<SmoothLife>().unwrap().cells().clone()
        });
        assert_eq!(&evolved, reference.cells());
    }

    #[test]
    fn run_to_tick_matches_manual_for_elementary() {
        let target = 10u64;
        let init = ElementaryInit::center(33, 90);

        let manual = {
            let mut ca = init.build();
            ca.steps((target + 1) as usize);
            ca.cells().to_vec()
        };

        let mut graph = Graph::new();
        let i = graph.add_node(init);
        let s = graph.add_node(ElementaryStep);
        graph.connect(i, 0, s, 0).unwrap();
        graph.connect_recurrence(s, 0, s, 0).unwrap();
        let mut state = FeedbackState::new();
        let r = graph
            .run_to_tick(target, &mut state, |_t| EvalContext::new())
            .unwrap();
        let resimulated = r
            .get(s, 0)
            .unwrap()
            .downcast_ref::<ElementaryCA>()
            .unwrap()
            .cells()
            .to_vec();
        assert_eq!(resimulated, manual);
    }
}
