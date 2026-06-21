//! Recurrent-graph integration for Wave Function Collapse.
//!
//! Ports the stateful [`WfcSolver`] onto the feedback-edge mechanism in
//! `unshape-core` (`docs/design/recurrent-graphs.md`). The entire mutable state
//! — the cell grid *and* the seeded RNG state — is carried on a feedback wire as
//! an opaque [`Value`], so the step node ([`WfcStep`]) is a pure `&self`
//! [`DynNode`]: it clones the previous solver, collapses one cell via the
//! existing native [`WfcSolver::step`], and returns the new solver. No state
//! lives in the node.
//!
//! # RNG on the feedback edge (determinism)
//!
//! WFC is RNG-driven: cell selection among min-entropy candidates and weighted
//! tile collapse both draw from the solver's internal `rng_state`. That state is
//! part of [`WfcSolver`] and therefore travels on the feedback edge, so each tick
//! continues the same deterministic stream — cloning-and-advancing reproduces the
//! native `run(seed)` loop bit-for-bit. [`WfcInit`] seeds the stream once (via
//! [`WfcSolver::set_seed`], matching `run`'s internal `seed.wrapping_add(1)`);
//! nothing reseeds per tick.
//!
//! # Progress / fixpoint and errors
//!
//! The native `step(&mut self) -> Result<bool, WfcError>` returns `Ok(false)`
//! once every cell is collapsed (complete). At that fixpoint the step is
//! idempotent, so the bool is not surfaced as a port: a completed solve re-emits
//! the same state on every later tick. A [`WfcError`] (contradiction /
//! out-of-bounds) is mapped to [`GraphError::ExecutionError`] — the variant
//! designated for node-execution failures — preserving the specific cause in the
//! message.

use std::any::Any;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use unshape_core::{
    DataLocation, DynNode, EvalContext, GraphError, GraphValue, PortDescriptor, Value, ValueType,
};

use crate::{Direction, TileId, TileSet, WfcError, WfcSolver};

/// The concrete WFC state type carried on the feedback edge.
///
/// WFC is generic over its adjacency source; the recurrent integration fixes it
/// to the base [`TileSet`] (explicit adjacency), which is the common case and is
/// fully reconstructible from serializable [`WfcInit`] config.
pub type WfcState = WfcSolver<TileSet>;

/// The opaque value type name for [`WfcState`] on a wire.
pub const WFC_STATE_NAME: &str = "WfcSolver<TileSet>";

impl GraphValue for WfcState {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn type_name(&self) -> &'static str {
        WFC_STATE_NAME
    }

    fn location(&self) -> DataLocation {
        DataLocation::Cpu
    }
}

/// Returns the [`ValueType`] used for [`WfcState`] on a wire.
pub fn wfc_state_type() -> ValueType {
    ValueType::of::<WfcState>(WFC_STATE_NAME)
}

/// A pure-data adjacency rule for a [`TilesetDesc`].
///
/// `direction` indexes [`Direction::all`] (`0..4`). `from`/`to` are tile indices.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct RuleDesc {
    /// Source tile index.
    pub from: usize,
    /// Direction index into [`Direction::all`] (`0..4`).
    pub direction: u8,
    /// Neighbor tile index allowed in that direction.
    pub to: usize,
}

/// Pure-data description of a [`TileSet`].
///
/// Carries tile weights and adjacency rules so a [`WfcInit`] node holds the
/// tileset as serializable data rather than the non-serializable [`TileSet`].
#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct TilesetDesc {
    /// Per-tile weights; length determines the tile count.
    pub weights: Vec<f32>,
    /// Adjacency rules.
    pub rules: Vec<RuleDesc>,
}

impl TilesetDesc {
    /// Creates a tileset descriptor with `count` unit-weight tiles and no rules.
    pub fn new(count: usize) -> Self {
        Self {
            weights: vec![1.0; count],
            rules: Vec::new(),
        }
    }

    /// Adds an adjacency rule (also adds the reverse rule, like
    /// [`TileSet::add_rule`]).
    pub fn with_rule(mut self, from: usize, direction: Direction, to: usize) -> Self {
        let d = Direction::all()
            .iter()
            .position(|x| *x == direction)
            .unwrap() as u8;
        self.rules.push(RuleDesc {
            from,
            direction: d,
            to,
        });
        self
    }

    fn build(&self) -> TileSet {
        let mut ts = TileSet::new();
        let ids: Vec<TileId> = self
            .weights
            .iter()
            .map(|&w| ts.add_tile_weighted(w))
            .collect();
        let dirs = Direction::all();
        for rule in &self.rules {
            let dir = dirs[rule.direction as usize % dirs.len()];
            ts.add_rule(ids[rule.from], dir, ids[rule.to]);
        }
        ts
    }
}

/// Pure in-graph **source** node producing the initial [`WfcState`].
///
/// Takes no inputs; outputs a fresh, fully-uncollapsed WFC solver of the given
/// size over the [`TilesetDesc`] adjacency, with its RNG seeded (matching
/// `run(seed)`). Deterministic — the same config yields the same seeded solver.
///
/// # Ports
/// - Output `0` `"state"`: `Custom(WfcSolver<TileSet>)` — initial solver.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct WfcInit {
    /// Grid width.
    pub width: usize,
    /// Grid height.
    pub height: usize,
    /// Tileset (adjacency + weights).
    pub tileset: TilesetDesc,
    /// RNG seed (deterministic; matches `WfcSolver::run`).
    pub seed: u64,
}

impl WfcInit {
    /// Creates an init node from size, tileset, and seed.
    pub fn new(width: usize, height: usize, tileset: TilesetDesc, seed: u64) -> Self {
        Self {
            width,
            height,
            tileset,
            seed,
        }
    }

    /// Builds the initial seeded [`WfcState`] from this config (pure).
    pub fn build(&self) -> WfcState {
        let mut solver = WfcSolver::new(self.width, self.height, self.tileset.build());
        solver.set_seed(self.seed);
        solver
    }
}

impl DynNode for WfcInit {
    fn type_name(&self) -> &'static str {
        "procgen::feedback::WfcInit"
    }

    fn inputs(&self) -> Vec<PortDescriptor> {
        vec![]
    }

    fn outputs(&self) -> Vec<PortDescriptor> {
        vec![PortDescriptor::new("state", wfc_state_type())]
    }

    fn execute(&self, _inputs: &[Value], _ctx: &EvalContext) -> Result<Vec<Value>, GraphError> {
        Ok(vec![Value::opaque(self.build())])
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Maps a [`WfcError`] to the [`GraphError`] variant for execution failures.
///
/// Matches each variant explicitly (no catch-all) so new `WfcError` variants
/// force a compile-time decision here.
fn wfc_error_to_graph_error(err: WfcError) -> GraphError {
    let detail = match err {
        WfcError::Contradiction => "WFC reached a contradiction".to_string(),
        WfcError::OutOfBounds(x, y) => format!("WFC position ({x}, {y}) out of bounds"),
    };
    GraphError::ExecutionError(format!("procgen::feedback::WfcStep: {detail}"))
}

/// Pure per-tick step node for Wave Function Collapse.
///
/// State (cell grid + seeded RNG) lives on the feedback edge, not in the node.
/// `execute` clones the previous [`WfcState`] and collapses one cell via the
/// native [`WfcSolver::step`]. On completion the step is idempotent; a
/// contradiction is reported as [`GraphError::ExecutionError`].
///
/// # Ports
/// - Input `0` `"state"`: `Custom(WfcSolver<TileSet>)` — previous-tick state.
/// - Output `0` `"state"`: `Custom(WfcSolver<TileSet>)` — advanced state.
#[derive(Debug, Clone, Copy, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct WfcStep;

impl DynNode for WfcStep {
    fn type_name(&self) -> &'static str {
        "procgen::feedback::WfcStep"
    }

    fn inputs(&self) -> Vec<PortDescriptor> {
        vec![PortDescriptor::new("state", wfc_state_type())]
    }

    fn outputs(&self) -> Vec<PortDescriptor> {
        vec![PortDescriptor::new("state", wfc_state_type())]
    }

    fn execute(&self, inputs: &[Value], _ctx: &EvalContext) -> Result<Vec<Value>, GraphError> {
        let prev = inputs[0].downcast_ref::<WfcState>().ok_or_else(|| {
            GraphError::ExecutionError(
                "procgen::feedback::WfcStep expects a WfcSolver<TileSet> state input".to_string(),
            )
        })?;
        let mut next = prev.clone();
        // Advance one collapse; the bool (more-work-remaining) is a progress
        // signal — at completion the state is unchanged, so we always emit next.
        next.step().map_err(wfc_error_to_graph_error)?;
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

    /// A small checkerboard-ish tileset: two tiles that must alternate.
    fn init_node() -> WfcInit {
        // Two tiles; each tile allows the *other* in all four directions, forcing
        // a deterministic checkerboard given a seed.
        let mut tileset = TilesetDesc::new(2);
        for &d in &Direction::all() {
            tileset = tileset.with_rule(0, d, 1);
            tileset = tileset.with_rule(1, d, 0);
        }
        WfcInit::new(6, 6, tileset, 0xC0FFEE)
    }

    fn result_grid(solver: &WfcState) -> Vec<Option<usize>> {
        let mut out = Vec::new();
        for y in 0..solver.height() {
            for x in 0..solver.width() {
                out.push(solver.get_tile(x, y).map(|t| t.index()));
            }
        }
        out
    }

    #[test]
    fn run_to_tick_matches_mut_run_loop() {
        // Reference: native run(seed) to completion.
        let mut reference = init_node().build();
        reference.run(0xC0FFEE).unwrap();
        assert!(reference.is_complete());
        let ref_grid = result_grid(&reference);

        // Feedback driver: step until complete (width*height collapses suffice),
        // then a few extra to confirm idempotence.
        let mut graph = Graph::new();
        let init = graph.add_node(init_node());
        let step = graph.add_node(WfcStep);
        graph.connect(init, 0, step, 0).unwrap();
        graph.connect_recurrence(step, 0, step, 0).unwrap();

        let total = 6 * 6 + 4;
        let mut state = FeedbackState::new();
        let r = graph
            .run_to_tick(total as u64, &mut state, |_t| EvalContext::new())
            .unwrap();
        let solver = r.get(step, 0).unwrap().downcast_ref::<WfcState>().unwrap();
        assert!(solver.is_complete());
        assert_eq!(result_grid(solver), ref_grid);
    }

    #[test]
    fn deterministic() {
        let run = || {
            let mut graph = Graph::new();
            let init = graph.add_node(init_node());
            let step = graph.add_node(WfcStep);
            graph.connect(init, 0, step, 0).unwrap();
            graph.connect_recurrence(step, 0, step, 0).unwrap();
            let mut state = FeedbackState::new();
            let r = graph
                .run_to_tick(40, &mut state, |_t| EvalContext::new())
                .unwrap();
            result_grid(r.get(step, 0).unwrap().downcast_ref::<WfcState>().unwrap())
        };
        assert_eq!(run(), run());
    }

    #[test]
    fn step_matches_native_step_loop_intermediate() {
        // Bit-for-bit match at an *intermediate* tick (not just completion),
        // proving the carried RNG continues the same stream.
        let steps = 10u64;

        let mut reference = init_node().build();
        reference.set_seed(0xC0FFEE);
        for _ in 0..steps {
            reference.step().unwrap();
        }
        let ref_grid = result_grid(&reference);

        let mut graph = Graph::new();
        let init = graph.add_node(init_node());
        let step = graph.add_node(WfcStep);
        graph.connect(init, 0, step, 0).unwrap();
        graph.connect_recurrence(step, 0, step, 0).unwrap();
        let mut state = FeedbackState::new();
        let mut last = None;
        for t in 0..steps {
            let r = graph.tick(t, &mut state, &EvalContext::new()).unwrap();
            last = Some(
                r.get(step, 0)
                    .unwrap()
                    .downcast_ref::<WfcState>()
                    .unwrap()
                    .clone(),
            );
        }
        assert_eq!(result_grid(&last.unwrap()), ref_grid);
    }
}
