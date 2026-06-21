//! Recurrent-graph integration for the space-colonization algorithm.
//!
//! Ports the stateful [`SpaceColonization`] grower onto the feedback-edge
//! mechanism in `unshape-core` (`docs/design/recurrent-graphs.md`). The entire
//! mutable state (attraction points + active set + grown nodes/edges + config) is
//! carried on a feedback wire as an opaque [`Value`], so the step node
//! ([`GrowStep`]) is a pure `&self` [`DynNode`]: it clones the previous state,
//! advances one growth iteration via the existing native
//! [`SpaceColonization::step`], and returns the new state. No state lives in the
//! node.
//!
//! # Progress / fixpoint
//!
//! The native `step(&mut self) -> bool` returns `false` once growth is complete
//! (no point still influences a node). At that fixpoint the state stops changing
//! (the step is idempotent), so the bool is *not* surfaced as a port: a converged
//! graph simply re-emits the same state on every later tick. Resimulating to any
//! tick beyond convergence yields the converged tree.
//!
//! # Determinism
//!
//! Attraction-point generation is fully seeded (`add_attraction_points_*` take a
//! `u64` seed), so the same [`GrowInit`] config always seeds the same points; no
//! RNG runs inside `step`. The native `step` sorts its `HashSet`/`HashMap`
//! iteration so growth is index-stable and bit-reproducible across runs and
//! drivers — the reproduction test compares grown node positions directly,
//! bit-for-bit.

use std::any::Any;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use unshape_core::{
    DataLocation, DynNode, EvalContext, GraphError, GraphValue, PortDescriptor, Value, ValueType,
};

use glam::Vec3;

use crate::{SpaceColonization, SpaceColonizationParams};

/// The opaque value type name for [`SpaceColonization`] state on a wire.
pub const SPACE_COLONIZATION_STATE_NAME: &str = "SpaceColonization";

impl GraphValue for SpaceColonization {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn type_name(&self) -> &'static str {
        SPACE_COLONIZATION_STATE_NAME
    }

    fn location(&self) -> DataLocation {
        DataLocation::Cpu
    }
}

/// Returns the [`ValueType`] used for [`SpaceColonization`] state on a wire.
pub fn space_colonization_state_type() -> ValueType {
    ValueType::of::<SpaceColonization>(SPACE_COLONIZATION_STATE_NAME)
}

/// Pure-data attraction-point source for a [`GrowInit`].
///
/// Each variant maps onto one of the crate's seeded `add_attraction_points_*`
/// helpers, so the node carries the volume as data. All are deterministic via
/// their `u64` seed.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum AttractionSource {
    /// Points in a spherical volume.
    Sphere {
        /// Center.
        center: Vec3,
        /// Radius.
        radius: f32,
        /// Number of points.
        count: usize,
        /// RNG seed (deterministic).
        seed: u64,
    },
    /// Points in an axis-aligned box volume.
    Box {
        /// Minimum corner.
        min: Vec3,
        /// Maximum corner.
        max: Vec3,
        /// Number of points.
        count: usize,
        /// RNG seed (deterministic).
        seed: u64,
    },
    /// Points in a cylinder volume.
    Cylinder {
        /// Base center.
        base: Vec3,
        /// Cylinder axis.
        axis: Vec3,
        /// Height along the axis.
        height: f32,
        /// Radius.
        radius: f32,
        /// Number of points.
        count: usize,
        /// RNG seed (deterministic).
        seed: u64,
    },
}

impl AttractionSource {
    fn add_to(&self, sc: &mut SpaceColonization) {
        match *self {
            AttractionSource::Sphere {
                center,
                radius,
                count,
                seed,
            } => sc.add_attraction_points_sphere(center, radius, count, seed),
            AttractionSource::Box {
                min,
                max,
                count,
                seed,
            } => sc.add_attraction_points_box(min, max, count, seed),
            AttractionSource::Cylinder {
                base,
                axis,
                height,
                radius,
                count,
                seed,
            } => sc.add_attraction_points_cylinder(base, axis, height, radius, count, seed),
        }
    }
}

/// Pure in-graph **source** node producing the initial [`SpaceColonization`].
///
/// Takes no inputs; outputs a grower seeded from its [`SpaceColonizationParams`],
/// a set of [`AttractionSource`]s, and a set of root positions. Deterministic —
/// the same config yields the same seeded state.
///
/// # Ports
/// - Output `0` `"state"`: `Custom(SpaceColonization)` — initial state.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct GrowInit {
    /// Algorithm parameters.
    pub params: SpaceColonizationParams,
    /// Attraction-point volumes to seed.
    pub sources: Vec<AttractionSource>,
    /// Root positions to start growth from.
    pub roots: Vec<Vec3>,
}

impl GrowInit {
    /// Creates an init node with the given params and no sources/roots.
    pub fn new(params: SpaceColonizationParams) -> Self {
        Self {
            params,
            sources: Vec::new(),
            roots: Vec::new(),
        }
    }

    /// Adds an attraction-point source.
    pub fn with_source(mut self, source: AttractionSource) -> Self {
        self.sources.push(source);
        self
    }

    /// Adds a root position.
    pub fn with_root(mut self, position: Vec3) -> Self {
        self.roots.push(position);
        self
    }

    /// Builds the initial [`SpaceColonization`] from this config (pure).
    pub fn build(&self) -> SpaceColonization {
        let mut sc = SpaceColonization::new(self.params.clone());
        for source in &self.sources {
            source.add_to(&mut sc);
        }
        for &root in &self.roots {
            sc.add_root(root);
        }
        sc
    }
}

impl DynNode for GrowInit {
    fn type_name(&self) -> &'static str {
        "space_colonization::feedback::GrowInit"
    }

    fn inputs(&self) -> Vec<PortDescriptor> {
        vec![]
    }

    fn outputs(&self) -> Vec<PortDescriptor> {
        vec![PortDescriptor::new(
            "state",
            space_colonization_state_type(),
        )]
    }

    fn execute(&self, _inputs: &[Value], _ctx: &EvalContext) -> Result<Vec<Value>, GraphError> {
        Ok(vec![Value::opaque(self.build())])
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Pure per-tick step node for space colonization.
///
/// State lives on the feedback edge, not in the node. `execute` clones the
/// previous [`SpaceColonization`] and advances it one growth iteration via the
/// native [`SpaceColonization::step`]. At the growth fixpoint the step is
/// idempotent and re-emits the same state.
///
/// # Ports
/// - Input `0` `"state"`: `Custom(SpaceColonization)` — previous-tick state.
/// - Output `0` `"state"`: `Custom(SpaceColonization)` — advanced state.
#[derive(Debug, Clone, Copy, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct GrowStep;

impl DynNode for GrowStep {
    fn type_name(&self) -> &'static str {
        "space_colonization::feedback::GrowStep"
    }

    fn inputs(&self) -> Vec<PortDescriptor> {
        vec![PortDescriptor::new(
            "state",
            space_colonization_state_type(),
        )]
    }

    fn outputs(&self) -> Vec<PortDescriptor> {
        vec![PortDescriptor::new(
            "state",
            space_colonization_state_type(),
        )]
    }

    fn execute(&self, inputs: &[Value], _ctx: &EvalContext) -> Result<Vec<Value>, GraphError> {
        let prev = inputs[0]
            .downcast_ref::<SpaceColonization>()
            .ok_or_else(|| {
                GraphError::ExecutionError(
                    "space_colonization::feedback::GrowStep expects a SpaceColonization state input"
                        .to_string(),
                )
            })?;
        let mut next = prev.clone();
        // The bool (more-growth-remaining) is a progress signal; at the fixpoint
        // the state is unchanged, so we always emit the (possibly identical) next.
        let _grew = next.step();
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

    fn init_node() -> GrowInit {
        GrowInit::new(SpaceColonizationParams {
            attraction_distance: 5.0,
            kill_distance: 1.0,
            segment_length: 0.5,
            ..Default::default()
        })
        .with_source(AttractionSource::Sphere {
            center: Vec3::new(0.0, 5.0, 0.0),
            radius: 3.0,
            count: 80,
            seed: 12345,
        })
        .with_root(Vec3::ZERO)
    }

    /// Index-stable sequence of node positions (bit-exact). `step` is fully
    /// deterministic, so this is reproducible run-to-run and across drivers.
    fn position_set(sc: &SpaceColonization) -> Vec<[u32; 3]> {
        sc.nodes()
            .iter()
            .map(|n| {
                [
                    n.position.x.to_bits(),
                    n.position.y.to_bits(),
                    n.position.z.to_bits(),
                ]
            })
            .collect()
    }

    #[test]
    fn evolves_like_mut_step_loop() {
        let n = 12u64;

        let mut reference = init_node().build();
        for _ in 0..n {
            reference.step();
        }

        let mut graph = Graph::new();
        let init = graph.add_node(init_node());
        let step = graph.add_node(GrowStep);
        graph.connect(init, 0, step, 0).unwrap();
        graph.connect_recurrence(step, 0, step, 0).unwrap();

        let mut state = FeedbackState::new();
        let mut last = None;
        for t in 0..n {
            let r = graph.tick(t, &mut state, &EvalContext::new()).unwrap();
            last = Some(
                r.get(step, 0)
                    .unwrap()
                    .downcast_ref::<SpaceColonization>()
                    .unwrap()
                    .clone(),
            );
        }
        let evolved = last.unwrap();

        // Same grown geometry (order-independent set of positions).
        assert_eq!(position_set(&evolved), position_set(&reference));
    }

    #[test]
    fn run_to_tick_matches_manual_stepping() {
        let target = 20u64;

        let manual = {
            let mut sc = init_node().build();
            for _ in 0..=target {
                sc.step();
            }
            position_set(&sc)
        };

        let mut graph = Graph::new();
        let init = graph.add_node(init_node());
        let step = graph.add_node(GrowStep);
        graph.connect(init, 0, step, 0).unwrap();
        graph.connect_recurrence(step, 0, step, 0).unwrap();
        let mut state = FeedbackState::new();
        let r = graph
            .run_to_tick(target, &mut state, |_t| EvalContext::new())
            .unwrap();
        let resimulated = position_set(
            r.get(step, 0)
                .unwrap()
                .downcast_ref::<SpaceColonization>()
                .unwrap(),
        );

        assert_eq!(resimulated, manual);
    }

    #[test]
    fn idempotent_at_fixpoint() {
        // Growing well past convergence yields the same geometry as growing to
        // convergence: the step is idempotent once growth is complete.
        let mut sc = init_node().build();
        // Run to convergence.
        while sc.step() {}
        let converged = position_set(&sc);

        // Extra steps change nothing.
        for _ in 0..5 {
            sc.step();
        }
        assert_eq!(position_set(&sc), converged);
    }
}
