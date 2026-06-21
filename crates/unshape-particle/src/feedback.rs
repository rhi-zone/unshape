//! Recurrent-graph integration for the particle simulation.
//!
//! Ports the stateful [`ParticleSystem`] onto the feedback-edge mechanism in
//! `unshape-core` (`docs/design/recurrent-graphs.md`). The entire mutable state
//! — the particle buffer *and* the seeded [`ParticleRng`] — is bundled into
//! [`ParticleSystem`] and carried on a feedback wire as an opaque [`Value`]. The
//! step node ([`Step`]) is a pure `&self` [`DynNode`]: it clones the previous
//! state, advances one tick (apply forces → integrate → age → cull dead), and
//! returns the new state.
//!
//! # Forces as data
//!
//! The runtime [`Force`](crate::Force) trait uses `&dyn Force` trait objects,
//! which are neither serializable nor faithfully storable on a node. To keep the
//! node pure-data, [`Step`] holds a `Vec<ForceKind>` — a serializable enum over
//! the built-in concrete forces — instead of boxed trait objects. Each variant
//! delegates to the same `Force::apply` as the imperative API, so behaviour is
//! identical.
//!
//! # Tick-0 state
//!
//! The initial [`ParticleSystem`] is produced by an in-graph [`ParticleInit`]
//! source node (a pure `&self` `DynNode` taking no inputs) wired into the
//! [`Step`]'s state port with a *direct* edge, alongside the feedback self-loop.
//! On tick 0 the `Init` node emits the initial particles (with a seeded RNG); on
//! later ticks the carried feedback value is used. This makes the sim rewindable
//! via [`Graph::run_to_tick`](unshape_core::Graph::run_to_tick) with no manual
//! seed.

use std::any::Any;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use unshape_core::{
    DataLocation, DynNode, EvalContext, GraphError, GraphValue, PortDescriptor, Value, ValueType,
};

use crate::{
    Attractor, Drag, EulerIntegrator, Force, Gravity, Integrator, ParticleSystem, PointEmitter,
    SemiImplicitEulerIntegrator, Vortex, Wind,
};

/// The opaque value type name for [`ParticleSystem`] state on a wire.
pub const PARTICLE_STATE_NAME: &str = "ParticleSystem";

impl GraphValue for ParticleSystem {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn type_name(&self) -> &'static str {
        PARTICLE_STATE_NAME
    }

    fn location(&self) -> DataLocation {
        DataLocation::Cpu
    }
}

/// Returns the [`ValueType`] used for [`ParticleSystem`] state on a wire.
pub fn particle_state_type() -> ValueType {
    ValueType::of::<ParticleSystem>(PARTICLE_STATE_NAME)
}

/// A serializable, pure-data force descriptor.
///
/// Mirrors the built-in [`Force`](crate::Force) implementations so a [`Step`]
/// node can hold its forces as data (not boxed trait objects). Each variant
/// delegates to the corresponding concrete force's `Force::apply`.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum ForceKind {
    /// Constant directional acceleration (see [`Gravity`]).
    Gravity(Gravity),
    /// Push toward a target velocity (see [`Wind`]).
    Wind(Wind),
    /// Velocity damping (see [`Drag`]).
    Drag(Drag),
    /// Inverse-square attraction/repulsion (see [`Attractor`]).
    Attractor(Attractor),
    /// Rotational force around an axis (see [`Vortex`]).
    Vortex(Vortex),
}

impl ForceKind {
    fn apply(&self, particle: &mut crate::Particle, dt: f32) {
        // Use UFCS: each force type also has an inherent `apply()` (Op sugar)
        // that would shadow the `Force` trait method.
        match self {
            ForceKind::Gravity(f) => Force::apply(f, particle, dt),
            ForceKind::Wind(f) => Force::apply(f, particle, dt),
            ForceKind::Drag(f) => Force::apply(f, particle, dt),
            ForceKind::Attractor(f) => Force::apply(f, particle, dt),
            ForceKind::Vortex(f) => Force::apply(f, particle, dt),
        }
    }
}

/// A serializable integrator selector for [`Step`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum IntegratorKind {
    /// Explicit Euler (see [`EulerIntegrator`]).
    #[default]
    Euler,
    /// Semi-implicit Euler (see [`SemiImplicitEulerIntegrator`]).
    SemiImplicit,
}

impl IntegratorKind {
    fn step(&self, particle: &mut crate::Particle, dt: f32) {
        match self {
            IntegratorKind::Euler => EulerIntegrator.step(particle, dt),
            IntegratorKind::SemiImplicit => SemiImplicitEulerIntegrator.step(particle, dt),
        }
    }
}

/// Pure in-graph **source** node producing the initial [`ParticleSystem`].
///
/// Takes no inputs; outputs a particle system seeded with a deterministic RNG
/// (`seed`) into which `count` particles have been emitted from a
/// [`PointEmitter`]. Deterministic — the same config yields the same particles
/// (including the seeded RNG carried in the output, so subsequent emission would
/// also reproduce).
///
/// Wire this into a [`Step`]'s state port with a *direct* edge (the tick-0 seed),
/// alongside the [`Step`]'s feedback self-loop.
///
/// # Ports
/// - Output `0` `"state"`: `Custom(ParticleSystem)` — initial system.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ParticleInit {
    /// Maximum particle capacity of the system.
    pub capacity: usize,
    /// RNG seed for reproducible emission.
    pub seed: u64,
    /// Emitter used to produce the initial particles.
    pub emitter: PointEmitter,
    /// Number of particles to emit initially.
    pub count: usize,
}

impl ParticleInit {
    /// Creates an init node from capacity, RNG seed, emitter, and emit count.
    pub fn new(capacity: usize, seed: u64, emitter: PointEmitter, count: usize) -> Self {
        Self {
            capacity,
            seed,
            emitter,
            count,
        }
    }

    /// Builds the initial [`ParticleSystem`] from this config (pure).
    pub fn build(&self) -> ParticleSystem {
        let mut sys = ParticleSystem::with_seed(self.capacity, self.seed);
        sys.emit(&self.emitter, self.count);
        sys
    }
}

impl DynNode for ParticleInit {
    fn type_name(&self) -> &'static str {
        "particle::feedback::ParticleInit"
    }

    fn inputs(&self) -> Vec<PortDescriptor> {
        vec![]
    }

    fn outputs(&self) -> Vec<PortDescriptor> {
        vec![PortDescriptor::new("state", particle_state_type())]
    }

    fn execute(&self, _inputs: &[Value], _ctx: &EvalContext) -> Result<Vec<Value>, GraphError> {
        Ok(vec![Value::opaque(self.build())])
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Pure per-tick step node for the particle simulation.
///
/// State (particles + seeded RNG) lives on the feedback edge, not in the node.
/// `execute` clones the previous [`ParticleSystem`], applies each force, runs the
/// integrator, ages particles by `dt`, culls dead ones, and returns the new
/// state — exactly mirroring
/// [`ParticleSystem::update_with_forces_and_integrator`].
///
/// # Ports
/// - Input `0` `"state"`: `Custom(ParticleSystem)` — previous-tick state.
/// - Output `0` `"state"`: `Custom(ParticleSystem)` — advanced state.
#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Step {
    /// Time step (seconds) per tick.
    pub dt: f32,
    /// Forces applied each tick, in order, before integration.
    pub forces: Vec<ForceKind>,
    /// Integrator used to advance positions.
    pub integrator: IntegratorKind,
}

impl Step {
    /// Creates a step node with the given time step and no forces.
    pub fn new(dt: f32) -> Self {
        Self {
            dt,
            forces: Vec::new(),
            integrator: IntegratorKind::default(),
        }
    }

    /// Adds a force to be applied each tick.
    pub fn with_force(mut self, force: ForceKind) -> Self {
        self.forces.push(force);
        self
    }

    /// Sets the integrator.
    pub fn with_integrator(mut self, integrator: IntegratorKind) -> Self {
        self.integrator = integrator;
        self
    }

    /// Advances a cloned system by one tick (pure).
    fn advance(&self, prev: &ParticleSystem) -> ParticleSystem {
        let mut next = prev.clone();
        for particle in next.particles_mut() {
            for force in &self.forces {
                force.apply(particle, self.dt);
            }
            self.integrator.step(particle, self.dt);
            particle.age += self.dt;
        }
        next.retain_alive();
        next
    }
}

impl DynNode for Step {
    fn type_name(&self) -> &'static str {
        "particle::feedback::Step"
    }

    fn inputs(&self) -> Vec<PortDescriptor> {
        vec![PortDescriptor::new("state", particle_state_type())]
    }

    fn outputs(&self) -> Vec<PortDescriptor> {
        vec![PortDescriptor::new("state", particle_state_type())]
    }

    fn execute(&self, inputs: &[Value], _ctx: &EvalContext) -> Result<Vec<Value>, GraphError> {
        let prev = inputs[0].downcast_ref::<ParticleSystem>().ok_or_else(|| {
            GraphError::ExecutionError(
                "particle::feedback::Step expects a ParticleSystem state input".to_string(),
            )
        })?;
        Ok(vec![Value::opaque(self.advance(prev))])
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PointEmitter;
    use glam::Vec3;
    use unshape_core::{FeedbackState, Graph};

    fn init_node() -> ParticleInit {
        // Seed 12345 matches `ParticleSystem::new`'s default RNG, so the Init
        // node reproduces `seeded_system()` exactly.
        let emitter = PointEmitter {
            lifetime_min: 100.0,
            lifetime_max: 100.0,
            ..Default::default()
        };
        ParticleInit::new(64, 12345, emitter, 20)
    }

    fn seeded_system() -> ParticleSystem {
        init_node().build()
    }

    fn step_node() -> Step {
        Step::new(0.1)
            .with_force(ForceKind::Gravity(Gravity::default()))
            .with_force(ForceKind::Drag(Drag { coefficient: 0.2 }))
    }

    /// A built feedback graph: `Init --direct--> Step.state`, `Step --feedback--> Step.state`.
    struct Built {
        graph: Graph,
        step: u32,
    }

    fn build() -> Built {
        let mut graph = Graph::new();
        let init = graph.add_node(init_node());
        let step = graph.add_node(step_node());
        graph.connect(init, 0, step, 0).unwrap();
        graph.connect_recurrence(step, 0, step, 0).unwrap();
        Built { graph, step }
    }

    fn total_momentum(sys: &ParticleSystem) -> Vec3 {
        sys.particles().iter().map(|p| p.velocity).sum()
    }

    #[test]
    fn evolves_like_mut_update_loop() {
        // (a) feedback stepping N times matches the imperative update loop.
        let n = 10u64;

        // Reference: imperative loop using the existing &mut API.
        let mut reference = seeded_system();
        let g = Gravity::default();
        let d = Drag { coefficient: 0.2 };
        for _ in 0..n {
            let force_refs: Vec<&dyn Force> = vec![&g, &d];
            reference.update_with_forces_and_integrator(0.1, &force_refs, &EulerIntegrator);
        }

        // Feedback driver.
        let Built { mut graph, step } = build();
        let mut state = FeedbackState::new();
        let mut last = None;
        for t in 0..n {
            let r = graph.tick(t, &mut state, &EvalContext::new()).unwrap();
            last = Some(
                r.get(step, 0)
                    .unwrap()
                    .downcast_ref::<ParticleSystem>()
                    .unwrap()
                    .clone(),
            );
        }
        let evolved = last.unwrap();

        assert_eq!(evolved.count(), reference.count());
        for (a, b) in evolved.particles().iter().zip(reference.particles()) {
            assert!((a.position - b.position).length() < 1e-5);
            assert!((a.velocity - b.velocity).length() < 1e-5);
            assert!((a.age - b.age).abs() < 1e-5);
        }
    }

    #[test]
    fn node_is_pure_fresh_state_restarts() {
        // (b) the node holds no mutable state: a fresh seed restarts identically
        // (the Init source re-seeds tick 0).
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
            .downcast_ref::<ParticleSystem>()
            .unwrap()
            .clone();

        // Equals a single advance from the seed.
        let expected = step_node().advance(&seeded_system());
        assert_eq!(one.count(), expected.count());
        for (a, b) in one.particles().iter().zip(expected.particles()) {
            assert!((a.position - b.position).length() < 1e-6);
            assert!((a.velocity - b.velocity).length() < 1e-6);
        }
    }

    #[test]
    fn deterministic() {
        // (c) same seed + inputs + N -> identical output.
        let run = || {
            let Built { mut graph, step } = build();
            let mut state = FeedbackState::new();
            let mut last = Vec3::ZERO;
            for t in 0..12 {
                let r = graph.tick(t, &mut state, &EvalContext::new()).unwrap();
                last = total_momentum(
                    r.get(step, 0)
                        .unwrap()
                        .downcast_ref::<ParticleSystem>()
                        .unwrap(),
                );
            }
            last
        };
        let a = run();
        let b = run();
        assert_eq!(a, b);
    }

    #[test]
    fn run_to_tick_matches_manual_stepping() {
        // (d) run_to_tick now SUCCEEDS: the in-graph ParticleInit source re-seeds
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
                        .downcast_ref::<ParticleSystem>()
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
            .downcast_ref::<ParticleSystem>()
            .unwrap();

        assert_eq!(resimulated.count(), manual.count());
        for (a, b) in resimulated.particles().iter().zip(manual.particles()) {
            assert!((a.position - b.position).length() < 1e-6);
            assert!((a.velocity - b.velocity).length() < 1e-6);
            assert!((a.age - b.age).abs() < 1e-6);
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
            total_momentum(
                r.get(step, 0)
                    .unwrap()
                    .downcast_ref::<ParticleSystem>()
                    .unwrap(),
            )
        };
        assert_eq!(seek_to(8), seek_to(8));
    }
}
