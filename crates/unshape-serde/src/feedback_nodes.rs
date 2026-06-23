//! Registration of recurrent (feedback) simulation nodes for serialization.
//!
//! The grid/particle/audio simulations are ported onto the feedback-edge
//! mechanism in `unshape-core` as pure `&self` `DynNode`s living in their own
//! crates (each behind a `feedback` feature). Their `Step` and `Init` nodes are
//! plain serde structs; this module implements [`SerializableNode`] for each
//! (the trait is local here, so the orphan rule permits implementing it for the
//! foreign node types) and exposes one `register_*_feedback_nodes` function per
//! crate that wires the stable `type_name` → factory mapping into a
//! [`NodeRegistry`].
//!
//! Only the node **config** is serialized — runtime [`FeedbackState`] is never
//! part of the graph; it is recomputed by re-running from the `Init` seed.
//!
//! [`SerializableNode`]: crate::SerializableNode
//! [`FeedbackState`]: unshape_core::FeedbackState

use crate::registry::NodeRegistry;

#[cfg(any(
    feature = "rd-feedback",
    feature = "particle-feedback",
    feature = "fluid-feedback",
    feature = "audio-feedback",
    feature = "automata-feedback",
    feature = "spring-feedback",
    feature = "physics-feedback",
    feature = "space-colonization-feedback",
    feature = "procgen-feedback"
))]
mod latch_node {
    use crate::error::SerdeError;
    use crate::registry::{NodeRegistry, SerializableNode};
    use serde_json::Value as JsonValue;
    use unshape_core::{Latch, Rate, ValueType};

    /// A `(state-type-name, constructor)` pair for resolving a latch's `ty`.
    #[allow(dead_code)] // Unused when only single-state-type features are enabled.
    type StateTypeCtor = (&'static str, fn() -> ValueType);

    /// Serializes a [`Latch`] as `{ "ty": <state-type-name>, "rate": <rate> }`.
    ///
    /// `ty` is stored by its display **name** because the opaque (`Custom`)
    /// state types carried by the sims cannot serialize their `TypeId`. The name
    /// is resolved back to a [`ValueType`] at load time against the set of state
    /// types registered for the enabled feedback features (the runtime stored
    /// value is never serialized — it is reproduced by replay from `init`).
    impl SerializableNode for Latch {
        fn params(&self) -> JsonValue {
            serde_json::json!({
                "ty": self.ty.to_string(),
                "rate": self.rate,
            })
        }
    }

    /// Resolves a Latch state-type **name** back to a [`ValueType`].
    ///
    /// Covers the opaque state types of every enabled feedback feature, plus the
    /// primitive value types (so a primitive-typed latch also round-trips).
    fn resolve_state_type(name: &str) -> Option<ValueType> {
        // Primitive types first.
        let primitive = match name {
            "f32" => Some(ValueType::F32),
            "f64" => Some(ValueType::F64),
            "i32" => Some(ValueType::I32),
            "bool" => Some(ValueType::Bool),
            "Vec2" => Some(ValueType::Vec2),
            "Vec3" => Some(ValueType::Vec3),
            "Vec4" => Some(ValueType::Vec4),
            _ => None,
        };
        if primitive.is_some() {
            return primitive;
        }

        // Opaque sim state types (one per enabled feedback feature).
        #[cfg(feature = "rd-feedback")]
        if name == unshape_rd::feedback::RD_STATE_NAME {
            return Some(unshape_rd::feedback::rd_state_type());
        }
        #[cfg(feature = "particle-feedback")]
        if name == unshape_particle::feedback::PARTICLE_STATE_NAME {
            return Some(unshape_particle::feedback::particle_state_type());
        }
        #[cfg(feature = "fluid-feedback")]
        {
            use unshape_fluid::feedback as f;
            let m: &[StateTypeCtor] = &[
                (f::FLUID_GRID_2D_NAME, f::fluid_grid_2d_type),
                (f::FLUID_GRID_3D_NAME, f::fluid_grid_3d_type),
                (f::SMOKE_GRID_2D_NAME, f::smoke_grid_2d_type),
                (f::SMOKE_GRID_3D_NAME, f::smoke_grid_3d_type),
                (f::SPH_2D_NAME, f::sph_2d_type),
                (f::SPH_3D_NAME, f::sph_3d_type),
            ];
            for (n, ctor) in m {
                if *n == name {
                    return Some(ctor());
                }
            }
        }
        #[cfg(feature = "audio-feedback")]
        {
            use unshape_audio::vocoder::feedback as v;
            if name == v::VOCODER_STATE_NAME {
                return Some(v::vocoder_state_type());
            }
            if name == v::AUDIO_BLOCK_NAME {
                return Some(v::audio_block_type());
            }
        }
        #[cfg(feature = "automata-feedback")]
        {
            use unshape_automata::feedback as a;
            let m: &[StateTypeCtor] = &[
                (a::ELEMENTARY_STATE_NAME, a::elementary_state_type),
                (a::LIFE_STATE_NAME, a::life_state_type),
                (a::SMOOTH_LIFE_STATE_NAME, a::smooth_life_state_type),
            ];
            for (n, ctor) in m {
                if *n == name {
                    return Some(ctor());
                }
            }
        }
        #[cfg(feature = "spring-feedback")]
        if name == unshape_spring::feedback::SPRING_STATE_NAME {
            return Some(unshape_spring::feedback::spring_state_type());
        }
        #[cfg(feature = "physics-feedback")]
        if name == unshape_physics::feedback::PHYSICS_STATE_NAME {
            return Some(unshape_physics::feedback::physics_state_type());
        }
        #[cfg(feature = "space-colonization-feedback")]
        if name == unshape_space_colonization::feedback::SPACE_COLONIZATION_STATE_NAME {
            return Some(unshape_space_colonization::feedback::space_colonization_state_type());
        }
        #[cfg(feature = "procgen-feedback")]
        if name == unshape_procgen::feedback::WFC_STATE_NAME {
            return Some(unshape_procgen::feedback::wfc_state_type());
        }

        None
    }

    /// Registers the `core::Latch` node into `registry`.
    ///
    /// The factory resolves the latch's `ty` name against the state types of all
    /// enabled feedback features (see [`resolve_state_type`]).
    pub fn register(registry: &mut NodeRegistry) {
        registry.register_factory("core::Latch", |params| {
            let name = params.get("ty").and_then(|v| v.as_str()).ok_or_else(|| {
                SerdeError::InvalidWireFormat("core::Latch params missing `ty`".to_string())
            })?;
            let ty = resolve_state_type(name).ok_or_else(|| {
                SerdeError::UnknownNodeType(format!("core::Latch with unknown state type {name:?}"))
            })?;
            let rate: Rate = params
                .get("rate")
                .cloned()
                .map(serde_json::from_value)
                .transpose()?
                .unwrap_or(Rate::Tick);
            Ok(Box::new(Latch { ty, rate }) as unshape_core::BoxedNode)
        });
    }
}

#[cfg(any(
    feature = "rd-feedback",
    feature = "particle-feedback",
    feature = "fluid-feedback",
    feature = "audio-feedback",
    feature = "automata-feedback",
    feature = "spring-feedback",
    feature = "physics-feedback",
    feature = "space-colonization-feedback",
    feature = "procgen-feedback"
))]
pub use latch_node::register as register_latch_node;

#[cfg(any(
    feature = "rd-feedback",
    feature = "particle-feedback",
    feature = "fluid-feedback",
    feature = "audio-feedback",
    feature = "automata-feedback",
    feature = "spring-feedback",
    feature = "physics-feedback",
    feature = "space-colonization-feedback",
    feature = "procgen-feedback"
))]
use crate::registry::SerializableNode;
#[cfg(any(
    feature = "rd-feedback",
    feature = "particle-feedback",
    feature = "fluid-feedback",
    feature = "audio-feedback",
    feature = "automata-feedback",
    feature = "spring-feedback",
    feature = "physics-feedback",
    feature = "space-colonization-feedback",
    feature = "procgen-feedback"
))]
use serde_json::Value as JsonValue;

/// Default `params()` for a serde node: serialize the whole node to JSON.
///
/// Unit step nodes serialize to `null`, which round-trips back to the unit
/// value; config init nodes serialize to their parameter object.
#[cfg(any(
    feature = "rd-feedback",
    feature = "particle-feedback",
    feature = "fluid-feedback",
    feature = "audio-feedback",
    feature = "automata-feedback",
    feature = "spring-feedback",
    feature = "physics-feedback",
    feature = "space-colonization-feedback",
    feature = "procgen-feedback"
))]
macro_rules! impl_serializable_node {
    ($($ty:ty),+ $(,)?) => {
        $(
            impl SerializableNode for $ty {
                fn params(&self) -> JsonValue {
                    serde_json::to_value(self)
                        .unwrap_or_else(|e| serde_json::json!({ "__error": e.to_string() }))
                }
            }
        )+
    };
}

// ===========================================================================
// Reaction-diffusion (unshape-rd)
// ===========================================================================

#[cfg(feature = "rd-feedback")]
mod rd {
    use super::*;
    use unshape_rd::feedback::{GrayScottInit, Step};

    impl_serializable_node!(GrayScottInit, Step);

    /// Registers the reaction-diffusion feedback nodes.
    ///
    /// Type names: `"rd::feedback::GrayScottInit"`, `"rd::feedback::Step"`.
    pub fn register(registry: &mut NodeRegistry) {
        super::register_latch_node(registry);
        registry.register_with_name::<GrayScottInit>("rd::feedback::GrayScottInit");
        registry.register_with_name::<Step>("rd::feedback::Step");
    }
}

#[cfg(feature = "rd-feedback")]
pub use rd::register as register_rd_feedback_nodes;

// ===========================================================================
// Particle systems (unshape-particle)
// ===========================================================================

#[cfg(feature = "particle-feedback")]
mod particle {
    use super::*;
    use unshape_particle::feedback::{ParticleInit, Step};

    impl_serializable_node!(ParticleInit, Step);

    /// Registers the particle-system feedback nodes.
    ///
    /// Type names: `"particle::feedback::ParticleInit"`,
    /// `"particle::feedback::Step"`.
    pub fn register(registry: &mut NodeRegistry) {
        super::register_latch_node(registry);
        registry.register_with_name::<ParticleInit>("particle::feedback::ParticleInit");
        registry.register_with_name::<Step>("particle::feedback::Step");
    }
}

#[cfg(feature = "particle-feedback")]
pub use particle::register as register_particle_feedback_nodes;

// ===========================================================================
// Fluid simulations (unshape-fluid)
// ===========================================================================

#[cfg(feature = "fluid-feedback")]
mod fluid {
    use super::*;
    use unshape_fluid::feedback::{
        Fluid3DInit, Fluid3DStep, FluidInit, Smoke2DInit, Smoke2DStep, Smoke3DInit, Smoke3DStep,
        Sph2DInit, Sph2DStep, Sph3DInit, Sph3DStep, Step,
    };

    impl_serializable_node!(
        FluidInit,
        Step,
        Fluid3DInit,
        Fluid3DStep,
        Smoke2DInit,
        Smoke2DStep,
        Smoke3DInit,
        Smoke3DStep,
        Sph2DInit,
        Sph2DStep,
        Sph3DInit,
        Sph3DStep,
    );

    /// Registers all fluid feedback nodes (2D/3D grid, smoke, SPH).
    ///
    /// Type names are `"fluid::feedback::<NodeName>"` for each node.
    pub fn register(registry: &mut NodeRegistry) {
        super::register_latch_node(registry);
        registry.register_with_name::<FluidInit>("fluid::feedback::FluidInit");
        registry.register_with_name::<Step>("fluid::feedback::Step");
        registry.register_with_name::<Fluid3DInit>("fluid::feedback::Fluid3DInit");
        registry.register_with_name::<Fluid3DStep>("fluid::feedback::Fluid3DStep");
        registry.register_with_name::<Smoke2DInit>("fluid::feedback::Smoke2DInit");
        registry.register_with_name::<Smoke2DStep>("fluid::feedback::Smoke2DStep");
        registry.register_with_name::<Smoke3DInit>("fluid::feedback::Smoke3DInit");
        registry.register_with_name::<Smoke3DStep>("fluid::feedback::Smoke3DStep");
        registry.register_with_name::<Sph2DInit>("fluid::feedback::Sph2DInit");
        registry.register_with_name::<Sph2DStep>("fluid::feedback::Sph2DStep");
        registry.register_with_name::<Sph3DInit>("fluid::feedback::Sph3DInit");
        registry.register_with_name::<Sph3DStep>("fluid::feedback::Sph3DStep");
    }
}

#[cfg(feature = "fluid-feedback")]
pub use fluid::register as register_fluid_feedback_nodes;

// ===========================================================================
// Audio vocoder (unshape-audio)
// ===========================================================================

#[cfg(feature = "audio-feedback")]
mod audio {
    use super::*;
    use unshape_audio::vocoder::feedback::{Step, VocoderInit};

    impl_serializable_node!(VocoderInit, Step);

    /// Registers the audio vocoder feedback nodes.
    ///
    /// Type names: `"vocoder::feedback::VocoderInit"`,
    /// `"vocoder::feedback::Step"`.
    pub fn register(registry: &mut NodeRegistry) {
        super::register_latch_node(registry);
        registry.register_with_name::<VocoderInit>("vocoder::feedback::VocoderInit");
        registry.register_with_name::<Step>("vocoder::feedback::Step");
    }
}

#[cfg(feature = "audio-feedback")]
pub use audio::register as register_audio_feedback_nodes;

// ===========================================================================
// Cellular automata (unshape-automata)
// ===========================================================================

#[cfg(feature = "automata-feedback")]
mod automata {
    use super::*;
    use unshape_automata::feedback::{
        ElementaryInit, ElementaryStep, LifeInit, LifeStep, SmoothLifeInit, SmoothLifeStep,
    };

    impl_serializable_node!(
        ElementaryInit,
        ElementaryStep,
        LifeInit,
        LifeStep,
        SmoothLifeInit,
        SmoothLifeStep,
    );

    /// Registers the cellular-automata feedback nodes (1D elementary, 2D
    /// life-like, continuous SmoothLife).
    ///
    /// Type names are `"automata::feedback::<NodeName>"` for each node.
    pub fn register(registry: &mut NodeRegistry) {
        super::register_latch_node(registry);
        registry.register_with_name::<ElementaryInit>("automata::feedback::ElementaryInit");
        registry.register_with_name::<ElementaryStep>("automata::feedback::ElementaryStep");
        registry.register_with_name::<LifeInit>("automata::feedback::LifeInit");
        registry.register_with_name::<LifeStep>("automata::feedback::LifeStep");
        registry.register_with_name::<SmoothLifeInit>("automata::feedback::SmoothLifeInit");
        registry.register_with_name::<SmoothLifeStep>("automata::feedback::SmoothLifeStep");
    }
}

#[cfg(feature = "automata-feedback")]
pub use automata::register as register_automata_feedback_nodes;

// ===========================================================================
// Spring soft bodies (unshape-spring)
// ===========================================================================

#[cfg(feature = "spring-feedback")]
mod spring {
    use super::*;
    use unshape_spring::feedback::{SpringInit, SpringStep};

    impl_serializable_node!(SpringInit, SpringStep);

    /// Registers the spring soft-body feedback nodes.
    ///
    /// Type names: `"spring::feedback::SpringInit"`, `"spring::feedback::SpringStep"`.
    pub fn register(registry: &mut NodeRegistry) {
        super::register_latch_node(registry);
        registry.register_with_name::<SpringInit>("spring::feedback::SpringInit");
        registry.register_with_name::<SpringStep>("spring::feedback::SpringStep");
    }
}

#[cfg(feature = "spring-feedback")]
pub use spring::register as register_spring_feedback_nodes;

// ===========================================================================
// Rigid-body physics (unshape-physics)
// ===========================================================================

#[cfg(feature = "physics-feedback")]
mod physics {
    use super::*;
    use unshape_physics::feedback::{PhysicsInit, PhysicsStep};

    impl_serializable_node!(PhysicsInit, PhysicsStep);

    /// Registers the rigid-body physics feedback nodes.
    ///
    /// Type names: `"physics::feedback::PhysicsInit"`, `"physics::feedback::PhysicsStep"`.
    pub fn register(registry: &mut NodeRegistry) {
        super::register_latch_node(registry);
        registry.register_with_name::<PhysicsInit>("physics::feedback::PhysicsInit");
        registry.register_with_name::<PhysicsStep>("physics::feedback::PhysicsStep");
    }
}

#[cfg(feature = "physics-feedback")]
pub use physics::register as register_physics_feedback_nodes;

// ===========================================================================
// Space colonization (unshape-space-colonization)
// ===========================================================================

#[cfg(feature = "space-colonization-feedback")]
mod space_colonization {
    use super::*;
    use unshape_space_colonization::feedback::{GrowInit, GrowStep};

    impl_serializable_node!(GrowInit, GrowStep);

    /// Registers the space-colonization feedback nodes.
    ///
    /// Type names: `"space_colonization::feedback::GrowInit"`,
    /// `"space_colonization::feedback::GrowStep"`.
    pub fn register(registry: &mut NodeRegistry) {
        super::register_latch_node(registry);
        registry.register_with_name::<GrowInit>("space_colonization::feedback::GrowInit");
        registry.register_with_name::<GrowStep>("space_colonization::feedback::GrowStep");
    }
}

#[cfg(feature = "space-colonization-feedback")]
pub use space_colonization::register as register_space_colonization_feedback_nodes;

// ===========================================================================
// Procedural generation / Wave Function Collapse (unshape-procgen)
// ===========================================================================

#[cfg(feature = "procgen-feedback")]
mod procgen {
    use super::*;
    use unshape_procgen::feedback::{WfcInit, WfcStep};

    impl_serializable_node!(WfcInit, WfcStep);

    /// Registers the procgen (Wave Function Collapse) feedback nodes.
    ///
    /// Type names: `"procgen::feedback::WfcInit"`, `"procgen::feedback::WfcStep"`.
    pub fn register(registry: &mut NodeRegistry) {
        super::register_latch_node(registry);
        registry.register_with_name::<WfcInit>("procgen::feedback::WfcInit");
        registry.register_with_name::<WfcStep>("procgen::feedback::WfcStep");
    }
}

#[cfg(feature = "procgen-feedback")]
pub use procgen::register as register_procgen_feedback_nodes;

// ===========================================================================
// Umbrella
// ===========================================================================

/// Registers every available feedback Step/Init node into `registry`.
///
/// Each crate's nodes are only registered when the corresponding
/// `<crate>-feedback` feature is enabled (`feedback-nodes` enables all).
pub fn register_all_feedback_nodes(registry: &mut NodeRegistry) {
    // Silences the unused-parameter warning when no feedback feature is on.
    let _ = &registry;
    #[cfg(feature = "rd-feedback")]
    register_rd_feedback_nodes(registry);
    #[cfg(feature = "particle-feedback")]
    register_particle_feedback_nodes(registry);
    #[cfg(feature = "fluid-feedback")]
    register_fluid_feedback_nodes(registry);
    #[cfg(feature = "audio-feedback")]
    register_audio_feedback_nodes(registry);
    #[cfg(feature = "automata-feedback")]
    register_automata_feedback_nodes(registry);
    #[cfg(feature = "spring-feedback")]
    register_spring_feedback_nodes(registry);
    #[cfg(feature = "physics-feedback")]
    register_physics_feedback_nodes(registry);
    #[cfg(feature = "space-colonization-feedback")]
    register_space_colonization_feedback_nodes(registry);
    #[cfg(feature = "procgen-feedback")]
    register_procgen_feedback_nodes(registry);
}

#[cfg(all(test, feature = "rd-feedback"))]
mod tests {
    use super::*;
    use crate::registry::SerializableNode;
    use crate::{JsonFormat, deserialize_graph, graph_to_serial, serialize_graph};
    use unshape_core::{EvalContext, Graph, Latch, LatchSnapshot};
    use unshape_rd::feedback::{GrayScottInit, Step, rd_state_type};

    fn extract_params(node: &dyn unshape_core::DynNode) -> Option<JsonValue> {
        let any = node.as_any();
        if let Some(n) = any.downcast_ref::<GrayScottInit>() {
            Some(n.params())
        } else if let Some(n) = any.downcast_ref::<Latch>() {
            Some(n.params())
        } else {
            any.downcast_ref::<Step>().map(|n| n.params())
        }
    }

    /// Builds Init -> latch.init; latch.out -> Step.state; Step.state -> latch.signal.
    fn build() -> (Graph, unshape_core::NodeId) {
        let mut graph = Graph::new();
        let init = graph.add_node(GrayScottInit::circle(48, 48, 24, 24, 6));
        let latch = graph.add_node(Latch::new(rd_state_type()));
        let step = graph.add_node(Step);
        graph.connect(init, 0, latch, 0).unwrap();
        graph.connect(latch, 0, step, 0).unwrap();
        graph.connect(step, 0, latch, 1).unwrap();
        (graph, step)
    }

    fn run(graph: &mut Graph, step: unshape_core::NodeId, n: u64) -> Vec<f32> {
        let mut state = LatchSnapshot::new();
        let r = graph
            .run_to_tick_latched(n, &mut state, |_t| EvalContext::new())
            .unwrap();
        let rd = r
            .get(step, 0)
            .unwrap()
            .downcast_ref::<unshape_rd::ReactionDiffusion>()
            .unwrap();
        rd.v_buffer().to_vec()
    }

    #[test]
    fn recurrent_graph_round_trips_and_reproduces() {
        let target = 12u64;

        // Original.
        let (mut original, ostep) = build();
        let expected = run(&mut original, ostep, target);

        // Serialize the original (rebuild fresh — run consumed the snapshot, but
        // the graph topology + node configs are unchanged).
        let (graph, _) = build();
        let bytes = serialize_graph(&graph, extract_params, &JsonFormat::default()).unwrap();

        // The serialized form carries the Latch as an ordinary node (no feedback
        // wire flag) and never writes the runtime stored value.
        let json = String::from_utf8(bytes.clone()).unwrap();
        assert!(
            json.contains("core::Latch"),
            "graph must serialize a core::Latch node:\n{json}"
        );
        assert!(
            !json.contains("\"feedback\":true"),
            "no feedback wire flag may be written:\n{json}"
        );

        // Deserialize and re-run.
        let mut registry = NodeRegistry::new();
        register_rd_feedback_nodes(&mut registry);
        let mut restored = deserialize_graph(&bytes, &registry, &JsonFormat::default()).unwrap();

        // The restored graph must contain the Latch node.
        assert!(
            restored
                .nodes_iter()
                .any(|(_, n)| n.type_name() == "core::Latch"),
            "restored graph lost its Latch node"
        );

        // Find the Step node id in the restored graph.
        let rstep = restored
            .nodes_iter()
            .find(|(_, n)| n.type_name() == "rd::feedback::Step")
            .map(|(id, _)| id)
            .unwrap();

        let actual = run(&mut restored, rstep, target);
        assert_eq!(expected, actual);
    }

    #[test]
    fn graph_to_serial_has_latch_node_and_direct_wires() {
        let (graph, _) = build();
        let serial = graph_to_serial(&graph, extract_params).unwrap();
        let latch_nodes = serial
            .nodes
            .iter()
            .filter(|n| n.type_name == "core::Latch")
            .count();
        assert_eq!(latch_nodes, 1, "exactly one Latch node expected");
        // All three wires are ordinary direct wires (no feedback flag).
        assert_eq!(serial.wires.len(), 3, "init/out/signal wires");
        assert!(
            serial.wires.iter().all(|w| !w.is_legacy_feedback()),
            "no wire may carry a feedback flag"
        );
    }
}

#[cfg(all(test, feature = "audio-feedback"))]
mod audio_tests {
    use super::*;
    use crate::registry::SerializableNode;
    use crate::{JsonFormat, deserialize_graph, graph_to_serial, serialize_graph};
    use serde_json::Value as JsonValue;
    use unshape_audio::vocoder::VocodeSynth;
    use unshape_audio::vocoder::feedback::{Step, VocoderInit, vocoder_state_type};
    use unshape_core::{Graph, Latch};

    fn extract_params(node: &dyn unshape_core::DynNode) -> Option<JsonValue> {
        let any = node.as_any();
        if let Some(n) = any.downcast_ref::<VocoderInit>() {
            Some(n.params())
        } else if let Some(n) = any.downcast_ref::<Latch>() {
            Some(n.params())
        } else {
            any.downcast_ref::<Step>().map(|n| n.params())
        }
    }

    /// A non-default config so the round-trip exercises real parameter values.
    fn config() -> VocodeSynth {
        VocodeSynth {
            window_size: 512,
            hop_size: 128,
            num_bands: 24,
            envelope_smoothing: 0.75,
        }
    }

    /// Builds Init -> latch.init; latch.out -> Step.state; Step.state -> latch.signal.
    fn build() -> Graph {
        let cfg = config();
        let mut graph = Graph::new();
        let init = graph.add_node(VocoderInit::new(cfg.clone()));
        let latch = graph.add_node(Latch::new(vocoder_state_type()));
        let step = graph.add_node(Step::new(cfg));
        graph.connect(init, 0, latch, 0).unwrap(); // Init -> latch.init (seed)
        graph.connect(latch, 0, step, 0).unwrap(); // latch.out -> step.state
        graph.connect(step, 0, latch, 1).unwrap(); // step.state -> latch.signal
        graph
    }

    #[test]
    fn vocoder_feedback_nodes_register() {
        let mut registry = NodeRegistry::new();
        register_audio_feedback_nodes(&mut registry);
        assert!(registry.contains("vocoder::feedback::VocoderInit"));
        assert!(registry.contains("vocoder::feedback::Step"));
        assert!(registry.contains("core::Latch"));
    }

    #[test]
    fn vocoder_recurrent_graph_round_trips() {
        let graph = build();
        let bytes = serialize_graph(&graph, extract_params, &JsonFormat::default()).unwrap();

        // The serialized form carries the Latch as an ordinary node, no feedback flag.
        let json = String::from_utf8(bytes.clone()).unwrap();
        assert!(
            json.contains("core::Latch"),
            "graph must serialize a core::Latch node:\n{json}"
        );
        assert!(
            !json.contains("\"feedback\":true"),
            "no feedback wire flag may be written:\n{json}"
        );

        // Deserialize against a registry with the audio vocoder nodes registered.
        let mut registry = NodeRegistry::new();
        register_audio_feedback_nodes(&mut registry);
        let restored = deserialize_graph(&bytes, &registry, &JsonFormat::default()).unwrap();

        // The restored graph must preserve the Latch node.
        assert!(
            restored
                .nodes_iter()
                .any(|(_, n)| n.type_name() == "core::Latch"),
            "restored graph lost its Latch node"
        );

        // Both vocoder node types must round-trip with their config intact.
        let restored_step = restored
            .nodes_iter()
            .find_map(|(_, n)| n.as_any().downcast_ref::<Step>())
            .expect("restored graph lost its Step node");
        assert_eq!(restored_step.config.window_size, config().window_size);
        assert_eq!(restored_step.config.num_bands, config().num_bands);

        let restored_init = restored
            .nodes_iter()
            .find_map(|(_, n)| n.as_any().downcast_ref::<VocoderInit>())
            .expect("restored graph lost its VocoderInit node");
        assert_eq!(restored_init.config.hop_size, config().hop_size);
    }

    #[test]
    fn graph_to_serial_has_latch_node() {
        let graph = build();
        let serial = graph_to_serial(&graph, extract_params).unwrap();
        let latch_nodes = serial
            .nodes
            .iter()
            .filter(|n| n.type_name == "core::Latch")
            .count();
        assert_eq!(latch_nodes, 1, "exactly one Latch node expected");
        assert_eq!(serial.wires.len(), 3, "init/out/signal wires");
        assert!(
            serial.wires.iter().all(|w| !w.is_legacy_feedback()),
            "no wire may carry a feedback flag"
        );
    }
}
