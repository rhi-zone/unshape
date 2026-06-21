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
    feature = "audio-feedback"
))]
use crate::registry::SerializableNode;
#[cfg(any(
    feature = "rd-feedback",
    feature = "particle-feedback",
    feature = "fluid-feedback",
    feature = "audio-feedback"
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
    feature = "audio-feedback"
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
        registry.register_with_name::<VocoderInit>("vocoder::feedback::VocoderInit");
        registry.register_with_name::<Step>("vocoder::feedback::Step");
    }
}

#[cfg(feature = "audio-feedback")]
pub use audio::register as register_audio_feedback_nodes;

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
}

#[cfg(all(test, feature = "rd-feedback"))]
mod tests {
    use super::*;
    use crate::{JsonFormat, deserialize_graph, graph_to_serial, serialize_graph};
    use unshape_core::{EvalContext, FeedbackState, Graph};
    use unshape_rd::feedback::{GrayScottInit, Step};

    fn extract_params(node: &dyn unshape_core::DynNode) -> Option<JsonValue> {
        let any = node.as_any();
        if let Some(n) = any.downcast_ref::<GrayScottInit>() {
            Some(n.params())
        } else {
            any.downcast_ref::<Step>().map(|n| n.params())
        }
    }

    /// Builds `Init --direct--> Step.state`, `Step --feedback--> Step.state`.
    fn build() -> (Graph, unshape_core::NodeId) {
        let mut graph = Graph::new();
        let init = graph.add_node(GrayScottInit::circle(48, 48, 24, 24, 6));
        let step = graph.add_node(Step);
        graph.connect(init, 0, step, 0).unwrap();
        graph.connect_recurrence(step, 0, step, 0).unwrap();
        (graph, step)
    }

    fn run(graph: &mut Graph, step: unshape_core::NodeId, n: u64) -> Vec<f32> {
        let mut state = FeedbackState::new();
        let r = graph
            .run_to_tick(n, &mut state, |_t| EvalContext::new())
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

        // Serialize the original (rebuild fresh — run consumed feedback state,
        // but the graph topology + node configs are unchanged).
        let (graph, _) = build();
        let bytes = serialize_graph(&graph, extract_params, &JsonFormat::default()).unwrap();

        // The serialized form must carry the recurrence wire as `feedback: true`.
        let json = String::from_utf8(bytes.clone()).unwrap();
        assert!(
            json.contains("\"feedback\":true"),
            "recurrence wire must serialize with feedback flag set:\n{json}"
        );

        // Deserialize and re-run.
        let mut registry = NodeRegistry::new();
        register_rd_feedback_nodes(&mut registry);
        let mut restored = deserialize_graph(&bytes, &registry, &JsonFormat::default()).unwrap();

        // The restored graph must contain a recurrence (feedback) wire.
        assert!(
            restored.wires().iter().any(|w| w.feedback),
            "restored graph lost its recurrence wire"
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
    fn graph_to_serial_marks_recurrence_wire() {
        let (graph, _) = build();
        let serial = graph_to_serial(&graph, extract_params).unwrap();
        let feedback_wires = serial.wires.iter().filter(|w| w.feedback).count();
        assert_eq!(feedback_wires, 1, "exactly one recurrence wire expected");
        let direct_wires = serial.wires.iter().filter(|w| !w.feedback).count();
        assert_eq!(direct_wires, 1, "exactly one direct seed wire expected");
    }
}
