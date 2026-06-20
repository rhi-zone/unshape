//! Recurrent-graph integration for the phase [`Vocoder`].
//!
//! Ports the stateful vocoder onto the feedback-edge mechanism in `unshape-core`
//! (`docs/design/recurrent-graphs.md`). The entire cross-block mutable state
//! ([`VocoderState`]: per-band envelopes, carrier/modulator input ring buffers,
//! overlap-add output buffer, and read position) rides a feedback wire as an
//! opaque [`Value`]. The carrier and modulator blocks are *per-tick inputs*, not
//! feedback. The step node ([`Step`]) is therefore a pure `&self` [`DynNode`]: it
//! reads the previous state plus this tick's carrier/modulator blocks, runs one
//! pure [`Vocoder::step`], and returns the new state and the vocoded output block.
//!
//! # Why this fits the per-tick node model
//!
//! `Vocoder::process` was already block-streaming: each call advances exactly the
//! five fields now bundled in [`VocoderState`] and returns one output block. The
//! static configuration (params, window, band layout) is immutable and lives in
//! the node ([`Vocoder`] rebuilt from [`VocodeSynth`]). So "one tick == one block"
//! maps cleanly: state on the wire, config in the node, signal blocks as inputs.
//!
//! # Tick-0 state
//!
//! Opaque types have no zero value, so the initial [`VocoderState`]
//! ([`VocoderState::new`]) must be pre-seeded into
//! [`FeedbackState`](unshape_core::FeedbackState) before the first tick.

use std::any::Any;

use unshape_core::{
    DataLocation, DynNode, EvalContext, GraphError, GraphValue, PortDescriptor, Value, ValueType,
};

use crate::vocoder::{VocodeSynth, Vocoder, VocoderState};

/// The opaque value type name for [`VocoderState`] on a wire.
pub const VOCODER_STATE_NAME: &str = "VocoderState";

/// The opaque value type name for an audio block (`Vec<f32>`) on a wire.
pub const AUDIO_BLOCK_NAME: &str = "AudioBlock";

impl GraphValue for VocoderState {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn type_name(&self) -> &'static str {
        VOCODER_STATE_NAME
    }

    fn location(&self) -> DataLocation {
        DataLocation::Cpu
    }
}

/// A block of audio samples flowing through the graph as an opaque value.
///
/// Used for the vocoder's carrier/modulator inputs and vocoded output. A thin
/// newtype so `Vec<f32>` can be carried in [`Value::Opaque`].
#[derive(Debug, Clone)]
pub struct AudioBlock(pub Vec<f32>);

impl AudioBlock {
    /// Borrow the samples.
    pub fn samples(&self) -> &[f32] {
        &self.0
    }

    /// Consume into the inner sample vector.
    pub fn into_inner(self) -> Vec<f32> {
        self.0
    }
}

impl GraphValue for AudioBlock {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn type_name(&self) -> &'static str {
        AUDIO_BLOCK_NAME
    }

    fn location(&self) -> DataLocation {
        DataLocation::Cpu
    }
}

/// Returns the [`ValueType`] used for [`VocoderState`] on a wire.
pub fn vocoder_state_type() -> ValueType {
    ValueType::of::<VocoderState>(VOCODER_STATE_NAME)
}

/// Returns the [`ValueType`] used for an [`AudioBlock`] on a wire.
pub fn audio_block_type() -> ValueType {
    ValueType::of::<AudioBlock>(AUDIO_BLOCK_NAME)
}

/// Pure per-tick vocoder step node.
///
/// State (envelopes + ring/overlap buffers + position) lives on the feedback
/// edge; the carrier and modulator blocks are per-tick inputs. `execute` rebuilds
/// the static [`Vocoder`] from `config`, runs one pure [`Vocoder::step`], and
/// returns the new state and the vocoded output block.
///
/// # Ports
/// - Input `0` `"state"`: `Custom(VocoderState)` — previous-tick state (feedback).
/// - Input `1` `"carrier"`: `Custom(AudioBlock)` — this tick's carrier block.
/// - Input `2` `"modulator"`: `Custom(AudioBlock)` — this tick's modulator block.
/// - Output `0` `"state"`: `Custom(VocoderState)` — advanced state.
/// - Output `1` `"output"`: `Custom(AudioBlock)` — vocoded output block.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Step {
    /// Static vocoder configuration (window/hop/bands/smoothing).
    pub config: VocodeSynth,
}

impl Step {
    /// Creates a step node from a vocoder configuration.
    pub fn new(config: VocodeSynth) -> Self {
        Self { config }
    }

    /// Returns the zero initial state matching this node's configuration.
    ///
    /// Pre-seed this into the [`FeedbackState`](unshape_core::FeedbackState) for
    /// the step node's state port before tick 0.
    pub fn initial_state(&self) -> VocoderState {
        VocoderState::new(&self.config)
    }
}

impl Default for Step {
    fn default() -> Self {
        Self::new(VocodeSynth::default())
    }
}

impl DynNode for Step {
    fn type_name(&self) -> &'static str {
        "vocoder::feedback::Step"
    }

    fn inputs(&self) -> Vec<PortDescriptor> {
        vec![
            PortDescriptor::new("state", vocoder_state_type()),
            PortDescriptor::new("carrier", audio_block_type()),
            PortDescriptor::new("modulator", audio_block_type()),
        ]
    }

    fn outputs(&self) -> Vec<PortDescriptor> {
        vec![
            PortDescriptor::new("state", vocoder_state_type()),
            PortDescriptor::new("output", audio_block_type()),
        ]
    }

    fn execute(&self, inputs: &[Value], _ctx: &EvalContext) -> Result<Vec<Value>, GraphError> {
        let state = inputs[0].downcast_ref::<VocoderState>().ok_or_else(|| {
            GraphError::ExecutionError(
                "vocoder::feedback::Step expects a VocoderState state input".to_string(),
            )
        })?;
        let carrier = inputs[1].downcast_ref::<AudioBlock>().ok_or_else(|| {
            GraphError::ExecutionError(
                "vocoder::feedback::Step expects an AudioBlock carrier input".to_string(),
            )
        })?;
        let modulator = inputs[2].downcast_ref::<AudioBlock>().ok_or_else(|| {
            GraphError::ExecutionError(
                "vocoder::feedback::Step expects an AudioBlock modulator input".to_string(),
            )
        })?;
        if carrier.0.len() != modulator.0.len() {
            return Err(GraphError::ExecutionError(format!(
                "vocoder::feedback::Step: carrier ({}) and modulator ({}) blocks must be equal length",
                carrier.0.len(),
                modulator.0.len()
            )));
        }

        let vocoder = Vocoder::new(self.config.clone());
        let step = vocoder.step(state, &carrier.0, &modulator.0);
        Ok(vec![
            Value::opaque(step.state),
            Value::opaque(AudioBlock(step.output)),
        ])
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;
    use unshape_core::{FeedbackState, Graph};

    fn config() -> VocodeSynth {
        // Small window for fast tests; hop divides block length.
        VocodeSynth {
            window_size: 256,
            hop_size: 64,
            num_bands: 8,
            envelope_smoothing: 0.8,
        }
    }

    fn carrier_block(block: usize, len: usize) -> Vec<f32> {
        (0..len)
            .map(|i| {
                let t = (block * len + i) as f32;
                (2.0 * PI * 440.0 * t / 44100.0).sin()
            })
            .collect()
    }

    fn modulator_block(block: usize, len: usize) -> Vec<f32> {
        (0..len)
            .map(|i| {
                let t = (block * len + i) as f32;
                (2.0 * PI * 110.0 * t / 44100.0).sin()
            })
            .collect()
    }

    /// A source node that emits a fixed audio block (host-injected per graph).
    struct BlockSource {
        block: Vec<f32>,
    }

    impl DynNode for BlockSource {
        fn type_name(&self) -> &'static str {
            "vocoder::test::BlockSource"
        }
        fn inputs(&self) -> Vec<PortDescriptor> {
            vec![]
        }
        fn outputs(&self) -> Vec<PortDescriptor> {
            vec![PortDescriptor::new("block", audio_block_type())]
        }
        fn execute(&self, _inputs: &[Value], _ctx: &EvalContext) -> Result<Vec<Value>, GraphError> {
            Ok(vec![Value::opaque(AudioBlock(self.block.clone()))])
        }
        fn as_any(&self) -> &dyn Any {
            self
        }
    }

    #[test]
    fn evolves_like_mut_process_loop() {
        // (a) feedback stepping N blocks matches calling Vocoder::process N times.
        let len = 256usize;
        let n = 6usize;

        // Reference: imperative &mut process loop across the same blocks.
        let mut reference = Vocoder::new(config());
        let mut ref_outputs = Vec::new();
        for b in 0..n {
            let c = carrier_block(b, len);
            let m = modulator_block(b, len);
            ref_outputs.push(reference.process(&c, &m));
        }

        // Feedback driver: state on a back-edge, carrier/modulator as inputs.
        // Rebuild per tick because the input blocks differ each tick.
        let mut state = FeedbackState::new();
        // Determine the step node id by building once.
        let mut feedback_outputs = Vec::new();
        for b in 0..n {
            let mut graph = Graph::new();
            let car = graph.add_node(BlockSource {
                block: carrier_block(b, len),
            });
            let modn = graph.add_node(BlockSource {
                block: modulator_block(b, len),
            });
            let step = graph.add_node(Step::new(config()));
            graph.connect(car, 0, step, 1).unwrap();
            graph.connect(modn, 0, step, 2).unwrap();
            graph.connect_feedback(step, 0, step, 0).unwrap();

            // Seed the state port: on the first block, the zero initial state;
            // afterwards, the carried state. Because we rebuild the graph each
            // tick, node ids are stable (same construction order), so `step` is
            // the same id every iteration and the feedback key lines up.
            if b == 0 {
                state.set(step, 0, Value::opaque(VocoderState::new(&config())));
            }

            let r = graph
                .tick(b as u64, &mut state, &EvalContext::new())
                .unwrap();
            let out = r
                .get(step, 1)
                .unwrap()
                .downcast_ref::<AudioBlock>()
                .unwrap()
                .0
                .clone();
            feedback_outputs.push(out);
        }

        for (rf, fb) in ref_outputs.iter().zip(&feedback_outputs) {
            assert_eq!(rf.len(), fb.len());
            for (a, b) in rf.iter().zip(fb) {
                assert!((a - b).abs() < 1e-5, "ref={a} fb={b}");
            }
        }
    }

    #[test]
    fn node_is_pure_fresh_state_restarts() {
        // (b) the node holds no cross-tick state: a fresh seeded state, same
        // input block -> identical output to a single pure step from zero state.
        let len = 256usize;
        let cfg = config();

        let mut graph = Graph::new();
        let car = graph.add_node(BlockSource {
            block: carrier_block(0, len),
        });
        let modn = graph.add_node(BlockSource {
            block: modulator_block(0, len),
        });
        let step = graph.add_node(Step::new(cfg.clone()));
        graph.connect(car, 0, step, 1).unwrap();
        graph.connect(modn, 0, step, 2).unwrap();
        graph.connect_feedback(step, 0, step, 0).unwrap();

        let mut state = FeedbackState::new();
        state.set(step, 0, Value::opaque(VocoderState::new(&cfg)));
        let r = graph.tick(0, &mut state, &EvalContext::new()).unwrap();
        let node_out = r
            .get(step, 1)
            .unwrap()
            .downcast_ref::<AudioBlock>()
            .unwrap()
            .0
            .clone();

        let voc = Vocoder::new(cfg.clone());
        let pure = voc.step(
            &VocoderState::new(&cfg),
            &carrier_block(0, len),
            &modulator_block(0, len),
        );
        assert_eq!(node_out.len(), pure.output.len());
        for (a, b) in node_out.iter().zip(&pure.output) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn deterministic() {
        // (c) same config + inputs + N -> identical output.
        let len = 256usize;
        let n = 5usize;
        let run = || {
            let mut state = FeedbackState::new();
            let mut last = Vec::new();
            for b in 0..n {
                let mut graph = Graph::new();
                let car = graph.add_node(BlockSource {
                    block: carrier_block(b, len),
                });
                let modn = graph.add_node(BlockSource {
                    block: modulator_block(b, len),
                });
                let step = graph.add_node(Step::new(config()));
                graph.connect(car, 0, step, 1).unwrap();
                graph.connect(modn, 0, step, 2).unwrap();
                graph.connect_feedback(step, 0, step, 0).unwrap();
                if b == 0 {
                    state.set(step, 0, Value::opaque(VocoderState::new(&config())));
                }
                let r = graph
                    .tick(b as u64, &mut state, &EvalContext::new())
                    .unwrap();
                last = r
                    .get(step, 1)
                    .unwrap()
                    .downcast_ref::<AudioBlock>()
                    .unwrap()
                    .0
                    .clone();
            }
            last
        };
        let a = run();
        let b = run();
        assert_eq!(a, b);
    }
}
