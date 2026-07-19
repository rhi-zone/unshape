//! Live sources — external, time-varying data entering the graph.
//!
//! This is the `unshape-live` crate from `docs/design/domain-subsumption.md`'s
//! Synthesis #3 (`LiveSource` / push scheduling). The graph stays pull/lazy;
//! a [`LiveSource`] only changes *when* a source-backed node's cached value
//! goes stale, not the evaluation model itself.
//!
//! # Core abstraction
//!
//! - [`LiveSource`] — trait for anything producing values from outside the
//!   graph: audio/video capture, MIDI, network streams, sensors, file
//!   watching. Domain-specific implementations (real capture devices, stream
//!   clients) belong in their own crates; this one holds only the trait, a
//!   few generic/test sources, and the push-to-pull bridge.
//! - [`LiveCache`] — caches a source's latest value and reports staleness,
//!   the policy referenced in the domain-subsumption doc as living here
//!   rather than in `unshape-core`.
//! - [`LiveSourceNode`] — wraps a `LiveSource` as a zero-input graph node,
//!   shaped like [`unshape_core::nodes::GraphInput`] but polling a source
//!   instead of reading `EvalContext`'s named-input map.
//!
//! # Built-in sources
//!
//! - [`ClockSource`] — wall-clock elapsed time and tick count.
//! - [`SignalGenerator`] (config: [`SignalGeneratorConfig`], shapes:
//!   [`Waveform`]) — sine/ramp/square/constant signal for testing live-source
//!   plumbing without real hardware.
//! - [`ChannelSource`] — generic `mpsc`-channel-backed source; the escape
//!   hatch for feeding arbitrary external data in without a dedicated impl.
//!
//! # Example
//!
//! ```
//! use unshape_core::{DynNode, EvalContext};
//! use unshape_live::{LiveSourceNode, SignalGenerator, Waveform, ValueType};
//!
//! let source = SignalGenerator::new(Waveform::Constant { value: 1.0 });
//! let node = LiveSourceNode::new("gen", ValueType::F32, source);
//! let outputs = node.execute(&[], &EvalContext::new()).unwrap();
//! assert_eq!(outputs[0].as_f32().unwrap(), 1.0);
//! ```

mod channel;
mod clock;
mod node;
mod signal;
mod source;

pub use channel::ChannelSource;
pub use clock::{ClockSample, ClockSource};
pub use node::LiveSourceNode;
pub use signal::{SignalGenerator, SignalGeneratorConfig, Waveform};
pub use source::{LiveCache, LiveSource};

// Re-exported for downstream convenience — most `LiveSourceNode` construction
// sites need `ValueType` (and often `Value`) alongside the types above.
pub use unshape_core::{Value, ValueType};
