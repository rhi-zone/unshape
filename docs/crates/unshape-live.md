# unshape-live

`LiveSource` abstraction for external, time-varying data entering the graph.

## Purpose

This is the `unshape-live` crate from `docs/design/domain-subsumption.md`'s
Synthesis #3 (`LiveSource` / push scheduling — the primitive OBS's camera/capture
inputs, ClickHouse's "rows landed," and Resolve's capture cards all share). The
graph stays pull/lazy; a `LiveSource` only changes *when* a source-backed node's
cached value goes stale, not the evaluation model itself. This is the mechanism
behind unshape's "any node input can be a live signal" philosophy.

The crate provides:
- **`LiveSource`** — the trait: `poll(&mut self) -> Option<Output>`, non-blocking,
  `None` if nothing new. Domain-specific implementations (real audio/video
  capture, network clients) belong in their own crates; this trait is only the
  shared shape they poll through.
- **`LiveCache`** — bridges a source's push/pull data into a cached "latest
  value," reporting staleness (`is_stale`) between updates. This is the
  cache-invalidation policy the domain-subsumption doc calls out as living
  here rather than in `unshape-core`.
- **`LiveSourceNode`** — wraps a `LiveSource` as a zero-input `DynNode`, shaped
  like `unshape_core::nodes::GraphInput` (one `"value"` output, optional
  default) but polling a source each execute instead of reading
  `EvalContext`'s named-input map.

Built-in concrete sources (test/utility-grade, not production capture):
- **`ClockSource`** — wall-clock elapsed time and tick count, as `ClockSample`
  (an opaque `Value` — see `GraphValue`/`Value::opaque`).
- **`SignalGenerator`** (config: `SignalGeneratorConfig`, shapes: `Waveform`) —
  sine/ramp/square/constant signal varying with wall-clock time, for testing
  live-source plumbing without real hardware. Follows the ops-as-values split:
  `Waveform`/`SignalGeneratorConfig` are serializable parameters, `start()`
  builds the (non-serializable, clock-holding) running source.
- **`ChannelSource<T>`** — generic `mpsc`-channel-backed source; the escape
  hatch for feeding arbitrary external data in without writing a dedicated
  `LiveSource` impl (hand the `Sender` to a capture thread, wrap the
  `Receiver` in a node).

## Related Crates

- **unshape-core** — `DynNode`, `Value`, `EvalContext`, `GraphInput` (the
  pull-only sibling this crate's push-aware `LiveSourceNode` parallels).
- **unshape-field** / **unshape-field-ops** — `Field<I, O>` for spatial/lazy
  sampling; a `LiveSource`'s cached output can feed a `Field` as a
  time-varying parameter (e.g. modulate a noise field's frequency from a
  `SignalGenerator`).
- Future capture-device crates (audio input, video capture, MIDI, screen
  capture) are expected to depend on `unshape-live` for the trait and wrap
  their own hardware/OS APIs behind it, per the "domain-specific
  implementations live elsewhere" split in `docs/design/domain-subsumption.md`.

## Example: wiring a live source into the graph

```rust
use unshape_core::{DynNode, EvalContext};
use unshape_live::{LiveSourceNode, SignalGenerator, Waveform, ValueType};

let source = SignalGenerator::new(Waveform::Sine {
    frequency_hz: 1.0,
    amplitude: 1.0,
    phase: 0.0,
});
let node = LiveSourceNode::new("lfo", ValueType::F32, source);

// Each execute() polls the source for a fresh reading.
let outputs = node.execute(&[], &EvalContext::new()).unwrap();
```

## Example: feeding external data via a channel

```rust
use unshape_live::ChannelSource;

let (tx, mut source) = ChannelSource::<f32>::channel();

// From another thread / callback:
tx.send(0.5).unwrap();

// The graph side (non-blocking, "latest wins"):
use unshape_live::LiveSource;
let latest = source.poll(); // Some(0.5)
```

## Compositions

### With unshape-field-ops
Use a `LiveSource`'s cached value as a time-varying parameter for a `Field`:
sample a `SignalGenerator` once per frame via `LiveCache`, then feed the
result into a `Field` combinator (e.g. `Scale`, `Translate`) as the
per-frame parameter, rather than baking a fixed value into the graph.

### With unshape-timeline (future)
A `LiveSource`-backed node can sit alongside `Timeline`-driven clips in a
composite graph — e.g. an OBS-style scene mixing a recorded clip (sampled by
local time) with a live capture input (polled from a `LiveSource`).
