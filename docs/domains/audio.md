# Audio

Audio synthesis and processing.

## Prior Art

### Pure Data (Pd) / Max/MSP
- **Dataflow**: objects connected by patch cords
- **Two rates**: audio rate (~44100 Hz) vs control rate (~64 samples)
- **Hot/cold inlets**: leftmost inlet triggers computation
- **Abstractions**: patches as reusable objects

### SuperCollider
- **SynthDef**: define synth as UGen graph, instantiate as Synth
- **UGens**: unit generators (oscillators, filters, etc.)
- **Demand rate**: generates values on demand (sequencers)
- **Buffers**: sample data, wavetables

### FAUST
- **Functional DSP**: audio as pure functions on streams
- **Block diagram algebra**: sequential `:`, parallel `,`, split `<:`, merge `:>`
- **Automatic differentiation**: for physical modeling

### VCV Rack / Modular Synths
- **Modules**: self-contained units with inputs/outputs/knobs
- **Polyphony**: multiple voices per cable
- **CV/Gate**: control voltage for parameters, gates for triggers

## Core Types

```rust
/// Audio processing context
struct AudioContext {
    sample_rate: f32,
    block_size: usize,
    time: f64,  // in samples
}

/// A node in the audio graph
trait AudioNode {
    fn process(&mut self, ctx: &AudioContext, inputs: &[&Buffer], outputs: &mut [Buffer]);
    fn input_count(&self) -> usize;
    fn output_count(&self) -> usize;
}

/// Audio buffer - block of samples
struct Buffer {
    samples: Vec<f32>,  // block_size samples
    // Or: channels: Vec<Vec<f32>> for multi-channel?
}

/// Control-rate value - updated once per block
struct Control {
    value: f32,
    // smoothing?
}

/// Trigger/gate signal
struct Gate {
    state: bool,
    triggered: bool,  // rising edge this block
}
```

## Primitives (Generators)

| Primitive | Parameters | Notes |
|-----------|------------|-------|
| Sine | frequency, phase | Pure tone |
| Square | frequency, duty_cycle | PWM |
| Saw | frequency | Rising sawtooth |
| Triangle | frequency | |
| Noise | color (white/pink/brown) | Random |
| Wavetable | table, position, frequency | Interpolated lookup |
| Sampler | buffer, position, speed | Sample playback |
| Impulse | frequency | Single-sample clicks |

## Operations (Processors)

### Filters
- **LowPass**: cutoff, resonance (Q)
- **HighPass**: cutoff, resonance
- **BandPass**: center, bandwidth
- **Notch**: center, bandwidth
- **StateVariable**: morph between LP/HP/BP
- **Comb**: delay, feedback
- **AllPass**: delay, coefficient

### Envelopes
- **ADSR**: attack, decay, sustain, release, gate input
- **AR**: attack, release
- **MultiStage**: arbitrary segments
- **Function**: f(t) -> value

### Effects
- **Delay**: time, feedback, mix
- **Reverb**: size, damping, mix
- **Chorus**: rate, depth, mix
- **Distortion**: drive, type (soft/hard clip, fold, etc.)
- **Bitcrush**: bit depth, sample rate reduction
- **Compressor**: threshold, ratio, attack, release

### Modulation
- **LFO**: frequency, shape -> control signal
- **EnvFollower**: audio -> control (amplitude tracking)
- **S&H**: sample and hold

### Mixing
- **Gain**: amplitude
- **Pan**: stereo position
- **Mix**: combine signals
- **Crossfade**: blend A/B

## Data Flow Pattern

```
Generator -> Processor -> Processor -> ... -> Output
    ↑           ↑
    param       param (can be audio-rate or control-rate)
```

Key distinction from meshes: **continuous streaming** vs discrete operations. Audio nodes process blocks continuously.

### Modulation Routing

```
LFO ──────────┐
              ↓
Oscillator ──-> Filter ──-> Output
              ↑
Envelope ─────┘
```

Parameters can be modulated by other signals. This is central to audio.

## Polyphony

Multiple voices playing the same patch:

```rust
struct Poly<N: AudioNode> {
    voices: Vec<N>,
    voice_allocation: VoiceAllocator,
}
```

Each voice has its own state. Voice allocation handles note-on/off.

## Open Questions

1. **Sample rate**: Fixed at graph creation, or runtime-configurable? FAUST compiles for specific rate, Pd is flexible.

2. **Block size**: Fixed or variable? Larger blocks = more efficient, smaller = less latency.

3. **Modulation depth**: Every parameter modulatable? Or explicit mod inputs? VCV Rack: everything is a cable. Pd: explicit inlets.

4. **Polyphony model**: Per-node (VCV poly cables)? Per-graph (Pd [clone])? Explicit voice management?

5. **Control vs audio rate**: Automatic promotion? Explicit types? SuperCollider has `.kr` vs `.ar` methods.

6. **State management**: Filters have state. How does this interact with "pure graph" model?
