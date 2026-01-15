# Audio Graph Primitives Architecture

This document describes the target architecture for audio effects: primitives as graph nodes with parameter modulation, eliminating dedicated composition structs.

## Problem Statement

### Current Architecture

```
Primitives (DelayLine, PhaseOsc, EnvelopeFollower, Allpass1, Smoother)
    ↓ hardcoded wiring in structs
Compositions (ModulatedDelay, AmplitudeMod, AllpassBank)
    ↓ impl AudioNode
Chain/AudioGraph (runtime, dyn dispatch)
```

### Issues

1. **Arbitrary groupings** - Why is `ModulatedDelay` a composition but `DynamicsProcessor` isn't? The boundaries feel arbitrary.

2. **Extensibility burden** - New effects require: new struct, impl AudioNode, export from lib.rs. Users can't easily create custom effects.

3. **Duplicate implementation paths** - Hardcoded compositions duplicate what the graph could do. Maintaining both is wasteful.

4. **Alternate implementations** - GPU/WASM backends must implement both primitives AND compositions.

## Target Architecture

```
Primitives (implement AudioNode directly)
    ↓
Graph with parameter modulation
    ↓
Effects = graph configurations (data, not code)
```

### Benefits

| Aspect | Current | Target |
|--------|---------|--------|
| New effect | New struct + impl + export | Graph configuration |
| Custom modulation | Not possible | Wire any output → any param |
| Alternate backends | Must impl compositions | Only impl primitives |
| Serialization | Code | Data (graph description) |

## Design

### Primitives as AudioNode

Each primitive implements `AudioNode` with modulatable parameters:

```rust
pub trait AudioNode: Send {
    /// Process one sample. Parameters come from context or modulation.
    fn process(&mut self, input: f32, ctx: &AudioContext) -> f32;

    /// Reset internal state.
    fn reset(&mut self) {}

    /// Declare modulatable parameters.
    fn params(&self) -> &[ParamDescriptor] { &[] }

    /// Set a parameter value (called by graph before process).
    fn set_param(&mut self, index: usize, value: f32);
}

pub struct ParamDescriptor {
    pub name: &'static str,
    pub default: f32,
    pub min: f32,
    pub max: f32,
}
```

Example primitive implementations:

```rust
// DelayLine - modulatable delay time
impl AudioNode for DelayLine<true> {
    fn process(&mut self, input: f32, _ctx: &AudioContext) -> f32 {
        self.write(input);
        self.read_interp(self.time)  // time set via set_param
    }

    fn params(&self) -> &[ParamDescriptor] {
        &[ParamDescriptor { name: "time", default: 0.0, min: 0.0, max: 1.0 }]
    }

    fn set_param(&mut self, index: usize, value: f32) {
        if index == 0 { self.time = value; }
    }
}

// PhaseOsc - outputs control signal, modulatable rate
impl AudioNode for PhaseOsc {
    fn process(&mut self, _input: f32, _ctx: &AudioContext) -> f32 {
        self.tick(self.phase_inc)  // phase_inc set via set_param
    }

    fn params(&self) -> &[ParamDescriptor] {
        &[ParamDescriptor { name: "rate", default: 1.0, min: 0.01, max: 100.0 }]
    }

    fn set_param(&mut self, index: usize, value: f32) {
        if index == 0 { self.phase_inc = value; }
    }
}
```

### Parameter Modulation in Graph

The graph supports two connection types:
- **Audio connections** - sample data flows between nodes
- **Param connections** - one node's output modulates another's parameter

```rust
pub struct AudioGraph {
    nodes: Vec<Box<dyn AudioNode>>,
    audio_wires: Vec<AudioWire>,
    param_wires: Vec<ParamWire>,
}

pub struct AudioWire {
    from_node: usize,
    to_node: usize,
}

pub struct ParamWire {
    from_node: usize,
    to_node: usize,
    to_param: usize,
    /// Transform: base + modulation * scale
    scale: f32,
    base: f32,
}

impl AudioGraph {
    /// Connect audio output → audio input
    pub fn connect(&mut self, from: usize, to: usize) { ... }

    /// Connect audio output → parameter (modulation)
    pub fn modulate(&mut self, from: usize, to: usize, param: &str, scale: f32, base: f32) { ... }
}
```

### Graph Execution

Per-sample execution with modulation:

```rust
impl AudioGraph {
    pub fn process(&mut self, input: f32, ctx: &AudioContext) -> f32 {
        // Store node outputs
        let mut outputs = vec![0.0; self.nodes.len()];
        outputs[0] = input;  // Input node

        for (i, node) in self.nodes.iter_mut().enumerate().skip(1) {
            // Apply parameter modulation from connected sources
            for wire in &self.param_wires {
                if wire.to_node == i {
                    let mod_value = outputs[wire.from_node];
                    let param_value = wire.base + mod_value * wire.scale;
                    node.set_param(wire.to_param, param_value);
                }
            }

            // Get audio input (sum of all connected sources)
            let audio_in: f32 = self.audio_wires
                .iter()
                .filter(|w| w.to_node == i)
                .map(|w| outputs[w.from_node])
                .sum();

            outputs[i] = node.process(audio_in, ctx);
        }

        outputs[self.output_node]
    }
}
```

### Effects as Graph Constructors

Effects become functions that build graphs:

```rust
pub fn chorus(sample_rate: f32) -> AudioGraph {
    let mut g = AudioGraph::new();

    let input = g.add_input();
    let lfo = g.add(PhaseOsc::new());
    let delay = g.add(DelayLine::new(4096));
    let output = g.add_output();

    // LFO modulates delay time: base 20ms, depth ±5ms
    g.modulate(lfo, delay, "time", 0.005 * sample_rate, 0.020 * sample_rate);

    // Audio path: input → delay → output (mixed with dry)
    g.connect(input, delay);
    g.connect_mix(input, delay, output, 0.5);  // 50% wet

    g
}

pub fn flanger(sample_rate: f32) -> AudioGraph {
    // Same structure as chorus, different parameters!
    let mut g = AudioGraph::new();
    // ... shorter delay, more feedback, faster LFO
    g
}
```

### Relationship to resin-core Graph

The `resin-core` Graph is for general node-based pipelines (meshes, textures, etc.) with typed ports and single execution.

The audio graph needs:
- Per-sample execution (not single-shot)
- Parameter modulation (control signals → params)
- Stateful nodes (delay lines, filters)

Options:

1. **Separate AudioGraph** (current direction) - Audio-specific, optimized for sample processing
2. **Extend resin-core** - Add rate/modulation concepts to general graph
3. **Shared modulation abstraction** - `Modulatable<T>` type used by both

Recommendation: Keep `AudioGraph` separate for now, but design `Modulatable<T>` as a shared concept that both can use later. The execution models are too different to force into one graph type.

## Migration Path

### Phase 1: Primitives as AudioNode
- Add `params()` and `set_param()` to AudioNode trait
- Implement AudioNode for primitives in `primitive.rs`
- Keep existing compositions working

### Phase 2: Graph with Modulation
- Add `param_wires` to AudioGraph
- Add `modulate()` method
- Update execution to apply modulation

### Phase 3: Effects as Graphs
- Rewrite `chorus()`, `flanger()`, etc. to return AudioGraph
- Deprecate composition structs (ModulatedDelay, etc.)
- Update benchmarks to verify no regression

### Phase 4: Cleanup
- Remove composition structs
- Primitives become the only building blocks
- Document graph-based effect creation

## Performance Considerations

### Overhead Sources

| Source | Impact | Mitigation |
|--------|--------|------------|
| Node iteration | Low | Small graphs (3-5 nodes) |
| Dyn dispatch | Low | Called once per sample per node |
| Param lookup | Low | Index-based, no string lookup at runtime |
| Modulation math | Negligible | One multiply-add per modulated param |

### Why This Should Be Fast

1. **Small graphs** - Chorus is ~3 nodes, not 100
2. **No allocation** - All storage pre-allocated
3. **Linear iteration** - Topological order cached
4. **Same math** - Modulation is just `base + output * scale`

The hardcoded compositions do the same work, just with the structure baked in. The graph makes the structure explicit but shouldn't add overhead.

### Benchmark Strategy

Before removing compositions:
1. Benchmark current compositions
2. Implement graph versions
3. Compare - should be within 10%
4. If slower, profile and optimize graph execution

## Future: Control Rate

Currently deferred, but the design supports it:

```rust
pub trait AudioNode {
    /// How often params need updating (default: every sample)
    fn control_rate(&self) -> ControlRate { ControlRate::Audio }
}

pub enum ControlRate {
    Audio,           // Every sample
    Block(usize),    // Every N samples
    OnChange,        // Only when modulator changes
}
```

The graph executor can batch param updates for `Block` rate nodes, reducing overhead for LFOs that don't need sample-accurate modulation.

## Open Questions

1. **Feedback loops** - Audio effects often have feedback (delay output → input). The graph needs to handle this without cycle errors. Solution: explicit `Feedback` node that provides one-sample delay.

2. **Multi-output nodes** - Some primitives (like stereo processors) have multiple outputs. Current design assumes single output. May need `outputs: Vec<f32>` instead of single return.

3. **Preset serialization** - Effects-as-graphs are naturally serializable (node types + connections). Need schema for saving/loading effect presets.

## Related Documents

- `unification.md` - Original analysis of audio/graphics effects
- `../domains/audio.md` - Audio domain overview
- `../open-questions.md` - Original modulation decisions (unimplemented)
