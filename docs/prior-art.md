# Prior Art

Existing tools and techniques that inform Resin's design.

## Mesh & Texture Generation

### .kkrieger (2004)

Demoscene FPS game in 96KB. Procedurally generates all meshes and textures at runtime using:
- Mesh generators (boxes, spheres, extrusions)
- Texture operators (noise, blur, combine)
- Everything described as operation graphs

Key insight: tiny code + parameters = rich content.

### Blender Geometry Nodes

Visual node-based system for procedural geometry:
- Attribute system (data on vertices, edges, faces)
- Fields (lazy-evaluated expressions over geometry)
- Instances (efficient duplication)

Key insight: fields/lazy evaluation enable complex procedural logic without explicit loops.

## Character & Animation

### MakeHuman

Open source character creator:
- Parametric body morphs
- Topology-preserving deformation
- Export to standard formats

Key insight: a well-designed base mesh + morph targets = infinite variation.

### Live2D Cubism

2D character rigging for animation:
- Mesh deformation of 2D artwork
- Parameters drive deformers
- Physics simulation on parameters

Key insight: 2D art can have skeletal-style rigging without 3D.

### Toon Boom Harmony

Professional 2D animation:
- Deformers (bone, envelope, bend)
- Drawing substitution
- Compositing

Key insight: deformers are the bridge between rig and render.

## Audio

### Pure Data (Pd)

Visual dataflow programming for audio:
- Objects connected by patch cords
- Everything is a signal or message
- Real-time synthesis and processing

Key insight: audio is naturally dataflow - sources -> processors -> output.

### Synths & Modular

- Oscillators, filters, envelopes as composable units
- Modulation routing (LFO -> filter cutoff)
- Polyphony as instance management

Key insight: a small set of primitives (osc, filter, env, lfo) covers vast sonic territory.

### MetaSynth

[Website](https://uisoftware.com/metasynth/)

Image-based audio synthesis (used in Dune, Inception, The Matrix):
- **Spectral painting**: x=time, y=frequency, brightness=amplitude, color=stereo
- **Bidirectional**: paint sound OR analyze audio→image, manipulate, resynthesize
- **Multi-synthesis**: wavetable, FM, additive, granular - all via image metaphor
- **16-bit images**: 65,536 amplitude levels for subtle spectral work

Key insight: **representation is transferable**. The same structure (2D field) can be an image OR a spectrogram. The "domain" (visual/audio) is just how you render it.

**Relevance to Resin**: We have `ImageField` + spectral audio. Natural connection: procedural textures → spectral content. Noise fields as timbral control. Reaction-diffusion as evolving spectra.

## Cross-Domain Reinterpretation

### Glitch Art / Databending

Intentionally "wrong" interpretation of data:
- **Databending**: import JPEG as raw audio in Audacity, process, export back
- **Hex editing**: corrupt headers, force partial decoding
- **Wrong codec**: decode MP3 as raw PCM, treat WAV as image pixels
- **Pixel sorting**: image rows as sortable numeric data
- **Circuit bending**: hardware-level misinterpretation

Key insight: **format is convention, not truth**. The same bytes are simultaneously valid (if weird) audio, image, mesh data. There's no "correct" interpretation - just useful ones.

**Relevance to Resin**:
- Raw buffer reinterpretation as creative tool (`&[f32]` → audio samples OR vertex positions)
- Intentional format mismatches (mesh normals as RGB, audio as heightfield)
- "Wrong" mappings that produce interesting results
- Corruption/noise injection as first-class operations

### The General Pattern

Both MetaSynth and glitch art point to the same underlying idea:

```
Structure (graph, field, buffer)
    ↓ interpretation A
  Image
    ↓ interpretation B
  Audio
    ↓ interpretation C
  Mesh
```

The structure is the real thing. Domain-specific outputs are just projections. This aligns with resin's core idea: everything is procedurally describable, and the "output format" is a rendering choice.

### TidalCycles / Strudel

[TidalCycles](https://tidalcycles.org/) (Haskell) · [Strudel](https://strudel.cc/) (JS)

Pattern-based live coding for music:

```javascript
s("bd sd [~ bd] sd").speed("1 2 1.5")
```

- **Mini-notation**: domain-specific string syntax for rhythms
- **Pattern transformations**: `fast()`, `slow()`, `rev()`, `jux()`
- **Patterns as values**: composable, higher-order
- **Time is fundamental**: patterns are functions of time

Key insight: **patterns as composable values**. The mini-notation is TidalCycles-specific; the real insight is that pattern transformations (`fast`, `slow`, `rev`) are composable ops - same model as resin.

## Image Processing

### Dithering Techniques

Classic dithering for quantization (Floyd-Steinberg, Atkinson, Sierra, etc.) is well-documented. More interesting prior art:

#### Werness/Koloth Dithering (Obra Dinn)

[GitHub](https://github.com/akavel/WernessDithering) · [Lucas Pope devlog](https://dukope.com/devlogs/obra-dinn/tig-18/)

Hybrid noise-threshold + error-diffusion invented by Brent Werness for Return of the Obra Dinn:
- **Inverted approach**: each pixel absorbs neighbors' errors rather than spreading its own
- **Blue noise seeding**: thresholds seeded with blue noise for better distribution
- **Edge preservation**: maintains detail where pattern dithering sacrifices it
- **GPU-friendly**: works as shader despite error diffusion being inherently serial

Limitations: poor at gradient extremes (light/dark ends), works best with detailed content.

Key insight: **algorithm inversion**. Flipping who "owns" the error (absorb vs spread) enables GPU parallelism while keeping diffusion benefits.

#### Surface-Stable Fractal Dithering

[GitHub](https://github.com/runevision/Dither3D) · [Playdate port](https://github.com/aras-p/playdate-dither3d)

3D rendering technique by Rune Skovbo Johansen:
- **Surface adhesion**: dither dots stick to 3D surfaces
- **Scale-invariant density**: dots add/remove with zoom, never pop
- **Fractal Bayer matrices**: exploits self-similarity for smooth transitions

Key insight: **self-similarity enables LOD**. Fractal properties of Bayer matrices allow seamless density transitions - dots only appear when zooming in, only vanish when zooming out.

Note: This is a rendering/shader technique, not 2D image processing - out of scope for resin-image but interesting for 3D applications.

## Animation & Motion

### Motion Canvas

[Website](https://motioncanvas.io/) · [GitHub](https://github.com/motion-canvas/motion-canvas)

TypeScript library for programmatic animation:
- **Generator-based timeline**: code execution IS the animation sequence
- **Procedural animation**: specify duration/easing, not keyframes
- **JSX-like scene graph**: declarative visual hierarchy
- **Real-time preview**: Vite-powered hot reload

```typescript
// Animation as generator
yield* circle().scale(2, 0.5);  // scale to 2 over 0.5s
yield* all(
  rect().x(100, 1),
  rect().fill('red', 1),
);
```

Key insight: generators turn imperative code into a timeline. Animation is just "what happens when this code runs."

**Relevance to Resin**: The generator model could work for animation sequencing. Less relevant for spatial generation (meshes, textures) but interesting for rigging/animation domain.

### Manim

[Docs](https://docs.manim.community/) · [GitHub (community)](https://github.com/ManimCommunity/manim) · [GitHub (3b1b)](https://github.com/3b1b/manim)

Python math animation engine (3blue1brown):
- **Scene-based**: each animation is a scene class with `construct()` method
- **Mobjects**: mathematical objects as primitives (shapes, LaTeX, graphs)
- **Transformations**: morph between mobjects with `Transform`, `ReplacementTransform`
- **Camera control**: pan, zoom, 3D rotation

```python
class Example(Scene):
    def construct(self):
        circle = Circle()
        square = Square()
        self.play(Create(circle))
        self.play(Transform(circle, square))
```

Key insight: scenes as composable units, transformations as first-class animations.

**Relevance to Resin**: Mobject concept maps to our generators. Transform animations relevant for morphing (mesh blend shapes, path morphing). Scene model less relevant - we're a library, not a framework.

## Node-Based Workflows

### ComfyUI

[Website](https://www.comfy.org/) · [GitHub](https://github.com/comfyanonymous/ComfyUI)

Node-based interface for generative AI:
- **Visual DAG**: connect nodes on canvas, data flows through
- **Multi-domain**: image, video, 3D, audio in one system
- **Portable workflows**: export includes full graph + metadata
- **Extensible**: custom nodes via Python
- **Built on LiteGraph**: uses typed slot system (see below)

```
[Load Image] -> [VAE Encode] -> [KSampler] -> [VAE Decode] -> [Save Image]
                    ↑
              [CLIP Text Encode] ← [Load Checkpoint]
```

Key insight: unified node graph across different media types. Workflows are shareable artifacts.

**Relevance to Resin**:
- Strong validation that node graphs work for multi-domain media
- "Portable workflows" as serializable artifacts - worth considering
- Difference: ComfyUI is UI-first, resin is library-first

### LiteGraph.js

[GitHub](https://github.com/jagenjo/litegraph.js)

JavaScript library powering ComfyUI and others:
- **Typed slots**: inputs/outputs declare type strings

```javascript
this.addInput("image", "IMAGE");
this.addOutput("latent", "LATENT");
this.addInput("mask", "MASK");
```

- **Connection validation**: only matching types can connect
- **Permissive runtime**: type checking at connection time, but data flow is dynamic
- **Namespaced nodes**: `"basic/sum"`, `"image/resize"` organization

Key insight: typed slots solve "different data types in same graph" - the graph knows what can connect to what, even if the underlying data is heterogeneous.

### Baklava.js

[Website](https://baklava.tech/) · [GitHub](https://github.com/newcat/baklavajs)

Vue-based alternative to LiteGraph with stronger typing:
- **TypeScript throughout**: compile-time + runtime type safety
- **Interface types**: formal definitions for connection points
- **Plugin system**: modular extensibility

Key insight: TypeScript enables stronger guarantees than LiteGraph's string-based types.

### maki (Baklava + generics)

[GitHub](https://github.com/pterror/maki)

Extension of Baklava with generic type inference:
- **Schema-driven types**: JSON Schema defines slot types
- **Generic parameters**: `map<T>` where T resolves at connection time
- **Connection-time inference**: connecting `string` output to `T` input -> T becomes string

```typescript
// Generic tool
tool: map<T>
  input: list<T>
  output: list<T>

// Connect list<Mesh> -> map.input
// -> T resolves to Mesh
// -> map.output becomes list<Mesh>
```

Key insight: static typing + generics gives full type safety with flexibility. Strictly stronger than Baklava alone.

**Type system spectrum:**
```
LiteGraph (strings) < Baklava (static) < maki (static + generics)
```

**Relevance to Resin**:
Typed slots solve "different data types in same graph":

| Approach | Extensible | Type-safe | Generics |
|----------|------------|-----------|----------|
| `enum SlotType` | No | Yes | No |
| Interned strings | Yes | No | No |
| `TypeId` | Yes | Yes | No |
| Schema-based | Yes | Yes | Yes |

Note: fast comparison is always solvable via interning, regardless of approach.

For plugins to define new types, avoid closed enums. Schema-based (like maki) is most powerful if we want `list<T>` style generics. Interned strings are simpler if we don't.

### Ryzome

AI-powered knowledge canvas:
- **Infinite canvas**: nodes on 2D space
- **Knowledge graph**: connections between ideas
- **Context-aware AI**: LLM with graph context

**Relevance to Resin**: Limited. It's a note-taking tool, not media generation. The canvas/graph UI pattern is common but not novel. Mentioned for completeness.

## Compute Abstraction

### Burn / CubeCL

[Burn GitHub](https://github.com/tracel-ai/burn) · [CubeCL GitHub](https://github.com/tracel-ai/cubecl)

Rust framework abstracting over compute backends:
- **Backend-agnostic**: same code runs on CPU, CUDA, wgpu, etc.
- **CubeCL**: compute language compiling to multiple targets
- **Backends**: `cubecl-cpu`, `cubecl-cuda`, `cubecl-wgpu`

```rust
// Write once, run anywhere
#[cube(launch)]
fn add_kernel(a: &Tensor<f32>, b: &Tensor<f32>, out: &mut Tensor<f32>) {
    out[ABSOLUTE_POS] = a[ABSOLUTE_POS] + b[ABSOLUTE_POS];
}
```

Key insight: abstract compute kernels over backends. No need to choose CPU vs GPU - support both.

**Relevance to Resin**:
- Texture operations (blur, noise, blend) are embarrassingly parallel
- Mesh operations less so (topology is tricky on GPU)
- Audio: typically CPU (real-time constraints, small blocks)
- Could use CubeCL for texture/noise, CPU for mesh topology

## Common Themes

1. **Small primitives, big results** - few building blocks, rich combinations
2. **Graphs over sequences** - declarative composition
3. **Parameters everywhere** - everything is tweakable
4. **Lazy/deferred evaluation** - describe, then compute
5. **Generators as timelines** - (Motion Canvas, Manim) code execution = animation
6. **Portable workflows** - (ComfyUI) graphs as shareable artifacts
7. **Domain as projection** - (MetaSynth, glitch art) structure is real, format is interpretation
