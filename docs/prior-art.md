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

Key insight: audio is naturally dataflow - sources → processors → output.

### Synths & Modular

- Oscillators, filters, envelopes as composable units
- Modulation routing (LFO → filter cutoff)
- Polyphony as instance management

Key insight: a small set of primitives (osc, filter, env, lfo) covers vast sonic territory.

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
[Load Image] → [VAE Encode] → [KSampler] → [VAE Decode] → [Save Image]
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
- **Connection-time inference**: connecting `string` output to `T` input → T becomes string

```typescript
// Generic tool
tool: map<T>
  input: list<T>
  output: list<T>

// Connect list<Mesh> → map.input
// → T resolves to Mesh
// → map.output becomes list<Mesh>
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

## Common Themes

1. **Small primitives, big results** - few building blocks, rich combinations
2. **Graphs over sequences** - declarative composition
3. **Parameters everywhere** - everything is tweakable
4. **Lazy/deferred evaluation** - describe, then compute
5. **Generators as timelines** - (Motion Canvas, Manim) code execution = animation
6. **Portable workflows** - (ComfyUI) graphs as shareable artifacts
