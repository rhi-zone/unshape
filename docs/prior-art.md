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

## Common Themes

1. **Small primitives, big results** - few building blocks, rich combinations
2. **Graphs over sequences** - declarative composition
3. **Parameters everywhere** - everything is tweakable
4. **Lazy/deferred evaluation** - describe, then compute
