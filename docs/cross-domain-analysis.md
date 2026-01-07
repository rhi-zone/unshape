# Cross-Domain Analysis

Comparing patterns across meshes, audio, textures, 2D vector, and rigging to find unifying abstractions.

## Summary Table

| Aspect | Meshes | Audio | Textures | 2D Vector | Rigging |
|--------|--------|-------|----------|-----------|---------|
| **Primary data** | Vertices, edges, faces | Sample buffers | Pixel grids / UV→Color | Paths, segments | Bones, transforms |
| **Evaluation** | Discrete ops | Continuous stream | Lazy sample or raster | Discrete ops | Parameter-driven |
| **Time** | Static (usually) | Continuous | Static (usually) | Static/animated | Animated |
| **Attributes** | Per-vertex/edge/face | Per-sample | Per-pixel | Per-point/path | Per-bone/vertex |
| **Topology** | Edges, connectivity | N/A (1D stream) | Grid (2D) | Segments, curves | Hierarchy (tree) |
| **Booleans** | Union, diff, intersect | N/A | Blend modes | Union, diff, intersect | N/A |
| **Transforms** | Affine 3D | Gain, pan | UV transform | Affine 2D | Bone transforms |

## Common Patterns

### 1. Generator → Modifier → Output

Every domain follows this pattern:

```
Generator → Modifier → Modifier → ... → Output
```

| Domain | Generators | Modifiers |
|--------|------------|-----------|
| Mesh | Box, Sphere, Cylinder | Subdivide, Extrude, Boolean |
| Audio | Oscillators, Noise | Filters, Effects, Envelopes |
| Texture | Noise, Patterns | Blend, Transform, Blur |
| Vector | Rectangle, Ellipse, Path | Boolean, Offset, Simplify |
| Rigging | Skeleton, Pose | Constraints, Deformers |

**Implication**: A common `Node` trait with inputs/outputs could work across domains. But the *data type* flowing through differs.

### 2. Attributes / Per-Element Data

Data attached to elements:

| Domain | Elements | Attributes |
|--------|----------|------------|
| Mesh | Vertex, Edge, Face, Corner | Position, Normal, UV, Color |
| Audio | Sample | Amplitude (the data itself) |
| Texture | Pixel / UV coord | Color channels |
| Vector | Point, Segment | Position, Tangent |
| Rigging | Bone, Vertex | Transform, Weight |

**Insight**: Meshes have the most complex attribute system (multiple domains, interpolation between them). Audio is simplest (just samples). Could we have a unified `Attribute<T>` that generalizes?

### 3. Parameters and Expressions

All domains parameterize operations:

| Domain | Parameter examples | Modulation? |
|--------|-------------------|-------------|
| Mesh | Subdivision level, extrude distance | Vertex expressions (fields) |
| Audio | Frequency, cutoff, mix | Yes - audio-rate modulation |
| Texture | Scale, octaves, blend factor | UV expressions |
| Vector | Corner radius, stroke width | Limited |
| Rigging | Bone rotation, blend weight | Yes - animation, physics |

**Insight**: Audio and rigging have the strongest "parameter modulation" stories. Blender's fields bring this to meshes/textures. This could be a unifying concept:

```rust
enum ParamValue<T> {
    Constant(T),
    Expression(Box<dyn Fn(Context) → T>),
    Animated(Track<T>),
    Modulated { base: T, modulator: Signal },
}
```

### 4. Continuous vs Discrete

| Domain | Evaluation model |
|--------|-----------------|
| Mesh | Discrete operations on geometry |
| Audio | Continuous stream processing |
| Texture | Hybrid: lazy sample() or rasterize |
| Vector | Discrete operations |
| Rigging | Discrete per-frame, continuous animation |

**Tension**: Audio is fundamentally streaming (real-time, block-by-block). Others are batch operations. A unified graph system needs to handle both.

Possible approach:
- **Lazy graphs** for mesh/vector/texture: build description, evaluate on demand
- **Streaming graphs** for audio: process() called each block
- **Ticked graphs** for rigging: evaluate at each frame

### 5. Hierarchies and Composition

| Domain | Hierarchy type |
|--------|---------------|
| Mesh | Instances (same geo, different transforms) |
| Audio | Polyphony (same patch, different state) |
| Texture | Layers (blend stack) |
| Vector | Groups (scene graph) |
| Rigging | Bone tree, deformer stack |

**Common need**: Instancing / cloning with variation. "Do this N times with different parameters."

### 6. Interpolation and Blending

All domains interpolate:

| Domain | Interpolation examples |
|--------|----------------------|
| Mesh | Morph targets, LOD blending |
| Audio | Crossfade, envelope curves |
| Texture | Blend modes, gradient mapping |
| Vector | Path morphing |
| Rigging | Pose blending, animation transitions |

Could unify around interpolatable types + easing functions.

## Key Differences

### Topology

- **Mesh**: Complex (edges, faces, adjacency)
- **Audio**: None (1D stream)
- **Texture**: Grid (trivial)
- **Vector**: Paths (ordered segments)
- **Rigging**: Tree (bones)

No single topology abstraction fits all.

### State

- **Audio**: Filters have internal state (previous samples)
- **Others**: Mostly stateless operations

Audio's state requirement affects graph execution model significantly.

### Time

- **Audio**: Intrinsically time-based
- **Rigging**: Animated over time
- **Others**: Usually static

Time could be an optional input to any node rather than baked into the system.

## Proposed Abstractions

### Core Trait: Node

```rust
trait Node {
    type Input;
    type Output;
    type Context;

    fn evaluate(&mut self, input: Self::Input, ctx: &Self::Context) → Self::Output;
}
```

Problem: Input/Output types differ wildly. Mesh node takes Mesh, audio node takes Buffer.

Alternative: Generic graph over specific data types per domain.

### Domain-Specific Data + Shared Utilities

```
resin-core/
├── math (Vec2, Vec3, Quat, Transform, etc.)
├── param (ParamValue, expressions, animation)
├── interpolate (Lerp trait, easing functions)
├── noise (Perlin, Simplex - used by mesh, texture)
└── color (Color, gradient, blend modes)

resin-mesh/
├── mesh (Mesh struct, attributes)
├── generators (Box, Sphere, etc.)
└── modifiers (Subdivide, Boolean, etc.)

resin-audio/
├── buffer (Buffer, Context)
├── node (AudioNode trait)
├── generators (Oscillators)
└── processors (Filters, Effects)

... etc
```

### Unified Parameter System

Parameters are universal. Every domain has knobs.

```rust
/// A named parameter with bounds and default
struct ParamDef {
    name: String,
    min: f32,
    max: f32,
    default: f32,
}

/// A parameter value (may be constant, animated, or driven)
enum ParamSource {
    Constant(f32),
    Animated(AnimationTrack),
    Expression(Box<dyn Fn(&EvalContext) → f32>),
    Linked(ParamId),  // from another param
}
```

This matches Live2D's model, audio's modulation, Blender's drivers.

### Fields / Expressions

Blender's field system is powerful: `position.z > 0` as a selection, `noise(position) * 0.1` as displacement.

Generalized:

```rust
/// A field evaluates to T at each element
trait Field<T> {
    fn eval(&self, ctx: &FieldContext) → T;
}

struct FieldContext {
    position: Vec3,      // for mesh/texture
    index: usize,        // element index
    uv: Vec2,            // for texture
    time: f32,           // for animation
    // etc.
}
```

This could unify:
- Mesh vertex expressions
- Texture UV → color
- Audio sample computation
- Path point expressions

## Recommended Approach

1. **Don't force unification** - each domain has legitimate differences
2. **Share utilities** - math, noise, interpolation, color
3. **Unify parameters** - ParamDef, ParamSource work everywhere
4. **Consider fields** - evaluate expressions over elements (powerful but complex)
5. **Domain-specific graphs** - MeshGraph, AudioGraph, etc. rather than one Graph<T>
6. **Cross-domain bridges** - explicit conversion points (texture → mesh displacement, path → mesh extrusion)

## Summary

| Pattern | Shared? | Notes |
|---------|---------|-------|
| Generator→Modifier→Output | **Yes** | All domains follow this |
| Parameters/knobs | **Yes** | Universal - good unification target |
| Attributes (per-element data) | Partially | Mesh has complex system, audio is trivial |
| Topology | **No** | Fundamentally different (mesh adjacency vs audio stream vs bone tree) |
| Time/streaming | **No** | Audio is continuous, others are batch |
| Interpolation | **Yes** | All domains blend/lerp |

**Should unify:**
- Shared math (Vec2, Vec3, Quat, Transform via glam - bevy-compatible)
- Parameter system (ParamDef, ParamSource: constant/animated/expression)
- Noise functions (used by mesh displacement, texture generation)
- Color utilities (used by texture, vector)
- Interpolation (Lerp trait, easing functions)

**Should NOT force:**
- Single `Node` trait - data types differ too much between domains
- Unified topology - legitimate structural differences
- Single graph system - audio streaming vs batch evaluation

## Next Steps

1. Implement `resin-core` with shared types (math, color, params)
2. Pick one domain to prototype (mesh or texture - good synergy)
3. Iterate on core abstractions as second domain is added
4. Look for forced abstractions and refactor

The goal: discover the right abstractions through implementation, not top-down design.
