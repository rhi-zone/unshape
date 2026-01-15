# Conventions

This document describes conventions used across the resin codebase.

## Coordinate Systems

Different domains use different coordinate conventions, matching their traditional usage:

### 2D Image/Raster (resin-image)

**Screen coordinates:**
- Origin `(0, 0)` at **top-left**
- X increases rightward
- Y increases **downward**
- UV coordinates: `(0, 0)` = top-left, `(1, 1)` = bottom-right

This matches standard image formats (PNG, JPEG) and screen rendering conventions.

### 2D Vector (resin-vector)

**Math coordinates:**
- Origin `(0, 0)` at **bottom-left** (or center, depending on context)
- X increases rightward
- Y increases **upward**
- Angles: counterclockwise from positive X-axis

This matches SVG, mathematical conventions, and most vector graphics software.

### 3D Mesh (resin-mesh)

**Right-handed, Y-up:**
- Origin `(0, 0, 0)` at center
- X increases rightward
- Y increases **upward**
- Z increases **toward viewer** (out of screen)

This matches Blender, glTF, and most 3D modeling software.

```
      +Y (up)
       |
       |
       +---- +X (right)
      /
     /
   +Z (toward viewer)
```

### Audio (resin-audio)

- Time is in **seconds** (f32)
- Sample indices are 0-based
- Frequencies in **Hz**
- Amplitudes normalized to **[-1.0, 1.0]**

## Units

Unless otherwise documented, these units are assumed:

| Domain | Unit |
|--------|------|
| 2D coordinates | Arbitrary units (often pixels or normalized [0,1]) |
| 3D coordinates | Meters (but arbitrary in practice) |
| Angles | Radians (use `to_radians()` / `to_degrees()` for conversion) |
| Time | Seconds |
| Color | Linear RGB [0,1], sRGB for display |
| Audio samples | Normalized [-1, 1] |

## Naming Conventions

### Types

- `*Config` - Configuration struct for an operation (e.g., `BlurConfig`, `StftConfig`)
- `*Error` - Error enum for a domain (e.g., `ImageFieldError`, `WavError`)
- `*Result<T>` - Type alias for `Result<T, *Error>`
- `*Id` - Identifier type (usually newtype around `u32`)

### Methods

- `new()` - Primary constructor with required parameters
- `default()` - Default configuration via `Default` trait
- `apply(&self, ...)` - Apply operation to input data
- `sample(&self, ...)` - Sample a value at a coordinate/time
- `with_*()` - Builder method (only for validation/derived values)

### Parameters

- `t` - Interpolation factor [0, 1]
- `u`, `v` - Texture/UV coordinates [0, 1]
- `x`, `y`, `z` - Spatial coordinates
- `time` - Time in seconds
- `ctx` - Evaluation context (`&EvalContext`)

## Graph Terminology

Different crates use graph-like structures for different purposes. To avoid confusion, each domain uses distinct terminology:

### Data Flow Domain (resin-core)

For node-based procedural graphs where data flows between processing units:

| Term | Meaning |
|------|---------|
| **Node** | Processing unit with typed inputs/outputs |
| **NodeId** | Identifier for a processing node |
| **Wire** | Connection from an output port to an input port |
| **Port** | Input or output slot on a node |

```rust
// Example: Wire connects port 0 of node A to port 1 of node B
let wire = Wire { from: (node_a, 0), to: (node_b, 1) };
```

### Vector Domain (resin-vector)

For 2D vector graphics networks where curves connect spatial positions:

| Term | Meaning |
|------|---------|
| **Anchor** | 2D position where curves meet (formerly "Node") |
| **AnchorId** | Identifier for an anchor point |
| **Edge** | Curve (line or bezier) connecting two anchors |
| **EdgeId** | Identifier for an edge/curve |
| **Region** | Closed area bounded by edges |

```rust
// Example: Anchor is a point with handle style
let anchor = Anchor { position: Vec2::new(10.0, 20.0), handle_style: HandleStyle::Smooth };
```

### Spatial Network Domain (resin-procgen)

For procedural road/river networks:

| Term | Meaning |
|------|---------|
| **NetworkNode** | Position in the spatial graph |
| **NetworkEdge** | Connection between positions |
| **NodeType** | Role: Source, Sink, Junction, Waypoint |

### Mesh Topology Domain (resin-mesh)

For 3D mesh topology traversal:

| Term | Meaning |
|------|---------|
| **Vertex** | 3D position with attributes |
| **VertexId** | Identifier for a vertex |
| **HalfEdge** | Directional edge (each edge has two half-edges) |
| **HalfEdgeId** | Identifier for a half-edge |
| **Face** | Polygon bounded by edges |
| **FaceId** | Identifier for a face |

### Skeletal Domain (resin-rig)

For character rigging and animation:

| Term | Meaning |
|------|---------|
| **Bone** | Joint in skeletal hierarchy |
| **BoneId** | Identifier for a bone |
| **Skeleton** | Tree of bones with parent-child relationships |

### Branching Domain (resin-space-colonization)

For procedural tree/branching structures:

| Term | Meaning |
|------|---------|
| **BranchNode** | Growth point in tree structure |
| **BranchEdge** | Parent-child connection |
