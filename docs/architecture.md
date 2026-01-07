# Architecture

## Crate Structure

```
resin/
├── crates/
│   ├── resin/           # umbrella, re-exports all domain crates
│   ├── resin-core/      # shared types, traits, math utilities
│   ├── resin-mesh/      # 3D mesh generation and manipulation
│   ├── resin-audio/     # audio synthesis and processing
│   ├── resin-texture/   # procedural texture generation
│   ├── resin-vector/    # 2D vector graphics
│   └── resin-rig/       # rigging and animation
└── docs/                # VitePress documentation
```

## Core Concepts

### Graphs

Operations are composed as directed acyclic graphs (DAGs):

```
Noise → Threshold → Displace
              ↓
           Sphere → Output
```

Evaluation is lazy - the graph is a description, computed on demand.

### Attributes

Data attached to geometric elements:
- Per-vertex: position, normal, UV, color
- Per-edge: crease, seam
- Per-face: material ID, smooth group

Attributes flow through operations, transformed as appropriate.

### Parameters

All operations expose parameters:
- Scalar values
- Expressions (computed at evaluation time)
- Animated values (function of time)
- Data-driven (from external sources)

## Bevy Integration

Types implement standard conversion traits:

```rust
// resin mesh → bevy mesh
let bevy_mesh: bevy::render::mesh::Mesh = resin_mesh.into();

// glam types used directly (shared with bevy)
let pos: glam::Vec3 = vertex.position;
```

No bevy dependency in core - conversions live in optional feature flags or adapter crates.
