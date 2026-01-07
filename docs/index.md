# Resin

Constructive generation and manipulation of media.

## What is Resin?

Resin is a Rust library for procedural generation and manipulation of media assets:

- **3D Meshes** - geometry generation, subdivision, deformation
- **2D Vector Art** - procedural shapes, paths, boolean operations
- **Audio** - synthesis, effects, sequencing
- **Textures/Noise** - procedural patterns, noise functions, compositing
- **Rigging** - skeletal animation, blend shapes, constraints

## Design Goals

- **Procedural first** - describe assets with parameters and expressions, not baked data
- **Composable** - small primitives that combine into complex results
- **Bevy-compatible** - works with bevy ecosystem without requiring it

## Quick Start

```toml
[dependencies]
resin = "0.1"
```

See [Getting Started](./getting-started.md) for detailed setup instructions.
