# Resin

Constructive generation and manipulation of media in Rust.

## Features

- **Meshes** - Procedural 3D geometry, booleans, decimation, LOD, topology analysis, UV atlas packing
- **Audio** - Synthesis, FM/wavetable/granular, effects, 3D spatial audio (HRTF), pattern sequencing
- **Textures & Noise** - Perlin, Simplex, fBm with lazy `Field` trait, 2D/3D signed distance fields
- **2D Vector** - SVG-like paths, bezier curves, booleans, hatching, rasterization
- **Rigging** - Skeletons, IK solvers, weight painting tools, heat diffusion skinning
- **Physics** - Rigid bodies, soft bodies, cloth, fluids, smoke simulation
- **Procedural** - L-systems, WFC, mazes, terrain erosion, space colonization
- **Node Graphs** - Dynamic evaluation with type-safe connections

## Quick Start

```toml
[dependencies]
rhizome-resin-core = "0.1"
rhizome-resin-mesh = "0.1"
rhizome-resin-audio = "0.1"
rhizome-resin-vector = "0.1"
rhizome-resin-rig = "0.1"
```

```rust
use rhizome_resin_mesh::{Cuboid, Icosphere};
use rhizome_resin_core::{Field, Perlin2D, EvalContext};
use glam::Vec2;

// Mesh primitives
let cube = Cuboid::unit().apply();
let ball = Icosphere::new(1.0, 3).apply();

// Lazy noise field
let noise = Perlin2D::new().scale(4.0);
let ctx = EvalContext::new();
let value = noise.sample(Vec2::new(0.5, 0.5), &ctx);
```

## Documentation

- [Online Docs](https://rhizome-lab.github.io/resin/)
- Local: `cd docs && bun install && bun run dev`

## Building

```bash
cargo build
cargo test
```

Or with Nix:

```bash
nix develop
cargo build
```

## License

MIT
