# Resin

Constructive generation and manipulation of media in Rust.

## Features

- **Meshes** - Procedural 3D geometry, primitives, indexed mesh representation
- **Audio** - Synthesis oscillators (sine, saw, square, triangle) with anti-aliasing
- **Textures & Noise** - Perlin, Simplex, fBm with lazy `Field` trait and combinators
- **2D Vector** - SVG-like paths, bezier curves, shapes
- **Rigging** - Skeletons, bones, poses, GPU-friendly skinning
- **Node Graphs** - Dynamic evaluation with type-safe connections

## Quick Start

```toml
[dependencies]
resin-core = "0.1"
resin-mesh = "0.1"
resin-audio = "0.1"
resin-vector = "0.1"
resin-rig = "0.1"
```

```rust
use resin_mesh::{box_mesh, sphere};
use resin_core::{Field, Perlin2D, EvalContext};
use glam::Vec2;

// Mesh primitives
let cube = box_mesh();
let ball = sphere();

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
