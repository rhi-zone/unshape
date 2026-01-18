# Resin

Constructive generation and manipulation of media.

## What is Resin?

Resin is a Rust library for procedural generation and manipulation of media assets:

- **3D Meshes** - geometry generation, primitives, indexed mesh representation
- **2D Vector Art** - paths, shapes, bezier curves
- **Audio** - oscillators, synthesis
- **Textures/Noise** - Perlin, Simplex, fBm, lazy field evaluation
- **Rigging** - skeletons, bones, skinning, poses

## Design Goals

- **Procedural first** - describe assets with parameters and expressions, not baked data
- **Composable** - small primitives that combine into complex results
- **Lazy evaluation** - build descriptions, evaluate on demand
- **Bevy-compatible** - works with bevy ecosystem without requiring it

## Quick Examples

### Mesh Generation

```rust
use rhizome_resin_mesh::{Cuboid, UvSphere};

// Unit box centered at origin
let cube = Cuboid::unit().apply();

// UV sphere with 32 segments, 16 rings
let ball = UvSphere::new(1.0, 32, 16).apply();
```

### Noise Fields

```rust
use rhizome_resin_core::{Field, Perlin2D, EvalContext};
use glam::Vec2;

// Lazy field - describes computation
let noise = Perlin2D::new().scale(4.0);

// Sample on demand
let ctx = EvalContext::new();
let value = noise.sample(Vec2::new(0.5, 0.5), &ctx);
```

### Audio Oscillators

```rust
use rhizome_resin_audio::{sine, saw, freq_to_phase};

let time = 0.5;  // seconds
let freq = 440.0;  // Hz

let phase = freq_to_phase(freq, time);
let sample = sine(phase);  // -1.0 to 1.0
```

### 2D Paths

```rust
use rhizome_resin_vector::{circle, rect, star, PathBuilder};
use glam::Vec2;

let c = circle(Vec2::ZERO, 1.0);
let r = rect(Vec2::new(-1.0, -1.0), Vec2::new(1.0, 1.0));
let s = star(Vec2::ZERO, 1.0, 0.5, 5);
```

### Skeletal Rigging

```rust
use rhizome_resin_rig::{Skeleton, Bone, Transform};
use glam::Vec3;

let mut skel = Skeleton::new();
let root = skel.add_bone(Bone::new("root")).id;
let arm = skel.add_bone(
    Bone::new("arm")
        .with_parent(root)
        .with_transform(Transform::from_translation(Vec3::Y))
).id;
```

## Quick Start

```toml
[dependencies]
rhizome-resin-core = "0.1"
rhizome-resin-mesh = "0.1"
rhizome-resin-audio = "0.1"
rhizome-resin-vector = "0.1"
rhizome-resin-rig = "0.1"
```

See [Getting Started](./getting-started.md) for detailed setup instructions.
