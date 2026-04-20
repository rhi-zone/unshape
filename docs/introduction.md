# Unshape

A substrate for operations on arbitrary media.

## What is Unshape?

Unshape is a Rust toolkit for any transformation between media that can be expressed as a parameterized, serializable operation. The medium types — meshes, images, audio, vector art, fields, and more — are open-ended: adding a new one means defining a representation and a set of ops. The graph, serialization, signal routing, and lazy/realtime evaluation come for free.

The one constraint on an op: **finite, named parameters**. If a transformation has a finite parameter set, it is serializable, replayable, composable in a graph, and inspectable at runtime. Stochastic ops qualify with a seed. Stateful ops (filters, simulations) qualify if their state is checkpointable.

This means Unshape is equally at home as:
- An archival/reproducible project file format (graph as `.psd`, fully rewindable)
- A live signal-driven experience (any input port can be driven by time, audio, sensors, or another graph's output)
- A scripting substrate for procedural content pipelines

Current media domains: 3D meshes, 2D vector art, audio, textures/noise, rigging.

## Design Goals

- **Operations as values** - every transformation is a serializable struct with named parameters; methods are sugar
- **Medium-agnostic graph** - the node graph, serialization, and signal machinery are not tied to any specific domain
- **Lazy evaluation** - build descriptions, evaluate on demand; no hidden materializations
- **Realtime first** - signal-driven use is not an afterthought; any input can be a live signal
- **Bevy-compatible** - works with the bevy ecosystem without requiring it

## Quick Examples

### Mesh Generation

```rust
use rhi_unshape_mesh::{Cuboid, UvSphere};

// Unit box centered at origin
let cube = Cuboid::unit().apply();

// UV sphere with 32 segments, 16 rings
let ball = UvSphere::new(1.0, 32, 16).apply();
```

### Noise Fields

```rust
use rhi_unshape_core::{Field, Perlin2D, EvalContext};
use glam::Vec2;

// Lazy field - describes computation
let noise = Perlin2D::new().scale(4.0);

// Sample on demand
let ctx = EvalContext::new();
let value = noise.sample(Vec2::new(0.5, 0.5), &ctx);
```

### Audio Oscillators

```rust
use rhi_unshape_audio::{sine, saw, freq_to_phase};

let time = 0.5;  // seconds
let freq = 440.0;  // Hz

let phase = freq_to_phase(freq, time);
let sample = sine(phase);  // -1.0 to 1.0
```

### 2D Paths

```rust
use rhi_unshape_vector::{circle, rect, star, PathBuilder};
use glam::Vec2;

let c = circle(Vec2::ZERO, 1.0);
let r = rect(Vec2::new(-1.0, -1.0), Vec2::new(1.0, 1.0));
let s = star(Vec2::ZERO, 1.0, 0.5, 5);
```

### Skeletal Rigging

```rust
use rhi_unshape_rig::{Skeleton, Bone, Transform};
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
rhi-unshape-core = "0.1"
rhi-unshape-mesh = "0.1"
rhi-unshape-audio = "0.1"
rhi-unshape-vector = "0.1"
rhi-unshape-rig = "0.1"
```

See [Getting Started](./getting-started.md) for detailed setup instructions.
