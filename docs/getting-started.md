# Getting Started

## Prerequisites

- Rust (edition 2024)
- Nix with flakes (optional, for dev environment)

## With Nix

```bash
# Enter dev shell (auto-detected via direnv, or manually)
nix develop

# Build
cargo build

# Run tests
cargo test
```

## Without Nix

Ensure you have:
- `rustc` and `cargo` (rustup recommended)
- `mold` or `lld` for fast linking (optional)

```bash
cargo build
cargo test
```

## Crate Structure

| Crate | Description |
|-------|-------------|
| `resin-core` | Graph evaluation, noise, Field trait, EvalContext |
| `resin-mesh` | 3D mesh generation, primitives (box, sphere) |
| `resin-audio` | Audio oscillators and synthesis |
| `resin-vector` | 2D paths, shapes, bezier curves |
| `resin-rig` | Skeleton, bones, poses, skinning |
| `resin-macros` | Derive macros (`#[derive(DynNode)]`) |

## Using as a Library

Add to your `Cargo.toml`:

```toml
[dependencies]
# Individual crates (recommended)
resin-core = "0.1"
resin-mesh = "0.1"
resin-audio = "0.1"
resin-vector = "0.1"
resin-rig = "0.1"
```

## Example: Creating a Node Graph

```rust
use resin_core::{Graph, DynNodeDerive, EvalContext, Value};

// Define a custom node using the derive macro
#[derive(DynNodeDerive, Clone, Default)]
struct AddNode {
    #[input]
    a: f32,
    #[input]
    b: f32,
    #[output]
    result: f32,
}

impl AddNode {
    fn compute(&mut self, _ctx: &EvalContext) {
        self.result = self.a + self.b;
    }
}

fn main() {
    let mut graph = Graph::new();
    let add = graph.add_node(AddNode::default());

    let ctx = EvalContext::new();
    // Execute with explicit inputs
    let outputs = graph.execute(add, &ctx).unwrap();
    println!("Result: {:?}", outputs[0]);
}
```

## Example: Procedural Texture with Fields

```rust
use resin_core::{Field, Perlin2D, Fbm2D, EvalContext};
use glam::Vec2;

fn main() {
    // Build a lazy noise field
    let base = Perlin2D::new();
    let fbm = Fbm2D::new(base).octaves(4).lacunarity(2.0);

    let ctx = EvalContext::new();

    // Sample a 64x64 grid
    for y in 0..64 {
        for x in 0..64 {
            let uv = Vec2::new(x as f32 / 64.0, y as f32 / 64.0);
            let value = fbm.sample(uv * 4.0, &ctx);
            // value is 0.0-1.0, use as brightness/height/etc
        }
    }
}
```

## Example: Simple Skeleton

```rust
use resin_rig::{Skeleton, Bone, Pose, Transform};
use glam::{Vec3, Quat};
use std::f32::consts::FRAC_PI_4;

fn main() {
    let mut skel = Skeleton::new();

    // Build hierarchy
    let hip = skel.add_bone(Bone::new("hip")).id;
    let spine = skel.add_bone(
        Bone::new("spine")
            .with_parent(hip)
            .with_transform(Transform::from_translation(Vec3::new(0.0, 1.0, 0.0)))
    ).id;
    let head = skel.add_bone(
        Bone::new("head")
            .with_parent(spine)
            .with_transform(Transform::from_translation(Vec3::new(0.0, 0.5, 0.0)))
    ).id;

    // Create a pose (animated state)
    let mut pose = skel.rest_pose();
    pose.set(spine, Transform::from_rotation(Quat::from_rotation_z(FRAC_PI_4)));

    // Get world position of head
    let head_world = pose.world_transform(&skel, head);
    println!("Head position: {:?}", head_world.translation);
}
```

## Documentation

```bash
cd docs
bun install
bun run dev
```

Opens at `http://localhost:5173/resin/`
