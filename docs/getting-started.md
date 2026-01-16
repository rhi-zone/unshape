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
| `rhizome-resin-core` | Graph container, DynNode trait, Value, EvalContext, evaluators |
| `rhizome-resin-mesh` | 3D mesh generation, primitives (box, sphere) |
| `rhizome-resin-audio` | Audio oscillators and synthesis |
| `rhizome-resin-vector` | 2D paths, shapes, bezier curves |
| `rhizome-resin-rig` | Skeleton, bones, poses, skinning |
| `rhizome-resin-macros` | Derive macros (`#[derive(DynNode)]`) |

## Using as a Library

Add to your `Cargo.toml`:

```toml
[dependencies]
# Individual crates (recommended)
rhizome-resin-core = "0.1"
rhizome-resin-mesh = "0.1"
rhizome-resin-audio = "0.1"
rhizome-resin-vector = "0.1"
rhizome-resin-rig = "0.1"
```

## Example: Creating a Node Graph

```rust
use rhizome_resin_core::{Graph, DynNodeDerive, EvalContext, Value};

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

    // Execute with default context
    let outputs = graph.execute(add).unwrap();
    println!("Result: {:?}", outputs[0]);

    // Or with custom context (for time, cancellation, etc.)
    let ctx = EvalContext::new().with_time(1.0, 60, 1.0/60.0);
    let outputs = graph.execute_with_context(add, &ctx).unwrap();
}
```

## Example: Procedural Texture with Fields

```rust
use rhizome_resin_field::{Field, EvalContext, Perlin2D};
use glam::Vec2;

fn main() {
    // Build a lazy noise field with combinators
    let noise = Perlin2D::new()
        .scale(4.0)    // Scale input coordinates
        .map(|v| v * 0.5 + 0.5);  // Remap from [-1,1] to [0,1]

    let ctx = EvalContext::new();

    // Sample a 64x64 grid
    for y in 0..64 {
        for x in 0..64 {
            let uv = Vec2::new(x as f32 / 64.0, y as f32 / 64.0);
            let value = noise.sample(uv, &ctx);
            // value is 0.0-1.0, use as brightness/height/etc
        }
    }
}
```

## Example: Simple Skeleton

```rust
use rhizome_resin_rig::{Skeleton, Bone, Pose, Transform};
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
