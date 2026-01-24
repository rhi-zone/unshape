# Domain Differences

Why certain abstractions can't (and shouldn't) be unified across domains.

## 1. Topology

Each domain has fundamentally different structural relationships.

### Mesh: Adjacency Graph

```rust
// Mesh topology is about connectivity
struct Mesh {
    vertices: Vec<Vertex>,
    half_edges: Vec<HalfEdge>,
    faces: Vec<Face>,
}

// Key operations rely on adjacency
fn get_adjacent_faces(mesh: &Mesh, face: FaceId) -> Vec<FaceId> {
    // Walk around face edges, get twin edges, get their faces
    let mut adjacent = vec![];
    for edge in mesh.face_edges(face) {
        if let Some(twin_face) = mesh.half_edges[edge.twin].face {
            adjacent.push(twin_face);
        }
    }
    adjacent
}

// Subdivision needs neighbors
fn subdivide_catmull_clark(mesh: &Mesh) -> Mesh {
    // For each face: new face point = average of face vertices
    // For each edge: new edge point = average of edge verts + adjacent face points
    // For each vertex: new position = (Q + 2R + (n-3)P) / n
    //   where Q = avg of adjacent face points
    //         R = avg of adjacent edge midpoints
    //         n = valence (number of adjacent edges)
    // ...
}
```

**Why it's special**: Operations need to know "what's next to what". Edge loops, face neighbors, vertex valence.

### Audio: 1D Stream (No Topology)

```rust
// Audio has no topology - just samples in sequence
struct Buffer {
    samples: Vec<f32>,  // That's it. No adjacency.
}

// Operations work sample-by-sample or block-by-block
fn lowpass_filter(input: &Buffer, output: &mut Buffer, cutoff: f32, state: &mut FilterState) {
    for i in 0..input.samples.len() {
        // Each sample depends only on previous samples (state), not neighbors
        let x = input.samples[i];
        let y = state.a0 * x + state.a1 * state.x1 + state.a2 * state.x2
                           - state.b1 * state.y1 - state.b2 * state.y2;
        output.samples[i] = y;
        state.update(x, y);
    }
}
```

**Why it's special**: No spatial relationships. Time is the only dimension, and it's strictly sequential.

### Texture: Regular Grid

```rust
// Texture is a 2D grid - trivial topology
struct Texture {
    width: u32,
    height: u32,
    pixels: Vec<Color>,  // row-major
}

fn get_pixel(tex: &Texture, x: u32, y: u32) -> Color {
    tex.pixels[(y * tex.width + x) as usize]
}

// Neighbors are trivial arithmetic
fn get_neighbors_3x3(tex: &Texture, x: u32, y: u32) -> [Color; 9] {
    // Just x±1, y±1 with bounds checking
}

// Blur needs neighbors but they're always at fixed offsets
fn gaussian_blur(input: &Texture, radius: u32) -> Texture {
    // For each pixel, sample in a grid pattern around it
    // No complex adjacency queries needed
}
```

**Why it's special**: Topology is implicit in the grid. No need for explicit edge/face structures.

### Vector: Ordered Segments

```rust
// Path is an ordered sequence of segments
struct Path {
    segments: Vec<Segment>,
    closed: bool,
}

// "Adjacency" is just index ± 1
fn next_segment(path: &Path, idx: usize) -> Option<&Segment> {
    if idx + 1 < path.segments.len() {
        Some(&path.segments[idx + 1])
    } else if path.closed {
        Some(&path.segments[0])
    } else {
        None
    }
}
```

**Why it's special**: Strictly linear. No branching (unlike mesh edges). Order matters.

### Rigging: Tree Hierarchy

```rust
// Skeleton is a tree
struct Skeleton {
    bones: Vec<Bone>,
    root: BoneId,
}

struct Bone {
    parent: Option<BoneId>,
    children: Vec<BoneId>,  // derived from parent links
    local_transform: Transform,
}

// Key operation: accumulate transforms down the tree
fn compute_world_transforms(skeleton: &Skeleton) -> Vec<Transform> {
    let mut world = vec![Transform::IDENTITY; skeleton.bones.len()];

    fn visit(skeleton: &Skeleton, bone: BoneId, parent_world: Transform, world: &mut [Transform]) {
        let local = skeleton.bones[bone].local_transform;
        world[bone] = parent_world * local;
        for &child in &skeleton.bones[bone].children {
            visit(skeleton, child, world[bone], world);
        }
    }

    visit(skeleton, skeleton.root, Transform::IDENTITY, &mut world);
    world
}
```

**Why it's special**: Hierarchical. Parent transforms affect all descendants. Not a general graph - strictly a tree.

### Summary: Topology

| Domain | Structure | Adjacency |
|--------|-----------|-----------|
| Mesh | Graph (V, E, F) | Complex: edge twins, face loops |
| Audio | Array | None (sequential only) |
| Texture | 2D Grid | Trivial: x±1, y±1 |
| Vector | Ordered list | Linear: prev/next |
| Rigging | Tree | Parent/children |

**No common abstraction makes sense.** A trait like `Topology { fn neighbors(&self, id: Id) -> Vec<Id> }` would be:
- Meaningless for audio (no neighbors)
- Wasteful for texture (just do arithmetic)
- Wrong for rigging (children ≠ neighbors)

---

## 2. Evaluation Model: Streaming vs Batch

### Batch Evaluation (Mesh, Vector, Texture)

```rust
// Mesh: apply operation, get new mesh
fn subdivide(input: Mesh) -> Mesh {
    // Process entire mesh at once
    // Allocate output, compute, return
    let mut output = Mesh::new();
    // ... compute all vertices, edges, faces ...
    output
}

// Compose operations
let result = extrude(subdivide(boolean_union(mesh_a, mesh_b)));

// Or as a lazy graph
let graph = Graph::new()
    .add(BooleanUnion::new(mesh_a, mesh_b))
    .add(Subdivide::new())
    .add(Extrude::new(1.0));
let result = graph.evaluate();
```

**Characteristics:**
- Input -> Output (functional)
- No time dimension
- Can evaluate lazily or eagerly
- No internal state (usually)

### Streaming Evaluation (Audio)

```rust
// Audio: process block by block, forever
trait AudioNode {
    fn process(&mut self, ctx: &AudioContext, input: &[Buffer], output: &mut [Buffer]);
}

// Called repeatedly by audio thread
loop {
    let now = Instant::now();
    oscillator.process(&ctx, &[], &mut [&mut osc_buf]);
    filter.process(&ctx, &[&osc_buf], &mut [&mut filter_buf]);
    output.process(&ctx, &[&filter_buf], &mut [&mut out_buf]);

    audio_device.write(&out_buf);

    // Must complete before deadline or audio glitches
    assert!(now.elapsed() < BLOCK_DURATION);
}
```

**Characteristics:**
- Continuous, never "done"
- Real-time constraints (can't take too long)
- Stateful (filters remember previous samples)
- Pull-based (audio callback requests samples)

### Why They Can't Unify

```rust
// Hypothetical unified trait - doesn't work well

trait Node {
    type Input;
    type Output;
    fn evaluate(&mut self, input: Self::Input) -> Self::Output;
}

// Mesh version - fine
impl Node for Subdivide {
    type Input = Mesh;
    type Output = Mesh;
    fn evaluate(&mut self, input: Mesh) -> Mesh { ... }
}

// Audio version - awkward
impl Node for LowPassFilter {
    type Input = Buffer;    // Just one block, not the stream
    type Output = Buffer;
    fn evaluate(&mut self, input: Buffer) -> Buffer {
        // Need to maintain state between calls
        // Who calls this? How often?
        // Real-time constraints not expressible in trait
    }
}
```

The trait works for one but not the other:
- Mesh: caller decides when to evaluate
- Audio: audio thread dictates timing, caller must keep up

### Rigging: Hybrid

```rust
// Rigging evaluates per-frame (like batch) but over time (like streaming)
fn animate(skeleton: &Skeleton, clip: &AnimationClip, time: f32) -> Pose {
    // Sample animation at time t
    let pose = clip.sample(time);

    // Apply constraints (may iterate)
    let mut final_pose = pose;
    for constraint in &skeleton.constraints {
        constraint.apply(&mut final_pose);
    }

    final_pose
}

// Called each frame
loop {
    let time = elapsed.as_secs_f32();
    let pose = animate(&skeleton, &walk_clip, time);
    let skinned_mesh = skin(&mesh, &skeleton, &pose);
    render(&skinned_mesh);
}
```

**Characteristics:**
- Evaluated at discrete times (frames)
- Time is explicit input, not implicit stream
- Stateless evaluation (state is in time parameter)
- Can skip frames, scrub backwards

---

## 3. Why Not One Node Trait?

Attempting a unified node:

```rust
trait Node {
    type Data;
    fn process(&mut self, inputs: &[&Self::Data], outputs: &mut [Self::Data], ctx: &Context);
}
```

### Problem 1: Data Types

```rust
// Mesh node
impl Node for Subdivide {
    type Data = Mesh;  // A single mesh
}

// Audio node
impl Node for Filter {
    type Data = Buffer;  // A block of samples
}

// Texture node
impl Node for Blur {
    type Data = Texture;  // A pixel grid
}
```

These can't be mixed in one graph. A `Graph<Mesh>` can't include audio nodes.

### Problem 2: Input/Output Counts

```rust
// Boolean needs 2 mesh inputs, 1 output
fn boolean_union(a: Mesh, b: Mesh) -> Mesh;

// Oscillator needs 0 inputs, 1 output
fn oscillator() -> Buffer;

// Delay needs 1 input, 2 outputs (dry + wet)
fn delay(input: Buffer) -> (Buffer, Buffer);
```

Fixed `&[&Data]` doesn't capture these constraints well.

### Problem 3: Context Differences

```rust
struct MeshContext {
    // ... nothing really needed
}

struct AudioContext {
    sample_rate: f32,
    block_size: usize,
    time_in_samples: u64,  // Crucial for audio
}

struct TextureContext {
    target_width: u32,
    target_height: u32,  // Needed for rasterization
}
```

Each domain needs different context. Cramming into one struct means every node sees irrelevant fields.

### Better: Domain-Specific Traits

```rust
// Mesh operations
trait MeshOp {
    fn apply(&self, input: &Mesh) -> Mesh;
}

// Audio processing
trait AudioProcessor {
    fn process(&mut self, ctx: &AudioContext, inputs: &[&Buffer], outputs: &mut [Buffer]);
    fn input_count(&self) -> usize;
    fn output_count(&self) -> usize;
}

// Texture generation
trait TextureGenerator {
    fn sample(&self, uv: Vec2) -> Color;
    // or
    fn rasterize(&self, width: u32, height: u32) -> Texture;
}
```

Each trait captures what that domain actually needs.

---

## 4. What CAN Be Shared

Despite the above, some things genuinely work across domains:

### Parameters

```rust
// Works for all domains
struct Param {
    name: String,
    value: f32,
    min: f32,
    max: f32,
}

enum ParamSource {
    Constant(f32),
    Animated(AnimationCurve),
    Expression(Box<dyn Fn(&EvalContext) -> f32>),
}

// Mesh: subdivision_levels: Param
// Audio: filter_cutoff: Param (can be modulated at audio rate)
// Texture: noise_scale: Param
// Vector: corner_radius: Param
// Rigging: blend_weight: Param (animated)
```

### Interpolation

```rust
trait Lerp {
    fn lerp(a: Self, b: Self, t: f32) -> Self;
}

impl Lerp for f32 { ... }
impl Lerp for Vec3 { ... }
impl Lerp for Color { ... }
impl Lerp for Transform { ... }

// Used everywhere:
// - Mesh: morph target blending
// - Audio: crossfade
// - Texture: gradient
// - Vector: path morphing
// - Rigging: pose blending
```

### Noise

```rust
trait NoiseFunction {
    fn sample(&self, pos: Vec3) -> f32;
}

struct Perlin { ... }
struct Simplex { ... }

// Used by:
// - Mesh: vertex displacement
// - Texture: procedural patterns
// - Audio: noise oscillator (1D slice)
// - Rigging: motion noise / jitter
```

### Math Types

```rust
// glam types used everywhere
use glam::{Vec2, Vec3, Vec4, Quat, Mat4};

// Transforms
struct Transform {
    translation: Vec3,
    rotation: Quat,
    scale: Vec3,
}
```

---

## Conclusion

The domains have genuine structural differences that make forced unification counterproductive:

1. **Topology** - each domain has different neighbor relationships
2. **Evaluation** - streaming (audio) vs batch (mesh) vs ticked (rigging)
3. **Node signatures** - different I/O patterns and context needs

But they share:
1. **Parameters** - everything has knobs
2. **Interpolation** - everything blends
3. **Noise** - several domains use it
4. **Math types** - universal

Design accordingly: shared utilities in `rhi-unshape-core`, domain-specific abstractions in domain crates.
