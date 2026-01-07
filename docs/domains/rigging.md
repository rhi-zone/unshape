# Rigging

Skeletal animation, deformation, and motion systems.

## Prior Art

### Blender Armatures
- **Bones**: head, tail, roll; parent-child hierarchy
- **Pose mode**: transform bones, store as keyframes
- **Constraints**: IK, copy rotation, track-to, etc.
- **Drivers**: expressions that control properties
- **Skinning**: vertex groups with weights

### Maya / 3ds Max
- **Joint hierarchies**: similar to Blender bones
- **IK solvers**: various algorithms (SC, RP, Spline)
- **Blend shapes**: morph targets for facial animation
- **Deformers**: lattice, wrap, cluster, nonlinear (bend, twist, etc.)

### Live2D Cubism
- **Art mesh**: triangulated 2D image regions
- **Deformers**: warp (grid), rotation
- **Parameters**: named floats that drive deformers
- **Physics**: spring simulation on parameters
- **Parts**: visibility groups

### Spine (2D)
- **Bones**: 2D skeleton
- **Meshes**: deformable image regions
- **Slots/attachments**: swap images per bone
- **Weights**: mesh vertices influenced by bones
- **IK constraints**: 2D inverse kinematics
- **Path constraints**: bones follow Bézier paths

### Animation Principles
- **Forward kinematics (FK)**: parent → child transforms
- **Inverse kinematics (IK)**: position end effector, solve chain
- **Blend shapes / morph targets**: interpolate between poses
- **Skinning**: vertices follow weighted bone influences

## Core Types

```rust
/// A skeleton
struct Skeleton {
    bones: Vec<Bone>,
    root: BoneId,
}

struct Bone {
    name: String,
    parent: Option<BoneId>,
    // Rest pose (bind pose)
    local_transform: Transform,
    length: f32,
}

/// A pose (bone transforms relative to rest)
struct Pose {
    bone_transforms: Vec<Transform>,  // indexed by BoneId
}

/// Transform (position, rotation, scale)
struct Transform {
    translation: Vec3,
    rotation: Quat,
    scale: Vec3,
}

/// Skinning weights for a mesh
struct Skin {
    // Per-vertex: which bones and how much
    influences: Vec<VertexInfluences>,
}

struct VertexInfluences {
    bones: [BoneId; 4],   // typically max 4
    weights: [f32; 4],    // sum to 1.0
}

/// Morph target (blend shape)
struct MorphTarget {
    name: String,
    // Per-vertex deltas from base
    deltas: Vec<Vec3>,
}
```

## Constraints

Constraints modify bone transforms based on rules:

| Constraint | Description |
|------------|-------------|
| CopyTransform | Copy position/rotation/scale from another bone |
| CopyRotation | Copy only rotation |
| TrackTo | Point axis at target |
| IK | Inverse kinematics chain |
| LookAt | Orient to face target |
| Limit | Clamp rotation/position/scale |
| Damped Track | Smooth tracking |
| Spline IK | Follow curve |

```rust
trait Constraint {
    fn evaluate(&self, skeleton: &Skeleton, pose: &mut Pose, targets: &Targets);
}
```

Constraints evaluated in order (stack).

## Inverse Kinematics

Given: end effector target position
Find: bone rotations that reach it

```rust
struct IKChain {
    bones: Vec<BoneId>,   // from root to tip
    target: Vec3,
    pole: Option<Vec3>,   // bend direction hint
    iterations: u32,
}
```

Algorithms:
- **CCD** (Cyclic Coordinate Descent): simple, iterative
- **FABRIK**: fast, position-based
- **Jacobian**: physically accurate, slower
- **Analytical**: closed-form for 2-bone

## Deformers (Non-skeletal)

| Deformer | Description |
|----------|-------------|
| Lattice | FFD - warp points in a control cage |
| Bend | Bend along axis |
| Twist | Rotate along axis |
| Taper | Scale along axis |
| Wave | Sinusoidal displacement |
| Shrinkwrap | Project onto target surface |

Deformers can be stacked. Applied after skinning or as alternative.

## 2D Rigging Specifics

Live2D / Spine style:

```rust
/// 2D mesh region
struct ArtMesh {
    texture_region: Rect,  // UV bounds in atlas
    vertices: Vec<Vec2>,   // in texture space
    indices: Vec<u32>,     // triangles
}

/// Warp deformer (grid)
struct WarpDeformer {
    rows: u32,
    cols: u32,
    control_points: Vec<Vec2>,  // rows × cols grid
}

/// Parameter (named control value)
struct Parameter {
    name: String,
    value: f32,
    min: f32,
    max: f32,
    default: f32,
}
```

Parameters drive deformers. Multiple parameters can affect same deformer (blended).

## Animation

```rust
/// Animation clip
struct AnimationClip {
    duration: f32,
    tracks: Vec<Track>,
}

/// Single animated property
struct Track {
    target: PropertyPath,  // e.g., "bones/arm/rotation"
    keyframes: Vec<Keyframe>,
}

struct Keyframe {
    time: f32,
    value: Value,
    interpolation: Interpolation,
}

enum Interpolation {
    Step,
    Linear,
    Bezier { in_tangent: Vec2, out_tangent: Vec2 },
}
```

## Data Flow Pattern

```
Skeleton (rest) + Animation → Pose
Pose + Constraints → Final Pose
Mesh + Skin + Final Pose → Deformed Mesh
```

Or with deformers:

```
Mesh → Deformer → Deformer → ... → Output
              ↑
         parameters
```

**Key insight**: rigging is about **mapping parameters to deformation**. Whether that's bones, blend shapes, or lattices.

## Physics

Spring/pendulum simulation on rig parameters:

```rust
struct PhysicsParam {
    target: PropertyPath,
    mass: f32,
    damping: f32,
    stiffness: f32,
}
```

Live2D uses this for hair, clothing physics. Parameters have inertia.

## Relationship to Other Domains

- **Meshes**: rigging deforms mesh vertices
- **2D Vector**: paths can be rigged (deform control points)
- **Audio**: animation timing, lip sync (parameter mapping)

## Open Questions

1. **Unified rig**: Can 2D and 3D rigging share abstractions? Bones work in both. Constraints work in both. Skinning works in both.

2. **Parameter system**: Live2D's parameter-centric model is elegant. Should all domains have "parameters" as first-class? (Audio already does via modulation.)

3. **Deformer stacking**: Order matters. How to represent? List? Graph?

4. **Procedural rigging**: Auto-rig from mesh topology? Useful for generated meshes.

5. **Animation blending**: Blend trees, state machines, layers. Include in scope or separate crate?

6. **Real-time vs offline**: Different constraints. Games need <16ms. Film can take hours per frame.
