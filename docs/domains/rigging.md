# Rigging

Skeletal animation, deformation, IK, procedural locomotion, and secondary motion.

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

### Game Animation Systems
- **Motion matching**: database of poses, find best match
- **Procedural animation**: IK-driven walk cycles, secondary motion
- **Animation layers**: blend multiple animations

## Module Structure

| Module | Purpose |
|--------|---------|
| `skeleton` | Bones, skeleton hierarchy, poses |
| `skin` | Mesh skinning with bone weights |
| `animation` | Animation clips, keyframes, playback |
| `blend` | Animation blending, layers, crossfades |
| `ik` | Inverse kinematics (CCD, FABRIK) |
| `constraint` | Bone constraints, path following |
| `locomotion` | Procedural walk cycles |
| `motion_matching` | Motion matching system |
| `secondary` | Jiggle physics, follow-through |

## Skeleton and Bones

### Creating a Skeleton

```rust
use rhi_unshape_rig::{Skeleton, Bone, BoneId, Transform};
use glam::{Vec3, Quat};

let mut skeleton = Skeleton::new();

// Add root bone
let root = skeleton.add_bone(Bone {
    name: "root".into(),
    parent: None,
    local_transform: Transform::IDENTITY,
    length: 0.0,
});

// Add spine
let spine = skeleton.add_bone(Bone {
    name: "spine".into(),
    parent: Some(root),
    local_transform: Transform::from_translation(Vec3::new(0.0, 1.0, 0.0)),
    length: 0.5,
});

// Add arm
let upper_arm = skeleton.add_bone(Bone {
    name: "upper_arm".into(),
    parent: Some(spine),
    local_transform: Transform::from_translation(Vec3::new(0.3, 0.0, 0.0)),
    length: 0.3,
});

let lower_arm = skeleton.add_bone(Bone {
    name: "lower_arm".into(),
    parent: Some(upper_arm),
    local_transform: Transform::from_translation(Vec3::new(0.3, 0.0, 0.0)),
    length: 0.3,
});
```

### Poses

```rust
use rhi_unshape_rig::Pose;

// Create pose from skeleton (rest pose)
let mut pose = Pose::from_skeleton(&skeleton);

// Modify bone transforms
pose.set_local_rotation(upper_arm, Quat::from_rotation_z(0.5));
pose.set_local_translation(root, Vec3::new(0.0, 0.1, 0.0));

// Get world-space transforms
let world_transform = pose.world_transform(&skeleton, upper_arm);
let world_position = pose.world_position(&skeleton, upper_arm);
```

## Mesh Skinning

```rust
use rhi_unshape_rig::{Skin, VertexInfluences, MAX_INFLUENCES};

// Create skin for mesh
let mut skin = Skin::new(vertex_count);

// Set influences for each vertex
skin.set_influences(vertex_index, VertexInfluences {
    bones: [bone_a, bone_b, BoneId::INVALID, BoneId::INVALID],
    weights: [0.7, 0.3, 0.0, 0.0],
});

// Apply skinning to mesh
let deformed_positions = skin.deform_positions(
    &original_positions,
    &skeleton,
    &pose,
);
```

## Animation

### Animation Clips

```rust
use rhi_unshape_rig::{AnimationClip, Track, Keyframe, Interpolation, AnimationTarget};

let mut clip = AnimationClip::new("walk", duration: 1.0);

// Add rotation track for a bone
clip.add_track(Track::new(
    AnimationTarget::BoneRotation(upper_arm),
    vec![
        Keyframe::new(0.0, Quat::IDENTITY, Interpolation::Linear),
        Keyframe::new(0.5, Quat::from_rotation_z(0.5), Interpolation::Linear),
        Keyframe::new(1.0, Quat::IDENTITY, Interpolation::Linear),
    ],
));

// Add position track
clip.add_track(Track::new(
    AnimationTarget::BonePosition(root),
    vec![
        Keyframe::new(0.0, Vec3::ZERO, Interpolation::Linear),
        Keyframe::new(1.0, Vec3::new(0.0, 0.0, 1.0), Interpolation::Linear),
    ],
));
```

### Animation Playback

```rust
use rhi_unshape_rig::AnimationPlayer;

let mut player = AnimationPlayer::new(&clip);

// Update
player.update(delta_time);

// Get current time
let time = player.time();

// Sample pose at current time
clip.sample(player.time(), &mut pose);

// Control playback
player.play();
player.pause();
player.set_time(0.5);
player.set_speed(2.0);
player.set_looping(true);
```

## Animation Blending

### Crossfade

```rust
use rhi_unshape_rig::Crossfade;

let mut crossfade = Crossfade::new(clip_a, clip_b, duration: 0.3);

crossfade.update(delta_time);
crossfade.sample(&mut pose);
```

### Animation Layers

```rust
use rhi_unshape_rig::{AnimationStack, AnimationLayer, BlendMode};

let mut stack = AnimationStack::new();

// Base layer (full body)
stack.add_layer(AnimationLayer {
    clip: walk_clip,
    weight: 1.0,
    blend_mode: BlendMode::Override,
    mask: None,  // All bones
});

// Additive layer (head look)
stack.add_layer(AnimationLayer {
    clip: look_clip,
    weight: 0.5,
    blend_mode: BlendMode::Additive,
    mask: Some(head_bone_mask),  // Only head bones
});

// Sample combined result
stack.sample(time, &mut pose);
```

### Blend Trees

```rust
use rhi_unshape_rig::BlendNode;

// 1D blend (walk speed)
let blend = BlendNode::Blend1D {
    clips: vec![idle_clip, walk_clip, run_clip],
    thresholds: vec![0.0, 0.5, 1.0],
    parameter: speed,
};

// 2D blend (movement direction)
let blend = BlendNode::Blend2D {
    clips: vec![
        (idle, Vec2::ZERO),
        (walk_forward, Vec2::new(0.0, 1.0)),
        (walk_back, Vec2::new(0.0, -1.0)),
        (walk_left, Vec2::new(-1.0, 0.0)),
        (walk_right, Vec2::new(1.0, 0.0)),
    ],
    parameter: movement_direction,
};

blend.sample(time, &mut pose);
```

## Inverse Kinematics

### CCD (Cyclic Coordinate Descent)

```rust
use rhi_unshape_rig::{IkChain, IkConfig, solve_ccd};

let chain = IkChain {
    bones: vec![upper_arm, lower_arm, hand],
    target: Vec3::new(1.0, 0.5, 0.0),
    pole: Some(Vec3::new(0.0, 0.0, -1.0)),  // Elbow direction hint
};

let config = IkConfig {
    iterations: 10,
    tolerance: 0.001,
};

let result = solve_ccd(&skeleton, &mut pose, &chain, &config);
```

### FABRIK

```rust
use rhi_unshape_rig::solve_fabrik;

let result = solve_fabrik(&skeleton, &mut pose, &chain, &config);

// Check result
if result.reached {
    println!("Target reached!");
} else {
    println!("Distance to target: {}", result.distance);
}
```

## Constraints

### Path Constraint

```rust
use rhi_unshape_rig::{PathConstraint, Path3D, ConstraintStack};

// Create path
let path = Path3D::from_points(&[
    Vec3::new(0.0, 0.0, 0.0),
    Vec3::new(1.0, 0.5, 0.0),
    Vec3::new(2.0, 0.0, 0.0),
]);

// Bone follows path
let constraint = PathConstraint {
    bone: spine,
    path: path,
    position: 0.5,  // 0.0-1.0 along path
    follow_rotation: true,
};

// Apply constraints
let mut stack = ConstraintStack::new();
stack.add(constraint);
stack.evaluate(&skeleton, &mut pose);
```

## Procedural Locomotion

### Walk Cycle Generator

```rust
use rhi_unshape_rig::{ProceduralWalk, GaitConfig, GaitPattern, WalkAnimator};

let config = GaitConfig {
    stride_length: 0.8,
    stride_height: 0.15,
    speed: 1.0,
    body_bob: 0.02,
    body_sway: 0.01,
};

let gait = GaitPattern::Walk;  // or Run, Trot, etc.

let mut walk = ProceduralWalk::new(&skeleton, config, gait);

// Configure legs
walk.add_leg(hip_l, knee_l, foot_l);
walk.add_leg(hip_r, knee_r, foot_r);

// Update
walk.update(delta_time, velocity);
walk.apply_to_pose(&mut pose);
```

### Foot Placement

```rust
use rhi_unshape_rig::FootPlacement;

let mut placement = FootPlacement::new(&skeleton);
placement.add_foot(foot_l, leg_chain_l);
placement.add_foot(foot_r, leg_chain_r);

// Adjust feet to terrain
placement.update(
    &mut pose,
    |position| terrain.height_at(position),  // Height query
    |position| terrain.normal_at(position),  // Normal query
);
```

## Motion Matching

### Database Setup

```rust
use rhi_unshape_rig::{
    MotionDatabase, MotionClip, MotionFrame, MotionFrameBuilder,
    MotionMatcher, MotionMatchingConfig, MotionQuery,
};

// Build motion database
let mut database = MotionDatabase::new();

// Add clips
let walk_clip = MotionClip::from_animation(&walk_animation, &skeleton);
database.add_clip("walk", walk_clip);

let run_clip = MotionClip::from_animation(&run_animation, &skeleton);
database.add_clip("run", run_clip);
```

### Matching and Playback

```rust
let config = MotionMatchingConfig {
    trajectory_weight: 1.0,
    pose_weight: 1.0,
    velocity_weight: 0.5,
    lookahead_time: 0.5,
};

let mut matcher = MotionMatcher::new(&database, config);

// Query for best matching pose
let query = MotionQuery {
    current_pose: &pose,
    desired_velocity: Vec3::new(0.0, 0.0, 1.0),
    desired_facing: Vec3::Z,
};

let result = matcher.find_best_match(&query);

// Apply matched frame
apply_frame_to_pose(&result.frame, &mut pose);

// Optionally blend between frames
let blended = blend_frames(&current_frame, &result.frame, blend_factor);
```

## Secondary Motion

### Jiggle Bones

```rust
use rhi_unshape_rig::{JiggleBone, SecondaryConfig};

let config = SecondaryConfig {
    stiffness: 0.5,
    damping: 0.8,
    gravity: Vec3::new(0.0, -9.8, 0.0),
};

let mut jiggle = JiggleBone::new(hair_bone, config);

// Update each frame
jiggle.update(&skeleton, &pose, delta_time);
jiggle.apply(&mut pose);
```

### Jiggle Chain

```rust
use rhi_unshape_rig::JiggleChain;

// Chain of bones (e.g., ponytail)
let mut chain = JiggleChain::new(
    vec![hair_1, hair_2, hair_3, hair_4],
    config,
);

chain.update(&skeleton, &pose, delta_time);
chain.apply(&mut pose);
```

### Jiggle Mesh

```rust
use rhi_unshape_rig::JiggleMesh;

let mut jiggle_mesh = JiggleMesh::new(
    &original_positions,
    config,
    anchored_vertices: vec![0, 1, 2],  // Fixed vertices
);

jiggle_mesh.update(&parent_transform, delta_time);
let deformed = jiggle_mesh.positions();
```

### Follow-Through and Overlap

```rust
use rhi_unshape_rig::{FollowThrough, OverlappingAction, RotationFollowThrough};

// Position follow-through (drag)
let mut follow = FollowThrough::new(bone, drag: 0.3);
follow.update(&skeleton, &pose, delta_time);

// Rotation overlap
let mut overlap = RotationFollowThrough::new(bone, lag: 0.2, damping: 0.9);
overlap.update(&skeleton, &pose, delta_time);
overlap.apply(&mut pose);

// Overlapping action (delay secondary elements)
let mut overlap = OverlappingAction::new(
    primary_bone: arm,
    secondary_bones: vec![sleeve_1, sleeve_2],
    delay: 0.1,
);
```

### Wind Forces

```rust
use rhi_unshape_rig::{WindForce, apply_wind_to_bone, apply_wind_to_chain};

let wind = WindForce {
    direction: Vec3::new(1.0, 0.0, 0.3),
    strength: 2.0,
    turbulence: 0.5,
    frequency: 1.0,
};

// Apply to single bone
apply_wind_to_bone(&wind, &mut jiggle_bone, time);

// Apply to chain
apply_wind_to_chain(&wind, &mut jiggle_chain, time);
```

## 3D Paths

```rust
use rhi_unshape_rig::{Path3D, Path3DBuilder, PathSample};

// Build path
let path = Path3DBuilder::new()
    .move_to(Vec3::ZERO)
    .line_to(Vec3::new(1.0, 0.0, 0.0))
    .cubic_to(
        Vec3::new(1.5, 0.5, 0.0),  // Control 1
        Vec3::new(2.0, 0.5, 0.0),  // Control 2
        Vec3::new(2.5, 0.0, 0.0),  // End
    )
    .build();

// Sample path
let sample: PathSample = path.sample(t: 0.5);
let position = sample.position;
let tangent = sample.tangent;
let normal = sample.normal;

// Get total length
let length = path.length();

// Sample at arc length
let sample = path.sample_at_length(distance: 1.5);
```

## Example: Complete Character Rig

```rust
use rhi_unshape_rig::*;

// Setup skeleton
let mut skeleton = Skeleton::new();
// ... add bones ...

// Load animations
let idle = AnimationClip::load("idle.anim")?;
let walk = AnimationClip::load("walk.anim")?;
let run = AnimationClip::load("run.anim")?;

// Setup blend tree
let locomotion = BlendNode::Blend1D {
    clips: vec![idle, walk, run],
    thresholds: vec![0.0, 0.5, 1.0],
    parameter: speed,
};

// Setup IK
let left_arm_ik = IkChain {
    bones: vec![shoulder_l, elbow_l, wrist_l],
    target: Vec3::ZERO,
    pole: Some(Vec3::new(0.0, 0.0, -1.0)),
};

// Setup secondary motion
let mut hair_physics = JiggleChain::new(
    vec![hair_1, hair_2, hair_3],
    SecondaryConfig::default(),
);

// Game loop
loop {
    // Animation
    locomotion.sample(time, &mut pose);

    // IK (reach for object)
    if reaching {
        left_arm_ik.target = target_position;
        solve_fabrik(&skeleton, &mut pose, &left_arm_ik, &ik_config);
    }

    // Secondary motion
    hair_physics.update(&skeleton, &pose, delta_time);
    hair_physics.apply(&mut pose);

    // Skin mesh
    let deformed = skin.deform_positions(&mesh_positions, &skeleton, &pose);
}
```
