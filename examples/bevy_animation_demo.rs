//! Bevy integration demo: Skeletal animation and IK.
//!
//! Demonstrates using resin-rig for bones, skeletons, poses, and IK
//! in a Bevy application.
//!
//! Run with: `cargo run --example bevy_animation_demo`

use bevy::{
    prelude::*,
    render::mesh::{Indices, PrimitiveTopology},
};
use rhizome_resin_rig::{
    Bone, BoneId, IkChain, IkConfig, Pose, Skeleton, Transform as ResinTransform, solve_fabrik,
};

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_systems(Startup, setup)
        .add_systems(Update, (update_ik_target, render_skeleton))
        .run();
}

/// Component holding the resin skeleton and pose.
#[derive(Component)]
struct ResinSkeleton {
    skeleton: Skeleton,
    pose: Pose,
    bone_ids: Vec<BoneId>,
}

/// Component for the IK target marker.
#[derive(Component)]
struct IkTarget;

/// Resource tracking time for animation.
#[derive(Resource)]
struct AnimationTime(f32);

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    // Create a simple 3-bone arm skeleton
    let mut skeleton = Skeleton::new();

    let shoulder = skeleton
        .add_bone(Bone {
            name: "shoulder".into(),
            local_transform: ResinTransform::from_translation(glam::Vec3::new(0.0, 0.0, 0.0)),
            length: 1.5,
            ..Default::default()
        })
        .id;

    let upper_arm = skeleton
        .add_bone(Bone {
            name: "upper_arm".into(),
            parent: Some(shoulder),
            local_transform: ResinTransform::from_translation(glam::Vec3::new(0.0, 1.5, 0.0)),
            length: 1.5,
        })
        .id;

    let forearm = skeleton
        .add_bone(Bone {
            name: "forearm".into(),
            parent: Some(upper_arm),
            local_transform: ResinTransform::from_translation(glam::Vec3::new(0.0, 1.5, 0.0)),
            length: 1.0,
        })
        .id;

    let pose = skeleton.rest_pose();

    commands.spawn(ResinSkeleton {
        skeleton,
        pose,
        bone_ids: vec![shoulder, upper_arm, forearm],
    });

    // Create IK target marker (sphere that moves in a circle)
    commands.spawn((
        Mesh3d(meshes.add(Sphere::new(0.2).mesh().ico(2).unwrap())),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(1.0, 0.3, 0.3),
            emissive: LinearRgba::new(1.0, 0.3, 0.3, 1.0),
            ..default()
        })),
        Transform::from_xyz(2.0, 3.0, 0.0),
        IkTarget,
    ));

    // Spawn debug meshes for each bone
    let bone_mesh = meshes.add(create_bone_mesh());
    let joint_mesh = meshes.add(Sphere::new(0.15).mesh().ico(2).unwrap());

    let bone_material = materials.add(StandardMaterial {
        base_color: Color::srgb(0.2, 0.6, 0.9),
        ..default()
    });

    let joint_material = materials.add(StandardMaterial {
        base_color: Color::srgb(0.9, 0.9, 0.2),
        ..default()
    });

    // Create visuals for 3 bones
    for i in 0..3 {
        commands.spawn((
            Mesh3d(bone_mesh.clone()),
            MeshMaterial3d(bone_material.clone()),
            Transform::default(),
            BoneVisual(i),
        ));
        commands.spawn((
            Mesh3d(joint_mesh.clone()),
            MeshMaterial3d(joint_material.clone()),
            Transform::default(),
            JointVisual(i),
        ));
    }

    // Camera
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(0.0, 3.0, 8.0).looking_at(Vec3::new(0.0, 2.0, 0.0), Vec3::Y),
    ));

    // Light
    commands.spawn((
        DirectionalLight {
            illuminance: 10000.0,
            ..default()
        },
        Transform::from_xyz(4.0, 8.0, 4.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));

    // Ambient light
    commands.insert_resource(AmbientLight {
        color: Color::WHITE,
        brightness: 300.0,
    });

    commands.insert_resource(AnimationTime(0.0));
}

/// Marker component for bone visuals.
#[derive(Component)]
struct BoneVisual(usize);

/// Marker component for joint visuals.
#[derive(Component)]
struct JointVisual(usize);

fn update_ik_target(
    time: Res<Time>,
    mut anim_time: ResMut<AnimationTime>,
    mut target_query: Query<&mut Transform, With<IkTarget>>,
    mut skeleton_query: Query<&mut ResinSkeleton>,
) {
    anim_time.0 += time.delta_secs();
    let t = anim_time.0;

    // Move target in a figure-8 pattern
    let target_x = (t * 0.5).sin() * 2.0;
    let target_y = 2.0 + (t * 1.0).sin() * 1.5;
    let target_z = (t * 0.7).cos() * 0.5;

    for mut transform in &mut target_query {
        transform.translation = Vec3::new(target_x, target_y, target_z);
    }

    // Solve IK
    let target = glam::Vec3::new(target_x, target_y, target_z);

    for mut resin_skel in &mut skeleton_query {
        // Clone skeleton to avoid borrow conflicts (skeleton isn't modified)
        let skeleton = resin_skel.skeleton.clone();
        let bone_ids = resin_skel.bone_ids.clone();

        // Create IK chain from shoulder to forearm
        let chain = IkChain::new(bone_ids);

        let config = IkConfig {
            max_iterations: 10,
            tolerance: 0.01,
        };

        // Solve IK - modifies pose in place
        let _result = solve_fabrik(&skeleton, &mut resin_skel.pose, &chain, target, &config);
    }
}

fn render_skeleton(
    skeleton_query: Query<&ResinSkeleton>,
    mut bone_query: Query<(&mut Transform, &BoneVisual), Without<JointVisual>>,
    mut joint_query: Query<(&mut Transform, &JointVisual), Without<BoneVisual>>,
) {
    for resin_skel in &skeleton_query {
        let skeleton = &resin_skel.skeleton;
        let pose = &resin_skel.pose;

        for (mut transform, BoneVisual(idx)) in &mut bone_query {
            if let Some(&bone_id) = resin_skel.bone_ids.get(*idx) {
                let world = pose.world_transform(skeleton, bone_id);
                let bone = skeleton.bone(bone_id).unwrap();

                // Position at bone head
                transform.translation = Vec3::new(
                    world.translation.x,
                    world.translation.y,
                    world.translation.z,
                );

                // Rotate to point along bone
                transform.rotation = Quat::from_xyzw(
                    world.rotation.x,
                    world.rotation.y,
                    world.rotation.z,
                    world.rotation.w,
                );

                // Scale by bone length
                transform.scale = Vec3::new(0.1, bone.length, 0.1);
            }
        }

        for (mut transform, JointVisual(idx)) in &mut joint_query {
            if let Some(&bone_id) = resin_skel.bone_ids.get(*idx) {
                let world = pose.world_transform(skeleton, bone_id);

                transform.translation = Vec3::new(
                    world.translation.x,
                    world.translation.y,
                    world.translation.z,
                );
            }
        }
    }
}

/// Create a simple bone mesh (tapered cylinder).
fn create_bone_mesh() -> Mesh {
    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        bevy::render::render_asset::RenderAssetUsages::default(),
    );

    let segments = 8;
    let mut positions = Vec::new();
    let mut normals = Vec::new();
    let mut indices = Vec::new();

    // Bottom (wider) and top (narrower) radii
    let r_bottom = 1.0;
    let r_top = 0.5;

    // Create vertices for bottom and top rings
    for i in 0..segments {
        let angle = (i as f32 / segments as f32) * std::f32::consts::TAU;
        let cos = angle.cos();
        let sin = angle.sin();

        // Bottom vertex
        positions.push([cos * r_bottom, 0.0, sin * r_bottom]);
        normals.push([cos, 0.0, sin]);

        // Top vertex
        positions.push([cos * r_top, 1.0, sin * r_top]);
        normals.push([cos, 0.0, sin]);
    }

    // Side triangles
    for i in 0..segments {
        let next = (i + 1) % segments;
        let b0 = (i * 2) as u32;
        let t0 = (i * 2 + 1) as u32;
        let b1 = (next * 2) as u32;
        let t1 = (next * 2 + 1) as u32;

        indices.extend_from_slice(&[b0, t0, b1]);
        indices.extend_from_slice(&[b1, t0, t1]);
    }

    // Bottom cap center
    let bottom_center = positions.len() as u32;
    positions.push([0.0, 0.0, 0.0]);
    normals.push([0.0, -1.0, 0.0]);

    // Top cap center
    let top_center = positions.len() as u32;
    positions.push([0.0, 1.0, 0.0]);
    normals.push([0.0, 1.0, 0.0]);

    // Cap triangles
    for i in 0..segments {
        let next = (i + 1) % segments;
        let b0 = (i * 2) as u32;
        let b1 = (next * 2) as u32;
        let t0 = (i * 2 + 1) as u32;
        let t1 = (next * 2 + 1) as u32;

        // Bottom cap
        indices.extend_from_slice(&[bottom_center, b1, b0]);
        // Top cap
        indices.extend_from_slice(&[top_center, t0, t1]);
    }

    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
    mesh.insert_indices(Indices::U32(indices));

    mesh
}
