//! Bevy integration demo: Procedural mesh generation.
//!
//! Demonstrates using unshape-mesh to generate procedural geometry
//! and display it in Bevy.
//!
//! Run with: `cargo run --example bevy_mesh_demo`

use bevy::{
    prelude::*,
    render::mesh::{Indices, PrimitiveTopology},
};
use unshape_mesh::{Cuboid, UvSphere, subdivide_loop_n};

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_systems(Startup, setup)
        .add_systems(Update, rotate_meshes)
        .run();
}

/// Marker component for rotating meshes.
#[derive(Component)]
struct Rotating {
    speed: f32,
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    // Create a basic unit box mesh from resin
    let resin_box = Cuboid::unit().apply();
    let bevy_box = resin_mesh_to_bevy(&resin_box);

    commands.spawn((
        Mesh3d(meshes.add(bevy_box)),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.8, 0.2, 0.2),
            ..default()
        })),
        Transform::from_xyz(-3.0, 0.0, 0.0),
        Rotating { speed: 0.5 },
    ));

    // Create a UV sphere with custom resolution
    let resin_sphere = UvSphere::new(1.0, 32, 16).apply();
    let bevy_sphere = resin_mesh_to_bevy(&resin_sphere);

    commands.spawn((
        Mesh3d(meshes.add(bevy_sphere)),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.2, 0.8, 0.2),
            ..default()
        })),
        Transform::from_xyz(0.0, 0.0, 0.0),
        Rotating { speed: 0.3 },
    ));

    // Create a subdivided box (smooth) using Loop subdivision
    let subdivided = Cuboid::unit().apply();
    let subdivided = subdivide_loop_n(&subdivided, 2);
    let bevy_subdivided = resin_mesh_to_bevy(&subdivided);

    commands.spawn((
        Mesh3d(meshes.add(bevy_subdivided)),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.2, 0.2, 0.8),
            ..default()
        })),
        Transform::from_xyz(3.0, 0.0, 0.0),
        Rotating { speed: 0.7 },
    ));

    // Camera
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(0.0, 5.0, 10.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));

    // Light
    commands.spawn((
        DirectionalLight {
            illuminance: 10000.0,
            shadows_enabled: true,
            ..default()
        },
        Transform::from_xyz(4.0, 8.0, 4.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));

    // Ambient light
    commands.insert_resource(AmbientLight {
        color: Color::WHITE,
        brightness: 200.0,
    });
}

fn rotate_meshes(time: Res<Time>, mut query: Query<(&mut Transform, &Rotating)>) {
    for (mut transform, rotating) in &mut query {
        transform.rotate_y(rotating.speed * time.delta_secs());
    }
}

/// Convert a resin Mesh to a Bevy Mesh.
fn resin_mesh_to_bevy(resin: &unshape_mesh::Mesh) -> Mesh {
    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        bevy::render::render_asset::RenderAssetUsages::default(),
    );

    // Convert positions (glam Vec3 is compatible)
    let positions: Vec<[f32; 3]> = resin.positions.iter().map(|v| [v.x, v.y, v.z]).collect();
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);

    // Convert normals if present
    if !resin.normals.is_empty() {
        let normals: Vec<[f32; 3]> = resin.normals.iter().map(|v| [v.x, v.y, v.z]).collect();
        mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
    }

    // Convert UVs if present
    if !resin.uvs.is_empty() {
        let uvs: Vec<[f32; 2]> = resin.uvs.iter().map(|v| [v.x, v.y]).collect();
        mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uvs);
    }

    // Set indices
    mesh.insert_indices(Indices::U32(resin.indices.clone()));

    mesh
}
