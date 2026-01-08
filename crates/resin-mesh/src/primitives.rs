//! Mesh primitives.

use glam::{Vec2, Vec3};
use std::f32::consts::{PI, TAU};

use crate::{Mesh, MeshBuilder};

/// Creates a unit box centered at the origin.
///
/// Each face has its own vertices (not shared) for correct normals.
/// The box extends from -0.5 to 0.5 on each axis.
pub fn box_mesh() -> Mesh {
    let mut builder = MeshBuilder::new();

    // Helper to add a face with 4 vertices and 2 triangles
    let mut add_face = |positions: [Vec3; 4], normal: Vec3, uvs: [Vec2; 4]| {
        let i0 = builder.vertex_with_normal_uv(positions[0], normal, uvs[0]);
        let i1 = builder.vertex_with_normal_uv(positions[1], normal, uvs[1]);
        let i2 = builder.vertex_with_normal_uv(positions[2], normal, uvs[2]);
        let i3 = builder.vertex_with_normal_uv(positions[3], normal, uvs[3]);
        builder.quad(i0, i1, i2, i3);
    };

    let uv = [
        Vec2::new(0.0, 0.0),
        Vec2::new(1.0, 0.0),
        Vec2::new(1.0, 1.0),
        Vec2::new(0.0, 1.0),
    ];

    // Front face (+Z)
    add_face(
        [
            Vec3::new(-0.5, -0.5, 0.5),
            Vec3::new(0.5, -0.5, 0.5),
            Vec3::new(0.5, 0.5, 0.5),
            Vec3::new(-0.5, 0.5, 0.5),
        ],
        Vec3::Z,
        uv,
    );

    // Back face (-Z)
    add_face(
        [
            Vec3::new(0.5, -0.5, -0.5),
            Vec3::new(-0.5, -0.5, -0.5),
            Vec3::new(-0.5, 0.5, -0.5),
            Vec3::new(0.5, 0.5, -0.5),
        ],
        Vec3::NEG_Z,
        uv,
    );

    // Right face (+X)
    add_face(
        [
            Vec3::new(0.5, -0.5, 0.5),
            Vec3::new(0.5, -0.5, -0.5),
            Vec3::new(0.5, 0.5, -0.5),
            Vec3::new(0.5, 0.5, 0.5),
        ],
        Vec3::X,
        uv,
    );

    // Left face (-X)
    add_face(
        [
            Vec3::new(-0.5, -0.5, -0.5),
            Vec3::new(-0.5, -0.5, 0.5),
            Vec3::new(-0.5, 0.5, 0.5),
            Vec3::new(-0.5, 0.5, -0.5),
        ],
        Vec3::NEG_X,
        uv,
    );

    // Top face (+Y)
    add_face(
        [
            Vec3::new(-0.5, 0.5, 0.5),
            Vec3::new(0.5, 0.5, 0.5),
            Vec3::new(0.5, 0.5, -0.5),
            Vec3::new(-0.5, 0.5, -0.5),
        ],
        Vec3::Y,
        uv,
    );

    // Bottom face (-Y)
    add_face(
        [
            Vec3::new(-0.5, -0.5, -0.5),
            Vec3::new(0.5, -0.5, -0.5),
            Vec3::new(0.5, -0.5, 0.5),
            Vec3::new(-0.5, -0.5, 0.5),
        ],
        Vec3::NEG_Y,
        uv,
    );

    builder.build()
}

/// Creates a UV sphere centered at the origin with radius 1.
///
/// # Arguments
/// * `segments` - Number of horizontal divisions (longitude). Minimum 3.
/// * `rings` - Number of vertical divisions (latitude). Minimum 2.
pub fn uv_sphere(segments: u32, rings: u32) -> Mesh {
    let segments = segments.max(3);
    let rings = rings.max(2);

    let mut builder = MeshBuilder::new();

    // Generate vertices
    for ring in 0..=rings {
        let v = ring as f32 / rings as f32;
        let phi = PI * v; // 0 to PI (top to bottom)

        for segment in 0..=segments {
            let u = segment as f32 / segments as f32;
            let theta = TAU * u; // 0 to TAU (around)

            let x = phi.sin() * theta.cos();
            let y = phi.cos();
            let z = phi.sin() * theta.sin();

            let position = Vec3::new(x, y, z);
            let normal = position; // For unit sphere, position = normal
            let uv = Vec2::new(u, v);

            builder.vertex_with_normal_uv(position, normal, uv);
        }
    }

    // Generate triangles
    let stride = segments + 1;

    for ring in 0..rings {
        for segment in 0..segments {
            let i0 = ring * stride + segment;
            let i1 = i0 + 1;
            let i2 = i0 + stride;
            let i3 = i2 + 1;

            // Top cap: only one triangle
            if ring == 0 {
                builder.triangle(i0, i2, i3);
            }
            // Bottom cap: only one triangle
            else if ring == rings - 1 {
                builder.triangle(i0, i2, i1);
            }
            // Middle: full quad
            else {
                builder.quad(i0, i2, i3, i1);
            }
        }
    }

    builder.build()
}

/// Creates a UV sphere with default resolution (32 segments, 16 rings).
pub fn sphere() -> Mesh {
    uv_sphere(32, 16)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_box_mesh() {
        let mesh = box_mesh();

        // Box has 6 faces * 4 vertices = 24 vertices
        assert_eq!(mesh.vertex_count(), 24);
        // Box has 6 faces * 2 triangles = 12 triangles
        assert_eq!(mesh.triangle_count(), 12);
        assert!(mesh.has_normals());
        assert!(mesh.has_uvs());
    }

    #[test]
    fn test_sphere() {
        let mesh = uv_sphere(8, 4);

        assert!(mesh.vertex_count() > 0);
        assert!(mesh.triangle_count() > 0);
        assert!(mesh.has_normals());
        assert!(mesh.has_uvs());

        // Check all normals are unit length (since it's a unit sphere)
        for normal in &mesh.normals {
            assert!((normal.length() - 1.0).abs() < 0.001);
        }
    }

    #[test]
    fn test_sphere_bounds() {
        let mesh = sphere();

        // All positions should be on unit sphere
        for pos in &mesh.positions {
            assert!((pos.length() - 1.0).abs() < 0.001);
        }
    }
}
