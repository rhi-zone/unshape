//! Mesh primitives.
//!
//! Each primitive is a serializable struct with an `apply()` method that generates
//! a mesh. See `docs/design/ops-as-values.md`.
//!
//! # Design Note
//!
//! These primitives are hardcoded shape generators. This is pragmatic but not ideal
//! from a "generative mindset" perspective.
//!
//! Ideally, primitives would be compositions of more fundamental operations or
//! expressed through a general-purpose expression system. However, Dew (our
//! expression language) doesn't support generating vertex lists, so there's no
//! viable alternative currently.
//!
//! If a more expressive primitive system emerges later (e.g., SDF-based generation,
//! parametric surface evaluation), these structs can become sugar over it.
//!
//! # Primitive Unification
//!
//! Some shapes are unified under a single primitive:
//! - `Cone` with low segments (3, 4, 5...) produces pyramids/tetrahedra
//! - `Cylinder` with low segments produces prisms (triangular, square, hexagonal...)
//!
//! This reduces redundancy while still allowing common shapes via segment count.

use glam::{Vec2, Vec3};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use std::f32::consts::{PI, TAU};

use crate::{Mesh, MeshBuilder};

// ============================================================================
// Cuboid
// ============================================================================

/// Generates a box/cuboid mesh centered at the origin.
///
/// Each face has its own vertices (not shared) for correct per-face normals.
/// UVs are mapped per-face (0-1 on each face).
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dynop", derive(unshape_op::Op))]
#[cfg_attr(feature = "dynop", op(input = (), output = Mesh))]
pub struct Cuboid {
    /// Size along the X axis.
    pub width: f32,
    /// Size along the Y axis.
    pub height: f32,
    /// Size along the Z axis.
    pub depth: f32,
}

impl Default for Cuboid {
    fn default() -> Self {
        Self {
            width: 1.0,
            height: 1.0,
            depth: 1.0,
        }
    }
}

impl Cuboid {
    /// Creates a new cuboid with the given dimensions.
    pub fn new(width: f32, height: f32, depth: f32) -> Self {
        Self {
            width,
            height,
            depth,
        }
    }

    /// Creates a unit cube (1x1x1).
    pub fn unit() -> Self {
        Self::default()
    }

    /// Creates a cube with equal sides.
    pub fn cube(size: f32) -> Self {
        Self::new(size, size, size)
    }

    /// Generates the cuboid mesh.
    pub fn apply(&self) -> Mesh {
        let mut builder = MeshBuilder::new();

        let hx = self.width / 2.0;
        let hy = self.height / 2.0;
        let hz = self.depth / 2.0;

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
                Vec3::new(-hx, -hy, hz),
                Vec3::new(hx, -hy, hz),
                Vec3::new(hx, hy, hz),
                Vec3::new(-hx, hy, hz),
            ],
            Vec3::Z,
            uv,
        );

        // Back face (-Z)
        add_face(
            [
                Vec3::new(hx, -hy, -hz),
                Vec3::new(-hx, -hy, -hz),
                Vec3::new(-hx, hy, -hz),
                Vec3::new(hx, hy, -hz),
            ],
            Vec3::NEG_Z,
            uv,
        );

        // Right face (+X)
        add_face(
            [
                Vec3::new(hx, -hy, hz),
                Vec3::new(hx, -hy, -hz),
                Vec3::new(hx, hy, -hz),
                Vec3::new(hx, hy, hz),
            ],
            Vec3::X,
            uv,
        );

        // Left face (-X)
        add_face(
            [
                Vec3::new(-hx, -hy, -hz),
                Vec3::new(-hx, -hy, hz),
                Vec3::new(-hx, hy, hz),
                Vec3::new(-hx, hy, -hz),
            ],
            Vec3::NEG_X,
            uv,
        );

        // Top face (+Y)
        add_face(
            [
                Vec3::new(-hx, hy, hz),
                Vec3::new(hx, hy, hz),
                Vec3::new(hx, hy, -hz),
                Vec3::new(-hx, hy, -hz),
            ],
            Vec3::Y,
            uv,
        );

        // Bottom face (-Y)
        add_face(
            [
                Vec3::new(-hx, -hy, -hz),
                Vec3::new(hx, -hy, -hz),
                Vec3::new(hx, -hy, hz),
                Vec3::new(-hx, -hy, hz),
            ],
            Vec3::NEG_Y,
            uv,
        );

        builder.build()
    }
}

// ============================================================================
// UvSphere
// ============================================================================

/// Generates a UV sphere mesh centered at the origin.
///
/// Uses latitude/longitude parameterization. For more uniform triangle
/// distribution, see [`Icosphere`].
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dynop", derive(unshape_op::Op))]
#[cfg_attr(feature = "dynop", op(input = (), output = Mesh))]
pub struct UvSphere {
    /// Radius of the sphere.
    pub radius: f32,
    /// Number of horizontal divisions (longitude). Minimum 3.
    pub segments: u32,
    /// Number of vertical divisions (latitude). Minimum 2.
    pub rings: u32,
}

impl Default for UvSphere {
    fn default() -> Self {
        Self {
            radius: 1.0,
            segments: 32,
            rings: 16,
        }
    }
}

impl UvSphere {
    /// Creates a new UV sphere with the given parameters.
    pub fn new(radius: f32, segments: u32, rings: u32) -> Self {
        Self {
            radius,
            segments,
            rings,
        }
    }

    /// Generates the sphere mesh.
    pub fn apply(&self) -> Mesh {
        let segments = self.segments.max(3);
        let rings = self.rings.max(2);
        let radius = self.radius;

        let mut builder = MeshBuilder::new();

        // Generate vertices
        for ring in 0..=rings {
            let v = ring as f32 / rings as f32;
            let phi = PI * v;

            for segment in 0..=segments {
                let u = segment as f32 / segments as f32;
                let theta = TAU * u;

                let x = phi.sin() * theta.cos();
                let y = phi.cos();
                let z = phi.sin() * theta.sin();

                let normal = Vec3::new(x, y, z);
                let position = normal * radius;
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

                if ring == 0 {
                    builder.triangle(i0, i2, i3);
                } else if ring == rings - 1 {
                    builder.triangle(i0, i2, i1);
                } else {
                    builder.quad(i0, i2, i3, i1);
                }
            }
        }

        builder.build()
    }
}

// ============================================================================
// Cylinder
// ============================================================================

/// Generates a cylinder mesh centered at the origin.
///
/// The cylinder extends from -height/2 to +height/2 on the Y axis.
/// Includes top and bottom caps with proper normals.
///
/// With low segment counts, produces prisms:
/// - 3 segments = triangular prism
/// - 4 segments = square prism
/// - 6 segments = hexagonal prism
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dynop", derive(unshape_op::Op))]
#[cfg_attr(feature = "dynop", op(input = (), output = Mesh))]
pub struct Cylinder {
    /// Radius of the cylinder.
    pub radius: f32,
    /// Height of the cylinder.
    pub height: f32,
    /// Number of divisions around the circumference. Minimum 3.
    pub segments: u32,
}

impl Default for Cylinder {
    fn default() -> Self {
        Self {
            radius: 1.0,
            height: 2.0,
            segments: 32,
        }
    }
}

impl Cylinder {
    /// Creates a new cylinder with the given dimensions.
    pub fn new(radius: f32, height: f32, segments: u32) -> Self {
        Self {
            radius,
            height,
            segments,
        }
    }

    /// Generates the cylinder mesh.
    pub fn apply(&self) -> Mesh {
        let segments = self.segments.max(3);
        let half_height = self.height / 2.0;
        let radius = self.radius;

        let mut builder = MeshBuilder::new();

        // Bottom cap center
        let bottom_center = builder.vertex_with_normal_uv(
            Vec3::new(0.0, -half_height, 0.0),
            Vec3::NEG_Y,
            Vec2::new(0.5, 0.5),
        );

        // Bottom cap ring
        let mut bottom_ring = Vec::with_capacity(segments as usize);
        for i in 0..segments {
            let angle = TAU * (i as f32) / (segments as f32);
            let x = angle.cos() * radius;
            let z = angle.sin() * radius;
            let u = 0.5 + angle.cos() * 0.5;
            let v = 0.5 + angle.sin() * 0.5;
            bottom_ring.push(builder.vertex_with_normal_uv(
                Vec3::new(x, -half_height, z),
                Vec3::NEG_Y,
                Vec2::new(u, v),
            ));
        }

        // Bottom cap triangles
        for i in 0..segments {
            let next = (i + 1) % segments;
            builder.triangle(
                bottom_center,
                bottom_ring[next as usize],
                bottom_ring[i as usize],
            );
        }

        // Top cap center
        let top_center = builder.vertex_with_normal_uv(
            Vec3::new(0.0, half_height, 0.0),
            Vec3::Y,
            Vec2::new(0.5, 0.5),
        );

        // Top cap ring
        let mut top_ring = Vec::with_capacity(segments as usize);
        for i in 0..segments {
            let angle = TAU * (i as f32) / (segments as f32);
            let x = angle.cos() * radius;
            let z = angle.sin() * radius;
            let u = 0.5 + angle.cos() * 0.5;
            let v = 0.5 + angle.sin() * 0.5;
            top_ring.push(builder.vertex_with_normal_uv(
                Vec3::new(x, half_height, z),
                Vec3::Y,
                Vec2::new(u, v),
            ));
        }

        // Top cap triangles
        for i in 0..segments {
            let next = (i + 1) % segments;
            builder.triangle(top_center, top_ring[i as usize], top_ring[next as usize]);
        }

        // Side surface
        for i in 0..=segments {
            let angle = TAU * (i as f32) / (segments as f32);
            let x = angle.cos();
            let z = angle.sin();
            let normal = Vec3::new(x, 0.0, z);
            let u = i as f32 / segments as f32;

            builder.vertex_with_normal_uv(
                Vec3::new(x * radius, -half_height, z * radius),
                normal,
                Vec2::new(u, 0.0),
            );
            builder.vertex_with_normal_uv(
                Vec3::new(x * radius, half_height, z * radius),
                normal,
                Vec2::new(u, 1.0),
            );
        }

        // Side quads
        let side_start = 1 + segments + 1 + segments;
        for i in 0..segments {
            let bl = side_start + i * 2;
            let tl = bl + 1;
            let br = bl + 2;
            let tr = bl + 3;
            builder.quad(bl, br, tr, tl);
        }

        builder.build()
    }
}

// ============================================================================
// Cone
// ============================================================================

/// Generates a cone mesh centered at the origin.
///
/// The cone has its base at -height/2 and apex at +height/2 on the Y axis.
/// Includes a bottom cap with proper normals.
///
/// With low segment counts, produces pyramids:
/// - 3 segments = tetrahedron (triangular pyramid)
/// - 4 segments = square pyramid
/// - 5 segments = pentagonal pyramid
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dynop", derive(unshape_op::Op))]
#[cfg_attr(feature = "dynop", op(input = (), output = Mesh))]
pub struct Cone {
    /// Radius of the base.
    pub radius: f32,
    /// Height of the cone.
    pub height: f32,
    /// Number of divisions around the circumference. Minimum 3.
    pub segments: u32,
}

impl Default for Cone {
    fn default() -> Self {
        Self {
            radius: 1.0,
            height: 2.0,
            segments: 32,
        }
    }
}

impl Cone {
    /// Creates a new cone with the given dimensions.
    pub fn new(radius: f32, height: f32, segments: u32) -> Self {
        Self {
            radius,
            height,
            segments,
        }
    }

    /// Generates the cone mesh.
    pub fn apply(&self) -> Mesh {
        let segments = self.segments.max(3);
        let half_height = self.height / 2.0;
        let radius = self.radius;

        let mut builder = MeshBuilder::new();

        // Bottom cap center
        let bottom_center = builder.vertex_with_normal_uv(
            Vec3::new(0.0, -half_height, 0.0),
            Vec3::NEG_Y,
            Vec2::new(0.5, 0.5),
        );

        // Bottom cap ring
        let mut bottom_ring = Vec::with_capacity(segments as usize);
        for i in 0..segments {
            let angle = TAU * (i as f32) / (segments as f32);
            let x = angle.cos() * radius;
            let z = angle.sin() * radius;
            let u = 0.5 + angle.cos() * 0.5;
            let v = 0.5 + angle.sin() * 0.5;
            bottom_ring.push(builder.vertex_with_normal_uv(
                Vec3::new(x, -half_height, z),
                Vec3::NEG_Y,
                Vec2::new(u, v),
            ));
        }

        // Bottom cap triangles
        for i in 0..segments {
            let next = (i + 1) % segments;
            builder.triangle(
                bottom_center,
                bottom_ring[next as usize],
                bottom_ring[i as usize],
            );
        }

        // Side surface
        let slope = (radius / self.height).atan();
        let normal_y = slope.sin();
        let normal_xz_scale = slope.cos();

        for i in 0..=segments {
            let angle = TAU * (i as f32) / (segments as f32);
            let x = angle.cos();
            let z = angle.sin();
            let normal = Vec3::new(x * normal_xz_scale, normal_y, z * normal_xz_scale).normalize();
            let u = i as f32 / segments as f32;

            builder.vertex_with_normal_uv(
                Vec3::new(x * radius, -half_height, z * radius),
                normal,
                Vec2::new(u, 0.0),
            );
            builder.vertex_with_normal_uv(
                Vec3::new(0.0, half_height, 0.0),
                normal,
                Vec2::new(u, 1.0),
            );
        }

        // Side triangles
        let side_start = 1 + segments;
        for i in 0..segments {
            let bl = side_start + i * 2;
            let apex = bl + 1;
            let br = bl + 2;
            builder.triangle(bl, br, apex);
        }

        builder.build()
    }
}

// ============================================================================
// Torus
// ============================================================================

/// Generates a torus (donut shape) mesh centered at the origin.
///
/// The torus lies in the XZ plane with the hole along the Y axis.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dynop", derive(unshape_op::Op))]
#[cfg_attr(feature = "dynop", op(input = (), output = Mesh))]
pub struct Torus {
    /// Distance from center of torus to center of tube.
    pub major_radius: f32,
    /// Radius of the tube.
    pub minor_radius: f32,
    /// Divisions around the main ring. Minimum 3.
    pub major_segments: u32,
    /// Divisions around the tube cross-section. Minimum 3.
    pub minor_segments: u32,
}

impl Default for Torus {
    fn default() -> Self {
        Self {
            major_radius: 1.0,
            minor_radius: 0.25,
            major_segments: 32,
            minor_segments: 16,
        }
    }
}

impl Torus {
    /// Creates a new torus with the given dimensions.
    pub fn new(
        major_radius: f32,
        minor_radius: f32,
        major_segments: u32,
        minor_segments: u32,
    ) -> Self {
        Self {
            major_radius,
            minor_radius,
            major_segments,
            minor_segments,
        }
    }

    /// Generates the torus mesh.
    pub fn apply(&self) -> Mesh {
        let major_segments = self.major_segments.max(3);
        let minor_segments = self.minor_segments.max(3);
        let major_radius = self.major_radius;
        let minor_radius = self.minor_radius;

        let mut builder = MeshBuilder::new();

        for i in 0..=major_segments {
            let u = i as f32 / major_segments as f32;
            let major_angle = TAU * u;
            let cos_major = major_angle.cos();
            let sin_major = major_angle.sin();

            for j in 0..=minor_segments {
                let v = j as f32 / minor_segments as f32;
                let minor_angle = TAU * v;
                let cos_minor = minor_angle.cos();
                let sin_minor = minor_angle.sin();

                let x = (major_radius + minor_radius * cos_minor) * cos_major;
                let y = minor_radius * sin_minor;
                let z = (major_radius + minor_radius * cos_minor) * sin_major;

                let nx = cos_minor * cos_major;
                let ny = sin_minor;
                let nz = cos_minor * sin_major;

                builder.vertex_with_normal_uv(
                    Vec3::new(x, y, z),
                    Vec3::new(nx, ny, nz),
                    Vec2::new(u, v),
                );
            }
        }

        let stride = minor_segments + 1;
        for i in 0..major_segments {
            for j in 0..minor_segments {
                let i0 = i * stride + j;
                let i1 = i0 + 1;
                let i2 = i0 + stride;
                let i3 = i2 + 1;
                builder.quad(i0, i2, i3, i1);
            }
        }

        builder.build()
    }
}

// ============================================================================
// Plane
// ============================================================================

/// Generates a flat plane mesh in the XZ plane, centered at the origin.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dynop", derive(unshape_op::Op))]
#[cfg_attr(feature = "dynop", op(input = (), output = Mesh))]
pub struct Plane {
    /// Size along the X axis.
    pub width: f32,
    /// Size along the Z axis.
    pub depth: f32,
    /// Number of divisions along X. Minimum 1.
    pub subdivisions_x: u32,
    /// Number of divisions along Z. Minimum 1.
    pub subdivisions_z: u32,
}

impl Default for Plane {
    fn default() -> Self {
        Self {
            width: 1.0,
            depth: 1.0,
            subdivisions_x: 1,
            subdivisions_z: 1,
        }
    }
}

impl Plane {
    /// Creates a new plane with the given dimensions.
    pub fn new(width: f32, depth: f32, subdivisions_x: u32, subdivisions_z: u32) -> Self {
        Self {
            width,
            depth,
            subdivisions_x,
            subdivisions_z,
        }
    }

    /// Generates the plane mesh.
    pub fn apply(&self) -> Mesh {
        let subdivisions_x = self.subdivisions_x.max(1);
        let subdivisions_z = self.subdivisions_z.max(1);

        let mut builder = MeshBuilder::new();

        let half_width = self.width / 2.0;
        let half_depth = self.depth / 2.0;

        for iz in 0..=subdivisions_z {
            let v = iz as f32 / subdivisions_z as f32;
            let z = -half_depth + self.depth * v;

            for ix in 0..=subdivisions_x {
                let u = ix as f32 / subdivisions_x as f32;
                let x = -half_width + self.width * u;

                builder.vertex_with_normal_uv(Vec3::new(x, 0.0, z), Vec3::Y, Vec2::new(u, v));
            }
        }

        let stride = subdivisions_x + 1;
        for iz in 0..subdivisions_z {
            for ix in 0..subdivisions_x {
                let i0 = iz * stride + ix;
                let i1 = i0 + 1;
                let i2 = i0 + stride;
                let i3 = i2 + 1;
                builder.quad(i0, i1, i3, i2);
            }
        }

        builder.build()
    }
}

// ============================================================================
// Icosphere
// ============================================================================

/// Generates an icosphere (geodesic sphere) mesh centered at the origin.
///
/// Starts from an icosahedron and subdivides each face recursively.
/// Produces more uniform triangle distribution than [`UvSphere`].
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dynop", derive(unshape_op::Op))]
#[cfg_attr(feature = "dynop", op(input = (), output = Mesh))]
pub struct Icosphere {
    /// Radius of the sphere.
    pub radius: f32,
    /// Number of subdivision iterations. 0 = icosahedron (20 faces).
    /// Each level multiplies face count by 4. Maximum 5 (practical limit).
    pub subdivisions: u32,
}

impl Default for Icosphere {
    fn default() -> Self {
        Self {
            radius: 1.0,
            subdivisions: 2,
        }
    }
}

impl Icosphere {
    /// Creates a new icosphere with the given parameters.
    pub fn new(radius: f32, subdivisions: u32) -> Self {
        Self {
            radius,
            subdivisions,
        }
    }

    /// Generates the icosphere mesh.
    pub fn apply(&self) -> Mesh {
        let subdivisions = self.subdivisions.min(5);
        let radius = self.radius;

        // Golden ratio
        let phi = (1.0 + 5.0_f32.sqrt()) / 2.0;
        let len = (1.0 + phi * phi).sqrt();
        let a = 1.0 / len;
        let b = phi / len;

        let vertices = [
            Vec3::new(-a, b, 0.0),
            Vec3::new(a, b, 0.0),
            Vec3::new(-a, -b, 0.0),
            Vec3::new(a, -b, 0.0),
            Vec3::new(0.0, -a, b),
            Vec3::new(0.0, a, b),
            Vec3::new(0.0, -a, -b),
            Vec3::new(0.0, a, -b),
            Vec3::new(b, 0.0, -a),
            Vec3::new(b, 0.0, a),
            Vec3::new(-b, 0.0, -a),
            Vec3::new(-b, 0.0, a),
        ];

        let mut faces: Vec<[usize; 3]> = vec![
            [0, 11, 5],
            [0, 5, 1],
            [0, 1, 7],
            [0, 7, 10],
            [0, 10, 11],
            [1, 5, 9],
            [5, 11, 4],
            [11, 10, 2],
            [10, 7, 6],
            [7, 1, 8],
            [3, 9, 4],
            [3, 4, 2],
            [3, 2, 6],
            [3, 6, 8],
            [3, 8, 9],
            [4, 9, 5],
            [2, 4, 11],
            [6, 2, 10],
            [8, 6, 7],
            [9, 8, 1],
        ];

        let mut current_vertices = vertices.to_vec();

        use std::collections::HashMap;
        for _ in 0..subdivisions {
            let mut new_faces = Vec::with_capacity(faces.len() * 4);
            let mut midpoint_cache: HashMap<(usize, usize), usize> = HashMap::new();

            let mut get_midpoint = |v1: usize, v2: usize, verts: &mut Vec<Vec3>| -> usize {
                let key = if v1 < v2 { (v1, v2) } else { (v2, v1) };
                if let Some(&idx) = midpoint_cache.get(&key) {
                    return idx;
                }
                let mid = ((verts[v1] + verts[v2]) / 2.0).normalize();
                let idx = verts.len();
                verts.push(mid);
                midpoint_cache.insert(key, idx);
                idx
            };

            for face in &faces {
                let a = face[0];
                let b = face[1];
                let c = face[2];

                let ab = get_midpoint(a, b, &mut current_vertices);
                let bc = get_midpoint(b, c, &mut current_vertices);
                let ca = get_midpoint(c, a, &mut current_vertices);

                new_faces.push([a, ab, ca]);
                new_faces.push([b, bc, ab]);
                new_faces.push([c, ca, bc]);
                new_faces.push([ab, bc, ca]);
            }

            faces = new_faces;
        }

        let mut builder = MeshBuilder::new();

        for &pos in &current_vertices {
            let u = 0.5 + pos.z.atan2(pos.x) / TAU;
            let v = 0.5 - pos.y.asin() / PI;
            builder.vertex_with_normal_uv(pos * radius, pos, Vec2::new(u, v));
        }

        for face in &faces {
            builder.triangle(face[0] as u32, face[1] as u32, face[2] as u32);
        }

        builder.build()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cuboid() {
        let mesh = Cuboid::default().apply();

        assert_eq!(mesh.vertex_count(), 24);
        assert_eq!(mesh.triangle_count(), 12);
        assert!(mesh.has_normals());
        assert!(mesh.has_uvs());
    }

    #[test]
    fn test_cuboid_dimensions() {
        let mesh = Cuboid::new(2.0, 4.0, 6.0).apply();

        for pos in &mesh.positions {
            assert!(pos.x.abs() <= 1.0 + 0.001);
            assert!(pos.y.abs() <= 2.0 + 0.001);
            assert!(pos.z.abs() <= 3.0 + 0.001);
        }
    }

    #[test]
    fn test_uv_sphere() {
        let mesh = UvSphere::new(1.0, 8, 4).apply();

        assert!(mesh.vertex_count() > 0);
        assert!(mesh.triangle_count() > 0);
        assert!(mesh.has_normals());
        assert!(mesh.has_uvs());

        for normal in &mesh.normals {
            assert!((normal.length() - 1.0).abs() < 0.001);
        }
    }

    #[test]
    fn test_uv_sphere_radius() {
        let mesh = UvSphere::new(2.5, 16, 8).apply();

        for pos in &mesh.positions {
            assert!((pos.length() - 2.5).abs() < 0.001);
        }
    }

    #[test]
    fn test_cylinder() {
        let mesh = Cylinder::new(1.0, 2.0, 8).apply();

        assert!(mesh.vertex_count() > 0);
        assert!(mesh.triangle_count() > 0);
        assert!(mesh.has_normals());
        assert!(mesh.has_uvs());

        for pos in &mesh.positions {
            assert!(pos.y >= -1.0 - 0.001 && pos.y <= 1.0 + 0.001);
        }
    }

    #[test]
    fn test_cylinder_as_prism() {
        // 4 segments = square prism
        let mesh = Cylinder::new(1.0, 1.0, 4).apply();
        assert!(mesh.triangle_count() > 0);
    }

    #[test]
    fn test_cone() {
        let mesh = Cone::new(1.0, 2.0, 8).apply();

        assert!(mesh.vertex_count() > 0);
        assert!(mesh.triangle_count() > 0);
        assert!(mesh.has_normals());
        assert!(mesh.has_uvs());

        for pos in &mesh.positions {
            assert!(pos.y >= -1.0 - 0.001 && pos.y <= 1.0 + 0.001);
        }
    }

    #[test]
    fn test_cone_as_pyramid() {
        // 4 segments = square pyramid
        let mesh = Cone::new(1.0, 1.0, 4).apply();
        assert!(mesh.triangle_count() > 0);
    }

    #[test]
    fn test_torus() {
        let mesh = Torus::new(1.0, 0.25, 16, 8).apply();

        assert!(mesh.vertex_count() > 0);
        assert!(mesh.triangle_count() > 0);
        assert!(mesh.has_normals());
        assert!(mesh.has_uvs());

        for pos in &mesh.positions {
            let dist_from_axis = (pos.x * pos.x + pos.z * pos.z).sqrt();
            assert!(dist_from_axis >= 0.75 - 0.001 && dist_from_axis <= 1.25 + 0.001);
        }
    }

    #[test]
    fn test_plane() {
        let mesh = Plane::new(2.0, 3.0, 4, 6).apply();

        assert_eq!(mesh.vertex_count(), 35);
        assert_eq!(mesh.triangle_count(), 48);
        assert!(mesh.has_normals());
        assert!(mesh.has_uvs());

        for pos in &mesh.positions {
            assert!(pos.y.abs() < 0.001);
        }

        for normal in &mesh.normals {
            assert!((normal.y - 1.0).abs() < 0.001);
        }
    }

    #[test]
    fn test_plane_single_quad() {
        let mesh = Plane::default().apply();
        assert_eq!(mesh.vertex_count(), 4);
        assert_eq!(mesh.triangle_count(), 2);
    }

    #[test]
    fn test_icosphere_base() {
        let mesh = Icosphere::new(1.0, 0).apply();

        assert_eq!(mesh.vertex_count(), 12);
        assert_eq!(mesh.triangle_count(), 20);
        assert!(mesh.has_normals());
        assert!(mesh.has_uvs());

        for pos in &mesh.positions {
            assert!((pos.length() - 1.0).abs() < 0.001);
        }
    }

    #[test]
    fn test_icosphere_subdivided() {
        let mesh = Icosphere::new(1.0, 2).apply();

        assert_eq!(mesh.triangle_count(), 320);
        assert!(mesh.has_normals());

        for pos in &mesh.positions {
            assert!((pos.length() - 1.0).abs() < 0.001);
        }
    }

    #[test]
    fn test_icosphere_radius() {
        let mesh = Icosphere::new(3.0, 1).apply();

        for pos in &mesh.positions {
            assert!((pos.length() - 3.0).abs() < 0.001);
        }
    }
}
