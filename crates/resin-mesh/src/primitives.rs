//! Mesh primitives.
//!
//! Operations are serializable structs with `apply` methods. Free functions
//! are sugar that delegate to these ops. See `docs/design/ops-as-values.md`.
//!
//! # Design Note
//!
//! These primitives are hardcoded shape generators - each struct (Cylinder, Cone,
//! Torus, etc.) directly encodes geometry generation logic. This is pragmatic but
//! not ideal from a "generative mindset" perspective.
//!
//! Ideally, primitives would be compositions of more fundamental operations or
//! expressed through a general-purpose expression system. However, Dew (our
//! expression language) doesn't support generating vertex lists, so there's no
//! viable alternative currently.
//!
//! If a more expressive primitive system emerges later (e.g., SDF-based generation,
//! parametric surface evaluation), these can become sugar over it. For now,
//! serializable hardcoded shapes beats non-serializable functions.

use glam::{Vec2, Vec3};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
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

// ============================================================================
// Cylinder
// ============================================================================

/// Generates a cylinder mesh centered at the origin.
///
/// The cylinder extends from -height/2 to +height/2 on the Y axis.
/// Includes top and bottom caps with proper normals.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dynop", derive(rhizome_resin_op::Op))]
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

        // Bottom cap triangles (wind clockwise when viewed from below)
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

        // Top cap triangles (wind counter-clockwise when viewed from above)
        for i in 0..segments {
            let next = (i + 1) % segments;
            builder.triangle(top_center, top_ring[i as usize], top_ring[next as usize]);
        }

        // Side surface - need separate vertices for different normals
        for i in 0..=segments {
            let angle = TAU * (i as f32) / (segments as f32);
            let x = angle.cos();
            let z = angle.sin();
            let normal = Vec3::new(x, 0.0, z);
            let u = i as f32 / segments as f32;

            // Bottom vertex of side
            builder.vertex_with_normal_uv(
                Vec3::new(x * radius, -half_height, z * radius),
                normal,
                Vec2::new(u, 0.0),
            );
            // Top vertex of side
            builder.vertex_with_normal_uv(
                Vec3::new(x * radius, half_height, z * radius),
                normal,
                Vec2::new(u, 1.0),
            );
        }

        // Side quads
        let side_start = 1 + segments + 1 + segments; // after caps
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

/// Creates a cylinder centered at the origin.
///
/// The cylinder extends from -height/2 to +height/2 on the Y axis.
/// Includes top and bottom caps with proper normals.
///
/// # Arguments
/// * `radius` - Radius of the cylinder
/// * `height` - Height of the cylinder
/// * `segments` - Number of divisions around the circumference. Minimum 3.
pub fn cylinder(radius: f32, height: f32, segments: u32) -> Mesh {
    Cylinder::new(radius, height, segments).apply()
}

// ============================================================================
// Cone
// ============================================================================

/// Generates a cone mesh centered at the origin.
///
/// The cone has its base at -height/2 and apex at +height/2 on the Y axis.
/// Includes a bottom cap with proper normals.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dynop", derive(rhizome_resin_op::Op))]
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
        // For a cone, the normal at each point tilts outward and upward
        // The slope angle is atan(radius / height)
        let slope = (radius / self.height).atan();
        let normal_y = slope.sin();
        let normal_xz_scale = slope.cos();

        for i in 0..=segments {
            let angle = TAU * (i as f32) / (segments as f32);
            let x = angle.cos();
            let z = angle.sin();
            let normal = Vec3::new(x * normal_xz_scale, normal_y, z * normal_xz_scale).normalize();
            let u = i as f32 / segments as f32;

            // Base vertex
            builder.vertex_with_normal_uv(
                Vec3::new(x * radius, -half_height, z * radius),
                normal,
                Vec2::new(u, 0.0),
            );
            // Apex vertex (same position, but separate for UV continuity)
            builder.vertex_with_normal_uv(
                Vec3::new(0.0, half_height, 0.0),
                normal,
                Vec2::new(u, 1.0),
            );
        }

        // Side triangles
        let side_start = 1 + segments; // after bottom cap
        for i in 0..segments {
            let bl = side_start + i * 2;
            let apex = bl + 1;
            let br = bl + 2;
            builder.triangle(bl, br, apex);
        }

        builder.build()
    }
}

/// Creates a cone centered at the origin.
///
/// The cone has its base at -height/2 and apex at +height/2 on the Y axis.
/// Includes a bottom cap with proper normals.
///
/// # Arguments
/// * `radius` - Radius of the base
/// * `height` - Height of the cone
/// * `segments` - Number of divisions around the circumference. Minimum 3.
pub fn cone(radius: f32, height: f32, segments: u32) -> Mesh {
    Cone::new(radius, height, segments).apply()
}

// ============================================================================
// Torus
// ============================================================================

/// Generates a torus (donut shape) mesh centered at the origin.
///
/// The torus lies in the XZ plane with the hole along the Y axis.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dynop", derive(rhizome_resin_op::Op))]
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

        // Generate vertices
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

                // Position on torus surface
                let x = (major_radius + minor_radius * cos_minor) * cos_major;
                let y = minor_radius * sin_minor;
                let z = (major_radius + minor_radius * cos_minor) * sin_major;

                // Normal points from tube center to surface
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

        // Generate quads
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

/// Creates a torus (donut shape) centered at the origin.
///
/// The torus lies in the XZ plane with the hole along the Y axis.
///
/// # Arguments
/// * `major_radius` - Distance from center of torus to center of tube
/// * `minor_radius` - Radius of the tube
/// * `major_segments` - Divisions around the main ring. Minimum 3.
/// * `minor_segments` - Divisions around the tube cross-section. Minimum 3.
pub fn torus(
    major_radius: f32,
    minor_radius: f32,
    major_segments: u32,
    minor_segments: u32,
) -> Mesh {
    Torus::new(major_radius, minor_radius, major_segments, minor_segments).apply()
}

// ============================================================================
// Plane
// ============================================================================

/// Generates a flat plane mesh in the XZ plane, centered at the origin.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dynop", derive(rhizome_resin_op::Op))]
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

        // Generate vertices
        for iz in 0..=subdivisions_z {
            let v = iz as f32 / subdivisions_z as f32;
            let z = -half_depth + self.depth * v;

            for ix in 0..=subdivisions_x {
                let u = ix as f32 / subdivisions_x as f32;
                let x = -half_width + self.width * u;

                builder.vertex_with_normal_uv(Vec3::new(x, 0.0, z), Vec3::Y, Vec2::new(u, v));
            }
        }

        // Generate quads
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

/// Creates a flat plane in the XZ plane, centered at the origin.
///
/// # Arguments
/// * `width` - Size along the X axis
/// * `depth` - Size along the Z axis
/// * `subdivisions_x` - Number of divisions along X. Minimum 1.
/// * `subdivisions_z` - Number of divisions along Z. Minimum 1.
pub fn plane(width: f32, depth: f32, subdivisions_x: u32, subdivisions_z: u32) -> Mesh {
    Plane::new(width, depth, subdivisions_x, subdivisions_z).apply()
}

// ============================================================================
// Icosphere
// ============================================================================

/// Generates an icosphere (geodesic sphere) mesh centered at the origin with radius 1.
///
/// Starts from an icosahedron and subdivides each face recursively.
/// Produces more uniform triangle distribution than UV sphere.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dynop", derive(rhizome_resin_op::Op))]
#[cfg_attr(feature = "dynop", op(input = (), output = Mesh))]
pub struct Icosphere {
    /// Number of subdivision iterations. 0 = icosahedron (20 faces).
    /// Each level multiplies face count by 4. Maximum 5 (practical limit).
    pub subdivisions: u32,
}

impl Default for Icosphere {
    fn default() -> Self {
        Self { subdivisions: 2 }
    }
}

impl Icosphere {
    /// Creates a new icosphere with the given subdivision level.
    pub fn new(subdivisions: u32) -> Self {
        Self { subdivisions }
    }

    /// Generates the icosphere mesh.
    pub fn apply(&self) -> Mesh {
        let subdivisions = self.subdivisions.min(5); // Prevent excessive geometry

        // Golden ratio
        let phi = (1.0 + 5.0_f32.sqrt()) / 2.0;
        let len = (1.0 + phi * phi).sqrt();
        let a = 1.0 / len;
        let b = phi / len;

        // Icosahedron vertices (normalized to unit sphere)
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

        // Icosahedron faces (20 triangles)
        let mut faces: Vec<[usize; 3]> = vec![
            // 5 faces around vertex 0
            [0, 11, 5],
            [0, 5, 1],
            [0, 1, 7],
            [0, 7, 10],
            [0, 10, 11],
            // 5 adjacent faces
            [1, 5, 9],
            [5, 11, 4],
            [11, 10, 2],
            [10, 7, 6],
            [7, 1, 8],
            // 5 faces around vertex 3
            [3, 9, 4],
            [3, 4, 2],
            [3, 2, 6],
            [3, 6, 8],
            [3, 8, 9],
            // 5 adjacent faces
            [4, 9, 5],
            [2, 4, 11],
            [6, 2, 10],
            [8, 6, 7],
            [9, 8, 1],
        ];

        let mut current_vertices = vertices.to_vec();

        // Subdivide
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

        // Build mesh
        let mut builder = MeshBuilder::new();

        // Add vertices with normals (position = normal for unit sphere)
        // Calculate UV from spherical coordinates
        for &pos in &current_vertices {
            let u = 0.5 + pos.z.atan2(pos.x) / TAU;
            let v = 0.5 - pos.y.asin() / PI;
            builder.vertex_with_normal_uv(pos, pos, Vec2::new(u, v));
        }

        // Add faces
        for face in &faces {
            builder.triangle(face[0] as u32, face[1] as u32, face[2] as u32);
        }

        builder.build()
    }
}

/// Creates an icosphere (geodesic sphere) centered at the origin with radius 1.
///
/// Starts from an icosahedron and subdivides each face recursively.
/// Produces more uniform triangle distribution than UV sphere.
///
/// # Arguments
/// * `subdivisions` - Number of subdivision iterations. 0 = icosahedron (20 faces).
///   Each level multiplies face count by 4. Maximum 5 (practical limit).
pub fn icosphere(subdivisions: u32) -> Mesh {
    Icosphere::new(subdivisions).apply()
}

// ============================================================================
// Pyramid
// ============================================================================

/// Generates a pyramid mesh with a square base, centered at the origin.
///
/// The base is at -height/2 and the apex at +height/2 on the Y axis.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dynop", derive(rhizome_resin_op::Op))]
#[cfg_attr(feature = "dynop", op(input = (), output = Mesh))]
pub struct Pyramid {
    /// Side length of the square base.
    pub base_size: f32,
    /// Height from base to apex.
    pub height: f32,
}

impl Default for Pyramid {
    fn default() -> Self {
        Self {
            base_size: 2.0,
            height: 2.0,
        }
    }
}

impl Pyramid {
    /// Creates a new pyramid with the given dimensions.
    pub fn new(base_size: f32, height: f32) -> Self {
        Self { base_size, height }
    }

    /// Generates the pyramid mesh.
    pub fn apply(&self) -> Mesh {
        let mut builder = MeshBuilder::new();

        let half_base = self.base_size / 2.0;
        let half_height = self.height / 2.0;

        // Base corners
        let corners = [
            Vec3::new(-half_base, -half_height, -half_base),
            Vec3::new(half_base, -half_height, -half_base),
            Vec3::new(half_base, -half_height, half_base),
            Vec3::new(-half_base, -half_height, half_base),
        ];
        let apex = Vec3::new(0.0, half_height, 0.0);

        // Base (facing down)
        let b0 = builder.vertex_with_normal_uv(corners[0], Vec3::NEG_Y, Vec2::new(0.0, 0.0));
        let b1 = builder.vertex_with_normal_uv(corners[1], Vec3::NEG_Y, Vec2::new(1.0, 0.0));
        let b2 = builder.vertex_with_normal_uv(corners[2], Vec3::NEG_Y, Vec2::new(1.0, 1.0));
        let b3 = builder.vertex_with_normal_uv(corners[3], Vec3::NEG_Y, Vec2::new(0.0, 1.0));
        builder.quad(b0, b1, b2, b3);

        // Side faces - each needs its own vertices for correct normals
        let side_indices = [(0, 1), (1, 2), (2, 3), (3, 0)];
        for &(c1, c2) in &side_indices {
            let p1 = corners[c1];
            let p2 = corners[c2];

            // Calculate face normal
            let edge1 = p2 - p1;
            let edge2 = apex - p1;
            let normal = edge1.cross(edge2).normalize();

            let v0 = builder.vertex_with_normal_uv(p1, normal, Vec2::new(0.0, 0.0));
            let v1 = builder.vertex_with_normal_uv(p2, normal, Vec2::new(1.0, 0.0));
            let v2 = builder.vertex_with_normal_uv(apex, normal, Vec2::new(0.5, 1.0));

            builder.triangle(v0, v1, v2);
        }

        builder.build()
    }
}

/// Creates a pyramid with a square base, centered at the origin.
///
/// The base is at -height/2 and the apex at +height/2 on the Y axis.
///
/// # Arguments
/// * `base_size` - Side length of the square base
/// * `height` - Height from base to apex
pub fn pyramid(base_size: f32, height: f32) -> Mesh {
    Pyramid::new(base_size, height).apply()
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

    #[test]
    fn test_cylinder() {
        let mesh = cylinder(1.0, 2.0, 8);

        assert!(mesh.vertex_count() > 0);
        assert!(mesh.triangle_count() > 0);
        assert!(mesh.has_normals());
        assert!(mesh.has_uvs());

        // Check height bounds
        for pos in &mesh.positions {
            assert!(pos.y >= -1.0 - 0.001 && pos.y <= 1.0 + 0.001);
        }
    }

    #[test]
    fn test_cylinder_struct() {
        let cyl = Cylinder::new(1.0, 2.0, 8);
        let mesh = cyl.apply();
        assert!(mesh.vertex_count() > 0);
    }

    #[test]
    fn test_cylinder_minimum_segments() {
        let mesh = cylinder(1.0, 1.0, 1); // Should clamp to 3
        assert!(mesh.triangle_count() >= 6); // At least 3 sides + 3 cap triangles * 2
    }

    #[test]
    fn test_cone() {
        let mesh = cone(1.0, 2.0, 8);

        assert!(mesh.vertex_count() > 0);
        assert!(mesh.triangle_count() > 0);
        assert!(mesh.has_normals());
        assert!(mesh.has_uvs());

        // Check height bounds
        for pos in &mesh.positions {
            assert!(pos.y >= -1.0 - 0.001 && pos.y <= 1.0 + 0.001);
        }
    }

    #[test]
    fn test_cone_struct() {
        let c = Cone::default();
        let mesh = c.apply();
        assert!(mesh.vertex_count() > 0);
    }

    #[test]
    fn test_torus() {
        let mesh = torus(1.0, 0.25, 16, 8);

        assert!(mesh.vertex_count() > 0);
        assert!(mesh.triangle_count() > 0);
        assert!(mesh.has_normals());
        assert!(mesh.has_uvs());

        // Check that all points are within expected bounds
        for pos in &mesh.positions {
            let dist_from_axis = (pos.x * pos.x + pos.z * pos.z).sqrt();
            // Should be between (major - minor) and (major + minor) from Y axis
            assert!(dist_from_axis >= 0.75 - 0.001 && dist_from_axis <= 1.25 + 0.001);
        }
    }

    #[test]
    fn test_torus_struct() {
        let t = Torus::default();
        let mesh = t.apply();
        assert!(mesh.vertex_count() > 0);
    }

    #[test]
    fn test_plane() {
        let mesh = plane(2.0, 3.0, 4, 6);

        // (4+1) * (6+1) = 35 vertices
        assert_eq!(mesh.vertex_count(), 35);
        // 4 * 6 * 2 = 48 triangles
        assert_eq!(mesh.triangle_count(), 48);
        assert!(mesh.has_normals());
        assert!(mesh.has_uvs());

        // All vertices should be at y=0
        for pos in &mesh.positions {
            assert!((pos.y).abs() < 0.001);
        }

        // All normals should point up
        for normal in &mesh.normals {
            assert!((normal.y - 1.0).abs() < 0.001);
        }
    }

    #[test]
    fn test_plane_struct() {
        let p = Plane::default();
        let mesh = p.apply();
        assert_eq!(mesh.vertex_count(), 4);
    }

    #[test]
    fn test_plane_single_quad() {
        let mesh = plane(1.0, 1.0, 1, 1);
        assert_eq!(mesh.vertex_count(), 4);
        assert_eq!(mesh.triangle_count(), 2);
    }

    #[test]
    fn test_icosphere_base() {
        let mesh = icosphere(0);

        // Icosahedron has 12 vertices and 20 faces
        assert_eq!(mesh.vertex_count(), 12);
        assert_eq!(mesh.triangle_count(), 20);
        assert!(mesh.has_normals());
        assert!(mesh.has_uvs());

        // All vertices should be on unit sphere
        for pos in &mesh.positions {
            assert!((pos.length() - 1.0).abs() < 0.001);
        }
    }

    #[test]
    fn test_icosphere_struct() {
        let ico = Icosphere::default();
        let mesh = ico.apply();
        assert!(mesh.vertex_count() > 12); // More than base icosahedron
    }

    #[test]
    fn test_icosphere_subdivided() {
        let mesh = icosphere(2);

        // Each subdivision multiplies faces by 4: 20 -> 80 -> 320
        assert_eq!(mesh.triangle_count(), 320);
        assert!(mesh.has_normals());

        // All vertices should still be on unit sphere
        for pos in &mesh.positions {
            assert!((pos.length() - 1.0).abs() < 0.001);
        }
    }

    #[test]
    fn test_pyramid() {
        let mesh = pyramid(2.0, 3.0);

        assert!(mesh.vertex_count() > 0);
        assert!(mesh.triangle_count() > 0);
        assert!(mesh.has_normals());
        assert!(mesh.has_uvs());

        // Check height bounds
        for pos in &mesh.positions {
            assert!(pos.y >= -1.5 - 0.001 && pos.y <= 1.5 + 0.001);
        }
    }

    #[test]
    fn test_pyramid_struct() {
        let p = Pyramid::default();
        let mesh = p.apply();
        assert!(mesh.vertex_count() > 0);
    }
}
