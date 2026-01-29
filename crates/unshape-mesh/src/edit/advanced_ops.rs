use std::collections::{HashMap, HashSet};

use glam::Vec3;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::Mesh;

/// Bridges two edge loops by creating connecting faces.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct BridgeEdgeLoops {
    /// Number of segments in the bridge.
    pub segments: u32,
    /// Twist amount (in edge loop positions).
    pub twist: i32,
}

impl Default for BridgeEdgeLoops {
    fn default() -> Self {
        Self {
            segments: 1,
            twist: 0,
        }
    }
}

impl BridgeEdgeLoops {
    /// Creates a bridge operation.
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a bridge with the given segment count.
    pub fn with_segments(segments: u32) -> Self {
        Self { segments, twist: 0 }
    }

    /// Applies this operation to two edge loops.
    ///
    /// Takes two lists of vertices representing edge loops.
    pub fn apply(&self, mesh: &Mesh, loop1: &[u32], loop2: &[u32]) -> Mesh {
        bridge_edge_loops(mesh, loop1, loop2, self.segments, self.twist)
    }
}

/// Bridges two edge loops by creating connecting faces.
pub fn bridge_edge_loops(
    mesh: &Mesh,
    loop1: &[u32],
    loop2: &[u32],
    segments: u32,
    twist: i32,
) -> Mesh {
    if loop1.is_empty() || loop2.is_empty() || loop1.len() != loop2.len() {
        return mesh.clone();
    }

    let segments = segments.max(1);
    let n = loop1.len();

    let mut result = mesh.clone();

    // Apply twist to loop2
    let twist = ((twist % n as i32) + n as i32) as usize % n;
    let twisted_loop2: Vec<u32> = (0..n).map(|i| loop2[(i + twist) % n]).collect();

    if segments == 1 {
        // Direct bridge
        for i in 0..n {
            let i_next = (i + 1) % n;

            let v0 = loop1[i];
            let v1 = loop1[i_next];
            let v2 = twisted_loop2[i_next];
            let v3 = twisted_loop2[i];

            // Create quad as two triangles
            result.indices.extend_from_slice(&[v0, v1, v2]);
            result.indices.extend_from_slice(&[v0, v2, v3]);
        }
    } else {
        // Multi-segment bridge - create intermediate vertices
        let mut prev_ring = loop1.to_vec();

        for seg in 1..=segments {
            let t = seg as f32 / segments as f32;

            // Create interpolated ring
            let mut new_ring = Vec::with_capacity(n);

            for i in 0..n {
                let p1 = mesh.positions[loop1[i] as usize];
                let p2 = mesh.positions[twisted_loop2[i] as usize];
                let interp_pos = p1.lerp(p2, t);

                let new_idx = result.positions.len() as u32;
                result.positions.push(interp_pos);

                // Interpolate normals
                if !mesh.normals.is_empty() {
                    let n1 = mesh
                        .normals
                        .get(loop1[i] as usize)
                        .copied()
                        .unwrap_or(Vec3::Y);
                    let n2 = mesh
                        .normals
                        .get(twisted_loop2[i] as usize)
                        .copied()
                        .unwrap_or(Vec3::Y);
                    result.normals.push(n1.lerp(n2, t).normalize_or_zero());
                }

                // Interpolate UVs
                if !mesh.uvs.is_empty() {
                    let uv1 = mesh.uvs.get(loop1[i] as usize).copied().unwrap_or_default();
                    let uv2 = mesh
                        .uvs
                        .get(twisted_loop2[i] as usize)
                        .copied()
                        .unwrap_or_default();
                    result.uvs.push(uv1.lerp(uv2, t));
                }

                new_ring.push(if seg < segments {
                    new_idx
                } else {
                    twisted_loop2[i]
                });
            }

            // Connect previous ring to current ring
            let current_ring = if seg < segments {
                &new_ring
            } else {
                &twisted_loop2
            };

            for i in 0..n {
                let i_next = (i + 1) % n;

                let v0 = prev_ring[i];
                let v1 = prev_ring[i_next];
                let v2 = current_ring[i_next];
                let v3 = current_ring[i];

                result.indices.extend_from_slice(&[v0, v1, v2]);
                result.indices.extend_from_slice(&[v0, v2, v3]);
            }

            if seg < segments {
                prev_ring = new_ring;
            }
        }
    }

    result.compute_smooth_normals();
    result
}

/// Knife cut point on the mesh.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct KnifePoint {
    /// Face index where the cut occurs.
    pub face: u32,
    /// Barycentric coordinates within the face.
    pub barycentric: [f32; 3],
}

/// Cuts the mesh along a path.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct KnifeCut {
    /// Points defining the cut path.
    pub points: Vec<KnifePoint>,
}

impl Default for KnifeCut {
    fn default() -> Self {
        Self { points: Vec::new() }
    }
}

impl KnifeCut {
    /// Creates a new knife cut operation.
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a cut point.
    pub fn add_point(&mut self, face: u32, barycentric: [f32; 3]) {
        self.points.push(KnifePoint { face, barycentric });
    }

    /// Applies this operation to a mesh.
    pub fn apply(&self, mesh: &Mesh) -> Mesh {
        knife_cut(mesh, &self.points)
    }
}

/// Cuts the mesh along a path defined by points on faces.
pub fn knife_cut(mesh: &Mesh, points: &[KnifePoint]) -> Mesh {
    if points.len() < 2 {
        return mesh.clone();
    }

    let mut result = mesh.clone();

    // Simple implementation: add vertices at cut points and split affected faces
    let mut face_cuts: HashMap<u32, Vec<(u32, [f32; 3])>> = HashMap::new();

    for (idx, point) in points.iter().enumerate() {
        face_cuts
            .entry(point.face)
            .or_default()
            .push((idx as u32, point.barycentric));
    }

    // Process each face with cuts
    let mut new_indices = Vec::new();
    let mut processed_faces: HashSet<u32> = HashSet::new();

    for (&face_idx, cuts) in &face_cuts {
        processed_faces.insert(face_idx);

        let base = face_idx as usize * 3;
        if base + 2 >= mesh.indices.len() {
            continue;
        }

        let i0 = mesh.indices[base];
        let i1 = mesh.indices[base + 1];
        let i2 = mesh.indices[base + 2];

        let p0 = mesh.positions[i0 as usize];
        let p1 = mesh.positions[i1 as usize];
        let p2 = mesh.positions[i2 as usize];

        // Create vertices for cut points
        let mut cut_vertex_indices = Vec::new();
        for &(_, bary) in cuts {
            let cut_pos = p0 * bary[0] + p1 * bary[1] + p2 * bary[2];
            let cut_idx = result.positions.len() as u32;
            result.positions.push(cut_pos);

            // Interpolate normal
            if !mesh.normals.is_empty() {
                let n0 = mesh.normals.get(i0 as usize).copied().unwrap_or(Vec3::Y);
                let n1 = mesh.normals.get(i1 as usize).copied().unwrap_or(Vec3::Y);
                let n2 = mesh.normals.get(i2 as usize).copied().unwrap_or(Vec3::Y);
                result
                    .normals
                    .push((n0 * bary[0] + n1 * bary[1] + n2 * bary[2]).normalize_or_zero());
            }

            // Interpolate UV
            if !mesh.uvs.is_empty() {
                let uv0 = mesh.uvs.get(i0 as usize).copied().unwrap_or_default();
                let uv1 = mesh.uvs.get(i1 as usize).copied().unwrap_or_default();
                let uv2 = mesh.uvs.get(i2 as usize).copied().unwrap_or_default();
                result
                    .uvs
                    .push(uv0 * bary[0] + uv1 * bary[1] + uv2 * bary[2]);
            }

            cut_vertex_indices.push(cut_idx);
        }

        // Simple triangulation: fan from first vertex through cut points
        if cut_vertex_indices.len() == 1 {
            // Single cut point - create 3 triangles
            let cut = cut_vertex_indices[0];
            new_indices.extend_from_slice(&[i0, i1, cut]);
            new_indices.extend_from_slice(&[i1, i2, cut]);
            new_indices.extend_from_slice(&[i2, i0, cut]);
        } else {
            // Multiple cut points - more complex triangulation needed
            // For now, just keep original face (proper implementation would be complex)
            new_indices.extend_from_slice(&[i0, i1, i2]);
        }
    }

    // Copy unaffected faces
    for face_idx in 0..mesh.triangle_count() {
        if !processed_faces.contains(&(face_idx as u32)) {
            let base = face_idx * 3;
            new_indices.extend_from_slice(&mesh.indices[base..base + 3]);
        }
    }

    result.indices = new_indices;
    result
}
