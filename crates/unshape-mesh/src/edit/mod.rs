//! Selection-aware mesh editing operations.
//!
//! These operations work with [`MeshSelection`] to modify only selected elements,
//! enabling Blender-style constructive modeling workflows.
//!
//! # Operations
//!
//! ## Simple
//! - [`DeleteFaces`] - Remove selected faces from mesh
//! - [`TransformVertices`] - Apply transformation to selected vertices
//! - [`TriangulateFaces`] - Convert selected quads/ngons to triangles
//! - [`PokeFaces`] - Add vertex at face center, create triangle fan
//!
//! ## Medium
//! - [`ScaleFaces`] - Scale faces around their centers
//! - [`SmoothVertices`] - Laplacian smoothing on selected vertices
//! - [`MergeVertices`] - Collapse vertices to single point
//! - [`SplitEdges`] - Duplicate edges for hard edges
//! - [`CreaseEdges`] - Mark edges for subdivision weighting
//!
//! ## Complex
//! - [`ExtrudeFaces`] - Extrude selected faces along normals
//! - [`InsetFaces`] - Inset selected faces toward centers
//! - [`SubdivideFaces`] - Subdivide only selected faces
//! - [`SlideEdges`] - Slide edges along adjacent faces
//! - [`RipVertices`] - Disconnect vertices from adjacent faces
//!
//! ## Advanced
//! - [`BridgeEdgeLoops`] - Connect two edge loops with faces
//! - [`KnifeCut`] - Cut mesh along arbitrary path

mod advanced_ops;
mod edge_ops;
mod face_ops;
mod vertex_ops;

pub use advanced_ops::*;
pub use edge_ops::*;
pub use face_ops::*;
pub use vertex_ops::*;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Cuboid;
    use crate::Mesh;
    use crate::selection::MeshSelection;
    use glam::{Vec2, Vec3};

    fn make_single_triangle() -> Mesh {
        Mesh {
            positions: vec![
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(0.5, 1.0, 0.0),
            ],
            normals: vec![Vec3::Z; 3],
            uvs: vec![Vec2::ZERO; 3],
            indices: vec![0, 1, 2],
        }
    }

    fn make_two_triangles() -> Mesh {
        Mesh {
            positions: vec![
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(0.5, 1.0, 0.0),
                Vec3::new(0.5, -1.0, 0.0),
            ],
            normals: vec![Vec3::Z; 4],
            uvs: vec![Vec2::ZERO; 4],
            indices: vec![0, 1, 2, 0, 3, 1],
        }
    }

    #[test]
    fn test_delete_faces_single() {
        let mesh = make_two_triangles();
        let mut selection = MeshSelection::new();
        selection.select_face(0);

        let result = delete_faces(&mesh, &selection, true);

        assert_eq!(result.triangle_count(), 1);
        // Should only have vertices needed for remaining face
        assert!(result.vertex_count() <= 4);
    }

    #[test]
    fn test_delete_faces_all() {
        let mesh = make_two_triangles();
        let mut selection = MeshSelection::new();
        selection.select_all_faces(&mesh);

        let result = delete_faces(&mesh, &selection, true);

        assert_eq!(result.triangle_count(), 0);
        assert_eq!(result.vertex_count(), 0);
    }

    #[test]
    fn test_transform_vertices() {
        let mesh = make_single_triangle();
        let mut selection = MeshSelection::new();
        selection.select_vertex(0);

        let transform = TransformVertices::translate(Vec3::new(1.0, 0.0, 0.0));
        let result = transform.apply(&mesh, &selection);

        assert_eq!(result.positions[0], Vec3::new(1.0, 0.0, 0.0));
        assert_eq!(result.positions[1], mesh.positions[1]); // Unchanged
    }

    #[test]
    fn test_poke_faces() {
        let mesh = make_single_triangle();
        let mut selection = MeshSelection::new();
        selection.select_face(0);

        let result = poke_faces(&mesh, &selection, 0.0);

        // 1 triangle becomes 3 triangles
        assert_eq!(result.triangle_count(), 3);
        // Added 1 center vertex
        assert_eq!(result.vertex_count(), 4);
    }

    #[test]
    fn test_scale_faces() {
        let cube = Cuboid::default().apply();
        let mut selection = MeshSelection::new();
        selection.select_face(0);

        let result = scale_faces(&cube, &selection, 0.5);

        // Should have more vertices due to face duplication
        assert!(result.vertex_count() >= cube.vertex_count());
    }

    #[test]
    fn test_smooth_vertices() {
        let mesh = make_two_triangles();
        let mut selection = MeshSelection::new();
        selection.select_vertex(0);

        let result = smooth_vertices(&mesh, &selection, 0.5, 1);

        // Position should have moved toward neighbors
        assert_ne!(result.positions[0], mesh.positions[0]);
        // Other vertices unchanged
        assert_eq!(result.positions[2], mesh.positions[2]);
    }

    #[test]
    fn test_merge_vertices() {
        let mesh = make_two_triangles();
        let mut selection = MeshSelection::new();
        selection.select_vertex(0);
        selection.select_vertex(1);

        let result = merge_vertices(&mesh, &selection, MergeMode::Center, Vec3::ZERO);

        // Should have fewer vertices
        assert!(result.vertex_count() < mesh.vertex_count());
    }

    #[test]
    fn test_extrude_faces() {
        let mesh = make_single_triangle();
        let mut selection = MeshSelection::new();
        selection.select_face(0);

        let result = extrude_faces(&mesh, &selection, 1.0);

        // Should have more vertices and triangles
        assert!(result.vertex_count() > mesh.vertex_count());
        assert!(result.triangle_count() > mesh.triangle_count());
    }

    #[test]
    fn test_inset_faces_individual() {
        let mesh = make_single_triangle();
        let mut selection = MeshSelection::new();
        selection.select_face(0);

        let result = inset_faces_individual(&mesh, &selection, 0.3);

        // 1 inner + 3 bridge quads (6 triangles) = 7 triangles
        assert_eq!(result.triangle_count(), 7);
    }

    #[test]
    fn test_subdivide_faces() {
        let mesh = make_single_triangle();
        let mut selection = MeshSelection::new();
        selection.select_face(0);

        let result = subdivide_faces(&mesh, &selection, 1);

        // 1 triangle becomes 4 triangles
        assert_eq!(result.triangle_count(), 4);
    }

    #[test]
    fn test_rip_vertices() {
        let mesh = make_two_triangles();
        let mut selection = MeshSelection::new();
        selection.select_vertex(0); // Shared vertex

        let result = rip_vertices(&mesh, &selection);

        // Vertex 0 should be duplicated
        assert!(result.vertex_count() > mesh.vertex_count());
    }

    #[test]
    fn test_bridge_edge_loops() {
        // Create two simple triangular loops (3 vertices each)
        let test_mesh = Mesh {
            positions: vec![
                // Loop 1 (at z=0)
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(0.5, 1.0, 0.0),
                // Loop 2 (at z=1)
                Vec3::new(0.0, 0.0, 1.0),
                Vec3::new(1.0, 0.0, 1.0),
                Vec3::new(0.5, 1.0, 1.0),
            ],
            normals: vec![Vec3::Z; 6],
            uvs: vec![Vec2::ZERO; 6],
            indices: vec![],
        };

        let loop1 = vec![0, 1, 2];
        let loop2 = vec![3, 4, 5];

        let result = bridge_edge_loops(&test_mesh, &loop1, &loop2, 1, 0);

        // Should create 3 quads = 6 triangles
        assert_eq!(result.triangle_count(), 6);
    }
}
