//! 3D mesh generation for resin.
//!
//! Provides mesh primitives and operations for procedural 3D geometry.

mod ao;
mod architecture;
mod bevel;
mod boolean;
mod curvature;
mod curve_mesh;
mod decimate;
mod edit;
pub mod geodesic;
mod halfedge;
mod lattice;
mod lod;
mod loft;
mod loops;
mod marching_cubes;
mod mesh;
mod morph;
mod navmesh;
mod obj;
mod ops;
mod primitives;
mod remesh;
pub mod repair;
mod sdf;
mod selection;
mod subdivision;
mod terrain;
mod topology;
mod uv;
mod weights;

pub use ao::{
    AoAccelerator, AoBakeConfig, AoTexture, BakeAo, ao_to_vertex_colors, bake_ao_texture,
    bake_ao_vertices, blur_ao_texture,
};
pub use architecture::{
    Building, DoorConfig, FloorPoint, RoofConfig, RoofStyle, WallConfig, WindowConfig,
    generate_building, generate_stairs, generate_wall_with_door, generate_wall_with_window,
    generate_walls,
};
pub use bevel::{
    Bevel, BevelConfig, bevel_edges, bevel_mesh_edges, bevel_mesh_vertices, bevel_vertices,
};
pub use boolean::{boolean_intersect, boolean_subtract, boolean_union};
pub use curvature::{
    CurvatureResult, compute_curvature, gaussian_curvature, mean_curvature, principal_curvatures,
};
pub use curve_mesh::{
    ExtrudeProfile, ExtrudeProfileConfig, Revolve, RevolveConfig, Sweep, SweepConfig,
    extrude_profile, extrude_profile_with_config, revolve_profile, revolve_profile_with_config,
    sweep_profile, sweep_profile_with_config,
};
pub use decimate::{Decimate, DecimateConfig, decimate};
pub use edit::{
    // Advanced operations
    BridgeEdgeLoops,
    CreaseEdges,
    // Simple operations
    DeleteFaces,
    EdgeCreases,
    // Complex operations
    ExtrudeFaces,
    InsetFaces,
    KnifeCut,
    KnifePoint,
    MergeMode,
    MergeVertices,
    PokeFaces,
    RipVertices,
    // Medium operations
    ScaleFaces,
    SlideEdges,
    SmoothVertices,
    SplitEdges,
    SubdivideFaces,
    TransformVertices,
    TriangulateFaces,
    bridge_edge_loops,
    delete_faces,
    extrude_faces,
    inset_faces_individual,
    inset_faces_region,
    knife_cut,
    merge_vertices,
    poke_faces,
    rip_vertices,
    scale_faces,
    slide_edges,
    smooth_vertices,
    split_edges,
    subdivide_faces,
    transform_vertices,
    transform_vertices_soft,
};
pub use halfedge::{
    Face, FaceId, HalfEdge, HalfEdgeId, HalfEdgeMesh, Vertex as HEVertex, VertexId,
};
pub use lattice::{
    Lattice, LatticeDeformConfig, bend_lattice, lattice_deform, lattice_deform_point,
    lattice_deform_points, lattice_deform_with_config, scale_lattice, taper_lattice, twist_lattice,
};
pub use lod::{
    GenerateLodChain, LodChain, LodConfig, LodLevel, estimate_bounding_radius, generate_lod_chain,
    generate_lod_chain_with_targets,
};
pub use loft::{
    Loft, LoftConfig, circle_profile, loft, loft_along_path, rect_profile, star_profile,
};
pub use loops::{
    edges_to_faces, edges_to_vertices, grow_edge_selection, loop_cut, select_boundary_edges,
    select_edge_loop, select_edge_ring, select_face_edges, select_vertex_edges,
};
pub use marching_cubes::{MarchingCubes, MarchingCubesConfig, marching_cubes};
pub use mesh::{Mesh, MeshBuilder};
pub use morph::{
    MorphTarget, MorphTargetSet, MorphWeights, apply_morph_targets,
    apply_morph_targets_with_normals, blend_positions,
};
pub use navmesh::{
    GenerateNavMesh, NavMesh, NavMeshConfig, NavPath, NavPolygon, create_grid_navmesh, find_path,
    smooth_path,
};
pub use obj::{ObjError, export_obj, export_obj_with_name, import_obj, import_obj_from_reader};
pub use ops::{
    Extrude, ExtrudeConfig, Inset, InsetConfig, NormalMode, Smooth, SmoothConfig, extrude,
    extrude_with_config, flip_normals, inset, inset_with_config, make_double_sided,
    recalculate_normals, smooth, smooth_taubin, smooth_with_config, solidify, split_faces,
    weld_vertices,
};
pub use primitives::{Cone, Cuboid, Cylinder, Icosphere, Plane, Torus, UvSphere};
pub use remesh::{
    QuadMesh, Quadify, QuadifyConfig, Remesh, RemeshConfig, average_edge_length, isotropic_remesh,
    quadify,
};
pub use sdf::{GenerateSdf, SdfConfig, SdfGrid, mesh_to_sdf, mesh_to_sdf_fast, raymarch};
pub use selection::{Edge, Falloff, MeshSelection, SelectionMode, SoftSelection};
pub use subdivision::{
    CatmullClark, subdivide_catmull_clark, subdivide_linear, subdivide_loop, subdivide_loop_n,
};
pub use terrain::{CombinedErosion, Heightfield, HydraulicErosion, ThermalErosion};
pub use topology::{
    TopologyInfo, analyze_topology, connected_components, euler_characteristic,
    extract_boundary_loops, find_boundary_edges, find_boundary_vertices, find_non_manifold_edges,
    genus, is_closed, is_manifold,
};
pub use uv::{
    AtlasPackConfig, AtlasPackResult, BoxConfig, CylindricalConfig, PackedChart, ProjectionAxis,
    SphericalConfig, UvChart, apply_atlas_pack, find_uv_islands, flip_u, flip_v, normalize_uvs,
    pack_mesh_uvs, pack_multi_mesh_uvs, pack_uv_charts, project_box, project_box_per_face,
    project_cylindrical, project_planar, project_planar_axis, project_spherical, rotate_uvs,
    scale_uvs, transform_uvs, translate_uvs,
};
pub use weights::{
    HeatDiffusionConfig, VertexWeights, blur_weights, compute_automatic_weights, gradient_weights,
    heat_diffusion, invert_weights, limit_influences, radial_weights, scale_weights,
    smooth_influence, smooth_weights, transfer_weights_nearest,
};

/// Registers all mesh operations with an [`OpRegistry`].
///
/// Call this to enable deserialization of mesh ops from saved pipelines.
#[cfg(feature = "dynop")]
pub fn register_ops(registry: &mut unshape_op::OpRegistry) {
    registry.register_type::<BakeAo>("resin::BakeAo");
    registry.register_type::<Bevel>("resin::Bevel");
    registry.register_type::<Decimate>("resin::Decimate");
    registry.register_type::<Extrude>("resin::Extrude");
    registry.register_type::<ExtrudeProfile>("resin::ExtrudeProfile");
    registry.register_type::<GenerateLodChain>("resin::GenerateLodChain");
    registry.register_type::<GenerateNavMesh>("resin::GenerateNavMesh");
    registry.register_type::<GenerateSdf>("resin::GenerateSdf");
    registry.register_type::<Inset>("resin::Inset");
    registry.register_type::<Loft>("resin::Loft");
    registry.register_type::<MarchingCubes>("resin::MarchingCubes");
    registry.register_type::<Quadify>("resin::Quadify");
    registry.register_type::<Remesh>("resin::Remesh");
    registry.register_type::<Revolve>("resin::Revolve");
    registry.register_type::<Smooth>("resin::Smooth");
    registry.register_type::<Sweep>("resin::Sweep");

    // Selection-aware editing operations
    registry.register_type::<DeleteFaces>("resin::DeleteFaces");
    registry.register_type::<TransformVertices>("resin::TransformVertices");
    registry.register_type::<TriangulateFaces>("resin::TriangulateFaces");
    registry.register_type::<PokeFaces>("resin::PokeFaces");
    registry.register_type::<ScaleFaces>("resin::ScaleFaces");
    registry.register_type::<SmoothVertices>("resin::SmoothVertices");
    registry.register_type::<MergeVertices>("resin::MergeVertices");
    registry.register_type::<SplitEdges>("resin::SplitEdges");
    registry.register_type::<ExtrudeFaces>("resin::ExtrudeFaces");
    registry.register_type::<InsetFaces>("resin::InsetFaces");
    registry.register_type::<SubdivideFaces>("resin::SubdivideFaces");
    registry.register_type::<SlideEdges>("resin::SlideEdges");
    registry.register_type::<RipVertices>("resin::RipVertices");
}

// ============================================================================
// Invariant tests
// ============================================================================

/// Invariant tests for mesh operations.
///
/// These tests verify mathematical and topological properties that should hold
/// for mesh primitives and operations. Run with:
///
/// ```sh
/// cargo test -p unshape-mesh --features invariant-tests
/// ```
#[cfg(all(test, feature = "invariant-tests"))]
mod invariant_tests {
    use super::*;
    use glam::Vec3;

    // ========================================================================
    // Euler characteristic tests
    // ========================================================================

    /// Sphere has Euler characteristic 2 (V - E + F = 2).
    #[test]
    fn test_uv_sphere_euler_characteristic() {
        for segments in [8, 16, 32] {
            for rings in [4, 8, 16] {
                let sphere = UvSphere::new(1.0, segments, rings).apply();
                let topo = analyze_topology(&sphere);

                assert_eq!(
                    topo.euler_characteristic, 2,
                    "UvSphere({segments}x{rings}) should have χ=2, got {}",
                    topo.euler_characteristic
                );
                assert_eq!(topo.genus, 0, "Sphere should have genus 0");
            }
        }
    }

    /// Icosphere has Euler characteristic 2.
    #[test]
    fn test_icosphere_euler_characteristic() {
        for subdivisions in 0..=4 {
            let sphere = Icosphere::new(1.0, subdivisions).apply();
            let topo = analyze_topology(&sphere);

            assert_eq!(
                topo.euler_characteristic, 2,
                "Icosphere(subdiv={subdivisions}) should have χ=2, got {}",
                topo.euler_characteristic
            );
            assert_eq!(topo.genus, 0, "Sphere should have genus 0");
        }
    }

    /// Cuboid has Euler characteristic 2.
    #[test]
    fn test_cuboid_euler_characteristic() {
        let cube = Cuboid::default().apply();
        let topo = analyze_topology(&cube);

        assert_eq!(topo.euler_characteristic, 2, "Cube should have χ=2");
        assert_eq!(topo.genus, 0, "Cube should have genus 0");
    }

    /// Cylinder has Euler characteristic 2 (with caps).
    #[test]
    fn test_cylinder_euler_characteristic() {
        for segments in [8, 16, 32] {
            let cylinder = Cylinder::new(1.0, 2.0, segments).apply();
            let topo = analyze_topology(&cylinder);

            assert_eq!(
                topo.euler_characteristic, 2,
                "Cylinder({segments} segments) should have χ=2, got {}",
                topo.euler_characteristic
            );
        }
    }

    /// Cone has Euler characteristic 2.
    #[test]
    fn test_cone_euler_characteristic() {
        for segments in [8, 16, 32] {
            let cone = Cone::new(1.0, 2.0, segments).apply();
            let topo = analyze_topology(&cone);

            assert_eq!(
                topo.euler_characteristic, 2,
                "Cone({segments} segments) should have χ=2, got {}",
                topo.euler_characteristic
            );
        }
    }

    /// Torus has Euler characteristic 0 (genus 1).
    #[test]
    fn test_torus_euler_characteristic() {
        let torus = Torus::new(1.0, 0.25, 16, 8).apply();
        let topo = analyze_topology(&torus);

        assert_eq!(
            topo.euler_characteristic, 0,
            "Torus should have χ=0 (genus 1)"
        );
        assert_eq!(topo.genus, 1, "Torus should have genus 1");
    }

    // ========================================================================
    // Manifold and closed tests
    // ========================================================================

    /// All closed primitives should be manifold.
    #[test]
    fn test_primitives_are_manifold() {
        let primitives: Vec<(&str, Mesh)> = vec![
            ("UvSphere", UvSphere::default().apply()),
            ("Icosphere", Icosphere::default().apply()),
            ("Cuboid", Cuboid::default().apply()),
            ("Cylinder", Cylinder::default().apply()),
            ("Cone", Cone::default().apply()),
            ("Torus", Torus::default().apply()),
        ];

        for (name, mesh) in primitives {
            let topo = analyze_topology(&mesh);
            assert!(topo.is_manifold, "{name} should be manifold");
        }
    }

    /// All closed primitives should be closed (no boundary edges).
    #[test]
    fn test_primitives_are_closed() {
        let primitives: Vec<(&str, Mesh)> = vec![
            ("UvSphere", UvSphere::default().apply()),
            ("Icosphere", Icosphere::default().apply()),
            ("Cuboid", Cuboid::default().apply()),
            ("Cylinder", Cylinder::default().apply()),
            ("Cone", Cone::default().apply()),
            ("Torus", Torus::default().apply()),
        ];

        for (name, mesh) in primitives {
            let topo = analyze_topology(&mesh);
            assert!(topo.is_closed, "{name} should be closed (no boundary)");
        }
    }

    /// Plane is open (has boundary).
    #[test]
    fn test_plane_is_open() {
        let plane = Plane::default().apply();
        let topo = analyze_topology(&plane);

        assert!(!topo.is_closed, "Plane should be open (have boundary)");
        assert!(topo.is_manifold, "Plane should be manifold");
        assert_eq!(
            topo.boundary_loop_count, 1,
            "Plane should have 1 boundary loop"
        );
    }

    // ========================================================================
    // Normals tests
    // ========================================================================

    /// Computed normals should be unit length.
    #[test]
    fn test_normals_are_unit_length() {
        let mut mesh = Icosphere::new(1.0, 2).apply();
        mesh.compute_smooth_normals();

        for (i, normal) in mesh.normals.iter().enumerate() {
            let length = normal.length();
            assert!(
                (length - 1.0).abs() < 0.001,
                "Normal {i} should be unit length, got {length}"
            );
        }
    }

    /// Flat normals should be perpendicular to faces.
    #[test]
    fn test_flat_normals_perpendicular_to_face() {
        let mut mesh = Cuboid::default().apply();
        mesh.compute_flat_normals();

        for tri in mesh.indices.chunks(3) {
            let v0 = mesh.positions[tri[0] as usize];
            let v1 = mesh.positions[tri[1] as usize];
            let v2 = mesh.positions[tri[2] as usize];

            let edge1 = v1 - v0;
            let edge2 = v2 - v0;
            let normal = mesh.normals[tri[0] as usize];

            // Normal should be perpendicular to both edges
            let dot1 = normal.dot(edge1).abs();
            let dot2 = normal.dot(edge2).abs();

            assert!(
                dot1 < 0.001,
                "Normal should be perpendicular to edge1, dot={dot1}"
            );
            assert!(
                dot2 < 0.001,
                "Normal should be perpendicular to edge2, dot={dot2}"
            );
        }
    }

    /// Normals of a sphere should point outward (same direction as position).
    #[test]
    fn test_sphere_normals_point_outward() {
        let sphere = UvSphere::new(1.0, 16, 8).apply();

        for (pos, normal) in sphere.positions.iter().zip(sphere.normals.iter()) {
            let pos_normalized = pos.normalize();
            let dot = pos_normalized.dot(*normal);

            // Normal should point in same direction as position (outward)
            assert!(
                dot > 0.99,
                "Sphere normal should point outward, dot with position = {dot}"
            );
        }
    }

    // ========================================================================
    // Subdivision tests
    // ========================================================================

    /// Linear subdivision multiplies triangle count by 4.
    #[test]
    fn test_linear_subdivision_triangle_count() {
        let mesh = Icosphere::new(1.0, 0).apply();
        let initial_tris = mesh.triangle_count();

        let sub1 = subdivide_linear(&mesh);
        assert_eq!(
            sub1.triangle_count(),
            initial_tris * 4,
            "1 level should give 4× triangles"
        );

        let sub2 = subdivide_linear(&sub1);
        assert_eq!(
            sub2.triangle_count(),
            initial_tris * 16,
            "2 levels should give 16× triangles"
        );
    }

    /// Loop subdivision multiplies triangle count by 4.
    #[test]
    fn test_loop_subdivision_triangle_count() {
        let mesh = Icosphere::new(1.0, 0).apply();
        let initial_tris = mesh.triangle_count();

        let sub1 = subdivide_loop(&mesh);
        assert_eq!(sub1.triangle_count(), initial_tris * 4);

        let sub2 = subdivide_loop(&sub1);
        assert_eq!(sub2.triangle_count(), initial_tris * 16);
    }

    /// Loop subdivision preserves topology (Euler characteristic, manifold-ness).
    #[test]
    fn test_loop_subdivision_preserves_topology() {
        let mesh = Icosphere::new(1.0, 1).apply();
        let topo_before = analyze_topology(&mesh);

        let subdivided = subdivide_loop(&mesh);
        let topo_after = analyze_topology(&subdivided);

        assert_eq!(
            topo_before.euler_characteristic, topo_after.euler_characteristic,
            "Euler characteristic should be preserved"
        );
        assert_eq!(
            topo_before.is_manifold, topo_after.is_manifold,
            "Manifold status should be preserved"
        );
        assert_eq!(
            topo_before.is_closed, topo_after.is_closed,
            "Closed status should be preserved"
        );
        assert_eq!(
            topo_before.genus, topo_after.genus,
            "Genus should be preserved"
        );
    }

    /// Catmull-Clark preserves topology (Euler characteristic, genus).
    #[test]
    fn test_catmull_clark_preserves_topology() {
        // Use Icosphere which has shared vertices (Cuboid has separate vertices per face)
        let mesh = Icosphere::new(1.0, 1).apply();
        let topo_before = analyze_topology(&mesh);

        let subdivided = CatmullClark::new(2).apply(&mesh);
        let topo_after = analyze_topology(&subdivided);

        assert_eq!(
            topo_before.euler_characteristic, topo_after.euler_characteristic,
            "Euler characteristic should be preserved"
        );
        assert_eq!(
            topo_before.genus, topo_after.genus,
            "Genus should be preserved"
        );
    }

    /// Catmull-Clark on torus preserves genus 1.
    #[test]
    fn test_catmull_clark_torus_preserves_genus() {
        let torus = Torus::new(1.0, 0.25, 8, 4).apply();
        let subdivided = CatmullClark::new(1).apply(&torus);
        let topo = analyze_topology(&subdivided);

        assert_eq!(
            topo.genus, 1,
            "Torus genus should be preserved after subdivision"
        );
    }

    // ========================================================================
    // Geometry tests
    // ========================================================================

    /// All vertices of a sphere should be at the specified radius.
    #[test]
    fn test_sphere_vertices_at_radius() {
        for radius in [0.5, 1.0, 2.0, 5.0] {
            let sphere = UvSphere::new(radius, 16, 8).apply();

            for (i, pos) in sphere.positions.iter().enumerate() {
                let dist = pos.length();
                assert!(
                    (dist - radius).abs() < 0.001,
                    "Sphere vertex {i} should be at radius {radius}, got {dist}"
                );
            }
        }
    }

    /// Icosphere vertices should all be at the specified radius.
    #[test]
    fn test_icosphere_vertices_at_radius() {
        for radius in [0.5, 1.0, 2.0] {
            for subdivisions in 0..=3 {
                let sphere = Icosphere::new(radius, subdivisions).apply();

                for (i, pos) in sphere.positions.iter().enumerate() {
                    let dist = pos.length();
                    assert!(
                        (dist - radius).abs() < 0.001,
                        "Icosphere({subdivisions}) vertex {i} at radius {radius}: got {dist}"
                    );
                }
            }
        }
    }

    /// Torus vertices should be within expected distance from axis.
    #[test]
    fn test_torus_vertices_in_bounds() {
        let major = 2.0;
        let minor = 0.5;
        let torus = Torus::new(major, minor, 16, 8).apply();

        let min_dist = major - minor;
        let max_dist = major + minor;

        for (i, pos) in torus.positions.iter().enumerate() {
            let dist_from_axis = (pos.x * pos.x + pos.z * pos.z).sqrt();
            assert!(
                dist_from_axis >= min_dist - 0.001 && dist_from_axis <= max_dist + 0.001,
                "Torus vertex {i} distance from axis should be in [{min_dist}, {max_dist}], got {dist_from_axis}"
            );
        }
    }

    /// Cuboid vertices should be within half-extents.
    #[test]
    fn test_cuboid_vertices_in_bounds() {
        let width = 2.0;
        let height = 3.0;
        let depth = 4.0;
        let cube = Cuboid::new(width, height, depth).apply();

        let hx = width / 2.0;
        let hy = height / 2.0;
        let hz = depth / 2.0;

        for (i, pos) in cube.positions.iter().enumerate() {
            assert!(
                pos.x.abs() <= hx + 0.001,
                "Cuboid vertex {i} X out of bounds: {}",
                pos.x
            );
            assert!(
                pos.y.abs() <= hy + 0.001,
                "Cuboid vertex {i} Y out of bounds: {}",
                pos.y
            );
            assert!(
                pos.z.abs() <= hz + 0.001,
                "Cuboid vertex {i} Z out of bounds: {}",
                pos.z
            );
        }
    }

    /// Cylinder vertices should be within radius and height bounds.
    #[test]
    fn test_cylinder_vertices_in_bounds() {
        let radius = 1.5;
        let height = 3.0;
        let cylinder = Cylinder::new(radius, height, 16).apply();

        let half_height = height / 2.0;

        for (i, pos) in cylinder.positions.iter().enumerate() {
            let dist_from_axis = (pos.x * pos.x + pos.z * pos.z).sqrt();
            assert!(
                dist_from_axis <= radius + 0.001,
                "Cylinder vertex {i} exceeds radius: {dist_from_axis}"
            );
            assert!(
                pos.y >= -half_height - 0.001 && pos.y <= half_height + 0.001,
                "Cylinder vertex {i} Y out of bounds: {}",
                pos.y
            );
        }
    }

    /// Plane vertices should all have Y = 0.
    #[test]
    fn test_plane_vertices_flat() {
        let plane = Plane::new(2.0, 3.0, 10, 10).apply();

        for (i, pos) in plane.positions.iter().enumerate() {
            assert!(
                pos.y.abs() < 0.001,
                "Plane vertex {i} should have Y=0, got {}",
                pos.y
            );
        }
    }

    // ========================================================================
    // Mesh merge and transform tests
    // ========================================================================

    /// Merging meshes should produce correct vertex and triangle counts.
    #[test]
    fn test_mesh_merge_counts() {
        let cube1 = Cuboid::default().apply();
        let cube2 = Cuboid::default().apply();

        let mut merged = cube1.clone();
        merged.merge(&cube2);

        assert_eq!(
            merged.vertex_count(),
            cube1.vertex_count() + cube2.vertex_count()
        );
        assert_eq!(
            merged.triangle_count(),
            cube1.triangle_count() + cube2.triangle_count()
        );
    }

    /// Transform should preserve mesh validity.
    #[test]
    fn test_transform_preserves_topology() {
        let mut mesh = Icosphere::new(1.0, 2).apply();
        let topo_before = analyze_topology(&mesh);

        mesh.transform(glam::Mat4::from_scale(Vec3::new(2.0, 3.0, 4.0)));
        let topo_after = analyze_topology(&mesh);

        assert_eq!(topo_before.vertex_count, topo_after.vertex_count);
        assert_eq!(topo_before.face_count, topo_after.face_count);
        assert_eq!(
            topo_before.euler_characteristic,
            topo_after.euler_characteristic
        );
    }

    /// Transform normals should remain unit length.
    #[test]
    fn test_transform_normals_remain_unit() {
        let mut mesh = UvSphere::new(1.0, 16, 8).apply();
        mesh.transform(glam::Mat4::from_scale(Vec3::new(2.0, 0.5, 1.0)));

        for (i, normal) in mesh.normals.iter().enumerate() {
            let length = normal.length();
            assert!(
                (length - 1.0).abs() < 0.01,
                "Transformed normal {i} should be unit length, got {length}"
            );
        }
    }

    // ========================================================================
    // Triangle winding and orientation tests
    // ========================================================================

    /// All primitives should have consistent winding (orientable).
    #[test]
    fn test_primitives_are_orientable() {
        let primitives: Vec<(&str, Mesh)> = vec![
            ("UvSphere", UvSphere::default().apply()),
            ("Icosphere", Icosphere::default().apply()),
            ("Cuboid", Cuboid::default().apply()),
            ("Cylinder", Cylinder::default().apply()),
            ("Cone", Cone::default().apply()),
            ("Torus", Torus::default().apply()),
        ];

        for (name, mesh) in primitives {
            let topo = analyze_topology(&mesh);
            assert!(topo.is_orientable, "{name} should be orientable");
        }
    }

    // ========================================================================
    // Connected components tests
    // ========================================================================

    /// Single primitives should have exactly 1 connected component.
    #[test]
    fn test_primitives_single_component() {
        let primitives: Vec<(&str, Mesh)> = vec![
            ("UvSphere", UvSphere::default().apply()),
            ("Icosphere", Icosphere::default().apply()),
            ("Cuboid", Cuboid::default().apply()),
            ("Cylinder", Cylinder::default().apply()),
            ("Cone", Cone::default().apply()),
            ("Torus", Torus::default().apply()),
            ("Plane", Plane::default().apply()),
        ];

        for (name, mesh) in primitives {
            let components = connected_components(&mesh);
            assert_eq!(components, 1, "{name} should have 1 connected component");
        }
    }

    /// Merged disjoint meshes should have multiple components.
    #[test]
    fn test_merged_meshes_multiple_components() {
        let mut cube1 = Cuboid::default().apply();
        let mut cube2 = Cuboid::default().apply();

        // Move cube2 far away so they don't overlap
        cube2.transform(glam::Mat4::from_translation(Vec3::new(10.0, 0.0, 0.0)));

        cube1.merge(&cube2);
        let components = connected_components(&cube1);

        assert_eq!(components, 2, "Two disjoint cubes should have 2 components");
    }

    // ========================================================================
    // Smooth operation tests
    // ========================================================================

    /// Smoothing should reduce vertex variance (converge toward centroid).
    #[test]
    fn test_smoothing_reduces_variance() {
        let mesh = Cuboid::default().apply();

        // Compute initial variance from centroid
        let centroid: Vec3 =
            mesh.positions.iter().copied().sum::<Vec3>() / mesh.positions.len() as f32;
        let initial_variance: f32 = mesh
            .positions
            .iter()
            .map(|p| (*p - centroid).length_squared())
            .sum::<f32>()
            / mesh.positions.len() as f32;

        // Apply smoothing
        let smoothed = smooth(&mesh, 0.5, 5);

        // Compute smoothed variance
        let smoothed_centroid: Vec3 =
            smoothed.positions.iter().copied().sum::<Vec3>() / smoothed.positions.len() as f32;
        let smoothed_variance: f32 = smoothed
            .positions
            .iter()
            .map(|p| (*p - smoothed_centroid).length_squared())
            .sum::<f32>()
            / smoothed.positions.len() as f32;

        assert!(
            smoothed_variance <= initial_variance,
            "Smoothing should reduce variance: initial={initial_variance}, smoothed={smoothed_variance}"
        );
    }

    // ========================================================================
    // Flip normals test
    // ========================================================================

    /// Flip normals should reverse all normals.
    #[test]
    fn test_flip_normals_reverses_direction() {
        let mesh = UvSphere::new(1.0, 16, 8).apply();
        let flipped = flip_normals(&mesh);

        for (original, flipped) in mesh.normals.iter().zip(flipped.normals.iter()) {
            let dot = original.dot(*flipped);
            assert!(
                (dot + 1.0).abs() < 0.001,
                "Flipped normal should be opposite: dot={dot}"
            );
        }
    }

    // ========================================================================
    // Double-sided mesh test
    // ========================================================================

    /// Make double-sided should double the triangle count.
    #[test]
    fn test_double_sided_doubles_triangles() {
        let mesh = Plane::default().apply();
        let double_sided = make_double_sided(&mesh);

        assert_eq!(
            double_sided.triangle_count(),
            mesh.triangle_count() * 2,
            "Double-sided should have 2× triangles"
        );
    }
}
