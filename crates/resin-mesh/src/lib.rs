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
pub use primitives::{
    Cone, Cylinder, Icosphere, Plane, Pyramid, Torus, box_mesh, cone, cylinder, icosphere, plane,
    pyramid, sphere, torus, uv_sphere,
};
pub use remesh::{
    QuadMesh, Quadify, QuadifyConfig, Remesh, RemeshConfig, average_edge_length, isotropic_remesh,
    quadify,
};
pub use sdf::{GenerateSdf, SdfConfig, SdfGrid, mesh_to_sdf, mesh_to_sdf_fast, raymarch};
pub use subdivision::{subdivide_linear, subdivide_loop, subdivide_loop_n};
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
pub fn register_ops(registry: &mut rhizome_resin_op::OpRegistry) {
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
}
