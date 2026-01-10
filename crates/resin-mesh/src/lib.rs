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
mod uv;

pub use ao::{
    AoAccelerator, AoBakeConfig, AoTexture, ao_to_vertex_colors, bake_ao_texture, bake_ao_vertices,
    blur_ao_texture,
};
pub use architecture::{
    Building, DoorConfig, FloorPoint, RoofConfig, RoofStyle, WallConfig, WindowConfig,
    generate_building, generate_stairs, generate_wall_with_door, generate_wall_with_window,
    generate_walls,
};
pub use bevel::{BevelConfig, bevel_edges, bevel_mesh_edges, bevel_mesh_vertices, bevel_vertices};
pub use boolean::{boolean_intersect, boolean_subtract, boolean_union};
pub use curvature::{
    CurvatureResult, compute_curvature, gaussian_curvature, mean_curvature, principal_curvatures,
};
pub use curve_mesh::{
    ExtrudeProfileConfig, RevolveConfig, SweepConfig, extrude_profile, extrude_profile_with_config,
    revolve_profile, revolve_profile_with_config, sweep_profile, sweep_profile_with_config,
};
pub use decimate::{DecimateConfig, decimate};
pub use halfedge::{
    Face, FaceId, HalfEdge, HalfEdgeId, HalfEdgeMesh, Vertex as HEVertex, VertexId,
};
pub use lattice::{
    Lattice, LatticeDeformConfig, bend_lattice, lattice_deform, lattice_deform_point,
    lattice_deform_points, lattice_deform_with_config, scale_lattice, taper_lattice, twist_lattice,
};
pub use loft::{LoftConfig, circle_profile, loft, loft_along_path, rect_profile, star_profile};
pub use loops::{
    edges_to_faces, edges_to_vertices, grow_edge_selection, loop_cut, select_boundary_edges,
    select_edge_loop, select_edge_ring, select_face_edges, select_vertex_edges,
};
pub use marching_cubes::{MarchingCubesConfig, marching_cubes};
pub use mesh::{Mesh, MeshBuilder};
pub use morph::{
    MorphTarget, MorphTargetSet, MorphWeights, apply_morph_targets,
    apply_morph_targets_with_normals, blend_positions,
};
pub use navmesh::{
    NavMesh, NavMeshConfig, NavPath, NavPolygon, create_grid_navmesh, find_path, smooth_path,
};
pub use obj::{ObjError, export_obj, export_obj_with_name, import_obj, import_obj_from_reader};
pub use ops::{
    ExtrudeConfig, InsetConfig, NormalMode, SmoothConfig, extrude, extrude_with_config,
    flip_normals, inset, inset_with_config, make_double_sided, recalculate_normals, smooth,
    smooth_taubin, smooth_with_config, solidify, split_faces, weld_vertices,
};
pub use primitives::{box_mesh, sphere, uv_sphere};
pub use remesh::{
    QuadMesh, QuadifyConfig, RemeshConfig, average_edge_length, isotropic_remesh, quadify,
};
pub use sdf::{SdfConfig, SdfGrid, mesh_to_sdf, mesh_to_sdf_fast, raymarch};
pub use subdivision::{subdivide_linear, subdivide_loop, subdivide_loop_n};
pub use terrain::{CombinedErosion, Heightfield, HydraulicErosion, ThermalErosion};
pub use uv::{
    AtlasPackConfig, AtlasPackResult, BoxConfig, CylindricalConfig, PackedChart, ProjectionAxis,
    SphericalConfig, UvChart, apply_atlas_pack, find_uv_islands, flip_u, flip_v, normalize_uvs,
    pack_mesh_uvs, pack_multi_mesh_uvs, pack_uv_charts, project_box, project_box_per_face,
    project_cylindrical, project_planar, project_planar_axis, project_spherical, rotate_uvs,
    scale_uvs, transform_uvs, translate_uvs,
};
