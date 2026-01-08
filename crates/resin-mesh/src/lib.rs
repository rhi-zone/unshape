//! 3D mesh generation for resin.
//!
//! Provides mesh primitives and operations for procedural 3D geometry.

mod bevel;
mod boolean;
mod halfedge;
mod loops;
mod mesh;
mod morph;
mod ops;
mod primitives;
mod subdivision;
mod uv;

pub use bevel::{BevelConfig, bevel_edges, bevel_mesh_edges, bevel_mesh_vertices, bevel_vertices};
pub use boolean::{boolean_intersect, boolean_subtract, boolean_union};
pub use halfedge::{
    Face, FaceId, HalfEdge, HalfEdgeId, HalfEdgeMesh, Vertex as HEVertex, VertexId,
};
pub use loops::{
    edges_to_faces, edges_to_vertices, grow_edge_selection, loop_cut, select_boundary_edges,
    select_edge_loop, select_edge_ring, select_face_edges, select_vertex_edges,
};
pub use mesh::{Mesh, MeshBuilder};
pub use morph::{
    MorphTarget, MorphTargetSet, MorphWeights, apply_morph_targets,
    apply_morph_targets_with_normals, blend_positions,
};
pub use ops::{
    ExtrudeConfig, InsetConfig, NormalMode, SmoothConfig, extrude, extrude_with_config,
    flip_normals, inset, inset_with_config, make_double_sided, recalculate_normals, smooth,
    smooth_taubin, smooth_with_config, solidify, split_faces, weld_vertices,
};
pub use primitives::{box_mesh, sphere, uv_sphere};
pub use subdivision::{subdivide_linear, subdivide_loop, subdivide_loop_n};
pub use uv::{
    BoxConfig, CylindricalConfig, ProjectionAxis, SphericalConfig, flip_u, flip_v, normalize_uvs,
    project_box, project_box_per_face, project_cylindrical, project_planar, project_planar_axis,
    project_spherical, rotate_uvs, scale_uvs, transform_uvs, translate_uvs,
};
