//! 2D vector graphics for resin.
//!
//! Provides path primitives, vector networks, and operations for 2D vector art.

mod boolean;
mod delaunay;
mod geometry;
mod gradient_mesh;
mod hatching;
mod network;
mod path;
pub mod rasterize;
mod stroke;
pub mod svg;
mod text;

pub use boolean::{
    CurveIntersection, CurveSegment, FillRule, closest_point_on_curve, cubic_self_intersections,
    curve_intersections, line_curve_intersections, path_contains_point, path_intersect,
    path_intersect_with_fill, path_subtract, path_subtract_with_fill, path_union,
    path_union_with_fill, path_winding_number, path_xor, path_xor_multi,
    polygon_contains_point_with_rule, winding_number,
};
pub use delaunay::{
    Triangle, VoronoiCell, VoronoiDiagram, delaunay_triangulation, triangles_to_indices,
    voronoi_diagram, voronoi_to_segments,
};
pub use geometry::{
    bounding_box, centroid, convex_hull, convex_hull_path, is_ccw, minimum_bounding_circle,
    point_in_polygon, point_on_hull, polygon_area, polygon_perimeter, signed_area,
};
pub use gradient_mesh::{
    GradientFace, GradientMesh, GradientPatch, GradientVertex, diagonal_gradient_mesh,
    four_corner_gradient_mesh, linear_gradient_mesh,
};
pub use hatching::{
    HatchConfig, HatchLine, cross_hatch_polygon, cross_hatch_rect, hatch_lines_to_paths,
    hatch_polygon, hatch_rect,
};
pub use network::{
    Edge, EdgeHandle, EdgeId, EdgeType, HandleStyle, Node, NodeId, Region, VectorNetwork,
};
pub use path::{
    Path,
    PathBuilder,
    PathCommand,
    // Primitives
    circle,
    ellipse,
    line,
    polygon,
    polyline,
    rect,
    rect_centered,
    regular_polygon,
    rounded_rect,
    star,
};
pub use stroke::{
    CapStyle, DashPattern, JoinStyle, StrokeConfig, dash_path, offset_path, path_length,
    point_at_length, resample_path, simplify_path, simplify_points, smooth_path, stroke_to_path,
    tangent_at_length,
};
pub use text::{
    Font, FontError, FontResult, TextConfig, TextMetrics, measure_text, text_to_path,
    text_to_path_outlined, text_to_paths, text_to_paths_outlined,
};
