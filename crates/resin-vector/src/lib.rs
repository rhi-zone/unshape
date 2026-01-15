//! 2D vector graphics for resin.
//!
//! Provides path primitives, vector networks, and operations for 2D vector art.
//!
//! Types implement the [`Curve`] trait from `resin-curve` for unified curve operations.

pub mod bezier;
mod boolean;
mod curve_impl;
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

// Re-export curve types for convenience
pub use rhizome_resin_curve::{
    Arc, ArcLengthPath, CubicBezier, Curve, Line, Path as CurvePath, QuadBezier, Segment2D,
    VectorSpace,
};

pub use boolean::{
    Bounds2D, ClosestPoint, CurveIntersection, CurveSegment, FillRule, SplitCurve,
    closest_point_on_curve, cubic_self_intersections, curve_intersections,
    line_curve_intersections, path_contains_point, path_intersect, path_intersect_with_fill,
    path_subtract, path_subtract_with_fill, path_union, path_union_with_fill, path_winding_number,
    path_xor, path_xor_multi, polygon_contains_point_with_rule, winding_number,
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
    Hatch, HatchConfig, HatchLine, cross_hatch_polygon, cross_hatch_rect, hatch_lines_to_paths,
    hatch_polygon, hatch_rect,
};
pub use network::{
    Edge, EdgeHandle, EdgeId, EdgeType, HandleStyle, Node, NodeId, Region, VectorNetwork,
};
pub use path::{
    CornerRadii,
    Path,
    PathBuilder,
    PathCommand,
    // Primitives
    circle,
    ellipse,
    line,
    pill,
    polygon,
    polyline,
    rect,
    rect_centered,
    regular_polygon,
    rounded_rect,
    rounded_rect_corners,
    squircle,
    squircle_uniform,
    squircle_with_segments,
    star,
};
pub use stroke::{
    CapStyle, DashPattern, JoinStyle, PressurePoint, PressureStroke, PressureStrokeConfig,
    PressureStrokeRender, Stroke, StrokeConfig, Trim, TrimResult, dash_path, offset_path,
    path_length, point_at_length, pressure_stroke_to_path, resample_path, simplify_path,
    simplify_points, simulate_natural_pressure, simulate_velocity_pressure, smooth_path,
    stroke_to_path, tangent_at_length, trim_path, trim_path_with_tangents, trim_segments,
};
pub use text::{
    Font, FontError, FontResult, TextConfig, TextLayout, TextMetrics, measure_text, text_to_path,
    text_to_path_outlined, text_to_paths, text_to_paths_outlined,
};

/// Registers all vector operations with an [`OpRegistry`].
///
/// Call this to enable deserialization of vector ops from saved pipelines.
#[cfg(feature = "dynop")]
pub fn register_ops(registry: &mut rhizome_resin_op::OpRegistry) {
    registry.register_type::<Stroke>("resin::Stroke");
    registry.register_type::<PressureStrokeRender>("resin::PressureStrokeRender");
    registry.register_type::<Trim>("resin::Trim");
}
