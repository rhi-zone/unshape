//! 2D vector graphics for resin.
//!
//! Provides path primitives, vector networks, and operations for 2D vector art.
//!
//! Types implement the [`Curve`] trait from `unshape-curve` for unified curve operations.

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
pub use unshape_curve::{
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
    Anchor, AnchorId, Edge, EdgeHandle, EdgeId, EdgeType, HandleStyle, Region, VectorNetwork,
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
pub fn register_ops(registry: &mut unshape_op::OpRegistry) {
    registry.register_type::<Stroke>("resin::Stroke");
    registry.register_type::<PressureStrokeRender>("resin::PressureStrokeRender");
    registry.register_type::<Trim>("resin::Trim");
}

/// Invariant tests for vector operations.
///
/// Run with: cargo test -p unshape-vector --features invariant-tests
#[cfg(all(test, feature = "invariant-tests"))]
mod invariant_tests {
    use super::*;
    use glam::Vec2;

    // ========================================================================
    // Path Invariants
    // ========================================================================

    /// Closed paths created by primitives should have their Close command.
    #[test]
    fn invariant_closed_path_has_close_command() {
        let shapes = [
            ("rect", rect(Vec2::ZERO, Vec2::new(10.0, 10.0))),
            ("circle", circle(Vec2::ZERO, 5.0)),
            ("polygon", polygon(&[Vec2::ZERO, Vec2::X, Vec2::Y])),
            ("regular_polygon", regular_polygon(Vec2::ZERO, 5.0, 6)),
            ("star", star(Vec2::ZERO, 10.0, 5.0, 5)),
            (
                "rounded_rect",
                rounded_rect(Vec2::ZERO, Vec2::new(10.0, 10.0), 2.0),
            ),
            ("squircle", squircle(Vec2::ZERO, Vec2::new(10.0, 10.0), 4.0)),
            ("ellipse", ellipse(Vec2::ZERO, Vec2::new(5.0, 3.0))),
        ];

        for (name, path) in shapes {
            let commands = path.commands();
            assert!(!commands.is_empty(), "{name}: path should not be empty");
            assert!(
                matches!(commands.last(), Some(PathCommand::Close)),
                "{name}: closed path should end with Close command, got {:?}",
                commands.last()
            );
        }
    }

    /// Path length (computed via stroke module) should be positive for non-empty paths.
    #[test]
    fn invariant_path_length_positive() {
        let paths = [
            ("line", line(Vec2::ZERO, Vec2::new(10.0, 0.0))),
            ("rect", rect(Vec2::ZERO, Vec2::new(10.0, 10.0))),
            ("circle", circle(Vec2::ZERO, 5.0)),
        ];

        for (name, path) in paths {
            let length = path_length(&path);
            assert!(
                length > 0.0,
                "{name}: path length should be positive, got {length}"
            );
        }
    }

    /// Polyline of sequential points should have length equal to sum of segment lengths.
    #[test]
    fn invariant_polyline_length_matches_segments() {
        let points = [
            Vec2::new(0.0, 0.0),
            Vec2::new(3.0, 0.0),
            Vec2::new(3.0, 4.0),
        ];
        let path = polyline(&points);

        // Expected: 3 + 4 = 7
        let expected_length = 7.0;
        let actual_length = path_length(&path);

        assert!(
            (actual_length - expected_length).abs() < 0.01,
            "polyline length should be {expected_length}, got {actual_length}"
        );
    }

    // ========================================================================
    // Boolean Operation Invariants
    // ========================================================================

    /// Union area should be >= max(area1, area2) for non-overlapping shapes.
    #[test]
    fn invariant_union_area_at_least_max() {
        let r1 = rect(Vec2::ZERO, Vec2::new(10.0, 10.0));
        let r2 = rect(Vec2::new(5.0, 5.0), Vec2::new(15.0, 15.0));

        let union = path_union(&r1, &r2, 8);
        let poly1 = flatten_path_for_area(&r1, 8);
        let poly2 = flatten_path_for_area(&r2, 8);
        let union_poly = flatten_path_for_area(&union, 8);

        let area1 = polygon_area(&poly1).abs();
        let area2 = polygon_area(&poly2).abs();
        let union_area = polygon_area(&union_poly).abs();

        let max_area = area1.max(area2);

        assert!(
            union_area >= max_area - 0.1,
            "union area ({union_area}) should be >= max of individual areas ({max_area})"
        );
    }

    /// Intersection area should be <= min(area1, area2).
    #[test]
    fn invariant_intersection_area_at_most_min() {
        let r1 = rect(Vec2::ZERO, Vec2::new(10.0, 10.0));
        let r2 = rect(Vec2::new(5.0, 5.0), Vec2::new(15.0, 15.0));

        let intersection = path_intersect(&r1, &r2, 8);
        let poly1 = flatten_path_for_area(&r1, 8);
        let poly2 = flatten_path_for_area(&r2, 8);
        let intersection_poly = flatten_path_for_area(&intersection, 8);

        let area1 = polygon_area(&poly1).abs();
        let area2 = polygon_area(&poly2).abs();
        let intersection_area = polygon_area(&intersection_poly).abs();

        let min_area = area1.min(area2);

        assert!(
            intersection_area <= min_area + 0.1,
            "intersection area ({intersection_area}) should be <= min of individual areas ({min_area})"
        );
    }

    /// Subtracting a shape completely containing A should result in empty path.
    #[test]
    fn invariant_subtract_complete_coverage_is_empty() {
        // Small rectangle completely inside a larger one
        let inner = rect(Vec2::new(2.0, 2.0), Vec2::new(8.0, 8.0));
        let outer = rect(Vec2::ZERO, Vec2::new(10.0, 10.0));

        let difference = path_subtract(&inner, &outer, 8);

        // When inner is completely inside outer, subtracting outer from inner
        // should yield an empty or nearly empty result
        let difference_poly = flatten_path_for_area(&difference, 8);
        let difference_area = polygon_area(&difference_poly).abs();

        assert!(
            difference_area < 1.0,
            "subtracting containing shape should yield near-empty result, got area {difference_area}"
        );
    }

    // ========================================================================
    // Convex Hull Invariants
    // ========================================================================

    /// All original points should be inside or on the convex hull.
    #[test]
    fn invariant_convex_hull_contains_all_points() {
        let points = vec![
            Vec2::new(0.0, 0.0),
            Vec2::new(10.0, 0.0),
            Vec2::new(10.0, 10.0),
            Vec2::new(0.0, 10.0),
            Vec2::new(5.0, 5.0), // Interior point
            Vec2::new(3.0, 7.0), // Interior point
            Vec2::new(8.0, 2.0), // Interior point
        ];

        let hull = convex_hull(&points);

        for (i, point) in points.iter().enumerate() {
            let inside = point_in_polygon(*point, &hull);
            let on_boundary = point_on_hull(*point, &hull, 0.001);
            assert!(
                inside || on_boundary,
                "point {i} ({point:?}) should be inside or on the convex hull"
            );
        }
    }

    /// Convex hull should have at most as many points as input (no duplicates added).
    #[test]
    fn invariant_convex_hull_size_bounded() {
        let points = vec![
            Vec2::new(0.0, 0.0),
            Vec2::new(1.0, 0.0),
            Vec2::new(2.0, 0.0),
            Vec2::new(0.0, 1.0),
            Vec2::new(1.0, 1.0),
            Vec2::new(2.0, 1.0),
            Vec2::new(0.0, 2.0),
            Vec2::new(1.0, 2.0),
            Vec2::new(2.0, 2.0),
        ];

        let hull = convex_hull(&points);

        assert!(
            hull.len() <= points.len(),
            "convex hull size ({}) should be <= input size ({})",
            hull.len(),
            points.len()
        );
    }

    /// Convex hull of a convex polygon should be the polygon itself (same vertices).
    #[test]
    fn invariant_convex_hull_of_convex_is_same() {
        // A square is convex
        let square = vec![
            Vec2::new(0.0, 0.0),
            Vec2::new(10.0, 0.0),
            Vec2::new(10.0, 10.0),
            Vec2::new(0.0, 10.0),
        ];

        let hull = convex_hull(&square);

        assert_eq!(
            hull.len(),
            square.len(),
            "convex hull of a convex polygon should have the same vertex count"
        );
    }

    // ========================================================================
    // Triangulation Invariants
    // ========================================================================

    /// Delaunay triangulation should produce valid triangle count for n points.
    /// For n >= 3 points in general position: 2n - 2 - h triangles,
    /// where h is the number of hull vertices. Approximately 2n - 5 for large n.
    #[test]
    fn invariant_triangulation_count_reasonable() {
        let points = vec![
            Vec2::new(0.0, 0.0),
            Vec2::new(10.0, 0.0),
            Vec2::new(10.0, 10.0),
            Vec2::new(0.0, 10.0),
            Vec2::new(5.0, 5.0),
        ];

        let triangles = delaunay_triangulation(&points);
        let n = points.len();

        // For n points: at most 2n - 5 triangles (for n >= 3)
        // Minimum is 1 for 3 points
        let min_triangles = 1;
        let max_triangles = if n >= 3 { 2 * n - 2 } else { 0 };

        assert!(
            triangles.len() >= min_triangles && triangles.len() <= max_triangles,
            "triangle count ({}) should be in range [{}, {}] for {} points",
            triangles.len(),
            min_triangles,
            max_triangles,
            n
        );
    }

    /// All triangle indices should be valid (within points array bounds).
    #[test]
    fn invariant_triangulation_valid_indices() {
        let points = vec![
            Vec2::new(0.0, 0.0),
            Vec2::new(10.0, 0.0),
            Vec2::new(10.0, 10.0),
            Vec2::new(0.0, 10.0),
            Vec2::new(5.0, 5.0),
            Vec2::new(3.0, 7.0),
        ];

        let triangles = delaunay_triangulation(&points);

        for (i, tri) in triangles.iter().enumerate() {
            assert!(
                tri.a < points.len(),
                "triangle {i}: vertex a ({}) out of bounds",
                tri.a
            );
            assert!(
                tri.b < points.len(),
                "triangle {i}: vertex b ({}) out of bounds",
                tri.b
            );
            assert!(
                tri.c < points.len(),
                "triangle {i}: vertex c ({}) out of bounds",
                tri.c
            );
        }
    }

    /// Triangles should have non-zero area (no degenerate triangles).
    #[test]
    fn invariant_triangulation_non_degenerate() {
        let points = vec![
            Vec2::new(0.0, 0.0),
            Vec2::new(10.0, 0.0),
            Vec2::new(10.0, 10.0),
            Vec2::new(0.0, 10.0),
            Vec2::new(5.0, 5.0),
        ];

        let triangles = delaunay_triangulation(&points);

        for (i, tri) in triangles.iter().enumerate() {
            let a = points[tri.a];
            let b = points[tri.b];
            let c = points[tri.c];
            let area = polygon_area(&[a, b, c]).abs();

            assert!(
                area > 1e-6,
                "triangle {i} should have non-zero area, got {area}"
            );
        }
    }

    /// Triangles should have distinct vertices (no repeated indices).
    #[test]
    fn invariant_triangulation_distinct_vertices() {
        let points = vec![
            Vec2::new(0.0, 0.0),
            Vec2::new(10.0, 0.0),
            Vec2::new(10.0, 10.0),
            Vec2::new(0.0, 10.0),
        ];

        let triangles = delaunay_triangulation(&points);

        for (i, tri) in triangles.iter().enumerate() {
            assert!(
                tri.a != tri.b && tri.b != tri.c && tri.a != tri.c,
                "triangle {i} should have distinct vertices: ({}, {}, {})",
                tri.a,
                tri.b,
                tri.c
            );
        }
    }

    // ========================================================================
    // Geometric Invariants
    // ========================================================================

    /// Bounding box should contain all input points.
    #[test]
    fn invariant_bounding_box_contains_all_points() {
        let points = vec![
            Vec2::new(-5.0, 3.0),
            Vec2::new(10.0, -2.0),
            Vec2::new(3.0, 8.0),
            Vec2::new(0.0, 0.0),
        ];

        let (min, max) = bounding_box(&points).unwrap();

        for (i, point) in points.iter().enumerate() {
            assert!(
                point.x >= min.x && point.x <= max.x,
                "point {i}: x ({}) outside bounding box [{}, {}]",
                point.x,
                min.x,
                max.x
            );
            assert!(
                point.y >= min.y && point.y <= max.y,
                "point {i}: y ({}) outside bounding box [{}, {}]",
                point.y,
                min.y,
                max.y
            );
        }
    }

    /// Centroid should be inside the bounding box of points.
    #[test]
    fn invariant_centroid_in_bounding_box() {
        let points = vec![
            Vec2::new(0.0, 0.0),
            Vec2::new(10.0, 0.0),
            Vec2::new(10.0, 10.0),
            Vec2::new(0.0, 10.0),
        ];

        let center = centroid(&points).unwrap();
        let (min, max) = bounding_box(&points).unwrap();

        assert!(
            center.x >= min.x && center.x <= max.x && center.y >= min.y && center.y <= max.y,
            "centroid ({center:?}) should be inside bounding box [{min:?}, {max:?}]"
        );
    }

    /// Minimum bounding circle should contain all points.
    #[test]
    fn invariant_min_bounding_circle_contains_all() {
        let points = vec![
            Vec2::new(0.0, 0.0),
            Vec2::new(10.0, 0.0),
            Vec2::new(5.0, 8.0),
            Vec2::new(3.0, 3.0),
        ];

        let (center, radius) = minimum_bounding_circle(&points).unwrap();

        for (i, point) in points.iter().enumerate() {
            let dist = (*point - center).length();
            assert!(
                dist <= radius + 0.01,
                "point {i} ({point:?}) at distance {dist} exceeds circle radius {radius}"
            );
        }
    }

    /// Polygon perimeter should be positive for non-trivial polygons.
    #[test]
    fn invariant_polygon_perimeter_positive() {
        let square = vec![
            Vec2::new(0.0, 0.0),
            Vec2::new(10.0, 0.0),
            Vec2::new(10.0, 10.0),
            Vec2::new(0.0, 10.0),
        ];

        let perimeter = polygon_perimeter(&square);
        assert!(
            perimeter > 0.0,
            "polygon perimeter should be positive, got {perimeter}"
        );

        // For a 10x10 square, perimeter should be 40
        assert!(
            (perimeter - 40.0).abs() < 0.01,
            "10x10 square perimeter should be 40, got {perimeter}"
        );
    }

    /// CCW polygon should have positive signed area, CW should have negative.
    #[test]
    fn invariant_signed_area_orientation() {
        let ccw = vec![
            Vec2::new(0.0, 0.0),
            Vec2::new(10.0, 0.0),
            Vec2::new(10.0, 10.0),
            Vec2::new(0.0, 10.0),
        ];

        let cw: Vec<Vec2> = ccw.iter().rev().copied().collect();

        let ccw_area = signed_area(&ccw);
        let cw_area = signed_area(&cw);

        assert!(
            ccw_area > 0.0,
            "CCW polygon should have positive signed area, got {ccw_area}"
        );
        assert!(
            cw_area < 0.0,
            "CW polygon should have negative signed area, got {cw_area}"
        );
    }

    // ========================================================================
    // Helper functions
    // ========================================================================

    fn flatten_path_for_area(path: &Path, segments: usize) -> Vec<Vec2> {
        use crate::bezier::{cubic_point, quadratic_point};

        let mut points = Vec::new();
        let mut current = Vec2::ZERO;

        for cmd in path.commands() {
            match cmd {
                PathCommand::MoveTo(p) => {
                    current = *p;
                    points.push(*p);
                }
                PathCommand::LineTo(p) => {
                    current = *p;
                    points.push(*p);
                }
                PathCommand::QuadTo { control, to } => {
                    for i in 1..=segments {
                        let t = i as f32 / segments as f32;
                        let p = quadratic_point(current, *control, *to, t);
                        points.push(p);
                    }
                    current = *to;
                }
                PathCommand::CubicTo {
                    control1,
                    control2,
                    to,
                } => {
                    for i in 1..=segments {
                        let t = i as f32 / segments as f32;
                        let p = cubic_point(current, *control1, *control2, *to, t);
                        points.push(p);
                    }
                    current = *to;
                }
                PathCommand::Close => {}
            }
        }

        points.dedup_by(|a, b| a.distance(*b) < 0.0001);
        points
    }
}
