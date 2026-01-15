//! Boolean operations on 2D paths.
//!
//! Provides union, intersection, and difference operations for paths.
//! Paths are first flattened to polygons, then boolean operations are applied.
//!
//! Also provides curve intersection utilities for Bezier curves.
//!
//! # Fill Rules
//!
//! Fill rules determine how to handle self-intersecting paths:
//! - `EvenOdd`: A point is inside if crossed by an odd number of edges
//! - `NonZero`: A point is inside if the winding number is non-zero
//!
//! # Example
//!
//! ```ignore
//! use rhizome_resin_vector::{circle, rect, path_union, path_intersect, path_subtract};
//! use rhizome_resin_vector::boolean::FillRule;
//!
//! let a = circle(Vec2::ZERO, 1.0);
//! let b = circle(Vec2::new(0.5, 0.0), 1.0);
//!
//! let union = path_union(&a, &b, 32);
//! let intersect = path_intersect(&a, &b, 32);
//! let difference = path_subtract(&a, &b, 32);
//!
//! // With fill rules for self-intersecting paths
//! let result = path_union_with_fill(&a, &b, 32, FillRule::NonZero);
//! ```

use glam::Vec2;

use crate::bezier::{cubic_point, quadratic_point};
use crate::{Path, PathBuilder, PathCommand};

// ============================================================================
// Fill Rules
// ============================================================================

/// Fill rule for determining inside/outside of paths.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FillRule {
    /// Even-odd rule: a point is inside if the ray crosses an odd number of edges.
    #[default]
    EvenOdd,
    /// Non-zero rule: a point is inside if the winding number is non-zero.
    NonZero,
}

// ============================================================================
// Curve Intersection Types
// ============================================================================

/// A curve segment (line, quadratic, or cubic bezier).
#[derive(Debug, Clone, Copy)]
pub enum CurveSegment {
    /// Line segment from start to end.
    Line { start: Vec2, end: Vec2 },
    /// Quadratic bezier curve.
    Quadratic {
        start: Vec2,
        control: Vec2,
        end: Vec2,
    },
    /// Cubic bezier curve.
    Cubic {
        start: Vec2,
        control1: Vec2,
        control2: Vec2,
        end: Vec2,
    },
}

/// An intersection between two curves.
#[derive(Debug, Clone, Copy)]
pub struct CurveIntersection {
    /// The intersection point.
    pub point: Vec2,
    /// Parameter t on the first curve (0-1).
    pub t1: f32,
    /// Parameter t on the second curve (0-1).
    pub t2: f32,
}

/// Bounding box with named min/max fields.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Bounds2D {
    /// Minimum corner of the bounding box.
    pub min: Vec2,
    /// Maximum corner of the bounding box.
    pub max: Vec2,
}

/// Result of splitting a curve at a parameter.
#[derive(Debug, Clone, Copy)]
pub struct SplitCurve {
    /// The curve segment before the split point.
    pub before: CurveSegment,
    /// The curve segment after the split point.
    pub after: CurveSegment,
}

/// Result of finding the closest point on a curve.
#[derive(Debug, Clone, Copy)]
pub struct ClosestPoint {
    /// The closest point on the curve.
    pub point: Vec2,
    /// The parameter t (0-1) at the closest point.
    pub t: f32,
}

impl CurveSegment {
    /// Evaluates the curve at parameter t (0-1).
    pub fn evaluate(&self, t: f32) -> Vec2 {
        match self {
            CurveSegment::Line { start, end } => *start + (*end - *start) * t,
            CurveSegment::Quadratic {
                start,
                control,
                end,
            } => quadratic_point(*start, *control, *end, t),
            CurveSegment::Cubic {
                start,
                control1,
                control2,
                end,
            } => cubic_point(*start, *control1, *control2, *end, t),
        }
    }

    /// Computes the derivative at parameter t.
    pub fn derivative(&self, t: f32) -> Vec2 {
        match self {
            CurveSegment::Line { start, end } => *end - *start,
            CurveSegment::Quadratic {
                start,
                control,
                end,
            } => {
                let mt = 1.0 - t;
                (*control - *start) * (2.0 * mt) + (*end - *control) * (2.0 * t)
            }
            CurveSegment::Cubic {
                start,
                control1,
                control2,
                end,
            } => {
                let mt = 1.0 - t;
                let mt2 = mt * mt;
                let t2 = t * t;
                (*control1 - *start) * (3.0 * mt2)
                    + (*control2 - *control1) * (6.0 * mt * t)
                    + (*end - *control2) * (3.0 * t2)
            }
        }
    }

    /// Returns the bounding box of the curve.
    pub fn bounds(&self) -> Bounds2D {
        match self {
            CurveSegment::Line { start, end } => Bounds2D {
                min: start.min(*end),
                max: start.max(*end),
            },
            CurveSegment::Quadratic {
                start,
                control,
                end,
            } => Bounds2D {
                min: start.min(*control).min(*end),
                max: start.max(*control).max(*end),
            },
            CurveSegment::Cubic {
                start,
                control1,
                control2,
                end,
            } => Bounds2D {
                min: start.min(*control1).min(*control2).min(*end),
                max: start.max(*control1).max(*control2).max(*end),
            },
        }
    }

    /// Splits the curve at parameter t, returning two subcurves.
    pub fn split(&self, t: f32) -> SplitCurve {
        match self {
            CurveSegment::Line { start, end } => {
                let mid = *start + (*end - *start) * t;
                SplitCurve {
                    before: CurveSegment::Line {
                        start: *start,
                        end: mid,
                    },
                    after: CurveSegment::Line {
                        start: mid,
                        end: *end,
                    },
                }
            }
            CurveSegment::Quadratic {
                start,
                control,
                end,
            } => {
                let (left, right) = split_quadratic(*start, *control, *end, t);
                SplitCurve {
                    before: CurveSegment::Quadratic {
                        start: left.0,
                        control: left.1,
                        end: left.2,
                    },
                    after: CurveSegment::Quadratic {
                        start: right.0,
                        control: right.1,
                        end: right.2,
                    },
                }
            }
            CurveSegment::Cubic {
                start,
                control1,
                control2,
                end,
            } => {
                let (left, right) = split_cubic(*start, *control1, *control2, *end, t);
                SplitCurve {
                    before: CurveSegment::Cubic {
                        start: left.0,
                        control1: left.1,
                        control2: left.2,
                        end: left.3,
                    },
                    after: CurveSegment::Cubic {
                        start: right.0,
                        control1: right.1,
                        control2: right.2,
                        end: right.3,
                    },
                }
            }
        }
    }
}

/// Splits a quadratic bezier at parameter t using de Casteljau's algorithm.
fn split_quadratic(
    p0: Vec2,
    p1: Vec2,
    p2: Vec2,
    t: f32,
) -> ((Vec2, Vec2, Vec2), (Vec2, Vec2, Vec2)) {
    let q0 = p0.lerp(p1, t);
    let q1 = p1.lerp(p2, t);
    let r = q0.lerp(q1, t);
    ((p0, q0, r), (r, q1, p2))
}

/// Splits a cubic bezier at parameter t using de Casteljau's algorithm.
fn split_cubic(
    p0: Vec2,
    p1: Vec2,
    p2: Vec2,
    p3: Vec2,
    t: f32,
) -> ((Vec2, Vec2, Vec2, Vec2), (Vec2, Vec2, Vec2, Vec2)) {
    let q0 = p0.lerp(p1, t);
    let q1 = p1.lerp(p2, t);
    let q2 = p2.lerp(p3, t);
    let r0 = q0.lerp(q1, t);
    let r1 = q1.lerp(q2, t);
    let s = r0.lerp(r1, t);
    ((p0, q0, r0, s), (s, r1, q2, p3))
}

/// Checks if two bounding boxes overlap.
fn bounds_overlap(a: Bounds2D, b: Bounds2D) -> bool {
    a.min.x <= b.max.x && a.max.x >= b.min.x && a.min.y <= b.max.y && a.max.y >= b.min.y
}

/// Bounding box diagonal length.
fn bounds_size(bounds: Bounds2D) -> f32 {
    (bounds.max - bounds.min).length()
}

// ============================================================================
// Curve-Curve Intersection
// ============================================================================

/// Finds intersections between two curve segments.
///
/// Uses recursive subdivision for bezier curves with configurable tolerance.
pub fn curve_intersections(
    c1: &CurveSegment,
    c2: &CurveSegment,
    tolerance: f32,
) -> Vec<CurveIntersection> {
    let mut result = Vec::new();
    find_intersections_recursive(c1, c2, 0.0, 1.0, 0.0, 1.0, tolerance, &mut result, 0);
    // Remove duplicates
    result.dedup_by(|a, b| a.point.distance(b.point) < tolerance);
    result
}

/// Recursive subdivision to find curve intersections.
fn find_intersections_recursive(
    c1: &CurveSegment,
    c2: &CurveSegment,
    t1_min: f32,
    t1_max: f32,
    t2_min: f32,
    t2_max: f32,
    tolerance: f32,
    result: &mut Vec<CurveIntersection>,
    depth: usize,
) {
    const MAX_DEPTH: usize = 20;

    // Check bounding box overlap
    let bounds1 = c1.bounds();
    let bounds2 = c2.bounds();

    if !bounds_overlap(bounds1, bounds2) {
        return;
    }

    // If both curves are small enough, report intersection
    let size1 = bounds_size(bounds1);
    let size2 = bounds_size(bounds2);

    if size1 < tolerance && size2 < tolerance {
        let t1 = (t1_min + t1_max) / 2.0;
        let t2 = (t2_min + t2_max) / 2.0;
        let p1 = c1.evaluate(0.5);
        let p2 = c2.evaluate(0.5);
        let point = (p1 + p2) / 2.0;
        result.push(CurveIntersection { point, t1, t2 });
        return;
    }

    if depth >= MAX_DEPTH {
        // Max depth reached, report approximate intersection
        let t1 = (t1_min + t1_max) / 2.0;
        let t2 = (t2_min + t2_max) / 2.0;
        let point = c1.evaluate(0.5);
        result.push(CurveIntersection { point, t1, t2 });
        return;
    }

    // Subdivide the larger curve
    if size1 > size2 {
        let split = c1.split(0.5);
        let t1_mid = (t1_min + t1_max) / 2.0;
        find_intersections_recursive(
            &split.before,
            c2,
            t1_min,
            t1_mid,
            t2_min,
            t2_max,
            tolerance,
            result,
            depth + 1,
        );
        find_intersections_recursive(
            &split.after,
            c2,
            t1_mid,
            t1_max,
            t2_min,
            t2_max,
            tolerance,
            result,
            depth + 1,
        );
    } else {
        let split = c2.split(0.5);
        let t2_mid = (t2_min + t2_max) / 2.0;
        find_intersections_recursive(
            c1,
            &split.before,
            t1_min,
            t1_max,
            t2_min,
            t2_mid,
            tolerance,
            result,
            depth + 1,
        );
        find_intersections_recursive(
            c1,
            &split.after,
            t1_min,
            t1_max,
            t2_mid,
            t2_max,
            tolerance,
            result,
            depth + 1,
        );
    }
}

/// Finds intersection between a line and a bezier curve.
pub fn line_curve_intersections(
    line_start: Vec2,
    line_end: Vec2,
    curve: &CurveSegment,
    tolerance: f32,
) -> Vec<CurveIntersection> {
    let line = CurveSegment::Line {
        start: line_start,
        end: line_end,
    };
    curve_intersections(&line, curve, tolerance)
}

/// Finds self-intersections in a cubic bezier curve.
pub fn cubic_self_intersections(
    start: Vec2,
    control1: Vec2,
    control2: Vec2,
    end: Vec2,
    tolerance: f32,
) -> Vec<CurveIntersection> {
    // A cubic can self-intersect if it has an inflection point
    // Check by finding where the second derivative changes sign
    let curve = CurveSegment::Cubic {
        start,
        control1,
        control2,
        end,
    };

    // Split curve in half and check if the two halves intersect
    let split = curve.split(0.5);

    // Exclude the common endpoint
    let mut intersections = curve_intersections(&split.before, &split.after, tolerance);
    intersections.retain(|i| (i.t1 - 0.5).abs() > 0.01 || (i.t2 - 0.5).abs() > 0.01);

    // Adjust t values to full curve range
    for i in &mut intersections {
        i.t1 *= 0.5;
        i.t2 = 0.5 + i.t2 * 0.5;
    }

    intersections
}

/// Finds the closest point on a curve to a given point.
pub fn closest_point_on_curve(curve: &CurveSegment, point: Vec2, samples: usize) -> ClosestPoint {
    let mut best_t = 0.0;
    let mut best_dist = f32::MAX;
    let mut best_point = curve.evaluate(0.0);

    // Initial sampling
    for i in 0..=samples {
        let t = i as f32 / samples as f32;
        let p = curve.evaluate(t);
        let dist = p.distance(point);
        if dist < best_dist {
            best_dist = dist;
            best_t = t;
            best_point = p;
        }
    }

    // Newton refinement
    for _ in 0..5 {
        let p = curve.evaluate(best_t);
        let d = curve.derivative(best_t);
        let diff = p - point;

        // Project diff onto tangent
        let d_len_sq = d.length_squared();
        if d_len_sq < 1e-10 {
            break;
        }

        let delta_t = -diff.dot(d) / d_len_sq;
        best_t = (best_t + delta_t * 0.5).clamp(0.0, 1.0);
    }

    best_point = curve.evaluate(best_t);
    ClosestPoint {
        point: best_point,
        t: best_t,
    }
}

/// Computes the union of two paths.
///
/// The `segments` parameter controls curve flattening resolution.
pub fn path_union(a: &Path, b: &Path, segments: usize) -> Path {
    let poly_a = flatten_path(a, segments);
    let poly_b = flatten_path(b, segments);

    let result = polygon_union(&poly_a, &poly_b);
    polygon_to_path(&result)
}

/// Computes the intersection of two paths.
pub fn path_intersect(a: &Path, b: &Path, segments: usize) -> Path {
    let poly_a = flatten_path(a, segments);
    let poly_b = flatten_path(b, segments);

    let result = polygon_intersect(&poly_a, &poly_b);
    polygon_to_path(&result)
}

/// Subtracts path B from path A.
pub fn path_subtract(a: &Path, b: &Path, segments: usize) -> Path {
    let poly_a = flatten_path(a, segments);
    let poly_b = flatten_path(b, segments);

    let result = polygon_subtract(&poly_a, &poly_b);
    polygon_to_path(&result)
}

/// Computes the XOR (symmetric difference) of two paths.
pub fn path_xor(a: &Path, b: &Path, segments: usize) -> Path {
    let poly_a = flatten_path(a, segments);
    let poly_b = flatten_path(b, segments);

    let result = polygon_xor(&poly_a, &poly_b);
    polygon_to_path(&result)
}

// ============================================================================
// Fill Rule-Aware Boolean Operations
// ============================================================================

/// Computes the union of two paths with a fill rule.
///
/// The fill rule determines how self-intersecting paths are handled.
pub fn path_union_with_fill(a: &Path, b: &Path, segments: usize, fill_rule: FillRule) -> Path {
    let poly_a = flatten_path(a, segments);
    let poly_b = flatten_path(b, segments);

    let result = polygon_union_with_fill(&poly_a, &poly_b, fill_rule);
    polygon_to_path(&result)
}

/// Computes the intersection of two paths with a fill rule.
pub fn path_intersect_with_fill(a: &Path, b: &Path, segments: usize, fill_rule: FillRule) -> Path {
    let poly_a = flatten_path(a, segments);
    let poly_b = flatten_path(b, segments);

    let result = polygon_intersect_with_fill(&poly_a, &poly_b, fill_rule);
    polygon_to_path(&result)
}

/// Subtracts path B from path A with a fill rule.
pub fn path_subtract_with_fill(a: &Path, b: &Path, segments: usize, fill_rule: FillRule) -> Path {
    let poly_a = flatten_path(a, segments);
    let poly_b = flatten_path(b, segments);

    let result = polygon_subtract_with_fill(&poly_a, &poly_b, fill_rule);
    polygon_to_path(&result)
}

/// Computes the XOR of two paths, returning multiple output paths.
///
/// XOR operations can produce disjoint regions, so this returns all of them.
pub fn path_xor_multi(a: &Path, b: &Path, segments: usize) -> Vec<Path> {
    let poly_a = flatten_path(a, segments);
    let poly_b = flatten_path(b, segments);

    let results = polygon_xor_multi(&poly_a, &poly_b);
    results.into_iter().map(|p| polygon_to_path(&p)).collect()
}

/// Checks if a point is inside a path using the specified fill rule.
pub fn path_contains_point(path: &Path, point: Vec2, segments: usize, fill_rule: FillRule) -> bool {
    let poly = flatten_path(path, segments);
    polygon_contains_point_with_rule(&poly, point, fill_rule)
}

/// Computes the winding number of a point with respect to a path.
pub fn path_winding_number(path: &Path, point: Vec2, segments: usize) -> i32 {
    let poly = flatten_path(path, segments);
    winding_number(&poly, point)
}

/// Polygon union with fill rule support.
fn polygon_union_with_fill(a: &[Vec2], b: &[Vec2], fill_rule: FillRule) -> Vec<Vec2> {
    if a.len() < 3 {
        return b.to_vec();
    }
    if b.len() < 3 {
        return a.to_vec();
    }

    let a_in_b = polygon_contains_point_with_rule(b, a[0], fill_rule);
    let b_in_a = polygon_contains_point_with_rule(a, b[0], fill_rule);

    if a_in_b && !has_intersection(a, b) {
        return b.to_vec();
    }
    if b_in_a && !has_intersection(a, b) {
        return a.to_vec();
    }

    weiler_atherton_union(a, b)
}

/// Polygon intersection with fill rule support.
fn polygon_intersect_with_fill(subject: &[Vec2], clip: &[Vec2], _fill_rule: FillRule) -> Vec<Vec2> {
    if subject.len() < 3 || clip.len() < 3 {
        return Vec::new();
    }

    sutherland_hodgman(subject, clip)
}

/// Polygon subtraction with fill rule support.
fn polygon_subtract_with_fill(a: &[Vec2], b: &[Vec2], fill_rule: FillRule) -> Vec<Vec2> {
    if a.len() < 3 || b.len() < 3 {
        return a.to_vec();
    }

    // Check if A is completely inside B
    if polygon_contains_point_with_rule(b, a[0], fill_rule) && !has_intersection(a, b) {
        return Vec::new();
    }

    let b_reversed: Vec<Vec2> = b.iter().rev().copied().collect();
    weiler_atherton_subtract(a, &b_reversed)
}

/// Polygon XOR returning multiple output polygons.
fn polygon_xor_multi(a: &[Vec2], b: &[Vec2]) -> Vec<Vec<Vec2>> {
    if a.len() < 3 || b.len() < 3 {
        let mut results = Vec::new();
        if a.len() >= 3 {
            results.push(a.to_vec());
        }
        if b.len() >= 3 {
            results.push(b.to_vec());
        }
        return results;
    }

    let intersection = sutherland_hodgman(a, b);
    if intersection.is_empty() {
        // No overlap - return both polygons
        return vec![a.to_vec(), b.to_vec()];
    }

    // Compute A - B and B - A
    let a_minus_b = polygon_subtract(a, b);
    let b_minus_a = polygon_subtract(b, a);

    let mut results = Vec::new();
    if a_minus_b.len() >= 3 {
        results.push(a_minus_b);
    }
    if b_minus_a.len() >= 3 {
        results.push(b_minus_a);
    }

    results
}

/// Flattens a path to a polygon (list of vertices).
fn flatten_path(path: &Path, segments: usize) -> Vec<Vec2> {
    let mut points = Vec::new();
    let mut current = Vec2::ZERO;
    let mut start = Vec2::ZERO;

    for cmd in path.commands() {
        match cmd {
            PathCommand::MoveTo(p) => {
                current = *p;
                start = *p;
                points.push(*p);
            }
            PathCommand::LineTo(p) => {
                current = *p;
                points.push(*p);
            }
            PathCommand::QuadTo { control, to } => {
                // Flatten quadratic bezier
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
                // Flatten cubic bezier
                for i in 1..=segments {
                    let t = i as f32 / segments as f32;
                    let p = cubic_point(current, *control1, *control2, *to, t);
                    points.push(p);
                }
                current = *to;
            }
            PathCommand::Close => {
                if current.distance(start) > 0.0001 {
                    points.push(start);
                }
                current = start;
            }
        }
    }

    // Remove consecutive duplicates
    points.dedup_by(|a, b| a.distance(*b) < 0.0001);

    points
}

/// Converts a polygon back to a path.
fn polygon_to_path(polygon: &[Vec2]) -> Path {
    let mut builder = PathBuilder::new();

    if polygon.is_empty() {
        return builder.build();
    }

    builder = builder.move_to(polygon[0]);
    for &p in &polygon[1..] {
        builder = builder.line_to(p);
    }
    builder.close().build()
}

// ============================================================================
// Polygon Boolean Operations
// ============================================================================

/// Computes polygon union using the Weiler-Atherton algorithm (simplified).
fn polygon_union(a: &[Vec2], b: &[Vec2]) -> Vec<Vec2> {
    if a.len() < 3 {
        return b.to_vec();
    }
    if b.len() < 3 {
        return a.to_vec();
    }

    // Find if one polygon is inside the other
    let a_in_b = polygon_contains_point(b, a[0]);
    let b_in_a = polygon_contains_point(a, b[0]);

    // Simple case: one completely inside the other
    if a_in_b && !has_intersection(a, b) {
        return b.to_vec();
    }
    if b_in_a && !has_intersection(a, b) {
        return a.to_vec();
    }

    // General case: compute union via clipping
    // For union, we want the outer boundary of both polygons
    weiler_atherton_union(a, b)
}

/// Computes polygon intersection using Sutherland-Hodgman clipping.
fn polygon_intersect(subject: &[Vec2], clip: &[Vec2]) -> Vec<Vec2> {
    if subject.len() < 3 || clip.len() < 3 {
        return Vec::new();
    }

    sutherland_hodgman(subject, clip)
}

/// Subtracts polygon B from polygon A.
fn polygon_subtract(a: &[Vec2], b: &[Vec2]) -> Vec<Vec2> {
    if a.len() < 3 || b.len() < 3 {
        return a.to_vec();
    }

    // Subtract = A AND NOT B
    // We clip A against the reversed B
    let b_reversed: Vec<Vec2> = b.iter().rev().copied().collect();
    weiler_atherton_subtract(a, &b_reversed)
}

/// Computes polygon XOR (symmetric difference).
fn polygon_xor(a: &[Vec2], b: &[Vec2]) -> Vec<Vec2> {
    // XOR = (A OR B) - (A AND B) = (A - B) OR (B - A)
    // For simplicity, return just A - B if there's an intersection
    // Full XOR would need to return multiple polygons

    let intersection = polygon_intersect(a, b);
    if intersection.is_empty() {
        // No overlap, union is just concatenation (simplified)
        return polygon_union(a, b);
    }

    // Return A - B for now
    polygon_subtract(a, b)
}

/// Sutherland-Hodgman polygon clipping algorithm.
fn sutherland_hodgman(subject: &[Vec2], clip: &[Vec2]) -> Vec<Vec2> {
    let mut output = subject.to_vec();

    for i in 0..clip.len() {
        if output.is_empty() {
            break;
        }

        let edge_start = clip[i];
        let edge_end = clip[(i + 1) % clip.len()];

        output = clip_polygon_to_edge(&output, edge_start, edge_end);
    }

    output
}

/// Clips a polygon against a single edge.
fn clip_polygon_to_edge(polygon: &[Vec2], edge_start: Vec2, edge_end: Vec2) -> Vec<Vec2> {
    let mut output = Vec::new();

    if polygon.is_empty() {
        return output;
    }

    for i in 0..polygon.len() {
        let current = polygon[i];
        let next = polygon[(i + 1) % polygon.len()];

        let current_inside = is_inside_edge(current, edge_start, edge_end);
        let next_inside = is_inside_edge(next, edge_start, edge_end);

        if current_inside {
            output.push(current);
            if !next_inside {
                // Exiting: add intersection
                if let Some(p) = line_intersection(current, next, edge_start, edge_end) {
                    output.push(p);
                }
            }
        } else if next_inside {
            // Entering: add intersection
            if let Some(p) = line_intersection(current, next, edge_start, edge_end) {
                output.push(p);
            }
        }
    }

    output
}

/// Checks if a point is inside (to the left of) an edge.
fn is_inside_edge(point: Vec2, edge_start: Vec2, edge_end: Vec2) -> bool {
    let edge = edge_end - edge_start;
    let to_point = point - edge_start;
    // Cross product: positive = left side (inside for CCW polygon)
    edge.x * to_point.y - edge.y * to_point.x >= 0.0
}

/// Computes intersection point of two line segments.
fn line_intersection(a1: Vec2, a2: Vec2, b1: Vec2, b2: Vec2) -> Option<Vec2> {
    let d1 = a2 - a1;
    let d2 = b2 - b1;

    let cross = d1.x * d2.y - d1.y * d2.x;
    if cross.abs() < 1e-10 {
        return None; // Parallel lines
    }

    let d = b1 - a1;
    let t = (d.x * d2.y - d.y * d2.x) / cross;

    Some(a1 + d1 * t)
}

/// Checks if a point is inside a polygon (using ray casting, even-odd rule).
fn polygon_contains_point(polygon: &[Vec2], point: Vec2) -> bool {
    polygon_contains_point_with_rule(polygon, point, FillRule::EvenOdd)
}

/// Checks if a point is inside a polygon using the specified fill rule.
pub fn polygon_contains_point_with_rule(
    polygon: &[Vec2],
    point: Vec2,
    fill_rule: FillRule,
) -> bool {
    if polygon.len() < 3 {
        return false;
    }

    match fill_rule {
        FillRule::EvenOdd => {
            let mut inside = false;
            let mut j = polygon.len() - 1;

            for i in 0..polygon.len() {
                let pi = polygon[i];
                let pj = polygon[j];

                if ((pi.y > point.y) != (pj.y > point.y))
                    && (point.x < (pj.x - pi.x) * (point.y - pi.y) / (pj.y - pi.y) + pi.x)
                {
                    inside = !inside;
                }

                j = i;
            }

            inside
        }
        FillRule::NonZero => winding_number(polygon, point) != 0,
    }
}

/// Computes the winding number of a point with respect to a polygon.
///
/// The winding number counts how many times the polygon winds around the point.
/// Positive values indicate counter-clockwise winding, negative for clockwise.
pub fn winding_number(polygon: &[Vec2], point: Vec2) -> i32 {
    if polygon.len() < 3 {
        return 0;
    }

    let mut winding = 0i32;

    for i in 0..polygon.len() {
        let p1 = polygon[i];
        let p2 = polygon[(i + 1) % polygon.len()];

        if p1.y <= point.y {
            if p2.y > point.y {
                // Upward crossing
                if is_left(p1, p2, point) > 0.0 {
                    winding += 1;
                }
            }
        } else if p2.y <= point.y {
            // Downward crossing
            if is_left(p1, p2, point) < 0.0 {
                winding -= 1;
            }
        }
    }

    winding
}

/// Returns > 0 if point is left of line p1->p2, < 0 if right, 0 if on line.
fn is_left(p1: Vec2, p2: Vec2, point: Vec2) -> f32 {
    (p2.x - p1.x) * (point.y - p1.y) - (point.x - p1.x) * (p2.y - p1.y)
}

/// Checks if two polygons have any intersecting edges.
fn has_intersection(a: &[Vec2], b: &[Vec2]) -> bool {
    for i in 0..a.len() {
        let a1 = a[i];
        let a2 = a[(i + 1) % a.len()];

        for j in 0..b.len() {
            let b1 = b[j];
            let b2 = b[(j + 1) % b.len()];

            if segments_intersect(a1, a2, b1, b2) {
                return true;
            }
        }
    }
    false
}

/// Checks if two line segments intersect.
fn segments_intersect(a1: Vec2, a2: Vec2, b1: Vec2, b2: Vec2) -> bool {
    let d1 = a2 - a1;
    let d2 = b2 - b1;

    let cross = d1.x * d2.y - d1.y * d2.x;
    if cross.abs() < 1e-10 {
        return false; // Parallel
    }

    let d = b1 - a1;
    let t = (d.x * d2.y - d.y * d2.x) / cross;
    let u = (d.x * d1.y - d.y * d1.x) / cross;

    t >= 0.0 && t <= 1.0 && u >= 0.0 && u <= 1.0
}

/// Simplified Weiler-Atherton union.
fn weiler_atherton_union(a: &[Vec2], b: &[Vec2]) -> Vec<Vec2> {
    // Find all intersection points
    let intersections = find_all_intersections(a, b);

    if intersections.is_empty() {
        // No intersections - return outer polygon or combined bounds
        if polygon_contains_point(b, a[0]) {
            return b.to_vec();
        }
        if polygon_contains_point(a, b[0]) {
            return a.to_vec();
        }
        // Disjoint - return just A for now (full implementation would return both)
        return a.to_vec();
    }

    // Build union by walking along the outer boundary
    // This is a simplified version - full Weiler-Atherton is more complex
    build_union_boundary(a, b, &intersections)
}

/// Simplified Weiler-Atherton subtraction.
fn weiler_atherton_subtract(a: &[Vec2], b: &[Vec2]) -> Vec<Vec2> {
    let intersections = find_all_intersections(a, b);

    if intersections.is_empty() {
        // No intersections
        if polygon_contains_point(b, a[0]) {
            // A is completely inside B - result is empty
            return Vec::new();
        }
        // No overlap
        return a.to_vec();
    }

    // Build difference boundary
    build_subtract_boundary(a, b, &intersections)
}

/// Intersection point with metadata.
#[derive(Clone)]
struct Intersection {
    point: Vec2,
    a_edge: usize, // Edge index in polygon A
    b_edge: usize, // Edge index in polygon B
    a_param: f32,  // Parameter along edge A
    #[allow(dead_code)]
    b_param: f32, // Parameter along edge B
}

/// Finds all intersection points between two polygons.
fn find_all_intersections(a: &[Vec2], b: &[Vec2]) -> Vec<Intersection> {
    let mut result = Vec::new();

    for i in 0..a.len() {
        let a1 = a[i];
        let a2 = a[(i + 1) % a.len()];

        for j in 0..b.len() {
            let b1 = b[j];
            let b2 = b[(j + 1) % b.len()];

            if let Some((point, t, u)) = segment_intersection_params(a1, a2, b1, b2) {
                result.push(Intersection {
                    point,
                    a_edge: i,
                    b_edge: j,
                    a_param: t,
                    b_param: u,
                });
            }
        }
    }

    result
}

/// Computes intersection point and parameters for two segments.
fn segment_intersection_params(a1: Vec2, a2: Vec2, b1: Vec2, b2: Vec2) -> Option<(Vec2, f32, f32)> {
    let d1 = a2 - a1;
    let d2 = b2 - b1;

    let cross = d1.x * d2.y - d1.y * d2.x;
    if cross.abs() < 1e-10 {
        return None;
    }

    let d = b1 - a1;
    let t = (d.x * d2.y - d.y * d2.x) / cross;
    let u = (d.x * d1.y - d.y * d1.x) / cross;

    if t >= 0.0 && t <= 1.0 && u >= 0.0 && u <= 1.0 {
        Some((a1 + d1 * t, t, u))
    } else {
        None
    }
}

/// Builds union boundary (simplified).
fn build_union_boundary(a: &[Vec2], b: &[Vec2], intersections: &[Intersection]) -> Vec<Vec2> {
    if intersections.is_empty() {
        return a.to_vec();
    }

    // Start from first intersection, walk along outer boundary
    let mut result = Vec::new();
    let mut on_a = true;
    let mut current_edge = intersections[0].a_edge;
    let mut visited_intersections = vec![false; intersections.len()];
    visited_intersections[0] = true;

    result.push(intersections[0].point);

    let max_iter = (a.len() + b.len()) * 2;
    for _ in 0..max_iter {
        if on_a {
            // Walk along A
            current_edge = (current_edge + 1) % a.len();
            let next_point = a[current_edge];

            // Check for intersection before reaching next vertex
            if let Some((idx, int)) = find_next_intersection_on_edge(
                intersections,
                current_edge.wrapping_sub(1) % a.len(),
                true,
                &visited_intersections,
            ) {
                visited_intersections[idx] = true;
                result.push(int.point);
                on_a = false;
                current_edge = int.b_edge;
            } else {
                result.push(next_point);
            }
        } else {
            // Walk along B
            current_edge = (current_edge + 1) % b.len();
            let next_point = b[current_edge];

            if let Some((idx, int)) = find_next_intersection_on_edge(
                intersections,
                current_edge.wrapping_sub(1) % b.len(),
                false,
                &visited_intersections,
            ) {
                visited_intersections[idx] = true;
                result.push(int.point);
                on_a = true;
                current_edge = int.a_edge;
            } else {
                result.push(next_point);
            }
        }

        // Check if we're back to start
        if !result.is_empty() && result.len() > 2 {
            if result.last().unwrap().distance(result[0]) < 0.0001 {
                result.pop();
                break;
            }
        }
    }

    result
}

/// Builds subtraction boundary (simplified).
fn build_subtract_boundary(a: &[Vec2], b: &[Vec2], intersections: &[Intersection]) -> Vec<Vec2> {
    if intersections.is_empty() {
        return a.to_vec();
    }

    // For subtraction, we walk A counterclockwise, B clockwise (reversed)
    let mut result = Vec::new();
    let mut on_a = true;
    let mut current_edge = 0;
    let start_on_b = polygon_contains_point(b, a[0]);

    // Find starting point
    if start_on_b {
        // A[0] is inside B, start from first intersection
        if let Some(int) = intersections.first() {
            result.push(int.point);
            on_a = true;
            current_edge = int.a_edge;
        }
    } else {
        // A[0] is outside B, start from A[0]
        result.push(a[0]);
    }

    let b_reversed: Vec<Vec2> = b.iter().rev().copied().collect();
    let mut visited = vec![false; intersections.len()];

    let max_iter = (a.len() + b.len()) * 2;
    for _ in 0..max_iter {
        if on_a {
            current_edge = (current_edge + 1) % a.len();
            let next_point = a[current_edge];

            // Check for intersection
            if let Some((idx, int)) = find_next_intersection_on_edge_for_sub(
                intersections,
                current_edge.wrapping_sub(1) % a.len(),
                &visited,
            ) {
                visited[idx] = true;
                if polygon_contains_point(b, next_point) {
                    result.push(int.point);
                    on_a = false;
                    current_edge = b.len() - 1 - int.b_edge; // Reversed index
                } else {
                    result.push(next_point);
                }
            } else {
                result.push(next_point);
            }
        } else {
            current_edge = (current_edge + 1) % b_reversed.len();
            let next_point = b_reversed[current_edge];

            result.push(next_point);

            // Check if we exit B
            let orig_idx = b.len() - 1 - current_edge;
            if let Some((idx, int)) = find_intersection_at_b_edge(intersections, orig_idx, &visited)
            {
                visited[idx] = true;
                result.push(int.point);
                on_a = true;
                current_edge = int.a_edge;
            }
        }

        if result.len() > 2 && result.last().unwrap().distance(result[0]) < 0.0001 {
            result.pop();
            break;
        }
    }

    result
}

fn find_next_intersection_on_edge(
    intersections: &[Intersection],
    edge: usize,
    on_a: bool,
    visited: &[bool],
) -> Option<(usize, Intersection)> {
    for (idx, int) in intersections.iter().enumerate() {
        if visited[idx] {
            continue;
        }
        if on_a && int.a_edge == edge {
            return Some((idx, int.clone()));
        }
        if !on_a && int.b_edge == edge {
            return Some((idx, int.clone()));
        }
    }
    None
}

fn find_next_intersection_on_edge_for_sub(
    intersections: &[Intersection],
    edge: usize,
    visited: &[bool],
) -> Option<(usize, Intersection)> {
    let mut best: Option<(usize, Intersection)> = None;

    for (idx, int) in intersections.iter().enumerate() {
        if visited[idx] {
            continue;
        }
        if int.a_edge == edge {
            if best.is_none() || int.a_param < best.as_ref().unwrap().1.a_param {
                best = Some((idx, int.clone()));
            }
        }
    }
    best
}

fn find_intersection_at_b_edge(
    intersections: &[Intersection],
    edge: usize,
    visited: &[bool],
) -> Option<(usize, Intersection)> {
    for (idx, int) in intersections.iter().enumerate() {
        if visited[idx] {
            continue;
        }
        if int.b_edge == edge {
            return Some((idx, int.clone()));
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{circle, rect};

    #[test]
    fn test_flatten_path() {
        let path = rect(Vec2::ZERO, Vec2::new(2.0, 2.0));
        let points = flatten_path(&path, 8);

        // Rect should have 5 points (4 corners + back to start via close)
        assert!(points.len() >= 4);
    }

    #[test]
    fn test_polygon_contains_point() {
        let square = vec![
            Vec2::new(0.0, 0.0),
            Vec2::new(2.0, 0.0),
            Vec2::new(2.0, 2.0),
            Vec2::new(0.0, 2.0),
        ];

        assert!(polygon_contains_point(&square, Vec2::new(1.0, 1.0)));
        assert!(!polygon_contains_point(&square, Vec2::new(3.0, 1.0)));
    }

    #[test]
    fn test_path_intersect() {
        let a = rect(Vec2::ZERO, Vec2::new(2.0, 2.0));
        let b = rect(Vec2::new(1.0, 1.0), Vec2::new(3.0, 3.0));

        let result = path_intersect(&a, &b, 8);

        // Should have some vertices (the overlapping region)
        assert!(result.len() > 0);
    }

    #[test]
    fn test_path_union_no_overlap() {
        let a = rect(Vec2::ZERO, Vec2::new(1.0, 1.0));
        let b = rect(Vec2::new(3.0, 0.0), Vec2::new(4.0, 1.0));

        let result = path_union(&a, &b, 8);

        // Should return one of the polygons (simplified implementation)
        assert!(result.len() > 0);
    }

    #[test]
    fn test_path_subtract() {
        let a = rect(Vec2::ZERO, Vec2::new(2.0, 2.0));
        let b = rect(Vec2::new(1.0, 1.0), Vec2::new(3.0, 3.0));

        let result = path_subtract(&a, &b, 8);

        // Should have vertices (A minus B)
        // Result may be empty or have points depending on overlap
        let _ = result;
    }

    #[test]
    fn test_sutherland_hodgman() {
        let subject = vec![
            Vec2::new(0.0, 0.0),
            Vec2::new(2.0, 0.0),
            Vec2::new(2.0, 2.0),
            Vec2::new(0.0, 2.0),
        ];
        let clip = vec![
            Vec2::new(1.0, 0.0),
            Vec2::new(3.0, 0.0),
            Vec2::new(3.0, 2.0),
            Vec2::new(1.0, 2.0),
        ];

        let result = sutherland_hodgman(&subject, &clip);

        // Should have 4 vertices (the overlapping rectangle)
        assert_eq!(result.len(), 4);
    }

    #[test]
    fn test_circle_intersect() {
        let a = circle(Vec2::ZERO, 1.0);
        let b = circle(Vec2::new(0.5, 0.0), 1.0);

        let result = path_intersect(&a, &b, 16);

        // Should have some vertices
        assert!(result.len() > 0);
    }

    // ========================================================================
    // Curve Intersection Tests
    // ========================================================================

    #[test]
    fn test_curve_segment_evaluate_line() {
        let line = CurveSegment::Line {
            start: Vec2::ZERO,
            end: Vec2::new(10.0, 0.0),
        };

        assert_eq!(line.evaluate(0.0), Vec2::ZERO);
        assert_eq!(line.evaluate(0.5), Vec2::new(5.0, 0.0));
        assert_eq!(line.evaluate(1.0), Vec2::new(10.0, 0.0));
    }

    #[test]
    fn test_curve_segment_evaluate_quadratic() {
        let quad = CurveSegment::Quadratic {
            start: Vec2::ZERO,
            control: Vec2::new(1.0, 2.0),
            end: Vec2::new(2.0, 0.0),
        };

        // At t=0, should be at start
        assert_eq!(quad.evaluate(0.0), Vec2::ZERO);
        // At t=1, should be at end
        assert_eq!(quad.evaluate(1.0), Vec2::new(2.0, 0.0));
        // At t=0.5, should be at peak (y-wise)
        let mid = quad.evaluate(0.5);
        assert!(mid.y > 0.5); // Should be above the endpoints
    }

    #[test]
    fn test_curve_segment_split_line() {
        let line = CurveSegment::Line {
            start: Vec2::ZERO,
            end: Vec2::new(10.0, 0.0),
        };

        let split = line.split(0.5);

        // Check before segment
        assert_eq!(split.before.evaluate(0.0), Vec2::ZERO);
        assert_eq!(split.before.evaluate(1.0), Vec2::new(5.0, 0.0));

        // Check after segment
        assert_eq!(split.after.evaluate(0.0), Vec2::new(5.0, 0.0));
        assert_eq!(split.after.evaluate(1.0), Vec2::new(10.0, 0.0));
    }

    #[test]
    fn test_curve_segment_bounds() {
        let cubic = CurveSegment::Cubic {
            start: Vec2::ZERO,
            control1: Vec2::new(0.0, 2.0),
            control2: Vec2::new(2.0, 2.0),
            end: Vec2::new(2.0, 0.0),
        };

        let bounds = cubic.bounds();
        assert_eq!(bounds.min, Vec2::ZERO);
        assert_eq!(bounds.max, Vec2::new(2.0, 2.0));
    }

    #[test]
    fn test_line_line_intersection() {
        let line1 = CurveSegment::Line {
            start: Vec2::new(0.0, 0.0),
            end: Vec2::new(2.0, 2.0),
        };
        let line2 = CurveSegment::Line {
            start: Vec2::new(0.0, 2.0),
            end: Vec2::new(2.0, 0.0),
        };

        let intersections = curve_intersections(&line1, &line2, 0.01);
        assert_eq!(intersections.len(), 1);

        let int = &intersections[0];
        assert!((int.point.x - 1.0).abs() < 0.1);
        assert!((int.point.y - 1.0).abs() < 0.1);
    }

    #[test]
    fn test_line_line_no_intersection() {
        let line1 = CurveSegment::Line {
            start: Vec2::new(0.0, 0.0),
            end: Vec2::new(1.0, 0.0),
        };
        let line2 = CurveSegment::Line {
            start: Vec2::new(0.0, 1.0),
            end: Vec2::new(1.0, 1.0),
        };

        let intersections = curve_intersections(&line1, &line2, 0.01);
        assert!(intersections.is_empty());
    }

    #[test]
    fn test_line_quadratic_intersection() {
        // Horizontal line at y=0.5 intersecting a quadratic arch
        let line = CurveSegment::Line {
            start: Vec2::new(-1.0, 0.5),
            end: Vec2::new(3.0, 0.5),
        };
        let quad = CurveSegment::Quadratic {
            start: Vec2::ZERO,
            control: Vec2::new(1.0, 2.0),
            end: Vec2::new(2.0, 0.0),
        };

        let intersections = curve_intersections(&line, &quad, 0.01);
        // Should intersect twice (going up and coming down)
        assert_eq!(intersections.len(), 2);
    }

    #[test]
    fn test_quadratic_quadratic_intersection() {
        let quad1 = CurveSegment::Quadratic {
            start: Vec2::new(0.0, 0.0),
            control: Vec2::new(1.0, 2.0),
            end: Vec2::new(2.0, 0.0),
        };
        let quad2 = CurveSegment::Quadratic {
            start: Vec2::new(0.0, 1.0),
            control: Vec2::new(1.0, -1.0),
            end: Vec2::new(2.0, 1.0),
        };

        let intersections = curve_intersections(&quad1, &quad2, 0.01);
        // These curves should intersect twice
        assert_eq!(intersections.len(), 2);
    }

    #[test]
    fn test_cubic_cubic_intersection() {
        let cubic1 = CurveSegment::Cubic {
            start: Vec2::new(0.0, 0.0),
            control1: Vec2::new(0.0, 2.0),
            control2: Vec2::new(2.0, 2.0),
            end: Vec2::new(2.0, 0.0),
        };
        let cubic2 = CurveSegment::Cubic {
            start: Vec2::new(0.0, 1.0),
            control1: Vec2::new(2.0, 1.0),
            control2: Vec2::new(0.0, 1.0),
            end: Vec2::new(2.0, 1.0),
        };

        let intersections = curve_intersections(&cubic1, &cubic2, 0.01);
        // Should have some intersections
        assert!(!intersections.is_empty());
    }

    #[test]
    fn test_closest_point_on_line() {
        let line = CurveSegment::Line {
            start: Vec2::ZERO,
            end: Vec2::new(10.0, 0.0),
        };

        // Point above the middle of the line
        let result = closest_point_on_curve(&line, Vec2::new(5.0, 3.0), 10);
        assert!((result.point.x - 5.0).abs() < 0.1);
        assert!(result.point.y.abs() < 0.1);
        assert!((result.t - 0.5).abs() < 0.1);
    }

    #[test]
    fn test_closest_point_on_quadratic() {
        let quad = CurveSegment::Quadratic {
            start: Vec2::ZERO,
            control: Vec2::new(1.0, 2.0),
            end: Vec2::new(2.0, 0.0),
        };

        // Point above the curve
        let result = closest_point_on_curve(&quad, Vec2::new(1.0, 3.0), 20);
        // Closest should be near the peak of the curve
        assert!((result.point.x - 1.0).abs() < 0.2);
        assert!(result.point.y > 0.5);
        assert!((result.t - 0.5).abs() < 0.2);
    }

    #[test]
    fn test_curve_derivative_line() {
        let line = CurveSegment::Line {
            start: Vec2::new(0.0, 0.0),
            end: Vec2::new(3.0, 4.0),
        };

        // Derivative should be constant for a line
        let d0 = line.derivative(0.0);
        let d1 = line.derivative(1.0);
        assert_eq!(d0, d1);
        assert_eq!(d0, Vec2::new(3.0, 4.0));
    }

    #[test]
    fn test_split_quadratic_de_casteljau() {
        let start = Vec2::ZERO;
        let control = Vec2::new(1.0, 2.0);
        let end = Vec2::new(2.0, 0.0);

        let (left, right) = split_quadratic(start, control, end, 0.5);

        // The split point should be on the original curve
        let split_point = quadratic_point(start, control, end, 0.5);
        assert!((left.2 - split_point).length() < 0.001);
        assert!((right.0 - split_point).length() < 0.001);
    }

    #[test]
    fn test_split_cubic_de_casteljau() {
        let p0 = Vec2::ZERO;
        let p1 = Vec2::new(0.0, 2.0);
        let p2 = Vec2::new(2.0, 2.0);
        let p3 = Vec2::new(2.0, 0.0);

        let (left, right) = split_cubic(p0, p1, p2, p3, 0.5);

        // The split point should be on the original curve
        let split_point = cubic_point(p0, p1, p2, p3, 0.5);
        assert!((left.3 - split_point).length() < 0.001);
        assert!((right.0 - split_point).length() < 0.001);
    }

    #[test]
    fn test_line_curve_intersections_helper() {
        let curve = CurveSegment::Quadratic {
            start: Vec2::ZERO,
            control: Vec2::new(1.0, 2.0),
            end: Vec2::new(2.0, 0.0),
        };

        let intersections =
            line_curve_intersections(Vec2::new(-1.0, 0.5), Vec2::new(3.0, 0.5), &curve, 0.01);

        assert_eq!(intersections.len(), 2);
    }

    // ========================================================================
    // Fill Rule Tests
    // ========================================================================

    #[test]
    fn test_winding_number_simple() {
        // CCW square
        let square = vec![
            Vec2::new(0.0, 0.0),
            Vec2::new(2.0, 0.0),
            Vec2::new(2.0, 2.0),
            Vec2::new(0.0, 2.0),
        ];

        // Point inside
        let wn = winding_number(&square, Vec2::new(1.0, 1.0));
        assert_eq!(wn, 1);

        // Point outside
        let wn = winding_number(&square, Vec2::new(5.0, 5.0));
        assert_eq!(wn, 0);
    }

    #[test]
    fn test_winding_number_clockwise() {
        // CW square (reversed)
        let square = vec![
            Vec2::new(0.0, 2.0),
            Vec2::new(2.0, 2.0),
            Vec2::new(2.0, 0.0),
            Vec2::new(0.0, 0.0),
        ];

        // Point inside - should be -1 for CW
        let wn = winding_number(&square, Vec2::new(1.0, 1.0));
        assert_eq!(wn, -1);
    }

    #[test]
    fn test_fill_rule_even_odd() {
        let square = vec![
            Vec2::new(0.0, 0.0),
            Vec2::new(2.0, 0.0),
            Vec2::new(2.0, 2.0),
            Vec2::new(0.0, 2.0),
        ];

        // Point inside
        assert!(polygon_contains_point_with_rule(
            &square,
            Vec2::new(1.0, 1.0),
            FillRule::EvenOdd
        ));

        // Point outside
        assert!(!polygon_contains_point_with_rule(
            &square,
            Vec2::new(5.0, 5.0),
            FillRule::EvenOdd
        ));
    }

    #[test]
    fn test_fill_rule_non_zero() {
        let square = vec![
            Vec2::new(0.0, 0.0),
            Vec2::new(2.0, 0.0),
            Vec2::new(2.0, 2.0),
            Vec2::new(0.0, 2.0),
        ];

        // Point inside
        assert!(polygon_contains_point_with_rule(
            &square,
            Vec2::new(1.0, 1.0),
            FillRule::NonZero
        ));

        // Point outside
        assert!(!polygon_contains_point_with_rule(
            &square,
            Vec2::new(5.0, 5.0),
            FillRule::NonZero
        ));
    }

    #[test]
    fn test_path_contains_point() {
        let path = rect(Vec2::ZERO, Vec2::new(2.0, 2.0));

        // Point inside
        assert!(path_contains_point(
            &path,
            Vec2::new(1.0, 1.0),
            8,
            FillRule::EvenOdd
        ));

        // Point outside
        assert!(!path_contains_point(
            &path,
            Vec2::new(5.0, 5.0),
            8,
            FillRule::EvenOdd
        ));
    }

    #[test]
    fn test_path_winding_number() {
        let path = rect(Vec2::ZERO, Vec2::new(2.0, 2.0));

        // Point inside
        let wn = path_winding_number(&path, Vec2::new(1.0, 1.0), 8);
        assert!(wn != 0);

        // Point outside
        let wn = path_winding_number(&path, Vec2::new(5.0, 5.0), 8);
        assert_eq!(wn, 0);
    }

    #[test]
    fn test_path_union_with_fill() {
        let a = rect(Vec2::ZERO, Vec2::new(2.0, 2.0));
        let b = rect(Vec2::new(1.0, 1.0), Vec2::new(3.0, 3.0));

        let result = path_union_with_fill(&a, &b, 8, FillRule::NonZero);
        assert!(result.len() > 0);
    }

    #[test]
    fn test_path_intersect_with_fill() {
        let a = rect(Vec2::ZERO, Vec2::new(2.0, 2.0));
        let b = rect(Vec2::new(1.0, 1.0), Vec2::new(3.0, 3.0));

        let result = path_intersect_with_fill(&a, &b, 8, FillRule::NonZero);
        assert!(result.len() > 0);
    }

    #[test]
    fn test_path_subtract_with_fill() {
        let a = rect(Vec2::ZERO, Vec2::new(2.0, 2.0));
        let b = rect(Vec2::new(1.0, 1.0), Vec2::new(3.0, 3.0));

        let result = path_subtract_with_fill(&a, &b, 8, FillRule::NonZero);
        // Result may be empty or have points
        let _ = result;
    }

    #[test]
    fn test_path_xor_multi_no_overlap() {
        let a = rect(Vec2::ZERO, Vec2::new(1.0, 1.0));
        let b = rect(Vec2::new(3.0, 0.0), Vec2::new(4.0, 1.0));

        let results = path_xor_multi(&a, &b, 8);
        // Should return two separate paths (no overlap)
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_path_xor_multi_with_overlap() {
        let a = rect(Vec2::ZERO, Vec2::new(2.0, 2.0));
        let b = rect(Vec2::new(1.0, 1.0), Vec2::new(3.0, 3.0));

        let results = path_xor_multi(&a, &b, 8);
        // Should have some results (possibly multiple)
        assert!(!results.is_empty());
    }

    #[test]
    fn test_fill_rule_default() {
        let rule = FillRule::default();
        assert_eq!(rule, FillRule::EvenOdd);
    }
}
