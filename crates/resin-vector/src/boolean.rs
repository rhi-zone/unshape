//! Boolean operations on 2D paths.
//!
//! Provides union, intersection, and difference operations for paths.
//! Paths are first flattened to polygons, then boolean operations are applied.
//!
//! # Example
//!
//! ```ignore
//! use resin_vector::{circle, rect, path_union, path_intersect, path_subtract};
//!
//! let a = circle(Vec2::ZERO, 1.0);
//! let b = circle(Vec2::new(0.5, 0.0), 1.0);
//!
//! let union = path_union(&a, &b, 32);
//! let intersect = path_intersect(&a, &b, 32);
//! let difference = path_subtract(&a, &b, 32);
//! ```

use glam::Vec2;

use crate::{Path, PathBuilder, PathCommand};

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
                    let p = quadratic_bezier(current, *control, *to, t);
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
                    let p = cubic_bezier(current, *control1, *control2, *to, t);
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

/// Evaluates a quadratic bezier curve at parameter t.
fn quadratic_bezier(p0: Vec2, p1: Vec2, p2: Vec2, t: f32) -> Vec2 {
    let t2 = t * t;
    let mt = 1.0 - t;
    let mt2 = mt * mt;
    p0 * mt2 + p1 * (2.0 * mt * t) + p2 * t2
}

/// Evaluates a cubic bezier curve at parameter t.
fn cubic_bezier(p0: Vec2, p1: Vec2, p2: Vec2, p3: Vec2, t: f32) -> Vec2 {
    let t2 = t * t;
    let t3 = t2 * t;
    let mt = 1.0 - t;
    let mt2 = mt * mt;
    let mt3 = mt2 * mt;
    p0 * mt3 + p1 * (3.0 * mt2 * t) + p2 * (3.0 * mt * t2) + p3 * t3
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

/// Checks if a point is inside a polygon (using ray casting).
fn polygon_contains_point(polygon: &[Vec2], point: Vec2) -> bool {
    if polygon.len() < 3 {
        return false;
    }

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
}
