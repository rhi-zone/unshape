//! Geometric algorithms for 2D vectors.
//!
//! Provides algorithms like convex hull, bounding box, and point-in-polygon tests.

use crate::{Path, PathBuilder};
use glam::Vec2;

/// Computes the convex hull of a set of 2D points using Andrew's monotone chain algorithm.
///
/// Returns points in counter-clockwise order.
///
/// # Example
/// ```
/// use glam::Vec2;
/// use unshape_vector::convex_hull;
///
/// let points = vec![
///     Vec2::new(0.0, 0.0),
///     Vec2::new(1.0, 0.0),
///     Vec2::new(0.5, 0.5), // Interior point
///     Vec2::new(0.0, 1.0),
///     Vec2::new(1.0, 1.0),
/// ];
///
/// let hull = convex_hull(&points);
/// assert_eq!(hull.len(), 4); // Square corners only
/// ```
pub fn convex_hull(points: &[Vec2]) -> Vec<Vec2> {
    if points.len() < 3 {
        return points.to_vec();
    }

    // Sort points by x, then by y
    let mut sorted: Vec<Vec2> = points.to_vec();
    sorted.sort_by(|a, b| {
        a.x.partial_cmp(&b.x)
            .unwrap()
            .then_with(|| a.y.partial_cmp(&b.y).unwrap())
    });

    // Remove duplicates
    sorted.dedup_by(|a, b| (a.x - b.x).abs() < 1e-10 && (a.y - b.y).abs() < 1e-10);

    if sorted.len() < 3 {
        return sorted;
    }

    // Build lower hull
    let mut lower = Vec::new();
    for p in &sorted {
        while lower.len() >= 2
            && cross_product(lower[lower.len() - 2], lower[lower.len() - 1], *p) <= 0.0
        {
            lower.pop();
        }
        lower.push(*p);
    }

    // Build upper hull
    let mut upper = Vec::new();
    for p in sorted.iter().rev() {
        while upper.len() >= 2
            && cross_product(upper[upper.len() - 2], upper[upper.len() - 1], *p) <= 0.0
        {
            upper.pop();
        }
        upper.push(*p);
    }

    // Remove last point of each half because it's repeated
    lower.pop();
    upper.pop();

    // Concatenate
    lower.extend(upper);
    lower
}

/// Computes the convex hull and returns it as a closed Path.
pub fn convex_hull_path(points: &[Vec2]) -> Path {
    let hull = convex_hull(points);

    if hull.is_empty() {
        return Path::new();
    }

    let mut builder = PathBuilder::new().move_to(hull[0]);
    for &point in &hull[1..] {
        builder = builder.line_to(point);
    }
    builder.close().build()
}

/// Cross product of vectors OA and OB where O is the origin.
/// Returns positive if counter-clockwise, negative if clockwise, 0 if collinear.
fn cross_product(o: Vec2, a: Vec2, b: Vec2) -> f32 {
    (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x)
}

/// Computes the axis-aligned bounding box of a set of points.
///
/// Returns (min, max) corners, or None if empty.
pub fn bounding_box(points: &[Vec2]) -> Option<(Vec2, Vec2)> {
    if points.is_empty() {
        return None;
    }

    let mut min = points[0];
    let mut max = points[0];

    for &p in &points[1..] {
        min = min.min(p);
        max = max.max(p);
    }

    Some((min, max))
}

/// Computes the centroid (center of mass) of a set of points.
pub fn centroid(points: &[Vec2]) -> Option<Vec2> {
    if points.is_empty() {
        return None;
    }

    let sum: Vec2 = points.iter().copied().sum();
    Some(sum / points.len() as f32)
}

/// Tests if a point is inside a polygon (defined by vertices in order).
///
/// Uses the ray casting algorithm.
pub fn point_in_polygon(point: Vec2, polygon: &[Vec2]) -> bool {
    if polygon.len() < 3 {
        return false;
    }

    let mut inside = false;
    let n = polygon.len();

    let mut j = n - 1;
    for i in 0..n {
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

/// Tests if a point is on the convex hull boundary.
pub fn point_on_hull(point: Vec2, hull: &[Vec2], tolerance: f32) -> bool {
    if hull.len() < 2 {
        return false;
    }

    let n = hull.len();
    for i in 0..n {
        let a = hull[i];
        let b = hull[(i + 1) % n];

        let dist = point_to_segment_distance(point, a, b);
        if dist < tolerance {
            return true;
        }
    }

    false
}

/// Computes distance from a point to a line segment.
fn point_to_segment_distance(point: Vec2, a: Vec2, b: Vec2) -> f32 {
    let ab = b - a;
    let len_sq = ab.length_squared();

    if len_sq < 1e-10 {
        return (point - a).length();
    }

    let t = ((point - a).dot(ab) / len_sq).clamp(0.0, 1.0);
    let projection = a + ab * t;

    (point - projection).length()
}

/// Computes the area of a polygon (can be negative for clockwise winding).
pub fn polygon_area(vertices: &[Vec2]) -> f32 {
    if vertices.len() < 3 {
        return 0.0;
    }

    let mut area = 0.0;
    let n = vertices.len();

    for i in 0..n {
        let j = (i + 1) % n;
        area += vertices[i].x * vertices[j].y;
        area -= vertices[j].x * vertices[i].y;
    }

    area / 2.0
}

/// Computes the signed area (positive = counter-clockwise, negative = clockwise).
pub fn signed_area(vertices: &[Vec2]) -> f32 {
    polygon_area(vertices)
}

/// Returns true if vertices are in counter-clockwise order.
pub fn is_ccw(vertices: &[Vec2]) -> bool {
    signed_area(vertices) > 0.0
}

/// Computes the perimeter of a polygon.
pub fn polygon_perimeter(vertices: &[Vec2]) -> f32 {
    if vertices.len() < 2 {
        return 0.0;
    }

    let mut perimeter = 0.0;
    let n = vertices.len();

    for i in 0..n {
        let j = (i + 1) % n;
        perimeter += (vertices[j] - vertices[i]).length();
    }

    perimeter
}

/// Computes the minimum bounding circle (smallest enclosing circle).
///
/// Uses Welzl's algorithm.
pub fn minimum_bounding_circle(points: &[Vec2]) -> Option<(Vec2, f32)> {
    if points.is_empty() {
        return None;
    }

    if points.len() == 1 {
        return Some((points[0], 0.0));
    }

    // Shuffle points for expected linear time
    let mut shuffled: Vec<Vec2> = points.to_vec();
    shuffle(&mut shuffled, 12345);

    welzl(&shuffled, &mut Vec::new())
}

/// Welzl's algorithm for minimum enclosing circle.
fn welzl(points: &[Vec2], boundary: &mut Vec<Vec2>) -> Option<(Vec2, f32)> {
    if points.is_empty() || boundary.len() == 3 {
        return circle_from_boundary(boundary);
    }

    let p = points[0];
    let rest = &points[1..];

    // Try to create circle without this point
    let circle_opt = welzl(rest, boundary);

    // If we got a valid circle and this point is inside, we're done
    if let Some((center, radius)) = circle_opt {
        if point_in_circle(p, center, radius) {
            return Some((center, radius));
        }
    }

    // Otherwise, this point must be on the boundary
    boundary.push(p);
    let result = welzl(rest, boundary);
    boundary.pop();

    result
}

/// Creates a circle from up to 3 boundary points.
fn circle_from_boundary(boundary: &[Vec2]) -> Option<(Vec2, f32)> {
    match boundary.len() {
        0 => None,
        1 => Some((boundary[0], 0.0)),
        2 => {
            let center = (boundary[0] + boundary[1]) / 2.0;
            let radius = (boundary[0] - center).length();
            Some((center, radius))
        }
        3 => circle_from_three_points(boundary[0], boundary[1], boundary[2]),
        _ => None,
    }
}

/// Creates a circle through three points.
fn circle_from_three_points(a: Vec2, b: Vec2, c: Vec2) -> Option<(Vec2, f32)> {
    let d = 2.0 * (a.x * (b.y - c.y) + b.x * (c.y - a.y) + c.x * (a.y - b.y));

    if d.abs() < 1e-10 {
        // Points are collinear
        return None;
    }

    let ux = ((a.x * a.x + a.y * a.y) * (b.y - c.y)
        + (b.x * b.x + b.y * b.y) * (c.y - a.y)
        + (c.x * c.x + c.y * c.y) * (a.y - b.y))
        / d;

    let uy = ((a.x * a.x + a.y * a.y) * (c.x - b.x)
        + (b.x * b.x + b.y * b.y) * (a.x - c.x)
        + (c.x * c.x + c.y * c.y) * (b.x - a.x))
        / d;

    let center = Vec2::new(ux, uy);
    let radius = (a - center).length();

    Some((center, radius))
}

/// Tests if a point is inside or on a circle.
fn point_in_circle(point: Vec2, center: Vec2, radius: f32) -> bool {
    (point - center).length_squared() <= radius * radius + 1e-10
}

/// Simple shuffle using LCG.
fn shuffle(points: &mut [Vec2], seed: u64) {
    let mut rng = seed;
    for i in (1..points.len()).rev() {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
        let j = (rng as usize) % (i + 1);
        points.swap(i, j);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_convex_hull_square() {
        let points = vec![
            Vec2::new(0.0, 0.0),
            Vec2::new(1.0, 0.0),
            Vec2::new(0.5, 0.5), // Interior
            Vec2::new(1.0, 1.0),
            Vec2::new(0.0, 1.0),
        ];

        let hull = convex_hull(&points);
        assert_eq!(hull.len(), 4);
    }

    #[test]
    fn test_convex_hull_triangle() {
        let points = vec![
            Vec2::new(0.0, 0.0),
            Vec2::new(2.0, 0.0),
            Vec2::new(1.0, 2.0),
        ];

        let hull = convex_hull(&points);
        assert_eq!(hull.len(), 3);
    }

    #[test]
    fn test_convex_hull_collinear() {
        let points = vec![
            Vec2::new(0.0, 0.0),
            Vec2::new(1.0, 0.0),
            Vec2::new(2.0, 0.0),
        ];

        let hull = convex_hull(&points);
        assert_eq!(hull.len(), 2); // Only endpoints
    }

    #[test]
    fn test_convex_hull_path() {
        let points = vec![
            Vec2::new(0.0, 0.0),
            Vec2::new(1.0, 0.0),
            Vec2::new(1.0, 1.0),
            Vec2::new(0.0, 1.0),
        ];

        let path = convex_hull_path(&points);
        assert!(!path.is_empty());
    }

    #[test]
    fn test_bounding_box() {
        let points = vec![
            Vec2::new(-1.0, 2.0),
            Vec2::new(3.0, -1.0),
            Vec2::new(0.0, 0.0),
        ];

        let (min, max) = bounding_box(&points).unwrap();
        assert_eq!(min, Vec2::new(-1.0, -1.0));
        assert_eq!(max, Vec2::new(3.0, 2.0));
    }

    #[test]
    fn test_centroid() {
        let points = vec![
            Vec2::new(0.0, 0.0),
            Vec2::new(2.0, 0.0),
            Vec2::new(2.0, 2.0),
            Vec2::new(0.0, 2.0),
        ];

        let c = centroid(&points).unwrap();
        assert!((c.x - 1.0).abs() < 0.01);
        assert!((c.y - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_point_in_polygon() {
        let square = vec![
            Vec2::new(0.0, 0.0),
            Vec2::new(1.0, 0.0),
            Vec2::new(1.0, 1.0),
            Vec2::new(0.0, 1.0),
        ];

        assert!(point_in_polygon(Vec2::new(0.5, 0.5), &square));
        assert!(!point_in_polygon(Vec2::new(2.0, 0.5), &square));
    }

    #[test]
    fn test_polygon_area() {
        let square = vec![
            Vec2::new(0.0, 0.0),
            Vec2::new(1.0, 0.0),
            Vec2::new(1.0, 1.0),
            Vec2::new(0.0, 1.0),
        ];

        let area = polygon_area(&square).abs();
        assert!((area - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_polygon_perimeter() {
        let square = vec![
            Vec2::new(0.0, 0.0),
            Vec2::new(1.0, 0.0),
            Vec2::new(1.0, 1.0),
            Vec2::new(0.0, 1.0),
        ];

        let perimeter = polygon_perimeter(&square);
        assert!((perimeter - 4.0).abs() < 0.01);
    }

    #[test]
    fn test_minimum_bounding_circle() {
        let points = vec![
            Vec2::new(0.0, 0.0),
            Vec2::new(1.0, 0.0),
            Vec2::new(0.5, 0.5),
        ];

        let (center, radius) = minimum_bounding_circle(&points).unwrap();

        // All points should be inside or on the circle
        for p in &points {
            assert!((*p - center).length() <= radius + 0.01);
        }
    }

    #[test]
    fn test_is_ccw() {
        let ccw = vec![
            Vec2::new(0.0, 0.0),
            Vec2::new(1.0, 0.0),
            Vec2::new(1.0, 1.0),
            Vec2::new(0.0, 1.0),
        ];

        let cw = vec![
            Vec2::new(0.0, 0.0),
            Vec2::new(0.0, 1.0),
            Vec2::new(1.0, 1.0),
            Vec2::new(1.0, 0.0),
        ];

        assert!(is_ccw(&ccw));
        assert!(!is_ccw(&cw));
    }
}
