//! Bezier curve utilities.
//!
//! Provides evaluation and manipulation functions for quadratic and cubic Bezier curves.

use glam::Vec2;

/// Evaluates a quadratic Bezier curve at parameter `t`.
///
/// # Arguments
///
/// * `p0` - Start point
/// * `p1` - Control point
/// * `p2` - End point
/// * `t` - Parameter in [0, 1]
///
/// # Example
///
/// ```
/// use rhizome_resin_vector::bezier::quadratic_point;
/// use glam::Vec2;
///
/// let start = Vec2::ZERO;
/// let control = Vec2::new(0.5, 1.0);
/// let end = Vec2::X;
///
/// let mid = quadratic_point(start, control, end, 0.5);
/// ```
#[inline]
pub fn quadratic_point(p0: Vec2, p1: Vec2, p2: Vec2, t: f32) -> Vec2 {
    let mt = 1.0 - t;
    let mt2 = mt * mt;
    let t2 = t * t;
    p0 * mt2 + p1 * (2.0 * mt * t) + p2 * t2
}

/// Evaluates the tangent (derivative) of a quadratic Bezier curve at parameter `t`.
///
/// Returns the unnormalized tangent vector.
#[inline]
pub fn quadratic_tangent(p0: Vec2, p1: Vec2, p2: Vec2, t: f32) -> Vec2 {
    let mt = 1.0 - t;
    2.0 * mt * (p1 - p0) + 2.0 * t * (p2 - p1)
}

/// Evaluates a cubic Bezier curve at parameter `t`.
///
/// # Arguments
///
/// * `p0` - Start point
/// * `p1` - First control point
/// * `p2` - Second control point
/// * `p3` - End point
/// * `t` - Parameter in [0, 1]
///
/// # Example
///
/// ```
/// use rhizome_resin_vector::bezier::cubic_point;
/// use glam::Vec2;
///
/// let p0 = Vec2::ZERO;
/// let p1 = Vec2::new(0.25, 1.0);
/// let p2 = Vec2::new(0.75, 1.0);
/// let p3 = Vec2::X;
///
/// let mid = cubic_point(p0, p1, p2, p3, 0.5);
/// ```
#[inline]
pub fn cubic_point(p0: Vec2, p1: Vec2, p2: Vec2, p3: Vec2, t: f32) -> Vec2 {
    let mt = 1.0 - t;
    let mt2 = mt * mt;
    let mt3 = mt2 * mt;
    let t2 = t * t;
    let t3 = t2 * t;
    p0 * mt3 + p1 * (3.0 * mt2 * t) + p2 * (3.0 * mt * t2) + p3 * t3
}

/// Evaluates the tangent (derivative) of a cubic Bezier curve at parameter `t`.
///
/// Returns the unnormalized tangent vector.
#[inline]
pub fn cubic_tangent(p0: Vec2, p1: Vec2, p2: Vec2, p3: Vec2, t: f32) -> Vec2 {
    let mt = 1.0 - t;
    let mt2 = mt * mt;
    let t2 = t * t;
    3.0 * mt2 * (p1 - p0) + 6.0 * mt * t * (p2 - p1) + 3.0 * t2 * (p3 - p2)
}

/// Splits a cubic Bezier curve at parameter `t` using de Casteljau's algorithm.
///
/// Returns two sets of control points: (left curve, right curve).
#[inline]
pub fn cubic_split(p0: Vec2, p1: Vec2, p2: Vec2, p3: Vec2, t: f32) -> ([Vec2; 4], [Vec2; 4]) {
    let p01 = p0.lerp(p1, t);
    let p12 = p1.lerp(p2, t);
    let p23 = p2.lerp(p3, t);
    let p012 = p01.lerp(p12, t);
    let p123 = p12.lerp(p23, t);
    let p0123 = p012.lerp(p123, t);

    ([p0, p01, p012, p0123], [p0123, p123, p23, p3])
}

/// Splits a quadratic Bezier curve at parameter `t`.
///
/// Returns two sets of control points: (left curve, right curve).
#[inline]
pub fn quadratic_split(p0: Vec2, p1: Vec2, p2: Vec2, t: f32) -> ([Vec2; 3], [Vec2; 3]) {
    let p01 = p0.lerp(p1, t);
    let p12 = p1.lerp(p2, t);
    let p012 = p01.lerp(p12, t);

    ([p0, p01, p012], [p012, p12, p2])
}

/// Computes the bounding box of a cubic Bezier curve.
///
/// Returns (min, max) corners of the axis-aligned bounding box.
pub fn cubic_bounds(p0: Vec2, p1: Vec2, p2: Vec2, p3: Vec2) -> (Vec2, Vec2) {
    // Start with endpoints
    let mut min = p0.min(p3);
    let mut max = p0.max(p3);

    // Find extrema by solving derivative = 0
    // Derivative: 3(1-t)²(p1-p0) + 6(1-t)t(p2-p1) + 3t²(p3-p2) = 0
    // This is a quadratic in t for each axis

    for axis in 0..2 {
        let a = -p0[axis] + 3.0 * p1[axis] - 3.0 * p2[axis] + p3[axis];
        let b = 2.0 * p0[axis] - 4.0 * p1[axis] + 2.0 * p2[axis];
        let c = -p0[axis] + p1[axis];

        if a.abs() < 1e-10 {
            // Linear case
            if b.abs() > 1e-10 {
                let t = -c / b;
                if t > 0.0 && t < 1.0 {
                    let pt = cubic_point(p0, p1, p2, p3, t);
                    min[axis] = min[axis].min(pt[axis]);
                    max[axis] = max[axis].max(pt[axis]);
                }
            }
        } else {
            // Quadratic case
            let discriminant = b * b - 4.0 * a * c;
            if discriminant >= 0.0 {
                let sqrt_d = discriminant.sqrt();
                for t in [(-b + sqrt_d) / (2.0 * a), (-b - sqrt_d) / (2.0 * a)] {
                    if t > 0.0 && t < 1.0 {
                        let pt = cubic_point(p0, p1, p2, p3, t);
                        min[axis] = min[axis].min(pt[axis]);
                        max[axis] = max[axis].max(pt[axis]);
                    }
                }
            }
        }
    }

    (min, max)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quadratic_endpoints() {
        let p0 = Vec2::ZERO;
        let p1 = Vec2::new(0.5, 1.0);
        let p2 = Vec2::X;

        let start = quadratic_point(p0, p1, p2, 0.0);
        let end = quadratic_point(p0, p1, p2, 1.0);

        assert!((start - p0).length() < 0.001);
        assert!((end - p2).length() < 0.001);
    }

    #[test]
    fn test_cubic_endpoints() {
        let p0 = Vec2::ZERO;
        let p1 = Vec2::new(0.25, 1.0);
        let p2 = Vec2::new(0.75, 1.0);
        let p3 = Vec2::X;

        let start = cubic_point(p0, p1, p2, p3, 0.0);
        let end = cubic_point(p0, p1, p2, p3, 1.0);

        assert!((start - p0).length() < 0.001);
        assert!((end - p3).length() < 0.001);
    }

    #[test]
    fn test_cubic_split_continuity() {
        let p0 = Vec2::ZERO;
        let p1 = Vec2::new(0.25, 1.0);
        let p2 = Vec2::new(0.75, 1.0);
        let p3 = Vec2::X;

        let (left, right) = cubic_split(p0, p1, p2, p3, 0.5);

        // Left curve should end where right curve starts
        assert!((left[3] - right[0]).length() < 0.001);

        // Point at split should match original curve
        let original_mid = cubic_point(p0, p1, p2, p3, 0.5);
        assert!((left[3] - original_mid).length() < 0.001);
    }

    #[test]
    fn test_cubic_bounds() {
        // A curve that bulges upward
        let p0 = Vec2::ZERO;
        let p1 = Vec2::new(0.0, 2.0);
        let p2 = Vec2::new(1.0, 2.0);
        let p3 = Vec2::X;

        let (min, max) = cubic_bounds(p0, p1, p2, p3);

        assert!(min.x <= 0.0);
        assert!(min.y <= 0.0);
        assert!(max.x >= 1.0);
        assert!(max.y >= 1.0); // Should catch the bulge
    }
}
