//! Spatial data structures for efficient queries and collision detection.
//!
//! This crate provides data structures for spatial partitioning and queries:
//!
//! - [`Quadtree`] - 2D spatial partitioning with point/region queries
//! - [`Octree`] - 3D spatial partitioning with point/region queries
//! - [`KdTree2D`] / [`KdTree3D`] - KD-trees for efficient nearest neighbor queries
//! - [`BallTree2D`] / [`BallTree3D`] - Ball trees using bounding spheres
//! - [`Bvh`] - Bounding volume hierarchy for ray/intersection queries
//! - [`SpatialHash`] - Grid-based broad phase collision detection
//! - [`Rtree`] - R-tree for rectangle/AABB queries
//!
//! # Example
//!
//! ```
//! use unshape_spatial::{Quadtree, Aabb2};
//! use glam::Vec2;
//!
//! let bounds = Aabb2::new(Vec2::ZERO, Vec2::splat(100.0));
//! let mut tree = Quadtree::new(bounds, 8, 4);
//!
//! // Insert points with associated data
//! tree.insert(Vec2::new(10.0, 20.0), "point A");
//! tree.insert(Vec2::new(50.0, 50.0), "point B");
//!
//! // Query points in a region
//! let query = Aabb2::new(Vec2::ZERO, Vec2::splat(30.0));
//! let results: Vec<_> = tree.query_region(&query).collect();
//! assert_eq!(results.len(), 1);
//! ```

use glam::{Vec2, Vec3};
use std::cmp::Ordering;

mod ball_tree;
mod bvh;
mod kdtree;
mod octree;
mod quadtree;
mod rtree;
mod spatial_hash;

pub use ball_tree::*;
pub use bvh::*;
pub use kdtree::*;
pub use octree::*;
pub use quadtree::*;
pub use rtree::*;
pub use spatial_hash::*;

// ============================================================================
// AABB Types
// ============================================================================

/// 2D axis-aligned bounding box.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Aabb2 {
    /// Minimum corner (lower-left).
    pub min: Vec2,
    /// Maximum corner (upper-right).
    pub max: Vec2,
}

impl Aabb2 {
    /// Creates a new AABB from min and max corners.
    pub fn new(min: Vec2, max: Vec2) -> Self {
        Self { min, max }
    }

    /// Creates an AABB from center and half-extents.
    pub fn from_center_half_extents(center: Vec2, half_extents: Vec2) -> Self {
        Self {
            min: center - half_extents,
            max: center + half_extents,
        }
    }

    /// Returns the center of the AABB.
    pub fn center(&self) -> Vec2 {
        (self.min + self.max) * 0.5
    }

    /// Returns the half-extents (half-size) of the AABB.
    pub fn half_extents(&self) -> Vec2 {
        (self.max - self.min) * 0.5
    }

    /// Returns the size of the AABB.
    pub fn size(&self) -> Vec2 {
        self.max - self.min
    }

    /// Checks if this AABB contains a point.
    pub fn contains_point(&self, point: Vec2) -> bool {
        point.x >= self.min.x
            && point.x <= self.max.x
            && point.y >= self.min.y
            && point.y <= self.max.y
    }

    /// Checks if this AABB intersects another AABB.
    pub fn intersects(&self, other: &Aabb2) -> bool {
        self.min.x <= other.max.x
            && self.max.x >= other.min.x
            && self.min.y <= other.max.y
            && self.max.y >= other.min.y
    }

    /// Returns the four quadrant AABBs when subdividing this AABB.
    pub fn quadrants(&self) -> [Aabb2; 4] {
        let center = self.center();
        [
            // Bottom-left (SW)
            Aabb2::new(self.min, center),
            // Bottom-right (SE)
            Aabb2::new(
                Vec2::new(center.x, self.min.y),
                Vec2::new(self.max.x, center.y),
            ),
            // Top-left (NW)
            Aabb2::new(
                Vec2::new(self.min.x, center.y),
                Vec2::new(center.x, self.max.y),
            ),
            // Top-right (NE)
            Aabb2::new(center, self.max),
        ]
    }
}

/// 3D axis-aligned bounding box.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Aabb3 {
    /// Minimum corner.
    pub min: Vec3,
    /// Maximum corner.
    pub max: Vec3,
}

impl Aabb3 {
    /// Creates a new AABB from min and max corners.
    pub fn new(min: Vec3, max: Vec3) -> Self {
        Self { min, max }
    }

    /// Creates an AABB from center and half-extents.
    pub fn from_center_half_extents(center: Vec3, half_extents: Vec3) -> Self {
        Self {
            min: center - half_extents,
            max: center + half_extents,
        }
    }

    /// Returns the center of the AABB.
    pub fn center(&self) -> Vec3 {
        (self.min + self.max) * 0.5
    }

    /// Returns the half-extents (half-size) of the AABB.
    pub fn half_extents(&self) -> Vec3 {
        (self.max - self.min) * 0.5
    }

    /// Returns the size of the AABB.
    pub fn size(&self) -> Vec3 {
        self.max - self.min
    }

    /// Checks if this AABB contains a point.
    pub fn contains_point(&self, point: Vec3) -> bool {
        point.x >= self.min.x
            && point.x <= self.max.x
            && point.y >= self.min.y
            && point.y <= self.max.y
            && point.z >= self.min.z
            && point.z <= self.max.z
    }

    /// Checks if this AABB intersects another AABB.
    pub fn intersects(&self, other: &Aabb3) -> bool {
        self.min.x <= other.max.x
            && self.max.x >= other.min.x
            && self.min.y <= other.max.y
            && self.max.y >= other.min.y
            && self.min.z <= other.max.z
            && self.max.z >= other.min.z
    }

    /// Returns the eight octant AABBs when subdividing this AABB.
    pub fn octants(&self) -> [Aabb3; 8] {
        let center = self.center();
        [
            // Bottom layer (z < center)
            Aabb3::new(self.min, center),
            Aabb3::new(
                Vec3::new(center.x, self.min.y, self.min.z),
                Vec3::new(self.max.x, center.y, center.z),
            ),
            Aabb3::new(
                Vec3::new(self.min.x, center.y, self.min.z),
                Vec3::new(center.x, self.max.y, center.z),
            ),
            Aabb3::new(
                Vec3::new(center.x, center.y, self.min.z),
                Vec3::new(self.max.x, self.max.y, center.z),
            ),
            // Top layer (z >= center)
            Aabb3::new(
                Vec3::new(self.min.x, self.min.y, center.z),
                Vec3::new(center.x, center.y, self.max.z),
            ),
            Aabb3::new(
                Vec3::new(center.x, self.min.y, center.z),
                Vec3::new(self.max.x, center.y, self.max.z),
            ),
            Aabb3::new(
                Vec3::new(self.min.x, center.y, center.z),
                Vec3::new(center.x, self.max.y, self.max.z),
            ),
            Aabb3::new(center, self.max),
        ]
    }

    /// Computes the surface area of the AABB.
    pub fn surface_area(&self) -> f32 {
        let size = self.size();
        2.0 * (size.x * size.y + size.y * size.z + size.z * size.x)
    }

    /// Returns the union of two AABBs.
    pub fn union(&self, other: &Aabb3) -> Aabb3 {
        Aabb3::new(self.min.min(other.min), self.max.max(other.max))
    }
}

// ============================================================================
// K-Nearest Helper Types
// ============================================================================

/// Candidate for k-nearest neighbor search in 2D.
/// Uses max-heap ordering (largest distance at top) for efficient pruning.
pub(crate) struct KNearestCandidate2D<'a, T> {
    pub position: Vec2,
    pub data: &'a T,
    pub distance: f32,
}

impl<T> PartialEq for KNearestCandidate2D<'_, T> {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}

impl<T> Eq for KNearestCandidate2D<'_, T> {}

impl<T> PartialOrd for KNearestCandidate2D<'_, T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<T> Ord for KNearestCandidate2D<'_, T> {
    fn cmp(&self, other: &Self) -> Ordering {
        // Max-heap: larger distance = higher priority
        self.distance
            .partial_cmp(&other.distance)
            .unwrap_or(Ordering::Equal)
    }
}

/// Candidate for k-nearest neighbor search in 3D.
/// Uses max-heap ordering (largest distance at top) for efficient pruning.
pub(crate) struct KNearestCandidate3D<'a, T> {
    pub position: Vec3,
    pub data: &'a T,
    pub distance: f32,
}

impl<T> PartialEq for KNearestCandidate3D<'_, T> {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}

impl<T> Eq for KNearestCandidate3D<'_, T> {}

impl<T> PartialOrd for KNearestCandidate3D<'_, T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<T> Ord for KNearestCandidate3D<'_, T> {
    fn cmp(&self, other: &Self) -> Ordering {
        // Max-heap: larger distance = higher priority
        self.distance
            .partial_cmp(&other.distance)
            .unwrap_or(Ordering::Equal)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // AABB2 tests

    #[test]
    fn test_aabb2_contains_point() {
        let aabb = Aabb2::new(Vec2::ZERO, Vec2::splat(10.0));
        assert!(aabb.contains_point(Vec2::new(5.0, 5.0)));
        assert!(aabb.contains_point(Vec2::ZERO));
        assert!(aabb.contains_point(Vec2::splat(10.0)));
        assert!(!aabb.contains_point(Vec2::new(-1.0, 5.0)));
        assert!(!aabb.contains_point(Vec2::new(11.0, 5.0)));
    }

    #[test]
    fn test_aabb2_intersects() {
        let a = Aabb2::new(Vec2::ZERO, Vec2::splat(10.0));
        let b = Aabb2::new(Vec2::splat(5.0), Vec2::splat(15.0));
        let c = Aabb2::new(Vec2::splat(20.0), Vec2::splat(30.0));

        assert!(a.intersects(&b));
        assert!(b.intersects(&a));
        assert!(!a.intersects(&c));
        assert!(!c.intersects(&a));
    }

    #[test]
    fn test_aabb2_quadrants() {
        let aabb = Aabb2::new(Vec2::ZERO, Vec2::splat(10.0));
        let quads = aabb.quadrants();

        // Check that quadrants cover the original AABB
        assert_eq!(quads[0].min, Vec2::ZERO);
        assert_eq!(quads[0].max, Vec2::splat(5.0));
        assert_eq!(quads[3].min, Vec2::splat(5.0));
        assert_eq!(quads[3].max, Vec2::splat(10.0));
    }

    // AABB3 tests

    #[test]
    fn test_aabb3_contains_point() {
        let aabb = Aabb3::new(Vec3::ZERO, Vec3::splat(10.0));
        assert!(aabb.contains_point(Vec3::new(5.0, 5.0, 5.0)));
        assert!(aabb.contains_point(Vec3::ZERO));
        assert!(aabb.contains_point(Vec3::splat(10.0)));
        assert!(!aabb.contains_point(Vec3::new(-1.0, 5.0, 5.0)));
        assert!(!aabb.contains_point(Vec3::new(11.0, 5.0, 5.0)));
    }

    #[test]
    fn test_aabb3_intersects() {
        let a = Aabb3::new(Vec3::ZERO, Vec3::splat(10.0));
        let b = Aabb3::new(Vec3::splat(5.0), Vec3::splat(15.0));
        let c = Aabb3::new(Vec3::splat(20.0), Vec3::splat(30.0));

        assert!(a.intersects(&b));
        assert!(b.intersects(&a));
        assert!(!a.intersects(&c));
        assert!(!c.intersects(&a));
    }

    #[test]
    fn test_aabb3_surface_area() {
        let aabb = Aabb3::new(Vec3::ZERO, Vec3::new(2.0, 3.0, 4.0));
        // 2 * (2*3 + 3*4 + 4*2) = 2 * (6 + 12 + 8) = 52
        assert!((aabb.surface_area() - 52.0).abs() < 1e-6);
    }

    // Quadtree tests

    #[test]
    fn test_quadtree_insert_and_query() {
        let bounds = Aabb2::new(Vec2::ZERO, Vec2::splat(100.0));
        let mut tree = Quadtree::new(bounds, 8, 4);

        tree.insert(Vec2::new(10.0, 10.0), 1);
        tree.insert(Vec2::new(20.0, 20.0), 2);
        tree.insert(Vec2::new(80.0, 80.0), 3);

        assert_eq!(tree.len(), 3);

        // Query region containing first two points
        let query = Aabb2::new(Vec2::ZERO, Vec2::splat(50.0));
        let results: Vec<_> = tree.query_region(&query).collect();
        assert_eq!(results.len(), 2);

        // Query region containing last point
        let query = Aabb2::new(Vec2::splat(60.0), Vec2::splat(100.0));
        let results: Vec<_> = tree.query_region(&query).collect();
        assert_eq!(results.len(), 1);
        assert_eq!(*results[0].1, 3);
    }

    #[test]
    fn test_quadtree_nearest() {
        let bounds = Aabb2::new(Vec2::ZERO, Vec2::splat(100.0));
        let mut tree = Quadtree::new(bounds, 8, 4);

        tree.insert(Vec2::new(10.0, 10.0), "A");
        tree.insert(Vec2::new(50.0, 50.0), "B");
        tree.insert(Vec2::new(90.0, 90.0), "C");

        let result = tree.nearest(Vec2::new(12.0, 12.0)).unwrap();
        assert_eq!(*result.1, "A");

        let result = tree.nearest(Vec2::new(48.0, 52.0)).unwrap();
        assert_eq!(*result.1, "B");
    }

    #[test]
    fn test_quadtree_k_nearest() {
        let bounds = Aabb2::new(Vec2::ZERO, Vec2::splat(100.0));
        let mut tree = Quadtree::new(bounds, 8, 4);

        tree.insert(Vec2::new(10.0, 10.0), "A");
        tree.insert(Vec2::new(20.0, 20.0), "B");
        tree.insert(Vec2::new(30.0, 30.0), "C");
        tree.insert(Vec2::new(90.0, 90.0), "D");

        // Query from origin - should get A, B, C in order
        let results = tree.k_nearest(Vec2::ZERO, 3);
        assert_eq!(results.len(), 3);
        assert_eq!(*results[0].1, "A"); // Closest
        assert_eq!(*results[1].1, "B");
        assert_eq!(*results[2].1, "C");

        // k=0 returns empty
        let results = tree.k_nearest(Vec2::ZERO, 0);
        assert!(results.is_empty());

        // k > len returns all
        let results = tree.k_nearest(Vec2::ZERO, 10);
        assert_eq!(results.len(), 4);

        // k=1 should match nearest()
        let k1 = tree.k_nearest(Vec2::new(12.0, 12.0), 1);
        let nearest = tree.nearest(Vec2::new(12.0, 12.0)).unwrap();
        assert_eq!(*k1[0].1, *nearest.1);
    }

    #[test]
    fn test_quadtree_out_of_bounds() {
        let bounds = Aabb2::new(Vec2::ZERO, Vec2::splat(100.0));
        let mut tree = Quadtree::new(bounds, 8, 4);

        assert!(!tree.insert(Vec2::new(-10.0, 50.0), 1));
        assert!(!tree.insert(Vec2::new(150.0, 50.0), 2));
        assert_eq!(tree.len(), 0);
    }

    #[test]
    fn test_quadtree_subdivision() {
        let bounds = Aabb2::new(Vec2::ZERO, Vec2::splat(100.0));
        let mut tree = Quadtree::new(bounds, 8, 2);

        // Insert enough points to trigger subdivision
        for i in 0..10 {
            tree.insert(Vec2::new(i as f32 * 5.0, i as f32 * 5.0), i);
        }

        assert_eq!(tree.len(), 10);

        // All points should still be queryable
        let query = Aabb2::new(Vec2::ZERO, Vec2::splat(100.0));
        let results: Vec<_> = tree.query_region(&query).collect();
        assert_eq!(results.len(), 10);
    }

    // Octree tests

    #[test]
    fn test_octree_insert_and_query() {
        let bounds = Aabb3::new(Vec3::ZERO, Vec3::splat(100.0));
        let mut tree = Octree::new(bounds, 8, 4);

        tree.insert(Vec3::new(10.0, 10.0, 10.0), 1);
        tree.insert(Vec3::new(20.0, 20.0, 20.0), 2);
        tree.insert(Vec3::new(80.0, 80.0, 80.0), 3);

        assert_eq!(tree.len(), 3);

        // Query region containing first two points
        let query = Aabb3::new(Vec3::ZERO, Vec3::splat(50.0));
        let results: Vec<_> = tree.query_region(&query).collect();
        assert_eq!(results.len(), 2);

        // Query region containing last point
        let query = Aabb3::new(Vec3::splat(60.0), Vec3::splat(100.0));
        let results: Vec<_> = tree.query_region(&query).collect();
        assert_eq!(results.len(), 1);
        assert_eq!(*results[0].1, 3);
    }

    #[test]
    fn test_octree_nearest() {
        let bounds = Aabb3::new(Vec3::ZERO, Vec3::splat(100.0));
        let mut tree = Octree::new(bounds, 8, 4);

        tree.insert(Vec3::new(10.0, 10.0, 10.0), "A");
        tree.insert(Vec3::new(50.0, 50.0, 50.0), "B");
        tree.insert(Vec3::new(90.0, 90.0, 90.0), "C");

        let result = tree.nearest(Vec3::new(12.0, 12.0, 12.0)).unwrap();
        assert_eq!(*result.1, "A");

        let result = tree.nearest(Vec3::new(48.0, 52.0, 50.0)).unwrap();
        assert_eq!(*result.1, "B");
    }

    #[test]
    fn test_octree_k_nearest() {
        let bounds = Aabb3::new(Vec3::ZERO, Vec3::splat(100.0));
        let mut tree = Octree::new(bounds, 8, 4);

        tree.insert(Vec3::new(10.0, 10.0, 10.0), "A");
        tree.insert(Vec3::new(20.0, 20.0, 20.0), "B");
        tree.insert(Vec3::new(30.0, 30.0, 30.0), "C");
        tree.insert(Vec3::new(90.0, 90.0, 90.0), "D");

        // Query from origin - should get A, B, C in order
        let results = tree.k_nearest(Vec3::ZERO, 3);
        assert_eq!(results.len(), 3);
        assert_eq!(*results[0].1, "A"); // Closest
        assert_eq!(*results[1].1, "B");
        assert_eq!(*results[2].1, "C");

        // k=0 returns empty
        let results = tree.k_nearest(Vec3::ZERO, 0);
        assert!(results.is_empty());

        // k > len returns all
        let results = tree.k_nearest(Vec3::ZERO, 10);
        assert_eq!(results.len(), 4);

        // k=1 should match nearest()
        let k1 = tree.k_nearest(Vec3::new(12.0, 12.0, 12.0), 1);
        let nearest = tree.nearest(Vec3::new(12.0, 12.0, 12.0)).unwrap();
        assert_eq!(*k1[0].1, *nearest.1);
    }

    #[test]
    fn test_octree_subdivision() {
        let bounds = Aabb3::new(Vec3::ZERO, Vec3::splat(100.0));
        let mut tree = Octree::new(bounds, 8, 2);

        // Insert enough points to trigger subdivision
        for i in 0..10 {
            let v = i as f32 * 5.0;
            tree.insert(Vec3::new(v, v, v), i);
        }

        assert_eq!(tree.len(), 10);

        // All points should still be queryable
        let query = Aabb3::new(Vec3::ZERO, Vec3::splat(100.0));
        let results: Vec<_> = tree.query_region(&query).collect();
        assert_eq!(results.len(), 10);
    }

    #[test]
    fn test_quadtree_clear() {
        let bounds = Aabb2::new(Vec2::ZERO, Vec2::splat(100.0));
        let mut tree = Quadtree::new(bounds, 8, 4);

        tree.insert(Vec2::new(10.0, 10.0), 1);
        tree.insert(Vec2::new(20.0, 20.0), 2);
        assert_eq!(tree.len(), 2);

        tree.clear();
        assert_eq!(tree.len(), 0);
        assert!(tree.is_empty());
    }

    #[test]
    fn test_octree_clear() {
        let bounds = Aabb3::new(Vec3::ZERO, Vec3::splat(100.0));
        let mut tree = Octree::new(bounds, 8, 4);

        tree.insert(Vec3::new(10.0, 10.0, 10.0), 1);
        tree.insert(Vec3::new(20.0, 20.0, 20.0), 2);
        assert_eq!(tree.len(), 2);

        tree.clear();
        assert_eq!(tree.len(), 0);
        assert!(tree.is_empty());
    }

    // Ray tests

    #[test]
    fn test_ray_at() {
        let ray = Ray::new(Vec3::ZERO, Vec3::Z);
        assert!((ray.at(5.0) - Vec3::new(0.0, 0.0, 5.0)).length() < 1e-6);
    }

    #[test]
    fn test_ray_intersect_aabb() {
        let aabb = Aabb3::new(Vec3::splat(-1.0), Vec3::splat(1.0));

        // Ray pointing at center
        let ray = Ray::new(Vec3::new(0.0, 0.0, -5.0), Vec3::Z);
        let hit = ray.intersect_aabb(&aabb);
        assert!(hit.is_some());
        let (t_min, t_max) = hit.unwrap();
        assert!((t_min - 4.0).abs() < 1e-6);
        assert!((t_max - 6.0).abs() < 1e-6);

        // Ray missing
        let ray = Ray::new(Vec3::new(10.0, 0.0, -5.0), Vec3::Z);
        assert!(ray.intersect_aabb(&aabb).is_none());
    }

    // BVH tests

    #[test]
    fn test_bvh_build_empty() {
        let bvh: Bvh<i32> = Bvh::build(vec![]);
        assert!(bvh.is_empty());
        assert_eq!(bvh.len(), 0);
    }

    #[test]
    fn test_bvh_build_single() {
        let primitives = vec![(Aabb3::new(Vec3::ZERO, Vec3::ONE), "box")];
        let bvh = Bvh::build(primitives);
        assert_eq!(bvh.len(), 1);
    }

    #[test]
    fn test_bvh_intersect_ray() {
        let primitives = vec![
            (Aabb3::new(Vec3::ZERO, Vec3::ONE), "box1"),
            (Aabb3::new(Vec3::splat(5.0), Vec3::splat(6.0)), "box2"),
            (Aabb3::new(Vec3::splat(10.0), Vec3::splat(11.0)), "box3"),
        ];
        let bvh = Bvh::build(primitives);
        assert_eq!(bvh.len(), 3);

        // Ray hitting box1
        let ray = Ray::new(Vec3::new(0.5, 0.5, -5.0), Vec3::Z);
        let hits = bvh.intersect_ray(&ray);
        assert_eq!(hits.len(), 1);
        assert_eq!(*hits[0].1, "box1");

        // Ray hitting box2
        let ray = Ray::new(Vec3::new(5.5, 5.5, -5.0), Vec3::Z);
        let hits = bvh.intersect_ray(&ray);
        assert_eq!(hits.len(), 1);
        assert_eq!(*hits[0].1, "box2");

        // Ray missing all
        let ray = Ray::new(Vec3::new(100.0, 100.0, -5.0), Vec3::Z);
        let hits = bvh.intersect_ray(&ray);
        assert!(hits.is_empty());
    }

    #[test]
    fn test_bvh_intersect_ray_closest() {
        let primitives = vec![
            (
                Aabb3::new(Vec3::new(0.0, 0.0, 2.0), Vec3::new(1.0, 1.0, 3.0)),
                "far",
            ),
            (
                Aabb3::new(Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 1.0, 1.0)),
                "near",
            ),
        ];
        let bvh = Bvh::build(primitives);

        let ray = Ray::new(Vec3::new(0.5, 0.5, -5.0), Vec3::Z);
        let closest = bvh.intersect_ray_closest(&ray);
        assert!(closest.is_some());
        let (_, data, _) = closest.unwrap();
        assert_eq!(*data, "near");
    }

    #[test]
    fn test_bvh_query_aabb() {
        let primitives = vec![
            (Aabb3::new(Vec3::ZERO, Vec3::ONE), "box1"),
            (Aabb3::new(Vec3::splat(5.0), Vec3::splat(6.0)), "box2"),
        ];
        let bvh = Bvh::build(primitives);

        // Query overlapping box1
        let query = Aabb3::new(Vec3::splat(-1.0), Vec3::splat(0.5));
        let results = bvh.query_aabb(&query);
        assert_eq!(results.len(), 1);
        assert_eq!(*results[0].1, "box1");

        // Query not overlapping anything
        let query = Aabb3::new(Vec3::splat(2.0), Vec3::splat(3.0));
        let results = bvh.query_aabb(&query);
        assert!(results.is_empty());
    }

    // SpatialHash tests

    #[test]
    fn test_spatial_hash_insert_query() {
        let mut hash = SpatialHash::new(10.0);

        hash.insert(Vec3::new(5.0, 5.0, 5.0), "A");
        hash.insert(Vec3::new(15.0, 5.0, 5.0), "B");
        hash.insert(Vec3::new(5.5, 5.5, 5.5), "C");

        assert_eq!(hash.len(), 3);

        // A and C in same cell
        let results: Vec<_> = hash.query_cell(Vec3::new(5.0, 5.0, 5.0)).collect();
        assert_eq!(results.len(), 2);

        // B in different cell
        let results: Vec<_> = hash.query_cell(Vec3::new(15.0, 5.0, 5.0)).collect();
        assert_eq!(results.len(), 1);
        assert_eq!(*results[0].1, "B");
    }

    #[test]
    fn test_spatial_hash_query_neighbors() {
        let mut hash = SpatialHash::new(10.0);

        hash.insert(Vec3::new(5.0, 5.0, 5.0), "A");
        hash.insert(Vec3::new(15.0, 5.0, 5.0), "B");
        hash.insert(Vec3::new(50.0, 50.0, 50.0), "C");

        // A and B are neighbors
        let results: Vec<_> = hash.query_neighbors(Vec3::new(5.0, 5.0, 5.0)).collect();
        assert_eq!(results.len(), 2);

        // C is isolated
        let results: Vec<_> = hash.query_neighbors(Vec3::new(50.0, 50.0, 50.0)).collect();
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_spatial_hash_query_radius() {
        let mut hash = SpatialHash::new(10.0);

        hash.insert(Vec3::new(0.0, 0.0, 0.0), "A");
        hash.insert(Vec3::new(5.0, 0.0, 0.0), "B");
        hash.insert(Vec3::new(20.0, 0.0, 0.0), "C");

        // Radius 6 should include A and B
        let results: Vec<_> = hash.query_radius(Vec3::ZERO, 6.0).collect();
        assert_eq!(results.len(), 2);

        // Radius 3 should only include A
        let results: Vec<_> = hash.query_radius(Vec3::ZERO, 3.0).collect();
        assert_eq!(results.len(), 1);
        assert_eq!(*results[0].1, "A");
    }

    #[test]
    fn test_spatial_hash_clear() {
        let mut hash = SpatialHash::new(10.0);
        hash.insert(Vec3::ZERO, 1);
        hash.insert(Vec3::ONE, 2);
        assert_eq!(hash.len(), 2);

        hash.clear();
        assert!(hash.is_empty());
    }

    // R-tree tests

    #[test]
    fn test_rtree_empty() {
        let tree: Rtree<i32> = Rtree::new(4);
        assert!(tree.is_empty());
        assert_eq!(tree.len(), 0);
    }

    #[test]
    fn test_rtree_insert_single() {
        let mut tree = Rtree::new(4);
        tree.insert(Aabb2::new(Vec2::ZERO, Vec2::splat(10.0)), "rect");
        assert_eq!(tree.len(), 1);
    }

    #[test]
    fn test_rtree_query() {
        let mut tree = Rtree::new(4);
        tree.insert(Aabb2::new(Vec2::ZERO, Vec2::splat(10.0)), "rect1");
        tree.insert(Aabb2::new(Vec2::splat(20.0), Vec2::splat(30.0)), "rect2");
        tree.insert(Aabb2::new(Vec2::splat(50.0), Vec2::splat(60.0)), "rect3");

        assert_eq!(tree.len(), 3);

        // Query overlapping rect1
        let query = Aabb2::new(Vec2::splat(5.0), Vec2::splat(15.0));
        let results = tree.query(&query);
        assert_eq!(results.len(), 1);
        assert_eq!(*results[0].1, "rect1");

        // Query overlapping rect2
        let query = Aabb2::new(Vec2::splat(25.0), Vec2::splat(35.0));
        let results = tree.query(&query);
        assert_eq!(results.len(), 1);
        assert_eq!(*results[0].1, "rect2");

        // Query not overlapping anything
        let query = Aabb2::new(Vec2::splat(100.0), Vec2::splat(110.0));
        let results = tree.query(&query);
        assert!(results.is_empty());
    }

    #[test]
    fn test_rtree_query_point() {
        let mut tree = Rtree::new(4);
        tree.insert(Aabb2::new(Vec2::ZERO, Vec2::splat(10.0)), "rect1");
        tree.insert(Aabb2::new(Vec2::splat(5.0), Vec2::splat(15.0)), "rect2");

        // Point inside both rectangles
        let results = tree.query_point(Vec2::splat(7.0));
        assert_eq!(results.len(), 2);

        // Point inside only rect1
        let results = tree.query_point(Vec2::splat(2.0));
        assert_eq!(results.len(), 1);
        assert_eq!(*results[0].1, "rect1");

        // Point outside all
        let results = tree.query_point(Vec2::splat(100.0));
        assert!(results.is_empty());
    }

    #[test]
    fn test_rtree_split() {
        let mut tree = Rtree::new(2); // Small max to force splits

        // Insert enough to trigger splits
        for i in 0..10 {
            let offset = i as f32 * 10.0;
            tree.insert(
                Aabb2::new(Vec2::splat(offset), Vec2::splat(offset + 5.0)),
                i,
            );
        }

        assert_eq!(tree.len(), 10);

        // All entries should still be queryable
        for i in 0..10 {
            let offset = i as f32 * 10.0;
            let results = tree.query_point(Vec2::splat(offset + 2.5));
            assert!(!results.is_empty());
        }
    }

    #[test]
    fn test_rtree_overlapping_results() {
        let mut tree = Rtree::new(4);

        // Create overlapping rectangles
        tree.insert(Aabb2::new(Vec2::ZERO, Vec2::splat(20.0)), "A");
        tree.insert(Aabb2::new(Vec2::splat(10.0), Vec2::splat(30.0)), "B");
        tree.insert(Aabb2::new(Vec2::splat(15.0), Vec2::splat(25.0)), "C");

        // Query region that overlaps all three
        let query = Aabb2::new(Vec2::splat(12.0), Vec2::splat(18.0));
        let results = tree.query(&query);
        assert_eq!(results.len(), 3);
    }

    // KD-tree 2D tests

    #[test]
    fn test_kdtree2d_empty() {
        let tree: KdTree2D<i32> = KdTree2D::build(vec![]);
        assert!(tree.is_empty());
        assert_eq!(tree.len(), 0);
        assert!(tree.nearest(Vec2::ZERO).is_none());
    }

    #[test]
    fn test_kdtree2d_single() {
        let tree = KdTree2D::build(vec![(Vec2::new(5.0, 5.0), "A")]);
        assert_eq!(tree.len(), 1);

        let (pos, data, dist) = tree.nearest(Vec2::ZERO).unwrap();
        assert_eq!(*data, "A");
        assert!((pos - Vec2::new(5.0, 5.0)).length() < 1e-5);
        assert!((dist - 5.0_f32.hypot(5.0)).abs() < 1e-5);
    }

    #[test]
    fn test_kdtree2d_nearest() {
        let points = vec![
            (Vec2::new(10.0, 10.0), "A"),
            (Vec2::new(50.0, 50.0), "B"),
            (Vec2::new(90.0, 90.0), "C"),
        ];
        let tree = KdTree2D::build(points);

        // Closest to origin is A
        let (_, data, _) = tree.nearest(Vec2::ZERO).unwrap();
        assert_eq!(*data, "A");

        // Closest to (60, 60) is B
        let (_, data, _) = tree.nearest(Vec2::new(60.0, 60.0)).unwrap();
        assert_eq!(*data, "B");

        // Closest to (100, 100) is C
        let (_, data, _) = tree.nearest(Vec2::splat(100.0)).unwrap();
        assert_eq!(*data, "C");
    }

    #[test]
    fn test_kdtree2d_k_nearest() {
        let points = vec![
            (Vec2::new(10.0, 10.0), "A"),
            (Vec2::new(20.0, 20.0), "B"),
            (Vec2::new(50.0, 50.0), "C"),
            (Vec2::new(90.0, 90.0), "D"),
        ];
        let tree = KdTree2D::build(points);

        let results = tree.k_nearest(Vec2::ZERO, 2);
        assert_eq!(results.len(), 2);
        assert_eq!(*results[0].1, "A"); // Closest
        assert_eq!(*results[1].1, "B"); // Second closest

        let results = tree.k_nearest(Vec2::ZERO, 10); // More than available
        assert_eq!(results.len(), 4);
    }

    #[test]
    fn test_kdtree2d_query_region() {
        let points = vec![
            (Vec2::new(10.0, 10.0), 1),
            (Vec2::new(20.0, 20.0), 2),
            (Vec2::new(80.0, 80.0), 3),
        ];
        let tree = KdTree2D::build(points);

        let region = Aabb2::new(Vec2::ZERO, Vec2::splat(50.0));
        let results = tree.query_region(&region);
        assert_eq!(results.len(), 2);

        let region = Aabb2::new(Vec2::splat(60.0), Vec2::splat(100.0));
        let results = tree.query_region(&region);
        assert_eq!(results.len(), 1);
        assert_eq!(*results[0].1, 3);
    }

    #[test]
    fn test_kdtree2d_query_radius() {
        let points = vec![
            (Vec2::new(0.0, 0.0), "origin"),
            (Vec2::new(5.0, 0.0), "near"),
            (Vec2::new(100.0, 0.0), "far"),
        ];
        let tree = KdTree2D::build(points);

        let results = tree.query_radius(Vec2::ZERO, 10.0);
        assert_eq!(results.len(), 2);

        let results = tree.query_radius(Vec2::ZERO, 1.0);
        assert_eq!(results.len(), 1);
        assert_eq!(*results[0].1, "origin");
    }

    // KD-tree 3D tests

    #[test]
    fn test_kdtree3d_empty() {
        let tree: KdTree3D<i32> = KdTree3D::build(vec![]);
        assert!(tree.is_empty());
        assert!(tree.nearest(Vec3::ZERO).is_none());
    }

    #[test]
    fn test_kdtree3d_nearest() {
        let points = vec![
            (Vec3::new(10.0, 10.0, 10.0), "A"),
            (Vec3::new(50.0, 50.0, 50.0), "B"),
            (Vec3::new(90.0, 90.0, 90.0), "C"),
        ];
        let tree = KdTree3D::build(points);

        let (_, data, _) = tree.nearest(Vec3::ZERO).unwrap();
        assert_eq!(*data, "A");

        let (_, data, _) = tree.nearest(Vec3::splat(60.0)).unwrap();
        assert_eq!(*data, "B");

        let (_, data, _) = tree.nearest(Vec3::splat(100.0)).unwrap();
        assert_eq!(*data, "C");
    }

    #[test]
    fn test_kdtree3d_k_nearest() {
        let points = vec![
            (Vec3::new(10.0, 10.0, 10.0), "A"),
            (Vec3::new(20.0, 20.0, 20.0), "B"),
            (Vec3::new(50.0, 50.0, 50.0), "C"),
        ];
        let tree = KdTree3D::build(points);

        let results = tree.k_nearest(Vec3::ZERO, 2);
        assert_eq!(results.len(), 2);
        assert_eq!(*results[0].1, "A");
        assert_eq!(*results[1].1, "B");
    }

    #[test]
    fn test_kdtree3d_query_region() {
        let points = vec![
            (Vec3::new(10.0, 10.0, 10.0), 1),
            (Vec3::new(20.0, 20.0, 20.0), 2),
            (Vec3::new(80.0, 80.0, 80.0), 3),
        ];
        let tree = KdTree3D::build(points);

        let region = Aabb3::new(Vec3::ZERO, Vec3::splat(50.0));
        let results = tree.query_region(&region);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_kdtree3d_query_radius() {
        let points = vec![
            (Vec3::ZERO, "origin"),
            (Vec3::new(5.0, 0.0, 0.0), "near"),
            (Vec3::new(100.0, 0.0, 0.0), "far"),
        ];
        let tree = KdTree3D::build(points);

        let results = tree.query_radius(Vec3::ZERO, 10.0);
        assert_eq!(results.len(), 2);
    }

    // Ball tree 2D tests

    #[test]
    fn test_balltree2d_empty() {
        let tree: BallTree2D<i32> = BallTree2D::build(vec![]);
        assert!(tree.is_empty());
        assert_eq!(tree.len(), 0);
        assert!(tree.nearest(Vec2::ZERO).is_none());
    }

    #[test]
    fn test_balltree2d_single() {
        let tree = BallTree2D::build(vec![(Vec2::new(5.0, 5.0), "A")]);
        assert_eq!(tree.len(), 1);

        let (pos, data, dist) = tree.nearest(Vec2::ZERO).unwrap();
        assert_eq!(*data, "A");
        assert!((pos - Vec2::new(5.0, 5.0)).length() < 1e-5);
        assert!((dist - 5.0_f32.hypot(5.0)).abs() < 1e-5);
    }

    #[test]
    fn test_balltree2d_nearest() {
        let points = vec![
            (Vec2::new(10.0, 10.0), "A"),
            (Vec2::new(50.0, 50.0), "B"),
            (Vec2::new(90.0, 90.0), "C"),
        ];
        let tree = BallTree2D::build(points);

        let (_, data, _) = tree.nearest(Vec2::ZERO).unwrap();
        assert_eq!(*data, "A");

        let (_, data, _) = tree.nearest(Vec2::new(60.0, 60.0)).unwrap();
        assert_eq!(*data, "B");

        let (_, data, _) = tree.nearest(Vec2::splat(100.0)).unwrap();
        assert_eq!(*data, "C");
    }

    #[test]
    fn test_balltree2d_k_nearest() {
        let points = vec![
            (Vec2::new(10.0, 10.0), "A"),
            (Vec2::new(20.0, 20.0), "B"),
            (Vec2::new(50.0, 50.0), "C"),
            (Vec2::new(90.0, 90.0), "D"),
        ];
        let tree = BallTree2D::build(points);

        let results = tree.k_nearest(Vec2::ZERO, 2);
        assert_eq!(results.len(), 2);
        assert_eq!(*results[0].1, "A");
        assert_eq!(*results[1].1, "B");

        let results = tree.k_nearest(Vec2::ZERO, 10);
        assert_eq!(results.len(), 4);
    }

    #[test]
    fn test_balltree2d_query_radius() {
        let points = vec![
            (Vec2::new(0.0, 0.0), "origin"),
            (Vec2::new(5.0, 0.0), "near"),
            (Vec2::new(100.0, 0.0), "far"),
        ];
        let tree = BallTree2D::build(points);

        let results = tree.query_radius(Vec2::ZERO, 10.0);
        assert_eq!(results.len(), 2);

        let results = tree.query_radius(Vec2::ZERO, 1.0);
        assert_eq!(results.len(), 1);
        assert_eq!(*results[0].1, "origin");
    }

    // Ball tree 3D tests

    #[test]
    fn test_balltree3d_empty() {
        let tree: BallTree3D<i32> = BallTree3D::build(vec![]);
        assert!(tree.is_empty());
        assert!(tree.nearest(Vec3::ZERO).is_none());
    }

    #[test]
    fn test_balltree3d_nearest() {
        let points = vec![
            (Vec3::new(10.0, 10.0, 10.0), "A"),
            (Vec3::new(50.0, 50.0, 50.0), "B"),
            (Vec3::new(90.0, 90.0, 90.0), "C"),
        ];
        let tree = BallTree3D::build(points);

        let (_, data, _) = tree.nearest(Vec3::ZERO).unwrap();
        assert_eq!(*data, "A");

        let (_, data, _) = tree.nearest(Vec3::splat(60.0)).unwrap();
        assert_eq!(*data, "B");

        let (_, data, _) = tree.nearest(Vec3::splat(100.0)).unwrap();
        assert_eq!(*data, "C");
    }

    #[test]
    fn test_balltree3d_k_nearest() {
        let points = vec![
            (Vec3::new(10.0, 10.0, 10.0), "A"),
            (Vec3::new(20.0, 20.0, 20.0), "B"),
            (Vec3::new(50.0, 50.0, 50.0), "C"),
        ];
        let tree = BallTree3D::build(points);

        let results = tree.k_nearest(Vec3::ZERO, 2);
        assert_eq!(results.len(), 2);
        assert_eq!(*results[0].1, "A");
        assert_eq!(*results[1].1, "B");
    }

    #[test]
    fn test_balltree3d_query_radius() {
        let points = vec![
            (Vec3::ZERO, "origin"),
            (Vec3::new(5.0, 0.0, 0.0), "near"),
            (Vec3::new(100.0, 0.0, 0.0), "far"),
        ];
        let tree = BallTree3D::build(points);

        let results = tree.query_radius(Vec3::ZERO, 10.0);
        assert_eq!(results.len(), 2);
    }
}

// ============================================================================
// Invariant tests
// ============================================================================

/// Invariant tests for spatial data structures.
///
/// These tests verify mathematical properties that should hold for spatial
/// data structures. Run with:
///
/// ```sh
/// cargo test -p unshape-spatial --features invariant-tests
/// ```
#[cfg(all(test, feature = "invariant-tests"))]
mod invariant_tests {
    use super::*;

    // ========================================================================
    // AABB invariants
    // ========================================================================

    /// AABB center is midpoint of min and max.
    #[test]
    fn test_aabb2_center_is_midpoint() {
        for _ in 0..100 {
            let min = Vec2::new(rand_f32(-100.0, 100.0), rand_f32(-100.0, 100.0));
            let max = min + Vec2::new(rand_f32(0.1, 50.0), rand_f32(0.1, 50.0));
            let aabb = Aabb2::new(min, max);

            let expected_center = (min + max) * 0.5;
            let center = aabb.center();

            assert!(
                (center - expected_center).length() < 1e-5,
                "Center should be midpoint: expected {expected_center:?}, got {center:?}"
            );
        }
    }

    /// AABB3 center is midpoint of min and max.
    #[test]
    fn test_aabb3_center_is_midpoint() {
        for _ in 0..100 {
            let min = Vec3::new(
                rand_f32(-100.0, 100.0),
                rand_f32(-100.0, 100.0),
                rand_f32(-100.0, 100.0),
            );
            let max = min
                + Vec3::new(
                    rand_f32(0.1, 50.0),
                    rand_f32(0.1, 50.0),
                    rand_f32(0.1, 50.0),
                );
            let aabb = Aabb3::new(min, max);

            let expected_center = (min + max) * 0.5;
            let center = aabb.center();

            assert!(
                (center - expected_center).length() < 1e-5,
                "Center should be midpoint"
            );
        }
    }

    /// AABB quadrants should cover the original AABB exactly.
    #[test]
    fn test_aabb2_quadrants_cover_original() {
        for _ in 0..50 {
            let min = Vec2::new(rand_f32(-100.0, 100.0), rand_f32(-100.0, 100.0));
            let max = min + Vec2::new(rand_f32(1.0, 50.0), rand_f32(1.0, 50.0));
            let aabb = Aabb2::new(min, max);
            let quadrants = aabb.quadrants();

            // Test random points - if in original, must be in exactly one quadrant
            for _ in 0..100 {
                let point = Vec2::new(
                    rand_f32(aabb.min.x, aabb.max.x),
                    rand_f32(aabb.min.y, aabb.max.y),
                );

                let containing_quads: Vec<_> = quadrants
                    .iter()
                    .enumerate()
                    .filter(|(_, q)| q.contains_point(point))
                    .collect();

                assert!(
                    !containing_quads.is_empty(),
                    "Point {point:?} should be in at least one quadrant"
                );
            }
        }
    }

    /// AABB octants should cover the original AABB exactly.
    #[test]
    fn test_aabb3_octants_cover_original() {
        for _ in 0..50 {
            let min = Vec3::new(
                rand_f32(-100.0, 100.0),
                rand_f32(-100.0, 100.0),
                rand_f32(-100.0, 100.0),
            );
            let max = min + Vec3::splat(rand_f32(1.0, 50.0));
            let aabb = Aabb3::new(min, max);
            let octants = aabb.octants();

            // Test random points
            for _ in 0..100 {
                let point = Vec3::new(
                    rand_f32(aabb.min.x, aabb.max.x),
                    rand_f32(aabb.min.y, aabb.max.y),
                    rand_f32(aabb.min.z, aabb.max.z),
                );

                let containing_octs: Vec<_> = octants
                    .iter()
                    .enumerate()
                    .filter(|(_, o)| o.contains_point(point))
                    .collect();

                assert!(
                    !containing_octs.is_empty(),
                    "Point {point:?} should be in at least one octant"
                );
            }
        }
    }

    /// AABB intersection is symmetric.
    #[test]
    fn test_aabb_intersection_symmetric() {
        for _ in 0..100 {
            let a = Aabb2::new(
                Vec2::new(rand_f32(-50.0, 50.0), rand_f32(-50.0, 50.0)),
                Vec2::new(rand_f32(0.0, 100.0), rand_f32(0.0, 100.0)),
            );
            let b = Aabb2::new(
                Vec2::new(rand_f32(-50.0, 50.0), rand_f32(-50.0, 50.0)),
                Vec2::new(rand_f32(0.0, 100.0), rand_f32(0.0, 100.0)),
            );

            assert_eq!(
                a.intersects(&b),
                b.intersects(&a),
                "Intersection should be symmetric"
            );
        }
    }

    // ========================================================================
    // Quadtree invariants
    // ========================================================================

    /// All inserted points should be found via range query.
    #[test]
    fn test_quadtree_all_points_queryable() {
        let bounds = Aabb2::new(Vec2::ZERO, Vec2::splat(100.0));
        let mut tree = Quadtree::new(bounds, 8, 4);

        let mut points = Vec::new();
        for i in 0..100 {
            let point = Vec2::new(rand_f32(0.0, 100.0), rand_f32(0.0, 100.0));
            points.push((point, i));
            tree.insert(point, i);
        }

        assert_eq!(tree.len(), 100);

        // Query entire bounds should return all points
        let results: Vec<_> = tree.query_region(&bounds).collect();
        assert_eq!(results.len(), 100);
    }

    /// Range query only returns points within query region.
    #[test]
    fn test_quadtree_range_query_correctness() {
        let bounds = Aabb2::new(Vec2::ZERO, Vec2::splat(100.0));
        let mut tree = Quadtree::new(bounds, 8, 4);

        for i in 0..100 {
            let point = Vec2::new(rand_f32(0.0, 100.0), rand_f32(0.0, 100.0));
            tree.insert(point, i);
        }

        // Query a sub-region
        let query = Aabb2::new(Vec2::new(25.0, 25.0), Vec2::new(75.0, 75.0));
        let results: Vec<_> = tree.query_region(&query).collect();

        // All returned points must be within query region
        for (pos, _) in &results {
            assert!(
                query.contains_point(*pos),
                "Returned point {:?} outside query region",
                pos
            );
        }
    }

    /// Nearest neighbor returns the truly closest point.
    #[test]
    fn test_quadtree_nearest_is_truly_closest() {
        let bounds = Aabb2::new(Vec2::ZERO, Vec2::splat(100.0));
        let mut tree = Quadtree::new(bounds, 8, 4);

        let mut points = Vec::new();
        for i in 0..50 {
            let point = Vec2::new(rand_f32(0.0, 100.0), rand_f32(0.0, 100.0));
            points.push(point);
            tree.insert(point, i);
        }

        // Test several query points
        for _ in 0..20 {
            let query = Vec2::new(rand_f32(0.0, 100.0), rand_f32(0.0, 100.0));

            // Find nearest via tree
            let tree_nearest = tree.nearest(query);
            assert!(tree_nearest.is_some());
            let (tree_pos, _, tree_dist) = tree_nearest.unwrap();

            // Find nearest via brute force
            let brute_nearest = points
                .iter()
                .map(|p| (*p, p.distance(query)))
                .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                .unwrap();

            assert!(
                (tree_dist - brute_nearest.1).abs() < 1e-5,
                "Tree nearest ({tree_dist}) != brute force ({}) for query {query:?}",
                brute_nearest.1
            );
            assert!(
                (tree_pos - brute_nearest.0).length() < 1e-5,
                "Tree returned wrong point"
            );
        }
    }

    /// k-nearest returns exactly k closest points, correctly ordered.
    #[test]
    fn test_quadtree_k_nearest_correctness() {
        let bounds = Aabb2::new(Vec2::ZERO, Vec2::splat(100.0));
        let mut tree = Quadtree::new(bounds, 8, 4);

        let mut points = Vec::new();
        for i in 0..50 {
            let point = Vec2::new(rand_f32(0.0, 100.0), rand_f32(0.0, 100.0));
            points.push(point);
            tree.insert(point, i);
        }

        // Test several k values and query points
        for k in [1, 3, 5, 10, 20] {
            for _ in 0..5 {
                let query = Vec2::new(rand_f32(0.0, 100.0), rand_f32(0.0, 100.0));

                // Find k-nearest via tree
                let tree_results = tree.k_nearest(query, k);

                // Find k-nearest via brute force
                let mut brute_results: Vec<_> =
                    points.iter().map(|p| (*p, p.distance(query))).collect();
                brute_results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
                let brute_k: Vec<_> = brute_results.into_iter().take(k).collect();

                // Should return same count
                assert_eq!(
                    tree_results.len(),
                    brute_k.len(),
                    "k={k}: tree returned {} points, brute force {}",
                    tree_results.len(),
                    brute_k.len()
                );

                // Should be sorted by distance (closest first)
                for i in 1..tree_results.len() {
                    assert!(
                        tree_results[i].2 >= tree_results[i - 1].2,
                        "k={k}: results not sorted at index {i}"
                    );
                }

                // Distances should match brute force
                for (i, ((_, _, tree_dist), (_, brute_dist))) in
                    tree_results.iter().zip(brute_k.iter()).enumerate()
                {
                    assert!(
                        (tree_dist - brute_dist).abs() < 1e-5,
                        "k={k}: distance mismatch at index {i}: tree={tree_dist}, brute={brute_dist}"
                    );
                }
            }
        }
    }

    // ========================================================================
    // Octree invariants
    // ========================================================================

    /// All inserted points should be found via range query.
    #[test]
    fn test_octree_all_points_queryable() {
        let bounds = Aabb3::new(Vec3::ZERO, Vec3::splat(100.0));
        let mut tree = Octree::new(bounds, 8, 4);

        for i in 0..100 {
            let point = Vec3::new(
                rand_f32(0.0, 100.0),
                rand_f32(0.0, 100.0),
                rand_f32(0.0, 100.0),
            );
            tree.insert(point, i);
        }

        assert_eq!(tree.len(), 100);

        // Query entire bounds should return all points
        let results: Vec<_> = tree.query_region(&bounds).collect();
        assert_eq!(results.len(), 100);
    }

    /// Nearest neighbor returns the truly closest point.
    #[test]
    fn test_octree_nearest_is_truly_closest() {
        let bounds = Aabb3::new(Vec3::ZERO, Vec3::splat(100.0));
        let mut tree = Octree::new(bounds, 8, 4);

        let mut points = Vec::new();
        for i in 0..50 {
            let point = Vec3::new(
                rand_f32(0.0, 100.0),
                rand_f32(0.0, 100.0),
                rand_f32(0.0, 100.0),
            );
            points.push(point);
            tree.insert(point, i);
        }

        // Test several query points
        for _ in 0..20 {
            let query = Vec3::new(
                rand_f32(0.0, 100.0),
                rand_f32(0.0, 100.0),
                rand_f32(0.0, 100.0),
            );

            let tree_nearest = tree.nearest(query);
            assert!(tree_nearest.is_some());
            let (_, _, tree_dist) = tree_nearest.unwrap();

            // Brute force
            let brute_dist = points
                .iter()
                .map(|p| p.distance(query))
                .min_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap();

            assert!(
                (tree_dist - brute_dist).abs() < 1e-5,
                "Tree nearest ({tree_dist}) != brute force ({brute_dist})"
            );
        }
    }

    /// k-nearest returns exactly k closest points, correctly ordered.
    #[test]
    fn test_octree_k_nearest_correctness() {
        let bounds = Aabb3::new(Vec3::ZERO, Vec3::splat(100.0));
        let mut tree = Octree::new(bounds, 8, 4);

        let mut points = Vec::new();
        for i in 0..50 {
            let point = Vec3::new(
                rand_f32(0.0, 100.0),
                rand_f32(0.0, 100.0),
                rand_f32(0.0, 100.0),
            );
            points.push(point);
            tree.insert(point, i);
        }

        // Test several k values and query points
        for k in [1, 3, 5, 10, 20] {
            for _ in 0..5 {
                let query = Vec3::new(
                    rand_f32(0.0, 100.0),
                    rand_f32(0.0, 100.0),
                    rand_f32(0.0, 100.0),
                );

                // Find k-nearest via tree
                let tree_results = tree.k_nearest(query, k);

                // Find k-nearest via brute force
                let mut brute_results: Vec<_> =
                    points.iter().map(|p| (*p, p.distance(query))).collect();
                brute_results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
                let brute_k: Vec<_> = brute_results.into_iter().take(k).collect();

                // Should return same count
                assert_eq!(
                    tree_results.len(),
                    brute_k.len(),
                    "k={k}: tree returned {} points, brute force {}",
                    tree_results.len(),
                    brute_k.len()
                );

                // Should be sorted by distance (closest first)
                for i in 1..tree_results.len() {
                    assert!(
                        tree_results[i].2 >= tree_results[i - 1].2,
                        "k={k}: results not sorted at index {i}"
                    );
                }

                // Distances should match brute force
                for (i, ((_, _, tree_dist), (_, brute_dist))) in
                    tree_results.iter().zip(brute_k.iter()).enumerate()
                {
                    assert!(
                        (tree_dist - brute_dist).abs() < 1e-5,
                        "k={k}: distance mismatch at index {i}: tree={tree_dist}, brute={brute_dist}"
                    );
                }
            }
        }
    }

    // ========================================================================
    // Ray invariants
    // ========================================================================

    /// Ray at(t) returns origin when t=0.
    #[test]
    fn test_ray_at_zero_is_origin() {
        for _ in 0..50 {
            let origin = Vec3::new(
                rand_f32(-100.0, 100.0),
                rand_f32(-100.0, 100.0),
                rand_f32(-100.0, 100.0),
            );
            let dir = Vec3::new(
                rand_f32(-1.0, 1.0),
                rand_f32(-1.0, 1.0),
                rand_f32(-1.0, 1.0),
            )
            .normalize();
            let ray = Ray::new(origin, dir);

            assert!(
                (ray.at(0.0) - origin).length() < 1e-5,
                "Ray at(0) should be origin"
            );
        }
    }

    /// Ray-AABB intersection points are on the AABB surface.
    #[test]
    fn test_ray_aabb_hit_point_on_surface() {
        let aabb = Aabb3::new(Vec3::splat(-5.0), Vec3::splat(5.0));

        for _ in 0..50 {
            // Create ray pointing toward AABB from outside
            let origin = Vec3::new(
                rand_f32(-20.0, -10.0),
                rand_f32(-3.0, 3.0),
                rand_f32(-3.0, 3.0),
            );
            let target = Vec3::new(
                rand_f32(-4.0, 4.0),
                rand_f32(-4.0, 4.0),
                rand_f32(-4.0, 4.0),
            );
            let ray = Ray::new(origin, (target - origin).normalize());

            if let Some((t_min, _)) = ray.intersect_aabb(&aabb) {
                let hit = ray.at(t_min);

                // Hit point should be on or very close to AABB surface
                let on_min_x = (hit.x - aabb.min.x).abs() < 1e-3;
                let on_max_x = (hit.x - aabb.max.x).abs() < 1e-3;
                let on_min_y = (hit.y - aabb.min.y).abs() < 1e-3;
                let on_max_y = (hit.y - aabb.max.y).abs() < 1e-3;
                let on_min_z = (hit.z - aabb.min.z).abs() < 1e-3;
                let on_max_z = (hit.z - aabb.max.z).abs() < 1e-3;

                let on_surface =
                    on_min_x || on_max_x || on_min_y || on_max_y || on_min_z || on_max_z;

                assert!(on_surface, "Hit point {hit:?} should be on AABB surface");
            }
        }
    }

    // ========================================================================
    // BVH invariants
    // ========================================================================

    /// All primitives should be reachable via AABB query covering all.
    #[test]
    fn test_bvh_all_primitives_queryable() {
        let mut primitives = Vec::new();
        for i in 0..50 {
            let center = Vec3::new(
                rand_f32(0.0, 100.0),
                rand_f32(0.0, 100.0),
                rand_f32(0.0, 100.0),
            );
            let half = Vec3::splat(rand_f32(0.5, 2.0));
            primitives.push((Aabb3::new(center - half, center + half), i));
        }

        let bvh = Bvh::build(primitives);
        assert_eq!(bvh.len(), 50);

        // Query entire space
        let query = Aabb3::new(Vec3::splat(-10.0), Vec3::splat(110.0));
        let results = bvh.query_aabb(&query);

        assert_eq!(results.len(), 50, "All primitives should be found");
    }

    /// BVH ray intersection returns only primitives the ray actually hits.
    #[test]
    fn test_bvh_ray_hits_are_valid() {
        let primitives = vec![
            (
                Aabb3::new(Vec3::new(0.0, 0.0, 0.0), Vec3::new(2.0, 2.0, 2.0)),
                "A",
            ),
            (
                Aabb3::new(Vec3::new(10.0, 0.0, 0.0), Vec3::new(12.0, 2.0, 2.0)),
                "B",
            ),
            (
                Aabb3::new(Vec3::new(20.0, 0.0, 0.0), Vec3::new(22.0, 2.0, 2.0)),
                "C",
            ),
        ];

        let bvh = Bvh::build(primitives);

        // Ray hitting A
        let ray = Ray::new(Vec3::new(1.0, 1.0, -5.0), Vec3::Z);
        let hits = bvh.intersect_ray(&ray);
        assert_eq!(hits.len(), 1);

        // Verify the hit is valid
        for (aabb, _) in &hits {
            assert!(
                ray.intersect_aabb(aabb).is_some(),
                "Returned primitive should actually be hit"
            );
        }
    }

    // ========================================================================
    // Spatial hash invariants
    // ========================================================================

    /// Points in same cell are within cell_size * sqrt(3) of each other.
    #[test]
    fn test_spatial_hash_cell_locality() {
        let cell_size = 10.0;
        let mut hash = SpatialHash::new(cell_size);

        // Insert points in same cell
        let base = Vec3::new(5.0, 5.0, 5.0);
        hash.insert(base, "A");
        hash.insert(base + Vec3::new(1.0, 1.0, 1.0), "B");
        hash.insert(base + Vec3::new(2.0, 2.0, 2.0), "C");

        let results: Vec<_> = hash.query_cell(base).collect();

        // All points in same cell should be within cell diagonal
        let max_dist = cell_size * 3.0_f32.sqrt();
        for (pos, _) in &results {
            let dist = pos.distance(base);
            assert!(
                dist <= max_dist,
                "Point in same cell should be within cell diagonal"
            );
        }
    }

    /// Radius query returns all points within radius.
    #[test]
    fn test_spatial_hash_radius_query_complete() {
        let mut hash = SpatialHash::new(10.0);

        let mut points = Vec::new();
        for i in 0..100 {
            let point = Vec3::new(
                rand_f32(0.0, 100.0),
                rand_f32(0.0, 100.0),
                rand_f32(0.0, 100.0),
            );
            points.push(point);
            hash.insert(point, i);
        }

        let query = Vec3::new(50.0, 50.0, 50.0);
        let radius = 20.0;

        let results: Vec<_> = hash.query_radius(query, radius).collect();

        // Count via brute force
        let brute_count = points
            .iter()
            .filter(|p| p.distance(query) <= radius)
            .count();

        assert_eq!(
            results.len(),
            brute_count,
            "Radius query should return all points within radius"
        );
    }

    // ========================================================================
    // R-tree invariants
    // ========================================================================

    /// All inserted rectangles should be found via full-bounds query.
    #[test]
    fn test_rtree_all_rects_queryable() {
        let mut tree = Rtree::new(4);

        for i in 0..50 {
            let min = Vec2::new(rand_f32(0.0, 90.0), rand_f32(0.0, 90.0));
            let max = min + Vec2::new(rand_f32(1.0, 10.0), rand_f32(1.0, 10.0));
            tree.insert(Aabb2::new(min, max), i);
        }

        assert_eq!(tree.len(), 50);

        // Query entire space
        let query = Aabb2::new(Vec2::splat(-10.0), Vec2::splat(200.0));
        let results = tree.query(&query);

        assert_eq!(results.len(), 50);
    }

    /// Query only returns rectangles that actually intersect.
    #[test]
    fn test_rtree_query_correctness() {
        let mut tree = Rtree::new(4);

        let mut rects = Vec::new();
        for i in 0..30 {
            let min = Vec2::new(rand_f32(0.0, 90.0), rand_f32(0.0, 90.0));
            let max = min + Vec2::new(rand_f32(1.0, 10.0), rand_f32(1.0, 10.0));
            let rect = Aabb2::new(min, max);
            rects.push(rect);
            tree.insert(rect, i);
        }

        let query = Aabb2::new(Vec2::new(40.0, 40.0), Vec2::new(60.0, 60.0));
        let results = tree.query(&query);

        // All returned rects must intersect query
        for (bounds, _) in &results {
            assert!(
                bounds.intersects(&query),
                "Returned rect should intersect query"
            );
        }

        // Count should match brute force
        let brute_count = rects.iter().filter(|r| r.intersects(&query)).count();
        assert_eq!(results.len(), brute_count);
    }

    // ========================================================================
    // Helper functions
    // ========================================================================

    /// Simple LCG random number generator for tests.
    fn rand_f32(min: f32, max: f32) -> f32 {
        use std::cell::Cell;
        thread_local! {
            static SEED: Cell<u64> = const { Cell::new(12345) };
        }
        SEED.with(|seed| {
            let s = seed.get().wrapping_mul(6364136223846793005).wrapping_add(1);
            seed.set(s);
            let t = ((s >> 33) as u32) as f32 / u32::MAX as f32;
            min + t * (max - min)
        })
    }
}
