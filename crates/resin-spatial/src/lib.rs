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
//! use rhizome_resin_spatial::{Quadtree, Aabb2};
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
use std::collections::BinaryHeap;

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
struct KNearestCandidate2D<'a, T> {
    position: Vec2,
    data: &'a T,
    distance: f32,
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
struct KNearestCandidate3D<'a, T> {
    position: Vec3,
    data: &'a T,
    distance: f32,
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
// Quadtree
// ============================================================================

/// A point with associated data stored in a quadtree.
#[derive(Debug, Clone)]
struct QuadtreeEntry<T> {
    position: Vec2,
    data: T,
}

/// A node in the quadtree.
#[derive(Debug)]
enum QuadtreeNode<T> {
    /// Leaf node containing points.
    Leaf { entries: Vec<QuadtreeEntry<T>> },
    /// Internal node with four children.
    Internal { children: Box<[QuadtreeNode<T>; 4]> },
}

/// A quadtree for 2D spatial partitioning.
///
/// Efficiently stores and queries points in 2D space by recursively subdividing
/// the space into quadrants.
///
/// # Type Parameters
///
/// * `T` - The type of data associated with each point.
///
/// # Example
///
/// ```
/// use rhizome_resin_spatial::{Quadtree, Aabb2};
/// use glam::Vec2;
///
/// let bounds = Aabb2::new(Vec2::ZERO, Vec2::splat(100.0));
/// let mut tree = Quadtree::new(bounds, 8, 4);
///
/// tree.insert(Vec2::new(10.0, 20.0), 1);
/// tree.insert(Vec2::new(50.0, 50.0), 2);
///
/// // Query all points in a region
/// let query = Aabb2::new(Vec2::ZERO, Vec2::splat(30.0));
/// for (pos, data) in tree.query_region(&query) {
///     println!("Found point at {:?} with data {}", pos, data);
/// }
/// ```
#[derive(Debug)]
pub struct Quadtree<T> {
    root: QuadtreeNode<T>,
    bounds: Aabb2,
    max_depth: usize,
    max_entries_per_leaf: usize,
}

impl<T> Quadtree<T> {
    /// Creates a new quadtree with the given bounds and parameters.
    ///
    /// # Arguments
    ///
    /// * `bounds` - The bounding box of the entire quadtree.
    /// * `max_depth` - Maximum depth of the tree (prevents infinite subdivision).
    /// * `max_entries_per_leaf` - Maximum entries per leaf before subdivision.
    pub fn new(bounds: Aabb2, max_depth: usize, max_entries_per_leaf: usize) -> Self {
        Self {
            root: QuadtreeNode::Leaf {
                entries: Vec::new(),
            },
            bounds,
            max_depth,
            max_entries_per_leaf,
        }
    }

    /// Returns the bounds of this quadtree.
    pub fn bounds(&self) -> Aabb2 {
        self.bounds
    }

    /// Inserts a point with associated data into the quadtree.
    ///
    /// Returns `true` if the point was inserted, `false` if it's outside bounds.
    pub fn insert(&mut self, position: Vec2, data: T) -> bool {
        if !self.bounds.contains_point(position) {
            return false;
        }

        Self::insert_recursive(
            &mut self.root,
            self.bounds,
            QuadtreeEntry { position, data },
            0,
            self.max_depth,
            self.max_entries_per_leaf,
        );
        true
    }

    fn insert_recursive(
        node: &mut QuadtreeNode<T>,
        bounds: Aabb2,
        entry: QuadtreeEntry<T>,
        depth: usize,
        max_depth: usize,
        max_entries: usize,
    ) {
        match node {
            QuadtreeNode::Leaf { entries } => {
                entries.push(entry);

                // Subdivide if we exceed capacity and haven't reached max depth
                if entries.len() > max_entries && depth < max_depth {
                    let quadrants = bounds.quadrants();
                    let old_entries = std::mem::take(entries);

                    let mut children: [QuadtreeNode<T>; 4] = [
                        QuadtreeNode::Leaf {
                            entries: Vec::new(),
                        },
                        QuadtreeNode::Leaf {
                            entries: Vec::new(),
                        },
                        QuadtreeNode::Leaf {
                            entries: Vec::new(),
                        },
                        QuadtreeNode::Leaf {
                            entries: Vec::new(),
                        },
                    ];

                    for e in old_entries {
                        for (i, quadrant) in quadrants.iter().enumerate() {
                            if quadrant.contains_point(e.position) {
                                Self::insert_recursive(
                                    &mut children[i],
                                    *quadrant,
                                    e,
                                    depth + 1,
                                    max_depth,
                                    max_entries,
                                );
                                break;
                            }
                        }
                    }

                    *node = QuadtreeNode::Internal {
                        children: Box::new(children),
                    };
                }
            }
            QuadtreeNode::Internal { children } => {
                let quadrants = bounds.quadrants();
                for (i, quadrant) in quadrants.iter().enumerate() {
                    if quadrant.contains_point(entry.position) {
                        Self::insert_recursive(
                            &mut children[i],
                            *quadrant,
                            entry,
                            depth + 1,
                            max_depth,
                            max_entries,
                        );
                        break;
                    }
                }
            }
        }
    }

    /// Queries all points within the given region.
    ///
    /// Returns an iterator over (position, data) pairs.
    pub fn query_region(&self, region: &Aabb2) -> impl Iterator<Item = (Vec2, &T)> {
        let mut results = Vec::new();
        Self::query_recursive(&self.root, self.bounds, region, &mut results);
        results.into_iter()
    }

    fn query_recursive<'a>(
        node: &'a QuadtreeNode<T>,
        bounds: Aabb2,
        region: &Aabb2,
        results: &mut Vec<(Vec2, &'a T)>,
    ) {
        if !bounds.intersects(region) {
            return;
        }

        match node {
            QuadtreeNode::Leaf { entries } => {
                for entry in entries {
                    if region.contains_point(entry.position) {
                        results.push((entry.position, &entry.data));
                    }
                }
            }
            QuadtreeNode::Internal { children } => {
                let quadrants = bounds.quadrants();
                for (i, child) in children.iter().enumerate() {
                    Self::query_recursive(child, quadrants[i], region, results);
                }
            }
        }
    }

    /// Finds the nearest point to the given position.
    ///
    /// Returns `None` if the tree is empty.
    pub fn nearest(&self, position: Vec2) -> Option<(Vec2, &T, f32)> {
        let mut best: Option<(Vec2, &T, f32)> = None;
        Self::nearest_recursive(&self.root, self.bounds, position, &mut best);
        best
    }

    fn nearest_recursive<'a>(
        node: &'a QuadtreeNode<T>,
        bounds: Aabb2,
        position: Vec2,
        best: &mut Option<(Vec2, &'a T, f32)>,
    ) {
        // Early exit if this node can't possibly contain a closer point
        if let Some((_, _, best_dist)) = best {
            let closest_in_bounds = Vec2::new(
                position.x.clamp(bounds.min.x, bounds.max.x),
                position.y.clamp(bounds.min.y, bounds.max.y),
            );
            if closest_in_bounds.distance(position) >= *best_dist {
                return;
            }
        }

        match node {
            QuadtreeNode::Leaf { entries } => {
                for entry in entries {
                    let dist = entry.position.distance(position);
                    if best.is_none() || dist < best.as_ref().unwrap().2 {
                        *best = Some((entry.position, &entry.data, dist));
                    }
                }
            }
            QuadtreeNode::Internal { children } => {
                let quadrants = bounds.quadrants();

                // Sort children by distance to query point for better pruning
                let mut indices: Vec<usize> = (0..4).collect();
                indices.sort_by(|&a, &b| {
                    let dist_a = quadrants[a].center().distance(position);
                    let dist_b = quadrants[b].center().distance(position);
                    dist_a.partial_cmp(&dist_b).unwrap()
                });

                for i in indices {
                    Self::nearest_recursive(&children[i], quadrants[i], position, best);
                }
            }
        }
    }

    /// Finds the k nearest points to the given position.
    ///
    /// Returns up to k points, sorted by distance (closest first).
    /// Returns fewer than k if the tree contains fewer points.
    ///
    /// # Example
    ///
    /// ```
    /// use rhizome_resin_spatial::{Quadtree, Aabb2};
    /// use glam::Vec2;
    ///
    /// let bounds = Aabb2::new(Vec2::ZERO, Vec2::splat(100.0));
    /// let mut tree = Quadtree::new(bounds, 8, 4);
    ///
    /// tree.insert(Vec2::new(10.0, 10.0), "A");
    /// tree.insert(Vec2::new(20.0, 20.0), "B");
    /// tree.insert(Vec2::new(50.0, 50.0), "C");
    ///
    /// let nearest = tree.k_nearest(Vec2::ZERO, 2);
    /// assert_eq!(nearest.len(), 2);
    /// assert_eq!(nearest[0].1, &"A"); // Closest
    /// assert_eq!(nearest[1].1, &"B"); // Second closest
    /// ```
    pub fn k_nearest(&self, position: Vec2, k: usize) -> Vec<(Vec2, &T, f32)> {
        if k == 0 {
            return Vec::new();
        }

        let mut heap: BinaryHeap<KNearestCandidate2D<T>> = BinaryHeap::new();
        Self::k_nearest_recursive(&self.root, self.bounds, position, k, &mut heap);

        // Convert heap to sorted vec (closest first)
        let mut results: Vec<_> = heap
            .into_iter()
            .map(|c| (c.position, c.data, c.distance))
            .collect();
        results.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap());
        results
    }

    fn k_nearest_recursive<'a>(
        node: &'a QuadtreeNode<T>,
        bounds: Aabb2,
        position: Vec2,
        k: usize,
        heap: &mut BinaryHeap<KNearestCandidate2D<'a, T>>,
    ) {
        // Early exit if this node can't possibly contain a closer point
        if heap.len() >= k {
            let worst_dist = heap.peek().unwrap().distance;
            let closest_in_bounds = Vec2::new(
                position.x.clamp(bounds.min.x, bounds.max.x),
                position.y.clamp(bounds.min.y, bounds.max.y),
            );
            if closest_in_bounds.distance(position) >= worst_dist {
                return;
            }
        }

        match node {
            QuadtreeNode::Leaf { entries } => {
                for entry in entries {
                    let dist = entry.position.distance(position);

                    if heap.len() < k {
                        heap.push(KNearestCandidate2D {
                            position: entry.position,
                            data: &entry.data,
                            distance: dist,
                        });
                    } else if dist < heap.peek().unwrap().distance {
                        heap.pop();
                        heap.push(KNearestCandidate2D {
                            position: entry.position,
                            data: &entry.data,
                            distance: dist,
                        });
                    }
                }
            }
            QuadtreeNode::Internal { children } => {
                let quadrants = bounds.quadrants();

                // Sort children by distance to query point for better pruning
                let mut indices: Vec<usize> = (0..4).collect();
                indices.sort_by(|&a, &b| {
                    let dist_a = quadrants[a].center().distance(position);
                    let dist_b = quadrants[b].center().distance(position);
                    dist_a.partial_cmp(&dist_b).unwrap()
                });

                for i in indices {
                    Self::k_nearest_recursive(&children[i], quadrants[i], position, k, heap);
                }
            }
        }
    }

    /// Returns the total number of entries in the tree.
    pub fn len(&self) -> usize {
        Self::count_recursive(&self.root)
    }

    /// Returns `true` if the tree is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn count_recursive(node: &QuadtreeNode<T>) -> usize {
        match node {
            QuadtreeNode::Leaf { entries } => entries.len(),
            QuadtreeNode::Internal { children } => children.iter().map(Self::count_recursive).sum(),
        }
    }

    /// Clears all entries from the tree.
    pub fn clear(&mut self) {
        self.root = QuadtreeNode::Leaf {
            entries: Vec::new(),
        };
    }
}

// ============================================================================
// Octree
// ============================================================================

/// A point with associated data stored in an octree.
#[derive(Debug, Clone)]
struct OctreeEntry<T> {
    position: Vec3,
    data: T,
}

/// A node in the octree.
#[derive(Debug)]
enum OctreeNode<T> {
    /// Leaf node containing points.
    Leaf { entries: Vec<OctreeEntry<T>> },
    /// Internal node with eight children.
    Internal { children: Box<[OctreeNode<T>; 8]> },
}

/// An octree for 3D spatial partitioning.
///
/// Efficiently stores and queries points in 3D space by recursively subdividing
/// the space into octants.
///
/// # Type Parameters
///
/// * `T` - The type of data associated with each point.
///
/// # Example
///
/// ```
/// use rhizome_resin_spatial::{Octree, Aabb3};
/// use glam::Vec3;
///
/// let bounds = Aabb3::new(Vec3::ZERO, Vec3::splat(100.0));
/// let mut tree = Octree::new(bounds, 8, 4);
///
/// tree.insert(Vec3::new(10.0, 20.0, 30.0), 1);
/// tree.insert(Vec3::new(50.0, 50.0, 50.0), 2);
///
/// // Query all points in a region
/// let query = Aabb3::new(Vec3::ZERO, Vec3::splat(30.0));
/// for (pos, data) in tree.query_region(&query) {
///     println!("Found point at {:?} with data {}", pos, data);
/// }
/// ```
#[derive(Debug)]
pub struct Octree<T> {
    root: OctreeNode<T>,
    bounds: Aabb3,
    max_depth: usize,
    max_entries_per_leaf: usize,
}

impl<T> Octree<T> {
    /// Creates a new octree with the given bounds and parameters.
    ///
    /// # Arguments
    ///
    /// * `bounds` - The bounding box of the entire octree.
    /// * `max_depth` - Maximum depth of the tree (prevents infinite subdivision).
    /// * `max_entries_per_leaf` - Maximum entries per leaf before subdivision.
    pub fn new(bounds: Aabb3, max_depth: usize, max_entries_per_leaf: usize) -> Self {
        Self {
            root: OctreeNode::Leaf {
                entries: Vec::new(),
            },
            bounds,
            max_depth,
            max_entries_per_leaf,
        }
    }

    /// Returns the bounds of this octree.
    pub fn bounds(&self) -> Aabb3 {
        self.bounds
    }

    /// Inserts a point with associated data into the octree.
    ///
    /// Returns `true` if the point was inserted, `false` if it's outside bounds.
    pub fn insert(&mut self, position: Vec3, data: T) -> bool {
        if !self.bounds.contains_point(position) {
            return false;
        }

        Self::insert_recursive(
            &mut self.root,
            self.bounds,
            OctreeEntry { position, data },
            0,
            self.max_depth,
            self.max_entries_per_leaf,
        );
        true
    }

    fn insert_recursive(
        node: &mut OctreeNode<T>,
        bounds: Aabb3,
        entry: OctreeEntry<T>,
        depth: usize,
        max_depth: usize,
        max_entries: usize,
    ) {
        match node {
            OctreeNode::Leaf { entries } => {
                entries.push(entry);

                // Subdivide if we exceed capacity and haven't reached max depth
                if entries.len() > max_entries && depth < max_depth {
                    let octants = bounds.octants();
                    let old_entries = std::mem::take(entries);

                    let mut children: [OctreeNode<T>; 8] = [
                        OctreeNode::Leaf {
                            entries: Vec::new(),
                        },
                        OctreeNode::Leaf {
                            entries: Vec::new(),
                        },
                        OctreeNode::Leaf {
                            entries: Vec::new(),
                        },
                        OctreeNode::Leaf {
                            entries: Vec::new(),
                        },
                        OctreeNode::Leaf {
                            entries: Vec::new(),
                        },
                        OctreeNode::Leaf {
                            entries: Vec::new(),
                        },
                        OctreeNode::Leaf {
                            entries: Vec::new(),
                        },
                        OctreeNode::Leaf {
                            entries: Vec::new(),
                        },
                    ];

                    for e in old_entries {
                        for (i, octant) in octants.iter().enumerate() {
                            if octant.contains_point(e.position) {
                                Self::insert_recursive(
                                    &mut children[i],
                                    *octant,
                                    e,
                                    depth + 1,
                                    max_depth,
                                    max_entries,
                                );
                                break;
                            }
                        }
                    }

                    *node = OctreeNode::Internal {
                        children: Box::new(children),
                    };
                }
            }
            OctreeNode::Internal { children } => {
                let octants = bounds.octants();
                for (i, octant) in octants.iter().enumerate() {
                    if octant.contains_point(entry.position) {
                        Self::insert_recursive(
                            &mut children[i],
                            *octant,
                            entry,
                            depth + 1,
                            max_depth,
                            max_entries,
                        );
                        break;
                    }
                }
            }
        }
    }

    /// Queries all points within the given region.
    ///
    /// Returns an iterator over (position, data) pairs.
    pub fn query_region(&self, region: &Aabb3) -> impl Iterator<Item = (Vec3, &T)> {
        let mut results = Vec::new();
        Self::query_recursive(&self.root, self.bounds, region, &mut results);
        results.into_iter()
    }

    fn query_recursive<'a>(
        node: &'a OctreeNode<T>,
        bounds: Aabb3,
        region: &Aabb3,
        results: &mut Vec<(Vec3, &'a T)>,
    ) {
        if !bounds.intersects(region) {
            return;
        }

        match node {
            OctreeNode::Leaf { entries } => {
                for entry in entries {
                    if region.contains_point(entry.position) {
                        results.push((entry.position, &entry.data));
                    }
                }
            }
            OctreeNode::Internal { children } => {
                let octants = bounds.octants();
                for (i, child) in children.iter().enumerate() {
                    Self::query_recursive(child, octants[i], region, results);
                }
            }
        }
    }

    /// Finds the nearest point to the given position.
    ///
    /// Returns `None` if the tree is empty.
    pub fn nearest(&self, position: Vec3) -> Option<(Vec3, &T, f32)> {
        let mut best: Option<(Vec3, &T, f32)> = None;
        Self::nearest_recursive(&self.root, self.bounds, position, &mut best);
        best
    }

    fn nearest_recursive<'a>(
        node: &'a OctreeNode<T>,
        bounds: Aabb3,
        position: Vec3,
        best: &mut Option<(Vec3, &'a T, f32)>,
    ) {
        // Early exit if this node can't possibly contain a closer point
        if let Some((_, _, best_dist)) = best {
            let closest_in_bounds = Vec3::new(
                position.x.clamp(bounds.min.x, bounds.max.x),
                position.y.clamp(bounds.min.y, bounds.max.y),
                position.z.clamp(bounds.min.z, bounds.max.z),
            );
            if closest_in_bounds.distance(position) >= *best_dist {
                return;
            }
        }

        match node {
            OctreeNode::Leaf { entries } => {
                for entry in entries {
                    let dist = entry.position.distance(position);
                    if best.is_none() || dist < best.as_ref().unwrap().2 {
                        *best = Some((entry.position, &entry.data, dist));
                    }
                }
            }
            OctreeNode::Internal { children } => {
                let octants = bounds.octants();

                // Sort children by distance to query point for better pruning
                let mut indices: Vec<usize> = (0..8).collect();
                indices.sort_by(|&a, &b| {
                    let dist_a = octants[a].center().distance(position);
                    let dist_b = octants[b].center().distance(position);
                    dist_a.partial_cmp(&dist_b).unwrap()
                });

                for i in indices {
                    Self::nearest_recursive(&children[i], octants[i], position, best);
                }
            }
        }
    }

    /// Finds the k nearest points to the given position.
    ///
    /// Returns up to k points, sorted by distance (closest first).
    /// Returns fewer than k if the tree contains fewer points.
    ///
    /// # Example
    ///
    /// ```
    /// use rhizome_resin_spatial::{Octree, Aabb3};
    /// use glam::Vec3;
    ///
    /// let bounds = Aabb3::new(Vec3::ZERO, Vec3::splat(100.0));
    /// let mut tree = Octree::new(bounds, 8, 4);
    ///
    /// tree.insert(Vec3::new(10.0, 10.0, 10.0), "A");
    /// tree.insert(Vec3::new(20.0, 20.0, 20.0), "B");
    /// tree.insert(Vec3::new(50.0, 50.0, 50.0), "C");
    ///
    /// let nearest = tree.k_nearest(Vec3::ZERO, 2);
    /// assert_eq!(nearest.len(), 2);
    /// assert_eq!(nearest[0].1, &"A"); // Closest
    /// assert_eq!(nearest[1].1, &"B"); // Second closest
    /// ```
    pub fn k_nearest(&self, position: Vec3, k: usize) -> Vec<(Vec3, &T, f32)> {
        if k == 0 {
            return Vec::new();
        }

        let mut heap: BinaryHeap<KNearestCandidate3D<T>> = BinaryHeap::new();
        Self::k_nearest_recursive(&self.root, self.bounds, position, k, &mut heap);

        // Convert heap to sorted vec (closest first)
        let mut results: Vec<_> = heap
            .into_iter()
            .map(|c| (c.position, c.data, c.distance))
            .collect();
        results.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap());
        results
    }

    fn k_nearest_recursive<'a>(
        node: &'a OctreeNode<T>,
        bounds: Aabb3,
        position: Vec3,
        k: usize,
        heap: &mut BinaryHeap<KNearestCandidate3D<'a, T>>,
    ) {
        // Early exit if this node can't possibly contain a closer point
        if heap.len() >= k {
            let worst_dist = heap.peek().unwrap().distance;
            let closest_in_bounds = Vec3::new(
                position.x.clamp(bounds.min.x, bounds.max.x),
                position.y.clamp(bounds.min.y, bounds.max.y),
                position.z.clamp(bounds.min.z, bounds.max.z),
            );
            if closest_in_bounds.distance(position) >= worst_dist {
                return;
            }
        }

        match node {
            OctreeNode::Leaf { entries } => {
                for entry in entries {
                    let dist = entry.position.distance(position);

                    if heap.len() < k {
                        heap.push(KNearestCandidate3D {
                            position: entry.position,
                            data: &entry.data,
                            distance: dist,
                        });
                    } else if dist < heap.peek().unwrap().distance {
                        heap.pop();
                        heap.push(KNearestCandidate3D {
                            position: entry.position,
                            data: &entry.data,
                            distance: dist,
                        });
                    }
                }
            }
            OctreeNode::Internal { children } => {
                let octants = bounds.octants();

                // Sort children by distance to query point for better pruning
                let mut indices: Vec<usize> = (0..8).collect();
                indices.sort_by(|&a, &b| {
                    let dist_a = octants[a].center().distance(position);
                    let dist_b = octants[b].center().distance(position);
                    dist_a.partial_cmp(&dist_b).unwrap()
                });

                for i in indices {
                    Self::k_nearest_recursive(&children[i], octants[i], position, k, heap);
                }
            }
        }
    }

    /// Returns the total number of entries in the tree.
    pub fn len(&self) -> usize {
        Self::count_recursive(&self.root)
    }

    /// Returns `true` if the tree is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn count_recursive(node: &OctreeNode<T>) -> usize {
        match node {
            OctreeNode::Leaf { entries } => entries.len(),
            OctreeNode::Internal { children } => children.iter().map(Self::count_recursive).sum(),
        }
    }

    /// Clears all entries from the tree.
    pub fn clear(&mut self) {
        self.root = OctreeNode::Leaf {
            entries: Vec::new(),
        };
    }
}

// ============================================================================
// Ray
// ============================================================================

/// A ray in 3D space.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Ray {
    /// Origin of the ray.
    pub origin: Vec3,
    /// Direction of the ray (should be normalized).
    pub direction: Vec3,
}

impl Ray {
    /// Creates a new ray.
    pub fn new(origin: Vec3, direction: Vec3) -> Self {
        Self {
            origin,
            direction: direction.normalize(),
        }
    }

    /// Returns the point at parameter t along the ray.
    pub fn at(&self, t: f32) -> Vec3 {
        self.origin + self.direction * t
    }

    /// Tests intersection with an AABB, returns (t_min, t_max) if hit.
    pub fn intersect_aabb(&self, aabb: &Aabb3) -> Option<(f32, f32)> {
        let inv_dir = Vec3::new(
            1.0 / self.direction.x,
            1.0 / self.direction.y,
            1.0 / self.direction.z,
        );

        let t1 = (aabb.min.x - self.origin.x) * inv_dir.x;
        let t2 = (aabb.max.x - self.origin.x) * inv_dir.x;
        let t3 = (aabb.min.y - self.origin.y) * inv_dir.y;
        let t4 = (aabb.max.y - self.origin.y) * inv_dir.y;
        let t5 = (aabb.min.z - self.origin.z) * inv_dir.z;
        let t6 = (aabb.max.z - self.origin.z) * inv_dir.z;

        let t_min = t1.min(t2).max(t3.min(t4)).max(t5.min(t6));
        let t_max = t1.max(t2).min(t3.max(t4)).min(t5.max(t6));

        if t_max >= t_min && t_max >= 0.0 {
            Some((t_min.max(0.0), t_max))
        } else {
            None
        }
    }
}

// ============================================================================
// BVH (Bounding Volume Hierarchy)
// ============================================================================

/// A node in the BVH.
#[derive(Debug)]
enum BvhNode<T> {
    /// Leaf node containing a single primitive.
    Leaf { bounds: Aabb3, data: T },
    /// Internal node with two children.
    Internal {
        bounds: Aabb3,
        left: Box<BvhNode<T>>,
        right: Box<BvhNode<T>>,
    },
}

/// A Bounding Volume Hierarchy for efficient ray intersection queries.
///
/// BVH builds a binary tree of bounding boxes, enabling O(log n) ray intersection
/// tests. Commonly used for ray tracing and collision detection.
///
/// # Type Parameters
///
/// * `T` - The type of data associated with each primitive.
///
/// # Example
///
/// ```
/// use rhizome_resin_spatial::{Bvh, Aabb3, Ray};
/// use glam::Vec3;
///
/// // Create primitives with their bounding boxes
/// let primitives = vec![
///     (Aabb3::new(Vec3::ZERO, Vec3::ONE), "box1"),
///     (Aabb3::new(Vec3::splat(5.0), Vec3::splat(6.0)), "box2"),
/// ];
///
/// let bvh = Bvh::build(primitives);
///
/// // Cast a ray and find intersections
/// let ray = Ray::new(Vec3::new(0.5, 0.5, -5.0), Vec3::Z);
/// for (aabb, data) in bvh.intersect_ray(&ray) {
///     println!("Hit {:?} with bounds {:?}", data, aabb);
/// }
/// ```
#[derive(Debug)]
pub struct Bvh<T> {
    root: Option<BvhNode<T>>,
}

impl<T> Bvh<T> {
    /// Builds a BVH from a list of (bounding box, data) pairs.
    ///
    /// Uses the Surface Area Heuristic (SAH) for optimal tree construction.
    pub fn build(primitives: Vec<(Aabb3, T)>) -> Self {
        if primitives.is_empty() {
            return Self { root: None };
        }

        Self {
            root: Some(Self::build_recursive(primitives)),
        }
    }

    fn build_recursive(mut primitives: Vec<(Aabb3, T)>) -> BvhNode<T> {
        if primitives.len() == 1 {
            let (bounds, data) = primitives.pop().unwrap();
            return BvhNode::Leaf { bounds, data };
        }

        // Compute overall bounds
        let bounds = primitives
            .iter()
            .fold(primitives[0].0, |acc, (b, _)| acc.union(b));

        // Find the longest axis and split there
        let size = bounds.size();
        let axis = if size.x >= size.y && size.x >= size.z {
            0
        } else if size.y >= size.z {
            1
        } else {
            2
        };

        // Sort by centroid along the chosen axis
        primitives.sort_by(|(a, _), (b, _)| {
            let ca = a.center();
            let cb = b.center();
            let va = match axis {
                0 => ca.x,
                1 => ca.y,
                _ => ca.z,
            };
            let vb = match axis {
                0 => cb.x,
                1 => cb.y,
                _ => cb.z,
            };
            va.partial_cmp(&vb).unwrap()
        });

        // Split in the middle
        let mid = primitives.len() / 2;
        let right_prims = primitives.split_off(mid);

        let left = Box::new(Self::build_recursive(primitives));
        let right = Box::new(Self::build_recursive(right_prims));

        BvhNode::Internal {
            bounds,
            left,
            right,
        }
    }

    /// Tests a ray against the BVH and returns all intersecting primitives.
    ///
    /// Returns an iterator over (bounds, data) pairs for primitives whose
    /// bounding boxes intersect the ray.
    pub fn intersect_ray(&self, ray: &Ray) -> Vec<(&Aabb3, &T)> {
        let mut results = Vec::new();
        if let Some(ref root) = self.root {
            Self::intersect_recursive(root, ray, &mut results);
        }
        results
    }

    fn intersect_recursive<'a>(
        node: &'a BvhNode<T>,
        ray: &Ray,
        results: &mut Vec<(&'a Aabb3, &'a T)>,
    ) {
        match node {
            BvhNode::Leaf { bounds, data } => {
                if ray.intersect_aabb(bounds).is_some() {
                    results.push((bounds, data));
                }
            }
            BvhNode::Internal {
                bounds,
                left,
                right,
            } => {
                if ray.intersect_aabb(bounds).is_some() {
                    Self::intersect_recursive(left, ray, results);
                    Self::intersect_recursive(right, ray, results);
                }
            }
        }
    }

    /// Tests a ray and returns the closest intersection.
    ///
    /// Returns `(bounds, data, t)` where `t` is the parameter along the ray.
    pub fn intersect_ray_closest(&self, ray: &Ray) -> Option<(&Aabb3, &T, f32)> {
        let mut closest: Option<(&Aabb3, &T, f32)> = None;
        if let Some(ref root) = self.root {
            Self::intersect_closest_recursive(root, ray, &mut closest);
        }
        closest
    }

    fn intersect_closest_recursive<'a>(
        node: &'a BvhNode<T>,
        ray: &Ray,
        closest: &mut Option<(&'a Aabb3, &'a T, f32)>,
    ) {
        let bounds = match node {
            BvhNode::Leaf { bounds, .. } => bounds,
            BvhNode::Internal { bounds, .. } => bounds,
        };

        // Early exit if ray doesn't hit this node's bounds
        let Some((t_min, _)) = ray.intersect_aabb(bounds) else {
            return;
        };

        // Early exit if we already have a closer hit
        if let Some((_, _, closest_t)) = closest {
            if t_min > *closest_t {
                return;
            }
        }

        match node {
            BvhNode::Leaf { bounds, data } => {
                if let Some((t_hit, _)) = ray.intersect_aabb(bounds) {
                    if closest.is_none() || t_hit < closest.unwrap().2 {
                        *closest = Some((bounds, data, t_hit));
                    }
                }
            }
            BvhNode::Internal { left, right, .. } => {
                // Recurse into children
                // For better performance, we could check which child to visit first
                Self::intersect_closest_recursive(left, ray, closest);
                Self::intersect_closest_recursive(right, ray, closest);
            }
        }
    }

    /// Queries all primitives whose bounds intersect the given AABB.
    pub fn query_aabb(&self, query: &Aabb3) -> Vec<(&Aabb3, &T)> {
        let mut results = Vec::new();
        if let Some(ref root) = self.root {
            Self::query_aabb_recursive(root, query, &mut results);
        }
        results
    }

    fn query_aabb_recursive<'a>(
        node: &'a BvhNode<T>,
        query: &Aabb3,
        results: &mut Vec<(&'a Aabb3, &'a T)>,
    ) {
        match node {
            BvhNode::Leaf { bounds, data } => {
                if bounds.intersects(query) {
                    results.push((bounds, data));
                }
            }
            BvhNode::Internal {
                bounds,
                left,
                right,
            } => {
                if bounds.intersects(query) {
                    Self::query_aabb_recursive(left, query, results);
                    Self::query_aabb_recursive(right, query, results);
                }
            }
        }
    }

    /// Returns the number of primitives in the BVH.
    pub fn len(&self) -> usize {
        match &self.root {
            None => 0,
            Some(root) => Self::count_recursive(root),
        }
    }

    /// Returns `true` if the BVH is empty.
    pub fn is_empty(&self) -> bool {
        self.root.is_none()
    }

    fn count_recursive(node: &BvhNode<T>) -> usize {
        match node {
            BvhNode::Leaf { .. } => 1,
            BvhNode::Internal { left, right, .. } => {
                Self::count_recursive(left) + Self::count_recursive(right)
            }
        }
    }
}

// ============================================================================
// Spatial Hash
// ============================================================================

/// A spatial hash grid for broad-phase collision detection.
///
/// Divides space into a uniform grid and maps objects to cells based on their
/// positions. Efficient for uniformly distributed objects.
///
/// # Type Parameters
///
/// * `T` - The type of data associated with each entry.
///
/// # Example
///
/// ```
/// use rhizome_resin_spatial::SpatialHash;
/// use glam::Vec3;
///
/// let mut hash = SpatialHash::new(10.0); // 10 unit cell size
///
/// hash.insert(Vec3::new(5.0, 5.0, 5.0), "A");
/// hash.insert(Vec3::new(15.0, 5.0, 5.0), "B");
/// hash.insert(Vec3::new(5.5, 5.5, 5.5), "C"); // Same cell as A
///
/// // Query nearby objects
/// let nearby: Vec<_> = hash.query_cell(Vec3::new(5.0, 5.0, 5.0)).collect();
/// assert_eq!(nearby.len(), 2); // A and C
/// ```
#[derive(Debug)]
pub struct SpatialHash<T> {
    cell_size: f32,
    inv_cell_size: f32,
    cells: std::collections::HashMap<(i32, i32, i32), Vec<SpatialHashEntry<T>>>,
}

#[derive(Debug, Clone)]
struct SpatialHashEntry<T> {
    position: Vec3,
    data: T,
}

impl<T> SpatialHash<T> {
    /// Creates a new spatial hash with the given cell size.
    pub fn new(cell_size: f32) -> Self {
        Self {
            cell_size,
            inv_cell_size: 1.0 / cell_size,
            cells: std::collections::HashMap::new(),
        }
    }

    /// Returns the cell size.
    pub fn cell_size(&self) -> f32 {
        self.cell_size
    }

    fn cell_key(&self, position: Vec3) -> (i32, i32, i32) {
        (
            (position.x * self.inv_cell_size).floor() as i32,
            (position.y * self.inv_cell_size).floor() as i32,
            (position.z * self.inv_cell_size).floor() as i32,
        )
    }

    /// Inserts an entry at the given position.
    pub fn insert(&mut self, position: Vec3, data: T) {
        let key = self.cell_key(position);
        self.cells
            .entry(key)
            .or_default()
            .push(SpatialHashEntry { position, data });
    }

    /// Queries all entries in the same cell as the given position.
    pub fn query_cell(&self, position: Vec3) -> impl Iterator<Item = (Vec3, &T)> {
        let key = self.cell_key(position);
        self.cells
            .get(&key)
            .map(|entries| {
                entries
                    .iter()
                    .map(|e| (e.position, &e.data))
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default()
            .into_iter()
    }

    /// Queries all entries in the cell containing position and all 26 neighboring cells.
    pub fn query_neighbors(&self, position: Vec3) -> impl Iterator<Item = (Vec3, &T)> {
        let (cx, cy, cz) = self.cell_key(position);
        let mut results = Vec::new();

        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    let key = (cx + dx, cy + dy, cz + dz);
                    if let Some(entries) = self.cells.get(&key) {
                        for e in entries {
                            results.push((e.position, &e.data));
                        }
                    }
                }
            }
        }

        results.into_iter()
    }

    /// Queries all entries within the given radius of a position.
    pub fn query_radius(&self, position: Vec3, radius: f32) -> impl Iterator<Item = (Vec3, &T)> {
        let radius_sq = radius * radius;
        let (cx, cy, cz) = self.cell_key(position);
        let cell_radius = (radius * self.inv_cell_size).ceil() as i32;

        let mut results = Vec::new();

        for dx in -cell_radius..=cell_radius {
            for dy in -cell_radius..=cell_radius {
                for dz in -cell_radius..=cell_radius {
                    let key = (cx + dx, cy + dy, cz + dz);
                    if let Some(entries) = self.cells.get(&key) {
                        for e in entries {
                            if e.position.distance_squared(position) <= radius_sq {
                                results.push((e.position, &e.data));
                            }
                        }
                    }
                }
            }
        }

        results.into_iter()
    }

    /// Returns the total number of entries.
    pub fn len(&self) -> usize {
        self.cells.values().map(|v| v.len()).sum()
    }

    /// Returns `true` if empty.
    pub fn is_empty(&self) -> bool {
        self.cells.is_empty()
    }

    /// Clears all entries.
    pub fn clear(&mut self) {
        self.cells.clear();
    }
}

// ============================================================================
// R-tree
// ============================================================================

/// An entry in the R-tree (a rectangle with associated data).
#[derive(Debug, Clone)]
struct RtreeEntry<T> {
    bounds: Aabb2,
    data: T,
}

/// A node in the R-tree.
#[derive(Debug)]
enum RtreeNode<T> {
    /// Leaf node containing entries.
    Leaf {
        bounds: Aabb2,
        entries: Vec<RtreeEntry<T>>,
    },
    /// Internal node with child nodes.
    Internal {
        bounds: Aabb2,
        children: Vec<RtreeNode<T>>,
    },
}

impl<T> RtreeNode<T> {
    fn bounds(&self) -> Aabb2 {
        match self {
            RtreeNode::Leaf { bounds, .. } => *bounds,
            RtreeNode::Internal { bounds, .. } => *bounds,
        }
    }
}

/// An R-tree for 2D rectangle/AABB queries.
///
/// R-trees are balanced search trees that organize spatial data by grouping
/// nearby objects in minimum bounding rectangles. Efficient for range queries
/// and nearest neighbor searches on rectangles.
///
/// # Type Parameters
///
/// * `T` - The type of data associated with each rectangle.
///
/// # Example
///
/// ```
/// use rhizome_resin_spatial::{Rtree, Aabb2};
/// use glam::Vec2;
///
/// let mut tree = Rtree::new(4); // max 4 entries per node
///
/// // Insert rectangles with data
/// tree.insert(Aabb2::new(Vec2::ZERO, Vec2::splat(10.0)), "rect1");
/// tree.insert(Aabb2::new(Vec2::splat(20.0), Vec2::splat(30.0)), "rect2");
///
/// // Query overlapping rectangles
/// let query = Aabb2::new(Vec2::splat(5.0), Vec2::splat(15.0));
/// for (bounds, data) in tree.query(&query) {
///     println!("Found {:?} with bounds {:?}", data, bounds);
/// }
/// ```
#[derive(Debug)]
pub struct Rtree<T> {
    root: Option<RtreeNode<T>>,
    max_entries: usize,
    len: usize,
}

impl<T> Rtree<T> {
    /// Creates a new R-tree with the given maximum entries per node.
    pub fn new(max_entries: usize) -> Self {
        Self {
            root: None,
            max_entries: max_entries.max(2),
            len: 0,
        }
    }

    /// Inserts a rectangle with associated data.
    pub fn insert(&mut self, bounds: Aabb2, data: T) {
        self.len += 1;
        let entry = RtreeEntry { bounds, data };

        match self.root.take() {
            None => {
                self.root = Some(RtreeNode::Leaf {
                    bounds,
                    entries: vec![entry],
                });
            }
            Some(root) => {
                self.root = Some(self.insert_recursive(root, entry));
            }
        }
    }

    fn insert_recursive(&self, mut node: RtreeNode<T>, entry: RtreeEntry<T>) -> RtreeNode<T> {
        match &mut node {
            RtreeNode::Leaf { bounds, entries } => {
                // Expand bounds to include new entry
                *bounds = union_aabb2(bounds, &entry.bounds);
                entries.push(entry);

                // Check if we need to split
                if entries.len() > self.max_entries {
                    self.split_leaf(std::mem::take(entries))
                } else {
                    node
                }
            }
            RtreeNode::Internal { bounds, children } => {
                // Find the child that needs least enlargement
                let best_idx = self.choose_subtree(children, &entry.bounds);

                // Expand bounds
                *bounds = union_aabb2(bounds, &entry.bounds);

                // Remove and reinsert into best child
                let child = children.remove(best_idx);
                let new_child = self.insert_recursive(child, entry);

                // Check if child split into an internal node
                match new_child {
                    RtreeNode::Internal {
                        children: mut split_children,
                        ..
                    } if split_children.len() == 2 => {
                        // Child was split, add both back
                        children.append(&mut split_children);
                    }
                    other => {
                        children.push(other);
                    }
                }

                // Check if we need to split this node
                if children.len() > self.max_entries {
                    self.split_internal(std::mem::take(children))
                } else {
                    // Recalculate bounds
                    let new_bounds = children
                        .iter()
                        .map(|c| c.bounds())
                        .reduce(|a, b| union_aabb2(&a, &b))
                        .unwrap();
                    *bounds = new_bounds;
                    node
                }
            }
        }
    }

    fn choose_subtree(&self, children: &[RtreeNode<T>], bounds: &Aabb2) -> usize {
        children
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| {
                let area_a = enlargement_area(&a.bounds(), bounds);
                let area_b = enlargement_area(&b.bounds(), bounds);
                area_a.partial_cmp(&area_b).unwrap()
            })
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    fn split_leaf(&self, entries: Vec<RtreeEntry<T>>) -> RtreeNode<T> {
        let (left, right) = self.quadratic_split_entries(entries);

        let left_bounds = left
            .iter()
            .map(|e| e.bounds)
            .reduce(|a, b| union_aabb2(&a, &b))
            .unwrap();
        let right_bounds = right
            .iter()
            .map(|e| e.bounds)
            .reduce(|a, b| union_aabb2(&a, &b))
            .unwrap();

        let bounds = union_aabb2(&left_bounds, &right_bounds);

        RtreeNode::Internal {
            bounds,
            children: vec![
                RtreeNode::Leaf {
                    bounds: left_bounds,
                    entries: left,
                },
                RtreeNode::Leaf {
                    bounds: right_bounds,
                    entries: right,
                },
            ],
        }
    }

    fn split_internal(&self, children: Vec<RtreeNode<T>>) -> RtreeNode<T> {
        let (left, right) = self.quadratic_split_nodes(children);

        let left_bounds = left
            .iter()
            .map(|n| n.bounds())
            .reduce(|a, b| union_aabb2(&a, &b))
            .unwrap();
        let right_bounds = right
            .iter()
            .map(|n| n.bounds())
            .reduce(|a, b| union_aabb2(&a, &b))
            .unwrap();

        let bounds = union_aabb2(&left_bounds, &right_bounds);

        RtreeNode::Internal {
            bounds,
            children: vec![
                RtreeNode::Internal {
                    bounds: left_bounds,
                    children: left,
                },
                RtreeNode::Internal {
                    bounds: right_bounds,
                    children: right,
                },
            ],
        }
    }

    fn quadratic_split_entries(
        &self,
        mut entries: Vec<RtreeEntry<T>>,
    ) -> (Vec<RtreeEntry<T>>, Vec<RtreeEntry<T>>) {
        // Find the two entries that would waste most area if grouped together
        let (seed1, seed2) = self.pick_seeds_entries(&entries);

        let entry2 = entries.remove(seed2);
        let entry1 = entries.remove(seed1);

        let mut left = vec![entry1];
        let mut right = vec![entry2];
        let mut left_bounds = left[0].bounds;
        let mut right_bounds = right[0].bounds;

        // Distribute remaining entries
        for entry in entries {
            let enlarge_left = enlargement_area(&left_bounds, &entry.bounds);
            let enlarge_right = enlargement_area(&right_bounds, &entry.bounds);

            if enlarge_left < enlarge_right {
                left_bounds = union_aabb2(&left_bounds, &entry.bounds);
                left.push(entry);
            } else {
                right_bounds = union_aabb2(&right_bounds, &entry.bounds);
                right.push(entry);
            }
        }

        (left, right)
    }

    fn quadratic_split_nodes(
        &self,
        mut nodes: Vec<RtreeNode<T>>,
    ) -> (Vec<RtreeNode<T>>, Vec<RtreeNode<T>>) {
        let (seed1, seed2) = self.pick_seeds_nodes(&nodes);

        let node2 = nodes.remove(seed2);
        let node1 = nodes.remove(seed1);

        let mut left = vec![node1];
        let mut right = vec![node2];
        let mut left_bounds = left[0].bounds();
        let mut right_bounds = right[0].bounds();

        for node in nodes {
            let enlarge_left = enlargement_area(&left_bounds, &node.bounds());
            let enlarge_right = enlargement_area(&right_bounds, &node.bounds());

            if enlarge_left < enlarge_right {
                left_bounds = union_aabb2(&left_bounds, &node.bounds());
                left.push(node);
            } else {
                right_bounds = union_aabb2(&right_bounds, &node.bounds());
                right.push(node);
            }
        }

        (left, right)
    }

    fn pick_seeds_entries(&self, entries: &[RtreeEntry<T>]) -> (usize, usize) {
        let mut max_waste = f32::NEG_INFINITY;
        let mut seeds = (0, 1);

        for i in 0..entries.len() {
            for j in (i + 1)..entries.len() {
                let combined = union_aabb2(&entries[i].bounds, &entries[j].bounds);
                let waste = area(&combined) - area(&entries[i].bounds) - area(&entries[j].bounds);
                if waste > max_waste {
                    max_waste = waste;
                    seeds = (i, j);
                }
            }
        }

        seeds
    }

    fn pick_seeds_nodes(&self, nodes: &[RtreeNode<T>]) -> (usize, usize) {
        let mut max_waste = f32::NEG_INFINITY;
        let mut seeds = (0, 1);

        for i in 0..nodes.len() {
            for j in (i + 1)..nodes.len() {
                let combined = union_aabb2(&nodes[i].bounds(), &nodes[j].bounds());
                let waste = area(&combined) - area(&nodes[i].bounds()) - area(&nodes[j].bounds());
                if waste > max_waste {
                    max_waste = waste;
                    seeds = (i, j);
                }
            }
        }

        seeds
    }

    /// Queries all rectangles that intersect the given region.
    pub fn query(&self, region: &Aabb2) -> Vec<(&Aabb2, &T)> {
        let mut results = Vec::new();
        if let Some(ref root) = self.root {
            Self::query_recursive(root, region, &mut results);
        }
        results
    }

    fn query_recursive<'a>(
        node: &'a RtreeNode<T>,
        region: &Aabb2,
        results: &mut Vec<(&'a Aabb2, &'a T)>,
    ) {
        if !node.bounds().intersects(region) {
            return;
        }

        match node {
            RtreeNode::Leaf { entries, .. } => {
                for entry in entries {
                    if entry.bounds.intersects(region) {
                        results.push((&entry.bounds, &entry.data));
                    }
                }
            }
            RtreeNode::Internal { children, .. } => {
                for child in children {
                    Self::query_recursive(child, region, results);
                }
            }
        }
    }

    /// Queries all rectangles that contain the given point.
    pub fn query_point(&self, point: Vec2) -> Vec<(&Aabb2, &T)> {
        let mut results = Vec::new();
        if let Some(ref root) = self.root {
            Self::query_point_recursive(root, point, &mut results);
        }
        results
    }

    fn query_point_recursive<'a>(
        node: &'a RtreeNode<T>,
        point: Vec2,
        results: &mut Vec<(&'a Aabb2, &'a T)>,
    ) {
        if !node.bounds().contains_point(point) {
            return;
        }

        match node {
            RtreeNode::Leaf { entries, .. } => {
                for entry in entries {
                    if entry.bounds.contains_point(point) {
                        results.push((&entry.bounds, &entry.data));
                    }
                }
            }
            RtreeNode::Internal { children, .. } => {
                for child in children {
                    Self::query_point_recursive(child, point, results);
                }
            }
        }
    }

    /// Returns the number of entries in the R-tree.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if the R-tree is empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

// Helper functions for Aabb2

fn union_aabb2(a: &Aabb2, b: &Aabb2) -> Aabb2 {
    Aabb2::new(a.min.min(b.min), a.max.max(b.max))
}

fn area(aabb: &Aabb2) -> f32 {
    let size = aabb.size();
    size.x * size.y
}

fn enlargement_area(base: &Aabb2, add: &Aabb2) -> f32 {
    let combined = union_aabb2(base, add);
    area(&combined) - area(base)
}

// ============================================================================
// KD-Tree (2D)
// ============================================================================

/// A point with associated data stored in a 2D KD-tree.
#[derive(Debug, Clone)]
struct KdEntry2D<T> {
    position: Vec2,
    data: T,
}

/// A node in the 2D KD-tree.
#[derive(Debug)]
enum KdNode2D<T> {
    /// Leaf node (empty or single point).
    Leaf(Option<KdEntry2D<T>>),
    /// Internal node with split value and children.
    Internal {
        /// The point at this node.
        entry: KdEntry2D<T>,
        /// Split dimension (0 = x, 1 = y).
        axis: usize,
        /// Left child (values < split).
        left: Box<KdNode2D<T>>,
        /// Right child (values >= split).
        right: Box<KdNode2D<T>>,
    },
}

/// A 2D KD-tree for efficient nearest neighbor queries.
///
/// KD-trees partition space by alternating splits along each dimension,
/// making them very efficient for nearest neighbor searches in low dimensions.
///
/// # Type Parameters
///
/// * `T` - The type of data associated with each point.
///
/// # Example
///
/// ```
/// use rhizome_resin_spatial::KdTree2D;
/// use glam::Vec2;
///
/// let points = vec![
///     (Vec2::new(10.0, 10.0), "A"),
///     (Vec2::new(20.0, 20.0), "B"),
///     (Vec2::new(50.0, 50.0), "C"),
/// ];
/// let tree = KdTree2D::build(points);
///
/// // Find nearest neighbor
/// let (pos, data, dist) = tree.nearest(Vec2::new(12.0, 12.0)).unwrap();
/// assert_eq!(*data, "A");
/// ```
#[derive(Debug)]
pub struct KdTree2D<T> {
    root: KdNode2D<T>,
    len: usize,
}

impl<T> KdTree2D<T> {
    /// Builds a KD-tree from a list of points.
    ///
    /// The tree is balanced by using median splits, giving O(log n) query time.
    pub fn build(points: Vec<(Vec2, T)>) -> Self {
        let len = points.len();
        let mut entries: Vec<KdEntry2D<T>> = points
            .into_iter()
            .map(|(position, data)| KdEntry2D { position, data })
            .collect();

        let root = Self::build_recursive(&mut entries, 0);
        Self { root, len }
    }

    fn build_recursive(entries: &mut [KdEntry2D<T>], depth: usize) -> KdNode2D<T> {
        if entries.is_empty() {
            return KdNode2D::Leaf(None);
        }
        if entries.len() == 1 {
            // Safety: we know there's exactly one element
            let entry = unsafe { std::ptr::read(&entries[0]) };
            return KdNode2D::Leaf(Some(entry));
        }

        let axis = depth % 2;

        // Sort by the current axis
        entries.sort_by(|a, b| {
            let va = if axis == 0 {
                a.position.x
            } else {
                a.position.y
            };
            let vb = if axis == 0 {
                b.position.x
            } else {
                b.position.y
            };
            va.partial_cmp(&vb).unwrap_or(std::cmp::Ordering::Equal)
        });

        let median = entries.len() / 2;

        // Split the array
        let (left_slice, right_slice) = entries.split_at_mut(median);
        let (median_entry, right_slice) = right_slice.split_first_mut().unwrap();

        // Safety: we're consuming the original slice
        let entry = unsafe { std::ptr::read(median_entry) };

        let left = Box::new(Self::build_recursive(left_slice, depth + 1));
        let right = Box::new(Self::build_recursive(right_slice, depth + 1));

        KdNode2D::Internal {
            entry,
            axis,
            left,
            right,
        }
    }

    /// Returns the number of points in the tree.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if the tree is empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Finds the nearest point to the given position.
    ///
    /// Returns `None` if the tree is empty.
    pub fn nearest(&self, position: Vec2) -> Option<(Vec2, &T, f32)> {
        let mut best: Option<(Vec2, &T, f32)> = None;
        Self::nearest_recursive(&self.root, position, 0, &mut best);
        best
    }

    fn nearest_recursive<'a>(
        node: &'a KdNode2D<T>,
        position: Vec2,
        depth: usize,
        best: &mut Option<(Vec2, &'a T, f32)>,
    ) {
        match node {
            KdNode2D::Leaf(None) => {}
            KdNode2D::Leaf(Some(entry)) => {
                let dist = entry.position.distance(position);
                if best.is_none() || dist < best.as_ref().unwrap().2 {
                    *best = Some((entry.position, &entry.data, dist));
                }
            }
            KdNode2D::Internal {
                entry,
                axis,
                left,
                right,
            } => {
                // Check current node
                let dist = entry.position.distance(position);
                if best.is_none() || dist < best.as_ref().unwrap().2 {
                    *best = Some((entry.position, &entry.data, dist));
                }

                let axis_val = if *axis == 0 { position.x } else { position.y };
                let split_val = if *axis == 0 {
                    entry.position.x
                } else {
                    entry.position.y
                };

                // Search the side that contains the query point first
                let (first, second) = if axis_val < split_val {
                    (left.as_ref(), right.as_ref())
                } else {
                    (right.as_ref(), left.as_ref())
                };

                Self::nearest_recursive(first, position, depth + 1, best);

                // Only search the other side if it could contain a closer point
                let axis_dist = (axis_val - split_val).abs();
                if best.is_none() || axis_dist < best.as_ref().unwrap().2 {
                    Self::nearest_recursive(second, position, depth + 1, best);
                }
            }
        }
    }

    /// Finds the k nearest points to the given position.
    ///
    /// Returns up to k points, sorted by distance (closest first).
    pub fn k_nearest(&self, position: Vec2, k: usize) -> Vec<(Vec2, &T, f32)> {
        if k == 0 {
            return Vec::new();
        }

        let mut heap: BinaryHeap<KNearestCandidate2D<T>> = BinaryHeap::new();
        Self::k_nearest_recursive(&self.root, position, 0, k, &mut heap);

        let mut results: Vec<_> = heap
            .into_iter()
            .map(|c| (c.position, c.data, c.distance))
            .collect();
        results.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap());
        results
    }

    fn k_nearest_recursive<'a>(
        node: &'a KdNode2D<T>,
        position: Vec2,
        depth: usize,
        k: usize,
        heap: &mut BinaryHeap<KNearestCandidate2D<'a, T>>,
    ) {
        match node {
            KdNode2D::Leaf(None) => {}
            KdNode2D::Leaf(Some(entry)) => {
                let dist = entry.position.distance(position);
                if heap.len() < k {
                    heap.push(KNearestCandidate2D {
                        position: entry.position,
                        data: &entry.data,
                        distance: dist,
                    });
                } else if dist < heap.peek().unwrap().distance {
                    heap.pop();
                    heap.push(KNearestCandidate2D {
                        position: entry.position,
                        data: &entry.data,
                        distance: dist,
                    });
                }
            }
            KdNode2D::Internal {
                entry,
                axis,
                left,
                right,
            } => {
                // Check current node
                let dist = entry.position.distance(position);
                if heap.len() < k {
                    heap.push(KNearestCandidate2D {
                        position: entry.position,
                        data: &entry.data,
                        distance: dist,
                    });
                } else if dist < heap.peek().unwrap().distance {
                    heap.pop();
                    heap.push(KNearestCandidate2D {
                        position: entry.position,
                        data: &entry.data,
                        distance: dist,
                    });
                }

                let axis_val = if *axis == 0 { position.x } else { position.y };
                let split_val = if *axis == 0 {
                    entry.position.x
                } else {
                    entry.position.y
                };

                let (first, second) = if axis_val < split_val {
                    (left.as_ref(), right.as_ref())
                } else {
                    (right.as_ref(), left.as_ref())
                };

                Self::k_nearest_recursive(first, position, depth + 1, k, heap);

                let axis_dist = (axis_val - split_val).abs();
                if heap.len() < k || axis_dist < heap.peek().unwrap().distance {
                    Self::k_nearest_recursive(second, position, depth + 1, k, heap);
                }
            }
        }
    }

    /// Queries all points within the given region.
    pub fn query_region(&self, region: &Aabb2) -> Vec<(Vec2, &T)> {
        let mut results = Vec::new();
        Self::query_region_recursive(&self.root, region, 0, &mut results);
        results
    }

    fn query_region_recursive<'a>(
        node: &'a KdNode2D<T>,
        region: &Aabb2,
        depth: usize,
        results: &mut Vec<(Vec2, &'a T)>,
    ) {
        match node {
            KdNode2D::Leaf(None) => {}
            KdNode2D::Leaf(Some(entry)) => {
                if region.contains_point(entry.position) {
                    results.push((entry.position, &entry.data));
                }
            }
            KdNode2D::Internal {
                entry,
                axis,
                left,
                right,
            } => {
                if region.contains_point(entry.position) {
                    results.push((entry.position, &entry.data));
                }

                let split_val = if *axis == 0 {
                    entry.position.x
                } else {
                    entry.position.y
                };
                let region_min = if *axis == 0 {
                    region.min.x
                } else {
                    region.min.y
                };
                let region_max = if *axis == 0 {
                    region.max.x
                } else {
                    region.max.y
                };

                // Search left if region overlaps
                if region_min < split_val {
                    Self::query_region_recursive(left, region, depth + 1, results);
                }
                // Search right if region overlaps
                if region_max >= split_val {
                    Self::query_region_recursive(right, region, depth + 1, results);
                }
            }
        }
    }

    /// Queries all points within a given radius of the center.
    pub fn query_radius(&self, center: Vec2, radius: f32) -> Vec<(Vec2, &T, f32)> {
        let mut results = Vec::new();
        Self::query_radius_recursive(&self.root, center, radius, 0, &mut results);
        results
    }

    fn query_radius_recursive<'a>(
        node: &'a KdNode2D<T>,
        center: Vec2,
        radius: f32,
        depth: usize,
        results: &mut Vec<(Vec2, &'a T, f32)>,
    ) {
        match node {
            KdNode2D::Leaf(None) => {}
            KdNode2D::Leaf(Some(entry)) => {
                let dist = entry.position.distance(center);
                if dist <= radius {
                    results.push((entry.position, &entry.data, dist));
                }
            }
            KdNode2D::Internal {
                entry,
                axis,
                left,
                right,
            } => {
                let dist = entry.position.distance(center);
                if dist <= radius {
                    results.push((entry.position, &entry.data, dist));
                }

                let axis_val = if *axis == 0 { center.x } else { center.y };
                let split_val = if *axis == 0 {
                    entry.position.x
                } else {
                    entry.position.y
                };

                // Search left if it could contain points within radius
                if axis_val - radius < split_val {
                    Self::query_radius_recursive(left, center, radius, depth + 1, results);
                }
                // Search right if it could contain points within radius
                if axis_val + radius >= split_val {
                    Self::query_radius_recursive(right, center, radius, depth + 1, results);
                }
            }
        }
    }
}

// ============================================================================
// KD-Tree (3D)
// ============================================================================

/// A point with associated data stored in a 3D KD-tree.
#[derive(Debug, Clone)]
struct KdEntry3D<T> {
    position: Vec3,
    data: T,
}

/// A node in the 3D KD-tree.
#[derive(Debug)]
enum KdNode3D<T> {
    /// Leaf node (empty or single point).
    Leaf(Option<KdEntry3D<T>>),
    /// Internal node with split value and children.
    Internal {
        /// The point at this node.
        entry: KdEntry3D<T>,
        /// Split dimension (0 = x, 1 = y, 2 = z).
        axis: usize,
        /// Left child (values < split).
        left: Box<KdNode3D<T>>,
        /// Right child (values >= split).
        right: Box<KdNode3D<T>>,
    },
}

/// A 3D KD-tree for efficient nearest neighbor queries.
///
/// KD-trees partition space by alternating splits along each dimension,
/// making them very efficient for nearest neighbor searches in low dimensions.
///
/// # Type Parameters
///
/// * `T` - The type of data associated with each point.
///
/// # Example
///
/// ```
/// use rhizome_resin_spatial::KdTree3D;
/// use glam::Vec3;
///
/// let points = vec![
///     (Vec3::new(10.0, 10.0, 10.0), "A"),
///     (Vec3::new(20.0, 20.0, 20.0), "B"),
///     (Vec3::new(50.0, 50.0, 50.0), "C"),
/// ];
/// let tree = KdTree3D::build(points);
///
/// // Find nearest neighbor
/// let (pos, data, dist) = tree.nearest(Vec3::new(12.0, 12.0, 12.0)).unwrap();
/// assert_eq!(*data, "A");
/// ```
#[derive(Debug)]
pub struct KdTree3D<T> {
    root: KdNode3D<T>,
    len: usize,
}

impl<T> KdTree3D<T> {
    /// Builds a KD-tree from a list of points.
    ///
    /// The tree is balanced by using median splits, giving O(log n) query time.
    pub fn build(points: Vec<(Vec3, T)>) -> Self {
        let len = points.len();
        let mut entries: Vec<KdEntry3D<T>> = points
            .into_iter()
            .map(|(position, data)| KdEntry3D { position, data })
            .collect();

        let root = Self::build_recursive(&mut entries, 0);
        Self { root, len }
    }

    fn build_recursive(entries: &mut [KdEntry3D<T>], depth: usize) -> KdNode3D<T> {
        if entries.is_empty() {
            return KdNode3D::Leaf(None);
        }
        if entries.len() == 1 {
            let entry = unsafe { std::ptr::read(&entries[0]) };
            return KdNode3D::Leaf(Some(entry));
        }

        let axis = depth % 3;

        // Sort by the current axis
        entries.sort_by(|a, b| {
            let va = match axis {
                0 => a.position.x,
                1 => a.position.y,
                _ => a.position.z,
            };
            let vb = match axis {
                0 => b.position.x,
                1 => b.position.y,
                _ => b.position.z,
            };
            va.partial_cmp(&vb).unwrap_or(std::cmp::Ordering::Equal)
        });

        let median = entries.len() / 2;
        let (left_slice, right_slice) = entries.split_at_mut(median);
        let (median_entry, right_slice) = right_slice.split_first_mut().unwrap();

        let entry = unsafe { std::ptr::read(median_entry) };

        let left = Box::new(Self::build_recursive(left_slice, depth + 1));
        let right = Box::new(Self::build_recursive(right_slice, depth + 1));

        KdNode3D::Internal {
            entry,
            axis,
            left,
            right,
        }
    }

    /// Returns the number of points in the tree.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if the tree is empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Finds the nearest point to the given position.
    ///
    /// Returns `None` if the tree is empty.
    pub fn nearest(&self, position: Vec3) -> Option<(Vec3, &T, f32)> {
        let mut best: Option<(Vec3, &T, f32)> = None;
        Self::nearest_recursive(&self.root, position, 0, &mut best);
        best
    }

    fn nearest_recursive<'a>(
        node: &'a KdNode3D<T>,
        position: Vec3,
        depth: usize,
        best: &mut Option<(Vec3, &'a T, f32)>,
    ) {
        match node {
            KdNode3D::Leaf(None) => {}
            KdNode3D::Leaf(Some(entry)) => {
                let dist = entry.position.distance(position);
                if best.is_none() || dist < best.as_ref().unwrap().2 {
                    *best = Some((entry.position, &entry.data, dist));
                }
            }
            KdNode3D::Internal {
                entry,
                axis,
                left,
                right,
            } => {
                let dist = entry.position.distance(position);
                if best.is_none() || dist < best.as_ref().unwrap().2 {
                    *best = Some((entry.position, &entry.data, dist));
                }

                let axis_val = match *axis {
                    0 => position.x,
                    1 => position.y,
                    _ => position.z,
                };
                let split_val = match *axis {
                    0 => entry.position.x,
                    1 => entry.position.y,
                    _ => entry.position.z,
                };

                let (first, second) = if axis_val < split_val {
                    (left.as_ref(), right.as_ref())
                } else {
                    (right.as_ref(), left.as_ref())
                };

                Self::nearest_recursive(first, position, depth + 1, best);

                let axis_dist = (axis_val - split_val).abs();
                if best.is_none() || axis_dist < best.as_ref().unwrap().2 {
                    Self::nearest_recursive(second, position, depth + 1, best);
                }
            }
        }
    }

    /// Finds the k nearest points to the given position.
    ///
    /// Returns up to k points, sorted by distance (closest first).
    pub fn k_nearest(&self, position: Vec3, k: usize) -> Vec<(Vec3, &T, f32)> {
        if k == 0 {
            return Vec::new();
        }

        let mut heap: BinaryHeap<KNearestCandidate3D<T>> = BinaryHeap::new();
        Self::k_nearest_recursive(&self.root, position, 0, k, &mut heap);

        let mut results: Vec<_> = heap
            .into_iter()
            .map(|c| (c.position, c.data, c.distance))
            .collect();
        results.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap());
        results
    }

    fn k_nearest_recursive<'a>(
        node: &'a KdNode3D<T>,
        position: Vec3,
        depth: usize,
        k: usize,
        heap: &mut BinaryHeap<KNearestCandidate3D<'a, T>>,
    ) {
        match node {
            KdNode3D::Leaf(None) => {}
            KdNode3D::Leaf(Some(entry)) => {
                let dist = entry.position.distance(position);
                if heap.len() < k {
                    heap.push(KNearestCandidate3D {
                        position: entry.position,
                        data: &entry.data,
                        distance: dist,
                    });
                } else if dist < heap.peek().unwrap().distance {
                    heap.pop();
                    heap.push(KNearestCandidate3D {
                        position: entry.position,
                        data: &entry.data,
                        distance: dist,
                    });
                }
            }
            KdNode3D::Internal {
                entry,
                axis,
                left,
                right,
            } => {
                let dist = entry.position.distance(position);
                if heap.len() < k {
                    heap.push(KNearestCandidate3D {
                        position: entry.position,
                        data: &entry.data,
                        distance: dist,
                    });
                } else if dist < heap.peek().unwrap().distance {
                    heap.pop();
                    heap.push(KNearestCandidate3D {
                        position: entry.position,
                        data: &entry.data,
                        distance: dist,
                    });
                }

                let axis_val = match *axis {
                    0 => position.x,
                    1 => position.y,
                    _ => position.z,
                };
                let split_val = match *axis {
                    0 => entry.position.x,
                    1 => entry.position.y,
                    _ => entry.position.z,
                };

                let (first, second) = if axis_val < split_val {
                    (left.as_ref(), right.as_ref())
                } else {
                    (right.as_ref(), left.as_ref())
                };

                Self::k_nearest_recursive(first, position, depth + 1, k, heap);

                let axis_dist = (axis_val - split_val).abs();
                if heap.len() < k || axis_dist < heap.peek().unwrap().distance {
                    Self::k_nearest_recursive(second, position, depth + 1, k, heap);
                }
            }
        }
    }

    /// Queries all points within the given region.
    pub fn query_region(&self, region: &Aabb3) -> Vec<(Vec3, &T)> {
        let mut results = Vec::new();
        Self::query_region_recursive(&self.root, region, 0, &mut results);
        results
    }

    fn query_region_recursive<'a>(
        node: &'a KdNode3D<T>,
        region: &Aabb3,
        depth: usize,
        results: &mut Vec<(Vec3, &'a T)>,
    ) {
        match node {
            KdNode3D::Leaf(None) => {}
            KdNode3D::Leaf(Some(entry)) => {
                if region.contains_point(entry.position) {
                    results.push((entry.position, &entry.data));
                }
            }
            KdNode3D::Internal {
                entry,
                axis,
                left,
                right,
            } => {
                if region.contains_point(entry.position) {
                    results.push((entry.position, &entry.data));
                }

                let split_val = match *axis {
                    0 => entry.position.x,
                    1 => entry.position.y,
                    _ => entry.position.z,
                };
                let region_min = match *axis {
                    0 => region.min.x,
                    1 => region.min.y,
                    _ => region.min.z,
                };
                let region_max = match *axis {
                    0 => region.max.x,
                    1 => region.max.y,
                    _ => region.max.z,
                };

                if region_min < split_val {
                    Self::query_region_recursive(left, region, depth + 1, results);
                }
                if region_max >= split_val {
                    Self::query_region_recursive(right, region, depth + 1, results);
                }
            }
        }
    }

    /// Queries all points within a given radius of the center.
    pub fn query_radius(&self, center: Vec3, radius: f32) -> Vec<(Vec3, &T, f32)> {
        let mut results = Vec::new();
        Self::query_radius_recursive(&self.root, center, radius, 0, &mut results);
        results
    }

    fn query_radius_recursive<'a>(
        node: &'a KdNode3D<T>,
        center: Vec3,
        radius: f32,
        depth: usize,
        results: &mut Vec<(Vec3, &'a T, f32)>,
    ) {
        match node {
            KdNode3D::Leaf(None) => {}
            KdNode3D::Leaf(Some(entry)) => {
                let dist = entry.position.distance(center);
                if dist <= radius {
                    results.push((entry.position, &entry.data, dist));
                }
            }
            KdNode3D::Internal {
                entry,
                axis,
                left,
                right,
            } => {
                let dist = entry.position.distance(center);
                if dist <= radius {
                    results.push((entry.position, &entry.data, dist));
                }

                let axis_val = match *axis {
                    0 => center.x,
                    1 => center.y,
                    _ => center.z,
                };
                let split_val = match *axis {
                    0 => entry.position.x,
                    1 => entry.position.y,
                    _ => entry.position.z,
                };

                if axis_val - radius < split_val {
                    Self::query_radius_recursive(left, center, radius, depth + 1, results);
                }
                if axis_val + radius >= split_val {
                    Self::query_radius_recursive(right, center, radius, depth + 1, results);
                }
            }
        }
    }
}

// ============================================================================
// Ball Tree (2D)
// ============================================================================

/// A point with associated data stored in a 2D Ball tree.
#[derive(Debug, Clone)]
struct BallEntry2D<T> {
    position: Vec2,
    data: T,
}

/// A node in the 2D Ball tree.
#[derive(Debug)]
enum BallNode2D<T> {
    /// Leaf node containing a single point.
    Leaf { entry: BallEntry2D<T> },
    /// Internal node containing a bounding ball and two children.
    Internal {
        center: Vec2,
        radius: f32,
        left: Box<BallNode2D<T>>,
        right: Box<BallNode2D<T>>,
    },
    /// Empty node.
    Empty,
}

/// A 2D Ball tree for efficient nearest neighbor queries.
///
/// Ball trees partition space using nested hyperspheres, which can be more
/// efficient than KD-trees for certain distance metrics and distributions.
/// Each node stores a bounding ball that contains all points in its subtree.
///
/// # Type Parameters
///
/// * `T` - The type of data associated with each point.
///
/// # Example
///
/// ```
/// use rhizome_resin_spatial::BallTree2D;
/// use glam::Vec2;
///
/// let points = vec![
///     (Vec2::new(10.0, 10.0), "A"),
///     (Vec2::new(20.0, 20.0), "B"),
///     (Vec2::new(50.0, 50.0), "C"),
/// ];
/// let tree = BallTree2D::build(points);
///
/// // Find nearest neighbor
/// let (pos, data, dist) = tree.nearest(Vec2::new(12.0, 12.0)).unwrap();
/// assert_eq!(*data, "A");
/// ```
#[derive(Debug)]
pub struct BallTree2D<T> {
    root: BallNode2D<T>,
    len: usize,
}

impl<T> BallTree2D<T> {
    /// Builds a Ball tree from a list of points.
    pub fn build(points: Vec<(Vec2, T)>) -> Self {
        let len = points.len();
        let mut entries: Vec<BallEntry2D<T>> = points
            .into_iter()
            .map(|(position, data)| BallEntry2D { position, data })
            .collect();

        let root = Self::build_recursive(&mut entries);
        Self { root, len }
    }

    fn build_recursive(entries: &mut [BallEntry2D<T>]) -> BallNode2D<T> {
        if entries.is_empty() {
            return BallNode2D::Empty;
        }
        if entries.len() == 1 {
            let entry = unsafe { std::ptr::read(&entries[0]) };
            return BallNode2D::Leaf { entry };
        }

        // Calculate bounding ball
        let (center, radius) = Self::compute_bounding_ball(entries);

        // Find the dimension with greatest spread
        let (mut min_x, mut max_x) = (f32::INFINITY, f32::NEG_INFINITY);
        let (mut min_y, mut max_y) = (f32::INFINITY, f32::NEG_INFINITY);

        for entry in entries.iter() {
            min_x = min_x.min(entry.position.x);
            max_x = max_x.max(entry.position.x);
            min_y = min_y.min(entry.position.y);
            max_y = max_y.max(entry.position.y);
        }

        let spread_x = max_x - min_x;
        let spread_y = max_y - min_y;

        // Sort by the dimension with greatest spread
        if spread_x >= spread_y {
            entries.sort_by(|a, b| a.position.x.partial_cmp(&b.position.x).unwrap());
        } else {
            entries.sort_by(|a, b| a.position.y.partial_cmp(&b.position.y).unwrap());
        }

        // Split at median
        let median = entries.len() / 2;
        let (left_slice, right_slice) = entries.split_at_mut(median);

        let left = Box::new(Self::build_recursive(left_slice));
        let right = Box::new(Self::build_recursive(right_slice));

        BallNode2D::Internal {
            center,
            radius,
            left,
            right,
        }
    }

    fn compute_bounding_ball(entries: &[BallEntry2D<T>]) -> (Vec2, f32) {
        // Simple approach: center is centroid, radius is max distance from center
        let mut center = Vec2::ZERO;
        for entry in entries {
            center += entry.position;
        }
        center /= entries.len() as f32;

        let mut radius = 0.0f32;
        for entry in entries {
            radius = radius.max(entry.position.distance(center));
        }

        (center, radius)
    }

    /// Returns the number of points in the tree.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if the tree is empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Finds the nearest point to the given position.
    ///
    /// Returns `None` if the tree is empty.
    pub fn nearest(&self, position: Vec2) -> Option<(Vec2, &T, f32)> {
        let mut best: Option<(Vec2, &T, f32)> = None;
        Self::nearest_recursive(&self.root, position, &mut best);
        best
    }

    fn nearest_recursive<'a>(
        node: &'a BallNode2D<T>,
        position: Vec2,
        best: &mut Option<(Vec2, &'a T, f32)>,
    ) {
        match node {
            BallNode2D::Empty => {}
            BallNode2D::Leaf { entry, .. } => {
                let dist = entry.position.distance(position);
                if best.is_none() || dist < best.as_ref().unwrap().2 {
                    *best = Some((entry.position, &entry.data, dist));
                }
            }
            BallNode2D::Internal {
                center,
                radius,
                left,
                right,
            } => {
                // Early exit: if the ball can't contain a closer point, skip
                let dist_to_center = position.distance(*center);
                let min_possible_dist = (dist_to_center - radius).max(0.0);

                if let Some((_, _, best_dist)) = best {
                    if min_possible_dist >= *best_dist {
                        return;
                    }
                }

                // Visit closer child first
                let left_center = Self::get_center(left);
                let right_center = Self::get_center(right);

                let (first, second) = match (left_center, right_center) {
                    (Some(lc), Some(rc)) => {
                        if position.distance(lc) < position.distance(rc) {
                            (left.as_ref(), right.as_ref())
                        } else {
                            (right.as_ref(), left.as_ref())
                        }
                    }
                    (Some(_), None) => (left.as_ref(), right.as_ref()),
                    (None, Some(_)) => (right.as_ref(), left.as_ref()),
                    (None, None) => return,
                };

                Self::nearest_recursive(first, position, best);
                Self::nearest_recursive(second, position, best);
            }
        }
    }

    fn get_center(node: &BallNode2D<T>) -> Option<Vec2> {
        match node {
            BallNode2D::Empty => None,
            BallNode2D::Leaf { entry } => Some(entry.position),
            BallNode2D::Internal { center, .. } => Some(*center),
        }
    }

    /// Finds the k nearest points to the given position.
    ///
    /// Returns up to k points, sorted by distance (closest first).
    pub fn k_nearest(&self, position: Vec2, k: usize) -> Vec<(Vec2, &T, f32)> {
        if k == 0 {
            return Vec::new();
        }

        let mut heap: BinaryHeap<KNearestCandidate2D<T>> = BinaryHeap::new();
        Self::k_nearest_recursive(&self.root, position, k, &mut heap);

        let mut results: Vec<_> = heap
            .into_iter()
            .map(|c| (c.position, c.data, c.distance))
            .collect();
        results.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap());
        results
    }

    fn k_nearest_recursive<'a>(
        node: &'a BallNode2D<T>,
        position: Vec2,
        k: usize,
        heap: &mut BinaryHeap<KNearestCandidate2D<'a, T>>,
    ) {
        match node {
            BallNode2D::Empty => {}
            BallNode2D::Leaf { entry, .. } => {
                let dist = entry.position.distance(position);
                if heap.len() < k {
                    heap.push(KNearestCandidate2D {
                        position: entry.position,
                        data: &entry.data,
                        distance: dist,
                    });
                } else if dist < heap.peek().unwrap().distance {
                    heap.pop();
                    heap.push(KNearestCandidate2D {
                        position: entry.position,
                        data: &entry.data,
                        distance: dist,
                    });
                }
            }
            BallNode2D::Internal {
                center,
                radius,
                left,
                right,
            } => {
                let dist_to_center = position.distance(*center);
                let min_possible_dist = (dist_to_center - radius).max(0.0);

                if heap.len() >= k && min_possible_dist >= heap.peek().unwrap().distance {
                    return;
                }

                // Visit closer child first
                let left_center = Self::get_center(left);
                let right_center = Self::get_center(right);

                let (first, second) = match (left_center, right_center) {
                    (Some(lc), Some(rc)) => {
                        if position.distance(lc) < position.distance(rc) {
                            (left.as_ref(), right.as_ref())
                        } else {
                            (right.as_ref(), left.as_ref())
                        }
                    }
                    (Some(_), None) => (left.as_ref(), right.as_ref()),
                    (None, Some(_)) => (right.as_ref(), left.as_ref()),
                    (None, None) => return,
                };

                Self::k_nearest_recursive(first, position, k, heap);
                Self::k_nearest_recursive(second, position, k, heap);
            }
        }
    }

    /// Queries all points within a given radius of the center.
    pub fn query_radius(&self, center: Vec2, radius: f32) -> Vec<(Vec2, &T, f32)> {
        let mut results = Vec::new();
        Self::query_radius_recursive(&self.root, center, radius, &mut results);
        results
    }

    fn query_radius_recursive<'a>(
        node: &'a BallNode2D<T>,
        center: Vec2,
        radius: f32,
        results: &mut Vec<(Vec2, &'a T, f32)>,
    ) {
        match node {
            BallNode2D::Empty => {}
            BallNode2D::Leaf { entry, .. } => {
                let dist = entry.position.distance(center);
                if dist <= radius {
                    results.push((entry.position, &entry.data, dist));
                }
            }
            BallNode2D::Internal {
                center: node_center,
                radius: node_radius,
                left,
                right,
            } => {
                // Skip if query ball doesn't intersect node ball
                let dist_to_center = center.distance(*node_center);
                if dist_to_center > radius + node_radius {
                    return;
                }

                Self::query_radius_recursive(left, center, radius, results);
                Self::query_radius_recursive(right, center, radius, results);
            }
        }
    }
}

// ============================================================================
// Ball Tree (3D)
// ============================================================================

/// A point with associated data stored in a 3D Ball tree.
#[derive(Debug, Clone)]
struct BallEntry3D<T> {
    position: Vec3,
    data: T,
}

/// A node in the 3D Ball tree.
#[derive(Debug)]
enum BallNode3D<T> {
    /// Leaf node containing a single point.
    Leaf { entry: BallEntry3D<T> },
    /// Internal node containing a bounding ball and two children.
    Internal {
        center: Vec3,
        radius: f32,
        left: Box<BallNode3D<T>>,
        right: Box<BallNode3D<T>>,
    },
    /// Empty node.
    Empty,
}

/// A 3D Ball tree for efficient nearest neighbor queries.
///
/// Ball trees partition space using nested hyperspheres, which can be more
/// efficient than KD-trees for certain distance metrics and distributions.
///
/// # Type Parameters
///
/// * `T` - The type of data associated with each point.
///
/// # Example
///
/// ```
/// use rhizome_resin_spatial::BallTree3D;
/// use glam::Vec3;
///
/// let points = vec![
///     (Vec3::new(10.0, 10.0, 10.0), "A"),
///     (Vec3::new(20.0, 20.0, 20.0), "B"),
///     (Vec3::new(50.0, 50.0, 50.0), "C"),
/// ];
/// let tree = BallTree3D::build(points);
///
/// // Find nearest neighbor
/// let (pos, data, dist) = tree.nearest(Vec3::new(12.0, 12.0, 12.0)).unwrap();
/// assert_eq!(*data, "A");
/// ```
#[derive(Debug)]
pub struct BallTree3D<T> {
    root: BallNode3D<T>,
    len: usize,
}

impl<T> BallTree3D<T> {
    /// Builds a Ball tree from a list of points.
    pub fn build(points: Vec<(Vec3, T)>) -> Self {
        let len = points.len();
        let mut entries: Vec<BallEntry3D<T>> = points
            .into_iter()
            .map(|(position, data)| BallEntry3D { position, data })
            .collect();

        let root = Self::build_recursive(&mut entries);
        Self { root, len }
    }

    fn build_recursive(entries: &mut [BallEntry3D<T>]) -> BallNode3D<T> {
        if entries.is_empty() {
            return BallNode3D::Empty;
        }
        if entries.len() == 1 {
            let entry = unsafe { std::ptr::read(&entries[0]) };
            return BallNode3D::Leaf { entry };
        }

        // Calculate bounding ball
        let (center, radius) = Self::compute_bounding_ball(entries);

        // Find the dimension with greatest spread
        let (mut min_x, mut max_x) = (f32::INFINITY, f32::NEG_INFINITY);
        let (mut min_y, mut max_y) = (f32::INFINITY, f32::NEG_INFINITY);
        let (mut min_z, mut max_z) = (f32::INFINITY, f32::NEG_INFINITY);

        for entry in entries.iter() {
            min_x = min_x.min(entry.position.x);
            max_x = max_x.max(entry.position.x);
            min_y = min_y.min(entry.position.y);
            max_y = max_y.max(entry.position.y);
            min_z = min_z.min(entry.position.z);
            max_z = max_z.max(entry.position.z);
        }

        let spread_x = max_x - min_x;
        let spread_y = max_y - min_y;
        let spread_z = max_z - min_z;

        // Sort by the dimension with greatest spread
        if spread_x >= spread_y && spread_x >= spread_z {
            entries.sort_by(|a, b| a.position.x.partial_cmp(&b.position.x).unwrap());
        } else if spread_y >= spread_z {
            entries.sort_by(|a, b| a.position.y.partial_cmp(&b.position.y).unwrap());
        } else {
            entries.sort_by(|a, b| a.position.z.partial_cmp(&b.position.z).unwrap());
        }

        // Split at median
        let median = entries.len() / 2;
        let (left_slice, right_slice) = entries.split_at_mut(median);

        let left = Box::new(Self::build_recursive(left_slice));
        let right = Box::new(Self::build_recursive(right_slice));

        BallNode3D::Internal {
            center,
            radius,
            left,
            right,
        }
    }

    fn compute_bounding_ball(entries: &[BallEntry3D<T>]) -> (Vec3, f32) {
        let mut center = Vec3::ZERO;
        for entry in entries {
            center += entry.position;
        }
        center /= entries.len() as f32;

        let mut radius = 0.0f32;
        for entry in entries {
            radius = radius.max(entry.position.distance(center));
        }

        (center, radius)
    }

    /// Returns the number of points in the tree.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if the tree is empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Finds the nearest point to the given position.
    ///
    /// Returns `None` if the tree is empty.
    pub fn nearest(&self, position: Vec3) -> Option<(Vec3, &T, f32)> {
        let mut best: Option<(Vec3, &T, f32)> = None;
        Self::nearest_recursive(&self.root, position, &mut best);
        best
    }

    fn nearest_recursive<'a>(
        node: &'a BallNode3D<T>,
        position: Vec3,
        best: &mut Option<(Vec3, &'a T, f32)>,
    ) {
        match node {
            BallNode3D::Empty => {}
            BallNode3D::Leaf { entry, .. } => {
                let dist = entry.position.distance(position);
                if best.is_none() || dist < best.as_ref().unwrap().2 {
                    *best = Some((entry.position, &entry.data, dist));
                }
            }
            BallNode3D::Internal {
                center,
                radius,
                left,
                right,
            } => {
                let dist_to_center = position.distance(*center);
                let min_possible_dist = (dist_to_center - radius).max(0.0);

                if let Some((_, _, best_dist)) = best {
                    if min_possible_dist >= *best_dist {
                        return;
                    }
                }

                let left_center = Self::get_center(left);
                let right_center = Self::get_center(right);

                let (first, second) = match (left_center, right_center) {
                    (Some(lc), Some(rc)) => {
                        if position.distance(lc) < position.distance(rc) {
                            (left.as_ref(), right.as_ref())
                        } else {
                            (right.as_ref(), left.as_ref())
                        }
                    }
                    (Some(_), None) => (left.as_ref(), right.as_ref()),
                    (None, Some(_)) => (right.as_ref(), left.as_ref()),
                    (None, None) => return,
                };

                Self::nearest_recursive(first, position, best);
                Self::nearest_recursive(second, position, best);
            }
        }
    }

    fn get_center(node: &BallNode3D<T>) -> Option<Vec3> {
        match node {
            BallNode3D::Empty => None,
            BallNode3D::Leaf { entry } => Some(entry.position),
            BallNode3D::Internal { center, .. } => Some(*center),
        }
    }

    /// Finds the k nearest points to the given position.
    ///
    /// Returns up to k points, sorted by distance (closest first).
    pub fn k_nearest(&self, position: Vec3, k: usize) -> Vec<(Vec3, &T, f32)> {
        if k == 0 {
            return Vec::new();
        }

        let mut heap: BinaryHeap<KNearestCandidate3D<T>> = BinaryHeap::new();
        Self::k_nearest_recursive(&self.root, position, k, &mut heap);

        let mut results: Vec<_> = heap
            .into_iter()
            .map(|c| (c.position, c.data, c.distance))
            .collect();
        results.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap());
        results
    }

    fn k_nearest_recursive<'a>(
        node: &'a BallNode3D<T>,
        position: Vec3,
        k: usize,
        heap: &mut BinaryHeap<KNearestCandidate3D<'a, T>>,
    ) {
        match node {
            BallNode3D::Empty => {}
            BallNode3D::Leaf { entry, .. } => {
                let dist = entry.position.distance(position);
                if heap.len() < k {
                    heap.push(KNearestCandidate3D {
                        position: entry.position,
                        data: &entry.data,
                        distance: dist,
                    });
                } else if dist < heap.peek().unwrap().distance {
                    heap.pop();
                    heap.push(KNearestCandidate3D {
                        position: entry.position,
                        data: &entry.data,
                        distance: dist,
                    });
                }
            }
            BallNode3D::Internal {
                center,
                radius,
                left,
                right,
            } => {
                let dist_to_center = position.distance(*center);
                let min_possible_dist = (dist_to_center - radius).max(0.0);

                if heap.len() >= k && min_possible_dist >= heap.peek().unwrap().distance {
                    return;
                }

                let left_center = Self::get_center(left);
                let right_center = Self::get_center(right);

                let (first, second) = match (left_center, right_center) {
                    (Some(lc), Some(rc)) => {
                        if position.distance(lc) < position.distance(rc) {
                            (left.as_ref(), right.as_ref())
                        } else {
                            (right.as_ref(), left.as_ref())
                        }
                    }
                    (Some(_), None) => (left.as_ref(), right.as_ref()),
                    (None, Some(_)) => (right.as_ref(), left.as_ref()),
                    (None, None) => return,
                };

                Self::k_nearest_recursive(first, position, k, heap);
                Self::k_nearest_recursive(second, position, k, heap);
            }
        }
    }

    /// Queries all points within a given radius of the center.
    pub fn query_radius(&self, center: Vec3, radius: f32) -> Vec<(Vec3, &T, f32)> {
        let mut results = Vec::new();
        Self::query_radius_recursive(&self.root, center, radius, &mut results);
        results
    }

    fn query_radius_recursive<'a>(
        node: &'a BallNode3D<T>,
        center: Vec3,
        radius: f32,
        results: &mut Vec<(Vec3, &'a T, f32)>,
    ) {
        match node {
            BallNode3D::Empty => {}
            BallNode3D::Leaf { entry, .. } => {
                let dist = entry.position.distance(center);
                if dist <= radius {
                    results.push((entry.position, &entry.data, dist));
                }
            }
            BallNode3D::Internal {
                center: node_center,
                radius: node_radius,
                left,
                right,
            } => {
                let dist_to_center = center.distance(*node_center);
                if dist_to_center > radius + node_radius {
                    return;
                }

                Self::query_radius_recursive(left, center, radius, results);
                Self::query_radius_recursive(right, center, radius, results);
            }
        }
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
/// cargo test -p rhizome-resin-spatial --features invariant-tests
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
