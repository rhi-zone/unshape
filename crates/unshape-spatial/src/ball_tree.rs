use std::collections::BinaryHeap;

use glam::{Vec2, Vec3};

use crate::{KNearestCandidate2D, KNearestCandidate3D};

// ============================================================================
// Ball Tree (2D)
// ============================================================================

/// A point with associated data stored in a 2D Ball tree.
#[derive(Debug, Clone)]
pub(crate) struct BallEntry2D<T> {
    pub position: Vec2,
    pub data: T,
}

/// A node in the 2D Ball tree.
#[derive(Debug)]
pub(crate) enum BallNode2D<T> {
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
/// use unshape_spatial::BallTree2D;
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
pub(crate) struct BallEntry3D<T> {
    pub position: Vec3,
    pub data: T,
}

/// A node in the 3D Ball tree.
#[derive(Debug)]
pub(crate) enum BallNode3D<T> {
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
/// use unshape_spatial::BallTree3D;
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
