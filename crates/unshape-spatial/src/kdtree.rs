use std::collections::BinaryHeap;

use glam::{Vec2, Vec3};

use crate::{Aabb2, Aabb3, KNearestCandidate2D, KNearestCandidate3D};

// ============================================================================
// KD-Tree (2D)
// ============================================================================

/// A point with associated data stored in a 2D KD-tree.
#[derive(Debug, Clone)]
pub(crate) struct KdEntry2D<T> {
    pub position: Vec2,
    pub data: T,
}

/// A node in the 2D KD-tree.
#[derive(Debug)]
pub(crate) enum KdNode2D<T> {
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
/// use unshape_spatial::KdTree2D;
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
pub(crate) struct KdEntry3D<T> {
    pub position: Vec3,
    pub data: T,
}

/// A node in the 3D KD-tree.
#[derive(Debug)]
pub(crate) enum KdNode3D<T> {
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
/// use unshape_spatial::KdTree3D;
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
