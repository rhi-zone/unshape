use std::collections::BinaryHeap;

use glam::Vec3;

use crate::{Aabb3, KNearestCandidate3D};

/// A point with associated data stored in an octree.
#[derive(Debug, Clone)]
pub(crate) struct OctreeEntry<T> {
    pub position: Vec3,
    pub data: T,
}

/// A node in the octree.
#[derive(Debug)]
pub(crate) enum OctreeNode<T> {
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
/// use unshape_spatial::{Octree, Aabb3};
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
    /// use unshape_spatial::{Octree, Aabb3};
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
