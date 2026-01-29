use glam::Vec2;

use crate::Aabb2;

/// An entry in the R-tree (a rectangle with associated data).
#[derive(Debug, Clone)]
pub(crate) struct RtreeEntry<T> {
    pub bounds: Aabb2,
    pub data: T,
}

/// A node in the R-tree.
#[derive(Debug)]
pub(crate) enum RtreeNode<T> {
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
    pub fn bounds(&self) -> Aabb2 {
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
/// use unshape_spatial::{Rtree, Aabb2};
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

pub(crate) fn union_aabb2(a: &Aabb2, b: &Aabb2) -> Aabb2 {
    Aabb2::new(a.min.min(b.min), a.max.max(b.max))
}

pub(crate) fn area(aabb: &Aabb2) -> f32 {
    let size = aabb.size();
    size.x * size.y
}

pub(crate) fn enlargement_area(base: &Aabb2, add: &Aabb2) -> f32 {
    let combined = union_aabb2(base, add);
    area(&combined) - area(base)
}
