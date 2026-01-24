//! Space colonization algorithm for generating branching structures.
//!
//! Creates natural-looking trees, blood vessels, lightning, and other
//! branching patterns by growing toward attraction points.
//!
//! Operations are serializable structs with `apply` methods.
//! See `docs/design/ops-as-values.md`.
//!
//! # Example
//!
//! ```
//! use unshape_space_colonization::{SpaceColonization, SpaceColonizationConfig};
//! use glam::Vec3;
//!
//! let config = SpaceColonizationConfig {
//!     attraction_distance: 5.0,
//!     kill_distance: 1.0,
//!     segment_length: 0.5,
//!     ..Default::default()
//! };
//!
//! let mut sc = SpaceColonization::new(config);
//!
//! // Add attraction points (e.g., in a sphere)
//! sc.add_attraction_points_sphere(Vec3::new(0.0, 5.0, 0.0), 3.0, 100, 12345);
//!
//! // Add root node
//! sc.add_root(Vec3::ZERO);
//!
//! // Run the algorithm
//! sc.run(100);
//!
//! // Get the resulting tree structure
//! let nodes = sc.nodes();
//! let edges = sc.edges();
//! ```

use glam::Vec3;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// Registers all space-colonization operations with an [`OpRegistry`].
///
/// Call this to enable deserialization of space-colonization ops from saved pipelines.
#[cfg(feature = "dynop")]
pub fn register_ops(registry: &mut unshape_op::OpRegistry) {
    registry.register_type::<SpaceColonizationParams>("resin::SpaceColonizationParams");
}

/// Configuration for the space colonization algorithm.
///
/// This is an operation struct - use [`SpaceColonizationParams::apply`] to create
/// a [`SpaceColonization`] instance ready for use.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dynop", derive(unshape_op::Op))]
#[cfg_attr(feature = "dynop", op(input = (), output = SpaceColonization))]
pub struct SpaceColonizationParams {
    /// Distance within which an attraction point influences a node.
    pub attraction_distance: f32,
    /// Distance at which an attraction point is removed (colonized).
    pub kill_distance: f32,
    /// Length of new segments when branching.
    pub segment_length: f32,
    /// Tropism direction and strength (e.g., gravity, light).
    pub tropism: Vec3,
    /// Tropism strength (0.0 = none, 1.0 = strong).
    pub tropism_strength: f32,
    /// Smoothing factor for growth direction (0.0 = sharp, 1.0 = smooth).
    pub smoothing: f32,
    /// Maximum number of iterations.
    pub max_iterations: usize,
}

impl Default for SpaceColonizationParams {
    fn default() -> Self {
        Self {
            attraction_distance: 5.0,
            kill_distance: 1.0,
            segment_length: 0.5,
            tropism: Vec3::ZERO,
            tropism_strength: 0.0,
            smoothing: 0.0,
            max_iterations: 1000,
        }
    }
}

impl SpaceColonizationParams {
    /// Applies this configuration to create a new [`SpaceColonization`] instance.
    pub fn apply(&self) -> SpaceColonization {
        SpaceColonization::new(self.clone())
    }
}

/// Backwards-compatible type alias.
pub type SpaceColonizationConfig = SpaceColonizationParams;

/// A node in the branching structure.
#[derive(Debug, Clone)]
pub struct BranchNode {
    /// Position in 3D space.
    pub position: Vec3,
    /// Parent node index (None for root).
    pub parent: Option<usize>,
    /// Depth from root.
    pub depth: usize,
    /// Radius (for thickness computation).
    pub radius: f32,
}

/// An edge connecting two nodes.
#[derive(Debug, Clone, Copy)]
pub struct BranchEdge {
    /// Start node index.
    pub from: usize,
    /// End node index.
    pub to: usize,
}

/// Space colonization algorithm for generating branching structures.
#[derive(Debug, Clone)]
pub struct SpaceColonization {
    /// Configuration.
    config: SpaceColonizationParams,
    /// Attraction points.
    attraction_points: Vec<Vec3>,
    /// Active attraction points (indices).
    active_points: HashSet<usize>,
    /// Branch nodes.
    nodes: Vec<BranchNode>,
    /// Edges (connections between nodes).
    edges: Vec<BranchEdge>,
}

impl SpaceColonization {
    /// Creates a new space colonization instance.
    pub fn new(config: SpaceColonizationParams) -> Self {
        Self {
            config,
            attraction_points: Vec::new(),
            active_points: HashSet::new(),
            nodes: Vec::new(),
            edges: Vec::new(),
        }
    }

    /// Adds attraction points in a spherical volume.
    pub fn add_attraction_points_sphere(
        &mut self,
        center: Vec3,
        radius: f32,
        count: usize,
        seed: u64,
    ) {
        let mut rng = seed;

        for _ in 0..count {
            // Generate random point in sphere using rejection sampling
            loop {
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                let x = (rng as f32 / u64::MAX as f32) * 2.0 - 1.0;
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                let y = (rng as f32 / u64::MAX as f32) * 2.0 - 1.0;
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                let z = (rng as f32 / u64::MAX as f32) * 2.0 - 1.0;

                let p = Vec3::new(x, y, z);
                if p.length_squared() <= 1.0 {
                    let idx = self.attraction_points.len();
                    self.attraction_points.push(center + p * radius);
                    self.active_points.insert(idx);
                    break;
                }
            }
        }
    }

    /// Adds attraction points in a box volume.
    pub fn add_attraction_points_box(&mut self, min: Vec3, max: Vec3, count: usize, seed: u64) {
        let mut rng = seed;
        let size = max - min;

        for _ in 0..count {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            let x = (rng as f32 / u64::MAX as f32) * size.x;
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            let y = (rng as f32 / u64::MAX as f32) * size.y;
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            let z = (rng as f32 / u64::MAX as f32) * size.z;

            let idx = self.attraction_points.len();
            self.attraction_points.push(min + Vec3::new(x, y, z));
            self.active_points.insert(idx);
        }
    }

    /// Adds attraction points in a cylinder volume.
    pub fn add_attraction_points_cylinder(
        &mut self,
        base: Vec3,
        axis: Vec3,
        height: f32,
        radius: f32,
        count: usize,
        seed: u64,
    ) {
        let mut rng = seed;
        let axis_norm = axis.normalize();

        // Create orthonormal basis
        let up = if axis_norm.y.abs() < 0.9 {
            Vec3::Y
        } else {
            Vec3::X
        };
        let u = axis_norm.cross(up).normalize();
        let v = u.cross(axis_norm);

        for _ in 0..count {
            // Random angle
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            let angle = (rng as f32 / u64::MAX as f32) * std::f32::consts::TAU;

            // Random radius (square root for uniform distribution)
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            let r = (rng as f32 / u64::MAX as f32).sqrt() * radius;

            // Random height
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            let h = (rng as f32 / u64::MAX as f32) * height;

            let offset = u * angle.cos() * r + v * angle.sin() * r + axis_norm * h;
            let idx = self.attraction_points.len();
            self.attraction_points.push(base + offset);
            self.active_points.insert(idx);
        }
    }

    /// Adds a single attraction point.
    pub fn add_attraction_point(&mut self, point: Vec3) {
        let idx = self.attraction_points.len();
        self.attraction_points.push(point);
        self.active_points.insert(idx);
    }

    /// Adds a root node.
    pub fn add_root(&mut self, position: Vec3) {
        self.nodes.push(BranchNode {
            position,
            parent: None,
            depth: 0,
            radius: 1.0,
        });
    }

    /// Runs the algorithm for a specified number of iterations.
    pub fn run(&mut self, iterations: usize) {
        let max_iter = iterations.min(self.config.max_iterations);

        for _ in 0..max_iter {
            if self.active_points.is_empty() {
                break;
            }

            if !self.step() {
                break;
            }
        }

        // Compute radii using pipe model
        self.compute_radii();
    }

    /// Runs a single iteration of the algorithm.
    pub fn step(&mut self) -> bool {
        if self.nodes.is_empty() || self.active_points.is_empty() {
            return false;
        }

        // Find which attraction points influence which nodes
        let mut influences: HashMap<usize, Vec<Vec3>> = HashMap::new();
        let mut points_to_remove: HashSet<usize> = HashSet::new();

        for &point_idx in &self.active_points {
            let point = self.attraction_points[point_idx];

            // Find nearest node
            let mut nearest_node = None;
            let mut nearest_dist = f32::INFINITY;

            for (node_idx, node) in self.nodes.iter().enumerate() {
                let dist = (point - node.position).length();

                if dist < self.config.kill_distance {
                    points_to_remove.insert(point_idx);
                    break;
                }

                if dist < self.config.attraction_distance && dist < nearest_dist {
                    nearest_dist = dist;
                    nearest_node = Some(node_idx);
                }
            }

            // Record influence
            if let Some(node_idx) = nearest_node {
                if !points_to_remove.contains(&point_idx) {
                    let direction = (point - self.nodes[node_idx].position).normalize();
                    influences.entry(node_idx).or_default().push(direction);
                }
            }
        }

        // Remove colonized points
        for idx in &points_to_remove {
            self.active_points.remove(idx);
        }

        // Grow new nodes
        let mut new_nodes = Vec::new();
        let mut new_edges = Vec::new();

        for (node_idx, directions) in influences {
            if directions.is_empty() {
                continue;
            }

            // Average direction
            let mut avg_direction: Vec3 = directions.iter().sum();
            avg_direction = avg_direction.normalize();

            // Apply tropism
            if self.config.tropism_strength > 0.0 {
                avg_direction = (avg_direction
                    + self.config.tropism * self.config.tropism_strength)
                    .normalize();
            }

            // Apply smoothing with parent direction
            if self.config.smoothing > 0.0 {
                if let Some(parent_idx) = self.nodes[node_idx].parent {
                    let parent_dir = (self.nodes[node_idx].position
                        - self.nodes[parent_idx].position)
                        .normalize();
                    avg_direction = avg_direction
                        .lerp(parent_dir, self.config.smoothing)
                        .normalize();
                }
            }

            // Create new node
            let new_position =
                self.nodes[node_idx].position + avg_direction * self.config.segment_length;

            let new_node_idx = self.nodes.len() + new_nodes.len();
            new_nodes.push(BranchNode {
                position: new_position,
                parent: Some(node_idx),
                depth: self.nodes[node_idx].depth + 1,
                radius: 1.0,
            });

            new_edges.push(BranchEdge {
                from: node_idx,
                to: new_node_idx,
            });
        }

        if new_nodes.is_empty() {
            return false;
        }

        self.nodes.extend(new_nodes);
        self.edges.extend(new_edges);

        true
    }

    /// Computes radii for all nodes using the pipe model.
    fn compute_radii(&mut self) {
        // Build children list
        let mut children: HashMap<usize, Vec<usize>> = HashMap::new();
        for node_idx in 0..self.nodes.len() {
            if let Some(parent) = self.nodes[node_idx].parent {
                children.entry(parent).or_default().push(node_idx);
            }
        }

        // Find leaf nodes
        let leaves: Vec<usize> = self
            .nodes
            .iter()
            .enumerate()
            .filter(|(idx, _)| !children.contains_key(idx))
            .map(|(idx, _)| idx)
            .collect();

        // Set leaf radii
        for &leaf in &leaves {
            self.nodes[leaf].radius = 0.1;
        }

        // Propagate up using pipe model: r_parent^2 = sum(r_child^2)
        let mut processed: HashSet<usize> = leaves.iter().copied().collect();
        let mut to_process: Vec<usize> = leaves.clone();

        while let Some(node_idx) = to_process.pop() {
            if let Some(parent_idx) = self.nodes[node_idx].parent {
                if processed.contains(&parent_idx) {
                    continue;
                }

                // Check if all children are processed
                let parent_children = children.get(&parent_idx);
                if let Some(kids) = parent_children {
                    if kids.iter().all(|c| processed.contains(c)) {
                        // Compute parent radius
                        let radius_sq: f32 =
                            kids.iter().map(|&c| self.nodes[c].radius.powi(2)).sum();
                        self.nodes[parent_idx].radius = radius_sq.sqrt();
                        processed.insert(parent_idx);
                        to_process.push(parent_idx);
                    }
                }
            }
        }
    }

    /// Returns the branch nodes.
    pub fn nodes(&self) -> &[BranchNode] {
        &self.nodes
    }

    /// Returns the edges.
    pub fn edges(&self) -> &[BranchEdge] {
        &self.edges
    }

    /// Returns the positions of all nodes.
    pub fn positions(&self) -> Vec<Vec3> {
        self.nodes.iter().map(|n| n.position).collect()
    }

    /// Returns the remaining attraction points.
    pub fn remaining_attraction_points(&self) -> Vec<Vec3> {
        self.active_points
            .iter()
            .map(|&idx| self.attraction_points[idx])
            .collect()
    }

    /// Returns the total length of all branches.
    pub fn total_length(&self) -> f32 {
        self.edges
            .iter()
            .map(|e| (self.nodes[e.to].position - self.nodes[e.from].position).length())
            .sum()
    }

    /// Returns the maximum depth of the tree.
    pub fn max_depth(&self) -> usize {
        self.nodes.iter().map(|n| n.depth).max().unwrap_or(0)
    }

    /// Clears all data.
    pub fn clear(&mut self) {
        self.attraction_points.clear();
        self.active_points.clear();
        self.nodes.clear();
        self.edges.clear();
    }

    /// Generates line segments for visualization.
    pub fn to_line_segments(&self) -> Vec<(Vec3, Vec3)> {
        self.edges
            .iter()
            .map(|e| (self.nodes[e.from].position, self.nodes[e.to].position))
            .collect()
    }
}

/// Creates a tree-like structure with default settings.
pub fn generate_tree(
    trunk_base: Vec3,
    crown_center: Vec3,
    crown_radius: f32,
    attraction_points: usize,
    seed: u64,
) -> SpaceColonization {
    let trunk_to_crown = (crown_center - trunk_base).length();

    let mut config = SpaceColonizationConfig::default();
    // Use distances relative to the total height for better coverage
    config.attraction_distance = (trunk_to_crown + crown_radius) * 0.8;
    config.kill_distance = crown_radius * 0.15;
    config.segment_length = crown_radius * 0.1;
    config.tropism = Vec3::Y; // Grow upward
    config.tropism_strength = 0.1;

    let mut sc = SpaceColonization::new(config);
    sc.add_attraction_points_sphere(crown_center, crown_radius, attraction_points, seed);
    sc.add_root(trunk_base);
    sc.run(500);

    sc
}

/// Creates a lightning-like branching pattern.
pub fn generate_lightning(
    start: Vec3,
    end: Vec3,
    spread: f32,
    attraction_points: usize,
    seed: u64,
) -> SpaceColonization {
    let direction = end - start;
    let length = direction.length();

    let mut config = SpaceColonizationConfig::default();
    config.attraction_distance = length * 0.3;
    config.kill_distance = length * 0.05;
    config.segment_length = length * 0.02;
    config.tropism = direction.normalize();
    config.tropism_strength = 0.3;

    let mut sc = SpaceColonization::new(config);

    // Add attraction points along a cylinder from start to end
    sc.add_attraction_points_cylinder(start, direction, length, spread, attraction_points, seed);
    sc.add_root(start);
    sc.run(200);

    sc
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = SpaceColonizationConfig::default();
        assert!(config.attraction_distance > 0.0);
        assert!(config.kill_distance > 0.0);
        assert!(config.segment_length > 0.0);
    }

    #[test]
    fn test_add_root() {
        let mut sc = SpaceColonization::new(SpaceColonizationConfig::default());
        sc.add_root(Vec3::ZERO);

        assert_eq!(sc.nodes().len(), 1);
        assert_eq!(sc.nodes()[0].position, Vec3::ZERO);
        assert!(sc.nodes()[0].parent.is_none());
    }

    #[test]
    fn test_add_attraction_points_sphere() {
        let mut sc = SpaceColonization::new(SpaceColonizationConfig::default());
        sc.add_attraction_points_sphere(Vec3::ZERO, 5.0, 100, 12345);

        // Should have 100 attraction points
        assert_eq!(sc.remaining_attraction_points().len(), 100);

        // All points should be within sphere
        for point in sc.remaining_attraction_points() {
            assert!(point.length() <= 5.0 + 0.01);
        }
    }

    #[test]
    fn test_add_attraction_points_box() {
        let mut sc = SpaceColonization::new(SpaceColonizationConfig::default());
        sc.add_attraction_points_box(Vec3::ZERO, Vec3::ONE * 5.0, 100, 12345);

        assert_eq!(sc.remaining_attraction_points().len(), 100);

        // All points should be within box
        for point in sc.remaining_attraction_points() {
            assert!(point.x >= 0.0 && point.x <= 5.0);
            assert!(point.y >= 0.0 && point.y <= 5.0);
            assert!(point.z >= 0.0 && point.z <= 5.0);
        }
    }

    #[test]
    fn test_single_step() {
        let config = SpaceColonizationConfig {
            attraction_distance: 10.0,
            kill_distance: 0.5,
            segment_length: 1.0,
            ..Default::default()
        };

        let mut sc = SpaceColonization::new(config);
        sc.add_root(Vec3::ZERO);
        sc.add_attraction_point(Vec3::new(0.0, 5.0, 0.0));

        let stepped = sc.step();
        assert!(stepped);
        assert!(sc.nodes().len() > 1);
    }

    #[test]
    fn test_run() {
        let config = SpaceColonizationConfig {
            attraction_distance: 5.0,
            kill_distance: 0.5,
            segment_length: 0.5,
            ..Default::default()
        };

        let mut sc = SpaceColonization::new(config);
        sc.add_attraction_points_sphere(Vec3::new(0.0, 3.0, 0.0), 2.0, 50, 12345);
        sc.add_root(Vec3::ZERO);

        sc.run(100);

        // Should have grown some branches
        assert!(sc.nodes().len() > 1);
        assert!(!sc.edges().is_empty());
    }

    #[test]
    fn test_edges_connect_nodes() {
        let config = SpaceColonizationConfig {
            attraction_distance: 5.0,
            kill_distance: 0.5,
            segment_length: 0.5,
            ..Default::default()
        };

        let mut sc = SpaceColonization::new(config);
        sc.add_attraction_points_sphere(Vec3::new(0.0, 3.0, 0.0), 2.0, 50, 12345);
        sc.add_root(Vec3::ZERO);
        sc.run(100);

        // All edges should connect valid nodes
        for edge in sc.edges() {
            assert!(edge.from < sc.nodes().len());
            assert!(edge.to < sc.nodes().len());
        }
    }

    #[test]
    fn test_total_length() {
        let config = SpaceColonizationConfig {
            attraction_distance: 5.0,
            kill_distance: 0.5,
            segment_length: 0.5,
            ..Default::default()
        };

        let mut sc = SpaceColonization::new(config);
        sc.add_attraction_points_sphere(Vec3::new(0.0, 3.0, 0.0), 2.0, 50, 12345);
        sc.add_root(Vec3::ZERO);
        sc.run(100);

        let length = sc.total_length();
        assert!(length > 0.0);
    }

    #[test]
    fn test_max_depth() {
        let config = SpaceColonizationConfig {
            attraction_distance: 5.0,
            kill_distance: 0.5,
            segment_length: 0.5,
            ..Default::default()
        };

        let mut sc = SpaceColonization::new(config);
        sc.add_attraction_points_sphere(Vec3::new(0.0, 3.0, 0.0), 2.0, 50, 12345);
        sc.add_root(Vec3::ZERO);
        sc.run(100);

        assert!(sc.max_depth() > 0);
    }

    #[test]
    fn test_generate_tree() {
        let sc = generate_tree(Vec3::ZERO, Vec3::new(0.0, 5.0, 0.0), 3.0, 100, 12345);

        assert!(sc.nodes().len() > 1);
        assert!(!sc.edges().is_empty());
    }

    #[test]
    fn test_generate_lightning() {
        let sc = generate_lightning(Vec3::ZERO, Vec3::new(0.0, 10.0, 0.0), 1.0, 100, 12345);

        assert!(sc.nodes().len() > 1);
    }

    #[test]
    fn test_to_line_segments() {
        let config = SpaceColonizationConfig {
            attraction_distance: 5.0,
            kill_distance: 0.5,
            segment_length: 0.5,
            ..Default::default()
        };

        let mut sc = SpaceColonization::new(config);
        sc.add_attraction_points_sphere(Vec3::new(0.0, 3.0, 0.0), 2.0, 50, 12345);
        sc.add_root(Vec3::ZERO);
        sc.run(100);

        let segments = sc.to_line_segments();
        assert_eq!(segments.len(), sc.edges().len());
    }

    #[test]
    fn test_clear() {
        let mut sc = SpaceColonization::new(SpaceColonizationConfig::default());
        sc.add_attraction_points_sphere(Vec3::ZERO, 5.0, 100, 12345);
        sc.add_root(Vec3::ZERO);
        sc.run(10);

        sc.clear();

        assert!(sc.nodes().is_empty());
        assert!(sc.edges().is_empty());
        assert!(sc.remaining_attraction_points().is_empty());
    }

    #[test]
    fn test_radii_computed() {
        let config = SpaceColonizationConfig {
            attraction_distance: 5.0,
            kill_distance: 0.5,
            segment_length: 0.5,
            ..Default::default()
        };

        let mut sc = SpaceColonization::new(config);
        sc.add_attraction_points_sphere(Vec3::new(0.0, 3.0, 0.0), 2.0, 50, 12345);
        sc.add_root(Vec3::ZERO);
        sc.run(100);

        // All nodes should have non-zero radius
        for node in sc.nodes() {
            assert!(node.radius > 0.0);
        }
    }
}
