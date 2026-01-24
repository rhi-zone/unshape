//! Road and river network generation.
//!
//! Provides graph-based path generation for roads, rivers, and other
//! linear network structures. Includes terrain-following for rivers
//! and hierarchical network generation for roads.
//!
//! # Example
//!
//! ```
//! use unshape_procgen::network::{RoadNetwork, RoadConfig, RoadType};
//! use glam::Vec2;
//!
//! // Create a road network
//! let config = RoadConfig::default();
//! let mut network = RoadNetwork::new(config);
//!
//! // Add some key nodes (cities, intersections)
//! let a = network.add_node(Vec2::new(0.0, 0.0));
//! let b = network.add_node(Vec2::new(10.0, 0.0));
//! let c = network.add_node(Vec2::new(5.0, 8.0));
//!
//! // Connect them
//! network.connect(a, b, RoadType::Highway);
//! network.connect(b, c, RoadType::Secondary);
//! network.connect(c, a, RoadType::Local);
//!
//! // Get edges for rendering
//! let segments = network.to_segments();
//! ```

use glam::Vec2;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet, VecDeque};

// ============================================================================
// Core Graph Types
// ============================================================================

/// A node in a network graph.
#[derive(Debug, Clone)]
pub struct NetworkNode {
    /// Position in 2D space.
    pub position: Vec2,
    /// Node type/category.
    pub node_type: NodeType,
    /// Optional elevation (for terrain following).
    pub elevation: f32,
    /// Connected edge indices.
    pub edges: Vec<usize>,
}

/// Type of network node.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum NodeType {
    /// Generic node.
    #[default]
    Generic,
    /// Source node (e.g., spring for rivers, city center for roads).
    Source,
    /// Sink node (e.g., ocean for rivers, terminus for roads).
    Sink,
    /// Junction/intersection.
    Junction,
    /// Waypoint along a path.
    Waypoint,
}

/// An edge in a network graph.
#[derive(Debug, Clone)]
pub struct NetworkEdge {
    /// Start node index.
    pub from: usize,
    /// End node index.
    pub to: usize,
    /// Edge weight/cost.
    pub weight: f32,
    /// Edge type.
    pub edge_type: EdgeType,
    /// Flow direction (for rivers).
    pub flow: f32,
}

/// Type of network edge.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum EdgeType {
    /// Generic edge.
    #[default]
    Generic,
    /// Main/primary route.
    Primary,
    /// Secondary route.
    Secondary,
    /// Tertiary/local route.
    Tertiary,
}

// ============================================================================
// Road Network
// ============================================================================

/// Configuration for road network generation.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct RoadConfig {
    /// Minimum distance between nodes.
    pub min_node_distance: f32,
    /// Maximum edge length before adding waypoints.
    pub max_edge_length: f32,
    /// Probability of creating branches at junctions.
    pub branch_probability: f32,
    /// Maximum branch angle (radians).
    pub max_branch_angle: f32,
    /// Grid snap distance (0 = no snap).
    pub grid_snap: f32,
}

impl Default for RoadConfig {
    fn default() -> Self {
        Self {
            min_node_distance: 1.0,
            max_edge_length: 10.0,
            branch_probability: 0.3,
            max_branch_angle: std::f32::consts::FRAC_PI_4,
            grid_snap: 0.0,
        }
    }
}

/// Road type hierarchy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RoadType {
    /// Major highway/motorway.
    Highway,
    /// Secondary road.
    Secondary,
    /// Local/residential road.
    Local,
    /// Path/trail.
    Path,
}

impl RoadType {
    /// Returns the default width for this road type.
    pub fn default_width(&self) -> f32 {
        match self {
            RoadType::Highway => 4.0,
            RoadType::Secondary => 2.5,
            RoadType::Local => 1.5,
            RoadType::Path => 0.5,
        }
    }

    /// Returns the edge type equivalent.
    pub fn to_edge_type(&self) -> EdgeType {
        match self {
            RoadType::Highway => EdgeType::Primary,
            RoadType::Secondary => EdgeType::Secondary,
            RoadType::Local | RoadType::Path => EdgeType::Tertiary,
        }
    }
}

/// A road network graph.
#[derive(Debug, Clone)]
pub struct RoadNetwork {
    /// Configuration.
    pub config: RoadConfig,
    /// Nodes in the network.
    nodes: Vec<NetworkNode>,
    /// Edges in the network.
    edges: Vec<NetworkEdge>,
    /// Spatial index for fast node lookup.
    spatial_index: HashMap<(i32, i32), Vec<usize>>,
    /// Grid cell size for spatial index.
    cell_size: f32,
}

impl RoadNetwork {
    /// Creates a new empty road network.
    pub fn new(config: RoadConfig) -> Self {
        Self {
            config,
            nodes: Vec::new(),
            edges: Vec::new(),
            spatial_index: HashMap::new(),
            cell_size: 10.0,
        }
    }

    /// Adds a node at the given position.
    pub fn add_node(&mut self, position: Vec2) -> usize {
        let position = self.snap_position(position);
        let idx = self.nodes.len();

        self.nodes.push(NetworkNode {
            position,
            node_type: NodeType::Generic,
            elevation: 0.0,
            edges: Vec::new(),
        });

        // Update spatial index
        let cell = self.position_to_cell(position);
        self.spatial_index.entry(cell).or_default().push(idx);

        idx
    }

    /// Adds a node with a specific type.
    pub fn add_node_typed(&mut self, position: Vec2, node_type: NodeType) -> usize {
        let idx = self.add_node(position);
        self.nodes[idx].node_type = node_type;
        idx
    }

    /// Connects two nodes with a road.
    pub fn connect(&mut self, from: usize, to: usize, road_type: RoadType) -> Option<usize> {
        if from >= self.nodes.len() || to >= self.nodes.len() || from == to {
            return None;
        }

        // Check if already connected
        if self.are_connected(from, to) {
            return None;
        }

        let edge_idx = self.edges.len();
        let distance = self.nodes[from].position.distance(self.nodes[to].position);

        self.edges.push(NetworkEdge {
            from,
            to,
            weight: distance,
            edge_type: road_type.to_edge_type(),
            flow: 0.0,
        });

        self.nodes[from].edges.push(edge_idx);
        self.nodes[to].edges.push(edge_idx);

        // Update node types
        if self.nodes[from].edges.len() > 2 {
            self.nodes[from].node_type = NodeType::Junction;
        }
        if self.nodes[to].edges.len() > 2 {
            self.nodes[to].node_type = NodeType::Junction;
        }

        Some(edge_idx)
    }

    /// Checks if two nodes are already connected.
    pub fn are_connected(&self, a: usize, b: usize) -> bool {
        for &edge_idx in &self.nodes[a].edges {
            let edge = &self.edges[edge_idx];
            if (edge.from == a && edge.to == b) || (edge.from == b && edge.to == a) {
                return true;
            }
        }
        false
    }

    /// Finds the nearest node to a position.
    pub fn find_nearest(&self, position: Vec2) -> Option<usize> {
        let cell = self.position_to_cell(position);
        let mut best_idx = None;
        let mut best_dist = f32::INFINITY;

        // Check nearby cells
        for dx in -1..=1 {
            for dy in -1..=1 {
                let check_cell = (cell.0 + dx, cell.1 + dy);
                if let Some(indices) = self.spatial_index.get(&check_cell) {
                    for &idx in indices {
                        let dist = self.nodes[idx].position.distance(position);
                        if dist < best_dist {
                            best_dist = dist;
                            best_idx = Some(idx);
                        }
                    }
                }
            }
        }

        // Also check all nodes if no nearby found
        if best_idx.is_none() && !self.nodes.is_empty() {
            for (idx, node) in self.nodes.iter().enumerate() {
                let dist = node.position.distance(position);
                if dist < best_dist {
                    best_dist = dist;
                    best_idx = Some(idx);
                }
            }
        }

        best_idx
    }

    /// Finds a path between two nodes using A*.
    pub fn find_path(&self, from: usize, to: usize) -> Option<Vec<usize>> {
        if from >= self.nodes.len() || to >= self.nodes.len() {
            return None;
        }

        let mut open_set = BinaryHeap::new();
        let mut came_from: HashMap<usize, usize> = HashMap::new();
        let mut g_score: HashMap<usize, f32> = HashMap::new();

        g_score.insert(from, 0.0);
        let h = self.nodes[from].position.distance(self.nodes[to].position);
        open_set.push(AStarNode {
            node: from,
            f_score: h,
        });

        while let Some(current) = open_set.pop() {
            if current.node == to {
                // Reconstruct path
                let mut path = vec![to];
                let mut node = to;
                while let Some(&prev) = came_from.get(&node) {
                    path.push(prev);
                    node = prev;
                }
                path.reverse();
                return Some(path);
            }

            let current_g = g_score[&current.node];

            for &edge_idx in &self.nodes[current.node].edges {
                let edge = &self.edges[edge_idx];
                let neighbor = if edge.from == current.node {
                    edge.to
                } else {
                    edge.from
                };

                let tentative_g = current_g + edge.weight;
                let neighbor_g = g_score.get(&neighbor).copied().unwrap_or(f32::INFINITY);

                if tentative_g < neighbor_g {
                    came_from.insert(neighbor, current.node);
                    g_score.insert(neighbor, tentative_g);
                    let h = self.nodes[neighbor]
                        .position
                        .distance(self.nodes[to].position);
                    open_set.push(AStarNode {
                        node: neighbor,
                        f_score: tentative_g + h,
                    });
                }
            }
        }

        None
    }

    /// Returns all nodes.
    pub fn nodes(&self) -> &[NetworkNode] {
        &self.nodes
    }

    /// Returns all edges.
    pub fn edges(&self) -> &[NetworkEdge] {
        &self.edges
    }

    /// Returns line segments for rendering.
    pub fn to_segments(&self) -> Vec<(Vec2, Vec2)> {
        self.edges
            .iter()
            .map(|e| (self.nodes[e.from].position, self.nodes[e.to].position))
            .collect()
    }

    /// Returns the total length of all roads.
    pub fn total_length(&self) -> f32 {
        self.edges.iter().map(|e| e.weight).sum()
    }

    /// Returns the number of nodes.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Returns the number of edges.
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    fn snap_position(&self, position: Vec2) -> Vec2 {
        if self.config.grid_snap > 0.0 {
            let snap = self.config.grid_snap;
            Vec2::new(
                (position.x / snap).round() * snap,
                (position.y / snap).round() * snap,
            )
        } else {
            position
        }
    }

    fn position_to_cell(&self, position: Vec2) -> (i32, i32) {
        (
            (position.x / self.cell_size).floor() as i32,
            (position.y / self.cell_size).floor() as i32,
        )
    }
}

/// Node for A* priority queue.
#[derive(Debug, Clone)]
struct AStarNode {
    node: usize,
    f_score: f32,
}

impl PartialEq for AStarNode {
    fn eq(&self, other: &Self) -> bool {
        self.node == other.node
    }
}

impl Eq for AStarNode {}

impl Ord for AStarNode {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .f_score
            .partial_cmp(&self.f_score)
            .unwrap_or(Ordering::Equal)
    }
}

impl PartialOrd for AStarNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

// ============================================================================
// River Network
// ============================================================================

/// Configuration for river network generation.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct RiverConfig {
    /// Minimum flow threshold for a river.
    pub min_flow: f32,
    /// Flow accumulation factor.
    pub flow_accumulation: f32,
    /// Meander strength (0 = straight, 1 = very curvy).
    pub meander_strength: f32,
    /// Number of erosion iterations.
    pub erosion_iterations: usize,
}

impl Default for RiverConfig {
    fn default() -> Self {
        Self {
            min_flow: 1.0,
            flow_accumulation: 1.0,
            meander_strength: 0.3,
            erosion_iterations: 0,
        }
    }
}

/// Operation to generate a simple procedural river from source to sink.
///
/// Takes a seed (u64) and produces a RiverNetwork.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dynop", derive(unshape_op::Op))]
#[cfg_attr(feature = "dynop", op(input = u64, output = RiverNetwork))]
pub struct GenerateRiver {
    /// Source position (upstream).
    pub source: Vec2,
    /// Sink position (downstream).
    pub sink: Vec2,
    /// River configuration parameters.
    pub config: RiverConfig,
}

impl GenerateRiver {
    /// Creates a new river generator with the given source and sink.
    pub fn new(source: Vec2, sink: Vec2) -> Self {
        Self {
            source,
            sink,
            config: RiverConfig::default(),
        }
    }

    /// Applies this operation to generate a river network.
    pub fn apply(&self, seed: &u64) -> RiverNetwork {
        RiverNetwork::generate_river(self.source, self.sink, self.config.clone(), *seed)
    }
}

/// A river network with flow-based generation.
#[derive(Debug, Clone)]
pub struct RiverNetwork {
    /// Configuration.
    pub config: RiverConfig,
    /// Nodes in the network.
    nodes: Vec<NetworkNode>,
    /// Edges in the network (directed, downstream).
    edges: Vec<NetworkEdge>,
}

impl RiverNetwork {
    /// Creates a new empty river network.
    pub fn new(config: RiverConfig) -> Self {
        Self {
            config,
            nodes: Vec::new(),
            edges: Vec::new(),
        }
    }

    /// Creates a river network from a heightfield.
    ///
    /// Uses flow accumulation to determine where rivers form.
    pub fn from_heightfield(
        heights: &[f32],
        width: usize,
        height: usize,
        config: RiverConfig,
    ) -> Self {
        let mut network = Self::new(config);

        // Calculate flow direction and accumulation
        let flow_dirs = calculate_flow_direction(heights, width, height);
        let flow_accum = calculate_flow_accumulation(&flow_dirs, width, height);

        // Find river cells (where flow > threshold)
        let mut river_cells: HashSet<(usize, usize)> = HashSet::new();
        for y in 0..height {
            for x in 0..width {
                let idx = y * width + x;
                if flow_accum[idx] >= network.config.min_flow {
                    river_cells.insert((x, y));
                }
            }
        }

        // Create nodes for river cells
        let mut cell_to_node: HashMap<(usize, usize), usize> = HashMap::new();
        for &(x, y) in &river_cells {
            let idx = y * width + x;
            let node_idx = network.nodes.len();
            network.nodes.push(NetworkNode {
                position: Vec2::new(x as f32, y as f32),
                node_type: NodeType::Waypoint,
                elevation: heights[idx],
                edges: Vec::new(),
            });
            cell_to_node.insert((x, y), node_idx);
        }

        // Create edges following flow direction
        for &(x, y) in &river_cells {
            let idx = y * width + x;
            let dir = flow_dirs[idx];

            if dir >= 0 {
                let (dx, dy) = flow_direction_delta(dir);
                let nx = (x as i32 + dx) as usize;
                let ny = (y as i32 + dy) as usize;

                if let (Some(&from_node), Some(&to_node)) =
                    (cell_to_node.get(&(x, y)), cell_to_node.get(&(nx, ny)))
                {
                    let edge_idx = network.edges.len();
                    let from_elev = network.nodes[from_node].elevation;
                    let to_elev = network.nodes[to_node].elevation;

                    network.edges.push(NetworkEdge {
                        from: from_node,
                        to: to_node,
                        weight: from_elev - to_elev,
                        edge_type: EdgeType::Primary,
                        flow: flow_accum[idx],
                    });

                    network.nodes[from_node].edges.push(edge_idx);
                    network.nodes[to_node].edges.push(edge_idx);
                }
            }
        }

        // Mark sources and sinks
        for node_idx in 0..network.nodes.len() {
            let mut outgoing = 0;
            let mut incoming = 0;

            for &edge_idx in &network.nodes[node_idx].edges {
                if network.edges[edge_idx].from == node_idx {
                    outgoing += 1;
                } else {
                    incoming += 1;
                }
            }

            if incoming == 0 && outgoing > 0 {
                network.nodes[node_idx].node_type = NodeType::Source;
            } else if outgoing == 0 && incoming > 0 {
                network.nodes[node_idx].node_type = NodeType::Sink;
            } else if network.nodes[node_idx].edges.len() > 2 {
                network.nodes[node_idx].node_type = NodeType::Junction;
            }
        }

        network
    }

    /// Generates a simple procedural river from source to sink.
    pub fn generate_river(source: Vec2, sink: Vec2, config: RiverConfig, seed: u64) -> Self {
        let mut network = Self::new(config);
        let mut rng = seed.wrapping_add(1);

        // Create source node
        let source_idx = network.add_node(source, NodeType::Source, 100.0);

        // Create meandering path to sink
        let direction = (sink - source).normalize();
        let distance = source.distance(sink);
        let num_segments = (distance / 5.0).max(3.0) as usize;

        let mut current_idx = source_idx;
        let mut current_elev = 100.0;

        for i in 1..num_segments {
            let t = i as f32 / num_segments as f32;
            let base_pos = source.lerp(sink, t);

            // Add meander
            rng = lcg_next(rng);
            let meander =
                (rng as f32 / u64::MAX as f32 - 0.5) * 2.0 * network.config.meander_strength;
            let perpendicular = Vec2::new(-direction.y, direction.x);
            let meandered_pos = base_pos + perpendicular * meander * distance * 0.1;

            current_elev -= 100.0 / num_segments as f32;
            let next_idx = network.add_node(meandered_pos, NodeType::Waypoint, current_elev);
            network.connect(
                current_idx,
                next_idx,
                current_elev - network.nodes[next_idx].elevation,
            );

            current_idx = next_idx;
        }

        // Create sink node
        let sink_idx = network.add_node(sink, NodeType::Sink, 0.0);
        network.connect(current_idx, sink_idx, current_elev);

        network
    }

    /// Adds a node with position, type, and elevation.
    fn add_node(&mut self, position: Vec2, node_type: NodeType, elevation: f32) -> usize {
        let idx = self.nodes.len();
        self.nodes.push(NetworkNode {
            position,
            node_type,
            elevation,
            edges: Vec::new(),
        });
        idx
    }

    /// Connects two nodes with a river segment.
    fn connect(&mut self, from: usize, to: usize, flow: f32) -> usize {
        let edge_idx = self.edges.len();
        let from_pos = self.nodes[from].position;
        let to_pos = self.nodes[to].position;

        self.edges.push(NetworkEdge {
            from,
            to,
            weight: from_pos.distance(to_pos),
            edge_type: EdgeType::Primary,
            flow,
        });

        self.nodes[from].edges.push(edge_idx);
        self.nodes[to].edges.push(edge_idx);

        edge_idx
    }

    /// Returns all nodes.
    pub fn nodes(&self) -> &[NetworkNode] {
        &self.nodes
    }

    /// Returns all edges.
    pub fn edges(&self) -> &[NetworkEdge] {
        &self.edges
    }

    /// Returns line segments for rendering.
    pub fn to_segments(&self) -> Vec<(Vec2, Vec2)> {
        self.edges
            .iter()
            .map(|e| (self.nodes[e.from].position, self.nodes[e.to].position))
            .collect()
    }

    /// Returns segments with flow values.
    pub fn to_segments_with_flow(&self) -> Vec<(Vec2, Vec2, f32)> {
        self.edges
            .iter()
            .map(|e| {
                (
                    self.nodes[e.from].position,
                    self.nodes[e.to].position,
                    e.flow,
                )
            })
            .collect()
    }

    /// Returns the total length of all rivers.
    pub fn total_length(&self) -> f32 {
        self.edges.iter().map(|e| e.weight).sum()
    }

    /// Returns source nodes.
    pub fn sources(&self) -> Vec<usize> {
        self.nodes
            .iter()
            .enumerate()
            .filter(|(_, n)| n.node_type == NodeType::Source)
            .map(|(i, _)| i)
            .collect()
    }

    /// Returns sink nodes.
    pub fn sinks(&self) -> Vec<usize> {
        self.nodes
            .iter()
            .enumerate()
            .filter(|(_, n)| n.node_type == NodeType::Sink)
            .map(|(i, _)| i)
            .collect()
    }
}

// ============================================================================
// Road Network Operations
// ============================================================================

/// Operation to generate a grid-based road network.
///
/// Takes a seed (u64, unused) and produces a RoadNetwork.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dynop", derive(unshape_op::Op))]
#[cfg_attr(feature = "dynop", op(input = u64, output = RoadNetwork))]
pub struct GenerateRoadNetworkGrid {
    /// Minimum corner of the bounds.
    pub bounds_min: Vec2,
    /// Maximum corner of the bounds.
    pub bounds_max: Vec2,
    /// Grid spacing between roads.
    pub spacing: f32,
}

impl GenerateRoadNetworkGrid {
    /// Creates a new grid road network generator.
    pub fn new(bounds_min: Vec2, bounds_max: Vec2, spacing: f32) -> Self {
        Self {
            bounds_min,
            bounds_max,
            spacing,
        }
    }

    /// Applies this operation to generate a road network.
    pub fn apply(&self, _seed: &u64) -> RoadNetwork {
        generate_road_network_grid(self.bounds_min, self.bounds_max, self.spacing)
    }
}

/// Operation to generate a hierarchical road network with main roads and side streets.
///
/// Takes a seed (u64) and produces a RoadNetwork.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dynop", derive(unshape_op::Op))]
#[cfg_attr(feature = "dynop", op(input = u64, output = RoadNetwork))]
pub struct GenerateRoadNetworkHierarchical {
    /// Minimum corner of the bounds.
    pub bounds_min: Vec2,
    /// Maximum corner of the bounds.
    pub bounds_max: Vec2,
    /// Road density factor.
    pub density: f32,
}

impl GenerateRoadNetworkHierarchical {
    /// Creates a new hierarchical road network generator.
    pub fn new(bounds_min: Vec2, bounds_max: Vec2, density: f32) -> Self {
        Self {
            bounds_min,
            bounds_max,
            density,
        }
    }

    /// Applies this operation to generate a road network.
    pub fn apply(&self, seed: &u64) -> RoadNetwork {
        generate_road_network_hierarchical(self.bounds_min, self.bounds_max, self.density, *seed)
    }
}

// ============================================================================
// Network Generation Algorithms
// ============================================================================

/// Generates a minimum spanning tree connecting the given points.
pub fn minimum_spanning_tree(points: &[Vec2]) -> Vec<(usize, usize)> {
    if points.len() < 2 {
        return Vec::new();
    }

    // Prim's algorithm
    let n = points.len();
    let mut in_tree = vec![false; n];
    let mut edges = Vec::new();

    in_tree[0] = true;
    let mut num_in_tree = 1;

    while num_in_tree < n {
        let mut best_edge = None;
        let mut best_dist = f32::INFINITY;

        for (i, &in_i) in in_tree.iter().enumerate() {
            if !in_i {
                continue;
            }
            for (j, &in_j) in in_tree.iter().enumerate() {
                if in_j {
                    continue;
                }
                let dist = points[i].distance(points[j]);
                if dist < best_dist {
                    best_dist = dist;
                    best_edge = Some((i, j));
                }
            }
        }

        if let Some((from, to)) = best_edge {
            edges.push((from, to));
            in_tree[to] = true;
            num_in_tree += 1;
        } else {
            break;
        }
    }

    edges
}

/// Generates a Delaunay-like road network (connects nearby points).
pub fn generate_road_network_delaunay(points: &[Vec2], max_distance: f32) -> RoadNetwork {
    let config = RoadConfig::default();
    let mut network = RoadNetwork::new(config);

    // Add all points as nodes
    let node_indices: Vec<usize> = points.iter().map(|&p| network.add_node(p)).collect();

    // Connect nearby points
    for i in 0..points.len() {
        for j in (i + 1)..points.len() {
            let dist = points[i].distance(points[j]);
            if dist <= max_distance {
                network.connect(node_indices[i], node_indices[j], RoadType::Local);
            }
        }
    }

    network
}

/// Generates a hierarchical road network with main roads and side streets.
pub fn generate_road_network_hierarchical(
    bounds_min: Vec2,
    bounds_max: Vec2,
    density: f32,
    seed: u64,
) -> RoadNetwork {
    let config = RoadConfig {
        grid_snap: 1.0 / density,
        ..Default::default()
    };
    let mut network = RoadNetwork::new(config);
    let mut rng = seed.wrapping_add(1);

    let size = bounds_max - bounds_min;
    let num_main_roads = ((size.x + size.y) * density * 0.1).max(2.0) as usize;

    // Generate main roads (roughly horizontal and vertical)
    let mut main_nodes: Vec<usize> = Vec::new();

    for i in 0..num_main_roads {
        rng = lcg_next(rng);
        let horizontal = i % 2 == 0;

        if horizontal {
            let y = bounds_min.y + (rng as f32 / u64::MAX as f32) * size.y;
            let start = network.add_node(Vec2::new(bounds_min.x, y));
            let end = network.add_node(Vec2::new(bounds_max.x, y));
            network.connect(start, end, RoadType::Highway);
            main_nodes.push(start);
            main_nodes.push(end);
        } else {
            let x = bounds_min.x + (rng as f32 / u64::MAX as f32) * size.x;
            let start = network.add_node(Vec2::new(x, bounds_min.y));
            let end = network.add_node(Vec2::new(x, bounds_max.y));
            network.connect(start, end, RoadType::Highway);
            main_nodes.push(start);
            main_nodes.push(end);
        }
    }

    // Add secondary roads at intersections
    let num_secondary = (num_main_roads * 2).min(main_nodes.len());
    for i in 0..num_secondary {
        rng = lcg_next(rng);
        let node_idx = main_nodes[i % main_nodes.len()];
        let node_pos = network.nodes[node_idx].position;

        // Create a perpendicular road
        rng = lcg_next(rng);
        let angle = (rng as f32 / u64::MAX as f32) * std::f32::consts::TAU;
        let length = 5.0 + (rng as f32 / u64::MAX as f32) * 10.0;

        let end_pos = node_pos + Vec2::new(angle.cos(), angle.sin()) * length;
        let end_pos = end_pos.clamp(bounds_min, bounds_max);

        let end_node = network.add_node(end_pos);
        network.connect(node_idx, end_node, RoadType::Secondary);
    }

    // Add local roads
    let num_local = (size.x * size.y * density * 0.01) as usize;
    for _ in 0..num_local {
        rng = lcg_next(rng);
        let x = bounds_min.x + (rng as f32 / u64::MAX as f32) * size.x;
        rng = lcg_next(rng);
        let y = bounds_min.y + (rng as f32 / u64::MAX as f32) * size.y;

        let pos = Vec2::new(x, y);

        // Connect to nearest existing node
        if let Some(nearest) = network.find_nearest(pos) {
            let dist = network.nodes[nearest].position.distance(pos);
            if dist > 1.0 && dist < 15.0 {
                let new_node = network.add_node(pos);
                network.connect(nearest, new_node, RoadType::Local);
            }
        }
    }

    network
}

/// Generates a grid-based road network.
pub fn generate_road_network_grid(bounds_min: Vec2, bounds_max: Vec2, spacing: f32) -> RoadNetwork {
    let config = RoadConfig {
        grid_snap: spacing,
        ..Default::default()
    };
    let mut network = RoadNetwork::new(config);

    let size = bounds_max - bounds_min;
    let cols = (size.x / spacing).ceil() as usize + 1;
    let rows = (size.y / spacing).ceil() as usize + 1;

    // Create grid nodes
    let mut grid: Vec<Vec<usize>> = Vec::new();

    for row in 0..rows {
        let mut row_nodes = Vec::new();
        for col in 0..cols {
            let x = bounds_min.x + col as f32 * spacing;
            let y = bounds_min.y + row as f32 * spacing;
            let idx = network.add_node(Vec2::new(x, y));
            row_nodes.push(idx);
        }
        grid.push(row_nodes);
    }

    // Connect horizontally
    for row in 0..rows {
        for col in 0..cols - 1 {
            network.connect(grid[row][col], grid[row][col + 1], RoadType::Local);
        }
    }

    // Connect vertically
    for row in 0..rows - 1 {
        for col in 0..cols {
            network.connect(grid[row][col], grid[row + 1][col], RoadType::Local);
        }
    }

    network
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Simple LCG random number generator.
fn lcg_next(state: u64) -> u64 {
    state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407)
}

/// Calculates flow direction for each cell (D8 algorithm).
fn calculate_flow_direction(heights: &[f32], width: usize, height: usize) -> Vec<i32> {
    let mut flow_dirs = vec![-1i32; width * height];

    for y in 0..height {
        for x in 0..width {
            let idx = y * width + x;
            let current_h = heights[idx];

            let mut best_dir = -1;
            let mut best_drop = 0.0f32;

            // Check 8 neighbors
            for dir in 0..8 {
                let (dx, dy) = flow_direction_delta(dir);
                let nx = x as i32 + dx;
                let ny = y as i32 + dy;

                if nx < 0 || ny < 0 || nx >= width as i32 || ny >= height as i32 {
                    continue;
                }

                let neighbor_idx = ny as usize * width + nx as usize;
                let neighbor_h = heights[neighbor_idx];
                let drop = current_h - neighbor_h;

                if drop > best_drop {
                    best_drop = drop;
                    best_dir = dir;
                }
            }

            flow_dirs[idx] = best_dir;
        }
    }

    flow_dirs
}

/// Calculates flow accumulation from flow directions.
fn calculate_flow_accumulation(flow_dirs: &[i32], width: usize, height: usize) -> Vec<f32> {
    let mut accum = vec![1.0f32; width * height];

    // Count incoming flows for each cell
    let mut incoming_count = vec![0usize; width * height];
    for y in 0..height {
        for x in 0..width {
            let idx = y * width + x;
            let dir = flow_dirs[idx];
            if dir >= 0 {
                let (dx, dy) = flow_direction_delta(dir);
                let nx = (x as i32 + dx) as usize;
                let ny = (y as i32 + dy) as usize;
                if nx < width && ny < height {
                    incoming_count[ny * width + nx] += 1;
                }
            }
        }
    }

    // Find cells with no incoming flow (headwaters)
    let mut queue: VecDeque<(usize, usize)> = VecDeque::new();
    let mut remaining = incoming_count.clone();

    for y in 0..height {
        for x in 0..width {
            if remaining[y * width + x] == 0 {
                queue.push_back((x, y));
            }
        }
    }

    // Process in topological order
    while let Some((x, y)) = queue.pop_front() {
        let idx = y * width + x;
        let dir = flow_dirs[idx];

        if dir >= 0 {
            let (dx, dy) = flow_direction_delta(dir);
            let nx = (x as i32 + dx) as usize;
            let ny = (y as i32 + dy) as usize;

            if nx < width && ny < height {
                let neighbor_idx = ny * width + nx;
                accum[neighbor_idx] += accum[idx];
                remaining[neighbor_idx] -= 1;

                if remaining[neighbor_idx] == 0 {
                    queue.push_back((nx, ny));
                }
            }
        }
    }

    accum
}

/// Returns delta (dx, dy) for D8 flow direction.
fn flow_direction_delta(dir: i32) -> (i32, i32) {
    match dir {
        0 => (1, 0),   // E
        1 => (1, 1),   // SE
        2 => (0, 1),   // S
        3 => (-1, 1),  // SW
        4 => (-1, 0),  // W
        5 => (-1, -1), // NW
        6 => (0, -1),  // N
        7 => (1, -1),  // NE
        _ => (0, 0),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_road_network_creation() {
        let config = RoadConfig::default();
        let network = RoadNetwork::new(config);
        assert_eq!(network.node_count(), 0);
        assert_eq!(network.edge_count(), 0);
    }

    #[test]
    fn test_road_network_add_nodes() {
        let config = RoadConfig::default();
        let mut network = RoadNetwork::new(config);

        let a = network.add_node(Vec2::new(0.0, 0.0));
        let b = network.add_node(Vec2::new(10.0, 0.0));

        assert_eq!(network.node_count(), 2);
        assert_eq!(a, 0);
        assert_eq!(b, 1);
    }

    #[test]
    fn test_road_network_connect() {
        let config = RoadConfig::default();
        let mut network = RoadNetwork::new(config);

        let a = network.add_node(Vec2::new(0.0, 0.0));
        let b = network.add_node(Vec2::new(10.0, 0.0));

        let edge = network.connect(a, b, RoadType::Highway);
        assert!(edge.is_some());
        assert_eq!(network.edge_count(), 1);
        assert!(network.are_connected(a, b));
    }

    #[test]
    fn test_road_network_no_self_connect() {
        let config = RoadConfig::default();
        let mut network = RoadNetwork::new(config);

        let a = network.add_node(Vec2::new(0.0, 0.0));
        let edge = network.connect(a, a, RoadType::Highway);

        assert!(edge.is_none());
    }

    #[test]
    fn test_road_network_no_duplicate_connect() {
        let config = RoadConfig::default();
        let mut network = RoadNetwork::new(config);

        let a = network.add_node(Vec2::new(0.0, 0.0));
        let b = network.add_node(Vec2::new(10.0, 0.0));

        network.connect(a, b, RoadType::Highway);
        let duplicate = network.connect(a, b, RoadType::Highway);

        assert!(duplicate.is_none());
        assert_eq!(network.edge_count(), 1);
    }

    #[test]
    fn test_road_network_find_nearest() {
        let config = RoadConfig::default();
        let mut network = RoadNetwork::new(config);

        let a = network.add_node(Vec2::new(0.0, 0.0));
        let b = network.add_node(Vec2::new(10.0, 0.0));
        let _c = network.add_node(Vec2::new(5.0, 10.0));

        let nearest = network.find_nearest(Vec2::new(1.0, 1.0));
        assert_eq!(nearest, Some(a));

        let nearest = network.find_nearest(Vec2::new(9.0, 1.0));
        assert_eq!(nearest, Some(b));
    }

    #[test]
    fn test_road_network_pathfinding() {
        let config = RoadConfig::default();
        let mut network = RoadNetwork::new(config);

        let a = network.add_node(Vec2::new(0.0, 0.0));
        let b = network.add_node(Vec2::new(5.0, 0.0));
        let c = network.add_node(Vec2::new(10.0, 0.0));

        network.connect(a, b, RoadType::Highway);
        network.connect(b, c, RoadType::Highway);

        let path = network.find_path(a, c);
        assert!(path.is_some());
        assert_eq!(path.unwrap(), vec![a, b, c]);
    }

    #[test]
    fn test_road_network_to_segments() {
        let config = RoadConfig::default();
        let mut network = RoadNetwork::new(config);

        let a = network.add_node(Vec2::new(0.0, 0.0));
        let b = network.add_node(Vec2::new(10.0, 0.0));
        network.connect(a, b, RoadType::Highway);

        let segments = network.to_segments();
        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].0, Vec2::new(0.0, 0.0));
        assert_eq!(segments[0].1, Vec2::new(10.0, 0.0));
    }

    #[test]
    fn test_road_type_width() {
        assert!(RoadType::Highway.default_width() > RoadType::Secondary.default_width());
        assert!(RoadType::Secondary.default_width() > RoadType::Local.default_width());
        assert!(RoadType::Local.default_width() > RoadType::Path.default_width());
    }

    #[test]
    fn test_river_network_creation() {
        let config = RiverConfig::default();
        let network = RiverNetwork::new(config);
        assert_eq!(network.nodes().len(), 0);
        assert_eq!(network.edges().len(), 0);
    }

    #[test]
    fn test_river_generation() {
        let config = RiverConfig::default();
        let network =
            RiverNetwork::generate_river(Vec2::new(0.0, 0.0), Vec2::new(100.0, 0.0), config, 12345);

        assert!(network.nodes().len() > 2);
        assert!(!network.edges().is_empty());
        assert_eq!(network.sources().len(), 1);
        assert_eq!(network.sinks().len(), 1);
    }

    #[test]
    fn test_river_to_segments() {
        let config = RiverConfig::default();
        let network =
            RiverNetwork::generate_river(Vec2::new(0.0, 0.0), Vec2::new(50.0, 50.0), config, 99999);

        let segments = network.to_segments();
        assert!(!segments.is_empty());
        assert_eq!(segments.len(), network.edges().len());
    }

    #[test]
    fn test_minimum_spanning_tree() {
        let points = vec![
            Vec2::new(0.0, 0.0),
            Vec2::new(1.0, 0.0),
            Vec2::new(0.0, 1.0),
            Vec2::new(1.0, 1.0),
        ];

        let edges = minimum_spanning_tree(&points);
        assert_eq!(edges.len(), 3); // N-1 edges for N points
    }

    #[test]
    fn test_minimum_spanning_tree_empty() {
        let points: Vec<Vec2> = vec![];
        let edges = minimum_spanning_tree(&points);
        assert!(edges.is_empty());
    }

    #[test]
    fn test_minimum_spanning_tree_single() {
        let points = vec![Vec2::new(0.0, 0.0)];
        let edges = minimum_spanning_tree(&points);
        assert!(edges.is_empty());
    }

    #[test]
    fn test_generate_road_network_delaunay() {
        let points = vec![
            Vec2::new(0.0, 0.0),
            Vec2::new(5.0, 0.0),
            Vec2::new(2.5, 4.0),
        ];

        let network = generate_road_network_delaunay(&points, 10.0);
        assert_eq!(network.node_count(), 3);
        assert!(network.edge_count() > 0);
    }

    #[test]
    fn test_generate_road_network_hierarchical() {
        let network = generate_road_network_hierarchical(
            Vec2::new(0.0, 0.0),
            Vec2::new(100.0, 100.0),
            0.5,
            12345,
        );

        assert!(network.node_count() > 0);
        assert!(network.edge_count() > 0);
    }

    #[test]
    fn test_generate_road_network_grid() {
        let network = generate_road_network_grid(Vec2::new(0.0, 0.0), Vec2::new(20.0, 20.0), 5.0);

        // 5x5 grid = 25 nodes
        assert_eq!(network.node_count(), 25);
        // 4 horizontal + 4 vertical per row/col = 4*5 + 5*4 = 40 edges
        assert_eq!(network.edge_count(), 40);
    }

    #[test]
    fn test_flow_direction_delta() {
        // E
        assert_eq!(flow_direction_delta(0), (1, 0));
        // S
        assert_eq!(flow_direction_delta(2), (0, 1));
        // W
        assert_eq!(flow_direction_delta(4), (-1, 0));
        // N
        assert_eq!(flow_direction_delta(6), (0, -1));
    }

    #[test]
    fn test_river_from_heightfield() {
        // Create a simple sloped heightfield
        let width = 10;
        let height = 10;
        let mut heights = vec![0.0f32; width * height];

        // Slope from top-left to bottom-right
        for y in 0..height {
            for x in 0..width {
                heights[y * width + x] = 100.0 - (x + y) as f32 * 5.0;
            }
        }

        let config = RiverConfig {
            min_flow: 3.0,
            ..Default::default()
        };

        let network = RiverNetwork::from_heightfield(&heights, width, height, config);

        // Should have some rivers
        assert!(network.nodes().len() > 0);
    }

    #[test]
    fn test_road_total_length() {
        let config = RoadConfig::default();
        let mut network = RoadNetwork::new(config);

        let a = network.add_node(Vec2::new(0.0, 0.0));
        let b = network.add_node(Vec2::new(3.0, 4.0)); // Distance = 5

        network.connect(a, b, RoadType::Highway);

        let length = network.total_length();
        assert!((length - 5.0).abs() < 0.001);
    }

    #[test]
    fn test_node_types() {
        let config = RoadConfig::default();
        let mut network = RoadNetwork::new(config);

        let a = network.add_node_typed(Vec2::new(0.0, 0.0), NodeType::Source);
        assert_eq!(network.nodes()[a].node_type, NodeType::Source);
    }

    #[test]
    fn test_junction_detection() {
        let config = RoadConfig::default();
        let mut network = RoadNetwork::new(config);

        let center = network.add_node(Vec2::new(0.0, 0.0));
        let a = network.add_node(Vec2::new(1.0, 0.0));
        let b = network.add_node(Vec2::new(-1.0, 0.0));
        let c = network.add_node(Vec2::new(0.0, 1.0));

        network.connect(center, a, RoadType::Highway);
        network.connect(center, b, RoadType::Highway);
        network.connect(center, c, RoadType::Highway);

        // Center should now be a junction
        assert_eq!(network.nodes()[center].node_type, NodeType::Junction);
    }
}
