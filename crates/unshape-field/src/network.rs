use glam::Vec2;

use crate::erosion::{Heightmap, SimpleRng};

/// A node in a network (intersection, city, water source).
#[derive(Debug, Clone)]
pub struct NetworkNode {
    /// Position in 2D space.
    pub position: Vec2,
    /// Optional height for terrain integration.
    pub height: f32,
    /// Node importance (affects connectivity priority).
    pub importance: f32,
}

impl NetworkNode {
    /// Creates a new network node.
    pub fn new(position: Vec2) -> Self {
        Self {
            position,
            height: 0.0,
            importance: 1.0,
        }
    }
}

/// An edge connecting two nodes with a path.
#[derive(Debug, Clone)]
pub struct NetworkEdge {
    /// Index of start node.
    pub start: usize,
    /// Index of end node.
    pub end: usize,
    /// Intermediate path points.
    pub path: Vec<Vec2>,
    /// Edge weight (distance, cost, etc.).
    pub weight: f32,
    /// Edge type (road width, river flow, etc.).
    pub edge_type: f32,
}

impl NetworkEdge {
    /// Creates a new edge.
    pub fn new(start: usize, end: usize) -> Self {
        Self {
            start,
            end,
            path: Vec::new(),
            weight: 0.0,
            edge_type: 1.0,
        }
    }
}

/// A network of connected nodes and edges.
#[derive(Debug, Clone)]
pub struct Network {
    /// Nodes in the network.
    pub nodes: Vec<NetworkNode>,
    /// Edges connecting nodes.
    pub edges: Vec<NetworkEdge>,
}

impl Network {
    /// Creates an empty network.
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            edges: Vec::new(),
        }
    }

    /// Adds a node and returns its index.
    pub fn add_node(&mut self, node: NetworkNode) -> usize {
        let idx = self.nodes.len();
        self.nodes.push(node);
        idx
    }

    /// Adds an edge between two nodes.
    pub fn add_edge(&mut self, start: usize, end: usize) -> &mut NetworkEdge {
        let edge = NetworkEdge::new(start, end);
        self.edges.push(edge);
        self.edges.last_mut().unwrap()
    }

    /// Returns all edges connected to a node.
    pub fn edges_for_node(&self, node_idx: usize) -> Vec<&NetworkEdge> {
        self.edges
            .iter()
            .filter(|e| e.start == node_idx || e.end == node_idx)
            .collect()
    }

    /// Returns neighboring node indices.
    pub fn neighbors(&self, node_idx: usize) -> Vec<usize> {
        self.edges
            .iter()
            .filter_map(|e| {
                if e.start == node_idx {
                    Some(e.end)
                } else if e.end == node_idx {
                    Some(e.start)
                } else {
                    None
                }
            })
            .collect()
    }

    /// Returns the full path for an edge including endpoints.
    pub fn edge_path(&self, edge: &NetworkEdge) -> Vec<Vec2> {
        let start_pos = self.nodes[edge.start].position;
        let end_pos = self.nodes[edge.end].position;

        let mut full_path = vec![start_pos];
        full_path.extend(edge.path.iter().cloned());
        full_path.push(end_pos);
        full_path
    }

    /// Samples all edge paths at regular intervals.
    pub fn sample_edges(&self, segment_length: f32) -> Vec<Vec<Vec2>> {
        self.edges
            .iter()
            .map(|edge| {
                let path = self.edge_path(edge);
                sample_path(&path, segment_length)
            })
            .collect()
    }
}

impl Default for Network {
    fn default() -> Self {
        Self::new()
    }
}

/// Samples a path at regular intervals.
fn sample_path(path: &[Vec2], segment_length: f32) -> Vec<Vec2> {
    if path.len() < 2 {
        return path.to_vec();
    }

    let mut result = vec![path[0]];
    let mut accumulated = 0.0;

    for window in path.windows(2) {
        let start = window[0];
        let end = window[1];
        let seg_len = (end - start).length();
        let dir = (end - start).normalize_or_zero();

        let mut pos = 0.0;
        while pos < seg_len {
            let remaining = segment_length - accumulated;
            if pos + remaining <= seg_len {
                pos += remaining;
                result.push(start + dir * pos);
                accumulated = 0.0;
            } else {
                accumulated += seg_len - pos;
                break;
            }
        }
    }

    if result.last() != path.last() {
        result.push(*path.last().unwrap());
    }

    result
}

/// Road network generation operation.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "dynop", derive(unshape_op::Op))]
#[cfg_attr(feature = "dynop", op(input = (), output = Network))]
pub struct RoadNetwork {
    /// Number of cities/intersections.
    pub num_nodes: usize,
    /// Area bounds (min_x, min_y, max_x, max_y).
    pub bounds: (f32, f32, f32, f32),
    /// Whether to generate minimum spanning tree first.
    pub use_mst: bool,
    /// Extra connections beyond MST (0.0 = none, 1.0 = full).
    pub extra_connectivity: f32,
    /// Number of path relaxation iterations.
    pub relaxation_iterations: usize,
    /// Path curvature amount.
    pub curvature: f32,
    /// Random seed for generation.
    pub seed: u64,
}

impl Default for RoadNetwork {
    fn default() -> Self {
        Self {
            num_nodes: 10,
            bounds: (0.0, 0.0, 100.0, 100.0),
            use_mst: true,
            extra_connectivity: 0.2,
            relaxation_iterations: 3,
            curvature: 0.3,
            seed: 0,
        }
    }
}

impl RoadNetwork {
    /// Applies this operation to generate a road network.
    pub fn apply(&self) -> Network {
        generate_road_network(self, self.seed)
    }
}

/// Backwards-compatible type alias.
pub type RoadNetworkConfig = RoadNetwork;

/// Generates a road network using random node placement and MST.
pub fn generate_road_network(config: &RoadNetwork, seed: u64) -> Network {
    let mut rng = SimpleRng::new(seed);
    let mut network = Network::new();

    // Generate random nodes
    let (min_x, min_y, max_x, max_y) = config.bounds;
    for _ in 0..config.num_nodes {
        let x = min_x + rng.next_f32() * (max_x - min_x);
        let y = min_y + rng.next_f32() * (max_y - min_y);
        network.add_node(NetworkNode::new(Vec2::new(x, y)));
    }

    if network.nodes.len() < 2 {
        return network;
    }

    if config.use_mst {
        // Build MST using Prim's algorithm
        build_mst(&mut network);
    }

    // Add extra connections
    if config.extra_connectivity > 0.0 {
        add_extra_connections(&mut network, config.extra_connectivity, &mut rng);
    }

    // Relax paths for natural curves
    for edge in &mut network.edges {
        relax_edge_path(
            &network.nodes,
            edge,
            config.relaxation_iterations,
            config.curvature,
        );
    }

    network
}

/// Builds a minimum spanning tree connecting all nodes.
fn build_mst(network: &mut Network) {
    let n = network.nodes.len();
    if n < 2 {
        return;
    }

    let mut in_tree = vec![false; n];
    let mut min_dist = vec![f32::MAX; n];
    let mut min_edge = vec![0usize; n];

    // Start from node 0
    in_tree[0] = true;

    // Initialize distances from node 0
    for i in 1..n {
        let d = (network.nodes[i].position - network.nodes[0].position).length();
        min_dist[i] = d;
        min_edge[i] = 0;
    }

    // Add n-1 edges
    for _ in 0..n - 1 {
        // Find closest node not in tree
        let mut best_dist = f32::MAX;
        let mut best_node = 0;

        for i in 0..n {
            if !in_tree[i] && min_dist[i] < best_dist {
                best_dist = min_dist[i];
                best_node = i;
            }
        }

        if best_dist == f32::MAX {
            break; // No more reachable nodes
        }

        // Add edge to tree
        in_tree[best_node] = true;
        network.add_edge(min_edge[best_node], best_node).weight = best_dist;

        // Update distances
        for i in 0..n {
            if !in_tree[i] {
                let d = (network.nodes[i].position - network.nodes[best_node].position).length();
                if d < min_dist[i] {
                    min_dist[i] = d;
                    min_edge[i] = best_node;
                }
            }
        }
    }
}

/// Adds extra connections beyond the MST.
fn add_extra_connections(network: &mut Network, connectivity: f32, rng: &mut SimpleRng) {
    let n = network.nodes.len();
    let existing_edges: std::collections::HashSet<(usize, usize)> = network
        .edges
        .iter()
        .flat_map(|e| [(e.start, e.end), (e.end, e.start)])
        .collect();

    // Calculate how many extra edges to add
    let max_extra = n * (n - 1) / 2 - network.edges.len();
    let num_extra = (max_extra as f32 * connectivity) as usize;

    // Collect possible edges sorted by distance
    let mut possible: Vec<(usize, usize, f32)> = Vec::new();
    for i in 0..n {
        for j in i + 1..n {
            if !existing_edges.contains(&(i, j)) {
                let d = (network.nodes[j].position - network.nodes[i].position).length();
                possible.push((i, j, d));
            }
        }
    }

    // Sort by distance
    possible.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap());

    // Add edges with probability favoring shorter ones
    let mut added = 0;
    for (i, j, d) in possible {
        if added >= num_extra {
            break;
        }
        // Higher probability for shorter edges
        let prob = 1.0 / (1.0 + d * 0.1);
        if rng.next_f32() < prob {
            network.add_edge(i, j).weight = d;
            added += 1;
        }
    }
}

/// Relaxes an edge path for natural curves.
fn relax_edge_path(
    nodes: &[NetworkNode],
    edge: &mut NetworkEdge,
    iterations: usize,
    curvature: f32,
) {
    if iterations == 0 || curvature <= 0.0 {
        return;
    }

    let start = nodes[edge.start].position;
    let end = nodes[edge.end].position;
    let dist = (end - start).length();

    // Add intermediate points
    let num_points = (dist / 10.0).max(2.0) as usize;
    edge.path = (1..num_points)
        .map(|i| {
            let t = i as f32 / num_points as f32;
            start.lerp(end, t)
        })
        .collect();

    // Perturb and relax
    let perp = Vec2::new(-(end - start).y, (end - start).x).normalize_or_zero();
    for (i, p) in edge.path.iter_mut().enumerate() {
        let t = (i + 1) as f32 / (num_points) as f32;
        let wave = (t * std::f32::consts::PI).sin() * curvature * dist * 0.1;
        *p += perp * wave;
    }

    // Smooth the path
    for _ in 0..iterations {
        let path_clone = edge.path.clone();
        for i in 0..edge.path.len() {
            let prev = if i == 0 { start } else { path_clone[i - 1] };
            let next = if i == edge.path.len() - 1 {
                end
            } else {
                path_clone[i + 1]
            };
            edge.path[i] = (prev + next) * 0.5;
        }
    }
}

/// River network generation operation.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "dynop", derive(unshape_op::Op))]
#[cfg_attr(feature = "dynop", op(input = Heightmap, output = Network))]
pub struct RiverNetwork {
    /// Number of river sources.
    pub num_sources: usize,
    /// Minimum height for sources.
    pub source_min_height: f32,
    /// Number of steps per river.
    pub max_steps: usize,
    /// Step size for gradient descent.
    pub step_size: f32,
    /// Whether to merge rivers that meet.
    pub merge_rivers: bool,
    /// Random seed for generation.
    pub seed: u64,
}

impl Default for RiverNetwork {
    fn default() -> Self {
        Self {
            num_sources: 3,
            source_min_height: 0.7,
            max_steps: 200,
            step_size: 1.0,
            merge_rivers: true,
            seed: 0,
        }
    }
}

impl RiverNetwork {
    /// Applies this operation to generate a river network from a heightmap.
    pub fn apply(&self, heightmap: &Heightmap) -> Network {
        generate_river_network(heightmap, self, self.seed)
    }
}

/// Backwards-compatible type alias.
pub type RiverNetworkConfig = RiverNetwork;

/// Generates a river network by following terrain gradients.
pub fn generate_river_network(heightmap: &Heightmap, config: &RiverNetwork, seed: u64) -> Network {
    let mut rng = SimpleRng::new(seed);
    let mut network = Network::new();

    // Find source points (high elevation)
    let mut sources: Vec<Vec2> = Vec::new();
    let (width, height) = (heightmap.width, heightmap.height);

    for _ in 0..config.num_sources * 10 {
        if sources.len() >= config.num_sources {
            break;
        }

        let x = rng.next_f32() * (width - 2) as f32 + 1.0;
        let y = rng.next_f32() * (height - 2) as f32 + 1.0;
        let h = heightmap.sample(x, y);

        // Accept high points not too close to existing sources
        if h >= config.source_min_height {
            let too_close = sources
                .iter()
                .any(|s| (*s - Vec2::new(x, y)).length() < 10.0);
            if !too_close {
                sources.push(Vec2::new(x, y));
            }
        }
    }

    // Trace each river
    let mut all_paths: Vec<Vec<Vec2>> = Vec::new();

    for source in &sources {
        let path = trace_river_path(heightmap, *source, config);
        if path.len() >= 2 {
            all_paths.push(path);
        }
    }

    // Build network from paths
    if config.merge_rivers {
        build_merged_river_network(&mut network, &all_paths);
    } else {
        for path in all_paths {
            if path.len() >= 2 {
                let start_idx = network.add_node(NetworkNode::new(path[0]));
                let end_idx = network.add_node(NetworkNode::new(*path.last().unwrap()));
                network.add_edge(start_idx, end_idx).path = path[1..path.len() - 1].to_vec();
            }
        }
    }

    network
}

/// Traces a river path following the steepest descent.
fn trace_river_path(heightmap: &Heightmap, start: Vec2, config: &RiverNetworkConfig) -> Vec<Vec2> {
    let mut path = vec![start];
    let mut pos = start;
    let (width, height) = (heightmap.width as f32, heightmap.height as f32);

    for _ in 0..config.max_steps {
        // Check bounds
        if pos.x <= 1.0 || pos.x >= width - 2.0 || pos.y <= 1.0 || pos.y >= height - 2.0 {
            break;
        }

        // Find steepest descent
        let current_h = heightmap.sample(pos.x, pos.y);
        let mut best_dir = Vec2::ZERO;
        let mut best_drop = 0.0;

        // Sample in 8 directions
        for angle_idx in 0..8 {
            let angle = angle_idx as f32 * std::f32::consts::TAU / 8.0;
            let dir = Vec2::new(angle.cos(), angle.sin());
            let sample_pos = pos + dir * config.step_size;

            if sample_pos.x > 0.0
                && sample_pos.x < width - 1.0
                && sample_pos.y > 0.0
                && sample_pos.y < height - 1.0
            {
                let sample_h = heightmap.sample(sample_pos.x, sample_pos.y);
                let drop = current_h - sample_h;
                if drop > best_drop {
                    best_drop = drop;
                    best_dir = dir;
                }
            }
        }

        // Stop if no downhill direction
        if best_drop <= 0.0 {
            break;
        }

        // Move in best direction
        pos += best_dir * config.step_size;
        path.push(pos);

        // Stop if reached low ground
        let h = heightmap.sample(pos.x, pos.y);
        if h < 0.05 {
            break;
        }
    }

    path
}

/// Builds a river network that merges rivers at intersection points.
fn build_merged_river_network(network: &mut Network, paths: &[Vec<Vec2>]) {
    let merge_threshold = 3.0;
    let mut node_positions: Vec<Vec2> = Vec::new();

    // Find or create node at position
    let find_or_create_node = |network: &mut Network, pos: Vec2, positions: &mut Vec<Vec2>| {
        for (i, &p) in positions.iter().enumerate() {
            if (p - pos).length() < merge_threshold {
                return i;
            }
        }
        let idx = network.add_node(NetworkNode::new(pos));
        positions.push(pos);
        idx
    };

    for path in paths {
        if path.len() < 2 {
            continue;
        }

        let mut prev_node = find_or_create_node(network, path[0], &mut node_positions);
        let mut segment_path: Vec<Vec2> = Vec::new();

        for &point in &path[1..] {
            // Check if this point is near an existing node
            let mut near_node = None;
            for (i, &p) in node_positions.iter().enumerate() {
                if (p - point).length() < merge_threshold && i != prev_node {
                    near_node = Some(i);
                    break;
                }
            }

            if let Some(node_idx) = near_node {
                // Create edge to existing node
                if prev_node != node_idx {
                    network.add_edge(prev_node, node_idx).path = segment_path.clone();
                }
                prev_node = node_idx;
                segment_path.clear();
            } else {
                segment_path.push(point);
            }
        }

        // Final segment
        let end_pos = *path.last().unwrap();
        let end_node = find_or_create_node(network, end_pos, &mut node_positions);
        if prev_node != end_node {
            network.add_edge(prev_node, end_node).path = segment_path;
        }
    }
}
