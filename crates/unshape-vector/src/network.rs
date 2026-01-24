//! Vector network representation for anchor-based vector editing.
//!
//! A vector network is a graph-based representation of 2D vector art where paths
//! can branch and merge, unlike traditional linear path representations.
//!
//! # Structure
//!
//! - Anchors are points in 2D space where curves meet
//! - Edges connect anchors and can be lines or Bezier curves
//! - Paths can fork (one anchor connects to multiple edges) and join
//! - Closed regions are cycles in the graph that can be filled
//!
//! # Terminology
//!
//! This module uses "Anchor" instead of "Node" to distinguish from data flow nodes
//! (see `docs/conventions.md` for terminology across resin crates).
//!
//! # Usage
//!
//! ```ignore
//! let mut net = VectorNetwork::new();
//!
//! // Add anchors
//! let a0 = net.add_anchor(Vec2::ZERO);
//! let a1 = net.add_anchor(Vec2::new(100.0, 0.0));
//! let a2 = net.add_anchor(Vec2::new(50.0, 100.0));
//!
//! // Connect with edges
//! net.add_line(a0, a1);
//! net.add_line(a1, a2);
//! net.add_line(a2, a0);
//!
//! // Detect closed regions for filling
//! let regions = net.find_regions();
//! ```

use crate::{Path, PathBuilder};
use glam::Vec2;
use std::collections::{HashMap, HashSet};

/// Index for an anchor in the network.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct AnchorId(pub u32);

/// Index for an edge in the network.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct EdgeId(pub u32);

/// Sentinel value for null references.
const NULL_ID: u32 = u32::MAX;

impl AnchorId {
    /// Sentinel value representing no anchor.
    pub const NULL: AnchorId = AnchorId(NULL_ID);

    /// Returns true if this is the null sentinel.
    pub fn is_null(self) -> bool {
        self.0 == NULL_ID
    }
}

impl EdgeId {
    /// Sentinel value representing no edge.
    pub const NULL: EdgeId = EdgeId(NULL_ID);

    /// Returns true if this is the null sentinel.
    pub fn is_null(self) -> bool {
        self.0 == NULL_ID
    }
}

/// Handle style at an anchor for connected edges.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum HandleStyle {
    /// No handles (sharp corner).
    None,
    /// Symmetric handles (same direction and length on both sides).
    Symmetric,
    /// Smooth handles (same direction but different lengths).
    Smooth,
    /// Free handles (independent directions and lengths).
    Free,
}

impl Default for HandleStyle {
    fn default() -> Self {
        HandleStyle::None
    }
}

/// An anchor point in the vector network.
///
/// Anchors are 2D positions where curves meet. They have handle styles
/// that control curve continuity at that point.
#[derive(Debug, Clone, Default)]
pub struct Anchor {
    /// Position in 2D space.
    pub position: Vec2,
    /// Handle style at this anchor.
    pub handle_style: HandleStyle,
    /// Connected edges (edge id, handle offset for that edge at this anchor).
    pub edges: Vec<EdgeHandle>,
}

/// An edge's handle at a specific anchor.
#[derive(Debug, Clone, Copy, Default)]
pub struct EdgeHandle {
    /// The edge this handle belongs to.
    pub edge: EdgeId,
    /// Handle offset relative to the anchor position.
    /// For the start of an edge, this is control1 offset.
    /// For the end of an edge, this is control2 offset.
    pub handle: Vec2,
}

/// Type of edge connection.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EdgeType {
    /// Straight line segment.
    Line,
    /// Cubic Bezier curve.
    Cubic,
}

impl Default for EdgeType {
    fn default() -> Self {
        EdgeType::Line
    }
}

/// An edge connecting two anchors.
#[derive(Debug, Clone, Default)]
pub struct Edge {
    /// Start anchor.
    pub start: AnchorId,
    /// End anchor.
    pub end: AnchorId,
    /// Type of edge.
    pub edge_type: EdgeType,
    /// Control point 1 (for cubic curves, absolute position).
    pub control1: Vec2,
    /// Control point 2 (for cubic curves, absolute position).
    pub control2: Vec2,
}

impl Edge {
    /// Returns true if this edge connects the given anchor.
    pub fn connects(&self, anchor: AnchorId) -> bool {
        self.start == anchor || self.end == anchor
    }

    /// Returns the other anchor connected by this edge.
    pub fn other_anchor(&self, anchor: AnchorId) -> AnchorId {
        if self.start == anchor {
            self.end
        } else {
            self.start
        }
    }
}

/// A closed region in the vector network (detected cycle).
#[derive(Debug, Clone)]
pub struct Region {
    /// Anchors forming the boundary of this region, in order.
    pub anchors: Vec<AnchorId>,
    /// Edges forming the boundary of this region, in order.
    pub edges: Vec<EdgeId>,
}

impl Region {
    /// Converts this region to a Path for rendering.
    pub fn to_path(&self, network: &VectorNetwork) -> Path {
        if self.edges.is_empty() {
            return Path::new();
        }

        let mut builder = PathBuilder::new();

        // Start at first anchor
        let first_anchor = self.anchors[0];
        builder = builder.move_to(network.anchors[first_anchor.0 as usize].position);

        // Walk around the region
        for i in 0..self.edges.len() {
            let edge_id = self.edges[i];
            let edge = &network.edges[edge_id.0 as usize];
            let current_anchor = self.anchors[i];
            let next_anchor = self.anchors[(i + 1) % self.anchors.len()];

            // Determine direction
            let forward = edge.start == current_anchor;

            match edge.edge_type {
                EdgeType::Line => {
                    builder = builder.line_to(network.anchors[next_anchor.0 as usize].position);
                }
                EdgeType::Cubic => {
                    let (c1, c2) = if forward {
                        (edge.control1, edge.control2)
                    } else {
                        (edge.control2, edge.control1)
                    };
                    builder =
                        builder.cubic_to(c1, c2, network.anchors[next_anchor.0 as usize].position);
                }
            }
        }

        builder.close().build()
    }
}

/// A vector network: a graph of anchors connected by edges.
#[derive(Debug, Clone, Default)]
pub struct VectorNetwork {
    /// All anchors.
    pub anchors: Vec<Anchor>,
    /// All edges.
    pub edges: Vec<Edge>,
}

impl VectorNetwork {
    /// Creates an empty vector network.
    pub fn new() -> Self {
        Self::default()
    }

    // ==================== Anchor Operations ====================

    /// Adds an anchor at the given position.
    pub fn add_anchor(&mut self, position: Vec2) -> AnchorId {
        let id = AnchorId(self.anchors.len() as u32);
        self.anchors.push(Anchor {
            position,
            handle_style: HandleStyle::None,
            edges: Vec::new(),
        });
        id
    }

    /// Returns the position of an anchor.
    pub fn anchor_position(&self, anchor: AnchorId) -> Vec2 {
        self.anchors[anchor.0 as usize].position
    }

    /// Sets the position of an anchor.
    pub fn set_anchor_position(&mut self, anchor: AnchorId, position: Vec2) {
        let delta = position - self.anchors[anchor.0 as usize].position;
        self.anchors[anchor.0 as usize].position = position;

        // Update connected edge control points
        for eh in &self.anchors[anchor.0 as usize].edges.clone() {
            let edge = &mut self.edges[eh.edge.0 as usize];
            if edge.edge_type == EdgeType::Cubic {
                if edge.start == anchor {
                    edge.control1 += delta;
                }
                if edge.end == anchor {
                    edge.control2 += delta;
                }
            }
        }
    }

    /// Returns the number of anchors.
    pub fn anchor_count(&self) -> usize {
        self.anchors.len()
    }

    /// Returns the valence (number of connected edges) of an anchor.
    pub fn anchor_valence(&self, anchor: AnchorId) -> usize {
        self.anchors[anchor.0 as usize].edges.len()
    }

    /// Returns the edges connected to an anchor.
    pub fn anchor_edges(&self, anchor: AnchorId) -> Vec<EdgeId> {
        self.anchors[anchor.0 as usize]
            .edges
            .iter()
            .map(|eh| eh.edge)
            .collect()
    }

    /// Returns anchors adjacent to a given anchor.
    pub fn anchor_neighbors(&self, anchor: AnchorId) -> Vec<AnchorId> {
        self.anchors[anchor.0 as usize]
            .edges
            .iter()
            .map(|eh| self.edges[eh.edge.0 as usize].other_anchor(anchor))
            .collect()
    }

    // ==================== Edge Operations ====================

    /// Adds a line edge between two anchors.
    pub fn add_line(&mut self, start: AnchorId, end: AnchorId) -> EdgeId {
        let id = EdgeId(self.edges.len() as u32);

        let start_pos = self.anchors[start.0 as usize].position;
        let end_pos = self.anchors[end.0 as usize].position;

        self.edges.push(Edge {
            start,
            end,
            edge_type: EdgeType::Line,
            control1: start_pos,
            control2: end_pos,
        });

        // Add to anchor edge lists
        self.anchors[start.0 as usize].edges.push(EdgeHandle {
            edge: id,
            handle: Vec2::ZERO,
        });
        self.anchors[end.0 as usize].edges.push(EdgeHandle {
            edge: id,
            handle: Vec2::ZERO,
        });

        id
    }

    /// Adds a cubic Bezier edge between two anchors.
    pub fn add_cubic(
        &mut self,
        start: AnchorId,
        control1: Vec2,
        control2: Vec2,
        end: AnchorId,
    ) -> EdgeId {
        let id = EdgeId(self.edges.len() as u32);

        let start_pos = self.anchors[start.0 as usize].position;
        let end_pos = self.anchors[end.0 as usize].position;

        self.edges.push(Edge {
            start,
            end,
            edge_type: EdgeType::Cubic,
            control1,
            control2,
        });

        // Add to anchor edge lists with handle offsets
        self.anchors[start.0 as usize].edges.push(EdgeHandle {
            edge: id,
            handle: control1 - start_pos,
        });
        self.anchors[end.0 as usize].edges.push(EdgeHandle {
            edge: id,
            handle: control2 - end_pos,
        });

        id
    }

    /// Adds a smooth cubic edge where control points are computed automatically.
    pub fn add_smooth_edge(&mut self, start: AnchorId, end: AnchorId) -> EdgeId {
        let start_pos = self.anchors[start.0 as usize].position;
        let end_pos = self.anchors[end.0 as usize].position;

        // Place control points at 1/3 and 2/3 along the line
        let control1 = start_pos.lerp(end_pos, 1.0 / 3.0);
        let control2 = start_pos.lerp(end_pos, 2.0 / 3.0);

        self.add_cubic(start, control1, control2, end)
    }

    /// Returns the number of edges.
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// Sets control points for an edge (converts to cubic if needed).
    pub fn set_edge_controls(&mut self, edge: EdgeId, control1: Vec2, control2: Vec2) {
        let e = &mut self.edges[edge.0 as usize];
        e.edge_type = EdgeType::Cubic;
        e.control1 = control1;
        e.control2 = control2;

        // Update handles in connected anchors
        let start = e.start;
        let end = e.end;
        let start_pos = self.anchors[start.0 as usize].position;
        let end_pos = self.anchors[end.0 as usize].position;

        for eh in &mut self.anchors[start.0 as usize].edges {
            if eh.edge == edge {
                eh.handle = control1 - start_pos;
            }
        }
        for eh in &mut self.anchors[end.0 as usize].edges {
            if eh.edge == edge {
                eh.handle = control2 - end_pos;
            }
        }
    }

    /// Removes an edge from the network.
    pub fn remove_edge(&mut self, edge: EdgeId) {
        let e = &self.edges[edge.0 as usize];
        let start = e.start;
        let end = e.end;

        // Remove from anchor edge lists
        self.anchors[start.0 as usize]
            .edges
            .retain(|eh| eh.edge != edge);
        self.anchors[end.0 as usize]
            .edges
            .retain(|eh| eh.edge != edge);

        // Mark edge as removed (we don't actually remove to keep indices stable)
        self.edges[edge.0 as usize].start = AnchorId::NULL;
        self.edges[edge.0 as usize].end = AnchorId::NULL;
    }

    /// Returns true if an edge is valid (not removed).
    pub fn is_edge_valid(&self, edge: EdgeId) -> bool {
        !self.edges[edge.0 as usize].start.is_null()
    }

    // ==================== Path Conversion ====================

    /// Converts a sequence of connected edges to a Path.
    pub fn edges_to_path(&self, edges: &[EdgeId]) -> Path {
        if edges.is_empty() {
            return Path::new();
        }

        let mut builder = PathBuilder::new();

        // Find starting anchor
        let first_edge = &self.edges[edges[0].0 as usize];
        let mut current_anchor = first_edge.start;
        builder = builder.move_to(self.anchors[current_anchor.0 as usize].position);

        for &edge_id in edges {
            let edge = &self.edges[edge_id.0 as usize];

            // Determine direction
            let forward = edge.start == current_anchor;
            let next_anchor = if forward { edge.end } else { edge.start };

            match edge.edge_type {
                EdgeType::Line => {
                    builder = builder.line_to(self.anchors[next_anchor.0 as usize].position);
                }
                EdgeType::Cubic => {
                    let (c1, c2) = if forward {
                        (edge.control1, edge.control2)
                    } else {
                        (edge.control2, edge.control1)
                    };
                    builder =
                        builder.cubic_to(c1, c2, self.anchors[next_anchor.0 as usize].position);
                }
            }

            current_anchor = next_anchor;
        }

        builder.build()
    }

    /// Converts the entire network to paths (one path per connected component stroke).
    pub fn to_paths(&self) -> Vec<Path> {
        let mut paths = Vec::new();
        let mut visited: HashSet<EdgeId> = HashSet::new();

        for i in 0..self.edges.len() {
            let edge_id = EdgeId(i as u32);
            if visited.contains(&edge_id) || !self.is_edge_valid(edge_id) {
                continue;
            }

            // Find a chain of edges
            let chain = self.trace_edge_chain(edge_id, &mut visited);
            if !chain.is_empty() {
                paths.push(self.edges_to_path(&chain));
            }
        }

        paths
    }

    /// Traces a chain of connected edges starting from the given edge.
    fn trace_edge_chain(&self, start: EdgeId, visited: &mut HashSet<EdgeId>) -> Vec<EdgeId> {
        let mut chain = vec![start];
        visited.insert(start);

        // Extend forward from end anchor
        let edge = &self.edges[start.0 as usize];
        let mut current = edge.end;

        loop {
            // Find an unvisited edge from current anchor
            let next = self.anchors[current.0 as usize]
                .edges
                .iter()
                .find(|eh| !visited.contains(&eh.edge) && self.is_edge_valid(eh.edge))
                .map(|eh| eh.edge);

            match next {
                Some(edge_id) => {
                    visited.insert(edge_id);
                    chain.push(edge_id);
                    current = self.edges[edge_id.0 as usize].other_anchor(current);
                }
                None => break,
            }
        }

        // Extend backward from start anchor
        let mut current = edge.start;
        let mut backward = Vec::new();

        loop {
            let next = self.anchors[current.0 as usize]
                .edges
                .iter()
                .find(|eh| !visited.contains(&eh.edge) && self.is_edge_valid(eh.edge))
                .map(|eh| eh.edge);

            match next {
                Some(edge_id) => {
                    visited.insert(edge_id);
                    backward.push(edge_id);
                    current = self.edges[edge_id.0 as usize].other_anchor(current);
                }
                None => break,
            }
        }

        // Combine: backward (reversed) + chain
        backward.reverse();
        backward.extend(chain);
        backward
    }

    // ==================== Region Detection ====================

    /// Finds all closed regions (faces) in the network.
    ///
    /// Uses a planar graph algorithm to detect minimal cycles.
    pub fn find_regions(&self) -> Vec<Region> {
        let mut regions = Vec::new();

        if self.edges.is_empty() {
            return regions;
        }

        // Build adjacency information sorted by angle
        let adjacency = self.build_angular_adjacency();

        // Track which directed edges have been used
        let mut used: HashSet<(AnchorId, AnchorId)> = HashSet::new();

        // For each directed edge, try to trace a face
        for i in 0..self.edges.len() {
            let edge = &self.edges[i];
            if edge.start.is_null() {
                continue;
            }

            // Try both directions
            for &(start, end) in &[(edge.start, edge.end), (edge.end, edge.start)] {
                if used.contains(&(start, end)) {
                    continue;
                }

                if let Some(region) = self.trace_region(start, end, &adjacency, &mut used) {
                    regions.push(region);
                }
            }
        }

        regions
    }

    /// Builds adjacency lists sorted by angle for planar face detection.
    fn build_angular_adjacency(&self) -> HashMap<AnchorId, Vec<(AnchorId, EdgeId)>> {
        let mut adj: HashMap<AnchorId, Vec<(AnchorId, EdgeId)>> = HashMap::new();

        for (i, edge) in self.edges.iter().enumerate() {
            if edge.start.is_null() {
                continue;
            }

            let edge_id = EdgeId(i as u32);
            adj.entry(edge.start).or_default().push((edge.end, edge_id));
            adj.entry(edge.end).or_default().push((edge.start, edge_id));
        }

        // Sort each adjacency list by angle
        for (anchor_id, neighbors) in &mut adj {
            let center = self.anchors[anchor_id.0 as usize].position;
            neighbors.sort_by(|a, b| {
                let angle_a = (self.anchors[a.0.0 as usize].position - center).angle();
                let angle_b = (self.anchors[b.0.0 as usize].position - center).angle();
                angle_a.partial_cmp(&angle_b).unwrap()
            });
        }

        adj
    }

    /// Traces a region starting from a directed edge.
    fn trace_region(
        &self,
        start: AnchorId,
        second: AnchorId,
        adjacency: &HashMap<AnchorId, Vec<(AnchorId, EdgeId)>>,
        used: &mut HashSet<(AnchorId, AnchorId)>,
    ) -> Option<Region> {
        let mut anchors = vec![start];
        let mut edges = Vec::new();

        let mut prev = start;
        let mut current = second;

        let max_steps = self.anchors.len() * 2;
        let mut steps = 0;

        loop {
            if steps > max_steps {
                return None; // Prevent infinite loops
            }
            steps += 1;

            // Find the edge between prev and current
            let edge_id = self.find_edge_between(prev, current)?;
            edges.push(edge_id);
            used.insert((prev, current));

            anchors.push(current);

            // If we've returned to start, we have a region
            if current == start {
                // Remove the duplicate start anchor at the end
                anchors.pop();
                return Some(Region { anchors, edges });
            }

            // Find next anchor: turn right (clockwise) from incoming direction
            let neighbors = adjacency.get(&current)?;
            let incoming_angle = (self.anchors[prev.0 as usize].position
                - self.anchors[current.0 as usize].position)
                .angle();

            // Find the neighbor just after the incoming direction (clockwise)
            let mut next = None;
            let mut best_angle = f32::MAX;

            for &(neighbor, _) in neighbors {
                if neighbor == prev {
                    continue;
                }

                let outgoing_angle = (self.anchors[neighbor.0 as usize].position
                    - self.anchors[current.0 as usize].position)
                    .angle();

                // Compute clockwise angle difference
                let mut diff = incoming_angle - outgoing_angle;
                if diff <= 0.0 {
                    diff += std::f32::consts::TAU;
                }

                if diff < best_angle {
                    best_angle = diff;
                    next = Some(neighbor);
                }
            }

            match next {
                Some(n) => {
                    prev = current;
                    current = n;
                }
                None => return None, // Dead end
            }
        }
    }

    /// Finds the edge connecting two anchors.
    fn find_edge_between(&self, a: AnchorId, b: AnchorId) -> Option<EdgeId> {
        for eh in &self.anchors[a.0 as usize].edges {
            let edge = &self.edges[eh.edge.0 as usize];
            if (edge.start == a && edge.end == b) || (edge.start == b && edge.end == a) {
                return Some(eh.edge);
            }
        }
        None
    }

    // ==================== Network Editing ====================

    /// Splits an edge at a parameter value t (0-1).
    pub fn split_edge(&mut self, edge: EdgeId, t: f32) -> AnchorId {
        let e = &self.edges[edge.0 as usize];
        let start = e.start;
        let end = e.end;
        let edge_type = e.edge_type;

        let new_pos = match edge_type {
            EdgeType::Line => {
                let start_pos = self.anchors[start.0 as usize].position;
                let end_pos = self.anchors[end.0 as usize].position;
                start_pos.lerp(end_pos, t)
            }
            EdgeType::Cubic => {
                // De Casteljau's algorithm for cubic Bezier
                let p0 = self.anchors[start.0 as usize].position;
                let p1 = e.control1;
                let p2 = e.control2;
                let p3 = self.anchors[end.0 as usize].position;

                let q0 = p0.lerp(p1, t);
                let q1 = p1.lerp(p2, t);
                let q2 = p2.lerp(p3, t);

                let r0 = q0.lerp(q1, t);
                let r1 = q1.lerp(q2, t);

                r0.lerp(r1, t)
            }
        };

        // Add new anchor
        let new_anchor = self.add_anchor(new_pos);

        // Get original control points for cubic
        let (c1, c2) = if edge_type == EdgeType::Cubic {
            let e = &self.edges[edge.0 as usize];
            (e.control1, e.control2)
        } else {
            (Vec2::ZERO, Vec2::ZERO)
        };

        // Remove old edge
        self.remove_edge(edge);

        // Add two new edges
        match edge_type {
            EdgeType::Line => {
                self.add_line(start, new_anchor);
                self.add_line(new_anchor, end);
            }
            EdgeType::Cubic => {
                let p0 = self.anchors[start.0 as usize].position;
                let p3 = self.anchors[end.0 as usize].position;

                let q0 = p0.lerp(c1, t);
                let q1 = c1.lerp(c2, t);
                let q2 = c2.lerp(p3, t);

                let r0 = q0.lerp(q1, t);
                let r1 = q1.lerp(q2, t);

                self.add_cubic(start, q0, r0, new_anchor);
                self.add_cubic(new_anchor, r1, q2, end);
            }
        }

        new_anchor
    }

    /// Welds two anchors into one, merging their edges.
    pub fn weld_anchors(&mut self, keep: AnchorId, remove: AnchorId) {
        // Update all edges pointing to `remove` to point to `keep`
        let edges_to_update: Vec<_> = self.anchors[remove.0 as usize]
            .edges
            .iter()
            .map(|eh| eh.edge)
            .collect();

        for edge_id in edges_to_update {
            let edge = &mut self.edges[edge_id.0 as usize];
            if edge.start == remove {
                edge.start = keep;
            }
            if edge.end == remove {
                edge.end = keep;
            }

            // Add edge to keep's list if not already there
            if !self.anchors[keep.0 as usize]
                .edges
                .iter()
                .any(|eh| eh.edge == edge_id)
            {
                self.anchors[keep.0 as usize].edges.push(EdgeHandle {
                    edge: edge_id,
                    handle: Vec2::ZERO,
                });
            }
        }

        // Clear removed anchor's edges
        self.anchors[remove.0 as usize].edges.clear();
    }

    /// Computes the bounding box of the network.
    pub fn bounds(&self) -> (Vec2, Vec2) {
        if self.anchors.is_empty() {
            return (Vec2::ZERO, Vec2::ZERO);
        }

        let mut min = Vec2::splat(f32::MAX);
        let mut max = Vec2::splat(f32::MIN);

        for anchor in &self.anchors {
            min = min.min(anchor.position);
            max = max.max(anchor.position);
        }

        // Include control points
        for edge in &self.edges {
            if edge.start.is_null() {
                continue;
            }
            if edge.edge_type == EdgeType::Cubic {
                min = min.min(edge.control1).min(edge.control2);
                max = max.max(edge.control1).max(edge.control2);
            }
        }

        (min, max)
    }
}

/// Trait extension for Vec2 to compute angle (used internally for region detection).
trait Vec2Angle {
    fn angle(self) -> f32;
}

impl Vec2Angle for Vec2 {
    fn angle(self) -> f32 {
        self.y.atan2(self.x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_anchors_and_line() {
        let mut net = VectorNetwork::new();
        let a0 = net.add_anchor(Vec2::ZERO);
        let a1 = net.add_anchor(Vec2::new(100.0, 0.0));
        let e = net.add_line(a0, a1);

        assert_eq!(net.anchor_count(), 2);
        assert_eq!(net.edge_count(), 1);
        assert_eq!(net.anchor_valence(a0), 1);
        assert_eq!(net.anchor_valence(a1), 1);
        assert!(net.is_edge_valid(e));
    }

    #[test]
    fn test_add_cubic() {
        let mut net = VectorNetwork::new();
        let a0 = net.add_anchor(Vec2::ZERO);
        let a1 = net.add_anchor(Vec2::new(100.0, 0.0));
        let e = net.add_cubic(a0, Vec2::new(30.0, 50.0), Vec2::new(70.0, 50.0), a1);

        assert_eq!(net.edges[e.0 as usize].edge_type, EdgeType::Cubic);
    }

    #[test]
    fn test_triangle_region() {
        let mut net = VectorNetwork::new();
        let a0 = net.add_anchor(Vec2::new(0.0, 0.0));
        let a1 = net.add_anchor(Vec2::new(100.0, 0.0));
        let a2 = net.add_anchor(Vec2::new(50.0, 100.0));

        net.add_line(a0, a1);
        net.add_line(a1, a2);
        net.add_line(a2, a0);

        let regions = net.find_regions();
        // Should find at least one region (the triangle)
        assert!(!regions.is_empty());
    }

    #[test]
    fn test_edges_to_path() {
        let mut net = VectorNetwork::new();
        let a0 = net.add_anchor(Vec2::ZERO);
        let a1 = net.add_anchor(Vec2::new(100.0, 0.0));
        let a2 = net.add_anchor(Vec2::new(100.0, 100.0));

        let e0 = net.add_line(a0, a1);
        let e1 = net.add_line(a1, a2);

        let path = net.edges_to_path(&[e0, e1]);
        assert_eq!(path.len(), 3); // move, line, line
    }

    #[test]
    fn test_region_to_path() {
        let mut net = VectorNetwork::new();
        let a0 = net.add_anchor(Vec2::new(0.0, 0.0));
        let a1 = net.add_anchor(Vec2::new(100.0, 0.0));
        let a2 = net.add_anchor(Vec2::new(50.0, 100.0));

        net.add_line(a0, a1);
        net.add_line(a1, a2);
        net.add_line(a2, a0);

        let regions = net.find_regions();
        if !regions.is_empty() {
            let path = regions[0].to_path(&net);
            assert!(path.len() >= 4); // move, 3 lines, close
        }
    }

    #[test]
    fn test_split_edge() {
        let mut net = VectorNetwork::new();
        let a0 = net.add_anchor(Vec2::ZERO);
        let a1 = net.add_anchor(Vec2::new(100.0, 0.0));
        let e = net.add_line(a0, a1);

        let a_mid = net.split_edge(e, 0.5);

        // Original edge should be removed, new edges created
        assert!(!net.is_edge_valid(e));
        assert_eq!(net.anchor_count(), 3);

        // New anchor should be at midpoint
        let mid_pos = net.anchor_position(a_mid);
        assert!((mid_pos.x - 50.0).abs() < 0.01);
    }

    #[test]
    fn test_remove_edge() {
        let mut net = VectorNetwork::new();
        let a0 = net.add_anchor(Vec2::ZERO);
        let a1 = net.add_anchor(Vec2::new(100.0, 0.0));
        let e = net.add_line(a0, a1);

        net.remove_edge(e);

        assert!(!net.is_edge_valid(e));
        assert_eq!(net.anchor_valence(a0), 0);
        assert_eq!(net.anchor_valence(a1), 0);
    }

    #[test]
    fn test_anchor_neighbors() {
        let mut net = VectorNetwork::new();
        let a0 = net.add_anchor(Vec2::ZERO);
        let a1 = net.add_anchor(Vec2::new(100.0, 0.0));
        let a2 = net.add_anchor(Vec2::new(0.0, 100.0));

        net.add_line(a0, a1);
        net.add_line(a0, a2);

        let neighbors = net.anchor_neighbors(a0);
        assert_eq!(neighbors.len(), 2);
    }

    #[test]
    fn test_bounds() {
        let mut net = VectorNetwork::new();
        net.add_anchor(Vec2::new(10.0, 20.0));
        net.add_anchor(Vec2::new(100.0, 50.0));
        net.add_anchor(Vec2::new(50.0, 200.0));

        let (min, max) = net.bounds();
        assert!((min.x - 10.0).abs() < 0.01);
        assert!((min.y - 20.0).abs() < 0.01);
        assert!((max.x - 100.0).abs() < 0.01);
        assert!((max.y - 200.0).abs() < 0.01);
    }

    #[test]
    fn test_to_paths() {
        let mut net = VectorNetwork::new();
        let a0 = net.add_anchor(Vec2::ZERO);
        let a1 = net.add_anchor(Vec2::new(100.0, 0.0));
        let a2 = net.add_anchor(Vec2::new(100.0, 100.0));

        net.add_line(a0, a1);
        net.add_line(a1, a2);

        let paths = net.to_paths();
        assert!(!paths.is_empty());
    }

    #[test]
    fn test_weld_anchors() {
        let mut net = VectorNetwork::new();
        let a0 = net.add_anchor(Vec2::ZERO);
        let a1 = net.add_anchor(Vec2::new(100.0, 0.0));
        let a2 = net.add_anchor(Vec2::new(100.0, 0.0)); // Same position as a1
        let a3 = net.add_anchor(Vec2::new(200.0, 0.0));

        net.add_line(a0, a1);
        net.add_line(a2, a3);

        net.weld_anchors(a1, a2);

        // a2 should have no edges
        assert_eq!(net.anchor_valence(a2), 0);
    }
}
