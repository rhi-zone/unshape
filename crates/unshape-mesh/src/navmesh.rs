//! Navigation mesh generation and pathfinding.
//!
//! Provides walkable surface generation and A* pathfinding for game AI.
//!
//! Operations are serializable structs with `apply` methods.
//! See `docs/design/ops-as-values.md`.

use crate::Mesh;
use glam::{Vec2, Vec3};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use std::collections::{BinaryHeap, HashMap, HashSet};

/// Generates a navigation mesh from a floor mesh.
///
/// Filters triangles based on slope and other walkability criteria
/// to create a navmesh for pathfinding.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dynop", derive(unshape_op::Op))]
#[cfg_attr(feature = "dynop", op(input = Mesh, output = NavMesh))]
pub struct GenerateNavMesh {
    /// Cell size for rasterization.
    pub cell_size: f32,
    /// Cell height for vertical sampling.
    pub cell_height: f32,
    /// Minimum walkable height (agent height).
    pub agent_height: f32,
    /// Maximum walkable slope in degrees.
    pub max_slope: f32,
    /// Agent radius for obstacle margin.
    pub agent_radius: f32,
    /// Maximum step height.
    pub max_step_height: f32,
}

impl Default for GenerateNavMesh {
    fn default() -> Self {
        Self {
            cell_size: 0.3,
            cell_height: 0.2,
            agent_height: 2.0,
            max_slope: 45.0,
            agent_radius: 0.5,
            max_step_height: 0.5,
        }
    }
}

impl GenerateNavMesh {
    /// Applies this operation to generate a navmesh from a floor mesh.
    pub fn apply(&self, mesh: &Mesh) -> NavMesh {
        NavMesh::from_floor(mesh, self)
    }
}

/// Backwards-compatible type alias.
pub type NavMeshConfig = GenerateNavMesh;

/// A navigation polygon (convex walkable area).
#[derive(Debug, Clone)]
pub struct NavPolygon {
    /// Vertex indices into the navmesh vertices array.
    pub vertices: Vec<usize>,
    /// Neighbor polygon indices (one per edge, -1 if no neighbor).
    pub neighbors: Vec<i32>,
    /// Center point for quick distance calculations.
    pub center: Vec3,
    /// Polygon area.
    pub area: f32,
}

impl NavPolygon {
    /// Creates a new nav polygon.
    pub fn new(vertices: Vec<usize>, positions: &[Vec3]) -> Self {
        let center = if vertices.is_empty() {
            Vec3::ZERO
        } else {
            let sum: Vec3 = vertices.iter().map(|&i| positions[i]).sum();
            sum / vertices.len() as f32
        };

        let area = Self::compute_area(&vertices, positions);

        Self {
            vertices,
            neighbors: Vec::new(),
            center,
            area,
        }
    }

    fn compute_area(vertices: &[usize], positions: &[Vec3]) -> f32 {
        if vertices.len() < 3 {
            return 0.0;
        }

        // Compute area using cross product (project to XZ plane)
        let mut area = 0.0;
        let n = vertices.len();
        for i in 0..n {
            let p1 = positions[vertices[i]];
            let p2 = positions[vertices[(i + 1) % n]];
            area += p1.x * p2.z - p2.x * p1.z;
        }
        (area / 2.0).abs()
    }
}

/// A navigation mesh for pathfinding.
#[derive(Debug, Clone)]
pub struct NavMesh {
    /// Vertex positions.
    pub vertices: Vec<Vec3>,
    /// Navigation polygons.
    pub polygons: Vec<NavPolygon>,
    /// Bounding box minimum.
    pub bounds_min: Vec3,
    /// Bounding box maximum.
    pub bounds_max: Vec3,
}

impl NavMesh {
    /// Creates an empty navmesh.
    pub fn new() -> Self {
        Self {
            vertices: Vec::new(),
            polygons: Vec::new(),
            bounds_min: Vec3::ZERO,
            bounds_max: Vec3::ZERO,
        }
    }

    /// Adds a vertex and returns its index.
    pub fn add_vertex(&mut self, pos: Vec3) -> usize {
        let idx = self.vertices.len();
        self.vertices.push(pos);
        idx
    }

    /// Adds a polygon.
    pub fn add_polygon(&mut self, vertices: Vec<usize>) -> usize {
        let idx = self.polygons.len();
        let poly = NavPolygon::new(vertices, &self.vertices);
        self.polygons.push(poly);
        idx
    }

    /// Computes bounds from vertices.
    pub fn compute_bounds(&mut self) {
        if self.vertices.is_empty() {
            self.bounds_min = Vec3::ZERO;
            self.bounds_max = Vec3::ZERO;
            return;
        }

        self.bounds_min = Vec3::splat(f32::MAX);
        self.bounds_max = Vec3::splat(f32::MIN);

        for v in &self.vertices {
            self.bounds_min = self.bounds_min.min(*v);
            self.bounds_max = self.bounds_max.max(*v);
        }
    }

    /// Finds the polygon containing a point (on XZ plane).
    pub fn find_polygon(&self, point: Vec3) -> Option<usize> {
        for (i, poly) in self.polygons.iter().enumerate() {
            if self.point_in_polygon(point, poly) {
                return Some(i);
            }
        }
        None
    }

    /// Checks if a point is inside a polygon (XZ plane).
    fn point_in_polygon(&self, point: Vec3, poly: &NavPolygon) -> bool {
        let n = poly.vertices.len();
        if n < 3 {
            return false;
        }

        // Ray casting algorithm
        let mut inside = false;
        let px = point.x;
        let pz = point.z;

        let mut j = n - 1;
        for i in 0..n {
            let vi = self.vertices[poly.vertices[i]];
            let vj = self.vertices[poly.vertices[j]];

            if ((vi.z > pz) != (vj.z > pz))
                && (px < (vj.x - vi.x) * (pz - vi.z) / (vj.z - vi.z) + vi.x)
            {
                inside = !inside;
            }
            j = i;
        }

        inside
    }

    /// Finds the nearest polygon to a point.
    pub fn find_nearest_polygon(&self, point: Vec3) -> Option<usize> {
        if self.polygons.is_empty() {
            return None;
        }

        let mut best_idx = 0;
        let mut best_dist = f32::MAX;

        for (i, poly) in self.polygons.iter().enumerate() {
            let dist = (poly.center - point).length_squared();
            if dist < best_dist {
                best_dist = dist;
                best_idx = i;
            }
        }

        Some(best_idx)
    }

    /// Builds neighbor connections between polygons.
    pub fn build_connections(&mut self) {
        // Clear existing neighbors
        for poly in &mut self.polygons {
            poly.neighbors = vec![-1; poly.vertices.len()];
        }

        // Build edge map (collect all edges first)
        let mut edge_map: HashMap<(usize, usize), (usize, usize)> = HashMap::new();
        let mut connections: Vec<(usize, usize, usize, usize)> = Vec::new();

        for poly_idx in 0..self.polygons.len() {
            let verts = self.polygons[poly_idx].vertices.clone();
            for edge_idx in 0..verts.len() {
                let v1 = verts[edge_idx];
                let v2 = verts[(edge_idx + 1) % verts.len()];

                // Use canonical edge (smaller vertex first)
                let edge = if v1 < v2 { (v1, v2) } else { (v2, v1) };

                if let Some(&(other_poly, other_edge)) = edge_map.get(&edge) {
                    // Found shared edge - record the connection
                    connections.push((poly_idx, edge_idx, other_poly, other_edge));
                } else {
                    edge_map.insert(edge, (poly_idx, edge_idx));
                }
            }
        }

        // Apply connections
        for (poly_a, edge_a, poly_b, edge_b) in connections {
            self.polygons[poly_a].neighbors[edge_a] = poly_b as i32;
            self.polygons[poly_b].neighbors[edge_b] = poly_a as i32;
        }
    }

    /// Creates a simple navmesh from a floor mesh.
    pub fn from_floor(mesh: &Mesh, config: &GenerateNavMesh) -> Self {
        let mut navmesh = NavMesh::new();

        // Filter walkable triangles
        for i in (0..mesh.indices.len()).step_by(3) {
            let i0 = mesh.indices[i] as usize;
            let i1 = mesh.indices[i + 1] as usize;
            let i2 = mesh.indices[i + 2] as usize;

            let v0 = mesh.positions[i0];
            let v1 = mesh.positions[i1];
            let v2 = mesh.positions[i2];

            // Check slope
            let edge1 = v1 - v0;
            let edge2 = v2 - v0;
            let normal = edge1.cross(edge2).normalize_or_zero();
            let up = Vec3::Y;
            let slope_cos = normal.dot(up).abs();
            let slope_angle = slope_cos.acos().to_degrees();

            if slope_angle <= config.max_slope {
                // Add triangle as walkable polygon
                let vi0 = navmesh.add_vertex(v0);
                let vi1 = navmesh.add_vertex(v1);
                let vi2 = navmesh.add_vertex(v2);
                navmesh.add_polygon(vec![vi0, vi1, vi2]);
            }
        }

        navmesh.compute_bounds();
        navmesh.build_connections();
        navmesh
    }
}

impl Default for NavMesh {
    fn default() -> Self {
        Self::new()
    }
}

/// A node in the A* search.
#[derive(Clone)]
struct PathNode {
    poly_idx: usize,
    f_cost: f32,
}

impl PartialEq for PathNode {
    fn eq(&self, other: &Self) -> bool {
        self.poly_idx == other.poly_idx
    }
}

impl Eq for PathNode {}

impl Ord for PathNode {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        other
            .f_cost
            .partial_cmp(&self.f_cost)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

impl PartialOrd for PathNode {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// Result of pathfinding.
#[derive(Debug, Clone)]
pub struct NavPath {
    /// Polygon indices along the path.
    pub polygons: Vec<usize>,
    /// Waypoints for following.
    pub waypoints: Vec<Vec3>,
    /// Total path length.
    pub length: f32,
}

impl NavPath {
    /// Returns true if the path is empty/invalid.
    pub fn is_empty(&self) -> bool {
        self.waypoints.is_empty()
    }
}

/// Finds a path between two points on the navmesh using A*.
pub fn find_path(navmesh: &NavMesh, start: Vec3, end: Vec3) -> Option<NavPath> {
    let start_poly = navmesh
        .find_polygon(start)
        .or_else(|| navmesh.find_nearest_polygon(start))?;
    let end_poly = navmesh
        .find_polygon(end)
        .or_else(|| navmesh.find_nearest_polygon(end))?;

    if start_poly == end_poly {
        return Some(NavPath {
            polygons: vec![start_poly],
            waypoints: vec![start, end],
            length: (end - start).length(),
        });
    }

    // A* search
    let mut open_set = BinaryHeap::new();
    let mut came_from: HashMap<usize, usize> = HashMap::new();
    let mut g_score: HashMap<usize, f32> = HashMap::new();
    let mut closed_set: HashSet<usize> = HashSet::new();

    let heuristic = |poly_idx: usize| -> f32 {
        (navmesh.polygons[poly_idx].center - navmesh.polygons[end_poly].center).length()
    };

    g_score.insert(start_poly, 0.0);
    open_set.push(PathNode {
        poly_idx: start_poly,
        f_cost: heuristic(start_poly),
    });

    while let Some(current) = open_set.pop() {
        if current.poly_idx == end_poly {
            // Reconstruct path
            let mut poly_path = vec![end_poly];
            let mut current_idx = end_poly;
            while let Some(&prev) = came_from.get(&current_idx) {
                poly_path.push(prev);
                current_idx = prev;
            }
            poly_path.reverse();

            // Generate waypoints (polygon centers + endpoints)
            let mut waypoints = vec![start];
            for &poly_idx in &poly_path[1..poly_path.len() - 1] {
                waypoints.push(navmesh.polygons[poly_idx].center);
            }
            waypoints.push(end);

            // Calculate total length
            let length: f32 = waypoints.windows(2).map(|w| (w[1] - w[0]).length()).sum();

            return Some(NavPath {
                polygons: poly_path,
                waypoints,
                length,
            });
        }

        if closed_set.contains(&current.poly_idx) {
            continue;
        }
        closed_set.insert(current.poly_idx);

        // Explore neighbors
        for &neighbor_idx in &navmesh.polygons[current.poly_idx].neighbors {
            if neighbor_idx < 0 {
                continue;
            }
            let neighbor = neighbor_idx as usize;

            if closed_set.contains(&neighbor) {
                continue;
            }

            let current_center = navmesh.polygons[current.poly_idx].center;
            let neighbor_center = navmesh.polygons[neighbor].center;
            let tentative_g =
                g_score[&current.poly_idx] + (neighbor_center - current_center).length();

            let current_g = g_score.get(&neighbor).copied().unwrap_or(f32::MAX);
            if tentative_g < current_g {
                came_from.insert(neighbor, current.poly_idx);
                g_score.insert(neighbor, tentative_g);
                let f = tentative_g + heuristic(neighbor);
                open_set.push(PathNode {
                    poly_idx: neighbor,
                    f_cost: f,
                });
            }
        }
    }

    None // No path found
}

/// Simplifies a path using string pulling (funnel algorithm).
pub fn smooth_path(navmesh: &NavMesh, path: &NavPath) -> Vec<Vec3> {
    if path.waypoints.len() <= 2 {
        return path.waypoints.clone();
    }

    // Simple line-of-sight based smoothing
    let mut smoothed = vec![path.waypoints[0]];
    let mut current = 0;

    while current < path.waypoints.len() - 1 {
        let mut furthest = current + 1;

        // Find furthest visible point
        for i in current + 2..path.waypoints.len() {
            if can_see(navmesh, path.waypoints[current], path.waypoints[i]) {
                furthest = i;
            }
        }

        smoothed.push(path.waypoints[furthest]);
        current = furthest;
    }

    smoothed
}

/// Checks if there's line of sight between two points.
fn can_see(navmesh: &NavMesh, from: Vec3, to: Vec3) -> bool {
    // Simple check: sample along the line and ensure we stay on navmesh
    let steps = ((to - from).length() / 0.5).ceil() as usize;
    if steps == 0 {
        return true;
    }

    for i in 0..=steps {
        let t = i as f32 / steps as f32;
        let pos = from.lerp(to, t);
        if navmesh.find_polygon(pos).is_none() {
            return false;
        }
    }

    true
}

/// Creates a simple grid-based navmesh.
pub fn create_grid_navmesh(min: Vec2, max: Vec2, height: f32, cell_size: f32) -> NavMesh {
    let mut navmesh = NavMesh::new();

    let cols = ((max.x - min.x) / cell_size).ceil() as usize;
    let rows = ((max.y - min.y) / cell_size).ceil() as usize;

    // Create vertices
    let mut vertex_indices = vec![vec![0usize; cols + 1]; rows + 1];
    for row in 0..=rows {
        for col in 0..=cols {
            let x = min.x + col as f32 * cell_size;
            let z = min.y + row as f32 * cell_size;
            vertex_indices[row][col] = navmesh.add_vertex(Vec3::new(x, height, z));
        }
    }

    // Create quads (as two triangles each)
    for row in 0..rows {
        for col in 0..cols {
            let v00 = vertex_indices[row][col];
            let v10 = vertex_indices[row][col + 1];
            let v01 = vertex_indices[row + 1][col];
            let v11 = vertex_indices[row + 1][col + 1];

            // Two triangles per cell
            navmesh.add_polygon(vec![v00, v10, v11]);
            navmesh.add_polygon(vec![v00, v11, v01]);
        }
    }

    navmesh.compute_bounds();
    navmesh.build_connections();
    navmesh
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_navmesh_config_default() {
        let config = GenerateNavMesh::default();
        assert!(config.cell_size > 0.0);
        assert!(config.agent_height > 0.0);
    }

    #[test]
    fn test_navmesh_creation() {
        let mut navmesh = NavMesh::new();
        let v0 = navmesh.add_vertex(Vec3::ZERO);
        let v1 = navmesh.add_vertex(Vec3::X);
        let v2 = navmesh.add_vertex(Vec3::Z);

        navmesh.add_polygon(vec![v0, v1, v2]);

        assert_eq!(navmesh.vertices.len(), 3);
        assert_eq!(navmesh.polygons.len(), 1);
    }

    #[test]
    fn test_grid_navmesh() {
        let navmesh = create_grid_navmesh(Vec2::ZERO, Vec2::splat(10.0), 0.0, 2.0);

        // Should have 5x5 cells = 50 triangles
        assert_eq!(navmesh.polygons.len(), 50);

        // Should have 6x6 vertices
        assert_eq!(navmesh.vertices.len(), 36);
    }

    #[test]
    fn test_find_polygon() {
        let navmesh = create_grid_navmesh(Vec2::ZERO, Vec2::splat(10.0), 0.0, 2.0);

        // Point in center should find a polygon
        let center = Vec3::new(5.0, 0.0, 5.0);
        let poly = navmesh.find_polygon(center);
        assert!(poly.is_some());
    }

    #[test]
    fn test_find_nearest_polygon() {
        let navmesh = create_grid_navmesh(Vec2::ZERO, Vec2::splat(10.0), 0.0, 2.0);

        // Point outside should find nearest
        let outside = Vec3::new(100.0, 0.0, 100.0);
        let poly = navmesh.find_nearest_polygon(outside);
        assert!(poly.is_some());
    }

    #[test]
    fn test_find_path_simple() {
        let navmesh = create_grid_navmesh(Vec2::ZERO, Vec2::splat(10.0), 0.0, 2.0);

        let start = Vec3::new(1.0, 0.0, 1.0);
        let end = Vec3::new(9.0, 0.0, 9.0);

        let path = find_path(&navmesh, start, end);
        assert!(path.is_some());

        let path = path.unwrap();
        assert!(!path.is_empty());
        assert!(path.length > 0.0);
    }

    #[test]
    fn test_find_path_same_polygon() {
        let navmesh = create_grid_navmesh(Vec2::ZERO, Vec2::splat(10.0), 0.0, 2.0);

        let start = Vec3::new(1.0, 0.0, 1.0);
        let end = Vec3::new(1.5, 0.0, 1.5);

        let path = find_path(&navmesh, start, end);
        assert!(path.is_some());

        let path = path.unwrap();
        assert_eq!(path.polygons.len(), 1);
    }

    #[test]
    fn test_smooth_path() {
        let navmesh = create_grid_navmesh(Vec2::ZERO, Vec2::splat(10.0), 0.0, 2.0);

        let start = Vec3::new(1.0, 0.0, 1.0);
        let end = Vec3::new(9.0, 0.0, 9.0);

        let path = find_path(&navmesh, start, end).unwrap();
        let smoothed = smooth_path(&navmesh, &path);

        // Smoothed path should be shorter or equal
        assert!(smoothed.len() <= path.waypoints.len());
    }

    #[test]
    fn test_polygon_connections() {
        let navmesh = create_grid_navmesh(Vec2::ZERO, Vec2::splat(4.0), 0.0, 2.0);

        // Interior polygons should have multiple neighbors
        let mut has_multiple_neighbors = false;
        for poly in &navmesh.polygons {
            let neighbor_count = poly.neighbors.iter().filter(|&&n| n >= 0).count();
            if neighbor_count >= 2 {
                has_multiple_neighbors = true;
                break;
            }
        }
        assert!(has_multiple_neighbors);
    }

    #[test]
    fn test_nav_polygon_area() {
        let mut navmesh = NavMesh::new();
        // 1x1 square at origin
        let v0 = navmesh.add_vertex(Vec3::ZERO);
        let v1 = navmesh.add_vertex(Vec3::new(1.0, 0.0, 0.0));
        let v2 = navmesh.add_vertex(Vec3::new(1.0, 0.0, 1.0));
        let v3 = navmesh.add_vertex(Vec3::new(0.0, 0.0, 1.0));

        navmesh.add_polygon(vec![v0, v1, v2, v3]);

        // Area should be 1.0
        let poly = &navmesh.polygons[0];
        assert!((poly.area - 1.0).abs() < 0.01);
    }
}
