//! Vertex weight painting tools.
//!
//! Tools for editing and manipulating vertex weights used in skinning/rigging:
//! - Weight smoothing (Laplacian smoothing)
//! - Heat diffusion (solving heat equation on mesh)
//! - Weight normalization
//! - Weight operations (add, subtract, scale, blur)
//!
//! # Example
//!
//! ```
//! use rhizome_resin_mesh::{UvSphere, VertexWeights, smooth_weights};
//!
//! let sphere = UvSphere::new(1.0, 16, 8).apply();
//! let mut weights = VertexWeights::new(sphere.vertex_count(), 2);
//!
//! // Set initial weights
//! weights.set(0, 0, 1.0); // Vertex 0, bone 0 = 1.0
//! weights.set(1, 1, 1.0); // Vertex 1, bone 1 = 1.0
//!
//! // Smooth the weights
//! let smoothed = smooth_weights(&sphere, &weights, 1.0, 3);
//! ```

use crate::Mesh;
use glam::Vec3;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// Vertex weights for skinning.
///
/// Stores per-vertex weights for multiple influences (bones).
#[derive(Debug, Clone)]
pub struct VertexWeights {
    /// Weights stored as [vertex][influence].
    weights: Vec<Vec<f32>>,
    /// Number of influences (bones).
    influence_count: usize,
}

impl VertexWeights {
    /// Creates new weights with the given vertex and influence count.
    pub fn new(vertex_count: usize, influence_count: usize) -> Self {
        Self {
            weights: vec![vec![0.0; influence_count]; vertex_count],
            influence_count,
        }
    }

    /// Returns the number of vertices.
    pub fn vertex_count(&self) -> usize {
        self.weights.len()
    }

    /// Returns the number of influences.
    pub fn influence_count(&self) -> usize {
        self.influence_count
    }

    /// Gets the weight for a vertex and influence.
    pub fn get(&self, vertex: usize, influence: usize) -> f32 {
        self.weights
            .get(vertex)
            .and_then(|v| v.get(influence))
            .copied()
            .unwrap_or(0.0)
    }

    /// Sets the weight for a vertex and influence.
    pub fn set(&mut self, vertex: usize, influence: usize, weight: f32) {
        if vertex < self.weights.len() && influence < self.influence_count {
            self.weights[vertex][influence] = weight;
        }
    }

    /// Adds to the weight for a vertex and influence.
    pub fn add(&mut self, vertex: usize, influence: usize, delta: f32) {
        if vertex < self.weights.len() && influence < self.influence_count {
            self.weights[vertex][influence] += delta;
        }
    }

    /// Gets all weights for a vertex.
    pub fn vertex_weights(&self, vertex: usize) -> &[f32] {
        self.weights
            .get(vertex)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    /// Gets all weights for an influence (bone).
    pub fn influence_weights(&self, influence: usize) -> Vec<f32> {
        self.weights
            .iter()
            .map(|v| v.get(influence).copied().unwrap_or(0.0))
            .collect()
    }

    /// Sets all weights for a single influence from a slice.
    pub fn set_influence(&mut self, influence: usize, weights: &[f32]) {
        for (vertex, &weight) in weights.iter().enumerate() {
            self.set(vertex, influence, weight);
        }
    }

    /// Normalizes weights so they sum to 1.0 for each vertex.
    pub fn normalize(&mut self) {
        for weights in &mut self.weights {
            let sum: f32 = weights.iter().sum();
            if sum > f32::EPSILON {
                for w in weights {
                    *w /= sum;
                }
            }
        }
    }

    /// Clamps all weights to [0, 1] range.
    pub fn clamp(&mut self) {
        for weights in &mut self.weights {
            for w in weights {
                *w = w.clamp(0.0, 1.0);
            }
        }
    }

    /// Clears all weights to zero.
    pub fn clear(&mut self) {
        for weights in &mut self.weights {
            weights.fill(0.0);
        }
    }

    /// Fills all vertices with a constant weight for an influence.
    pub fn fill_influence(&mut self, influence: usize, weight: f32) {
        for weights in &mut self.weights {
            if influence < weights.len() {
                weights[influence] = weight;
            }
        }
    }
}

/// Builds vertex adjacency from mesh topology.
fn build_adjacency(mesh: &Mesh) -> Vec<HashSet<usize>> {
    let mut adj = vec![HashSet::new(); mesh.positions.len()];

    for tri in mesh.indices.chunks(3) {
        let i0 = tri[0] as usize;
        let i1 = tri[1] as usize;
        let i2 = tri[2] as usize;

        adj[i0].insert(i1);
        adj[i0].insert(i2);
        adj[i1].insert(i0);
        adj[i1].insert(i2);
        adj[i2].insert(i0);
        adj[i2].insert(i1);
    }

    adj
}

/// Smooths vertex weights using Laplacian smoothing.
///
/// # Arguments
/// * `mesh` - The mesh to smooth on
/// * `weights` - The weights to smooth
/// * `factor` - Smoothing factor (0-1, higher = more smoothing)
/// * `iterations` - Number of smoothing iterations
pub fn smooth_weights(
    mesh: &Mesh,
    weights: &VertexWeights,
    factor: f32,
    iterations: usize,
) -> VertexWeights {
    let adj = build_adjacency(mesh);
    let factor = factor.clamp(0.0, 1.0);

    let mut result = weights.clone();

    for _ in 0..iterations {
        let current = result.clone();

        for v in 0..mesh.positions.len() {
            if adj[v].is_empty() {
                continue;
            }

            for influence in 0..weights.influence_count() {
                // Compute average of neighbors
                let neighbor_sum: f32 = adj[v].iter().map(|&n| current.get(n, influence)).sum();
                let neighbor_avg = neighbor_sum / adj[v].len() as f32;

                // Blend current value with neighbor average
                let current_val = current.get(v, influence);
                let smoothed = current_val * (1.0 - factor) + neighbor_avg * factor;
                result.set(v, influence, smoothed);
            }
        }
    }

    result
}

/// Smooths weights for a single influence.
pub fn smooth_influence(mesh: &Mesh, weights: &[f32], factor: f32, iterations: usize) -> Vec<f32> {
    let adj = build_adjacency(mesh);
    let factor = factor.clamp(0.0, 1.0);

    let mut result = weights.to_vec();

    for _ in 0..iterations {
        let current = result.clone();

        for v in 0..mesh.positions.len() {
            if adj[v].is_empty() || v >= current.len() {
                continue;
            }

            let neighbor_sum: f32 = adj[v]
                .iter()
                .filter(|&&n| n < current.len())
                .map(|&n| current[n])
                .sum();
            let neighbor_count = adj[v].iter().filter(|&&n| n < current.len()).count();

            if neighbor_count > 0 {
                let neighbor_avg = neighbor_sum / neighbor_count as f32;
                result[v] = current[v] * (1.0 - factor) + neighbor_avg * factor;
            }
        }
    }

    result
}

/// Heat diffusion configuration.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct HeatDiffusionConfig {
    /// Time step for diffusion simulation.
    pub time_step: f32,
    /// Number of iterations.
    pub iterations: usize,
    /// Diffusion rate (higher = faster diffusion).
    pub diffusion_rate: f32,
}

impl Default for HeatDiffusionConfig {
    fn default() -> Self {
        Self {
            time_step: 0.1,
            iterations: 50,
            diffusion_rate: 1.0,
        }
    }
}

/// Performs heat diffusion on vertex weights.
///
/// This simulates heat spreading from hot vertices (weight = 1.0) to cold ones (weight = 0.0).
/// Useful for creating smooth weight transitions from seed vertices.
///
/// # Arguments
/// * `mesh` - The mesh to diffuse on
/// * `initial` - Initial heat values (vertex -> heat)
/// * `config` - Diffusion parameters
pub fn heat_diffusion(
    mesh: &Mesh,
    initial: &HashMap<usize, f32>,
    config: &HeatDiffusionConfig,
) -> Vec<f32> {
    let adj = build_adjacency(mesh);
    let n = mesh.positions.len();

    // Initialize heat values
    let mut heat = vec![0.0f32; n];
    for (&v, &h) in initial {
        if v < n {
            heat[v] = h;
        }
    }

    // Fixed vertices (heat sources)
    let fixed: HashSet<usize> = initial.keys().copied().collect();

    // Run diffusion simulation
    for _ in 0..config.iterations {
        let current = heat.clone();

        for v in 0..n {
            // Don't update fixed vertices
            if fixed.contains(&v) {
                continue;
            }

            if adj[v].is_empty() {
                continue;
            }

            // Laplacian: average of neighbors minus current
            let neighbor_avg: f32 =
                adj[v].iter().map(|&n| current[n]).sum::<f32>() / adj[v].len() as f32;

            let laplacian = neighbor_avg - current[v];
            heat[v] = current[v] + config.diffusion_rate * config.time_step * laplacian;
            heat[v] = heat[v].clamp(0.0, 1.0);
        }
    }

    heat
}

/// Computes automatic weights from bones using heat diffusion.
///
/// # Arguments
/// * `mesh` - The mesh to weight
/// * `bone_vertices` - Map of bone index -> seed vertex indices
/// * `config` - Diffusion configuration
pub fn compute_automatic_weights(
    mesh: &Mesh,
    bone_vertices: &HashMap<usize, Vec<usize>>,
    config: &HeatDiffusionConfig,
) -> VertexWeights {
    let bone_count = bone_vertices.keys().max().map(|&m| m + 1).unwrap_or(0);
    let mut weights = VertexWeights::new(mesh.positions.len(), bone_count);

    // Compute heat diffusion from each bone's seed vertices
    for (&bone, vertices) in bone_vertices {
        let mut initial = HashMap::new();
        for &v in vertices {
            initial.insert(v, 1.0);
        }

        let heat = heat_diffusion(mesh, &initial, config);
        weights.set_influence(bone, &heat);
    }

    // Normalize so weights sum to 1
    weights.normalize();

    weights
}

/// Applies weight gradient based on vertex positions.
///
/// Creates a gradient from 0 to 1 along an axis.
pub fn gradient_weights(mesh: &Mesh, axis: Vec3, min_pos: f32, max_pos: f32) -> Vec<f32> {
    let axis = axis.normalize();
    let range = max_pos - min_pos;

    if range.abs() < f32::EPSILON {
        return vec![0.5; mesh.positions.len()];
    }

    mesh.positions
        .iter()
        .map(|pos| {
            let proj = pos.dot(axis);
            ((proj - min_pos) / range).clamp(0.0, 1.0)
        })
        .collect()
}

/// Creates radial weights from a center point.
///
/// Weight is 1.0 at center and falls off with distance.
pub fn radial_weights(mesh: &Mesh, center: Vec3, radius: f32, falloff: f32) -> Vec<f32> {
    if radius <= 0.0 {
        return vec![0.0; mesh.positions.len()];
    }

    mesh.positions
        .iter()
        .map(|pos| {
            let dist = pos.distance(center);
            let normalized = (dist / radius).clamp(0.0, 1.0);
            (1.0 - normalized).powf(falloff)
        })
        .collect()
}

/// Blurs weights using a distance-weighted average.
pub fn blur_weights(mesh: &Mesh, weights: &[f32], radius: f32) -> Vec<f32> {
    if radius <= 0.0 {
        return weights.to_vec();
    }

    let mut result = vec![0.0f32; mesh.positions.len()];

    for (i, pos_i) in mesh.positions.iter().enumerate() {
        let mut sum = 0.0f32;
        let mut weight_sum = 0.0f32;

        for (j, pos_j) in mesh.positions.iter().enumerate() {
            let dist = pos_i.distance(*pos_j);
            if dist <= radius {
                let w = 1.0 - dist / radius;
                sum += weights.get(j).copied().unwrap_or(0.0) * w;
                weight_sum += w;
            }
        }

        if weight_sum > 0.0 {
            result[i] = sum / weight_sum;
        }
    }

    result
}

/// Transfers weights from one mesh to another using nearest point.
pub fn transfer_weights_nearest(
    source_mesh: &Mesh,
    source_weights: &VertexWeights,
    target_mesh: &Mesh,
) -> VertexWeights {
    let mut result = VertexWeights::new(
        target_mesh.positions.len(),
        source_weights.influence_count(),
    );

    for (i, target_pos) in target_mesh.positions.iter().enumerate() {
        // Find nearest vertex in source mesh
        let mut nearest = 0;
        let mut nearest_dist = f32::MAX;

        for (j, source_pos) in source_mesh.positions.iter().enumerate() {
            let dist = target_pos.distance(*source_pos);
            if dist < nearest_dist {
                nearest_dist = dist;
                nearest = j;
            }
        }

        // Copy weights from nearest vertex
        for influence in 0..source_weights.influence_count() {
            result.set(i, influence, source_weights.get(nearest, influence));
        }
    }

    result
}

/// Limits the number of influences per vertex.
///
/// Keeps the N highest weights and normalizes.
pub fn limit_influences(weights: &mut VertexWeights, max_influences: usize) {
    for v in 0..weights.vertex_count() {
        // Get all weights for this vertex with their indices
        let mut indexed: Vec<(usize, f32)> = (0..weights.influence_count())
            .map(|i| (i, weights.get(v, i)))
            .collect();

        // Sort by weight descending
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Zero out weights beyond the limit
        for (i, _) in indexed.iter().skip(max_influences) {
            weights.set(v, *i, 0.0);
        }
    }

    // Renormalize
    weights.normalize();
}

/// Scales weights by a factor for selected vertices.
pub fn scale_weights(
    weights: &mut VertexWeights,
    influence: usize,
    vertices: &[usize],
    factor: f32,
) {
    for &v in vertices {
        let current = weights.get(v, influence);
        weights.set(v, influence, (current * factor).clamp(0.0, 1.0));
    }
}

/// Inverts weights for an influence.
pub fn invert_weights(weights: &mut VertexWeights, influence: usize) {
    for v in 0..weights.vertex_count() {
        let current = weights.get(v, influence);
        weights.set(v, influence, 1.0 - current);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::UvSphere;

    #[test]
    fn test_vertex_weights_basic() {
        let mut weights = VertexWeights::new(4, 2);

        weights.set(0, 0, 1.0);
        weights.set(0, 1, 0.0);
        weights.set(1, 0, 0.5);
        weights.set(1, 1, 0.5);

        assert_eq!(weights.get(0, 0), 1.0);
        assert_eq!(weights.get(0, 1), 0.0);
        assert_eq!(weights.get(1, 0), 0.5);
    }

    #[test]
    fn test_normalize_weights() {
        let mut weights = VertexWeights::new(2, 3);

        weights.set(0, 0, 2.0);
        weights.set(0, 1, 2.0);
        weights.set(0, 2, 0.0);

        weights.normalize();

        assert!((weights.get(0, 0) - 0.5).abs() < 0.001);
        assert!((weights.get(0, 1) - 0.5).abs() < 0.001);
        assert_eq!(weights.get(0, 2), 0.0);
    }

    #[test]
    fn test_smooth_weights() {
        let mesh = UvSphere::new(1.0, 8, 4).apply();
        let mut weights = VertexWeights::new(mesh.vertex_count(), 1);

        // Set one vertex to 1.0, rest to 0.0
        weights.set(0, 0, 1.0);

        let smoothed = smooth_weights(&mesh, &weights, 0.5, 3);

        // Original vertex should have reduced weight
        assert!(smoothed.get(0, 0) < 1.0);
        // Some neighbors should have gained weight
        let total: f32 = (0..mesh.vertex_count()).map(|v| smoothed.get(v, 0)).sum();
        assert!(total > 0.0);
    }

    #[test]
    fn test_heat_diffusion() {
        let mesh = UvSphere::new(1.0, 8, 4).apply();

        let mut initial = HashMap::new();
        initial.insert(0, 1.0);

        let heat = heat_diffusion(&mesh, &initial, &HeatDiffusionConfig::default());

        // Source should be 1.0
        assert_eq!(heat[0], 1.0);
        // Heat should spread to neighbors
        let avg: f32 = heat.iter().sum::<f32>() / heat.len() as f32;
        assert!(avg > 0.0);
        assert!(avg < 1.0);
    }

    #[test]
    fn test_gradient_weights() {
        let mesh = UvSphere::new(1.0, 8, 4).apply();

        let weights = gradient_weights(&mesh, Vec3::Y, -1.0, 1.0);

        // Should have values between 0 and 1
        for &w in &weights {
            assert!(w >= 0.0 && w <= 1.0);
        }
    }

    #[test]
    fn test_radial_weights() {
        let mesh = UvSphere::new(1.0, 8, 4).apply();

        let weights = radial_weights(&mesh, Vec3::ZERO, 2.0, 1.0);

        // All weights should be positive (sphere is radius 1)
        for &w in &weights {
            assert!(w >= 0.0);
        }
    }

    #[test]
    fn test_limit_influences() {
        let mut weights = VertexWeights::new(1, 4);

        weights.set(0, 0, 0.4);
        weights.set(0, 1, 0.3);
        weights.set(0, 2, 0.2);
        weights.set(0, 3, 0.1);

        limit_influences(&mut weights, 2);

        // Only 2 highest should remain (normalized)
        let sum: f32 = (0..4).map(|i| weights.get(0, i)).sum();
        assert!((sum - 1.0).abs() < 0.001);

        // Lower influences should be zero
        assert!(weights.get(0, 2).abs() < 0.001);
        assert!(weights.get(0, 3).abs() < 0.001);
    }

    #[test]
    fn test_invert_weights() {
        let mut weights = VertexWeights::new(2, 1);
        weights.set(0, 0, 0.3);
        weights.set(1, 0, 0.8);

        invert_weights(&mut weights, 0);

        assert!((weights.get(0, 0) - 0.7).abs() < 0.001);
        assert!((weights.get(1, 0) - 0.2).abs() < 0.001);
    }

    #[test]
    fn test_transfer_weights_nearest() {
        let source = UvSphere::new(1.0, 4, 2).apply();
        let mut source_weights = VertexWeights::new(source.vertex_count(), 1);
        for v in 0..source.vertex_count() {
            source_weights.set(v, 0, (v as f32) / (source.vertex_count() as f32));
        }

        let target = UvSphere::new(1.0, 8, 4).apply();
        let transferred = transfer_weights_nearest(&source, &source_weights, &target);

        // Should have same influence count
        assert_eq!(transferred.influence_count(), 1);
        // Should have weights for all target vertices
        assert_eq!(transferred.vertex_count(), target.vertex_count());
    }

    #[test]
    fn test_automatic_weights() {
        let mesh = UvSphere::new(1.0, 8, 4).apply();

        let mut bone_vertices = HashMap::new();
        bone_vertices.insert(0, vec![0]);
        bone_vertices.insert(1, vec![mesh.vertex_count() / 2]);

        let weights =
            compute_automatic_weights(&mesh, &bone_vertices, &HeatDiffusionConfig::default());

        // Should have 2 influences
        assert_eq!(weights.influence_count(), 2);

        // Weights should sum to 1 for each vertex (unless both are near zero)
        for v in 0..mesh.vertex_count() {
            let sum: f32 = (0..2).map(|i| weights.get(v, i)).sum();
            // Either sums to 1 (normalized) or both weights are very small
            assert!(
                (sum - 1.0).abs() < 0.01 || sum < 0.01,
                "vertex {}: sum = {}",
                v,
                sum
            );
        }
    }

    #[test]
    fn test_influence_weights() {
        let mut weights = VertexWeights::new(4, 2);
        weights.set(0, 0, 1.0);
        weights.set(1, 0, 0.5);
        weights.set(2, 0, 0.0);
        weights.set(3, 0, 0.25);

        let influence_0 = weights.influence_weights(0);
        assert_eq!(influence_0, vec![1.0, 0.5, 0.0, 0.25]);
    }
}
