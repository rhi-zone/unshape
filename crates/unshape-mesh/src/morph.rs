//! Morph targets / blend shapes for mesh deformation.
//!
//! Morph targets store per-vertex position (and optionally normal) deltas
//! that can be blended together to deform a mesh.

use glam::Vec3;

/// A single morph target storing vertex deltas.
#[derive(Debug, Clone)]
pub struct MorphTarget {
    /// Target name (e.g., "smile", "blink_left").
    pub name: String,
    /// Position deltas for each vertex.
    pub position_deltas: Vec<Vec3>,
    /// Normal deltas for each vertex (optional).
    pub normal_deltas: Vec<Vec3>,
}

impl MorphTarget {
    /// Creates a new morph target with the given name and vertex count.
    pub fn new(name: impl Into<String>, vertex_count: usize) -> Self {
        Self {
            name: name.into(),
            position_deltas: vec![Vec3::ZERO; vertex_count],
            normal_deltas: Vec::new(),
        }
    }

    /// Creates a morph target from position deltas.
    pub fn from_positions(name: impl Into<String>, deltas: Vec<Vec3>) -> Self {
        Self {
            name: name.into(),
            position_deltas: deltas,
            normal_deltas: Vec::new(),
        }
    }

    /// Creates a morph target from position and normal deltas.
    pub fn from_positions_and_normals(
        name: impl Into<String>,
        positions: Vec<Vec3>,
        normals: Vec<Vec3>,
    ) -> Self {
        Self {
            name: name.into(),
            position_deltas: positions,
            normal_deltas: normals,
        }
    }

    /// Creates a morph target by computing deltas between two meshes.
    pub fn from_difference(
        name: impl Into<String>,
        base_positions: &[Vec3],
        target_positions: &[Vec3],
    ) -> Self {
        assert_eq!(
            base_positions.len(),
            target_positions.len(),
            "vertex counts must match"
        );

        let deltas: Vec<_> = base_positions
            .iter()
            .zip(target_positions)
            .map(|(base, target)| *target - *base)
            .collect();

        Self::from_positions(name, deltas)
    }

    /// Returns the number of vertices this target affects.
    pub fn vertex_count(&self) -> usize {
        self.position_deltas.len()
    }

    /// Returns true if this target has normal deltas.
    pub fn has_normal_deltas(&self) -> bool {
        self.normal_deltas.len() == self.position_deltas.len()
    }

    /// Sets the position delta for a vertex.
    pub fn set_position_delta(&mut self, vertex: usize, delta: Vec3) {
        if let Some(d) = self.position_deltas.get_mut(vertex) {
            *d = delta;
        }
    }

    /// Gets the position delta for a vertex.
    pub fn position_delta(&self, vertex: usize) -> Vec3 {
        self.position_deltas
            .get(vertex)
            .copied()
            .unwrap_or(Vec3::ZERO)
    }
}

/// A collection of morph targets for a mesh.
#[derive(Debug, Clone, Default)]
pub struct MorphTargetSet {
    targets: Vec<MorphTarget>,
}

impl MorphTargetSet {
    /// Creates an empty morph target set.
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a morph target to the set.
    pub fn add(&mut self, target: MorphTarget) -> usize {
        let index = self.targets.len();
        self.targets.push(target);
        index
    }

    /// Returns the number of targets.
    pub fn len(&self) -> usize {
        self.targets.len()
    }

    /// Returns true if empty.
    pub fn is_empty(&self) -> bool {
        self.targets.is_empty()
    }

    /// Gets a target by index.
    pub fn get(&self, index: usize) -> Option<&MorphTarget> {
        self.targets.get(index)
    }

    /// Gets a target by name.
    pub fn get_by_name(&self, name: &str) -> Option<&MorphTarget> {
        self.targets.iter().find(|t| t.name == name)
    }

    /// Finds a target index by name.
    pub fn find_index(&self, name: &str) -> Option<usize> {
        self.targets.iter().position(|t| t.name == name)
    }

    /// Returns all targets.
    pub fn targets(&self) -> &[MorphTarget] {
        &self.targets
    }
}

/// Weights for blending morph targets.
#[derive(Debug, Clone, Default)]
pub struct MorphWeights {
    weights: Vec<f32>,
}

impl MorphWeights {
    /// Creates weights for a morph target set (all zero).
    pub fn new(target_count: usize) -> Self {
        Self {
            weights: vec![0.0; target_count],
        }
    }

    /// Sets a weight by index.
    pub fn set(&mut self, index: usize, weight: f32) {
        if let Some(w) = self.weights.get_mut(index) {
            *w = weight;
        }
    }

    /// Gets a weight by index.
    pub fn get(&self, index: usize) -> f32 {
        self.weights.get(index).copied().unwrap_or(0.0)
    }

    /// Returns all weights.
    pub fn weights(&self) -> &[f32] {
        &self.weights
    }

    /// Resets all weights to zero.
    pub fn reset(&mut self) {
        self.weights.fill(0.0);
    }
}

/// Applies morph targets to vertex positions.
///
/// This modifies the positions in-place by adding weighted target deltas.
pub fn apply_morph_targets(
    positions: &mut [Vec3],
    targets: &MorphTargetSet,
    weights: &MorphWeights,
) {
    for (i, target) in targets.targets().iter().enumerate() {
        let weight = weights.get(i);
        if weight.abs() < f32::EPSILON {
            continue;
        }

        for (pos, delta) in positions.iter_mut().zip(&target.position_deltas) {
            *pos += *delta * weight;
        }
    }
}

/// Applies morph targets to both positions and normals.
pub fn apply_morph_targets_with_normals(
    positions: &mut [Vec3],
    normals: &mut [Vec3],
    targets: &MorphTargetSet,
    weights: &MorphWeights,
) {
    for (i, target) in targets.targets().iter().enumerate() {
        let weight = weights.get(i);
        if weight.abs() < f32::EPSILON {
            continue;
        }

        for (pos, delta) in positions.iter_mut().zip(&target.position_deltas) {
            *pos += *delta * weight;
        }

        if target.has_normal_deltas() {
            for (normal, delta) in normals.iter_mut().zip(&target.normal_deltas) {
                *normal += *delta * weight;
            }
        }
    }

    // Renormalize normals
    for normal in normals.iter_mut() {
        *normal = normal.normalize_or_zero();
    }
}

/// Computes blended positions without modifying the original.
pub fn blend_positions(
    base_positions: &[Vec3],
    targets: &MorphTargetSet,
    weights: &MorphWeights,
) -> Vec<Vec3> {
    let mut result = base_positions.to_vec();
    apply_morph_targets(&mut result, targets, weights);
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_morph_target_creation() {
        let target = MorphTarget::new("test", 4);
        assert_eq!(target.name, "test");
        assert_eq!(target.vertex_count(), 4);
    }

    #[test]
    fn test_morph_from_difference() {
        let base = vec![Vec3::ZERO, Vec3::X, Vec3::Y];
        let target = vec![Vec3::X, Vec3::new(2.0, 0.0, 0.0), Vec3::new(0.0, 2.0, 0.0)];

        let morph = MorphTarget::from_difference("move", &base, &target);

        assert_eq!(morph.position_delta(0), Vec3::X);
        assert_eq!(morph.position_delta(1), Vec3::X);
        assert_eq!(morph.position_delta(2), Vec3::Y);
    }

    #[test]
    fn test_morph_target_set() {
        let mut set = MorphTargetSet::new();

        set.add(MorphTarget::new("a", 4));
        set.add(MorphTarget::new("b", 4));

        assert_eq!(set.len(), 2);
        assert_eq!(set.find_index("b"), Some(1));
        assert_eq!(
            set.get_by_name("a").map(|t| &t.name),
            Some(&"a".to_string())
        );
    }

    #[test]
    fn test_apply_single_target() {
        let mut positions = vec![Vec3::ZERO, Vec3::X];

        let mut set = MorphTargetSet::new();
        set.add(MorphTarget::from_positions("test", vec![Vec3::Y, Vec3::Y]));

        let mut weights = MorphWeights::new(1);
        weights.set(0, 1.0);

        apply_morph_targets(&mut positions, &set, &weights);

        assert_eq!(positions[0], Vec3::Y);
        assert_eq!(positions[1], Vec3::new(1.0, 1.0, 0.0));
    }

    #[test]
    fn test_apply_partial_weight() {
        let mut positions = vec![Vec3::ZERO];

        let mut set = MorphTargetSet::new();
        set.add(MorphTarget::from_positions(
            "test",
            vec![Vec3::new(10.0, 0.0, 0.0)],
        ));

        let mut weights = MorphWeights::new(1);
        weights.set(0, 0.5);

        apply_morph_targets(&mut positions, &set, &weights);

        assert!((positions[0].x - 5.0).abs() < 0.001);
    }

    #[test]
    fn test_blend_multiple_targets() {
        let base = vec![Vec3::ZERO];

        let mut set = MorphTargetSet::new();
        set.add(MorphTarget::from_positions("x", vec![Vec3::X]));
        set.add(MorphTarget::from_positions("y", vec![Vec3::Y]));

        let mut weights = MorphWeights::new(2);
        weights.set(0, 1.0);
        weights.set(1, 1.0);

        let result = blend_positions(&base, &set, &weights);

        assert_eq!(result[0], Vec3::new(1.0, 1.0, 0.0));
    }

    #[test]
    fn test_zero_weight_skipped() {
        let mut positions = vec![Vec3::ZERO];

        let mut set = MorphTargetSet::new();
        set.add(MorphTarget::from_positions("test", vec![Vec3::X]));

        let weights = MorphWeights::new(1); // all zeros

        apply_morph_targets(&mut positions, &set, &weights);

        assert_eq!(positions[0], Vec3::ZERO);
    }
}
