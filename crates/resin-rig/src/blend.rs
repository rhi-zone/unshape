//! Animation blending and layering.
//!
//! Provides tools for combining multiple animations:
//! - Crossfade blending between clips
//! - Additive animation layers
//! - 1D/2D blend trees for parametric blending

use crate::{AnimationClip, AnimationTarget, Interpolate, Lerp, Transform};
use glam::Vec3;
use std::collections::HashMap;

/// How animations are combined.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BlendMode {
    /// Replace the previous value completely.
    #[default]
    Replace,
    /// Add to the previous value (for additive animations).
    Additive,
}

/// A sampled pose from an animation.
#[derive(Debug, Clone, Default)]
pub struct AnimationPose {
    /// Transform values by target.
    pub transforms: HashMap<AnimationTarget, Transform>,
    /// Float values by target.
    pub floats: HashMap<AnimationTarget, f32>,
    /// Vec3 values by target.
    pub vec3s: HashMap<AnimationTarget, Vec3>,
}

impl AnimationPose {
    /// Creates an empty pose.
    pub fn new() -> Self {
        Self::default()
    }

    /// Samples an animation clip at a given time into this pose.
    pub fn sample_clip(&mut self, clip: &AnimationClip, time: f32) {
        for (target, track) in &clip.transform_tracks {
            self.transforms.insert(target.clone(), track.sample(time));
        }
        for (target, track) in &clip.float_tracks {
            self.floats.insert(target.clone(), track.sample(time));
        }
        for (target, track) in &clip.vec3_tracks {
            self.vec3s.insert(target.clone(), track.sample(time));
        }
    }

    /// Blends another pose into this one.
    pub fn blend(&mut self, other: &AnimationPose, weight: f32, mode: BlendMode) {
        match mode {
            BlendMode::Replace => self.blend_replace(other, weight),
            BlendMode::Additive => self.blend_additive(other, weight),
        }
    }

    /// Blends with replace mode (lerp towards other).
    fn blend_replace(&mut self, other: &AnimationPose, weight: f32) {
        // Blend existing transforms
        for (target, other_transform) in &other.transforms {
            if let Some(self_transform) = self.transforms.get_mut(target) {
                *self_transform = self_transform.lerp_to(other_transform, weight);
            } else {
                // New target - lerp from identity
                self.transforms.insert(
                    target.clone(),
                    Transform::IDENTITY.lerp_to(other_transform, weight),
                );
            }
        }

        // Blend existing floats
        for (target, other_value) in &other.floats {
            if let Some(self_value) = self.floats.get_mut(target) {
                *self_value = self_value.lerp_to(other_value, weight);
            } else {
                self.floats
                    .insert(target.clone(), 0.0_f32.lerp_to(other_value, weight));
            }
        }

        // Blend existing vec3s
        for (target, other_value) in &other.vec3s {
            if let Some(self_value) = self.vec3s.get_mut(target) {
                *self_value = self_value.lerp_to(other_value, weight);
            } else {
                self.vec3s
                    .insert(target.clone(), Vec3::ZERO.lerp_to(other_value, weight));
            }
        }
    }

    /// Blends with additive mode (add scaled values).
    fn blend_additive(&mut self, other: &AnimationPose, weight: f32) {
        for (target, other_transform) in &other.transforms {
            if let Some(self_transform) = self.transforms.get_mut(target) {
                // Additive: add weighted difference from identity
                let delta = Transform {
                    translation: other_transform.translation * weight,
                    rotation: glam::Quat::IDENTITY.slerp(other_transform.rotation, weight),
                    scale: Vec3::ONE.lerp(other_transform.scale, weight),
                };
                *self_transform = self_transform.then(&delta);
            } else {
                // Apply additive to identity
                self.transforms.insert(
                    target.clone(),
                    Transform {
                        translation: other_transform.translation * weight,
                        rotation: glam::Quat::IDENTITY.slerp(other_transform.rotation, weight),
                        scale: Vec3::ONE.lerp(other_transform.scale, weight),
                    },
                );
            }
        }

        for (target, other_value) in &other.floats {
            let delta = other_value * weight;
            if let Some(self_value) = self.floats.get_mut(target) {
                *self_value += delta;
            } else {
                self.floats.insert(target.clone(), delta);
            }
        }

        for (target, other_value) in &other.vec3s {
            let delta = *other_value * weight;
            if let Some(self_value) = self.vec3s.get_mut(target) {
                *self_value += delta;
            } else {
                self.vec3s.insert(target.clone(), delta);
            }
        }
    }

    /// Clears all values.
    pub fn clear(&mut self) {
        self.transforms.clear();
        self.floats.clear();
        self.vec3s.clear();
    }
}

/// An animation layer with its own clip, weight, and blend mode.
#[derive(Debug, Clone)]
pub struct AnimationLayer {
    /// The animation clip for this layer.
    pub clip: AnimationClip,
    /// Current playback time.
    pub time: f32,
    /// Layer weight (0.0 = no effect, 1.0 = full effect).
    pub weight: f32,
    /// How this layer blends with layers below it.
    pub blend_mode: BlendMode,
    /// Playback speed.
    pub speed: f32,
    /// Whether this layer loops.
    pub looping: bool,
    /// Whether this layer is active.
    pub active: bool,
    /// Optional mask - if set, only these targets are affected.
    pub mask: Option<Vec<AnimationTarget>>,
}

impl AnimationLayer {
    /// Creates a new layer with the given clip.
    pub fn new(clip: AnimationClip) -> Self {
        Self {
            clip,
            time: 0.0,
            weight: 1.0,
            blend_mode: BlendMode::Replace,
            speed: 1.0,
            looping: true,
            active: true,
            mask: None,
        }
    }

    /// Creates an additive layer.
    pub fn additive(clip: AnimationClip) -> Self {
        Self {
            blend_mode: BlendMode::Additive,
            ..Self::new(clip)
        }
    }

    /// Sets the weight.
    pub fn with_weight(mut self, weight: f32) -> Self {
        self.weight = weight;
        self
    }

    /// Sets the blend mode.
    pub fn with_blend_mode(mut self, mode: BlendMode) -> Self {
        self.blend_mode = mode;
        self
    }

    /// Sets a mask of targets this layer affects.
    pub fn with_mask(mut self, targets: Vec<AnimationTarget>) -> Self {
        self.mask = Some(targets);
        self
    }

    /// Updates the layer's playback time.
    pub fn update(&mut self, dt: f32) {
        if !self.active {
            return;
        }

        let duration = self.clip.duration();
        if duration <= 0.0 {
            return;
        }

        self.time += dt * self.speed;

        if self.looping {
            while self.time >= duration {
                self.time -= duration;
            }
            while self.time < 0.0 {
                self.time += duration;
            }
        } else {
            self.time = self.time.clamp(0.0, duration);
        }
    }

    /// Samples this layer into a pose.
    pub fn sample(&self) -> AnimationPose {
        let mut pose = AnimationPose::new();
        if self.active && self.weight > 0.0 {
            pose.sample_clip(&self.clip, self.time);

            // Apply mask if present
            if let Some(mask) = &self.mask {
                pose.transforms.retain(|k, _| mask.contains(k));
                pose.floats.retain(|k, _| mask.contains(k));
                pose.vec3s.retain(|k, _| mask.contains(k));
            }
        }
        pose
    }
}

/// A stack of animation layers that are blended together.
#[derive(Debug, Clone, Default)]
pub struct AnimationStack {
    /// Layers in order (bottom to top).
    pub layers: Vec<AnimationLayer>,
}

impl AnimationStack {
    /// Creates an empty animation stack.
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a layer to the top of the stack.
    pub fn push(&mut self, layer: AnimationLayer) {
        self.layers.push(layer);
    }

    /// Adds a base layer (first layer, replace mode).
    pub fn with_base(mut self, clip: AnimationClip) -> Self {
        self.layers.push(AnimationLayer::new(clip));
        self
    }

    /// Adds an additive layer on top.
    pub fn with_additive(mut self, clip: AnimationClip, weight: f32) -> Self {
        self.layers
            .push(AnimationLayer::additive(clip).with_weight(weight));
        self
    }

    /// Updates all layers.
    pub fn update(&mut self, dt: f32) {
        for layer in &mut self.layers {
            layer.update(dt);
        }
    }

    /// Evaluates the stack and returns the final blended pose.
    pub fn evaluate(&self) -> AnimationPose {
        let mut result = AnimationPose::new();

        for layer in &self.layers {
            if !layer.active || layer.weight <= 0.0 {
                continue;
            }

            let layer_pose = layer.sample();
            result.blend(&layer_pose, layer.weight, layer.blend_mode);
        }

        result
    }

    /// Gets a mutable reference to a layer by index.
    pub fn layer_mut(&mut self, index: usize) -> Option<&mut AnimationLayer> {
        self.layers.get_mut(index)
    }
}

/// A node in a blend tree.
#[derive(Debug, Clone)]
pub enum BlendNode {
    /// A single animation clip.
    Clip {
        /// The animation clip to play.
        clip: AnimationClip,
        /// Current playback time in seconds.
        time: f32,
        /// Playback speed multiplier.
        speed: f32,
    },
    /// Linear blend between nodes based on a 1D parameter.
    Blend1D {
        /// Child nodes with their threshold values (threshold, node).
        children: Vec<(f32, Box<BlendNode>)>,
        /// Current blend parameter value.
        parameter: f32,
    },
    /// 2D blend between nodes based on two parameters.
    Blend2D {
        /// Child nodes with their 2D positions (x, y, node).
        children: Vec<(f32, f32, Box<BlendNode>)>,
        /// Current X parameter value.
        parameter_x: f32,
        /// Current Y parameter value.
        parameter_y: f32,
    },
    /// Additive blend layering.
    Additive {
        /// Base animation node.
        base: Box<BlendNode>,
        /// Additive layer to apply on top.
        additive: Box<BlendNode>,
        /// Blend weight for the additive layer (0-1).
        weight: f32,
    },
}

impl BlendNode {
    /// Creates a clip node.
    pub fn clip(clip: AnimationClip) -> Self {
        Self::Clip {
            clip,
            time: 0.0,
            speed: 1.0,
        }
    }

    /// Creates a 1D blend node.
    pub fn blend_1d(children: Vec<(f32, BlendNode)>) -> Self {
        Self::Blend1D {
            children: children
                .into_iter()
                .map(|(t, n)| (t, Box::new(n)))
                .collect(),
            parameter: 0.0,
        }
    }

    /// Creates a 2D blend node.
    pub fn blend_2d(children: Vec<(f32, f32, BlendNode)>) -> Self {
        Self::Blend2D {
            children: children
                .into_iter()
                .map(|(x, y, n)| (x, y, Box::new(n)))
                .collect(),
            parameter_x: 0.0,
            parameter_y: 0.0,
        }
    }

    /// Creates an additive blend node.
    pub fn additive(base: BlendNode, additive: BlendNode, weight: f32) -> Self {
        Self::Additive {
            base: Box::new(base),
            additive: Box::new(additive),
            weight,
        }
    }

    /// Updates playback time.
    pub fn update(&mut self, dt: f32) {
        match self {
            BlendNode::Clip { clip, time, speed } => {
                let duration = clip.duration();
                if duration > 0.0 {
                    *time += dt * *speed;
                    while *time >= duration {
                        *time -= duration;
                    }
                }
            }
            BlendNode::Blend1D { children, .. } => {
                for (_, child) in children {
                    child.update(dt);
                }
            }
            BlendNode::Blend2D { children, .. } => {
                for (_, _, child) in children {
                    child.update(dt);
                }
            }
            BlendNode::Additive { base, additive, .. } => {
                base.update(dt);
                additive.update(dt);
            }
        }
    }

    /// Sets the 1D blend parameter.
    pub fn set_parameter(&mut self, value: f32) {
        if let BlendNode::Blend1D { parameter, .. } = self {
            *parameter = value;
        }
    }

    /// Sets the 2D blend parameters.
    pub fn set_parameters(&mut self, x: f32, y: f32) {
        if let BlendNode::Blend2D {
            parameter_x,
            parameter_y,
            ..
        } = self
        {
            *parameter_x = x;
            *parameter_y = y;
        }
    }

    /// Evaluates the blend tree and returns a pose.
    pub fn evaluate(&self) -> AnimationPose {
        match self {
            BlendNode::Clip { clip, time, .. } => {
                let mut pose = AnimationPose::new();
                pose.sample_clip(clip, *time);
                pose
            }
            BlendNode::Blend1D {
                children,
                parameter,
            } => evaluate_blend_1d(children, *parameter),
            BlendNode::Blend2D {
                children,
                parameter_x,
                parameter_y,
            } => evaluate_blend_2d(children, *parameter_x, *parameter_y),
            BlendNode::Additive {
                base,
                additive,
                weight,
            } => {
                let mut pose = base.evaluate();
                let additive_pose = additive.evaluate();
                pose.blend(&additive_pose, *weight, BlendMode::Additive);
                pose
            }
        }
    }
}

/// Evaluates a 1D blend between children.
fn evaluate_blend_1d(children: &[(f32, Box<BlendNode>)], parameter: f32) -> AnimationPose {
    if children.is_empty() {
        return AnimationPose::new();
    }
    if children.len() == 1 {
        return children[0].1.evaluate();
    }

    // Find the two surrounding thresholds
    let mut sorted: Vec<_> = children.iter().collect();
    sorted.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    // Clamp to range
    if parameter <= sorted[0].0 {
        return sorted[0].1.evaluate();
    }
    if parameter >= sorted.last().unwrap().0 {
        return sorted.last().unwrap().1.evaluate();
    }

    // Find surrounding nodes
    for i in 0..sorted.len() - 1 {
        let (t0, node0) = sorted[i];
        let (t1, node1) = sorted[i + 1];

        if parameter >= *t0 && parameter <= *t1 {
            let blend = (parameter - t0) / (t1 - t0);
            let mut pose0 = node0.evaluate();
            let pose1 = node1.evaluate();
            pose0.blend(&pose1, blend, BlendMode::Replace);
            return pose0;
        }
    }

    sorted.last().unwrap().1.evaluate()
}

/// Evaluates a 2D blend using inverse distance weighting.
fn evaluate_blend_2d(
    children: &[(f32, f32, Box<BlendNode>)],
    param_x: f32,
    param_y: f32,
) -> AnimationPose {
    if children.is_empty() {
        return AnimationPose::new();
    }
    if children.len() == 1 {
        return children[0].2.evaluate();
    }

    // Inverse distance weighting
    let mut weights = Vec::with_capacity(children.len());
    let mut total_weight = 0.0;

    for (x, y, _) in children {
        let dx = param_x - x;
        let dy = param_y - y;
        let dist_sq = dx * dx + dy * dy;

        // Handle exact match
        if dist_sq < 0.0001 {
            let idx = children
                .iter()
                .position(|(cx, cy, _)| (*cx - x).abs() < 0.001 && (*cy - y).abs() < 0.001)
                .unwrap();
            return children[idx].2.evaluate();
        }

        let w = 1.0 / dist_sq;
        weights.push(w);
        total_weight += w;
    }

    // Normalize weights and blend
    let mut result = AnimationPose::new();
    let mut first = true;

    for (i, (_, _, node)) in children.iter().enumerate() {
        let weight = weights[i] / total_weight;
        let pose = node.evaluate();

        if first {
            result = pose;
            first = false;
        } else {
            result.blend(&pose, weight, BlendMode::Replace);
        }
    }

    result
}

/// Crossfade between two animations.
#[derive(Debug, Clone)]
pub struct Crossfade {
    /// Source animation.
    pub from: AnimationClip,
    /// Target animation.
    pub to: AnimationClip,
    /// Current blend weight (0 = from, 1 = to).
    pub blend: f32,
    /// Fade duration in seconds.
    pub duration: f32,
    /// Current time in the crossfade.
    pub elapsed: f32,
    /// Playback time in source clip.
    pub from_time: f32,
    /// Playback time in target clip.
    pub to_time: f32,
}

impl Crossfade {
    /// Creates a new crossfade.
    pub fn new(from: AnimationClip, to: AnimationClip, duration: f32) -> Self {
        Self {
            from,
            to,
            blend: 0.0,
            duration,
            elapsed: 0.0,
            from_time: 0.0,
            to_time: 0.0,
        }
    }

    /// Starts a crossfade from the current time.
    pub fn start_from(&mut self, from_time: f32) {
        self.from_time = from_time;
        self.to_time = 0.0;
        self.elapsed = 0.0;
        self.blend = 0.0;
    }

    /// Updates the crossfade.
    pub fn update(&mut self, dt: f32) {
        self.elapsed += dt;
        self.blend = (self.elapsed / self.duration).clamp(0.0, 1.0);

        // Update playback times
        let from_duration = self.from.duration();
        let to_duration = self.to.duration();

        if from_duration > 0.0 {
            self.from_time += dt;
            if self.from_time >= from_duration {
                self.from_time -= from_duration;
            }
        }

        if to_duration > 0.0 {
            self.to_time += dt;
            if self.to_time >= to_duration {
                self.to_time -= to_duration;
            }
        }
    }

    /// Returns true if the crossfade is complete.
    pub fn is_complete(&self) -> bool {
        self.elapsed >= self.duration
    }

    /// Evaluates the crossfade.
    pub fn evaluate(&self) -> AnimationPose {
        let mut from_pose = AnimationPose::new();
        from_pose.sample_clip(&self.from, self.from_time);

        let mut to_pose = AnimationPose::new();
        to_pose.sample_clip(&self.to, self.to_time);

        from_pose.blend(&to_pose, self.blend, BlendMode::Replace);
        from_pose
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Track;

    fn make_test_clip(name: &str, end_y: f32) -> AnimationClip {
        let mut clip = AnimationClip::new(name);
        let mut track = Track::new();
        track.add(0.0, Transform::from_translation(Vec3::ZERO));
        track.add(1.0, Transform::from_translation(Vec3::new(0.0, end_y, 0.0)));
        clip.add_transform_track(AnimationTarget::BoneTransform(0), track);
        clip
    }

    #[test]
    fn test_animation_pose_blend_replace() {
        let clip_a = make_test_clip("a", 10.0);
        let clip_b = make_test_clip("b", 20.0);

        let mut pose_a = AnimationPose::new();
        pose_a.sample_clip(&clip_a, 1.0); // y = 10

        let mut pose_b = AnimationPose::new();
        pose_b.sample_clip(&clip_b, 1.0); // y = 20

        pose_a.blend(&pose_b, 0.5, BlendMode::Replace);

        let transform = pose_a
            .transforms
            .get(&AnimationTarget::BoneTransform(0))
            .unwrap();
        assert!((transform.translation.y - 15.0).abs() < 0.001);
    }

    #[test]
    fn test_animation_layer() {
        let clip = make_test_clip("walk", 5.0);
        let mut layer = AnimationLayer::new(clip);

        layer.update(0.5);
        let pose = layer.sample();

        let transform = pose
            .transforms
            .get(&AnimationTarget::BoneTransform(0))
            .unwrap();
        assert!((transform.translation.y - 2.5).abs() < 0.001);
    }

    #[test]
    fn test_animation_stack() {
        let base_clip = make_test_clip("base", 10.0);
        let additive_clip = make_test_clip("additive", 5.0);

        let mut stack = AnimationStack::new()
            .with_base(base_clip)
            .with_additive(additive_clip, 0.5);

        // Move to midpoint (0.5s) to avoid loop wrap
        stack.update(0.5);

        let pose = stack.evaluate();
        let transform = pose
            .transforms
            .get(&AnimationTarget::BoneTransform(0))
            .unwrap();

        // Base at t=0.5: y = 5.0
        // Additive at t=0.5: y = 2.5, with weight 0.5 = 1.25 added
        // Total = 5.0 + 1.25 = 6.25
        assert!((transform.translation.y - 6.25).abs() < 0.1);
    }

    #[test]
    fn test_blend_node_1d() {
        let walk = make_test_clip("walk", 5.0);
        let run = make_test_clip("run", 15.0);

        let mut blend = BlendNode::blend_1d(vec![
            (0.0, BlendNode::clip(walk)),
            (1.0, BlendNode::clip(run)),
        ]);

        // Update to midpoint to avoid loop wrap
        blend.update(0.5);

        // Test at parameter 0.5 (blend between walk and run)
        blend.set_parameter(0.5);
        let pose = blend.evaluate();
        let transform = pose
            .transforms
            .get(&AnimationTarget::BoneTransform(0))
            .unwrap();

        // Walk at t=0.5: y = 2.5, Run at t=0.5: y = 7.5
        // Blend 0.5 between them: 2.5 + 0.5 * (7.5 - 2.5) = 5.0
        assert!((transform.translation.y - 5.0).abs() < 0.1);
    }

    #[test]
    fn test_crossfade() {
        let from = make_test_clip("from", 0.0);
        let to = make_test_clip("to", 10.0);

        let mut fade = Crossfade::new(from, to, 0.5);

        // At start
        let pose = fade.evaluate();
        let t = pose
            .transforms
            .get(&AnimationTarget::BoneTransform(0))
            .unwrap();
        assert!((t.translation.y - 0.0).abs() < 0.1);

        // Update halfway through both clips AND halfway through fade
        fade.from_time = 0.5;
        fade.to_time = 0.5;
        fade.blend = 0.5;

        let pose = fade.evaluate();
        let t = pose
            .transforms
            .get(&AnimationTarget::BoneTransform(0))
            .unwrap();
        // from at 0.5 = 0.0, to at 0.5 = 5.0, blend 0.5 = 2.5
        assert!((t.translation.y - 2.5).abs() < 0.1);
    }

    #[test]
    fn test_layer_mask() {
        let mut clip = AnimationClip::new("masked");

        let mut track0 = Track::new();
        track0.add(0.0, Transform::from_translation(Vec3::new(1.0, 0.0, 0.0)));
        clip.add_transform_track(AnimationTarget::BoneTransform(0), track0);

        let mut track1 = Track::new();
        track1.add(0.0, Transform::from_translation(Vec3::new(2.0, 0.0, 0.0)));
        clip.add_transform_track(AnimationTarget::BoneTransform(1), track1);

        let layer = AnimationLayer::new(clip).with_mask(vec![AnimationTarget::BoneTransform(0)]);

        let pose = layer.sample();

        assert!(
            pose.transforms
                .contains_key(&AnimationTarget::BoneTransform(0))
        );
        assert!(
            !pose
                .transforms
                .contains_key(&AnimationTarget::BoneTransform(1))
        );
    }
}
