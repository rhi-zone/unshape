//! Animation clips and tracks with keyframe interpolation.
//!
//! Provides a system for storing and sampling animated values over time.

use crate::Transform3D;
use glam::Vec3;
pub use unshape_easing::Lerp;
use std::collections::HashMap;

/// Interpolation method between keyframes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Interpolation {
    /// Hold previous value until next keyframe.
    Step,
    /// Linear interpolation.
    #[default]
    Linear,
    /// Cubic bezier interpolation (not yet implemented, falls back to linear).
    Cubic,
}

/// A value that can be interpolated in animation tracks.
///
/// This extends [`Lerp`] with `Clone + Default` requirements needed for
/// animation sampling (cloning keyframe values, providing defaults for empty tracks).
///
/// Blanket implementation is provided for any `T: Lerp + Clone + Default`.
pub trait Interpolate: Lerp + Clone + Default {}

impl<T: Lerp + Clone + Default> Interpolate for T {}

/// A keyframe with a time and value.
#[derive(Debug, Clone)]
pub struct Keyframe<T> {
    /// Time in seconds.
    pub time: f32,
    /// Value at this keyframe.
    pub value: T,
    /// Interpolation to next keyframe.
    pub interpolation: Interpolation,
}

impl<T> Keyframe<T> {
    /// Creates a new keyframe.
    pub fn new(time: f32, value: T) -> Self {
        Self {
            time,
            value,
            interpolation: Interpolation::Linear,
        }
    }

    /// Creates a keyframe with step interpolation.
    pub fn step(time: f32, value: T) -> Self {
        Self {
            time,
            value,
            interpolation: Interpolation::Step,
        }
    }
}

/// A track of keyframes for a single animated value.
#[derive(Debug, Clone, Default)]
pub struct Track<T> {
    keyframes: Vec<Keyframe<T>>,
}

impl<T: Interpolate> Track<T> {
    /// Creates an empty track.
    pub fn new() -> Self {
        Self {
            keyframes: Vec::new(),
        }
    }

    /// Creates a track from keyframes.
    pub fn from_keyframes(keyframes: Vec<Keyframe<T>>) -> Self {
        let mut track = Self { keyframes };
        track.sort();
        track
    }

    /// Adds a keyframe and keeps the track sorted.
    pub fn add_keyframe(&mut self, keyframe: Keyframe<T>) {
        self.keyframes.push(keyframe);
        self.sort();
    }

    /// Adds a keyframe at a time with a value.
    pub fn add(&mut self, time: f32, value: T) {
        self.add_keyframe(Keyframe::new(time, value));
    }

    /// Returns the number of keyframes.
    pub fn len(&self) -> usize {
        self.keyframes.len()
    }

    /// Returns true if empty.
    pub fn is_empty(&self) -> bool {
        self.keyframes.is_empty()
    }

    /// Returns the duration of this track.
    pub fn duration(&self) -> f32 {
        self.keyframes.last().map(|k| k.time).unwrap_or(0.0)
    }

    /// Sorts keyframes by time.
    fn sort(&mut self) {
        self.keyframes
            .sort_by(|a, b| a.time.partial_cmp(&b.time).unwrap());
    }

    /// Samples the track at a given time.
    pub fn sample(&self, time: f32) -> T {
        if self.keyframes.is_empty() {
            return T::default();
        }

        // Before first keyframe
        if time <= self.keyframes[0].time {
            return self.keyframes[0].value.clone();
        }

        // After last keyframe
        if time >= self.keyframes.last().unwrap().time {
            return self.keyframes.last().unwrap().value.clone();
        }

        // Find surrounding keyframes
        for i in 0..self.keyframes.len() - 1 {
            let curr = &self.keyframes[i];
            let next = &self.keyframes[i + 1];

            if time >= curr.time && time < next.time {
                let t = (time - curr.time) / (next.time - curr.time);

                return match curr.interpolation {
                    Interpolation::Step => curr.value.clone(),
                    Interpolation::Linear | Interpolation::Cubic => {
                        curr.value.lerp_to(&next.value, t)
                    }
                };
            }
        }

        // Fallback (shouldn't reach here)
        self.keyframes.last().unwrap().value.clone()
    }
}

/// Target for an animation track.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum AnimationTarget {
    /// Bone transform by bone index.
    BoneTransform(u32),
    /// Bone transform by name.
    BoneTransformNamed(String),
    /// Morph target weight by index.
    MorphWeight(usize),
    /// Morph target weight by name.
    MorphWeightNamed(String),
    /// Custom property.
    Property(String),
}

/// An animation clip containing multiple tracks.
#[derive(Debug, Clone, Default)]
pub struct AnimationClip {
    /// Clip name.
    pub name: String,
    /// Transform tracks (for bones).
    pub transform_tracks: HashMap<AnimationTarget, Track<Transform3D>>,
    /// Float tracks (for morph weights, etc).
    pub float_tracks: HashMap<AnimationTarget, Track<f32>>,
    /// Vec3 tracks (for positions, colors, etc).
    pub vec3_tracks: HashMap<AnimationTarget, Track<Vec3>>,
}

impl AnimationClip {
    /// Creates a new animation clip.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            ..Default::default()
        }
    }

    /// Adds a transform track.
    pub fn add_transform_track(&mut self, target: AnimationTarget, track: Track<Transform3D>) {
        self.transform_tracks.insert(target, track);
    }

    /// Adds a float track.
    pub fn add_float_track(&mut self, target: AnimationTarget, track: Track<f32>) {
        self.float_tracks.insert(target, track);
    }

    /// Adds a vec3 track.
    pub fn add_vec3_track(&mut self, target: AnimationTarget, track: Track<Vec3>) {
        self.vec3_tracks.insert(target, track);
    }

    /// Returns the duration of the clip (max of all tracks).
    pub fn duration(&self) -> f32 {
        let mut max = 0.0f32;
        for track in self.transform_tracks.values() {
            max = max.max(track.duration());
        }
        for track in self.float_tracks.values() {
            max = max.max(track.duration());
        }
        for track in self.vec3_tracks.values() {
            max = max.max(track.duration());
        }
        max
    }

    /// Samples a transform track at a given time.
    pub fn sample_transform(&self, target: &AnimationTarget, time: f32) -> Option<Transform3D> {
        self.transform_tracks.get(target).map(|t| t.sample(time))
    }

    /// Samples a float track at a given time.
    pub fn sample_float(&self, target: &AnimationTarget, time: f32) -> Option<f32> {
        self.float_tracks.get(target).map(|t| t.sample(time))
    }

    /// Samples a vec3 track at a given time.
    pub fn sample_vec3(&self, target: &AnimationTarget, time: f32) -> Option<Vec3> {
        self.vec3_tracks.get(target).map(|t| t.sample(time))
    }
}

/// Animation playback state.
#[derive(Debug, Clone)]
pub struct AnimationPlayer {
    /// Current time in the animation.
    pub time: f32,
    /// Playback speed multiplier.
    pub speed: f32,
    /// Whether the animation loops.
    pub looping: bool,
    /// Whether the animation is playing.
    pub playing: bool,
}

impl Default for AnimationPlayer {
    fn default() -> Self {
        Self {
            time: 0.0,
            speed: 1.0,
            looping: true,
            playing: true,
        }
    }
}

impl AnimationPlayer {
    /// Creates a new animation player.
    pub fn new() -> Self {
        Self::default()
    }

    /// Advances the animation by delta time.
    pub fn update(&mut self, dt: f32, duration: f32) {
        if !self.playing || duration <= 0.0 {
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
            if self.time >= duration || self.time <= 0.0 {
                self.playing = false;
            }
        }
    }

    /// Resets playback to the beginning.
    pub fn reset(&mut self) {
        self.time = 0.0;
        self.playing = true;
    }

    /// Seeks to a specific time.
    pub fn seek(&mut self, time: f32) {
        self.time = time;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_track_sample_empty() {
        let track: Track<f32> = Track::new();
        assert_eq!(track.sample(0.0), 0.0);
    }

    #[test]
    fn test_track_sample_single() {
        let mut track = Track::new();
        track.add(0.0, 5.0);

        assert_eq!(track.sample(-1.0), 5.0);
        assert_eq!(track.sample(0.0), 5.0);
        assert_eq!(track.sample(1.0), 5.0);
    }

    #[test]
    fn test_track_sample_linear() {
        let mut track: Track<f32> = Track::new();
        track.add(0.0, 0.0);
        track.add(1.0, 10.0);

        assert_eq!(track.sample(0.0), 0.0);
        assert!((track.sample(0.5) - 5.0).abs() < 0.001);
        assert_eq!(track.sample(1.0), 10.0);
    }

    #[test]
    fn test_track_sample_step() {
        let mut track = Track::new();
        track.add_keyframe(Keyframe::step(0.0, 0.0));
        track.add_keyframe(Keyframe::step(1.0, 10.0));

        assert_eq!(track.sample(0.0), 0.0);
        assert_eq!(track.sample(0.5), 0.0); // Step holds value
        assert_eq!(track.sample(1.0), 10.0);
    }

    #[test]
    fn test_track_duration() {
        let mut track = Track::new();
        track.add(0.0, 0.0);
        track.add(2.5, 10.0);

        assert!((track.duration() - 2.5).abs() < 0.001);
    }

    #[test]
    fn test_transform_interpolation() {
        let mut track = Track::new();
        track.add(0.0, Transform3D::from_translation(Vec3::ZERO));
        track.add(
            1.0,
            Transform3D::from_translation(Vec3::new(10.0, 0.0, 0.0)),
        );

        let mid = track.sample(0.5);
        assert!((mid.translation.x - 5.0).abs() < 0.001);
    }

    #[test]
    fn test_animation_clip() {
        let mut clip = AnimationClip::new("walk");

        let mut track = Track::new();
        track.add(0.0, Transform3D::IDENTITY);
        track.add(1.0, Transform3D::from_translation(Vec3::Y));

        clip.add_transform_track(AnimationTarget::BoneTransform(0), track);

        assert!((clip.duration() - 1.0).abs() < 0.001);

        let sample = clip
            .sample_transform(&AnimationTarget::BoneTransform(0), 0.5)
            .unwrap();
        assert!((sample.translation.y - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_animation_player_update() {
        let mut player = AnimationPlayer::new();
        player.looping = false;

        player.update(0.5, 1.0);
        assert!((player.time - 0.5).abs() < 0.001);

        player.update(0.6, 1.0);
        assert!((player.time - 1.0).abs() < 0.001);
        assert!(!player.playing);
    }

    #[test]
    fn test_animation_player_loop() {
        let mut player = AnimationPlayer::new();
        player.looping = true;

        player.update(1.5, 1.0);
        assert!((player.time - 0.5).abs() < 0.001);
    }
}
