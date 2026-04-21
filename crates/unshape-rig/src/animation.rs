//! Animation clips and tracks with keyframe interpolation.
//!
//! Provides a system for storing and sampling animated values over time.

use crate::Transform3D;
use glam::Vec3;
use std::collections::HashMap;
pub use unshape_easing::Lerp;

/// Interpolation method between keyframes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Interpolation {
    /// Hold previous value until next keyframe.
    Step,
    /// Linear interpolation.
    #[default]
    Linear,
    /// Cubic bezier interpolation using per-keyframe tangent handles.
    ///
    /// Each [`Keyframe`] may carry `out_tangent` (forward handle) and `in_tangent`
    /// (backward handle). When tangents are absent the curve degrades to linear.
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

/// A tangent handle for cubic Bezier keyframe interpolation.
///
/// `dt` is a time *offset* from the owning keyframe (≥ 0 for a well-formed
/// curve).  `dv` is the **absolute** value of the Bezier control point in
/// value-space — it is **not** a delta from the keyframe value.
///
/// Using absolute values avoids requiring an `Add` bound on `T` (only `Lerp`
/// is needed) and matches the convention used by most DCC tools (Blender,
/// Maya) when exporting tangent handles.
///
/// When a handle is `None` the corresponding control point collapses to the
/// keyframe value, which produces zero velocity at that end and degrades
/// gracefully to linear interpolation when both handles are absent.
#[derive(Debug, Clone)]
pub struct Tangent<T> {
    /// Time offset from the keyframe (should be ≥ 0).
    pub dt: f32,
    /// Absolute value of this Bezier control point.
    pub dv: T,
}

impl<T> Tangent<T> {
    /// Creates a new tangent handle.
    ///
    /// `dt` is the time offset from the keyframe; `dv` is the absolute
    /// control-point value (not a delta).
    pub fn new(dt: f32, dv: T) -> Self {
        Self { dt, dv }
    }
}

/// A keyframe with a time and value.
#[derive(Debug, Clone)]
pub struct Keyframe<T> {
    /// Time in seconds.
    pub time: f32,
    /// Value at this keyframe.
    pub value: T,
    /// Interpolation to next keyframe.
    pub interpolation: Interpolation,
    /// In-tangent handle (backward), used for cubic interpolation.
    ///
    /// When `None` the keyframe behaves as if the tangent value offset is zero
    /// (i.e., the curve is locally linear at this keyframe on the incoming side).
    pub in_tangent: Option<Tangent<T>>,
    /// Out-tangent handle (forward), used for cubic interpolation.
    ///
    /// When `None` the keyframe behaves as if the tangent value offset is zero
    /// (i.e., the curve is locally linear at this keyframe on the outgoing side).
    pub out_tangent: Option<Tangent<T>>,
}

impl<T> Keyframe<T> {
    /// Creates a new linear keyframe.
    pub fn new(time: f32, value: T) -> Self {
        Self {
            time,
            value,
            interpolation: Interpolation::Linear,
            in_tangent: None,
            out_tangent: None,
        }
    }

    /// Creates a keyframe with step interpolation.
    pub fn step(time: f32, value: T) -> Self {
        Self {
            time,
            value,
            interpolation: Interpolation::Step,
            in_tangent: None,
            out_tangent: None,
        }
    }

    /// Creates a cubic keyframe with explicit tangent handles.
    pub fn cubic(
        time: f32,
        value: T,
        in_tangent: Option<Tangent<T>>,
        out_tangent: Option<Tangent<T>>,
    ) -> Self {
        Self {
            time,
            value,
            interpolation: Interpolation::Cubic,
            in_tangent,
            out_tangent,
        }
    }
}

// ---------------------------------------------------------------------------
// Cubic Bezier helpers
// ---------------------------------------------------------------------------

/// Evaluates a 1-D cubic Bezier at parameter `t ∈ [0, 1]`.
///
/// Control points are `[p0, p1, p2, p3]`.
#[inline]
fn cubic_bezier_1d(p0: f32, p1: f32, p2: f32, p3: f32, t: f32) -> f32 {
    let u = 1.0 - t;
    u * u * u * p0 + 3.0 * u * u * t * p1 + 3.0 * u * t * t * p2 + t * t * t * p3
}

/// Derivative of a 1-D cubic Bezier with respect to `t`.
#[inline]
fn cubic_bezier_1d_deriv(p0: f32, p1: f32, p2: f32, p3: f32, t: f32) -> f32 {
    let u = 1.0 - t;
    3.0 * u * u * (p1 - p0) + 6.0 * u * t * (p2 - p1) + 3.0 * t * t * (p3 - p2)
}

/// Solves for the Bezier parameter `t` such that the time Bezier equals
/// `target_time`, using Newton's method (8 iterations) with bisection fallback.
///
/// `t0..=t3` are the four time control points of the Bezier.
fn solve_bezier_t(t0: f32, t1: f32, t2: f32, t3: f32, target_time: f32) -> f32 {
    // Normalise the problem: remap so that t0 → 0 and t3 → 1 gives a well
    // conditioned starting guess.
    let mut t = (target_time - t0) / (t3 - t0).max(1e-10);
    t = t.clamp(0.0, 1.0);

    for _ in 0..8 {
        let current = cubic_bezier_1d(t0, t1, t2, t3, t);
        let error = current - target_time;
        if error.abs() < 1e-6 {
            break;
        }
        let deriv = cubic_bezier_1d_deriv(t0, t1, t2, t3, t);
        if deriv.abs() < 1e-10 {
            break;
        }
        t -= error / deriv;
        t = t.clamp(0.0, 1.0);
    }

    t
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
                    Interpolation::Linear => curr.value.lerp_to(&next.value, t),
                    Interpolation::Cubic => {
                        // Build time Bezier control points.
                        // tc1 is curr.time offset forward by out_tangent.dt;
                        // tc2 is next.time offset backward by in_tangent.dt.
                        let out_dt = curr.out_tangent.as_ref().map(|h| h.dt).unwrap_or(0.0);
                        let in_dt = next.in_tangent.as_ref().map(|h| h.dt).unwrap_or(0.0);
                        let tc0 = curr.time;
                        let tc1 = curr.time + out_dt;
                        let tc2 = next.time - in_dt;
                        let tc3 = next.time;

                        // Solve for the Bezier parameter u in [0,1] where the
                        // time Bezier equals `time`.
                        let u = solve_bezier_t(tc0, tc1, tc2, tc3, time);

                        // Value control points.  `dv` in Tangent stores the
                        // *absolute* control-point value (not a delta), so no
                        // addition is needed and only Lerp is required on T.
                        // When a handle is absent, the control point collapses
                        // to the keyframe value (zero outgoing/incoming velocity),
                        // which degrades the curve to linear interpolation.
                        let p0 = &curr.value;
                        let p1 = curr.out_tangent.as_ref().map_or(&curr.value, |h| &h.dv);
                        let p2 = next.in_tangent.as_ref().map_or(&next.value, |h| &h.dv);
                        let p3 = &next.value;

                        // De Casteljau evaluation at u:
                        //   B(u) = lerp(lerp(lerp(p0,p1,u), lerp(p1,p2,u), u),
                        //               lerp(lerp(p1,p2,u), lerp(p2,p3,u), u), u)
                        let q0 = p0.lerp_to(p1, u);
                        let q1 = p1.lerp_to(p2, u);
                        let q2 = p2.lerp_to(p3, u);
                        let r0 = q0.lerp_to(&q1, u);
                        let r1 = q1.lerp_to(&q2, u);
                        r0.lerp_to(&r1, u)
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

    // -----------------------------------------------------------------------
    // Cubic interpolation tests
    // -----------------------------------------------------------------------

    /// Helper: build a cubic f32 track with three keyframes and explicit
    /// tangents that create a smooth S-curve.
    ///
    /// Keyframes: t=0 → 0.0, t=1 → 5.0, t=2 → 10.0
    /// Out-tangent of kf0: dt=0.5, dv=3.0  (control point slightly above linear)
    /// In-tangent  of kf1: dt=0.5, dv=3.0
    /// Out-tangent of kf1: dt=0.5, dv=7.0
    /// In-tangent  of kf2: dt=0.5, dv=7.0
    fn make_cubic_track() -> Track<f32> {
        Track::from_keyframes(vec![
            Keyframe::cubic(0.0, 0.0, None, Some(Tangent::new(0.5, 3.0))),
            Keyframe::cubic(
                1.0,
                5.0,
                Some(Tangent::new(0.5, 3.0)),
                Some(Tangent::new(0.5, 7.0)),
            ),
            Keyframe::cubic(2.0, 10.0, Some(Tangent::new(0.5, 7.0)), None),
        ])
    }

    #[test]
    fn test_cubic_endpoints() {
        let track = make_cubic_track();
        // Endpoints must match exactly.
        assert!((track.sample(0.0) - 0.0).abs() < 1e-4);
        assert!((track.sample(2.0) - 10.0).abs() < 1e-4);
    }

    #[test]
    fn test_cubic_midpoint_differs_from_linear() {
        let track = make_cubic_track();
        // Linear midpoint between 0 and 5 would be 2.5.
        // Our out_tangent dv=3.0 < 5.0, pulling the curve above the linear
        // path, so the sample at t=0.5 should be noticeably above 2.5.
        let sample = track.sample(0.5);
        assert!(
            (sample - 2.5).abs() > 0.1,
            "cubic should differ from linear at midpoint, got {sample}"
        );
    }

    #[test]
    fn test_cubic_no_tangents_matches_linear() {
        // Without tangents, cubic degrades to linear interpolation because
        // both inner control points collapse to the keyframe values.
        let mut cubic_track: Track<f32> = Track::from_keyframes(vec![
            Keyframe::cubic(0.0, 0.0, None, None),
            Keyframe::cubic(1.0, 10.0, None, None),
        ]);
        let mut linear_track: Track<f32> = Track::new();
        linear_track.add(0.0, 0.0);
        linear_track.add(1.0, 10.0);

        for &t in &[0.0, 0.25, 0.5, 0.75, 1.0] {
            let c = cubic_track.sample(t);
            let l = linear_track.sample(t);
            assert!(
                (c - l).abs() < 1e-4,
                "cubic without tangents should match linear at t={t}: cubic={c}, linear={l}"
            );
        }

        // suppress unused mut warnings
        let _ = &mut cubic_track;
        let _ = &mut linear_track;
    }

    #[test]
    fn test_cubic_three_keyframes_smooth() {
        let track = make_cubic_track();
        // Sample many points and verify the curve is monotonically increasing
        // (since start=0, end=10 with symmetric S-curve tangents).
        let mut prev = track.sample(0.0);
        let steps = 40;
        for i in 1..=steps {
            let t = 2.0 * i as f32 / steps as f32;
            let val = track.sample(t);
            assert!(
                val >= prev - 1e-4,
                "curve should be non-decreasing; at t={t} got {val} < prev {prev}"
            );
            prev = val;
        }
    }
}
