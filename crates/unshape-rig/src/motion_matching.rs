//! Motion matching system.
//!
//! Motion matching finds the best-fitting pose from a database of motion clips
//! based on the current pose and desired trajectory.

use crate::{Pose, Skeleton, Transform3D};
use glam::{Quat, Vec3};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Configuration for motion matching.
///
/// Controls weights for different matching criteria and timing parameters
/// for the motion matching algorithm.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dynop", derive(unshape_op::Op))]
#[cfg_attr(feature = "dynop", op(input = (), output = MotionMatching))]
pub struct MotionMatching {
    /// Weight for bone position matching.
    pub position_weight: f32,
    /// Weight for bone velocity matching.
    pub velocity_weight: f32,
    /// Weight for trajectory matching.
    pub trajectory_weight: f32,
    /// Weight for facing direction matching.
    pub facing_weight: f32,
    /// Number of future trajectory samples.
    pub trajectory_samples: usize,
    /// Time between trajectory samples.
    pub trajectory_sample_time: f32,
    /// Minimum time between transitions.
    pub min_transition_time: f32,
    /// Blend time for transitions.
    pub blend_time: f32,
}

/// Backwards-compatible type alias.
pub type MotionMatchingConfig = MotionMatching;

impl MotionMatching {
    /// Applies this generator, returning the configuration.
    pub fn apply(&self) -> MotionMatching {
        self.clone()
    }
}

impl Default for MotionMatching {
    fn default() -> Self {
        Self {
            position_weight: 1.0,
            velocity_weight: 1.0,
            trajectory_weight: 2.0,
            facing_weight: 1.0,
            trajectory_samples: 3,
            trajectory_sample_time: 0.2,
            min_transition_time: 0.2,
            blend_time: 0.2,
        }
    }
}

/// A single frame in a motion clip.
#[derive(Debug, Clone)]
pub struct MotionFrame {
    /// Time in the clip.
    pub time: f32,
    /// Root position.
    pub root_position: Vec3,
    /// Root velocity.
    pub root_velocity: Vec3,
    /// Facing direction (normalized XZ).
    pub facing: Vec3,
    /// Bone local positions (relative to parent).
    pub bone_positions: Vec<Vec3>,
    /// Bone local rotations.
    pub bone_rotations: Vec<Quat>,
    /// Bone velocities.
    pub bone_velocities: Vec<Vec3>,
    /// Future trajectory positions (for matching).
    pub trajectory_positions: Vec<Vec3>,
    /// Future trajectory facing directions.
    pub trajectory_facings: Vec<Vec3>,
}

impl MotionFrame {
    /// Creates a new motion frame.
    pub fn new(time: f32, bone_count: usize) -> Self {
        Self {
            time,
            root_position: Vec3::ZERO,
            root_velocity: Vec3::ZERO,
            facing: Vec3::Z,
            bone_positions: vec![Vec3::ZERO; bone_count],
            bone_rotations: vec![Quat::IDENTITY; bone_count],
            bone_velocities: vec![Vec3::ZERO; bone_count],
            trajectory_positions: Vec::new(),
            trajectory_facings: Vec::new(),
        }
    }
}

/// A motion clip for motion matching.
#[derive(Debug, Clone)]
pub struct MotionClip {
    /// Clip name.
    pub name: String,
    /// Frames in the clip.
    pub frames: Vec<MotionFrame>,
    /// Frame rate.
    pub frame_rate: f32,
    /// Whether the clip loops.
    pub looping: bool,
    /// Tags for this clip (e.g., "walk", "run", "idle").
    pub tags: Vec<String>,
}

impl MotionClip {
    /// Creates a new motion clip.
    pub fn new(name: impl Into<String>, frame_rate: f32) -> Self {
        Self {
            name: name.into(),
            frames: Vec::new(),
            frame_rate,
            looping: false,
            tags: Vec::new(),
        }
    }

    /// Duration of the clip.
    pub fn duration(&self) -> f32 {
        if self.frames.is_empty() {
            0.0
        } else {
            self.frames.last().unwrap().time
        }
    }

    /// Gets the frame index for a time.
    pub fn frame_index(&self, time: f32) -> usize {
        if self.frames.is_empty() {
            return 0;
        }

        let t = if self.looping {
            time.rem_euclid(self.duration())
        } else {
            time.clamp(0.0, self.duration())
        };

        // Binary search for frame
        let mut lo = 0;
        let mut hi = self.frames.len() - 1;

        while lo < hi {
            let mid = (lo + hi + 1) / 2;
            if self.frames[mid].time <= t {
                lo = mid;
            } else {
                hi = mid - 1;
            }
        }

        lo
    }

    /// Gets a frame by index.
    pub fn frame(&self, index: usize) -> Option<&MotionFrame> {
        self.frames.get(index)
    }
}

/// A reference to a specific frame in the motion database.
#[derive(Debug, Clone, Copy)]
pub struct FrameRef {
    /// Clip index in the database.
    pub clip: usize,
    /// Frame index in the clip.
    pub frame: usize,
}

/// A motion database containing all clips.
#[derive(Debug, Clone)]
pub struct MotionDatabase {
    /// All clips in the database.
    pub clips: Vec<MotionClip>,
    // KD-tree or similar structure could be added for faster searching.
}

impl MotionDatabase {
    /// Creates a new empty database.
    pub fn new() -> Self {
        Self { clips: Vec::new() }
    }

    /// Adds a clip to the database.
    pub fn add_clip(&mut self, clip: MotionClip) -> usize {
        let index = self.clips.len();
        self.clips.push(clip);
        index
    }

    /// Gets a clip by index.
    pub fn clip(&self, index: usize) -> Option<&MotionClip> {
        self.clips.get(index)
    }

    /// Gets a frame reference.
    pub fn frame(&self, frame_ref: FrameRef) -> Option<&MotionFrame> {
        self.clips.get(frame_ref.clip)?.frame(frame_ref.frame)
    }

    /// Finds clips with a specific tag.
    pub fn clips_with_tag(&self, tag: &str) -> Vec<usize> {
        self.clips
            .iter()
            .enumerate()
            .filter(|(_, clip)| clip.tags.iter().any(|t| t == tag))
            .map(|(i, _)| i)
            .collect()
    }

    /// Total number of frames across all clips.
    pub fn total_frames(&self) -> usize {
        self.clips.iter().map(|c| c.frames.len()).sum()
    }
}

impl Default for MotionDatabase {
    fn default() -> Self {
        Self::new()
    }
}

/// Query for motion matching search.
#[derive(Debug, Clone)]
pub struct MotionQuery {
    /// Current bone positions.
    pub bone_positions: Vec<Vec3>,
    /// Current bone velocities.
    pub bone_velocities: Vec<Vec3>,
    /// Desired future trajectory positions.
    pub trajectory_positions: Vec<Vec3>,
    /// Desired future facing directions.
    pub trajectory_facings: Vec<Vec3>,
    /// Current facing direction.
    pub facing: Vec3,
}

impl MotionQuery {
    /// Creates a new query with the given bone count.
    pub fn new(bone_count: usize, trajectory_samples: usize) -> Self {
        Self {
            bone_positions: vec![Vec3::ZERO; bone_count],
            bone_velocities: vec![Vec3::ZERO; bone_count],
            trajectory_positions: vec![Vec3::ZERO; trajectory_samples],
            trajectory_facings: vec![Vec3::Z; trajectory_samples],
            facing: Vec3::Z,
        }
    }

    /// Sets trajectory from a direction and speed.
    pub fn set_trajectory_linear(&mut self, direction: Vec3, speed: f32, sample_time: f32) {
        let dir = direction.normalize_or_zero();
        for (i, pos) in self.trajectory_positions.iter_mut().enumerate() {
            *pos = dir * speed * (i + 1) as f32 * sample_time;
        }
        for facing in &mut self.trajectory_facings {
            *facing = if dir.length_squared() > 0.0 {
                dir
            } else {
                Vec3::Z
            };
        }
    }
}

/// Result of a motion matching search.
#[derive(Debug, Clone, Copy)]
pub struct MatchResult {
    /// Best matching frame.
    pub frame_ref: FrameRef,
    /// Cost/distance of the match.
    pub cost: f32,
}

/// Searches for the best matching frame.
pub fn find_best_match(
    database: &MotionDatabase,
    query: &MotionQuery,
    config: &MotionMatchingConfig,
    allowed_clips: Option<&[usize]>,
) -> Option<MatchResult> {
    let mut best: Option<MatchResult> = None;

    let clips_to_search: Vec<usize> = allowed_clips
        .map(|c| c.to_vec())
        .unwrap_or_else(|| (0..database.clips.len()).collect());

    for &clip_idx in &clips_to_search {
        let clip = match database.clip(clip_idx) {
            Some(c) => c,
            None => continue,
        };

        for (frame_idx, frame) in clip.frames.iter().enumerate() {
            let cost = compute_match_cost(query, frame, config);

            let is_better = best.as_ref().map(|b| cost < b.cost).unwrap_or(true);
            if is_better {
                best = Some(MatchResult {
                    frame_ref: FrameRef {
                        clip: clip_idx,
                        frame: frame_idx,
                    },
                    cost,
                });
            }
        }
    }

    best
}

/// Computes the cost between a query and a frame.
pub fn compute_match_cost(
    query: &MotionQuery,
    frame: &MotionFrame,
    config: &MotionMatchingConfig,
) -> f32 {
    let mut cost = 0.0;

    // Bone position cost
    if config.position_weight > 0.0 {
        let pos_cost: f32 = query
            .bone_positions
            .iter()
            .zip(frame.bone_positions.iter())
            .map(|(q, f)| (*q - *f).length_squared())
            .sum();
        cost += pos_cost * config.position_weight;
    }

    // Bone velocity cost
    if config.velocity_weight > 0.0 {
        let vel_cost: f32 = query
            .bone_velocities
            .iter()
            .zip(frame.bone_velocities.iter())
            .map(|(q, f)| (*q - *f).length_squared())
            .sum();
        cost += vel_cost * config.velocity_weight;
    }

    // Trajectory cost
    if config.trajectory_weight > 0.0 {
        let traj_cost: f32 = query
            .trajectory_positions
            .iter()
            .zip(frame.trajectory_positions.iter())
            .map(|(q, f)| (*q - *f).length_squared())
            .sum();
        cost += traj_cost * config.trajectory_weight;
    }

    // Facing cost
    if config.facing_weight > 0.0 {
        let facing_cost = 1.0 - query.facing.dot(frame.facing).max(0.0);
        cost += facing_cost * config.facing_weight;

        // Future facing
        let future_facing_cost: f32 = query
            .trajectory_facings
            .iter()
            .zip(frame.trajectory_facings.iter())
            .map(|(q, f)| 1.0 - q.dot(*f).max(0.0))
            .sum();
        cost += future_facing_cost * config.facing_weight;
    }

    cost
}

/// Motion matching controller.
#[derive(Debug, Clone)]
pub struct MotionMatcher {
    /// Configuration.
    pub config: MotionMatchingConfig,
    /// Current clip index.
    pub current_clip: usize,
    /// Current time in clip.
    pub current_time: f32,
    /// Previous clip (for blending).
    pub blend_clip: Option<usize>,
    /// Previous time (for blending).
    pub blend_time_left: f32,
    /// Time since last transition.
    pub time_since_transition: f32,
    /// Allowed clip tags (None = all).
    pub allowed_tags: Option<Vec<String>>,
}

impl MotionMatcher {
    /// Creates a new motion matcher.
    pub fn new(config: MotionMatchingConfig) -> Self {
        Self {
            config,
            current_clip: 0,
            current_time: 0.0,
            blend_clip: None,
            blend_time_left: 0.0,
            time_since_transition: 0.0,
            allowed_tags: None,
        }
    }

    /// Updates the motion matcher and returns the current pose.
    pub fn update(
        &mut self,
        database: &MotionDatabase,
        query: &MotionQuery,
        dt: f32,
    ) -> Option<MotionFrame> {
        // Advance time
        self.current_time += dt;
        self.time_since_transition += dt;

        if self.blend_time_left > 0.0 {
            self.blend_time_left = (self.blend_time_left - dt).max(0.0);
        }

        // Get allowed clips
        let allowed_clips: Option<Vec<usize>> = self.allowed_tags.as_ref().map(|tags| {
            tags.iter()
                .flat_map(|tag| database.clips_with_tag(tag))
                .collect()
        });

        // Search for better match if enough time has passed
        if self.time_since_transition >= self.config.min_transition_time {
            if let Some(result) =
                find_best_match(database, query, &self.config, allowed_clips.as_deref())
            {
                // Check if new match is significantly better
                let current_clip = database.clip(self.current_clip)?;
                let current_frame_idx = current_clip.frame_index(self.current_time);
                let current_frame = current_clip.frame(current_frame_idx)?;
                let current_cost = compute_match_cost(query, current_frame, &self.config);

                // Transition if new match is at least 10% better
                if result.cost < current_cost * 0.9 {
                    self.transition_to(result.frame_ref.clip, database);
                    self.current_time = database
                        .clip(result.frame_ref.clip)?
                        .frames
                        .get(result.frame_ref.frame)?
                        .time;
                }
            }
        }

        // Get current frame
        let clip = database.clip(self.current_clip)?;

        // Handle looping or clip end
        if self.current_time > clip.duration() {
            if clip.looping {
                self.current_time = self.current_time.rem_euclid(clip.duration());
            } else {
                self.current_time = clip.duration();
            }
        }

        let frame_idx = clip.frame_index(self.current_time);
        clip.frame(frame_idx).cloned()
    }

    /// Transitions to a new clip.
    pub fn transition_to(&mut self, clip_idx: usize, _database: &MotionDatabase) {
        if clip_idx != self.current_clip {
            self.blend_clip = Some(self.current_clip);
            self.blend_time_left = self.config.blend_time;
            self.current_clip = clip_idx;
            self.current_time = 0.0;
            self.time_since_transition = 0.0;
        }
    }

    /// Sets allowed tags for clip filtering.
    pub fn set_allowed_tags(&mut self, tags: Vec<String>) {
        self.allowed_tags = Some(tags);
    }

    /// Clears tag filter (allows all clips).
    pub fn clear_tag_filter(&mut self) {
        self.allowed_tags = None;
    }

    /// Gets the current blend weight (0 = previous clip, 1 = current clip).
    pub fn blend_weight(&self) -> f32 {
        if self.blend_time_left > 0.0 {
            1.0 - (self.blend_time_left / self.config.blend_time)
        } else {
            1.0
        }
    }
}

/// Builds motion frames from an animation.
pub struct MotionFrameBuilder {
    trajectory_samples: usize,
    trajectory_sample_time: f32,
}

impl MotionFrameBuilder {
    /// Creates a new builder.
    pub fn new(config: &MotionMatchingConfig) -> Self {
        Self {
            trajectory_samples: config.trajectory_samples,
            trajectory_sample_time: config.trajectory_sample_time,
        }
    }

    /// Builds a motion clip from keyframed animation data.
    ///
    /// # Arguments
    /// * `name` - Clip name
    /// * `frame_rate` - Target frame rate
    /// * `duration` - Total duration
    /// * `sample_fn` - Function that returns (root_pos, bone_positions, bone_rotations) for a time
    pub fn build_clip<F>(
        &self,
        name: impl Into<String>,
        frame_rate: f32,
        duration: f32,
        mut sample_fn: F,
    ) -> MotionClip
    where
        F: FnMut(f32) -> (Vec3, Vec<Vec3>, Vec<Quat>),
    {
        let mut clip = MotionClip::new(name, frame_rate);
        let frame_time = 1.0 / frame_rate;
        let frame_count = (duration / frame_time).ceil() as usize + 1;

        // Sample all frames first
        let samples: Vec<_> = (0..frame_count)
            .map(|i| {
                let t = (i as f32 * frame_time).min(duration);
                let (root_pos, bone_pos, bone_rot) = sample_fn(t);
                (t, root_pos, bone_pos, bone_rot)
            })
            .collect();

        // Build frames with velocities and trajectories
        for i in 0..frame_count {
            let (time, root_pos, bone_pos, bone_rot) = &samples[i];
            let bone_count = bone_pos.len();

            let mut frame = MotionFrame::new(*time, bone_count);
            frame.root_position = *root_pos;
            frame.bone_positions = bone_pos.clone();
            frame.bone_rotations = bone_rot.clone();

            // Compute velocities from neighboring frames
            if i > 0 {
                let (prev_time, prev_root, prev_pos, _) = &samples[i - 1];
                let dt = time - prev_time;
                if dt > 0.0 {
                    frame.root_velocity = (*root_pos - *prev_root) / dt;
                    frame.bone_velocities = bone_pos
                        .iter()
                        .zip(prev_pos.iter())
                        .map(|(curr, prev)| (*curr - *prev) / dt)
                        .collect();
                }
            }

            // Compute facing direction from root velocity
            let facing_2d = Vec3::new(frame.root_velocity.x, 0.0, frame.root_velocity.z);
            frame.facing = if facing_2d.length_squared() > 0.01 {
                facing_2d.normalize()
            } else {
                Vec3::Z
            };

            // Build trajectory
            for j in 0..self.trajectory_samples {
                let future_time = time + (j + 1) as f32 * self.trajectory_sample_time;
                let future_idx = ((future_time / frame_time) as usize).min(frame_count - 1);
                let (_, future_root, _, _) = &samples[future_idx];

                frame.trajectory_positions.push(*future_root - *root_pos);
                frame.trajectory_facings.push(frame.facing);
            }

            clip.frames.push(frame);
        }

        clip
    }
}

/// Applies a motion frame to a skeleton pose.
pub fn apply_frame_to_pose(frame: &MotionFrame, pose: &mut Pose, skeleton: &Skeleton) {
    let bone_count = skeleton.bone_count();
    for i in 0..bone_count
        .min(frame.bone_positions.len())
        .min(frame.bone_rotations.len())
    {
        let bone_id = crate::skeleton::BoneId(i as u32);
        let transform = Transform3D {
            translation: frame.bone_positions[i],
            rotation: frame.bone_rotations[i],
            scale: Vec3::ONE,
        };
        pose.set(bone_id, transform);
    }
}

/// Blends two motion frames.
pub fn blend_frames(a: &MotionFrame, b: &MotionFrame, t: f32) -> MotionFrame {
    let bone_count = a.bone_positions.len().min(b.bone_positions.len());
    let traj_count = a
        .trajectory_positions
        .len()
        .min(b.trajectory_positions.len());

    MotionFrame {
        time: a.time + (b.time - a.time) * t,
        root_position: a.root_position.lerp(b.root_position, t),
        root_velocity: a.root_velocity.lerp(b.root_velocity, t),
        facing: a.facing.lerp(b.facing, t).normalize_or_zero(),
        bone_positions: (0..bone_count)
            .map(|i| a.bone_positions[i].lerp(b.bone_positions[i], t))
            .collect(),
        bone_rotations: (0..bone_count)
            .map(|i| a.bone_rotations[i].slerp(b.bone_rotations[i], t))
            .collect(),
        bone_velocities: (0..bone_count)
            .map(|i| a.bone_velocities[i].lerp(b.bone_velocities[i], t))
            .collect(),
        trajectory_positions: (0..traj_count)
            .map(|i| a.trajectory_positions[i].lerp(b.trajectory_positions[i], t))
            .collect(),
        trajectory_facings: (0..traj_count)
            .map(|i| {
                a.trajectory_facings[i]
                    .lerp(b.trajectory_facings[i], t)
                    .normalize_or_zero()
            })
            .collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_motion_matching_config_default() {
        let config = MotionMatchingConfig::default();
        assert!(config.position_weight > 0.0);
        assert!(config.trajectory_samples > 0);
    }

    #[test]
    fn test_motion_frame_new() {
        let frame = MotionFrame::new(0.0, 5);
        assert_eq!(frame.bone_positions.len(), 5);
        assert_eq!(frame.bone_rotations.len(), 5);
    }

    #[test]
    fn test_motion_clip_creation() {
        let mut clip = MotionClip::new("walk", 30.0);
        clip.looping = true;
        clip.tags.push("locomotion".to_string());

        assert_eq!(clip.name, "walk");
        assert!(clip.looping);
        assert!(clip.tags.contains(&"locomotion".to_string()));
    }

    #[test]
    fn test_motion_clip_duration() {
        let mut clip = MotionClip::new("test", 30.0);

        assert_eq!(clip.duration(), 0.0);

        clip.frames.push(MotionFrame::new(0.0, 1));
        clip.frames.push(MotionFrame::new(0.5, 1));
        clip.frames.push(MotionFrame::new(1.0, 1));

        assert_eq!(clip.duration(), 1.0);
    }

    #[test]
    fn test_motion_clip_frame_index() {
        let mut clip = MotionClip::new("test", 30.0);
        clip.frames.push(MotionFrame::new(0.0, 1));
        clip.frames.push(MotionFrame::new(0.5, 1));
        clip.frames.push(MotionFrame::new(1.0, 1));

        assert_eq!(clip.frame_index(0.0), 0);
        assert_eq!(clip.frame_index(0.3), 0);
        assert_eq!(clip.frame_index(0.5), 1);
        assert_eq!(clip.frame_index(0.7), 1);
        assert_eq!(clip.frame_index(1.0), 2);
    }

    #[test]
    fn test_motion_database() {
        let mut db = MotionDatabase::new();

        let mut clip = MotionClip::new("walk", 30.0);
        clip.tags.push("locomotion".to_string());
        let idx = db.add_clip(clip);

        assert_eq!(idx, 0);
        assert!(db.clip(0).is_some());
        assert_eq!(db.clips_with_tag("locomotion"), vec![0]);
        assert!(db.clips_with_tag("combat").is_empty());
    }

    #[test]
    fn test_motion_query() {
        let mut query = MotionQuery::new(5, 3);

        assert_eq!(query.bone_positions.len(), 5);
        assert_eq!(query.trajectory_positions.len(), 3);

        query.set_trajectory_linear(Vec3::X, 5.0, 0.2);

        assert!(query.trajectory_positions[0].x > 0.0);
    }

    #[test]
    fn test_compute_match_cost() {
        let config = MotionMatchingConfig::default();
        let query = MotionQuery::new(2, 2);
        let frame = MotionFrame::new(0.0, 2);

        // Same pose should have low cost
        let cost = compute_match_cost(&query, &frame, &config);
        assert!(cost >= 0.0);
    }

    #[test]
    fn test_find_best_match() {
        let config = MotionMatchingConfig::default();
        let mut db = MotionDatabase::new();

        let mut clip = MotionClip::new("test", 30.0);
        clip.frames.push(MotionFrame::new(0.0, 2));
        clip.frames.push(MotionFrame::new(0.5, 2));
        db.add_clip(clip);

        let query = MotionQuery::new(2, 3);

        let result = find_best_match(&db, &query, &config, None);
        assert!(result.is_some());
    }

    #[test]
    fn test_motion_matcher_creation() {
        let config = MotionMatchingConfig::default();
        let matcher = MotionMatcher::new(config);

        assert_eq!(matcher.current_clip, 0);
        assert_eq!(matcher.current_time, 0.0);
    }

    #[test]
    fn test_motion_matcher_blend_weight() {
        let mut config = MotionMatchingConfig::default();
        config.blend_time = 1.0;

        let mut matcher = MotionMatcher::new(config);
        matcher.blend_time_left = 0.5;

        let weight = matcher.blend_weight();
        assert!(weight > 0.0 && weight < 1.0);
    }

    #[test]
    fn test_blend_frames() {
        let a = MotionFrame::new(0.0, 2);
        let mut b = MotionFrame::new(1.0, 2);
        b.root_position = Vec3::X;

        let blended = blend_frames(&a, &b, 0.5);

        assert!((blended.root_position.x - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_motion_frame_builder() {
        let config = MotionMatchingConfig::default();
        let builder = MotionFrameBuilder::new(&config);

        let clip = builder.build_clip("test", 10.0, 1.0, |t| {
            let pos = Vec3::new(t, 0.0, 0.0);
            (
                pos,
                vec![Vec3::ZERO, Vec3::Y],
                vec![Quat::IDENTITY, Quat::IDENTITY],
            )
        });

        assert!(!clip.frames.is_empty());
        assert!(clip.frames.len() >= 10);
    }

    #[test]
    fn test_motion_matcher_tag_filter() {
        let config = MotionMatchingConfig::default();
        let mut matcher = MotionMatcher::new(config);

        matcher.set_allowed_tags(vec!["walk".to_string()]);
        assert!(matcher.allowed_tags.is_some());

        matcher.clear_tag_filter();
        assert!(matcher.allowed_tags.is_none());
    }

    #[test]
    fn test_motion_database_total_frames() {
        let mut db = MotionDatabase::new();

        let mut clip1 = MotionClip::new("a", 30.0);
        clip1.frames.push(MotionFrame::new(0.0, 1));
        clip1.frames.push(MotionFrame::new(0.5, 1));
        db.add_clip(clip1);

        let mut clip2 = MotionClip::new("b", 30.0);
        clip2.frames.push(MotionFrame::new(0.0, 1));
        db.add_clip(clip2);

        assert_eq!(db.total_frames(), 3);
    }

    #[test]
    fn test_frame_ref() {
        let frame_ref = FrameRef { clip: 1, frame: 5 };
        assert_eq!(frame_ref.clip, 1);
        assert_eq!(frame_ref.frame, 5);
    }
}
