//! Procedural locomotion and walk cycles.
//!
//! Provides gait generation, foot placement, and body motion for bipeds and quadrupeds.

use crate::{IkChain, IkConfig, Pose, Skeleton, Transform, solve_fabrik};
use glam::{Quat, Vec3};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Configuration for a walking gait.
///
/// Controls stride, step height, timing, and body motion for procedural
/// locomotion of bipeds and quadrupeds.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dynop", derive(rhizome_resin_op::Op))]
#[cfg_attr(feature = "dynop", op(input = (), output = Gait))]
pub struct Gait {
    /// Length of a single stride.
    pub stride_length: f32,
    /// Maximum height of foot during step.
    pub step_height: f32,
    /// Duration of a full gait cycle in seconds.
    pub cycle_duration: f32,
    /// Fraction of cycle spent with foot planted (0.0-1.0).
    pub stance_ratio: f32,
    /// Vertical bob amount for body.
    pub body_bob: f32,
    /// Lateral sway amount for body.
    pub body_sway: f32,
    /// Forward lean when moving.
    pub lean_amount: f32,
}

/// Backwards-compatible type alias.
pub type GaitConfig = Gait;

impl Gait {
    /// Applies this generator, returning the configuration.
    pub fn apply(&self) -> Gait {
        *self
    }
}

impl Default for Gait {
    fn default() -> Self {
        Self {
            stride_length: 0.6,
            step_height: 0.15,
            cycle_duration: 1.0,
            stance_ratio: 0.5,
            body_bob: 0.03,
            body_sway: 0.02,
            lean_amount: 0.05,
        }
    }
}

impl Gait {
    /// Quick walking gait.
    pub fn walk() -> Self {
        Self::default()
    }

    /// Running gait with longer strides and higher steps.
    pub fn run() -> Self {
        Self {
            stride_length: 1.2,
            step_height: 0.25,
            cycle_duration: 0.5,
            stance_ratio: 0.35,
            body_bob: 0.05,
            body_sway: 0.03,
            lean_amount: 0.15,
        }
    }

    /// Sneaking gait with small, careful steps.
    pub fn sneak() -> Self {
        Self {
            stride_length: 0.3,
            step_height: 0.08,
            cycle_duration: 1.5,
            stance_ratio: 0.65,
            body_bob: 0.01,
            body_sway: 0.01,
            lean_amount: 0.02,
        }
    }

    /// Quadruped walking gait.
    pub fn quadruped_walk() -> Self {
        Self {
            stride_length: 0.4,
            step_height: 0.1,
            cycle_duration: 0.8,
            stance_ratio: 0.65,
            body_bob: 0.02,
            body_sway: 0.01,
            lean_amount: 0.03,
        }
    }

    /// Quadruped trot (diagonal pairs move together).
    pub fn quadruped_trot() -> Self {
        Self {
            stride_length: 0.7,
            step_height: 0.15,
            cycle_duration: 0.5,
            stance_ratio: 0.45,
            body_bob: 0.04,
            body_sway: 0.02,
            lean_amount: 0.08,
        }
    }
}

/// Phase offsets for different gait patterns.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GaitPattern {
    /// Biped walk (alternating legs).
    BipedWalk,
    /// Quadruped walk (each leg offset by 0.25).
    QuadrupedWalk,
    /// Quadruped trot (diagonal pairs).
    QuadrupedTrot,
    /// Quadruped pace (lateral pairs).
    QuadrupedPace,
    /// Quadruped bound (front pair, then back pair).
    QuadrupedBound,
}

impl GaitPattern {
    /// Returns phase offsets for each leg in the pattern.
    pub fn phase_offsets(&self) -> Vec<f32> {
        match self {
            GaitPattern::BipedWalk => vec![0.0, 0.5],
            GaitPattern::QuadrupedWalk => vec![0.0, 0.5, 0.75, 0.25], // FL, FR, BL, BR
            GaitPattern::QuadrupedTrot => vec![0.0, 0.5, 0.5, 0.0],   // Diagonal pairs
            GaitPattern::QuadrupedPace => vec![0.0, 0.5, 0.0, 0.5],   // Lateral pairs
            GaitPattern::QuadrupedBound => vec![0.0, 0.0, 0.5, 0.5],  // Front, then back
        }
    }

    /// Returns number of legs for this pattern.
    pub fn leg_count(&self) -> usize {
        match self {
            GaitPattern::BipedWalk => 2,
            _ => 4,
        }
    }
}

/// State of a single leg during locomotion.
#[derive(Debug, Clone)]
pub struct LegState {
    /// Current foot position in world space.
    pub foot_position: Vec3,
    /// Target foot position for current step.
    pub target_position: Vec3,
    /// Previous planted position.
    pub previous_position: Vec3,
    /// Current phase in gait cycle (0.0-1.0).
    pub phase: f32,
    /// Whether foot is currently planted.
    pub planted: bool,
    /// Offset from body center to hip (local space).
    pub hip_offset: Vec3,
    /// Default standing position relative to hip.
    pub rest_offset: Vec3,
}

impl LegState {
    /// Creates a new leg state.
    pub fn new(hip_offset: Vec3, rest_offset: Vec3) -> Self {
        Self {
            foot_position: hip_offset + rest_offset,
            target_position: hip_offset + rest_offset,
            previous_position: hip_offset + rest_offset,
            phase: 0.0,
            planted: true,
            hip_offset,
            rest_offset,
        }
    }

    /// Computes step progress (0.0 when lifting, 1.0 when planted).
    fn step_progress(&self, stance_ratio: f32) -> f32 {
        if self.planted {
            return 1.0;
        }
        let swing_phase = (self.phase - stance_ratio) / (1.0 - stance_ratio);
        swing_phase.clamp(0.0, 1.0)
    }
}

/// Procedural walk cycle controller.
#[derive(Debug, Clone)]
pub struct ProceduralWalk {
    /// Gait configuration.
    pub config: GaitConfig,
    /// Gait pattern defining leg timing.
    pub pattern: GaitPattern,
    /// Per-leg states.
    pub legs: Vec<LegState>,
    /// Current body position.
    pub body_position: Vec3,
    /// Current body rotation.
    pub body_rotation: Quat,
    /// Base body position (without bob/sway).
    base_position: Vec3,
    /// Base body rotation (without lean).
    base_rotation: Quat,
    /// Current movement velocity.
    velocity: Vec3,
    /// Master phase for cycle.
    master_phase: f32,
}

impl ProceduralWalk {
    /// Creates a biped walk controller.
    pub fn biped(
        body_position: Vec3,
        body_rotation: Quat,
        hip_width: f32,
        leg_length: f32,
    ) -> Self {
        let half_width = hip_width / 2.0;
        let legs = vec![
            LegState::new(
                Vec3::new(-half_width, 0.0, 0.0),
                Vec3::new(0.0, -leg_length, 0.0),
            ),
            LegState::new(
                Vec3::new(half_width, 0.0, 0.0),
                Vec3::new(0.0, -leg_length, 0.0),
            ),
        ];

        Self {
            config: GaitConfig::walk(),
            pattern: GaitPattern::BipedWalk,
            legs,
            body_position,
            body_rotation,
            base_position: body_position,
            base_rotation: body_rotation,
            velocity: Vec3::ZERO,
            master_phase: 0.0,
        }
    }

    /// Creates a quadruped walk controller.
    pub fn quadruped(
        body_position: Vec3,
        body_rotation: Quat,
        hip_width: f32,
        body_length: f32,
        leg_length: f32,
    ) -> Self {
        let half_width = hip_width / 2.0;
        let half_length = body_length / 2.0;
        let legs = vec![
            // Front left
            LegState::new(
                Vec3::new(-half_width, 0.0, half_length),
                Vec3::new(0.0, -leg_length, 0.0),
            ),
            // Front right
            LegState::new(
                Vec3::new(half_width, 0.0, half_length),
                Vec3::new(0.0, -leg_length, 0.0),
            ),
            // Back left
            LegState::new(
                Vec3::new(-half_width, 0.0, -half_length),
                Vec3::new(0.0, -leg_length, 0.0),
            ),
            // Back right
            LegState::new(
                Vec3::new(half_width, 0.0, -half_length),
                Vec3::new(0.0, -leg_length, 0.0),
            ),
        ];

        Self {
            config: GaitConfig::quadruped_walk(),
            pattern: GaitPattern::QuadrupedWalk,
            legs,
            body_position,
            body_rotation,
            base_position: body_position,
            base_rotation: body_rotation,
            velocity: Vec3::ZERO,
            master_phase: 0.0,
        }
    }

    /// Updates the walk cycle with new velocity.
    pub fn update(&mut self, dt: f32, velocity: Vec3) {
        self.velocity = velocity;
        let speed = velocity.length();

        // Only cycle when moving
        if speed > 0.001 {
            // Advance phase based on speed
            let phase_delta = (speed / self.config.stride_length) * dt / self.config.cycle_duration;
            self.master_phase = (self.master_phase + phase_delta) % 1.0;

            // Update base position
            self.base_position += velocity * dt;

            // Calculate movement direction
            let move_dir = velocity.normalize_or_zero();

            // Update base rotation to face movement direction
            if move_dir.length_squared() > 0.0001 {
                let target_rotation = rotation_from_direction(move_dir);
                self.base_rotation = self.base_rotation.slerp(target_rotation, 5.0 * dt);
            }
        }

        // Update each leg
        let phase_offsets = self.pattern.phase_offsets();
        for (i, leg) in self.legs.iter_mut().enumerate() {
            let leg_phase = (self.master_phase + phase_offsets[i]) % 1.0;
            let was_planted = leg.planted;

            // Determine if leg should be planted or swinging
            leg.planted = leg_phase < self.config.stance_ratio;
            leg.phase = leg_phase;

            if !was_planted && leg.planted {
                // Just planted - lock in position
                leg.previous_position = leg.foot_position;
            } else if was_planted && !leg.planted {
                // Just lifted - compute new target
                leg.previous_position = leg.foot_position;
                let hip_world = self.base_position + self.base_rotation * leg.hip_offset;

                // Target is stride ahead in movement direction
                let move_dir = if speed > 0.001 {
                    velocity.normalize()
                } else {
                    Vec3::Z
                };
                let stride_forward = move_dir * self.config.stride_length * 0.5;

                // Rest position under hip, plus stride
                leg.target_position =
                    hip_world + self.base_rotation * leg.rest_offset + stride_forward;
            }

            // Interpolate foot position
            if !leg.planted {
                let t = leg.step_progress(self.config.stance_ratio);

                // Horizontal interpolation
                let horizontal = leg.previous_position.lerp(leg.target_position, t);

                // Vertical arc
                let arc = (t * std::f32::consts::PI).sin() * self.config.step_height;

                leg.foot_position = Vec3::new(horizontal.x, horizontal.y + arc, horizontal.z);
            }
        }

        // Calculate body motion
        self.update_body_motion(speed);
    }

    fn update_body_motion(&mut self, speed: f32) {
        if speed < 0.001 {
            self.body_position = self.base_position;
            self.body_rotation = self.base_rotation;
            return;
        }

        // Body bob (twice per cycle - once per leg plant)
        let bob_phase = self.master_phase * std::f32::consts::TAU * 2.0;
        let bob = bob_phase.sin().abs() * self.config.body_bob;

        // Body sway (once per cycle)
        let sway_phase = self.master_phase * std::f32::consts::TAU;
        let sway = sway_phase.sin() * self.config.body_sway;

        // Apply motion
        let up = self.base_rotation * Vec3::Y;
        let right = self.base_rotation * Vec3::X;

        self.body_position = self.base_position + up * bob + right * sway;

        // Forward lean based on speed
        let normalized_speed = (speed / 2.0).clamp(0.0, 1.0);
        let lean = self.config.lean_amount * normalized_speed;
        let lean_rotation = Quat::from_axis_angle(Vec3::X, lean);
        self.body_rotation = self.base_rotation * lean_rotation;
    }

    /// Returns current foot positions.
    pub fn foot_positions(&self) -> Vec<Vec3> {
        self.legs.iter().map(|l| l.foot_position).collect()
    }

    /// Returns which legs are currently planted.
    pub fn planted_legs(&self) -> Vec<bool> {
        self.legs.iter().map(|l| l.planted).collect()
    }

    /// Resets the walk cycle to standing position.
    pub fn reset(&mut self, position: Vec3, rotation: Quat) {
        self.base_position = position;
        self.base_rotation = rotation;
        self.body_position = position;
        self.body_rotation = rotation;
        self.velocity = Vec3::ZERO;
        self.master_phase = 0.0;

        for leg in &mut self.legs {
            let hip_world = position + rotation * leg.hip_offset;
            let foot_world = hip_world + rotation * leg.rest_offset;
            leg.foot_position = foot_world;
            leg.target_position = foot_world;
            leg.previous_position = foot_world;
            leg.phase = 0.0;
            leg.planted = true;
        }
    }

    /// Returns ground height at current position (for terrain adaptation).
    /// Override this for actual terrain.
    pub fn ground_height_at(&self, _position: Vec3) -> f32 {
        0.0
    }
}

/// Computes rotation facing a direction.
fn rotation_from_direction(dir: Vec3) -> Quat {
    if dir.length_squared() < 0.0001 {
        return Quat::IDENTITY;
    }
    let forward = dir.normalize();
    let up = Vec3::Y;
    let right = up.cross(forward).normalize_or_zero();
    let corrected_up = forward.cross(right);

    Quat::from_mat3(&glam::Mat3::from_cols(right, corrected_up, forward))
}

/// Foot placement for terrain adaptation.
#[derive(Debug, Clone)]
pub struct FootPlacement {
    /// Maximum raycast distance for ground detection.
    pub max_distance: f32,
    /// Offset above ground for foot.
    pub foot_height: f32,
    /// Speed of foot adaptation to terrain.
    pub adaptation_speed: f32,
}

impl Default for FootPlacement {
    fn default() -> Self {
        Self {
            max_distance: 2.0,
            foot_height: 0.0,
            adaptation_speed: 10.0,
        }
    }
}

impl FootPlacement {
    /// Adjusts a foot target to meet ground height.
    pub fn adjust_to_ground(&self, target: Vec3, ground_height: f32) -> Vec3 {
        Vec3::new(target.x, ground_height + self.foot_height, target.z)
    }
}

/// Applies procedural walk to a skeleton pose.
pub struct WalkAnimator {
    /// The walk controller.
    pub walk: ProceduralWalk,
    /// IK chains for each leg (root to foot).
    pub leg_chains: Vec<IkChain>,
    /// IK configuration.
    pub ik_config: IkConfig,
}

impl WalkAnimator {
    /// Creates a new walk animator.
    pub fn new(walk: ProceduralWalk, leg_chains: Vec<IkChain>) -> Self {
        assert_eq!(
            walk.legs.len(),
            leg_chains.len(),
            "Leg count must match IK chain count"
        );
        Self {
            walk,
            leg_chains,
            ik_config: IkConfig::default(),
        }
    }

    /// Updates and applies walk animation to pose.
    pub fn update(&mut self, skeleton: &Skeleton, pose: &mut Pose, dt: f32, velocity: Vec3) {
        self.walk.update(dt, velocity);

        // Apply IK for each leg
        let foot_positions = self.walk.foot_positions();
        for (i, chain) in self.leg_chains.iter().enumerate() {
            solve_fabrik(skeleton, pose, chain, foot_positions[i], &self.ik_config);
        }
    }

    /// Returns the current body transform.
    pub fn body_transform(&self) -> Transform {
        Transform {
            translation: self.walk.body_position,
            rotation: self.walk.body_rotation,
            scale: Vec3::ONE,
        }
    }
}

/// Simple procedural hop (for small creatures or jump).
#[derive(Debug, Clone)]
pub struct ProceduralHop {
    /// Height of hop.
    pub height: f32,
    /// Duration of hop cycle.
    pub duration: f32,
    /// Current phase.
    phase: f32,
    /// Is currently hopping.
    hopping: bool,
    /// Start position.
    start_position: Vec3,
    /// Target position.
    target_position: Vec3,
}

impl ProceduralHop {
    /// Creates a new hop controller.
    pub fn new(height: f32, duration: f32) -> Self {
        Self {
            height,
            duration,
            phase: 0.0,
            hopping: false,
            start_position: Vec3::ZERO,
            target_position: Vec3::ZERO,
        }
    }

    /// Starts a hop toward a target.
    pub fn hop_to(&mut self, from: Vec3, to: Vec3) {
        self.start_position = from;
        self.target_position = to;
        self.phase = 0.0;
        self.hopping = true;
    }

    /// Updates the hop and returns current position.
    pub fn update(&mut self, dt: f32) -> Vec3 {
        if !self.hopping {
            return self.target_position;
        }

        self.phase += dt / self.duration;
        if self.phase >= 1.0 {
            self.phase = 1.0;
            self.hopping = false;
            return self.target_position;
        }

        // Horizontal interpolation
        let horizontal = self.start_position.lerp(self.target_position, self.phase);

        // Vertical arc
        let arc = (self.phase * std::f32::consts::PI).sin() * self.height;

        Vec3::new(horizontal.x, horizontal.y + arc, horizontal.z)
    }

    /// Returns true if currently hopping.
    pub fn is_hopping(&self) -> bool {
        self.hopping
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gait_config_default() {
        let config = GaitConfig::default();
        assert!(config.stride_length > 0.0);
        assert!(config.step_height > 0.0);
        assert!(config.cycle_duration > 0.0);
        assert!(config.stance_ratio > 0.0 && config.stance_ratio < 1.0);
    }

    #[test]
    fn test_gait_config_presets() {
        let walk = GaitConfig::walk();
        let run = GaitConfig::run();
        let sneak = GaitConfig::sneak();

        // Run should be faster and higher
        assert!(run.stride_length > walk.stride_length);
        assert!(run.step_height > walk.step_height);
        assert!(run.cycle_duration < walk.cycle_duration);

        // Sneak should be slower and lower
        assert!(sneak.stride_length < walk.stride_length);
        assert!(sneak.step_height < walk.step_height);
        assert!(sneak.cycle_duration > walk.cycle_duration);
    }

    #[test]
    fn test_gait_pattern_offsets() {
        assert_eq!(GaitPattern::BipedWalk.phase_offsets().len(), 2);
        assert_eq!(GaitPattern::QuadrupedWalk.phase_offsets().len(), 4);
        assert_eq!(GaitPattern::QuadrupedTrot.phase_offsets().len(), 4);
    }

    #[test]
    fn test_biped_creation() {
        let walk = ProceduralWalk::biped(Vec3::new(0.0, 1.0, 0.0), Quat::IDENTITY, 0.4, 0.8);

        assert_eq!(walk.legs.len(), 2);
        assert_eq!(walk.pattern, GaitPattern::BipedWalk);
    }

    #[test]
    fn test_quadruped_creation() {
        let walk =
            ProceduralWalk::quadruped(Vec3::new(0.0, 0.5, 0.0), Quat::IDENTITY, 0.3, 0.6, 0.4);

        assert_eq!(walk.legs.len(), 4);
        assert_eq!(walk.pattern, GaitPattern::QuadrupedWalk);
    }

    #[test]
    fn test_biped_update() {
        let mut walk = ProceduralWalk::biped(Vec3::new(0.0, 1.0, 0.0), Quat::IDENTITY, 0.4, 0.8);

        // Simulate walking forward
        let velocity = Vec3::new(0.0, 0.0, 1.0);
        for _ in 0..100 {
            walk.update(1.0 / 60.0, velocity);
        }

        // Should have moved forward
        assert!(walk.body_position.z > 0.5);
    }

    #[test]
    fn test_leg_alternation() {
        let mut walk = ProceduralWalk::biped(Vec3::new(0.0, 1.0, 0.0), Quat::IDENTITY, 0.4, 0.8);

        let velocity = Vec3::new(0.0, 0.0, 1.0);

        // Track if both legs ever swap
        let mut left_was_up = false;
        let mut right_was_up = false;

        for _ in 0..120 {
            walk.update(1.0 / 60.0, velocity);
            if !walk.legs[0].planted {
                left_was_up = true;
            }
            if !walk.legs[1].planted {
                right_was_up = true;
            }
        }

        assert!(left_was_up);
        assert!(right_was_up);
    }

    #[test]
    fn test_foot_positions() {
        let walk = ProceduralWalk::biped(Vec3::new(0.0, 1.0, 0.0), Quat::IDENTITY, 0.4, 0.8);

        let positions = walk.foot_positions();
        assert_eq!(positions.len(), 2);

        // Feet should be below body
        assert!(positions[0].y < walk.body_position.y);
        assert!(positions[1].y < walk.body_position.y);
    }

    #[test]
    fn test_planted_legs() {
        let walk = ProceduralWalk::biped(Vec3::new(0.0, 1.0, 0.0), Quat::IDENTITY, 0.4, 0.8);

        let planted = walk.planted_legs();
        assert_eq!(planted.len(), 2);

        // Both legs planted when standing
        assert!(planted[0]);
        assert!(planted[1]);
    }

    #[test]
    fn test_walk_reset() {
        let mut walk = ProceduralWalk::biped(Vec3::new(0.0, 1.0, 0.0), Quat::IDENTITY, 0.4, 0.8);

        // Move it around
        for _ in 0..60 {
            walk.update(1.0 / 60.0, Vec3::new(1.0, 0.0, 1.0));
        }

        // Reset
        let new_pos = Vec3::new(5.0, 1.0, 5.0);
        walk.reset(new_pos, Quat::IDENTITY);

        assert_eq!(walk.body_position, new_pos);
        assert_eq!(walk.master_phase, 0.0);
    }

    #[test]
    fn test_body_bob() {
        let mut walk = ProceduralWalk::biped(Vec3::new(0.0, 1.0, 0.0), Quat::IDENTITY, 0.4, 0.8);

        let velocity = Vec3::new(0.0, 0.0, 2.0);
        let mut min_y = f32::MAX;
        let mut max_y = f32::MIN;

        for _ in 0..120 {
            walk.update(1.0 / 60.0, velocity);
            min_y = min_y.min(walk.body_position.y);
            max_y = max_y.max(walk.body_position.y);
        }

        // Should have some vertical variation
        let variation = max_y - min_y;
        assert!(variation > 0.01);
    }

    #[test]
    fn test_quadruped_trot() {
        let mut walk =
            ProceduralWalk::quadruped(Vec3::new(0.0, 0.5, 0.0), Quat::IDENTITY, 0.3, 0.6, 0.4);
        walk.pattern = GaitPattern::QuadrupedTrot;
        walk.config = GaitConfig::quadruped_trot();

        let velocity = Vec3::new(0.0, 0.0, 1.5);

        for _ in 0..120 {
            walk.update(1.0 / 60.0, velocity);
        }

        // Should move forward
        assert!(walk.body_position.z > 1.0);
    }

    #[test]
    fn test_foot_placement() {
        let placement = FootPlacement::default();
        let target = Vec3::new(1.0, 0.5, 2.0);
        let ground = 0.2;

        let adjusted = placement.adjust_to_ground(target, ground);
        assert_eq!(adjusted.x, target.x);
        assert_eq!(adjusted.y, ground + placement.foot_height);
        assert_eq!(adjusted.z, target.z);
    }

    #[test]
    fn test_procedural_hop() {
        let mut hop = ProceduralHop::new(1.0, 0.5);

        let from = Vec3::ZERO;
        let to = Vec3::new(2.0, 0.0, 0.0);
        hop.hop_to(from, to);

        assert!(hop.is_hopping());

        // Midway should be elevated
        hop.update(0.25);
        let mid = hop.update(0.0);
        assert!(mid.y > 0.5);

        // Finish hop
        for _ in 0..30 {
            hop.update(0.02);
        }

        assert!(!hop.is_hopping());
    }

    #[test]
    fn test_step_height() {
        let mut walk = ProceduralWalk::biped(Vec3::new(0.0, 1.0, 0.0), Quat::IDENTITY, 0.4, 0.8);

        let velocity = Vec3::new(0.0, 0.0, 1.0);
        let mut max_foot_height = f32::MIN;

        for _ in 0..120 {
            walk.update(1.0 / 60.0, velocity);
            for leg in &walk.legs {
                if !leg.planted {
                    max_foot_height = max_foot_height.max(leg.foot_position.y);
                }
            }
        }

        // Foot should lift during step
        assert!(max_foot_height > 0.0);
    }
}
