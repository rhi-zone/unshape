//! L-systems (Lindenmayer systems) for procedural generation.
//!
//! L-systems are string rewriting systems that can generate fractal-like
//! structures, plant models, and other procedural content.
//!
//! # Example
//!
//! ```
//! use unshape_lsystem::{LSystem, Rule, Turtle2D};
//!
//! // Koch curve
//! let lsystem = LSystem::new("F")
//!     .with_rule(Rule::simple('F', "F+F-F-F+F"));
//!
//! let result = lsystem.generate(3);
//! let turtle = Turtle2D { angle: 90.0, ..Default::default() };
//! let segments = turtle.apply(&result);
//! ```

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use glam::{Vec2, Vec3};

/// Registers all lsystem operations with an [`OpRegistry`].
///
/// Call this to enable deserialization of lsystem ops from saved pipelines.
#[cfg(feature = "dynop")]
pub fn register_ops(registry: &mut unshape_op::OpRegistry) {
    registry.register_type::<Turtle2D>("resin::Turtle2D");
    registry.register_type::<Turtle3D>("resin::Turtle3D");
}

use std::collections::HashMap;
use std::f32::consts::PI;

/// A production rule for an L-system.
#[derive(Debug, Clone)]
pub struct Rule {
    /// The symbol to replace.
    pub predecessor: char,
    /// The replacement string.
    pub successor: String,
    /// Optional probability (for stochastic L-systems).
    pub probability: f32,
}

impl Rule {
    /// Creates a simple deterministic rule.
    pub fn simple(predecessor: char, successor: &str) -> Self {
        Self {
            predecessor,
            successor: successor.to_string(),
            probability: 1.0,
        }
    }

    /// Creates a stochastic rule with a given probability.
    pub fn stochastic(predecessor: char, successor: &str, probability: f32) -> Self {
        Self {
            predecessor,
            successor: successor.to_string(),
            probability,
        }
    }
}

/// An L-system definition.
#[derive(Debug, Clone)]
pub struct LSystem {
    /// The starting string.
    pub axiom: String,
    /// Production rules.
    rules: Vec<Rule>,
    /// Random seed for stochastic L-systems.
    pub seed: u64,
}

impl LSystem {
    /// Creates a new L-system with the given axiom.
    pub fn new(axiom: &str) -> Self {
        Self {
            axiom: axiom.to_string(),
            rules: Vec::new(),
            seed: 0,
        }
    }

    /// Adds a production rule.
    pub fn with_rule(mut self, rule: Rule) -> Self {
        self.rules.push(rule);
        self
    }

    /// Sets the random seed for stochastic rules.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Generates the L-system string after n iterations.
    pub fn generate(&self, iterations: usize) -> String {
        let mut current = self.axiom.clone();
        let mut rng = SimpleRng::new(self.seed);

        // Build a map from predecessor to rules for quick lookup
        let mut rule_map: HashMap<char, Vec<&Rule>> = HashMap::new();
        for rule in &self.rules {
            rule_map.entry(rule.predecessor).or_default().push(rule);
        }

        for _ in 0..iterations {
            let mut next = String::with_capacity(current.len() * 2);

            for c in current.chars() {
                if let Some(rules) = rule_map.get(&c) {
                    if rules.len() == 1 {
                        // Deterministic case
                        next.push_str(&rules[0].successor);
                    } else {
                        // Stochastic case - choose based on probability
                        let r = rng.next_f32();
                        let mut cumulative = 0.0;
                        let mut chosen = &rules[0].successor;

                        for rule in rules {
                            cumulative += rule.probability;
                            if r < cumulative {
                                chosen = &rule.successor;
                                break;
                            }
                        }
                        next.push_str(chosen);
                    }
                } else {
                    // No rule - keep the symbol
                    next.push(c);
                }
            }

            current = next;
        }

        current
    }

    /// Returns the length of the generated string after n iterations
    /// without actually generating it.
    pub fn estimate_length(&self, iterations: usize) -> usize {
        // Simple estimate assuming each rule roughly doubles the length
        self.axiom.len() * (2_usize.pow(iterations as u32))
    }
}

/// Interprets an L-system string using 2D turtle graphics.
///
/// Operations are serializable structs with `apply` methods.
/// See `docs/design/ops-as-values.md`.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dynop", derive(unshape_op::Op))]
#[cfg_attr(feature = "dynop", op(input = String, output = Vec<TurtleSegment2D>))]
pub struct Turtle2D {
    /// Rotation angle in degrees for + and - commands.
    pub angle: f32,
    /// Step distance for F and f commands.
    pub step: f32,
    /// Scale factor for push/pop.
    pub scale_factor: f32,
}

impl Default for Turtle2D {
    fn default() -> Self {
        Self {
            angle: 25.0,
            step: 1.0,
            scale_factor: 0.9,
        }
    }
}

impl Turtle2D {
    /// Applies this operation to interpret an L-system string.
    pub fn apply(&self, input: &str) -> Vec<TurtleSegment2D> {
        interpret_turtle_2d(input, &self.into())
    }
}

/// Interprets an L-system string using 3D turtle graphics.
///
/// Operations are serializable structs with `apply` methods.
/// See `docs/design/ops-as-values.md`.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dynop", derive(unshape_op::Op))]
#[cfg_attr(feature = "dynop", op(input = String, output = Vec<TurtleSegment3D>))]
pub struct Turtle3D {
    /// Rotation angle in degrees for + and - commands.
    pub angle: f32,
    /// Step distance for F and f commands.
    pub step: f32,
    /// Scale factor for push/pop.
    pub scale_factor: f32,
}

impl Default for Turtle3D {
    fn default() -> Self {
        Self {
            angle: 25.0,
            step: 1.0,
            scale_factor: 0.9,
        }
    }
}

impl Turtle3D {
    /// Applies this operation to interpret an L-system string.
    pub fn apply(&self, input: &str) -> Vec<TurtleSegment3D> {
        interpret_turtle_3d(input, &self.into())
    }
}

/// Configuration for turtle interpretation.
///
/// Backwards-compatible type alias for `Turtle2D`.
pub type TurtleConfig = Turtle2D;

/// Conversion from Turtle2D to internal config for interpret functions.
impl From<&Turtle2D> for TurtleConfigInternal {
    fn from(t: &Turtle2D) -> Self {
        TurtleConfigInternal {
            angle: t.angle,
            step: t.step,
            scale_factor: t.scale_factor,
        }
    }
}

/// Conversion from Turtle3D to internal config for interpret functions.
impl From<&Turtle3D> for TurtleConfigInternal {
    fn from(t: &Turtle3D) -> Self {
        TurtleConfigInternal {
            angle: t.angle,
            step: t.step,
            scale_factor: t.scale_factor,
        }
    }
}

/// Internal configuration for turtle interpretation functions.
#[derive(Debug, Clone)]
struct TurtleConfigInternal {
    /// Rotation angle in degrees for + and - commands.
    angle: f32,
    /// Step distance for F and f commands.
    step: f32,
    /// Scale factor for push/pop.
    scale_factor: f32,
}

/// 2D turtle state.
#[derive(Debug, Clone, Copy)]
pub struct TurtleState2D {
    /// Current position.
    pub position: Vec2,
    /// Current heading in radians.
    pub angle: f32,
    /// Distance to move per step.
    pub step: f32,
}

/// Segment produced by turtle interpretation.
#[derive(Debug, Clone)]
pub struct TurtleSegment2D {
    /// Start point of the segment.
    pub start: Vec2,
    /// End point of the segment.
    pub end: Vec2,
    /// Recursion depth when this segment was drawn.
    pub depth: usize,
}

/// Interprets an L-system string using 2D turtle graphics.
///
/// Standard turtle commands:
/// - F: Move forward, drawing a line
/// - f: Move forward, no line
/// - +: Turn left by angle
/// - -: Turn right by angle
/// - [: Push state
/// - ]: Pop state
/// - |: Turn around (180 degrees)
fn interpret_turtle_2d(input: &str, config: &TurtleConfigInternal) -> Vec<TurtleSegment2D> {
    let mut segments = Vec::new();
    let mut stack = Vec::new();
    let angle_rad = config.angle * PI / 180.0;

    let mut state = TurtleState2D {
        position: Vec2::ZERO,
        angle: PI / 2.0, // Start pointing up
        step: config.step,
    };

    let mut depth = 0;

    for c in input.chars() {
        match c {
            'F' | 'G' => {
                // Move forward, drawing a line
                let dir = Vec2::new(state.angle.cos(), state.angle.sin());
                let new_pos = state.position + dir * state.step;

                segments.push(TurtleSegment2D {
                    start: state.position,
                    end: new_pos,
                    depth,
                });

                state.position = new_pos;
            }
            'f' | 'g' => {
                // Move forward, no line
                let dir = Vec2::new(state.angle.cos(), state.angle.sin());
                state.position = state.position + dir * state.step;
            }
            '+' => {
                state.angle += angle_rad;
            }
            '-' => {
                state.angle -= angle_rad;
            }
            '|' => {
                state.angle += PI;
            }
            '[' => {
                stack.push((state, depth));
                state.step *= config.scale_factor;
                depth += 1;
            }
            ']' => {
                if let Some((saved_state, saved_depth)) = stack.pop() {
                    state = saved_state;
                    depth = saved_depth;
                }
            }
            _ => {} // Ignore other characters
        }
    }

    segments
}

/// 3D turtle state.
#[derive(Debug, Clone, Copy)]
pub struct TurtleState3D {
    /// Current position.
    pub position: Vec3,
    /// Heading (forward direction).
    pub heading: Vec3,
    /// Left direction.
    pub left: Vec3,
    /// Up direction.
    pub up: Vec3,
    /// Distance to move per step.
    pub step: f32,
}

impl Default for TurtleState3D {
    fn default() -> Self {
        Self {
            position: Vec3::ZERO,
            heading: Vec3::Y, // Forward is +Y
            left: Vec3::NEG_X,
            up: Vec3::Z,
            step: 1.0,
        }
    }
}

/// Segment produced by 3D turtle interpretation.
#[derive(Debug, Clone)]
pub struct TurtleSegment3D {
    /// Start point of the segment.
    pub start: Vec3,
    /// End point of the segment.
    pub end: Vec3,
    /// Recursion depth when this segment was drawn.
    pub depth: usize,
}

/// Interprets an L-system string using 3D turtle graphics.
///
/// Additional 3D commands:
/// - &: Pitch down
/// - ^: Pitch up
/// - \\: Roll left
/// - /: Roll right
fn interpret_turtle_3d(input: &str, config: &TurtleConfigInternal) -> Vec<TurtleSegment3D> {
    let mut segments = Vec::new();
    let mut stack = Vec::new();
    let angle_rad = config.angle * PI / 180.0;

    let mut state = TurtleState3D {
        step: config.step,
        ..Default::default()
    };

    let mut depth = 0;

    for c in input.chars() {
        match c {
            'F' | 'G' => {
                let new_pos = state.position + state.heading * state.step;

                segments.push(TurtleSegment3D {
                    start: state.position,
                    end: new_pos,
                    depth,
                });

                state.position = new_pos;
            }
            'f' | 'g' => {
                state.position = state.position + state.heading * state.step;
            }
            '+' => {
                // Turn left (yaw)
                let axis = state.up;
                rotate_turtle_3d(&mut state, axis, angle_rad);
            }
            '-' => {
                // Turn right
                let axis = state.up;
                rotate_turtle_3d(&mut state, axis, -angle_rad);
            }
            '&' => {
                // Pitch down
                let axis = state.left;
                rotate_turtle_3d(&mut state, axis, angle_rad);
            }
            '^' => {
                // Pitch up
                let axis = state.left;
                rotate_turtle_3d(&mut state, axis, -angle_rad);
            }
            '\\' => {
                // Roll left
                let axis = state.heading;
                rotate_turtle_3d(&mut state, axis, angle_rad);
            }
            '/' => {
                // Roll right
                let axis = state.heading;
                rotate_turtle_3d(&mut state, axis, -angle_rad);
            }
            '|' => {
                // Turn around
                let axis = state.up;
                rotate_turtle_3d(&mut state, axis, PI);
            }
            '[' => {
                stack.push((state, depth));
                state.step *= config.scale_factor;
                depth += 1;
            }
            ']' => {
                if let Some((saved_state, saved_depth)) = stack.pop() {
                    state = saved_state;
                    depth = saved_depth;
                }
            }
            _ => {}
        }
    }

    segments
}

/// Rotates the turtle state around an axis.
fn rotate_turtle_3d(state: &mut TurtleState3D, axis: Vec3, angle: f32) {
    let cos_a = angle.cos();
    let sin_a = angle.sin();

    let rotate = |v: Vec3| -> Vec3 {
        let dot = axis.dot(v);
        let cross = axis.cross(v);
        v * cos_a + cross * sin_a + axis * dot * (1.0 - cos_a)
    };

    state.heading = rotate(state.heading).normalize();
    state.left = rotate(state.left).normalize();
    state.up = rotate(state.up).normalize();
}

/// Converts 2D turtle segments to a list of Vec2 paths.
pub fn segments_to_paths_2d(segments: &[TurtleSegment2D]) -> Vec<Vec<Vec2>> {
    if segments.is_empty() {
        return vec![];
    }

    let mut paths = Vec::new();
    let mut current_path = vec![segments[0].start, segments[0].end];

    for window in segments.windows(2) {
        let prev = &window[0];
        let next = &window[1];

        if (prev.end - next.start).length() < 0.0001 {
            // Connected - extend current path
            current_path.push(next.end);
        } else {
            // Disconnected - start new path
            paths.push(current_path);
            current_path = vec![next.start, next.end];
        }
    }

    if !current_path.is_empty() {
        paths.push(current_path);
    }

    paths
}

/// Preset L-systems for common patterns.
pub mod presets {
    use super::{LSystem, Rule};

    /// Koch curve - classic snowflake fractal.
    pub fn koch_curve() -> LSystem {
        LSystem::new("F").with_rule(Rule::simple('F', "F+F-F-F+F"))
    }

    /// Koch snowflake.
    pub fn koch_snowflake() -> LSystem {
        LSystem::new("F--F--F").with_rule(Rule::simple('F', "F+F--F+F"))
    }

    /// Sierpinski triangle.
    pub fn sierpinski_triangle() -> LSystem {
        LSystem::new("F-G-G")
            .with_rule(Rule::simple('F', "F-G+F+G-F"))
            .with_rule(Rule::simple('G', "GG"))
    }

    /// Dragon curve.
    pub fn dragon_curve() -> LSystem {
        LSystem::new("F")
            .with_rule(Rule::simple('F', "F+G"))
            .with_rule(Rule::simple('G', "F-G"))
    }

    /// Hilbert curve.
    pub fn hilbert_curve() -> LSystem {
        LSystem::new("A")
            .with_rule(Rule::simple('A', "-BF+AFA+FB-"))
            .with_rule(Rule::simple('B', "+AF-BFB-FA+"))
    }

    /// Simple tree (2D).
    pub fn simple_tree() -> LSystem {
        LSystem::new("X")
            .with_rule(Rule::simple('X', "F[+X][-X]FX"))
            .with_rule(Rule::simple('F', "FF"))
    }

    /// Binary tree.
    pub fn binary_tree() -> LSystem {
        LSystem::new("X")
            .with_rule(Rule::simple('X', "F[+X]F[-X]+X"))
            .with_rule(Rule::simple('F', "FF"))
    }

    /// Fractal plant.
    pub fn fractal_plant() -> LSystem {
        LSystem::new("X")
            .with_rule(Rule::simple('X', "F+[[X]-X]-F[-FX]+X"))
            .with_rule(Rule::simple('F', "FF"))
    }

    /// Stochastic tree with variation.
    pub fn stochastic_tree() -> LSystem {
        LSystem::new("X")
            .with_rule(Rule::stochastic('X', "F[+X][-X]FX", 0.5))
            .with_rule(Rule::stochastic('X', "F[-X]FX", 0.3))
            .with_rule(Rule::stochastic('X', "F[+X]FX", 0.2))
            .with_rule(Rule::simple('F', "FF"))
    }

    /// 3D tree.
    pub fn tree_3d() -> LSystem {
        LSystem::new("A")
            .with_rule(Rule::simple('A', "[&FL!A]/////'[&FL!A]///////'[&FL!A]"))
            .with_rule(Rule::simple('F', "S//F"))
            .with_rule(Rule::simple('S', "FL"))
            .with_rule(Rule::simple('L', "['''^^{-f+f+f-|-f+f+f}]"))
    }
}

/// Simple RNG for stochastic L-systems.
struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_add(1),
        }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.state
    }

    fn next_f32(&mut self) -> f32 {
        (self.next_u64() as f64 / u64::MAX as f64) as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_rule() {
        let lsystem = LSystem::new("A")
            .with_rule(Rule::simple('A', "AB"))
            .with_rule(Rule::simple('B', "A"));

        assert_eq!(lsystem.generate(0), "A");
        assert_eq!(lsystem.generate(1), "AB");
        assert_eq!(lsystem.generate(2), "ABA");
        assert_eq!(lsystem.generate(3), "ABAAB");
    }

    #[test]
    fn test_no_rule() {
        let lsystem = LSystem::new("ABC").with_rule(Rule::simple('A', "AA"));

        assert_eq!(lsystem.generate(1), "AABC");
    }

    #[test]
    fn test_koch_curve() {
        let lsystem = presets::koch_curve();
        let result = lsystem.generate(1);
        assert_eq!(result, "F+F-F-F+F");
    }

    #[test]
    fn test_turtle_2d() {
        let lsystem = LSystem::new("F+F+F+F").with_rule(Rule::simple('F', "F"));

        let result = lsystem.generate(0);
        let turtle = Turtle2D {
            angle: 90.0,
            ..Default::default()
        };
        let segments = turtle.apply(&result);

        // Should produce 4 segments forming a square
        assert_eq!(segments.len(), 4);
    }

    #[test]
    fn test_turtle_push_pop() {
        let result = "F[+F]F";
        let turtle = Turtle2D {
            angle: 90.0,
            ..Default::default()
        };
        let segments = turtle.apply(result);

        // Should produce 3 segments
        assert_eq!(segments.len(), 3);
    }

    #[test]
    fn test_stochastic() {
        let lsystem = LSystem::new("A")
            .with_rule(Rule::stochastic('A', "X", 0.5))
            .with_rule(Rule::stochastic('A', "Y", 0.5))
            .with_seed(12345);

        let result1 = lsystem.generate(1);

        let lsystem2 = lsystem.clone().with_seed(67890);
        let result2 = lsystem2.generate(1);

        // With different seeds, may get different results
        // (though with only one iteration and 50/50 odds, might be same)
        assert!(result1 == "X" || result1 == "Y");
        assert!(result2 == "X" || result2 == "Y");
    }

    #[test]
    fn test_presets() {
        // Just verify all presets can be generated
        let _koch = presets::koch_curve().generate(2);
        let _snowflake = presets::koch_snowflake().generate(2);
        let _sierpinski = presets::sierpinski_triangle().generate(2);
        let _dragon = presets::dragon_curve().generate(2);
        let _hilbert = presets::hilbert_curve().generate(2);
        let _tree = presets::simple_tree().generate(2);
        let _binary = presets::binary_tree().generate(2);
        let _plant = presets::fractal_plant().generate(2);
        let _stoch = presets::stochastic_tree().generate(2);
        let _tree3d = presets::tree_3d().generate(2);
    }

    #[test]
    fn test_segments_to_paths() {
        let segments = vec![
            TurtleSegment2D {
                start: Vec2::ZERO,
                end: Vec2::X,
                depth: 0,
            },
            TurtleSegment2D {
                start: Vec2::X,
                end: Vec2::new(2.0, 0.0),
                depth: 0,
            },
            // Gap
            TurtleSegment2D {
                start: Vec2::Y,
                end: Vec2::new(0.0, 2.0),
                depth: 0,
            },
        ];

        let paths = segments_to_paths_2d(&segments);
        assert_eq!(paths.len(), 2);
        assert_eq!(paths[0].len(), 3);
        assert_eq!(paths[1].len(), 2);
    }

    #[test]
    fn test_turtle_3d() {
        let result = "F+F&F";
        let turtle = Turtle3D {
            angle: 90.0,
            ..Default::default()
        };
        let segments = turtle.apply(result);

        assert_eq!(segments.len(), 3);
    }
}
