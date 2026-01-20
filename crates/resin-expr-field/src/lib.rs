//! Bridge between expressions and fields.
//!
//! Provides `ExprField` which evaluates expressions as spatial fields,
//! and noise expression functions for use in expressions.
//!
//! This crate also provides JIT compilation for `FieldExpr` via the `cranelift` feature.
//!
//! # Example
//!
//! ```
//! use rhizome_resin_expr_field::{ExprField, register_noise, scalar_registry};
//! use rhizome_resin_field::{Field, EvalContext};
//! use glam::Vec2;
//!
//! // Create registry with standard math + noise functions
//! let mut registry = scalar_registry();
//! register_noise(&mut registry);
//!
//! let field = ExprField::parse("sin(x * 3.14159) + noise(x, y)", registry).unwrap();
//! let ctx = EvalContext::new();
//! let value: f32 = field.sample(Vec2::new(0.5, 0.5), &ctx);
//! ```

#[cfg(feature = "cranelift")]
pub mod jit_impl;

#[cfg(feature = "cranelift")]
pub use jit_impl::{CompiledFieldExpr, FieldExprCompiler, FieldJitError, FieldJitResult};

use glam::{Vec2, Vec3};
use rhizome_dew_core::{Expr, ParseError};
use rhizome_dew_scalar::{FunctionRegistry, ScalarFn};
use rhizome_resin_field::{EvalContext, Field};
use std::collections::{HashMap, HashSet};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

// Re-export dew types for convenience
pub use rhizome_dew_core::{Ast, BinOp, UnaryOp};
pub use rhizome_dew_scalar::{Error as EvalError, scalar_registry};

/// Built-in variables that are automatically bound during field evaluation.
pub const BUILTIN_VARS: &[&str] = &["x", "y", "z", "t", "time"];

// ============================================================================
// Noise expression functions
// ============================================================================

/// 2D Perlin noise: noise(x, y)
pub struct Noise;
impl ScalarFn<f32> for Noise {
    fn name(&self) -> &str {
        "noise"
    }
    fn arg_count(&self) -> usize {
        2
    }
    fn call(&self, args: &[f32]) -> f32 {
        use rhizome_resin_noise::Noise2D;
        let [x, y] = args else { return 0.0 };
        rhizome_resin_noise::Perlin2D::new().sample(*x, *y)
    }
}

/// 2D Perlin noise: perlin(x, y)
pub struct Perlin;
impl ScalarFn<f32> for Perlin {
    fn name(&self) -> &str {
        "perlin"
    }
    fn arg_count(&self) -> usize {
        2
    }
    fn call(&self, args: &[f32]) -> f32 {
        use rhizome_resin_noise::Noise2D;
        let [x, y] = args else { return 0.0 };
        rhizome_resin_noise::Perlin2D::new().sample(*x, *y)
    }
}

/// 3D Perlin noise: perlin3(x, y, z)
pub struct Perlin3;
impl ScalarFn<f32> for Perlin3 {
    fn name(&self) -> &str {
        "perlin3"
    }
    fn arg_count(&self) -> usize {
        3
    }
    fn call(&self, args: &[f32]) -> f32 {
        use rhizome_resin_noise::Noise3D;
        let [x, y, z] = args else { return 0.0 };
        rhizome_resin_noise::Perlin3D::new().sample(*x, *y, *z)
    }
}

/// 2D Simplex noise: simplex(x, y)
pub struct Simplex;
impl ScalarFn<f32> for Simplex {
    fn name(&self) -> &str {
        "simplex"
    }
    fn arg_count(&self) -> usize {
        2
    }
    fn call(&self, args: &[f32]) -> f32 {
        use rhizome_resin_noise::Noise2D;
        let [x, y] = args else { return 0.0 };
        rhizome_resin_noise::Simplex2D::new().sample(*x, *y)
    }
}

/// 3D Simplex noise: simplex3(x, y, z)
pub struct Simplex3;
impl ScalarFn<f32> for Simplex3 {
    fn name(&self) -> &str {
        "simplex3"
    }
    fn arg_count(&self) -> usize {
        3
    }
    fn call(&self, args: &[f32]) -> f32 {
        use rhizome_resin_noise::Noise3D;
        let [x, y, z] = args else { return 0.0 };
        rhizome_resin_noise::Simplex3D::new().sample(*x, *y, *z)
    }
}

/// 2D FBM noise: fbm(x, y, octaves)
pub struct Fbm;
impl ScalarFn<f32> for Fbm {
    fn name(&self) -> &str {
        "fbm"
    }
    fn arg_count(&self) -> usize {
        3
    }
    fn call(&self, args: &[f32]) -> f32 {
        use rhizome_resin_noise::Noise2D;
        let [x, y, octaves] = args else { return 0.0 };
        rhizome_resin_noise::Fbm::new(rhizome_resin_noise::Perlin2D::new())
            .octaves(*octaves as u32)
            .sample(*x, *y)
    }
}

/// Registers noise functions into a FunctionRegistry.
pub fn register_noise(registry: &mut FunctionRegistry<f32>) {
    registry.register(Noise);
    registry.register(Perlin);
    registry.register(Perlin3);
    registry.register(Simplex);
    registry.register(Simplex3);
    registry.register(Fbm);
}

// ============================================================================
// ExprField
// ============================================================================

/// An expression bundled with its function registry for use as a Field.
///
/// ExprField bridges the expression language to the Field system by:
/// - Storing an expression and its function registry
/// - Mapping input positions to variable bindings (x, y, z)
/// - Mapping EvalContext to the `time` variable
pub struct ExprField {
    expr: Expr,
    registry: FunctionRegistry<f32>,
}

impl ExprField {
    /// Creates a new ExprField with the given registry.
    pub fn new(expr: Expr, registry: FunctionRegistry<f32>) -> Self {
        Self { expr, registry }
    }

    /// Parses an expression and creates an ExprField with the given registry.
    pub fn parse(input: &str, registry: FunctionRegistry<f32>) -> Result<Self, ParseError> {
        Ok(Self::new(Expr::parse(input)?, registry))
    }

    /// Evaluates with explicit variable bindings.
    pub fn eval(&self, vars: &HashMap<String, f32>) -> Result<f32, EvalError> {
        rhizome_dew_scalar::eval(self.expr.ast(), vars, &self.registry)
    }

    /// Returns all free variables referenced in the expression.
    pub fn free_vars(&self) -> HashSet<&str> {
        self.expr.free_vars()
    }

    /// Returns user-defined variables (free vars minus builtins like x, y, z, t, time).
    ///
    /// These are the variables that need to be bound by the user.
    pub fn user_inputs(&self) -> HashSet<&str> {
        self.expr
            .free_vars()
            .into_iter()
            .filter(|v| !BUILTIN_VARS.contains(v))
            .collect()
    }

    /// Returns the underlying expression.
    pub fn expr(&self) -> &Expr {
        &self.expr
    }
}

impl Field<Vec2, f32> for ExprField {
    fn sample(&self, input: Vec2, ctx: &EvalContext) -> f32 {
        let vars: HashMap<String, f32> = [
            ("x".to_string(), input.x),
            ("y".to_string(), input.y),
            ("time".to_string(), ctx.time),
            ("t".to_string(), ctx.time),
        ]
        .into();
        rhizome_dew_scalar::eval(self.expr.ast(), &vars, &self.registry).unwrap_or(0.0)
    }
}

impl Field<Vec3, f32> for ExprField {
    fn sample(&self, input: Vec3, ctx: &EvalContext) -> f32 {
        let vars: HashMap<String, f32> = [
            ("x".to_string(), input.x),
            ("y".to_string(), input.y),
            ("z".to_string(), input.z),
            ("time".to_string(), ctx.time),
            ("t".to_string(), ctx.time),
        ]
        .into();
        rhizome_dew_scalar::eval(self.expr.ast(), &vars, &self.registry).unwrap_or(0.0)
    }
}

// ============================================================================
// FieldExpr - Typed AST for field expressions
// ============================================================================

/// A typed expression AST for spatial fields ((x, y, z) → value).
///
/// Unlike raw dew AST, this has typed variants for each field function,
/// enabling UI introspection, JSON serialization, and GPU compilation.
///
/// # Example
///
/// ```
/// use rhizome_resin_expr_field::FieldExpr;
///
/// // Build expression: perlin(x * 4, y * 4) + 0.5 * simplex(x * 8, y * 8)
/// let expr = FieldExpr::Add(
///     Box::new(FieldExpr::Perlin2 {
///         x: Box::new(FieldExpr::Mul(
///             Box::new(FieldExpr::X),
///             Box::new(FieldExpr::Constant(4.0)),
///         )),
///         y: Box::new(FieldExpr::Mul(
///             Box::new(FieldExpr::Y),
///             Box::new(FieldExpr::Constant(4.0)),
///         )),
///     }),
///     Box::new(FieldExpr::Mul(
///         Box::new(FieldExpr::Constant(0.5)),
///         Box::new(FieldExpr::Simplex2 {
///             x: Box::new(FieldExpr::Mul(
///                 Box::new(FieldExpr::X),
///                 Box::new(FieldExpr::Constant(8.0)),
///             )),
///             y: Box::new(FieldExpr::Mul(
///                 Box::new(FieldExpr::Y),
///                 Box::new(FieldExpr::Constant(8.0)),
///             )),
///         }),
///     )),
/// );
///
/// // Evaluate at position (0.5, 0.5, 0.0)
/// let value = expr.eval(0.5, 0.5, 0.0, 0.0, &Default::default());
/// ```
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum FieldExpr {
    // === Coordinates ===
    /// X coordinate.
    X,

    /// Y coordinate.
    Y,

    /// Z coordinate.
    Z,

    /// Time coordinate.
    T,

    // === Literals ===
    /// Constant value.
    Constant(f32),

    /// Variable reference (user-defined).
    Var(String),

    // === Binary operations ===
    /// Addition.
    Add(Box<FieldExpr>, Box<FieldExpr>),

    /// Subtraction.
    Sub(Box<FieldExpr>, Box<FieldExpr>),

    /// Multiplication.
    Mul(Box<FieldExpr>, Box<FieldExpr>),

    /// Division.
    Div(Box<FieldExpr>, Box<FieldExpr>),

    /// Modulo.
    Mod(Box<FieldExpr>, Box<FieldExpr>),

    /// Power.
    Pow(Box<FieldExpr>, Box<FieldExpr>),

    // === Unary operations ===
    /// Negation.
    Neg(Box<FieldExpr>),

    // === Noise functions ===
    /// 2D Perlin noise.
    Perlin2 {
        x: Box<FieldExpr>,
        y: Box<FieldExpr>,
    },

    /// 3D Perlin noise.
    Perlin3 {
        x: Box<FieldExpr>,
        y: Box<FieldExpr>,
        z: Box<FieldExpr>,
    },

    /// 2D Simplex noise.
    Simplex2 {
        x: Box<FieldExpr>,
        y: Box<FieldExpr>,
    },

    /// 3D Simplex noise.
    Simplex3 {
        x: Box<FieldExpr>,
        y: Box<FieldExpr>,
        z: Box<FieldExpr>,
    },

    /// 2D FBM (fractional Brownian motion) noise.
    Fbm2 {
        x: Box<FieldExpr>,
        y: Box<FieldExpr>,
        octaves: u32,
    },

    /// 3D FBM noise.
    Fbm3 {
        x: Box<FieldExpr>,
        y: Box<FieldExpr>,
        z: Box<FieldExpr>,
        octaves: u32,
    },

    // === Distance functions ===
    /// 2D distance from point.
    Distance2 {
        x: Box<FieldExpr>,
        y: Box<FieldExpr>,
        px: f32,
        py: f32,
    },

    /// 3D distance from point.
    Distance3 {
        x: Box<FieldExpr>,
        y: Box<FieldExpr>,
        z: Box<FieldExpr>,
        px: f32,
        py: f32,
        pz: f32,
    },

    /// 2D length (distance from origin).
    Length2 {
        x: Box<FieldExpr>,
        y: Box<FieldExpr>,
    },

    /// 3D length (distance from origin).
    Length3 {
        x: Box<FieldExpr>,
        y: Box<FieldExpr>,
        z: Box<FieldExpr>,
    },

    // === SDF operations ===
    /// SDF circle: length(p) - radius.
    SdfCircle {
        x: Box<FieldExpr>,
        y: Box<FieldExpr>,
        radius: f32,
    },

    /// SDF sphere: length(p) - radius.
    SdfSphere {
        x: Box<FieldExpr>,
        y: Box<FieldExpr>,
        z: Box<FieldExpr>,
        radius: f32,
    },

    /// SDF box (2D).
    SdfBox2 {
        x: Box<FieldExpr>,
        y: Box<FieldExpr>,
        half_width: f32,
        half_height: f32,
    },

    /// SDF box (3D).
    SdfBox3 {
        x: Box<FieldExpr>,
        y: Box<FieldExpr>,
        z: Box<FieldExpr>,
        half_x: f32,
        half_y: f32,
        half_z: f32,
    },

    /// Smooth union of two SDFs.
    SdfSmoothUnion {
        a: Box<FieldExpr>,
        b: Box<FieldExpr>,
        k: f32,
    },

    /// Smooth intersection of two SDFs.
    SdfSmoothIntersection {
        a: Box<FieldExpr>,
        b: Box<FieldExpr>,
        k: f32,
    },

    /// Smooth subtraction of two SDFs.
    SdfSmoothSubtraction {
        a: Box<FieldExpr>,
        b: Box<FieldExpr>,
        k: f32,
    },

    // === Math functions ===
    /// Sine.
    Sin(Box<FieldExpr>),

    /// Cosine.
    Cos(Box<FieldExpr>),

    /// Tangent.
    Tan(Box<FieldExpr>),

    /// Absolute value.
    Abs(Box<FieldExpr>),

    /// Floor.
    Floor(Box<FieldExpr>),

    /// Ceiling.
    Ceil(Box<FieldExpr>),

    /// Fractional part.
    Fract(Box<FieldExpr>),

    /// Square root.
    Sqrt(Box<FieldExpr>),

    /// Exponential.
    Exp(Box<FieldExpr>),

    /// Natural logarithm.
    Ln(Box<FieldExpr>),

    /// Sign (-1, 0, or 1).
    Sign(Box<FieldExpr>),

    /// Minimum of two values.
    Min(Box<FieldExpr>, Box<FieldExpr>),

    /// Maximum of two values.
    Max(Box<FieldExpr>, Box<FieldExpr>),

    /// Clamp value to range.
    Clamp {
        value: Box<FieldExpr>,
        min: Box<FieldExpr>,
        max: Box<FieldExpr>,
    },

    /// Linear interpolation (mix).
    Lerp {
        a: Box<FieldExpr>,
        b: Box<FieldExpr>,
        t: Box<FieldExpr>,
    },

    /// Smooth step (Hermite interpolation).
    SmoothStep {
        edge0: Box<FieldExpr>,
        edge1: Box<FieldExpr>,
        x: Box<FieldExpr>,
    },

    /// Step function.
    Step {
        edge: Box<FieldExpr>,
        x: Box<FieldExpr>,
    },

    // === Conditionals ===
    /// If-then-else.
    IfThenElse {
        condition: Box<FieldExpr>,
        then_expr: Box<FieldExpr>,
        else_expr: Box<FieldExpr>,
    },

    /// Greater than (returns 1.0 or 0.0).
    Gt(Box<FieldExpr>, Box<FieldExpr>),

    /// Less than (returns 1.0 or 0.0).
    Lt(Box<FieldExpr>, Box<FieldExpr>),

    /// Equal (within epsilon, returns 1.0 or 0.0).
    Eq(Box<FieldExpr>, Box<FieldExpr>),
}

impl FieldExpr {
    /// Evaluate the expression at position (x, y, z) and time t with variable bindings.
    pub fn eval(&self, x: f32, y: f32, z: f32, t: f32, vars: &HashMap<String, f32>) -> f32 {
        match self {
            // Coordinates
            Self::X => x,
            Self::Y => y,
            Self::Z => z,
            Self::T => t,

            // Literals
            Self::Constant(v) => *v,
            Self::Var(name) => *vars.get(name).unwrap_or(&0.0),

            // Binary ops
            Self::Add(a, b) => a.eval(x, y, z, t, vars) + b.eval(x, y, z, t, vars),
            Self::Sub(a, b) => a.eval(x, y, z, t, vars) - b.eval(x, y, z, t, vars),
            Self::Mul(a, b) => a.eval(x, y, z, t, vars) * b.eval(x, y, z, t, vars),
            Self::Div(a, b) => a.eval(x, y, z, t, vars) / b.eval(x, y, z, t, vars),
            Self::Mod(a, b) => a.eval(x, y, z, t, vars) % b.eval(x, y, z, t, vars),
            Self::Pow(a, b) => a.eval(x, y, z, t, vars).powf(b.eval(x, y, z, t, vars)),

            // Unary ops
            Self::Neg(a) => -a.eval(x, y, z, t, vars),

            // Noise functions
            Self::Perlin2 { x: ex, y: ey } => {
                use rhizome_resin_noise::Noise2D;
                rhizome_resin_noise::Perlin2D::new()
                    .sample(ex.eval(x, y, z, t, vars), ey.eval(x, y, z, t, vars))
            }
            Self::Perlin3 {
                x: ex,
                y: ey,
                z: ez,
            } => {
                use rhizome_resin_noise::Noise3D;
                rhizome_resin_noise::Perlin3D::new().sample(
                    ex.eval(x, y, z, t, vars),
                    ey.eval(x, y, z, t, vars),
                    ez.eval(x, y, z, t, vars),
                )
            }
            Self::Simplex2 { x: ex, y: ey } => {
                use rhizome_resin_noise::Noise2D;
                rhizome_resin_noise::Simplex2D::new()
                    .sample(ex.eval(x, y, z, t, vars), ey.eval(x, y, z, t, vars))
            }
            Self::Simplex3 {
                x: ex,
                y: ey,
                z: ez,
            } => {
                use rhizome_resin_noise::Noise3D;
                rhizome_resin_noise::Simplex3D::new().sample(
                    ex.eval(x, y, z, t, vars),
                    ey.eval(x, y, z, t, vars),
                    ez.eval(x, y, z, t, vars),
                )
            }
            Self::Fbm2 {
                x: ex,
                y: ey,
                octaves,
            } => {
                use rhizome_resin_noise::Noise2D;
                rhizome_resin_noise::Fbm::new(rhizome_resin_noise::Perlin2D::new())
                    .octaves(*octaves)
                    .sample(ex.eval(x, y, z, t, vars), ey.eval(x, y, z, t, vars))
            }
            Self::Fbm3 {
                x: ex,
                y: ey,
                z: ez,
                octaves,
            } => {
                use rhizome_resin_noise::Noise3D;
                rhizome_resin_noise::Fbm::new(rhizome_resin_noise::Perlin3D::new())
                    .octaves(*octaves)
                    .sample(
                        ex.eval(x, y, z, t, vars),
                        ey.eval(x, y, z, t, vars),
                        ez.eval(x, y, z, t, vars),
                    )
            }

            // Distance functions
            Self::Distance2 {
                x: ex,
                y: ey,
                px,
                py,
            } => {
                let dx = ex.eval(x, y, z, t, vars) - px;
                let dy = ey.eval(x, y, z, t, vars) - py;
                (dx * dx + dy * dy).sqrt()
            }
            Self::Distance3 {
                x: ex,
                y: ey,
                z: ez,
                px,
                py,
                pz,
            } => {
                let dx = ex.eval(x, y, z, t, vars) - px;
                let dy = ey.eval(x, y, z, t, vars) - py;
                let dz = ez.eval(x, y, z, t, vars) - pz;
                (dx * dx + dy * dy + dz * dz).sqrt()
            }
            Self::Length2 { x: ex, y: ey } => {
                let vx = ex.eval(x, y, z, t, vars);
                let vy = ey.eval(x, y, z, t, vars);
                (vx * vx + vy * vy).sqrt()
            }
            Self::Length3 {
                x: ex,
                y: ey,
                z: ez,
            } => {
                let vx = ex.eval(x, y, z, t, vars);
                let vy = ey.eval(x, y, z, t, vars);
                let vz = ez.eval(x, y, z, t, vars);
                (vx * vx + vy * vy + vz * vz).sqrt()
            }

            // SDF operations
            Self::SdfCircle {
                x: ex,
                y: ey,
                radius,
            } => {
                let vx = ex.eval(x, y, z, t, vars);
                let vy = ey.eval(x, y, z, t, vars);
                (vx * vx + vy * vy).sqrt() - radius
            }
            Self::SdfSphere {
                x: ex,
                y: ey,
                z: ez,
                radius,
            } => {
                let vx = ex.eval(x, y, z, t, vars);
                let vy = ey.eval(x, y, z, t, vars);
                let vz = ez.eval(x, y, z, t, vars);
                (vx * vx + vy * vy + vz * vz).sqrt() - radius
            }
            Self::SdfBox2 {
                x: ex,
                y: ey,
                half_width,
                half_height,
            } => {
                let vx = ex.eval(x, y, z, t, vars).abs();
                let vy = ey.eval(x, y, z, t, vars).abs();
                let dx = vx - half_width;
                let dy = vy - half_height;
                let outside = (dx.max(0.0) * dx.max(0.0) + dy.max(0.0) * dy.max(0.0)).sqrt();
                let inside = dx.max(dy).min(0.0);
                outside + inside
            }
            Self::SdfBox3 {
                x: ex,
                y: ey,
                z: ez,
                half_x,
                half_y,
                half_z,
            } => {
                let vx = ex.eval(x, y, z, t, vars).abs();
                let vy = ey.eval(x, y, z, t, vars).abs();
                let vz = ez.eval(x, y, z, t, vars).abs();
                let dx = vx - half_x;
                let dy = vy - half_y;
                let dz = vz - half_z;
                let outside = (dx.max(0.0) * dx.max(0.0)
                    + dy.max(0.0) * dy.max(0.0)
                    + dz.max(0.0) * dz.max(0.0))
                .sqrt();
                let inside = dx.max(dy).max(dz).min(0.0);
                outside + inside
            }
            Self::SdfSmoothUnion { a, b, k } => {
                let d1 = a.eval(x, y, z, t, vars);
                let d2 = b.eval(x, y, z, t, vars);
                let h = (0.5 + 0.5 * (d2 - d1) / k).clamp(0.0, 1.0);
                d2 * (1.0 - h) + d1 * h - k * h * (1.0 - h)
            }
            Self::SdfSmoothIntersection { a, b, k } => {
                let d1 = a.eval(x, y, z, t, vars);
                let d2 = b.eval(x, y, z, t, vars);
                let h = (0.5 - 0.5 * (d2 - d1) / k).clamp(0.0, 1.0);
                d2 * (1.0 - h) + d1 * h + k * h * (1.0 - h)
            }
            Self::SdfSmoothSubtraction { a, b, k } => {
                let d1 = a.eval(x, y, z, t, vars);
                let d2 = b.eval(x, y, z, t, vars);
                let h = (0.5 - 0.5 * (d2 + d1) / k).clamp(0.0, 1.0);
                -d1 * (1.0 - h) + d2 * h + k * h * (1.0 - h)
            }

            // Math functions
            Self::Sin(a) => a.eval(x, y, z, t, vars).sin(),
            Self::Cos(a) => a.eval(x, y, z, t, vars).cos(),
            Self::Tan(a) => a.eval(x, y, z, t, vars).tan(),
            Self::Abs(a) => a.eval(x, y, z, t, vars).abs(),
            Self::Floor(a) => a.eval(x, y, z, t, vars).floor(),
            Self::Ceil(a) => a.eval(x, y, z, t, vars).ceil(),
            Self::Fract(a) => a.eval(x, y, z, t, vars).fract(),
            Self::Sqrt(a) => a.eval(x, y, z, t, vars).sqrt(),
            Self::Exp(a) => a.eval(x, y, z, t, vars).exp(),
            Self::Ln(a) => a.eval(x, y, z, t, vars).ln(),
            Self::Sign(a) => {
                let v = a.eval(x, y, z, t, vars);
                if v > 0.0 {
                    1.0
                } else if v < 0.0 {
                    -1.0
                } else {
                    0.0
                }
            }
            Self::Min(a, b) => a.eval(x, y, z, t, vars).min(b.eval(x, y, z, t, vars)),
            Self::Max(a, b) => a.eval(x, y, z, t, vars).max(b.eval(x, y, z, t, vars)),
            Self::Clamp { value, min, max } => value
                .eval(x, y, z, t, vars)
                .clamp(min.eval(x, y, z, t, vars), max.eval(x, y, z, t, vars)),
            Self::Lerp {
                a: ea,
                b: eb,
                t: et,
            } => {
                let va = ea.eval(x, y, z, t, vars);
                let vb = eb.eval(x, y, z, t, vars);
                let vt = et.eval(x, y, z, t, vars);
                va + (vb - va) * vt
            }
            Self::SmoothStep {
                edge0,
                edge1,
                x: ex,
            } => {
                let e0 = edge0.eval(x, y, z, t, vars);
                let e1 = edge1.eval(x, y, z, t, vars);
                let v = ex.eval(x, y, z, t, vars);
                let t = ((v - e0) / (e1 - e0)).clamp(0.0, 1.0);
                t * t * (3.0 - 2.0 * t)
            }
            Self::Step { edge, x: ex } => {
                let e = edge.eval(x, y, z, t, vars);
                let v = ex.eval(x, y, z, t, vars);
                if v < e { 0.0 } else { 1.0 }
            }

            // Conditionals
            Self::IfThenElse {
                condition,
                then_expr,
                else_expr,
            } => {
                if condition.eval(x, y, z, t, vars) > 0.5 {
                    then_expr.eval(x, y, z, t, vars)
                } else {
                    else_expr.eval(x, y, z, t, vars)
                }
            }
            Self::Gt(a, b) => {
                if a.eval(x, y, z, t, vars) > b.eval(x, y, z, t, vars) {
                    1.0
                } else {
                    0.0
                }
            }
            Self::Lt(a, b) => {
                if a.eval(x, y, z, t, vars) < b.eval(x, y, z, t, vars) {
                    1.0
                } else {
                    0.0
                }
            }
            Self::Eq(a, b) => {
                if (a.eval(x, y, z, t, vars) - b.eval(x, y, z, t, vars)).abs() < 1e-6 {
                    1.0
                } else {
                    0.0
                }
            }
        }
    }

    /// Returns the free variables in this expression (excluding x, y, z, t).
    pub fn free_vars(&self) -> HashSet<String> {
        let mut vars = HashSet::new();
        self.collect_vars(&mut vars);
        vars
    }

    fn collect_vars(&self, vars: &mut HashSet<String>) {
        match self {
            Self::Var(name) => {
                vars.insert(name.clone());
            }
            Self::X | Self::Y | Self::Z | Self::T | Self::Constant(_) => {}

            Self::Add(a, b)
            | Self::Sub(a, b)
            | Self::Mul(a, b)
            | Self::Div(a, b)
            | Self::Mod(a, b)
            | Self::Pow(a, b)
            | Self::Min(a, b)
            | Self::Max(a, b)
            | Self::Gt(a, b)
            | Self::Lt(a, b)
            | Self::Eq(a, b) => {
                a.collect_vars(vars);
                b.collect_vars(vars);
            }

            Self::Neg(a)
            | Self::Sin(a)
            | Self::Cos(a)
            | Self::Tan(a)
            | Self::Abs(a)
            | Self::Floor(a)
            | Self::Ceil(a)
            | Self::Fract(a)
            | Self::Sqrt(a)
            | Self::Exp(a)
            | Self::Ln(a)
            | Self::Sign(a) => {
                a.collect_vars(vars);
            }

            Self::Perlin2 { x, y } | Self::Simplex2 { x, y } | Self::Length2 { x, y } => {
                x.collect_vars(vars);
                y.collect_vars(vars);
            }

            Self::Fbm2 { x, y, .. }
            | Self::Distance2 { x, y, .. }
            | Self::SdfCircle { x, y, .. } => {
                x.collect_vars(vars);
                y.collect_vars(vars);
            }

            Self::SdfBox2 { x, y, .. } => {
                x.collect_vars(vars);
                y.collect_vars(vars);
            }

            Self::Perlin3 { x, y, z } | Self::Simplex3 { x, y, z } | Self::Length3 { x, y, z } => {
                x.collect_vars(vars);
                y.collect_vars(vars);
                z.collect_vars(vars);
            }

            Self::Fbm3 { x, y, z, .. }
            | Self::Distance3 { x, y, z, .. }
            | Self::SdfSphere { x, y, z, .. }
            | Self::SdfBox3 { x, y, z, .. } => {
                x.collect_vars(vars);
                y.collect_vars(vars);
                z.collect_vars(vars);
            }

            Self::SdfSmoothUnion { a, b, .. }
            | Self::SdfSmoothIntersection { a, b, .. }
            | Self::SdfSmoothSubtraction { a, b, .. } => {
                a.collect_vars(vars);
                b.collect_vars(vars);
            }

            Self::Clamp { value, min, max } => {
                value.collect_vars(vars);
                min.collect_vars(vars);
                max.collect_vars(vars);
            }

            Self::Lerp { a, b, t } => {
                a.collect_vars(vars);
                b.collect_vars(vars);
                t.collect_vars(vars);
            }

            Self::SmoothStep { edge0, edge1, x } => {
                edge0.collect_vars(vars);
                edge1.collect_vars(vars);
                x.collect_vars(vars);
            }

            Self::Step { edge, x } => {
                edge.collect_vars(vars);
                x.collect_vars(vars);
            }

            Self::IfThenElse {
                condition,
                then_expr,
                else_expr,
            } => {
                condition.collect_vars(vars);
                then_expr.collect_vars(vars);
                else_expr.collect_vars(vars);
            }
        }
    }

    /// Convert to dew AST for compilation to WGSL/Cranelift.
    pub fn to_dew_ast(&self) -> Ast {
        match self {
            // Coordinates
            Self::X => Ast::Var("x".into()),
            Self::Y => Ast::Var("y".into()),
            Self::Z => Ast::Var("z".into()),
            Self::T => Ast::Var("t".into()),

            // Literals
            Self::Constant(v) => Ast::Num(*v as f64),
            Self::Var(name) => Ast::Var(name.clone()),

            // Binary ops
            Self::Add(a, b) => Ast::BinOp(
                BinOp::Add,
                Box::new(a.to_dew_ast()),
                Box::new(b.to_dew_ast()),
            ),
            Self::Sub(a, b) => Ast::BinOp(
                BinOp::Sub,
                Box::new(a.to_dew_ast()),
                Box::new(b.to_dew_ast()),
            ),
            Self::Mul(a, b) => Ast::BinOp(
                BinOp::Mul,
                Box::new(a.to_dew_ast()),
                Box::new(b.to_dew_ast()),
            ),
            Self::Div(a, b) => Ast::BinOp(
                BinOp::Div,
                Box::new(a.to_dew_ast()),
                Box::new(b.to_dew_ast()),
            ),
            Self::Mod(a, b) => Ast::Call("mod".into(), vec![a.to_dew_ast(), b.to_dew_ast()]),
            Self::Pow(a, b) => Ast::BinOp(
                BinOp::Pow,
                Box::new(a.to_dew_ast()),
                Box::new(b.to_dew_ast()),
            ),

            Self::Neg(a) => Ast::UnaryOp(UnaryOp::Neg, Box::new(a.to_dew_ast())),

            // Noise functions
            Self::Perlin2 { x, y } => {
                Ast::Call("perlin".into(), vec![x.to_dew_ast(), y.to_dew_ast()])
            }
            Self::Perlin3 { x, y, z } => Ast::Call(
                "perlin3".into(),
                vec![x.to_dew_ast(), y.to_dew_ast(), z.to_dew_ast()],
            ),
            Self::Simplex2 { x, y } => {
                Ast::Call("simplex".into(), vec![x.to_dew_ast(), y.to_dew_ast()])
            }
            Self::Simplex3 { x, y, z } => Ast::Call(
                "simplex3".into(),
                vec![x.to_dew_ast(), y.to_dew_ast(), z.to_dew_ast()],
            ),
            Self::Fbm2 { x, y, octaves } => Ast::Call(
                "fbm".into(),
                vec![x.to_dew_ast(), y.to_dew_ast(), Ast::Num(*octaves as f64)],
            ),
            Self::Fbm3 { x, y, z, octaves } => Ast::Call(
                "fbm3".into(),
                vec![
                    x.to_dew_ast(),
                    y.to_dew_ast(),
                    z.to_dew_ast(),
                    Ast::Num(*octaves as f64),
                ],
            ),

            // Distance functions
            Self::Distance2 { x, y, px, py } => Ast::Call(
                "distance2".into(),
                vec![
                    x.to_dew_ast(),
                    y.to_dew_ast(),
                    Ast::Num(*px as f64),
                    Ast::Num(*py as f64),
                ],
            ),
            Self::Distance3 {
                x,
                y,
                z,
                px,
                py,
                pz,
            } => Ast::Call(
                "distance3".into(),
                vec![
                    x.to_dew_ast(),
                    y.to_dew_ast(),
                    z.to_dew_ast(),
                    Ast::Num(*px as f64),
                    Ast::Num(*py as f64),
                    Ast::Num(*pz as f64),
                ],
            ),
            Self::Length2 { x, y } => {
                Ast::Call("length2".into(), vec![x.to_dew_ast(), y.to_dew_ast()])
            }
            Self::Length3 { x, y, z } => Ast::Call(
                "length3".into(),
                vec![x.to_dew_ast(), y.to_dew_ast(), z.to_dew_ast()],
            ),

            // SDF operations
            Self::SdfCircle { x, y, radius } => Ast::Call(
                "sdf_circle".into(),
                vec![x.to_dew_ast(), y.to_dew_ast(), Ast::Num(*radius as f64)],
            ),
            Self::SdfSphere { x, y, z, radius } => Ast::Call(
                "sdf_sphere".into(),
                vec![
                    x.to_dew_ast(),
                    y.to_dew_ast(),
                    z.to_dew_ast(),
                    Ast::Num(*radius as f64),
                ],
            ),
            Self::SdfBox2 {
                x,
                y,
                half_width,
                half_height,
            } => Ast::Call(
                "sdf_box2".into(),
                vec![
                    x.to_dew_ast(),
                    y.to_dew_ast(),
                    Ast::Num(*half_width as f64),
                    Ast::Num(*half_height as f64),
                ],
            ),
            Self::SdfBox3 {
                x,
                y,
                z,
                half_x,
                half_y,
                half_z,
            } => Ast::Call(
                "sdf_box3".into(),
                vec![
                    x.to_dew_ast(),
                    y.to_dew_ast(),
                    z.to_dew_ast(),
                    Ast::Num(*half_x as f64),
                    Ast::Num(*half_y as f64),
                    Ast::Num(*half_z as f64),
                ],
            ),
            Self::SdfSmoothUnion { a, b, k } => Ast::Call(
                "sdf_smooth_union".into(),
                vec![a.to_dew_ast(), b.to_dew_ast(), Ast::Num(*k as f64)],
            ),
            Self::SdfSmoothIntersection { a, b, k } => Ast::Call(
                "sdf_smooth_intersection".into(),
                vec![a.to_dew_ast(), b.to_dew_ast(), Ast::Num(*k as f64)],
            ),
            Self::SdfSmoothSubtraction { a, b, k } => Ast::Call(
                "sdf_smooth_subtraction".into(),
                vec![a.to_dew_ast(), b.to_dew_ast(), Ast::Num(*k as f64)],
            ),

            // Math functions
            Self::Sin(a) => Ast::Call("sin".into(), vec![a.to_dew_ast()]),
            Self::Cos(a) => Ast::Call("cos".into(), vec![a.to_dew_ast()]),
            Self::Tan(a) => Ast::Call("tan".into(), vec![a.to_dew_ast()]),
            Self::Abs(a) => Ast::Call("abs".into(), vec![a.to_dew_ast()]),
            Self::Floor(a) => Ast::Call("floor".into(), vec![a.to_dew_ast()]),
            Self::Ceil(a) => Ast::Call("ceil".into(), vec![a.to_dew_ast()]),
            Self::Fract(a) => Ast::Call("fract".into(), vec![a.to_dew_ast()]),
            Self::Sqrt(a) => Ast::Call("sqrt".into(), vec![a.to_dew_ast()]),
            Self::Exp(a) => Ast::Call("exp".into(), vec![a.to_dew_ast()]),
            Self::Ln(a) => Ast::Call("ln".into(), vec![a.to_dew_ast()]),
            Self::Sign(a) => Ast::Call("sign".into(), vec![a.to_dew_ast()]),
            Self::Min(a, b) => Ast::Call("min".into(), vec![a.to_dew_ast(), b.to_dew_ast()]),
            Self::Max(a, b) => Ast::Call("max".into(), vec![a.to_dew_ast(), b.to_dew_ast()]),
            Self::Clamp { value, min, max } => Ast::Call(
                "clamp".into(),
                vec![value.to_dew_ast(), min.to_dew_ast(), max.to_dew_ast()],
            ),
            Self::Lerp { a, b, t } => Ast::Call(
                "lerp".into(),
                vec![a.to_dew_ast(), b.to_dew_ast(), t.to_dew_ast()],
            ),
            Self::SmoothStep { edge0, edge1, x } => Ast::Call(
                "smoothstep".into(),
                vec![edge0.to_dew_ast(), edge1.to_dew_ast(), x.to_dew_ast()],
            ),
            Self::Step { edge, x } => {
                Ast::Call("step".into(), vec![edge.to_dew_ast(), x.to_dew_ast()])
            }

            // Conditionals
            Self::IfThenElse {
                condition,
                then_expr,
                else_expr,
            } => Ast::If(
                Box::new(condition.to_dew_ast()),
                Box::new(then_expr.to_dew_ast()),
                Box::new(else_expr.to_dew_ast()),
            ),
            Self::Gt(a, b) => Ast::Compare(
                rhizome_dew_core::CompareOp::Gt,
                Box::new(a.to_dew_ast()),
                Box::new(b.to_dew_ast()),
            ),
            Self::Lt(a, b) => Ast::Compare(
                rhizome_dew_core::CompareOp::Lt,
                Box::new(a.to_dew_ast()),
                Box::new(b.to_dew_ast()),
            ),
            Self::Eq(a, b) => Ast::Compare(
                rhizome_dew_core::CompareOp::Eq,
                Box::new(a.to_dew_ast()),
                Box::new(b.to_dew_ast()),
            ),
        }
    }
}

impl Field<Vec2, f32> for FieldExpr {
    fn sample(&self, input: Vec2, ctx: &EvalContext) -> f32 {
        self.eval(input.x, input.y, 0.0, ctx.time, &HashMap::new())
    }
}

impl Field<Vec3, f32> for FieldExpr {
    fn sample(&self, input: Vec3, ctx: &EvalContext) -> f32 {
        self.eval(input.x, input.y, input.z, ctx.time, &HashMap::new())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_expr_field_2d() {
        let registry = FunctionRegistry::<f32>::new();
        let field = ExprField::parse("x + y", registry).unwrap();
        let ctx = EvalContext::new();
        let v: f32 = field.sample(Vec2::new(3.0, 4.0), &ctx);
        assert_eq!(v, 7.0);
    }

    #[test]
    fn test_expr_field_3d() {
        let registry = FunctionRegistry::<f32>::new();
        let field = ExprField::parse("x + y + z", registry).unwrap();
        let ctx = EvalContext::new();
        let v: f32 = field.sample(Vec3::new(1.0, 2.0, 3.0), &ctx);
        assert_eq!(v, 6.0);
    }

    #[test]
    fn test_expr_field_time() {
        let registry = FunctionRegistry::<f32>::new();
        let field = ExprField::parse("time", registry).unwrap();
        let ctx = EvalContext::new().with_time(5.0);
        let v: f32 = field.sample(Vec2::ZERO, &ctx);
        assert_eq!(v, 5.0);
    }

    #[test]
    fn test_noise_functions() {
        let mut registry = FunctionRegistry::<f32>::new();
        register_noise(&mut registry);

        let expr = Expr::parse("noise(0.5, 0.5)").unwrap();
        let vars = HashMap::new();
        let v = rhizome_dew_scalar::eval(expr.ast(), &vars, &registry).unwrap();
        assert!((0.0..=1.0).contains(&v));

        let expr = Expr::parse("perlin(0.5, 0.5)").unwrap();
        let v = rhizome_dew_scalar::eval(expr.ast(), &vars, &registry).unwrap();
        assert!((0.0..=1.0).contains(&v));

        let expr = Expr::parse("simplex(0.5, 0.5)").unwrap();
        let v = rhizome_dew_scalar::eval(expr.ast(), &vars, &registry).unwrap();
        assert!((0.0..=1.0).contains(&v));
    }

    #[test]
    fn test_free_vars() {
        let registry = FunctionRegistry::<f32>::new();
        let field = ExprField::parse("sin(t * speed) * amplitude + x", registry).unwrap();

        let free = field.free_vars();
        assert!(free.contains("t"));
        assert!(free.contains("speed"));
        assert!(free.contains("amplitude"));
        assert!(free.contains("x"));
    }

    #[test]
    fn test_user_inputs() {
        let registry = FunctionRegistry::<f32>::new();
        let field = ExprField::parse("sin(t * speed) * amplitude + x", registry).unwrap();

        let inputs = field.user_inputs();
        // Builtins (t, x) should be filtered out
        assert!(!inputs.contains("t"));
        assert!(!inputs.contains("x"));
        // User inputs remain
        assert!(inputs.contains("speed"));
        assert!(inputs.contains("amplitude"));
        assert_eq!(inputs.len(), 2);
    }

    #[test]
    fn test_eval_with_bindings() {
        let registry = FunctionRegistry::<f32>::new();
        let field = ExprField::parse("a + b", registry).unwrap();

        let mut vars = HashMap::new();
        vars.insert("a".to_string(), 3.0);
        vars.insert("b".to_string(), 4.0);

        let result = field.eval(&vars).unwrap();
        assert_eq!(result, 7.0);
    }

    // FieldExpr tests

    #[test]
    fn test_field_expr_coordinates() {
        let expr_x = FieldExpr::X;
        assert_eq!(expr_x.eval(1.0, 2.0, 3.0, 0.0, &Default::default()), 1.0);

        let expr_y = FieldExpr::Y;
        assert_eq!(expr_y.eval(1.0, 2.0, 3.0, 0.0, &Default::default()), 2.0);

        let expr_z = FieldExpr::Z;
        assert_eq!(expr_z.eval(1.0, 2.0, 3.0, 0.0, &Default::default()), 3.0);
    }

    #[test]
    fn test_field_expr_arithmetic() {
        // x + y * 2
        let expr = FieldExpr::Add(
            Box::new(FieldExpr::X),
            Box::new(FieldExpr::Mul(
                Box::new(FieldExpr::Y),
                Box::new(FieldExpr::Constant(2.0)),
            )),
        );
        // 3 + 4 * 2 = 11
        assert_eq!(expr.eval(3.0, 4.0, 0.0, 0.0, &Default::default()), 11.0);
    }

    #[test]
    fn test_field_expr_perlin() {
        let expr = FieldExpr::Perlin2 {
            x: Box::new(FieldExpr::X),
            y: Box::new(FieldExpr::Y),
        };
        let v = expr.eval(0.5, 0.5, 0.0, 0.0, &Default::default());
        // Perlin returns 0..1
        assert!((0.0..=1.0).contains(&v), "Perlin out of range: {}", v);
    }

    #[test]
    fn test_field_expr_sdf_circle() {
        let expr = FieldExpr::SdfCircle {
            x: Box::new(FieldExpr::X),
            y: Box::new(FieldExpr::Y),
            radius: 1.0,
        };
        // At origin, distance = -1 (inside)
        assert!((expr.eval(0.0, 0.0, 0.0, 0.0, &Default::default()) - (-1.0)).abs() < 0.01);
        // At (1, 0), distance = 0 (on surface)
        assert!((expr.eval(1.0, 0.0, 0.0, 0.0, &Default::default()) - 0.0).abs() < 0.01);
        // At (2, 0), distance = 1 (outside)
        assert!((expr.eval(2.0, 0.0, 0.0, 0.0, &Default::default()) - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_field_expr_smooth_union() {
        let circle1 = FieldExpr::SdfCircle {
            x: Box::new(FieldExpr::Sub(
                Box::new(FieldExpr::X),
                Box::new(FieldExpr::Constant(-0.5)),
            )),
            y: Box::new(FieldExpr::Y),
            radius: 0.5,
        };
        let circle2 = FieldExpr::SdfCircle {
            x: Box::new(FieldExpr::Sub(
                Box::new(FieldExpr::X),
                Box::new(FieldExpr::Constant(0.5)),
            )),
            y: Box::new(FieldExpr::Y),
            radius: 0.5,
        };
        let expr = FieldExpr::SdfSmoothUnion {
            a: Box::new(circle1),
            b: Box::new(circle2),
            k: 0.2,
        };
        // Should produce smooth blended result
        let v = expr.eval(0.0, 0.0, 0.0, 0.0, &Default::default());
        assert!(v < 0.0, "Point should be inside smooth union: {}", v);
    }

    #[test]
    fn test_field_expr_free_vars() {
        let expr = FieldExpr::Add(
            Box::new(FieldExpr::Var("scale".into())),
            Box::new(FieldExpr::Mul(
                Box::new(FieldExpr::X),
                Box::new(FieldExpr::Var("freq".into())),
            )),
        );
        let vars = expr.free_vars();
        assert!(vars.contains("scale"));
        assert!(vars.contains("freq"));
        assert_eq!(vars.len(), 2);
    }

    #[test]
    fn test_field_expr_as_field() {
        let expr = FieldExpr::Add(Box::new(FieldExpr::X), Box::new(FieldExpr::Y));
        let ctx = EvalContext::new();

        // Test Vec2
        let v: f32 = Field::<Vec2, f32>::sample(&expr, Vec2::new(3.0, 4.0), &ctx);
        assert_eq!(v, 7.0);

        // Test Vec3
        let v: f32 = Field::<Vec3, f32>::sample(&expr, Vec3::new(1.0, 2.0, 3.0), &ctx);
        assert_eq!(v, 3.0);
    }

    #[test]
    fn test_field_expr_to_dew_ast() {
        let expr = FieldExpr::Add(Box::new(FieldExpr::X), Box::new(FieldExpr::Constant(1.0)));
        let ast = expr.to_dew_ast();
        assert!(matches!(ast, Ast::BinOp(..)));
    }

    #[test]
    fn test_field_expr_smoothstep() {
        let expr = FieldExpr::SmoothStep {
            edge0: Box::new(FieldExpr::Constant(0.0)),
            edge1: Box::new(FieldExpr::Constant(1.0)),
            x: Box::new(FieldExpr::X),
        };
        // At x=0, smoothstep = 0
        assert!((expr.eval(0.0, 0.0, 0.0, 0.0, &Default::default()) - 0.0).abs() < 0.01);
        // At x=1, smoothstep = 1
        assert!((expr.eval(1.0, 0.0, 0.0, 0.0, &Default::default()) - 1.0).abs() < 0.01);
        // At x=0.5, smoothstep = 0.5
        assert!((expr.eval(0.5, 0.0, 0.0, 0.0, &Default::default()) - 0.5).abs() < 0.01);
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_field_expr_serde_roundtrip() {
        let expr = FieldExpr::Perlin2 {
            x: Box::new(FieldExpr::Mul(
                Box::new(FieldExpr::X),
                Box::new(FieldExpr::Var("freq".into())),
            )),
            y: Box::new(FieldExpr::Y),
        };
        let json = serde_json::to_string(&expr).unwrap();
        let parsed: FieldExpr = serde_json::from_str(&json).unwrap();
        assert_eq!(expr, parsed);
    }
}
