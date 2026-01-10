//! Bridge between expressions and fields.
//!
//! Provides `ExprField` which evaluates expressions as spatial fields,
//! and noise expression functions for use in expressions.
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

use glam::{Vec2, Vec3};
use rhizome_dew_core::{Expr, ParseError};
use rhizome_dew_scalar::{FunctionRegistry, ScalarFn};
use rhizome_resin_field::{EvalContext, Field};
use std::collections::HashMap;

use std::collections::HashSet;

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
        let [x, y] = args else { return 0.0 };
        rhizome_resin_noise::perlin2(*x, *y)
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
        let [x, y] = args else { return 0.0 };
        rhizome_resin_noise::perlin2(*x, *y)
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
        let [x, y, z] = args else { return 0.0 };
        rhizome_resin_noise::perlin3(*x, *y, *z)
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
        let [x, y] = args else { return 0.0 };
        rhizome_resin_noise::simplex2(*x, *y)
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
        let [x, y, z] = args else { return 0.0 };
        rhizome_resin_noise::simplex3(*x, *y, *z)
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
        let [x, y, octaves] = args else { return 0.0 };
        rhizome_resin_noise::fbm_perlin2(*x, *y, *octaves as u32)
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
}
