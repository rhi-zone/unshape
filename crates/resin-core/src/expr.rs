//! Expression language integration with Fields.
//!
//! Re-exports the core expression types from `resin_expr` and provides
//! `ExprField` which bridges expressions to the `Field` trait.
//!
//! # Example
//!
//! ```ignore
//! use resin_core::expr::{ExprField, Expr, FunctionRegistry};
//! use resin_core::field::Field;
//! use resin_core::EvalContext;
//! use resin_expr_std::std_registry;
//! use glam::Vec2;
//!
//! let registry = std_registry();
//! let field = ExprField::parse("sin(x * pi()) + y", registry).unwrap();
//! let ctx = EvalContext::new();
//! let value: f32 = field.sample(Vec2::new(0.5, 0.0), &ctx);
//! ```

use crate::context::EvalContext;
use crate::field::Field;
use glam::{Vec2, Vec3};
use resin_noise;
use std::collections::HashMap;

// Re-export core expression types
pub use resin_expr::{Ast, BinOp, EvalError, Expr, ExprFn, FunctionRegistry, ParseError, UnaryOp};

// ============================================================================
// Resin-specific expression functions (noise, etc.)
// ============================================================================

/// 2D Perlin noise: noise(x, y)
pub struct Noise;
impl ExprFn for Noise {
    fn name(&self) -> &str {
        "noise"
    }
    fn arg_count(&self) -> usize {
        2
    }
    fn call(&self, args: &[f32]) -> f32 {
        let [x, y] = args else { return 0.0 };
        resin_noise::perlin2(*x, *y)
    }
}

/// 2D Perlin noise: perlin(x, y)
pub struct Perlin;
impl ExprFn for Perlin {
    fn name(&self) -> &str {
        "perlin"
    }
    fn arg_count(&self) -> usize {
        2
    }
    fn call(&self, args: &[f32]) -> f32 {
        let [x, y] = args else { return 0.0 };
        resin_noise::perlin2(*x, *y)
    }
}

/// 3D Perlin noise: perlin3(x, y, z)
pub struct Perlin3;
impl ExprFn for Perlin3 {
    fn name(&self) -> &str {
        "perlin3"
    }
    fn arg_count(&self) -> usize {
        3
    }
    fn call(&self, args: &[f32]) -> f32 {
        let [x, y, z] = args else { return 0.0 };
        resin_noise::perlin3(*x, *y, *z)
    }
}

/// 2D Simplex noise: simplex(x, y)
pub struct Simplex;
impl ExprFn for Simplex {
    fn name(&self) -> &str {
        "simplex"
    }
    fn arg_count(&self) -> usize {
        2
    }
    fn call(&self, args: &[f32]) -> f32 {
        let [x, y] = args else { return 0.0 };
        resin_noise::simplex2(*x, *y)
    }
}

/// 3D Simplex noise: simplex3(x, y, z)
pub struct Simplex3;
impl ExprFn for Simplex3 {
    fn name(&self) -> &str {
        "simplex3"
    }
    fn arg_count(&self) -> usize {
        3
    }
    fn call(&self, args: &[f32]) -> f32 {
        let [x, y, z] = args else { return 0.0 };
        resin_noise::simplex3(*x, *y, *z)
    }
}

/// 2D FBM noise: fbm(x, y, octaves)
pub struct Fbm;
impl ExprFn for Fbm {
    fn name(&self) -> &str {
        "fbm"
    }
    fn arg_count(&self) -> usize {
        3
    }
    fn call(&self, args: &[f32]) -> f32 {
        let [x, y, octaves] = args else { return 0.0 };
        resin_noise::fbm_perlin2(*x, *y, *octaves as u32)
    }
}

/// Registers resin-specific expression functions (noise, etc.)
pub fn register_resin(registry: &mut FunctionRegistry) {
    registry.register(Noise);
    registry.register(Perlin);
    registry.register(Perlin3);
    registry.register(Simplex);
    registry.register(Simplex3);
    registry.register(Fbm);
}

/// An expression bundled with its function registry for use as a Field.
///
/// ExprField bridges the expression language to the Field system by:
/// - Storing an expression and its function registry
/// - Mapping input positions to variable bindings (x, y, z)
/// - Mapping EvalContext to the `time` variable
pub struct ExprField {
    expr: Expr,
    registry: FunctionRegistry,
}

impl ExprField {
    /// Creates a new ExprField with the given registry.
    pub fn new(expr: Expr, registry: FunctionRegistry) -> Self {
        Self { expr, registry }
    }

    /// Parses an expression and creates an ExprField with the given registry.
    pub fn parse(input: &str, registry: FunctionRegistry) -> Result<Self, ParseError> {
        Ok(Self::new(Expr::parse(input)?, registry))
    }

    /// Evaluates with explicit variable bindings.
    pub fn eval(&self, vars: &HashMap<String, f32>) -> Result<f32, EvalError> {
        self.expr.eval(vars, &self.registry)
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
        self.expr.eval(&vars, &self.registry).unwrap_or(0.0)
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
        self.expr.eval(&vars, &self.registry).unwrap_or(0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_expr_field_2d() {
        let registry = FunctionRegistry::new();
        let field = ExprField::parse("x + y", registry).unwrap();
        let ctx = EvalContext::new();
        let v: f32 = field.sample(Vec2::new(3.0, 4.0), &ctx);
        assert_eq!(v, 7.0);
    }

    #[test]
    fn test_expr_field_3d() {
        let registry = FunctionRegistry::new();
        let field = ExprField::parse("x + y + z", registry).unwrap();
        let ctx = EvalContext::new();
        let v: f32 = field.sample(Vec3::new(1.0, 2.0, 3.0), &ctx);
        assert_eq!(v, 6.0);
    }

    #[test]
    fn test_expr_field_time() {
        let registry = FunctionRegistry::new();
        let field = ExprField::parse("time", registry).unwrap();
        let ctx = EvalContext::new().with_time(5.0);
        let v: f32 = field.sample(Vec2::ZERO, &ctx);
        assert_eq!(v, 5.0);
    }

    #[test]
    fn test_noise_functions() {
        let mut registry = FunctionRegistry::new();
        register_resin(&mut registry);

        let expr = Expr::parse("noise(0.5, 0.5)").unwrap();
        let vars = HashMap::new();
        let v = expr.eval(&vars, &registry).unwrap();
        assert!((0.0..=1.0).contains(&v));

        let expr = Expr::parse("perlin(0.5, 0.5)").unwrap();
        let v = expr.eval(&vars, &registry).unwrap();
        assert!((0.0..=1.0).contains(&v));

        let expr = Expr::parse("simplex(0.5, 0.5)").unwrap();
        let v = expr.eval(&vars, &registry).unwrap();
        assert!((0.0..=1.0).contains(&v));
    }
}
