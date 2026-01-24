//! Easing expressions as AST builders.
//!
//! These functions return [`FieldExpr`] AST nodes representing easing curves.
//! Unlike runtime easing functions, these expressions can be:
//! - Optimized by constant folding when input is known
//! - JIT-compiled via Cranelift
//! - Serialized to JSON for project files
//!
//! # Example
//!
//! ```
//! use unshape_expr_field::{FieldExpr, easing};
//!
//! // Build a quad_in easing: t * t
//! let t = FieldExpr::T;
//! let eased = easing::quad_in(t);
//!
//! // Evaluate at time 0.5
//! let value = eased.eval(0.0, 0.0, 0.0, 0.5, &Default::default());
//! assert!((value - 0.25).abs() < 0.001); // 0.5 * 0.5 = 0.25
//! ```
//!
//! # Constant Folding
//!
//! When the input is a constant, the optimizer can fold the expression:
//!
//! ```
//! use unshape_expr_field::{FieldExpr, easing};
//!
//! // quad_in(0.5) = Mul(0.5, 0.5) -> can be folded to 0.25
//! let eased = easing::quad_in(FieldExpr::Constant(0.5));
//! ```

use crate::FieldExpr;
use std::f32::consts::PI;

/// Helper to box an expression.
fn b(expr: FieldExpr) -> Box<FieldExpr> {
    Box::new(expr)
}

/// Constant value.
fn c(v: f32) -> FieldExpr {
    FieldExpr::Constant(v)
}

// ============================================================================
// Linear
// ============================================================================

/// Linear interpolation (identity): `t`
#[inline]
pub fn linear(t: FieldExpr) -> FieldExpr {
    t
}

// ============================================================================
// Quadratic
// ============================================================================

/// Quadratic ease in: `t * t`
#[inline]
pub fn quad_in(t: FieldExpr) -> FieldExpr {
    FieldExpr::Mul(b(t.clone()), b(t))
}

/// Quadratic ease out: `1 - (1 - t)^2`
#[inline]
pub fn quad_out(t: FieldExpr) -> FieldExpr {
    // 1 - (1 - t) * (1 - t)
    let one_minus_t = FieldExpr::Sub(b(c(1.0)), b(t));
    FieldExpr::Sub(
        b(c(1.0)),
        b(FieldExpr::Mul(b(one_minus_t.clone()), b(one_minus_t))),
    )
}

/// Quadratic ease in-out.
///
/// Returns: `if t < 0.5 then 2*t*t else 1 - ((-2*t + 2)^2) / 2`
#[inline]
pub fn quad_in_out(t: FieldExpr) -> FieldExpr {
    // First half: 2 * t * t
    let two_t = FieldExpr::Mul(b(c(2.0)), b(t.clone()));
    let first_half = FieldExpr::Mul(b(two_t.clone()), b(t.clone()));

    // Second half: 1 - ((-2*t + 2)^2) / 2
    let neg_two_t_plus_two = FieldExpr::Add(b(FieldExpr::Mul(b(c(-2.0)), b(t.clone()))), b(c(2.0)));
    let squared = FieldExpr::Mul(b(neg_two_t_plus_two.clone()), b(neg_two_t_plus_two));
    let second_half = FieldExpr::Sub(b(c(1.0)), b(FieldExpr::Div(b(squared), b(c(2.0)))));

    // Condition: t < 0.5
    let condition = FieldExpr::Lt(b(t), b(c(0.5)));

    FieldExpr::IfThenElse {
        condition: b(condition),
        then_expr: b(first_half),
        else_expr: b(second_half),
    }
}

// ============================================================================
// Cubic
// ============================================================================

/// Cubic ease in: `t * t * t`
#[inline]
pub fn cubic_in(t: FieldExpr) -> FieldExpr {
    FieldExpr::Mul(b(FieldExpr::Mul(b(t.clone()), b(t.clone()))), b(t))
}

/// Cubic ease out: `1 - (1 - t)^3`
#[inline]
pub fn cubic_out(t: FieldExpr) -> FieldExpr {
    let one_minus_t = FieldExpr::Sub(b(c(1.0)), b(t));
    let cubed = FieldExpr::Mul(
        b(FieldExpr::Mul(
            b(one_minus_t.clone()),
            b(one_minus_t.clone()),
        )),
        b(one_minus_t),
    );
    FieldExpr::Sub(b(c(1.0)), b(cubed))
}

/// Cubic ease in-out.
#[inline]
pub fn cubic_in_out(t: FieldExpr) -> FieldExpr {
    // First half: 4 * t * t * t
    let t_cubed = FieldExpr::Mul(b(FieldExpr::Mul(b(t.clone()), b(t.clone()))), b(t.clone()));
    let first_half = FieldExpr::Mul(b(c(4.0)), b(t_cubed));

    // Second half: 1 - ((-2*t + 2)^3) / 2
    let neg_two_t_plus_two = FieldExpr::Add(b(FieldExpr::Mul(b(c(-2.0)), b(t.clone()))), b(c(2.0)));
    let cubed = FieldExpr::Mul(
        b(FieldExpr::Mul(
            b(neg_two_t_plus_two.clone()),
            b(neg_two_t_plus_two.clone()),
        )),
        b(neg_two_t_plus_two),
    );
    let second_half = FieldExpr::Sub(b(c(1.0)), b(FieldExpr::Div(b(cubed), b(c(2.0)))));

    let condition = FieldExpr::Lt(b(t), b(c(0.5)));

    FieldExpr::IfThenElse {
        condition: b(condition),
        then_expr: b(first_half),
        else_expr: b(second_half),
    }
}

// ============================================================================
// Quartic
// ============================================================================

/// Quartic ease in: `t^4`
#[inline]
pub fn quart_in(t: FieldExpr) -> FieldExpr {
    let t2 = FieldExpr::Mul(b(t.clone()), b(t));
    FieldExpr::Mul(b(t2.clone()), b(t2))
}

/// Quartic ease out: `1 - (1 - t)^4`
#[inline]
pub fn quart_out(t: FieldExpr) -> FieldExpr {
    let one_minus_t = FieldExpr::Sub(b(c(1.0)), b(t));
    let t2 = FieldExpr::Mul(b(one_minus_t.clone()), b(one_minus_t));
    let t4 = FieldExpr::Mul(b(t2.clone()), b(t2));
    FieldExpr::Sub(b(c(1.0)), b(t4))
}

/// Quartic ease in-out.
#[inline]
pub fn quart_in_out(t: FieldExpr) -> FieldExpr {
    // First half: 8 * t^4
    let t2 = FieldExpr::Mul(b(t.clone()), b(t.clone()));
    let t4 = FieldExpr::Mul(b(t2.clone()), b(t2));
    let first_half = FieldExpr::Mul(b(c(8.0)), b(t4));

    // Second half: 1 - ((-2*t + 2)^4) / 2
    let neg_two_t_plus_two = FieldExpr::Add(b(FieldExpr::Mul(b(c(-2.0)), b(t.clone()))), b(c(2.0)));
    let p2 = FieldExpr::Mul(b(neg_two_t_plus_two.clone()), b(neg_two_t_plus_two));
    let p4 = FieldExpr::Mul(b(p2.clone()), b(p2));
    let second_half = FieldExpr::Sub(b(c(1.0)), b(FieldExpr::Div(b(p4), b(c(2.0)))));

    let condition = FieldExpr::Lt(b(t), b(c(0.5)));

    FieldExpr::IfThenElse {
        condition: b(condition),
        then_expr: b(first_half),
        else_expr: b(second_half),
    }
}

// ============================================================================
// Quintic
// ============================================================================

/// Quintic ease in: `t^5`
#[inline]
pub fn quint_in(t: FieldExpr) -> FieldExpr {
    let t2 = FieldExpr::Mul(b(t.clone()), b(t.clone()));
    let t4 = FieldExpr::Mul(b(t2.clone()), b(t2));
    FieldExpr::Mul(b(t4), b(t))
}

/// Quintic ease out: `1 - (1 - t)^5`
#[inline]
pub fn quint_out(t: FieldExpr) -> FieldExpr {
    let one_minus_t = FieldExpr::Sub(b(c(1.0)), b(t));
    let t2 = FieldExpr::Mul(b(one_minus_t.clone()), b(one_minus_t.clone()));
    let t4 = FieldExpr::Mul(b(t2.clone()), b(t2));
    let t5 = FieldExpr::Mul(b(t4), b(one_minus_t));
    FieldExpr::Sub(b(c(1.0)), b(t5))
}

/// Quintic ease in-out.
#[inline]
pub fn quint_in_out(t: FieldExpr) -> FieldExpr {
    // First half: 16 * t^5
    let t2 = FieldExpr::Mul(b(t.clone()), b(t.clone()));
    let t4 = FieldExpr::Mul(b(t2.clone()), b(t2));
    let t5 = FieldExpr::Mul(b(t4), b(t.clone()));
    let first_half = FieldExpr::Mul(b(c(16.0)), b(t5));

    // Second half: 1 - ((-2*t + 2)^5) / 2
    let neg_two_t_plus_two = FieldExpr::Add(b(FieldExpr::Mul(b(c(-2.0)), b(t.clone()))), b(c(2.0)));
    let p2 = FieldExpr::Mul(b(neg_two_t_plus_two.clone()), b(neg_two_t_plus_two.clone()));
    let p4 = FieldExpr::Mul(b(p2.clone()), b(p2));
    let p5 = FieldExpr::Mul(b(p4), b(neg_two_t_plus_two));
    let second_half = FieldExpr::Sub(b(c(1.0)), b(FieldExpr::Div(b(p5), b(c(2.0)))));

    let condition = FieldExpr::Lt(b(t), b(c(0.5)));

    FieldExpr::IfThenElse {
        condition: b(condition),
        then_expr: b(first_half),
        else_expr: b(second_half),
    }
}

// ============================================================================
// Sine
// ============================================================================

/// Sine ease in: `1 - cos(t * PI / 2)`
#[inline]
pub fn sine_in(t: FieldExpr) -> FieldExpr {
    let angle = FieldExpr::Mul(b(t), b(c(PI / 2.0)));
    FieldExpr::Sub(b(c(1.0)), b(FieldExpr::Cos(b(angle))))
}

/// Sine ease out: `sin(t * PI / 2)`
#[inline]
pub fn sine_out(t: FieldExpr) -> FieldExpr {
    let angle = FieldExpr::Mul(b(t), b(c(PI / 2.0)));
    FieldExpr::Sin(b(angle))
}

/// Sine ease in-out: `-cos(t * PI) / 2 + 0.5`
#[inline]
pub fn sine_in_out(t: FieldExpr) -> FieldExpr {
    let angle = FieldExpr::Mul(b(t), b(c(PI)));
    let cos_val = FieldExpr::Cos(b(angle));
    let neg_cos_half = FieldExpr::Div(b(FieldExpr::Neg(b(cos_val))), b(c(2.0)));
    FieldExpr::Add(b(neg_cos_half), b(c(0.5)))
}

// ============================================================================
// Exponential
// ============================================================================

/// Exponential ease in: `2^(10 * t - 10)` (returns 0 at t=0).
#[inline]
pub fn expo_in(t: FieldExpr) -> FieldExpr {
    // if t == 0 then 0 else 2^(10*t - 10)
    let exponent = FieldExpr::Sub(b(FieldExpr::Mul(b(c(10.0)), b(t.clone()))), b(c(10.0)));
    let pow_result = FieldExpr::Pow(b(c(2.0)), b(exponent));

    let condition = FieldExpr::Eq(b(t), b(c(0.0)));
    FieldExpr::IfThenElse {
        condition: b(condition),
        then_expr: b(c(0.0)),
        else_expr: b(pow_result),
    }
}

/// Exponential ease out: `1 - 2^(-10 * t)` (returns 1 at t=1).
#[inline]
pub fn expo_out(t: FieldExpr) -> FieldExpr {
    // if t == 1 then 1 else 1 - 2^(-10*t)
    let exponent = FieldExpr::Mul(b(c(-10.0)), b(t.clone()));
    let pow_result = FieldExpr::Pow(b(c(2.0)), b(exponent));
    let result = FieldExpr::Sub(b(c(1.0)), b(pow_result));

    let condition = FieldExpr::Eq(b(t), b(c(1.0)));
    FieldExpr::IfThenElse {
        condition: b(condition),
        then_expr: b(c(1.0)),
        else_expr: b(result),
    }
}

/// Exponential ease in-out.
#[inline]
pub fn expo_in_out(t: FieldExpr) -> FieldExpr {
    // Complex piecewise - use nested conditionals
    let is_zero = FieldExpr::Eq(b(t.clone()), b(c(0.0)));
    let is_one = FieldExpr::Eq(b(t.clone()), b(c(1.0)));
    let is_first_half = FieldExpr::Lt(b(t.clone()), b(c(0.5)));

    // First half: 2^(20*t - 10) / 2
    let exp1 = FieldExpr::Sub(b(FieldExpr::Mul(b(c(20.0)), b(t.clone()))), b(c(10.0)));
    let first_half = FieldExpr::Div(b(FieldExpr::Pow(b(c(2.0)), b(exp1))), b(c(2.0)));

    // Second half: (2 - 2^(-20*t + 10)) / 2
    let exp2 = FieldExpr::Add(b(FieldExpr::Mul(b(c(-20.0)), b(t))), b(c(10.0)));
    let second_half = FieldExpr::Div(
        b(FieldExpr::Sub(
            b(c(2.0)),
            b(FieldExpr::Pow(b(c(2.0)), b(exp2))),
        )),
        b(c(2.0)),
    );

    let mid_result = FieldExpr::IfThenElse {
        condition: b(is_first_half),
        then_expr: b(first_half),
        else_expr: b(second_half),
    };

    let not_one = FieldExpr::IfThenElse {
        condition: b(is_one),
        then_expr: b(c(1.0)),
        else_expr: b(mid_result),
    };

    FieldExpr::IfThenElse {
        condition: b(is_zero),
        then_expr: b(c(0.0)),
        else_expr: b(not_one),
    }
}

// ============================================================================
// Circular
// ============================================================================

/// Circular ease in: `1 - sqrt(1 - t^2)`
#[inline]
pub fn circ_in(t: FieldExpr) -> FieldExpr {
    let t_sq = FieldExpr::Mul(b(t.clone()), b(t));
    let sqrt_arg = FieldExpr::Sub(b(c(1.0)), b(t_sq));
    FieldExpr::Sub(b(c(1.0)), b(FieldExpr::Sqrt(b(sqrt_arg))))
}

/// Circular ease out: `sqrt(1 - (t - 1)^2)`
#[inline]
pub fn circ_out(t: FieldExpr) -> FieldExpr {
    let t_minus_one = FieldExpr::Sub(b(t), b(c(1.0)));
    let sq = FieldExpr::Mul(b(t_minus_one.clone()), b(t_minus_one));
    FieldExpr::Sqrt(b(FieldExpr::Sub(b(c(1.0)), b(sq))))
}

/// Circular ease in-out.
#[inline]
pub fn circ_in_out(t: FieldExpr) -> FieldExpr {
    let condition = FieldExpr::Lt(b(t.clone()), b(c(0.5)));

    // First half: (1 - sqrt(1 - (2*t)^2)) / 2
    let two_t = FieldExpr::Mul(b(c(2.0)), b(t.clone()));
    let two_t_sq = FieldExpr::Mul(b(two_t.clone()), b(two_t));
    let first_half = FieldExpr::Div(
        b(FieldExpr::Sub(
            b(c(1.0)),
            b(FieldExpr::Sqrt(b(FieldExpr::Sub(b(c(1.0)), b(two_t_sq))))),
        )),
        b(c(2.0)),
    );

    // Second half: (sqrt(1 - (-2*t + 2)^2) + 1) / 2
    let neg_two_t_plus_two = FieldExpr::Add(b(FieldExpr::Mul(b(c(-2.0)), b(t))), b(c(2.0)));
    let sq = FieldExpr::Mul(b(neg_two_t_plus_two.clone()), b(neg_two_t_plus_two));
    let second_half = FieldExpr::Div(
        b(FieldExpr::Add(
            b(FieldExpr::Sqrt(b(FieldExpr::Sub(b(c(1.0)), b(sq))))),
            b(c(1.0)),
        )),
        b(c(2.0)),
    );

    FieldExpr::IfThenElse {
        condition: b(condition),
        then_expr: b(first_half),
        else_expr: b(second_half),
    }
}

// ============================================================================
// Back (overshoot)
// ============================================================================

const BACK_C1: f32 = 1.70158;
const BACK_C2: f32 = BACK_C1 * 1.525;
const BACK_C3: f32 = BACK_C1 + 1.0;

/// Back ease in: overshoots backwards at start.
#[inline]
pub fn back_in(t: FieldExpr) -> FieldExpr {
    // c3 * t^3 - c1 * t^2
    let t2 = FieldExpr::Mul(b(t.clone()), b(t.clone()));
    let t3 = FieldExpr::Mul(b(t2.clone()), b(t));
    FieldExpr::Sub(
        b(FieldExpr::Mul(b(c(BACK_C3)), b(t3))),
        b(FieldExpr::Mul(b(c(BACK_C1)), b(t2))),
    )
}

/// Back ease out: overshoots at end.
#[inline]
pub fn back_out(t: FieldExpr) -> FieldExpr {
    // 1 + c3 * (t-1)^3 + c1 * (t-1)^2
    let t_minus_1 = FieldExpr::Sub(b(t), b(c(1.0)));
    let p2 = FieldExpr::Mul(b(t_minus_1.clone()), b(t_minus_1.clone()));
    let p3 = FieldExpr::Mul(b(p2.clone()), b(t_minus_1));
    FieldExpr::Add(
        b(c(1.0)),
        b(FieldExpr::Add(
            b(FieldExpr::Mul(b(c(BACK_C3)), b(p3))),
            b(FieldExpr::Mul(b(c(BACK_C1)), b(p2))),
        )),
    )
}

/// Back ease in-out.
#[inline]
pub fn back_in_out(t: FieldExpr) -> FieldExpr {
    let condition = FieldExpr::Lt(b(t.clone()), b(c(0.5)));

    // First half: ((2*t)^2 * ((c2+1)*2*t - c2)) / 2
    let two_t = FieldExpr::Mul(b(c(2.0)), b(t.clone()));
    let two_t_sq = FieldExpr::Mul(b(two_t.clone()), b(two_t.clone()));
    let inner = FieldExpr::Sub(
        b(FieldExpr::Mul(b(c(BACK_C2 + 1.0)), b(two_t))),
        b(c(BACK_C2)),
    );
    let first_half = FieldExpr::Div(b(FieldExpr::Mul(b(two_t_sq), b(inner))), b(c(2.0)));

    // Second half: ((2*t - 2)^2 * ((c2+1)*(2*t-2) + c2) + 2) / 2
    let two_t_minus_2 = FieldExpr::Sub(b(FieldExpr::Mul(b(c(2.0)), b(t))), b(c(2.0)));
    let p2 = FieldExpr::Mul(b(two_t_minus_2.clone()), b(two_t_minus_2.clone()));
    let inner2 = FieldExpr::Add(
        b(FieldExpr::Mul(b(c(BACK_C2 + 1.0)), b(two_t_minus_2))),
        b(c(BACK_C2)),
    );
    let second_half = FieldExpr::Div(
        b(FieldExpr::Add(
            b(FieldExpr::Mul(b(p2), b(inner2))),
            b(c(2.0)),
        )),
        b(c(2.0)),
    );

    FieldExpr::IfThenElse {
        condition: b(condition),
        then_expr: b(first_half),
        else_expr: b(second_half),
    }
}

// ============================================================================
// Smoothstep variants
// ============================================================================

/// Smoothstep (cubic Hermite): `t * t * (3 - 2 * t)`
#[inline]
pub fn smoothstep(t: FieldExpr) -> FieldExpr {
    let t2 = FieldExpr::Mul(b(t.clone()), b(t.clone()));
    let three_minus_2t = FieldExpr::Sub(b(c(3.0)), b(FieldExpr::Mul(b(c(2.0)), b(t))));
    FieldExpr::Mul(b(t2), b(three_minus_2t))
}

/// Smootherstep (quintic Hermite, Ken Perlin): `t^3 * (t * (6*t - 15) + 10)`
#[inline]
pub fn smootherstep(t: FieldExpr) -> FieldExpr {
    let t2 = FieldExpr::Mul(b(t.clone()), b(t.clone()));
    let t3 = FieldExpr::Mul(b(t2.clone()), b(t.clone()));

    // 6*t - 15
    let six_t_minus_15 = FieldExpr::Sub(b(FieldExpr::Mul(b(c(6.0)), b(t.clone()))), b(c(15.0)));

    // t * (6*t - 15) + 10
    let inner = FieldExpr::Add(b(FieldExpr::Mul(b(t), b(six_t_minus_15))), b(c(10.0)));

    FieldExpr::Mul(b(t3), b(inner))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn eval(expr: &FieldExpr, t: f32) -> f32 {
        expr.eval(0.0, 0.0, 0.0, t, &Default::default())
    }

    fn test_bounds(f: fn(FieldExpr) -> FieldExpr, name: &str) {
        let expr_0 = f(FieldExpr::Constant(0.0));
        let expr_1 = f(FieldExpr::Constant(1.0));

        let at_0 = eval(&expr_0, 0.0);
        let at_1 = eval(&expr_1, 0.0);

        assert!(at_0.abs() < 0.01, "{name}(0) = {at_0}, expected ~0");
        assert!((at_1 - 1.0).abs() < 0.01, "{name}(1) = {at_1}, expected ~1");
    }

    #[test]
    fn test_all_easings_bounds() {
        let easings: &[(fn(FieldExpr) -> FieldExpr, &str)] = &[
            (linear, "linear"),
            (quad_in, "quad_in"),
            (quad_out, "quad_out"),
            (quad_in_out, "quad_in_out"),
            (cubic_in, "cubic_in"),
            (cubic_out, "cubic_out"),
            (cubic_in_out, "cubic_in_out"),
            (quart_in, "quart_in"),
            (quart_out, "quart_out"),
            (quart_in_out, "quart_in_out"),
            (quint_in, "quint_in"),
            (quint_out, "quint_out"),
            (quint_in_out, "quint_in_out"),
            (sine_in, "sine_in"),
            (sine_out, "sine_out"),
            (sine_in_out, "sine_in_out"),
            (expo_in, "expo_in"),
            (expo_out, "expo_out"),
            (expo_in_out, "expo_in_out"),
            (circ_in, "circ_in"),
            (circ_out, "circ_out"),
            (circ_in_out, "circ_in_out"),
            (back_in, "back_in"),
            (back_out, "back_out"),
            (back_in_out, "back_in_out"),
            (smoothstep, "smoothstep"),
            (smootherstep, "smootherstep"),
        ];

        for (f, name) in easings {
            test_bounds(*f, name);
        }
    }

    #[test]
    fn test_quad_in_values() {
        let expr = quad_in(FieldExpr::T);
        // quad_in(0.5) = 0.25
        let value = eval(&expr, 0.5);
        assert!((value - 0.25).abs() < 0.001, "quad_in(0.5) = {value}");
    }

    #[test]
    fn test_quad_out_values() {
        let expr = quad_out(FieldExpr::T);
        // quad_out(0.5) = 0.75
        let value = eval(&expr, 0.5);
        assert!((value - 0.75).abs() < 0.001, "quad_out(0.5) = {value}");
    }

    #[test]
    fn test_smoothstep_midpoint() {
        let expr = smoothstep(FieldExpr::T);
        let value = eval(&expr, 0.5);
        assert!((value - 0.5).abs() < 0.001, "smoothstep(0.5) = {value}");
    }

    #[test]
    fn test_back_overshoots() {
        let expr_in = back_in(FieldExpr::T);
        let value = eval(&expr_in, 0.2);
        assert!(value < 0.0, "back_in(0.2) should be negative: {value}");

        let expr_out = back_out(FieldExpr::T);
        let value = eval(&expr_out, 0.8);
        assert!(value > 1.0, "back_out(0.8) should be > 1: {value}");
    }

    #[test]
    fn test_constant_folding_potential() {
        // When input is constant, the expression can be fully evaluated
        let expr = quad_in(FieldExpr::Constant(0.5));

        // The expression should evaluate to 0.25
        let value = eval(&expr, 0.0); // t doesn't matter, input is constant
        assert!((value - 0.25).abs() < 0.001);
    }
}
