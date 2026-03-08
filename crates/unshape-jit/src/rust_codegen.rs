//! Rust source code generation from wick/dew expression ASTs.
//!
//! Generates Rust source strings (and optionally `proc_macro2::TokenStream`) from
//! `wick_core::Ast` nodes. All values are modeled as `f64`, matching wick's evaluation
//! semantics.
//!
//! # Example
//!
//! ```
//! use wick_core::Ast;
//! use unshape_jit::rust_codegen::ast_to_rust;
//!
//! let ast = Ast::BinOp(
//!     wick_core::BinOp::Add,
//!     Box::new(Ast::Var("x".into())),
//!     Box::new(Ast::Num(1.0)),
//! );
//! assert_eq!(ast_to_rust(&ast), "(x + 1_f64)");
//! ```

use wick_core::{Ast, BinOp, Expr, UnaryOp};

#[cfg(feature = "cond")]
use wick_core::CompareOp;

/// Generate a Rust source string for the given AST node.
///
/// All values are `f64`. The output is a valid Rust expression that can be embedded
/// in a function body.
pub fn ast_to_rust(ast: &Ast) -> String {
    match ast {
        Ast::Num(n) => format!("{}_f64", n),
        Ast::Var(name) => name.clone(),
        Ast::BinOp(op, l, r) => {
            let l = ast_to_rust(l);
            let r = ast_to_rust(r);
            match op {
                BinOp::Add => format!("({l} + {r})"),
                BinOp::Sub => format!("({l} - {r})"),
                BinOp::Mul => format!("({l} * {r})"),
                BinOp::Div => format!("({l} / {r})"),
                BinOp::Pow => format!("f64::powf({l}, {r})"),
                BinOp::Rem => format!("({l} % {r})"),
                BinOp::BitAnd => format!("(({l} as i64 & {r} as i64) as f64)"),
                BinOp::BitOr => format!("(({l} as i64 | {r} as i64) as f64)"),
                BinOp::Shl => format!("(({l} as i64) << ({r} as u32)) as f64"),
                BinOp::Shr => format!("(({l} as i64) >> ({r} as u32)) as f64"),
            }
        }
        Ast::UnaryOp(op, x) => {
            let x = ast_to_rust(x);
            match op {
                UnaryOp::Neg => format!("(-{x})"),
                UnaryOp::BitNot => format!("(!(({x}) as i64) as f64)"),
                #[cfg(feature = "cond")]
                UnaryOp::Not => {
                    format!("(if ({x}) != 0.0_f64 {{ 0.0_f64 }} else {{ 1.0_f64 }})")
                }
            }
        }
        #[cfg(feature = "func")]
        Ast::Call(name, args) => {
            let args: Vec<String> = args.iter().map(ast_to_rust).collect();
            format!("{}({})", name, args.join(", "))
        }
        #[cfg(feature = "cond")]
        Ast::Compare(op, l, r) => {
            let l = ast_to_rust(l);
            let r = ast_to_rust(r);
            let cmp = match op {
                CompareOp::Lt => "<",
                CompareOp::Le => "<=",
                CompareOp::Gt => ">",
                CompareOp::Ge => ">=",
                CompareOp::Eq => "==",
                CompareOp::Ne => "!=",
            };
            format!("(if ({l}) {cmp} ({r}) {{ 1.0_f64 }} else {{ 0.0_f64 }})")
        }
        #[cfg(feature = "cond")]
        Ast::And(l, r) => {
            let l = ast_to_rust(l);
            let r = ast_to_rust(r);
            format!("(if ({l}) != 0.0_f64 && ({r}) != 0.0_f64 {{ 1.0_f64 }} else {{ 0.0_f64 }})")
        }
        #[cfg(feature = "cond")]
        Ast::Or(l, r) => {
            let l = ast_to_rust(l);
            let r = ast_to_rust(r);
            format!("(if ({l}) != 0.0_f64 || ({r}) != 0.0_f64 {{ 1.0_f64 }} else {{ 0.0_f64 }})")
        }
        #[cfg(feature = "cond")]
        Ast::If(cond, then_branch, else_branch) => {
            let cond = ast_to_rust(cond);
            let then_branch = ast_to_rust(then_branch);
            let else_branch = ast_to_rust(else_branch);
            format!("(if ({cond}) != 0.0_f64 {{ {then_branch} }} else {{ {else_branch} }})")
        }
        Ast::Let { name, value, body } => {
            let value = ast_to_rust(value);
            let body = ast_to_rust(body);
            format!("{{ let {name}: f64 = {value}; {body} }}")
        }
    }
}

/// Generate a Rust source string for the given `Expr`.
///
/// Thin wrapper around [`ast_to_rust`].
pub fn expr_to_rust(expr: &Expr) -> String {
    ast_to_rust(expr.ast())
}

/// Generates a complete Rust function from a dew expression,
/// with only the context fields actually used as parameters.
///
/// Only parameters actually referenced in the expression are included,
/// providing dead code elimination at the function signature level.
///
/// # Arguments
/// * `fn_name` - Name of the generated function
/// * `expr` - The dew expression
/// * `all_context_fields` - All possible context field names in declaration order
///   (e.g. `&["u", "v", "x", "y", "width", "height", "time"]`)
///
/// # Returns
/// A complete Rust function like:
/// ```text
/// fn my_fn(u: f64, v: f64) -> f64 {
///     (u * 2.0_f64 + v)
/// }
/// ```
/// Only parameters actually referenced in the expression are included.
#[cfg(feature = "introspect")]
pub fn expr_to_rust_fn(fn_name: &str, expr: &Expr, all_context_fields: &[&str]) -> String {
    let used = expr.free_vars();
    let params: Vec<&str> = all_context_fields
        .iter()
        .copied()
        .filter(|f| used.contains(*f))
        .collect();
    let param_list = params
        .iter()
        .map(|p| format!("{p}: f64"))
        .collect::<Vec<_>>()
        .join(", ");
    let body = expr_to_rust(expr);
    format!("fn {fn_name}({param_list}) -> f64 {{\n    {body}\n}}")
}

/// Returns which fields from `all_context_fields` are actually used by the expression.
///
/// Useful for callers to know which values they need to compute before calling eval,
/// enabling dead code elimination in the caller (skip computing z and t if unused).
#[cfg(feature = "introspect")]
pub fn used_fields<'a>(expr: &Expr, all_context_fields: &[&'a str]) -> Vec<&'a str> {
    let free = expr.free_vars();
    all_context_fields
        .iter()
        .copied()
        .filter(|f| free.contains(*f))
        .collect()
}

/// Generate a `proc_macro2::TokenStream` for the given AST node.
///
/// Equivalent to [`ast_to_rust`] but produces token trees instead of a string.
/// Gate behind the `proc-macro2` feature.
#[cfg(feature = "proc-macro2")]
pub fn ast_to_tokens(ast: &Ast) -> proc_macro2::TokenStream {
    use proc_macro2::TokenStream;
    use quote::quote;

    match ast {
        Ast::Num(n) => {
            // Build the literal as a string and parse it into tokens.
            // quote! can't suffix a variable as `#n_f64`, so we construct the
            // token stream from the formatted string.
            let lit: proc_macro2::TokenStream = format!("{n}_f64")
                .parse()
                .expect("numeric literal is always valid tokens");
            lit
        }
        Ast::Var(name) => {
            let ident = proc_macro2::Ident::new(name, proc_macro2::Span::call_site());
            quote! { #ident }
        }
        Ast::BinOp(op, l, r) => {
            let l = ast_to_tokens(l);
            let r = ast_to_tokens(r);
            match op {
                BinOp::Add => quote! { (#l + #r) },
                BinOp::Sub => quote! { (#l - #r) },
                BinOp::Mul => quote! { (#l * #r) },
                BinOp::Div => quote! { (#l / #r) },
                BinOp::Pow => quote! { f64::powf(#l, #r) },
                BinOp::Rem => quote! { (#l % #r) },
                BinOp::BitAnd => quote! { ((#l as i64 & #r as i64) as f64) },
                BinOp::BitOr => quote! { ((#l as i64 | #r as i64) as f64) },
                BinOp::Shl => quote! { ((#l as i64) << (#r as u32)) as f64 },
                BinOp::Shr => quote! { ((#l as i64) >> (#r as u32)) as f64 },
            }
        }
        Ast::UnaryOp(op, x) => {
            let x = ast_to_tokens(x);
            match op {
                UnaryOp::Neg => quote! { (-#x) },
                UnaryOp::BitNot => quote! { (!((#x) as i64) as f64) },
                #[cfg(feature = "cond")]
                UnaryOp::Not => {
                    quote! { (if (#x) != 0.0_f64 { 0.0_f64 } else { 1.0_f64 }) }
                }
            }
        }
        #[cfg(feature = "func")]
        Ast::Call(name, args) => {
            let ident = proc_macro2::Ident::new(name, proc_macro2::Span::call_site());
            let args: Vec<TokenStream> = args.iter().map(ast_to_tokens).collect();
            quote! { #ident(#(#args),*) }
        }
        #[cfg(feature = "cond")]
        Ast::Compare(op, l, r) => {
            let l = ast_to_tokens(l);
            let r = ast_to_tokens(r);
            match op {
                CompareOp::Lt => quote! { (if (#l) < (#r) { 1.0_f64 } else { 0.0_f64 }) },
                CompareOp::Le => quote! { (if (#l) <= (#r) { 1.0_f64 } else { 0.0_f64 }) },
                CompareOp::Gt => quote! { (if (#l) > (#r) { 1.0_f64 } else { 0.0_f64 }) },
                CompareOp::Ge => quote! { (if (#l) >= (#r) { 1.0_f64 } else { 0.0_f64 }) },
                CompareOp::Eq => quote! { (if (#l) == (#r) { 1.0_f64 } else { 0.0_f64 }) },
                CompareOp::Ne => quote! { (if (#l) != (#r) { 1.0_f64 } else { 0.0_f64 }) },
            }
        }
        #[cfg(feature = "cond")]
        Ast::And(l, r) => {
            let l = ast_to_tokens(l);
            let r = ast_to_tokens(r);
            quote! { (if (#l) != 0.0_f64 && (#r) != 0.0_f64 { 1.0_f64 } else { 0.0_f64 }) }
        }
        #[cfg(feature = "cond")]
        Ast::Or(l, r) => {
            let l = ast_to_tokens(l);
            let r = ast_to_tokens(r);
            quote! { (if (#l) != 0.0_f64 || (#r) != 0.0_f64 { 1.0_f64 } else { 0.0_f64 }) }
        }
        #[cfg(feature = "cond")]
        Ast::If(cond, then_branch, else_branch) => {
            let cond = ast_to_tokens(cond);
            let then_branch = ast_to_tokens(then_branch);
            let else_branch = ast_to_tokens(else_branch);
            quote! { (if (#cond) != 0.0_f64 { #then_branch } else { #else_branch }) }
        }
        Ast::Let { name, value, body } => {
            let ident = proc_macro2::Ident::new(name, proc_macro2::Span::call_site());
            let value = ast_to_tokens(value);
            let body = ast_to_tokens(body);
            quote! { { let #ident: f64 = #value; #body } }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wick_core::Ast;

    #[cfg(feature = "cond")]
    use wick_core::CompareOp;

    // ========================================================================
    // ast_to_rust tests
    // ========================================================================

    #[test]
    fn test_num_integer() {
        assert_eq!(ast_to_rust(&Ast::Num(3.0)), "3_f64");
    }

    #[test]
    fn test_num_float() {
        assert_eq!(ast_to_rust(&Ast::Num(1.5)), "1.5_f64");
    }

    #[test]
    fn test_num_negative() {
        assert_eq!(ast_to_rust(&Ast::Num(-1.0)), "-1_f64");
    }

    #[test]
    fn test_var() {
        assert_eq!(ast_to_rust(&Ast::Var("x".into())), "x");
        assert_eq!(ast_to_rust(&Ast::Var("my_var".into())), "my_var");
    }

    #[test]
    fn test_binop_add() {
        let ast = Ast::BinOp(
            BinOp::Add,
            Box::new(Ast::Var("x".into())),
            Box::new(Ast::Num(1.0)),
        );
        assert_eq!(ast_to_rust(&ast), "(x + 1_f64)");
    }

    #[test]
    fn test_binop_sub() {
        let ast = Ast::BinOp(
            BinOp::Sub,
            Box::new(Ast::Var("x".into())),
            Box::new(Ast::Num(2.0)),
        );
        assert_eq!(ast_to_rust(&ast), "(x - 2_f64)");
    }

    #[test]
    fn test_binop_mul() {
        let ast = Ast::BinOp(
            BinOp::Mul,
            Box::new(Ast::Num(2.0)),
            Box::new(Ast::Var("x".into())),
        );
        assert_eq!(ast_to_rust(&ast), "(2_f64 * x)");
    }

    #[test]
    fn test_binop_div() {
        let ast = Ast::BinOp(
            BinOp::Div,
            Box::new(Ast::Var("x".into())),
            Box::new(Ast::Num(4.0)),
        );
        assert_eq!(ast_to_rust(&ast), "(x / 4_f64)");
    }

    #[test]
    fn test_binop_pow() {
        let ast = Ast::BinOp(
            BinOp::Pow,
            Box::new(Ast::Var("x".into())),
            Box::new(Ast::Num(2.0)),
        );
        assert_eq!(ast_to_rust(&ast), "f64::powf(x, 2_f64)");
    }

    #[test]
    fn test_binop_rem() {
        let ast = Ast::BinOp(
            BinOp::Rem,
            Box::new(Ast::Var("x".into())),
            Box::new(Ast::Num(3.0)),
        );
        assert_eq!(ast_to_rust(&ast), "(x % 3_f64)");
    }

    #[test]
    fn test_binop_bitand() {
        let ast = Ast::BinOp(
            BinOp::BitAnd,
            Box::new(Ast::Var("a".into())),
            Box::new(Ast::Var("b".into())),
        );
        assert_eq!(ast_to_rust(&ast), "((a as i64 & b as i64) as f64)");
    }

    #[test]
    fn test_binop_bitor() {
        let ast = Ast::BinOp(
            BinOp::BitOr,
            Box::new(Ast::Var("a".into())),
            Box::new(Ast::Var("b".into())),
        );
        assert_eq!(ast_to_rust(&ast), "((a as i64 | b as i64) as f64)");
    }

    #[test]
    fn test_binop_shl() {
        let ast = Ast::BinOp(
            BinOp::Shl,
            Box::new(Ast::Var("x".into())),
            Box::new(Ast::Num(2.0)),
        );
        assert_eq!(ast_to_rust(&ast), "((x as i64) << (2_f64 as u32)) as f64");
    }

    #[test]
    fn test_binop_shr() {
        let ast = Ast::BinOp(
            BinOp::Shr,
            Box::new(Ast::Var("x".into())),
            Box::new(Ast::Num(1.0)),
        );
        assert_eq!(ast_to_rust(&ast), "((x as i64) >> (1_f64 as u32)) as f64");
    }

    #[test]
    fn test_unary_neg() {
        let ast = Ast::UnaryOp(UnaryOp::Neg, Box::new(Ast::Var("x".into())));
        assert_eq!(ast_to_rust(&ast), "(-x)");
    }

    #[test]
    fn test_unary_bitnot() {
        let ast = Ast::UnaryOp(UnaryOp::BitNot, Box::new(Ast::Var("x".into())));
        assert_eq!(ast_to_rust(&ast), "(!((x) as i64) as f64)");
    }

    #[test]
    fn test_let() {
        let ast = Ast::Let {
            name: "t".into(),
            value: Box::new(Ast::Num(2.0)),
            body: Box::new(Ast::BinOp(
                BinOp::Mul,
                Box::new(Ast::Var("t".into())),
                Box::new(Ast::Var("t".into())),
            )),
        };
        assert_eq!(ast_to_rust(&ast), "{ let t: f64 = 2_f64; (t * t) }");
    }

    #[test]
    fn test_nested_binop() {
        // (x + 1) * (x - 1)
        let ast = Ast::BinOp(
            BinOp::Mul,
            Box::new(Ast::BinOp(
                BinOp::Add,
                Box::new(Ast::Var("x".into())),
                Box::new(Ast::Num(1.0)),
            )),
            Box::new(Ast::BinOp(
                BinOp::Sub,
                Box::new(Ast::Var("x".into())),
                Box::new(Ast::Num(1.0)),
            )),
        );
        assert_eq!(ast_to_rust(&ast), "((x + 1_f64) * (x - 1_f64))");
    }

    #[cfg(feature = "func")]
    #[test]
    fn test_call_no_args() {
        let ast = Ast::Call("rand".into(), vec![]);
        assert_eq!(ast_to_rust(&ast), "rand()");
    }

    #[cfg(feature = "func")]
    #[test]
    fn test_call_single_arg() {
        let ast = Ast::Call("sin".into(), vec![Ast::Var("x".into())]);
        assert_eq!(ast_to_rust(&ast), "sin(x)");
    }

    #[cfg(feature = "func")]
    #[test]
    fn test_call_multi_args() {
        let ast = Ast::Call("pow".into(), vec![Ast::Var("x".into()), Ast::Num(2.0)]);
        assert_eq!(ast_to_rust(&ast), "pow(x, 2_f64)");
    }

    #[cfg(feature = "cond")]
    #[test]
    fn test_unary_not() {
        let ast = Ast::UnaryOp(UnaryOp::Not, Box::new(Ast::Var("x".into())));
        assert_eq!(
            ast_to_rust(&ast),
            "(if (x) != 0.0_f64 { 0.0_f64 } else { 1.0_f64 })"
        );
    }

    #[cfg(feature = "cond")]
    #[test]
    fn test_compare_lt() {
        let ast = Ast::Compare(
            CompareOp::Lt,
            Box::new(Ast::Var("x".into())),
            Box::new(Ast::Num(0.0)),
        );
        assert_eq!(
            ast_to_rust(&ast),
            "(if (x) < (0_f64) { 1.0_f64 } else { 0.0_f64 })"
        );
    }

    #[cfg(feature = "cond")]
    #[test]
    fn test_compare_le() {
        let ast = Ast::Compare(
            CompareOp::Le,
            Box::new(Ast::Var("x".into())),
            Box::new(Ast::Num(1.0)),
        );
        assert_eq!(
            ast_to_rust(&ast),
            "(if (x) <= (1_f64) { 1.0_f64 } else { 0.0_f64 })"
        );
    }

    #[cfg(feature = "cond")]
    #[test]
    fn test_compare_gt() {
        let ast = Ast::Compare(
            CompareOp::Gt,
            Box::new(Ast::Var("x".into())),
            Box::new(Ast::Num(0.0)),
        );
        assert_eq!(
            ast_to_rust(&ast),
            "(if (x) > (0_f64) { 1.0_f64 } else { 0.0_f64 })"
        );
    }

    #[cfg(feature = "cond")]
    #[test]
    fn test_compare_ge() {
        let ast = Ast::Compare(
            CompareOp::Ge,
            Box::new(Ast::Var("x".into())),
            Box::new(Ast::Num(0.0)),
        );
        assert_eq!(
            ast_to_rust(&ast),
            "(if (x) >= (0_f64) { 1.0_f64 } else { 0.0_f64 })"
        );
    }

    #[cfg(feature = "cond")]
    #[test]
    fn test_compare_eq() {
        let ast = Ast::Compare(
            CompareOp::Eq,
            Box::new(Ast::Var("x".into())),
            Box::new(Ast::Var("y".into())),
        );
        assert_eq!(
            ast_to_rust(&ast),
            "(if (x) == (y) { 1.0_f64 } else { 0.0_f64 })"
        );
    }

    #[cfg(feature = "cond")]
    #[test]
    fn test_compare_ne() {
        let ast = Ast::Compare(
            CompareOp::Ne,
            Box::new(Ast::Var("x".into())),
            Box::new(Ast::Var("y".into())),
        );
        assert_eq!(
            ast_to_rust(&ast),
            "(if (x) != (y) { 1.0_f64 } else { 0.0_f64 })"
        );
    }

    #[cfg(feature = "cond")]
    #[test]
    fn test_and() {
        let ast = Ast::And(
            Box::new(Ast::Var("a".into())),
            Box::new(Ast::Var("b".into())),
        );
        assert_eq!(
            ast_to_rust(&ast),
            "(if (a) != 0.0_f64 && (b) != 0.0_f64 { 1.0_f64 } else { 0.0_f64 })"
        );
    }

    #[cfg(feature = "cond")]
    #[test]
    fn test_or() {
        let ast = Ast::Or(
            Box::new(Ast::Var("a".into())),
            Box::new(Ast::Var("b".into())),
        );
        assert_eq!(
            ast_to_rust(&ast),
            "(if (a) != 0.0_f64 || (b) != 0.0_f64 { 1.0_f64 } else { 0.0_f64 })"
        );
    }

    #[cfg(feature = "cond")]
    #[test]
    fn test_if() {
        let ast = Ast::If(
            Box::new(Ast::Var("cond".into())),
            Box::new(Ast::Num(1.0)),
            Box::new(Ast::Num(0.0)),
        );
        assert_eq!(
            ast_to_rust(&ast),
            "(if (cond) != 0.0_f64 { 1_f64 } else { 0_f64 })"
        );
    }

    // ========================================================================
    // expr_to_rust_fn / used_fields tests
    // ========================================================================

    #[cfg(feature = "introspect")]
    #[test]
    fn test_expr_to_rust_fn_filters_params() {
        let expr = wick_core::Expr::parse("u * 2.0 + v").unwrap();
        let all_fields = &["u", "v", "x", "y", "time"];
        let code = expr_to_rust_fn("my_fn", &expr, all_fields);
        assert!(
            code.starts_with("fn my_fn("),
            "expected fn header, got: {code}"
        );
        assert!(code.contains("u: f64"), "expected u param in: {code}");
        assert!(code.contains("v: f64"), "expected v param in: {code}");
        assert!(!code.contains("x: f64"), "unexpected x param in: {code}");
        assert!(!code.contains("y: f64"), "unexpected y param in: {code}");
        assert!(
            !code.contains("time: f64"),
            "unexpected time param in: {code}"
        );
        assert!(code.contains("-> f64"), "expected return type in: {code}");
    }

    #[cfg(feature = "introspect")]
    #[test]
    fn test_expr_to_rust_fn_single_param() {
        let expr = wick_core::Expr::parse("sin(time)").unwrap();
        let all_fields = &["x", "y", "z", "time"];
        let code = expr_to_rust_fn("audio_fn", &expr, all_fields);
        assert!(code.contains("time: f64"), "expected time param in: {code}");
        assert!(!code.contains("x: f64"), "unexpected x param in: {code}");
        assert!(!code.contains("y: f64"), "unexpected y param in: {code}");
        assert!(!code.contains("z: f64"), "unexpected z param in: {code}");
    }

    #[cfg(feature = "introspect")]
    #[test]
    fn test_expr_to_rust_fn_no_params() {
        let expr = wick_core::Expr::parse("3.14").unwrap();
        let all_fields = &["u", "v", "x", "y", "time"];
        let code = expr_to_rust_fn("const_fn", &expr, all_fields);
        // No parameters — empty parameter list
        assert!(
            code.contains("fn const_fn() -> f64"),
            "expected empty params in: {code}"
        );
    }

    #[cfg(feature = "introspect")]
    #[test]
    fn test_used_fields_subset() {
        let expr = wick_core::Expr::parse("u * 2.0 + v").unwrap();
        let all_fields = &["u", "v", "x", "y", "time"];
        let result = used_fields(&expr, all_fields);
        assert_eq!(result, vec!["u", "v"]);
    }

    #[cfg(feature = "introspect")]
    #[test]
    fn test_used_fields_preserves_order() {
        let expr = wick_core::Expr::parse("y + x").unwrap();
        let all_fields = &["x", "y", "z", "t"];
        let result = used_fields(&expr, all_fields);
        // Should preserve declaration order: x before y
        assert_eq!(result, vec!["x", "y"]);
    }

    #[cfg(feature = "introspect")]
    #[test]
    fn test_used_fields_none() {
        let expr = wick_core::Expr::parse("42.0").unwrap();
        let all_fields = &["x", "y", "z", "t"];
        let result = used_fields(&expr, all_fields);
        assert!(result.is_empty());
    }

    // ========================================================================
    // expr_to_rust tests
    // ========================================================================

    #[test]
    fn test_expr_to_rust_simple() {
        let expr = wick_core::Expr::parse("x + 1").unwrap();
        let code = expr_to_rust(&expr);
        // Should produce valid Rust — assert it contains the key parts
        assert!(code.contains("x"), "missing 'x' in: {code}");
        assert!(code.contains("1_f64"), "missing literal in: {code}");
        assert!(code.contains('+'), "missing '+' in: {code}");
    }

    #[test]
    fn test_expr_to_rust_constant() {
        let expr = wick_core::Expr::parse("1.5").unwrap();
        let code = expr_to_rust(&expr);
        assert!(code.contains("1.5_f64"), "expected 1.5_f64, got: {code}");
    }

    #[test]
    fn test_expr_to_rust_nested() {
        // (x * 2) + 1 - parsed from string
        let expr = wick_core::Expr::parse("x * 2 + 1").unwrap();
        let code = expr_to_rust(&expr);
        // Should be well-formed (parens balance)
        let open = code.chars().filter(|&c| c == '(').count();
        let close = code.chars().filter(|&c| c == ')').count();
        assert_eq!(open, close, "unbalanced parens in: {code}");
    }
}
