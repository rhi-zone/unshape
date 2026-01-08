//! Expression language for field evaluation.
//!
//! A simple expression parser that compiles string expressions into evaluable fields.
//!
//! # Syntax
//!
//! ```text
//! // Variables
//! x, y, z          // Coordinates
//! time             // Context time
//!
//! // Operators (precedence low to high)
//! a + b, a - b     // Addition, subtraction
//! a * b, a / b     // Multiplication, division
//! a ^ b            // Exponentiation
//! -a               // Negation
//!
//! // Functions
//! sin(x), cos(x), tan(x)
//! sqrt(x), abs(x)
//! floor(x), ceil(x), fract(x)
//! min(a, b), max(a, b)
//! clamp(x, lo, hi)
//! lerp(a, b, t)    // Linear interpolation
//! noise(x, y)      // 2D Perlin noise
//! ```
//!
//! # Example
//!
//! ```ignore
//! use resin_core::expr::Expr;
//! use resin_core::EvalContext;
//! use glam::Vec2;
//!
//! let expr = Expr::parse("sin(x * 3.14) + 0.5").unwrap();
//! let ctx = EvalContext::new();
//! let value = expr.eval(Vec2::new(0.5, 0.0), &ctx);
//! ```

use crate::context::EvalContext;
use crate::noise;
use glam::Vec2;
use std::f32::consts::PI;

/// Expression parse error.
#[derive(Debug, Clone, PartialEq)]
pub enum ParseError {
    UnexpectedChar(char),
    UnexpectedEnd,
    UnexpectedToken(String),
    UnknownFunction(String),
    UnknownVariable(String),
    WrongArgCount {
        func: String,
        expected: usize,
        got: usize,
    },
    InvalidNumber(String),
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ParseError::UnexpectedChar(c) => write!(f, "unexpected character: '{}'", c),
            ParseError::UnexpectedEnd => write!(f, "unexpected end of expression"),
            ParseError::UnexpectedToken(t) => write!(f, "unexpected token: '{}'", t),
            ParseError::UnknownFunction(name) => write!(f, "unknown function: '{}'", name),
            ParseError::UnknownVariable(name) => write!(f, "unknown variable: '{}'", name),
            ParseError::WrongArgCount {
                func,
                expected,
                got,
            } => {
                write!(
                    f,
                    "function '{}' expects {} args, got {}",
                    func, expected, got
                )
            }
            ParseError::InvalidNumber(s) => write!(f, "invalid number: '{}'", s),
        }
    }
}

impl std::error::Error for ParseError {}

/// Token types for lexing.
#[derive(Debug, Clone, PartialEq)]
enum Token {
    Number(f32),
    Ident(String),
    Plus,
    Minus,
    Star,
    Slash,
    Caret,
    LParen,
    RParen,
    Comma,
    Eof,
}

/// Tokenizer for expressions.
struct Lexer<'a> {
    input: &'a str,
    pos: usize,
}

impl<'a> Lexer<'a> {
    fn new(input: &'a str) -> Self {
        Self { input, pos: 0 }
    }

    fn peek_char(&self) -> Option<char> {
        self.input[self.pos..].chars().next()
    }

    fn next_char(&mut self) -> Option<char> {
        let c = self.peek_char()?;
        self.pos += c.len_utf8();
        Some(c)
    }

    fn skip_whitespace(&mut self) {
        while let Some(c) = self.peek_char() {
            if c.is_whitespace() {
                self.next_char();
            } else {
                break;
            }
        }
    }

    fn read_number(&mut self) -> Result<f32, ParseError> {
        let start = self.pos;
        while let Some(c) = self.peek_char() {
            if c.is_ascii_digit() || c == '.' {
                self.next_char();
            } else {
                break;
            }
        }
        let s = &self.input[start..self.pos];
        s.parse()
            .map_err(|_| ParseError::InvalidNumber(s.to_string()))
    }

    fn read_ident(&mut self) -> String {
        let start = self.pos;
        while let Some(c) = self.peek_char() {
            if c.is_alphanumeric() || c == '_' {
                self.next_char();
            } else {
                break;
            }
        }
        self.input[start..self.pos].to_string()
    }

    fn next_token(&mut self) -> Result<Token, ParseError> {
        self.skip_whitespace();

        let Some(c) = self.peek_char() else {
            return Ok(Token::Eof);
        };

        match c {
            '+' => {
                self.next_char();
                Ok(Token::Plus)
            }
            '-' => {
                self.next_char();
                Ok(Token::Minus)
            }
            '*' => {
                self.next_char();
                Ok(Token::Star)
            }
            '/' => {
                self.next_char();
                Ok(Token::Slash)
            }
            '^' => {
                self.next_char();
                Ok(Token::Caret)
            }
            '(' => {
                self.next_char();
                Ok(Token::LParen)
            }
            ')' => {
                self.next_char();
                Ok(Token::RParen)
            }
            ',' => {
                self.next_char();
                Ok(Token::Comma)
            }
            '0'..='9' | '.' => Ok(Token::Number(self.read_number()?)),
            'a'..='z' | 'A'..='Z' | '_' => Ok(Token::Ident(self.read_ident())),
            _ => Err(ParseError::UnexpectedChar(c)),
        }
    }
}

/// AST node for expressions.
#[derive(Debug, Clone)]
pub enum Ast {
    Num(f32),
    Var(Var),
    BinOp(BinOp, Box<Ast>, Box<Ast>),
    UnaryOp(UnaryOp, Box<Ast>),
    Call(Func, Vec<Ast>),
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Var {
    X,
    Y,
    Z,
    Time,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum UnaryOp {
    Neg,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Func {
    Sin,
    Cos,
    Tan,
    Sqrt,
    Abs,
    Floor,
    Ceil,
    Fract,
    Min,
    Max,
    Clamp,
    Lerp,
    Noise,
}

impl Func {
    fn from_name(name: &str) -> Option<Self> {
        match name {
            "sin" => Some(Func::Sin),
            "cos" => Some(Func::Cos),
            "tan" => Some(Func::Tan),
            "sqrt" => Some(Func::Sqrt),
            "abs" => Some(Func::Abs),
            "floor" => Some(Func::Floor),
            "ceil" => Some(Func::Ceil),
            "fract" => Some(Func::Fract),
            "min" => Some(Func::Min),
            "max" => Some(Func::Max),
            "clamp" => Some(Func::Clamp),
            "lerp" | "mix" => Some(Func::Lerp),
            "noise" | "perlin" => Some(Func::Noise),
            _ => None,
        }
    }

    fn arg_count(&self) -> usize {
        match self {
            Func::Sin
            | Func::Cos
            | Func::Tan
            | Func::Sqrt
            | Func::Abs
            | Func::Floor
            | Func::Ceil
            | Func::Fract => 1,
            Func::Min | Func::Max | Func::Noise => 2,
            Func::Clamp | Func::Lerp => 3,
        }
    }
}

/// Parser for expressions.
struct Parser<'a> {
    lexer: Lexer<'a>,
    current: Token,
}

impl<'a> Parser<'a> {
    fn new(input: &'a str) -> Result<Self, ParseError> {
        let mut lexer = Lexer::new(input);
        let current = lexer.next_token()?;
        Ok(Self { lexer, current })
    }

    fn advance(&mut self) -> Result<(), ParseError> {
        self.current = self.lexer.next_token()?;
        Ok(())
    }

    fn expect(&mut self, expected: Token) -> Result<(), ParseError> {
        if self.current == expected {
            self.advance()
        } else {
            Err(ParseError::UnexpectedToken(format!("{:?}", self.current)))
        }
    }

    fn parse_expr(&mut self) -> Result<Ast, ParseError> {
        self.parse_add_sub()
    }

    fn parse_add_sub(&mut self) -> Result<Ast, ParseError> {
        let mut left = self.parse_mul_div()?;

        loop {
            match &self.current {
                Token::Plus => {
                    self.advance()?;
                    let right = self.parse_mul_div()?;
                    left = Ast::BinOp(BinOp::Add, Box::new(left), Box::new(right));
                }
                Token::Minus => {
                    self.advance()?;
                    let right = self.parse_mul_div()?;
                    left = Ast::BinOp(BinOp::Sub, Box::new(left), Box::new(right));
                }
                _ => break,
            }
        }

        Ok(left)
    }

    fn parse_mul_div(&mut self) -> Result<Ast, ParseError> {
        let mut left = self.parse_power()?;

        loop {
            match &self.current {
                Token::Star => {
                    self.advance()?;
                    let right = self.parse_power()?;
                    left = Ast::BinOp(BinOp::Mul, Box::new(left), Box::new(right));
                }
                Token::Slash => {
                    self.advance()?;
                    let right = self.parse_power()?;
                    left = Ast::BinOp(BinOp::Div, Box::new(left), Box::new(right));
                }
                _ => break,
            }
        }

        Ok(left)
    }

    fn parse_power(&mut self) -> Result<Ast, ParseError> {
        let base = self.parse_unary()?;

        if self.current == Token::Caret {
            self.advance()?;
            let exp = self.parse_power()?; // Right associative
            Ok(Ast::BinOp(BinOp::Pow, Box::new(base), Box::new(exp)))
        } else {
            Ok(base)
        }
    }

    fn parse_unary(&mut self) -> Result<Ast, ParseError> {
        if self.current == Token::Minus {
            self.advance()?;
            let inner = self.parse_unary()?;
            Ok(Ast::UnaryOp(UnaryOp::Neg, Box::new(inner)))
        } else {
            self.parse_primary()
        }
    }

    fn parse_primary(&mut self) -> Result<Ast, ParseError> {
        match &self.current {
            Token::Number(n) => {
                let n = *n;
                self.advance()?;
                Ok(Ast::Num(n))
            }
            Token::Ident(name) => {
                let name = name.clone();
                self.advance()?;

                // Check if it's a function call
                if self.current == Token::LParen {
                    self.advance()?;
                    let func = Func::from_name(&name)
                        .ok_or_else(|| ParseError::UnknownFunction(name.clone()))?;

                    let mut args = Vec::new();
                    if self.current != Token::RParen {
                        args.push(self.parse_expr()?);
                        while self.current == Token::Comma {
                            self.advance()?;
                            args.push(self.parse_expr()?);
                        }
                    }
                    self.expect(Token::RParen)?;

                    let expected = func.arg_count();
                    if args.len() != expected {
                        return Err(ParseError::WrongArgCount {
                            func: name,
                            expected,
                            got: args.len(),
                        });
                    }

                    Ok(Ast::Call(func, args))
                } else {
                    // It's a variable
                    let var = match name.as_str() {
                        "x" => Var::X,
                        "y" => Var::Y,
                        "z" => Var::Z,
                        "time" | "t" => Var::Time,
                        "pi" | "PI" => return Ok(Ast::Num(PI)),
                        "e" | "E" => return Ok(Ast::Num(std::f32::consts::E)),
                        _ => return Err(ParseError::UnknownVariable(name)),
                    };
                    Ok(Ast::Var(var))
                }
            }
            Token::LParen => {
                self.advance()?;
                let inner = self.parse_expr()?;
                self.expect(Token::RParen)?;
                Ok(inner)
            }
            Token::Eof => Err(ParseError::UnexpectedEnd),
            _ => Err(ParseError::UnexpectedToken(format!("{:?}", self.current))),
        }
    }
}

/// A compiled expression that can be evaluated.
#[derive(Debug, Clone)]
pub struct Expr {
    ast: Ast,
}

impl Expr {
    /// Parses an expression from a string.
    pub fn parse(input: &str) -> Result<Self, ParseError> {
        let mut parser = Parser::new(input)?;
        let ast = parser.parse_expr()?;
        if parser.current != Token::Eof {
            return Err(ParseError::UnexpectedToken(format!("{:?}", parser.current)));
        }
        Ok(Self { ast })
    }

    /// Evaluates the expression at a 2D point.
    pub fn eval(&self, pos: Vec2, ctx: &EvalContext) -> f32 {
        Self::eval_ast(&self.ast, pos.x, pos.y, 0.0, ctx)
    }

    /// Evaluates the expression at a 3D point.
    pub fn eval3(&self, x: f32, y: f32, z: f32, ctx: &EvalContext) -> f32 {
        Self::eval_ast(&self.ast, x, y, z, ctx)
    }

    fn eval_ast(ast: &Ast, x: f32, y: f32, z: f32, ctx: &EvalContext) -> f32 {
        match ast {
            Ast::Num(n) => *n,
            Ast::Var(v) => match v {
                Var::X => x,
                Var::Y => y,
                Var::Z => z,
                Var::Time => ctx.time,
            },
            Ast::BinOp(op, l, r) => {
                let l = Self::eval_ast(l, x, y, z, ctx);
                let r = Self::eval_ast(r, x, y, z, ctx);
                match op {
                    BinOp::Add => l + r,
                    BinOp::Sub => l - r,
                    BinOp::Mul => l * r,
                    BinOp::Div => l / r,
                    BinOp::Pow => l.powf(r),
                }
            }
            Ast::UnaryOp(op, inner) => {
                let v = Self::eval_ast(inner, x, y, z, ctx);
                match op {
                    UnaryOp::Neg => -v,
                }
            }
            Ast::Call(func, args) => {
                let a: Vec<f32> = args
                    .iter()
                    .map(|a| Self::eval_ast(a, x, y, z, ctx))
                    .collect();
                match func {
                    Func::Sin => a[0].sin(),
                    Func::Cos => a[0].cos(),
                    Func::Tan => a[0].tan(),
                    Func::Sqrt => a[0].sqrt(),
                    Func::Abs => a[0].abs(),
                    Func::Floor => a[0].floor(),
                    Func::Ceil => a[0].ceil(),
                    Func::Fract => a[0].fract(),
                    Func::Min => a[0].min(a[1]),
                    Func::Max => a[0].max(a[1]),
                    Func::Clamp => a[0].clamp(a[1], a[2]),
                    Func::Lerp => a[0] + (a[1] - a[0]) * a[2],
                    Func::Noise => noise::perlin2(a[0], a[1]),
                }
            }
        }
    }
}

/// Implements Field for Expr (2D input).
impl crate::field::Field<Vec2, f32> for Expr {
    fn sample(&self, input: Vec2, ctx: &EvalContext) -> f32 {
        self.eval(input, ctx)
    }
}

/// Implements Field for Expr (3D input).
impl crate::field::Field<glam::Vec3, f32> for Expr {
    fn sample(&self, input: glam::Vec3, ctx: &EvalContext) -> f32 {
        self.eval3(input.x, input.y, input.z, ctx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_number() {
        let expr = Expr::parse("42").unwrap();
        let ctx = EvalContext::new();
        assert_eq!(expr.eval(Vec2::ZERO, &ctx), 42.0);
    }

    #[test]
    fn test_parse_float() {
        let expr = Expr::parse("1.234").unwrap();
        let ctx = EvalContext::new();
        assert!((expr.eval(Vec2::ZERO, &ctx) - 1.234).abs() < 0.001);
    }

    #[test]
    fn test_parse_variable() {
        let expr = Expr::parse("x").unwrap();
        let ctx = EvalContext::new();
        assert_eq!(expr.eval(Vec2::new(5.0, 0.0), &ctx), 5.0);
    }

    #[test]
    fn test_parse_add() {
        let expr = Expr::parse("1 + 2").unwrap();
        let ctx = EvalContext::new();
        assert_eq!(expr.eval(Vec2::ZERO, &ctx), 3.0);
    }

    #[test]
    fn test_parse_mul() {
        let expr = Expr::parse("3 * 4").unwrap();
        let ctx = EvalContext::new();
        assert_eq!(expr.eval(Vec2::ZERO, &ctx), 12.0);
    }

    #[test]
    fn test_precedence() {
        let expr = Expr::parse("2 + 3 * 4").unwrap();
        let ctx = EvalContext::new();
        assert_eq!(expr.eval(Vec2::ZERO, &ctx), 14.0);
    }

    #[test]
    fn test_parentheses() {
        let expr = Expr::parse("(2 + 3) * 4").unwrap();
        let ctx = EvalContext::new();
        assert_eq!(expr.eval(Vec2::ZERO, &ctx), 20.0);
    }

    #[test]
    fn test_negation() {
        let expr = Expr::parse("-5").unwrap();
        let ctx = EvalContext::new();
        assert_eq!(expr.eval(Vec2::ZERO, &ctx), -5.0);
    }

    #[test]
    fn test_power() {
        let expr = Expr::parse("2 ^ 3").unwrap();
        let ctx = EvalContext::new();
        assert_eq!(expr.eval(Vec2::ZERO, &ctx), 8.0);
    }

    #[test]
    fn test_function_sin() {
        let expr = Expr::parse("sin(0)").unwrap();
        let ctx = EvalContext::new();
        assert!(expr.eval(Vec2::ZERO, &ctx).abs() < 0.001);
    }

    #[test]
    fn test_function_sqrt() {
        let expr = Expr::parse("sqrt(16)").unwrap();
        let ctx = EvalContext::new();
        assert_eq!(expr.eval(Vec2::ZERO, &ctx), 4.0);
    }

    #[test]
    fn test_function_min_max() {
        let expr = Expr::parse("min(3, 7)").unwrap();
        let ctx = EvalContext::new();
        assert_eq!(expr.eval(Vec2::ZERO, &ctx), 3.0);

        let expr = Expr::parse("max(3, 7)").unwrap();
        assert_eq!(expr.eval(Vec2::ZERO, &ctx), 7.0);
    }

    #[test]
    fn test_function_clamp() {
        let expr = Expr::parse("clamp(5, 0, 3)").unwrap();
        let ctx = EvalContext::new();
        assert_eq!(expr.eval(Vec2::ZERO, &ctx), 3.0);
    }

    #[test]
    fn test_function_lerp() {
        let expr = Expr::parse("lerp(0, 10, 0.5)").unwrap();
        let ctx = EvalContext::new();
        assert_eq!(expr.eval(Vec2::ZERO, &ctx), 5.0);
    }

    #[test]
    fn test_complex_expression() {
        let expr = Expr::parse("sin(x * 3.14) + y / 2").unwrap();
        let ctx = EvalContext::new();
        let v = expr.eval(Vec2::new(0.5, 4.0), &ctx);
        // sin(0.5 * 3.14) + 4.0 / 2 = sin(1.57) + 2 ≈ 1 + 2 = 3
        assert!((v - 3.0).abs() < 0.1);
    }

    #[test]
    fn test_time_variable() {
        let expr = Expr::parse("time").unwrap();
        let ctx = EvalContext::new().with_time(5.0);
        assert_eq!(expr.eval(Vec2::ZERO, &ctx), 5.0);
    }

    #[test]
    fn test_pi_constant() {
        let expr = Expr::parse("pi").unwrap();
        let ctx = EvalContext::new();
        assert!((expr.eval(Vec2::ZERO, &ctx) - PI).abs() < 0.001);
    }

    #[test]
    fn test_noise_function() {
        let expr = Expr::parse("noise(x, y)").unwrap();
        let ctx = EvalContext::new();
        let v = expr.eval(Vec2::new(0.5, 0.5), &ctx);
        // Noise should return values in [0, 1] (our perlin is normalized)
        assert!((0.0..=1.0).contains(&v));
    }

    #[test]
    fn test_unknown_function() {
        let result = Expr::parse("unknown(1)");
        assert!(matches!(result, Err(ParseError::UnknownFunction(_))));
    }

    #[test]
    fn test_wrong_arg_count() {
        let result = Expr::parse("sin(1, 2)");
        assert!(matches!(result, Err(ParseError::WrongArgCount { .. })));
    }

    #[test]
    fn test_as_field() {
        use crate::field::Field;

        let expr = Expr::parse("x + y").unwrap();
        let ctx = EvalContext::new();
        let v: f32 = expr.sample(Vec2::new(3.0, 4.0), &ctx);
        assert_eq!(v, 7.0);
    }
}
