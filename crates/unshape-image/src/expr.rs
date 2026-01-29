use crate::ImageField;
use crate::colorspace::{
    hsl_to_rgb, hsv_to_rgb, hwb_to_rgb, lab_to_rgb, lch_to_rgb, oklab_to_rgb, oklch_to_rgb,
    rgb_to_hsl, rgb_to_hsv, rgb_to_hwb, rgb_to_lab, rgb_to_lch, rgb_to_oklab, rgb_to_oklch,
    rgb_to_ycbcr, ycbcr_to_rgb,
};
use crate::kernel::{MapPixels, RemapUv};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// A typed expression AST for UV coordinate remapping (Vec2 -> Vec2).
///
/// This is the expression language for the `remap_uv` primitive. Each variant
/// represents an operation that transforms UV coordinates.
///
/// # Design
///
/// Unlike raw closures, `UvExpr` is:
/// - **Serializable** - Save/load effect pipelines
/// - **Interpretable** - Direct CPU evaluation
/// - **Inspectable** - Debug and optimize transforms
/// - **Future JIT/GPU** - Will compile to Cranelift/WGSL when dew-linalg is ready
///
/// # Example
///
/// ```
/// use unshape_image::{ImageField, UvExpr, remap_uv};
///
/// let image = ImageField::from_raw(vec![[0.5, 0.5, 0.5, 1.0]; 16], 4, 4);
///
/// // Wave distortion: offset U by sin(V * frequency) * amplitude
/// let wave = UvExpr::Add(
///     Box::new(UvExpr::Uv),
///     Box::new(UvExpr::Vec2 {
///         x: Box::new(UvExpr::Mul(
///             Box::new(UvExpr::Constant(0.05)),
///             Box::new(UvExpr::Sin(Box::new(UvExpr::Mul(
///                 Box::new(UvExpr::V),
///                 Box::new(UvExpr::Constant(6.28 * 4.0)),
///             )))),
///         )),
///         y: Box::new(UvExpr::Constant(0.0)),
///     }),
/// );
///
/// let result = remap_uv(&image, &wave);
/// ```
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum UvExpr {
    // === Coordinates ===
    /// The input UV coordinate as Vec2.
    Uv,
    /// Just the U coordinate (x).
    U,
    /// Just the V coordinate (y).
    V,

    // === Constructors ===
    /// Construct a Vec2 from two scalar expressions.
    Vec2 { x: Box<UvExpr>, y: Box<UvExpr> },

    // === Literals ===
    /// A constant scalar value.
    Constant(f32),
    /// A constant Vec2 value.
    Constant2(f32, f32),

    // === Vec2 operations ===
    /// Component-wise addition of two Vec2 expressions.
    Add(Box<UvExpr>, Box<UvExpr>),
    /// Component-wise subtraction.
    Sub(Box<UvExpr>, Box<UvExpr>),
    /// Component-wise multiplication.
    Mul(Box<UvExpr>, Box<UvExpr>),
    /// Component-wise division.
    Div(Box<UvExpr>, Box<UvExpr>),
    /// Negate (flip sign).
    Neg(Box<UvExpr>),

    // === Scalar math functions (applied component-wise or to scalars) ===
    /// Sine.
    Sin(Box<UvExpr>),
    /// Cosine.
    Cos(Box<UvExpr>),
    /// Absolute value.
    Abs(Box<UvExpr>),
    /// Floor.
    Floor(Box<UvExpr>),
    /// Fractional part.
    Fract(Box<UvExpr>),
    /// Square root.
    Sqrt(Box<UvExpr>),
    /// Power.
    Pow(Box<UvExpr>, Box<UvExpr>),
    /// Minimum.
    Min(Box<UvExpr>, Box<UvExpr>),
    /// Maximum.
    Max(Box<UvExpr>, Box<UvExpr>),
    /// Clamp to range.
    Clamp {
        value: Box<UvExpr>,
        min: Box<UvExpr>,
        max: Box<UvExpr>,
    },
    /// Linear interpolation.
    Lerp {
        a: Box<UvExpr>,
        b: Box<UvExpr>,
        t: Box<UvExpr>,
    },

    // === Vec2-specific operations ===
    /// Length of the vector.
    Length(Box<UvExpr>),
    /// Normalize to unit vector.
    Normalize(Box<UvExpr>),
    /// Dot product.
    Dot(Box<UvExpr>, Box<UvExpr>),
    /// Distance between two points.
    Distance(Box<UvExpr>, Box<UvExpr>),

    // === Common UV transforms (sugar for complex expressions) ===
    /// Rotate around a center point by angle (radians).
    Rotate {
        center: Box<UvExpr>,
        angle: Box<UvExpr>,
    },
    /// Scale around a center point.
    Scale {
        center: Box<UvExpr>,
        scale: Box<UvExpr>,
    },
}

impl UvExpr {
    /// Evaluate the expression at the given UV coordinate.
    ///
    /// Returns the transformed UV as (u', v').
    pub fn eval(&self, u: f32, v: f32) -> (f32, f32) {
        match self {
            // Coordinates
            Self::Uv => (u, v),
            Self::U => (u, u), // scalar → vec2 broadcast
            Self::V => (v, v),

            // Constructors
            Self::Vec2 { x, y } => {
                let (xu, _) = x.eval(u, v);
                let (yu, _) = y.eval(u, v);
                (xu, yu)
            }

            // Literals
            Self::Constant(c) => (*c, *c),
            Self::Constant2(x, y) => (*x, *y),

            // Vec2 operations
            Self::Add(a, b) => {
                let (au, av) = a.eval(u, v);
                let (bu, bv) = b.eval(u, v);
                (au + bu, av + bv)
            }
            Self::Sub(a, b) => {
                let (au, av) = a.eval(u, v);
                let (bu, bv) = b.eval(u, v);
                (au - bu, av - bv)
            }
            Self::Mul(a, b) => {
                let (au, av) = a.eval(u, v);
                let (bu, bv) = b.eval(u, v);
                (au * bu, av * bv)
            }
            Self::Div(a, b) => {
                let (au, av) = a.eval(u, v);
                let (bu, bv) = b.eval(u, v);
                (au / bu, av / bv)
            }
            Self::Neg(a) => {
                let (au, av) = a.eval(u, v);
                (-au, -av)
            }

            // Scalar math
            Self::Sin(a) => {
                let (au, av) = a.eval(u, v);
                (au.sin(), av.sin())
            }
            Self::Cos(a) => {
                let (au, av) = a.eval(u, v);
                (au.cos(), av.cos())
            }
            Self::Abs(a) => {
                let (au, av) = a.eval(u, v);
                (au.abs(), av.abs())
            }
            Self::Floor(a) => {
                let (au, av) = a.eval(u, v);
                (au.floor(), av.floor())
            }
            Self::Fract(a) => {
                let (au, av) = a.eval(u, v);
                (au.fract(), av.fract())
            }
            Self::Sqrt(a) => {
                let (au, av) = a.eval(u, v);
                (au.sqrt(), av.sqrt())
            }
            Self::Pow(a, b) => {
                let (au, av) = a.eval(u, v);
                let (bu, bv) = b.eval(u, v);
                (au.powf(bu), av.powf(bv))
            }
            Self::Min(a, b) => {
                let (au, av) = a.eval(u, v);
                let (bu, bv) = b.eval(u, v);
                (au.min(bu), av.min(bv))
            }
            Self::Max(a, b) => {
                let (au, av) = a.eval(u, v);
                let (bu, bv) = b.eval(u, v);
                (au.max(bu), av.max(bv))
            }
            Self::Clamp { value, min, max } => {
                let (vu, vv) = value.eval(u, v);
                let (minu, minv) = min.eval(u, v);
                let (maxu, maxv) = max.eval(u, v);
                (vu.clamp(minu, maxu), vv.clamp(minv, maxv))
            }
            Self::Lerp { a, b, t } => {
                let (au, av) = a.eval(u, v);
                let (bu, bv) = b.eval(u, v);
                let (tu, tv) = t.eval(u, v);
                (au + (bu - au) * tu, av + (bv - av) * tv)
            }

            // Vec2-specific
            Self::Length(a) => {
                let (au, av) = a.eval(u, v);
                let len = (au * au + av * av).sqrt();
                (len, len)
            }
            Self::Normalize(a) => {
                let (au, av) = a.eval(u, v);
                let len = (au * au + av * av).sqrt();
                if len > 0.0 {
                    (au / len, av / len)
                } else {
                    (0.0, 0.0)
                }
            }
            Self::Dot(a, b) => {
                let (au, av) = a.eval(u, v);
                let (bu, bv) = b.eval(u, v);
                let d = au * bu + av * bv;
                (d, d)
            }
            Self::Distance(a, b) => {
                let (au, av) = a.eval(u, v);
                let (bu, bv) = b.eval(u, v);
                let dx = au - bu;
                let dy = av - bv;
                let d = (dx * dx + dy * dy).sqrt();
                (d, d)
            }

            // Common transforms
            Self::Rotate { center, angle } => {
                let (cx, cy) = center.eval(u, v);
                let (angle_val, _) = angle.eval(u, v);
                let dx = u - cx;
                let dy = v - cy;
                let cos_a = angle_val.cos();
                let sin_a = angle_val.sin();
                (cx + dx * cos_a - dy * sin_a, cy + dx * sin_a + dy * cos_a)
            }
            Self::Scale { center, scale } => {
                let (cx, cy) = center.eval(u, v);
                let (sx, sy) = scale.eval(u, v);
                (cx + (u - cx) * sx, cy + (v - cy) * sy)
            }
        }
    }

    /// Creates an identity transform (returns UV unchanged).
    pub fn identity() -> Self {
        Self::Uv
    }

    /// Creates a translation transform.
    pub fn translate(offset_x: f32, offset_y: f32) -> Self {
        Self::Add(
            Box::new(Self::Uv),
            Box::new(Self::Constant2(offset_x, offset_y)),
        )
    }

    /// Creates a scale transform around the center (0.5, 0.5).
    pub fn scale_centered(scale_x: f32, scale_y: f32) -> Self {
        Self::Scale {
            center: Box::new(Self::Constant2(0.5, 0.5)),
            scale: Box::new(Self::Constant2(scale_x, scale_y)),
        }
    }

    /// Creates a rotation transform around the center (0.5, 0.5).
    pub fn rotate_centered(angle: f32) -> Self {
        Self::Rotate {
            center: Box::new(Self::Constant2(0.5, 0.5)),
            angle: Box::new(Self::Constant(angle)),
        }
    }

    /// Converts this expression to a Dew AST for JIT/WGSL compilation.
    ///
    /// The resulting AST expects a `uv` Vec2 variable and returns Vec2.
    /// Use with `dew-linalg` for evaluation or compilation.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use rhizome_dew_linalg::{Value, eval, linalg_registry};
    /// use std::collections::HashMap;
    ///
    /// let expr = UvExpr::translate(0.1, 0.0);
    /// let ast = expr.to_dew_ast();
    ///
    /// let mut vars = HashMap::new();
    /// vars.insert("uv".into(), Value::Vec2([0.5, 0.5]));
    ///
    /// let result = eval(&ast, &vars, &linalg_registry()).unwrap();
    /// // result = Value::Vec2([0.6, 0.5])
    /// ```
    #[cfg(feature = "dew")]
    pub fn to_dew_ast(&self) -> rhizome_dew_core::Ast {
        use rhizome_dew_core::{Ast, BinOp, UnaryOp};

        match self {
            // Coordinates - uv is a Vec2 variable
            Self::Uv => Ast::Var("uv".into()),
            // U and V need component extraction - use helper functions
            Self::U => Ast::Call("x".into(), vec![Ast::Var("uv".into())]),
            Self::V => Ast::Call("y".into(), vec![Ast::Var("uv".into())]),

            // Constructors
            Self::Vec2 { x, y } => Ast::Call("vec2".into(), vec![x.to_dew_ast(), y.to_dew_ast()]),

            // Literals
            Self::Constant(c) => Ast::Num(*c as f64),
            Self::Constant2(x, y) => Ast::Call(
                "vec2".into(),
                vec![Ast::Num(*x as f64), Ast::Num(*y as f64)],
            ),

            // Binary operations
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
            Self::Neg(a) => Ast::UnaryOp(UnaryOp::Neg, Box::new(a.to_dew_ast())),

            // Math functions
            Self::Sin(a) => Ast::Call("sin".into(), vec![a.to_dew_ast()]),
            Self::Cos(a) => Ast::Call("cos".into(), vec![a.to_dew_ast()]),
            Self::Abs(a) => Ast::Call("abs".into(), vec![a.to_dew_ast()]),
            Self::Floor(a) => Ast::Call("floor".into(), vec![a.to_dew_ast()]),
            Self::Fract(a) => Ast::Call("fract".into(), vec![a.to_dew_ast()]),
            Self::Sqrt(a) => Ast::Call("sqrt".into(), vec![a.to_dew_ast()]),
            Self::Pow(a, b) => Ast::BinOp(
                BinOp::Pow,
                Box::new(a.to_dew_ast()),
                Box::new(b.to_dew_ast()),
            ),
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

            // Vec2-specific operations
            Self::Length(a) => Ast::Call("length".into(), vec![a.to_dew_ast()]),
            Self::Normalize(a) => Ast::Call("normalize".into(), vec![a.to_dew_ast()]),
            Self::Dot(a, b) => Ast::Call("dot".into(), vec![a.to_dew_ast(), b.to_dew_ast()]),
            Self::Distance(a, b) => {
                Ast::Call("distance".into(), vec![a.to_dew_ast(), b.to_dew_ast()])
            }

            // Transforms - these expand to their mathematical equivalents
            Self::Rotate { center, angle } => {
                // center + rotate_vec(uv - center, angle)
                // where rotate_vec(v, a) = vec2(v.x * cos(a) - v.y * sin(a), v.x * sin(a) + v.y * cos(a))
                let c = center.to_dew_ast();
                let a = angle.to_dew_ast();
                let delta = Ast::BinOp(
                    BinOp::Sub,
                    Box::new(Ast::Var("uv".into())),
                    Box::new(c.clone()),
                );
                // For now, use a rotate2d function that backends should implement
                let rotated = Ast::Call("rotate2d".into(), vec![delta, a]);
                Ast::BinOp(BinOp::Add, Box::new(c), Box::new(rotated))
            }
            Self::Scale { center, scale } => {
                // center + (uv - center) * scale
                let c = center.to_dew_ast();
                let s = scale.to_dew_ast();
                let delta = Ast::BinOp(
                    BinOp::Sub,
                    Box::new(Ast::Var("uv".into())),
                    Box::new(c.clone()),
                );
                let scaled = Ast::BinOp(BinOp::Mul, Box::new(delta), Box::new(s));
                Ast::BinOp(BinOp::Add, Box::new(c), Box::new(scaled))
            }
        }
    }
}

/// A typed expression AST for per-pixel color transforms (Vec4 → Vec4).
///
/// This is the expression language for the `map_pixels` primitive. Each variant
/// represents an operation that transforms RGBA color values.
///
/// # Design
///
/// Unlike raw closures, `ColorExpr` is:
/// - **Serializable** - Save/load effect pipelines
/// - **Interpretable** - Direct CPU evaluation
/// - **Inspectable** - Debug and optimize transforms
/// - **Future JIT/GPU** - Will compile to Cranelift/WGSL when dew-linalg is ready
///
/// # Example
///
/// ```
/// use unshape_image::{ImageField, ColorExpr, map_pixels};
///
/// let image = ImageField::from_raw(vec![[0.5, 0.3, 0.7, 1.0]; 16], 4, 4);
///
/// // Grayscale: luminance weighted average
/// let grayscale = ColorExpr::grayscale();
/// let result = map_pixels(&image, &grayscale);
/// ```
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum ColorExpr {
    // === Input ===
    /// The input RGBA color as Vec4.
    Rgba,
    /// Just the red channel.
    R,
    /// Just the green channel.
    G,
    /// Just the blue channel.
    B,
    /// Just the alpha channel.
    A,
    /// Computed luminance (0.2126*R + 0.7152*G + 0.0722*B).
    Luminance,

    // === Constructors ===
    /// Construct RGBA from four scalar expressions.
    Vec4 {
        r: Box<ColorExpr>,
        g: Box<ColorExpr>,
        b: Box<ColorExpr>,
        a: Box<ColorExpr>,
    },
    /// Construct RGB with explicit alpha.
    Vec3A {
        r: Box<ColorExpr>,
        g: Box<ColorExpr>,
        b: Box<ColorExpr>,
        a: Box<ColorExpr>,
    },

    // === Literals ===
    /// A constant scalar value (broadcasts to all channels).
    Constant(f32),
    /// A constant RGBA value.
    Constant4(f32, f32, f32, f32),

    // === Arithmetic ===
    /// Component-wise addition.
    Add(Box<ColorExpr>, Box<ColorExpr>),
    /// Component-wise subtraction.
    Sub(Box<ColorExpr>, Box<ColorExpr>),
    /// Component-wise multiplication.
    Mul(Box<ColorExpr>, Box<ColorExpr>),
    /// Component-wise division.
    Div(Box<ColorExpr>, Box<ColorExpr>),
    /// Negate.
    Neg(Box<ColorExpr>),

    // === Math functions ===
    /// Absolute value.
    Abs(Box<ColorExpr>),
    /// Floor.
    Floor(Box<ColorExpr>),
    /// Fractional part.
    Fract(Box<ColorExpr>),
    /// Square root.
    Sqrt(Box<ColorExpr>),
    /// Power.
    Pow(Box<ColorExpr>, Box<ColorExpr>),
    /// Minimum.
    Min(Box<ColorExpr>, Box<ColorExpr>),
    /// Maximum.
    Max(Box<ColorExpr>, Box<ColorExpr>),
    /// Clamp to range.
    Clamp {
        value: Box<ColorExpr>,
        min: Box<ColorExpr>,
        max: Box<ColorExpr>,
    },
    /// Linear interpolation (mix).
    Lerp {
        a: Box<ColorExpr>,
        b: Box<ColorExpr>,
        t: Box<ColorExpr>,
    },
    /// Smooth step.
    SmoothStep {
        edge0: Box<ColorExpr>,
        edge1: Box<ColorExpr>,
        x: Box<ColorExpr>,
    },
    /// Step function.
    Step {
        edge: Box<ColorExpr>,
        x: Box<ColorExpr>,
    },

    // === Conditionals ===
    /// If-then-else based on comparison > 0.5.
    IfThenElse {
        condition: Box<ColorExpr>,
        then_expr: Box<ColorExpr>,
        else_expr: Box<ColorExpr>,
    },
    /// Greater than (returns 1.0 or 0.0).
    Gt(Box<ColorExpr>, Box<ColorExpr>),
    /// Less than (returns 1.0 or 0.0).
    Lt(Box<ColorExpr>, Box<ColorExpr>),

    // === Colorspace conversions ===
    // These operate on RGB channels, preserving alpha.
    // Input: vec4 RGBA, Output: vec4 where RGB is converted, A is preserved.
    /// Convert RGB to HSL (Hue, Saturation, Lightness).
    RgbToHsl(Box<ColorExpr>),
    /// Convert HSL to RGB.
    HslToRgb(Box<ColorExpr>),
    /// Convert RGB to HSV (Hue, Saturation, Value).
    RgbToHsv(Box<ColorExpr>),
    /// Convert HSV to RGB.
    HsvToRgb(Box<ColorExpr>),
    /// Convert RGB to HWB (Hue, Whiteness, Blackness).
    RgbToHwb(Box<ColorExpr>),
    /// Convert HWB to RGB.
    HwbToRgb(Box<ColorExpr>),
    /// Convert RGB to CIE LAB (Lightness, a*, b*).
    RgbToLab(Box<ColorExpr>),
    /// Convert CIE LAB to RGB.
    LabToRgb(Box<ColorExpr>),
    /// Convert RGB to LCH (Lightness, Chroma, Hue) - cylindrical LAB.
    RgbToLch(Box<ColorExpr>),
    /// Convert LCH to RGB.
    LchToRgb(Box<ColorExpr>),
    /// Convert RGB to OkLab (perceptually uniform).
    RgbToOklab(Box<ColorExpr>),
    /// Convert OkLab to RGB.
    OklabToRgb(Box<ColorExpr>),
    /// Convert RGB to OkLCH (cylindrical OkLab).
    RgbToOklch(Box<ColorExpr>),
    /// Convert OkLCH to RGB.
    OklchToRgb(Box<ColorExpr>),
    /// Convert RGB to YCbCr (luma, chroma).
    RgbToYcbcr(Box<ColorExpr>),
    /// Convert YCbCr to RGB.
    YcbcrToRgb(Box<ColorExpr>),

    // === HSL/HSV adjustments ===
    // These perform colorspace conversion, adjustment, and conversion back in one step.
    /// Adjust hue, saturation, and lightness.
    ///
    /// This is more efficient than manual RGB→HSL→adjust→RGB because it's
    /// a single operation that can be optimized.
    ///
    /// - `hue_shift`: Hue adjustment in [0, 1] range (wraps around)
    /// - `saturation`: Saturation multiplier (1.0 = no change, 0.0 = grayscale)
    /// - `lightness`: Lightness offset (-1 to +1)
    AdjustHsl {
        input: Box<ColorExpr>,
        hue_shift: f32,
        saturation: f32,
        lightness: f32,
    },

    /// Adjust hue, saturation, and value.
    ///
    /// - `hue_shift`: Hue adjustment in [0, 1] range (wraps around)
    /// - `saturation`: Saturation multiplier (1.0 = no change)
    /// - `value`: Value multiplier (1.0 = no change)
    AdjustHsv {
        input: Box<ColorExpr>,
        hue_shift: f32,
        saturation: f32,
        value: f32,
    },

    /// Adjust brightness and contrast.
    ///
    /// - `brightness`: Additive brightness offset (-1 to +1, 0 = no change)
    /// - `contrast`: Contrast adjustment (-1 to +1, 0 = no change)
    ///
    /// Formula: `result = (value - 0.5) * (1 + contrast) + 0.5 + brightness`
    AdjustBrightnessContrast {
        input: Box<ColorExpr>,
        brightness: f32,
        contrast: f32,
    },

    /// Apply a 4x4 color matrix transform.
    ///
    /// The matrix transforms RGBA values: `[r', g', b', a'] = matrix * [r, g, b, a]`.
    /// Row-major layout: `matrix[row][col]`.
    Matrix {
        input: Box<ColorExpr>,
        /// Row-major 4x4 matrix.
        matrix: [[f32; 4]; 4],
    },
}

impl ColorExpr {
    /// Evaluate the expression for the given RGBA color.
    ///
    /// Returns the transformed color as [r', g', b', a'].
    pub fn eval(&self, r: f32, g: f32, b: f32, a: f32) -> [f32; 4] {
        match self {
            // Input
            Self::Rgba => [r, g, b, a],
            Self::R => [r, r, r, r],
            Self::G => [g, g, g, g],
            Self::B => [b, b, b, b],
            Self::A => [a, a, a, a],
            Self::Luminance => {
                let lum = 0.2126 * r + 0.7152 * g + 0.0722 * b;
                [lum, lum, lum, lum]
            }

            // Constructors
            Self::Vec4 {
                r: er,
                g: eg,
                b: eb,
                a: ea,
            } => {
                let [rv, _, _, _] = er.eval(r, g, b, a);
                let [gv, _, _, _] = eg.eval(r, g, b, a);
                let [bv, _, _, _] = eb.eval(r, g, b, a);
                let [av, _, _, _] = ea.eval(r, g, b, a);
                [rv, gv, bv, av]
            }
            Self::Vec3A {
                r: er,
                g: eg,
                b: eb,
                a: ea,
            } => {
                let [rv, _, _, _] = er.eval(r, g, b, a);
                let [gv, _, _, _] = eg.eval(r, g, b, a);
                let [bv, _, _, _] = eb.eval(r, g, b, a);
                let [av, _, _, _] = ea.eval(r, g, b, a);
                [rv, gv, bv, av]
            }

            // Literals
            Self::Constant(c) => [*c, *c, *c, *c],
            Self::Constant4(cr, cg, cb, ca) => [*cr, *cg, *cb, *ca],

            // Arithmetic
            Self::Add(ea, eb) => {
                let [ar, ag, ab, aa] = ea.eval(r, g, b, a);
                let [br, bg, bb, ba] = eb.eval(r, g, b, a);
                [ar + br, ag + bg, ab + bb, aa + ba]
            }
            Self::Sub(ea, eb) => {
                let [ar, ag, ab, aa] = ea.eval(r, g, b, a);
                let [br, bg, bb, ba] = eb.eval(r, g, b, a);
                [ar - br, ag - bg, ab - bb, aa - ba]
            }
            Self::Mul(ea, eb) => {
                let [ar, ag, ab, aa] = ea.eval(r, g, b, a);
                let [br, bg, bb, ba] = eb.eval(r, g, b, a);
                [ar * br, ag * bg, ab * bb, aa * ba]
            }
            Self::Div(ea, eb) => {
                let [ar, ag, ab, aa] = ea.eval(r, g, b, a);
                let [br, bg, bb, ba] = eb.eval(r, g, b, a);
                [ar / br, ag / bg, ab / bb, aa / ba]
            }
            Self::Neg(e) => {
                let [er, eg, eb, ea] = e.eval(r, g, b, a);
                [-er, -eg, -eb, -ea]
            }

            // Math functions
            Self::Abs(e) => {
                let [er, eg, eb, ea] = e.eval(r, g, b, a);
                [er.abs(), eg.abs(), eb.abs(), ea.abs()]
            }
            Self::Floor(e) => {
                let [er, eg, eb, ea] = e.eval(r, g, b, a);
                [er.floor(), eg.floor(), eb.floor(), ea.floor()]
            }
            Self::Fract(e) => {
                let [er, eg, eb, ea] = e.eval(r, g, b, a);
                [er.fract(), eg.fract(), eb.fract(), ea.fract()]
            }
            Self::Sqrt(e) => {
                let [er, eg, eb, ea] = e.eval(r, g, b, a);
                [er.sqrt(), eg.sqrt(), eb.sqrt(), ea.sqrt()]
            }
            Self::Pow(ea, eb) => {
                let [ar, ag, ab, aa] = ea.eval(r, g, b, a);
                let [br, bg, bb, ba] = eb.eval(r, g, b, a);
                [ar.powf(br), ag.powf(bg), ab.powf(bb), aa.powf(ba)]
            }
            Self::Min(ea, eb) => {
                let [ar, ag, ab, aa] = ea.eval(r, g, b, a);
                let [br, bg, bb, ba] = eb.eval(r, g, b, a);
                [ar.min(br), ag.min(bg), ab.min(bb), aa.min(ba)]
            }
            Self::Max(ea, eb) => {
                let [ar, ag, ab, aa] = ea.eval(r, g, b, a);
                let [br, bg, bb, ba] = eb.eval(r, g, b, a);
                [ar.max(br), ag.max(bg), ab.max(bb), aa.max(ba)]
            }
            Self::Clamp { value, min, max } => {
                let [vr, vg, vb, va] = value.eval(r, g, b, a);
                let [minr, ming, minb, mina] = min.eval(r, g, b, a);
                let [maxr, maxg, maxb, maxa] = max.eval(r, g, b, a);
                [
                    vr.clamp(minr, maxr),
                    vg.clamp(ming, maxg),
                    vb.clamp(minb, maxb),
                    va.clamp(mina, maxa),
                ]
            }
            Self::Lerp {
                a: ea,
                b: eb,
                t: et,
            } => {
                let [ar, ag, ab, aa] = ea.eval(r, g, b, a);
                let [br, bg, bb, ba] = eb.eval(r, g, b, a);
                let [tr, tg, tb, ta] = et.eval(r, g, b, a);
                [
                    ar + (br - ar) * tr,
                    ag + (bg - ag) * tg,
                    ab + (bb - ab) * tb,
                    aa + (ba - aa) * ta,
                ]
            }
            Self::SmoothStep { edge0, edge1, x } => {
                let [e0r, e0g, e0b, e0a] = edge0.eval(r, g, b, a);
                let [e1r, e1g, e1b, e1a] = edge1.eval(r, g, b, a);
                let [xr, xg, xb, xa] = x.eval(r, g, b, a);

                let smooth = |x: f32, e0: f32, e1: f32| {
                    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
                    t * t * (3.0 - 2.0 * t)
                };

                [
                    smooth(xr, e0r, e1r),
                    smooth(xg, e0g, e1g),
                    smooth(xb, e0b, e1b),
                    smooth(xa, e0a, e1a),
                ]
            }
            Self::Step { edge, x } => {
                let [er, eg, eb, ea] = edge.eval(r, g, b, a);
                let [xr, xg, xb, xa] = x.eval(r, g, b, a);
                [
                    if xr < er { 0.0 } else { 1.0 },
                    if xg < eg { 0.0 } else { 1.0 },
                    if xb < eb { 0.0 } else { 1.0 },
                    if xa < ea { 0.0 } else { 1.0 },
                ]
            }

            // Conditionals
            Self::IfThenElse {
                condition,
                then_expr,
                else_expr,
            } => {
                let [cr, cg, cb, ca] = condition.eval(r, g, b, a);
                let [tr, tg, tb, ta] = then_expr.eval(r, g, b, a);
                let [er, eg, eb, ea] = else_expr.eval(r, g, b, a);
                [
                    if cr > 0.5 { tr } else { er },
                    if cg > 0.5 { tg } else { eg },
                    if cb > 0.5 { tb } else { eb },
                    if ca > 0.5 { ta } else { ea },
                ]
            }
            Self::Gt(ea, eb) => {
                let [ar, ag, ab, aa] = ea.eval(r, g, b, a);
                let [br, bg, bb, ba] = eb.eval(r, g, b, a);
                [
                    if ar > br { 1.0 } else { 0.0 },
                    if ag > bg { 1.0 } else { 0.0 },
                    if ab > bb { 1.0 } else { 0.0 },
                    if aa > ba { 1.0 } else { 0.0 },
                ]
            }
            Self::Lt(ea, eb) => {
                let [ar, ag, ab, aa] = ea.eval(r, g, b, a);
                let [br, bg, bb, ba] = eb.eval(r, g, b, a);
                [
                    if ar < br { 1.0 } else { 0.0 },
                    if ag < bg { 1.0 } else { 0.0 },
                    if ab < bb { 1.0 } else { 0.0 },
                    if aa < ba { 1.0 } else { 0.0 },
                ]
            }

            // Colorspace conversions
            Self::RgbToHsl(e) => {
                let [er, eg, eb, ea] = e.eval(r, g, b, a);
                let (h, s, l) = rgb_to_hsl(er, eg, eb);
                [h, s, l, ea]
            }
            Self::HslToRgb(e) => {
                let [eh, es, el, ea] = e.eval(r, g, b, a);
                let (r, g, b) = hsl_to_rgb(eh, es, el);
                [r, g, b, ea]
            }
            Self::RgbToHsv(e) => {
                let [er, eg, eb, ea] = e.eval(r, g, b, a);
                let (h, s, v) = rgb_to_hsv(er, eg, eb);
                [h, s, v, ea]
            }
            Self::HsvToRgb(e) => {
                let [eh, es, ev, ea] = e.eval(r, g, b, a);
                let (r, g, b) = hsv_to_rgb(eh, es, ev);
                [r, g, b, ea]
            }
            Self::RgbToHwb(e) => {
                let [er, eg, eb, ea] = e.eval(r, g, b, a);
                let (h, w, b_val) = rgb_to_hwb(er, eg, eb);
                [h, w, b_val, ea]
            }
            Self::HwbToRgb(e) => {
                let [eh, ew, eb_val, ea] = e.eval(r, g, b, a);
                let (r, g, b) = hwb_to_rgb(eh, ew, eb_val);
                [r, g, b, ea]
            }
            Self::RgbToLab(e) => {
                let [er, eg, eb, ea] = e.eval(r, g, b, a);
                let (l, a_val, b_val) = rgb_to_lab(er, eg, eb);
                [l, a_val, b_val, ea]
            }
            Self::LabToRgb(e) => {
                let [el, ea_val, eb_val, ea] = e.eval(r, g, b, a);
                let (r, g, b) = lab_to_rgb(el, ea_val, eb_val);
                [r, g, b, ea]
            }
            Self::RgbToLch(e) => {
                let [er, eg, eb, ea] = e.eval(r, g, b, a);
                let (l, c, h) = rgb_to_lch(er, eg, eb);
                [l, c, h, ea]
            }
            Self::LchToRgb(e) => {
                let [el, ec, eh, ea] = e.eval(r, g, b, a);
                let (r, g, b) = lch_to_rgb(el, ec, eh);
                [r, g, b, ea]
            }
            Self::RgbToOklab(e) => {
                let [er, eg, eb, ea] = e.eval(r, g, b, a);
                let (l, a_val, b_val) = rgb_to_oklab(er, eg, eb);
                [l, a_val, b_val, ea]
            }
            Self::OklabToRgb(e) => {
                let [el, ea_val, eb_val, ea] = e.eval(r, g, b, a);
                let (r, g, b) = oklab_to_rgb(el, ea_val, eb_val);
                [r, g, b, ea]
            }
            Self::RgbToOklch(e) => {
                let [er, eg, eb, ea] = e.eval(r, g, b, a);
                let (l, c, h) = rgb_to_oklch(er, eg, eb);
                [l, c, h, ea]
            }
            Self::OklchToRgb(e) => {
                let [el, ec, eh, ea] = e.eval(r, g, b, a);
                let (r, g, b) = oklch_to_rgb(el, ec, eh);
                [r, g, b, ea]
            }
            Self::RgbToYcbcr(e) => {
                let [er, eg, eb, ea] = e.eval(r, g, b, a);
                let (y, cb, cr) = rgb_to_ycbcr(er, eg, eb);
                [y, cb, cr, ea]
            }
            Self::YcbcrToRgb(e) => {
                let [ey, ecb, ecr, ea] = e.eval(r, g, b, a);
                let (r, g, b) = ycbcr_to_rgb(ey, ecb, ecr);
                [r, g, b, ea]
            }

            // HSL/HSV adjustments
            Self::AdjustHsl {
                input,
                hue_shift,
                saturation,
                lightness,
            } => {
                let [ir, ig, ib, ia] = input.eval(r, g, b, a);
                let (h, s, l) = rgb_to_hsl(ir, ig, ib);
                let new_h = (h + hue_shift).rem_euclid(1.0);
                let new_s = (s * saturation).clamp(0.0, 1.0);
                let new_l = (l + lightness).clamp(0.0, 1.0);
                let (nr, ng, nb) = hsl_to_rgb(new_h, new_s, new_l);
                [nr, ng, nb, ia]
            }
            Self::AdjustHsv {
                input,
                hue_shift,
                saturation,
                value,
            } => {
                let [ir, ig, ib, ia] = input.eval(r, g, b, a);
                let (h, s, v) = rgb_to_hsv(ir, ig, ib);
                let new_h = (h + hue_shift).rem_euclid(1.0);
                let new_s = (s * saturation).clamp(0.0, 1.0);
                let new_v = (v * value).clamp(0.0, 1.0);
                let (nr, ng, nb) = hsv_to_rgb(new_h, new_s, new_v);
                [nr, ng, nb, ia]
            }

            // Brightness/contrast adjustment
            Self::AdjustBrightnessContrast {
                input,
                brightness,
                contrast,
            } => {
                let [ir, ig, ib, ia] = input.eval(r, g, b, a);
                // Convert contrast to multiplier: 0 = 1x, 1 = 2x, -1 = 0x
                let contrast_factor = (1.0 + contrast).max(0.0);

                let adjust = |v: f32| -> f32 {
                    // Apply contrast around midpoint, then brightness
                    let contrasted = (v - 0.5) * contrast_factor + 0.5;
                    (contrasted + brightness).clamp(0.0, 1.0)
                };

                [adjust(ir), adjust(ig), adjust(ib), ia]
            }

            // Matrix transform
            Self::Matrix { input, matrix } => {
                let [ir, ig, ib, ia] = input.eval(r, g, b, a);
                let m = matrix;
                [
                    m[0][0] * ir + m[0][1] * ig + m[0][2] * ib + m[0][3] * ia,
                    m[1][0] * ir + m[1][1] * ig + m[1][2] * ib + m[1][3] * ia,
                    m[2][0] * ir + m[2][1] * ig + m[2][2] * ib + m[2][3] * ia,
                    m[3][0] * ir + m[3][1] * ig + m[3][2] * ib + m[3][3] * ia,
                ]
            }
        }
    }

    /// Creates an identity transform (returns RGBA unchanged).
    pub fn identity() -> Self {
        Self::Rgba
    }

    /// Creates a grayscale transform using ITU-R BT.709 luminance.
    pub fn grayscale() -> Self {
        // luminance, luminance, luminance, alpha
        Self::Vec4 {
            r: Box::new(Self::Luminance),
            g: Box::new(Self::Luminance),
            b: Box::new(Self::Luminance),
            a: Box::new(Self::A),
        }
    }

    /// Creates an invert transform (1 - RGB, preserves alpha).
    pub fn invert() -> Self {
        Self::Vec4 {
            r: Box::new(Self::Sub(Box::new(Self::Constant(1.0)), Box::new(Self::R))),
            g: Box::new(Self::Sub(Box::new(Self::Constant(1.0)), Box::new(Self::G))),
            b: Box::new(Self::Sub(Box::new(Self::Constant(1.0)), Box::new(Self::B))),
            a: Box::new(Self::A),
        }
    }

    /// Creates a threshold transform (luminance > threshold ? white : black).
    pub fn threshold(threshold: f32) -> Self {
        let lum_gt_threshold = Self::Gt(
            Box::new(Self::Luminance),
            Box::new(Self::Constant(threshold)),
        );
        Self::Vec4 {
            r: Box::new(lum_gt_threshold.clone()),
            g: Box::new(lum_gt_threshold.clone()),
            b: Box::new(lum_gt_threshold),
            a: Box::new(Self::A),
        }
    }

    /// Creates a brightness adjustment (multiply RGB by factor).
    pub fn brightness(factor: f32) -> Self {
        Self::Vec4 {
            r: Box::new(Self::Mul(
                Box::new(Self::R),
                Box::new(Self::Constant(factor)),
            )),
            g: Box::new(Self::Mul(
                Box::new(Self::G),
                Box::new(Self::Constant(factor)),
            )),
            b: Box::new(Self::Mul(
                Box::new(Self::B),
                Box::new(Self::Constant(factor)),
            )),
            a: Box::new(Self::A),
        }
    }

    /// Creates a contrast adjustment (scale RGB around 0.5).
    pub fn contrast(factor: f32) -> Self {
        // (color - 0.5) * factor + 0.5
        let adjust = |channel: Self| {
            Self::Add(
                Box::new(Self::Mul(
                    Box::new(Self::Sub(Box::new(channel), Box::new(Self::Constant(0.5)))),
                    Box::new(Self::Constant(factor)),
                )),
                Box::new(Self::Constant(0.5)),
            )
        };
        Self::Vec4 {
            r: Box::new(adjust(Self::R)),
            g: Box::new(adjust(Self::G)),
            b: Box::new(adjust(Self::B)),
            a: Box::new(Self::A),
        }
    }

    /// Creates a posterize effect (quantize to N levels).
    pub fn posterize(levels: u32) -> Self {
        let factor = (levels.max(2) - 1) as f32;
        // floor(color * factor) / factor
        let quantize = |channel: Self| {
            Self::Div(
                Box::new(Self::Floor(Box::new(Self::Mul(
                    Box::new(channel),
                    Box::new(Self::Constant(factor)),
                )))),
                Box::new(Self::Constant(factor)),
            )
        };
        Self::Vec4 {
            r: Box::new(quantize(Self::R)),
            g: Box::new(quantize(Self::G)),
            b: Box::new(quantize(Self::B)),
            a: Box::new(Self::A),
        }
    }

    /// Creates a gamma correction transform.
    pub fn gamma(gamma: f32) -> Self {
        let inv_gamma = 1.0 / gamma;
        Self::Vec4 {
            r: Box::new(Self::Pow(
                Box::new(Self::R),
                Box::new(Self::Constant(inv_gamma)),
            )),
            g: Box::new(Self::Pow(
                Box::new(Self::G),
                Box::new(Self::Constant(inv_gamma)),
            )),
            b: Box::new(Self::Pow(
                Box::new(Self::B),
                Box::new(Self::Constant(inv_gamma)),
            )),
            a: Box::new(Self::A),
        }
    }

    /// Creates a color tint (multiply by a color).
    pub fn tint(tint_r: f32, tint_g: f32, tint_b: f32) -> Self {
        Self::Vec4 {
            r: Box::new(Self::Mul(
                Box::new(Self::R),
                Box::new(Self::Constant(tint_r)),
            )),
            g: Box::new(Self::Mul(
                Box::new(Self::G),
                Box::new(Self::Constant(tint_g)),
            )),
            b: Box::new(Self::Mul(
                Box::new(Self::B),
                Box::new(Self::Constant(tint_b)),
            )),
            a: Box::new(Self::A),
        }
    }

    /// Creates an HSL adjustment expression.
    ///
    /// # Arguments
    ///
    /// * `hue_shift` - Hue shift in [0, 1] range (wraps around the color wheel)
    /// * `saturation` - Saturation multiplier (1.0 = no change, 0.0 = grayscale)
    /// * `lightness` - Lightness offset (-1 to +1)
    ///
    /// # Example
    ///
    /// ```
    /// use unshape_image::{ImageField, ColorExpr, map_pixels};
    ///
    /// let data = vec![[1.0, 0.0, 0.0, 1.0]; 4]; // Red
    /// let img = ImageField::from_raw(data, 2, 2);
    ///
    /// // Shift hue by 180 degrees (to cyan)
    /// let result = map_pixels(&img, &ColorExpr::hsl_adjust(0.5, 1.0, 0.0));
    /// ```
    pub fn hsl_adjust(hue_shift: f32, saturation: f32, lightness: f32) -> Self {
        Self::AdjustHsl {
            input: Box::new(Self::Rgba),
            hue_shift,
            saturation,
            lightness,
        }
    }

    /// Creates a hue shift expression.
    ///
    /// Shorthand for `hsl_adjust(hue_shift, 1.0, 0.0)`.
    pub fn hue_shift(amount: f32) -> Self {
        Self::hsl_adjust(amount, 1.0, 0.0)
    }

    /// Creates a saturation adjustment expression.
    ///
    /// Shorthand for `hsl_adjust(0.0, multiplier, 0.0)`.
    pub fn saturate(multiplier: f32) -> Self {
        Self::hsl_adjust(0.0, multiplier, 0.0)
    }

    /// Creates a lightness adjustment expression.
    ///
    /// Shorthand for `hsl_adjust(0.0, 1.0, offset)`.
    pub fn lighten(offset: f32) -> Self {
        Self::hsl_adjust(0.0, 1.0, offset)
    }

    /// Creates an HSV adjustment expression.
    ///
    /// # Arguments
    ///
    /// * `hue_shift` - Hue shift in [0, 1] range (wraps around)
    /// * `saturation` - Saturation multiplier (1.0 = no change)
    /// * `value` - Value multiplier (1.0 = no change)
    pub fn hsv_adjust(hue_shift: f32, saturation: f32, value: f32) -> Self {
        Self::AdjustHsv {
            input: Box::new(Self::Rgba),
            hue_shift,
            saturation,
            value,
        }
    }

    /// Creates a brightness/contrast adjustment expression.
    ///
    /// # Arguments
    ///
    /// * `brightness` - Additive brightness offset (-1 to +1, 0 = no change)
    /// * `contrast` - Contrast adjustment (-1 to +1, 0 = no change)
    ///
    /// Formula: `result = (value - 0.5) * (1 + contrast) + 0.5 + brightness`
    pub fn brightness_contrast(brightness: f32, contrast: f32) -> Self {
        Self::AdjustBrightnessContrast {
            input: Box::new(Self::Rgba),
            brightness,
            contrast,
        }
    }

    /// Creates a 4x4 color matrix transform expression.
    ///
    /// The matrix transforms RGBA values: `[r', g', b', a'] = matrix * [r, g, b, a]`.
    /// Row-major layout: `matrix[row][col]`.
    ///
    /// # Example
    ///
    /// ```
    /// use unshape_image::ColorExpr;
    ///
    /// // Grayscale matrix
    /// let gray = ColorExpr::matrix([
    ///     [0.299, 0.587, 0.114, 0.0],
    ///     [0.299, 0.587, 0.114, 0.0],
    ///     [0.299, 0.587, 0.114, 0.0],
    ///     [0.0,   0.0,   0.0,   1.0],
    /// ]);
    /// ```
    pub fn matrix(matrix: [[f32; 4]; 4]) -> Self {
        Self::Matrix {
            input: Box::new(Self::Rgba),
            matrix,
        }
    }

    /// Converts this expression to a Dew AST for JIT/WGSL compilation.
    ///
    /// The resulting AST expects an `rgba` Vec4 variable and returns Vec4.
    /// Use with `dew-linalg` for evaluation or compilation.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use rhizome_dew_linalg::{Value, eval, linalg_registry};
    /// use std::collections::HashMap;
    ///
    /// let expr = ColorExpr::grayscale();
    /// let ast = expr.to_dew_ast();
    ///
    /// let mut vars = HashMap::new();
    /// vars.insert("rgba".into(), Value::Vec4([1.0, 0.0, 0.0, 1.0]));
    ///
    /// let result = eval(&ast, &vars, &linalg_registry()).unwrap();
    /// // result = Value::Vec4([0.2126, 0.2126, 0.2126, 1.0])
    /// ```
    #[cfg(feature = "dew")]
    pub fn to_dew_ast(&self) -> rhizome_dew_core::Ast {
        use rhizome_dew_core::{Ast, BinOp, UnaryOp};

        match self {
            // Input - rgba is a Vec4 variable
            Self::Rgba => Ast::Var("rgba".into()),
            // Component extraction using x/y/z/w (or could use r/g/b/a if supported)
            Self::R => Ast::Call("x".into(), vec![Ast::Var("rgba".into())]),
            Self::G => Ast::Call("y".into(), vec![Ast::Var("rgba".into())]),
            Self::B => Ast::Call("z".into(), vec![Ast::Var("rgba".into())]),
            Self::A => Ast::Call("w".into(), vec![Ast::Var("rgba".into())]),
            // Luminance: 0.2126*R + 0.7152*G + 0.0722*B
            Self::Luminance => {
                let r = Ast::Call("x".into(), vec![Ast::Var("rgba".into())]);
                let g = Ast::Call("y".into(), vec![Ast::Var("rgba".into())]);
                let b = Ast::Call("z".into(), vec![Ast::Var("rgba".into())]);
                let term_r = Ast::BinOp(BinOp::Mul, Box::new(Ast::Num(0.2126)), Box::new(r));
                let term_g = Ast::BinOp(BinOp::Mul, Box::new(Ast::Num(0.7152)), Box::new(g));
                let term_b = Ast::BinOp(BinOp::Mul, Box::new(Ast::Num(0.0722)), Box::new(b));
                let sum_rg = Ast::BinOp(BinOp::Add, Box::new(term_r), Box::new(term_g));
                Ast::BinOp(BinOp::Add, Box::new(sum_rg), Box::new(term_b))
            }

            // Constructors
            Self::Vec4 { r, g, b, a } => Ast::Call(
                "vec4".into(),
                vec![
                    r.to_dew_ast(),
                    g.to_dew_ast(),
                    b.to_dew_ast(),
                    a.to_dew_ast(),
                ],
            ),
            Self::Vec3A { r, g, b, a } => {
                // Same as Vec4 - construct vec4 from components
                Ast::Call(
                    "vec4".into(),
                    vec![
                        r.to_dew_ast(),
                        g.to_dew_ast(),
                        b.to_dew_ast(),
                        a.to_dew_ast(),
                    ],
                )
            }

            // Literals
            Self::Constant(c) => Ast::Num(*c as f64),
            Self::Constant4(r, g, b, a) => Ast::Call(
                "vec4".into(),
                vec![
                    Ast::Num(*r as f64),
                    Ast::Num(*g as f64),
                    Ast::Num(*b as f64),
                    Ast::Num(*a as f64),
                ],
            ),

            // Binary operations
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
            Self::Neg(a) => Ast::UnaryOp(UnaryOp::Neg, Box::new(a.to_dew_ast())),

            // Math functions
            Self::Abs(a) => Ast::Call("abs".into(), vec![a.to_dew_ast()]),
            Self::Floor(a) => Ast::Call("floor".into(), vec![a.to_dew_ast()]),
            Self::Fract(a) => Ast::Call("fract".into(), vec![a.to_dew_ast()]),
            Self::Sqrt(a) => Ast::Call("sqrt".into(), vec![a.to_dew_ast()]),
            Self::Pow(a, b) => Ast::BinOp(
                BinOp::Pow,
                Box::new(a.to_dew_ast()),
                Box::new(b.to_dew_ast()),
            ),
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

            // Conditionals - these map to select/mix based on comparison
            Self::IfThenElse {
                condition,
                then_expr,
                else_expr,
            } => {
                // select(else, then, condition > 0.5)
                // or we can use: lerp(else, then, step(0.5, condition))
                let cond_gt_half =
                    Ast::Call("step".into(), vec![Ast::Num(0.5), condition.to_dew_ast()]);
                Ast::Call(
                    "lerp".into(),
                    vec![else_expr.to_dew_ast(), then_expr.to_dew_ast(), cond_gt_half],
                )
            }
            Self::Gt(a, b) => {
                // step(b, a) returns 1.0 if a >= b, 0.0 otherwise
                // We want a > b strictly, but step is >=, close enough for floats
                Ast::Call("step".into(), vec![b.to_dew_ast(), a.to_dew_ast()])
            }
            Self::Lt(a, b) => {
                // step(a, b) returns 1.0 if b >= a, i.e., a <= b
                Ast::Call("step".into(), vec![a.to_dew_ast(), b.to_dew_ast()])
            }

            // Colorspace conversions - emit function calls that must be registered
            Self::RgbToHsl(e) => Ast::Call("rgb_to_hsl".into(), vec![e.to_dew_ast()]),
            Self::HslToRgb(e) => Ast::Call("hsl_to_rgb".into(), vec![e.to_dew_ast()]),
            Self::RgbToHsv(e) => Ast::Call("rgb_to_hsv".into(), vec![e.to_dew_ast()]),
            Self::HsvToRgb(e) => Ast::Call("hsv_to_rgb".into(), vec![e.to_dew_ast()]),
            Self::RgbToHwb(e) => Ast::Call("rgb_to_hwb".into(), vec![e.to_dew_ast()]),
            Self::HwbToRgb(e) => Ast::Call("hwb_to_rgb".into(), vec![e.to_dew_ast()]),
            Self::RgbToLab(e) => Ast::Call("rgb_to_lab".into(), vec![e.to_dew_ast()]),
            Self::LabToRgb(e) => Ast::Call("lab_to_rgb".into(), vec![e.to_dew_ast()]),
            Self::RgbToLch(e) => Ast::Call("rgb_to_lch".into(), vec![e.to_dew_ast()]),
            Self::LchToRgb(e) => Ast::Call("lch_to_rgb".into(), vec![e.to_dew_ast()]),
            Self::RgbToOklab(e) => Ast::Call("rgb_to_oklab".into(), vec![e.to_dew_ast()]),
            Self::OklabToRgb(e) => Ast::Call("oklab_to_rgb".into(), vec![e.to_dew_ast()]),
            Self::RgbToOklch(e) => Ast::Call("rgb_to_oklch".into(), vec![e.to_dew_ast()]),
            Self::OklchToRgb(e) => Ast::Call("oklch_to_rgb".into(), vec![e.to_dew_ast()]),
            Self::RgbToYcbcr(e) => Ast::Call("rgb_to_ycbcr".into(), vec![e.to_dew_ast()]),
            Self::YcbcrToRgb(e) => Ast::Call("ycbcr_to_rgb".into(), vec![e.to_dew_ast()]),

            // HSL/HSV adjustments
            Self::AdjustHsl {
                input,
                hue_shift,
                saturation,
                lightness,
            } => Ast::Call(
                "adjust_hsl".into(),
                vec![
                    input.to_dew_ast(),
                    Ast::Num(*hue_shift as f64),
                    Ast::Num(*saturation as f64),
                    Ast::Num(*lightness as f64),
                ],
            ),
            Self::AdjustHsv {
                input,
                hue_shift,
                saturation,
                value,
            } => Ast::Call(
                "adjust_hsv".into(),
                vec![
                    input.to_dew_ast(),
                    Ast::Num(*hue_shift as f64),
                    Ast::Num(*saturation as f64),
                    Ast::Num(*value as f64),
                ],
            ),

            Self::AdjustBrightnessContrast {
                input,
                brightness,
                contrast,
            } => Ast::Call(
                "adjust_brightness_contrast".into(),
                vec![
                    input.to_dew_ast(),
                    Ast::Num(*brightness as f64),
                    Ast::Num(*contrast as f64),
                ],
            ),

            Self::Matrix { input, matrix } => {
                // Emit matrix as a flat 16-element call: color_matrix(input, m00, m01, ..., m33)
                let mut args = vec![input.to_dew_ast()];
                for row in matrix {
                    for val in row {
                        args.push(Ast::Num(*val as f64));
                    }
                }
                Ast::Call("color_matrix".into(), args)
            }
        }
    }
}

/// Colorspace conversion functions for dew expression evaluation.
///
/// These functions allow colorspace conversions to be used in dew expressions
/// when evaluated via `rhizome_dew_linalg`.
///
/// # Example
///
/// ```ignore
/// use rhizome_dew_linalg::{linalg_registry, eval, Value};
/// use unshape_image::register_colorspace;
///
/// let mut registry = linalg_registry();
/// register_colorspace(&mut registry);
///
/// // Now you can use rgb_to_hsl, hsl_to_rgb, etc. in expressions
/// ```
#[cfg(feature = "dew")]
pub mod colorspace_dew {
    use num_traits::NumCast;
    use rhizome_dew_core::Numeric;
    use rhizome_dew_linalg::{FunctionRegistry, LinalgFn, LinalgValue, Signature, Type};

    macro_rules! colorspace_fn {
        ($name:ident, $fn_name:literal, $convert:expr) => {
            /// Colorspace conversion function for dew.
            pub struct $name;

            impl<T, V> LinalgFn<T, V> for $name
            where
                T: Numeric,
                V: LinalgValue<T>,
            {
                fn name(&self) -> &str {
                    $fn_name
                }

                fn signatures(&self) -> Vec<Signature> {
                    // Takes vec4 (RGBA), returns vec4 (converted RGB + preserved A)
                    vec![Signature {
                        args: vec![Type::Vec4],
                        ret: Type::Vec4,
                    }]
                }

                fn call(&self, args: &[V]) -> V {
                    let rgba = args[0].as_vec4().unwrap();
                    let r: f32 = NumCast::from(rgba[0]).unwrap_or(0.0);
                    let g: f32 = NumCast::from(rgba[1]).unwrap_or(0.0);
                    let b: f32 = NumCast::from(rgba[2]).unwrap_or(0.0);
                    let a: f32 = NumCast::from(rgba[3]).unwrap_or(1.0);

                    let convert: fn(f32, f32, f32) -> (f32, f32, f32) = $convert;
                    let (c0, c1, c2) = convert(r, g, b);

                    V::from_vec4([
                        NumCast::from(c0).unwrap_or_else(T::zero),
                        NumCast::from(c1).unwrap_or_else(T::zero),
                        NumCast::from(c2).unwrap_or_else(T::zero),
                        NumCast::from(a).unwrap_or_else(T::one),
                    ])
                }
            }
        };
    }

    colorspace_fn!(RgbToHsl, "rgb_to_hsl", super::rgb_to_hsl);
    colorspace_fn!(HslToRgb, "hsl_to_rgb", super::hsl_to_rgb);
    colorspace_fn!(RgbToHsv, "rgb_to_hsv", super::rgb_to_hsv);
    colorspace_fn!(HsvToRgb, "hsv_to_rgb", super::hsv_to_rgb);
    colorspace_fn!(RgbToHwb, "rgb_to_hwb", super::rgb_to_hwb);
    colorspace_fn!(HwbToRgb, "hwb_to_rgb", super::hwb_to_rgb);
    colorspace_fn!(RgbToLab, "rgb_to_lab", super::rgb_to_lab);
    colorspace_fn!(LabToRgb, "lab_to_rgb", super::lab_to_rgb);
    colorspace_fn!(RgbToLch, "rgb_to_lch", super::rgb_to_lch);
    colorspace_fn!(LchToRgb, "lch_to_rgb", super::lch_to_rgb);
    colorspace_fn!(RgbToOklab, "rgb_to_oklab", super::rgb_to_oklab);
    colorspace_fn!(OklabToRgb, "oklab_to_rgb", super::oklab_to_rgb);
    colorspace_fn!(RgbToOklch, "rgb_to_oklch", super::rgb_to_oklch);
    colorspace_fn!(OklchToRgb, "oklch_to_rgb", super::oklch_to_rgb);
    colorspace_fn!(RgbToYcbcr, "rgb_to_ycbcr", super::rgb_to_ycbcr);
    colorspace_fn!(YcbcrToRgb, "ycbcr_to_rgb", super::ycbcr_to_rgb);

    /// Registers all colorspace conversion functions into a dew-linalg registry.
    pub fn register_colorspace<T, V>(registry: &mut FunctionRegistry<T, V>)
    where
        T: Numeric,
        V: LinalgValue<T>,
    {
        registry.register(RgbToHsl);
        registry.register(HslToRgb);
        registry.register(RgbToHsv);
        registry.register(HsvToRgb);
        registry.register(RgbToHwb);
        registry.register(HwbToRgb);
        registry.register(RgbToLab);
        registry.register(LabToRgb);
        registry.register(RgbToLch);
        registry.register(LchToRgb);
        registry.register(RgbToOklab);
        registry.register(OklabToRgb);
        registry.register(RgbToOklch);
        registry.register(OklchToRgb);
        registry.register(RgbToYcbcr);
        registry.register(YcbcrToRgb);
    }
}

#[cfg(feature = "dew")]
pub use colorspace_dew::register_colorspace;

/// Remaps UV coordinates using an expression.
///
/// This is a fundamental primitive for geometric image transforms. All UV-based
/// effects (distortions, transforms, warps) can be expressed through this function.
///
/// # Arguments
///
/// * `image` - The source image to sample from
/// * `expr` - A `UvExpr` that maps (u, v) → (u', v')
///
/// # How It Works
///
/// For each output pixel at position (u, v):
/// 1. Evaluate `expr` to get the source coordinates (u', v')
/// 2. Sample the source image at (u', v')
/// 3. Write that color to the output pixel
///
/// # Example
///
/// ```
/// use unshape_image::{ImageField, UvExpr, remap_uv};
///
/// let image = ImageField::from_raw(vec![[0.5, 0.5, 0.5, 1.0]; 16], 4, 4);
///
/// // Simple translation
/// let translated = remap_uv(&image, &UvExpr::translate(0.1, 0.0));
///
/// // Rotation around center
/// let rotated = remap_uv(&image, &UvExpr::rotate_centered(0.5));
/// ```
pub fn remap_uv(image: &ImageField, expr: &UvExpr) -> ImageField {
    RemapUv::new(expr.clone()).apply(image)
}

/// Internal remap_uv implementation.
pub(crate) fn remap_uv_impl(image: &ImageField, expr: &UvExpr) -> ImageField {
    let (width, height) = image.dimensions();
    let mut data = Vec::with_capacity((width * height) as usize);

    for y in 0..height {
        for x in 0..width {
            let u = (x as f32 + 0.5) / width as f32;
            let v = (y as f32 + 0.5) / height as f32;

            let (src_u, src_v) = expr.eval(u, v);
            let color = image.sample_uv(src_u, src_v);
            data.push([color.r, color.g, color.b, color.a]);
        }
    }

    ImageField::from_raw(data, width, height)
        .with_wrap_mode(image.wrap_mode)
        .with_filter_mode(image.filter_mode)
}

/// Applies a per-pixel color transform using an expression.
///
/// This is a fundamental primitive for color image transforms. All per-pixel
/// color effects (grayscale, invert, threshold, color grading) can be expressed
/// through this function.
///
/// # Arguments
///
/// * `image` - The source image to transform
/// * `expr` - A `ColorExpr` that maps (r, g, b, a) → (r', g', b', a')
///
/// # How It Works
///
/// For each pixel in the image:
/// 1. Read the RGBA color
/// 2. Evaluate `expr` to transform the color
/// 3. Write the transformed color back
///
/// # Example
///
/// ```
/// use unshape_image::{ImageField, ColorExpr, map_pixels};
///
/// let image = ImageField::from_raw(vec![[0.5, 0.3, 0.7, 1.0]; 16], 4, 4);
///
/// // Convert to grayscale
/// let gray = map_pixels(&image, &ColorExpr::grayscale());
///
/// // Invert colors
/// let inverted = map_pixels(&image, &ColorExpr::invert());
///
/// // Apply threshold
/// let binary = map_pixels(&image, &ColorExpr::threshold(0.5));
/// ```
pub fn map_pixels(image: &ImageField, expr: &ColorExpr) -> ImageField {
    MapPixels::new(expr.clone()).apply(image)
}

/// Internal map_pixels implementation.
pub(crate) fn map_pixels_impl(image: &ImageField, expr: &ColorExpr) -> ImageField {
    let (width, height) = image.dimensions();
    let mut data = Vec::with_capacity((width * height) as usize);

    for y in 0..height {
        for x in 0..width {
            let pixel = image.get_pixel(x, y);
            let [r, g, b, a] = expr.eval(pixel[0], pixel[1], pixel[2], pixel[3]);
            data.push([r, g, b, a]);
        }
    }

    ImageField::from_raw(data, width, height)
        .with_wrap_mode(image.wrap_mode)
        .with_filter_mode(image.filter_mode)
}

/// Applies a UV coordinate remapping using a closure.
///
/// This is the runtime closure variant of [`remap_uv`]. Use this when:
/// - The transformation doesn't need to be serialized
/// - The transformation is a one-off custom effect
/// - The transformation references external state (like another image in [`displace`])
///
/// For serializable/compilable transforms, use [`remap_uv`] with [`UvExpr`] instead.
///
/// # Arguments
///
/// * `image` - The source image to transform
/// * `f` - A function that maps output UV → source UV coordinates
///
/// # Example
///
/// ```
/// use unshape_image::{ImageField, remap_uv_fn};
///
/// let image = ImageField::from_raw(vec![[0.5, 0.5, 0.5, 1.0]; 16], 4, 4);
///
/// // Custom swirl effect
/// let swirled = remap_uv_fn(&image, |u, v| {
///     let dx = u - 0.5;
///     let dy = v - 0.5;
///     let dist = (dx * dx + dy * dy).sqrt();
///     let angle = dist * 3.0;
///     let cos_a = angle.cos();
///     let sin_a = angle.sin();
///     (0.5 + dx * cos_a - dy * sin_a, 0.5 + dx * sin_a + dy * cos_a)
/// });
/// ```
pub fn remap_uv_fn(image: &ImageField, f: impl Fn(f32, f32) -> (f32, f32)) -> ImageField {
    let (width, height) = image.dimensions();
    let mut data = Vec::with_capacity((width * height) as usize);

    for y in 0..height {
        for x in 0..width {
            let u = (x as f32 + 0.5) / width as f32;
            let v = (y as f32 + 0.5) / height as f32;

            let (src_u, src_v) = f(u, v);
            let color = image.sample_uv(src_u, src_v);
            data.push([color.r, color.g, color.b, color.a]);
        }
    }

    ImageField::from_raw(data, width, height)
        .with_wrap_mode(image.wrap_mode)
        .with_filter_mode(image.filter_mode)
}
