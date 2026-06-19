//! Projectional editor MVP.
//!
//! A single live texture is produced by an ordered list of image modifiers
//! (an *op stack*). The SAME modifier is shown in two co-equal projections:
//!
//! 1. an editable op-stack row (sliders over the modifier's scalar parameters), and
//! 2. an editable formula view, whose text drives the GPU directly.
//!
//! The model here is the source of truth. A typed [`ImageMod`] (e.g.
//! [`ImageMod::UvScale`]) stores its editable scalar parameters as plain
//! serializable values; the heavier `UvExpr` / `ColorExpr` representation is
//! *derived* from those parameters on demand, and the formula text is derived
//! from that.
//!
//! When the user edits a typed modifier's formula, it is **promoted** to an
//! [`ImageMod::Expr`] whose source of truth is the formula *string* itself. That
//! string is parsed to a dew [`Ast`](wick_core::Ast) and fed straight to the GPU
//! via the raw-AST entry points, so editing a formula recompiles and repaints
//! live, exactly like editing a slider.

use serde::{Deserialize, Serialize};
use unshape_image::{ColorExpr, UvExpr};

pub mod render;

/// A single image modifier in the op stack.
///
/// Each variant carries the small set of scalar parameters a user edits. The
/// corresponding `UvExpr` / `ColorExpr` (and thus the formula projection) is
/// derived from those parameters, never stored.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ImageMod {
    /// Uniform scale around the texture center.
    UvScale {
        /// Horizontal scale factor.
        x: f32,
        /// Vertical scale factor.
        y: f32,
    },
    /// Rotation around the texture center, in radians.
    UvRotate {
        /// Angle in radians.
        angle: f32,
    },
    /// Per-channel brightness multiply.
    ColorBrightness {
        /// Multiplier applied to R, G, B.
        factor: f32,
    },
    /// Hue rotation, in radians.
    ColorHueShift {
        /// Hue rotation amount in radians.
        amount: f32,
    },
    /// Per-channel gamma correction.
    ColorGamma {
        /// Gamma exponent.
        gamma: f32,
    },
    /// A free-form expression node whose source of truth is the formula string.
    ///
    /// The `source` text is parsed to a dew [`Ast`](wick_core::Ast) and applied
    /// directly on the GPU via the raw-AST entry points, using the same bound
    /// variables as the typed passes (`uv: vec2` for [`PassKind::Uv`],
    /// `rgba: vec4` for [`PassKind::Color`]). This means a typed modifier and an
    /// `Expr` carrying its derived formula produce identical output.
    Expr {
        /// Which GPU pass this expression drives.
        kind: PassKind,
        /// The formula source. This is the editable source of truth.
        source: String,
    },
}

/// Which GPU pass a modifier maps to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PassKind {
    /// A UV remap pass (`remap_uv_gpu` / `remap_uv_ast_gpu`).
    Uv,
    /// A per-pixel color pass (`map_pixels_gpu` / `map_pixels_ast_gpu`).
    Color,
}

/// The derived GPU expression for a modifier: either a UV remap or a color map.
pub enum ModExpr {
    /// UV remap expression.
    Uv(UvExpr),
    /// Per-pixel color expression.
    Color(ColorExpr),
}

impl ImageMod {
    /// Short human-readable kind label for the op-stack row.
    pub fn label(&self) -> &'static str {
        match self {
            ImageMod::UvScale { .. } => "UV Scale",
            ImageMod::UvRotate { .. } => "UV Rotate",
            ImageMod::ColorBrightness { .. } => "Color Brightness",
            ImageMod::ColorHueShift { .. } => "Color Hue Shift",
            ImageMod::ColorGamma { .. } => "Color Gamma",
            ImageMod::Expr {
                kind: PassKind::Uv, ..
            } => "Formula (uv)",
            ImageMod::Expr {
                kind: PassKind::Color,
                ..
            } => "Formula (color)",
        }
    }

    /// Whether this modifier is a UV pass or a color pass.
    pub fn kind(&self) -> PassKind {
        match self {
            ImageMod::UvScale { .. } | ImageMod::UvRotate { .. } => PassKind::Uv,
            ImageMod::ColorBrightness { .. }
            | ImageMod::ColorHueShift { .. }
            | ImageMod::ColorGamma { .. } => PassKind::Color,
            ImageMod::Expr { kind, .. } => *kind,
        }
    }

    /// Derives the typed GPU expression for a *typed* modifier from its scalar
    /// parameters.
    ///
    /// Returns `None` for [`ImageMod::Expr`] nodes, whose source of truth is the
    /// formula string and which are applied via the raw-AST GPU path instead.
    pub fn to_expr(&self) -> Option<ModExpr> {
        match *self {
            ImageMod::UvScale { x, y } => Some(ModExpr::Uv(UvExpr::scale_centered(x, y))),
            ImageMod::UvRotate { angle } => Some(ModExpr::Uv(UvExpr::rotate_centered(angle))),
            ImageMod::ColorBrightness { factor } => {
                Some(ModExpr::Color(ColorExpr::brightness(factor)))
            }
            ImageMod::ColorHueShift { amount } => {
                Some(ModExpr::Color(ColorExpr::hue_shift(amount)))
            }
            ImageMod::ColorGamma { gamma } => Some(ModExpr::Color(ColorExpr::gamma(gamma))),
            ImageMod::Expr { .. } => None,
        }
    }

    /// The formula text projection of this modifier.
    ///
    /// For a typed modifier this is *derived* from its dew AST (the formula the
    /// GPU pass actually evaluates). For an [`ImageMod::Expr`] it is the editable
    /// source string itself.
    pub fn to_formula(&self) -> String {
        match self {
            ImageMod::Expr { source, .. } => source.clone(),
            _ => match self.to_expr() {
                Some(ModExpr::Uv(e)) => format!("{}", e.to_dew_ast()),
                Some(ModExpr::Color(e)) => format!("{}", e.to_dew_ast()),
                None => String::new(),
            },
        }
    }
}

/// The editor document: an ordered op stack of image modifiers.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Document {
    /// Modifiers applied in order, top-to-bottom, to the source texture.
    pub mods: Vec<ImageMod>,
}

impl Default for Document {
    fn default() -> Self {
        // Seed with a few modifiers so the editor shows something immediately.
        Self {
            mods: vec![
                ImageMod::UvScale { x: 1.5, y: 1.5 },
                ImageMod::ColorHueShift {
                    amount: std::f32::consts::FRAC_PI_2,
                },
                ImageMod::ColorBrightness { factor: 1.2 },
            ],
        }
    }
}
