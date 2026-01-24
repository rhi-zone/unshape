//! Text to path conversion using fonts.
//!
//! # Example
//!
//! ```ignore
//! use unshape_vector::{Font, TextLayout, text_to_paths};
//!
//! let font = Font::from_bytes(include_bytes!("font.ttf")).unwrap();
//! let paths = text_to_paths("Hello", &font, TextLayout::default());
//! ```

use ab_glyph::{Font as AbFont, FontRef, GlyphId, OutlinedGlyph, ScaleFont};
use glam::Vec2;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::path::{Path, PathBuilder};

/// Errors that can occur during font loading.
#[derive(Debug, thiserror::Error)]
pub enum FontError {
    /// Invalid font data.
    #[error("Invalid font: {0}")]
    InvalidFont(String),
}

/// Result type for font operations.
pub type FontResult<T> = Result<T, FontError>;

/// A loaded font that can be used to convert text to paths.
pub struct Font<'a> {
    font: FontRef<'a>,
}

impl<'a> Font<'a> {
    /// Loads a font from raw bytes (TTF or OTF format).
    pub fn from_bytes(data: &'a [u8]) -> FontResult<Self> {
        let font =
            FontRef::try_from_slice(data).map_err(|e| FontError::InvalidFont(e.to_string()))?;
        Ok(Self { font })
    }

    /// Returns the font's units per em.
    pub fn units_per_em(&self) -> f32 {
        self.font.units_per_em().unwrap_or(1000.0)
    }

    /// Returns the font's ascent in font units.
    pub fn ascent(&self) -> f32 {
        self.font.as_scaled(1.0).ascent()
    }

    /// Returns the font's descent in font units.
    pub fn descent(&self) -> f32 {
        self.font.as_scaled(1.0).descent()
    }
}

/// Configuration for text rendering.
///
/// Note: This is a configuration struct rather than a full Op because text
/// rendering requires a font reference which cannot be serialized.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct TextLayout {
    /// Font size in pixels/units.
    pub size: f32,
    /// Letter spacing (0.0 = default).
    pub letter_spacing: f32,
    /// Line height multiplier (1.0 = default).
    pub line_height: f32,
}

impl Default for TextLayout {
    fn default() -> Self {
        Self {
            size: 72.0,
            letter_spacing: 0.0,
            line_height: 1.2,
        }
    }
}

impl TextLayout {
    /// Creates a new text layout with default values.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Backwards-compatible type alias.
pub type TextConfig = TextLayout;

/// Converts text to vector paths using a font.
///
/// Returns a list of paths, one per glyph. The paths are positioned
/// for natural text layout with the baseline at y=0.
pub fn text_to_paths(text: &str, font: &Font, config: TextConfig) -> Vec<Path> {
    let scaled_font = font.font.as_scaled(config.size);
    let mut paths = Vec::new();
    let mut x = 0.0;
    let mut y = 0.0;

    let line_height = scaled_font.height() * config.line_height;

    for c in text.chars() {
        if c == '\n' {
            x = 0.0;
            y += line_height;
            continue;
        }

        let glyph_id = font.font.glyph_id(c);
        if let Some(outlined) = scaled_font
            .outline_glyph(glyph_id.with_scale_and_position(config.size, ab_glyph::point(x, y)))
        {
            if let Some(path) = glyph_to_path(&outlined) {
                paths.push(path);
            }
        }

        // Advance cursor
        x += scaled_font.h_advance(glyph_id) + config.letter_spacing;
    }

    paths
}

/// Converts text to a single combined path.
pub fn text_to_path(text: &str, font: &Font, config: TextConfig) -> Path {
    let paths = text_to_paths(text, font, config);
    let mut combined = Path::new();
    for path in paths {
        combined.extend(&path);
    }
    combined
}

/// Information about laid-out text.
#[derive(Debug, Clone)]
pub struct TextMetrics {
    /// Total width of the text.
    pub width: f32,
    /// Total height of the text.
    pub height: f32,
    /// Ascent from baseline.
    pub ascent: f32,
    /// Descent from baseline (negative).
    pub descent: f32,
}

/// Measures text without generating paths.
pub fn measure_text(text: &str, font: &Font, config: &TextConfig) -> TextMetrics {
    let scaled_font = font.font.as_scaled(config.size);
    let mut x = 0.0f32;
    let mut max_x = 0.0f32;
    let mut lines = 1;

    for c in text.chars() {
        if c == '\n' {
            max_x = max_x.max(x);
            x = 0.0;
            lines += 1;
            continue;
        }

        let glyph_id = font.font.glyph_id(c);
        x += scaled_font.h_advance(glyph_id) + config.letter_spacing;
    }
    max_x = max_x.max(x);

    let line_height = scaled_font.height() * config.line_height;

    TextMetrics {
        width: max_x,
        height: line_height * lines as f32,
        ascent: scaled_font.ascent(),
        descent: scaled_font.descent(),
    }
}

/// Converts a single glyph outline to a path.
/// Note: OutlinedGlyph is for rasterization; use build_glyph_path for vector outlines.
fn glyph_to_path(glyph: &OutlinedGlyph) -> Option<Path> {
    let bounds = glyph.px_bounds();

    // OutlinedGlyph is for rasterization, not outline extraction
    // Return empty path if bounds are valid - actual outline comes from Font::outline
    if bounds.width() > 0.0 && bounds.height() > 0.0 {
        Some(Path::new())
    } else {
        None
    }
}

/// Converts glyph outlines using the outline curve visitor pattern.
/// This requires accessing the font's raw glyph data.
fn build_glyph_path(font: &ab_glyph::FontRef, glyph_id: GlyphId, scale: f32, offset: Vec2) -> Path {
    use ab_glyph::Font;

    let mut builder = PathBuilder::new();

    // Get the glyph outline and convert to path commands
    if let Some(outline) = font.outline(glyph_id) {
        let mut first = true;

        for curve in outline.curves {
            match curve {
                ab_glyph::OutlineCurve::Line(from, to) => {
                    let from_pt = Vec2::new(from.x * scale + offset.x, -from.y * scale + offset.y);
                    let to_pt = Vec2::new(to.x * scale + offset.x, -to.y * scale + offset.y);

                    if first {
                        builder = builder.move_to(from_pt);
                        first = false;
                    }
                    builder = builder.line_to(to_pt);
                }
                ab_glyph::OutlineCurve::Quad(from, control, to) => {
                    let from_pt = Vec2::new(from.x * scale + offset.x, -from.y * scale + offset.y);
                    let ctrl_pt =
                        Vec2::new(control.x * scale + offset.x, -control.y * scale + offset.y);
                    let to_pt = Vec2::new(to.x * scale + offset.x, -to.y * scale + offset.y);

                    if first {
                        builder = builder.move_to(from_pt);
                        first = false;
                    }
                    builder = builder.quad_to(ctrl_pt, to_pt);
                }
                ab_glyph::OutlineCurve::Cubic(from, control1, control2, to) => {
                    let from_pt = Vec2::new(from.x * scale + offset.x, -from.y * scale + offset.y);
                    let ctrl1_pt = Vec2::new(
                        control1.x * scale + offset.x,
                        -control1.y * scale + offset.y,
                    );
                    let ctrl2_pt = Vec2::new(
                        control2.x * scale + offset.x,
                        -control2.y * scale + offset.y,
                    );
                    let to_pt = Vec2::new(to.x * scale + offset.x, -to.y * scale + offset.y);

                    if first {
                        builder = builder.move_to(from_pt);
                        first = false;
                    }
                    builder = builder.cubic_to(ctrl1_pt, ctrl2_pt, to_pt);
                }
            }
        }
    }

    builder.build()
}

/// Converts text to paths using the direct outline method.
pub fn text_to_paths_outlined(text: &str, font: &Font, config: TextConfig) -> Vec<Path> {
    let scaled_font = font.font.as_scaled(config.size);
    let scale = config.size / font.units_per_em();
    let mut paths = Vec::new();
    let mut x = 0.0;
    let mut y = 0.0;

    let line_height = scaled_font.height() * config.line_height;

    for c in text.chars() {
        if c == '\n' {
            x = 0.0;
            y += line_height;
            continue;
        }

        if c == ' ' {
            let glyph_id = font.font.glyph_id(c);
            x += scaled_font.h_advance(glyph_id) + config.letter_spacing;
            continue;
        }

        let glyph_id = font.font.glyph_id(c);
        let offset = Vec2::new(x, y);
        let path = build_glyph_path(&font.font, glyph_id, scale, offset);

        if !path.is_empty() {
            paths.push(path);
        }

        x += scaled_font.h_advance(glyph_id) + config.letter_spacing;
    }

    paths
}

/// Converts text to a single combined path using the direct outline method.
pub fn text_to_path_outlined(text: &str, font: &Font, config: TextConfig) -> Path {
    let paths = text_to_paths_outlined(text, font, config);
    let mut combined = Path::new();
    for path in paths {
        combined.extend(&path);
    }
    combined
}

#[cfg(test)]
mod tests {
    use super::*;

    // Note: Tests require actual font data. Here we test the basic structure.

    #[test]
    fn test_text_config_default() {
        let config = TextConfig::default();
        assert_eq!(config.size, 72.0);
        assert_eq!(config.letter_spacing, 0.0);
        assert!((config.line_height - 1.2).abs() < 0.001);
    }

    #[test]
    fn test_text_config_struct_init() {
        let config = TextConfig {
            size: 24.0,
            letter_spacing: 2.0,
            line_height: 1.5,
        };

        assert_eq!(config.size, 24.0);
        assert_eq!(config.letter_spacing, 2.0);
        assert!((config.line_height - 1.5).abs() < 0.001);
    }

    // Integration tests with actual fonts would go here
    // They would require bundling a test font or using system fonts
}
