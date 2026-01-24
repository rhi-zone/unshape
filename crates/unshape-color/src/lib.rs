//! Color types and gradients for procedural generation.
//!
//! Provides color spaces, conversions, and gradient interpolation.

use glam::Vec3;
pub use unshape_easing::Lerp;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Linear RGB color (0-1 range, not gamma corrected).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct LinearRgb {
    /// Red channel (0-1).
    pub r: f32,
    /// Green channel (0-1).
    pub g: f32,
    /// Blue channel (0-1).
    pub b: f32,
}

impl LinearRgb {
    /// Black (0, 0, 0).
    pub const BLACK: Self = Self::new(0.0, 0.0, 0.0);
    /// White (1, 1, 1).
    pub const WHITE: Self = Self::new(1.0, 1.0, 1.0);
    /// Red (1, 0, 0).
    pub const RED: Self = Self::new(1.0, 0.0, 0.0);
    /// Green (0, 1, 0).
    pub const GREEN: Self = Self::new(0.0, 1.0, 0.0);
    /// Blue (0, 0, 1).
    pub const BLUE: Self = Self::new(0.0, 0.0, 1.0);

    /// Creates a new RGB color.
    pub const fn new(r: f32, g: f32, b: f32) -> Self {
        Self { r, g, b }
    }

    /// Creates from a Vec3.
    pub fn from_vec3(v: Vec3) -> Self {
        Self::new(v.x, v.y, v.z)
    }

    /// Converts to Vec3.
    pub fn to_vec3(self) -> Vec3 {
        Vec3::new(self.r, self.g, self.b)
    }

    /// Creates from sRGB (gamma corrected) values.
    pub fn from_srgb(r: f32, g: f32, b: f32) -> Self {
        Self::new(srgb_to_linear(r), srgb_to_linear(g), srgb_to_linear(b))
    }

    /// Creates from a hex color code (e.g., 0xFF5500).
    pub fn from_hex(hex: u32) -> Self {
        let r = ((hex >> 16) & 0xFF) as f32 / 255.0;
        let g = ((hex >> 8) & 0xFF) as f32 / 255.0;
        let b = (hex & 0xFF) as f32 / 255.0;
        Self::from_srgb(r, g, b)
    }

    /// Converts to sRGB (gamma corrected).
    pub fn to_srgb(self) -> (f32, f32, f32) {
        (
            linear_to_srgb(self.r),
            linear_to_srgb(self.g),
            linear_to_srgb(self.b),
        )
    }

    /// Converts to hex color code.
    pub fn to_hex(self) -> u32 {
        let (r, g, b) = self.to_srgb();
        let r = (r.clamp(0.0, 1.0) * 255.0) as u32;
        let g = (g.clamp(0.0, 1.0) * 255.0) as u32;
        let b = (b.clamp(0.0, 1.0) * 255.0) as u32;
        (r << 16) | (g << 8) | b
    }

    /// Converts to HSL color space.
    pub fn to_hsl(self) -> Hsl {
        let max = self.r.max(self.g).max(self.b);
        let min = self.r.min(self.g).min(self.b);
        let l = (max + min) / 2.0;

        if (max - min).abs() < 0.0001 {
            return Hsl::new(0.0, 0.0, l);
        }

        let d = max - min;
        let s = if l > 0.5 {
            d / (2.0 - max - min)
        } else {
            d / (max + min)
        };

        let h = if (max - self.r).abs() < 0.0001 {
            (self.g - self.b) / d + (if self.g < self.b { 6.0 } else { 0.0 })
        } else if (max - self.g).abs() < 0.0001 {
            (self.b - self.r) / d + 2.0
        } else {
            (self.r - self.g) / d + 4.0
        };

        Hsl::new(h / 6.0, s, l)
    }

    /// Converts to HSV color space.
    pub fn to_hsv(self) -> Hsv {
        let max = self.r.max(self.g).max(self.b);
        let min = self.r.min(self.g).min(self.b);
        let v = max;

        if (max - min).abs() < 0.0001 {
            return Hsv::new(0.0, 0.0, v);
        }

        let d = max - min;
        let s = d / max;

        let h = if (max - self.r).abs() < 0.0001 {
            (self.g - self.b) / d + (if self.g < self.b { 6.0 } else { 0.0 })
        } else if (max - self.g).abs() < 0.0001 {
            (self.b - self.r) / d + 2.0
        } else {
            (self.r - self.g) / d + 4.0
        };

        Hsv::new(h / 6.0, s, v)
    }

    /// Linear interpolation between two colors.
    pub fn lerp(self, other: Self, t: f32) -> Self {
        Self::new(
            self.r + (other.r - self.r) * t,
            self.g + (other.g - self.g) * t,
            self.b + (other.b - self.b) * t,
        )
    }

    /// Returns the luminance of the color.
    pub fn luminance(self) -> f32 {
        0.2126 * self.r + 0.7152 * self.g + 0.0722 * self.b
    }

    /// Clamps all components to [0, 1].
    pub fn clamp(self) -> Self {
        Self::new(
            self.r.clamp(0.0, 1.0),
            self.g.clamp(0.0, 1.0),
            self.b.clamp(0.0, 1.0),
        )
    }
}

/// HSL (Hue, Saturation, Lightness) color space.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Hsl {
    /// Hue (0-1, wraps around).
    pub h: f32,
    /// Saturation (0-1).
    pub s: f32,
    /// Lightness (0-1).
    pub l: f32,
}

impl Hsl {
    /// Creates a new HSL color.
    pub const fn new(h: f32, s: f32, l: f32) -> Self {
        Self { h, s, l }
    }

    /// Converts to linear RGB.
    pub fn to_rgb(self) -> LinearRgb {
        if self.s < 0.0001 {
            return LinearRgb::new(self.l, self.l, self.l);
        }

        let q = if self.l < 0.5 {
            self.l * (1.0 + self.s)
        } else {
            self.l + self.s - self.l * self.s
        };
        let p = 2.0 * self.l - q;

        LinearRgb::new(
            hue_to_rgb(p, q, self.h + 1.0 / 3.0),
            hue_to_rgb(p, q, self.h),
            hue_to_rgb(p, q, self.h - 1.0 / 3.0),
        )
    }

    /// Linear interpolation in HSL space.
    pub fn lerp(self, other: Self, t: f32) -> Self {
        // Handle hue wrapping
        let mut dh = other.h - self.h;
        if dh > 0.5 {
            dh -= 1.0;
        } else if dh < -0.5 {
            dh += 1.0;
        }

        Self::new(
            (self.h + dh * t).rem_euclid(1.0),
            self.s + (other.s - self.s) * t,
            self.l + (other.l - self.l) * t,
        )
    }
}

/// HSV (Hue, Saturation, Value) color space.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Hsv {
    /// Hue (0-1, wraps around).
    pub h: f32,
    /// Saturation (0-1).
    pub s: f32,
    /// Value/Brightness (0-1).
    pub v: f32,
}

impl Hsv {
    /// Creates a new HSV color.
    pub const fn new(h: f32, s: f32, v: f32) -> Self {
        Self { h, s, v }
    }

    /// Converts to linear RGB.
    pub fn to_rgb(self) -> LinearRgb {
        if self.s < 0.0001 {
            return LinearRgb::new(self.v, self.v, self.v);
        }

        let h = self.h * 6.0;
        let i = h.floor() as i32;
        let f = h - i as f32;
        let p = self.v * (1.0 - self.s);
        let q = self.v * (1.0 - self.s * f);
        let t = self.v * (1.0 - self.s * (1.0 - f));

        match i % 6 {
            0 => LinearRgb::new(self.v, t, p),
            1 => LinearRgb::new(q, self.v, p),
            2 => LinearRgb::new(p, self.v, t),
            3 => LinearRgb::new(p, q, self.v),
            4 => LinearRgb::new(t, p, self.v),
            _ => LinearRgb::new(self.v, p, q),
        }
    }

    /// Linear interpolation in HSV space.
    pub fn lerp(self, other: Self, t: f32) -> Self {
        // Handle hue wrapping
        let mut dh = other.h - self.h;
        if dh > 0.5 {
            dh -= 1.0;
        } else if dh < -0.5 {
            dh += 1.0;
        }

        Self::new(
            (self.h + dh * t).rem_euclid(1.0),
            self.s + (other.s - self.s) * t,
            self.v + (other.v - self.v) * t,
        )
    }
}

/// RGBA color with alpha channel.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Rgba {
    /// Red channel (0-1).
    pub r: f32,
    /// Green channel (0-1).
    pub g: f32,
    /// Blue channel (0-1).
    pub b: f32,
    /// Alpha channel (0-1).
    pub a: f32,
}

impl Rgba {
    /// Fully transparent black.
    pub const TRANSPARENT: Self = Self::new(0.0, 0.0, 0.0, 0.0);
    /// Opaque black.
    pub const BLACK: Self = Self::new(0.0, 0.0, 0.0, 1.0);
    /// Opaque white.
    pub const WHITE: Self = Self::new(1.0, 1.0, 1.0, 1.0);

    /// Creates a new RGBA color.
    pub const fn new(r: f32, g: f32, b: f32, a: f32) -> Self {
        Self { r, g, b, a }
    }

    /// Creates from RGB with full opacity.
    pub fn from_rgb(rgb: LinearRgb) -> Self {
        Self::new(rgb.r, rgb.g, rgb.b, 1.0)
    }

    /// Converts to RGB (discards alpha).
    pub fn to_rgb(self) -> LinearRgb {
        LinearRgb::new(self.r, self.g, self.b)
    }

    /// Creates from hex with optional alpha (0xRRGGBB or 0xRRGGBBAA).
    pub fn from_hex(hex: u32, has_alpha: bool) -> Self {
        if has_alpha {
            let r = ((hex >> 24) & 0xFF) as f32 / 255.0;
            let g = ((hex >> 16) & 0xFF) as f32 / 255.0;
            let b = ((hex >> 8) & 0xFF) as f32 / 255.0;
            let a = (hex & 0xFF) as f32 / 255.0;
            Self::new(srgb_to_linear(r), srgb_to_linear(g), srgb_to_linear(b), a)
        } else {
            let rgb = LinearRgb::from_hex(hex);
            Self::from_rgb(rgb)
        }
    }

    /// Linear interpolation with alpha.
    pub fn lerp(self, other: Self, t: f32) -> Self {
        Self::new(
            self.r + (other.r - self.r) * t,
            self.g + (other.g - self.g) * t,
            self.b + (other.b - self.b) * t,
            self.a + (other.a - self.a) * t,
        )
    }

    /// Premultiplies RGB by alpha.
    pub fn premultiply(self) -> Self {
        Self::new(self.r * self.a, self.g * self.a, self.b * self.a, self.a)
    }
}

// ============================================================================
// Lerp Trait Implementations
// ============================================================================

impl Lerp for LinearRgb {
    fn lerp_to(&self, other: &Self, t: f32) -> Self {
        self.lerp(*other, t)
    }
}

impl Lerp for Hsl {
    fn lerp_to(&self, other: &Self, t: f32) -> Self {
        self.lerp(*other, t)
    }
}

impl Lerp for Hsv {
    fn lerp_to(&self, other: &Self, t: f32) -> Self {
        self.lerp(*other, t)
    }
}

impl Lerp for Rgba {
    fn lerp_to(&self, other: &Self, t: f32) -> Self {
        self.lerp(*other, t)
    }
}

// ============================================================================
// From Conversions for Arrays
// ============================================================================

impl From<[f32; 3]> for LinearRgb {
    fn from(arr: [f32; 3]) -> Self {
        Self::new(arr[0], arr[1], arr[2])
    }
}

impl From<LinearRgb> for [f32; 3] {
    fn from(c: LinearRgb) -> Self {
        [c.r, c.g, c.b]
    }
}

impl From<[f32; 4]> for Rgba {
    fn from(arr: [f32; 4]) -> Self {
        Self::new(arr[0], arr[1], arr[2], arr[3])
    }
}

impl From<Rgba> for [f32; 4] {
    fn from(c: Rgba) -> Self {
        [c.r, c.g, c.b, c.a]
    }
}

// ============================================================================
// Color Gradients
// ============================================================================

/// A color stop in a gradient.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ColorStop {
    /// Position in the gradient (0-1).
    pub position: f32,
    /// Color at this position.
    pub color: Rgba,
}

impl ColorStop {
    /// Creates a new color stop.
    pub fn new(position: f32, color: Rgba) -> Self {
        Self { position, color }
    }

    /// Creates from RGB color.
    pub fn rgb(position: f32, color: LinearRgb) -> Self {
        Self::new(position, Rgba::from_rgb(color))
    }
}

/// A linear gradient with multiple color stops.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Gradient {
    /// Color stops (should be sorted by position).
    stops: Vec<ColorStop>,
}

impl Gradient {
    /// Creates an empty gradient.
    pub fn new() -> Self {
        Self { stops: Vec::new() }
    }

    /// Creates a gradient from two colors.
    pub fn two_color(start: LinearRgb, end: LinearRgb) -> Self {
        Self {
            stops: vec![ColorStop::rgb(0.0, start), ColorStop::rgb(1.0, end)],
        }
    }

    /// Creates a gradient from multiple colors (evenly spaced).
    pub fn from_colors(colors: &[LinearRgb]) -> Self {
        if colors.is_empty() {
            return Self::new();
        }
        if colors.len() == 1 {
            return Self {
                stops: vec![ColorStop::rgb(0.0, colors[0])],
            };
        }

        let step = 1.0 / (colors.len() - 1) as f32;
        let stops = colors
            .iter()
            .enumerate()
            .map(|(i, &c)| ColorStop::rgb(i as f32 * step, c))
            .collect();

        Self { stops }
    }

    /// Adds a color stop.
    pub fn add_stop(&mut self, stop: ColorStop) {
        self.stops.push(stop);
        self.stops
            .sort_by(|a, b| a.position.partial_cmp(&b.position).unwrap());
    }

    /// Adds an RGB color stop.
    pub fn add_rgb(&mut self, position: f32, color: LinearRgb) {
        self.add_stop(ColorStop::rgb(position, color));
    }

    /// Samples the gradient at position t (0-1).
    pub fn sample(&self, t: f32) -> Rgba {
        if self.stops.is_empty() {
            return Rgba::BLACK;
        }
        if self.stops.len() == 1 {
            return self.stops[0].color;
        }

        let t = t.clamp(0.0, 1.0);

        // Find surrounding stops
        let mut lower = &self.stops[0];
        let mut upper = &self.stops[self.stops.len() - 1];

        for i in 0..self.stops.len() - 1 {
            if t >= self.stops[i].position && t <= self.stops[i + 1].position {
                lower = &self.stops[i];
                upper = &self.stops[i + 1];
                break;
            }
        }

        // Handle edge cases
        if t <= lower.position {
            return lower.color;
        }
        if t >= upper.position {
            return upper.color;
        }

        // Interpolate
        let range = upper.position - lower.position;
        let local_t = if range > 0.0001 {
            (t - lower.position) / range
        } else {
            0.0
        };

        lower.color.lerp(upper.color, local_t)
    }

    /// Samples and returns RGB (discards alpha).
    pub fn sample_rgb(&self, t: f32) -> LinearRgb {
        self.sample(t).to_rgb()
    }

    /// Returns the number of stops.
    pub fn len(&self) -> usize {
        self.stops.len()
    }

    /// Returns true if the gradient has no stops.
    pub fn is_empty(&self) -> bool {
        self.stops.is_empty()
    }
}

impl Default for Gradient {
    fn default() -> Self {
        Self::new()
    }
}

/// Preset gradients for common use cases.
pub mod presets {
    use super::*;

    /// Black to white.
    pub fn grayscale() -> Gradient {
        Gradient::two_color(LinearRgb::BLACK, LinearRgb::WHITE)
    }

    /// Rainbow gradient (ROYGBIV).
    pub fn rainbow() -> Gradient {
        Gradient::from_colors(&[
            LinearRgb::from_hex(0xFF0000), // Red
            LinearRgb::from_hex(0xFF8000), // Orange
            LinearRgb::from_hex(0xFFFF00), // Yellow
            LinearRgb::from_hex(0x00FF00), // Green
            LinearRgb::from_hex(0x0080FF), // Blue
            LinearRgb::from_hex(0x8000FF), // Indigo
            LinearRgb::from_hex(0xFF00FF), // Violet
        ])
    }

    /// Heat map (blue -> cyan -> green -> yellow -> red).
    pub fn heat() -> Gradient {
        Gradient::from_colors(&[
            LinearRgb::from_hex(0x0000FF),
            LinearRgb::from_hex(0x00FFFF),
            LinearRgb::from_hex(0x00FF00),
            LinearRgb::from_hex(0xFFFF00),
            LinearRgb::from_hex(0xFF0000),
        ])
    }

    /// Viridis colormap (perceptually uniform).
    pub fn viridis() -> Gradient {
        Gradient::from_colors(&[
            LinearRgb::from_hex(0x440154),
            LinearRgb::from_hex(0x3B528B),
            LinearRgb::from_hex(0x21918C),
            LinearRgb::from_hex(0x5EC962),
            LinearRgb::from_hex(0xFDE725),
        ])
    }

    /// Inferno colormap.
    pub fn inferno() -> Gradient {
        Gradient::from_colors(&[
            LinearRgb::from_hex(0x000004),
            LinearRgb::from_hex(0x420A68),
            LinearRgb::from_hex(0x932667),
            LinearRgb::from_hex(0xDD513A),
            LinearRgb::from_hex(0xFCA50A),
            LinearRgb::from_hex(0xFCFFA4),
        ])
    }
}

// ============================================================================
// Blend Modes
// ============================================================================

/// Color blend modes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum BlendMode {
    /// Normal alpha blending.
    #[default]
    Normal,
    /// Multiply colors.
    Multiply,
    /// Screen (inverse multiply).
    Screen,
    /// Overlay (multiply or screen based on base).
    Overlay,
    /// Darken (minimum).
    Darken,
    /// Lighten (maximum).
    Lighten,
    /// Add colors.
    Add,
    /// Subtract.
    Subtract,
    /// Difference.
    Difference,
}

/// Blends two colors using the specified mode.
pub fn blend(base: LinearRgb, blend: LinearRgb, mode: BlendMode) -> LinearRgb {
    match mode {
        BlendMode::Normal => blend,
        BlendMode::Multiply => LinearRgb::new(base.r * blend.r, base.g * blend.g, base.b * blend.b),
        BlendMode::Screen => LinearRgb::new(
            1.0 - (1.0 - base.r) * (1.0 - blend.r),
            1.0 - (1.0 - base.g) * (1.0 - blend.g),
            1.0 - (1.0 - base.b) * (1.0 - blend.b),
        ),
        BlendMode::Overlay => LinearRgb::new(
            overlay_channel(base.r, blend.r),
            overlay_channel(base.g, blend.g),
            overlay_channel(base.b, blend.b),
        ),
        BlendMode::Darken => LinearRgb::new(
            base.r.min(blend.r),
            base.g.min(blend.g),
            base.b.min(blend.b),
        ),
        BlendMode::Lighten => LinearRgb::new(
            base.r.max(blend.r),
            base.g.max(blend.g),
            base.b.max(blend.b),
        ),
        BlendMode::Add => LinearRgb::new(
            (base.r + blend.r).min(1.0),
            (base.g + blend.g).min(1.0),
            (base.b + blend.b).min(1.0),
        ),
        BlendMode::Subtract => LinearRgb::new(
            (base.r - blend.r).max(0.0),
            (base.g - blend.g).max(0.0),
            (base.b - blend.b).max(0.0),
        ),
        BlendMode::Difference => LinearRgb::new(
            (base.r - blend.r).abs(),
            (base.g - blend.g).abs(),
            (base.b - blend.b).abs(),
        ),
    }
}

/// Blends with alpha and opacity.
pub fn blend_with_alpha(base: Rgba, overlay: Rgba, mode: BlendMode, opacity: f32) -> Rgba {
    let blended_rgb = blend(base.to_rgb(), overlay.to_rgb(), mode);
    let result_rgb = base.to_rgb().lerp(blended_rgb, opacity * overlay.a);

    // Combine alphas
    let result_a = base.a + overlay.a * opacity * (1.0 - base.a);

    Rgba::new(result_rgb.r, result_rgb.g, result_rgb.b, result_a)
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Converts sRGB to linear RGB (single channel).
fn srgb_to_linear(c: f32) -> f32 {
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

/// Converts linear RGB to sRGB (single channel).
fn linear_to_srgb(c: f32) -> f32 {
    if c <= 0.0031308 {
        c * 12.92
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    }
}

/// Helper for HSL to RGB conversion.
fn hue_to_rgb(p: f32, q: f32, t: f32) -> f32 {
    let t = t.rem_euclid(1.0);
    if t < 1.0 / 6.0 {
        p + (q - p) * 6.0 * t
    } else if t < 0.5 {
        q
    } else if t < 2.0 / 3.0 {
        p + (q - p) * (2.0 / 3.0 - t) * 6.0
    } else {
        p
    }
}

/// Helper for overlay blend mode.
fn overlay_channel(base: f32, blend: f32) -> f32 {
    if base < 0.5 {
        2.0 * base * blend
    } else {
        1.0 - 2.0 * (1.0 - base) * (1.0 - blend)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rgb_hex_roundtrip() {
        let original = 0xFF5500;
        let color = LinearRgb::from_hex(original);
        let result = color.to_hex();

        // Allow small differences due to gamma conversion
        let diff = (original as i32 - result as i32).abs();
        assert!(diff < 0x020202);
    }

    #[test]
    fn test_hsl_roundtrip() {
        let original = LinearRgb::new(0.8, 0.4, 0.2);
        let hsl = original.to_hsl();
        let back = hsl.to_rgb();

        assert!((original.r - back.r).abs() < 0.01);
        assert!((original.g - back.g).abs() < 0.01);
        assert!((original.b - back.b).abs() < 0.01);
    }

    #[test]
    fn test_hsv_roundtrip() {
        let original = LinearRgb::new(0.8, 0.4, 0.2);
        let hsv = original.to_hsv();
        let back = hsv.to_rgb();

        assert!((original.r - back.r).abs() < 0.01);
        assert!((original.g - back.g).abs() < 0.01);
        assert!((original.b - back.b).abs() < 0.01);
    }

    #[test]
    fn test_gradient_two_color() {
        let grad = Gradient::two_color(LinearRgb::BLACK, LinearRgb::WHITE);

        let start = grad.sample_rgb(0.0);
        assert!((start.r - 0.0).abs() < 0.01);

        let end = grad.sample_rgb(1.0);
        assert!((end.r - 1.0).abs() < 0.01);

        let mid = grad.sample_rgb(0.5);
        assert!((mid.r - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_gradient_multi_stop() {
        let grad = Gradient::from_colors(&[LinearRgb::RED, LinearRgb::GREEN, LinearRgb::BLUE]);

        // At 0, should be red
        let c0 = grad.sample_rgb(0.0);
        assert!((c0.r - 1.0).abs() < 0.01);

        // At 0.5, should be green
        let c50 = grad.sample_rgb(0.5);
        assert!((c50.g - 1.0).abs() < 0.01);

        // At 1.0, should be blue
        let c100 = grad.sample_rgb(1.0);
        assert!((c100.b - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_blend_modes() {
        let base = LinearRgb::new(0.5, 0.5, 0.5);
        let blend_color = LinearRgb::new(0.8, 0.2, 0.4);

        // Multiply
        let mult = blend(base, blend_color, BlendMode::Multiply);
        assert!((mult.r - 0.4).abs() < 0.01);

        // Add
        let add = blend(base, blend_color, BlendMode::Add);
        assert!((add.r - 1.0).abs() < 0.01); // clamped

        // Difference
        let diff = blend(base, blend_color, BlendMode::Difference);
        assert!((diff.r - 0.3).abs() < 0.01);
    }

    #[test]
    fn test_hsl_lerp_hue_wrap() {
        // Test that hue interpolation takes the short path
        let red = Hsl::new(0.0, 1.0, 0.5);
        let magenta = Hsl::new(0.9, 1.0, 0.5);

        let mid = red.lerp(magenta, 0.5);
        // Should go through 0.95 (wrap around), not 0.45
        assert!(mid.h > 0.9 || mid.h < 0.1);
    }

    #[test]
    fn test_luminance() {
        // White should have luminance ~1.0
        assert!((LinearRgb::WHITE.luminance() - 1.0).abs() < 0.01);

        // Black should have luminance ~0.0
        assert!(LinearRgb::BLACK.luminance() < 0.01);

        // Green has highest luminance weight
        let green_lum = LinearRgb::GREEN.luminance();
        let red_lum = LinearRgb::RED.luminance();
        assert!(green_lum > red_lum);
    }

    #[test]
    fn test_preset_gradients() {
        // Just verify they don't panic
        let _ = presets::grayscale().sample_rgb(0.5);
        let _ = presets::rainbow().sample_rgb(0.5);
        let _ = presets::heat().sample_rgb(0.5);
        let _ = presets::viridis().sample_rgb(0.5);
        let _ = presets::inferno().sample_rgb(0.5);
    }

    #[test]
    fn test_lerp_trait() {
        // Test LinearRgb Lerp
        let a = LinearRgb::BLACK;
        let b = LinearRgb::WHITE;
        let mid = a.lerp_to(&b, 0.5);
        assert!((mid.r - 0.5).abs() < 0.01);
        assert!((mid.g - 0.5).abs() < 0.01);
        assert!((mid.b - 0.5).abs() < 0.01);

        // Test Rgba Lerp
        let a = Rgba::new(0.0, 0.0, 0.0, 0.0);
        let b = Rgba::new(1.0, 1.0, 1.0, 1.0);
        let mid = a.lerp_to(&b, 0.5);
        assert!((mid.r - 0.5).abs() < 0.01);
        assert!((mid.a - 0.5).abs() < 0.01);

        // Test Hsl Lerp (with hue wrapping)
        let a = Hsl::new(0.1, 0.5, 0.5);
        let b = Hsl::new(0.3, 0.5, 0.5);
        let mid = a.lerp_to(&b, 0.5);
        assert!((mid.h - 0.2).abs() < 0.01);

        // Test Hsv Lerp
        let a = Hsv::new(0.0, 0.0, 0.0);
        let b = Hsv::new(0.5, 1.0, 1.0);
        let mid = a.lerp_to(&b, 0.5);
        assert!((mid.h - 0.25).abs() < 0.01);
        assert!((mid.s - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_array_conversions() {
        // LinearRgb <-> [f32; 3]
        let arr = [0.2, 0.4, 0.6];
        let color: LinearRgb = arr.into();
        assert!((color.r - 0.2).abs() < 0.001);
        assert!((color.g - 0.4).abs() < 0.001);
        assert!((color.b - 0.6).abs() < 0.001);

        let back: [f32; 3] = color.into();
        assert!((back[0] - 0.2).abs() < 0.001);
        assert!((back[1] - 0.4).abs() < 0.001);
        assert!((back[2] - 0.6).abs() < 0.001);

        // Rgba <-> [f32; 4]
        let arr = [0.1, 0.2, 0.3, 0.4];
        let color: Rgba = arr.into();
        assert!((color.r - 0.1).abs() < 0.001);
        assert!((color.g - 0.2).abs() < 0.001);
        assert!((color.b - 0.3).abs() < 0.001);
        assert!((color.a - 0.4).abs() < 0.001);

        let back: [f32; 4] = color.into();
        assert!((back[0] - 0.1).abs() < 0.001);
        assert!((back[1] - 0.2).abs() < 0.001);
        assert!((back[2] - 0.3).abs() < 0.001);
        assert!((back[3] - 0.4).abs() < 0.001);
    }
}
