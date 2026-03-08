#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::ImageField;
use crate::channel::Channel;

/// Integer range specification for pixel value conversion.
///
/// Different color spaces and use cases require different integer ranges:
/// - RGB channels: 0-255 (u8)
/// - Hue values: 0-359 (degrees)
/// - Custom ranges for specific quantization schemes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum IntRange {
    /// Standard 8-bit range (0-255). Most common for RGB.
    U8,
    /// Hue in degrees (0-359). For HSL/HSV/LCH/OkLCH operations.
    Hue360,
    /// Custom range with min and max values.
    Custom { min: i32, max: i32 },
}

impl IntRange {
    /// Returns the minimum value of this range.
    pub fn min(&self) -> i32 {
        match self {
            IntRange::U8 => 0,
            IntRange::Hue360 => 0,
            IntRange::Custom { min, .. } => *min,
        }
    }

    /// Returns the maximum value of this range.
    pub fn max(&self) -> i32 {
        match self {
            IntRange::U8 => 255,
            IntRange::Hue360 => 359,
            IntRange::Custom { max, .. } => *max,
        }
    }

    /// Returns the span of this range (max - min).
    pub fn span(&self) -> i32 {
        self.max() - self.min()
    }
}

/// Converts a floating-point image (0-1) to integer representation.
///
/// This is the foundation for bit-level manipulation. After conversion,
/// you can use `IntColorExpr` for bitwise operations.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ToInt {
    /// The integer range to convert to.
    pub range: IntRange,
    /// Which channel to convert (None = all channels).
    pub channel: Option<Channel>,
}

impl Default for ToInt {
    fn default() -> Self {
        Self {
            range: IntRange::U8,
            channel: None,
        }
    }
}

impl ToInt {
    /// Creates a ToInt conversion for standard 0-255 range.
    pub fn u8() -> Self {
        Self::default()
    }

    /// Creates a ToInt conversion for hue (0-359).
    pub fn hue() -> Self {
        Self {
            range: IntRange::Hue360,
            channel: None,
        }
    }

    /// Creates a ToInt conversion with a custom range.
    pub fn custom(min: i32, max: i32) -> Self {
        Self {
            range: IntRange::Custom { min, max },
            channel: None,
        }
    }

    /// Restricts conversion to a specific channel.
    pub fn with_channel(mut self, channel: Channel) -> Self {
        self.channel = Some(channel);
        self
    }

    /// Applies the conversion to a single value.
    #[inline]
    pub fn convert(&self, value: f32) -> i32 {
        let min = self.range.min();
        let span = self.range.span() as f32;
        let scaled = value.clamp(0.0, 1.0) * span + min as f32;
        scaled.round() as i32
    }

    /// Applies this operation to an image, returning integer pixel data.
    ///
    /// Note: This returns raw i32 data, not an ImageField. Use for
    /// intermediate processing before converting back with FromInt.
    pub fn apply(&self, image: &ImageField) -> Vec<[i32; 4]> {
        image
            .data
            .iter()
            .map(|pixel| {
                [
                    self.convert(pixel[0]),
                    self.convert(pixel[1]),
                    self.convert(pixel[2]),
                    self.convert(pixel[3]),
                ]
            })
            .collect()
    }
}

/// Converts integer pixel data back to floating-point image (0-1).
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct FromInt {
    /// The integer range to convert from.
    pub range: IntRange,
}

impl Default for FromInt {
    fn default() -> Self {
        Self {
            range: IntRange::U8,
        }
    }
}

impl FromInt {
    /// Creates a FromInt conversion for standard 0-255 range.
    pub fn u8() -> Self {
        Self::default()
    }

    /// Creates a FromInt conversion for hue (0-359).
    pub fn hue() -> Self {
        Self {
            range: IntRange::Hue360,
        }
    }

    /// Creates a FromInt conversion with a custom range.
    pub fn custom(min: i32, max: i32) -> Self {
        Self {
            range: IntRange::Custom { min, max },
        }
    }

    /// Applies the conversion to a single value.
    #[inline]
    pub fn convert(&self, value: i32) -> f32 {
        let min = self.range.min();
        let span = self.range.span() as f32;
        ((value - min) as f32 / span).clamp(0.0, 1.0)
    }

    /// Applies this operation to integer pixel data, returning an ImageField.
    pub fn apply(&self, data: &[[i32; 4]], width: u32, height: u32) -> ImageField {
        let float_data: Vec<[f32; 4]> = data
            .iter()
            .map(|pixel| {
                [
                    self.convert(pixel[0]),
                    self.convert(pixel[1]),
                    self.convert(pixel[2]),
                    self.convert(pixel[3]),
                ]
            })
            .collect();

        ImageField::from_raw(float_data, width, height)
    }
}

/// Extracts a single bit plane from an image channel.
///
/// Each pixel in the output is 0.0 or 1.0 based on whether the specified
/// bit is set in the source channel.
///
/// # Use Cases
/// - Steganography: extract hidden data from LSBs
/// - Glitch art: visualize bit planes
/// - Analysis: see contribution of each bit to final image
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ExtractBitPlane {
    /// Which channel to extract from.
    pub channel: Channel,
    /// Which bit to extract (0 = LSB, 7 = MSB for 8-bit).
    pub bit: u8,
}

impl ExtractBitPlane {
    /// Creates a bit plane extractor for the specified channel and bit.
    pub fn new(channel: Channel, bit: u8) -> Self {
        Self { channel, bit }
    }

    /// Extracts the MSB (most significant bit) of a channel.
    pub fn msb(channel: Channel) -> Self {
        Self { channel, bit: 7 }
    }

    /// Extracts the LSB (least significant bit) of a channel.
    pub fn lsb(channel: Channel) -> Self {
        Self { channel, bit: 0 }
    }

    /// Applies this operation to an image.
    pub fn apply(&self, image: &ImageField) -> ImageField {
        let channel_idx = match self.channel {
            Channel::Red => 0,
            Channel::Green => 1,
            Channel::Blue => 2,
            Channel::Alpha => 3,
        };
        let mask = 1u8 << self.bit;

        let data: Vec<[f32; 4]> = image
            .data
            .iter()
            .map(|pixel| {
                let byte = (pixel[channel_idx].clamp(0.0, 1.0) * 255.0) as u8;
                let bit_value = if (byte & mask) != 0 { 1.0 } else { 0.0 };
                [bit_value, bit_value, bit_value, 1.0]
            })
            .collect();

        ImageField::from_raw(data, image.width, image.height)
            .with_wrap_mode(image.wrap_mode)
            .with_filter_mode(image.filter_mode)
    }
}

/// Sets a single bit plane in an image channel from a source image.
///
/// The source image is treated as binary (>0.5 = 1, <=0.5 = 0).
///
/// # Use Cases
/// - Steganography: embed data in LSBs
/// - Glitch art: replace bit planes between images
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct SetBitPlane {
    /// Which channel to modify.
    pub channel: Channel,
    /// Which bit to set (0 = LSB, 7 = MSB for 8-bit).
    pub bit: u8,
}

impl SetBitPlane {
    /// Creates a bit plane setter for the specified channel and bit.
    pub fn new(channel: Channel, bit: u8) -> Self {
        Self { channel, bit }
    }

    /// Sets the MSB (most significant bit) of a channel.
    pub fn msb(channel: Channel) -> Self {
        Self { channel, bit: 7 }
    }

    /// Sets the LSB (least significant bit) of a channel.
    pub fn lsb(channel: Channel) -> Self {
        Self { channel, bit: 0 }
    }

    /// Applies this operation to an image, using source for bit values.
    ///
    /// Source values > 0.5 set the bit to 1, otherwise 0.
    pub fn apply(&self, image: &ImageField, source: &ImageField) -> ImageField {
        let channel_idx = match self.channel {
            Channel::Red => 0,
            Channel::Green => 1,
            Channel::Blue => 2,
            Channel::Alpha => 3,
        };
        let mask = 1u8 << self.bit;
        let inv_mask = !mask;

        let data: Vec<[f32; 4]> = image
            .data
            .iter()
            .enumerate()
            .map(|(i, pixel)| {
                let x = (i as u32) % image.width;
                let y = (i as u32) / image.width;
                let src_pixel = source.get_pixel(x, y);

                // Get source bit value (treat as binary: >0.5 = 1)
                let src_bit = if src_pixel[0] > 0.5 { mask } else { 0 };

                // Get current byte value
                let byte = (pixel[channel_idx].clamp(0.0, 1.0) * 255.0) as u8;

                // Clear the bit and set from source
                let new_byte = (byte & inv_mask) | src_bit;

                let mut result = *pixel;
                result[channel_idx] = new_byte as f32 / 255.0;
                result
            })
            .collect();

        ImageField::from_raw(data, image.width, image.height)
            .with_wrap_mode(image.wrap_mode)
            .with_filter_mode(image.filter_mode)
    }
}

// Note: LsbEmbed was intentionally not included as it's not a primitive.
// It's a composition of ExtractBitPlane/SetBitPlane loops over pixels.
// Users can compose these primitives to build their own embedding schemes.
// See docs/archive/decomposition-audit.md for the principle of identifying true primitives.

/// An image stored as `[u8; 4]` per pixel (RGBA, 0–255 per channel).
///
/// This is the integer counterpart to [`ImageField`], suited for bit-level
/// manipulation, steganography, compression analysis, and glitch art.
///
/// # Conversion
///
/// Use [`ImageFieldU8::from_image_field`] and [`ImageFieldU8::to_image_field`]
/// to convert between the two representations.
#[derive(Debug, Clone)]
pub struct ImageFieldU8 {
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
    /// Raw pixel data in row-major order, RGBA 0–255.
    pub pixels: Vec<[u8; 4]>,
}

impl ImageFieldU8 {
    /// Creates a new blank (all-zero) image.
    pub fn new(width: u32, height: u32) -> Self {
        let pixels = vec![[0u8; 4]; (width * height) as usize];
        Self {
            width,
            height,
            pixels,
        }
    }

    /// Creates an `ImageFieldU8` from raw pixel data.
    ///
    /// # Panics
    ///
    /// Panics if `pixels.len() != width * height`.
    pub fn from_raw(pixels: Vec<[u8; 4]>, width: u32, height: u32) -> Self {
        assert_eq!(
            pixels.len(),
            (width * height) as usize,
            "pixel count does not match dimensions"
        );
        Self {
            width,
            height,
            pixels,
        }
    }

    /// Converts an [`ImageField`] (f32 0–1) to `ImageFieldU8` (u8 0–255).
    pub fn from_image_field(image: &ImageField) -> Self {
        let pixels = image
            .data
            .iter()
            .map(|px| {
                [
                    (px[0].clamp(0.0, 1.0) * 255.0).round() as u8,
                    (px[1].clamp(0.0, 1.0) * 255.0).round() as u8,
                    (px[2].clamp(0.0, 1.0) * 255.0).round() as u8,
                    (px[3].clamp(0.0, 1.0) * 255.0).round() as u8,
                ]
            })
            .collect();
        Self {
            width: image.width,
            height: image.height,
            pixels,
        }
    }

    /// Converts this `ImageFieldU8` back to an [`ImageField`] (f32 0–1).
    pub fn to_image_field(&self) -> ImageField {
        let data = self
            .pixels
            .iter()
            .map(|px| {
                [
                    px[0] as f32 / 255.0,
                    px[1] as f32 / 255.0,
                    px[2] as f32 / 255.0,
                    px[3] as f32 / 255.0,
                ]
            })
            .collect();
        ImageField::from_raw(data, self.width, self.height)
    }

    /// Returns the pixel at `(x, y)`.
    ///
    /// # Panics
    ///
    /// Panics if `x >= width` or `y >= height`.
    pub fn get(&self, x: u32, y: u32) -> [u8; 4] {
        self.pixels[(y * self.width + x) as usize]
    }

    /// Sets the pixel at `(x, y)`.
    ///
    /// # Panics
    ///
    /// Panics if `x >= width` or `y >= height`.
    pub fn set(&mut self, x: u32, y: u32, pixel: [u8; 4]) {
        self.pixels[(y * self.width + x) as usize] = pixel;
    }

    /// Returns image dimensions as `(width, height)`.
    pub fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// Applies a per-pixel transform, returning a new image.
    pub fn map(&self, f: impl Fn([u8; 4]) -> [u8; 4]) -> Self {
        let pixels = self.pixels.iter().map(|&px| f(px)).collect();
        Self {
            width: self.width,
            height: self.height,
            pixels,
        }
    }

    /// Applies a per-pixel transform combining two same-size images.
    ///
    /// # Panics
    ///
    /// Panics if `self` and `other` have different dimensions.
    pub fn zip_map(&self, other: &Self, f: impl Fn([u8; 4], [u8; 4]) -> [u8; 4]) -> Self {
        assert_eq!(
            (self.width, self.height),
            (other.width, other.height),
            "image dimensions must match for zip_map"
        );
        let pixels = self
            .pixels
            .iter()
            .zip(other.pixels.iter())
            .map(|(&a, &b)| f(a, b))
            .collect();
        Self {
            width: self.width,
            height: self.height,
            pixels,
        }
    }
}

/// A typed expression AST for integer pixel manipulation (`[u8; 4]` → `[u8; 4]`).
///
/// This is the integer counterpart to `ColorExpr`, suited for bit-level operations
/// like steganography, glitch art, and compression analysis.
///
/// Channels are named `r`, `g`, `b`, `a` and hold u8 values (0–255). All
/// arithmetic wraps on overflow to match u8 semantics.
///
/// # Example
///
/// ```
/// use unshape_image::{ImageFieldU8, IntColorExpr};
///
/// // XOR all channels with 0xFF (invert bits)
/// let invert = IntColorExpr::Vec4 {
///     r: Box::new(IntColorExpr::BitXor(
///         Box::new(IntColorExpr::R),
///         Box::new(IntColorExpr::Constant(0xFF)),
///     )),
///     g: Box::new(IntColorExpr::BitXor(
///         Box::new(IntColorExpr::G),
///         Box::new(IntColorExpr::Constant(0xFF)),
///     )),
///     b: Box::new(IntColorExpr::BitXor(
///         Box::new(IntColorExpr::B),
///         Box::new(IntColorExpr::Constant(0xFF)),
///     )),
///     a: Box::new(IntColorExpr::A),
/// };
///
/// let img = ImageFieldU8::from_raw(vec![[0x12, 0x34, 0x56, 0xFF]; 4], 2, 2);
/// let result = invert.apply(&img);
/// assert_eq!(result.get(0, 0), [0xED, 0xCB, 0xA9, 0xFF]);
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum IntColorExpr {
    // === Channel inputs ===
    /// Red channel of the current pixel.
    R,
    /// Green channel of the current pixel.
    G,
    /// Blue channel of the current pixel.
    B,
    /// Alpha channel of the current pixel.
    A,

    // === Constant ===
    /// A constant u8 value.
    Constant(u8),

    // === Output constructors ===
    /// Construct a `[u8; 4]` output from four channel expressions.
    Vec4 {
        r: Box<IntColorExpr>,
        g: Box<IntColorExpr>,
        b: Box<IntColorExpr>,
        a: Box<IntColorExpr>,
    },

    // === Arithmetic (wrapping) ===
    /// Wrapping addition.
    Add(Box<IntColorExpr>, Box<IntColorExpr>),
    /// Wrapping subtraction.
    Sub(Box<IntColorExpr>, Box<IntColorExpr>),
    /// Wrapping multiplication.
    Mul(Box<IntColorExpr>, Box<IntColorExpr>),
    /// Integer division (saturates; division by zero yields 0).
    Div(Box<IntColorExpr>, Box<IntColorExpr>),

    // === Bitwise ops ===
    /// Bitwise AND.
    BitAnd(Box<IntColorExpr>, Box<IntColorExpr>),
    /// Bitwise OR.
    BitOr(Box<IntColorExpr>, Box<IntColorExpr>),
    /// Bitwise XOR.
    BitXor(Box<IntColorExpr>, Box<IntColorExpr>),
    /// Bitwise NOT.
    BitNot(Box<IntColorExpr>),
    /// Logical right shift (shift count is taken mod 8).
    Shr(Box<IntColorExpr>, Box<IntColorExpr>),
    /// Left shift (shift count is taken mod 8).
    Shl(Box<IntColorExpr>, Box<IntColorExpr>),

    // === Comparisons (return 0xFF if true, 0x00 if false) ===
    /// Returns 0xFF if left > right, else 0x00.
    Gt(Box<IntColorExpr>, Box<IntColorExpr>),
    /// Returns 0xFF if left < right, else 0x00.
    Lt(Box<IntColorExpr>, Box<IntColorExpr>),
    /// Returns 0xFF if left == right, else 0x00.
    Eq(Box<IntColorExpr>, Box<IntColorExpr>),

    // === Selection ===
    /// `if condition != 0 { then_expr } else { else_expr }`
    IfNonZero {
        condition: Box<IntColorExpr>,
        then_expr: Box<IntColorExpr>,
        else_expr: Box<IntColorExpr>,
    },

    // === Min / Max ===
    /// Minimum of two values.
    Min(Box<IntColorExpr>, Box<IntColorExpr>),
    /// Maximum of two values.
    Max(Box<IntColorExpr>, Box<IntColorExpr>),
}

impl IntColorExpr {
    /// Evaluates the expression for a single pixel `[r, g, b, a]`.
    ///
    /// Returns the scalar result; use [`IntColorExpr::Vec4`] at the top level
    /// to produce a `[u8; 4]` output, or call [`IntColorExpr::apply`].
    pub fn eval(&self, px: [u8; 4]) -> u8 {
        let [r, g, b, a] = px;
        match self {
            Self::R => r,
            Self::G => g,
            Self::B => b,
            Self::A => a,
            Self::Constant(c) => *c,
            // Vec4 — pick the red component as the scalar result for inner use
            Self::Vec4 { r: re, .. } => re.eval(px),
            Self::Add(lhs, rhs) => lhs.eval(px).wrapping_add(rhs.eval(px)),
            Self::Sub(lhs, rhs) => lhs.eval(px).wrapping_sub(rhs.eval(px)),
            Self::Mul(lhs, rhs) => lhs.eval(px).wrapping_mul(rhs.eval(px)),
            Self::Div(lhs, rhs) => {
                let d = rhs.eval(px);
                if d == 0 { 0 } else { lhs.eval(px) / d }
            }
            Self::BitAnd(lhs, rhs) => lhs.eval(px) & rhs.eval(px),
            Self::BitOr(lhs, rhs) => lhs.eval(px) | rhs.eval(px),
            Self::BitXor(lhs, rhs) => lhs.eval(px) ^ rhs.eval(px),
            Self::BitNot(inner) => !inner.eval(px),
            Self::Shr(val, shift) => val.eval(px) >> (shift.eval(px) % 8),
            Self::Shl(val, shift) => val.eval(px) << (shift.eval(px) % 8),
            Self::Gt(lhs, rhs) => {
                if lhs.eval(px) > rhs.eval(px) {
                    0xFF
                } else {
                    0x00
                }
            }
            Self::Lt(lhs, rhs) => {
                if lhs.eval(px) < rhs.eval(px) {
                    0xFF
                } else {
                    0x00
                }
            }
            Self::Eq(lhs, rhs) => {
                if lhs.eval(px) == rhs.eval(px) {
                    0xFF
                } else {
                    0x00
                }
            }
            Self::IfNonZero {
                condition,
                then_expr,
                else_expr,
            } => {
                if condition.eval(px) != 0 {
                    then_expr.eval(px)
                } else {
                    else_expr.eval(px)
                }
            }
            Self::Min(lhs, rhs) => lhs.eval(px).min(rhs.eval(px)),
            Self::Max(lhs, rhs) => lhs.eval(px).max(rhs.eval(px)),
        }
    }

    /// Evaluates this expression as a `[u8; 4]` output for a single pixel.
    ///
    /// If the top-level node is [`IntColorExpr::Vec4`], all four channels are
    /// evaluated independently. Otherwise the scalar result is broadcast to all
    /// four channels (alpha is set to 0xFF).
    pub fn eval_pixel(&self, px: [u8; 4]) -> [u8; 4] {
        match self {
            Self::Vec4 { r, g, b, a } => [r.eval(px), g.eval(px), b.eval(px), a.eval(px)],
            other => {
                let v = other.eval(px);
                [v, v, v, 0xFF]
            }
        }
    }

    /// Applies this expression to every pixel of an [`ImageFieldU8`].
    pub fn apply(&self, image: &ImageFieldU8) -> ImageFieldU8 {
        image.map(|px| self.eval_pixel(px))
    }

    // === Convenience constructors ===

    /// Returns pixels unchanged.
    pub fn identity() -> Self {
        Self::Vec4 {
            r: Box::new(Self::R),
            g: Box::new(Self::G),
            b: Box::new(Self::B),
            a: Box::new(Self::A),
        }
    }

    /// Inverts all RGB channels; preserves alpha.
    pub fn invert_rgb() -> Self {
        Self::Vec4 {
            r: Box::new(Self::BitNot(Box::new(Self::R))),
            g: Box::new(Self::BitNot(Box::new(Self::G))),
            b: Box::new(Self::BitNot(Box::new(Self::B))),
            a: Box::new(Self::A),
        }
    }

    /// Extracts bit `bit` (0 = LSB, 7 = MSB) of channel `channel_idx` (0=R,
    /// 1=G, 2=B, 3=A) as 0x00 / 0xFF, broadcast to all channels.
    pub fn extract_bit(channel_idx: u8, bit: u8) -> Self {
        let channel = match channel_idx {
            0 => Self::R,
            1 => Self::G,
            2 => Self::B,
            _ => Self::A,
        };
        let extracted = Self::BitAnd(
            Box::new(Self::Shr(Box::new(channel), Box::new(Self::Constant(bit)))),
            Box::new(Self::Constant(1)),
        );
        // Scale 0/1 → 0x00/0xFF
        let scaled = Self::Mul(Box::new(extracted), Box::new(Self::Constant(0xFF)));
        Self::Vec4 {
            r: Box::new(scaled.clone()),
            g: Box::new(scaled.clone()),
            b: Box::new(scaled.clone()),
            a: Box::new(Self::Constant(0xFF)),
        }
    }
}
