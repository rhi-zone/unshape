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
