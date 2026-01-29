use crate::ImageField;
use crate::expr::{ColorExpr, map_pixels};

/// Which channel to extract or operate on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Channel {
    /// Red channel.
    Red,
    /// Green channel.
    Green,
    /// Blue channel.
    Blue,
    /// Alpha channel.
    Alpha,
}

/// Extracts a single channel as a grayscale image.
///
/// The extracted channel is stored in all RGB channels of the output,
/// with alpha set to 1.0.
///
/// # Example
///
/// ```
/// use unshape_image::{ImageField, Channel, extract_channel};
///
/// let data = vec![[1.0, 0.5, 0.25, 1.0]; 4];
/// let img = ImageField::from_raw(data, 2, 2);
///
/// let red = extract_channel(&img, Channel::Red);
/// assert_eq!(red.get_pixel(0, 0)[0], 1.0);
///
/// let green = extract_channel(&img, Channel::Green);
/// assert_eq!(green.get_pixel(0, 0)[0], 0.5);
/// ```
pub fn extract_channel(image: &ImageField, channel: Channel) -> ImageField {
    // Use ColorExpr to extract channel as grayscale
    let ch = match channel {
        Channel::Red => ColorExpr::R,
        Channel::Green => ColorExpr::G,
        Channel::Blue => ColorExpr::B,
        Channel::Alpha => ColorExpr::A,
    };
    let expr = ColorExpr::Vec4 {
        r: Box::new(ch.clone()),
        g: Box::new(ch.clone()),
        b: Box::new(ch),
        a: Box::new(ColorExpr::Constant(1.0)),
    };
    map_pixels(image, &expr)
}

/// Splits an image into separate R, G, B, A grayscale images.
///
/// Returns a tuple of (red, green, blue, alpha) images.
///
/// # Example
///
/// ```
/// use unshape_image::{ImageField, split_channels};
///
/// let data = vec![[1.0, 0.5, 0.25, 0.75]; 4];
/// let img = ImageField::from_raw(data, 2, 2);
///
/// let (r, g, b, a) = split_channels(&img);
/// assert_eq!(r.get_pixel(0, 0)[0], 1.0);
/// assert_eq!(g.get_pixel(0, 0)[0], 0.5);
/// assert_eq!(b.get_pixel(0, 0)[0], 0.25);
/// assert_eq!(a.get_pixel(0, 0)[0], 0.75);
/// ```
pub fn split_channels(image: &ImageField) -> (ImageField, ImageField, ImageField, ImageField) {
    (
        extract_channel(image, Channel::Red),
        extract_channel(image, Channel::Green),
        extract_channel(image, Channel::Blue),
        extract_channel(image, Channel::Alpha),
    )
}

/// Merges separate grayscale images into a single RGBA image.
///
/// Each input image's red channel is used as that channel's value.
///
/// # Example
///
/// ```
/// use unshape_image::{ImageField, merge_channels};
///
/// let r = ImageField::from_raw(vec![[1.0, 1.0, 1.0, 1.0]; 4], 2, 2);
/// let g = ImageField::from_raw(vec![[0.5, 0.5, 0.5, 1.0]; 4], 2, 2);
/// let b = ImageField::from_raw(vec![[0.25, 0.25, 0.25, 1.0]; 4], 2, 2);
/// let a = ImageField::from_raw(vec![[1.0, 1.0, 1.0, 1.0]; 4], 2, 2);
///
/// let merged = merge_channels(&r, &g, &b, &a);
/// let pixel = merged.get_pixel(0, 0);
/// assert_eq!(pixel[0], 1.0);
/// assert_eq!(pixel[1], 0.5);
/// assert_eq!(pixel[2], 0.25);
/// ```
pub fn merge_channels(
    red: &ImageField,
    green: &ImageField,
    blue: &ImageField,
    alpha: &ImageField,
) -> ImageField {
    let (width, height) = red.dimensions();
    let mut data = Vec::with_capacity((width * height) as usize);

    for y in 0..height {
        for x in 0..width {
            let r = red.get_pixel(x, y)[0];
            let g = green.get_pixel(x, y)[0];
            let b = blue.get_pixel(x, y)[0];
            let a = alpha.get_pixel(x, y)[0];
            data.push([r, g, b, a]);
        }
    }

    ImageField::from_raw(data, width, height)
        .with_wrap_mode(red.wrap_mode)
        .with_filter_mode(red.filter_mode)
}

/// Replaces a single channel in an image.
///
/// The source image's red channel is used as the replacement value.
///
/// # Example
///
/// ```
/// use unshape_image::{ImageField, Channel, set_channel};
///
/// let img = ImageField::from_raw(vec![[0.0, 0.0, 0.0, 1.0]; 4], 2, 2);
/// let new_red = ImageField::from_raw(vec![[1.0, 1.0, 1.0, 1.0]; 4], 2, 2);
///
/// let result = set_channel(&img, Channel::Red, &new_red);
/// assert_eq!(result.get_pixel(0, 0)[0], 1.0);
/// assert_eq!(result.get_pixel(0, 0)[1], 0.0); // Green unchanged
/// ```
pub fn set_channel(image: &ImageField, channel: Channel, source: &ImageField) -> ImageField {
    let (width, height) = image.dimensions();
    let mut data = Vec::with_capacity((width * height) as usize);

    let idx = match channel {
        Channel::Red => 0,
        Channel::Green => 1,
        Channel::Blue => 2,
        Channel::Alpha => 3,
    };

    for y in 0..height {
        for x in 0..width {
            let mut pixel = image.get_pixel(x, y);
            pixel[idx] = source.get_pixel(x, y)[0];
            data.push(pixel);
        }
    }

    ImageField::from_raw(data, width, height)
        .with_wrap_mode(image.wrap_mode)
        .with_filter_mode(image.filter_mode)
}

/// Applies a function to a single channel of an image.
///
/// This primitive extracts a channel as a grayscale image, transforms it,
/// and puts it back. Useful for per-channel effects like independent blur,
/// noise, or distortion.
///
/// # Arguments
///
/// * `image` - Source image
/// * `channel` - Channel to transform
/// * `f` - Function that transforms a grayscale `ImageField` and returns a new one
///
/// # Example
///
/// ```
/// use unshape_image::{ImageField, Channel, map_channel, convolve, Kernel};
///
/// let img = ImageField::from_raw(vec![[0.5, 0.3, 0.7, 1.0]; 64], 8, 8);
///
/// // Blur only the red channel
/// let result = map_channel(&img, Channel::Red, |ch| {
///     convolve(&ch, &Kernel::box_blur())
/// });
///
/// // Apply noise only to the blue channel
/// let result = map_channel(&img, Channel::Blue, |ch| {
///     // ch is a grayscale image of the blue channel
///     ch // return transformed channel
/// });
/// ```
pub fn map_channel(
    image: &ImageField,
    channel: Channel,
    f: impl FnOnce(ImageField) -> ImageField,
) -> ImageField {
    let extracted = extract_channel(image, channel);
    let transformed = f(extracted);
    set_channel(image, channel, &transformed)
}

/// Swaps two channels in an image.
///
/// # Example
///
/// ```
/// use unshape_image::{ImageField, Channel, swap_channels};
///
/// let img = ImageField::from_raw(vec![[1.0, 0.5, 0.0, 1.0]; 4], 2, 2);
/// let swapped = swap_channels(&img, Channel::Red, Channel::Blue);
///
/// assert_eq!(swapped.get_pixel(0, 0)[0], 0.0);  // Was blue
/// assert_eq!(swapped.get_pixel(0, 0)[2], 1.0);  // Was red
/// ```
pub fn swap_channels(image: &ImageField, a: Channel, b: Channel) -> ImageField {
    // Build channel expressions, swapping a and b
    let channel_expr = |ch: Channel| -> ColorExpr {
        if ch == a {
            // Use b's value for a's position
            match b {
                Channel::Red => ColorExpr::R,
                Channel::Green => ColorExpr::G,
                Channel::Blue => ColorExpr::B,
                Channel::Alpha => ColorExpr::A,
            }
        } else if ch == b {
            // Use a's value for b's position
            match a {
                Channel::Red => ColorExpr::R,
                Channel::Green => ColorExpr::G,
                Channel::Blue => ColorExpr::B,
                Channel::Alpha => ColorExpr::A,
            }
        } else {
            // Keep original
            match ch {
                Channel::Red => ColorExpr::R,
                Channel::Green => ColorExpr::G,
                Channel::Blue => ColorExpr::B,
                Channel::Alpha => ColorExpr::A,
            }
        }
    };

    let expr = ColorExpr::Vec4 {
        r: Box::new(channel_expr(Channel::Red)),
        g: Box::new(channel_expr(Channel::Green)),
        b: Box::new(channel_expr(Channel::Blue)),
        a: Box::new(channel_expr(Channel::Alpha)),
    };
    map_pixels(image, &expr)
}
