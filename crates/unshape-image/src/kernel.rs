#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use unshape_color::BlendMode;

use crate::ImageField;
use crate::composite::composite_impl;
use crate::expr::{ColorExpr, UvExpr, map_pixels_impl, remap_uv_impl};
use crate::pyramid::resize_impl;

/// A convolution kernel for image filtering.
///
/// Kernels are square matrices of odd dimensions (3x3, 5x5, etc.).
#[derive(Debug, Clone)]
pub struct Kernel {
    /// Kernel weights in row-major order.
    pub weights: Vec<f32>,
    /// Kernel size (width and height).
    pub size: usize,
}

impl Kernel {
    /// Creates a new kernel from weights.
    ///
    /// The weights array must have length `size * size`.
    pub fn new(weights: Vec<f32>, size: usize) -> Self {
        assert_eq!(
            weights.len(),
            size * size,
            "Kernel weights must be size*size"
        );
        assert!(size % 2 == 1, "Kernel size must be odd");
        Self { weights, size }
    }

    /// Creates a 3x3 identity kernel (no change).
    pub fn identity() -> Self {
        Self::new(vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], 3)
    }

    /// Creates a 3x3 box blur kernel.
    pub fn box_blur() -> Self {
        let w = 1.0 / 9.0;
        Self::new(vec![w, w, w, w, w, w, w, w, w], 3)
    }

    /// Creates a 3x3 Gaussian blur kernel.
    pub fn gaussian_blur_3x3() -> Self {
        let weights = vec![
            1.0 / 16.0,
            2.0 / 16.0,
            1.0 / 16.0,
            2.0 / 16.0,
            4.0 / 16.0,
            2.0 / 16.0,
            1.0 / 16.0,
            2.0 / 16.0,
            1.0 / 16.0,
        ];
        Self::new(weights, 3)
    }

    /// Creates a 5x5 Gaussian blur kernel.
    pub fn gaussian_blur_5x5() -> Self {
        let weights = vec![
            1.0, 4.0, 6.0, 4.0, 1.0, 4.0, 16.0, 24.0, 16.0, 4.0, 6.0, 24.0, 36.0, 24.0, 6.0, 4.0,
            16.0, 24.0, 16.0, 4.0, 1.0, 4.0, 6.0, 4.0, 1.0,
        ]
        .iter()
        .map(|&x| x / 256.0)
        .collect();
        Self::new(weights, 5)
    }

    /// Creates a 3x3 sharpen kernel.
    pub fn sharpen() -> Self {
        Self::new(vec![0.0, -1.0, 0.0, -1.0, 5.0, -1.0, 0.0, -1.0, 0.0], 3)
    }

    /// Creates a 3x3 unsharp mask kernel.
    pub fn unsharp_mask() -> Self {
        Self::new(
            vec![
                -1.0 / 9.0,
                -1.0 / 9.0,
                -1.0 / 9.0,
                -1.0 / 9.0,
                17.0 / 9.0,
                -1.0 / 9.0,
                -1.0 / 9.0,
                -1.0 / 9.0,
                -1.0 / 9.0,
            ],
            3,
        )
    }

    /// Creates a 3x3 Sobel edge detection kernel (horizontal edges).
    pub fn sobel_horizontal() -> Self {
        Self::new(vec![-1.0, -2.0, -1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 1.0], 3)
    }

    /// Creates a 3x3 Sobel edge detection kernel (vertical edges).
    pub fn sobel_vertical() -> Self {
        Self::new(vec![-1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -1.0, 0.0, 1.0], 3)
    }

    /// Creates a 3x3 Laplacian edge detection kernel.
    pub fn laplacian() -> Self {
        Self::new(vec![0.0, -1.0, 0.0, -1.0, 4.0, -1.0, 0.0, -1.0, 0.0], 3)
    }

    /// Creates a 3x3 Laplacian of Gaussian (LoG) approximation.
    pub fn laplacian_of_gaussian() -> Self {
        Self::new(vec![-1.0, -1.0, -1.0, -1.0, 8.0, -1.0, -1.0, -1.0, -1.0], 3)
    }

    /// Creates a 3x3 emboss kernel.
    pub fn emboss() -> Self {
        Self::new(vec![-2.0, -1.0, 0.0, -1.0, 1.0, 1.0, 0.0, 1.0, 2.0], 3)
    }

    /// Creates a 3x3 edge enhancement kernel.
    pub fn edge_enhance() -> Self {
        Self::new(vec![0.0, 0.0, 0.0, -1.0, 1.0, 0.0, 0.0, 0.0, 0.0], 3)
    }

    /// Returns the kernel radius (distance from center to edge).
    pub fn radius(&self) -> usize {
        self.size / 2
    }
}

/// Applies a convolution kernel to an image.
///
/// # Example
///
/// ```
/// use unshape_image::{ImageField, Kernel, convolve};
/// use unshape_color::Rgba;
///
/// // Create a simple 3x3 test image
/// let data = vec![
///     [0.5, 0.5, 0.5, 1.0]; 9
/// ];
/// let img = ImageField::from_raw(data, 3, 3);
///
/// let blurred = convolve(&img, &Kernel::box_blur());
/// ```
pub fn convolve(image: &ImageField, kernel: &Kernel) -> ImageField {
    Convolve::new(kernel.clone()).apply(image)
}

/// 2D spatial convolution operation.
///
/// A true image primitive - neighborhood operations cannot be decomposed further.
/// All blur, sharpen, and edge detection effects are implemented via this primitive.
///
/// # Example
///
/// ```
/// use unshape_image::{ImageField, Kernel, Convolve};
///
/// let img = ImageField::from_raw(vec![[0.5, 0.5, 0.5, 1.0]; 9], 3, 3);
/// let blur = Convolve::new(Kernel::box_blur());
/// let blurred = blur.apply(&img);
/// ```
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Convolve {
    /// The convolution kernel.
    pub kernel: Kernel,
}

impl Convolve {
    /// Creates a new convolution operation.
    pub fn new(kernel: Kernel) -> Self {
        Self { kernel }
    }

    /// Applies the convolution to an image.
    pub fn apply(&self, image: &ImageField) -> ImageField {
        convolve_impl(image, &self.kernel)
    }
}

/// Internal convolution implementation.
fn convolve_impl(image: &ImageField, kernel: &Kernel) -> ImageField {
    let (width, height) = image.dimensions();
    let radius = kernel.radius() as i32;
    let mut data = Vec::with_capacity((width * height) as usize);

    for y in 0..height {
        for x in 0..width {
            let mut r = 0.0;
            let mut g = 0.0;
            let mut b = 0.0;
            let mut a = 0.0;

            for ky in 0..kernel.size {
                for kx in 0..kernel.size {
                    let weight = kernel.weights[ky * kernel.size + kx];

                    // Sample with clamping at edges
                    let sx = (x as i32 + kx as i32 - radius).clamp(0, width as i32 - 1) as u32;
                    let sy = (y as i32 + ky as i32 - radius).clamp(0, height as i32 - 1) as u32;

                    let pixel = image.get_pixel(sx, sy);
                    r += pixel[0] * weight;
                    g += pixel[1] * weight;
                    b += pixel[2] * weight;
                    a += pixel[3] * weight;
                }
            }

            data.push([r, g, b, a]);
        }
    }

    ImageField::from_raw(data, width, height)
        .with_wrap_mode(image.wrap_mode)
        .with_filter_mode(image.filter_mode)
}

/// Image resizing operation.
///
/// A true image primitive - resampling requires interpolation which cannot
/// be decomposed further.
///
/// # Example
///
/// ```
/// use unshape_image::{ImageField, Resize};
///
/// let img = ImageField::from_raw(vec![[0.5, 0.5, 0.5, 1.0]; 16], 4, 4);
/// let resize = Resize::new(8, 6);
/// let resized = resize.apply(&img);
/// assert_eq!(resized.dimensions(), (8, 6));
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Resize {
    /// Target width in pixels.
    pub width: u32,
    /// Target height in pixels.
    pub height: u32,
}

impl Resize {
    /// Creates a new resize operation.
    pub fn new(width: u32, height: u32) -> Self {
        Self { width, height }
    }

    /// Applies the resize to an image.
    pub fn apply(&self, image: &ImageField) -> ImageField {
        resize_impl(image, self.width, self.height)
    }
}

/// Image compositing operation.
///
/// A true image primitive - blending requires per-pixel blend mode computation
/// which cannot be decomposed further.
///
/// # Example
///
/// ```
/// use unshape_image::{ImageField, Composite};
/// use unshape_color::BlendMode;
///
/// let base = ImageField::from_raw(vec![[0.2, 0.3, 0.4, 1.0]; 16], 4, 4);
/// let overlay = ImageField::from_raw(vec![[1.0, 0.0, 0.0, 0.5]; 16], 4, 4);
///
/// let comp = Composite::new(BlendMode::Normal, 0.8);
/// let result = comp.apply(&base, &overlay);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Composite {
    /// The blend mode to use.
    pub mode: BlendMode,
    /// Opacity of the overlay (0.0 = transparent, 1.0 = opaque).
    pub opacity: f32,
}

impl Composite {
    /// Creates a new composite operation.
    pub fn new(mode: BlendMode, opacity: f32) -> Self {
        Self { mode, opacity }
    }

    /// Applies the composite operation.
    pub fn apply(&self, base: &ImageField, overlay: &ImageField) -> ImageField {
        composite_impl(base, overlay, self.mode, self.opacity)
    }
}

/// UV coordinate remapping operation.
///
/// A true image primitive - spatial transforms on pixel coordinates.
/// All geometric effects (rotation, scale, distortion, lens effects) are
/// implemented via this primitive.
///
/// # Example
///
/// ```
/// use unshape_image::{ImageField, UvExpr, RemapUv};
///
/// let img = ImageField::from_raw(vec![[0.5, 0.5, 0.5, 1.0]; 16], 4, 4);
///
/// // Rotate around center
/// let rotate = RemapUv::new(UvExpr::rotate_centered(0.5));
/// let rotated = rotate.apply(&img);
/// ```
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct RemapUv {
    /// The UV remapping expression.
    pub expr: UvExpr,
}

impl RemapUv {
    /// Creates a new UV remapping operation.
    pub fn new(expr: UvExpr) -> Self {
        Self { expr }
    }

    /// Applies the UV remapping to an image.
    pub fn apply(&self, image: &ImageField) -> ImageField {
        remap_uv_impl(image, &self.expr)
    }
}

/// Per-pixel color transform operation.
///
/// A true image primitive - per-pixel color computation.
/// All color effects (grayscale, invert, levels, curves, color grading) are
/// implemented via this primitive.
///
/// # Example
///
/// ```
/// use unshape_image::{ImageField, ColorExpr, MapPixels};
///
/// let img = ImageField::from_raw(vec![[0.5, 0.3, 0.7, 1.0]; 16], 4, 4);
///
/// // Convert to grayscale
/// let gray_op = MapPixels::new(ColorExpr::grayscale());
/// let gray = gray_op.apply(&img);
/// ```
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct MapPixels {
    /// The color transform expression.
    pub expr: ColorExpr,
}

impl MapPixels {
    /// Creates a new per-pixel color transform.
    pub fn new(expr: ColorExpr) -> Self {
        Self { expr }
    }

    /// Applies the color transform to an image.
    pub fn apply(&self, image: &ImageField) -> ImageField {
        map_pixels_impl(image, &self.expr)
    }
}

/// Applies Sobel edge detection and returns the edge magnitude.
///
/// Combines horizontal and vertical Sobel kernels to detect edges in all directions.
pub fn detect_edges(image: &ImageField) -> ImageField {
    let horizontal = convolve(image, &Kernel::sobel_horizontal());
    let vertical = convolve(image, &Kernel::sobel_vertical());

    let (width, height) = image.dimensions();
    let mut data = Vec::with_capacity((width * height) as usize);

    for y in 0..height {
        for x in 0..width {
            let h = horizontal.get_pixel(x, y);
            let v = vertical.get_pixel(x, y);

            // Compute magnitude for each channel
            let mag_r = (h[0] * h[0] + v[0] * v[0]).sqrt();
            let mag_g = (h[1] * h[1] + v[1] * v[1]).sqrt();
            let mag_b = (h[2] * h[2] + v[2] * v[2]).sqrt();

            // Average the channels for grayscale edge output
            let mag = (mag_r + mag_g + mag_b) / 3.0;
            data.push([mag, mag, mag, 1.0]);
        }
    }

    ImageField::from_raw(data, width, height)
}

/// Applies a Gaussian blur with the specified number of passes.
///
/// Multiple passes of a small kernel approximate a larger blur radius.
pub fn blur(image: &ImageField, passes: u32) -> ImageField {
    let kernel = Kernel::gaussian_blur_3x3();
    let mut result = image.clone();

    for _ in 0..passes {
        result = convolve(&result, &kernel);
    }

    result
}

/// Sharpens an image.
pub fn sharpen(image: &ImageField) -> ImageField {
    convolve(image, &Kernel::sharpen())
}

/// Applies emboss effect to an image.
pub fn emboss(image: &ImageField) -> ImageField {
    let embossed = convolve(image, &Kernel::emboss());

    // Normalize emboss output to visible range (add 0.5 bias)
    let (width, height) = embossed.dimensions();
    let mut data = Vec::with_capacity((width * height) as usize);

    for y in 0..height {
        for x in 0..width {
            let pixel = embossed.get_pixel(x, y);
            data.push([
                (pixel[0] + 0.5).clamp(0.0, 1.0),
                (pixel[1] + 0.5).clamp(0.0, 1.0),
                (pixel[2] + 0.5).clamp(0.0, 1.0),
                pixel[3].clamp(0.0, 1.0),
            ]);
        }
    }

    ImageField::from_raw(data, width, height)
}
