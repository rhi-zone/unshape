use crate::kernel::{Resize, blur};
use crate::{FilterMode, ImageField};

/// Downsamples an image by half using box filtering (averaging 2x2 blocks).
///
/// The output dimensions are `(width / 2, height / 2)`.
/// If dimensions are odd, the last row/column is included in the final average.
///
/// # Example
///
/// ```
/// use unshape_image::{ImageField, downsample};
///
/// let data = vec![[0.5, 0.5, 0.5, 1.0]; 16];
/// let img = ImageField::from_raw(data, 4, 4);
///
/// let half = downsample(&img);
/// assert_eq!(half.dimensions(), (2, 2));
/// ```
pub fn downsample(image: &ImageField) -> ImageField {
    let (width, height) = image.dimensions();
    let new_width = (width / 2).max(1);
    let new_height = (height / 2).max(1);

    let mut data = Vec::with_capacity((new_width * new_height) as usize);

    for y in 0..new_height {
        for x in 0..new_width {
            // Average 2x2 block
            let x0 = x * 2;
            let y0 = y * 2;
            let x1 = (x0 + 1).min(width - 1);
            let y1 = (y0 + 1).min(height - 1);

            let p00 = image.get_pixel(x0, y0);
            let p10 = image.get_pixel(x1, y0);
            let p01 = image.get_pixel(x0, y1);
            let p11 = image.get_pixel(x1, y1);

            let avg = [
                (p00[0] + p10[0] + p01[0] + p11[0]) / 4.0,
                (p00[1] + p10[1] + p01[1] + p11[1]) / 4.0,
                (p00[2] + p10[2] + p01[2] + p11[2]) / 4.0,
                (p00[3] + p10[3] + p01[3] + p11[3]) / 4.0,
            ];

            data.push(avg);
        }
    }

    ImageField::from_raw(data, new_width, new_height)
        .with_wrap_mode(image.wrap_mode)
        .with_filter_mode(image.filter_mode)
}

/// Upsamples an image by 2x using bilinear interpolation.
///
/// The output dimensions are `(width * 2, height * 2)`.
///
/// # Deprecation
///
/// This function is equivalent to `Resize::new(w * 2, h * 2).apply(image)`.
/// Consider using [`Resize`] directly for more control over dimensions.
///
/// # Example
///
/// ```
/// use unshape_image::{ImageField, upsample};
///
/// let data = vec![[0.5, 0.5, 0.5, 1.0]; 4];
/// let img = ImageField::from_raw(data, 2, 2);
///
/// let double = upsample(&img);
/// assert_eq!(double.dimensions(), (4, 4));
/// ```
#[deprecated(
    since = "0.2.0",
    note = "Use Resize::new(w * 2, h * 2).apply(image) instead"
)]
pub fn upsample(image: &ImageField) -> ImageField {
    let (width, height) = image.dimensions();
    let new_width = width * 2;
    let new_height = height * 2;

    let mut data = Vec::with_capacity((new_width * new_height) as usize);

    // Use bilinear sampling
    let bilinear_image = ImageField {
        filter_mode: FilterMode::Bilinear,
        ..image.clone()
    };

    for y in 0..new_height {
        for x in 0..new_width {
            // Map to source coordinates
            let u = (x as f32 + 0.5) / new_width as f32;
            let v = (y as f32 + 0.5) / new_height as f32;

            let color = bilinear_image.sample_uv(u, v);
            data.push([color.r, color.g, color.b, color.a]);
        }
    }

    ImageField::from_raw(data, new_width, new_height)
        .with_wrap_mode(image.wrap_mode)
        .with_filter_mode(image.filter_mode)
}

/// An image pyramid for multi-scale processing.
///
/// Contains progressively downsampled versions of the original image.
#[derive(Debug, Clone)]
pub struct ImagePyramid {
    /// Pyramid levels, from finest (original) to coarsest.
    pub levels: Vec<ImageField>,
}

impl ImagePyramid {
    /// Creates a Gaussian pyramid by repeatedly downsampling.
    ///
    /// # Arguments
    /// * `image` - Source image (becomes level 0)
    /// * `num_levels` - Total number of pyramid levels (including original)
    ///
    /// # Example
    ///
    /// ```
    /// use unshape_image::{ImageField, ImagePyramid};
    ///
    /// let data = vec![[0.5, 0.5, 0.5, 1.0]; 64];
    /// let img = ImageField::from_raw(data, 8, 8);
    ///
    /// let pyramid = ImagePyramid::gaussian(&img, 4);
    /// assert_eq!(pyramid.levels.len(), 4);
    /// assert_eq!(pyramid.levels[0].dimensions(), (8, 8));
    /// assert_eq!(pyramid.levels[1].dimensions(), (4, 4));
    /// ```
    pub fn gaussian(image: &ImageField, num_levels: usize) -> Self {
        let num_levels = num_levels.max(1);
        let mut levels = Vec::with_capacity(num_levels);

        // Level 0 is the original (optionally blurred)
        let blurred = blur(image, 1);
        levels.push(blurred);

        // Build remaining levels
        for _ in 1..num_levels {
            let prev = levels.last().unwrap();
            let (w, h) = prev.dimensions();

            // Stop if we can't downsample further
            if w <= 1 && h <= 1 {
                break;
            }

            let downsampled = downsample(prev);
            let blurred = blur(&downsampled, 1);
            levels.push(blurred);
        }

        Self { levels }
    }

    /// Creates a Laplacian pyramid (difference-of-Gaussians).
    ///
    /// Each level stores the difference between consecutive Gaussian levels,
    /// which captures detail at that scale.
    ///
    /// The final level stores the coarsest Gaussian level (the residual).
    pub fn laplacian(image: &ImageField, num_levels: usize) -> Self {
        let gaussian = Self::gaussian(image, num_levels);
        let mut levels = Vec::with_capacity(gaussian.levels.len());

        for i in 0..gaussian.levels.len() - 1 {
            let current = &gaussian.levels[i];
            let (w, h) = gaussian.levels[i + 1].dimensions();
            let next_upsampled = Resize::new(w * 2, h * 2).apply(&gaussian.levels[i + 1]);

            // Compute difference (detail at this level)
            let (width, height) = current.dimensions();
            let mut diff_data = Vec::with_capacity((width * height) as usize);

            for y in 0..height {
                for x in 0..width {
                    let u = (x as f32 + 0.5) / width as f32;
                    let v = (y as f32 + 0.5) / height as f32;

                    let c1 = current.get_pixel(x, y);
                    let c2 = next_upsampled.sample_uv(u, v);

                    // Store difference + 0.5 offset to keep in [0, 1] range
                    diff_data.push([
                        (c1[0] - c2.r) * 0.5 + 0.5,
                        (c1[1] - c2.g) * 0.5 + 0.5,
                        (c1[2] - c2.b) * 0.5 + 0.5,
                        c1[3],
                    ]);
                }
            }

            levels.push(
                ImageField::from_raw(diff_data, width, height)
                    .with_wrap_mode(current.wrap_mode)
                    .with_filter_mode(current.filter_mode),
            );
        }

        // Final level is the residual (coarsest Gaussian level)
        levels.push(gaussian.levels.last().unwrap().clone());

        Self { levels }
    }

    /// Returns the number of levels in the pyramid.
    pub fn len(&self) -> usize {
        self.levels.len()
    }

    /// Returns true if the pyramid is empty.
    pub fn is_empty(&self) -> bool {
        self.levels.is_empty()
    }

    /// Returns the finest (largest) level.
    pub fn finest(&self) -> Option<&ImageField> {
        self.levels.first()
    }

    /// Returns the coarsest (smallest) level.
    pub fn coarsest(&self) -> Option<&ImageField> {
        self.levels.last()
    }

    /// Reconstructs an image from a Laplacian pyramid.
    ///
    /// Starts from the coarsest level and progressively adds detail.
    pub fn reconstruct_laplacian(&self) -> Option<ImageField> {
        if self.levels.is_empty() {
            return None;
        }

        // Start with the coarsest (residual) level
        let mut current = self.levels.last().unwrap().clone();

        // Add detail from each level
        for i in (0..self.levels.len() - 1).rev() {
            let detail = &self.levels[i];
            let (width, height) = detail.dimensions();

            // Upsample current
            let (cw, ch) = current.dimensions();
            let upsampled = Resize::new(cw * 2, ch * 2).apply(&current);

            // Add detail
            let mut data = Vec::with_capacity((width * height) as usize);

            for y in 0..height {
                for x in 0..width {
                    let u = (x as f32 + 0.5) / width as f32;
                    let v = (y as f32 + 0.5) / height as f32;

                    let base = upsampled.sample_uv(u, v);
                    let diff = detail.get_pixel(x, y);

                    // Undo the 0.5 offset and add
                    data.push([
                        base.r + (diff[0] - 0.5) * 2.0,
                        base.g + (diff[1] - 0.5) * 2.0,
                        base.b + (diff[2] - 0.5) * 2.0,
                        diff[3],
                    ]);
                }
            }

            current = ImageField::from_raw(data, width, height)
                .with_wrap_mode(detail.wrap_mode)
                .with_filter_mode(detail.filter_mode);
        }

        Some(current)
    }
}

/// Resizes an image to a specific size using bilinear interpolation.
///
/// # Example
///
/// ```
/// use unshape_image::{ImageField, resize};
///
/// let data = vec![[0.5, 0.5, 0.5, 1.0]; 16];
/// let img = ImageField::from_raw(data, 4, 4);
///
/// let resized = resize(&img, 8, 6);
/// assert_eq!(resized.dimensions(), (8, 6));
/// ```
pub fn resize(image: &ImageField, new_width: u32, new_height: u32) -> ImageField {
    Resize::new(new_width, new_height).apply(image)
}

/// Internal resize implementation.
pub(crate) fn resize_impl(image: &ImageField, new_width: u32, new_height: u32) -> ImageField {
    let new_width = new_width.max(1);
    let new_height = new_height.max(1);

    let mut data = Vec::with_capacity((new_width * new_height) as usize);

    // Use bilinear sampling
    let bilinear_image = ImageField {
        filter_mode: FilterMode::Bilinear,
        ..image.clone()
    };

    for y in 0..new_height {
        for x in 0..new_width {
            let u = (x as f32 + 0.5) / new_width as f32;
            let v = (y as f32 + 0.5) / new_height as f32;

            let color = bilinear_image.sample_uv(u, v);
            data.push([color.r, color.g, color.b, color.a]);
        }
    }

    ImageField::from_raw(data, new_width, new_height)
        .with_wrap_mode(image.wrap_mode)
        .with_filter_mode(image.filter_mode)
}
