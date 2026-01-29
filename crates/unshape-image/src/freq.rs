#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use unshape_spectral::{Complex, dct2d, fft_shift, fft2d, idct2d, ifft2d};

use crate::ImageField;

/// 2D Fast Fourier Transform.
///
/// Transforms an image from spatial domain to frequency domain.
/// Output is two images: real and imaginary components.
#[derive(Debug, Clone, Copy, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Fft2d;

impl Fft2d {
    /// Applies 2D FFT to an image.
    ///
    /// Returns (real, imaginary) image pair representing the frequency domain.
    /// Low frequencies are at corners, high frequencies at center.
    /// Use `FftShift` to center low frequencies.
    ///
    /// Note: Dimensions must be powers of 2. Non-power-of-2 images are padded.
    pub fn apply(&self, image: &ImageField) -> (ImageField, ImageField) {
        let (w, h) = (image.width as usize, image.height as usize);
        let (pw, ph) = (w.next_power_of_two(), h.next_power_of_two());

        // Process each channel separately, then combine
        let mut real_data = vec![[0.0f32; 4]; pw * ph];
        let mut imag_data = vec![[0.0f32; 4]; pw * ph];

        for ch in 0..4 {
            // Extract channel as grayscale, pad to power of 2
            let mut pixels = vec![0.0f32; pw * ph];
            for y in 0..h {
                for x in 0..w {
                    pixels[y * pw + x] = image.data[y * w + x][ch];
                }
            }

            // Apply 2D FFT
            let spectrum = fft2d(&pixels, pw, ph);

            // Split into real/imag
            for (i, c) in spectrum.iter().enumerate() {
                real_data[i][ch] = c.re;
                imag_data[i][ch] = c.im;
            }
        }

        let real = ImageField::from_raw(real_data, pw as u32, ph as u32)
            .with_wrap_mode(image.wrap_mode)
            .with_filter_mode(image.filter_mode);
        let imag = ImageField::from_raw(imag_data, pw as u32, ph as u32)
            .with_wrap_mode(image.wrap_mode)
            .with_filter_mode(image.filter_mode);

        (real, imag)
    }
}

/// 2D Inverse Fast Fourier Transform.
///
/// Transforms from frequency domain back to spatial domain.
#[derive(Debug, Clone, Copy, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Ifft2d;

impl Ifft2d {
    /// Applies 2D IFFT to frequency domain images.
    ///
    /// Takes (real, imaginary) pair and returns spatial domain image.
    pub fn apply(&self, real: &ImageField, imag: &ImageField) -> ImageField {
        assert_eq!(real.width, imag.width);
        assert_eq!(real.height, imag.height);

        let (w, h) = (real.width as usize, real.height as usize);
        let mut result_data = vec![[0.0f32; 4]; w * h];

        for ch in 0..4 {
            // Combine real/imag into complex spectrum
            let spectrum: Vec<Complex> = (0..w * h)
                .map(|i| Complex::new(real.data[i][ch], imag.data[i][ch]))
                .collect();

            // Apply 2D IFFT
            let pixels = ifft2d(&spectrum, w, h);

            // Store result
            for (i, &p) in pixels.iter().enumerate() {
                result_data[i][ch] = p;
            }
        }

        ImageField::from_raw(result_data, real.width, real.height)
            .with_wrap_mode(real.wrap_mode)
            .with_filter_mode(real.filter_mode)
    }
}

/// Shifts zero frequency to center of spectrum.
///
/// Swaps quadrants so DC component is at center instead of corners.
/// Apply before visualization or frequency-domain filtering.
#[derive(Debug, Clone, Copy, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct FftShift;

impl FftShift {
    /// Shifts spectrum so DC is at center.
    ///
    /// Works on both real and imaginary parts of FFT output.
    pub fn apply(&self, image: &ImageField) -> ImageField {
        let (w, h) = (image.width as usize, image.height as usize);
        let mut data = image.data.clone();

        for ch in 0..4 {
            // Extract channel
            let mut channel: Vec<Complex> = data.iter().map(|p| Complex::new(p[ch], 0.0)).collect();

            // Apply shift
            fft_shift(&mut channel, w, h);

            // Put back
            for (i, c) in channel.iter().enumerate() {
                data[i][ch] = c.re;
            }
        }

        ImageField::from_raw(data, image.width, image.height)
            .with_wrap_mode(image.wrap_mode)
            .with_filter_mode(image.filter_mode)
    }
}

/// 2D Discrete Cosine Transform.
///
/// Used in JPEG compression. Can operate on whole image or in blocks.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Dct2d {
    /// Block size for block-based DCT. None = whole image.
    pub block_size: Option<u32>,
}

impl Default for Dct2d {
    fn default() -> Self {
        Self { block_size: None }
    }
}

impl Dct2d {
    /// Creates a whole-image DCT.
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a block-based DCT (like JPEG uses 8x8).
    pub fn with_block_size(block_size: u32) -> Self {
        Self {
            block_size: Some(block_size),
        }
    }

    /// Applies 2D DCT to an image.
    pub fn apply(&self, image: &ImageField) -> ImageField {
        let (w, h) = (image.width as usize, image.height as usize);
        let block_size = self.block_size.map(|b| b as usize);

        let mut result_data = vec![[0.0f32; 4]; w * h];

        for ch in 0..4 {
            // Extract channel
            let pixels: Vec<f32> = image.data.iter().map(|p| p[ch]).collect();

            // Apply 2D DCT
            let spectrum = dct2d(&pixels, w, h, block_size);

            // Store result
            for (i, &s) in spectrum.iter().enumerate() {
                result_data[i][ch] = s;
            }
        }

        ImageField::from_raw(result_data, image.width, image.height)
            .with_wrap_mode(image.wrap_mode)
            .with_filter_mode(image.filter_mode)
    }
}

/// 2D Inverse Discrete Cosine Transform.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Idct2d {
    /// Block size for block-based IDCT. None = whole image.
    pub block_size: Option<u32>,
}

impl Default for Idct2d {
    fn default() -> Self {
        Self { block_size: None }
    }
}

impl Idct2d {
    /// Creates a whole-image IDCT.
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a block-based IDCT.
    pub fn with_block_size(block_size: u32) -> Self {
        Self {
            block_size: Some(block_size),
        }
    }

    /// Applies 2D IDCT to a DCT image.
    pub fn apply(&self, image: &ImageField) -> ImageField {
        let (w, h) = (image.width as usize, image.height as usize);
        let block_size = self.block_size.map(|b| b as usize);

        let mut result_data = vec![[0.0f32; 4]; w * h];

        for ch in 0..4 {
            // Extract channel
            let spectrum: Vec<f32> = image.data.iter().map(|p| p[ch]).collect();

            // Apply 2D IDCT
            let pixels = idct2d(&spectrum, w, h, block_size);

            // Store result
            for (i, &p) in pixels.iter().enumerate() {
                result_data[i][ch] = p;
            }
        }

        ImageField::from_raw(result_data, image.width, image.height)
            .with_wrap_mode(image.wrap_mode)
            .with_filter_mode(image.filter_mode)
    }
}

/// Spreads image data using a pseudorandom sequence.
///
/// Multiplies pixel values by a deterministic pseudorandom ±1 sequence.
/// Used for robust watermarking that survives compression.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct SpreadSpectrum {
    /// Seed for the pseudorandom sequence.
    pub seed: u64,
    /// Spreading factor (higher = more robust, lower visual quality).
    pub factor: f32,
}

impl Default for SpreadSpectrum {
    fn default() -> Self {
        Self {
            seed: 0,
            factor: 0.1,
        }
    }
}

impl SpreadSpectrum {
    /// Creates a spread spectrum op with the given seed.
    pub fn new(seed: u64) -> Self {
        Self {
            seed,
            ..Default::default()
        }
    }

    /// Sets the spreading factor.
    pub fn with_factor(mut self, factor: f32) -> Self {
        self.factor = factor;
        self
    }

    /// Applies spread spectrum to an image.
    ///
    /// Each pixel is multiplied by factor * sign, where sign is ±1
    /// determined by a pseudorandom sequence seeded by `self.seed`.
    pub fn apply(&self, image: &ImageField) -> ImageField {
        let data: Vec<[f32; 4]> = image
            .data
            .iter()
            .enumerate()
            .map(|(i, pixel)| {
                // Simple PRNG: xorshift with position-based seed
                let mut state = self.seed.wrapping_add(i as u64);
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                let sign = if state & 1 == 0 { 1.0 } else { -1.0 };

                [
                    pixel[0] + self.factor * sign,
                    pixel[1] + self.factor * sign,
                    pixel[2] + self.factor * sign,
                    pixel[3], // Don't modify alpha
                ]
            })
            .collect();

        ImageField::from_raw(data, image.width, image.height)
            .with_wrap_mode(image.wrap_mode)
            .with_filter_mode(image.filter_mode)
    }
}

/// Reverses spread spectrum operation.
///
/// Since spread spectrum adds factor * sign, unspread subtracts it.
/// This is the inverse operation when using the same seed.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct UnspreadSpectrum {
    /// Seed for the pseudorandom sequence (must match SpreadSpectrum).
    pub seed: u64,
    /// Factor used in original spread (must match).
    pub factor: f32,
}

impl UnspreadSpectrum {
    /// Creates an unspread op with the given seed.
    pub fn new(seed: u64) -> Self {
        Self { seed, factor: 0.1 }
    }

    /// Sets the factor (must match the spread factor).
    pub fn with_factor(mut self, factor: f32) -> Self {
        self.factor = factor;
        self
    }

    /// Applies unspread to recover original.
    pub fn apply(&self, image: &ImageField) -> ImageField {
        let data: Vec<[f32; 4]> = image
            .data
            .iter()
            .enumerate()
            .map(|(i, pixel)| {
                // Same PRNG as SpreadSpectrum
                let mut state = self.seed.wrapping_add(i as u64);
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                let sign = if state & 1 == 0 { 1.0 } else { -1.0 };

                [
                    pixel[0] - self.factor * sign,
                    pixel[1] - self.factor * sign,
                    pixel[2] - self.factor * sign,
                    pixel[3],
                ]
            })
            .collect();

        ImageField::from_raw(data, image.width, image.height)
            .with_wrap_mode(image.wrap_mode)
            .with_filter_mode(image.filter_mode)
    }
}

/// Quantizes pixel values with a bias toward specific values.
///
/// Used for embedding data in quantization decisions.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct QuantizeWithBias {
    /// Number of quantization levels.
    pub levels: u32,
}

impl Default for QuantizeWithBias {
    fn default() -> Self {
        Self { levels: 8 }
    }
}

impl QuantizeWithBias {
    /// Creates a quantizer with the specified number of levels.
    pub fn new(levels: u32) -> Self {
        Self { levels }
    }

    /// Applies biased quantization.
    ///
    /// Each pixel is quantized to `levels` discrete values.
    /// The bias image controls rounding: bias > 0.5 rounds up, <= 0.5 rounds down.
    /// This allows embedding binary data in quantization decisions.
    pub fn apply(&self, image: &ImageField, bias: &ImageField) -> ImageField {
        assert_eq!(image.width, bias.width);
        assert_eq!(image.height, bias.height);

        let levels = self.levels as f32;
        let data: Vec<[f32; 4]> = image
            .data
            .iter()
            .zip(bias.data.iter())
            .map(|(pixel, bias_pixel)| {
                let mut result = [0.0f32; 4];
                for ch in 0..4 {
                    // Scale to level range
                    let scaled = pixel[ch] * (levels - 1.0);
                    let floor = scaled.floor();
                    let frac = scaled - floor;

                    // Use bias to decide rounding
                    let quantized = if bias_pixel[ch] > 0.5 {
                        // Bias toward ceiling
                        if frac > 0.0 { floor + 1.0 } else { floor }
                    } else {
                        // Bias toward floor
                        floor
                    };

                    // Scale back to 0-1
                    result[ch] = (quantized / (levels - 1.0)).clamp(0.0, 1.0);
                }
                result
            })
            .collect();

        ImageField::from_raw(data, image.width, image.height)
            .with_wrap_mode(image.wrap_mode)
            .with_filter_mode(image.filter_mode)
    }
}
