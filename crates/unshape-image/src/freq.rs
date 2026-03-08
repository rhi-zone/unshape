#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use unshape_spectral::{Complex, dct2d, fft_shift, fft2d, idct2d, ifft2d};

use crate::ImageField;

// ============================================================================
// FreqImage — typed container for frequency-domain data
// ============================================================================

/// A pair of images representing the real and imaginary components of a 2D FFT.
///
/// Used as the intermediate type between [`Fft2d`] and [`Ifft2d`] / [`FreqRadialMul`].
/// Both images have identical dimensions (power-of-2 padded).
#[derive(Debug, Clone)]
pub struct FreqImage {
    /// Real component of the frequency-domain spectrum.
    pub real: ImageField,
    /// Imaginary component of the frequency-domain spectrum.
    pub imag: ImageField,
}

impl FreqImage {
    /// Creates a `FreqImage` from real and imaginary component images.
    pub fn new(real: ImageField, imag: ImageField) -> Self {
        Self { real, imag }
    }
}

// ============================================================================
// FreqRadialMul — radial frequency mask multiplication
// ============================================================================

/// Applies a radial frequency mask to a frequency-domain image pair.
///
/// The mask is a smooth step function based on distance from the DC component
/// (center of the spectrum after FFT shift). Used to implement low-pass and
/// high-pass frequency filters.
///
/// Typically used between [`Fft2d`] and [`Ifft2d`] in a pipeline.
/// The optimizer recognizes this pattern and replaces it with
/// [`LowPassFreqOptimized`](crate::optimizer::LowPassFreqOptimized) or
/// [`HighPassFreqOptimized`](crate::optimizer::HighPassFreqOptimized).
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct FreqRadialMul {
    /// Cutoff frequency as a fraction of Nyquist (0.0–1.0).
    pub cutoff: f32,
    /// If `true`, frequencies below cutoff are passed (low-pass).
    /// If `false`, frequencies above cutoff are passed (high-pass).
    pub low_pass: bool,
}

impl FreqRadialMul {
    /// Creates a low-pass radial mask with the given cutoff.
    pub fn low_pass(cutoff: f32) -> Self {
        Self {
            cutoff,
            low_pass: true,
        }
    }

    /// Creates a high-pass radial mask with the given cutoff.
    pub fn high_pass(cutoff: f32) -> Self {
        Self {
            cutoff,
            low_pass: false,
        }
    }

    /// Applies the radial mask to a [`FreqImage`].
    pub fn apply(&self, freq: &FreqImage) -> FreqImage {
        let mut real = freq.real.clone();
        let mut imag = freq.imag.clone();
        self.apply_inplace(&mut real, &mut imag);
        FreqImage::new(real, imag)
    }

    /// Applies the radial mask in-place to real and imaginary image components.
    ///
    /// This is the fused path used by the optimizer to avoid extra allocations.
    pub fn apply_inplace(&self, real: &mut ImageField, imag: &mut ImageField) {
        let (w, h) = (real.width as usize, real.height as usize);
        let cx = w as f32 / 2.0;
        let cy = h as f32 / 2.0;
        let max_radius = cx.min(cy);

        for y in 0..h {
            for x in 0..w {
                // Distance from DC (corner in un-shifted FFT = cx, cy after shift).
                // Since we don't apply fft_shift, DC is at (0, 0) corner.
                // Compute normalized distance from nearest corner.
                let dx = (x as f32).min(w as f32 - x as f32);
                let dy = (y as f32).min(h as f32 - y as f32);
                let dist = (dx * dx + dy * dy).sqrt() / max_radius;

                let weight = if self.low_pass {
                    // Smooth step: pass low frequencies, attenuate high.
                    if dist < self.cutoff {
                        1.0
                    } else {
                        let t = ((dist - self.cutoff) / (1.0 - self.cutoff + 1e-8)).clamp(0.0, 1.0);
                        1.0 - t * t * (3.0 - 2.0 * t) // smoothstep
                    }
                } else {
                    // High-pass: pass high frequencies, attenuate low.
                    if dist > self.cutoff {
                        1.0
                    } else {
                        let t = (dist / (self.cutoff + 1e-8)).clamp(0.0, 1.0);
                        t * t * (3.0 - 2.0 * t) // smoothstep
                    }
                };

                let i = y * w + x;
                for ch in 0..4 {
                    real.data[i][ch] *= weight;
                    imag.data[i][ch] *= weight;
                }
            }
        }
    }
}

#[cfg(feature = "dynop")]
impl unshape_op::DynOp for FreqRadialMul {
    fn type_name(&self) -> &'static str {
        "resin::FreqRadialMul"
    }

    fn input_type(&self) -> unshape_op::OpType {
        unshape_op::OpType::of::<FreqImage>("FreqImage")
    }

    fn output_type(&self) -> unshape_op::OpType {
        unshape_op::OpType::of::<FreqImage>("FreqImage")
    }

    fn apply_dyn(
        &self,
        input: unshape_op::OpValue,
    ) -> Result<unshape_op::OpValue, unshape_op::OpError> {
        let freq: FreqImage = input.downcast()?;
        let result = self.apply(&freq);
        Ok(unshape_op::OpValue::new(
            unshape_op::OpType::of::<FreqImage>("FreqImage"),
            result,
        ))
    }

    fn params(&self) -> serde_json::Value {
        serde_json::json!({
            "cutoff": self.cutoff,
            "low_pass": self.low_pass,
        })
    }
}

// ============================================================================
// FreqRingMul — ring/annulus mask in frequency domain
// ============================================================================

/// Applies a ring (annulus) frequency mask to a frequency-domain image pair.
///
/// Passes frequencies whose normalized radius falls between `lo` and `hi`.
/// Used to implement band-pass filtering in the frequency domain.
///
/// Typically used between [`Fft2d`] and [`Ifft2d`] in a pipeline.
/// The optimizer recognizes this pattern and replaces it with
/// [`BandPassFreqOptimized`](crate::optimizer::BandPassFreqOptimized).
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct FreqRingMul {
    /// Lower bound of the pass-band as a fraction of Nyquist (0.0–1.0).
    pub lo: f32,
    /// Upper bound of the pass-band as a fraction of Nyquist (0.0–1.0).
    pub hi: f32,
}

impl FreqRingMul {
    /// Creates a ring mask that passes frequencies between `lo` and `hi`.
    pub fn new(lo: f32, hi: f32) -> Self {
        Self { lo, hi }
    }

    /// Applies the ring mask to a [`FreqImage`].
    pub fn apply(&self, freq: &FreqImage) -> FreqImage {
        let mut real = freq.real.clone();
        let mut imag = freq.imag.clone();
        self.apply_inplace(&mut real, &mut imag);
        FreqImage::new(real, imag)
    }

    /// Applies the ring mask in-place to real and imaginary image components.
    ///
    /// This is the fused path used by the optimizer to avoid extra allocations.
    pub fn apply_inplace(&self, real: &mut ImageField, imag: &mut ImageField) {
        let (w, h) = (real.width as usize, real.height as usize);
        let cx = w as f32 / 2.0;
        let cy = h as f32 / 2.0;
        let max_radius = cx.min(cy);

        for y in 0..h {
            for x in 0..w {
                let dx = (x as f32).min(w as f32 - x as f32);
                let dy = (y as f32).min(h as f32 - y as f32);
                let dist = (dx * dx + dy * dy).sqrt() / max_radius;

                // Smooth step for lower edge (lo): ramp up from 0 near 0 to 1 at lo.
                let lower_weight = if dist < self.lo {
                    let t = (dist / (self.lo + 1e-8)).clamp(0.0, 1.0);
                    t * t * (3.0 - 2.0 * t)
                } else {
                    1.0
                };

                // Smooth step for upper edge (hi): ramp down from 1 at hi to 0 above hi.
                let upper_weight = if dist > self.hi {
                    let t = ((dist - self.hi) / (1.0 - self.hi + 1e-8)).clamp(0.0, 1.0);
                    1.0 - t * t * (3.0 - 2.0 * t)
                } else {
                    1.0
                };

                let weight = lower_weight * upper_weight;

                let i = y * w + x;
                for ch in 0..4 {
                    real.data[i][ch] *= weight;
                    imag.data[i][ch] *= weight;
                }
            }
        }
    }
}

#[cfg(feature = "dynop")]
impl unshape_op::DynOp for FreqRingMul {
    fn type_name(&self) -> &'static str {
        "resin::FreqRingMul"
    }

    fn input_type(&self) -> unshape_op::OpType {
        unshape_op::OpType::of::<FreqImage>("FreqImage")
    }

    fn output_type(&self) -> unshape_op::OpType {
        unshape_op::OpType::of::<FreqImage>("FreqImage")
    }

    fn apply_dyn(
        &self,
        input: unshape_op::OpValue,
    ) -> Result<unshape_op::OpValue, unshape_op::OpError> {
        let freq: FreqImage = input.downcast()?;
        let result = self.apply(&freq);
        Ok(unshape_op::OpValue::new(
            unshape_op::OpType::of::<FreqImage>("FreqImage"),
            result,
        ))
    }

    fn params(&self) -> serde_json::Value {
        serde_json::json!({
            "lo": self.lo,
            "hi": self.hi,
        })
    }
}

/// 2D Fast Fourier Transform.
///
/// Transforms an image from spatial domain to frequency domain.
/// Output is a [`FreqImage`] containing real and imaginary components.
/// Low frequencies are at corners; use [`FftShift`] to center them.
///
/// Note: Dimensions must be powers of 2. Non-power-of-2 images are padded.
#[derive(Debug, Clone, Copy, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Fft2d;

#[cfg(feature = "dynop")]
impl unshape_op::DynOp for Fft2d {
    fn type_name(&self) -> &'static str {
        "resin::Fft2d"
    }

    fn input_type(&self) -> unshape_op::OpType {
        unshape_op::OpType::of::<ImageField>("ImageField")
    }

    fn output_type(&self) -> unshape_op::OpType {
        unshape_op::OpType::of::<FreqImage>("FreqImage")
    }

    fn apply_dyn(
        &self,
        input: unshape_op::OpValue,
    ) -> Result<unshape_op::OpValue, unshape_op::OpError> {
        let img: ImageField = input.downcast()?;
        let (real, imag) = self.apply(&img);
        Ok(unshape_op::OpValue::new(
            unshape_op::OpType::of::<FreqImage>("FreqImage"),
            FreqImage::new(real, imag),
        ))
    }

    fn params(&self) -> serde_json::Value {
        serde_json::json!({})
    }
}

impl Fft2d {
    /// Applies 2D FFT to an image.
    ///
    /// Returns (real, imaginary) image pair representing the frequency domain.
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
/// Accepts a [`FreqImage`] (or the raw (real, imag) pair directly).
#[derive(Debug, Clone, Copy, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Ifft2d;

#[cfg(feature = "dynop")]
impl unshape_op::DynOp for Ifft2d {
    fn type_name(&self) -> &'static str {
        "resin::Ifft2d"
    }

    fn input_type(&self) -> unshape_op::OpType {
        unshape_op::OpType::of::<FreqImage>("FreqImage")
    }

    fn output_type(&self) -> unshape_op::OpType {
        unshape_op::OpType::of::<ImageField>("ImageField")
    }

    fn apply_dyn(
        &self,
        input: unshape_op::OpValue,
    ) -> Result<unshape_op::OpValue, unshape_op::OpError> {
        let freq: FreqImage = input.downcast()?;
        let result = self.apply(&freq.real, &freq.imag);
        Ok(unshape_op::OpValue::new(
            unshape_op::OpType::of::<ImageField>("ImageField"),
            result,
        ))
    }

    fn params(&self) -> serde_json::Value {
        serde_json::json!({})
    }
}

impl Ifft2d {
    /// Applies 2D IFFT to frequency domain images.
    ///
    /// Takes (real, imaginary) pair and returns spatial domain image.
    pub fn apply(&self, real: &ImageField, imag: &ImageField) -> ImageField {
        assert_eq!(real.width, imag.width);
        assert_eq!(real.height, imag.height);

        let (w, h) = (real.width as usize, real.height as usize);
        let mut result_data = vec![[0.0f32; 4]; w * h];

        #[allow(clippy::needless_range_loop)]
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
#[derive(Default)]
pub struct Dct2d {
    /// Block size for block-based DCT. None = whole image.
    pub block_size: Option<u32>,
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
#[derive(Default)]
pub struct Idct2d {
    /// Block size for block-based IDCT. None = whole image.
    pub block_size: Option<u32>,
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
