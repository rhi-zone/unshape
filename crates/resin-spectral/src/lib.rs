//! FFT and frequency-domain primitives.
//!
//! This crate provides core spectral operations shared between audio and image processing:
//! - FFT/IFFT for 1D signals
//! - FFT2D/IFFT2D for 2D images
//! - DCT/IDCT for compression and watermarking
//! - Window functions (Hann, Hamming, Blackman)
//! - Pre-allocated workspaces for real-time use
//!
//! # Example
//!
//! ```
//! use rhizome_resin_spectral::{fft, ifft, hann_window};
//!
//! // Generate a simple signal
//! let n = 1024;
//! let signal: Vec<f32> = (0..n)
//!     .map(|i| (2.0 * std::f32::consts::PI * 10.0 * i as f32 / n as f32).sin())
//!     .collect();
//!
//! // Apply FFT
//! let spectrum = fft(&signal);
//!
//! // The spectrum has n/2+1 complex values (positive frequencies)
//! assert_eq!(spectrum.len(), n / 2 + 1);
//! ```

use std::f32::consts::PI;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

// ============================================================================
// Complex number type
// ============================================================================

/// A complex number for FFT operations.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Complex {
    /// Real part.
    pub re: f32,
    /// Imaginary part.
    pub im: f32,
}

impl Complex {
    /// Creates a new complex number.
    pub fn new(re: f32, im: f32) -> Self {
        Self { re, im }
    }

    /// Creates a complex number from polar coordinates.
    pub fn from_polar(mag: f32, phase: f32) -> Self {
        Self {
            re: mag * phase.cos(),
            im: mag * phase.sin(),
        }
    }

    /// Returns the magnitude (absolute value).
    pub fn mag(&self) -> f32 {
        (self.re * self.re + self.im * self.im).sqrt()
    }

    /// Returns the squared magnitude (avoids sqrt).
    pub fn mag_sq(&self) -> f32 {
        self.re * self.re + self.im * self.im
    }

    /// Returns the phase angle.
    pub fn phase(&self) -> f32 {
        self.im.atan2(self.re)
    }

    /// Returns the complex conjugate.
    pub fn conj(&self) -> Self {
        Self {
            re: self.re,
            im: -self.im,
        }
    }
}

impl std::ops::Add for Complex {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self {
            re: self.re + rhs.re,
            im: self.im + rhs.im,
        }
    }
}

impl std::ops::Sub for Complex {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self {
            re: self.re - rhs.re,
            im: self.im - rhs.im,
        }
    }
}

impl std::ops::Mul for Complex {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        Self {
            re: self.re * rhs.re - self.im * rhs.im,
            im: self.re * rhs.im + self.im * rhs.re,
        }
    }
}

impl std::ops::Mul<f32> for Complex {
    type Output = Self;
    fn mul(self, rhs: f32) -> Self {
        Self {
            re: self.re * rhs,
            im: self.im * rhs,
        }
    }
}

impl std::ops::Div<f32> for Complex {
    type Output = Self;
    fn div(self, rhs: f32) -> Self {
        Self {
            re: self.re / rhs,
            im: self.im / rhs,
        }
    }
}

// ============================================================================
// Window functions
// ============================================================================

/// Generates a Hann (raised cosine) window.
pub fn hann_window(size: usize) -> Vec<f32> {
    (0..size)
        .map(|n| 0.5 * (1.0 - (2.0 * PI * n as f32 / (size - 1) as f32).cos()))
        .collect()
}

/// Generates a Hamming window.
pub fn hamming_window(size: usize) -> Vec<f32> {
    (0..size)
        .map(|n| 0.54 - 0.46 * (2.0 * PI * n as f32 / (size - 1) as f32).cos())
        .collect()
}

/// Generates a Blackman window.
pub fn blackman_window(size: usize) -> Vec<f32> {
    (0..size)
        .map(|n| {
            let t = 2.0 * PI * n as f32 / (size - 1) as f32;
            0.42 - 0.5 * t.cos() + 0.08 * (2.0 * t).cos()
        })
        .collect()
}

/// Generates a rectangular (flat) window.
pub fn rect_window(size: usize) -> Vec<f32> {
    vec![1.0; size]
}

/// Applies a window function to a signal in-place.
pub fn apply_window(signal: &mut [f32], window: &[f32]) {
    assert_eq!(signal.len(), window.len());
    for (s, w) in signal.iter_mut().zip(window.iter()) {
        *s *= w;
    }
}

// ============================================================================
// 1D FFT (Cooley-Tukey radix-2)
// ============================================================================

/// Computes the FFT of a real signal.
///
/// Returns N/2+1 complex values representing positive frequencies.
/// Input length must be a power of 2.
pub fn fft(signal: &[f32]) -> Vec<Complex> {
    let n = signal.len();
    assert!(n.is_power_of_two(), "FFT size must be power of 2");

    // Convert to complex
    let mut x: Vec<Complex> = signal.iter().map(|&s| Complex::new(s, 0.0)).collect();

    // In-place FFT
    fft_in_place(&mut x, false);

    // Return only positive frequencies (N/2+1 bins)
    x.truncate(n / 2 + 1);
    x
}

/// Computes the inverse FFT, returning a real signal.
///
/// Input should be N/2+1 complex values from fft().
/// Output length is 2 * (input.len() - 1).
pub fn ifft(spectrum: &[Complex]) -> Vec<f32> {
    let n = (spectrum.len() - 1) * 2;
    assert!(n.is_power_of_two(), "IFFT size must be power of 2");

    // Reconstruct full spectrum (conjugate symmetry)
    let mut x = Vec::with_capacity(n);
    x.extend_from_slice(spectrum);

    // Add conjugate mirrored frequencies
    for i in (1..n / 2).rev() {
        x.push(spectrum[i].conj());
    }

    // In-place IFFT
    fft_in_place(&mut x, true);

    // Return real part, scaled
    x.iter().map(|c| c.re / n as f32).collect()
}

/// Computes the full complex FFT (not just positive frequencies).
pub fn fft_complex(signal: &[Complex]) -> Vec<Complex> {
    let n = signal.len();
    assert!(n.is_power_of_two(), "FFT size must be power of 2");

    let mut x = signal.to_vec();
    fft_in_place(&mut x, false);
    x
}

/// Computes the full complex IFFT.
pub fn ifft_complex(spectrum: &[Complex]) -> Vec<Complex> {
    let n = spectrum.len();
    assert!(n.is_power_of_two(), "IFFT size must be power of 2");

    let mut x = spectrum.to_vec();
    fft_in_place(&mut x, true);

    // Scale
    for c in &mut x {
        *c = *c * (1.0 / n as f32);
    }
    x
}

/// In-place FFT using Cooley-Tukey radix-2 algorithm.
fn fft_in_place(x: &mut [Complex], inverse: bool) {
    let n = x.len();
    if n <= 1 {
        return;
    }

    // Bit-reversal permutation
    let mut j = 0;
    for i in 0..n {
        if i < j {
            x.swap(i, j);
        }
        let mut m = n >> 1;
        while m >= 1 && j >= m {
            j -= m;
            m >>= 1;
        }
        j += m;
    }

    // Cooley-Tukey iterative FFT
    let sign = if inverse { 1.0 } else { -1.0 };

    let mut len = 2;
    while len <= n {
        let half = len / 2;
        let angle = sign * 2.0 * PI / len as f32;

        for i in (0..n).step_by(len) {
            let mut w = Complex::new(1.0, 0.0);
            let wn = Complex::from_polar(1.0, angle);

            for k in 0..half {
                let t = x[i + k + half] * w;
                let u = x[i + k];
                x[i + k] = u + t;
                x[i + k + half] = u - t;
                w = w * wn;
            }
        }

        len <<= 1;
    }
}

// ============================================================================
// 2D FFT for images
// ============================================================================

/// Computes 2D FFT of a real image.
///
/// Input: row-major grayscale pixels, width × height must both be powers of 2.
/// Output: complex spectrum in row-major order.
pub fn fft2d(pixels: &[f32], width: usize, height: usize) -> Vec<Complex> {
    assert!(width.is_power_of_two(), "Width must be power of 2");
    assert!(height.is_power_of_two(), "Height must be power of 2");
    assert_eq!(pixels.len(), width * height);

    // Convert to complex
    let mut data: Vec<Complex> = pixels.iter().map(|&p| Complex::new(p, 0.0)).collect();

    // FFT each row
    for row in 0..height {
        let start = row * width;
        fft_in_place(&mut data[start..start + width], false);
    }

    // FFT each column (transpose, FFT, transpose back)
    let mut col_buffer = vec![Complex::default(); height];
    for col in 0..width {
        // Extract column
        for row in 0..height {
            col_buffer[row] = data[row * width + col];
        }

        // FFT column
        fft_in_place(&mut col_buffer, false);

        // Put back
        for row in 0..height {
            data[row * width + col] = col_buffer[row];
        }
    }

    data
}

/// Computes inverse 2D FFT, returning real pixels.
///
/// Input: complex spectrum from fft2d().
/// Output: grayscale pixels in row-major order.
pub fn ifft2d(spectrum: &[Complex], width: usize, height: usize) -> Vec<f32> {
    assert!(width.is_power_of_two(), "Width must be power of 2");
    assert!(height.is_power_of_two(), "Height must be power of 2");
    assert_eq!(spectrum.len(), width * height);

    let mut data = spectrum.to_vec();
    let scale = 1.0 / (width * height) as f32;

    // IFFT each row
    for row in 0..height {
        let start = row * width;
        fft_in_place(&mut data[start..start + width], true);
    }

    // IFFT each column
    let mut col_buffer = vec![Complex::default(); height];
    for col in 0..width {
        // Extract column
        for row in 0..height {
            col_buffer[row] = data[row * width + col];
        }

        // IFFT column
        fft_in_place(&mut col_buffer, true);

        // Put back
        for row in 0..height {
            data[row * width + col] = col_buffer[row];
        }
    }

    // Return real part, scaled
    data.iter().map(|c| c.re * scale).collect()
}

/// Shifts zero frequency to center (for visualization).
///
/// Swaps quadrants: top-left ↔ bottom-right, top-right ↔ bottom-left.
pub fn fft_shift(spectrum: &mut [Complex], width: usize, height: usize) {
    assert_eq!(spectrum.len(), width * height);

    let half_w = width / 2;
    let half_h = height / 2;

    for row in 0..half_h {
        for col in 0..half_w {
            // Swap quadrant 1 (top-left) with quadrant 3 (bottom-right)
            let i1 = row * width + col;
            let i3 = (row + half_h) * width + (col + half_w);
            spectrum.swap(i1, i3);

            // Swap quadrant 2 (top-right) with quadrant 4 (bottom-left)
            let i2 = row * width + (col + half_w);
            let i4 = (row + half_h) * width + col;
            spectrum.swap(i2, i4);
        }
    }
}

// ============================================================================
// DCT (Discrete Cosine Transform) for compression
// ============================================================================

/// Computes 1D DCT-II (the "standard" DCT used in JPEG).
pub fn dct(signal: &[f32]) -> Vec<f32> {
    let n = signal.len();
    let mut result = vec![0.0; n];

    for k in 0..n {
        let mut sum = 0.0;
        for i in 0..n {
            sum += signal[i] * (PI * (2 * i + 1) as f32 * k as f32 / (2 * n) as f32).cos();
        }
        result[k] = sum
            * if k == 0 {
                1.0 / (n as f32).sqrt()
            } else {
                (2.0 / n as f32).sqrt()
            };
    }

    result
}

/// Computes 1D inverse DCT-II (DCT-III).
pub fn idct(spectrum: &[f32]) -> Vec<f32> {
    let n = spectrum.len();
    let mut result = vec![0.0; n];

    let scale_0 = 1.0 / (n as f32).sqrt();
    let scale_k = (2.0 / n as f32).sqrt();

    for i in 0..n {
        let mut sum = spectrum[0] * scale_0;
        for k in 1..n {
            sum +=
                spectrum[k] * scale_k * (PI * (2 * i + 1) as f32 * k as f32 / (2 * n) as f32).cos();
        }
        result[i] = sum;
    }

    result
}

/// Computes 2D DCT on an image (block-based or full).
///
/// If `block_size` is Some, applies DCT to non-overlapping blocks.
/// If None, applies DCT to the entire image.
pub fn dct2d(pixels: &[f32], width: usize, height: usize, block_size: Option<usize>) -> Vec<f32> {
    match block_size {
        Some(bs) => dct2d_blocked(pixels, width, height, bs),
        None => dct2d_full(pixels, width, height),
    }
}

fn dct2d_full(pixels: &[f32], width: usize, height: usize) -> Vec<f32> {
    assert_eq!(pixels.len(), width * height);

    // DCT each row
    let mut data: Vec<f32> = Vec::with_capacity(width * height);
    for row in 0..height {
        let start = row * width;
        data.extend(dct(&pixels[start..start + width]));
    }

    // DCT each column
    let mut col_buffer = vec![0.0; height];
    for col in 0..width {
        // Extract column
        for row in 0..height {
            col_buffer[row] = data[row * width + col];
        }

        // DCT column
        let transformed = dct(&col_buffer);

        // Put back
        for row in 0..height {
            data[row * width + col] = transformed[row];
        }
    }

    data
}

fn dct2d_blocked(pixels: &[f32], width: usize, height: usize, block_size: usize) -> Vec<f32> {
    assert_eq!(pixels.len(), width * height);

    let mut result = vec![0.0; width * height];

    for block_y in (0..height).step_by(block_size) {
        for block_x in (0..width).step_by(block_size) {
            let bw = block_size.min(width - block_x);
            let bh = block_size.min(height - block_y);

            // Extract block
            let mut block = vec![0.0; bw * bh];
            for row in 0..bh {
                for col in 0..bw {
                    block[row * bw + col] = pixels[(block_y + row) * width + (block_x + col)];
                }
            }

            // Apply 2D DCT to block
            let transformed = dct2d_full(&block, bw, bh);

            // Put back
            for row in 0..bh {
                for col in 0..bw {
                    result[(block_y + row) * width + (block_x + col)] = transformed[row * bw + col];
                }
            }
        }
    }

    result
}

/// Computes inverse 2D DCT.
pub fn idct2d(
    spectrum: &[f32],
    width: usize,
    height: usize,
    block_size: Option<usize>,
) -> Vec<f32> {
    match block_size {
        Some(bs) => idct2d_blocked(spectrum, width, height, bs),
        None => idct2d_full(spectrum, width, height),
    }
}

fn idct2d_full(spectrum: &[f32], width: usize, height: usize) -> Vec<f32> {
    assert_eq!(spectrum.len(), width * height);

    // IDCT each row
    let mut data: Vec<f32> = Vec::with_capacity(width * height);
    for row in 0..height {
        let start = row * width;
        data.extend(idct(&spectrum[start..start + width]));
    }

    // IDCT each column
    let mut col_buffer = vec![0.0; height];
    for col in 0..width {
        // Extract column
        for row in 0..height {
            col_buffer[row] = data[row * width + col];
        }

        // IDCT column
        let transformed = idct(&col_buffer);

        // Put back
        for row in 0..height {
            data[row * width + col] = transformed[row];
        }
    }

    data
}

fn idct2d_blocked(spectrum: &[f32], width: usize, height: usize, block_size: usize) -> Vec<f32> {
    assert_eq!(spectrum.len(), width * height);

    let mut result = vec![0.0; width * height];

    for block_y in (0..height).step_by(block_size) {
        for block_x in (0..width).step_by(block_size) {
            let bw = block_size.min(width - block_x);
            let bh = block_size.min(height - block_y);

            // Extract block
            let mut block = vec![0.0; bw * bh];
            for row in 0..bh {
                for col in 0..bw {
                    block[row * bw + col] = spectrum[(block_y + row) * width + (block_x + col)];
                }
            }

            // Apply 2D IDCT to block
            let transformed = idct2d_full(&block, bw, bh);

            // Put back
            for row in 0..bh {
                for col in 0..bw {
                    result[(block_y + row) * width + (block_x + col)] = transformed[row * bw + col];
                }
            }
        }
    }

    result
}

// ============================================================================
// Pre-allocated workspace for real-time spectral processing
// ============================================================================

/// Pre-allocated buffers for FFT/IFFT operations.
///
/// Use this to avoid allocations in real-time callbacks or tight loops.
#[derive(Debug, Clone)]
pub struct SpectralWorkspace {
    /// FFT size (must be power of 2).
    fft_size: usize,
    /// Complex buffer for FFT operations.
    complex_buffer: Vec<Complex>,
    /// Real output buffer for IFFT.
    real_buffer: Vec<f32>,
    /// Spectrum output (N/2+1 bins).
    spectrum: Vec<Complex>,
}

impl SpectralWorkspace {
    /// Creates a new workspace for the given FFT size.
    ///
    /// # Panics
    /// Panics if `fft_size` is not a power of 2.
    pub fn new(fft_size: usize) -> Self {
        assert!(fft_size.is_power_of_two(), "FFT size must be power of 2");
        Self {
            fft_size,
            complex_buffer: vec![Complex::new(0.0, 0.0); fft_size],
            real_buffer: vec![0.0; fft_size],
            spectrum: vec![Complex::new(0.0, 0.0); fft_size / 2 + 1],
        }
    }

    /// Returns the FFT size this workspace was created for.
    pub fn fft_size(&self) -> usize {
        self.fft_size
    }

    /// Returns the computed spectrum (N/2+1 complex bins).
    pub fn spectrum(&self) -> &[Complex] {
        &self.spectrum
    }

    /// Returns mutable access to the spectrum for modification.
    pub fn spectrum_mut(&mut self) -> &mut [Complex] {
        &mut self.spectrum
    }

    /// Returns the real output buffer (after IFFT).
    pub fn real_buffer(&self) -> &[f32] {
        &self.real_buffer
    }

    /// Computes IFFT from the workspace's current spectrum.
    ///
    /// Results are stored in `real_buffer()`.
    pub fn ifft_from_spectrum(&mut self) {
        let n = self.fft_size;

        // Reconstruct full spectrum (conjugate symmetry)
        for i in 0..self.spectrum.len() {
            self.complex_buffer[i] = self.spectrum[i];
        }
        for i in (1..n / 2).rev() {
            self.complex_buffer[n - i] = self.spectrum[i].conj();
        }

        // In-place IFFT
        fft_in_place(&mut self.complex_buffer, true);

        // Copy real part to output, scaled
        let scale = 1.0 / n as f32;
        for (i, c) in self.complex_buffer.iter().enumerate() {
            self.real_buffer[i] = c.re * scale;
        }
    }
}

/// Computes FFT into a pre-allocated workspace.
///
/// Results are stored in `workspace.spectrum()`.
pub fn fft_into(signal: &[f32], workspace: &mut SpectralWorkspace) {
    let n = workspace.fft_size;
    assert_eq!(
        signal.len(),
        n,
        "Signal length must match workspace FFT size"
    );

    // Copy signal to complex buffer
    for (i, &s) in signal.iter().enumerate() {
        workspace.complex_buffer[i] = Complex::new(s, 0.0);
    }

    // In-place FFT
    fft_in_place(&mut workspace.complex_buffer, false);

    // Copy positive frequencies to spectrum
    workspace.spectrum[..n / 2 + 1].copy_from_slice(&workspace.complex_buffer[..n / 2 + 1]);
}

/// Computes IFFT into a pre-allocated workspace.
///
/// Results are stored in `workspace.real_buffer()`.
pub fn ifft_into(spectrum: &[Complex], workspace: &mut SpectralWorkspace) {
    let n = workspace.fft_size;
    assert_eq!(
        spectrum.len(),
        n / 2 + 1,
        "Spectrum length must be N/2+1 for workspace FFT size N"
    );

    // Reconstruct full spectrum (conjugate symmetry)
    workspace.complex_buffer[..spectrum.len()].copy_from_slice(spectrum);
    for i in (1..n / 2).rev() {
        workspace.complex_buffer[n - i] = spectrum[i].conj();
    }

    // In-place IFFT
    fft_in_place(&mut workspace.complex_buffer, true);

    // Copy real part to output, scaled
    let scale = 1.0 / n as f32;
    for (i, c) in workspace.complex_buffer.iter().enumerate() {
        workspace.real_buffer[i] = c.re * scale;
    }
}

// ============================================================================
// 2D workspace for image processing
// ============================================================================

/// Pre-allocated buffers for 2D FFT operations on images.
#[derive(Debug, Clone)]
pub struct SpectralWorkspace2d {
    width: usize,
    height: usize,
    /// Complex buffer for the full image.
    buffer: Vec<Complex>,
    /// Column buffer for vertical passes.
    col_buffer: Vec<Complex>,
}

impl SpectralWorkspace2d {
    /// Creates a new workspace for the given image dimensions.
    ///
    /// # Panics
    /// Panics if width or height is not a power of 2.
    pub fn new(width: usize, height: usize) -> Self {
        assert!(width.is_power_of_two(), "Width must be power of 2");
        assert!(height.is_power_of_two(), "Height must be power of 2");
        Self {
            width,
            height,
            buffer: vec![Complex::default(); width * height],
            col_buffer: vec![Complex::default(); height],
        }
    }

    /// Returns the buffer containing the spectrum after fft2d_into.
    pub fn spectrum(&self) -> &[Complex] {
        &self.buffer
    }

    /// Returns mutable access to the spectrum.
    pub fn spectrum_mut(&mut self) -> &mut [Complex] {
        &mut self.buffer
    }

    /// Returns image dimensions.
    pub fn dimensions(&self) -> (usize, usize) {
        (self.width, self.height)
    }
}

/// Computes 2D FFT into a pre-allocated workspace.
pub fn fft2d_into(pixels: &[f32], workspace: &mut SpectralWorkspace2d) {
    let (width, height) = (workspace.width, workspace.height);
    assert_eq!(pixels.len(), width * height);

    // Convert to complex
    for (i, &p) in pixels.iter().enumerate() {
        workspace.buffer[i] = Complex::new(p, 0.0);
    }

    // FFT each row
    for row in 0..height {
        let start = row * width;
        fft_in_place(&mut workspace.buffer[start..start + width], false);
    }

    // FFT each column
    for col in 0..width {
        // Extract column
        for row in 0..height {
            workspace.col_buffer[row] = workspace.buffer[row * width + col];
        }

        // FFT column
        fft_in_place(&mut workspace.col_buffer, false);

        // Put back
        for row in 0..height {
            workspace.buffer[row * width + col] = workspace.col_buffer[row];
        }
    }
}

/// Computes inverse 2D FFT from workspace spectrum, returns real pixels.
pub fn ifft2d_from_workspace(workspace: &mut SpectralWorkspace2d) -> Vec<f32> {
    let (width, height) = (workspace.width, workspace.height);
    let scale = 1.0 / (width * height) as f32;

    // IFFT each row
    for row in 0..height {
        let start = row * width;
        fft_in_place(&mut workspace.buffer[start..start + width], true);
    }

    // IFFT each column
    for col in 0..width {
        // Extract column
        for row in 0..height {
            workspace.col_buffer[row] = workspace.buffer[row * width + col];
        }

        // IFFT column
        fft_in_place(&mut workspace.col_buffer, true);

        // Put back
        for row in 0..height {
            workspace.buffer[row * width + col] = workspace.col_buffer[row];
        }
    }

    // Return real part, scaled
    workspace.buffer.iter().map(|c| c.re * scale).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_complex_ops() {
        let a = Complex::new(3.0, 4.0);
        assert!((a.mag() - 5.0).abs() < 0.001);

        let b = Complex::new(1.0, 2.0);
        let c = a + b;
        assert!((c.re - 4.0).abs() < 0.001);
        assert!((c.im - 6.0).abs() < 0.001);
    }

    #[test]
    fn test_hann_window() {
        let window = hann_window(8);
        assert_eq!(window.len(), 8);
        // Hann window should be symmetric
        assert!((window[0] - window[7]).abs() < 0.001);
        assert!((window[1] - window[6]).abs() < 0.001);
        // Peak at center
        assert!(window[3] > window[0]);
        assert!(window[4] > window[0]);
    }

    #[test]
    fn test_fft_ifft_roundtrip() {
        let signal: Vec<f32> = (0..64)
            .map(|i| (2.0 * PI * 5.0 * i as f32 / 64.0).sin())
            .collect();

        let spectrum = fft(&signal);
        let recovered = ifft(&spectrum);

        assert_eq!(recovered.len(), signal.len());
        for (a, b) in signal.iter().zip(recovered.iter()) {
            assert!((a - b).abs() < 0.001);
        }
    }

    #[test]
    fn test_fft2d_ifft2d_roundtrip() {
        let width = 16;
        let height = 16;
        let pixels: Vec<f32> = (0..width * height)
            .map(|i| (i as f32 * 0.1).sin())
            .collect();

        let spectrum = fft2d(&pixels, width, height);
        let recovered = ifft2d(&spectrum, width, height);

        assert_eq!(recovered.len(), pixels.len());
        for (a, b) in pixels.iter().zip(recovered.iter()) {
            assert!((a - b).abs() < 0.01, "Mismatch: {} vs {}", a, b);
        }
    }

    #[test]
    fn test_dct_idct_roundtrip() {
        let signal: Vec<f32> = (0..8).map(|i| (i as f32 * 0.5).sin()).collect();

        let spectrum = dct(&signal);
        let recovered = idct(&spectrum);

        assert_eq!(recovered.len(), signal.len());
        for (a, b) in signal.iter().zip(recovered.iter()) {
            assert!((a - b).abs() < 0.01, "Mismatch: {} vs {}", a, b);
        }
    }

    #[test]
    fn test_dct2d_idct2d_roundtrip() {
        let width = 8;
        let height = 8;
        let pixels: Vec<f32> = (0..width * height)
            .map(|i| (i as f32 * 0.1).sin())
            .collect();

        let spectrum = dct2d(&pixels, width, height, None);
        let recovered = idct2d(&spectrum, width, height, None);

        assert_eq!(recovered.len(), pixels.len());
        for (a, b) in pixels.iter().zip(recovered.iter()) {
            assert!((a - b).abs() < 0.01, "Mismatch: {} vs {}", a, b);
        }
    }

    #[test]
    fn test_workspace_roundtrip() {
        let signal: Vec<f32> = (0..64)
            .map(|i| (2.0 * PI * 5.0 * i as f32 / 64.0).sin())
            .collect();

        let mut workspace = SpectralWorkspace::new(64);

        // FFT
        fft_into(&signal, &mut workspace);

        // IFFT
        workspace.ifft_from_spectrum();
        let recovered = workspace.real_buffer();

        for (a, b) in signal.iter().zip(recovered.iter()) {
            assert!((a - b).abs() < 0.001, "Mismatch: {} vs {}", a, b);
        }
    }

    #[test]
    fn test_workspace_2d_roundtrip() {
        let width = 16;
        let height = 16;
        let pixels: Vec<f32> = (0..width * height)
            .map(|i| (i as f32 * 0.1).sin())
            .collect();

        let mut workspace = SpectralWorkspace2d::new(width, height);

        // FFT
        fft2d_into(&pixels, &mut workspace);

        // IFFT
        let recovered = ifft2d_from_workspace(&mut workspace);

        for (a, b) in pixels.iter().zip(recovered.iter()) {
            assert!((a - b).abs() < 0.01, "Mismatch: {} vs {}", a, b);
        }
    }
}
