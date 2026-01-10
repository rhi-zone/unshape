//! Cross-domain data interpretation and conversion utilities.
//!
//! Inspired by MetaSynth and glitch art - the insight that structure is transferable
//! between domains. An image can become audio, audio can become vertices, noise can
//! be anything.
//!
//! # Example
//!
//! ```ignore
//! use rhizome_resin_crossdomain::{image_to_audio, audio_to_image, ImageToAudioConfig};
//!
//! // Convert an image to audio (MetaSynth-style spectral painting)
//! let config = ImageToAudioConfig::new(44100, 10.0);
//! let audio = image_to_audio(&image, &config);
//!
//! // Convert audio back to a spectrogram image
//! let spectrogram = audio_to_image(&audio, 44100, 512, 256);
//! ```

use glam::{Vec2, Vec3};
use rhizome_resin_audio::spectral::{StftConfig, stft_with_sample_rate};
use rhizome_resin_color::Rgba;
use rhizome_resin_field::{EvalContext, Field};
use rhizome_resin_image::ImageField;

// ============================================================================
// Buffer Reinterpretation
// ============================================================================

/// A view of float data as audio samples.
#[derive(Debug, Clone)]
pub struct AudioView<'a> {
    /// The underlying sample data.
    pub samples: &'a [f32],
    /// Sample rate in Hz.
    pub sample_rate: u32,
}

impl<'a> AudioView<'a> {
    /// Creates a new audio view from float data.
    pub fn new(samples: &'a [f32], sample_rate: u32) -> Self {
        Self {
            samples,
            sample_rate,
        }
    }

    /// Duration in seconds.
    pub fn duration(&self) -> f32 {
        self.samples.len() as f32 / self.sample_rate as f32
    }
}

/// A view of float data as 2D vertices.
#[derive(Debug, Clone)]
pub struct Vertices2DView<'a> {
    data: &'a [f32],
}

impl<'a> Vertices2DView<'a> {
    /// Creates a new 2D vertex view. Data length must be even.
    pub fn new(data: &'a [f32]) -> Option<Self> {
        if data.len() % 2 == 0 {
            Some(Self { data })
        } else {
            None
        }
    }

    /// Number of vertices.
    pub fn len(&self) -> usize {
        self.data.len() / 2
    }

    /// Returns true if there are no vertices.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Gets vertex at index.
    pub fn get(&self, index: usize) -> Option<Vec2> {
        if index < self.len() {
            let i = index * 2;
            Some(Vec2::new(self.data[i], self.data[i + 1]))
        } else {
            None
        }
    }

    /// Iterates over all vertices.
    pub fn iter(&self) -> impl Iterator<Item = Vec2> + '_ {
        (0..self.len()).map(|i| self.get(i).unwrap())
    }
}

/// A view of float data as 3D vertices.
#[derive(Debug, Clone)]
pub struct Vertices3DView<'a> {
    data: &'a [f32],
}

impl<'a> Vertices3DView<'a> {
    /// Creates a new 3D vertex view. Data length must be divisible by 3.
    pub fn new(data: &'a [f32]) -> Option<Self> {
        if data.len() % 3 == 0 {
            Some(Self { data })
        } else {
            None
        }
    }

    /// Number of vertices.
    pub fn len(&self) -> usize {
        self.data.len() / 3
    }

    /// Returns true if there are no vertices.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Gets vertex at index.
    pub fn get(&self, index: usize) -> Option<Vec3> {
        if index < self.len() {
            let i = index * 3;
            Some(Vec3::new(self.data[i], self.data[i + 1], self.data[i + 2]))
        } else {
            None
        }
    }

    /// Iterates over all vertices.
    pub fn iter(&self) -> impl Iterator<Item = Vec3> + '_ {
        (0..self.len()).map(|i| self.get(i).unwrap())
    }
}

/// A view of float data as RGBA pixels.
#[derive(Debug, Clone)]
pub struct PixelView<'a> {
    data: &'a [f32],
}

impl<'a> PixelView<'a> {
    /// Creates a new pixel view. Data length must be divisible by 4.
    pub fn new(data: &'a [f32]) -> Option<Self> {
        if data.len() % 4 == 0 {
            Some(Self { data })
        } else {
            None
        }
    }

    /// Number of pixels.
    pub fn len(&self) -> usize {
        self.data.len() / 4
    }

    /// Returns true if there are no pixels.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Gets pixel at index as [r, g, b, a].
    pub fn get(&self, index: usize) -> Option<[f32; 4]> {
        if index < self.len() {
            let i = index * 4;
            Some([
                self.data[i],
                self.data[i + 1],
                self.data[i + 2],
                self.data[i + 3],
            ])
        } else {
            None
        }
    }

    /// Iterates over all pixels.
    pub fn iter(&self) -> impl Iterator<Item = [f32; 4]> + '_ {
        (0..self.len()).map(|i| self.get(i).unwrap())
    }
}

// ============================================================================
// Raw Byte Casting (requires `bytemuck` feature)
// ============================================================================

/// Raw byte reinterpretation utilities for glitch-art style domain crossing.
///
/// These functions let you interpret any byte stream as audio samples, pixel data,
/// or vertex positions - regardless of what the bytes originally represented.
#[cfg(feature = "bytemuck")]
pub mod raw {
    use super::*;

    /// Reinterprets raw bytes as f32 audio samples.
    ///
    /// The bytes are cast directly to f32 values. Length must be divisible by 4.
    /// Values are not normalized - they may be outside [-1, 1].
    ///
    /// # Example
    /// ```ignore
    /// let jpeg_bytes = std::fs::read("image.jpg")?;
    /// let samples = bytes_as_f32(&jpeg_bytes)?;
    /// // Now you can hear what the JPEG "sounds like"
    /// ```
    pub fn bytes_as_f32(bytes: &[u8]) -> Option<&[f32]> {
        if bytes.len() % 4 != 0 {
            return None;
        }
        Some(bytemuck::cast_slice(bytes))
    }

    /// Reinterprets raw bytes as i16 audio samples.
    ///
    /// Common for raw PCM audio. Length must be divisible by 2.
    pub fn bytes_as_i16(bytes: &[u8]) -> Option<&[i16]> {
        if bytes.len() % 2 != 0 {
            return None;
        }
        Some(bytemuck::cast_slice(bytes))
    }

    /// Reinterprets raw bytes as u8 samples (0-255 range).
    ///
    /// No alignment requirements - every byte is a sample.
    pub fn bytes_as_u8(bytes: &[u8]) -> &[u8] {
        bytes
    }

    /// Converts i16 samples to f32 in [-1, 1] range.
    pub fn i16_to_f32(samples: &[i16]) -> Vec<f32> {
        samples
            .iter()
            .map(|&s| s as f32 / i16::MAX as f32)
            .collect()
    }

    /// Converts u8 samples to f32 in [-1, 1] range.
    pub fn u8_to_f32(samples: &[u8]) -> Vec<f32> {
        samples.iter().map(|&s| (s as f32 / 128.0) - 1.0).collect()
    }

    /// Normalizes f32 samples to [-1, 1] range.
    ///
    /// Useful after raw byte casting since values may be arbitrary floats.
    pub fn normalize_samples(samples: &[f32]) -> Vec<f32> {
        let max = samples
            .iter()
            .filter(|s| s.is_finite())
            .map(|s| s.abs())
            .fold(0.0f32, |a, b| a.max(b));

        if max == 0.0 {
            return samples.to_vec();
        }

        samples
            .iter()
            .map(|&s| if s.is_finite() { s / max } else { 0.0 })
            .collect()
    }

    /// Reinterprets raw bytes as RGBA pixels (4 bytes per pixel).
    ///
    /// Returns pixel data as [u8; 4] arrays.
    pub fn bytes_as_rgba_u8(bytes: &[u8]) -> Option<&[[u8; 4]]> {
        if bytes.len() % 4 != 0 {
            return None;
        }
        Some(bytemuck::cast_slice(bytes))
    }

    /// Reinterprets raw bytes as RGB pixels (3 bytes per pixel).
    ///
    /// Returns pixel data as [u8; 3] arrays.
    pub fn bytes_as_rgb_u8(bytes: &[u8]) -> Option<&[[u8; 3]]> {
        if bytes.len() % 3 != 0 {
            return None;
        }
        Some(bytemuck::cast_slice(bytes))
    }

    /// Reinterprets raw bytes as Vec2 vertices (8 bytes per vertex).
    pub fn bytes_as_vec2(bytes: &[u8]) -> Option<&[[f32; 2]]> {
        if bytes.len() % 8 != 0 {
            return None;
        }
        Some(bytemuck::cast_slice(bytes))
    }

    /// Reinterprets raw bytes as Vec3 vertices (12 bytes per vertex).
    pub fn bytes_as_vec3(bytes: &[u8]) -> Option<&[[f32; 3]]> {
        if bytes.len() % 12 != 0 {
            return None;
        }
        Some(bytemuck::cast_slice(bytes))
    }

    /// Creates an ImageField from raw bytes interpreted as RGBA pixels.
    ///
    /// Each 4 bytes become one RGBA pixel (u8 values scaled to 0-1).
    pub fn bytes_to_image(bytes: &[u8], width: u32, height: u32) -> Option<ImageField> {
        let expected = (width * height * 4) as usize;
        if bytes.len() < expected {
            return None;
        }

        let pixels: Vec<[f32; 4]> = bytes[..expected]
            .chunks_exact(4)
            .map(|chunk| {
                [
                    chunk[0] as f32 / 255.0,
                    chunk[1] as f32 / 255.0,
                    chunk[2] as f32 / 255.0,
                    chunk[3] as f32 / 255.0,
                ]
            })
            .collect();

        Some(ImageField::from_raw(pixels, width, height))
    }

    /// Creates an ImageField from raw bytes with automatic dimensions.
    ///
    /// Assumes square image, rounds down to nearest valid size.
    pub fn bytes_to_image_auto(bytes: &[u8]) -> Option<ImageField> {
        let num_pixels = bytes.len() / 4;
        if num_pixels == 0 {
            return None;
        }
        let side = (num_pixels as f32).sqrt() as u32;
        bytes_to_image(bytes, side, side)
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn test_bytes_as_f32() {
            let bytes: Vec<u8> = vec![0, 0, 128, 63, 0, 0, 0, 64]; // 1.0f32, 2.0f32 in little-endian
            let floats = bytes_as_f32(&bytes).unwrap();
            assert_eq!(floats.len(), 2);
            assert!((floats[0] - 1.0).abs() < 0.0001);
            assert!((floats[1] - 2.0).abs() < 0.0001);
        }

        #[test]
        fn test_bytes_as_f32_invalid_len() {
            let bytes = vec![0, 0, 128]; // 3 bytes, not divisible by 4
            assert!(bytes_as_f32(&bytes).is_none());
        }

        #[test]
        fn test_bytes_as_i16() {
            let bytes: Vec<u8> = vec![0, 0, 255, 127]; // 0i16, 32767i16 in little-endian
            let samples = bytes_as_i16(&bytes).unwrap();
            assert_eq!(samples.len(), 2);
            assert_eq!(samples[0], 0);
            assert_eq!(samples[1], i16::MAX);
        }

        #[test]
        fn test_i16_to_f32() {
            let samples = vec![0i16, i16::MAX, i16::MIN];
            let floats = i16_to_f32(&samples);
            assert!((floats[0] - 0.0).abs() < 0.0001);
            assert!((floats[1] - 1.0).abs() < 0.0001);
            assert!((floats[2] - (-1.0)).abs() < 0.01);
        }

        #[test]
        fn test_u8_to_f32() {
            let samples = vec![0u8, 128, 255];
            let floats = u8_to_f32(&samples);
            assert!((floats[0] - (-1.0)).abs() < 0.01);
            assert!((floats[1] - 0.0).abs() < 0.01);
            assert!((floats[2] - 1.0).abs() < 0.01);
        }

        #[test]
        fn test_normalize_samples() {
            let samples = vec![0.0, 5.0, -10.0, 2.5];
            let normalized = normalize_samples(&samples);
            assert!((normalized[0] - 0.0).abs() < 0.0001);
            assert!((normalized[1] - 0.5).abs() < 0.0001);
            assert!((normalized[2] - (-1.0)).abs() < 0.0001);
            assert!((normalized[3] - 0.25).abs() < 0.0001);
        }

        #[test]
        fn test_normalize_with_nan() {
            let samples = vec![1.0, f32::NAN, -2.0];
            let normalized = normalize_samples(&samples);
            assert!((normalized[0] - 0.5).abs() < 0.0001);
            assert_eq!(normalized[1], 0.0); // NaN becomes 0
            assert!((normalized[2] - (-1.0)).abs() < 0.0001);
        }

        #[test]
        fn test_bytes_as_rgba_u8() {
            let bytes = vec![255, 0, 0, 255, 0, 255, 0, 128];
            let pixels = bytes_as_rgba_u8(&bytes).unwrap();
            assert_eq!(pixels.len(), 2);
            assert_eq!(pixels[0], [255, 0, 0, 255]);
            assert_eq!(pixels[1], [0, 255, 0, 128]);
        }

        #[test]
        fn test_bytes_as_vec3() {
            // 12 bytes = 3 floats = 1 Vec3
            let v = 1.0f32;
            let mut bytes = Vec::new();
            bytes.extend_from_slice(&v.to_le_bytes());
            bytes.extend_from_slice(&2.0f32.to_le_bytes());
            bytes.extend_from_slice(&3.0f32.to_le_bytes());

            let verts = bytes_as_vec3(&bytes).unwrap();
            assert_eq!(verts.len(), 1);
            assert!((verts[0][0] - 1.0).abs() < 0.0001);
            assert!((verts[0][1] - 2.0).abs() < 0.0001);
            assert!((verts[0][2] - 3.0).abs() < 0.0001);
        }

        #[test]
        fn test_bytes_to_image() {
            // 4 pixels (2x2)
            let bytes = vec![
                255, 0, 0, 255, // red
                0, 255, 0, 255, // green
                0, 0, 255, 255, // blue
                255, 255, 0, 255, // yellow
            ];
            let img = bytes_to_image(&bytes, 2, 2).unwrap();
            assert_eq!(img.dimensions(), (2, 2));
        }

        #[test]
        fn test_bytes_to_image_auto() {
            // 16 pixels worth of bytes = 4x4 image
            let bytes = vec![128u8; 64];
            let img = bytes_to_image_auto(&bytes).unwrap();
            assert_eq!(img.dimensions(), (4, 4));
        }
    }
}

// ============================================================================
// Image to Audio Conversion
// ============================================================================

/// Configuration for image-to-audio conversion.
#[derive(Debug, Clone)]
pub struct ImageToAudioConfig {
    /// Sample rate in Hz.
    pub sample_rate: u32,
    /// Duration of the output audio in seconds.
    pub duration: f32,
    /// Minimum frequency (Hz) for the bottom of the image.
    pub min_freq: f32,
    /// Maximum frequency (Hz) for the top of the image.
    pub max_freq: f32,
    /// Whether to use logarithmic frequency scaling.
    pub log_frequency: bool,
}

impl ImageToAudioConfig {
    /// Creates a new config with default frequency range (80 Hz - 8000 Hz).
    pub fn new(sample_rate: u32, duration: f32) -> Self {
        Self {
            sample_rate,
            duration,
            min_freq: 80.0,
            max_freq: 8000.0,
            log_frequency: true,
        }
    }

    /// Sets the frequency range.
    pub fn with_frequency_range(mut self, min: f32, max: f32) -> Self {
        self.min_freq = min;
        self.max_freq = max;
        self
    }

    /// Sets whether to use logarithmic frequency scaling.
    pub fn with_log_frequency(mut self, log: bool) -> Self {
        self.log_frequency = log;
        self
    }
}

/// Converts an image to audio using additive synthesis.
///
/// Each row of the image represents a frequency band. Brightness controls amplitude.
/// Time progresses from left to right across the image.
///
/// This is inspired by MetaSynth's "paint with sound" approach.
pub fn image_to_audio(image: &ImageField, config: &ImageToAudioConfig) -> Vec<f32> {
    let num_samples = (config.sample_rate as f32 * config.duration) as usize;
    let mut output = vec![0.0f32; num_samples];

    let (width, height) = image.dimensions();
    if width == 0 || height == 0 {
        return output;
    }

    let ctx = EvalContext::new();

    // For each frequency band (row of the image)
    for y in 0..height {
        // Map row to frequency
        let t = y as f32 / (height - 1).max(1) as f32;
        let freq = if config.log_frequency {
            // Logarithmic mapping for perceptually uniform distribution
            config.min_freq * (config.max_freq / config.min_freq).powf(1.0 - t)
        } else {
            // Linear mapping
            config.min_freq + (config.max_freq - config.min_freq) * (1.0 - t)
        };

        // Phase increment per sample for this frequency
        let phase_inc = 2.0 * std::f32::consts::PI * freq / config.sample_rate as f32;

        // Generate this frequency band's contribution
        for (sample_idx, sample) in output.iter_mut().enumerate() {
            // Map sample position to image X coordinate
            let time_t = sample_idx as f32 / num_samples as f32;
            let uv = Vec2::new(time_t, y as f32 / height as f32);
            let color: Rgba = image.sample(uv, &ctx);

            // Use brightness as amplitude (average of RGB)
            let brightness = (color.r + color.g + color.b) / 3.0;

            // Add this oscillator's contribution
            let phase = phase_inc * sample_idx as f32;
            *sample += brightness * phase.sin() * 0.5 / (height as f32).sqrt();
        }
    }

    // Normalize to prevent clipping
    let max_amp = output.iter().map(|s| s.abs()).fold(0.0f32, |a, b| a.max(b));
    if max_amp > 0.0 {
        let scale = 0.9 / max_amp;
        for sample in &mut output {
            *sample *= scale;
        }
    }

    output
}

/// Configuration for audio-to-image conversion.
#[derive(Debug, Clone)]
pub struct AudioToImageConfig {
    /// Output image width in pixels.
    pub width: u32,
    /// Output image height in pixels (number of frequency bins).
    pub height: u32,
    /// FFT window size.
    pub window_size: usize,
    /// Hop size between windows.
    pub hop_size: usize,
    /// Whether to use logarithmic magnitude scaling.
    pub log_magnitude: bool,
    /// Gain applied to magnitude values.
    pub gain: f32,
}

impl AudioToImageConfig {
    /// Creates a new config with reasonable defaults.
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            window_size: 2048,
            hop_size: 512,
            log_magnitude: true,
            gain: 1.0,
        }
    }

    /// Sets the FFT parameters.
    pub fn with_fft(mut self, window_size: usize, hop_size: usize) -> Self {
        self.window_size = window_size;
        self.hop_size = hop_size;
        self
    }

    /// Sets the gain.
    pub fn with_gain(mut self, gain: f32) -> Self {
        self.gain = gain;
        self
    }
}

/// Converts audio to a spectrogram image.
///
/// The resulting image represents frequency (vertical) vs time (horizontal),
/// with brightness indicating magnitude.
pub fn audio_to_image(audio: &[f32], sample_rate: u32, config: &AudioToImageConfig) -> ImageField {
    let stft_config = StftConfig::new(config.window_size).with_hop_size(config.hop_size);
    let stft = stft_with_sample_rate(audio, &stft_config, Some(sample_rate));
    let magnitudes = stft.magnitude();

    let num_frames = magnitudes.len();
    let num_bins = if magnitudes.is_empty() {
        0
    } else {
        magnitudes[0].len()
    };

    // Create output image
    let mut pixels = vec![[0.0f32; 4]; (config.width * config.height) as usize];

    if num_frames == 0 || num_bins == 0 {
        return ImageField::from_raw(pixels, config.width, config.height);
    }

    // Find max magnitude for normalization
    let max_mag = magnitudes
        .iter()
        .flat_map(|frame| frame.iter())
        .fold(0.0f32, |a, &b| a.max(b));

    // Fill pixels
    for y in 0..config.height {
        for x in 0..config.width {
            // Map pixel to spectrogram position
            let time_t = x as f32 / config.width as f32;
            let freq_t = 1.0 - (y as f32 / config.height as f32); // Flip so high freq is at top

            let frame_idx = ((time_t * num_frames as f32) as usize).min(num_frames - 1);
            let bin_idx = ((freq_t * num_bins as f32) as usize).min(num_bins - 1);

            let mut mag = magnitudes[frame_idx][bin_idx] / max_mag.max(0.0001);

            if config.log_magnitude {
                // Convert to dB-like scale
                mag = (1.0 + mag * config.gain).ln() / (1.0 + config.gain).ln();
            } else {
                mag *= config.gain;
            }

            mag = mag.clamp(0.0, 1.0);

            let pixel_idx = (y * config.width + x) as usize;
            pixels[pixel_idx] = [mag, mag, mag, 1.0];
        }
    }

    ImageField::from_raw(pixels, config.width, config.height)
}

/// Converts audio to a colored spectrogram based on phase.
///
/// Magnitude controls brightness, phase controls hue.
pub fn audio_to_image_colored(
    audio: &[f32],
    sample_rate: u32,
    config: &AudioToImageConfig,
) -> ImageField {
    let stft_config = StftConfig::new(config.window_size).with_hop_size(config.hop_size);
    let stft = stft_with_sample_rate(audio, &stft_config, Some(sample_rate));
    let magnitudes = stft.magnitude();
    let phases = stft.phase();

    let num_frames = magnitudes.len();
    let num_bins = if magnitudes.is_empty() {
        0
    } else {
        magnitudes[0].len()
    };

    let mut pixels = vec![[0.0f32; 4]; (config.width * config.height) as usize];

    if num_frames == 0 || num_bins == 0 {
        return ImageField::from_raw(pixels, config.width, config.height);
    }

    let max_mag = magnitudes
        .iter()
        .flat_map(|frame| frame.iter())
        .fold(0.0f32, |a, &b| a.max(b));

    for y in 0..config.height {
        for x in 0..config.width {
            let time_t = x as f32 / config.width as f32;
            let freq_t = 1.0 - (y as f32 / config.height as f32);

            let frame_idx = ((time_t * num_frames as f32) as usize).min(num_frames - 1);
            let bin_idx = ((freq_t * num_bins as f32) as usize).min(num_bins - 1);

            let mut mag = magnitudes[frame_idx][bin_idx] / max_mag.max(0.0001);
            let phase = phases[frame_idx][bin_idx];

            if config.log_magnitude {
                mag = (1.0 + mag * config.gain).ln() / (1.0 + config.gain).ln();
            } else {
                mag *= config.gain;
            }
            mag = mag.clamp(0.0, 1.0);

            // Phase to hue (0 to 2π maps to 0 to 1)
            let hue = (phase + std::f32::consts::PI) / (2.0 * std::f32::consts::PI);
            let (r, g, b) = hsl_to_rgb(hue, 0.8, mag * 0.5);

            let pixel_idx = (y * config.width + x) as usize;
            pixels[pixel_idx] = [r, g, b, 1.0];
        }
    }

    ImageField::from_raw(pixels, config.width, config.height)
}

/// Converts HSL to RGB.
fn hsl_to_rgb(h: f32, s: f32, l: f32) -> (f32, f32, f32) {
    if s == 0.0 {
        return (l, l, l);
    }

    let q = if l < 0.5 {
        l * (1.0 + s)
    } else {
        l + s - l * s
    };
    let p = 2.0 * l - q;

    let r = hue_to_rgb(p, q, h + 1.0 / 3.0);
    let g = hue_to_rgb(p, q, h);
    let b = hue_to_rgb(p, q, h - 1.0 / 3.0);

    (r, g, b)
}

fn hue_to_rgb(p: f32, q: f32, mut t: f32) -> f32 {
    if t < 0.0 {
        t += 1.0;
    }
    if t > 1.0 {
        t -= 1.0;
    }

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

// ============================================================================
// Noise as Anything
// ============================================================================

/// Samples a 1D field to generate audio samples.
///
/// Treats the field's X axis as time, normalized to [0, 1] over the duration.
pub fn field_to_audio<F>(field: &F, sample_rate: u32, duration: f32) -> Vec<f32>
where
    F: Field<f32, f32>,
{
    let num_samples = (sample_rate as f32 * duration) as usize;
    let ctx = EvalContext::new();

    (0..num_samples)
        .map(|i| {
            let t = i as f32 / num_samples as f32;
            field.sample(t, &ctx)
        })
        .collect()
}

/// Samples a 2D field to generate audio using the Y axis for stereo panning.
///
/// Y=0 is full left, Y=1 is full right.
pub fn field_to_audio_stereo<F>(field: &F, sample_rate: u32, duration: f32) -> Vec<(f32, f32)>
where
    F: Field<Vec2, f32>,
{
    let num_samples = (sample_rate as f32 * duration) as usize;
    let ctx = EvalContext::new();

    (0..num_samples)
        .map(|i| {
            let t = i as f32 / num_samples as f32;
            let left = field.sample(Vec2::new(t, 0.0), &ctx);
            let right = field.sample(Vec2::new(t, 1.0), &ctx);
            (left, right)
        })
        .collect()
}

/// Samples a 2D field to generate an image.
pub fn field_to_image<F>(field: &F, width: u32, height: u32) -> ImageField
where
    F: Field<Vec2, f32>,
{
    let ctx = EvalContext::new();
    let mut pixels = Vec::with_capacity((width * height) as usize);

    for y in 0..height {
        for x in 0..width {
            let uv = Vec2::new(x as f32 / width as f32, y as f32 / height as f32);
            let value = field.sample(uv, &ctx).clamp(0.0, 1.0);
            pixels.push([value, value, value, 1.0]);
        }
    }

    ImageField::from_raw(pixels, width, height)
}

/// Samples an RGBA field to generate an image.
pub fn field_rgba_to_image<F>(field: &F, width: u32, height: u32) -> ImageField
where
    F: Field<Vec2, Rgba>,
{
    let ctx = EvalContext::new();
    let mut pixels = Vec::with_capacity((width * height) as usize);

    for y in 0..height {
        for x in 0..width {
            let uv = Vec2::new(x as f32 / width as f32, y as f32 / height as f32);
            let color = field.sample(uv, &ctx);
            pixels.push([color.r, color.g, color.b, color.a]);
        }
    }

    ImageField::from_raw(pixels, width, height)
}

/// Samples a 2D field along a path to generate vertices.
pub fn field_to_vertices_2d<F>(field: &F, num_points: usize, scale: f32) -> Vec<Vec2>
where
    F: Field<f32, Vec2>,
{
    let ctx = EvalContext::new();

    (0..num_points)
        .map(|i| {
            let t = i as f32 / num_points as f32;
            field.sample(t, &ctx) * scale
        })
        .collect()
}

/// Samples a 3D field along a path to generate vertices.
pub fn field_to_vertices_3d<F>(field: &F, num_points: usize, scale: f32) -> Vec<Vec3>
where
    F: Field<f32, Vec3>,
{
    let ctx = EvalContext::new();

    (0..num_points)
        .map(|i| {
            let t = i as f32 / num_points as f32;
            field.sample(t, &ctx) * scale
        })
        .collect()
}

/// Samples a 2D scalar field on a grid to get displacement values.
pub fn field_to_displacement<F>(field: &F, width: usize, height: usize) -> Vec<f32>
where
    F: Field<Vec2, f32>,
{
    let ctx = EvalContext::new();
    let mut values = Vec::with_capacity(width * height);

    for y in 0..height {
        for x in 0..width {
            let uv = Vec2::new(x as f32 / width as f32, y as f32 / height as f32);
            values.push(field.sample(uv, &ctx));
        }
    }

    values
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_audio_view() {
        let samples = vec![0.1, 0.2, 0.3, 0.4];
        let view = AudioView::new(&samples, 44100);
        assert_eq!(view.samples.len(), 4);
        assert_eq!(view.sample_rate, 44100);
        assert!((view.duration() - 4.0 / 44100.0).abs() < 0.0001);
    }

    #[test]
    fn test_vertices_2d_view() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let view = Vertices2DView::new(&data).unwrap();
        assert_eq!(view.len(), 3);
        assert_eq!(view.get(0), Some(Vec2::new(1.0, 2.0)));
        assert_eq!(view.get(1), Some(Vec2::new(3.0, 4.0)));
        assert_eq!(view.get(2), Some(Vec2::new(5.0, 6.0)));
        assert_eq!(view.get(3), None);
    }

    #[test]
    fn test_vertices_2d_view_invalid() {
        let data = vec![1.0, 2.0, 3.0]; // Odd length
        assert!(Vertices2DView::new(&data).is_none());
    }

    #[test]
    fn test_vertices_3d_view() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let view = Vertices3DView::new(&data).unwrap();
        assert_eq!(view.len(), 2);
        assert_eq!(view.get(0), Some(Vec3::new(1.0, 2.0, 3.0)));
        assert_eq!(view.get(1), Some(Vec3::new(4.0, 5.0, 6.0)));
    }

    #[test]
    fn test_vertices_3d_view_invalid() {
        let data = vec![1.0, 2.0]; // Not divisible by 3
        assert!(Vertices3DView::new(&data).is_none());
    }

    #[test]
    fn test_pixel_view() {
        let data = vec![1.0, 0.5, 0.0, 1.0, 0.0, 0.5, 1.0, 0.8];
        let view = PixelView::new(&data).unwrap();
        assert_eq!(view.len(), 2);
        assert_eq!(view.get(0), Some([1.0, 0.5, 0.0, 1.0]));
        assert_eq!(view.get(1), Some([0.0, 0.5, 1.0, 0.8]));
    }

    #[test]
    fn test_pixel_view_invalid() {
        let data = vec![1.0, 0.5, 0.0]; // Not divisible by 4
        assert!(PixelView::new(&data).is_none());
    }

    #[test]
    fn test_image_to_audio_config() {
        let config = ImageToAudioConfig::new(44100, 2.0)
            .with_frequency_range(100.0, 4000.0)
            .with_log_frequency(false);
        assert_eq!(config.sample_rate, 44100);
        assert_eq!(config.duration, 2.0);
        assert_eq!(config.min_freq, 100.0);
        assert_eq!(config.max_freq, 4000.0);
        assert!(!config.log_frequency);
    }

    #[test]
    fn test_image_to_audio_empty() {
        let image = ImageField::from_raw(vec![], 0, 0);
        let config = ImageToAudioConfig::new(44100, 1.0);
        let audio = image_to_audio(&image, &config);
        assert_eq!(audio.len(), 44100);
        assert!(audio.iter().all(|&s| s == 0.0));
    }

    #[test]
    fn test_image_to_audio_simple() {
        // Create a simple 4x4 white image
        let pixels: Vec<[f32; 4]> = vec![[1.0, 1.0, 1.0, 1.0]; 16];
        let image = ImageField::from_raw(pixels, 4, 4);
        let config = ImageToAudioConfig::new(44100, 0.1);
        let audio = image_to_audio(&image, &config);

        // Should have generated audio
        assert_eq!(audio.len(), 4410);
        // Should have some non-zero values
        assert!(audio.iter().any(|&s| s != 0.0));
        // Should be normalized
        assert!(audio.iter().all(|&s| s.abs() <= 1.0));
    }

    #[test]
    fn test_audio_to_image_config() {
        let config = AudioToImageConfig::new(512, 256)
            .with_fft(1024, 256)
            .with_gain(2.0);
        assert_eq!(config.width, 512);
        assert_eq!(config.height, 256);
        assert_eq!(config.window_size, 1024);
        assert_eq!(config.hop_size, 256);
        assert_eq!(config.gain, 2.0);
    }

    #[test]
    fn test_audio_to_image_empty() {
        let audio: Vec<f32> = vec![];
        let config = AudioToImageConfig::new(64, 32);
        let image = audio_to_image(&audio, 44100, &config);
        let (w, h) = image.dimensions();
        assert_eq!(w, 64);
        assert_eq!(h, 32);
    }

    #[test]
    fn test_audio_to_image_sine() {
        // Generate a simple sine wave
        let sample_rate = 44100;
        let freq = 440.0;
        let duration = 0.5;
        let num_samples = (sample_rate as f32 * duration) as usize;

        let audio: Vec<f32> = (0..num_samples)
            .map(|i| {
                let t = i as f32 / sample_rate as f32;
                (2.0 * std::f32::consts::PI * freq * t).sin()
            })
            .collect();

        let config = AudioToImageConfig::new(128, 64).with_fft(1024, 256);
        let image = audio_to_image(&audio, sample_rate, &config);

        let (w, h) = image.dimensions();
        assert_eq!(w, 128);
        assert_eq!(h, 64);
    }

    #[test]
    fn test_hsl_to_rgb() {
        // Red
        let (r, g, b) = hsl_to_rgb(0.0, 1.0, 0.5);
        assert!((r - 1.0).abs() < 0.01);
        assert!(g < 0.01);
        assert!(b < 0.01);

        // Green
        let (r, g, b) = hsl_to_rgb(1.0 / 3.0, 1.0, 0.5);
        assert!(r < 0.01);
        assert!((g - 1.0).abs() < 0.01);
        assert!(b < 0.01);

        // Gray (no saturation)
        let (r, g, b) = hsl_to_rgb(0.5, 0.0, 0.5);
        assert!((r - 0.5).abs() < 0.01);
        assert!((g - 0.5).abs() < 0.01);
        assert!((b - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_vertices_2d_iter() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let view = Vertices2DView::new(&data).unwrap();
        let verts: Vec<_> = view.iter().collect();
        assert_eq!(verts.len(), 2);
        assert_eq!(verts[0], Vec2::new(1.0, 2.0));
        assert_eq!(verts[1], Vec2::new(3.0, 4.0));
    }

    #[test]
    fn test_vertices_3d_iter() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let view = Vertices3DView::new(&data).unwrap();
        let verts: Vec<_> = view.iter().collect();
        assert_eq!(verts.len(), 2);
        assert_eq!(verts[0], Vec3::new(1.0, 2.0, 3.0));
        assert_eq!(verts[1], Vec3::new(4.0, 5.0, 6.0));
    }

    #[test]
    fn test_pixel_iter() {
        let data = vec![1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0];
        let view = PixelView::new(&data).unwrap();
        let pixels: Vec<_> = view.iter().collect();
        assert_eq!(pixels.len(), 2);
        assert_eq!(pixels[0], [1.0, 0.0, 0.0, 1.0]);
        assert_eq!(pixels[1], [0.0, 1.0, 0.0, 1.0]);
    }
}
