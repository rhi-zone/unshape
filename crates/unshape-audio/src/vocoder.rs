//! Vocoder for spectral cross-synthesis.
//!
//! A vocoder analyzes the spectral envelope of a modulator signal (typically speech)
//! and applies it to a carrier signal (typically a synthesizer), creating the
//! classic "talking robot" effect.
//!
//! # Example
//!
//! ```
//! use unshape_audio::vocoder::{Vocoder, VocodeSynth};
//!
//! let config = VocodeSynth::default();
//! let mut vocoder = Vocoder::new(config);
//!
//! // Process a block of audio
//! let carrier = vec![0.5; 1024]; // Example carrier
//! let modulator = vec![0.3; 1024]; // Example modulator
//! let output = vocoder.process(&carrier, &modulator);
//! ```

use crate::spectral::{Complex, fft, hann_window, ifft};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Input for vocoder operation.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct VocodeInput {
    /// Carrier signal (synthesizer, noise, etc.).
    pub carrier: Vec<f32>,
    /// Modulator signal (speech, etc.).
    pub modulator: Vec<f32>,
}

/// Configuration for the vocoder.
///
/// Operations on audio buffers use the ops-as-values pattern.
/// See `docs/design/ops-as-values.md`.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dynop", derive(unshape_op::Op))]
#[cfg_attr(feature = "dynop", op(input = VocodeInput, output = Vec<f32>))]
pub struct VocodeSynth {
    /// FFT window size (must be power of 2).
    pub window_size: usize,
    /// Hop size between consecutive frames.
    pub hop_size: usize,
    /// Number of frequency bands for the filterbank.
    pub num_bands: usize,
    /// Envelope follower smoothing (0-1, higher = smoother).
    pub envelope_smoothing: f32,
}

impl Default for VocodeSynth {
    fn default() -> Self {
        Self {
            window_size: 1024,
            hop_size: 256,
            num_bands: 16,
            envelope_smoothing: 0.9,
        }
    }
}

impl VocodeSynth {
    /// Creates a new vocoder config with the given window size.
    pub fn new(window_size: usize) -> Self {
        Self {
            window_size,
            hop_size: window_size / 4,
            ..Default::default()
        }
    }

    /// Sets the number of bands.
    pub fn with_bands(mut self, num_bands: usize) -> Self {
        self.num_bands = num_bands.max(2);
        self
    }

    /// Sets the envelope smoothing.
    pub fn with_smoothing(mut self, smoothing: f32) -> Self {
        self.envelope_smoothing = smoothing.clamp(0.0, 0.999);
        self
    }

    /// Applies this vocoder configuration to process carrier and modulator signals.
    pub fn apply(&self, input: &VocodeInput) -> Vec<f32> {
        let mut vocoder = Vocoder::new(self.clone());
        vocoder.process(&input.carrier, &input.modulator)
    }
}

/// Backwards-compatible type alias.
pub type VocoderConfig = VocodeSynth;

/// Band definition for filterbank vocoder.
#[derive(Debug, Clone)]
struct Band {
    /// Start bin index (inclusive).
    start_bin: usize,
    /// End bin index (exclusive).
    end_bin: usize,
    /// Current envelope value.
    envelope: f32,
}

/// A phase vocoder for spectral cross-synthesis.
pub struct Vocoder {
    config: VocodeSynth,
    window: Vec<f32>,
    bands: Vec<Band>,
    /// Input buffer for carrier.
    carrier_buffer: Vec<f32>,
    /// Input buffer for modulator.
    modulator_buffer: Vec<f32>,
    /// Output buffer with overlap-add.
    output_buffer: Vec<f32>,
    /// Position in the output buffer.
    output_position: usize,
}

impl Vocoder {
    /// Creates a new vocoder with the given configuration.
    pub fn new(config: VocodeSynth) -> Self {
        let window = hann_window(config.window_size);
        let num_bins = config.window_size / 2 + 1;

        // Create logarithmically spaced frequency bands
        let bands = create_bands(num_bins, config.num_bands);

        Self {
            config: config.clone(),
            window,
            bands,
            carrier_buffer: vec![0.0; config.window_size],
            modulator_buffer: vec![0.0; config.window_size],
            output_buffer: vec![0.0; config.window_size * 2],
            output_position: 0,
        }
    }

    /// Processes a block of audio.
    ///
    /// Both carrier and modulator should have the same length.
    /// Returns the vocoded output.
    pub fn process(&mut self, carrier: &[f32], modulator: &[f32]) -> Vec<f32> {
        assert_eq!(carrier.len(), modulator.len());

        let mut output = Vec::with_capacity(carrier.len());
        let mut pos = 0;

        while pos < carrier.len() {
            // Fill buffers
            let remaining = carrier.len() - pos;
            let to_copy = remaining.min(self.config.hop_size);

            // Shift buffers left
            let shift = self.config.window_size - self.config.hop_size;
            self.carrier_buffer.copy_within(self.config.hop_size.., 0);
            self.modulator_buffer.copy_within(self.config.hop_size.., 0);

            // Add new samples
            self.carrier_buffer[shift..shift + to_copy]
                .copy_from_slice(&carrier[pos..pos + to_copy]);
            self.modulator_buffer[shift..shift + to_copy]
                .copy_from_slice(&modulator[pos..pos + to_copy]);

            // Process when we have enough samples
            if to_copy == self.config.hop_size {
                let frame = self.process_frame();

                // Overlap-add
                for (i, &sample) in frame.iter().enumerate() {
                    let out_idx = (self.output_position + i) % self.output_buffer.len();
                    self.output_buffer[out_idx] += sample;
                }
            }

            // Output samples
            for _ in 0..to_copy {
                let out_idx = self.output_position % self.output_buffer.len();
                output.push(self.output_buffer[out_idx]);
                self.output_buffer[out_idx] = 0.0;
                self.output_position += 1;
            }

            pos += to_copy;
        }

        output
    }

    /// Processes a single frame.
    fn process_frame(&mut self) -> Vec<f32> {
        // Apply window to both signals
        let carrier_windowed: Vec<f32> = self
            .carrier_buffer
            .iter()
            .zip(&self.window)
            .map(|(s, w)| s * w)
            .collect();

        let modulator_windowed: Vec<f32> = self
            .modulator_buffer
            .iter()
            .zip(&self.window)
            .map(|(s, w)| s * w)
            .collect();

        // FFT
        let carrier_spectrum = fft(&carrier_windowed);
        let modulator_spectrum = fft(&modulator_windowed);

        // Compute band envelopes from modulator
        let modulator_envelopes = self.compute_band_envelopes(&modulator_spectrum);

        // Update smoothed envelopes
        for (band, env) in self.bands.iter_mut().zip(modulator_envelopes.iter()) {
            band.envelope = band.envelope * self.config.envelope_smoothing
                + env * (1.0 - self.config.envelope_smoothing);
        }

        // Apply envelopes to carrier
        let output_spectrum = self.apply_envelopes(&carrier_spectrum);

        // IFFT
        let output_frame = ifft(&output_spectrum);

        // Apply synthesis window
        output_frame
            .iter()
            .zip(&self.window)
            .map(|(s, w)| s * w)
            .collect()
    }

    /// Computes band envelopes from a spectrum.
    fn compute_band_envelopes(&self, spectrum: &[Complex]) -> Vec<f32> {
        self.bands
            .iter()
            .map(|band| {
                let mut sum = 0.0;
                for bin in band.start_bin..band.end_bin.min(spectrum.len()) {
                    sum += spectrum[bin].mag();
                }
                let count = (band.end_bin - band.start_bin).max(1);
                sum / count as f32
            })
            .collect()
    }

    /// Applies band envelopes to a spectrum.
    fn apply_envelopes(&self, spectrum: &[Complex]) -> Vec<Complex> {
        let mut output = spectrum.to_vec();

        for band in &self.bands {
            // Compute current band energy in carrier
            let mut carrier_energy = 0.0f32;
            for bin in band.start_bin..band.end_bin.min(spectrum.len()) {
                carrier_energy += spectrum[bin].mag();
            }
            let count = (band.end_bin - band.start_bin).max(1);
            let avg_carrier = carrier_energy / count as f32;

            // Compute gain to match modulator envelope
            let gain = if avg_carrier > 1e-10 {
                band.envelope / avg_carrier
            } else {
                0.0
            };

            // Apply gain to bins in this band
            for bin in band.start_bin..band.end_bin.min(output.len()) {
                output[bin] = output[bin] * gain;
            }
        }

        output
    }

    /// Resets the vocoder state.
    pub fn reset(&mut self) {
        self.carrier_buffer.fill(0.0);
        self.modulator_buffer.fill(0.0);
        self.output_buffer.fill(0.0);
        self.output_position = 0;
        for band in &mut self.bands {
            band.envelope = 0.0;
        }
    }
}

/// Creates logarithmically spaced frequency bands.
fn create_bands(num_bins: usize, num_bands: usize) -> Vec<Band> {
    let mut bands = Vec::with_capacity(num_bands);

    // Logarithmic spacing (Bark-like scale approximation)
    let min_freq: f32 = 1.0;
    let max_freq: f32 = num_bins as f32;
    let log_min = min_freq.ln();
    let log_max = max_freq.ln();

    for i in 0..num_bands {
        let t0 = i as f32 / num_bands as f32;
        let t1 = (i + 1) as f32 / num_bands as f32;

        let start = (log_min + t0 * (log_max - log_min)).exp().round() as usize;
        let end = (log_min + t1 * (log_max - log_min)).exp().round() as usize;

        bands.push(Band {
            start_bin: start.min(num_bins - 1),
            end_bin: end.min(num_bins),
            envelope: 0.0,
        });
    }

    bands
}

// ============================================================================
// Simple filterbank vocoder (alternative implementation)
// ============================================================================

/// A simple filterbank-based vocoder.
///
/// This is a more traditional vocoder implementation using bandpass filters
/// and envelope followers, operating in the time domain.
pub struct FilterbankVocoder {
    /// Number of bands.
    num_bands: usize,
    /// Band envelope followers.
    envelopes: Vec<f32>,
    /// Attack coefficient.
    attack: f32,
    /// Release coefficient.
    release: f32,
    /// Band center frequencies (normalized 0-0.5).
    band_centers: Vec<f32>,
    /// Band widths (Q factor inverse).
    band_widths: Vec<f32>,
    /// Filter states for carrier (2 per band for biquad).
    carrier_states: Vec<[f32; 4]>,
    /// Filter states for modulator.
    modulator_states: Vec<[f32; 4]>,
}

impl FilterbankVocoder {
    /// Creates a new filterbank vocoder.
    pub fn new(num_bands: usize, sample_rate: f32) -> Self {
        let attack = (-1.0 / (sample_rate * 0.001)).exp(); // 1ms attack
        let release = (-1.0 / (sample_rate * 0.02)).exp(); // 20ms release

        // Create logarithmically spaced bands from 100Hz to Nyquist
        let min_freq: f32 = 100.0 / sample_rate;
        let max_freq: f32 = 0.45; // Just below Nyquist

        let log_min = min_freq.ln();
        let log_max = max_freq.ln();

        let mut band_centers = Vec::with_capacity(num_bands);
        let mut band_widths = Vec::with_capacity(num_bands);

        for i in 0..num_bands {
            let t = (i as f32 + 0.5) / num_bands as f32;
            let freq = (log_min + t * (log_max - log_min)).exp();
            band_centers.push(freq);
            // Q increases with frequency
            band_widths.push(0.5 / (2.0 + i as f32 * 0.5));
        }

        Self {
            num_bands,
            envelopes: vec![0.0; num_bands],
            attack,
            release,
            band_centers,
            band_widths,
            carrier_states: vec![[0.0; 4]; num_bands],
            modulator_states: vec![[0.0; 4]; num_bands],
        }
    }

    /// Processes a single sample.
    pub fn process_sample(&mut self, carrier: f32, modulator: f32) -> f32 {
        let mut output = 0.0;

        for i in 0..self.num_bands {
            let center = self.band_centers[i];
            let width = self.band_widths[i];

            // Filter modulator through bandpass
            let mod_filtered =
                bandpass_filter(modulator, center, width, &mut self.modulator_states[i]);

            // Envelope follower
            let envelope_input = mod_filtered.abs();
            if envelope_input > self.envelopes[i] {
                self.envelopes[i] =
                    self.attack * self.envelopes[i] + (1.0 - self.attack) * envelope_input;
            } else {
                self.envelopes[i] =
                    self.release * self.envelopes[i] + (1.0 - self.release) * envelope_input;
            }

            // Filter carrier through same bandpass
            let car_filtered = bandpass_filter(carrier, center, width, &mut self.carrier_states[i]);

            // Apply envelope
            output += car_filtered * self.envelopes[i];
        }

        output
    }

    /// Processes a block of samples.
    pub fn process(&mut self, carrier: &[f32], modulator: &[f32]) -> Vec<f32> {
        assert_eq!(carrier.len(), modulator.len());
        carrier
            .iter()
            .zip(modulator.iter())
            .map(|(&c, &m)| self.process_sample(c, m))
            .collect()
    }

    /// Resets the vocoder state.
    pub fn reset(&mut self) {
        self.envelopes.fill(0.0);
        for state in &mut self.carrier_states {
            *state = [0.0; 4];
        }
        for state in &mut self.modulator_states {
            *state = [0.0; 4];
        }
    }
}

/// Simple 2-pole bandpass filter (biquad).
fn bandpass_filter(input: f32, center: f32, width: f32, state: &mut [f32; 4]) -> f32 {
    use std::f32::consts::PI;

    // Compute coefficients (simplified bandpass)
    let omega = 2.0 * PI * center;
    let alpha = omega.sin() * width;
    let cos_omega = omega.cos();

    let b0 = alpha;
    let b1 = 0.0;
    let b2 = -alpha;
    let a0 = 1.0 + alpha;
    let a1 = -2.0 * cos_omega;
    let a2 = 1.0 - alpha;

    // Normalize
    let b0 = b0 / a0;
    let b1 = b1 / a0;
    let b2 = b2 / a0;
    let a1 = a1 / a0;
    let a2 = a2 / a0;

    // Direct Form II Transposed
    let output = b0 * input + state[0];
    state[0] = b1 * input - a1 * output + state[1];
    state[1] = b2 * input - a2 * output;

    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    #[test]
    fn test_vocoder_config() {
        let config = VocodeSynth::new(2048).with_bands(32).with_smoothing(0.95);

        assert_eq!(config.window_size, 2048);
        assert_eq!(config.num_bands, 32);
        assert!((config.envelope_smoothing - 0.95).abs() < 0.001);
    }

    #[test]
    fn test_vocoder_process() {
        let config = VocodeSynth::default();
        let mut vocoder = Vocoder::new(config);

        let n = 2048;
        let carrier: Vec<f32> = (0..n)
            .map(|i| (2.0 * PI * 440.0 * i as f32 / 44100.0).sin())
            .collect();
        let modulator: Vec<f32> = (0..n)
            .map(|i| (2.0 * PI * 100.0 * i as f32 / 44100.0).sin())
            .collect();

        let output = vocoder.process(&carrier, &modulator);
        assert_eq!(output.len(), n);

        // Output should not be silent
        let energy: f32 = output.iter().map(|x| x * x).sum();
        assert!(energy > 0.001);
    }

    #[test]
    fn test_filterbank_vocoder() {
        let mut vocoder = FilterbankVocoder::new(16, 44100.0);

        let n = 1024;
        let carrier: Vec<f32> = (0..n)
            .map(|i| (2.0 * PI * 440.0 * i as f32 / 44100.0).sin())
            .collect();
        let modulator: Vec<f32> = (0..n)
            .map(|i| (2.0 * PI * 100.0 * i as f32 / 44100.0).sin())
            .collect();

        let output = vocoder.process(&carrier, &modulator);
        assert_eq!(output.len(), n);
    }

    #[test]
    fn test_vocoder_reset() {
        let config = VocodeSynth::default();
        let mut vocoder = Vocoder::new(config);

        let signal = vec![0.5; 512];
        vocoder.process(&signal, &signal);

        vocoder.reset();

        // After reset, internal buffers should be zero
        assert!(vocoder.carrier_buffer.iter().all(|&x| x == 0.0));
        assert!(vocoder.modulator_buffer.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_create_bands() {
        let bands = create_bands(513, 16);

        assert_eq!(bands.len(), 16);
        // Bands should cover the full range
        assert_eq!(bands[0].start_bin, 1);
        // Last band should reach near the end
        assert!(bands.last().unwrap().end_bin <= 513);
    }
}
