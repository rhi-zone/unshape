//! Spectral processing: STFT and frequency-domain analysis.
//!
//! This module re-exports core FFT primitives from `resin-spectral` and adds
//! audio-specific functionality:
//! - STFT for time-frequency analysis
//! - Time stretching and pitch shifting
//! - Pitch detection
//!
//! # Example
//!
//! ```
//! use rhizome_resin_audio::spectral::{fft, ifft, hann_window};
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

// Re-export core spectral primitives from resin-spectral
pub use rhizome_resin_spectral::{
    // Complex number type
    Complex,
    // Workspace
    SpectralWorkspace,
    // Window functions
    apply_window,
    blackman_window,
    // 1D FFT
    fft,
    fft_complex,
    fft_into,
    hamming_window,
    hann_window,
    ifft,
    ifft_complex,
    ifft_into,
    rect_window,
};

// ============================================================================
// STFT (Short-Time Fourier Transform)
// ============================================================================

/// Configuration for STFT analysis.
///
/// Operations on audio buffers use the ops-as-values pattern.
/// See `docs/design/ops-as-values.md`.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dynop", derive(rhizome_resin_op::Op))]
#[cfg_attr(feature = "dynop", op(input = Vec<f32>, output = StftResult))]
pub struct Stft {
    /// FFT window size (must be power of 2).
    pub window_size: usize,
    /// Hop size between consecutive frames.
    pub hop_size: usize,
    /// Window function.
    pub window: Vec<f32>,
}

impl Default for Stft {
    fn default() -> Self {
        let window_size = 1024;
        Self {
            window_size,
            hop_size: 256,
            window: hann_window(window_size),
        }
    }
}

impl Stft {
    /// Creates a new STFT config with the given window size.
    ///
    /// Hop size defaults to window_size / 4.
    pub fn new(window_size: usize) -> Self {
        Self {
            window_size,
            hop_size: window_size / 4,
            window: hann_window(window_size),
        }
    }

    /// Creates a new STFT config with explicit window and hop sizes.
    ///
    /// Uses a Hann window function.
    pub fn with_hop(window_size: usize, hop_size: usize) -> Self {
        Self {
            window_size,
            hop_size,
            window: hann_window(window_size),
        }
    }

    /// Applies this STFT configuration to a signal.
    pub fn apply(&self, signal: &[f32]) -> StftResult {
        stft(signal, self)
    }
}

/// Backwards-compatible type alias.
pub type StftConfig = Stft;

/// Result of STFT analysis.
#[derive(Debug, Clone)]
pub struct StftResult {
    /// Complex spectrogram: `frames[frame_index][bin_index]`
    pub frames: Vec<Vec<Complex>>,
    /// Sample rate (if known).
    pub sample_rate: Option<u32>,
    /// Configuration used.
    pub config: Stft,
}

impl StftResult {
    /// Returns the number of frames.
    pub fn num_frames(&self) -> usize {
        self.frames.len()
    }

    /// Returns the number of frequency bins per frame.
    pub fn num_bins(&self) -> usize {
        if self.frames.is_empty() {
            0
        } else {
            self.frames[0].len()
        }
    }

    /// Computes the magnitude spectrogram.
    pub fn magnitude(&self) -> Vec<Vec<f32>> {
        self.frames
            .iter()
            .map(|frame| frame.iter().map(|c| c.mag()).collect())
            .collect()
    }

    /// Computes the power spectrogram (magnitude squared).
    pub fn power(&self) -> Vec<Vec<f32>> {
        self.frames
            .iter()
            .map(|frame| frame.iter().map(|c| c.re * c.re + c.im * c.im).collect())
            .collect()
    }

    /// Computes the phase spectrogram.
    pub fn phase(&self) -> Vec<Vec<f32>> {
        self.frames
            .iter()
            .map(|frame| frame.iter().map(|c| c.phase()).collect())
            .collect()
    }

    /// Returns the frequency for a given bin index.
    pub fn bin_to_freq(&self, bin: usize) -> Option<f32> {
        self.sample_rate
            .map(|sr| bin as f32 * sr as f32 / self.config.window_size as f32)
    }

    /// Returns the approximate bin index for a given frequency.
    pub fn freq_to_bin(&self, freq: f32) -> Option<usize> {
        self.sample_rate
            .map(|sr| (freq * self.config.window_size as f32 / sr as f32).round() as usize)
    }
}

/// Computes the Short-Time Fourier Transform of a signal.
pub fn stft(signal: &[f32], config: &Stft) -> StftResult {
    stft_with_sample_rate(signal, config, None)
}

/// Computes the STFT with a known sample rate.
pub fn stft_with_sample_rate(
    signal: &[f32],
    config: &Stft,
    sample_rate: Option<u32>,
) -> StftResult {
    let mut frames = Vec::new();
    let mut pos = 0;

    let mut frame_buffer = vec![0.0f32; config.window_size];

    while pos + config.window_size <= signal.len() {
        // Copy frame and apply window
        frame_buffer.copy_from_slice(&signal[pos..pos + config.window_size]);
        apply_window(&mut frame_buffer, &config.window);

        // Compute FFT
        let spectrum = fft(&frame_buffer);
        frames.push(spectrum);

        pos += config.hop_size;
    }

    StftResult {
        frames,
        sample_rate,
        config: config.clone(),
    }
}

/// Reconstructs a signal from STFT using overlap-add synthesis.
pub fn istft(stft: &StftResult) -> Vec<f32> {
    if stft.frames.is_empty() {
        return Vec::new();
    }

    let config = &stft.config;
    let n_frames = stft.frames.len();
    let output_len = (n_frames - 1) * config.hop_size + config.window_size;

    let mut output = vec![0.0f32; output_len];
    let mut window_sum = vec![0.0f32; output_len];

    // Synthesis window (for perfect reconstruction with Hann window and 75% overlap)
    let synthesis_window = &config.window;

    for (i, frame) in stft.frames.iter().enumerate() {
        let pos = i * config.hop_size;

        // Inverse FFT
        let time_domain = ifft(frame);

        // Overlap-add with synthesis window
        for (j, &sample) in time_domain.iter().enumerate() {
            output[pos + j] += sample * synthesis_window[j];
            window_sum[pos + j] += synthesis_window[j] * synthesis_window[j];
        }
    }

    // Normalize by window sum
    for (o, &w) in output.iter_mut().zip(window_sum.iter()) {
        if w > 1e-8 {
            *o /= w;
        }
    }

    output
}

// ============================================================================
// Spectral analysis utilities
// ============================================================================

/// Finds the peak frequency in a magnitude spectrum.
pub fn find_peak_frequency(magnitudes: &[f32], sample_rate: u32, fft_size: usize) -> Option<f32> {
    if magnitudes.is_empty() {
        return None;
    }

    let (peak_bin, _) = magnitudes
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())?;

    Some(peak_bin as f32 * sample_rate as f32 / fft_size as f32)
}

/// Estimates the fundamental frequency using normalized autocorrelation.
pub fn estimate_pitch(
    signal: &[f32],
    sample_rate: u32,
    min_freq: f32,
    max_freq: f32,
) -> Option<f32> {
    let n = signal.len();
    if n == 0 {
        return None;
    }

    let min_lag = (sample_rate as f32 / max_freq).ceil() as usize;
    let max_lag = (sample_rate as f32 / min_freq).floor() as usize;

    if max_lag >= n || min_lag >= max_lag {
        return None;
    }

    // Compute energy at lag 0 for normalization
    let energy: f32 = signal.iter().map(|&x| x * x).sum();
    if energy < 1e-10 {
        return None;
    }

    // Compute normalized autocorrelation and find first peak after first zero crossing
    let mut correlations = Vec::with_capacity(max_lag - min_lag + 1);

    for lag in min_lag..=max_lag.min(n - 1) {
        let mut sum = 0.0;
        let mut energy1 = 0.0;
        let mut energy2 = 0.0;

        for i in 0..(n - lag) {
            sum += signal[i] * signal[i + lag];
            energy1 += signal[i] * signal[i];
            energy2 += signal[i + lag] * signal[i + lag];
        }

        let denom = (energy1 * energy2).sqrt();
        let corr = if denom > 1e-10 { sum / denom } else { 0.0 };
        correlations.push((lag, corr));
    }

    // Find the first significant peak (not at lag 0)
    // Use a simple peak detection: correlation > threshold and higher than neighbors
    let threshold = 0.5;
    let mut best_lag = min_lag;
    let mut best_corr = 0.0f32;

    for i in 1..correlations.len().saturating_sub(1) {
        let (lag, corr) = correlations[i];
        let prev_corr = correlations[i - 1].1;
        let next_corr = correlations[i + 1].1;

        // Is this a local maximum above threshold?
        if corr > threshold && corr > prev_corr && corr >= next_corr {
            // Take the first peak found (fundamental, not harmonic)
            if best_corr < threshold || corr > best_corr * 0.9 {
                best_lag = lag;
                best_corr = corr;
                break; // First strong peak is likely the fundamental
            }
        }
    }

    if best_corr < threshold {
        // Fallback: find global maximum
        for &(lag, corr) in &correlations {
            if corr > best_corr {
                best_corr = corr;
                best_lag = lag;
            }
        }
    }

    Some(sample_rate as f32 / best_lag as f32)
}

/// Computes the spectral centroid (center of mass of the spectrum).
pub fn spectral_centroid(magnitudes: &[f32], sample_rate: u32, fft_size: usize) -> f32 {
    let mut weighted_sum = 0.0;
    let mut total_magnitude = 0.0;

    for (bin, &mag) in magnitudes.iter().enumerate() {
        let freq = bin as f32 * sample_rate as f32 / fft_size as f32;
        weighted_sum += freq * mag;
        total_magnitude += mag;
    }

    if total_magnitude > 1e-8 {
        weighted_sum / total_magnitude
    } else {
        0.0
    }
}

/// Computes the spectral flatness (tonality coefficient).
///
/// Returns 0 for pure tones, 1 for white noise.
pub fn spectral_flatness(magnitudes: &[f32]) -> f32 {
    if magnitudes.is_empty() {
        return 0.0;
    }

    let n = magnitudes.len() as f32;

    // Geometric mean
    let log_sum: f32 = magnitudes.iter().map(|&m| (m + 1e-10).ln()).sum();
    let geometric_mean = (log_sum / n).exp();

    // Arithmetic mean
    let arithmetic_mean: f32 = magnitudes.iter().sum::<f32>() / n;

    if arithmetic_mean > 1e-10 {
        geometric_mean / arithmetic_mean
    } else {
        0.0
    }
}

// ============================================================================
// Time-stretching via Phase Vocoder
// ============================================================================

/// Configuration for time-stretching.
///
/// Operations on audio buffers use the ops-as-values pattern.
/// See `docs/design/ops-as-values.md`.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dynop", derive(rhizome_resin_op::Op))]
#[cfg_attr(feature = "dynop", op(input = Vec<f32>, output = Vec<f32>))]
pub struct TimeStretch {
    /// FFT window size (must be power of 2).
    pub window_size: usize,
    /// Analysis hop size.
    pub analysis_hop: usize,
    /// Time stretch factor (< 1.0 = faster, > 1.0 = slower).
    pub stretch_factor: f32,
    /// Whether to preserve transients.
    pub preserve_transients: bool,
    /// Transient detection threshold.
    pub transient_threshold: f32,
}

impl Default for TimeStretch {
    fn default() -> Self {
        Self {
            window_size: 2048,
            analysis_hop: 512,
            stretch_factor: 1.0,
            preserve_transients: true,
            transient_threshold: 1.5,
        }
    }
}

impl TimeStretch {
    /// Creates a config with the given stretch factor.
    pub fn with_factor(factor: f32) -> Self {
        Self {
            stretch_factor: factor.max(0.1).min(10.0),
            ..Default::default()
        }
    }

    /// Sets the window size.
    pub fn with_window_size(mut self, size: usize) -> Self {
        assert!(size.is_power_of_two(), "Window size must be power of 2");
        self.window_size = size;
        self.analysis_hop = size / 4;
        self
    }

    /// Applies this time-stretch configuration to a signal.
    pub fn apply(&self, signal: &[f32]) -> Vec<f32> {
        time_stretch(signal, self)
    }
}

/// Backwards-compatible type alias.
pub type TimeStretchConfig = TimeStretch;

/// Pre-allocated workspace for time-stretch operations.
///
/// Use this to avoid allocations when time-stretching in real-time or in loops.
/// Create once with your window size, reuse for multiple time-stretch calls.
///
/// # Example
///
/// ```
/// use rhizome_resin_audio::spectral::{TimeStretch, TimeStretchWorkspace, time_stretch_with_workspace};
///
/// let config = TimeStretch::with_factor(1.5);
/// let mut workspace = TimeStretchWorkspace::new(config.window_size);
///
/// let audio: Vec<f32> = (0..8192).map(|i| (i as f32 * 0.1).sin()).collect();
/// let stretched = time_stretch_with_workspace(&audio, &config, &mut workspace);
/// ```
#[derive(Debug, Clone)]
pub struct TimeStretchWorkspace {
    /// Spectral workspace for FFT/IFFT.
    pub spectral: SpectralWorkspace,
    /// Window function (cached).
    window: Vec<f32>,
    /// Frame buffer for windowed input.
    frame_buffer: Vec<f32>,
    /// Previous phase for each bin.
    prev_phase: Vec<f32>,
    /// Synthesis phase accumulator for each bin.
    synth_phase: Vec<f32>,
    /// Adjusted spectrum buffer.
    adjusted_spectrum: Vec<Complex>,
}

impl TimeStretchWorkspace {
    /// Creates a new workspace for the given window size.
    ///
    /// # Panics
    /// Panics if `window_size` is not a power of 2.
    pub fn new(window_size: usize) -> Self {
        assert!(
            window_size.is_power_of_two(),
            "Window size must be power of 2"
        );
        let num_bins = window_size / 2 + 1;
        Self {
            spectral: SpectralWorkspace::new(window_size),
            window: hann_window(window_size),
            frame_buffer: vec![0.0; window_size],
            prev_phase: vec![0.0; num_bins],
            synth_phase: vec![0.0; num_bins],
            adjusted_spectrum: vec![Complex::new(0.0, 0.0); num_bins],
        }
    }

    /// Resets phase accumulators for a new stretch operation.
    pub fn reset(&mut self) {
        self.prev_phase.fill(0.0);
        self.synth_phase.fill(0.0);
    }

    /// Returns the window size this workspace was created for.
    pub fn window_size(&self) -> usize {
        self.spectral.fft_size()
    }
}

/// Time-stretches audio using a phase vocoder with pre-allocated workspace.
///
/// This is the allocation-free variant for real-time use.
/// The workspace must have been created with the same window_size as the config.
pub fn time_stretch_with_workspace(
    signal: &[f32],
    config: &TimeStretch,
    workspace: &mut TimeStretchWorkspace,
) -> Vec<f32> {
    assert_eq!(
        config.window_size,
        workspace.window_size(),
        "Config window_size must match workspace"
    );

    if signal.is_empty() || (config.stretch_factor - 1.0).abs() < 0.001 {
        return signal.to_vec();
    }

    workspace.reset();

    let window_size = config.window_size;
    let analysis_hop = config.analysis_hop;
    let synthesis_hop = (analysis_hop as f32 * config.stretch_factor) as usize;

    // Estimate output length
    let num_frames = (signal.len().saturating_sub(window_size)) / analysis_hop + 1;
    let output_len = (num_frames.saturating_sub(1)) * synthesis_hop + window_size;

    // Output buffers (these are the actual output, can't avoid allocating)
    let mut output = vec![0.0f32; output_len];
    let mut window_sum = vec![0.0f32; output_len];

    let freq_per_bin = 2.0 * PI / window_size as f32;

    let mut analysis_pos = 0;
    let mut synthesis_pos = 0;

    while analysis_pos + window_size <= signal.len() {
        // Copy and window the analysis frame
        workspace
            .frame_buffer
            .copy_from_slice(&signal[analysis_pos..analysis_pos + window_size]);
        apply_window(&mut workspace.frame_buffer, &workspace.window);

        // FFT (no allocation)
        fft_into(&workspace.frame_buffer, &mut workspace.spectral);

        // Phase vocoder: adjust phases
        for (bin, c) in workspace.spectral.spectrum().iter().enumerate() {
            let mag = c.mag();
            let phase = c.phase();

            let expected_phase_advance = bin as f32 * analysis_hop as f32 * freq_per_bin;
            let phase_diff = phase - workspace.prev_phase[bin] - expected_phase_advance;
            let wrapped_diff = wrap_phase(phase_diff);
            let true_freq = bin as f32 * freq_per_bin + wrapped_diff / analysis_hop as f32;

            workspace.synth_phase[bin] += true_freq * synthesis_hop as f32;
            workspace.synth_phase[bin] = wrap_phase(workspace.synth_phase[bin]);
            workspace.prev_phase[bin] = phase;

            workspace.adjusted_spectrum[bin] = Complex::from_polar(mag, workspace.synth_phase[bin]);
        }

        // Transient detection
        let is_transient = if config.preserve_transients {
            detect_transient(
                workspace.spectral.spectrum(),
                &workspace.prev_phase,
                config.transient_threshold,
            )
        } else {
            false
        };

        // For transients, copy original spectrum to adjusted_spectrum to preserve attack
        if is_transient {
            workspace
                .adjusted_spectrum
                .copy_from_slice(workspace.spectral.spectrum());
        }

        // IFFT (no allocation) - always use adjusted_spectrum
        ifft_into(&workspace.adjusted_spectrum, &mut workspace.spectral);

        // Overlap-add at synthesis position
        if synthesis_pos + window_size <= output_len {
            for (j, &sample) in workspace.spectral.real_buffer().iter().enumerate() {
                output[synthesis_pos + j] += sample * workspace.window[j];
                window_sum[synthesis_pos + j] += workspace.window[j] * workspace.window[j];
            }
        }

        analysis_pos += analysis_hop;
        synthesis_pos += synthesis_hop;
    }

    // Normalize by window sum
    for (o, &w) in output.iter_mut().zip(window_sum.iter()) {
        if w > 1e-8 {
            *o /= w;
        }
    }

    output
}

/// Time-stretches audio using a phase vocoder.
///
/// Changes the duration without changing the pitch.
///
/// # Example
///
/// ```
/// use rhizome_resin_audio::spectral::{time_stretch, TimeStretch};
///
/// let audio: Vec<f32> = (0..8192).map(|i| (i as f32 * 0.1).sin()).collect();
///
/// // Slow down to 1.5x duration
/// let slower = time_stretch(&audio, &TimeStretch::with_factor(1.5));
///
/// // Speed up to 0.5x duration
/// let faster = time_stretch(&audio, &TimeStretch::with_factor(0.5));
/// ```
pub fn time_stretch(signal: &[f32], config: &TimeStretch) -> Vec<f32> {
    if signal.is_empty() || (config.stretch_factor - 1.0).abs() < 0.001 {
        return signal.to_vec();
    }

    let window_size = config.window_size;
    let analysis_hop = config.analysis_hop;
    let synthesis_hop = (analysis_hop as f32 * config.stretch_factor) as usize;

    // Window functions
    let window = hann_window(window_size);
    let num_bins = window_size / 2 + 1;

    // Estimate output length
    let num_frames = (signal.len().saturating_sub(window_size)) / analysis_hop + 1;
    let output_len = (num_frames.saturating_sub(1)) * synthesis_hop + window_size;

    let mut output = vec![0.0f32; output_len];
    let mut window_sum = vec![0.0f32; output_len];

    // Phase accumulators for phase vocoder
    let mut prev_phase = vec![0.0f32; num_bins];
    let mut synth_phase = vec![0.0f32; num_bins];

    // Frequency per bin (for phase unwrapping)
    let freq_per_bin = 2.0 * PI / window_size as f32;

    // Process frames
    let mut frame_buffer = vec![0.0f32; window_size];
    let mut analysis_pos = 0;
    let mut synthesis_pos = 0;

    while analysis_pos + window_size <= signal.len() {
        // Copy and window the analysis frame
        frame_buffer.copy_from_slice(&signal[analysis_pos..analysis_pos + window_size]);
        apply_window(&mut frame_buffer, &window);

        // FFT
        let spectrum = fft(&frame_buffer);

        // Phase vocoder: adjust phases
        let mut adjusted_spectrum = Vec::with_capacity(num_bins);

        for (bin, c) in spectrum.iter().enumerate() {
            let mag = c.mag();
            let phase = c.phase();

            // Calculate phase difference
            let expected_phase_advance = bin as f32 * analysis_hop as f32 * freq_per_bin;
            let phase_diff = phase - prev_phase[bin] - expected_phase_advance;

            // Wrap to [-PI, PI]
            let wrapped_diff = wrap_phase(phase_diff);

            // True frequency deviation
            let true_freq = bin as f32 * freq_per_bin + wrapped_diff / analysis_hop as f32;

            // Accumulate synthesis phase
            synth_phase[bin] += true_freq * synthesis_hop as f32;
            synth_phase[bin] = wrap_phase(synth_phase[bin]);

            // Store for next frame
            prev_phase[bin] = phase;

            // Reconstruct complex number with new phase
            adjusted_spectrum.push(Complex::from_polar(mag, synth_phase[bin]));
        }

        // Transient detection (optional)
        let is_transient = if config.preserve_transients {
            detect_transient(&spectrum, &prev_phase, config.transient_threshold)
        } else {
            false
        };

        // For transients, use original phase to preserve attack
        let final_spectrum = if is_transient {
            spectrum
        } else {
            adjusted_spectrum
        };

        // IFFT
        let time_frame = ifft(&final_spectrum);

        // Overlap-add at synthesis position
        if synthesis_pos + window_size <= output_len {
            for (j, &sample) in time_frame.iter().enumerate() {
                output[synthesis_pos + j] += sample * window[j];
                window_sum[synthesis_pos + j] += window[j] * window[j];
            }
        }

        analysis_pos += analysis_hop;
        synthesis_pos += synthesis_hop;
    }

    // Normalize by window sum
    for (o, &w) in output.iter_mut().zip(window_sum.iter()) {
        if w > 1e-8 {
            *o /= w;
        }
    }

    output
}

/// Pitch-shifts audio by time-stretching and resampling.
///
/// Changes the pitch without changing the duration.
///
/// # Arguments
/// * `signal` - Input audio samples
/// * `semitones` - Pitch shift in semitones (positive = higher, negative = lower)
pub fn pitch_shift(signal: &[f32], semitones: f32) -> Vec<f32> {
    if signal.is_empty() || semitones.abs() < 0.001 {
        return signal.to_vec();
    }

    // Pitch ratio (2^(semitones/12))
    let pitch_ratio = (2.0f32).powf(semitones / 12.0);

    // Time-stretch to compensate for resampling
    let stretch_config = TimeStretch::with_factor(pitch_ratio);
    let stretched = time_stretch(signal, &stretch_config);

    // Resample to original length
    resample_linear(&stretched, signal.len())
}

/// Granular time-stretch for extreme ratios.
///
/// Better quality than phase vocoder for very large stretch factors.
pub fn time_stretch_granular(
    signal: &[f32],
    stretch_factor: f32,
    grain_size: usize,
    overlap: f32,
) -> Vec<f32> {
    if signal.is_empty() || stretch_factor <= 0.0 {
        return signal.to_vec();
    }

    let hop = (grain_size as f32 * (1.0 - overlap.clamp(0.0, 0.95))) as usize;
    let output_hop = (hop as f32 * stretch_factor) as usize;

    // Estimate output length
    let num_grains = signal.len().saturating_sub(grain_size) / hop + 1;
    let output_len = (num_grains.saturating_sub(1)) * output_hop + grain_size;

    let mut output = vec![0.0f32; output_len];
    let mut window_sum = vec![0.0f32; output_len];

    let window = hann_window(grain_size);

    let mut read_pos = 0;
    let mut write_pos = 0;

    while read_pos + grain_size <= signal.len() && write_pos + grain_size <= output_len {
        // Copy grain with window
        for i in 0..grain_size {
            output[write_pos + i] += signal[read_pos + i] * window[i];
            window_sum[write_pos + i] += window[i] * window[i];
        }

        read_pos += hop;
        write_pos += output_hop;
    }

    // Normalize
    for (o, &w) in output.iter_mut().zip(window_sum.iter()) {
        if w > 1e-8 {
            *o /= w;
        }
    }

    output
}

/// Wraps a phase value to [-PI, PI].
fn wrap_phase(phase: f32) -> f32 {
    let mut p = phase;
    while p > PI {
        p -= 2.0 * PI;
    }
    while p < -PI {
        p += 2.0 * PI;
    }
    p
}

/// Detects if a frame is a transient.
fn detect_transient(spectrum: &[Complex], prev_phase: &[f32], threshold: f32) -> bool {
    let mut phase_deviation_sum = 0.0f32;
    let mut count = 0;

    for (bin, c) in spectrum.iter().enumerate().skip(1) {
        if c.mag() > 0.001 {
            let phase = c.phase();
            let diff = (phase - prev_phase[bin]).abs();
            phase_deviation_sum += diff;
            count += 1;
        }
    }

    if count > 0 {
        let avg_deviation = phase_deviation_sum / count as f32;
        avg_deviation > threshold
    } else {
        false
    }
}

/// Linear resampling to a target length.
fn resample_linear(signal: &[f32], target_len: usize) -> Vec<f32> {
    if signal.is_empty() || target_len == 0 {
        return vec![];
    }

    if target_len == signal.len() {
        return signal.to_vec();
    }

    let ratio = (signal.len() - 1) as f32 / (target_len - 1).max(1) as f32;

    (0..target_len)
        .map(|i| {
            let pos = i as f32 * ratio;
            let idx = pos.floor() as usize;
            let frac = pos.fract();

            if idx + 1 < signal.len() {
                signal[idx] * (1.0 - frac) + signal[idx + 1] * frac
            } else {
                signal[signal.len() - 1]
            }
        })
        .collect()
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
    fn test_fft_sine_peak() {
        let n = 1024;
        let freq = 100.0;
        let sample_rate = 1024.0;

        let signal: Vec<f32> = (0..n)
            .map(|i| (2.0 * PI * freq * i as f32 / sample_rate).sin())
            .collect();

        let spectrum = fft(&signal);
        let magnitudes: Vec<f32> = spectrum.iter().map(|c| c.mag()).collect();

        // Peak should be at bin 100 (freq * n / sample_rate)
        let peak_bin = magnitudes
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap();

        assert_eq!(peak_bin, 100);
    }

    #[test]
    fn test_stft_istft_roundtrip() {
        let n = 4096;
        let signal: Vec<f32> = (0..n)
            .map(|i| (2.0 * PI * 440.0 * i as f32 / 44100.0).sin())
            .collect();

        let config = Stft::new(1024); // hop_size defaults to window_size/4 = 256
        let stft_result = stft(&signal, &config);

        assert!(stft_result.num_frames() > 0);
        assert_eq!(stft_result.num_bins(), 513); // 1024/2 + 1

        let recovered = istft(&stft_result);

        // Check reconstruction error (allowing for edge effects)
        let start = config.window_size;
        let end = signal.len() - config.window_size;
        for i in start..end.min(recovered.len()) {
            assert!(
                (signal[i] - recovered[i]).abs() < 0.1,
                "Mismatch at {}: {} vs {}",
                i,
                signal[i],
                recovered[i]
            );
        }
    }

    #[test]
    fn test_spectral_centroid() {
        // Low frequency dominated
        let low_mags = vec![10.0, 5.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let centroid_low = spectral_centroid(&low_mags, 8000, 16);

        // High frequency dominated
        let high_mags = vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 5.0, 10.0];
        let centroid_high = spectral_centroid(&high_mags, 8000, 16);

        assert!(centroid_low < centroid_high);
    }

    #[test]
    fn test_spectral_flatness() {
        // Pure tone (very unflat)
        let tone = vec![0.0, 0.0, 100.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let flatness_tone = spectral_flatness(&tone);

        // Flat spectrum (very flat)
        let flat = vec![10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0];
        let flatness_flat = spectral_flatness(&flat);

        assert!(flatness_tone < flatness_flat);
        assert!(flatness_flat > 0.9); // Should be close to 1
    }

    #[test]
    fn test_stft_magnitude() {
        let signal: Vec<f32> = (0..2048)
            .map(|i| (2.0 * PI * 100.0 * i as f32 / 1024.0).sin())
            .collect();

        let config = Stft::new(1024);
        let result = stft(&signal, &config);
        let mags = result.magnitude();

        assert!(!mags.is_empty());
        assert!(!mags[0].is_empty());
    }

    #[test]
    fn test_estimate_pitch() {
        let sample_rate = 44100;
        let freq = 440.0;

        let signal: Vec<f32> = (0..4096)
            .map(|i| (2.0 * PI * freq * i as f32 / sample_rate as f32).sin())
            .collect();

        let estimated = estimate_pitch(&signal, sample_rate, 100.0, 1000.0);
        assert!(estimated.is_some());

        let est_freq = estimated.unwrap();
        assert!(
            (est_freq - freq).abs() < 10.0,
            "Estimated {} vs actual {}",
            est_freq,
            freq
        );
    }

    #[test]
    fn test_time_stretch_slower() {
        let signal: Vec<f32> = (0..8192)
            .map(|i| (2.0 * PI * 440.0 * i as f32 / 44100.0).sin())
            .collect();

        let config = TimeStretch::with_factor(2.0);
        let stretched = time_stretch(&signal, &config);

        // Should be approximately twice as long
        let ratio = stretched.len() as f32 / signal.len() as f32;
        assert!(
            (ratio - 2.0).abs() < 0.3,
            "Expected ratio ~2.0, got {}",
            ratio
        );
    }

    #[test]
    fn test_time_stretch_faster() {
        let signal: Vec<f32> = (0..8192)
            .map(|i| (2.0 * PI * 440.0 * i as f32 / 44100.0).sin())
            .collect();

        let config = TimeStretch::with_factor(0.5);
        let stretched = time_stretch(&signal, &config);

        // Should be approximately half as long
        let ratio = stretched.len() as f32 / signal.len() as f32;
        assert!(
            (ratio - 0.5).abs() < 0.2,
            "Expected ratio ~0.5, got {}",
            ratio
        );
    }

    #[test]
    fn test_time_stretch_unity() {
        let signal: Vec<f32> = (0..4096)
            .map(|i| (2.0 * PI * 440.0 * i as f32 / 44100.0).sin())
            .collect();

        let config = TimeStretch::with_factor(1.0);
        let stretched = time_stretch(&signal, &config);

        // Should be same length for unity stretch
        assert_eq!(stretched.len(), signal.len());
    }

    #[test]
    fn test_pitch_shift() {
        let signal: Vec<f32> = (0..4096)
            .map(|i| (2.0 * PI * 440.0 * i as f32 / 44100.0).sin())
            .collect();

        // Shift up an octave
        let shifted = pitch_shift(&signal, 12.0);

        // Should be same length
        assert_eq!(shifted.len(), signal.len());
    }

    #[test]
    fn test_time_stretch_granular() {
        let signal: Vec<f32> = (0..8192)
            .map(|i| (2.0 * PI * 440.0 * i as f32 / 44100.0).sin())
            .collect();

        let stretched = time_stretch_granular(&signal, 2.0, 2048, 0.75);

        // Should be approximately twice as long
        let ratio = stretched.len() as f32 / signal.len() as f32;
        assert!(
            (ratio - 2.0).abs() < 0.5,
            "Expected ratio ~2.0, got {}",
            ratio
        );
    }

    #[test]
    fn test_time_stretch_empty() {
        let signal: Vec<f32> = vec![];
        let stretched = time_stretch(&signal, &TimeStretch::with_factor(2.0));
        assert!(stretched.is_empty());
    }

    #[test]
    fn test_wrap_phase() {
        assert!((wrap_phase(0.0) - 0.0).abs() < 0.001);
        assert!((wrap_phase(PI) - PI).abs() < 0.001);
        assert!((wrap_phase(-PI) - (-PI)).abs() < 0.001);
        assert!((wrap_phase(3.0 * PI) - PI).abs() < 0.001);
        assert!((wrap_phase(-3.0 * PI) - (-PI)).abs() < 0.001);
    }

    #[test]
    fn test_resample_linear() {
        let signal = vec![0.0, 1.0, 2.0, 3.0, 4.0];

        // Upsample
        let upsampled = resample_linear(&signal, 9);
        assert_eq!(upsampled.len(), 9);
        assert!((upsampled[0] - 0.0).abs() < 0.001);
        assert!((upsampled[8] - 4.0).abs() < 0.001);

        // Downsample
        let downsampled = resample_linear(&signal, 3);
        assert_eq!(downsampled.len(), 3);
    }

    #[test]
    fn test_spectral_workspace_roundtrip() {
        let signal: Vec<f32> = (0..64)
            .map(|i| (2.0 * PI * 5.0 * i as f32 / 64.0).sin())
            .collect();

        let mut workspace = SpectralWorkspace::new(64);

        // FFT
        fft_into(&signal, &mut workspace);

        // IFFT
        workspace.ifft_from_spectrum();
        let recovered = workspace.real_buffer();

        // Verify roundtrip
        for (a, b) in signal.iter().zip(recovered.iter()) {
            assert!((a - b).abs() < 0.001, "Mismatch: {} vs {}", a, b);
        }
    }

    #[test]
    fn test_time_stretch_with_workspace() {
        let signal: Vec<f32> = (0..4096)
            .map(|i| (2.0 * PI * 440.0 * i as f32 / 44100.0).sin())
            .collect();

        let config = TimeStretch::with_factor(2.0);
        let mut workspace = TimeStretchWorkspace::new(config.window_size);

        // Stretch with workspace
        let stretched_ws = time_stretch_with_workspace(&signal, &config, &mut workspace);

        // Stretch without workspace (for comparison)
        let stretched = time_stretch(&signal, &config);

        // Results should be identical
        assert_eq!(stretched_ws.len(), stretched.len());
        for (a, b) in stretched_ws.iter().zip(stretched.iter()) {
            assert!((a - b).abs() < 0.001, "Mismatch: {} vs {}", a, b);
        }
    }
}
