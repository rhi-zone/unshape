//! Audio filters.
//!
//! Provides common audio filters for sound design and synthesis.
//!
//! Filters are implemented in two forms:
//! - Pure functions for one-sample-at-a-time processing (you manage state)
//! - State structs for convenient multi-sample processing

use std::f32::consts::PI;

// ============================================================================
// One-pole filters (simple, stateful via returned value)
// ============================================================================

/// One-pole low-pass filter coefficient from cutoff frequency.
///
/// # Arguments
/// * `cutoff` - Cutoff frequency in Hz
/// * `sample_rate` - Sample rate in Hz
///
/// # Returns
/// Coefficient `a` where: `y[n] = a * x[n] + (1 - a) * y[n-1]`
#[inline]
pub fn lowpass_coeff(cutoff: f32, sample_rate: f32) -> f32 {
    let rc = 1.0 / (2.0 * PI * cutoff);
    let dt = 1.0 / sample_rate;
    dt / (rc + dt)
}

/// One-pole high-pass filter coefficient from cutoff frequency.
///
/// # Arguments
/// * `cutoff` - Cutoff frequency in Hz
/// * `sample_rate` - Sample rate in Hz
///
/// # Returns
/// Coefficient `a` where high-pass is applied
#[inline]
pub fn highpass_coeff(cutoff: f32, sample_rate: f32) -> f32 {
    let rc = 1.0 / (2.0 * PI * cutoff);
    let dt = 1.0 / sample_rate;
    rc / (rc + dt)
}

/// Apply one-pole low-pass filter to a single sample.
///
/// # Arguments
/// * `input` - Current input sample
/// * `prev_output` - Previous output sample
/// * `coeff` - Filter coefficient from `lowpass_coeff()`
///
/// # Returns
/// Filtered sample (also becomes next `prev_output`)
#[inline]
pub fn lowpass_sample(input: f32, prev_output: f32, coeff: f32) -> f32 {
    coeff * input + (1.0 - coeff) * prev_output
}

/// Apply one-pole high-pass filter to a single sample.
///
/// # Arguments
/// * `input` - Current input sample
/// * `prev_input` - Previous input sample
/// * `prev_output` - Previous output sample
/// * `coeff` - Filter coefficient from `highpass_coeff()`
///
/// # Returns
/// Filtered sample (also becomes next `prev_output`)
#[inline]
pub fn highpass_sample(input: f32, prev_input: f32, prev_output: f32, coeff: f32) -> f32 {
    coeff * (prev_output + input - prev_input)
}

// ============================================================================
// Stateful filter structs
// ============================================================================

/// One-pole low-pass filter.
#[derive(Debug, Clone)]
pub struct LowPass {
    coeff: f32,
    prev: f32,
}

impl LowPass {
    /// Creates a new low-pass filter.
    pub fn new(cutoff: f32, sample_rate: f32) -> Self {
        Self {
            coeff: lowpass_coeff(cutoff, sample_rate),
            prev: 0.0,
        }
    }

    /// Sets the cutoff frequency.
    pub fn set_cutoff(&mut self, cutoff: f32, sample_rate: f32) {
        self.coeff = lowpass_coeff(cutoff, sample_rate);
    }

    /// Processes a single sample.
    #[inline]
    pub fn process(&mut self, input: f32) -> f32 {
        self.prev = lowpass_sample(input, self.prev, self.coeff);
        self.prev
    }

    /// Processes a buffer of samples in-place.
    pub fn process_buffer(&mut self, buffer: &mut [f32]) {
        for sample in buffer {
            *sample = self.process(*sample);
        }
    }

    /// Resets the filter state.
    pub fn reset(&mut self) {
        self.prev = 0.0;
    }
}

/// One-pole high-pass filter.
#[derive(Debug, Clone)]
pub struct HighPass {
    coeff: f32,
    prev_input: f32,
    prev_output: f32,
}

impl HighPass {
    /// Creates a new high-pass filter.
    pub fn new(cutoff: f32, sample_rate: f32) -> Self {
        Self {
            coeff: highpass_coeff(cutoff, sample_rate),
            prev_input: 0.0,
            prev_output: 0.0,
        }
    }

    /// Sets the cutoff frequency.
    pub fn set_cutoff(&mut self, cutoff: f32, sample_rate: f32) {
        self.coeff = highpass_coeff(cutoff, sample_rate);
    }

    /// Processes a single sample.
    #[inline]
    pub fn process(&mut self, input: f32) -> f32 {
        self.prev_output = highpass_sample(input, self.prev_input, self.prev_output, self.coeff);
        self.prev_input = input;
        self.prev_output
    }

    /// Processes a buffer of samples in-place.
    pub fn process_buffer(&mut self, buffer: &mut [f32]) {
        for sample in buffer {
            *sample = self.process(*sample);
        }
    }

    /// Resets the filter state.
    pub fn reset(&mut self) {
        self.prev_input = 0.0;
        self.prev_output = 0.0;
    }
}

// ============================================================================
// Biquad filter
// ============================================================================

/// Biquad filter coefficients.
#[derive(Debug, Clone, Copy)]
pub struct BiquadCoeffs {
    /// Feedforward coefficient b0.
    pub b0: f32,
    /// Feedforward coefficient b1.
    pub b1: f32,
    /// Feedforward coefficient b2.
    pub b2: f32,
    /// Feedback coefficient a1.
    pub a1: f32,
    /// Feedback coefficient a2.
    pub a2: f32,
}

impl BiquadCoeffs {
    /// Creates low-pass biquad coefficients.
    pub fn lowpass(cutoff: f32, q: f32, sample_rate: f32) -> Self {
        let omega = 2.0 * PI * cutoff / sample_rate;
        let sin_omega = omega.sin();
        let cos_omega = omega.cos();
        let alpha = sin_omega / (2.0 * q);

        let b0 = (1.0 - cos_omega) / 2.0;
        let b1 = 1.0 - cos_omega;
        let b2 = (1.0 - cos_omega) / 2.0;
        let a0 = 1.0 + alpha;
        let a1 = -2.0 * cos_omega;
        let a2 = 1.0 - alpha;

        Self {
            b0: b0 / a0,
            b1: b1 / a0,
            b2: b2 / a0,
            a1: a1 / a0,
            a2: a2 / a0,
        }
    }

    /// Creates high-pass biquad coefficients.
    pub fn highpass(cutoff: f32, q: f32, sample_rate: f32) -> Self {
        let omega = 2.0 * PI * cutoff / sample_rate;
        let sin_omega = omega.sin();
        let cos_omega = omega.cos();
        let alpha = sin_omega / (2.0 * q);

        let b0 = (1.0 + cos_omega) / 2.0;
        let b1 = -(1.0 + cos_omega);
        let b2 = (1.0 + cos_omega) / 2.0;
        let a0 = 1.0 + alpha;
        let a1 = -2.0 * cos_omega;
        let a2 = 1.0 - alpha;

        Self {
            b0: b0 / a0,
            b1: b1 / a0,
            b2: b2 / a0,
            a1: a1 / a0,
            a2: a2 / a0,
        }
    }

    /// Creates band-pass biquad coefficients (constant peak gain).
    pub fn bandpass(center: f32, q: f32, sample_rate: f32) -> Self {
        let omega = 2.0 * PI * center / sample_rate;
        let sin_omega = omega.sin();
        let cos_omega = omega.cos();
        let alpha = sin_omega / (2.0 * q);

        let b0 = alpha;
        let b1 = 0.0;
        let b2 = -alpha;
        let a0 = 1.0 + alpha;
        let a1 = -2.0 * cos_omega;
        let a2 = 1.0 - alpha;

        Self {
            b0: b0 / a0,
            b1: b1 / a0,
            b2: b2 / a0,
            a1: a1 / a0,
            a2: a2 / a0,
        }
    }

    /// Creates notch (band-reject) biquad coefficients.
    pub fn notch(center: f32, q: f32, sample_rate: f32) -> Self {
        let omega = 2.0 * PI * center / sample_rate;
        let sin_omega = omega.sin();
        let cos_omega = omega.cos();
        let alpha = sin_omega / (2.0 * q);

        let b0 = 1.0;
        let b1 = -2.0 * cos_omega;
        let b2 = 1.0;
        let a0 = 1.0 + alpha;
        let a1 = -2.0 * cos_omega;
        let a2 = 1.0 - alpha;

        Self {
            b0: b0 / a0,
            b1: b1 / a0,
            b2: b2 / a0,
            a1: a1 / a0,
            a2: a2 / a0,
        }
    }

    /// Creates all-pass biquad coefficients.
    pub fn allpass(center: f32, q: f32, sample_rate: f32) -> Self {
        let omega = 2.0 * PI * center / sample_rate;
        let sin_omega = omega.sin();
        let cos_omega = omega.cos();
        let alpha = sin_omega / (2.0 * q);

        let b0 = 1.0 - alpha;
        let b1 = -2.0 * cos_omega;
        let b2 = 1.0 + alpha;
        let a0 = 1.0 + alpha;
        let a1 = -2.0 * cos_omega;
        let a2 = 1.0 - alpha;

        Self {
            b0: b0 / a0,
            b1: b1 / a0,
            b2: b2 / a0,
            a1: a1 / a0,
            a2: a2 / a0,
        }
    }

    /// Creates peaking EQ biquad coefficients (bell curve).
    ///
    /// Boosts or cuts frequencies around `center` by `gain_db` decibels
    /// with bandwidth controlled by `q`.
    pub fn peaking(center: f32, q: f32, gain_db: f32, sample_rate: f32) -> Self {
        let a = 10.0_f32.powf(gain_db / 40.0);
        let omega = 2.0 * PI * center / sample_rate;
        let sin_omega = omega.sin();
        let cos_omega = omega.cos();
        let alpha = sin_omega / (2.0 * q);

        let b0 = 1.0 + alpha * a;
        let b1 = -2.0 * cos_omega;
        let b2 = 1.0 - alpha * a;
        let a0 = 1.0 + alpha / a;
        let a1 = -2.0 * cos_omega;
        let a2 = 1.0 - alpha / a;

        Self {
            b0: b0 / a0,
            b1: b1 / a0,
            b2: b2 / a0,
            a1: a1 / a0,
            a2: a2 / a0,
        }
    }

    /// Creates low shelf biquad coefficients.
    ///
    /// Boosts or cuts frequencies below `cutoff` by `gain_db` decibels.
    pub fn low_shelf(cutoff: f32, q: f32, gain_db: f32, sample_rate: f32) -> Self {
        let a = 10.0_f32.powf(gain_db / 40.0);
        let omega = 2.0 * PI * cutoff / sample_rate;
        let sin_omega = omega.sin();
        let cos_omega = omega.cos();
        let alpha = sin_omega / (2.0 * q);
        let two_sqrt_a_alpha = 2.0 * a.sqrt() * alpha;

        let b0 = a * ((a + 1.0) - (a - 1.0) * cos_omega + two_sqrt_a_alpha);
        let b1 = 2.0 * a * ((a - 1.0) - (a + 1.0) * cos_omega);
        let b2 = a * ((a + 1.0) - (a - 1.0) * cos_omega - two_sqrt_a_alpha);
        let a0 = (a + 1.0) + (a - 1.0) * cos_omega + two_sqrt_a_alpha;
        let a1 = -2.0 * ((a - 1.0) + (a + 1.0) * cos_omega);
        let a2 = (a + 1.0) + (a - 1.0) * cos_omega - two_sqrt_a_alpha;

        Self {
            b0: b0 / a0,
            b1: b1 / a0,
            b2: b2 / a0,
            a1: a1 / a0,
            a2: a2 / a0,
        }
    }

    /// Creates high shelf biquad coefficients.
    ///
    /// Boosts or cuts frequencies above `cutoff` by `gain_db` decibels.
    pub fn high_shelf(cutoff: f32, q: f32, gain_db: f32, sample_rate: f32) -> Self {
        let a = 10.0_f32.powf(gain_db / 40.0);
        let omega = 2.0 * PI * cutoff / sample_rate;
        let sin_omega = omega.sin();
        let cos_omega = omega.cos();
        let alpha = sin_omega / (2.0 * q);
        let two_sqrt_a_alpha = 2.0 * a.sqrt() * alpha;

        let b0 = a * ((a + 1.0) + (a - 1.0) * cos_omega + two_sqrt_a_alpha);
        let b1 = -2.0 * a * ((a - 1.0) + (a + 1.0) * cos_omega);
        let b2 = a * ((a + 1.0) + (a - 1.0) * cos_omega - two_sqrt_a_alpha);
        let a0 = (a + 1.0) - (a - 1.0) * cos_omega + two_sqrt_a_alpha;
        let a1 = 2.0 * ((a - 1.0) - (a + 1.0) * cos_omega);
        let a2 = (a + 1.0) - (a - 1.0) * cos_omega - two_sqrt_a_alpha;

        Self {
            b0: b0 / a0,
            b1: b1 / a0,
            b2: b2 / a0,
            a1: a1 / a0,
            a2: a2 / a0,
        }
    }
}

/// Biquad filter (second-order IIR).
#[derive(Debug, Clone)]
pub struct Biquad {
    coeffs: BiquadCoeffs,
    x1: f32,
    x2: f32,
    y1: f32,
    y2: f32,
}

impl Biquad {
    /// Creates a new biquad filter with the given coefficients.
    pub fn new(coeffs: BiquadCoeffs) -> Self {
        Self {
            coeffs,
            x1: 0.0,
            x2: 0.0,
            y1: 0.0,
            y2: 0.0,
        }
    }

    /// Creates a low-pass biquad filter.
    pub fn lowpass(cutoff: f32, q: f32, sample_rate: f32) -> Self {
        Self::new(BiquadCoeffs::lowpass(cutoff, q, sample_rate))
    }

    /// Creates a high-pass biquad filter.
    pub fn highpass(cutoff: f32, q: f32, sample_rate: f32) -> Self {
        Self::new(BiquadCoeffs::highpass(cutoff, q, sample_rate))
    }

    /// Creates a band-pass biquad filter.
    pub fn bandpass(center: f32, q: f32, sample_rate: f32) -> Self {
        Self::new(BiquadCoeffs::bandpass(center, q, sample_rate))
    }

    /// Creates a notch biquad filter.
    pub fn notch(center: f32, q: f32, sample_rate: f32) -> Self {
        Self::new(BiquadCoeffs::notch(center, q, sample_rate))
    }

    /// Creates an all-pass biquad filter.
    pub fn allpass(center: f32, q: f32, sample_rate: f32) -> Self {
        Self::new(BiquadCoeffs::allpass(center, q, sample_rate))
    }

    /// Creates a peaking EQ biquad filter (bell curve).
    pub fn peaking(center: f32, q: f32, gain_db: f32, sample_rate: f32) -> Self {
        Self::new(BiquadCoeffs::peaking(center, q, gain_db, sample_rate))
    }

    /// Creates a low shelf biquad filter.
    pub fn low_shelf(cutoff: f32, q: f32, gain_db: f32, sample_rate: f32) -> Self {
        Self::new(BiquadCoeffs::low_shelf(cutoff, q, gain_db, sample_rate))
    }

    /// Creates a high shelf biquad filter.
    pub fn high_shelf(cutoff: f32, q: f32, gain_db: f32, sample_rate: f32) -> Self {
        Self::new(BiquadCoeffs::high_shelf(cutoff, q, gain_db, sample_rate))
    }

    /// Sets new coefficients.
    pub fn set_coeffs(&mut self, coeffs: BiquadCoeffs) {
        self.coeffs = coeffs;
    }

    /// Processes a single sample.
    #[inline]
    pub fn process(&mut self, input: f32) -> f32 {
        let c = &self.coeffs;
        let output =
            c.b0 * input + c.b1 * self.x1 + c.b2 * self.x2 - c.a1 * self.y1 - c.a2 * self.y2;

        self.x2 = self.x1;
        self.x1 = input;
        self.y2 = self.y1;
        self.y1 = output;

        output
    }

    /// Processes a buffer of samples in-place.
    pub fn process_buffer(&mut self, buffer: &mut [f32]) {
        for sample in buffer {
            *sample = self.process(*sample);
        }
    }

    /// Resets the filter state.
    pub fn reset(&mut self) {
        self.x1 = 0.0;
        self.x2 = 0.0;
        self.y1 = 0.0;
        self.y2 = 0.0;
    }
}

// ============================================================================
// Delay
// ============================================================================

/// Simple delay line.
#[derive(Debug, Clone)]
pub struct Delay {
    buffer: Vec<f32>,
    write_pos: usize,
    delay_samples: usize,
}

impl Delay {
    /// Creates a new delay line.
    ///
    /// # Arguments
    /// * `max_delay_samples` - Maximum delay in samples
    /// * `delay_samples` - Initial delay in samples
    pub fn new(max_delay_samples: usize, delay_samples: usize) -> Self {
        Self {
            buffer: vec![0.0; max_delay_samples],
            write_pos: 0,
            delay_samples: delay_samples.min(max_delay_samples),
        }
    }

    /// Creates a delay from time and sample rate.
    pub fn from_time(max_delay_seconds: f32, delay_seconds: f32, sample_rate: f32) -> Self {
        let max_samples = (max_delay_seconds * sample_rate) as usize;
        let delay_samples = (delay_seconds * sample_rate) as usize;
        Self::new(max_samples, delay_samples)
    }

    /// Sets the delay time in samples.
    pub fn set_delay(&mut self, delay_samples: usize) {
        self.delay_samples = delay_samples.min(self.buffer.len());
    }

    /// Sets the delay time from seconds.
    pub fn set_delay_time(&mut self, delay_seconds: f32, sample_rate: f32) {
        self.set_delay((delay_seconds * sample_rate) as usize);
    }

    /// Reads the delayed sample.
    #[inline]
    pub fn read(&self) -> f32 {
        let read_pos =
            (self.write_pos + self.buffer.len() - self.delay_samples) % self.buffer.len();
        self.buffer[read_pos]
    }

    /// Writes a sample and returns the delayed output.
    #[inline]
    pub fn process(&mut self, input: f32) -> f32 {
        let output = self.read();
        self.buffer[self.write_pos] = input;
        self.write_pos = (self.write_pos + 1) % self.buffer.len();
        output
    }

    /// Processes a buffer of samples in-place.
    pub fn process_buffer(&mut self, buffer: &mut [f32]) {
        for sample in buffer {
            *sample = self.process(*sample);
        }
    }

    /// Clears the delay buffer.
    pub fn clear(&mut self) {
        self.buffer.fill(0.0);
    }
}

/// Delay with feedback (for echoes/reverb tails).
#[derive(Debug, Clone)]
pub struct FeedbackDelay {
    delay: Delay,
    feedback: f32,
}

impl FeedbackDelay {
    /// Creates a new feedback delay.
    ///
    /// # Arguments
    /// * `max_delay_samples` - Maximum delay in samples
    /// * `delay_samples` - Initial delay in samples
    /// * `feedback` - Feedback amount (0.0 to <1.0)
    pub fn new(max_delay_samples: usize, delay_samples: usize, feedback: f32) -> Self {
        Self {
            delay: Delay::new(max_delay_samples, delay_samples),
            feedback: feedback.clamp(0.0, 0.999),
        }
    }

    /// Creates from time and sample rate.
    pub fn from_time(
        max_delay_seconds: f32,
        delay_seconds: f32,
        feedback: f32,
        sample_rate: f32,
    ) -> Self {
        Self {
            delay: Delay::from_time(max_delay_seconds, delay_seconds, sample_rate),
            feedback: feedback.clamp(0.0, 0.999),
        }
    }

    /// Sets the feedback amount.
    pub fn set_feedback(&mut self, feedback: f32) {
        self.feedback = feedback.clamp(0.0, 0.999);
    }

    /// Sets the delay time in samples.
    pub fn set_delay(&mut self, delay_samples: usize) {
        self.delay.set_delay(delay_samples);
    }

    /// Processes a single sample.
    #[inline]
    pub fn process(&mut self, input: f32) -> f32 {
        let delayed = self.delay.read();
        let output = input + delayed * self.feedback;
        self.delay.buffer[self.delay.write_pos] = output;
        self.delay.write_pos = (self.delay.write_pos + 1) % self.delay.buffer.len();
        delayed
    }

    /// Processes a buffer of samples in-place.
    pub fn process_buffer(&mut self, buffer: &mut [f32]) {
        for sample in buffer {
            *sample = self.process(*sample);
        }
    }

    /// Clears the delay buffer.
    pub fn clear(&mut self) {
        self.delay.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lowpass_coeff() {
        let coeff = lowpass_coeff(1000.0, 44100.0);
        assert!(coeff > 0.0 && coeff < 1.0);
    }

    #[test]
    fn test_lowpass_filter() {
        let mut filter = LowPass::new(1000.0, 44100.0);

        // Step response should approach 1.0
        let mut output = 0.0;
        for _ in 0..1000 {
            output = filter.process(1.0);
        }
        assert!((output - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_highpass_filter() {
        let mut filter = HighPass::new(1000.0, 44100.0);

        // DC should be blocked
        for _ in 0..1000 {
            filter.process(1.0);
        }
        let output = filter.process(1.0);
        assert!(output.abs() < 0.01);
    }

    #[test]
    fn test_biquad_lowpass() {
        let mut filter = Biquad::lowpass(1000.0, 0.707, 44100.0);

        // Process some samples
        let mut output = 0.0;
        for _ in 0..1000 {
            output = filter.process(1.0);
        }
        assert!((output - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_delay() {
        let mut delay = Delay::new(100, 10);

        // First 10 samples should be zero
        for _ in 0..10 {
            let out = delay.process(1.0);
            assert_eq!(out, 0.0);
        }

        // After delay, should output 1.0
        let out = delay.process(1.0);
        assert_eq!(out, 1.0);
    }

    #[test]
    fn test_feedback_delay() {
        let mut delay = FeedbackDelay::new(100, 10, 0.5);

        // Write a single impulse
        delay.process(1.0);
        for _ in 0..9 {
            delay.process(0.0);
        }

        // First echo
        let echo1 = delay.process(0.0);
        assert!((echo1 - 1.0).abs() < 0.001);

        // Second echo (after feedback)
        for _ in 0..9 {
            delay.process(0.0);
        }
        let echo2 = delay.process(0.0);
        assert!((echo2 - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_filter_reset() {
        let mut filter = LowPass::new(1000.0, 44100.0);
        filter.process(1.0);
        filter.process(1.0);
        filter.reset();

        // After reset, should start fresh
        let out = filter.process(1.0);
        assert!(out < 0.5); // Should be ramping up from 0
    }

    #[test]
    fn test_peaking_eq_boost() {
        let sample_rate = 44100.0;
        let center = 1000.0;
        let mut filter = Biquad::peaking(center, 1.0, 6.0, sample_rate);

        // Generate sine at center frequency
        let mut signal: Vec<f32> = (0..4410)
            .map(|i| (2.0 * std::f32::consts::PI * center * i as f32 / sample_rate).sin())
            .collect();

        // Warm up filter
        for sample in signal.iter_mut().take(100) {
            *sample = filter.process(*sample);
        }

        // Measure RMS of remaining signal
        let rms: f32 = signal[100..]
            .iter()
            .map(|&x| filter.process(x).powi(2))
            .sum::<f32>()
            .sqrt();

        // With +6dB boost, output should be louder than input
        // Input RMS of sine is ~0.707, boosted should be ~1.4
        assert!(rms > 50.0, "Peaking filter should boost center frequency");
    }

    #[test]
    fn test_peaking_eq_cut() {
        let sample_rate = 44100.0;
        let center = 1000.0;

        // Generate sine at center frequency
        let signal: Vec<f32> = (0..4410)
            .map(|i| (2.0 * std::f32::consts::PI * center * i as f32 / sample_rate).sin())
            .collect();

        // Measure with boost
        let mut boost_filter = Biquad::peaking(center, 1.0, 12.0, sample_rate);
        let mut boost_max = 0.0f32;
        for &sample in signal.iter() {
            boost_max = boost_max.max(boost_filter.process(sample).abs());
        }

        // Measure with cut
        let mut cut_filter = Biquad::peaking(center, 1.0, -12.0, sample_rate);
        let mut cut_max = 0.0f32;
        for &sample in signal.iter() {
            cut_max = cut_max.max(cut_filter.process(sample).abs());
        }

        // Boost should increase, cut should decrease relative to unity (1.0)
        assert!(
            boost_max > 1.0,
            "Boost should exceed unity, got {}",
            boost_max
        );
        assert!(cut_max < 1.0, "Cut should be below unity, got {}", cut_max);
        assert!(
            boost_max > cut_max * 2.0,
            "Boost should be significantly more than cut"
        );
    }

    #[test]
    fn test_low_shelf() {
        let sample_rate = 44100.0;
        let mut filter = Biquad::low_shelf(200.0, 0.707, 6.0, sample_rate);

        // Low frequency should be boosted
        let low_freq = 100.0;
        let signal: Vec<f32> = (0..4410)
            .map(|i| (2.0 * std::f32::consts::PI * low_freq * i as f32 / sample_rate).sin())
            .collect();

        let mut max_out = 0.0f32;
        for &sample in signal.iter().skip(100) {
            max_out = max_out.max(filter.process(sample).abs());
        }

        // Boosted signal should exceed unity
        assert!(max_out > 1.0, "Low shelf should boost low frequencies");
    }

    #[test]
    fn test_high_shelf() {
        let sample_rate = 44100.0;
        let mut filter = Biquad::high_shelf(2000.0, 0.707, 6.0, sample_rate);

        // High frequency should be boosted
        let high_freq = 4000.0;
        let signal: Vec<f32> = (0..4410)
            .map(|i| (2.0 * std::f32::consts::PI * high_freq * i as f32 / sample_rate).sin())
            .collect();

        let mut max_out = 0.0f32;
        for &sample in signal.iter().skip(100) {
            max_out = max_out.max(filter.process(sample).abs());
        }

        // Boosted signal should exceed unity
        assert!(max_out > 1.0, "High shelf should boost high frequencies");
    }

    #[test]
    fn test_peaking_zero_gain_is_identity() {
        let sample_rate = 44100.0;
        let mut filter = Biquad::peaking(1000.0, 1.0, 0.0, sample_rate);

        // With 0 dB gain, filter should pass signal unchanged
        let input = 0.5;
        // Warm up
        for _ in 0..100 {
            filter.process(input);
        }
        let output = filter.process(input);

        assert!(
            (output - input).abs() < 0.01,
            "Peaking with 0dB should be identity"
        );
    }
}

// ============================================================================
// Invariant tests - frequency response validation via FFT
// ============================================================================

#[cfg(all(test, feature = "invariant-tests"))]
mod invariant_tests {
    use super::*;
    use rustfft::{FftPlanner, num_complex::Complex};

    const SAMPLE_RATE: f32 = 44100.0;
    const FFT_SIZE: usize = 4096;

    /// Compute magnitude frequency response of a filter using impulse response
    fn frequency_response(filter: &mut Biquad) -> Vec<f32> {
        filter.reset();

        // Generate impulse response
        let mut ir = vec![0.0f32; FFT_SIZE];
        ir[0] = filter.process(1.0);
        for i in 1..FFT_SIZE {
            ir[i] = filter.process(0.0);
        }

        // FFT
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(FFT_SIZE);

        let mut buffer: Vec<Complex<f32>> = ir.iter().map(|&x| Complex::new(x, 0.0)).collect();
        fft.process(&mut buffer);

        // Magnitude spectrum (first half)
        buffer[..FFT_SIZE / 2].iter().map(|c| c.norm()).collect()
    }

    /// Convert bin index to frequency
    fn bin_to_freq(bin: usize) -> f32 {
        bin as f32 * SAMPLE_RATE / FFT_SIZE as f32
    }

    /// Find bin index closest to frequency
    fn freq_to_bin(freq: f32) -> usize {
        (freq * FFT_SIZE as f32 / SAMPLE_RATE).round() as usize
    }

    #[test]
    fn test_lowpass_attenuates_high_frequencies() {
        let cutoff = 1000.0;
        let mut filter = Biquad::lowpass(cutoff, 0.707, SAMPLE_RATE);
        let response = frequency_response(&mut filter);

        let dc_mag = response[1]; // Avoid bin 0 for DC
        let high_freq_bin = freq_to_bin(10000.0);
        let high_freq_mag = response[high_freq_bin.min(response.len() - 1)];

        // High frequencies should be significantly attenuated
        assert!(
            high_freq_mag < dc_mag * 0.1,
            "Lowpass should attenuate 10kHz by >20dB: DC={}, 10kHz={}",
            dc_mag,
            high_freq_mag
        );
    }

    #[test]
    fn test_highpass_attenuates_low_frequencies() {
        let cutoff = 1000.0;
        let mut filter = Biquad::highpass(cutoff, 0.707, SAMPLE_RATE);
        let response = frequency_response(&mut filter);

        let low_freq_bin = freq_to_bin(100.0);
        let low_freq_mag = response[low_freq_bin.max(1)];
        let high_freq_bin = freq_to_bin(10000.0);
        let high_freq_mag = response[high_freq_bin.min(response.len() - 1)];

        // Low frequencies should be significantly attenuated
        assert!(
            low_freq_mag < high_freq_mag * 0.2,
            "Highpass should attenuate 100Hz: 100Hz={}, 10kHz={}",
            low_freq_mag,
            high_freq_mag
        );
    }

    #[test]
    fn test_bandpass_peaks_at_center() {
        let center = 2000.0;
        let mut filter = Biquad::bandpass(center, 2.0, SAMPLE_RATE);
        let response = frequency_response(&mut filter);

        let center_bin = freq_to_bin(center);
        let low_bin = freq_to_bin(200.0);
        let high_bin = freq_to_bin(15000.0);

        let center_mag = response[center_bin.min(response.len() - 1)];
        let low_mag = response[low_bin.max(1)];
        let high_mag = response[high_bin.min(response.len() - 1)];

        // Center frequency should be loudest
        assert!(
            center_mag > low_mag * 2.0 && center_mag > high_mag * 2.0,
            "Bandpass should peak at center: low={}, center={}, high={}",
            low_mag,
            center_mag,
            high_mag
        );
    }

    #[test]
    fn test_notch_attenuates_center() {
        let center = 2000.0;
        let mut filter = Biquad::notch(center, 2.0, SAMPLE_RATE);
        let response = frequency_response(&mut filter);

        let center_bin = freq_to_bin(center);
        let nearby_bin = freq_to_bin(center * 2.0);

        let center_mag = response[center_bin.min(response.len() - 1)];
        let nearby_mag = response[nearby_bin.min(response.len() - 1)];

        // Center frequency should be attenuated
        assert!(
            center_mag < nearby_mag * 0.5,
            "Notch should attenuate center: center={}, nearby={}",
            center_mag,
            nearby_mag
        );
    }

    #[test]
    fn test_allpass_preserves_magnitude() {
        let center = 2000.0;
        let mut filter = Biquad::allpass(center, 1.0, SAMPLE_RATE);
        let response = frequency_response(&mut filter);

        // All-pass should have unity magnitude at all frequencies
        let low_mag = response[freq_to_bin(100.0).max(1)];
        let mid_mag = response[freq_to_bin(2000.0)];
        let high_mag = response[freq_to_bin(10000.0).min(response.len() - 1)];

        assert!(
            (low_mag - 1.0).abs() < 0.1,
            "Allpass should be ~1.0 at 100Hz, got {}",
            low_mag
        );
        assert!(
            (mid_mag - 1.0).abs() < 0.1,
            "Allpass should be ~1.0 at 2kHz, got {}",
            mid_mag
        );
        assert!(
            (high_mag - 1.0).abs() < 0.1,
            "Allpass should be ~1.0 at 10kHz, got {}",
            high_mag
        );
    }

    #[test]
    fn test_filter_stability() {
        // Filters should not explode with random input
        let mut lowpass = Biquad::lowpass(1000.0, 0.707, SAMPLE_RATE);
        let mut highpass = Biquad::highpass(1000.0, 0.707, SAMPLE_RATE);
        let mut bandpass = Biquad::bandpass(1000.0, 2.0, SAMPLE_RATE);

        // Process random-ish signal
        for i in 0..10000 {
            let input = ((i * 7919) % 1000) as f32 / 500.0 - 1.0;
            let lp_out = lowpass.process(input);
            let hp_out = highpass.process(input);
            let bp_out = bandpass.process(input);

            assert!(
                lp_out.is_finite() && lp_out.abs() < 100.0,
                "Lowpass unstable at sample {}: {}",
                i,
                lp_out
            );
            assert!(
                hp_out.is_finite() && hp_out.abs() < 100.0,
                "Highpass unstable at sample {}: {}",
                i,
                hp_out
            );
            assert!(
                bp_out.is_finite() && bp_out.abs() < 100.0,
                "Bandpass unstable at sample {}: {}",
                i,
                bp_out
            );
        }
    }

    #[test]
    fn test_lowpass_cutoff_ordering() {
        // Lower cutoff should attenuate high frequencies more
        let test_freq = 5000.0;
        let test_bin = freq_to_bin(test_freq);

        let mut filter_500 = Biquad::lowpass(500.0, 0.707, SAMPLE_RATE);
        let mut filter_2000 = Biquad::lowpass(2000.0, 0.707, SAMPLE_RATE);
        let mut filter_8000 = Biquad::lowpass(8000.0, 0.707, SAMPLE_RATE);

        let resp_500 = frequency_response(&mut filter_500);
        let resp_2000 = frequency_response(&mut filter_2000);
        let resp_8000 = frequency_response(&mut filter_8000);

        let mag_500 = resp_500[test_bin.min(resp_500.len() - 1)];
        let mag_2000 = resp_2000[test_bin.min(resp_2000.len() - 1)];
        let mag_8000 = resp_8000[test_bin.min(resp_8000.len() - 1)];

        assert!(
            mag_500 < mag_2000 && mag_2000 < mag_8000,
            "Lower cutoff should attenuate more at {}: 500Hz={}, 2kHz={}, 8kHz={}",
            test_freq,
            mag_500,
            mag_2000,
            mag_8000
        );
    }
}
