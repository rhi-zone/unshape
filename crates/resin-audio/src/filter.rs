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
    pub b0: f32,
    pub b1: f32,
    pub b2: f32,
    pub a1: f32,
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
}
