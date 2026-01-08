//! Audio oscillators.
//!
//! Pure functions of phase - no internal state.
//! Phase is in [0, 1] representing one cycle.

use std::f32::consts::TAU;

/// Sine wave oscillator.
///
/// # Arguments
/// * `phase` - Phase in [0, 1], wraps automatically.
///
/// # Returns
/// Value in [-1, 1].
#[inline]
pub fn sine(phase: f32) -> f32 {
    (phase * TAU).sin()
}

/// Square wave oscillator.
///
/// # Arguments
/// * `phase` - Phase in [0, 1].
///
/// # Returns
/// Value in [-1, 1].
#[inline]
pub fn square(phase: f32) -> f32 {
    if phase.fract() < 0.5 { 1.0 } else { -1.0 }
}

/// Pulse wave oscillator with variable duty cycle.
///
/// # Arguments
/// * `phase` - Phase in [0, 1].
/// * `duty` - Duty cycle in [0, 1]. 0.5 = square wave.
///
/// # Returns
/// Value in [-1, 1].
#[inline]
pub fn pulse(phase: f32, duty: f32) -> f32 {
    if phase.fract() < duty { 1.0 } else { -1.0 }
}

/// Sawtooth wave oscillator (rising).
///
/// # Arguments
/// * `phase` - Phase in [0, 1].
///
/// # Returns
/// Value in [-1, 1].
#[inline]
pub fn saw(phase: f32) -> f32 {
    2.0 * phase.fract() - 1.0
}

/// Reverse sawtooth wave oscillator (falling).
///
/// # Arguments
/// * `phase` - Phase in [0, 1].
///
/// # Returns
/// Value in [-1, 1].
#[inline]
pub fn saw_rev(phase: f32) -> f32 {
    1.0 - 2.0 * phase.fract()
}

/// Triangle wave oscillator.
///
/// # Arguments
/// * `phase` - Phase in [0, 1].
///
/// # Returns
/// Value in [-1, 1].
#[inline]
pub fn triangle(phase: f32) -> f32 {
    let p = phase.fract();
    if p < 0.5 {
        4.0 * p - 1.0
    } else {
        3.0 - 4.0 * p
    }
}

/// Converts frequency (Hz) and time (seconds) to phase.
///
/// # Example
/// ```
/// use resin_audio::osc::{freq_to_phase, sine};
///
/// let frequency = 440.0; // A4
/// let time = 0.001; // 1ms
/// let phase = freq_to_phase(frequency, time);
/// let sample = sine(phase);
/// ```
#[inline]
pub fn freq_to_phase(frequency: f32, time: f32) -> f32 {
    frequency * time
}

/// Generates a phase value from sample index and sample rate.
///
/// # Arguments
/// * `frequency` - Oscillator frequency in Hz.
/// * `sample_index` - Current sample number.
/// * `sample_rate` - Samples per second.
#[inline]
pub fn sample_to_phase(frequency: f32, sample_index: u64, sample_rate: f32) -> f32 {
    frequency * (sample_index as f32 / sample_rate)
}

/// Polyblep anti-aliasing correction for naive waveforms.
///
/// Reduces aliasing artifacts at discontinuities.
#[inline]
fn poly_blep(t: f32, dt: f32) -> f32 {
    if t < dt {
        let t = t / dt;
        2.0 * t - t * t - 1.0
    } else if t > 1.0 - dt {
        let t = (t - 1.0) / dt;
        t * t + 2.0 * t + 1.0
    } else {
        0.0
    }
}

/// Band-limited square wave using polyblep.
///
/// Reduces aliasing compared to naive square.
#[inline]
pub fn square_blep(phase: f32, dt: f32) -> f32 {
    let p = phase.fract();
    let mut value = if p < 0.5 { 1.0 } else { -1.0 };
    value += poly_blep(p, dt);
    value -= poly_blep((p + 0.5).fract(), dt);
    value
}

/// Band-limited sawtooth wave using polyblep.
#[inline]
pub fn saw_blep(phase: f32, dt: f32) -> f32 {
    let p = phase.fract();
    let mut value = 2.0 * p - 1.0;
    value -= poly_blep(p, dt);
    value
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sine_range() {
        for i in 0..100 {
            let phase = i as f32 / 100.0;
            let v = sine(phase);
            assert!(
                v >= -1.0 && v <= 1.0,
                "sine({}) = {} out of range",
                phase,
                v
            );
        }
    }

    #[test]
    fn test_sine_zero_crossing() {
        assert!((sine(0.0)).abs() < 0.001);
        assert!((sine(0.5)).abs() < 0.001);
    }

    #[test]
    fn test_sine_peaks() {
        assert!((sine(0.25) - 1.0).abs() < 0.001);
        assert!((sine(0.75) + 1.0).abs() < 0.001);
    }

    #[test]
    fn test_square_range() {
        for i in 0..100 {
            let phase = i as f32 / 100.0;
            let v = square(phase);
            assert!(v == 1.0 || v == -1.0);
        }
    }

    #[test]
    fn test_square_duty() {
        assert_eq!(square(0.0), 1.0);
        assert_eq!(square(0.49), 1.0);
        assert_eq!(square(0.51), -1.0);
        assert_eq!(square(0.99), -1.0);
    }

    #[test]
    fn test_saw_range() {
        for i in 0..100 {
            let phase = i as f32 / 100.0;
            let v = saw(phase);
            assert!(v >= -1.0 && v <= 1.0, "saw({}) = {} out of range", phase, v);
        }
    }

    #[test]
    fn test_triangle_range() {
        for i in 0..100 {
            let phase = i as f32 / 100.0;
            let v = triangle(phase);
            assert!(
                v >= -1.0 && v <= 1.0,
                "triangle({}) = {} out of range",
                phase,
                v
            );
        }
    }

    #[test]
    fn test_triangle_peaks() {
        assert!((triangle(0.0) + 1.0).abs() < 0.001);
        assert!((triangle(0.25)).abs() < 0.001);
        assert!((triangle(0.5) - 1.0).abs() < 0.001);
        assert!((triangle(0.75)).abs() < 0.001);
    }

    #[test]
    fn test_pulse_duty() {
        // 25% duty cycle
        assert_eq!(pulse(0.0, 0.25), 1.0);
        assert_eq!(pulse(0.2, 0.25), 1.0);
        assert_eq!(pulse(0.3, 0.25), -1.0);
        assert_eq!(pulse(0.9, 0.25), -1.0);
    }

    #[test]
    fn test_freq_to_phase() {
        // 1 Hz at t=1 should give phase 1.0 (one full cycle)
        assert!((freq_to_phase(1.0, 1.0) - 1.0).abs() < 0.001);
        // 440 Hz at t=1/440 should give phase 1.0
        assert!((freq_to_phase(440.0, 1.0 / 440.0) - 1.0).abs() < 0.001);
    }
}
