//! [`SignalGenerator`] — a synthetic, wall-clock-driven [`LiveSource`] for testing
//! live-source plumbing without a real capture device.

use std::f64::consts::TAU;
use std::time::Instant;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::source::LiveSource;

/// Waveform shape sampled by a [`SignalGenerator`]. Per house style this is the
/// serializable parameter set (the op); [`SignalGeneratorConfig::start`] builds
/// the running [`SignalGenerator`] source from it.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum Waveform {
    /// A sine wave: `amplitude * sin(TAU * frequency_hz * t + phase)`.
    Sine {
        /// Oscillation frequency in Hz.
        frequency_hz: f32,
        /// Peak amplitude.
        amplitude: f32,
        /// Phase offset in radians.
        phase: f32,
    },
    /// A sawtooth ramp cycling through `[0, amplitude)` at `frequency_hz`.
    Ramp {
        /// Cycles per second.
        frequency_hz: f32,
        /// Value at the top of the ramp before it wraps back to zero.
        amplitude: f32,
    },
    /// A two-level square wave alternating between `-amplitude` and `amplitude`.
    Square {
        /// Cycles per second.
        frequency_hz: f32,
        /// Magnitude of each level.
        amplitude: f32,
    },
    /// A fixed value that never changes — useful as a control in tests.
    Constant {
        /// The value always returned.
        value: f32,
    },
}

impl Waveform {
    fn sample(&self, elapsed_secs: f64) -> f32 {
        match *self {
            Waveform::Sine {
                frequency_hz,
                amplitude,
                phase,
            } => {
                let angle = TAU * frequency_hz as f64 * elapsed_secs + phase as f64;
                amplitude * angle.sin() as f32
            }
            Waveform::Ramp {
                frequency_hz,
                amplitude,
            } => {
                let phase = (elapsed_secs * frequency_hz as f64).rem_euclid(1.0);
                amplitude * phase as f32
            }
            Waveform::Square {
                frequency_hz,
                amplitude,
            } => {
                let phase = (elapsed_secs * frequency_hz as f64).rem_euclid(1.0);
                if phase < 0.5 { amplitude } else { -amplitude }
            }
            Waveform::Constant { value } => value,
        }
    }
}

/// Serializable configuration for a [`SignalGenerator`]. The running generator
/// itself holds a start `Instant` and so is not serializable; save/load this
/// config and call [`start`](SignalGeneratorConfig::start) to rebuild it.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct SignalGeneratorConfig {
    /// The waveform to generate.
    pub waveform: Waveform,
}

impl SignalGeneratorConfig {
    /// Creates a config for the given waveform.
    pub fn new(waveform: Waveform) -> Self {
        Self { waveform }
    }

    /// Builds a running [`SignalGenerator`], with its clock starting now.
    #[allow(
        clippy::disallowed_methods,
        reason = "SignalGenerator is a deliberate wall-clock-driven LiveSource for \
                  testing live-source plumbing; graph nodes/ops stay deterministic by \
                  reading its polled output, not the clock directly"
    )]
    pub fn start(&self) -> SignalGenerator {
        SignalGenerator {
            waveform: self.waveform,
            started_at: Instant::now(),
        }
    }
}

/// A [`LiveSource`] that produces a synthetic value varying with wall-clock
/// time — a sine wave, ramp, square wave, or constant. Intended for testing
/// live-source plumbing (graph wiring, staleness policy, node integration)
/// without a real capture device.
///
/// Always has a fresh reading — every [`poll`](LiveSource::poll) call returns
/// `Some`, same as [`crate::ClockSource`].
pub struct SignalGenerator {
    waveform: Waveform,
    started_at: Instant,
}

impl SignalGenerator {
    /// Starts a generator for the given waveform, with its clock starting now.
    pub fn new(waveform: Waveform) -> Self {
        SignalGeneratorConfig::new(waveform).start()
    }
}

impl LiveSource for SignalGenerator {
    type Output = f32;

    fn poll(&mut self) -> Option<f32> {
        let elapsed_secs = self.started_at.elapsed().as_secs_f64();
        Some(self.waveform.sample(elapsed_secs))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constant_waveform_never_changes() {
        let mut generator = SignalGenerator::new(Waveform::Constant { value: 4.0 });
        assert_eq!(generator.poll(), Some(4.0));
        assert_eq!(generator.poll(), Some(4.0));
    }

    #[test]
    fn sine_starts_at_phase_zero() {
        let mut generator = SignalGenerator::new(Waveform::Sine {
            frequency_hz: 1.0,
            amplitude: 1.0,
            phase: 0.0,
        });
        let first = generator.poll().unwrap();
        assert!(first.abs() < 0.1, "expected near zero, got {first}");
    }

    #[test]
    fn ramp_stays_within_amplitude_bounds() {
        let mut generator = SignalGenerator::new(Waveform::Ramp {
            frequency_hz: 100.0,
            amplitude: 2.0,
        });
        for _ in 0..5 {
            let v = generator.poll().unwrap();
            assert!((0.0..2.0).contains(&v), "ramp value out of bounds: {v}");
        }
    }

    #[test]
    fn square_only_takes_two_levels() {
        let mut generator = SignalGenerator::new(Waveform::Square {
            frequency_hz: 1.0,
            amplitude: 3.0,
        });
        for _ in 0..5 {
            let v = generator.poll().unwrap();
            assert!(v == 3.0 || v == -3.0, "unexpected square level: {v}");
        }
    }

    #[test]
    fn config_round_trips_to_a_running_generator() {
        let config = SignalGeneratorConfig::new(Waveform::Constant { value: 9.0 });
        let mut generator = config.start();
        assert_eq!(generator.poll(), Some(9.0));
    }

    #[cfg(feature = "serde")]
    #[test]
    fn config_serializes() {
        let config = SignalGeneratorConfig::new(Waveform::Sine {
            frequency_hz: 440.0,
            amplitude: 1.0,
            phase: 0.0,
        });
        let json = serde_json::to_string(&config).unwrap();
        let round_tripped: SignalGeneratorConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(round_tripped, config);
    }
}
