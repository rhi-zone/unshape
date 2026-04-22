//! Utility DSP building blocks.
//!
//! Provides common signal-processing utilities:
//! - [`PitchQuantizer`] — snap continuous pitch to the nearest scale note
//! - [`SampleAndHold`] — hold a value on trigger
//! - [`SlewLimiter`] — rate-limit signal rise and fall
//! - [`MidSideEncode`] / [`MidSideDecode`] — stereo L/R ↔ M/S matrix

// ============================================================================
// PitchQuantizer
// ============================================================================

/// Scale mode for [`PitchQuantizer`].
#[derive(Debug, Clone, PartialEq)]
pub enum ScaleMode {
    /// All 12 semitones (no quantization effect).
    Chromatic,
    /// Major (Ionian) scale intervals.
    Major,
    /// Natural minor (Aeolian) scale intervals.
    NaturalMinor,
    /// Harmonic minor scale intervals.
    HarmonicMinor,
    /// Melodic minor scale intervals.
    MelodicMinor,
    /// Major pentatonic scale intervals.
    Pentatonic,
    /// Blues scale intervals.
    Blues,
    /// Custom set of intervals in semitones within one octave.
    Custom(Vec<u8>),
}

impl ScaleMode {
    /// Returns the semitone intervals (within one octave) for this scale mode.
    pub fn intervals(&self) -> &[u8] {
        match self {
            ScaleMode::Chromatic => &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            ScaleMode::Major => &[0, 2, 4, 5, 7, 9, 11],
            ScaleMode::NaturalMinor => &[0, 2, 3, 5, 7, 8, 10],
            ScaleMode::HarmonicMinor => &[0, 2, 3, 5, 7, 8, 11],
            ScaleMode::MelodicMinor => &[0, 2, 3, 5, 7, 9, 11],
            ScaleMode::Pentatonic => &[0, 2, 4, 7, 9],
            ScaleMode::Blues => &[0, 3, 5, 6, 7, 10],
            ScaleMode::Custom(v) => v.as_slice(),
        }
    }
}

/// Quantizes a continuous pitch to the nearest note in a musical scale.
///
/// Pitches can be expressed either in semitones (relative to C4 = 0) or in Hz.
/// When `octave_wrap` is `true`, the quantized note is always chosen within the
/// same octave as the input before octave-wrapping is applied.
///
/// # Example
/// ```
/// use unshape_audio::utility::{PitchQuantizer, ScaleMode};
///
/// let q = PitchQuantizer { root: 0.0, scale: ScaleMode::Major, octave_wrap: true };
/// // C#4 (1 semitone above C4) → C4 or D4 in major scale
/// let snapped = q.quantize_semitones(1.0);
/// assert!(snapped == 0.0 || snapped == 2.0);
/// ```
#[derive(Debug, Clone)]
pub struct PitchQuantizer {
    /// Root note in semitones (0 = C4 = 261.63 Hz).
    pub root: f32,
    /// Scale mode.
    pub scale: ScaleMode,
    /// When true, wrap the quantized pitch to the same octave as the input.
    pub octave_wrap: bool,
}

impl PitchQuantizer {
    /// Quantizes a pitch given in semitones (relative to C4) to the nearest
    /// scale note.
    pub fn quantize_semitones(&self, semitones: f32) -> f32 {
        let intervals = self.scale.intervals();
        if intervals.is_empty() {
            return semitones;
        }

        // Express input relative to the root note.
        let relative = semitones - self.root;

        // Determine octave and semitone within octave.
        let octave = relative.div_euclid(12.0).floor();
        let semi_in_oct = relative.rem_euclid(12.0);

        // Find the nearest interval.
        let mut best_interval = intervals[0] as f32;
        let mut best_dist = (semi_in_oct - best_interval).abs();

        for &iv in intervals.iter().skip(1) {
            let dist = (semi_in_oct - iv as f32).abs();
            if dist < best_dist {
                best_dist = dist;
                best_interval = iv as f32;
            }
        }

        // Also consider the first interval of the next octave (wraps around).
        let next_oct_first = intervals[0] as f32 + 12.0;
        let dist_wrap = (semi_in_oct - next_oct_first).abs();
        let (quantized_in_oct, octave_adj) = if dist_wrap < best_dist {
            (intervals[0] as f32, octave + 1.0)
        } else {
            (best_interval, octave)
        };

        if self.octave_wrap {
            self.root + octave * 12.0 + quantized_in_oct
        } else {
            self.root + octave_adj * 12.0 + quantized_in_oct
        }
    }

    /// Quantizes a frequency in Hz to the nearest scale note, returning Hz.
    pub fn quantize_hz(&self, freq: f32) -> f32 {
        // C4 = 261.63 Hz, 0 semitones
        let semitones = 12.0 * (freq / 261.630_5_f32).log2();
        let snapped = self.quantize_semitones(semitones);
        261.630_5_f32 * 2.0_f32.powf(snapped / 12.0)
    }
}

// ============================================================================
// SampleAndHold
// ============================================================================

/// Holds the last sampled value until a new trigger arrives.
///
/// When `trigger > 0.5` the current `input` is captured; otherwise the
/// previously held value is returned.
///
/// # Example
/// ```
/// use unshape_audio::utility::SampleAndHold;
///
/// let mut sh = SampleAndHold::default();
/// // Trigger high → sample the input.
/// assert_eq!(sh.process(3.14, 1.0), 3.14);
/// // Trigger low → return held value.
/// assert_eq!(sh.process(9.99, 0.0), 3.14);
/// ```
#[derive(Debug, Clone, Default)]
pub struct SampleAndHold {
    /// Currently held value.
    held: f32,
}

impl SampleAndHold {
    /// Creates a new sample-and-hold starting at zero.
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a new sample-and-hold with an initial held value.
    pub fn with_initial(value: f32) -> Self {
        Self { held: value }
    }

    /// Processes one sample.
    ///
    /// Returns `input` (and stores it) when `trigger > 0.5`; otherwise returns
    /// the previously held value.
    #[inline]
    pub fn process(&mut self, input: f32, trigger: f32) -> f32 {
        if trigger > 0.5 {
            self.held = input;
        }
        self.held
    }

    /// Returns the currently held value without processing a new sample.
    #[inline]
    pub fn held(&self) -> f32 {
        self.held
    }

    /// Resets the held value to zero.
    pub fn reset(&mut self) {
        self.held = 0.0;
    }
}

// ============================================================================
// SlewLimiter
// ============================================================================

/// Rate-limits a signal's rise and fall.
///
/// The output tracks the target but is limited to rising by at most `rise_rate`
/// per sample and falling by at most `fall_rate` per sample.
///
/// # Example
/// ```
/// use unshape_audio::utility::SlewLimiter;
///
/// let mut sl = SlewLimiter { rise_rate: 0.1, fall_rate: 0.1, current: 0.0 };
/// let out = sl.process(1.0);
/// assert!((out - 0.1).abs() < 1e-6);
/// ```
#[derive(Debug, Clone)]
pub struct SlewLimiter {
    /// Maximum increase per sample.
    pub rise_rate: f32,
    /// Maximum decrease per sample.
    pub fall_rate: f32,
    /// Current (output) value.
    pub current: f32,
}

impl SlewLimiter {
    /// Creates a new slew limiter.
    pub fn new(rise_rate: f32, fall_rate: f32) -> Self {
        Self {
            rise_rate,
            fall_rate,
            current: 0.0,
        }
    }

    /// Creates a symmetric slew limiter suitable for pitch portamento.
    ///
    /// `time_samples` is the number of samples to glide one octave (12 semitones).
    /// The rate is expressed per sample.
    pub fn portamento(time_samples: f32) -> Self {
        let rate = if time_samples > 0.0 {
            1.0 / time_samples
        } else {
            f32::INFINITY
        };
        Self::new(rate, rate)
    }

    /// Processes one sample, returning the slew-rate-limited output.
    #[inline]
    pub fn process(&mut self, target: f32) -> f32 {
        let delta = target - self.current;
        self.current += delta.clamp(-self.fall_rate, self.rise_rate);
        self.current
    }

    /// Resets the current value to zero.
    pub fn reset(&mut self) {
        self.current = 0.0;
    }

    /// Resets the current value to a specific starting point.
    pub fn reset_to(&mut self, value: f32) {
        self.current = value;
    }
}

// ============================================================================
// Mid-Side matrix
// ============================================================================

/// Encoded mid-side signal pair.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MidSidePair {
    /// Mid channel: (L + R) / √2.
    pub mid: f32,
    /// Side channel: (L − R) / √2.
    pub side: f32,
}

/// Decoded stereo signal pair.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StereoPair {
    /// Left channel.
    pub left: f32,
    /// Right channel.
    pub right: f32,
}

/// Encodes a stereo L/R signal into M/S representation.
///
/// M = (L + R) / √2, S = (L − R) / √2
#[derive(Debug, Clone, Copy, Default)]
pub struct MidSideEncode;

impl MidSideEncode {
    /// Encodes a stereo sample pair into mid/side.
    #[inline]
    pub fn apply(&self, left: f32, right: f32) -> MidSidePair {
        let scale = std::f32::consts::FRAC_1_SQRT_2;
        MidSidePair {
            mid: (left + right) * scale,
            side: (left - right) * scale,
        }
    }
}

/// Decodes a M/S signal back into stereo L/R representation.
///
/// L = (M + S) / √2, R = (M − S) / √2
#[derive(Debug, Clone, Copy, Default)]
pub struct MidSideDecode;

impl MidSideDecode {
    /// Decodes a mid/side pair back into stereo.
    #[inline]
    pub fn apply(&self, mid: f32, side: f32) -> StereoPair {
        let scale = std::f32::consts::FRAC_1_SQRT_2;
        StereoPair {
            left: (mid + side) * scale,
            right: (mid - side) * scale,
        }
    }
}

// ============================================================================
// Graph wrapper for SampleAndHold
// ============================================================================

/// Graph node wrapper for [`SampleAndHold`].
///
/// Expects two inputs: (input, trigger). Because the graph system is mono
/// (single `f32` per sample), the trigger is stored separately and updated
/// via [`SampleAndHoldNode::set_trigger`].
#[derive(Debug, Clone, Default)]
pub struct SampleAndHoldNode {
    inner: SampleAndHold,
    trigger: f32,
}

impl SampleAndHoldNode {
    /// Creates a new sample-and-hold node.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the trigger value for the next `process` call.
    pub fn set_trigger(&mut self, trigger: f32) {
        self.trigger = trigger;
    }

    /// Processes a sample using the currently stored trigger value.
    pub fn process(&mut self, input: f32) -> f32 {
        self.inner.process(input, self.trigger)
    }

    /// Returns the currently held value.
    pub fn held(&self) -> f32 {
        self.inner.held()
    }

    /// Resets both held value and trigger to zero.
    pub fn reset(&mut self) {
        self.inner.reset();
        self.trigger = 0.0;
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ---- PitchQuantizer ----

    #[test]
    fn pitch_quantizer_major_snaps_c_sharp_to_scale_note() {
        let q = PitchQuantizer {
            root: 0.0,
            scale: ScaleMode::Major,
            octave_wrap: true,
        };
        // C#4 = 1 semitone. In C major the adjacent notes are C (0) and D (2).
        let snapped = q.quantize_semitones(1.0);
        assert!(
            snapped == 0.0 || snapped == 2.0,
            "Expected C4 (0) or D4 (2), got {snapped}"
        );
    }

    #[test]
    fn pitch_quantizer_chromatic_is_identity() {
        let q = PitchQuantizer {
            root: 0.0,
            scale: ScaleMode::Chromatic,
            octave_wrap: true,
        };
        for i in 0..12 {
            let semi = i as f32;
            let snapped = q.quantize_semitones(semi);
            assert!(
                (snapped - semi).abs() < 1e-4,
                "Chromatic should not change {semi}, got {snapped}"
            );
        }
    }

    #[test]
    fn pitch_quantizer_hz_round_trips_via_c4() {
        let q = PitchQuantizer {
            root: 0.0,
            scale: ScaleMode::Chromatic,
            octave_wrap: true,
        };
        let c4 = 261.630_5_f32;
        let snapped = q.quantize_hz(c4);
        assert!(
            (snapped - c4).abs() < 0.5,
            "Expected ~{c4} Hz, got {snapped}"
        );
    }

    #[test]
    fn pitch_quantizer_major_snapped_note_is_in_scale() {
        let q = PitchQuantizer {
            root: 0.0,
            scale: ScaleMode::Major,
            octave_wrap: true,
        };
        let intervals: std::collections::HashSet<u8> =
            ScaleMode::Major.intervals().iter().copied().collect();

        for i in 0..24 {
            let snapped = q.quantize_semitones(i as f32);
            let semi_in_oct = snapped.rem_euclid(12.0).round() as u8;
            assert!(
                intervals.contains(&semi_in_oct),
                "Snapped {i} → {snapped}, semitone in octave {semi_in_oct} not in major scale"
            );
        }
    }

    // ---- SampleAndHold ----

    #[test]
    fn sample_and_hold_captures_on_trigger() {
        let mut sh = SampleAndHold::new();
        let out = sh.process(3.14, 1.0);
        assert!((out - 3.14).abs() < 1e-6);
    }

    #[test]
    fn sample_and_hold_holds_between_triggers() {
        let mut sh = SampleAndHold::new();
        sh.process(3.14, 1.0); // capture 3.14
        let out1 = sh.process(9.99, 0.0); // no trigger → hold
        let out2 = sh.process(0.0, 0.0); // still no trigger → hold
        assert!((out1 - 3.14).abs() < 1e-6, "Expected 3.14, got {out1}");
        assert!((out2 - 3.14).abs() < 1e-6, "Expected 3.14, got {out2}");
    }

    #[test]
    fn sample_and_hold_updates_on_second_trigger() {
        let mut sh = SampleAndHold::new();
        sh.process(1.0, 1.0);
        let out = sh.process(2.0, 1.0);
        assert!((out - 2.0).abs() < 1e-6, "Expected 2.0, got {out}");
    }

    // ---- SlewLimiter ----

    #[test]
    fn slew_limiter_limits_rise_rate() {
        let mut sl = SlewLimiter::new(0.1, 0.1);
        // Jump from 0 → 1: each sample should advance by at most 0.1.
        let out = sl.process(1.0);
        assert!(
            (out - 0.1).abs() < 1e-6,
            "Rise should be limited to 0.1, got {out}"
        );
    }

    #[test]
    fn slew_limiter_limits_fall_rate() {
        let mut sl = SlewLimiter::new(1.0, 0.2);
        sl.reset_to(1.0); // start at 1
        let out = sl.process(0.0); // fall toward 0, max 0.2 per sample
        assert!(
            (out - 0.8).abs() < 1e-6,
            "Fall should be limited to 0.2, got {out}"
        );
    }

    #[test]
    fn slew_limiter_reaches_target_eventually() {
        let mut sl = SlewLimiter::new(0.05, 0.05);
        for _ in 0..25 {
            sl.process(1.0);
        }
        // After 25 samples at rate 0.05 we should be at 1.0 (25 * 0.05 = 1.25 > 1).
        assert!(
            (sl.current - 1.0).abs() < 1e-5,
            "Should have reached target, got {}",
            sl.current
        );
    }

    #[test]
    fn slew_limiter_asymmetric_rates() {
        let mut sl = SlewLimiter::new(0.5, 0.1);
        // Rising fast
        let up = sl.process(1.0);
        assert!((up - 0.5).abs() < 1e-6, "Rise: expected 0.5, got {up}");
        sl.reset_to(1.0);
        // Falling slow
        let down = sl.process(0.0);
        assert!((down - 0.9).abs() < 1e-6, "Fall: expected 0.9, got {down}");
    }

    // ---- MidSide ----

    #[test]
    fn mid_side_encode_then_decode_round_trips() {
        let enc = MidSideEncode;
        let dec = MidSideDecode;

        let left = 0.6_f32;
        let right = 0.2_f32;

        let ms = enc.apply(left, right);
        let stereo = dec.apply(ms.mid, ms.side);

        assert!(
            (stereo.left - left).abs() < 1e-6,
            "Left round-trip failed: {} → {}",
            left,
            stereo.left
        );
        assert!(
            (stereo.right - right).abs() < 1e-6,
            "Right round-trip failed: {} → {}",
            right,
            stereo.right
        );
    }

    #[test]
    fn mid_side_mono_signal_has_zero_side() {
        let enc = MidSideEncode;
        // Mono: L == R → S should be 0.
        let ms = enc.apply(0.5, 0.5);
        assert!(
            ms.side.abs() < 1e-6,
            "Mono should produce zero side, got {}",
            ms.side
        );
    }

    #[test]
    fn mid_side_out_of_phase_has_zero_mid() {
        let enc = MidSideEncode;
        // Perfectly out-of-phase: L == -R → M should be 0.
        let ms = enc.apply(0.5, -0.5);
        assert!(
            ms.mid.abs() < 1e-6,
            "Out-of-phase should produce zero mid, got {}",
            ms.mid
        );
    }
}
