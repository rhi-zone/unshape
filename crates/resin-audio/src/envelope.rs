//! Envelope generators for audio synthesis.
//!
//! Provides ADSR envelopes, LFOs, and other modulation sources.

/// ADSR envelope state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdsrState {
    /// Not active.
    Idle,
    /// Rising to peak.
    Attack,
    /// Falling to sustain level.
    Decay,
    /// Holding at sustain level.
    Sustain,
    /// Falling to zero after note off.
    Release,
}

/// ADSR (Attack, Decay, Sustain, Release) envelope generator.
#[derive(Debug, Clone)]
pub struct Adsr {
    /// Attack time in seconds.
    pub attack: f32,
    /// Decay time in seconds.
    pub decay: f32,
    /// Sustain level (0.0 to 1.0).
    pub sustain: f32,
    /// Release time in seconds.
    pub release: f32,
    /// Current state.
    state: AdsrState,
    /// Current envelope value.
    value: f32,
    /// Value when release started (for proper release curve).
    release_start_value: f32,
    /// Time in current state.
    time: f32,
}

impl Default for Adsr {
    fn default() -> Self {
        Self {
            attack: 0.01,
            decay: 0.1,
            sustain: 0.7,
            release: 0.3,
            state: AdsrState::Idle,
            value: 0.0,
            release_start_value: 0.0,
            time: 0.0,
        }
    }
}

impl Adsr {
    /// Creates a new ADSR envelope with default parameters.
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates an ADSR with specified parameters.
    pub fn with_params(attack: f32, decay: f32, sustain: f32, release: f32) -> Self {
        Self {
            attack,
            decay,
            sustain: sustain.clamp(0.0, 1.0),
            release,
            ..Default::default()
        }
    }

    /// Triggers the envelope (note on).
    pub fn trigger(&mut self) {
        self.state = AdsrState::Attack;
        self.time = 0.0;
    }

    /// Releases the envelope (note off).
    pub fn release(&mut self) {
        if self.state != AdsrState::Idle {
            self.state = AdsrState::Release;
            self.release_start_value = self.value;
            self.time = 0.0;
        }
    }

    /// Returns current state.
    pub fn state(&self) -> AdsrState {
        self.state
    }

    /// Returns current envelope value.
    pub fn value(&self) -> f32 {
        self.value
    }

    /// Returns true if the envelope is active (not idle).
    pub fn is_active(&self) -> bool {
        self.state != AdsrState::Idle
    }

    /// Processes one sample and returns the envelope value.
    pub fn process(&mut self, dt: f32) -> f32 {
        self.time += dt;

        match self.state {
            AdsrState::Idle => {
                self.value = 0.0;
            }
            AdsrState::Attack => {
                if self.attack <= 0.0 {
                    self.value = 1.0;
                    self.state = AdsrState::Decay;
                    self.time = 0.0;
                } else {
                    self.value = (self.time / self.attack).min(1.0);
                    if self.time >= self.attack {
                        self.state = AdsrState::Decay;
                        self.time = 0.0;
                    }
                }
            }
            AdsrState::Decay => {
                if self.decay <= 0.0 {
                    self.value = self.sustain;
                    self.state = AdsrState::Sustain;
                } else {
                    let t = (self.time / self.decay).min(1.0);
                    self.value = 1.0 + (self.sustain - 1.0) * t;
                    if self.time >= self.decay {
                        self.state = AdsrState::Sustain;
                    }
                }
            }
            AdsrState::Sustain => {
                self.value = self.sustain;
            }
            AdsrState::Release => {
                if self.release <= 0.0 {
                    self.value = 0.0;
                    self.state = AdsrState::Idle;
                } else {
                    let t = (self.time / self.release).min(1.0);
                    self.value = self.release_start_value * (1.0 - t);
                    if self.time >= self.release {
                        self.state = AdsrState::Idle;
                        self.value = 0.0;
                    }
                }
            }
        }

        self.value
    }

    /// Resets the envelope to idle state.
    pub fn reset(&mut self) {
        self.state = AdsrState::Idle;
        self.value = 0.0;
        self.time = 0.0;
    }
}

/// LFO (Low Frequency Oscillator) waveform type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LfoWaveform {
    /// Smooth sine wave.
    #[default]
    Sine,
    /// Linear triangle wave.
    Triangle,
    /// Sawtooth wave.
    Saw,
    /// Square wave.
    Square,
    /// Sample-and-hold random.
    Random,
}

/// Low Frequency Oscillator for modulation.
#[derive(Debug, Clone)]
pub struct Lfo {
    /// Frequency in Hz.
    pub frequency: f32,
    /// Amplitude (0.0 to 1.0).
    pub amplitude: f32,
    /// DC offset (-1.0 to 1.0).
    pub offset: f32,
    /// Waveform type.
    pub waveform: LfoWaveform,
    /// Current phase (0.0 to 1.0).
    phase: f32,
    /// Last random value (for S&H random).
    random_value: f32,
    /// Random state.
    random_state: u32,
}

impl Default for Lfo {
    fn default() -> Self {
        Self {
            frequency: 1.0,
            amplitude: 1.0,
            offset: 0.0,
            waveform: LfoWaveform::Sine,
            phase: 0.0,
            random_value: 0.0,
            random_state: 12345,
        }
    }
}

impl Lfo {
    /// Creates a new LFO with default parameters.
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates an LFO with specified frequency.
    pub fn with_frequency(frequency: f32) -> Self {
        Self {
            frequency,
            ..Default::default()
        }
    }

    /// Returns the current phase (0.0 to 1.0).
    pub fn phase(&self) -> f32 {
        self.phase
    }

    /// Resets the phase to zero.
    pub fn reset(&mut self) {
        self.phase = 0.0;
    }

    /// Processes one sample and returns the LFO value.
    pub fn process(&mut self, dt: f32) -> f32 {
        self.phase += self.frequency * dt;

        // Wrap phase
        while self.phase >= 1.0 {
            self.phase -= 1.0;
            // Update random value on phase wrap
            if self.waveform == LfoWaveform::Random {
                self.random_state = self
                    .random_state
                    .wrapping_mul(1103515245)
                    .wrapping_add(12345);
                self.random_value = (self.random_state as f32 / u32::MAX as f32) * 2.0 - 1.0;
            }
        }

        let raw = match self.waveform {
            LfoWaveform::Sine => (self.phase * std::f32::consts::TAU).sin(),
            LfoWaveform::Triangle => {
                let t = self.phase * 4.0;
                if t < 1.0 {
                    t
                } else if t < 3.0 {
                    2.0 - t
                } else {
                    t - 4.0
                }
            }
            LfoWaveform::Saw => self.phase * 2.0 - 1.0,
            LfoWaveform::Square => {
                if self.phase < 0.5 {
                    1.0
                } else {
                    -1.0
                }
            }
            LfoWaveform::Random => self.random_value,
        };

        raw * self.amplitude + self.offset
    }

    /// Samples the LFO at a specific phase (0.0 to 1.0) without advancing.
    pub fn sample_at(&self, phase: f32) -> f32 {
        let phase = phase.fract();
        let raw = match self.waveform {
            LfoWaveform::Sine => (phase * std::f32::consts::TAU).sin(),
            LfoWaveform::Triangle => {
                let t = phase * 4.0;
                if t < 1.0 {
                    t
                } else if t < 3.0 {
                    2.0 - t
                } else {
                    t - 4.0
                }
            }
            LfoWaveform::Saw => phase * 2.0 - 1.0,
            LfoWaveform::Square => {
                if phase < 0.5 {
                    1.0
                } else {
                    -1.0
                }
            }
            LfoWaveform::Random => {
                // Can't sample random at arbitrary phase
                0.0
            }
        };

        raw * self.amplitude + self.offset
    }
}

/// Simple AR (Attack-Release) envelope.
#[derive(Debug, Clone)]
pub struct Ar {
    /// Attack time in seconds.
    pub attack: f32,
    /// Release time in seconds.
    pub release: f32,
    /// Current value.
    value: f32,
    /// Time accumulator.
    time: f32,
    /// Whether triggered.
    triggered: bool,
}

impl Default for Ar {
    fn default() -> Self {
        Self {
            attack: 0.01,
            release: 0.1,
            value: 0.0,
            time: 0.0,
            triggered: false,
        }
    }
}

impl Ar {
    /// Creates a new AR envelope.
    pub fn new(attack: f32, release: f32) -> Self {
        Self {
            attack,
            release,
            ..Default::default()
        }
    }

    /// Triggers the envelope.
    pub fn trigger(&mut self) {
        self.triggered = true;
        self.time = 0.0;
    }

    /// Returns the current value.
    pub fn value(&self) -> f32 {
        self.value
    }

    /// Processes one sample.
    pub fn process(&mut self, dt: f32) -> f32 {
        if !self.triggered {
            return 0.0;
        }

        self.time += dt;
        let total = self.attack + self.release;

        if self.time >= total {
            self.triggered = false;
            self.value = 0.0;
        } else if self.time < self.attack {
            self.value = if self.attack > 0.0 {
                self.time / self.attack
            } else {
                1.0
            };
        } else {
            let release_time = self.time - self.attack;
            self.value = if self.release > 0.0 {
                1.0 - (release_time / self.release)
            } else {
                0.0
            };
        }

        self.value
    }

    /// Resets the envelope.
    pub fn reset(&mut self) {
        self.triggered = false;
        self.value = 0.0;
        self.time = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adsr_trigger() {
        let mut env = Adsr::with_params(0.1, 0.1, 0.5, 0.1);
        assert_eq!(env.state(), AdsrState::Idle);

        env.trigger();
        assert_eq!(env.state(), AdsrState::Attack);
    }

    #[test]
    fn test_adsr_attack() {
        let mut env = Adsr::with_params(0.1, 0.1, 0.5, 0.1);
        env.trigger();

        // Process through attack
        for _ in 0..10 {
            env.process(0.01);
        }

        // Should be at or near peak
        assert!(env.value() > 0.9);
    }

    #[test]
    fn test_adsr_sustain() {
        let mut env = Adsr::with_params(0.01, 0.01, 0.7, 0.1);
        env.trigger();

        // Process through attack and decay
        for _ in 0..100 {
            env.process(0.001);
        }

        // Should be at sustain level
        assert!((env.value() - 0.7).abs() < 0.05);
        assert_eq!(env.state(), AdsrState::Sustain);
    }

    #[test]
    fn test_adsr_release() {
        let mut env = Adsr::with_params(0.01, 0.01, 0.7, 0.1);
        env.trigger();

        // Process to sustain
        for _ in 0..100 {
            env.process(0.001);
        }

        env.release();
        assert_eq!(env.state(), AdsrState::Release);

        // Process through release
        for _ in 0..200 {
            env.process(0.001);
        }

        assert_eq!(env.state(), AdsrState::Idle);
        assert!(env.value() < 0.01);
    }

    #[test]
    fn test_lfo_sine() {
        let mut lfo = Lfo::with_frequency(1.0);

        // At phase 0, sine should be 0
        assert!(lfo.process(0.0).abs() < 0.01);

        // At phase 0.25, sine should be 1
        lfo.phase = 0.25;
        let v = lfo.sample_at(0.25);
        assert!((v - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_lfo_waveforms() {
        let mut lfo = Lfo::new();

        // Triangle
        lfo.waveform = LfoWaveform::Triangle;
        lfo.phase = 0.0;
        assert!(lfo.sample_at(0.0).abs() < 0.01);
        assert!((lfo.sample_at(0.25) - 1.0).abs() < 0.01);

        // Square
        lfo.waveform = LfoWaveform::Square;
        assert!((lfo.sample_at(0.0) - 1.0).abs() < 0.01);
        assert!((lfo.sample_at(0.5) - (-1.0)).abs() < 0.01);

        // Saw
        lfo.waveform = LfoWaveform::Saw;
        assert!((lfo.sample_at(0.0) - (-1.0)).abs() < 0.01);
        assert!((lfo.sample_at(0.5) - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_lfo_amplitude_offset() {
        let mut lfo = Lfo::new();
        lfo.amplitude = 0.5;
        lfo.offset = 0.5;

        // Should range from 0 to 1 instead of -1 to 1
        let min = lfo.sample_at(0.75); // sine minimum
        let max = lfo.sample_at(0.25); // sine maximum

        assert!(min >= -0.01);
        assert!(max <= 1.01);
    }

    #[test]
    fn test_ar_envelope() {
        let mut env = Ar::new(0.1, 0.1);

        env.trigger();

        // Process through attack
        for _ in 0..10 {
            env.process(0.01);
        }
        assert!(env.value() > 0.9);

        // Process through release
        for _ in 0..20 {
            env.process(0.01);
        }
        assert!(env.value() < 0.1);
    }
}

// ============================================================================
// Invariant tests - envelope smoothness and timing
// ============================================================================

#[cfg(all(test, feature = "invariant-tests"))]
mod invariant_tests {
    use super::*;

    const SAMPLE_RATE: f32 = 44100.0;
    const DT: f32 = 1.0 / SAMPLE_RATE;

    #[test]
    fn test_adsr_range_bounds() {
        // ADSR output should always be in [0, 1]
        let mut env = Adsr::with_params(0.01, 0.1, 0.7, 0.2);
        env.trigger();

        for _ in 0..10000 {
            let value = env.process(DT);
            assert!(value >= 0.0 && value <= 1.0, "ADSR out of range: {}", value);
        }

        env.release();
        for _ in 0..10000 {
            let value = env.process(DT);
            assert!(
                value >= 0.0 && value <= 1.0,
                "ADSR out of range during release: {}",
                value
            );
        }
    }

    #[test]
    fn test_adsr_smoothness() {
        // ADSR should be smooth (no large discontinuities)
        let mut env = Adsr::with_params(0.01, 0.05, 0.7, 0.1);
        env.trigger();

        let mut prev = env.process(DT);
        for _ in 0..5000 {
            let curr = env.process(DT);
            let delta = (curr - prev).abs();
            // Max change per sample should be bounded
            assert!(
                delta < 0.01,
                "ADSR discontinuity: prev={}, curr={}, delta={}",
                prev,
                curr,
                delta
            );
            prev = curr;
        }

        env.release();
        for _ in 0..5000 {
            let curr = env.process(DT);
            let delta = (curr - prev).abs();
            assert!(
                delta < 0.01,
                "ADSR release discontinuity: prev={}, curr={}, delta={}",
                prev,
                curr,
                delta
            );
            prev = curr;
        }
    }

    #[test]
    fn test_adsr_attack_reaches_peak() {
        // Attack phase should reach 1.0
        let attack_time = 0.1;
        let mut env = Adsr::with_params(attack_time, 0.1, 0.5, 0.1);
        env.trigger();

        // Process through attack
        let samples = (attack_time * SAMPLE_RATE * 1.5) as usize;
        let mut max_value = 0.0f32;
        for _ in 0..samples {
            let value = env.process(DT);
            max_value = max_value.max(value);
        }

        assert!(
            max_value > 0.95,
            "ADSR should reach peak during attack, got max={}",
            max_value
        );
    }

    #[test]
    fn test_adsr_sustain_level() {
        // After attack+decay, should hold at sustain level
        let sustain = 0.6;
        let mut env = Adsr::with_params(0.01, 0.05, sustain, 0.1);
        env.trigger();

        // Process through attack and decay
        for _ in 0..5000 {
            env.process(DT);
        }

        // Should be at sustain level
        let value = env.value();
        assert!(
            (value - sustain).abs() < 0.05,
            "ADSR should sustain at {}, got {}",
            sustain,
            value
        );
    }

    #[test]
    fn test_adsr_release_to_zero() {
        // After release, should decay to zero
        let mut env = Adsr::with_params(0.01, 0.01, 0.5, 0.1);
        env.trigger();

        // Process to sustain
        for _ in 0..2000 {
            env.process(DT);
        }

        env.release();

        // Process through release
        for _ in 0..10000 {
            env.process(DT);
        }

        let value = env.value();
        assert!(
            value < 0.01,
            "ADSR should release to near zero, got {}",
            value
        );
    }

    #[test]
    fn test_lfo_range() {
        // LFO with amplitude 1.0 and offset 0.0 should be in [-1, 1]
        let lfo = Lfo::new();

        for i in 0..1000 {
            let phase = i as f32 / 1000.0;
            let value = lfo.sample_at(phase);
            assert!(
                value >= -1.0 && value <= 1.0,
                "LFO out of range at phase {}: {}",
                phase,
                value
            );
        }
    }

    #[test]
    fn test_lfo_waveform_ranges() {
        // All LFO waveforms should stay in range
        let waveforms = [
            LfoWaveform::Sine,
            LfoWaveform::Triangle,
            LfoWaveform::Square,
            LfoWaveform::Saw,
            LfoWaveform::Random,
        ];

        for waveform in waveforms {
            let mut lfo = Lfo::new();
            lfo.waveform = waveform;

            for i in 0..1000 {
                let phase = i as f32 / 1000.0;
                let value = lfo.sample_at(phase);
                assert!(
                    value >= -1.0 && value <= 1.0,
                    "{:?} LFO out of range at phase {}: {}",
                    waveform,
                    phase,
                    value
                );
            }
        }
    }

    #[test]
    fn test_lfo_amplitude_scaling() {
        // Amplitude should scale the output
        let mut lfo = Lfo::new();
        lfo.amplitude = 0.5;

        let mut max_value = 0.0f32;
        for i in 0..1000 {
            let phase = i as f32 / 1000.0;
            max_value = max_value.max(lfo.sample_at(phase).abs());
        }

        assert!(
            (max_value - 0.5).abs() < 0.01,
            "LFO max should be ~0.5 with amplitude 0.5, got {}",
            max_value
        );
    }

    #[test]
    fn test_lfo_offset() {
        // Offset should shift the center
        let mut lfo = Lfo::new();
        lfo.offset = 0.5;
        lfo.amplitude = 0.5;

        let mut min_value = f32::INFINITY;
        let mut max_value = f32::NEG_INFINITY;

        for i in 0..1000 {
            let phase = i as f32 / 1000.0;
            let value = lfo.sample_at(phase);
            min_value = min_value.min(value);
            max_value = max_value.max(value);
        }

        // With offset 0.5 and amplitude 0.5, should range from 0 to 1
        assert!(
            min_value >= -0.01 && max_value <= 1.01,
            "LFO with offset 0.5, amp 0.5 should be in [0, 1], got [{}, {}]",
            min_value,
            max_value
        );
    }

    #[test]
    fn test_ar_range_bounds() {
        // AR envelope should be in [0, 1]
        let mut env = Ar::new(0.1, 0.2);
        env.trigger();

        for _ in 0..20000 {
            let value = env.process(DT);
            assert!(value >= 0.0 && value <= 1.0, "AR out of range: {}", value);
        }
    }

    #[test]
    fn test_ar_smoothness() {
        // AR should be smooth
        let mut env = Ar::new(0.05, 0.1);
        env.trigger();

        let mut prev = env.process(DT);
        for _ in 0..10000 {
            let curr = env.process(DT);
            let delta = (curr - prev).abs();
            assert!(
                delta < 0.01,
                "AR discontinuity: prev={}, curr={}, delta={}",
                prev,
                curr,
                delta
            );
            prev = curr;
        }
    }
}
