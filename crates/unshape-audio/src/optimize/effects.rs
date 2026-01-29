use crate::graph::{AudioContext, AudioNode};
use crate::primitive::{DelayLine, PhaseOsc};

use super::engine::{
    GraphFingerprint, MatchResult, NodeType, Pattern, PatternNode, PatternStructure,
};

/// Optimized tremolo effect (LFO modulating gain).
///
/// This is the monomorphized version that eliminates dyn dispatch.
pub struct TremoloOptimized {
    lfo: crate::primitive::PhaseOsc,
    phase_inc: f32,
    base: f32,
    scale: f32,
}

impl TremoloOptimized {
    /// Create a new optimized tremolo.
    pub fn new(rate: f32, depth: f32, sample_rate: f32) -> Self {
        Self {
            lfo: crate::primitive::PhaseOsc::new(),
            phase_inc: rate / sample_rate,
            base: 1.0 - depth * 0.5,
            scale: depth * 0.5,
        }
    }

    /// Create from match result (used by pattern).
    pub fn from_match(m: &MatchResult, sample_rate: f32) -> Self {
        // Extract rate from LFO node (pattern idx 0)
        let rate = m.get_param(0, "rate").unwrap_or(5.0);
        // Extract modulation depth from param wire
        let (base, scale) = m.get_modulation(0, 1).unwrap_or((0.5, 0.5));

        Self {
            lfo: crate::primitive::PhaseOsc::new(),
            phase_inc: rate / sample_rate,
            base,
            scale,
        }
    }
}

impl crate::graph::AudioNode for TremoloOptimized {
    #[inline]
    fn process(&mut self, input: f32, _ctx: &crate::graph::AudioContext) -> f32 {
        let lfo_out = self.lfo.sine();
        self.lfo.advance(self.phase_inc);
        let gain = self.base + lfo_out * self.scale;
        input * gain
    }

    fn reset(&mut self) {
        self.lfo.reset();
    }
}

/// Optimized flanger effect (LFO modulating delay time).
///
/// Flanger = LFO → delay time, with feedback and dry/wet mix.
pub struct FlangerOptimized {
    lfo: crate::primitive::PhaseOsc,
    delay: crate::primitive::DelayLine<true>,
    phase_inc: f32,
    base_delay: f32,
    depth: f32,
    feedback: f32,
    // Pre-computed mix factors (constant folding)
    wet_mix: f32,
    dry_mix: f32,
}

impl FlangerOptimized {
    /// Create a new optimized flanger.
    pub fn new(
        rate: f32,
        base_delay_ms: f32,
        depth_ms: f32,
        feedback: f32,
        mix: f32,
        sample_rate: f32,
    ) -> Self {
        let base_delay = base_delay_ms * sample_rate / 1000.0;
        let depth = depth_ms * sample_rate / 1000.0;
        let max_delay = ((base_delay_ms + depth_ms * 2.0) * sample_rate / 1000.0) as usize + 1;

        Self {
            lfo: crate::primitive::PhaseOsc::new(),
            delay: crate::primitive::DelayLine::new(max_delay),
            phase_inc: rate / sample_rate,
            base_delay,
            depth,
            feedback,
            wet_mix: mix,
            dry_mix: 1.0 - mix,
        }
    }

    /// Create from match result.
    pub fn from_match(m: &MatchResult, sample_rate: f32) -> Self {
        let rate = m.get_param(0, "rate").unwrap_or(0.3);
        let (base_delay, depth) = m.get_modulation(0, 1).unwrap_or((220.0, 130.0)); // ~5ms, ~3ms at 44.1kHz
        let feedback = m.get_param(1, "feedback").unwrap_or(0.7);
        let mix = m.get_param(2, "mix").unwrap_or(0.5);

        let max_delay = (base_delay + depth * 2.0) as usize + 1;

        Self {
            lfo: crate::primitive::PhaseOsc::new(),
            delay: crate::primitive::DelayLine::new(max_delay),
            phase_inc: rate / sample_rate,
            base_delay,
            depth,
            feedback,
            wet_mix: mix,
            dry_mix: 1.0 - mix,
        }
    }
}

impl crate::graph::AudioNode for FlangerOptimized {
    #[inline]
    fn process(&mut self, input: f32, _ctx: &crate::graph::AudioContext) -> f32 {
        let lfo_out = self.lfo.sine();
        self.lfo.advance(self.phase_inc);

        let delay_time = self.base_delay + lfo_out * self.depth;
        let delayed = self.delay.read_interp(delay_time);

        // Write with feedback
        self.delay.write(input + delayed * self.feedback);

        // Mix dry and wet (pre-computed factors)
        input * self.dry_mix + delayed * self.wet_mix
    }

    fn reset(&mut self) {
        self.lfo.reset();
        self.delay.clear();
    }
}

/// Optimized chorus effect (LFO modulating delay time with mix).
///
/// Chorus = LFO → delay time, mixed with dry signal.
pub struct ChorusOptimized {
    lfo: crate::primitive::PhaseOsc,
    delay: crate::primitive::DelayLine<true>,
    phase_inc: f32,
    base_delay: f32,
    depth: f32,
    // Pre-computed mix factors (constant folding)
    wet_mix: f32,
    dry_mix: f32,
}

impl ChorusOptimized {
    /// Create a new optimized chorus.
    pub fn new(rate: f32, base_delay_ms: f32, depth_ms: f32, mix: f32, sample_rate: f32) -> Self {
        let base_delay = base_delay_ms * sample_rate / 1000.0;
        let depth = depth_ms * sample_rate / 1000.0;
        let max_delay = ((base_delay_ms + depth_ms * 2.0) * sample_rate / 1000.0) as usize + 1;

        Self {
            lfo: crate::primitive::PhaseOsc::new(),
            delay: crate::primitive::DelayLine::new(max_delay),
            phase_inc: rate / sample_rate,
            base_delay,
            depth,
            wet_mix: mix,
            dry_mix: 1.0 - mix,
        }
    }

    /// Create from match result.
    pub fn from_match(m: &MatchResult, sample_rate: f32) -> Self {
        let rate = m.get_param(0, "rate").unwrap_or(0.8);
        let (base_delay, depth) = m.get_modulation(0, 1).unwrap_or((880.0, 220.0)); // ~20ms, ~5ms at 44.1kHz
        let mix = m.get_param(2, "mix").unwrap_or(0.5);

        let max_delay = (base_delay + depth * 2.0) as usize + 1;

        Self {
            lfo: crate::primitive::PhaseOsc::new(),
            delay: crate::primitive::DelayLine::new(max_delay),
            phase_inc: rate / sample_rate,
            base_delay,
            depth,
            wet_mix: mix,
            dry_mix: 1.0 - mix,
        }
    }
}

impl crate::graph::AudioNode for ChorusOptimized {
    #[inline]
    fn process(&mut self, input: f32, _ctx: &crate::graph::AudioContext) -> f32 {
        let lfo_out = self.lfo.sine();
        self.lfo.advance(self.phase_inc);

        let delay_time = self.base_delay + lfo_out * self.depth;
        self.delay.write(input);
        let wet = self.delay.read_interp(delay_time);

        // Mix dry and wet (pre-computed factors)
        input * self.dry_mix + wet * self.wet_mix
    }

    fn reset(&mut self) {
        self.lfo.reset();
        self.delay.clear();
    }
}

/// Returns the default set of effect patterns.
pub fn default_patterns() -> Vec<Pattern> {
    vec![tremolo_pattern(), flanger_pattern(), chorus_pattern()]
}

/// Pattern for tremolo: LFO modulating an affine (gain) node.
fn tremolo_pattern() -> Pattern {
    Pattern {
        name: "tremolo",
        required: fingerprint!(Lfo: 1, Affine: 1),
        structure: PatternStructure {
            nodes: vec![
                PatternNode {
                    node_type: NodeType::Lfo,
                    constraints: vec![],
                },
                PatternNode {
                    node_type: NodeType::Affine,
                    constraints: vec![],
                },
            ],
            audio_wires: vec![],               // LFO doesn't send audio to Affine
            param_wires: vec![(0, 1, "gain")], // LFO modulates Affine's gain param
            external_inputs: vec![1],          // External audio enters at Affine
            external_outputs: vec![1],         // Audio leaves from Affine
        },
        build: |m| Box::new(TremoloOptimized::from_match(m, 44100.0)),
        priority: 0,
    }
}

/// Pattern for flanger: LFO modulating delay time (no mixer).
fn flanger_pattern() -> Pattern {
    Pattern {
        name: "flanger",
        required: fingerprint!(Lfo: 1, Delay: 1),
        structure: PatternStructure {
            nodes: vec![
                PatternNode {
                    node_type: NodeType::Lfo,
                    constraints: vec![],
                },
                PatternNode {
                    node_type: NodeType::Delay,
                    constraints: vec![],
                },
            ],
            audio_wires: vec![],               // LFO doesn't send audio to Delay
            param_wires: vec![(0, 1, "time")], // LFO modulates Delay's time param
            external_inputs: vec![1],          // External audio enters at Delay
            external_outputs: vec![1],         // Audio leaves from Delay
        },
        build: |m| Box::new(FlangerOptimized::from_match(m, 44100.0)),
        priority: 0,
    }
}

/// Pattern for chorus: LFO modulating delay time with mixer.
fn chorus_pattern() -> Pattern {
    Pattern {
        name: "chorus",
        required: fingerprint!(Lfo: 1, Delay: 1, Mix: 1),
        structure: PatternStructure {
            nodes: vec![
                PatternNode {
                    node_type: NodeType::Lfo,
                    constraints: vec![],
                },
                PatternNode {
                    node_type: NodeType::Delay,
                    constraints: vec![],
                },
                PatternNode {
                    node_type: NodeType::Mix,
                    constraints: vec![],
                },
            ],
            audio_wires: vec![(1, 2)],         // Delay → Mixer
            param_wires: vec![(0, 1, "time")], // LFO modulates Delay's time param
            external_inputs: vec![1],          // External audio enters at Delay
            external_outputs: vec![2],         // Audio leaves from Mixer
        },
        build: |m| Box::new(ChorusOptimized::from_match(m, 44100.0)),
        priority: 10, // Higher priority than flanger (more specific)
    }
}
