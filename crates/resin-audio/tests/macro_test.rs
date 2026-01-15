//! Integration test for the graph_effect! proc macro.

use rhizome_resin_audio::graph::{AudioContext, AudioNode};
use rhizome_resin_audio::primitive::{GainNode, LfoNode};
use rhizome_resin_audio_macros::graph_effect;

// Define a simple tremolo effect using the macro
graph_effect! {
    name: StaticTremolo,
    nodes: {
        lfo: LfoNode::with_freq(5.0, 44100.0),
        gain: GainNode::new(1.0),
    },
    audio: [input -> gain],
    modulation: [lfo -> gain.gain(base: 0.5, scale: 0.5)],
    output: gain,
}

#[test]
fn test_static_tremolo_compiles() {
    let _tremolo = StaticTremolo::new();
}

#[test]
fn test_static_tremolo_processes() {
    let ctx = AudioContext::new(44100.0);
    let mut tremolo = StaticTremolo::new();

    // Process some samples
    let mut samples = Vec::new();
    for _ in 0..44100 {
        samples.push(tremolo.process(1.0, &ctx));
    }

    // Should have modulated output (not all the same)
    let first = samples[0];
    let has_variation = samples.iter().any(|&s| (s - first).abs() > 0.01);
    assert!(has_variation, "tremolo should modulate the signal");

    // Output should be in reasonable range (base 0.5 +/- 0.5 scale)
    let min = samples.iter().cloned().fold(f32::MAX, f32::min);
    let max = samples.iter().cloned().fold(f32::MIN, f32::max);

    assert!(min >= 0.0, "min {} should be >= 0", min);
    assert!(max <= 1.0, "max {} should be <= 1", max);
}

#[test]
fn test_static_tremolo_reset() {
    let ctx = AudioContext::new(44100.0);
    let mut tremolo = StaticTremolo::new();

    // Process some samples
    for _ in 0..1000 {
        tremolo.process(1.0, &ctx);
    }

    // Reset should not panic
    tremolo.reset();
}
