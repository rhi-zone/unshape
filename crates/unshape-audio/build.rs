//! Build script for unshape-audio.
//!
//! Generates optimized effect structs for benchmarking Tier 4 codegen.

fn main() {
    #[cfg(feature = "codegen-bench")]
    generate_benchmark_effects();
}

#[cfg(feature = "codegen-bench")]
fn generate_benchmark_effects() {
    use unshape_audio_codegen::{
        SerialAudioGraph, SerialAudioNode, SerialParamWire, generate_effect, generate_header,
    };
    use std::env;
    use std::fs;
    use std::path::Path;

    let out_dir = env::var("OUT_DIR").unwrap();
    let dest_path = Path::new(&out_dir).join("codegen_bench.rs");

    // Start with the common header (imports)
    let mut code = generate_header();

    // Tremolo: LFO (5Hz) modulating gain
    // base=0.5, scale=0.5 means gain oscillates from 0.0 to 1.0
    let tremolo_graph = SerialAudioGraph {
        nodes: vec![
            SerialAudioNode::Lfo { rate: 5.0 },
            SerialAudioNode::Gain { gain: 1.0 },
        ],
        audio_wires: vec![],
        param_wires: vec![SerialParamWire {
            from: 0,
            to: 1,
            param: 0,
            base: 0.5,
            scale: 0.5,
        }],
        input_node: Some(1),
        output_node: Some(1),
    };
    code.push_str(&generate_effect(&tremolo_graph, "GeneratedTremolo"));
    code.push_str("\n\n");

    // Chorus: LFO (0.5Hz) modulating delay time, mixed with dry
    // Match ChorusOptimized: rate=0.5, base=7ms, depth=3ms, mix=0.5
    let sample_rate = 44100.0_f32;
    let base_delay = 7.0 * sample_rate / 1000.0;
    let depth = 3.0 * sample_rate / 1000.0;
    let max_delay = ((7.0 + 3.0 * 2.0) * sample_rate / 1000.0) as usize + 1;

    let chorus_graph = SerialAudioGraph {
        nodes: vec![
            SerialAudioNode::Lfo { rate: 0.5 },
            SerialAudioNode::Delay {
                max_samples: max_delay,
                feedback: 0.0,
                mix: 0.5,
            },
        ],
        audio_wires: vec![],
        param_wires: vec![SerialParamWire {
            from: 0,
            to: 1,
            param: 0,
            base: base_delay,
            scale: depth,
        }],
        input_node: Some(1),
        output_node: Some(1),
    };
    code.push_str(&generate_effect(&chorus_graph, "GeneratedChorus"));
    code.push_str("\n\n");

    // Flanger: LFO (0.3Hz) modulating delay time with feedback
    // Match FlangerOptimized: rate=0.3, base=3ms, depth=2ms, feedback=0.7, mix=0.5
    let base_delay = 3.0 * sample_rate / 1000.0;
    let depth = 2.0 * sample_rate / 1000.0;
    let max_delay = ((3.0 + 2.0 * 2.0) * sample_rate / 1000.0) as usize + 1;

    let flanger_graph = SerialAudioGraph {
        nodes: vec![
            SerialAudioNode::Lfo { rate: 0.3 },
            SerialAudioNode::Delay {
                max_samples: max_delay,
                feedback: 0.7,
                mix: 0.5,
            },
        ],
        audio_wires: vec![],
        param_wires: vec![SerialParamWire {
            from: 0,
            to: 1,
            param: 0,
            base: base_delay,
            scale: depth,
        }],
        input_node: Some(1),
        output_node: Some(1),
    };
    code.push_str(&generate_effect(&flanger_graph, "GeneratedFlanger"));

    fs::write(&dest_path, code).unwrap();
    println!("cargo:rerun-if-changed=build.rs");
}
