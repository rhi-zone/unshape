//! Benchmarks for audio effects.
//!
//! Run with: cargo bench -p rhizome-resin-audio

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use rhizome_resin_audio::effects::{
    Bitcrusher, Compressor, Distortion, Limiter, NoiseGate, Reverb, chorus, chorus_graph, flanger,
    flanger_graph, phaser, tremolo, tremolo_graph,
};
use rhizome_resin_audio::graph::AudioContext;

const SAMPLE_RATE: f32 = 44100.0;
const ONE_SECOND: usize = 44100;

/// Generate a test signal (sine wave with some harmonics).
fn test_signal(samples: usize) -> Vec<f32> {
    (0..samples)
        .map(|i| {
            let t = i as f32 / SAMPLE_RATE;
            let fundamental = (t * 440.0 * std::f32::consts::TAU).sin();
            let harmonic2 = (t * 880.0 * std::f32::consts::TAU).sin() * 0.5;
            let harmonic3 = (t * 1320.0 * std::f32::consts::TAU).sin() * 0.25;
            (fundamental + harmonic2 + harmonic3) * 0.3
        })
        .collect()
}

fn bench_chorus(c: &mut Criterion) {
    let signal = test_signal(ONE_SECOND);

    c.bench_function("chorus_1sec", |b| {
        let mut effect = chorus(SAMPLE_RATE);
        b.iter(|| {
            for &sample in &signal {
                black_box(effect.process(sample));
            }
        });
    });
}

fn bench_flanger(c: &mut Criterion) {
    let signal = test_signal(ONE_SECOND);

    c.bench_function("flanger_1sec", |b| {
        let mut effect = flanger(SAMPLE_RATE);
        b.iter(|| {
            for &sample in &signal {
                black_box(effect.process(sample));
            }
        });
    });
}

fn bench_phaser(c: &mut Criterion) {
    let signal = test_signal(ONE_SECOND);

    c.bench_function("phaser_1sec", |b| {
        let mut effect = phaser(SAMPLE_RATE);
        b.iter(|| {
            for &sample in &signal {
                black_box(effect.process(sample));
            }
        });
    });
}

fn bench_tremolo(c: &mut Criterion) {
    let signal = test_signal(ONE_SECOND);

    c.bench_function("tremolo_1sec", |b| {
        let mut effect = tremolo(SAMPLE_RATE, 5.0, 0.5);
        b.iter(|| {
            for &sample in &signal {
                black_box(effect.process(sample));
            }
        });
    });
}

fn bench_reverb(c: &mut Criterion) {
    let signal = test_signal(ONE_SECOND);

    c.bench_function("reverb_1sec", |b| {
        let mut reverb = Reverb::new(SAMPLE_RATE);
        b.iter(|| {
            for &sample in &signal {
                black_box(reverb.process(sample));
            }
        });
    });
}

fn bench_distortion(c: &mut Criterion) {
    let signal = test_signal(ONE_SECOND);

    c.bench_function("distortion_1sec", |b| {
        let mut distortion = Distortion::new(SAMPLE_RATE);
        b.iter(|| {
            for &sample in &signal {
                black_box(distortion.process(sample));
            }
        });
    });
}

fn bench_compressor(c: &mut Criterion) {
    let signal = test_signal(ONE_SECOND);

    c.bench_function("compressor_1sec", |b| {
        let mut compressor = Compressor::new(SAMPLE_RATE);
        b.iter(|| {
            for &sample in &signal {
                black_box(compressor.process(sample));
            }
        });
    });
}

fn bench_limiter(c: &mut Criterion) {
    let signal = test_signal(ONE_SECOND);

    c.bench_function("limiter_1sec", |b| {
        let mut limiter = Limiter::new(SAMPLE_RATE);
        b.iter(|| {
            for &sample in &signal {
                black_box(limiter.process(sample));
            }
        });
    });
}

fn bench_noise_gate(c: &mut Criterion) {
    let signal = test_signal(ONE_SECOND);

    c.bench_function("noise_gate_1sec", |b| {
        let mut gate = NoiseGate::new(SAMPLE_RATE);
        b.iter(|| {
            for &sample in &signal {
                black_box(gate.process(sample));
            }
        });
    });
}

fn bench_bitcrusher(c: &mut Criterion) {
    let signal = test_signal(ONE_SECOND);

    c.bench_function("bitcrusher_1sec", |b| {
        let mut crusher = Bitcrusher::new(SAMPLE_RATE);
        b.iter(|| {
            for &sample in &signal {
                black_box(crusher.process(sample));
            }
        });
    });
}

// ============================================================================
// Graph-based effects (for comparison)
// ============================================================================

fn bench_tremolo_graph(c: &mut Criterion) {
    let signal = test_signal(ONE_SECOND);

    c.bench_function("tremolo_graph_1sec", |b| {
        let mut effect = tremolo_graph(SAMPLE_RATE, 5.0, 0.5);
        let ctx = AudioContext::new(SAMPLE_RATE);
        b.iter(|| {
            for &sample in &signal {
                black_box(effect.process(sample, &ctx));
            }
        });
    });
}

fn bench_chorus_graph(c: &mut Criterion) {
    let signal = test_signal(ONE_SECOND);

    c.bench_function("chorus_graph_1sec", |b| {
        let mut effect = chorus_graph(SAMPLE_RATE);
        let ctx = AudioContext::new(SAMPLE_RATE);
        b.iter(|| {
            for &sample in &signal {
                black_box(effect.process(sample, &ctx));
            }
        });
    });
}

fn bench_flanger_graph(c: &mut Criterion) {
    let signal = test_signal(ONE_SECOND);

    c.bench_function("flanger_graph_1sec", |b| {
        let mut effect = flanger_graph(SAMPLE_RATE);
        let ctx = AudioContext::new(SAMPLE_RATE);
        b.iter(|| {
            for &sample in &signal {
                black_box(effect.process(sample, &ctx));
            }
        });
    });
}

// ============================================================================
// Tier 2: Optimized patterns (pattern matching + monomorphized)
// ============================================================================

#[cfg(feature = "optimize")]
fn bench_tremolo_optimized(c: &mut Criterion) {
    use rhizome_resin_audio::graph::AudioNode;
    use rhizome_resin_audio::optimize::TremoloOptimized;

    let signal = test_signal(ONE_SECOND);
    let ctx = AudioContext::new(SAMPLE_RATE);

    c.bench_function("tremolo_optimized_1sec", |b| {
        // rate=5.0 Hz, depth=0.5, sample_rate
        let mut effect = TremoloOptimized::new(5.0, 0.5, SAMPLE_RATE);
        b.iter(|| {
            for &sample in &signal {
                black_box(effect.process(sample, &ctx));
            }
        });
    });
}

#[cfg(feature = "optimize")]
fn bench_chorus_optimized(c: &mut Criterion) {
    use rhizome_resin_audio::graph::AudioNode;
    use rhizome_resin_audio::optimize::ChorusOptimized;

    let signal = test_signal(ONE_SECOND);
    let ctx = AudioContext::new(SAMPLE_RATE);

    c.bench_function("chorus_optimized_1sec", |b| {
        // rate=0.5 Hz, base_delay=7ms, depth=3ms, mix=0.5
        let mut effect = ChorusOptimized::new(0.5, 7.0, 3.0, 0.5, SAMPLE_RATE);
        b.iter(|| {
            for &sample in &signal {
                black_box(effect.process(sample, &ctx));
            }
        });
    });
}

#[cfg(feature = "optimize")]
fn bench_flanger_optimized(c: &mut Criterion) {
    use rhizome_resin_audio::graph::AudioNode;
    use rhizome_resin_audio::optimize::FlangerOptimized;

    let signal = test_signal(ONE_SECOND);
    let ctx = AudioContext::new(SAMPLE_RATE);

    c.bench_function("flanger_optimized_1sec", |b| {
        // rate=0.3 Hz, base_delay=3ms, depth=2ms, feedback=0.7, mix=0.5 (matches original flanger)
        let mut effect = FlangerOptimized::new(0.3, 3.0, 2.0, 0.7, 0.5, SAMPLE_RATE);
        b.iter(|| {
            for &sample in &signal {
                black_box(effect.process(sample, &ctx));
            }
        });
    });
}

// ============================================================================
// Tier 3: Cranelift JIT
// ============================================================================

#[cfg(feature = "cranelift")]
fn bench_tremolo_jit(c: &mut Criterion) {
    use rhizome_resin_audio::jit::JitCompiler;
    use rhizome_resin_audio::primitive::PhaseOsc;

    let signal = test_signal(ONE_SECOND);

    c.bench_function("tremolo_jit_1sec", |b| {
        let mut compiler = JitCompiler::new().unwrap();
        let compiled = compiler.compile_tremolo(0.5, 0.5).unwrap();
        let mut lfo = PhaseOsc::new();
        let phase_inc = 5.0 / SAMPLE_RATE;

        b.iter(|| {
            for &sample in &signal {
                let lfo_val = lfo.sine();
                lfo.advance(phase_inc);
                black_box(compiled.process(sample, lfo_val));
            }
        });
    });
}

#[cfg(feature = "cranelift")]
fn bench_gain_jit(c: &mut Criterion) {
    use rhizome_resin_audio::jit::JitCompiler;

    let signal = test_signal(ONE_SECOND);

    c.bench_function("gain_jit_1sec", |b| {
        let mut compiler = JitCompiler::new().unwrap();
        let compiled = compiler.compile_gain(0.5).unwrap();

        b.iter(|| {
            for &sample in &signal {
                black_box(compiled.process(sample));
            }
        });
    });
}

// Baseline: pure Rust gain for comparison
fn bench_gain_rust(c: &mut Criterion) {
    let signal = test_signal(ONE_SECOND);

    c.bench_function("gain_rust_1sec", |b| {
        let gain = 0.5f32;
        b.iter(|| {
            for &sample in &signal {
                black_box(sample * gain);
            }
        });
    });
}

// ============================================================================
// Tier 4: Build.rs codegen
// ============================================================================

#[cfg(feature = "codegen-bench")]
mod codegen_bench {
    include!(concat!(env!("OUT_DIR"), "/codegen_bench.rs"));
}

#[cfg(feature = "codegen-bench")]
fn bench_tremolo_codegen(c: &mut Criterion) {
    use codegen_bench::GeneratedTremolo;
    use rhizome_resin_audio::graph::AudioNode;

    let signal = test_signal(ONE_SECOND);
    let ctx = AudioContext::new(SAMPLE_RATE);

    c.bench_function("tremolo_codegen_1sec", |b| {
        let mut effect = GeneratedTremolo::new(SAMPLE_RATE);
        b.iter(|| {
            for &sample in &signal {
                black_box(effect.process(sample, &ctx));
            }
        });
    });
}

#[cfg(feature = "codegen-bench")]
fn bench_chorus_codegen(c: &mut Criterion) {
    use codegen_bench::GeneratedChorus;
    use rhizome_resin_audio::graph::AudioNode;

    let signal = test_signal(ONE_SECOND);
    let ctx = AudioContext::new(SAMPLE_RATE);

    c.bench_function("chorus_codegen_1sec", |b| {
        let mut effect = GeneratedChorus::new(SAMPLE_RATE);
        b.iter(|| {
            for &sample in &signal {
                black_box(effect.process(sample, &ctx));
            }
        });
    });
}

#[cfg(feature = "codegen-bench")]
fn bench_flanger_codegen(c: &mut Criterion) {
    use codegen_bench::GeneratedFlanger;
    use rhizome_resin_audio::graph::AudioNode;

    let signal = test_signal(ONE_SECOND);
    let ctx = AudioContext::new(SAMPLE_RATE);

    c.bench_function("flanger_codegen_1sec", |b| {
        let mut effect = GeneratedFlanger::new(SAMPLE_RATE);
        b.iter(|| {
            for &sample in &signal {
                black_box(effect.process(sample, &ctx));
            }
        });
    });
}

criterion_group!(
    benches,
    // Tier 1: Concrete effect structs (baseline)
    bench_chorus,
    bench_flanger,
    bench_phaser,
    bench_tremolo,
    bench_reverb,
    bench_distortion,
    bench_compressor,
    bench_limiter,
    bench_noise_gate,
    bench_bitcrusher,
    // Dynamic graph versions
    bench_tremolo_graph,
    bench_chorus_graph,
    bench_flanger_graph,
    // Rust baseline
    bench_gain_rust,
);

#[cfg(feature = "codegen-bench")]
criterion_group!(
    codegen_benches,
    bench_tremolo_codegen,
    bench_chorus_codegen,
    bench_flanger_codegen,
);

#[cfg(feature = "optimize")]
criterion_group!(
    optimized_benches,
    bench_tremolo_optimized,
    bench_chorus_optimized,
    bench_flanger_optimized,
);

#[cfg(feature = "cranelift")]
criterion_group!(jit_benches, bench_tremolo_jit, bench_gain_jit,);

// Main macro combinations for different feature sets
#[cfg(all(feature = "optimize", feature = "cranelift", feature = "codegen-bench"))]
criterion_main!(benches, optimized_benches, jit_benches, codegen_benches);

#[cfg(all(
    feature = "optimize",
    feature = "cranelift",
    not(feature = "codegen-bench")
))]
criterion_main!(benches, optimized_benches, jit_benches);

#[cfg(all(
    feature = "optimize",
    not(feature = "cranelift"),
    feature = "codegen-bench"
))]
criterion_main!(benches, optimized_benches, codegen_benches);

#[cfg(all(
    feature = "optimize",
    not(feature = "cranelift"),
    not(feature = "codegen-bench")
))]
criterion_main!(benches, optimized_benches);

#[cfg(all(
    not(feature = "optimize"),
    feature = "cranelift",
    feature = "codegen-bench"
))]
criterion_main!(benches, jit_benches, codegen_benches);

#[cfg(all(
    not(feature = "optimize"),
    feature = "cranelift",
    not(feature = "codegen-bench")
))]
criterion_main!(benches, jit_benches);

#[cfg(all(
    not(feature = "optimize"),
    not(feature = "cranelift"),
    feature = "codegen-bench"
))]
criterion_main!(benches, codegen_benches);

#[cfg(not(any(feature = "optimize", feature = "cranelift", feature = "codegen-bench")))]
criterion_main!(benches);
