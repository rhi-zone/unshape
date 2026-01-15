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

criterion_group!(
    benches,
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
    // Graph-based versions
    bench_tremolo_graph,
    bench_chorus_graph,
    bench_flanger_graph,
);
criterion_main!(benches);
