//! JIT compilation benchmarks.

use criterion::{Criterion, black_box, criterion_group, criterion_main};

const SAMPLE_COUNT: usize = 44100; // 1 second of audio

#[cfg(feature = "cranelift")]
fn bench_compile_affine(c: &mut Criterion) {
    use rhizome_resin_jit::{JitCompiler, JitConfig};

    c.bench_function("compile_affine_scalar", |b| {
        b.iter(|| {
            let mut compiler = JitCompiler::new(JitConfig::default()).unwrap();
            compiler.compile_affine(0.5, 1.0).unwrap()
        })
    });

    c.bench_function("compile_affine_simd", |b| {
        b.iter(|| {
            let mut compiler = JitCompiler::new(JitConfig::default()).unwrap();
            compiler.compile_affine_simd(0.5, 1.0).unwrap()
        })
    });
}

#[cfg(feature = "cranelift")]
fn bench_execute_scalar(c: &mut Criterion) {
    use rhizome_resin_jit::{JitCompiler, JitConfig};

    let mut compiler = JitCompiler::new(JitConfig::default()).unwrap();
    let affine = compiler.compile_affine(0.5, 1.0).unwrap();

    c.bench_function("jit_scalar_44100", |b| {
        b.iter(|| {
            let mut sum = 0.0f32;
            for i in 0..SAMPLE_COUNT {
                sum += affine.call_f32(i as f32);
            }
            black_box(sum)
        })
    });
}

#[cfg(feature = "cranelift")]
fn bench_execute_simd(c: &mut Criterion) {
    use rhizome_resin_jit::{JitCompiler, JitConfig};

    let mut compiler = JitCompiler::new(JitConfig::default()).unwrap();
    let simd = compiler.compile_affine_simd(0.5, 1.0).unwrap();

    let input: Vec<f32> = (0..SAMPLE_COUNT).map(|i| i as f32).collect();
    let mut output = vec![0.0f32; SAMPLE_COUNT];

    c.bench_function("jit_simd_44100", |b| {
        b.iter(|| {
            simd.process(black_box(&input), black_box(&mut output));
            black_box(output[0])
        })
    });
}

#[cfg(feature = "cranelift")]
fn bench_native(c: &mut Criterion) {
    let gain = 0.5f32;
    let offset = 1.0f32;

    let input: Vec<f32> = (0..SAMPLE_COUNT).map(|i| i as f32).collect();
    let mut output = vec![0.0f32; SAMPLE_COUNT];

    c.bench_function("native_44100", |b| {
        b.iter(|| {
            for i in 0..SAMPLE_COUNT {
                output[i] = input[i] * gain + offset;
            }
            black_box(output[0])
        })
    });
}

#[cfg(feature = "cranelift")]
criterion_group!(
    benches,
    bench_compile_affine,
    bench_execute_scalar,
    bench_execute_simd,
    bench_native
);

#[cfg(not(feature = "cranelift"))]
fn placeholder(_c: &mut Criterion) {}

#[cfg(not(feature = "cranelift"))]
criterion_group!(benches, placeholder);

criterion_main!(benches);
