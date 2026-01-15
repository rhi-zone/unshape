//! JIT compilation benchmarks.

use criterion::{Criterion, criterion_group, criterion_main};

#[cfg(feature = "cranelift")]
fn bench_compile_affine(c: &mut Criterion) {
    use rhizome_resin_jit::{JitCompiler, JitConfig};

    c.bench_function("compile_affine", |b| {
        b.iter(|| {
            let mut compiler = JitCompiler::new(JitConfig::default()).unwrap();
            compiler.compile_affine(0.5, 1.0).unwrap()
        })
    });
}

#[cfg(feature = "cranelift")]
fn bench_execute_affine(c: &mut Criterion) {
    use rhizome_resin_jit::{JitCompiler, JitConfig};

    let mut compiler = JitCompiler::new(JitConfig::default()).unwrap();
    let affine = compiler.compile_affine(0.5, 1.0).unwrap();

    c.bench_function("execute_affine_1k", |b| {
        b.iter(|| {
            let mut sum = 0.0f32;
            for i in 0..1000 {
                sum += affine.call_f32(i as f32);
            }
            sum
        })
    });
}

#[cfg(feature = "cranelift")]
fn bench_native_affine(c: &mut Criterion) {
    let gain = 0.5f32;
    let offset = 1.0f32;

    c.bench_function("native_affine_1k", |b| {
        b.iter(|| {
            let mut sum = 0.0f32;
            for i in 0..1000 {
                sum += (i as f32) * gain + offset;
            }
            sum
        })
    });
}

#[cfg(feature = "cranelift")]
criterion_group!(
    benches,
    bench_compile_affine,
    bench_execute_affine,
    bench_native_affine
);

#[cfg(not(feature = "cranelift"))]
fn placeholder(_c: &mut Criterion) {}

#[cfg(not(feature = "cranelift"))]
criterion_group!(benches, placeholder);

criterion_main!(benches);
