//! Benchmarks for noise functions.

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use unshape_noise::{
    fbm_perlin2, fbm_perlin3, fbm_simplex2, fbm_simplex3, perlin2, perlin3, simplex2, simplex3,
};

fn bench_perlin(c: &mut Criterion) {
    c.bench_function("perlin2", |b| {
        b.iter(|| perlin2(black_box(1.234), black_box(5.678)))
    });

    c.bench_function("perlin3", |b| {
        b.iter(|| perlin3(black_box(1.234), black_box(5.678), black_box(9.012)))
    });
}

fn bench_simplex(c: &mut Criterion) {
    c.bench_function("simplex2", |b| {
        b.iter(|| simplex2(black_box(1.234), black_box(5.678)))
    });

    c.bench_function("simplex3", |b| {
        b.iter(|| simplex3(black_box(1.234), black_box(5.678), black_box(9.012)))
    });
}

fn bench_fbm(c: &mut Criterion) {
    c.bench_function("fbm_perlin2_4oct", |b| {
        b.iter(|| fbm_perlin2(black_box(1.234), black_box(5.678), black_box(4)))
    });

    c.bench_function("fbm_perlin2_8oct", |b| {
        b.iter(|| fbm_perlin2(black_box(1.234), black_box(5.678), black_box(8)))
    });

    c.bench_function("fbm_perlin3_4oct", |b| {
        b.iter(|| {
            fbm_perlin3(
                black_box(1.234),
                black_box(5.678),
                black_box(9.012),
                black_box(4),
            )
        })
    });

    c.bench_function("fbm_simplex2_4oct", |b| {
        b.iter(|| fbm_simplex2(black_box(1.234), black_box(5.678), black_box(4)))
    });

    c.bench_function("fbm_simplex3_4oct", |b| {
        b.iter(|| {
            fbm_simplex3(
                black_box(1.234),
                black_box(5.678),
                black_box(9.012),
                black_box(4),
            )
        })
    });
}

fn bench_bulk(c: &mut Criterion) {
    c.bench_function("perlin2_1000_samples", |b| {
        b.iter(|| {
            for i in 0..1000 {
                let x = i as f32 * 0.01;
                let y = i as f32 * 0.017;
                black_box(perlin2(x, y));
            }
        })
    });

    c.bench_function("simplex2_1000_samples", |b| {
        b.iter(|| {
            for i in 0..1000 {
                let x = i as f32 * 0.01;
                let y = i as f32 * 0.017;
                black_box(simplex2(x, y));
            }
        })
    });

    c.bench_function("fbm_perlin2_4oct_1000_samples", |b| {
        b.iter(|| {
            for i in 0..1000 {
                let x = i as f32 * 0.01;
                let y = i as f32 * 0.017;
                black_box(fbm_perlin2(x, y, 4));
            }
        })
    });
}

criterion_group!(benches, bench_perlin, bench_simplex, bench_fbm, bench_bulk);
criterion_main!(benches);
