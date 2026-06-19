//! Benchmarks for noise functions.

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use unshape_noise::{Fbm, Noise2D, Noise3D, Perlin2D, Perlin3D, Simplex2D, Simplex3D};

fn bench_perlin(c: &mut Criterion) {
    let perlin2 = Perlin2D::new();
    let perlin3 = Perlin3D::new();

    c.bench_function("perlin2", |b| {
        b.iter(|| perlin2.sample(black_box(1.234), black_box(5.678)))
    });

    c.bench_function("perlin3", |b| {
        b.iter(|| perlin3.sample(black_box(1.234), black_box(5.678), black_box(9.012)))
    });
}

fn bench_simplex(c: &mut Criterion) {
    let simplex2 = Simplex2D::new();
    let simplex3 = Simplex3D::new();

    c.bench_function("simplex2", |b| {
        b.iter(|| simplex2.sample(black_box(1.234), black_box(5.678)))
    });

    c.bench_function("simplex3", |b| {
        b.iter(|| simplex3.sample(black_box(1.234), black_box(5.678), black_box(9.012)))
    });
}

fn bench_fbm(c: &mut Criterion) {
    let fbm_perlin2_4 = Fbm::new(Perlin2D::new()).octaves(4);
    let fbm_perlin2_8 = Fbm::new(Perlin2D::new()).octaves(8);
    let fbm_perlin3_4 = Fbm::new(Perlin3D::new()).octaves(4);
    let fbm_simplex2_4 = Fbm::new(Simplex2D::new()).octaves(4);
    let fbm_simplex3_4 = Fbm::new(Simplex3D::new()).octaves(4);

    c.bench_function("fbm_perlin2_4oct", |b| {
        b.iter(|| fbm_perlin2_4.sample(black_box(1.234), black_box(5.678)))
    });

    c.bench_function("fbm_perlin2_8oct", |b| {
        b.iter(|| fbm_perlin2_8.sample(black_box(1.234), black_box(5.678)))
    });

    c.bench_function("fbm_perlin3_4oct", |b| {
        b.iter(|| fbm_perlin3_4.sample(black_box(1.234), black_box(5.678), black_box(9.012)))
    });

    c.bench_function("fbm_simplex2_4oct", |b| {
        b.iter(|| fbm_simplex2_4.sample(black_box(1.234), black_box(5.678)))
    });

    c.bench_function("fbm_simplex3_4oct", |b| {
        b.iter(|| fbm_simplex3_4.sample(black_box(1.234), black_box(5.678), black_box(9.012)))
    });
}

fn bench_bulk(c: &mut Criterion) {
    let perlin2 = Perlin2D::new();
    let simplex2 = Simplex2D::new();
    let fbm_perlin2_4 = Fbm::new(Perlin2D::new()).octaves(4);

    c.bench_function("perlin2_1000_samples", |b| {
        b.iter(|| {
            for i in 0..1000 {
                let x = i as f32 * 0.01;
                let y = i as f32 * 0.017;
                black_box(perlin2.sample(x, y));
            }
        })
    });

    c.bench_function("simplex2_1000_samples", |b| {
        b.iter(|| {
            for i in 0..1000 {
                let x = i as f32 * 0.01;
                let y = i as f32 * 0.017;
                black_box(simplex2.sample(x, y));
            }
        })
    });

    c.bench_function("fbm_perlin2_4oct_1000_samples", |b| {
        b.iter(|| {
            for i in 0..1000 {
                let x = i as f32 * 0.01;
                let y = i as f32 * 0.017;
                black_box(fbm_perlin2_4.sample(x, y));
            }
        })
    });
}

criterion_group!(benches, bench_perlin, bench_simplex, bench_fbm, bench_bulk);
criterion_main!(benches);
