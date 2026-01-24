//! Benchmarks for mesh operations.

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use glam::Vec3;
use unshape_mesh::{
    DecimateConfig, MarchingCubesConfig, decimate, marching_cubes, subdivide_loop,
    subdivide_loop_n, uv_sphere,
};

fn bench_primitives(c: &mut Criterion) {
    c.bench_function("uv_sphere_16x8", |b| {
        b.iter(|| uv_sphere(black_box(16), black_box(8)))
    });

    c.bench_function("uv_sphere_32x16", |b| {
        b.iter(|| uv_sphere(black_box(32), black_box(16)))
    });

    c.bench_function("uv_sphere_64x32", |b| {
        b.iter(|| uv_sphere(black_box(64), black_box(32)))
    });
}

fn bench_marching_cubes(c: &mut Criterion) {
    // Sphere SDF
    let sphere_sdf = |p: Vec3| -> f32 { p.length() - 0.5 };

    c.bench_function("marching_cubes_16", |b| {
        let config = MarchingCubesConfig {
            resolution: 16,
            min: Vec3::splat(-1.0),
            max: Vec3::splat(1.0),
            iso_value: 0.0,
        };
        b.iter(|| marching_cubes(black_box(&sphere_sdf), black_box(config.clone())))
    });

    c.bench_function("marching_cubes_32", |b| {
        let config = MarchingCubesConfig {
            resolution: 32,
            min: Vec3::splat(-1.0),
            max: Vec3::splat(1.0),
            iso_value: 0.0,
        };
        b.iter(|| marching_cubes(black_box(&sphere_sdf), black_box(config.clone())))
    });

    c.bench_function("marching_cubes_64", |b| {
        let config = MarchingCubesConfig {
            resolution: 64,
            min: Vec3::splat(-1.0),
            max: Vec3::splat(1.0),
            iso_value: 0.0,
        };
        b.iter(|| marching_cubes(black_box(&sphere_sdf), black_box(config.clone())))
    });
}

fn bench_subdivision(c: &mut Criterion) {
    let mesh = uv_sphere(8, 8);

    c.bench_function("subdivide_loop_1x", |b| {
        b.iter(|| subdivide_loop(black_box(&mesh)))
    });

    c.bench_function("subdivide_loop_2x", |b| {
        b.iter(|| subdivide_loop_n(black_box(&mesh), 2))
    });

    c.bench_function("subdivide_loop_3x", |b| {
        b.iter(|| subdivide_loop_n(black_box(&mesh), 3))
    });
}

fn bench_decimation(c: &mut Criterion) {
    // Create a high-poly mesh to decimate
    let mesh = uv_sphere(32, 32);

    c.bench_function("decimate_50_percent", |b| {
        let config = DecimateConfig {
            target_ratio: Some(0.5),
            ..Default::default()
        };
        b.iter(|| decimate(black_box(&mesh), black_box(config.clone())))
    });

    c.bench_function("decimate_25_percent", |b| {
        let config = DecimateConfig {
            target_ratio: Some(0.25),
            ..Default::default()
        };
        b.iter(|| decimate(black_box(&mesh), black_box(config.clone())))
    });
}

criterion_group!(
    benches,
    bench_primitives,
    bench_marching_cubes,
    bench_subdivision,
    bench_decimation
);
criterion_main!(benches);
