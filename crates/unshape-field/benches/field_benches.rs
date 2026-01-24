use criterion::{Criterion, black_box, criterion_group, criterion_main};
use glam::Vec2;
use unshape_field::*;

// ============================================================================
// Noise field sampling
// ============================================================================

fn bench_perlin2d(c: &mut Criterion) {
    let field = Perlin2D::new();
    let ctx = EvalContext::new();

    c.bench_function("perlin2d_sample", |b| {
        b.iter(|| {
            for i in 0..1000 {
                let x = (i as f32) * 0.1;
                let y = (i as f32) * 0.07;
                black_box(field.sample(Vec2::new(x, y), &ctx));
            }
        })
    });
}

fn bench_simplex2d(c: &mut Criterion) {
    let field = Simplex2D::new();
    let ctx = EvalContext::new();

    c.bench_function("simplex2d_sample", |b| {
        b.iter(|| {
            for i in 0..1000 {
                let x = (i as f32) * 0.1;
                let y = (i as f32) * 0.07;
                black_box(field.sample(Vec2::new(x, y), &ctx));
            }
        })
    });
}

fn bench_worley2d(c: &mut Criterion) {
    let field = Worley2D::new();
    let ctx = EvalContext::new();

    c.bench_function("worley2d_sample", |b| {
        b.iter(|| {
            for i in 0..1000 {
                let x = (i as f32) * 0.1;
                let y = (i as f32) * 0.07;
                black_box(field.sample(Vec2::new(x, y), &ctx));
            }
        })
    });
}

// ============================================================================
// FBM composition with different octave counts
// ============================================================================

fn bench_fbm_octaves(c: &mut Criterion) {
    let ctx = EvalContext::new();
    let mut group = c.benchmark_group("fbm_octaves");

    for octaves in [2, 4, 6, 8] {
        let field = Fbm2D::new(Perlin2D::new()).octaves(octaves);

        group.bench_function(format!("fbm2d_{}_octaves", octaves), |b| {
            b.iter(|| {
                for i in 0..1000 {
                    let x = (i as f32) * 0.1;
                    let y = (i as f32) * 0.07;
                    black_box(field.sample(Vec2::new(x, y), &ctx));
                }
            })
        });
    }

    group.finish();
}

// ============================================================================
// Field combinators
// ============================================================================

fn bench_add_combinator(c: &mut Criterion) {
    let field = Perlin2D::new().add(Simplex2D::new());
    let ctx = EvalContext::new();

    c.bench_function("add_combinator", |b| {
        b.iter(|| {
            for i in 0..1000 {
                let x = (i as f32) * 0.1;
                let y = (i as f32) * 0.07;
                black_box(field.sample(Vec2::new(x, y), &ctx));
            }
        })
    });
}

fn bench_mul_combinator(c: &mut Criterion) {
    let field = Perlin2D::new().mul(Simplex2D::new());
    let ctx = EvalContext::new();

    c.bench_function("mul_combinator", |b| {
        b.iter(|| {
            for i in 0..1000 {
                let x = (i as f32) * 0.1;
                let y = (i as f32) * 0.07;
                black_box(field.sample(Vec2::new(x, y), &ctx));
            }
        })
    });
}

fn bench_scale_combinator(c: &mut Criterion) {
    let field = Perlin2D::new().scale(4.0);
    let ctx = EvalContext::new();

    c.bench_function("scale_combinator", |b| {
        b.iter(|| {
            for i in 0..1000 {
                let x = (i as f32) * 0.1;
                let y = (i as f32) * 0.07;
                black_box(field.sample(Vec2::new(x, y), &ctx));
            }
        })
    });
}

fn bench_translate_combinator(c: &mut Criterion) {
    let field = Perlin2D::new().translate(Vec2::new(10.0, 10.0));
    let ctx = EvalContext::new();

    c.bench_function("translate_combinator", |b| {
        b.iter(|| {
            for i in 0..1000 {
                let x = (i as f32) * 0.1;
                let y = (i as f32) * 0.07;
                black_box(field.sample(Vec2::new(x, y), &ctx));
            }
        })
    });
}

fn bench_chained_combinators(c: &mut Criterion) {
    let field = Perlin2D::new()
        .scale(4.0)
        .translate(Vec2::new(10.0, 10.0))
        .add(Simplex2D::new().scale(2.0));
    let ctx = EvalContext::new();

    c.bench_function("chained_combinators", |b| {
        b.iter(|| {
            for i in 0..1000 {
                let x = (i as f32) * 0.1;
                let y = (i as f32) * 0.07;
                black_box(field.sample(Vec2::new(x, y), &ctx));
            }
        })
    });
}

// ============================================================================
// Terrain generation
// ============================================================================

fn bench_terrain2d(c: &mut Criterion) {
    let field = Terrain2D::new();
    let ctx = EvalContext::new();

    c.bench_function("terrain2d_sample", |b| {
        b.iter(|| {
            for i in 0..1000 {
                let x = (i as f32) * 0.1;
                let y = (i as f32) * 0.07;
                black_box(field.sample(Vec2::new(x, y), &ctx));
            }
        })
    });
}

fn bench_ridged_terrain2d(c: &mut Criterion) {
    let field = RidgedTerrain2D::new();
    let ctx = EvalContext::new();

    c.bench_function("ridged_terrain2d_sample", |b| {
        b.iter(|| {
            for i in 0..1000 {
                let x = (i as f32) * 0.1;
                let y = (i as f32) * 0.07;
                black_box(field.sample(Vec2::new(x, y), &ctx));
            }
        })
    });
}

fn bench_terrain_presets(c: &mut Criterion) {
    let ctx = EvalContext::new();
    let mut group = c.benchmark_group("terrain_presets");

    let presets = [
        ("rolling_hills", Terrain2D::rolling_hills()),
        ("mountains", Terrain2D::mountains()),
        ("plains", Terrain2D::plains()),
        ("canyons", Terrain2D::canyons()),
    ];

    for (name, field) in presets {
        group.bench_function(name, |b| {
            b.iter(|| {
                for i in 0..1000 {
                    let x = (i as f32) * 0.1;
                    let y = (i as f32) * 0.07;
                    black_box(field.sample(Vec2::new(x, y), &ctx));
                }
            })
        });
    }

    group.finish();
}

// ============================================================================
// Heightmap generation
// ============================================================================

fn bench_heightmap_from_field(c: &mut Criterion) {
    let mut group = c.benchmark_group("heightmap_generation");

    let sizes = [(64, 64), (128, 128), (256, 256)];

    for (width, height) in sizes {
        let field = Terrain2D::new();

        group.bench_function(format!("heightmap_{}x{}", width, height), |b| {
            b.iter(|| {
                black_box(Heightmap::from_field(&field, width, height, 0.01));
            })
        });
    }

    group.finish();
}

fn bench_heightmap_sampling(c: &mut Criterion) {
    let field = Terrain2D::new();
    let heightmap = Heightmap::from_field(&field, 256, 256, 0.01);

    c.bench_function("heightmap_bilinear_sample", |b| {
        b.iter(|| {
            for i in 0..1000 {
                let x = (i as f32) * 0.25;
                let y = (i as f32) * 0.18;
                black_box(heightmap.sample(x, y));
            }
        })
    });
}

fn bench_heightmap_gradient(c: &mut Criterion) {
    let field = Terrain2D::new();
    let heightmap = Heightmap::from_field(&field, 256, 256, 0.01);

    c.bench_function("heightmap_gradient", |b| {
        b.iter(|| {
            for i in 0..1000 {
                let x = (i as f32) * 0.25;
                let y = (i as f32) * 0.18;
                black_box(heightmap.gradient(x, y));
            }
        })
    });
}

criterion_group!(
    benches,
    // Noise sampling
    bench_perlin2d,
    bench_simplex2d,
    bench_worley2d,
    // FBM
    bench_fbm_octaves,
    // Combinators
    bench_add_combinator,
    bench_mul_combinator,
    bench_scale_combinator,
    bench_translate_combinator,
    bench_chained_combinators,
    // Terrain
    bench_terrain2d,
    bench_ridged_terrain2d,
    bench_terrain_presets,
    // Heightmap
    bench_heightmap_from_field,
    bench_heightmap_sampling,
    bench_heightmap_gradient,
);

criterion_main!(benches);
