use glam::{Vec2, Vec3};

use crate::*;

const SAMPLE_COUNT: usize = 10_000;

// ========================================================================
// Noise field range tests
// ========================================================================

/// All 2D noise fields should produce values in [0, 1].
#[test]
fn test_noise_2d_range() {
    let ctx = EvalContext::new();
    #[allow(clippy::type_complexity)]
    let fields: Vec<(&str, Box<dyn Field<Vec2, f32>>)> = vec![
        ("Perlin2D", Box::new(Perlin2D::new())),
        ("Simplex2D", Box::new(Simplex2D::new())),
        ("Value2D", Box::new(Value2D::new())),
        ("Worley2D", Box::new(Worley2D::new())),
        ("WorleyF2_2D", Box::new(WorleyF2_2D::new())),
        ("WorleyEdge2D", Box::new(WorleyEdge2D::new())),
        ("WhiteNoise2D", Box::new(WhiteNoise2D::new())),
        ("PinkNoise2D", Box::new(PinkNoise2D::new())),
        ("BrownNoise2D", Box::new(BrownNoise2D::new())),
    ];

    for (name, field) in fields {
        let mut min = f32::MAX;
        let mut max = f32::MIN;

        for i in 0..SAMPLE_COUNT {
            let x = (i as f32 / SAMPLE_COUNT as f32) * 100.0 - 50.0;
            let y = ((i * 7) as f32 / SAMPLE_COUNT as f32) * 100.0 - 50.0;
            let v = field.sample(Vec2::new(x, y), &ctx);
            min = min.min(v);
            max = max.max(v);
        }

        assert!(
            min >= -0.01 && max <= 1.01,
            "{name}: values out of range [0, 1], got [{min:.3}, {max:.3}]"
        );
    }
}

/// All 3D noise fields should produce values in [0, 1].
#[test]
fn test_noise_3d_range() {
    let ctx = EvalContext::new();
    #[allow(clippy::type_complexity)]
    let fields: Vec<(&str, Box<dyn Field<Vec3, f32>>)> = vec![
        ("Perlin3D", Box::new(Perlin3D::new())),
        ("Simplex3D", Box::new(Simplex3D::new())),
        ("Value3D", Box::new(Value3D::new())),
        ("Worley3D", Box::new(Worley3D::new())),
    ];

    for (name, field) in fields {
        let mut min = f32::MAX;
        let mut max = f32::MIN;

        for i in 0..SAMPLE_COUNT {
            let x = (i as f32 / SAMPLE_COUNT as f32) * 100.0 - 50.0;
            let y = ((i * 7) as f32 / SAMPLE_COUNT as f32) * 100.0 - 50.0;
            let z = ((i * 13) as f32 / SAMPLE_COUNT as f32) * 100.0 - 50.0;
            let v = field.sample(Vec3::new(x, y, z), &ctx);
            min = min.min(v);
            max = max.max(v);
        }

        assert!(
            min >= -0.01 && max <= 1.01,
            "{name}: values out of range [0, 1], got [{min:.3}, {max:.3}]"
        );
    }
}

/// 1D noise fields should produce values in [0, 1].
#[test]
fn test_noise_1d_range() {
    let ctx = EvalContext::new();
    #[allow(clippy::type_complexity)]
    let fields: Vec<(&str, Box<dyn Field<f32, f32>>)> = vec![
        ("Perlin1D", Box::new(Perlin1D::new())),
        ("Simplex1D", Box::new(Simplex1D::new())),
        ("Value1D", Box::new(Value1D::new())),
        ("Worley1D", Box::new(Worley1D::new())),
        ("WhiteNoise1D", Box::new(WhiteNoise1D::new())),
        ("PinkNoise1D", Box::new(PinkNoise1D::new())),
        ("BrownNoise1D", Box::new(BrownNoise1D::new())),
        ("VioletNoise1D", Box::new(VioletNoise1D::new())),
        ("GreyNoise1D", Box::new(GreyNoise1D::new())),
        ("VelvetNoise1D", Box::new(VelvetNoise1D::new())),
    ];

    for (name, field) in fields {
        let mut min = f32::MAX;
        let mut max = f32::MIN;

        for i in 0..SAMPLE_COUNT {
            let x = (i as f32 / SAMPLE_COUNT as f32) * 100.0 - 50.0;
            let v = field.sample(x, &ctx);
            min = min.min(v);
            max = max.max(v);
        }

        assert!(
            min >= -0.01 && max <= 1.01,
            "{name}: values out of range [0, 1], got [{min:.3}, {max:.3}]"
        );
    }
}

// ========================================================================
// Determinism tests
// ========================================================================

/// Noise fields with the same seed should produce identical output.
#[test]
fn test_noise_determinism() {
    let ctx = EvalContext::new();
    let seeds = [0, 42, 12345, -999];
    let positions = [
        Vec2::new(0.0, 0.0),
        Vec2::new(1.5, -2.3),
        Vec2::new(100.0, 100.0),
        Vec2::new(-50.0, 25.0),
    ];

    for seed in seeds {
        for pos in positions {
            // Create two instances with same seed
            let a = Perlin2D::with_seed(seed);
            let b = Perlin2D::with_seed(seed);

            let va = a.sample(pos, &ctx);
            let vb = b.sample(pos, &ctx);

            assert_eq!(
                va, vb,
                "Same seed should produce identical output: seed={seed}, pos={pos:?}"
            );
        }
    }
}

/// Different seeds should produce different output (with high probability).
#[test]
fn test_different_seeds_differ() {
    let ctx = EvalContext::new();
    let pos = Vec2::new(5.5, -3.2);

    let v0 = Perlin2D::with_seed(0).sample(pos, &ctx);
    let v1 = Perlin2D::with_seed(1).sample(pos, &ctx);
    let v2 = Perlin2D::with_seed(42).sample(pos, &ctx);

    // At least 2 of 3 should be different
    let different = (v0 != v1) as u32 + (v1 != v2) as u32 + (v0 != v2) as u32;
    assert!(
        different >= 2,
        "Different seeds should produce different outputs"
    );
}

// ========================================================================
// FBM property tests
// ========================================================================

/// FBM should still produce values in [0, 1] after octave composition.
#[test]
fn test_fbm_range() {
    let ctx = EvalContext::new();

    for octaves in [1, 2, 4, 8] {
        let fbm = Fbm2D::new(Perlin2D::new()).octaves(octaves);

        let mut min = f32::MAX;
        let mut max = f32::MIN;

        for i in 0..SAMPLE_COUNT {
            let x = (i as f32 / SAMPLE_COUNT as f32) * 50.0 - 25.0;
            let y = ((i * 7) as f32 / SAMPLE_COUNT as f32) * 50.0 - 25.0;
            let v = fbm.sample(Vec2::new(x, y), &ctx);
            min = min.min(v);
            max = max.max(v);
        }

        assert!(
            min >= -0.01 && max <= 1.01,
            "FBM({octaves} octaves): values out of range, got [{min:.3}, {max:.3}]"
        );
    }
}

// ========================================================================
// Terrain property tests
// ========================================================================

/// Terrain fields should produce values in [0, 1].
#[test]
fn test_terrain_range() {
    let ctx = EvalContext::new();
    #[allow(clippy::type_complexity)]
    let terrains: Vec<(&str, Box<dyn Field<Vec2, f32>>)> = vec![
        ("Terrain2D::default", Box::new(Terrain2D::default())),
        ("Terrain2D::mountains", Box::new(Terrain2D::mountains())),
        ("Terrain2D::plains", Box::new(Terrain2D::plains())),
        ("Terrain2D::canyons", Box::new(Terrain2D::canyons())),
        ("RidgedTerrain2D", Box::new(RidgedTerrain2D::default())),
        ("BillowyTerrain2D", Box::new(BillowyTerrain2D::default())),
    ];

    for (name, terrain) in terrains {
        let mut min = f32::MAX;
        let mut max = f32::MIN;

        for i in 0..SAMPLE_COUNT {
            let x = (i as f32 / SAMPLE_COUNT as f32) * 20.0 - 10.0;
            let y = ((i * 7) as f32 / SAMPLE_COUNT as f32) * 20.0 - 10.0;
            let v = terrain.sample(Vec2::new(x, y), &ctx);
            min = min.min(v);
            max = max.max(v);
        }

        assert!(
            min >= -0.01 && max <= 1.01,
            "{name}: values out of range [0, 1], got [{min:.3}, {max:.3}]"
        );
    }
}

// ========================================================================
// Combinator property tests
// ========================================================================

/// Scale combinator should scale input coordinates (zoom effect).
#[test]
fn test_scale_combinator() {
    let ctx = EvalContext::new();
    // Use checkerboard which has visible scale effect
    let field = Checkerboard::with_scale(1.0);

    // At scale 2.0, pattern should appear twice as large (sample at 1.0 = unscaled at 2.0)
    let scaled = Scale { field, factor: 2.0 };

    // Sample scaled at (0.25, 0.25) = unscaled at (0.5, 0.5)
    let v_scaled = scaled.sample(Vec2::new(0.25, 0.25), &ctx);
    let v_unscaled = field.sample(Vec2::new(0.5, 0.5), &ctx);

    assert_eq!(
        v_scaled, v_unscaled,
        "Scale should multiply input coordinates"
    );
}

/// Add helper should sum outputs correctly.
#[test]
fn test_add_combinator() {
    let ctx = EvalContext::new();
    let a = Constant::<f32>::new(0.3);
    let b = Constant::<f32>::new(0.4);

    let sum = add::<Vec2, _, _, _>(a, b);
    let v = sum.sample(Vec2::ZERO, &ctx);

    assert!((v - 0.7).abs() < 0.001, "add: expected 0.7, got {v}");
}

/// Translate combinator should shift input coordinates.
#[test]
fn test_translate_combinator() {
    let ctx = EvalContext::new();
    // Use a field that depends on position
    let field = Checkerboard::with_scale(1.0);

    // Translate shifts input by subtracting offset
    // So translated.sample(p) = field.sample(p - offset)
    let translated = Translate {
        field,
        offset: Vec2::new(1.0, 1.0),
    };

    let v1 = translated.sample(Vec2::new(1.5, 1.5), &ctx);
    let v2 = field.sample(Vec2::new(0.5, 0.5), &ctx);

    assert_eq!(v1, v2, "Translate should shift input coordinates");
}

// ========================================================================
// Statistical distribution tests
// ========================================================================

/// White noise should have approximately uniform distribution.
#[test]
fn test_white_noise_distribution() {
    let ctx = EvalContext::new();
    let noise = WhiteNoise2D::new();

    let mut sum = 0.0;
    let mut sum_sq = 0.0;

    for i in 0..SAMPLE_COUNT {
        let x = i as f32 * 0.1;
        let y = (i * 7) as f32 * 0.1;
        let v = noise.sample(Vec2::new(x, y), &ctx);
        sum += v;
        sum_sq += v * v;
    }

    let mean = sum / SAMPLE_COUNT as f32;
    let variance = sum_sq / SAMPLE_COUNT as f32 - mean * mean;

    // Uniform [0,1] has mean 0.5 and variance 1/12 ≈ 0.0833
    assert!(
        (mean - 0.5).abs() < 0.05,
        "White noise mean should be ~0.5, got {mean}"
    );
    assert!(
        (variance - 0.0833).abs() < 0.02,
        "White noise variance should be ~0.0833, got {variance}"
    );
}

/// Velvet noise should be mostly neutral (0.5) with occasional impulses.
#[test]
fn test_velvet_noise_sparsity() {
    let ctx = EvalContext::new();
    let velvet = VelvetNoise1D::new().density(0.1);

    let mut neutral_count = 0;

    for i in 0..SAMPLE_COUNT {
        let x = i as f32 * 0.01;
        let v = velvet.sample(x, &ctx);
        if (v - 0.5).abs() < 0.01 {
            neutral_count += 1;
        }
    }

    let neutral_ratio = neutral_count as f32 / SAMPLE_COUNT as f32;

    // With 10% density, ~90% should be neutral
    assert!(
        neutral_ratio > 0.85,
        "Velvet noise should be mostly neutral, got {:.1}% neutral",
        neutral_ratio * 100.0
    );
}
