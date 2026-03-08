use glam::{Vec2, Vec3};
use std::marker::PhantomData;

use crate::*;

#[test]
fn test_constant_field() {
    let field = Constant::new(42.0f32);
    let ctx = EvalContext::new();

    assert_eq!(field.sample(Vec2::ZERO, &ctx), 42.0);
    assert_eq!(field.sample(Vec2::new(100.0, 100.0), &ctx), 42.0);
}

#[test]
fn test_coordinates_field() {
    let field = Coordinates;
    let ctx = EvalContext::new();

    assert_eq!(field.sample(Vec2::new(1.0, 2.0), &ctx), Vec2::new(1.0, 2.0));
}

#[test]
fn test_scale_combinator() {
    let field = <Coordinates as Field<Vec2, Vec2>>::scale(Coordinates, 2.0);
    let ctx = EvalContext::new();

    // Scaling input by 2 means we query at 2x the position
    let result: Vec2 = field.sample(Vec2::new(1.0, 1.0), &ctx);
    assert_eq!(result, Vec2::new(2.0, 2.0));
}

#[test]
fn test_translate_combinator() {
    let field = <Coordinates as Field<Vec2, Vec2>>::translate(Coordinates, Vec2::new(1.0, 1.0));
    let ctx = EvalContext::new();

    // Translating subtracts from input before sampling
    let result: Vec2 = field.sample(Vec2::new(2.0, 2.0), &ctx);
    assert_eq!(result, Vec2::new(1.0, 1.0));
}

#[test]
fn test_map_combinator() {
    let field = <Constant<f32> as Field<Vec2, f32>>::map(Constant::new(5.0f32), |x| x * 2.0);
    let ctx = EvalContext::new();

    assert_eq!(field.sample(Vec2::ZERO, &ctx), 10.0);
}

#[test]
fn test_add_combinator() {
    let a = Constant::new(3.0f32);
    let b = Constant::new(4.0f32);
    let field = add::<Vec2, _, _, _>(a, b);
    let ctx = EvalContext::new();

    assert_eq!(field.sample(Vec2::ZERO, &ctx), 7.0);
}

#[test]
fn test_zip_combinator() {
    let a = Constant::new(3.0f32);
    let b = Constant::new(4.0f32);
    let zipped = Zip::new(a, b);
    let ctx = EvalContext::new();

    let (va, vb): (f32, f32) = Field::<Vec2, (f32, f32)>::sample(&zipped, Vec2::ZERO, &ctx);
    assert_eq!(va, 3.0);
    assert_eq!(vb, 4.0);
}

#[test]
fn test_zip_with_map_equals_add() {
    let a = Constant::new(3.0f32);
    let b = Constant::new(4.0f32);
    let zipped = Zip::new(a, b);
    let sum = Map {
        field: zipped,
        f: |(x, y): (f32, f32)| x + y,
        _phantom: PhantomData::<(f32, f32)>,
    };
    let ctx = EvalContext::new();

    assert_eq!(Field::<Vec2, f32>::sample(&sum, Vec2::ZERO, &ctx), 7.0);
}

#[test]
fn test_zip3_combinator() {
    let a = Constant::new(1.0f32);
    let b = Constant::new(2.0f32);
    let c = Constant::new(3.0f32);
    let zipped = Zip3::new(a, b, c);
    let ctx = EvalContext::new();

    let (va, vb, vc): (f32, f32, f32) =
        Field::<Vec2, (f32, f32, f32)>::sample(&zipped, Vec2::ZERO, &ctx);
    assert_eq!(va, 1.0);
    assert_eq!(vb, 2.0);
    assert_eq!(vc, 3.0);
}

#[test]
fn test_zip3_lerp() {
    let a = Constant::new(0.0f32);
    let b = Constant::new(10.0f32);
    let t = Constant::new(0.5f32);
    let zipped = Zip3::new(a, b, t);
    let result = Map {
        field: zipped,
        f: |(a, b, t): (f32, f32, f32)| a * (1.0 - t) + b * t,
        _phantom: PhantomData::<(f32, f32, f32)>,
    };
    let ctx = EvalContext::new();

    assert_eq!(Field::<Vec2, f32>::sample(&result, Vec2::ZERO, &ctx), 5.0);
}

#[test]
fn test_lerp_helper() {
    let a = Constant::new(0.0f32);
    let b = Constant::new(10.0f32);
    let t = Constant::new(0.25f32);
    let result = lerp::<Vec2, _, _, _, _>(a, b, t);
    let ctx = EvalContext::new();

    assert_eq!(result.sample(Vec2::ZERO, &ctx), 2.5);
}

#[test]
fn test_lerp_vec2_output() {
    // Generic lerp works with any Lerp-implementing type
    let a = Constant::new(Vec2::new(0.0, 0.0));
    let b = Constant::new(Vec2::new(10.0, 20.0));
    let t = Constant::new(0.5f32);
    let result = lerp::<Vec2, _, _, _, _>(a, b, t);
    let ctx = EvalContext::new();

    let v = result.sample(Vec2::ZERO, &ctx);
    assert_eq!(v, Vec2::new(5.0, 10.0));
}

#[test]
fn test_add_vec2_output() {
    // Generic add works with any Add-implementing type
    let a = Constant::new(Vec2::new(1.0, 2.0));
    let b = Constant::new(Vec2::new(3.0, 4.0));
    let result = add::<Vec2, _, _, _>(a, b);
    let ctx = EvalContext::new();

    let v = result.sample(Vec2::ZERO, &ctx);
    assert_eq!(v, Vec2::new(4.0, 6.0));
}

#[test]
fn test_zip_standalone_fn() {
    let a = Constant::new(5.0f32);
    let b = Constant::new(7.0f32);
    let zipped = zip(a, b);
    let ctx = EvalContext::new();

    let (va, vb): (f32, f32) = zipped.sample(0.0f32, &ctx);
    assert_eq!(va, 5.0);
    assert_eq!(vb, 7.0);
}

#[test]
fn test_zip3_standalone_fn() {
    let a = Constant::new(1.0f32);
    let b = Constant::new(2.0f32);
    let c = Constant::new(3.0f32);
    let zipped = zip3(a, b, c);
    let ctx = EvalContext::new();

    let (va, vb, vc): (f32, f32, f32) = zipped.sample(0.0f32, &ctx);
    assert_eq!(va, 1.0);
    assert_eq!(vb, 2.0);
    assert_eq!(vc, 3.0);
}

#[test]
fn test_perlin_field() {
    let field = Perlin2D::new().scale(4.0);
    let ctx = EvalContext::new();

    let v1 = field.sample(Vec2::new(0.0, 0.0), &ctx);
    let v2 = field.sample(Vec2::new(1.0, 1.0), &ctx);

    assert!((0.0..=1.0).contains(&v1));
    assert!((0.0..=1.0).contains(&v2));
}

#[test]
fn test_white_noise_1d() {
    let field = WhiteNoise1D::with_seed(42);
    let ctx = EvalContext::new();

    // Values should be in [0, 1]
    for i in 0..100 {
        let v = field.sample(i as f32 * 0.1, &ctx);
        assert!((0.0..=1.0).contains(&v), "Value {} out of range", v);
    }

    // Same input = same output (deterministic)
    let v1 = field.sample(0.5, &ctx);
    let v2 = field.sample(0.5, &ctx);
    assert_eq!(v1, v2);

    // Different seeds = different output
    let field2 = WhiteNoise1D::with_seed(123);
    let v3 = field2.sample(0.5, &ctx);
    assert_ne!(v1, v3);
}

#[test]
fn test_white_noise_2d() {
    let field = WhiteNoise2D::with_seed(42);
    let ctx = EvalContext::new();

    // Values should be in [0, 1]
    for y in 0..10 {
        for x in 0..10 {
            let v = field.sample(Vec2::new(x as f32 * 0.1, y as f32 * 0.1), &ctx);
            assert!((0.0..=1.0).contains(&v), "Value {} out of range", v);
        }
    }

    // Deterministic
    let v1 = field.sample(Vec2::new(0.5, 0.5), &ctx);
    let v2 = field.sample(Vec2::new(0.5, 0.5), &ctx);
    assert_eq!(v1, v2);
}

#[test]
fn test_white_noise_3d() {
    let field = WhiteNoise3D::with_seed(42);
    let ctx = EvalContext::new();

    // Values should be in [0, 1]
    for z in 0..5 {
        for y in 0..5 {
            for x in 0..5 {
                let v = field.sample(
                    Vec3::new(x as f32 * 0.1, y as f32 * 0.1, z as f32 * 0.1),
                    &ctx,
                );
                assert!((0.0..=1.0).contains(&v), "Value {} out of range", v);
            }
        }
    }

    // Useful for temporal dithering: same (x,y,t) = same value
    let v1 = field.sample(Vec3::new(0.5, 0.5, 0.0), &ctx);
    let v2 = field.sample(Vec3::new(0.5, 0.5, 0.0), &ctx);
    assert_eq!(v1, v2);

    // Different time = different value (for animation)
    let v3 = field.sample(Vec3::new(0.5, 0.5, 1.0), &ctx);
    assert_ne!(v1, v3);
}

#[test]
fn test_checkerboard() {
    let field = Checkerboard::new();
    let ctx = EvalContext::new();

    assert_eq!(field.sample(Vec2::new(0.5, 0.5), &ctx), 1.0);
    assert_eq!(field.sample(Vec2::new(1.5, 0.5), &ctx), 0.0);
}

#[test]
fn test_distance_circle() {
    let field = DistanceCircle::new(Vec2::ZERO, 1.0);
    let ctx = EvalContext::new();

    assert!(field.sample(Vec2::ZERO, &ctx) < 0.0);
    assert!((field.sample(Vec2::new(1.0, 0.0), &ctx) - 0.0).abs() < 0.001);
    assert!(field.sample(Vec2::new(2.0, 0.0), &ctx) > 0.0);
}

#[test]
fn test_metaballs() {
    let balls = vec![Metaball::new_2d(Vec2::ZERO, 1.0)];
    let field = Metaballs2D::new(balls);
    let ctx = EvalContext::new();

    let center_val = field.sample(Vec2::ZERO, &ctx);
    assert!(center_val > 10.0);

    let radius_val = field.sample(Vec2::new(1.0, 0.0), &ctx);
    assert!((radius_val - 1.0).abs() < 0.01);
}

// Terrain generation tests

#[test]
fn test_terrain_basic() {
    let terrain = Terrain2D::new();
    let ctx = EvalContext::new();

    // Sample at various points
    let v1 = terrain.sample(Vec2::new(0.0, 0.0), &ctx);
    let v2 = terrain.sample(Vec2::new(1.0, 1.0), &ctx);
    let v3 = terrain.sample(Vec2::new(10.0, 10.0), &ctx);

    // All values should be in [0, 1] range
    assert!((0.0..=1.0).contains(&v1));
    assert!((0.0..=1.0).contains(&v2));
    assert!((0.0..=1.0).contains(&v3));

    // Values should vary (not all the same)
    assert!((v1 - v2).abs() > 0.001 || (v2 - v3).abs() > 0.001);
}

#[test]
fn test_terrain_presets() {
    let ctx = EvalContext::new();
    let point = Vec2::new(0.5, 0.5);

    // Test all presets produce valid values
    let hills = Terrain2D::rolling_hills().sample(point, &ctx);
    let mountains = Terrain2D::mountains().sample(point, &ctx);
    let plains = Terrain2D::plains().sample(point, &ctx);
    let canyons = Terrain2D::canyons().sample(point, &ctx);

    assert!((0.0..=1.0).contains(&hills));
    assert!((0.0..=1.0).contains(&mountains));
    assert!((0.0..=1.0).contains(&plains));
    assert!((0.0..=1.0).contains(&canyons));
}

#[test]
fn test_terrain_deterministic() {
    let ctx = EvalContext::new();
    let terrain1 = Terrain2D {
        seed: 42,
        ..Default::default()
    };
    let terrain2 = Terrain2D {
        seed: 42,
        ..Default::default()
    };

    let point = Vec2::new(0.5, 0.5);
    assert_eq!(terrain1.sample(point, &ctx), terrain2.sample(point, &ctx));
}

#[test]
fn test_ridged_terrain() {
    let terrain = RidgedTerrain2D::new();
    let ctx = EvalContext::new();

    let v = terrain.sample(Vec2::new(0.5, 0.5), &ctx);
    assert!((0.0..=1.0).contains(&v));
}

#[test]
fn test_billowy_terrain() {
    let terrain = BillowyTerrain2D::new();
    let ctx = EvalContext::new();

    let v = terrain.sample(Vec2::new(0.5, 0.5), &ctx);
    assert!((0.0..=1.0).contains(&v));
}

#[test]
fn test_island_terrain() {
    let terrain = IslandTerrain2D {
        radius: 1.0,
        ..Default::default()
    };
    let ctx = EvalContext::new();

    // Center should have terrain
    let center = terrain.sample(Vec2::ZERO, &ctx);
    assert!(center >= 0.0);

    // Far from center should be zero (outside island)
    let far = terrain.sample(Vec2::new(10.0, 10.0), &ctx);
    assert_eq!(far, 0.0);
}

#[test]
fn test_terraced_terrain() {
    let base = Terrain2D::new();
    let terraced = TerracedTerrain2D::new(base, 5);
    let ctx = EvalContext::new();

    let v = terraced.sample(Vec2::new(0.5, 0.5), &ctx);
    assert!((0.0..=1.0).contains(&v));
}

#[test]
fn test_terrain_struct_literal() {
    let terrain = Terrain2D {
        seed: 123,
        octaves: 4,
        lacunarity: 2.5,
        persistence: 0.4,
        scale: 2.0,
        exponent: 1.5,
    };

    assert_eq!(terrain.seed, 123);
    assert_eq!(terrain.octaves, 4);
    assert!((terrain.lacunarity - 2.5).abs() < 0.001);
    assert!((terrain.persistence - 0.4).abs() < 0.001);
    assert!((terrain.scale - 2.0).abs() < 0.001);
    assert!((terrain.exponent - 1.5).abs() < 0.001);
}

// ========================================================================
// Heightmap tests
// ========================================================================

#[test]
fn test_heightmap_new() {
    let hm = Heightmap::new(10, 20);
    assert_eq!(hm.width, 10);
    assert_eq!(hm.height, 20);
    assert_eq!(hm.data.len(), 200);
    assert!(hm.data.iter().all(|&h| h == 0.0));
}

#[test]
fn test_heightmap_get_set() {
    let mut hm = Heightmap::new(10, 10);
    hm.set(5, 5, 1.0);
    assert_eq!(hm.get(5, 5), 1.0);
    assert_eq!(hm.get(0, 0), 0.0);

    // Out of bounds returns 0
    assert_eq!(hm.get(100, 100), 0.0);
}

#[test]
fn test_heightmap_sample() {
    let mut hm = Heightmap::new(4, 4);
    hm.set(1, 1, 0.0);
    hm.set(2, 1, 1.0);
    hm.set(1, 2, 0.0);
    hm.set(2, 2, 1.0);

    // Sample at integer position
    assert!((hm.sample(1.0, 1.0) - 0.0).abs() < 0.001);
    assert!((hm.sample(2.0, 1.0) - 1.0).abs() < 0.001);

    // Sample at midpoint - should interpolate
    let mid = hm.sample(1.5, 1.5);
    assert!((mid - 0.5).abs() < 0.001);
}

#[test]
fn test_heightmap_gradient() {
    let mut hm = Heightmap::new(10, 10);
    // Create a slope: height increases with x
    for y in 0..10 {
        for x in 0..10 {
            hm.set(x, y, x as f32);
        }
    }

    let grad = hm.gradient(5.0, 5.0);
    // Gradient should point in +x direction
    assert!(grad.x > 0.5);
    assert!(grad.y.abs() < 0.1);
}

#[test]
fn test_heightmap_bounds() {
    let mut hm = Heightmap::new(5, 5);
    hm.set(0, 0, -10.0);
    hm.set(2, 2, 20.0);

    let (min, max) = hm.bounds();
    assert_eq!(min, -10.0);
    assert_eq!(max, 20.0);
}

#[test]
fn test_heightmap_normalize() {
    let mut hm = Heightmap::new(5, 5);
    hm.set(0, 0, -10.0);
    hm.set(2, 2, 20.0);
    hm.set(4, 4, 5.0);

    hm.normalize();

    let (min, max) = hm.bounds();
    assert!((min - 0.0).abs() < 0.001);
    assert!((max - 1.0).abs() < 0.001);

    // Check specific values
    assert!((hm.get(0, 0) - 0.0).abs() < 0.001); // -10 -> 0
    assert!((hm.get(2, 2) - 1.0).abs() < 0.001); // 20 -> 1
    assert!((hm.get(4, 4) - 0.5).abs() < 0.001); // 5 -> 0.5
}

#[test]
fn test_heightmap_from_field() {
    let field = Constant::new(0.5f32);
    let hm = Heightmap::from_field(&field, 10, 10, 1.0);

    assert_eq!(hm.width, 10);
    assert_eq!(hm.height, 10);
    assert!(hm.data.iter().all(|&h| (h - 0.5).abs() < 0.001));
}

// ========================================================================
// Hydraulic erosion tests
// ========================================================================

#[test]
fn test_hydraulic_erosion_config_default() {
    let config = HydraulicErosionConfig::default();
    assert_eq!(config.iterations, 10000);
    assert_eq!(config.max_lifetime, 64);
    assert_eq!(config.brush_radius, 3);
}

#[test]
fn test_hydraulic_erosion_modifies_terrain() {
    // Create a simple cone terrain
    let mut hm = Heightmap::new(32, 32);
    let cx = 16.0;
    let cy = 16.0;

    for y in 0..32 {
        for x in 0..32 {
            let dx = x as f32 - cx;
            let dy = y as f32 - cy;
            let dist = (dx * dx + dy * dy).sqrt();
            hm.set(x, y, (16.0 - dist).max(0.0));
        }
    }

    let original_sum: f32 = hm.data.iter().sum();

    let config = HydraulicErosionConfig {
        iterations: 1000,
        max_lifetime: 30,
        brush_radius: 2,
        ..Default::default()
    };

    hydraulic_erosion(&mut hm, &config, 12345);

    // Terrain should be modified (not identical to original)
    let new_sum: f32 = hm.data.iter().sum();
    assert!(new_sum != original_sum, "terrain should be modified");

    // Most material should remain (within bounds)
    assert!(new_sum > 0.0, "terrain should have material");

    // Heights should still be reasonable
    assert!(hm.data.iter().all(|&h| (-5.0..25.0).contains(&h)));
}

#[test]
fn test_hydraulic_erosion_deterministic() {
    let create_terrain = || {
        let mut hm = Heightmap::new(16, 16);
        for y in 0..16 {
            for x in 0..16 {
                hm.set(x, y, ((x + y) as f32 / 30.0).sin() * 5.0);
            }
        }
        hm
    };

    let config = HydraulicErosionConfig {
        iterations: 500,
        ..Default::default()
    };

    let mut hm1 = create_terrain();
    let mut hm2 = create_terrain();

    hydraulic_erosion(&mut hm1, &config, 42);
    hydraulic_erosion(&mut hm2, &config, 42);

    // Same seed should produce same result
    assert_eq!(hm1.data, hm2.data);

    // Different seed should produce different result
    let mut hm3 = create_terrain();
    hydraulic_erosion(&mut hm3, &config, 999);
    assert_ne!(hm1.data, hm3.data);
}

// ========================================================================
// Thermal erosion tests
// ========================================================================

#[test]
fn test_thermal_erosion_config_default() {
    let config = ThermalErosionConfig::default();
    assert_eq!(config.iterations, 50);
    assert!((config.talus_angle - 0.8).abs() < 0.001);
    assert!((config.transfer_rate - 0.5).abs() < 0.001);
}

#[test]
fn test_thermal_erosion_smooths_steep_slopes() {
    // Create terrain with a sharp spike
    let mut hm = Heightmap::new(16, 16);
    hm.set(8, 8, 10.0); // Sharp spike

    // Calculate initial max slope
    let initial_spike = hm.get(8, 8);

    let config = ThermalErosionConfig {
        iterations: 100,
        talus_angle: 0.5,
        transfer_rate: 0.5,
    };

    thermal_erosion(&mut hm, &config);

    // Spike should be reduced
    assert!(hm.get(8, 8) < initial_spike);

    // Material should have spread to neighbors
    assert!(hm.get(7, 8) > 0.0);
    assert!(hm.get(9, 8) > 0.0);
    assert!(hm.get(8, 7) > 0.0);
    assert!(hm.get(8, 9) > 0.0);
}

#[test]
fn test_thermal_erosion_preserves_mass() {
    let mut hm = Heightmap::new(16, 16);
    // Create terrain where mass is away from edges
    for y in 4..12 {
        for x in 4..12 {
            hm.set(x, y, 5.0 + ((x + y) as f32 * 0.1));
        }
    }

    let original_sum: f32 = hm.data.iter().sum();

    let config = ThermalErosionConfig::default();
    thermal_erosion(&mut hm, &config);

    let new_sum: f32 = hm.data.iter().sum();

    // Mass should be conserved (within floating point tolerance)
    assert!((new_sum - original_sum).abs() < 0.001);
}

#[test]
fn test_thermal_erosion_flat_terrain_unchanged() {
    let mut hm = Heightmap::new(16, 16);
    // Flat terrain
    for y in 0..16 {
        for x in 0..16 {
            hm.set(x, y, 5.0);
        }
    }

    let original_data = hm.data.clone();

    let config = ThermalErosionConfig::default();
    thermal_erosion(&mut hm, &config);

    // Flat terrain should remain unchanged
    assert_eq!(hm.data, original_data);
}

#[test]
fn test_combined_erosion() {
    // Create initial terrain from noise
    let terrain = Terrain2D {
        seed: 42,
        ..Default::default()
    };
    let mut hm = Heightmap::from_field(&terrain, 32, 32, 0.1);
    hm.normalize();

    // Scale up for erosion
    for h in &mut hm.data {
        *h *= 10.0;
    }

    // Apply both erosion types
    let hydro_config = HydraulicErosionConfig {
        iterations: 500,
        ..Default::default()
    };
    hydraulic_erosion(&mut hm, &hydro_config, 12345);

    let thermal_config = ThermalErosionConfig::default();
    thermal_erosion(&mut hm, &thermal_config);

    // Result should be valid terrain
    assert!(hm.data.iter().all(|&h| h.is_finite()));
    let (min, max) = hm.bounds();
    assert!(min < max);
}

// ====== Network Tests ======

#[test]
fn test_network_node() {
    let node = NetworkNode {
        position: Vec2::new(10.0, 20.0),
        height: 5.0,
        importance: 2.0,
    };

    assert_eq!(node.position, Vec2::new(10.0, 20.0));
    assert_eq!(node.height, 5.0);
    assert_eq!(node.importance, 2.0);
}

#[test]
fn test_network_edge() {
    let edge = NetworkEdge {
        start: 0,
        end: 1,
        weight: 10.0,
        path: vec![Vec2::new(5.0, 5.0)],
        edge_type: 1.0,
    };

    assert_eq!(edge.start, 0);
    assert_eq!(edge.end, 1);
    assert_eq!(edge.weight, 10.0);
    assert_eq!(edge.path.len(), 1);
}

#[test]
fn test_network_creation() {
    let mut network = Network::new();
    let n0 = network.add_node(NetworkNode::new(Vec2::ZERO));
    let n1 = network.add_node(NetworkNode::new(Vec2::new(10.0, 0.0)));
    let n2 = network.add_node(NetworkNode::new(Vec2::new(5.0, 10.0)));

    network.add_edge(n0, n1);
    network.add_edge(n1, n2);
    network.add_edge(n2, n0);

    assert_eq!(network.nodes.len(), 3);
    assert_eq!(network.edges.len(), 3);
}

#[test]
fn test_network_neighbors() {
    let mut network = Network::new();
    let n0 = network.add_node(NetworkNode::new(Vec2::ZERO));
    let n1 = network.add_node(NetworkNode::new(Vec2::new(10.0, 0.0)));
    let n2 = network.add_node(NetworkNode::new(Vec2::new(0.0, 10.0)));

    network.add_edge(n0, n1);
    network.add_edge(n0, n2);

    let neighbors = network.neighbors(n0);
    assert_eq!(neighbors.len(), 2);
    assert!(neighbors.contains(&n1));
    assert!(neighbors.contains(&n2));
}

#[test]
fn test_network_edge_path() {
    let mut network = Network::new();
    let n0 = network.add_node(NetworkNode::new(Vec2::ZERO));
    let n1 = network.add_node(NetworkNode::new(Vec2::new(10.0, 0.0)));

    network.add_edge(n0, n1).path = vec![Vec2::new(3.0, 1.0), Vec2::new(7.0, 1.0)];

    let full_path = network.edge_path(&network.edges[0]);
    assert_eq!(full_path.len(), 4); // start + 2 intermediate + end
    assert_eq!(full_path[0], Vec2::ZERO);
    assert_eq!(full_path[3], Vec2::new(10.0, 0.0));
}

#[test]
fn test_road_network_config_default() {
    let config = RoadNetworkConfig::default();
    assert!(config.num_nodes > 0);
    assert!(config.use_mst);
}

#[test]
fn test_generate_road_network() {
    let config = RoadNetworkConfig {
        num_nodes: 8,
        bounds: (0.0, 0.0, 50.0, 50.0),
        ..Default::default()
    };

    let network = generate_road_network(&config, 42);

    assert_eq!(network.nodes.len(), 8);
    // MST should have n-1 edges plus some extras
    assert!(network.edges.len() >= 7);

    // All nodes should be within bounds
    for node in &network.nodes {
        assert!(node.position.x >= 0.0 && node.position.x <= 50.0);
        assert!(node.position.y >= 0.0 && node.position.y <= 50.0);
    }
}

#[test]
fn test_road_network_connected() {
    let config = RoadNetworkConfig {
        num_nodes: 5,
        use_mst: true,
        extra_connectivity: 0.0,
        ..Default::default()
    };

    let network = generate_road_network(&config, 123);

    // MST with 5 nodes should have exactly 4 edges
    assert_eq!(network.edges.len(), 4);
}

#[test]
fn test_river_network_config_default() {
    let config = RiverNetworkConfig::default();
    assert!(config.num_sources > 0);
    assert!(config.max_steps > 0);
}

#[test]
fn test_generate_river_network() {
    // Create a sloped terrain
    let mut hm = Heightmap::new(32, 32);
    for y in 0..32 {
        for x in 0..32 {
            // Height decreases from top-left to bottom-right
            let h = 1.0 - (x + y) as f32 / 62.0;
            hm.set(x, y, h);
        }
    }

    let config = RiverNetworkConfig {
        num_sources: 2,
        source_min_height: 0.7,
        ..Default::default()
    };

    let network = generate_river_network(&hm, &config, 42);

    // Should have at least some nodes and edges
    assert!(!network.nodes.is_empty());
}

#[test]
fn test_flow_accumulation() {
    // Create a simple bowl-shaped terrain
    let mut hm = Heightmap::new(16, 16);
    for y in 0..16 {
        for x in 0..16 {
            let dx = x as f32 - 7.5;
            let dy = y as f32 - 7.5;
            let h = (dx * dx + dy * dy).sqrt() / 10.0;
            hm.set(x, y, h);
        }
    }

    let flow = compute_flow_accumulation(&hm);

    // Center should have highest flow (everything drains there)
    let center_flow = flow.get(7, 7);
    let edge_flow = flow.get(0, 0);
    assert!(center_flow > edge_flow);
}

#[test]
fn test_network_sample_edges() {
    let mut network = Network::new();
    let n0 = network.add_node(NetworkNode::new(Vec2::ZERO));
    let n1 = network.add_node(NetworkNode::new(Vec2::new(20.0, 0.0)));

    network.add_edge(n0, n1);

    let sampled = network.sample_edges(5.0);
    assert_eq!(sampled.len(), 1);

    // Should have multiple points along the edge
    assert!(sampled[0].len() >= 4);
}

#[test]
fn test_road_network_deterministic() {
    let config = RoadNetworkConfig::default();

    let network1 = generate_road_network(&config, 12345);
    let network2 = generate_road_network(&config, 12345);

    // Same seed should produce same network
    assert_eq!(network1.nodes.len(), network2.nodes.len());
    assert_eq!(network1.edges.len(), network2.edges.len());

    for (n1, n2) in network1.nodes.iter().zip(network2.nodes.iter()) {
        assert_eq!(n1.position, n2.position);
    }
}

// 2D SDF primitive tests

#[test]
fn test_distance_rounded_box() {
    let ctx = EvalContext::new();
    let sdf = DistanceRoundedBox::new(Vec2::ZERO, Vec2::new(1.0, 0.5), 0.1);

    // Center should be inside (negative)
    assert!(sdf.sample(Vec2::ZERO, &ctx) < 0.0);

    // Far away should be outside (positive)
    assert!(sdf.sample(Vec2::new(5.0, 0.0), &ctx) > 0.0);

    // At edge should be close to zero
    let edge_val = sdf.sample(Vec2::new(1.0, 0.0), &ctx).abs();
    assert!(edge_val < 0.2);
}

#[test]
fn test_distance_ellipse() {
    let ctx = EvalContext::new();
    let sdf = DistanceEllipse::new(Vec2::ZERO, Vec2::new(2.0, 1.0));

    // Center should be inside (negative)
    assert!(sdf.sample(Vec2::ZERO, &ctx) < 0.0);

    // On the ellipse boundary should be close to zero
    let on_x = sdf.sample(Vec2::new(2.0, 0.0), &ctx);
    let on_y = sdf.sample(Vec2::new(0.0, 1.0), &ctx);
    assert!(on_x.abs() < 0.1);
    assert!(on_y.abs() < 0.1);

    // Outside should be positive
    assert!(sdf.sample(Vec2::new(5.0, 0.0), &ctx) > 0.0);
}

#[test]
fn test_distance_capsule() {
    let ctx = EvalContext::new();
    let sdf = DistanceCapsule::new(Vec2::new(-1.0, 0.0), Vec2::new(1.0, 0.0), 0.5);

    // Center should be inside
    assert!(sdf.sample(Vec2::ZERO, &ctx) < 0.0);

    // At ends should be at edge
    let end_val = sdf.sample(Vec2::new(1.5, 0.0), &ctx).abs();
    assert!(end_val < 0.1);

    // Outside should be positive
    assert!(sdf.sample(Vec2::new(3.0, 0.0), &ctx) > 0.0);
}

#[test]
fn test_distance_triangle() {
    let ctx = EvalContext::new();
    let sdf = DistanceTriangle::new(
        Vec2::new(0.0, 1.0),
        Vec2::new(-1.0, -1.0),
        Vec2::new(1.0, -1.0),
    );

    // Center should be inside (negative)
    assert!(sdf.sample(Vec2::ZERO, &ctx) < 0.0);

    // Far outside should be positive
    assert!(sdf.sample(Vec2::new(5.0, 0.0), &ctx) > 0.0);
}

#[test]
fn test_distance_regular_polygon() {
    let ctx = EvalContext::new();

    // Hexagon
    let hex = DistanceRegularPolygon::new(Vec2::ZERO, 1.0, 6);

    // Center should be inside
    assert!(hex.sample(Vec2::ZERO, &ctx) < 0.0);

    // Outside should be positive
    assert!(hex.sample(Vec2::new(2.0, 0.0), &ctx) > 0.0);

    // Square
    let square = DistanceRegularPolygon::new(Vec2::ZERO, 1.0, 4);
    assert!(square.sample(Vec2::ZERO, &ctx) < 0.0);
}

#[test]
fn test_distance_arc() {
    let ctx = EvalContext::new();

    // Pie shape (filled arc)
    let pie = DistanceArc::pie(Vec2::ZERO, 1.0, std::f32::consts::FRAC_PI_4);

    // Center of pie should be inside
    assert!(pie.sample(Vec2::new(0.3, 0.0), &ctx) < 0.0);

    // Outside the angle should be positive
    assert!(pie.sample(Vec2::new(0.0, 0.5), &ctx) > 0.0);

    // Far outside radius should be positive
    assert!(pie.sample(Vec2::new(3.0, 0.0), &ctx) > 0.0);
}
