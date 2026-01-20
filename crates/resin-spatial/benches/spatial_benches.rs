//! Benchmarks for spatial data structures.

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use glam::{Vec2, Vec3};
use rhizome_resin_spatial::{Aabb2, Aabb3, Bvh, Octree, Quadtree, Ray, Rtree, SpatialHash};

// ============================================================================
// Quadtree Benchmarks
// ============================================================================

fn bench_quadtree_insert(c: &mut Criterion) {
    c.bench_function("quadtree_insert_1000", |b| {
        b.iter(|| {
            let bounds = Aabb2::new(Vec2::ZERO, Vec2::splat(100.0));
            let mut tree = Quadtree::new(bounds, 8, 4);
            for i in 0..1000 {
                let x = (i as f32) % 100.0;
                let y = (i as f32 / 100.0).floor();
                tree.insert(Vec2::new(x, y), i);
            }
            black_box(tree)
        })
    });
}

fn bench_quadtree_insert_10000(c: &mut Criterion) {
    c.bench_function("quadtree_insert_10000", |b| {
        b.iter(|| {
            let bounds = Aabb2::new(Vec2::ZERO, Vec2::splat(1000.0));
            let mut tree = Quadtree::new(bounds, 10, 8);
            for i in 0..10000 {
                let x = (i as f32) % 1000.0;
                let y = (i as f32 / 1000.0).floor();
                tree.insert(Vec2::new(x, y), i);
            }
            black_box(tree)
        })
    });
}

fn bench_quadtree_query_point(c: &mut Criterion) {
    let bounds = Aabb2::new(Vec2::ZERO, Vec2::splat(100.0));
    let mut tree = Quadtree::new(bounds, 8, 4);
    for i in 0..1000 {
        let x = (i as f32) % 100.0;
        let y = (i as f32 / 100.0).floor();
        tree.insert(Vec2::new(x, y), i);
    }

    c.bench_function("quadtree_nearest_1000", |b| {
        b.iter(|| {
            let query = Vec2::new(50.0, 50.0);
            black_box(tree.nearest(query))
        })
    });
}

fn bench_quadtree_query_region(c: &mut Criterion) {
    let bounds = Aabb2::new(Vec2::ZERO, Vec2::splat(100.0));
    let mut tree = Quadtree::new(bounds, 8, 4);
    for i in 0..1000 {
        let x = (i as f32) % 100.0;
        let y = (i as f32 / 100.0).floor();
        tree.insert(Vec2::new(x, y), i);
    }

    c.bench_function("quadtree_query_region_1000", |b| {
        b.iter(|| {
            let query = Aabb2::new(Vec2::new(25.0, 25.0), Vec2::new(75.0, 75.0));
            let results: Vec<_> = tree.query_region(&query).collect();
            black_box(results)
        })
    });
}

fn bench_quadtree_query_region_small(c: &mut Criterion) {
    let bounds = Aabb2::new(Vec2::ZERO, Vec2::splat(100.0));
    let mut tree = Quadtree::new(bounds, 8, 4);
    for i in 0..1000 {
        let x = (i as f32) % 100.0;
        let y = (i as f32 / 100.0).floor();
        tree.insert(Vec2::new(x, y), i);
    }

    c.bench_function("quadtree_query_region_small_1000", |b| {
        b.iter(|| {
            let query = Aabb2::new(Vec2::new(45.0, 45.0), Vec2::new(55.0, 55.0));
            let results: Vec<_> = tree.query_region(&query).collect();
            black_box(results)
        })
    });
}

fn bench_quadtree_k_nearest(c: &mut Criterion) {
    let bounds = Aabb2::new(Vec2::ZERO, Vec2::splat(100.0));
    let mut tree = Quadtree::new(bounds, 8, 4);
    for i in 0..1000 {
        let x = (i as f32) % 100.0;
        let y = (i as f32 / 100.0).floor();
        tree.insert(Vec2::new(x, y), i);
    }

    c.bench_function("quadtree_k_nearest_10_from_1000", |b| {
        b.iter(|| {
            let query = Vec2::new(50.0, 50.0);
            black_box(tree.k_nearest(query, 10))
        })
    });
}

// ============================================================================
// Octree Benchmarks
// ============================================================================

fn bench_octree_insert(c: &mut Criterion) {
    c.bench_function("octree_insert_1000", |b| {
        b.iter(|| {
            let bounds = Aabb3::new(Vec3::ZERO, Vec3::splat(100.0));
            let mut tree = Octree::new(bounds, 8, 4);
            for i in 0..1000 {
                let x = (i as f32) % 10.0 * 10.0;
                let y = ((i as f32 / 10.0).floor() % 10.0) * 10.0;
                let z = (i as f32 / 100.0).floor() * 10.0;
                tree.insert(Vec3::new(x, y, z), i);
            }
            black_box(tree)
        })
    });
}

fn bench_octree_insert_10000(c: &mut Criterion) {
    c.bench_function("octree_insert_10000", |b| {
        b.iter(|| {
            let bounds = Aabb3::new(Vec3::ZERO, Vec3::splat(1000.0));
            let mut tree = Octree::new(bounds, 10, 8);
            for i in 0..10000 {
                let x = (i as f32) % 100.0 * 10.0;
                let y = ((i as f32 / 100.0).floor() % 100.0) * 10.0;
                let z = (i as f32 / 10000.0).floor() * 10.0;
                tree.insert(Vec3::new(x, y, z), i);
            }
            black_box(tree)
        })
    });
}

fn bench_octree_query_region(c: &mut Criterion) {
    let bounds = Aabb3::new(Vec3::ZERO, Vec3::splat(100.0));
    let mut tree = Octree::new(bounds, 8, 4);
    for i in 0..1000 {
        let x = (i as f32) % 10.0 * 10.0;
        let y = ((i as f32 / 10.0).floor() % 10.0) * 10.0;
        let z = (i as f32 / 100.0).floor() * 10.0;
        tree.insert(Vec3::new(x, y, z), i);
    }

    c.bench_function("octree_query_region_1000", |b| {
        b.iter(|| {
            let query = Aabb3::new(Vec3::new(25.0, 25.0, 25.0), Vec3::new(75.0, 75.0, 75.0));
            let results: Vec<_> = tree.query_region(&query).collect();
            black_box(results)
        })
    });
}

fn bench_octree_nearest(c: &mut Criterion) {
    let bounds = Aabb3::new(Vec3::ZERO, Vec3::splat(100.0));
    let mut tree = Octree::new(bounds, 8, 4);
    for i in 0..1000 {
        let x = (i as f32) % 10.0 * 10.0;
        let y = ((i as f32 / 10.0).floor() % 10.0) * 10.0;
        let z = (i as f32 / 100.0).floor() * 10.0;
        tree.insert(Vec3::new(x, y, z), i);
    }

    c.bench_function("octree_nearest_1000", |b| {
        b.iter(|| {
            let query = Vec3::new(50.0, 50.0, 50.0);
            black_box(tree.nearest(query))
        })
    });
}

fn bench_octree_k_nearest(c: &mut Criterion) {
    let bounds = Aabb3::new(Vec3::ZERO, Vec3::splat(100.0));
    let mut tree = Octree::new(bounds, 8, 4);
    for i in 0..1000 {
        let x = (i as f32) % 10.0 * 10.0;
        let y = ((i as f32 / 10.0).floor() % 10.0) * 10.0;
        let z = (i as f32 / 100.0).floor() * 10.0;
        tree.insert(Vec3::new(x, y, z), i);
    }

    c.bench_function("octree_k_nearest_10_from_1000", |b| {
        b.iter(|| {
            let query = Vec3::new(50.0, 50.0, 50.0);
            black_box(tree.k_nearest(query, 10))
        })
    });
}

// ============================================================================
// BVH Benchmarks
// ============================================================================

fn bench_bvh_build(c: &mut Criterion) {
    c.bench_function("bvh_build_1000", |b| {
        b.iter(|| {
            let primitives: Vec<_> = (0..1000)
                .map(|i| {
                    let x = (i as f32) % 10.0 * 10.0;
                    let y = ((i as f32 / 10.0).floor() % 10.0) * 10.0;
                    let z = (i as f32 / 100.0).floor() * 10.0;
                    let min = Vec3::new(x, y, z);
                    let max = min + Vec3::splat(5.0);
                    (Aabb3::new(min, max), i)
                })
                .collect();
            black_box(Bvh::build(primitives))
        })
    });
}

fn bench_bvh_build_10000(c: &mut Criterion) {
    c.bench_function("bvh_build_10000", |b| {
        b.iter(|| {
            let primitives: Vec<_> = (0..10000)
                .map(|i| {
                    let x = (i as f32) % 100.0 * 10.0;
                    let y = ((i as f32 / 100.0).floor() % 100.0) * 10.0;
                    let z = (i as f32 / 10000.0).floor() * 10.0;
                    let min = Vec3::new(x, y, z);
                    let max = min + Vec3::splat(5.0);
                    (Aabb3::new(min, max), i)
                })
                .collect();
            black_box(Bvh::build(primitives))
        })
    });
}

fn bench_bvh_ray_query(c: &mut Criterion) {
    let primitives: Vec<_> = (0..1000)
        .map(|i| {
            let x = (i as f32) % 10.0 * 10.0;
            let y = ((i as f32 / 10.0).floor() % 10.0) * 10.0;
            let z = (i as f32 / 100.0).floor() * 10.0;
            let min = Vec3::new(x, y, z);
            let max = min + Vec3::splat(5.0);
            (Aabb3::new(min, max), i)
        })
        .collect();
    let bvh = Bvh::build(primitives);

    c.bench_function("bvh_ray_query_1000", |b| {
        b.iter(|| {
            let ray = Ray::new(Vec3::new(50.0, 50.0, -10.0), Vec3::Z);
            black_box(bvh.intersect_ray(&ray))
        })
    });
}

fn bench_bvh_ray_query_closest(c: &mut Criterion) {
    let primitives: Vec<_> = (0..1000)
        .map(|i| {
            let x = (i as f32) % 10.0 * 10.0;
            let y = ((i as f32 / 10.0).floor() % 10.0) * 10.0;
            let z = (i as f32 / 100.0).floor() * 10.0;
            let min = Vec3::new(x, y, z);
            let max = min + Vec3::splat(5.0);
            (Aabb3::new(min, max), i)
        })
        .collect();
    let bvh = Bvh::build(primitives);

    c.bench_function("bvh_ray_query_closest_1000", |b| {
        b.iter(|| {
            let ray = Ray::new(Vec3::new(50.0, 50.0, -10.0), Vec3::Z);
            black_box(bvh.intersect_ray_closest(&ray))
        })
    });
}

fn bench_bvh_aabb_query(c: &mut Criterion) {
    let primitives: Vec<_> = (0..1000)
        .map(|i| {
            let x = (i as f32) % 10.0 * 10.0;
            let y = ((i as f32 / 10.0).floor() % 10.0) * 10.0;
            let z = (i as f32 / 100.0).floor() * 10.0;
            let min = Vec3::new(x, y, z);
            let max = min + Vec3::splat(5.0);
            (Aabb3::new(min, max), i)
        })
        .collect();
    let bvh = Bvh::build(primitives);

    c.bench_function("bvh_aabb_query_1000", |b| {
        b.iter(|| {
            let query = Aabb3::new(Vec3::new(40.0, 40.0, 40.0), Vec3::new(60.0, 60.0, 60.0));
            black_box(bvh.query_aabb(&query))
        })
    });
}

// ============================================================================
// SpatialHash Benchmarks
// ============================================================================

fn bench_spatial_hash_insert(c: &mut Criterion) {
    c.bench_function("spatial_hash_insert_1000", |b| {
        b.iter(|| {
            let mut hash = SpatialHash::new(10.0);
            for i in 0..1000 {
                let x = (i as f32) % 10.0 * 10.0;
                let y = ((i as f32 / 10.0).floor() % 10.0) * 10.0;
                let z = (i as f32 / 100.0).floor() * 10.0;
                hash.insert(Vec3::new(x, y, z), i);
            }
            black_box(hash)
        })
    });
}

fn bench_spatial_hash_insert_10000(c: &mut Criterion) {
    c.bench_function("spatial_hash_insert_10000", |b| {
        b.iter(|| {
            let mut hash = SpatialHash::new(10.0);
            for i in 0..10000 {
                let x = (i as f32) % 100.0 * 10.0;
                let y = ((i as f32 / 100.0).floor() % 100.0) * 10.0;
                let z = (i as f32 / 10000.0).floor() * 10.0;
                hash.insert(Vec3::new(x, y, z), i);
            }
            black_box(hash)
        })
    });
}

fn bench_spatial_hash_query_cell(c: &mut Criterion) {
    let mut hash = SpatialHash::new(10.0);
    for i in 0..1000 {
        let x = (i as f32) % 10.0 * 10.0;
        let y = ((i as f32 / 10.0).floor() % 10.0) * 10.0;
        let z = (i as f32 / 100.0).floor() * 10.0;
        hash.insert(Vec3::new(x, y, z), i);
    }

    c.bench_function("spatial_hash_query_cell_1000", |b| {
        b.iter(|| {
            let query = Vec3::new(50.0, 50.0, 50.0);
            let results: Vec<_> = hash.query_cell(query).collect();
            black_box(results)
        })
    });
}

fn bench_spatial_hash_query_neighbors(c: &mut Criterion) {
    let mut hash = SpatialHash::new(10.0);
    for i in 0..1000 {
        let x = (i as f32) % 10.0 * 10.0;
        let y = ((i as f32 / 10.0).floor() % 10.0) * 10.0;
        let z = (i as f32 / 100.0).floor() * 10.0;
        hash.insert(Vec3::new(x, y, z), i);
    }

    c.bench_function("spatial_hash_query_neighbors_1000", |b| {
        b.iter(|| {
            let query = Vec3::new(50.0, 50.0, 50.0);
            let results: Vec<_> = hash.query_neighbors(query).collect();
            black_box(results)
        })
    });
}

fn bench_spatial_hash_query_radius(c: &mut Criterion) {
    let mut hash = SpatialHash::new(10.0);
    for i in 0..1000 {
        let x = (i as f32) % 10.0 * 10.0;
        let y = ((i as f32 / 10.0).floor() % 10.0) * 10.0;
        let z = (i as f32 / 100.0).floor() * 10.0;
        hash.insert(Vec3::new(x, y, z), i);
    }

    c.bench_function("spatial_hash_query_radius_1000", |b| {
        b.iter(|| {
            let query = Vec3::new(50.0, 50.0, 50.0);
            let results: Vec<_> = hash.query_radius(query, 25.0).collect();
            black_box(results)
        })
    });
}

// ============================================================================
// R-tree Benchmarks
// ============================================================================

fn bench_rtree_insert(c: &mut Criterion) {
    c.bench_function("rtree_insert_1000", |b| {
        b.iter(|| {
            let mut tree = Rtree::new(4);
            for i in 0..1000 {
                let x = (i as f32) % 100.0;
                let y = (i as f32 / 100.0).floor();
                let min = Vec2::new(x, y);
                let max = min + Vec2::splat(5.0);
                tree.insert(Aabb2::new(min, max), i);
            }
            black_box(tree)
        })
    });
}

fn bench_rtree_insert_10000(c: &mut Criterion) {
    c.bench_function("rtree_insert_10000", |b| {
        b.iter(|| {
            let mut tree = Rtree::new(8);
            for i in 0..10000 {
                let x = (i as f32) % 1000.0;
                let y = (i as f32 / 1000.0).floor();
                let min = Vec2::new(x, y);
                let max = min + Vec2::splat(5.0);
                tree.insert(Aabb2::new(min, max), i);
            }
            black_box(tree)
        })
    });
}

fn bench_rtree_query(c: &mut Criterion) {
    let mut tree = Rtree::new(4);
    for i in 0..1000 {
        let x = (i as f32) % 100.0;
        let y = (i as f32 / 100.0).floor();
        let min = Vec2::new(x, y);
        let max = min + Vec2::splat(5.0);
        tree.insert(Aabb2::new(min, max), i);
    }

    c.bench_function("rtree_query_region_1000", |b| {
        b.iter(|| {
            let query = Aabb2::new(Vec2::new(25.0, 2.0), Vec2::new(75.0, 8.0));
            black_box(tree.query(&query))
        })
    });
}

fn bench_rtree_query_point(c: &mut Criterion) {
    let mut tree = Rtree::new(4);
    for i in 0..1000 {
        let x = (i as f32) % 100.0;
        let y = (i as f32 / 100.0).floor();
        let min = Vec2::new(x, y);
        let max = min + Vec2::splat(5.0);
        tree.insert(Aabb2::new(min, max), i);
    }

    c.bench_function("rtree_query_point_1000", |b| {
        b.iter(|| {
            let query = Vec2::new(50.0, 5.0);
            black_box(tree.query_point(query))
        })
    });
}

criterion_group!(
    quadtree_benches,
    bench_quadtree_insert,
    bench_quadtree_insert_10000,
    bench_quadtree_query_point,
    bench_quadtree_query_region,
    bench_quadtree_query_region_small,
    bench_quadtree_k_nearest
);

criterion_group!(
    octree_benches,
    bench_octree_insert,
    bench_octree_insert_10000,
    bench_octree_query_region,
    bench_octree_nearest,
    bench_octree_k_nearest
);

criterion_group!(
    bvh_benches,
    bench_bvh_build,
    bench_bvh_build_10000,
    bench_bvh_ray_query,
    bench_bvh_ray_query_closest,
    bench_bvh_aabb_query
);

criterion_group!(
    spatial_hash_benches,
    bench_spatial_hash_insert,
    bench_spatial_hash_insert_10000,
    bench_spatial_hash_query_cell,
    bench_spatial_hash_query_neighbors,
    bench_spatial_hash_query_radius
);

criterion_group!(
    rtree_benches,
    bench_rtree_insert,
    bench_rtree_insert_10000,
    bench_rtree_query,
    bench_rtree_query_point
);

criterion_main!(
    quadtree_benches,
    octree_benches,
    bvh_benches,
    spatial_hash_benches,
    rtree_benches
);
