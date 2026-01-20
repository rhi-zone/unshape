//! Benchmarks for point cloud operations.

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use glam::{Mat4, Vec3};
use rhizome_resin_mesh::Cuboid;
use rhizome_resin_pointcloud::{
    PointCloud, Poisson, RemoveOutliers, crop_to_bounds, estimate_normals, remove_outliers,
    sample_mesh_poisson, sample_mesh_uniform, transform, voxel_downsample,
};

// ============================================================================
// Point Cloud Creation
// ============================================================================

fn bench_pointcloud_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("pointcloud_creation");

    for size in [100, 1000, 10000] {
        group.bench_with_input(BenchmarkId::new("from_positions", size), &size, |b, &n| {
            let points: Vec<Vec3> = (0..n)
                .map(|i| {
                    Vec3::new(
                        (i % 100) as f32,
                        ((i / 100) % 100) as f32,
                        (i / 10000) as f32,
                    )
                })
                .collect();
            b.iter(|| black_box(PointCloud::from_positions(points.clone())))
        });

        group.bench_with_input(
            BenchmarkId::new("from_positions_normals", size),
            &size,
            |b, &n| {
                let points: Vec<Vec3> = (0..n)
                    .map(|i| {
                        Vec3::new(
                            (i % 100) as f32,
                            ((i / 100) % 100) as f32,
                            (i / 10000) as f32,
                        )
                    })
                    .collect();
                let normals: Vec<Vec3> = (0..n).map(|_| Vec3::Y).collect();
                b.iter(|| {
                    black_box(PointCloud::from_positions_normals(
                        points.clone(),
                        normals.clone(),
                    ))
                })
            },
        );
    }

    group.finish();
}

fn bench_point_insertion(c: &mut Criterion) {
    let mut group = c.benchmark_group("point_insertion");

    for count in [100, 1000] {
        group.bench_with_input(BenchmarkId::new("add_point", count), &count, |b, &n| {
            b.iter(|| {
                let mut cloud = PointCloud::new();
                for i in 0..n {
                    cloud.add_point(Vec3::new(i as f32, 0.0, 0.0));
                }
                black_box(cloud)
            })
        });

        group.bench_with_input(
            BenchmarkId::new("add_point_with_normal", count),
            &count,
            |b, &n| {
                b.iter(|| {
                    let mut cloud = PointCloud::new();
                    for i in 0..n {
                        cloud.add_point_with_normal(Vec3::new(i as f32, 0.0, 0.0), Vec3::Y);
                    }
                    black_box(cloud)
                })
            },
        );
    }

    group.finish();
}

// ============================================================================
// Queries and Computations
// ============================================================================

fn bench_bounding_box(c: &mut Criterion) {
    let mut group = c.benchmark_group("bounding_box");

    for size in [100, 1000, 10000] {
        let points: Vec<Vec3> = (0..size)
            .map(|i| {
                Vec3::new(
                    (i % 100) as f32,
                    ((i / 100) % 100) as f32,
                    (i / 10000) as f32,
                )
            })
            .collect();
        let cloud = PointCloud::from_positions(points);

        group.bench_with_input(BenchmarkId::from_parameter(size), &cloud, |b, cloud| {
            b.iter(|| black_box(cloud.bounding_box()))
        });
    }

    group.finish();
}

fn bench_centroid(c: &mut Criterion) {
    let mut group = c.benchmark_group("centroid");

    for size in [100, 1000, 10000] {
        let points: Vec<Vec3> = (0..size)
            .map(|i| {
                Vec3::new(
                    (i % 100) as f32,
                    ((i / 100) % 100) as f32,
                    (i / 10000) as f32,
                )
            })
            .collect();
        let cloud = PointCloud::from_positions(points);

        group.bench_with_input(BenchmarkId::from_parameter(size), &cloud, |b, cloud| {
            b.iter(|| black_box(cloud.centroid()))
        });
    }

    group.finish();
}

// ============================================================================
// Mesh Sampling
// ============================================================================

fn bench_sample_mesh_uniform(c: &mut Criterion) {
    let mut group = c.benchmark_group("sample_mesh_uniform");

    let mesh = Cuboid::unit().apply();

    for count in [100, 1000, 5000] {
        group.bench_with_input(BenchmarkId::from_parameter(count), &count, |b, &n| {
            b.iter(|| black_box(sample_mesh_uniform(&mesh, n)))
        });
    }

    group.finish();
}

fn bench_sample_mesh_poisson(c: &mut Criterion) {
    let mut group = c.benchmark_group("sample_mesh_poisson");
    group.sample_size(20); // Poisson sampling is slower

    let mesh = Cuboid::unit().apply();

    for min_distance in [0.05, 0.1, 0.2] {
        let config = Poisson::new(min_distance);
        group.bench_with_input(
            BenchmarkId::new("min_dist", format!("{:.2}", min_distance)),
            &config,
            |b, config| b.iter(|| black_box(sample_mesh_poisson(&mesh, config))),
        );
    }

    group.finish();
}

// ============================================================================
// Normal Estimation
// ============================================================================

fn bench_estimate_normals(c: &mut Criterion) {
    let mut group = c.benchmark_group("estimate_normals");
    group.sample_size(20); // Normal estimation is O(n^2) so limit iterations

    // Create point clouds without normals
    let mesh = Cuboid::unit().apply();

    for size in [50, 100, 200] {
        let cloud = sample_mesh_uniform(&mesh, size);
        let cloud_no_normals = PointCloud::from_positions(cloud.positions);

        for k in [5, 10] {
            group.bench_with_input(
                BenchmarkId::new(format!("k{}", k), size),
                &(cloud_no_normals.clone(), k),
                |b, (cloud, k)| b.iter(|| black_box(estimate_normals(cloud, *k))),
            );
        }
    }

    group.finish();
}

// ============================================================================
// Filtering Operations
// ============================================================================

fn bench_voxel_downsample(c: &mut Criterion) {
    let mut group = c.benchmark_group("voxel_downsample");

    let mesh = Cuboid::unit().apply();
    let cloud = sample_mesh_uniform(&mesh, 5000);

    for voxel_size in [0.05, 0.1, 0.2] {
        group.bench_with_input(
            BenchmarkId::new("voxel_size", format!("{:.2}", voxel_size)),
            &voxel_size,
            |b, &size| b.iter(|| black_box(voxel_downsample(&cloud, size))),
        );
    }

    group.finish();
}

fn bench_remove_outliers(c: &mut Criterion) {
    let mut group = c.benchmark_group("remove_outliers");
    group.sample_size(20); // O(n^2) operation

    let mesh = Cuboid::unit().apply();

    for size in [100, 200, 500] {
        let cloud = sample_mesh_uniform(&mesh, size);

        for k in [5, 10] {
            let config = RemoveOutliers::new(k);
            group.bench_with_input(
                BenchmarkId::new(format!("k{}", k), size),
                &(cloud.clone(), config),
                |b, (cloud, config)| b.iter(|| black_box(remove_outliers(cloud, config))),
            );
        }
    }

    group.finish();
}

fn bench_crop_to_bounds(c: &mut Criterion) {
    let mut group = c.benchmark_group("crop_to_bounds");

    let mesh = Cuboid::unit().apply();

    for size in [1000, 5000, 10000] {
        let cloud = sample_mesh_uniform(&mesh, size);

        group.bench_with_input(BenchmarkId::from_parameter(size), &cloud, |b, cloud| {
            b.iter(|| {
                black_box(crop_to_bounds(
                    cloud,
                    Vec3::new(-0.25, -0.25, -0.25),
                    Vec3::new(0.25, 0.25, 0.25),
                ))
            })
        });
    }

    group.finish();
}

// ============================================================================
// Transform and Merge
// ============================================================================

fn bench_transform(c: &mut Criterion) {
    let mut group = c.benchmark_group("transform");

    let mesh = Cuboid::unit().apply();
    let matrix = Mat4::from_translation(Vec3::new(1.0, 2.0, 3.0))
        * Mat4::from_rotation_y(std::f32::consts::PI / 4.0);

    for size in [1000, 5000, 10000] {
        let cloud = sample_mesh_uniform(&mesh, size);

        group.bench_with_input(
            BenchmarkId::new("positions_only", size),
            &PointCloud::from_positions(cloud.positions.clone()),
            |b, cloud| b.iter(|| black_box(transform(cloud, matrix))),
        );

        group.bench_with_input(
            BenchmarkId::new("with_normals", size),
            &cloud,
            |b, cloud| b.iter(|| black_box(transform(cloud, matrix))),
        );
    }

    group.finish();
}

fn bench_merge(c: &mut Criterion) {
    let mut group = c.benchmark_group("merge");

    let mesh = Cuboid::unit().apply();

    for size in [1000, 5000] {
        let cloud1 = sample_mesh_uniform(&mesh, size);
        let cloud2 = sample_mesh_uniform(&mesh, size);

        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            &(cloud1, cloud2),
            |b, (c1, c2)| {
                b.iter(|| {
                    let mut cloud = c1.clone();
                    cloud.merge(c2);
                    black_box(cloud)
                })
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_pointcloud_creation,
    bench_point_insertion,
    bench_bounding_box,
    bench_centroid,
    bench_sample_mesh_uniform,
    bench_sample_mesh_poisson,
    bench_estimate_normals,
    bench_voxel_downsample,
    bench_remove_outliers,
    bench_crop_to_bounds,
    bench_transform,
    bench_merge,
);

criterion_main!(benches);
