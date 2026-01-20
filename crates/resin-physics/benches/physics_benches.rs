//! Benchmarks for physics simulation.

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use glam::Vec3;
use rhizome_resin_physics::{Collider, Physics, PhysicsWorld, RigidBody};

fn bench_world_step(c: &mut Criterion) {
    c.bench_function("physics_step_100_bodies", |b| {
        let mut world = PhysicsWorld::new(Physics::default());
        // Add 100 dynamic sphere bodies in a grid
        for i in 0..100 {
            let pos = Vec3::new((i % 10) as f32 * 2.0, (i / 10) as f32 * 2.0 + 5.0, 0.0);
            let body = RigidBody::new(pos, Collider::sphere(0.5), 1.0);
            world.add_body(body);
        }

        b.iter(|| {
            world.step();
            black_box(&world);
        })
    });

    c.bench_function("physics_step_500_bodies", |b| {
        let mut world = PhysicsWorld::new(Physics::default());
        // Add 500 dynamic sphere bodies in a grid
        for i in 0..500 {
            let x = (i % 25) as f32 * 2.0;
            let y = ((i / 25) % 20) as f32 * 2.0 + 5.0;
            let z = (i / 500) as f32 * 2.0;
            let pos = Vec3::new(x, y, z);
            let body = RigidBody::new(pos, Collider::sphere(0.5), 1.0);
            world.add_body(body);
        }

        b.iter(|| {
            world.step();
            black_box(&world);
        })
    });
}

fn bench_collision_detection(c: &mut Criterion) {
    // Sphere-sphere collision detection (many potential pairs)
    c.bench_function("collision_sphere_sphere_100", |b| {
        let mut world = PhysicsWorld::new(Physics::default());
        // Place spheres close together to generate contacts
        for i in 0..100 {
            let pos = Vec3::new((i % 10) as f32 * 1.5, (i / 10) as f32 * 1.5 + 1.0, 0.0);
            let body = RigidBody::new(pos, Collider::sphere(1.0), 1.0);
            world.add_body(body);
        }

        b.iter(|| {
            world.step();
            black_box(&world);
        })
    });

    // Sphere-plane collision detection
    c.bench_function("collision_sphere_plane_50", |b| {
        let mut world = PhysicsWorld::new(Physics::default());
        // Add ground plane
        let ground = RigidBody::new_static(Vec3::ZERO, Collider::ground());
        world.add_body(ground);

        // Add spheres above ground
        for i in 0..50 {
            let pos = Vec3::new((i % 10) as f32 * 2.0, 0.5 + (i / 10) as f32 * 0.1, 0.0);
            let body = RigidBody::new(pos, Collider::sphere(0.5), 1.0);
            world.add_body(body);
        }

        b.iter(|| {
            world.step();
            black_box(&world);
        })
    });

    // Box-plane collision detection
    c.bench_function("collision_box_plane_50", |b| {
        let mut world = PhysicsWorld::new(Physics::default());
        // Add ground plane
        let ground = RigidBody::new_static(Vec3::ZERO, Collider::ground());
        world.add_body(ground);

        // Add boxes above ground
        for i in 0..50 {
            let pos = Vec3::new((i % 10) as f32 * 3.0, 1.0 + (i / 10) as f32 * 0.1, 0.0);
            let body = RigidBody::new(pos, Collider::box_shape(Vec3::splat(0.5)), 1.0);
            world.add_body(body);
        }

        b.iter(|| {
            world.step();
            black_box(&world);
        })
    });
}

fn bench_integration(c: &mut Criterion) {
    // Velocity Verlet integration (pure motion, no collisions)
    c.bench_function("integration_100_bodies", |b| {
        let mut world = PhysicsWorld::new(Physics::default());
        // Add bodies spaced far apart to avoid collisions
        for i in 0..100 {
            let pos = Vec3::new((i % 10) as f32 * 100.0, (i / 10) as f32 * 100.0, 0.0);
            let mut body = RigidBody::new(pos, Collider::sphere(0.5), 1.0);
            // Give initial velocity
            body.velocity = Vec3::new(1.0, 2.0, 0.0);
            world.add_body(body);
        }

        b.iter(|| {
            world.step();
            black_box(&world);
        })
    });

    c.bench_function("integration_1000_bodies", |b| {
        let mut world = PhysicsWorld::new(Physics::default());
        // Add bodies spaced far apart to avoid collisions
        for i in 0..1000 {
            let pos = Vec3::new(
                (i % 100) as f32 * 100.0,
                ((i / 100) % 10) as f32 * 100.0,
                (i / 1000) as f32 * 100.0,
            );
            let mut body = RigidBody::new(pos, Collider::sphere(0.5), 1.0);
            body.velocity = Vec3::new(1.0, 2.0, 0.0);
            world.add_body(body);
        }

        b.iter(|| {
            world.step();
            black_box(&world);
        })
    });
}

fn bench_rigid_body_updates(c: &mut Criterion) {
    c.bench_function("rigid_body_apply_force_1000", |b| {
        let mut body = RigidBody::new(Vec3::ZERO, Collider::sphere(1.0), 1.0);
        let force = Vec3::new(10.0, 5.0, 0.0);

        b.iter(|| {
            for _ in 0..1000 {
                body.apply_force(black_box(force));
            }
            black_box(&body);
        })
    });

    c.bench_function("rigid_body_apply_impulse_at_point_1000", |b| {
        let mut body = RigidBody::new(Vec3::ZERO, Collider::sphere(1.0), 1.0);
        let impulse = Vec3::new(1.0, 0.5, 0.0);
        let point = Vec3::new(0.5, 0.0, 0.0);

        b.iter(|| {
            for _ in 0..1000 {
                body.apply_impulse_at_point(black_box(impulse), black_box(point));
            }
            black_box(&body);
        })
    });

    c.bench_function("rigid_body_velocity_at_point_1000", |b| {
        let mut body = RigidBody::new(Vec3::ZERO, Collider::sphere(1.0), 1.0);
        body.velocity = Vec3::new(1.0, 2.0, 0.0);
        body.angular_velocity = Vec3::new(0.0, 0.0, 1.0);
        let point = Vec3::new(1.0, 0.0, 0.0);

        b.iter(|| {
            for _ in 0..1000 {
                black_box(body.velocity_at_point(black_box(point)));
            }
        })
    });
}

fn bench_constraints(c: &mut Criterion) {
    c.bench_function("constraints_spring_10_bodies", |b| {
        let mut world = PhysicsWorld::new(Physics::default());

        // Create a chain of bodies connected by springs
        for i in 0..10 {
            let pos = Vec3::new(i as f32 * 2.0, 5.0, 0.0);
            let body = RigidBody::new(pos, Collider::sphere(0.5), 1.0);
            world.add_body(body);
        }

        // Connect adjacent bodies with springs
        for i in 0..9 {
            world.add_spring(i, i + 1, 2.0, 100.0, 1.0);
        }

        b.iter(|| {
            world.step();
            black_box(&world);
        })
    });

    c.bench_function("constraints_distance_20_bodies", |b| {
        let mut world = PhysicsWorld::new(Physics::default());

        // Create bodies
        for i in 0..20 {
            let pos = Vec3::new((i % 5) as f32 * 2.0, (i / 5) as f32 * 2.0 + 5.0, 0.0);
            let body = RigidBody::new(pos, Collider::sphere(0.5), 1.0);
            world.add_body(body);
        }

        // Add distance constraints in a grid pattern
        for i in 0..20 {
            if i % 5 < 4 {
                world.add_distance_constraint(i, i + 1, 2.0);
            }
            if i < 15 {
                world.add_distance_constraint(i, i + 5, 2.0);
            }
        }

        b.iter(|| {
            world.step();
            black_box(&world);
        })
    });
}

criterion_group!(
    benches,
    bench_world_step,
    bench_collision_detection,
    bench_integration,
    bench_rigid_body_updates,
    bench_constraints
);
criterion_main!(benches);
