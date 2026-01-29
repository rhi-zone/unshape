//! Fluid simulation for resin.
//!
//! Provides grid-based and particle-based fluid simulation:
//! - `FluidGrid2D` - 2D Eulerian grid-based simulation (stable fluids)
//! - `FluidGrid3D` - 3D Eulerian grid-based simulation
//! - `Sph2D` - 2D Smoothed Particle Hydrodynamics
//! - `Sph3D` - 3D Smoothed Particle Hydrodynamics

mod helpers;

mod grid;
mod smoke;
mod sph;

pub use grid::*;
pub use smoke::*;
pub use sph::*;

/// Registers all fluid operations with an [`OpRegistry`].
///
/// Call this to enable deserialization of fluid ops from saved pipelines.
#[cfg(feature = "dynop")]
pub fn register_ops(registry: &mut unshape_op::OpRegistry) {
    registry.register_type::<Fluid>("resin::Fluid");
    registry.register_type::<Sph>("resin::Sph");
    registry.register_type::<SphParams3D>("resin::SphParams3D");
    registry.register_type::<Smoke>("resin::Smoke");
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::{Vec2, Vec3};

    #[test]
    fn test_fluid_grid_2d_creation() {
        let fluid = FluidGrid2D::new(64, 64, FluidConfig::default());
        assert_eq!(fluid.size(), (64, 64));
    }

    #[test]
    fn test_fluid_grid_2d_add_density() {
        let mut fluid = FluidGrid2D::new(32, 32, FluidConfig::default());
        fluid.add_density(16, 16, 100.0);
        assert_eq!(fluid.density(16, 16), 100.0);
    }

    #[test]
    fn test_fluid_grid_2d_add_velocity() {
        let mut fluid = FluidGrid2D::new(32, 32, FluidConfig::default());
        fluid.add_velocity(16, 16, 10.0, 5.0);
        let vel = fluid.velocity(16, 16);
        assert_eq!(vel.x, 10.0);
        assert_eq!(vel.y, 5.0);
    }

    #[test]
    fn test_fluid_grid_2d_step() {
        let mut fluid = FluidGrid2D::new(32, 32, FluidConfig::default());
        fluid.add_density(16, 16, 100.0);
        fluid.add_velocity(16, 16, 5.0, 0.0);

        // Step simulation
        fluid.step();

        // Density should have spread/advected
        // Just verify it runs without panic
    }

    #[test]
    fn test_fluid_grid_2d_sample() {
        let mut fluid = FluidGrid2D::new(32, 32, FluidConfig::default());
        fluid.add_density(16, 16, 100.0);

        let sampled = fluid.sample_density(Vec2::new(16.0, 16.0));
        assert!(sampled > 0.0);
    }

    #[test]
    fn test_fluid_grid_3d_creation() {
        let fluid = FluidGrid3D::new(16, 16, 16, FluidConfig::default());
        assert_eq!(fluid.size(), (16, 16, 16));
    }

    #[test]
    fn test_fluid_grid_3d_step() {
        let mut fluid = FluidGrid3D::new(16, 16, 16, FluidConfig::default());
        fluid.add_density(8, 8, 8, 100.0);
        fluid.add_velocity(8, 8, 8, Vec3::new(1.0, 0.0, 0.0));
        fluid.step();
        // Just verify it runs
    }

    #[test]
    fn test_sph_2d_creation() {
        let sph = Sph2D::new(SphConfig::default(), (Vec2::ZERO, Vec2::new(100.0, 100.0)));
        assert_eq!(sph.particles.len(), 0);
    }

    #[test]
    fn test_sph_2d_add_particle() {
        let mut sph = Sph2D::new(SphConfig::default(), (Vec2::ZERO, Vec2::new(100.0, 100.0)));
        sph.add_particle(Vec2::new(50.0, 50.0), 1.0);
        assert_eq!(sph.particles.len(), 1);
    }

    #[test]
    fn test_sph_2d_add_block() {
        let mut sph = Sph2D::new(SphConfig::default(), (Vec2::ZERO, Vec2::new(100.0, 100.0)));
        sph.add_block(Vec2::new(10.0, 10.0), Vec2::new(30.0, 30.0), 5.0, 1.0);
        assert!(sph.particles.len() > 0);
    }

    #[test]
    fn test_sph_2d_step() {
        let mut sph = Sph2D::new(SphConfig::default(), (Vec2::ZERO, Vec2::new(100.0, 100.0)));
        sph.add_block(Vec2::new(20.0, 50.0), Vec2::new(40.0, 70.0), 8.0, 1.0);

        let initial_pos = sph.positions();
        sph.step();
        let final_pos = sph.positions();

        // Particles should have moved (gravity)
        assert!(initial_pos != final_pos);
    }

    #[test]
    fn test_sph_3d_creation() {
        let sph = Sph3D::new(
            SphConfig3D::default(),
            (Vec3::ZERO, Vec3::new(1.0, 1.0, 1.0)),
        );
        assert_eq!(sph.particles.len(), 0);
    }

    #[test]
    fn test_sph_3d_add_block() {
        let mut sph = Sph3D::new(
            SphConfig3D::default(),
            (Vec3::ZERO, Vec3::new(1.0, 1.0, 1.0)),
        );
        sph.add_block(
            Vec3::new(0.1, 0.5, 0.1),
            Vec3::new(0.3, 0.7, 0.3),
            0.05,
            0.001,
        );
        assert!(sph.particles.len() > 0);
    }

    #[test]
    fn test_sph_3d_step() {
        let mut sph = Sph3D::new(
            SphConfig3D::default(),
            (Vec3::ZERO, Vec3::new(1.0, 1.0, 1.0)),
        );
        sph.add_block(
            Vec3::new(0.2, 0.5, 0.2),
            Vec3::new(0.4, 0.7, 0.4),
            0.05,
            0.001,
        );

        let initial_pos = sph.positions();
        sph.step();
        let final_pos = sph.positions();

        // Particles should have moved
        assert!(initial_pos != final_pos);
    }

    #[test]
    fn test_smoke_2d_creation() {
        let smoke = SmokeGrid2D::new(64, 64, SmokeConfig::default());
        assert_eq!(smoke.size(), (64, 64));
    }

    #[test]
    fn test_smoke_2d_add_smoke() {
        let mut smoke = SmokeGrid2D::new(32, 32, SmokeConfig::default());
        smoke.add_smoke(16, 16, 100.0, 50.0);
        assert_eq!(smoke.density(16, 16), 100.0);
        assert_eq!(smoke.temperature(16, 16), 50.0);
    }

    #[test]
    fn test_smoke_2d_buoyancy() {
        let mut smoke = SmokeGrid2D::new(32, 32, SmokeConfig::default());
        // Add hot smoke
        smoke.add_smoke(16, 5, 100.0, 100.0);

        // Step simulation
        for _ in 0..10 {
            smoke.step();
        }

        // Hot smoke should have risen (positive y velocity somewhere)
        let vel = smoke.velocity(16, 10);
        assert!(vel.y > 0.0 || smoke.density(16, 10) > 0.0);
    }

    #[test]
    fn test_smoke_2d_dissipation() {
        let mut smoke = SmokeGrid2D::new(32, 32, SmokeConfig::default());
        smoke.add_smoke(16, 16, 100.0, 100.0);

        let initial_density = smoke.density(16, 16);
        smoke.step();
        let final_density = smoke.density(16, 16);

        // Density should have decreased due to dissipation
        assert!(final_density < initial_density);
    }

    #[test]
    fn test_smoke_3d_creation() {
        let smoke = SmokeGrid3D::new(16, 16, 16, SmokeConfig::default());
        assert_eq!(smoke.size(), (16, 16, 16));
    }

    #[test]
    fn test_smoke_3d_step() {
        let mut smoke = SmokeGrid3D::new(16, 16, 16, SmokeConfig::default());
        smoke.add_smoke(8, 4, 8, 100.0, 100.0);

        // Step simulation
        smoke.step();

        // Just verify it runs
    }
}
