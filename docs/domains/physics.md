# Physics

Physical simulation systems for rigid bodies, fluids, cloth, and soft bodies.

## Prior Art

### Bullet Physics / PhysX
- **Rigid bodies**: impulse-based collision resolution, constraints
- **Soft bodies**: mass-spring systems, FEM deformation
- **Collision detection**: broadphase (AABB), narrowphase (GJK/EPA)

### Jos Stam's Stable Fluids
- **Grid-based simulation**: velocity/density fields on regular grid
- **Semi-Lagrangian advection**: unconditionally stable
- **Pressure projection**: enforce incompressibility via Poisson solve

### SPH (Smoothed Particle Hydrodynamics)
- **Particle-based**: fluid as moving particles with kernel smoothing
- **Navier-Stokes**: pressure, viscosity forces between particles
- **Free surfaces**: natural handling of splashing, merging

### Position-Based Dynamics
- **Verlet integration**: position-based, implicit velocity
- **Constraint projection**: directly solve position constraints
- **Stable cloth/rope**: no stiff spring oscillation

## Crate Structure

| Crate | Purpose |
|-------|---------|
| `unshape-physics` | Rigid body dynamics, cloth, soft body FEM |
| `unshape-fluid` | Grid-based and particle-based fluid simulation |
| `unshape-spring` | Position-based dynamics, Verlet integration |

## Rigid Body Physics

### Core Types

```rust
use rhi_unshape_physics::{RigidBody, Collider, PhysicsWorld};

// Create collider shapes
let sphere = Collider::sphere(1.0);
let box_shape = Collider::box_shape(Vec3::new(0.5, 0.5, 0.5));
let ground = Collider::ground();  // Infinite plane at y=0

// Create bodies
let dynamic = RigidBody::new(Vec3::new(0.0, 10.0, 0.0), sphere, 1.0);
let static_body = RigidBody::new_static(Vec3::ZERO, ground);

// Simulation world
let mut world = PhysicsWorld::new();
world.set_gravity(Vec3::new(0.0, -9.8, 0.0));
world.add_body(dynamic);
world.add_body(static_body);

// Step simulation
world.step(1.0 / 60.0);
```

### Collider Types

| Shape | Constructor | Parameters |
|-------|-------------|------------|
| Sphere | `Collider::sphere(radius)` | Radius |
| Box | `Collider::box_shape(half_extents)` | Half-extents Vec3 |
| Plane | `Collider::plane(normal, distance)` | Normal, distance from origin |
| Ground | `Collider::ground()` | Infinite plane at y=0 |

### Body Properties

```rust
let mut body = RigidBody::new(position, collider, mass);
body.restitution = 0.8;      // Bounciness (0-1)
body.friction = 0.3;         // Friction coefficient
body.linear_damping = 0.01;  // Air resistance
body.angular_damping = 0.05; // Rotational damping
```

### Constraints

```rust
use rhi_unshape_physics::{DistanceConstraint, HingeConstraint, SpringConstraint};

// Distance constraint (rigid connection)
let distance = DistanceConstraint::new(body_a, body_b, 2.0);

// Hinge constraint (rotates around axis)
let hinge = HingeConstraint::new(body_a, body_b, pivot, axis);

// Spring constraint (elastic connection)
let spring = SpringConstraint::new(body_a, body_b, stiffness, damping);
```

## Spring Physics (Position-Based Dynamics)

### Verlet Integration

```rust
use rhi_unshape_spring::{SpringSystem, SpringConfig};

let mut system = SpringSystem::new();

// Add particles
let p0 = system.add_particle(Vec3::new(0.0, 5.0, 0.0), 1.0);
let p1 = system.add_particle(Vec3::new(1.0, 5.0, 0.0), 1.0);

// Pin first particle (infinite mass)
system.pin_particle(p0);

// Connect with spring
system.add_spring(p0, p1, SpringConfig {
    rest_length: 1.0,
    stiffness: 0.9,
    damping: 0.1,
});

// Simulate
system.set_gravity(Vec3::new(0.0, -9.8, 0.0));
system.step(0.016);  // 60 FPS timestep
```

### Preset Shapes

```rust
use rhi_unshape_spring::{create_rope, create_cloth, create_soft_sphere};

// Rope: chain of particles
let rope = create_rope(
    start: Vec3::new(0.0, 5.0, 0.0),
    end: Vec3::new(5.0, 5.0, 0.0),
    segments: 20,
    config: SpringConfig::default(),
);

// Cloth: 2D grid of particles
let cloth = create_cloth(
    origin: Vec3::ZERO,
    width: 5.0,
    height: 5.0,
    resolution: (20, 20),
    config: SpringConfig::default(),
);

// Soft sphere: 3D particle shell
let ball = create_soft_sphere(
    center: Vec3::ZERO,
    radius: 1.0,
    resolution: 10,
    config: SpringConfig::default(),
);
```

## Cloth Simulation

### High-Level API

```rust
use rhi_unshape_physics::{Cloth, ClothConfig};

let config = ClothConfig {
    structural_stiffness: 0.9,
    shear_stiffness: 0.5,
    bend_stiffness: 0.1,
    damping: 0.02,
    iterations: 4,
};

let mut cloth = Cloth::new(width, height, resolution, config);

// Pin corners
cloth.pin(0, 0);
cloth.pin(resolution - 1, 0);

// Simulate
cloth.step(dt, gravity);
```

### Cloth-Object Collision

```rust
use rhi_unshape_physics::{ClothCollider, query_collision, solve_self_collision};

// Create collider for cloth to interact with
let sphere_collider = ClothCollider::sphere(center, radius);

// Query collisions
let collisions = query_collision(&cloth, &sphere_collider);

// Resolve self-collision (cloth folding on itself)
solve_self_collision(&mut cloth, grid_cell_size);
```

## Soft Body (FEM)

Finite Element Method deformation for volumetric soft bodies.

```rust
use rhi_unshape_physics::{
    SoftBody, SoftBodyConfig, LameParameters, tetrahedralize_surface
};

// Lame parameters control material properties
let rubber = LameParameters::new(
    youngs_modulus: 1000.0,  // Stiffness
    poisson_ratio: 0.45,     // Volume preservation (0-0.5)
);

// Create from mesh by tetrahedralizing
let tets = tetrahedralize_surface(&mesh);
let config = SoftBodyConfig {
    lame: rubber,
    damping: 0.1,
    iterations: 10,
};

let mut soft = SoftBody::new(tets, config);
soft.step(dt);
```

## Fluid Simulation

### Grid-Based (Stable Fluids)

Best for: smoke, fire, contained liquids, real-time effects.

```rust
use rhi_unshape_fluid::{FluidGrid2D, FluidConfig};

let config = FluidConfig {
    diffusion: 0.0001,  // Viscosity
    iterations: 20,     // Solver iterations
    dt: 0.1,            // Time step
};

let mut fluid = FluidGrid2D::new(128, 128, config);

// Add density (ink, smoke)
fluid.add_density(64, 64, 100.0);

// Add velocity (forces)
fluid.add_velocity(64, 64, Vec2::new(0.0, 10.0));

// Step simulation
fluid.step();

// Query results
let d = fluid.density(x, y);
let v = fluid.velocity(x, y);
```

### 3D Grid Fluids

```rust
use rhi_unshape_fluid::FluidGrid3D;

let mut fluid = FluidGrid3D::new(64, 64, 64, config);
fluid.add_density(32, 32, 32, 100.0);
fluid.add_velocity(32, 32, 32, Vec3::new(0.0, 5.0, 0.0));
fluid.step();
```

### SPH (Particle-Based)

Best for: splashing liquids, free-surface flows, interactions.

```rust
use rhi_unshape_fluid::{Sph2D, SphConfig};

let config = SphConfig {
    rest_density: 1000.0,    // Water density
    gas_constant: 2000.0,    // Pressure response
    viscosity: 0.001,        // Fluid thickness
    kernel_radius: 0.1,      // Smoothing radius
};

let mut sph = Sph2D::new(config);

// Add particles
sph.add_particle(Vec2::new(0.0, 1.0));
sph.add_particle(Vec2::new(0.1, 1.0));
// ... add more

// Simulate
sph.step(dt);

// Get particle positions for rendering
let positions = sph.positions();
```

### 3D SPH

```rust
use rhi_unshape_fluid::Sph3D;

let mut sph = Sph3D::new(config);
sph.add_particle(Vec3::new(0.0, 1.0, 0.0));
sph.step(dt);
```

## Smoke/Gas Simulation

```rust
use rhi_unshape_fluid::{SmokeGrid2D, SmokeConfig};

let config = SmokeConfig {
    diffusion: 0.0001,
    dissipation: 0.99,      // Smoke fades over time
    buoyancy: 1.0,          // Hot smoke rises
    temperature_diffusion: 0.0001,
};

let mut smoke = SmokeGrid2D::new(128, 128, config);

// Add smoke source with temperature
smoke.add_source(64, 10, density: 1.0, temperature: 100.0);

// Step simulation
smoke.step();

// Query for rendering
let d = smoke.density(x, y);
let t = smoke.temperature(x, y);
```

## Performance Tips

1. **Rigid bodies**: Use simple colliders (sphere, box) when possible
2. **Springs**: Reduce iteration count for soft constraints
3. **Cloth**: Lower resolution, increase constraint stiffness
4. **Fluids**: Smaller grid = faster but less detail
5. **SPH**: Use spatial hashing for neighbor queries

## Integration with Meshes

```rust
// Convert cloth to mesh for rendering
let mesh = cloth.to_mesh();

// Sample fluid velocity at mesh vertices for displacement
for vertex in mesh.vertices_mut() {
    let vel = fluid.sample_velocity(vertex.position);
    vertex.position += vel * dt;
}

// Soft body already provides mesh access
let deformed_mesh = soft.mesh();
```
