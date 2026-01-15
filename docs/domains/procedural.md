# Procedural Generation

Algorithms for generating content procedurally: terrain, mazes, networks, plants, and patterns.

## Prior Art

### Wave Function Collapse
- **Constraint propagation**: tiles constrain neighbors recursively
- **Entropy-based selection**: collapse lowest-entropy cells first
- **Backtracking**: recover from contradictions

### L-Systems (Lindenmayer Systems)
- **String rewriting**: parallel rule application
- **Turtle graphics**: interpret strings as drawing commands
- **Parametric/stochastic**: extensions for natural variation

### Space Colonization
- **Attraction points**: growth targets distributed in space
- **Incremental growth**: nodes extend toward attractors
- **Kill distance**: attractors removed when reached

### Erosion Simulation
- **Hydraulic**: water carries sediment downhill
- **Thermal**: material slides when slope exceeds threshold
- **Particle-based**: droplet simulation for detail

## Crate Structure

| Crate | Purpose |
|-------|---------|
| `resin-procgen` | Maze generation, road/river networks, WFC |
| `resin-lsystem` | L-system string rewriting and turtle interpretation |
| `resin-space-colonization` | Tree/vessel/lightning generation |
| `resin-automata` | Cellular automata (1D elementary, 2D Game of Life) |
| `resin-rd` | Reaction-diffusion pattern generation |
| `resin-mesh` (terrain module) | Heightfield generation and erosion |

## Maze Generation

### Algorithms

```rust
use rhizome_resin_procgen::maze::{generate_maze, MazeAlgorithm, Maze};

// Available algorithms
let maze = generate_maze(width, height, MazeAlgorithm::RecursiveBacktracker, seed);
let maze = generate_maze(width, height, MazeAlgorithm::Prim, seed);
let maze = generate_maze(width, height, MazeAlgorithm::Kruskal, seed);
let maze = generate_maze(width, height, MazeAlgorithm::Eller, seed);
let maze = generate_maze(width, height, MazeAlgorithm::BinaryTree, seed);
let maze = generate_maze(width, height, MazeAlgorithm::Sidewinder, seed);
```

| Algorithm | Character | Best For |
|-----------|-----------|----------|
| RecursiveBacktracker | Long, winding passages | Natural caves |
| Prim | Short dead ends, uniform | Dungeons |
| Kruskal | Similar to Prim | Random feel |
| Eller | Row-by-row generation | Infinite mazes |
| BinaryTree | Diagonal bias | Fast generation |
| Sidewinder | Horizontal bias | Corridors |

### Working with Mazes

```rust
// Query cells
let is_open = maze.is_passage(x, y);
let is_wall = maze.is_wall(x, y);

// Get full grid (includes walls between cells)
let grid: Vec<Vec<bool>> = maze.to_grid();

// Dimensions
let (w, h) = maze.dimensions();        // Logical cells
let (gw, gh) = maze.grid_dimensions(); // Including walls
```

## Road and River Networks

### Road Networks

```rust
use rhizome_resin_procgen::network::{RoadNetwork, RoadConfig, RoadType};
use glam::Vec2;

let config = RoadConfig::default();
let mut network = RoadNetwork::new(config);

// Add nodes (cities, intersections)
let a = network.add_node(Vec2::new(0.0, 0.0));
let b = network.add_node(Vec2::new(100.0, 0.0));
let c = network.add_node(Vec2::new(50.0, 80.0));

// Connect with different road types
network.connect(a, b, RoadType::Highway);
network.connect(b, c, RoadType::Secondary);
network.connect(c, a, RoadType::Local);

// Get segments for rendering
let segments = network.to_segments();
```

### River Networks

```rust
use rhizome_resin_procgen::network::{RiverNetwork, RiverConfig};

let config = RiverConfig {
    meander_strength: 0.3,
    branch_probability: 0.1,
    ..Default::default()
};

let mut rivers = RiverNetwork::new(config);
rivers.add_source(Vec2::new(50.0, 100.0));
rivers.flow_to(Vec2::new(50.0, 0.0), &heightfield);

let segments = rivers.to_segments();
```

## L-Systems

### Basic Usage

```rust
use rhizome_resin_lsystem::{LSystem, Rule, TurtleConfig, interpret_turtle_2d};

// Define system
let lsystem = LSystem::new("F")
    .with_rule(Rule::simple('F', "F+F-F-F+F"));

// Generate string
let result = lsystem.generate(4);  // 4 iterations

// Interpret as 2D paths
let turtle = Turtle2D {
    angle: 90.0,   // Turn angle in degrees
    step: 1.0,     // Segment length
    ..Default::default()
};

let paths = turtle.apply(&result);
```

### Turtle Commands

| Symbol | Action |
|--------|--------|
| `F` | Move forward, draw line |
| `f` | Move forward, no line |
| `+` | Turn left by angle |
| `-` | Turn right by angle |
| `[` | Push state (position, angle) |
| `]` | Pop state |
| `&` | Pitch down (3D) |
| `^` | Pitch up (3D) |
| `\` | Roll left (3D) |
| `/` | Roll right (3D) |

### Presets

```rust
use rhizome_resin_lsystem::presets;

// Classic fractals
let koch = presets::koch_curve();
let sierpinski = presets::sierpinski_triangle();
let dragon = presets::dragon_curve();

// Plants
let bush = presets::bush();
let tree = presets::simple_tree();
let fern = presets::barnsley_fern();
```

### 3D Interpretation

```rust
use rhizome_resin_lsystem::interpret_turtle_3d;

let tree_string = lsystem.generate(5);
let branches = interpret_turtle_3d(&tree_string, &config);

// branches contains Vec<(Vec3, Vec3)> line segments
```

## Space Colonization

### Tree Generation

```rust
use rhizome_resin_space_colonization::{SpaceColonization, SpaceColonizationConfig};
use glam::Vec3;

let config = SpaceColonizationConfig {
    attraction_distance: 5.0,  // Range of influence
    kill_distance: 1.0,        // Remove attractor when reached
    segment_length: 0.5,       // Branch segment length
    tropism: Vec3::new(0.0, 1.0, 0.0),  // Grow upward
    tropism_strength: 0.1,
    ..Default::default()
};

let mut sc = SpaceColonization::new(config);

// Define crown shape with attraction points
sc.add_attraction_points_sphere(
    center: Vec3::new(0.0, 8.0, 0.0),
    radius: 4.0,
    count: 500,
    seed: 12345,
);

// Add trunk root
sc.add_root(Vec3::ZERO);

// Run algorithm
sc.run(200);

// Get results
let nodes = sc.nodes();  // Vec<BranchNode>
let edges = sc.edges();  // Vec<(usize, usize)>

// Convert to mesh with pipe model radii
let mesh = sc.to_mesh_with_radii(base_radius: 0.1, taper: 0.7);
```

### Other Shapes

```rust
// Lightning bolt
sc.add_attraction_points_line(start, end, count, seed);
sc.add_root(start);

// Blood vessels / roots
sc.add_attraction_points_box(min, max, count, seed);

// Custom distribution
for point in my_points {
    sc.add_attraction_point(point);
}
```

## Cellular Automata

### 1D Elementary CA

```rust
use rhizome_resin_automata::{ElementaryCA, elementary_rules};

// Create with rule number (0-255)
let mut ca = ElementaryCA::new(width: 100, rule: 30);

// Initial condition
ca.set_center();  // Single cell in middle
// or
ca.randomize(seed);

// Step simulation
ca.step();

// Generate 2D pattern (time as Y axis)
let pattern = ca.generate_pattern(generations: 100);
```

| Rule | Pattern |
|------|---------|
| 30 | Chaotic (random number generation) |
| 90 | Sierpinski triangle |
| 110 | Turing complete |
| 184 | Traffic flow |

### 2D Cellular Automata

```rust
use rhizome_resin_automata::{CellularAutomaton2D, GameOfLife, rules};

// Conway's Game of Life
let mut life = GameOfLife::life(50, 50);
life.randomize(seed, density: 0.3);
life.step();

// Custom rules (Birth/Survival notation)
let (birth, survive) = rules::HIGH_LIFE;  // B36/S23
let mut ca = CellularAutomaton2D::new(50, 50, birth, survive);

// Available presets
rules::LIFE;       // B3/S23 - Classic
rules::HIGH_LIFE;  // B36/S23 - More action
rules::SEEDS;      // B2/S - Explosive
rules::DAY_NIGHT;  // B3678/S34678 - Symmetric
rules::MAZE;       // B3/S12345 - Maze-like
rules::DIAMOEBA;   // B35678/S5678 - Amoeba
```

## Reaction-Diffusion

### Gray-Scott Model

```rust
use rhizome_resin_rd::{ReactionDiffusion, GrayScottPreset};

let mut rd = ReactionDiffusion::new(256, 256);

// Use preset parameters
rd.set_preset(GrayScottPreset::Coral);
rd.set_preset(GrayScottPreset::Mitosis);
rd.set_preset(GrayScottPreset::Fingerprint);
rd.set_preset(GrayScottPreset::Spots);
rd.set_preset(GrayScottPreset::Stripes);

// Or set manually
rd.feed = 0.055;
rd.kill = 0.062;
rd.du = 1.0;
rd.dv = 0.5;

// Seed the pattern
rd.add_seed_circle(x: 128, y: 128, radius: 10);
rd.add_seed_square(x: 64, y: 64, size: 20);

// Simulate (needs many iterations)
for _ in 0..5000 {
    rd.step();
}

// Get results
let pattern = rd.get_v_normalized();  // 0.0-1.0 values
```

## Terrain Generation

### Heightfield Creation

```rust
use rhizome_resin_mesh::{Heightfield, HydraulicErosion, ThermalErosion};

// Create heightfield
let mut hf = Heightfield::new(256, 256);

// Generate base terrain
hf.apply_diamond_square(roughness: 0.5, seed: 12345);

// Or use noise
hf.apply_noise(|x, y| {
    fbm_perlin2(x * 0.01, y * 0.01, octaves: 6)
});
```

### Erosion

```rust
// Hydraulic erosion (water flow)
let mut hydraulic = HydraulicErosion {
    iterations: 50000,
    inertia: 0.05,
    capacity: 4.0,
    deposition: 0.3,
    erosion: 0.3,
    evaporation: 0.01,
    min_slope: 0.01,
    ..Default::default()
};
hydraulic.erode(&mut hf, droplets: 50000);

// Thermal erosion (talus slopes)
let mut thermal = ThermalErosion {
    iterations: 50,
    talus_angle: 0.5,  // Maximum stable slope
    erosion_rate: 0.5,
};
thermal.erode(&mut hf);
```

### Convert to Mesh

```rust
// Generate mesh with scale
let mesh = hf.to_mesh(
    horizontal_scale: 100.0,  // World units per sample
    vertical_scale: 20.0,     // Height multiplier
);

// Generate normal map for detail
let normal_map = hf.to_normal_map();
```

## Building Generation

```rust
use rhizome_resin_procgen::{Building, generate_building, generate_stairs};

// Simple building from footprint
let footprint = vec![
    Vec2::new(0.0, 0.0),
    Vec2::new(10.0, 0.0),
    Vec2::new(10.0, 8.0),
    Vec2::new(0.0, 8.0),
];

let building = generate_building(
    footprint: &footprint,
    floors: 3,
    floor_height: 3.0,
    roof_style: RoofStyle::Gabled,
);

// Stairs
let stairs = generate_stairs(
    start: Vec3::ZERO,
    end: Vec3::new(0.0, 3.0, 4.0),
    width: 1.2,
    step_height: 0.18,
);
```

## Combining Systems

### Terrain + Vegetation

```rust
// Generate terrain
let mut hf = Heightfield::new(256, 256);
hf.apply_diamond_square(0.5, seed);
HydraulicErosion::default().erode(&mut hf, 10000);

// Place trees using space colonization
let mut trees = Vec::new();
for _ in 0..100 {
    let x = rng.gen_range(0.0..256.0);
    let z = rng.gen_range(0.0..256.0);
    let y = hf.sample(x, z);

    // Only on gentle slopes
    if hf.slope_at(x, z) < 0.3 {
        let tree = generate_tree_at(Vec3::new(x, y, z));
        trees.push(tree);
    }
}
```

### Maze + L-System Decorations

```rust
// Generate maze
let maze = generate_maze(20, 20, MazeAlgorithm::Prim, seed);

// Add vines on walls using L-system
let vine = presets::vine();
for wall in maze.wall_positions() {
    let vine_string = vine.generate(3);
    let paths = interpret_turtle_2d(&vine_string, &config);
    // Transform paths to wall position...
}
```
