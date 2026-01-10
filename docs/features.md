# Feature Catalog

Complete listing of resin's capabilities organized by domain.

## Audio

### resin-audio

Complete procedural audio synthesis and processing.

**Synthesis:**
- `Oscillator` - sine, saw, square, triangle, pulse with BLEP antialiasing
- `FmSynth`, `FmOsc`, `FmOperator`, `FmAlgorithm` - FM synthesis with preset algorithms
- `Wavetable`, `WavetableBank`, `WavetableOsc` - wavetable synthesis
- `KarplusStrong`, `ExtendedKarplusStrong`, `PolyStrings` - physical string modeling
- `Bar`, `Membrane`, `Plate` - physical percussion (modal synthesis)
- `GrainCloud`, `GrainConfig`, `GrainScheduler` - granular synthesis

**Envelopes & Modulation:**
- `Adsr`, `AdsrState` - ADSR envelope generator
- `Ar` - attack-release envelope
- `Lfo`, `LfoWaveform` - low-frequency oscillator

**Effects:**
- `Reverb`, `ConvolutionReverb` - algorithmic and convolution reverb
- `Delay`, `FeedbackDelay` - delay lines
- `Chorus`, `Flanger`, `Phaser`, `Tremolo` - modulation effects
- `Distortion`, `DistortionMode`, `SoftClip` - distortion/saturation
- `Biquad`, `LowPass`, `HighPass` - resonant filters
- `FilterbankVocoder`, `Vocoder` - vocoding

**Spectral:**
- `fft()`, `ifft()`, `stft()`, `istft()` - FFT operations
- `StftConfig`, `StftResult` - STFT configuration
- `estimate_pitch()`, `spectral_centroid()` - analysis
- `hamming_window()`, `hann_window()`, `blackman_window()` - window functions

**Infrastructure:**
- `AudioGraph`, `Chain`, `Mixer` - signal routing
- `RoomAcoustics`, `RoomGeometry` - room simulation
- `SynthPatch`, `PatchBank`, `ModRouting` - patch management
- `WavFile`, `WavFormat` - WAV import/export
- `note_to_freq()`, `freq_to_note()`, `velocity_to_amplitude()` - MIDI utilities

---

## Mesh & 3D Geometry

### resin-mesh

Comprehensive 3D mesh operations.

**Primitives:**
- `box_mesh()`, `sphere()`, `uv_sphere()`, `cylinder()`, `torus()`, `plane()`
- `MeshBuilder` - fluent mesh construction

**Boolean Operations:**
- `boolean_union()`, `boolean_subtract()`, `boolean_intersect()` - CSG operations

**Subdivision & Smoothing:**
- `subdivide_loop()`, `subdivide_linear()` - subdivision
- `smooth()`, `smooth_taubin()` - Laplacian smoothing
- `decimate()` - edge-collapse decimation
- `isotropic_remesh()`, `quadify()` - remeshing

**Editing:**
- `inset()`, `extrude()`, `bevel_edges()`, `split_faces()`
- `flip_normals()`, `recalculate_normals()`

**From Curves:**
- `loft()`, `extrude_profile()`, `revolve_profile()`, `sweep_profile()`
- `marching_cubes()` - isosurface extraction

**Terrain:**
- `Heightfield` - heightfield mesh generation
- `HydraulicErosion`, `ThermalErosion`, `CombinedErosion` - erosion simulation

**UV & Baking:**
- UV projection (planar, cylindrical, spherical, box)
- `bake_ao_vertices()`, `bake_ao_texture()` - ambient occlusion

**Navigation:**
- `NavMesh`, `NavPath` - navigation mesh
- `create_grid_navmesh()`, `find_path()`, `smooth_path()`

**Architecture:**
- `generate_building()`, `generate_walls()`, `generate_stairs()`

**Topology:**
- `HalfEdgeMesh` - half-edge representation for topology ops
- `Mesh` - indexed triangle mesh

**SDF:**
- `SdfGrid` - signed distance field
- `mesh_to_sdf()`, `mesh_to_sdf_fast()` - mesh to SDF
- `raymarch()` - SDF ray marching

**I/O:**
- `import_obj()`, `export_obj()` - OBJ format

### resin-gltf

glTF 2.0 import/export.

- `GltfExporter` - export meshes to .gltf/.glb
- `import_gltf()`, `import_gltf_from_bytes()` - import
- `GltfMaterial` - PBR material support

### resin-pointcloud

Point cloud operations.

- `PointCloud` - positions, normals, colors
- `sample_mesh_uniform()`, `sample_mesh_random()` - sample from mesh
- `sample_sdf_uniform()`, `sample_sdf_random()` - sample from SDF
- `estimate_normals()` - normal estimation
- `downsample_random()`, `downsample_grid()` - downsampling

### resin-voxel

Voxel operations.

- `VoxelGrid<T>` - dense voxel grid
- `SparseVoxelGrid` - sparse storage
- `fill_sphere()`, `fill_box()`, `fill_ellipsoid()` - shape filling
- `voxel_to_mesh()` - greedy meshing
- `dilate()`, `erode()` - morphological ops

### resin-surface

NURBS surfaces.

- `NurbsSurface` - tensor product NURBS
- `evaluate()`, `normal()`, `tessellate()`
- `nurbs_sphere()`, `nurbs_cylinder()`, `nurbs_torus()` - primitives

### resin-spline

Curve types.

- `CubicBezier<T>` - cubic Bezier segments
- `CatmullRom<T>` - Catmull-Rom spline
- `BSpline<T>` - B-spline
- `Nurbs<T>` - NURBS curve
- `evaluate()`, `derivative()`, `split()`, `arc_length()`

---

## 2D Vector Graphics

### resin-vector

Comprehensive 2D path operations.

**Primitives:**
- `line()`, `rect()`, `circle()`, `ellipse()`, `polygon()`, `star()`
- `Path`, `PathBuilder`, `PathCommand`

**Boolean Operations:**
- `path_union()`, `path_intersect()`, `path_subtract()`, `path_xor()`
- `FillRule` - even-odd and non-zero winding

**Stroke & Offset:**
- `stroke_to_path()` - stroke to outline
- `offset_path()` - parallel offset
- `dash_path()` - dashed pattern

**Path Operations:**
- `simplify_path()`, `smooth_path()`, `resample_path()`
- `path_length()`, `point_at_length()`, `tangent_at_length()`

**Geometry:**
- `convex_hull()`, `bounding_box()`, `centroid()`
- `minimum_bounding_circle()`, `polygon_area()`
- `path_contains_point()`, `path_winding_number()`

**Triangulation:**
- `delaunay_triangulation()` - Delaunay from points
- `voronoi_diagram()` - Voronoi partitioning

**Vector Networks:**
- `VectorNetwork` - node/edge/region structure
- `Node`, `Edge`, `Region` - network primitives

**Gradient Mesh:**
- `GradientMesh`, `GradientPatch`
- `linear_gradient_mesh()`, `four_corner_gradient_mesh()`

**Text:**
- `text_to_path()`, `text_to_paths_outlined()`
- `Font`, `measure_text()`

**Hatching:**
- `hatch_polygon()`, `cross_hatch_polygon()`
- `HatchConfig` - spacing, angle

**Rasterization:**
- `rasterize_path()`, `rasterize_polygon()`

**SVG:**
- `export_svg()`, `import_svg()`

---

## Image & Texture

### resin-image

Image as field + processing.

**Sampling:**
- `ImageField` - image as `Field<Vec2, Rgba>`
- `WrapMode` - Repeat, Clamp, Mirror
- `FilterMode` - Nearest, Bilinear

**Convolution:**
- `Kernel` - convolution kernel
- `convolve()` - apply kernel
- `blur()`, `sharpen()`, `detect_edges()`, `emboss()`
- Presets: `box_blur()`, `gaussian_blur_3x3()`, `gaussian_blur_5x5()`, `sobel_horizontal()`, `sobel_vertical()`, `laplacian()`

**Channel Operations:**
- `Channel` - Red, Green, Blue, Alpha enum
- `extract_channel()`, `split_channels()`, `merge_channels()` - channel manipulation
- `set_channel()`, `swap_channels()` - in-place channel operations

**Color Adjustments:**
- `LevelsConfig`, `adjust_levels()` - input/output mapping, gamma
- `adjust_brightness_contrast()` - simple brightness/contrast
- `HslAdjustment`, `adjust_hsl()` - hue shift, saturation, lightness
- `grayscale()`, `invert()`, `posterize()`, `threshold()` - color effects

**Distortion:**
- `LensDistortionConfig`, `lens_distortion()` - barrel/pincushion distortion
- `WaveDistortionConfig`, `wave_distortion()` - sine wave distortion
- `displace()` - displacement mapping from another image
- `swirl()` - twist/swirl effect
- `spherize()` - spherical bulge/pinch
- `ChromaticAberrationConfig`, `chromatic_aberration()` - RGB channel offset

**Image Pyramid:**
- `downsample()`, `upsample()` - 2x scaling with filtering
- `ImagePyramid` - multi-scale image representation
- `ImagePyramid::gaussian()`, `ImagePyramid::laplacian()` - pyramid types
- `resize()` - arbitrary resize with bilinear interpolation

**Baking:**
- `bake_scalar()`, `bake_rgba()`, `bake_vec4()` - field to image
- `BakeConfig` - resolution, anti-aliasing samples

**Utilities:**
- `heightfield_to_normal_map()` - normal map generation
- `export_png()` - PNG export

---

## Color

### resin-color

Color spaces and gradients.

**Color Types:**
- `LinearRgb` - linear RGB (0-1)
- `Hsl` - hue, saturation, lightness
- `Hsv` - hue, saturation, value
- `Rgba` - RGB with alpha

**Operations:**
- Color space conversions (RGB ↔ HSL ↔ HSV)
- `lerp()` - interpolation
- `blend()` - blend modes
- `luminance()`, `clamp()`

**Gradients:**
- `Gradient`, `ColorStop`
- `grayscale()`, `rainbow()`, `heat()`, `viridis()`, `inferno()`

**Blend Modes:**
- Normal, Multiply, Screen, Overlay, Add, Subtract, Difference, etc.

---

## Animation & Rigging

### resin-rig

Complete character rigging system.

**Skeleton:**
- `Skeleton`, `Bone`, `BoneId`
- `Pose`, `Transform`

**Animation:**
- `AnimationClip`, `Track<T>`, `Keyframe<T>`
- `AnimationPlayer` - playback with blending
- `Interpolation` - Linear, Cubic, Step

**Blending:**
- `AnimationStack`, `AnimationLayer`
- `BlendNode` - tree-based blending
- `Crossfade` - smooth transitions

**IK:**
- `IkChain`, `IkConfig`
- `solve_fabrik()` - FABRIK solver
- `solve_ccd()` - CCD solver

**Motion Matching:**
- `MotionDatabase`, `MotionClip`, `MotionFrame`
- `MotionMatcher`, `MotionQuery`
- `compute_match_cost()`

**Procedural:**
- `ProceduralWalk`, `ProceduralHop`
- `WalkAnimator`, `GaitPattern`, `GaitConfig`

**Secondary Motion:**
- `JiggleBone`, `JiggleChain`, `JiggleMesh`
- `FollowThrough`, `Drag`
- `WindForce`, `apply_wind_to_chain()`

**Constraints:**
- `Constraint`, `ConstraintStack`
- `PathConstraint` - constrain to path

**Skinning:**
- `Skin`, `VertexInfluences`
- `apply_skin()` - mesh deformation

### resin-easing

Animation easing functions.

- 31 easing types: Linear, Quad, Cubic, Quart, Quint, Sine, Expo, Circ, Back, Elastic, Bounce
- Each with In/Out/InOut variants
- `smoothstep()`, `smootherstep()`

---

## Physics

### resin-physics

Rigid body and soft body simulation.

**Rigid Bodies:**
- `RigidBody` - dynamic/static body
- `Collider` - Sphere, Box, Plane
- `PhysicsWorld` - simulation container
- `Contact` - collision contact

**Constraints:**
- `DistanceConstraint`, `PointConstraint`
- `HingeConstraint`, `SpringConstraint`

**Soft Bodies:**
- `SoftBody` - tetrahedral FEM
- `Cloth` - cloth simulation
- `Tetrahedron`, `LameParameters`
- `solve_self_collision()`

### resin-spring

Spring physics with Verlet integration.

- `Particle` - point mass
- `SpringConfig` - stiffness, damping
- `SpringSystem` - particles + springs
- `AngleConstraint` - angular constraint
- `pin_particle()`, `step()`

### resin-particle

Particle systems.

- `Particle` - position, velocity, lifetime, size, color
- `Emitter` trait - custom emission
- `Force` trait - custom forces

### resin-fluid

Fluid simulation.

**Grid-Based:**
- `FluidGrid2D`, `FluidGrid3D` - stable fluids
- `FluidConfig` - diffusion, iterations

**Particle-Based:**
- `Sph2D`, `Sph3D` - smoothed particle hydrodynamics
- `SphConfig` - particle parameters

---

## Procedural Generation

### resin-noise

Core noise functions.

- `perlin2()`, `perlin3()` - Perlin noise
- `simplex2()`, `simplex3()` - Simplex noise
- `fbm_perlin2()`, `fbm_simplex2()` - fractional Brownian motion

### resin-automata

Cellular automata.

- `ElementaryCA` - 1D Wolfram rules (0-255)
- `CellularAutomaton2D` - 2D with birth/survival rules
- `GameOfLife` - Conway's Game of Life
- Presets: `LIFE`, `HIGH_LIFE`, `SEEDS`, `MAZE`, `DIAMOEBA`

### resin-procgen

Procedural generation algorithms.

**Maze Generation:**
- `Maze`, `MazeAlgorithm`
- Algorithms: Recursive Backtracker, Prim's, Kruskal's, Eller's

**Wave Function Collapse:**
- `TileSet`, `TileId` - type-safe tile system
- `NamedTileSet` - ergonomic string-based API
- `WfcSolver` - WFC algorithm
- `platformer_tileset()`, `maze_tileset()` - presets

**Networks:**
- `RoadNetwork`, `RiverNetwork`
- Graph-based path generation

### resin-rd

Reaction-diffusion.

- `ReactionDiffusion` - Gray-Scott model
- `GrayScottPreset` - Mitosis, Coral, Maze, Worms, Fingerprint, Spots
- `add_seed_circle()`, `add_seed_rect()`, `step()`

### resin-lsystem

L-systems.

- `LSystem` - axiom + rules
- `Rule` - deterministic or stochastic
- `TurtleConfig` - turtle graphics
- `interpret_turtle_2d()`, `interpret_turtle_3d()`

### resin-space-colonization

Space colonization for branching.

- `SpaceColonization`, `SpaceColonizationConfig`
- `BranchNode`, `BranchEdge`
- `add_attraction_points_sphere()`, `run()`

---

## Fields & Expressions

### resin-field

Lazy spatial computation.

- `Field<I, O>` trait - core abstraction
- `sample()`, `map()`, `scale()`, `translate()`
- `add()`, `mul()`, `mix()` - combinators
- `EvalContext` - evaluation with time

### resin-expr-field

Expression language for fields.

- `ExprField` - parsed expression as field
- Math: `+`, `-`, `*`, `/`, `sin`, `cos`, `sqrt`, `pow`, `abs`, `min`, `max`
- Noise: `noise()`, `perlin()`, `simplex()`, `fbm()`

---

## Instancing & Scattering

### resin-scatter

Instance placement.

- `Instance` - transform (position, rotation, scale)
- `scatter_random()` - random in volume
- `scatter_grid()` - grid-based
- `scatter_poisson()` - Poisson disk sampling

---

## GPU Acceleration

### resin-gpu

GPU compute via wgpu.

- `GpuContext` - GPU initialization
- `GpuTexture` - texture resource
- `generate_noise_texture()` - GPU noise generation
- `NoiseType`, `NoiseConfig` - noise parameters

---

## Summary

| Domain | Crates | Highlights |
|--------|--------|------------|
| Audio | 1 | FM, wavetable, granular, physical modeling, effects, spectral |
| Mesh | 6 | Booleans, subdivision, remeshing, terrain, NURBS, voxels |
| 2D Vector | 1 | Paths, booleans, networks, gradients, text, SVG |
| Image | 1 | Convolution, baking, normal maps |
| Color | 1 | Color spaces, gradients, blend modes |
| Animation | 2 | Skeleton, IK, motion matching, secondary motion, easing |
| Physics | 4 | Rigid body, soft body, cloth, springs, particles, fluids |
| Procedural | 5 | Noise, automata, WFC, L-systems, reaction-diffusion |
| Fields | 2 | Lazy evaluation, expression language |
| GPU | 1 | wgpu compute for noise/textures |
| Export | 1 | glTF import/export |
