# TODO

## Next Up

- [x] Documentation audit - ensure all public APIs have docs (367 items fixed)
- [x] Test coverage audit - 750+ tests passing, all crates covered
  - Well-tested: resin-mesh (153), resin-audio (123), resin-vector (117)
  - Improved: resin-core (29), resin-noise (17), resin-surface (25)
- [x] Polish pass - examples, benchmarks, integration tests
  - Added 3 standalone examples: audio_synthesis, noise_texture, procgen_lsystem
  - Added criterion benchmarks for mesh and noise operations
- [x] Graph serialization - evaluated, registry approach recommended (see Backlog)
- [x] Identify new features - surveyed gaps (see New Features below)

### New Features (prioritized suggestions)

**High Value / Moderate Effort:**
- [x] UV atlas packing - pack multiple UV charts efficiently for game dev
- [x] LOD generation - automatic level-of-detail from high-poly meshes
- [x] Mesh curvature - Gaussian/mean curvature calculation
- [x] 2D signed distance fields - SDF operations for 2D (have 3D already)

**Audio Extensions:**
- [x] Audio time-stretching - phase vocoder, granular time-stretch
- [x] 3D audio / HRTF - spatial audio, binaural rendering

**Pattern/Sequencing (TidalCycles-inspired):**
- [x] Pattern combinators - `fast()`, `slow()`, `rev()`, `jux()` transformations
- ~~Pattern mini-notation~~ - Not planned (DSLs avoided per docs/philosophy.md)

**Quality of Life:**
- [x] Weight painting tools - weight smoothing, heat diffusion for skinning
- [x] Topology analysis - genus detection, manifold testing, boundary classification

## Backlog

### Geometry / Mesh

- [x] Terrain generation - heightfield, hydraulic/thermal erosion (Heightfield, HydraulicErosion, ThermalErosion)
- [x] Remeshing/retopology - uniform triangle distribution (isotropic_remesh, quadify)
- [x] Lattice deformation - FFD (free-form deformation) - already implemented
- [x] SDF from mesh - mesh → distance field conversion (mesh_to_sdf, mesh_to_sdf_fast)
- [x] Navigation meshes - walkable surface generation (NavMesh, find_path, smooth_path)

### Physics

- [x] Constraints/joints - hinges, springs, motors (DistanceConstraint, HingeConstraint, SpringConstraint)
- [x] Cloth-object collision - two-way interaction (ClothCollider, query_collision, solve_self_collision)
- [x] Soft body FEM - finite element method deformation (SoftBody, Tetrahedron, LameParameters)

### Procedural

- [x] Procedural architecture - buildings, rooms, floor plans (Building, generate_building, generate_stairs)
- [x] Road/river networks - graph-based path generation (RoadNetwork, RiverNetwork, resin-procgen)
- [x] Terrain erosion - hydraulic, thermal simulation (part of Heightfield)
- [x] Maze generation - recursive backtracker, Prim's, Kruskal's, Eller's (resin-procgen)

### Animation

- [x] Procedural walk cycles - parametric locomotion (ProceduralWalk, WalkAnimator, GaitPattern)
- [x] Secondary motion - jiggle physics, follow-through, overlap (JiggleBone, JiggleChain, FollowThrough)
- [x] Motion matching - animation database lookup (MotionDatabase, MotionMatcher, find_best_match)

### Image / Texture

- [x] Convolution filters - blur, sharpen, edge detection, emboss (Kernel, convolve, detect_edges)
- [x] Normal map from heightfield - Sobel-based normal generation (heightfield_to_normal_map)
- [x] Ambient occlusion baking - ray-based AO (bake_ao_vertices, bake_ao_texture, AoAccelerator)
- [x] Channel operations - split/merge/process R/G/B/A independently (extract_channel, split_channels, merge_channels, set_channel, swap_channels)
- [x] Chromatic aberration - RGB channel offset (radial from center) (chromatic_aberration, ChromaticAberrationConfig)
- [x] Distortion effects - barrel, pincushion, wave, displacement map (lens_distortion, wave_distortion, displace, swirl, spherize)
- [x] Inpainting - diffusion-based fill, multi-scale PatchMatch (inpaint_diffusion, inpaint_patchmatch, create_color_key_mask, dilate_mask)
- [x] Image pyramid - downsample/upsample, coarse-to-fine processing (downsample, upsample, ImagePyramid, resize)
- [x] Color adjustments - levels, curves, hue/saturation, color balance (adjust_levels, adjust_brightness_contrast, adjust_hsl, grayscale, invert, posterize, threshold)

### Audio

- [x] Convolution reverb - load and apply impulse responses (ConvolutionReverb, generate_room_ir)
- [x] Room acoustics simulation - image-source early reflections (RoomAcoustics, calculate_early_reflections, generate_ir)
- [x] Synthesizer patch system - preset save/load, modulation routing (SynthPatch, PatchBank, ModRouting)

### 2D Vector

- [x] Curve booleans - proper path intersection with winding rules (FillRule, winding_number, path_xor_multi)
- [x] Pressure curves - pen tool simulation, variable stroke width (PressureStroke, simulate_velocity_pressure)
- [x] Gradient meshes - interpolated color regions (GradientMesh, GradientPatch)

### Spatial (new crate: resin-spatial)

- [x] Quadtree - 2D spatial partitioning, point/region queries
- [x] Octree - 3D spatial partitioning, point/region queries
- [x] BVH - bounding volume hierarchy for ray/intersection queries
- [x] Spatial hash - grid-based broad phase collision detection
- [x] R-tree - rectangle/AABB queries

### Cross-Domain

> Inspired by MetaSynth, glitch art - structure is transferable between domains.

- [x] Image↔Audio - spectral painting, sonification, audio-to-image (image_to_audio, audio_to_image)
- [x] Buffer reinterpretation - treat any `&[f32]` as audio/vertices/pixels (AudioView, Vertices2DView, Vertices3DView, PixelView)
- [x] Noise-as-anything - same noise field as texture, audio modulation, displacement (field_to_audio, field_to_image, field_to_vertices_*)

### File Formats

> **Out of scope.** Complex file formats (FBX, USD, Alembic, video) are [Cambium](https://github.com/rhizome-lab/cambium)'s responsibility. Resin focuses on generation and manipulation, not I/O for proprietary formats.

### Graph Serialization

> **Status:** ✅ Fully implemented

**Crates:**
- `resin-op` - DynOp trait, OpRegistry, Pipeline, `#[derive(Op)]` macro
- `resin-serde` - SerialGraph format, NodeRegistry, JSON/bincode formats
- `resin-history` - SnapshotHistory (full snapshots), EventHistory (event sourcing)

**DynOp System (ops-as-values):**
- All domain ops derive `Op` macro: `#[derive(Op)] #[op(input = T, output = U)]`
- Each crate exports `register_ops(registry)` for pipeline deserialization
- 20 crates with dynop feature: mesh, audio, vector, image, field, procgen, physics, fluid, rig, pointcloud, scatter, lsystem, space-colonization, spring, crossdomain, rd, automata, voxel, particle, gpu

**How it works:**
1. `SerialGraph` stores nodes as `(id, type_name, params_json)` + edges
2. `NodeRegistry` maps type_name → deserializer factory function
3. `OpRegistry` maps op names → op deserializer for pipelines
4. Supports JSON (human-readable) and bincode (compact binary)
5. History: snapshots for simple undo/redo, events for fine-grained tracking

### Post-Features

- [x] Documentation pass - comprehensive docs for all new features
- [x] Test coverage pass - tests for all new features
- [x] Architecture review - evaluate patterns, identify inconsistencies (see docs/architecture-review.md)
- [x] Refactoring - HIGH: tuple returns → named structs, panics → Option; MEDIUM: collision dedup, missing traits, step() split

### Architecture / Future Extraction

- [ ] Scene graph generalization - evaluate if resin-motion's scene graph should be extracted to resin-scene for general use (2D/3D hierarchy, transforms, parent-child relationships)
- [ ] Expression AST extensibility - figure out how users can add custom functions to MotionExpr/FieldExpr without forking

### Motion Graphics

> **Status:** In progress

**Goal:** After Effects-style 2D motion graphics with vector-first approach.

**Crates:**
- `resin-motion-fn` - Motion functions: Spring, Oscillate, Wiggle, Eased, Lerp (✅ implemented)
- `resin-motion` - Scene graph: Transform2D, Layer, hierarchy, blend modes (✅ implemented)

**Features needed:**
- [x] Transform2D with anchor point (pivot for rotation/scale)
- [x] Layer hierarchy with parent-child transforms
- [x] Path trim (animate stroke reveal 0-100%) - Trim, trim_path, trim_segments in resin-vector
- [x] Stagger/offset timing for instances - Stagger, StaggerPattern, stagger_timing in resin-scatter
- [ ] Drop shadow, glow effects

**Typed Expression AST:** ✅ Implemented
- [x] MotionExpr enum - typed variants for motion functions (resin-motion-fn)
- [x] FieldExpr enum - typed variants for field functions (resin-expr-field)
- [x] Dew AST ↔ our AST conversion (to_dew_ast() implemented, from_dew_ast() future work)

## Done
- [x] Rigid body physics (RigidBody, Collider shapes, PhysicsWorld, impulse-based collision resolution)
- [x] Fluid simulation (FluidGrid2D/3D stable fluids, Sph2D/3D particle hydrodynamics)
- [x] Smoke/gas simulation (SmokeGrid2D/3D, buoyancy, temperature, dissipation)
- [x] Animation export (AnimationConfig, render_animation, export_image_sequence, export_gif)
- [x] Physical modeling percussion (Membrane, Bar, Plate, modal synthesis, drum/cymbal/bell presets)
- [x] Vocoder (Vocoder, FilterbankVocoder, spectral cross-synthesis, envelope follower)
- [x] Spectral processing (FFT, IFFT, STFT, ISTFT, window functions, pitch detection, spectral analysis)
- [x] Voxels (VoxelGrid, SparseVoxels, sdf_to_voxels, editing, dilate/erode, mesh generation)
- [x] Point clouds (PointCloud, sampling from meshes/SDFs, normal estimation, filtering, voxel downsampling)
- [x] Texture baking (BakeConfig, bake_scalar, bake_rgba, bake_vec4, export_png, anti-aliasing)

- [x] 2D rasterization (Rasterizer, scanline algorithm, FillRule, alpha compositing, stroke rasterization)

- [x] Space colonization (SpaceColonization, BranchNode, generate_tree, generate_lightning, pipe model radii)
- [x] Wave function collapse (WfcSolver, TileSet, constraint propagation, resin-procgen)
- [x] Geodesic distance (Dijkstra, fast marching, geodesic_path, find_mesh_center, iso_distance)
- [x] Mesh repair (find_boundary_loops, fill_hole_fan, fill_hole_ear_clip, fill_hole_minimum_area)
- [x] Spring physics (SpringSystem, Verlet integration, create_rope, create_cloth, create_soft_sphere)
- [x] Reaction-diffusion (ReactionDiffusion, GrayScottPreset, MultiChannelRD, 5-point Laplacian)
- [x] SVG import/parsing (parse_path_data, M/L/H/V/C/S/Q/T/A/Z commands, arc to cubic conversion)
- [x] WAV import/export (WavFile, PCM 8/16/24/32-bit, Float32, resampling)
- [x] Granular synthesis (GrainCloud, GrainConfig, GrainScheduler, Hann envelope)
- [x] Hatching patterns (HatchConfig, hatch_rect, cross_hatch_rect, hatch_polygon)
- [x] Metaballs 2D/3D (Metaball, Metaballs2D, Metaballs3D, MetaballSdf2D/3D)
- [x] Voronoi/Delaunay diagrams (Bowyer-Watson algorithm, VoronoiDiagram, VoronoiCell)
- [x] Convex hull and geometry utils (bounding_box, centroid, point_in_polygon, minimum_bounding_circle)
- [x] Cellular automata (ElementaryCA, CellularAutomaton2D, GameOfLife, rule presets)
- [x] Path simplification (simplify_path, simplify_points, smooth_path, resample_path)
- [x] Karplus-Strong string synthesis (KarplusStrong, ExtendedKarplusStrong, PolyStrings, PluckConfig)
- [x] MIDI support (MidiMessage parsing, note/freq conversion, CC constants, velocity/amplitude)

- [x] L-systems (LSystem, Rule, TurtleConfig, interpret_turtle_2d/3d, presets)
- [x] FM synthesis (FmOperator, FmOsc, FmSynth, FmAlgorithm, fm_presets)
- [x] Wavetable oscillators (Wavetable, WavetableBank, WavetableOsc, additive_wavetable, supersaw_wavetable)
- [x] Instancing/scattering (scatter_random, scatter_grid, scatter_sphere, scatter_poisson_2d, Instance)
- [x] Text to paths (Font, text_to_path, text_to_paths_outlined, TextConfig, measure_text)
- [x] Marching cubes (MarchingCubesConfig, sphere/box/torus SDFs, iso-value support)
- [x] glTF import (import_gltf, import_gltf_from_bytes, GltfScene with merge)
- [x] Mesh from curves (extrude_profile, revolve_profile, sweep_profile with caps and scaling)
- [x] Curve lofting (loft, loft_along_path, interpolation, profile helpers)
- [x] Boolean operations on 2D paths (union, intersect, subtract, xor via Sutherland-Hodgman)
- [x] Mesh decimation (edge collapse, DecimateConfig, boundary preservation, max error threshold)
- [x] Image/texture loading as fields (ImageField, WrapMode, FilterMode, bilinear sampling)
- [x] OBJ import/export (import_obj, export_obj, normals, UVs, quad triangulation)
- [x] Laplacian mesh smoothing (smooth, smooth_taubin, preserve boundary, SmoothConfig)
- [x] Animation blending and layering (layers, crossfade, 1D/2D blend trees, additive blending)
- [x] glTF export for meshes (GLB binary, embedded base64, PBR materials)

- [x] NURBS surfaces (tensor product surfaces, sphere/cylinder/torus/cone primitives, tessellation)
- [x] SVG export for 2D vector graphics (paths, shapes, styles, viewBox)
- [x] GPU compute backend for fields (wgpu compute shaders, noise generation)

- [x] NURBS curves (rational B-splines with weights, circle/ellipse/arc primitives)
- [x] Bevy integration examples (mesh generation, skeletal animation with IK)
- [x] Path offset/stroke operations (offset, stroke-to-path, dash patterns, point/tangent at length)
- [x] Edge loop/ring selection and loop cut
- [x] Bevel operations (edge/vertex bevel with configurable amount)
- [x] Mesh boolean operations (BSP-based union, subtract, intersect)
- [x] Vector network (node-based vector editing, region detection)
- [x] Half-edge mesh representation (Catmull-Clark subdivision)
- [x] Easing functions (quad, cubic, elastic, bounce, etc.)
- [x] Audio effects (reverb, chorus, phaser, flanger, distortion)
- [x] Audio graph / signal chain (Chain, Mixer, nodes)
- [x] Color types and gradients (RGB, HSL, HSV, blend modes)
- [x] Spline curves (Bezier, Catmull-Rom, B-spline)
- [x] UV projection (planar, cylindrical, spherical, box)
- [x] Mesh ops (extrude, inset, solidify, weld, flip)
- [x] Particle system (emitters, forces, turbulence)
- [x] Path constraints (bones follow curves)
- [x] Morph targets / blend shapes
- [x] Animation clips and tracks
- [x] IK solvers (CCD, FABRIK)
- [x] Subdivision surfaces (Loop)
- [x] SDF operations (smooth union, intersection, subtraction)
- [x] Domain modifiers (twist, bend, repeat, mirror)
- [x] Envelope generators (ADSR, AR, LFO)
- [x] Expression language parser (expr::Expr)
- [x] Basic texture nodes (Checkerboard, Voronoi, Stripes, Brick, Dots, SDFs)
- [x] Audio effects/filters (LowPass, HighPass, Biquad, Delay, FeedbackDelay)
- [x] Skeleton/bone types
- [x] Derive macro for DynNode
- [x] Field<I, O> trait for lazy evaluation
- [x] Noise functions (Perlin, Simplex)
- [x] Basic oscillators
- [x] 2D path primitives
- [x] Implement basic mesh primitives (box, sphere)
- [x] Set up graph evaluation system
- [x] Define attribute traits
- [x] Review all design decisions for consistency, extract patterns/philosophy to docs/philosophy.md
