# TODO

## Next Up

### Mesh Primitives

> **Goal:** Complete the basic primitive set for constructive modeling.

- [x] Cylinder primitive - `cylinder(radius, height, segments)`
- [x] Cone primitive - `cone(radius, height, segments)`
- [x] Torus primitive - `torus(major_radius, minor_radius, major_segments, minor_segments)`
- [x] Plane/grid primitive - `plane(width, depth, subdivisions_x, subdivisions_z)`
- [x] Icosphere primitive - `icosphere(subdivisions)` (geodesic sphere)
- [x] Pyramid primitive - `pyramid(base_size, height)`

### Warning Cleanup (2025-01-16)

- [x] Analyzed and fixed all compile warnings. See `docs/design/dead-code-patterns.md` for patterns and lessons learned.
  - `SvgElement::Group` kept with `#[allow(dead_code)]` - rendering support exists, just needs API

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

### Compute Backend Architecture

- [x] NodeExecutor trait - decouple node execution from evaluation logic
  - `NodeExecutor` trait in resin-core with `execute(node, inputs, ctx)` method
  - `LazyEvaluator<E: NodeExecutor>` is now generic, delegates to executor
  - `DefaultNodeExecutor` calls `node.execute()` directly (default behavior)
  - `BackendNodeExecutor` in resin-backend wraps `Scheduler` for GPU routing
  - No more duplication - just use `LazyEvaluator::with_executor(BackendNodeExecutor::new(scheduler))`
- [x] Pass node reference to GpuKernel::execute() - kernels need access to node params (expressions, configs)
  - `GpuKernel::execute()` now receives `&dyn DynNode` parameter
  - Kernels can downcast to access node-specific fields (e.g., `RemapUvNode::expr`)

### Constructive Modeling (Blender-style workflow)

> **Goal:** Interactive mesh editing - start from primitive, select elements, transform, extrude, etc.

**Primitives:** *(Moved to Next Up)*

**Selection System:**
- [ ] MeshSelection struct - store selected vertices/edges/faces as index sets
- [ ] Selection modes - vertex, edge, face selection
- [ ] Selection operations - select_all, deselect_all, invert_selection
- [ ] Selection expansion - grow_selection, shrink_selection, select_linked
- [ ] Selection by trait - select_by_normal, select_by_area, select_random
- [ ] Soft selection / proportional editing - falloff weights for smooth transforms

**Face Operations:**
- [ ] Extrude faces - `extrude_faces(selection, amount)` with new side faces
- [ ] Inset faces - `inset_faces(selection, amount, depth)` individual face inset
- [ ] Scale faces - `scale_faces(selection, factor)` scale around face centers
- [ ] Delete faces - `delete_faces(selection)` with cleanup options (verts, edges)
- [ ] Subdivide faces - `subdivide_faces(selection)` selective subdivision
- [ ] Poke faces - `poke_faces(selection)` add vertex at face center
- [ ] Triangulate faces - `triangulate_faces(selection)` convert quads to tris

**Edge Operations:**
- [ ] Bridge edge loops - `bridge_edges(loop1, loop2)` connect two edge loops
- [ ] Edge slide - `slide_edges(selection, factor)` slide along adjacent faces
- [ ] Edge crease - `crease_edges(selection, weight)` for subdivision control
- [ ] Knife/cut - `knife_cut(path)` cut through mesh with arbitrary path
- [ ] Edge split - `split_edges(selection)` duplicate edges (hard edge)

**Vertex Operations:**
- [ ] Transform vertices - `transform_vertices(selection, matrix)` arbitrary transform
- [ ] Merge vertices - `merge_vertices(selection, mode)` at center/cursor/collapse
- [ ] Rip vertices - `rip_vertices(selection)` disconnect from adjacent faces
- [ ] Smooth vertices - `smooth_vertices(selection, factor)` relax positions

**Subdivision:**
- [ ] Catmull-Clark subdivision - `subdivide_catmull_clark(mesh, levels)`
- [ ] Selective subdivision - `subdivide_selected(selection, method)`

**Edit History:** Use `resin-history` (SnapshotHistory or EventHistory) - already implemented.

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
- [x] Color matrix - linear RGBA transforms (grayscale, sepia, channel mixing) (color_matrix)
- [x] Position transform - 2D affine transforms on pixel positions (rotate, scale, translate) (transform_image, TransformConfig)
- [x] LUT support - 1D per-channel curves and 3D trilinear color grading (Lut1D, Lut3D, apply_lut_1d, apply_lut_3d)

### Audio

- [x] Convolution reverb - load and apply impulse responses (ConvolutionReverb, generate_room_ir)
- [x] Room acoustics simulation - image-source early reflections (RoomAcoustics, calculate_early_reflections, generate_ir)
- [x] Synthesizer patch system - preset save/load, modulation routing (SynthPatch, PatchBank, ModRouting)

### Graph Compilation (general, not audio-specific)

- [x] Cranelift JIT - compile graphs to native code at runtime
  - Feature-gated: `cranelift` feature in resin-audio
  - Block processing via `BlockProcessor` trait to amortize function call overhead
  - Per-sample JIT has ~15 cycle overhead; block processing amortizes across 64+ samples
  - Proof-of-concept in `resin-audio/src/jit.rs`, stateful nodes supported
  - See `docs/design/jit-optimization-log.md` for performance analysis
- [x] Static graph compilation - build.rs codegen from SerialAudioGraph
  - `resin-audio-codegen` crate with `generate_effect()` function
  - Topological sort + generic node processing (not pattern-matching)
  - ~0-7% overhead vs hand-optimized Tier 2 code
- [x] Feature-gated pre-monomorphized compositions - `optimize` feature in resin-audio
  - Pattern matching identifies tremolo/chorus/flanger graphs
  - Replaces with optimized `TremoloOptimized`, `ChorusOptimized`, `FlangerOptimized`
- [x] BlockProcessor trait - unified block-based processing interface
  - All tiers implement `BlockProcessor::process_block()`
  - Blanket impl for `AudioNode` types loops over per-sample `process()`
  - JIT uses native block impl for efficiency

### Graph Optimization Passes (pre-codegen)

> **Goal:** Transform graphs to reduce node count before execution/compilation.
> Works at graph level, not tied to any codegen backend. Benefits both dynamic execution and JIT.

**Algebraic Fusion:**
- [x] Affine chain fusion - `Gain(a) -> Offset(b) -> Gain(c) -> Offset(d)` → single multiply-add
  - `AffineNode` struct with `then()` for composition, `fuse_affine_chains()` pass
  - Reduces N nodes to 1 fused node
- [x] Delay merging - consecutive delays become single buffer with combined length
  - `merge_delays()` pass merges zero-feedback delays
- [ ] Filter cascading - NOT FEASIBLE: two cascaded 2nd-order biquads = 4th-order filter
  - Would require higher-order filter struct; biquads can't be combined into one biquad

**Simplification:**
- [x] Identity elimination - remove `Gain(1.0)`, `Offset(0.0)`, `PassThrough`
  - `eliminate_identities()` pass rewires around no-op nodes
- [x] Dead node elimination - remove nodes not connected to output
  - `eliminate_dead_nodes()` walks backwards from output via audio/param wires
- [x] Constant folding - `Constant(a) -> Gain(b)` → `Constant(a*b)`
  - `fold_constants()` pass folds Constant through Affine nodes
- [x] Constant propagation - track known-constant inputs through graph
  - `propagate_constants()` iteratively folds and fuses until fixpoint

**Implementation:**
- [x] Composable passes - `run_optimization_passes()` runs all passes until no changes
- [x] Preserve semantics - tests verify output unchanged after optimization
- [ ] `GraphOptimizer` trait - `fn optimize(&self, graph: &mut Graph)` (for extensibility)
- [ ] Generic over graph type - works on `AudioGraph`, `FieldGraph`, etc.

**JIT Compilation:** ✅
- [x] Generalized JIT in `resin-jit` crate - extract from audio, apply to any optimized graph
  - Generic traits: `JitCompilable`, `SimdCompilable`, `JitGraph`
  - Node classification: `JitCategory::PureMath`, `Stateful`, `External`
- [x] SIMD codegen - vectorize pure-math chains (4 samples at once via f32x4)
  - `compile_affine_simd()` generates SIMD loop + scalar tail
  - 41x faster than scalar JIT (5.1µs vs 209µs for 44100 samples)
  - 6.6x faster than native Rust (no bounds checking overhead)
  - Parity tests verify scalar == SIMD == native
- [x] Field expression JIT - compile `FieldExpr` AST to native `fn(x,y,z,t) -> f32`
  - **Pure Cranelift perlin2**: No Rust boundary crossing, exact parity with Rust impl
  - **Polynomial transcendentals**: sin, cos, tan, exp, ln via optimized approximations
  - Other noise (simplex, perlin3, fbm) still use external calls (future: inline these)
  - Math ops, SDF operations, conditionals compiled inline
  - 29 parity tests verify JIT == interpreted eval

### Audio Effects (guitar pedals / studio)

- [x] Compressor - dynamic range compression with attack/release/threshold/ratio (Compressor)
- [x] Limiter - brickwall limiting, lookahead (Limiter)
- [x] Noise gate - threshold-based gating with attack/hold/release (NoiseGate)
- [x] Bitcrusher - bit depth reduction, sample rate reduction (Bitcrusher)
- [ ] Wah-wah - envelope follower + bandpass, auto-wah
- [ ] Octaver - pitch tracking, sub-octave generation
- [ ] Pitch shifter - real-time pitch shifting (simple, not time-stretch)
- [ ] Ring modulator - carrier frequency modulation
- [ ] Parametric EQ - multi-band with Q control
- [ ] Graphic EQ - fixed frequency bands
- [ ] Cabinet simulation - speaker impulse responses, mic placement

### Glitch Art (image/video)

- [x] Pixel sorting - sort pixels by brightness/hue/saturation along rows/columns (pixel_sort, PixelSort)
- [x] RGB channel shift - independent X/Y offset per channel (rgb_shift, RgbShift)
- [x] Scan lines - CRT-style horizontal lines with configurable gap/intensity (scan_lines, ScanLines)
- [x] Static/noise overlay - TV static, film grain, digital noise (static_noise)
- [x] VHS tracking - horizontal displacement bands, color bleeding (vhs_tracking, VhsTracking)
- [x] JPEG artifacts - DCT block corruption, quantization artifacts (jpeg_artifacts, JpegArtifacts)
- [x] Bit manipulation - XOR/AND/OR on raw pixel bytes (bit_manip, BitManip, BitOperation)
- [ ] Datamosh (video) - P-frame/I-frame manipulation, motion vector corruption
- [x] Corrupt bytes - random byte insertion/deletion/swap in image data (byte_corrupt, ByteCorrupt, CorruptMode)

### Image Primitive Refactoring

> **Goal:** Expose minimal surface area with low-level, highly composable primitives.
> Sugar functions remain for ergonomics but delegate to primitives.

**True Primitives:**
- [ ] `remap_uv(image, Fn(Vec2) -> Vec2)` - UV coordinate remapping (runtime, not serializable)
- [ ] `map_pixels(image, Fn([f32;4]) -> [f32;4])` - per-pixel color transform
- [x] `convolve(image, Kernel)` - neighborhood operation (already exists)
- [x] `composite(image, image, BlendMode, opacity)` - blending (already exists)
- [x] `sample_uv` - texture sampling (already exists on ImageField)

**Refactor to use primitives:**
- [ ] `lens_distortion`, `wave_distortion`, `swirl`, `spherize`, `transform_image` → use `remap_uv`
- [ ] `grayscale`, `invert`, `threshold`, `posterize`, `bit_manip` → use `map_pixels`
- [ ] `blur`, `sharpen`, `emboss`, `edge_detect` → already use `convolve`

**Serialization & compilation:**
- Use Dew expressions for UV remapping and pixel transforms
- Dew AST is serializable AND compilable to:
  - Interpreter (CPU, fallback)
  - Cranelift JIT (fast CPU)
  - WGSL/GLSL (GPU shaders)
- Config structs (`LensDistortion`, etc.) remain as ergonomic sugar that generates Dew AST
- Example: `remap_uv(image, "vec2(u + sin(v * 6.28) * 0.1, v)")`

### Buffer / Channel Operations

- [ ] Per-channel transform - `map_channel(image, channel, Fn(ImageField) -> ImageField)`
- [x] Colorspace decomposition - decompose/reconstruct in HSL/HSV/LAB/YCbCr (decompose_colorspace, reconstruct_colorspace, Colorspace)
- [ ] Arbitrary channel reorder - swap/permute channels across colorspaces
- [ ] Buffer map - `map_buffer(&[f32], Fn(f32) -> f32)` for audio/image/mesh
- [ ] Buffer zip - `zip_buffers(&[f32], &[f32], Fn(f32, f32) -> f32)`
- [ ] Buffer fold - `fold_buffer(&[f32], init, Fn(acc, f32) -> acc)`
- [ ] Windowed operations - sliding window with arbitrary kernel function

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

### Complexity Hotspots

> **Status:** ✅ Reviewed - 57 functions allowed in `.moss/complexity-allow`

All high-complexity library functions reviewed. Complexity reflects inherent algorithmic needs:
- Noise/mesh algorithms (simplex, marching cubes, subdivision, remeshing)
- Physics simulations (fluid, cloth, soft body, erosion)
- Audio processing (FFT, time-stretch, room acoustics, pitch detection)
- Geometry (arc-to-cubic, Delaunay, clipping, IK solvers)
- Parsers (SVG path, OBJ) and procedural generation (mazes, roads, architecture)

Only `examples/*/main` functions remain above threshold (intentionally verbose).

### Codebase Normalization

> **Status:** ✅ Complete - see `docs/design/normalization.md`

- [x] Transform representations - added `From` conversions between Mat4/Transform2D/Transform
- [x] Interpolation trait - `Lerp` in resin-easing, implemented for common types
- [x] Cubic bezier dedup - consolidated in `resin-vector/src/bezier.rs`
- [x] Color conversions - `From<[f32; 4]> for Rgba` and vice versa
- [x] Config builders - removed ~170 boilerplate builders, kept only useful ones
- [x] Error handling - standardized on thiserror across all crates
- [x] Coordinate docs - documented in `docs/conventions.md`

### Type Unification

> **Status:** 🟡 In progress - see `docs/design/unification.md` for full analysis

**High Priority:**
- [x] Curve trait unification - implemented trait-based `Curve` design from `docs/design/curve-types.md`
  - Created `resin-curve` crate with `Curve` trait, `VectorSpace`, segment enums
  - Integrated with `resin-spline`, `resin-vector`, `resin-rig`
  - `Path3D` now uses `ArcLengthPath<Segment3D>` from unified types

**Medium Priority:**
- [x] Graph terminology clarification - distinguish node/edge meanings across domains
  - Data flow graphs (resin-core): Node, Wire (port connections)
  - Vector graphics (resin-vector): Anchor, Edge (spatial curves)
  - Spatial networks (resin-procgen): NetworkNode, NetworkEdge
  - Topology (resin-mesh): Vertex, HalfEdge, Face
  - Documented in `docs/conventions.md` under "Graph Terminology"
- [x] Transform unification - `resin-transform` crate with `SpatialTransform` trait
  - Implemented for `Transform3D` and `Transform2D`
  - Enables generic algorithms over both transform types

**Low Priority:**
- [x] Vertex attribute unification - use traits instead of per-subsystem Vertex structs
  - Implemented `HasPositions`, `HasNormals`, `HasColors` on `PointCloud`
  - Added `HasPositions2D` trait for 2D types
  - Documented SoA vs AoS limitation: traits only work with Struct-of-Arrays storage
  - AoS types (`SoftBody`, `GradientMesh`) cannot implement without restructuring

### Compute Backends

> **Goal:** Heterogeneous execution - GPU when available, CPU fallback, policy-based selection.
> See `docs/design/compute-backends.md` for full design.

- [ ] `ComputeBackend` trait - extensible backend abstraction (not enum)
- [ ] `BackendRegistry` - register/query available backends
- [ ] `CpuBackend` - default, always available
- [ ] `GpuComputeBackend` - wgpu-based, registers GPU kernels per node type
- [ ] `ExecutionPolicy` - Auto, PreferKind, Named, LocalFirst, MinimizeCost
- [ ] `BackendScheduler` - matches policy to node capabilities
- [ ] `WorkloadHint` - nodes advertise workload size for scheduling
- [ ] `Cost` model - estimate compute + transfer costs
- [ ] `DataLocation` tracking in `Value` - know where data lives
- [ ] Integration with `EvalContext` - backends + policy fields

### Architecture / Future Extraction

- [ ] Scene graph generalization - evaluate if resin-motion's scene graph should be extracted to resin-scene for general use (2D/3D hierarchy, transforms, parent-child relationships)
- [ ] Expression AST extensibility - figure out how users can add custom functions to MotionExpr/FieldExpr without forking

### Extensibility / User-Defined Processing

- [ ] Arbitrary shaders - user-provided WGSL/GLSL for image/field processing via resin-gpu
- [ ] Arbitrary waveform manipulation - user-defined sample-by-sample audio processing (Fn(f32) -> f32)
- [ ] Custom field functions - user-defined Field<I, O> implementations without forking
- [ ] Plugin system - dynamic loading of user effects (.so/.dll/.dylib)

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
- [x] Drop shadow, glow effects - drop_shadow, glow, bloom, composite in resin-image

**Typed Expression AST:** ✅ Implemented
- [x] MotionExpr enum - typed variants for motion functions (resin-motion-fn)
- [x] FieldExpr enum - typed variants for field functions (resin-expr-field)
- [x] Dew AST ↔ our AST conversion (to_dew_ast() implemented, from_dew_ast() future work)

### Live Coding / Performance Parity

> **Goal:** Performance and generality sufficient to build a live coding environment (like Strudel/TidalCycles) on top of resin.

**Why this matters:**
- Live coding requires sub-frame latency for audio, real-time for visuals
- Strudel/TidalCycles achieve this with careful architecture, not just "fast code"
- We want to be a foundation for live coding tools, not just offline generation

**Research needed:**
- [ ] Analyze Strudel architecture - how do they achieve low-latency pattern evaluation?
- [ ] Analyze TidalCycles/Haskell approach - lazy evaluation, demand-driven computation
- [ ] Profile our pattern system (`resin-audio` patterns) vs Strudel for equivalent patterns

**Potential optimizations:**
- [ ] Incremental pattern evaluation - only recompute changed parts of pattern graph
- [ ] Pre-allocated sample buffers - avoid allocation in audio callback
- [ ] Lock-free parameter updates - atomic floats for real-time safe modulation
- [ ] SIMD pattern evaluation - vectorize pattern sampling (already have SIMD JIT for audio)
- [ ] WebAssembly target - for browser-based live coding (Strudel runs in browser)

**Architecture considerations:**
- [ ] Separation of "scheduling" from "synthesis" - patterns schedule events, audio graph renders
- [ ] Hot-swappable graphs - replace effect chain without glitches
- [ ] Time model - absolute vs relative time, tempo changes, pattern alignment

**Feature parity targets (from Strudel/Tidal):**
- [x] Pattern combinators - `fast()`, `slow()`, `rev()`, `jux()` (already implemented)
- [ ] Polymetric patterns - patterns of different lengths running in parallel
- [ ] Pattern randomness - `rand`, `choose`, `shuffle` with reproducible seeds
- [ ] Euclidean rhythms - `euclid(k, n)` pattern generator
- [ ] Continuous patterns - patterns that evaluate at any point in time (not just events)

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
