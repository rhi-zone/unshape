# TODO

## Next Up

### Selection System (2025-01-17) ✅

> **Goal:** Implement mesh selection for constructive modeling workflow.

- [x] MeshSelection struct - store selected vertices/edges/faces as index sets
- [x] Selection modes - vertex, edge, face selection
- [x] Selection operations - select_all, deselect_all, invert_selection
- [x] Selection expansion - grow_selection, shrink_selection, select_linked
- [x] Selection by trait - select_by_normal, select_by_area, select_random
- [x] Soft selection / proportional editing - falloff weights for smooth transforms

Implemented in `selection.rs`. Key types: `MeshSelection`, `Edge`, `SelectionMode`, `SoftSelection`, `Falloff`.

### Mesh Primitives (2025-01-17) ✅

> **Goal:** Complete the basic primitive set for constructive modeling.

- [x] `Cuboid` - box/cube with width, height, depth
- [x] `UvSphere` - latitude/longitude sphere with radius
- [x] `Icosphere` - geodesic sphere with radius and subdivisions
- [x] `Cylinder` - with caps (low segments = prisms)
- [x] `Cone` - with base cap (low segments = pyramids)
- [x] `Torus` - donut shape
- [x] `Plane` - subdivided flat grid

All primitives are serializable structs with `apply()` method (ops-as-values pattern).
Pyramid removed - use `Cone { segments: 4, .. }` instead.

### File Splits (2025-01-29)

> **Goal:** Split massive monolithic files into submodules for maintainability.

**Critical:**
- [ ] `unshape-image/src/lib.rs` (12,984 lines) → submodules by domain
  - baking, convolve, channel, colorspace, adjust, dither/, distortion, pyramid, normals, inpaint, effects/, glitch/, frequency, expressions
  - 52 sections with completely unrelated concerns (dithering, glitch art, frequency domain, inpainting, etc.)

**High:**
- [ ] `unshape-audio/src/optimize.rs` (2,435 lines) → engine + passes
  - Separate pattern matching engine (fingerprinting, structural match) from optimization passes (affine fusion, dead code, constant fold)
  - Tier 2 optimized effects (Tremolo/Flanger/Chorus) into own module
- [ ] `unshape-audio/src/graph.rs` (2,180 lines) → params, chain, mixer, swappable, nodes/
  - 5 distinct subsystems: lock-free params, signal chain, mixer, AudioGraph, ~20 built-in node types

**Medium:**
- [ ] `unshape-mesh/src/edit.rs` (2,126 lines) → face_ops, vertex_ops, edge_ops
  - Each operation is an independent op struct; natural split by target element type

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
- [x] `Warp` op - time remapping via Dew expr (covers swing, humanize, quantize)
- See `docs/design/pattern-primitives.md` for primitive design decisions

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

**Selection System:** *(Moved to Next Up)*

**Face Operations:**
- [x] Extrude faces - `ExtrudeFaces::apply()` with side faces connecting to selection boundary
- [x] Inset faces - `InsetFaces::apply()` individual or region inset
- [x] Scale faces - `ScaleFaces::apply()` scale around face centers
- [x] Delete faces - `DeleteFaces::apply()` with orphaned vertex cleanup
- [x] Subdivide faces - `SubdivideFaces::apply()` selective midpoint subdivision
- [x] Poke faces - `PokeFaces::apply()` add vertex at face center with optional offset
- [x] Triangulate faces - `TriangulateFaces::apply()` (no-op for IndexedMesh, for HalfEdgeMesh)

**Edge Operations:**
- [x] Bridge edge loops - `BridgeEdgeLoops::apply()` connect two edge loops with segments/twist
- [x] Edge slide - `SlideEdges::apply()` slide along adjacent faces
- [x] Edge crease - `CreaseEdges::apply()` + `EdgeCreases` for subdivision control
- [x] Knife/cut - `KnifeCut::apply()` cut through mesh at barycentric points
- [x] Edge split - `SplitEdges::apply()` duplicate edges for hard edges

**Vertex Operations:**
- [x] Transform vertices - `TransformVertices::apply()` with soft selection support
- [x] Merge vertices - `MergeVertices::apply()` at center/position/first/last
- [x] Rip vertices - `RipVertices::apply()` disconnect from adjacent faces
- [x] Smooth vertices - `SmoothVertices::apply()` selection-aware Laplacian smoothing

**Subdivision:**
- [x] Catmull-Clark subdivision - `CatmullClark` op struct, `subdivide_catmull_clark(mesh, levels)`
- [x] Crease-aware Catmull-Clark - `CatmullClark { levels, creases: Option<EdgeCreases> }`
  - `catmull_clark(creases: Option<&EdgeCreases>)` on HalfEdgeMesh
  - Creases propagate to child edges through subdivision levels
  - Selection → crease: `CreaseEdges::apply(selection)` converts selection to creases

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
- [x] Dithering - quantization with various dithering patterns (dither, DitherConfig, DitherMethod)
  - [x] Ordered (Bayer 2x2, 4x4, 8x8) - fast threshold-based dithering
  - [x] Error diffusion (Floyd-Steinberg, Atkinson, Sierra, SierraTwoRow, SierraLite, JarvisJudiceNinke, Stucki, Burkes)
  - [x] Blue noise dithering - perceptually optimal, no banding artifacts
  - [x] Void-and-cluster - generates blue noise patterns (generate_blue_noise)
  - [x] Werness dithering - hybrid noise-threshold + error-absorption, preserves edges (Obra Dinn style)
    - Prior art: https://github.com/akavel/WernessDithering, https://dukope.com/devlogs/obra-dinn/tig-18/
  - [x] Riemersma dithering - error diffusion along Hilbert curve, eliminates directional artifacts
    - Prior art: https://surma.dev/things/ditherpunk/
  - [x] Temporal dithering - for animation/video (TemporalBayer, InterleavedGradientNoise, TemporalBlueNoise, QuantizeWithTemporalThreshold)

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
- [x] `GraphOptimizer` trait - `fn apply(&self, graph: &mut AudioGraph) -> usize` with `OptimizerPipeline` for composing
- [x] Generic `Optimizer<G>` trait in resin-core - works with any graph type
  - `OptimizerPipeline<G>` and `TrackedPipeline<G>` for composing passes
  - Audio `GraphOptimizer` is now blanket impl over `Optimizer<AudioGraph>`

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
- [x] Ring modulator - RingMod takes any AudioNode as carrier
- [x] Pitch shifter - pitch_shift() in spectral.rs
- [x] Peaking/Parametric EQ - BiquadCoeffs::peaking(freq, q, gain_db)
- [x] Shelf filters - BiquadCoeffs::low_shelf(), high_shelf()
- ~~Wah-wah~~ - composable: EnvelopeFollower + Biquad::bandpass
- ~~Octaver~~ - composable: pitch detection + synthesis at half freq
- ~~Graphic EQ~~ - composable: array of peaking filters at fixed frequencies
- ~~Cabinet simulation~~ - composable: ConvolutionReverb with cabinet IRs (IRs are data, not primitives)

### Glitch Art (image/video)

- [x] Pixel sorting - sort pixels by brightness/hue/saturation along rows/columns (pixel_sort, PixelSort)
- [x] RGB channel shift - independent X/Y offset per channel (rgb_shift, RgbShift)
- [x] Scan lines - CRT-style horizontal lines with configurable gap/intensity (scan_lines, ScanLines)
- [x] Static/noise overlay - TV static, film grain, digital noise (static_noise)
- [x] VHS tracking - horizontal displacement bands, color bleeding (vhs_tracking, VhsTracking)
- [x] JPEG artifacts - DCT block corruption, quantization artifacts (jpeg_artifacts, JpegArtifacts)
- [x] Bit manipulation - XOR/AND/OR on raw pixel bytes (bit_manip, BitManip, BitOperation)
- [x] Datamosh - P-frame/I-frame simulation with motion vectors, freeze blocks, decay
  - `datamosh()` for single-frame simulation, `datamosh_frames()` for two-frame
  - Motion patterns: Random, Directional, Radial, Vortex, Brightness-based
  - Builder API: block_size, intensity, decay, freeze probability, seed
- [x] Corrupt bytes - random byte insertion/deletion/swap in image data (byte_corrupt, ByteCorrupt, CorruptMode)

### Image Primitive Refactoring

> **Goal:** Expose minimal surface area with low-level, highly composable primitives.
> Sugar functions remain for ergonomics but delegate to primitives.

**True Primitives:**
- [x] `remap_uv(image, &UvExpr)` - UV coordinate remapping (serializable Dew expression)
- [x] `remap_uv_fn(image, Fn)` - UV coordinate remapping (internal, closure-based)
- [x] `map_pixels(image, &ColorExpr)` - per-pixel color transform (serializable Dew expression)
- [x] `convolve(image, Kernel)` - neighborhood operation (already exists)
- [x] `composite(image, image, BlendMode, opacity)` - blending (already exists)
- [x] `sample_uv` - texture sampling (already exists on ImageField)

**Refactor to use primitives:**
- [x] `swirl`, `spherize`, `transform_image` → use `remap_uv_fn`
- [x] `grayscale`, `invert`, `threshold`, `posterize` → use `map_pixels` + `ColorExpr`
- [x] `bit_manip` → inlines pixel iteration (bit ops not in ColorExpr, not worth int monomorphization)
- [x] `blur`, `sharpen`, `emboss`, `edge_detect` → already use `convolve`

**Serialization & compilation:**
- Use Dew expressions for UV remapping and pixel transforms
- Dew AST is serializable AND compilable to:
  - Interpreter (CPU, fallback)
  - Cranelift JIT (fast CPU)
  - WGSL/GLSL (GPU shaders)
- Config structs (`LensDistortion`, etc.) remain as ergonomic sugar that generates Dew AST
- Example: `remap_uv(image, "vec2(u + sin(v * 6.28) * 0.1, v)")`

### Buffer / Channel Operations

- [x] Per-channel transform - `map_channel(image, channel, Fn(ImageField) -> ImageField)`
- [x] Colorspace decomposition - decompose/reconstruct in HSL/HSV/LAB/YCbCr (decompose_colorspace, reconstruct_colorspace, Colorspace)
- ~~Arbitrary channel reorder~~ - N/A, use `map_pixels(img, &ColorExpr::Vec4 { r: B, g: G, b: R, a: A })`
- [x] Buffer map - `map_buffer(&[f32], Fn(f32) -> f32)` in resin-bytes
- [x] Buffer zip - `zip_buffers(&[f32], &[f32], Fn(f32, f32) -> f32)` in resin-bytes
- [x] Buffer fold - `fold_buffer(&[f32], init, Fn(acc, f32) -> acc)` in resin-bytes
- [x] Windowed operations - `windowed_buffer(&[f32], size, Fn(&[f32]) -> f32)` in resin-bytes

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

> **Out of scope.** Complex file formats (FBX, USD, Alembic, video) are [Paraphase](https://github.com/rhi-zone/paraphase)'s responsibility. Unshape focuses on generation and manipulation, not I/O for proprietary formats.

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

### Invariant Tests ✅

> **Goal:** Feature-gated statistical/mathematical property tests for modules where simple unit tests are insufficient.
> Run with `cargo test -p crate --features invariant-tests`. Keep normal test runs fast.

**resin-noise:** ✅ Implemented
- [x] Spectral slope tests via FFT (white≈0, pink≈-1, brown<-1, violet>1)
- [x] Spectral ordering test (brown < pink < white < violet)
- [x] Autocorrelation tests (white≈0, perlin/value>0.5, brown>0.8)
- [x] Distribution tests (mean, variance for white/perlin/value)
- [x] Histogram uniformity (chi-squared)
- [x] Worley properties (has zeros, F2 >= F1)
- [x] Determinism tests

**resin-image:** ✅ Implemented
- [x] Blue noise 1D/2D negative autocorrelation
- [x] Blue noise uniform distribution
- [x] Blur kernels sum to 1, preserve uniform images, reduce variance
- [x] Dithering preserves average brightness, produces binary output
- [x] Bayer field range and unique values
- [x] Grayscale idempotent, invert is involution

**resin-audio:** ✅ Implemented
- [x] Filter frequency response via FFT (lowpass/highpass/bandpass/notch/allpass)
- [x] Filter stability (bounded output), cutoff ordering
- [x] Oscillator output range, frequency accuracy via FFT peak
- [x] Phase continuity, duty cycle behavior
- [x] Envelope range, smoothness, timing accuracy
- [x] LFO range, smoothness, bipolar modes

**resin-mesh:** ✅ Implemented
- [x] Euler characteristic (sphere=2, torus=0, etc.)
- [x] Primitives are manifold, closed, orientable
- [x] Normals unit length, perpendicular to faces, outward for sphere
- [x] Subdivision triangle count, topology preservation
- [x] Geometry bounds (sphere radius, torus, cylinder, cuboid)
- [x] Merge counts, transform preservation, smoothing reduces variance

**resin-spatial:** ✅ Implemented
- [x] AABB center is midpoint, quadrants/octants cover original
- [x] Quadtree/Octree all points queryable, range query correctness, nearest is truly closest
- [x] Ray at(0) is origin, AABB hit points on surface
- [x] BVH all primitives queryable, ray hits are valid
- [x] SpatialHash cell locality, radius query complete
- [x] R-tree all rects queryable, query correctness

**resin-easing:** ✅ Implemented
- [x] All functions pass through (0,0) and (1,1)
- [x] In/out duality, in-out point symmetry
- [x] Monotonicity for polynomial ease-in/out
- [x] Smoothstep/smootherstep zero derivatives at endpoints
- [x] Lerp boundary conditions, linearity, extrapolation
- [x] Reverse/mirror preserve boundary conditions

**resin-curve:** ✅ Implemented
- [x] Curve endpoints: position_at(0) = start(), position_at(1) = end()
- [x] Position and tangent continuity
- [x] Arc length non-negative, line = Euclidean, circle = r * θ
- [x] Split preserves continuity, approximates original
- [x] Bezier degree elevation exact, control polygon >= arc length
- [x] Arc radius invariant, tangent perpendicular to radius
- [x] Path length = sum of segments, ArcLengthPath uniform speed

**resin-automata:** ✅ Implemented
- [x] Neighborhood counts (Moore=8, VN=4, Hex=6, ExtendedMoore formula)
- [x] Neighborhoods exclude origin (0,0)
- [x] GoL: block still life, blinker period-2, glider displacement (1,1)/4 gens
- [x] GoL empty stays empty
- [x] Elementary CA: Rule 90 from center, determinism, Rule 184 conservation
- [x] SmoothLife values in [0,1], empty stays near zero
- [x] HashLife matches CellularAutomaton2D (blinker, glider)
- [x] Langton's Ant: cell flip, grid values bounded, step count monotonic
- [x] LargerThanLife: boolean values
- [x] 3D CA: empty stays empty, single cell dies (B4/S5)
- [x] Turmite: grid values bounded

### Spatial Additions

- [x] k-nearest neighbor queries - `k_nearest(position, k)` for Quadtree/Octree
- [x] KD-tree - `KdTree2D`, `KdTree3D` with nearest, k_nearest, query_region, query_radius
- [x] Ball tree - `BallTree2D`, `BallTree3D` with nearest, k_nearest, query_radius

### Crate Pattern Gaps

> **Goal:** Ensure consistent patterns across all crates for serde, dynop, invariant-tests, Field trait, etc.

**serde feature missing:**
- [x] resin-curve - added `serde` feature with Serialize/Deserialize for all curve types
- [x] resin-noise - added `serde` feature for all noise structs (Perlin2D, Simplex3D, Fbm, etc.)
- [x] resin-easing - added `serde` feature for Easing enum
- [x] resin-geometry - N/A (traits only, no structs)
- [x] resin-spatial - added `serde` feature for Aabb2, Aabb3, Ray
- [x] resin-spline - added `serde` feature for all spline types
- [x] resin-surface - added `serde` feature for all surface types

**dynop feature missing:**
- [x] resin-audio - already has `dynop` feature properly configured
- [x] resin-curve - N/A (data types only, no operations)
- [x] resin-noise - N/A (serde sufficient, not used in Pipeline pattern)
- [x] resin-easing - N/A (pure functions only, no operations)
- [x] resin-spatial - N/A (data structures for queries, not operations)
- [x] resin-transform - N/A (trait only, no structs)
- [x] resin-motion - N/A (scene graph containers, motion types in motion-fn)
- [x] resin-motion-fn - N/A (generic types, serde sufficient, composed via Field trait)

**invariant-tests added:**
- [x] resin-voxel - 11 invariant tests for voxel operations
- [x] resin-field - 12 invariant tests for field composition
- [x] resin-vector - 18 invariant tests for vector operations
- [x] resin-spring - 9 invariant tests for spring physics
- [x] resin-particle - 18 invariant tests for particle systems
- [x] resin-physics - 16 invariant tests for rigid body physics

**Field trait implementations:**
- [x] resin-noise - Field implementations in resin-field (Perlin2D, Simplex2D, etc.)
- [x] resin-automata - N/A (discrete grid simulation, not continuous field)
- [x] resin-rd - N/A (discrete grid simulation, not continuous field)

**benchmarks added:**
- [x] resin-field - criterion benchmarks for noise, FBM, combinators, terrain, heightmap
- [x] resin-spatial - criterion benchmarks for quadtree, octree, BVH, spatial hash, R-tree
- [x] resin-physics - criterion benchmarks for world step, collision, integration, constraints
- [x] resin-pointcloud - criterion benchmarks for creation, sampling, normals, transforms

### Context Field Dead Code Elimination

> **Goal:** Unified context types (PixelContext, AudioContext) with all fields, but optimizers eliminate unused fields across all backends.

**Problem:** Passing `(u, v, x, y, width, height, time)` when only `(u, v)` is used wastes cycles.

**Solution:** Static analysis + DCE in all backends:
- [ ] AST analysis pass - determine which context fields are actually referenced
- [ ] Cranelift JIT - eliminate unused parameter loads, dead stores
- [ ] WGSL codegen - omit unused uniforms/varyings from generated shaders
- [ ] GLSL codegen - same as WGSL
- [ ] AOT/build.rs codegen - generate specialized functions without unused params
- [ ] Interpreter - could skip field computation but lower priority

**Context types affected:**
- `PixelContext { u, v, x, y, width, height, time }` for image expressions
- `AudioContext { sample_rate, time, dt, sample_index }` for audio expressions
- Future: `FieldContext { x, y, z, time }` for 3D fields

### Integer Image Processing / Bit-Level Primitives

> **Goal:** Low-level bit manipulation for images - enables steganography, glitch art, compression analysis.

**Types:**
- [ ] `ImageFieldU8` - `[u8; 4]` per pixel, integer representation
- [ ] `IntColorExpr` - integer color expressions with bitwise ops via dew

**Conversion ops:**
- [ ] `ToInt { range: IntRange }` - f32 0-1 → integer in range
- [ ] `FromInt { range: IntRange }` - integer in range → f32 0-1
- [ ] `IntRange` enum: `U8` (0-255), `Hue360` (0-359), `Custom { min: i32, max: i32 }`

**Bit-level ops:**
- [ ] `IntColorExpr` evaluation - full integer expression with bitwise ops via dew
  - Bit plane extract: `(r >> bit) & 1`
  - Bit plane set: `(r & ~(1 << bit)) | (source << bit)`
  - LSB embed: `(r & 0xFE) | data_bit`
  - These are expression patterns, not special ops - optimizer can recognize

**Use cases:**
- RGB LSB manipulation (0-255 range)
- Hue LSB hiding (0-359 range) - low visual impact, detectable but obscure
- Custom ranges for specific quantization schemes

### Frequency Domain Image Processing ✅

> **Goal:** FFT/DCT for frequency analysis, filtering, and manipulation.

**Primitives:** (implemented in resin-spectral shared crate)
- [x] `Fft2d` - image → complex frequency domain (two fields: real, imaginary)
- [x] `Ifft2d` - complex frequency domain → image
- [x] `Dct2d { block_size: Option<u32> }` - discrete cosine transform (JPEG-style)
- [x] `Idct2d { block_size: Option<u32> }` - inverse DCT
- [x] `FftShift` - shift DC to center for visualization/filtering

**Complex number representation:** Two separate fields (real, imaginary) - explicit, composable.

**Not primitives (compositions for optimizer):**
- ~~FrequencyMask~~ - `fft → zip(mask) → ifft`
- ~~HighPassFreq~~ - `fft → zip(radial_mask(cutoff, 0→1)) → ifft`
- ~~LowPassFreq~~ - `fft → zip(radial_mask(cutoff, 1→0)) → ifft`
- ~~BandPassFreq~~ - `fft → zip(ring_mask(lo, hi)) → ifft`

Add to image pattern-matching optimizer (like audio's tremolo/chorus recognition).

### Image Pattern-Matching Optimizer

> **Goal:** Recognize common image operation patterns and replace with optimized implementations.
> Parallel to audio's `GraphOptimizer` that recognizes tremolo/chorus/flanger.

**Frequency domain patterns:**
- [ ] `fft → mul(radial_mask) → ifft` → `LowPassFreqOptimized` / `HighPassFreqOptimized`
- [ ] `fft → mul(ring_mask) → ifft` → `BandPassFreqOptimized`

**Convolution patterns:**
- [ ] Separable kernel detection → two 1D passes instead of 2D
- [ ] `blur(blur(x))` → single blur with combined sigma

**Bit manipulation patterns:**
- [ ] `(r >> N) & 1` → `ExtractBitPlaneOptimized`
- [ ] `(r & ~(1 << N)) | (src << N)` → `SetBitPlaneOptimized`
- [ ] `(r & 0xFE) | bit` → `LsbEmbedOptimized`

**Composite patterns:**
- [ ] `original - blur(original)` → `HighPassOptimized` (unsharp component)
- [ ] `original + k * (original - blur(original))` → `UnsharpMaskOptimized`
- [ ] `lerp(original, transformed, mask)` → fused masked transform

**Implementation:**
- [ ] `ImageOptimizer` trait (parallel to `GraphOptimizer`)
- [ ] Pattern matchers for each recognized pattern
- [ ] Optimized implementations that fuse operations / use SIMD / avoid intermediate allocations

### Comprehensive Primitive Decomposition Pass

> **Goal:** Audit all ops across resin to find decomposition opportunities. Many "primitives" are compositions.

**Mindset examples (what we've found so far):**

| Looked like primitive | Actually is |
|-----------------------|-------------|
| `SdfToMask` | SDF is already a field, just use it (threshold = `map(\|d\| d < 0.0)`) |
| `ApplyMasked<Op>` | `Mix<original, transformed, mask>` |
| `MaskUnion/Intersect/Subtract` | Field math: `max(a,b)`, `min(a,b)`, `a*(1-b)` |
| `HighPassFreq { cutoff }` | `fft → zip(radial_mask) → ifft` |
| `LowPassFreq { cutoff }` | `fft → zip(radial_mask_inv) → ifft` |
| `ExtractBitPlane { bit }` | IntColorExpr: `(r >> bit) & 1` |
| `SetBitPlane { bit, src }` | IntColorExpr: `(r & ~(1 << bit)) \| (src << bit)` |
| `Add<A, B>` | `Zip<A, B, "a + b">` |
| `Mul<A, B>` | `Zip<A, B, "a * b">` |
| `GaussianBlur { sigma }` | `convolve(gaussian_kernel(sigma))` - kernel is data, not op |
| `UnsharpMask { amount }` | `original + amount * (original - blur(original))` |

**Questions to ask for each op:**
1. Can this be expressed as a composition of existing ops?
2. Is this just an expression pattern the optimizer should recognize?
3. Is the "parameter" actually data (kernel, LUT, mask) rather than configuration?
4. Would a more general op subsume this and others?

**Audit targets:**
- [ ] resin-image - all ops, filters, effects
- [ ] resin-audio - effects, filters (already did some: wah = envelope + bandpass)
- [ ] resin-mesh - operations, modifiers
- [ ] resin-vector - path operations
- [ ] resin-field - combinators (are Add/Mul/Mix all needed, or just Zip + Map?)

**Outcome:** Minimal primitive set + pattern-matching optimizer for common compositions.

### Field Crate Organization

> **Goal:** Separate trait definitions from combinator implementations.

**Problem:** resin-field currently defines both the `Field` trait AND the combinator ops (Map, Add, Mul, etc.). This conflates interface and implementation.

**Proposed split:**
- [ ] `resin-field` - just the `Field<I, O>` trait and core types
- [ ] `resin-field-ops` - combinator structs (Map, Zip, Mix, domain transforms)

**Benefits:**
- Cleaner dependency graph
- Trait can be implemented without pulling in all combinators
- Matches pattern used elsewhere (e.g., Iterator trait vs itertools)

### Field Combinator Simplification

> **Goal:** Reduce field combinators to true primitives.

**True primitives (implemented):**
- [x] `Zip<A, B>` - general binary: two fields → tuple, compose with Map for any op
- [x] `Zip3<A, B, C>` - general ternary: three fields → triple
- [x] `Map<F, M>` - general unary: transform output
- [x] Helper functions: `add()`, `mul()`, `sub()`, `div()`, `lerp()`, `mix()` — all Zip + Map sugar

**Removed (deprecated → deleted):**
- [x] ~~`Add<A, B>`~~ — removed, use `add(a, b)` or `zip(a, b).map(|(x, y)| x + y)`
- [x] ~~`Mul<A, B>`~~ — removed, use `mul(a, b)` or `zip(a, b).map(|(x, y)| x * y)`
- [x] ~~`Mix<A, B, Blend>`~~ — removed, use `lerp(a, b, t)` or `zip3(a, b, t).map(...)`
- [x] ~~Field trait methods `.add()`, `.mul()`, `.mix()`~~ — removed from trait

**Generic helpers (output type via traits):**
- [x] `lerp()`, `mix()` — generic over `Lerp` trait (f32, Vec2, Vec3, Rgba, etc.)
- [x] `add()`, `sub()`, `mul()`, `div()` — generic over arithmetic traits (f32, Vec2, Vec3, etc.)
- Note: Rgba doesn't implement Add/Mul (color arithmetic is nonsensical), but Lerp works

**Not needed (just use combinators):**
- ~~SdfToMask~~ - SDF is already a field, threshold via `map(|d| if d < 0.0 { 1.0 } else { 0.0 })`
- ~~ApplyMasked~~ - masking is just lerp with a second value:
  - `lerp(original, transparent, mask)` = alpha mask
  - `lerp(original, black, mask)` = fade to black
  - `lerp(original, transformed, mask)` = selective transform
  - `lerp(a, b, Constant::new(0.5))` = 50% blend
- ~~MaskUnion/Intersect/Subtract~~ - just field math: `max(a,b)`, `min(a,b)`, `a*(1-b)`

### Spread Spectrum / Quantization ✅

> **Goal:** Primitives for robust signal embedding.

**Ops:** (implemented in resin-image)
- [x] `SpreadSpectrum { seed, factor }` - add factor × pseudorandom ±1 sequence
- [x] `UnspreadSpectrum { seed, factor }` - reverse spread (same seed)
- [x] `QuantizeWithBias { levels }` - round toward bias image values

### Image Ops-as-Values Refactor ✅

> **Goal:** Convert remaining function-based image ops to struct-based ops.

**Converted (functions now delegate to op structs):**
- [x] `map_pixels` → `MapPixels { expr }`
- [x] `remap_uv` → `RemapUv { expr }`
- [x] `convolve` → `Convolve { kernel }`
- [x] `composite` → `Composite { mode, opacity }`
- [x] `resize` → `Resize { width, height }`

**Remaining:**
- [ ] `color_matrix` → `ColorMatrix { matrix }` (if needed - may be ColorExpr composition)

See DECOMPOSITION-AUDIT.md for which are true primitives vs compositions.

### Dew → Rust Codegen (AOT)

> **Goal:** Generate Rust source from dew expressions for compile-time optimization.

**Outputs:**
- [ ] `expr.to_rust_code() -> String` - text output for build.rs
- [ ] `expr.to_tokens() -> TokenStream` - for proc-macros (feature-gated on `proc-macro2`)

**Use cases:**
- build.rs generates optimized effect implementations (parallel to audio codegen)
- `#[derive(CompiledField)]` generates Field impl at compile time
- Avoids JIT overhead, rustc optimizes further, better debugging
- Pattern-matching optimizer runs at build time, emits fused implementations

**Integration with optimizers:**
- Run pattern recognition at build time
- Emit specialized Rust for recognized patterns
- Example: detect `fft → mask → ifft`, emit single fused function

### Graph Format Specification

> **Goal:** Document the JSON graph serialization format - cross-language spec for interop.

**Problem:** `SerialGraph` in resin-serde defines a JSON format, but it's only documented in Rust code. Other languages need a spec to read/write resin graphs.

**Proposed spec (`docs/spec/graph-format.md` + JSON Schema):**
- [ ] `SerialGraph` structure: nodes array, edges array, metadata
- [ ] Node format: `{ "id": "...", "type": "...", "params": {...} }`
- [ ] Edge format: `{ "from": "node:port", "to": "node:port" }`
- [ ] Type registry: how node types map to implementations
- [ ] Expression format: dew AST serialization

**Example:**
```json
{
  "nodes": [
    { "id": "n1", "type": "resin::Oscillator", "params": { "waveform": "sine", "frequency": 440.0 } },
    { "id": "n2", "type": "resin::LowPass", "params": { "cutoff": 1000.0, "resonance": 0.7 } }
  ],
  "edges": [
    { "from": "n1:out", "to": "n2:in" }
  ]
}
```

**Optimizer hints (advisory, separate doc):**
- [ ] Common patterns ranked by frequency (lerp, bit ops, etc.)
- [ ] Suggested optimizations (not mandatory)
- [ ] GPU intrinsic mappings where applicable

**Sync requirement:** Update when SerialGraph format changes.

**Benefits:**
- Cross-language: JS/Python/etc can read/write graphs
- Auditable: implementations can validate against schema
- Tooling: editors, visualizers, converters

### Cellular Automata Extensions (2025-01-25) ✅

> **Goal:** Expand unshape-automata with custom neighborhoods, new CA types, and advanced algorithms.

**Foundation:**
- [x] `Neighborhood2D` / `Neighborhood3D` traits - abstract neighborhood shapes
  - `fn offsets(&self) -> &[(i32, i32)]` for 2D, `&[(i32, i32, i32)]` for 3D
  - Implementations: `Moore`, `VonNeumann`, `Hexagonal`, `ExtendedMoore`, `CustomNeighborhood2D`
  - `CellularAutomaton2D` refactored to use `Neighborhood2D` trait

**Larger than Life (LtL):**
- [x] `LtlRules` - range-based birth/survive rules with configurable radius
- [x] `LargerThanLife` struct - generalization with extended Moore neighborhoods
- [x] Presets: BUGS, WAFFLE, GLOBE, MAJORITY, MAJORITY_SMOOTH

**3D Cellular Automata:**
- [x] `CellularAutomaton3D` - 3D grid with B/S rules
  - `Moore3D` (26-neighbor) and `VonNeumann3D` (6-neighbor) neighborhoods
  - `CustomNeighborhood3D` for arbitrary 3D offsets
- [x] Presets: LIFE_3D_445, LIFE_3D_4555, LIFE_3D_AMOEBA, CLOUDS_3D, PYROCLASTIC_3D, CRYSTAL_3D

**Turmites / Langton's Ant:**
- [x] `LangtonsAnt` - classic multi-state ant with turn rules
- [x] `Turmite` - generalized multi-state ant with state machine
  - `TurmiteRule`: `(cell_color, ant_state) -> (new_color, turn, new_state)`
- [x] Presets: classic RL, LLRR, LRRRRRLLR, Fibonacci, Cardioid

**SmoothLife:**
- [x] `SmoothLife` - continuous-space, continuous-state Life
  - `f32` state values (0.0-1.0) with smooth sigmoid transitions
  - Inner/outer disk radii for neighbor computation
- [x] `SmoothLifeConfig` - birth/death thresholds, sigmoid steepness, radii
- [x] Presets: `standard()`, `fluid()`, `slow()`

**Wang Tiles (in unshape-procgen):**
- [x] `WangTile` - tile with colored edges (N/E/S/W)
- [x] `WangTileSet` - collection with `to_tileset()` conversion to WFC
- [x] `solve_wang_tiling(tileset, width, height, seed)` - wrapper around `WfcSolver`
- [x] Presets: `two_color_corners`, `blob_tiles`, `three_color`

**HashLife:**
- [x] `HashLife` - quadtree-based universe for Conway's Game of Life
  - Structural sharing (hash-consed nodes)
  - Arbitrary coordinate support (including negative)
  - Population tracking and bounds computation

**HashLife (Optimization) ✅:**
- [x] Memoized recursive algorithm with `advance(node, step_log2)`
  - Base case: `advance_level2` computes GoL on 4×4 grid
  - Full speed: `advance_full` - two rounds of recursion for 2^(L-2) generations
  - Slow mode: `advance_slow` - one round + center extraction for arbitrary 2^n
  - 9 overlapping sub-squares via `nine_sub_squares()` helper
  - Cache: `HashMap<(node_id, step_log2), result_node_id>`
- [x] `step_pow2(n)` - advance 2^n generations with memoization
- [x] `steps(n)` - decomposes n into powers of 2 for optimal memoization
- [x] `from_ca2d(ca)` - convert from CellularAutomaton2D
- [x] Cache eviction: clear when exceeding 1M entries
- [x] Cache invalidation: `set_cell()` clears result cache

### WFC AdjacencySource Refactoring ✅

> **Goal:** Make `WangTileSet` a first-class primitive instead of converting to `TileSet`.

**Problem:** Converting 1000 Wang tiles → TileSet = 1,000,000 adjacency rules (O(N²)). Not viable.

**Solution:** Co-equal primitives unified by a trait (see `docs/design/general-internal-constrained-api.md`):

- [x] `AdjacencySource` trait with `tile_count()`, `weight()`, `valid_neighbors()`
- [x] `ValidNeighbors` enum iterator (HashSet::Iter or slice::Iter, no boxing)
- [x] Implement `AdjacencySource` for `TileSet` (existing HashMap lookup)
- [x] Implement `AdjacencySource` for `WangTileSet`:
  - Edge-color indices: `west_index`, `north_index`, `east_index`, `south_index`
  - Adjacency lookup via index: O(C³) where C = colors
- [x] Make `WfcSolver<A: AdjacencySource>` generic
- [x] `solve_wang_tiling()` now uses `WangTileSet` directly (no O(N²) conversion)
- [x] Keep `to_tileset()` for backwards compatibility (still useful for debugging/serialization)
- [x] Tests for AdjacencySource implementations

### Architecture / Future Extraction

- [ ] Scene graph generalization - evaluate if resin-motion's scene graph should be extracted to resin-scene for general use
- [ ] Expression AST extensibility - user-defined functions in MotionExpr/FieldExpr without forking
- [ ] End-to-end shader compilation - dew/ColorExpr/UvExpr to standalone WGSL/GLSL shaders

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
- [x] Pre-allocated sample buffers - `SpectralWorkspace`, `TimeStretchWorkspace`, `fft_into`, `ifft_into`
- [x] Lock-free parameter updates - `AtomicF32`, `AtomicParams` for real-time safe modulation
- [ ] SIMD pattern evaluation - vectorize pattern sampling (already have SIMD JIT for audio)
- [ ] WebAssembly target - for browser-based live coding (Strudel runs in browser)

**Architecture considerations:**
- [ ] Separation of "scheduling" from "synthesis" - patterns schedule events, audio graph renders
- [x] Hot-swappable graphs - `SwappableGraph` for glitch-free effect chain replacement
  - Equal-power crossfade during transition
  - Configurable crossfade duration
  - Cancel support for reverting to previous graph
- [ ] Time model - absolute vs relative time, tempo changes, pattern alignment

**Feature parity targets (from Strudel/Tidal):**
- [x] Pattern combinators - `fast()`, `slow()`, `rev()`, `jux()` (already implemented)
- [x] Polymetric patterns - `polymeter()`, `polyrhythm()` for different-length patterns
- [x] Pattern randomness - `rand`, `choose`, `shuffle` with reproducible seeds
- [x] Euclidean rhythms - `euclid(k, n)` pattern generator
- [x] Continuous patterns - `Continuous<T>` for LFOs and control signals (evaluate at any time point)

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
