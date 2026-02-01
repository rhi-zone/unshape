# Primitive Decomposition Audit

Track progress auditing each crate for decomposition opportunities.

## Status

| Crate | Status | Primitives Found | Key Insight |
|-------|--------|------------------|-------------|
| resin-audio | done | 9 | DelayLine, PhaseOsc, Biquad, EnvelopeFollower, Allpass1, FFT/IFFT, AffineNode, Smoother, Mix |
| resin-audio-codegen | skip | — | Codegen, not ops |
| resin-automata | done | 12 | Step ops per CA type + HashLife memoized advance. Neighborhoods are pluggable, configs/presets are helpers |
| resin-backend | skip | — | Infrastructure |
| resin-bytes | skip | — | Infrastructure (byte casting utilities) |
| resin-color | done | 3 | Gamma, HSL/HSV transform, hue-wrapped lerp. Blend modes are component-wise ops |
| resin-core | skip | — | Infrastructure (node graph system, DynNode, Value) |
| resin-crossdomain | done | 2 | ImageToAudio (additive synthesis), AudioToImage (STFT spectrogram) |
| resin-curve | done | 4 | Line, QuadBezier, CubicBezier, Arc. All ops decompose to position_at/tangent_at |
| resin-easing | done | 0 | All easing = dew expressions (t*t, sin, pow, etc). Ergonomic presets only |
| resin-expr-field | skip | — | Infrastructure (dew→field bridge, FieldExpr AST) |
| resin-field | done | 5 | Map, Zip, Zip3, FnField, Twist/Bend/Repeat. Add/Mul/Mix removed (were Zip+Map) |
| resin-fluid | done | 4 | Stable Fluids (advect/diffuse/project), SPH, Smoke buoyancy, Dissipation |
| resin-geometry | skip | — | Traits only |
| resin-gltf | skip | — | I/O |
| resin-gpu | done | 1 | NoiseConfig (GPU noise). Kernels are impl detail |
| resin-history | skip | — | Infrastructure |
| resin-image | done | 5+spectral | MapPixels, RemapUv, Convolve, Composite, Resize + spectral ops from resin-spectral |
| resin-jit | skip | — | Codegen |
| resin-spectral | done | 6 | FFT, IFFT, FFT2D, IFFT2D, DCT, IDCT (shared by audio/image) |
| resin-lsystem | done | 3 | LSystem.generate, Turtle2D, Turtle3D. Presets are config |
| resin-macros | skip | — | Proc macros |
| resin-mesh | done | 7 | Poke, SplitEdge, RipVertex, Transform, Linear/Loop/CC subdivision |
| resin-motion | done | 3 | Spring, Oscillate, Wiggle. Transform motions = Zip of components |
| resin-motion-fn | done | 3 | Same as resin-motion (Spring, Oscillate, Wiggle). Delay/Loop are time transforms |
| resin-noise | done | 5 | Perlin, Simplex(2D/3D), Value, Worley, Velvet |
| resin-op | skip | — | Infrastructure |
| resin-op-macros | skip | — | Proc macros |
| resin-particle | done | ~8 | CompositeEmitter pattern, 4 force primitives, Integrator missing |
| resin-physics | done | 5 | RigidBody, Force/Impulse, Integration, Collision shapes, Constraint |
| resin-pointcloud | done | 2 | Poisson, RemoveOutliers. Other ops need struct wrappers |
| resin-procgen | done | 8 | WfcSolver generic over AdjacencySource (co-equal: TileSet, WangTileSet) + 6 maze algs + river |
| resin-rd | done | 2 | Step (PDE integration), Laplacian (internal). Seeds are buffer writes |
| resin-rig | done | 8 | Skeleton, Pose, Skin, CCD, FABRIK, JiggleBone, Track. Locomotion decomposes |
| resin-scatter | done | 6 | Random, Grid, Sphere, PoissonDisk2D, Line, Circle |
| resin-serde | skip | — | Serialization |
| resin-space-colonization | done | 2 | SpaceColonizationStep, ComputeRadii (pipe model) |
| resin-spatial | done | 9 | Quadtree, Octree, KdTree2D/3D, BallTree2D/3D, Bvh, SpatialHash, Rtree |
| resin-spline | done | 4 | CubicBezier, CatmullRom, BSpline, Nurbs. Value containers, not ops |
| resin-spring | done | 2 | Verlet integrate, SpringConstraint solve. Damping is parametric |
| resin-surface | done | 1 | NurbsSurface::evaluate (De Boor). Value container like resin-spline |
| resin-transform | skip | — | Traits only |
| resin-vector | done | ~8 | Line/Curve segments, polygon algorithms. Heavy flattening |
| resin-voxel | done | 8 | FillSphere/Box, Dilate, Erode, VoxelsToMesh, SDF conversion |

**Legend:** `-` = not started, `partial` = in progress, `done` = complete, `skip` = not applicable

---

## Findings by Crate

### resin-image (done)

**True Primitives (Spatial, 5):**
1. `MapPixels { expr: ColorExpr }` - per-pixel color transform
2. `RemapUv { expr: UvExpr }` - UV coordinate remapping
3. `Convolve { kernel: Kernel }` - 2D spatial convolution
4. `Composite { blend_mode, opacity }` - image blending
5. `Resize { width, height }` - resampling

**Frequency Domain (via resin-spectral, 5):**
6. `Fft2d` - 2D FFT to frequency domain
7. `Ifft2d` - 2D IFFT back to spatial
8. `FftShift` - shift DC to center
9. `Dct2d { block_size }` - 2D DCT (JPEG-style)
10. `Idct2d { block_size }` - inverse DCT

**Integer/Bit-Level (4):**
11. `ToInt { range }` - float to integer conversion
12. `FromInt { range }` - integer to float conversion
13. `ExtractBitPlane { channel, bit }` - extract single bit as image
14. `SetBitPlane { channel, bit }` - set single bit from image

**Ops-as-Values:** ✅ All spatial primitives now have struct forms (Convolve, Resize, Composite, RemapUv, MapPixels). Functions are sugar that delegate to op structs.

**Decompositions Found:**

| Op | Decomposes To |
|----------|---------------|
| Blur | `Convolve { kernel: Gaussian }` in loop |
| Sharpen | `Convolve { kernel: Sharpen }` |
| Emboss | `Convolve { kernel: Emboss } + MapPixels { expr: +0.5 }` |
| DetectEdges | `Convolve { SobelH } + Convolve { SobelV } + MapPixels { magnitude }` |
| Grayscale | `MapPixels { expr: grayscale }` |
| Invert | `MapPixels { expr: invert }` |
| Posterize | `MapPixels { expr: posterize(levels) }` |
| Threshold | `MapPixels { expr: threshold(t) }` |
| LensDistortion | `RemapUv { expr: lens_expr }` |
| WaveDistortion | `RemapUv { expr: wave_expr }` |
| Glow | `Threshold + Blur + Colorize + Composite` |
| Bloom | `Threshold + PyramidBlur + Composite` |
| DropShadow | `ExtractChannel + Displace + Blur + Colorize + Composite` |
| ChromaticAberration | `SetChannel { R, RemapUv } + SetChannel { B, RemapUv }` |
| Downsample | `Resize { w/2, h/2 }` - trivial wrapper |
| Upsample | `Resize { w*2, h*2 }` - trivial wrapper |

**Issues Found:**
- ~~`AdjustHsl` duplicates ColorExpr colorspace logic~~ ✅ Fixed: Added `ColorExpr::AdjustHsl` and `ColorExpr::AdjustHsv` variants with convenience constructors (`hsl_adjust`, `hue_shift`, `saturate`, `lighten`, `hsv_adjust`). The `adjust_hsl()` function now delegates to `ColorExpr`.
- ~~`AdjustBrightnessContrast` duplicates per-pixel math~~ ✅ Fixed: Added `ColorExpr::AdjustBrightnessContrast` variant with `brightness_contrast()` convenience constructor. The `adjust_brightness_contrast()` function now delegates to `ColorExpr`.
- ~~`color_matrix` duplicates per-pixel math~~ ✅ Fixed: Added `ColorExpr::Matrix` variant with `matrix()` convenience constructor. The `color_matrix()` function now delegates to `ColorExpr`.
- ~~Channel operations duplicate sampling patterns~~ ✅ Partially fixed:
  - `extract_channel` and `swap_channels` now use `map_pixels` + `ColorExpr`
  - `set_channel` and `merge_channels` need multi-image input support (future: extend ColorExpr or use Composite ops)

---

### resin-audio (done)

**True Primitives (9):**
1. `DelayLine<T>` - circular buffer
2. `PhaseOsc` - phase accumulator with waveforms
3. `Biquad` - 2nd-order IIR filter
4. `EnvelopeFollower` - attack/release envelope
5. `Allpass1` - 1st-order allpass
6. `FFT/IFFT` - frequency domain
7. `AffineNode` - gain/offset
8. `Smoother` - 1-pole smoothing
9. `Mix::blend` - linear interpolation

**Decompositions Found:**

| Effect | Decomposes To |
|--------|---------------|
| Tremolo | `Input × PhaseOsc (LFO)` |
| Chorus | `DelayLine + PhaseOsc modulating delay + mix` |
| Flanger | Same as Chorus, shorter delay, higher feedback |
| Phaser | `Allpass1 cascade + PhaseOsc modulating coefficients` |
| Reverb | `Parallel CombFilters → Series AllpassFilters` |
| Distortion | `Gain × Waveshaper × Biquad(tone)` |
| Bitcrusher | `Sample-hold + Quantizer` |
| Compressor | `EnvelopeFollower + GainComputer + AffineNode` |
| Limiter | `DelayLine(lookahead) + EnvelopeFollower + AffineNode` |
| NoiseGate | `EnvelopeFollower + Threshold + AffineNode` |
| Karplus-Strong | `DelayLine + LowPass in feedback` |
| Ring Modulator | `Input × Oscillator` |
| Wah | `EnvelopeFollower → Biquad::bandpass` (already noted) |

**Status:** Excellent decomposition. Architecture is sound.

---

### resin-field (done)

**True Primitives (5):**
1. `Map<F, M>` - output transformation
2. `Zip<A, B>` - evaluate both fields at same input, yield tuple ✅
3. `Zip3<A, B, C>` - evaluate three fields at same input, yield triple ✅
4. `FnField<I, O, F>` - universal closure adapter
5. `Twist/Bend/Repeat` - irreducible domain transforms

**Ergonomic Helpers (Layer 2):**
- `add(a, b)`, `mul(a, b)`, `sub(a, b)`, `div(a, b)` - expand to `Zip + Map` ✅
- `lerp(a, b, t)`, `mix(a, b, t)` - expand to `Zip3 + Map` ✅
- `zip(a, b)`, `zip3(a, b, c)` - standalone functions ✅

**Removed (were Zip+Map):** ✅
- ~~`Add<A, B>`~~ — replaced by `add()` helper
- ~~`Mul<A, B>`~~ — replaced by `mul()` helper
- ~~`Mix<A, B, T>`~~ — replaced by `lerp()` / `mix()` helpers
- ~~`Field::add()`, `Field::mul()`, `Field::mix()`~~ trait methods — removed

**Remaining redundancies (not yet removed):**

| Current | Becomes |
|---------|---------|
| `SdfUnion` | `Zip.map(min)` |
| `SdfIntersection` | `Zip.map(max)` |
| `SdfSubtraction` | `Zip.map(\|(a,b)\| max(a, -b))` |
| `SdfSmoothUnion` | `Zip + smooth_min` |
| All SdfSmooth* | `Zip + smooth variant` |

**Convenience (keep for ergonomics):**
- `Scale`, `Translate`, `Rotate2D`, `Mirror` - specialized Map
- `Constant`, `Coordinates` - trivial FnField

---

### resin-mesh (done)

**True Primitives (7):**
1. `Poke` - add vertex at face center, create triangle fan
2. `SplitEdge` - duplicate vertices on edges
3. `RipVertex` - disconnect vertex from shared edges
4. `TransformVertex` - apply matrix to vertices
5. `LinearSubdivision` - 1→4 triangles via midpoints
6. `LoopSubdivision` - triangle-specific smoothing
7. `CatmullClarkSubdivision` - polygon-agnostic smoothing

**Decompositions Found:**

| Operation | Decomposes To |
|-----------|---------------|
| Extrude | Duplicate verts + Transform + Bridge edges |
| Inset | Poke + Scale + Bridge |
| Bevel (edges) | Split + Scale + Create caps |
| Bevel (vertices) | Split edges + Scale + Create edge faces |
| Merge | Index rewiring + Remove degenerates |
| Smooth | Iterate: Transform toward neighbor centroid |
| Slide | Find neighbors + Transform along direction |
| Bridge | Interpolate rings + Create quads |

---

### resin-vector (done)

**True Primitives (~8):**
1. `LineTo` - line segment
2. `QuadraticTo` - quadratic bezier
3. `CubicTo` - cubic bezier
4. `Close` - close path
5. `sutherland_hodgman` - polygon intersection
6. `weiler_atherton` - polygon union/subtract
7. `ramer_douglas_peucker` - simplification
8. `andrews_monotone_chain` - convex hull

**Decompositions Found:**

| Operation | Decomposes To |
|-----------|---------------|
| circle | 4× CubicTo (arc approximation) + Close |
| ellipse | circle + scale_xy |
| rect | 4× LineTo + Close |
| rounded_rect | LineTo (edges) + CubicTo (corners) + Close |
| star | polygon(interleaved vertices) |
| polygon | N× LineTo + Close |
| stroke_to_path | offset_path (both sides) + cap geometry |
| dash_path | trim_path (repeated) |
| path_union | flatten + polygon_union + rebuild |
| path_intersect | flatten + sutherland_hodgman + rebuild |
| path_subtract | flatten + weiler_atherton + rebuild |

**Key Insight:** Heavy flattening - all ops convert curves→polygons→algorithm→curves.

---

### resin-noise (done)

**True Primitives (5):**
1. `Perlin` (1D/2D/3D) - gradient noise
2. `Simplex` (2D/3D only) - simplex algorithm
3. `Value` (1D/2D/3D) - value noise
4. `Worley` (1D/2D/3D) - cellular/voronoi
5. `Velvet` (1D) - sparse impulse

**Decompositions Found:**

| Noise | Decomposes To |
|-------|---------------|
| Simplex1D | = Perlin1D (alias) |
| Fbm | Octave sum of any base noise |
| Pink | Value + octave sum (equal energy/octave) |
| Brown | Value at multiple low frequencies (1/f²) |
| Violet | Second differences of permutation (f²) |
| Grey | 0.5×Pink + 0.3×White + 0.2×Brown |

---

### resin-particle (done)

**True Primitives (~8):**

*Emitters (compositional pattern):*
- `PositionProvider` - where to emit
- `DirectionProvider` - which direction
- `SpeedProvider` - how fast
- `LifetimeProvider` - how long

*Forces (4):*
1. `Acceleration` - constant (covers Gravity, Wind)
2. `DistanceField` - inverse-power (covers Attractor)
3. `NoisePerturbation` - noise-based (covers Turbulence, CurlNoise)
4. `Damping` - exponential decay

**Decompositions Found:**

| Component | Decomposes To |
|-----------|---------------|
| PointEmitter | FixedPosition + ConeDirection + SpeedRange + LifetimeRange |
| SphereEmitter | SpherePosition + RadialDirection + SpeedRange + LifetimeRange |
| ConeEmitter | ConePosition + ConeBoundedDirection + SpeedRange + LifetimeRange |
| Gravity | `Acceleration { vector: (0,-9.81,0) }` |
| Wind | `Acceleration { target_velocity, strength }` |
| Vortex | `DistanceField + TangentTransform` |
| CurlNoise | `NoisePerturbation + CurlOperator` |

**Issues Found:**
- ~~Integrator hardcoded to Euler (should be trait)~~ ✅ Fixed: Added `Integrator` trait with `EulerIntegrator` and `SemiImplicitEulerIntegrator` implementations
- ~~All emitters duplicate attribute initialization~~ ✅ Fixed: Added composable provider traits (`PositionProvider`, `VelocityProvider`, `LifetimeProvider`, `AttributeProvider`) and `CompositeEmitter`

**Ops-as-Values:** ✅ Provider traits for composable emission: `FixedPosition`, `SpherePosition`, `BoxPosition`, `ConeVelocity`, `RadialVelocity`, `LifetimeRange`, `FixedAttributes`. `CompositeEmitter<P,V,L,A>` combines these.

---

### resin-physics (done)

**True Primitives (5):**
1. `RigidBody` - position, orientation, velocity, mass, inertia
2. `Force/Impulse` - impulse_at_point, apply_torque
3. `Integration` - verlet, quaternion rotation
4. `CollisionShapes` - Sphere, Box, Plane
5. `Constraint` - generic anchor-based solver

**Decompositions Found:**

| Component | Decomposes To |
|-----------|---------------|
| DistanceConstraint | 2× PointConstraint with distance error |
| HingeConstraint | PointConstraint + AxisAlignment + Limits |
| SpringConstraint | Hooke's law force via apply_force_at_point |
| Cloth | N× Particle + DistanceConstraints + Collision |
| SoftBody | N× Vertex + Tetrahedra + ElasticConstraints |

**Key Insight:** All constraints follow same pattern: compute_error → solve_jacobian → apply_correction

**Ops-as-Values:** ✅ Added `ConstraintSolver` trait unifying all constraint types with:
- `compute_error(&self, bodies) -> Option<ConstraintError>` - compute position/angular error
- `apply_correction(&self, bodies, stiffness)` - apply corrections to bodies
- `stiffness(&self) -> f32` - get constraint stiffness

Implemented for: `DistanceConstraint`, `PointConstraint`, `HingeConstraint`, `Constraint` enum. Enables custom constraints and explicit solver pattern.

---

### resin-curve (done)

**True Primitives (4):**
1. `Line<V>` - linear segment (trivial lerp)
2. `QuadBezier<V>` - quadratic Bézier (3 control points)
3. `CubicBezier<V>` - cubic Bézier (4 control points)
4. `Arc` - 2D elliptical arc (angular parameterization)

**Composition Containers:**
- `Path<C>` - connected sequence of curves
- `ArcLengthPath<C>` - arc-length parameterized wrapper
- `Segment2D` / `Segment3D` - enum dispatchers

**Decompositions Found:**

| Operation | Decomposes To |
|-----------|---------------|
| `Path::position_at(t)` | Segment indexing + `.position_at()` |
| `Path::length()` | Sum of segment lengths |
| `ArcLengthPath` | Path + cumulative length cache + binary search |
| `Curve::flatten(tol)` | Recursive bisection via `position_at()` |
| `Curve::length()` | Gaussian quadrature over `tangent_at()` |

**Key Insight:** All operations decompose to repeated queries of `position_at()` and `tangent_at()`.

---

### resin-easing (done)

**True Primitives: 0** - All easing functions are dew expressions.

| Easing | Dew Expression |
|--------|----------------|
| `linear` | `t` |
| `quad_in` | `t * t` |
| `cubic_in` | `t * t * t` |
| `quart_in` | `pow(t, 4)` |
| `sine_in` | `1 - cos(t * PI / 2)` |
| `expo_in` | `pow(2, 10 * (t - 1))` |
| `circ_in` | `1 - sqrt(1 - t * t)` |
| `back_in` | `t * t * (2.70158 * t - 1.70158)` |
| `elastic_in` | `pow(2, 10*(t-1)) * sin(...)` |
| `smoothstep` | `t * t * (3 - 2 * t)` |
| `smootherstep` | `t * t * t * (t * (6*t - 15) + 10)` |

**Decompositions:**
- All `*_out(t)` = `1 - *_in(1-t)`
- All `*_in_out(t)` = piecewise `*_in` scaled

**Implementation:** ✅ Easing expression builders now live in `unshape-expr-field::easing` module. These return `FieldExpr` AST nodes (e.g., `quad_in(t)` returns `Mul(t, t)`) enabling constant folding by the optimizer.

**Three-Layer Architecture:**
1. **Primitives**: dew's `+`, `*`, `sin`, `cos`, `pow`, `sqrt`
2. **Ergonomics**: `quad_in(t)` constructs the expression `t * t`
3. **Optimizer**: Pattern-matches `t*t*(3-2*t)` → GPU `smoothstep` intrinsic

**Key Insight:** resin-easing provides ergonomic presets, not primitives. The optimizer handles efficient codegen.

---

### resin-color (done)

**True Primitives (3):**
1. `srgb_to_linear` / `linear_to_srgb` - gamma curve conversion
2. `LinearRgb::to_hsl()` / `to_hsv()` - coordinate system transform with hue calculation
3. `Hsl::lerp()` / `Hsv::lerp()` - interpolation with hue wrapping (short path)

**Decompositions Found:**

| Blend Mode | Decomposes To |
|------------|---------------|
| Multiply | `a * b` |
| Screen | `1 - (1-a)*(1-b)` |
| Overlay | `if a<0.5 then 2*a*b else 1-2*(1-a)*(1-b)` |
| Darken | `min(a, b)` |
| Lighten | `max(a, b)` |
| Add | `min(a+b, 1)` |
| Subtract | `max(a-b, 0)` |
| Difference | `abs(a-b)` |
| `clamp()` | Per-component bounding |
| `premultiply()` | Per-channel alpha scaling |
| `blend_with_alpha()` | Lerp + alpha compositing |

**Key Insight:** Color space conversions are irreducible (unique math), but blend modes decompose to component-wise operations.

---

### resin-spatial (done)

**True Primitives (9 structures):**
1. `Aabb2` / `Aabb3` - bounding box geometry
2. `Quadtree` - 2D hierarchical partitioning (4-way)
3. `Octree` - 3D hierarchical partitioning (8-way)
4. `KdTree2D` / `KdTree3D` - axis-aligned binary partitioning
5. `BallTree2D` / `BallTree3D` - metric ball partitioning
6. `Bvh` - SAH-based ray hierarchy
7. `SpatialHash` - uniform grid hashing
8. `Rtree` - dynamic rectangle tree

**Decompositions Found:**

| Operation | Decomposes To |
|-----------|---------------|
| `nearest(point)` | `k_nearest(point, 1)` |
| `query_radius(center, r)` | `query_region(sphere_aabb) + distance_filter` |

**Key Insight:** These are data structures, not operations. Each uses fundamentally different algorithms (tree vs grid vs hash).

---

### resin-spline (done)

**True Primitives (4 algorithms):**
1. `CubicBezier::evaluate()` - De Casteljau (4-point basis)
2. `CatmullRom::evaluate()` - centripetal parameterization with tension
3. `BSpline::evaluate()` - De Boor algorithm (arbitrary degree/knots)
4. `Nurbs::evaluate()` - Rational De Boor (homogeneous coordinates)

**Compositional:**

| Item | Decomposes To |
|------|---------------|
| `BezierSpline` | N× CubicBezier segments |
| `nurbs_circle` | Nurbs::new(9 weighted points, degree=2) |
| `nurbs_arc` | Nurbs::new(N weighted points by angle) |
| `nurbs_ellipse` | Nurbs::new(9 scaled weighted points) |
| `smooth_through_points` | CatmullRom + sample |

**Note:** These are value containers (spline data), not operations. Not part of ops-as-values pattern.

---

### resin-automata (done)

**True Primitives (12):**

*Step operations (one per CA type):*
1. `StepElementaryCA { steps }` - 1D rule application (3-cell → rule bit lookup)
2. `StepCellularAutomaton2D { steps }` - 2D neighbor counting + birth/survival rules
3. `LargerThanLife::step()` - 2D extended-range CA with range-based birth/survival thresholds
4. `CellularAutomaton3D::step()` - 3D neighbor counting + birth/survival rules
5. `SmoothLife::step(dt)` - continuous-state 2D CA with sigmoid transitions and disk/ring integrals
6. `LangtonsAnt::step()` - 2D Turing machine (ant on grid, state → turn rule)
7. `Turmite::step()` - generalized multi-state ant with (grid_state, ant_state) → (new_grid, turn, new_ant) transitions

*HashLife:*
8. `HashLife::advance(node, step_log2)` - memoized recursive quadtree advance (Gosper's algorithm)
9. `HashLife::advance_level2(node)` - base case: manual GoL on 4×4 grid → 2×2 center
10. `HashLife::advance_full(node)` - full-speed: 9 sub-squares → recurse → 4 intermediates → recurse
11. `HashLife::advance_slow(node, step_log2)` - parameterized: 9 sub-squares → recurse → extract centers

*Neighborhood abstraction:*
12. `Neighborhood2D` / `Neighborhood3D` traits - pluggable neighborhood patterns (Moore, VonNeumann, Hexagonal, ExtendedMoore, custom)

**Decompositions Found:**

| Operation | Decomposes To |
|-----------|---------------|
| `GeneratePattern` | Loop: `StepElementaryCA` × N, collect states |
| `ElementaryCAConfig` | `ElementaryCA::new()` + `randomize()` or `set_center()` |
| `CellularAutomaton2DConfig` | `CellularAutomaton2D::new()` + `randomize()` |
| `HashLife::step()` | `step_pow2(0)` |
| `HashLife::steps(n)` | Binary decomposition: `step_pow2` per set bit in n |
| `HashLife::from_ca2d()` | Tree construction from `CellularAutomaton2D` grid |
| Rule presets (LIFE, etc.) | Data constants, not operations |
| LtlRules presets (BUGS, etc.) | Data constants for Larger than Life |
| Ant/turmite presets | Rule string constants |
| SmoothLifeConfig presets | Parameter configurations (standard, fluid, slow) |

**Key Insights:**
- Step operations are truly primitive per CA type (each has fundamentally different state/transition logic).
- HashLife decomposes step into memoized recursive `advance` — three cases (base, full-speed, parameterized) are irreducible.
- Neighborhoods are pluggable via traits, not hardcoded — `CellularAutomaton2D` and `CellularAutomaton3D` accept any `Neighborhood2D`/`Neighborhood3D`.
- Binary step decomposition in `steps(n)` maximizes HashLife memoization hits.

---

### resin-spring (done)

**True Primitives (2):**
1. `Verlet::integrate` - position update from history (implicit velocity)
2. `SpringConstraint::solve` - Hooke's law with per-spring damping

**Decompositions Found:**

| Operation | Decomposes To |
|-----------|---------------|
| `create_rope` | SpawnLine of particles + sequential spring connections |
| `create_cloth` | SpawnGrid + structural springs (H, V) + shear springs (diagonals) |
| `create_soft_sphere` | GenerateSphereGeometry + distance-based connectivity |
| `pin_particle` | Set `mass = 0` (infinite mass) |
| Damping behavior | Parametric: `global_damping` + `spring_damping` compose to under/over/critical |

**Key Insight:** No separate damped/overdamped types needed - damping emerges from parameter combinations.

---

### resin-voxel (done)

**True Primitives (8):**
1. `FillSphere` - distance-based voxel selection
2. `FillBox` - axis-aligned bounds iteration
3. `Dilate` - 6-connected morphological expansion
4. `Erode` - 6-connected morphological shrinking
5. `VoxelsToMesh` - binary grid → mesh with face culling
6. `SparseVoxelsToMesh` - sparse variant
7. `sdf_to_voxels` - field sampling + threshold
8. `sdf_to_density` - field sampling (raw values)

**Data Structures:** `VoxelGrid<T>` (dense), `SparseVoxels<T>` (hashmap-based)

**Key Insight:** Morphological ops (dilate/erode) are true primitives. Fill operations have distinct geometric algorithms.

---

### resin-lsystem (done)

**True Primitives (3):**
1. `LSystem.generate(iterations)` - parallel string rewriting with stochastic rule selection
2. `Turtle2D.apply(string)` - 2D turtle state machine with stack
3. `Turtle3D.apply(string)` - 3D turtle with heading/left/up vectors + Rodrigues rotation

**Decompositions Found:**

| Operation | Decomposes To |
|-----------|---------------|
| All presets (koch_curve, etc.) | `LSystem::new().with_rule()` configurations |
| `segments_to_paths_2d()` | Group connected segments into paths |

**Key Insight:** String rewriting + turtle interpretation are irreducible. All presets are just parameter configurations.

---

### resin-rd (done)

**True Primitives (2):**
1. `Step { count }` - Gray-Scott PDE numerical integration
2. `Laplacian` (internal) - 5-point stencil finite difference

**Decompositions Found:**

| Operation | Decomposes To |
|-----------|---------------|
| `SeedCircle` / `SeedRect` | Buffer writes (no simulation) |
| `SeedRandom` | N × SeedCircle |
| `SetFeed` / `SetKill` | State mutation only |
| `ApplyPreset` | `SetFeed + SetKill` |
| `Clear` | `fill(u=1.0, v=0.0)` |

**Key Insight:** Only Step performs actual simulation. All seeding/parameter ops are state manipulation.

---

### resin-motion / resin-motion-fn (done)

**True Primitives (3):**
1. `Spring<T>` - critically/underdamped spring ODE solver via `spring_value()`
2. `Oscillate<T>` - sine wave parameterization (t → phase → sin)
3. `Wiggle` / `Wiggle2D` - noise-based motion via Perlin sampling

**Time Transformations (not primitives):**
- `Delay<M>` → `motion.at(t - delay)`
- `TimeScale<M>` → `motion.at(t * scale)`
- `Loop<M>` → `motion.at(t % duration)`
- `PingPong<M>` → bidirectional looping

**Decompositions Found:**

| Operation | Decomposes To |
|-----------|---------------|
| `OscillateTransform2D` | `Oscillate<f32>` per component + reassemble |
| `WiggleTransform2D` | `Wiggle` per component + reassemble |
| `Eased<T>` | `Lerp<T>` + easing expression (dew) |
| `Constant<T>` | Trivial (return clone) |
| `Lerp<T>` | Standard linear interpolation |

**Key Insight:** Transform motion types decompose to Zip of per-component scalar motions. Time wrappers are orthogonal transformations.

---

### resin-pointcloud (done)

**True Primitives (2 Op structs):**
1. `Poisson { min_distance, max_attempts }` - Bridson's algorithm with spatial hashing
2. `RemoveOutliers { k, std_ratio }` - statistical distance-based filtering

**Should-Be-Ops (currently free functions):**
- `UniformSampling { count }` - area-weighted surface sampling
- `SdfSampling { count, threshold }` - rejection sampling on implicit surface
- `EstimateNormals { k }` - local PCA normal inference
- `VoxelDownsample { voxel_size }` - spatial quantization + averaging
- `CropBounds { min, max }` - AABB filtering

**Key Insight:** Only 2 ops follow ops-as-values pattern. 5 free functions need struct wrappers for serialization.

---

### resin-scatter (done)

**True Primitives (6):**
1. `ScatterRandom` - uniform random distribution in bounds
2. `ScatterGrid` - regular 3D lattice
3. `ScatterSphere` - Fibonacci sphere parameterization
4. `ScatterPoissonDisk2D` - Bridson's 2D blue noise
5. `ScatterLine` - linear interpolation along segment
6. `ScatterCircle` - polar angle enumeration

**Decompositions Found:**

| Operation | Decomposes To |
|-----------|---------------|
| `scatter_grid_2d` | `scatter_grid([rx, ry, 1])` with z fixed |
| `randomize_scale` | Post-process: per-instance scale multiplication |
| `randomize_rotation` | Post-process: random quaternion assignment |
| `jitter_positions` | Post-process: per-axis offset perturbation |
| Stagger system | Animation timing (belongs in resin-motion) |

**Key Insight:** 6 distinct distribution algorithms. Post-processing helpers (randomize_*, jitter_*) are transforms, not scatter ops.

---

### resin-procgen (done)

**True Primitives (8):**
1. `WfcSolver<A: AdjacencySource>` - generic Wave Function Collapse with entropy-driven constraint propagation
2. `RecursiveBacktracker` - DFS maze (long winding passages)
3. `Prim` - randomized Prim's spanning tree maze
4. `Kruskal` - union-find spanning tree maze
5. `Eller` - row-by-row efficient maze generation
6. `BinaryTree` - simple diagonal-bias maze
7. `Sidewinder` - horizontal-bias maze variant
8. `RiverNetwork::generate_river()` - procedural river with meandering

**Co-Equal Primitives (AdjacencySource trait):**

`WfcSolver` is generic over the `AdjacencySource` trait, which has two co-equal implementations:

| Type | Storage | Adjacency Lookup | Best For |
|------|---------|-------------------|----------|
| `TileSet` | O(R) explicit rules in HashMap | O(1) per direction | Custom/irregular adjacency |
| `WangTileSet` | O(N) tiles + O(C⁴) edge-color index | O(C³) where C = colors | Regular tiling patterns |

Converting 1000 Wang tiles to explicit rules = ~1,000,000 entries. Neither subsumes the other efficiently. Both implement `AdjacencySource`, and `WfcSolver` works with either without conversion. `ValidNeighbors` enum avoids boxing (returns `HashSet::Iter` or `slice::Iter` depending on backing type).

See `docs/design/general-internal-constrained-api.md` § "Exception: Co-Equal Primitives".

**Op Structs (Layer 2):**
- `GenerateMaze { width, height, algorithm, add_entrance, add_exit }`
- `GenerateRiver { source, sink, config }`
- `GenerateRoadNetworkGrid { bounds_min, bounds_max, spacing }`
- `GenerateRoadNetworkHierarchical { bounds_min, bounds_max, density }`

**Decompositions Found:**

| Operation | Decomposes To |
|-----------|---------------|
| `solve_wang_tiling()` | `WfcSolver::new(WangTileSet)` + `run()` (no O(N²) conversion) |
| `WangTileSet::to_tileset()` | Explicit rule expansion (available but avoided in solver) |
| `NamedTileSet` | String-keyed wrapper around `TileSet` |
| Preset tilesets (`platformer_tileset`, etc.) | `NamedTileSet` configurations |
| Wang presets (`two_color_corners`, etc.) | `WangTileSet` configurations |

**Key Insights:**
- WFC and maze algorithms are irreducible — each uses fundamentally different traversal patterns.
- `AdjacencySource` is the unifying trait for tile adjacency, not a general/constrained pair. Both `TileSet` and `WangTileSet` are first-class primitives with different performance tradeoffs.
- `WfcSolver` being generic over `AdjacencySource` eliminates the previous O(N²) conversion path for Wang tile solving.

---

### resin-rig (done)

**True Primitives (8):**
1. `Skeleton` - hierarchical bone structure with parent-child relationships
2. `Pose` - per-bone transforms relative to rest pose
3. `Transform3D` - TRS composition with lerp via quaternion slerp
4. `Skin` - linear blend skinning (weighted bone transforms)
5. `SolveCcd` - Cyclic Coordinate Descent IK ✅
6. `SolveFabrik` - Forward And Backward Reaching IK ✅
7. `JiggleBone` - spring-damper physics for soft body
8. `Track<T>` - keyframe storage with interpolation

**Ops-as-Values:** ✅ IK solvers wrapped as op structs (`SolveCcd`, `SolveFabrik`) with serializable config.

**Decompositions Found:**

| Operation | Decomposes To |
|-----------|---------------|
| `ProceduralWalk` | Gait config + IK per leg + parametric body motion |
| `MotionMatcher` | Database search + linear blend |
| `AnimationStack` | Layered Track sampling + Lerp |
| `PathConstraint` | Path sampling + transform composition |
| `BlendNode` | Tree container + recursive sampling |

**Key Insight:** IK solvers (CCD, FABRIK) and physics (JiggleBone) are irreducible. Locomotion and animation blending decompose to primitives.

---

### resin-fluid (done)

**True Primitives (4):**
1. `Stable Fluids Step` - Eulerian grid PDE (advect + diffuse + project)
2. `SPH Particle Integration` - Smoothed Particle Hydrodynamics (density + forces + integrate)
3. `Smoke Buoyancy Force` - temperature-driven velocity (`apply_buoyancy`)
4. `Dissipation` - exponential energy decay

**Configuration Structs (not primitives):**
- `Fluid` - grid simulation parameters
- `Sph` / `SphParams3D` - particle simulation parameters
- `Smoke` - smoke simulation parameters

**Decompositions Found:**

| Operation | Decomposes To |
|-----------|---------------|
| `add_density` / `add_velocity` | Buffer write + field access |
| `add_particle` / `add_block` | Vec::push + spatial iteration |
| `sample_density` / `sample_velocity` | Bilinear interpolation |
| `clear` | `fill(0.0)` on all buffers |

**Key Insight:** Fluid methods (Stable Fluids vs SPH vs Smoke) are fundamentally different algorithms. Cannot decompose further.

---

### resin-crossdomain (done)

**True Primitives (2):**
1. `ImageToAudio` - per-row frequency bands → additive synthesis
2. `AudioToImage` - STFT → spectrogram visualization

**Helper Functions (not primitives):**
- `audio_to_image_colored()` - AudioToImage + colorization
- `field_to_audio()` / `field_to_audio_stereo()` - field sampling → audio
- `field_to_image()` / `field_rgba_to_image()` - field sampling → image
- `field_to_vertices_2d/3d()` - field sampling → vertex positions
- `field_to_displacement()` - field → mesh displacement

**View Types (reinterpretation, not ops):**
- `AudioView`, `PixelView`, `Vertices2DView`, `Vertices3DView`

**Key Insight:** Two core domain-crossing primitives (image↔audio via spectral). Helpers are field sampling + domain conversion.

---

### resin-space-colonization (done)

**True Primitives (2):**
1. `SpaceColonizationStep` - core iteration (influence finding → direction averaging → node creation)
2. `ComputeRadii` - pipe model radius calculation (leaf → root propagation)

**Decompositions Found:**

| Operation | Decomposes To |
|-----------|---------------|
| `generate_tree()` | `SpaceColonizationParams` + sphere attractions + gravity tropism |
| `generate_lightning()` | `SpaceColonizationParams` + cylinder attractions + directional tropism |
| `add_attraction_points_*` | Geometric sampling utilities |
| Tropism | Directional bias (could be field modifier) |

**Key Insight:** Step + radii computation are irreducible. Tree/lightning presets are parameter configurations.

---

### resin-surface (done)

**True Primitives (1):**
1. `NurbsSurface::evaluate(u, v)` - De Boor algorithm for tensor product B-spline + rational weighting

**Decompositions Found:**

| Operation | Decomposes To |
|-----------|---------------|
| `derivative_u(u, v)` | `evaluate(u+ε, v) - evaluate(u-ε, v)` / 2ε |
| `derivative_v(u, v)` | `evaluate(u, v+ε) - evaluate(u, v-ε)` / 2ε |
| `normal(u, v)` | `derivative_u.cross(derivative_v).normalize()` |
| `tessellate(div_u, div_v)` | Grid loop: `evaluate()` + quad triangulation |

**Constructors (presets):**
- `nurbs_sphere`, `nurbs_cylinder`, `nurbs_torus`, `nurbs_cone`, `nurbs_bilinear_patch`

**Key Insight:** Value container like resin-spline. Evaluation is the only primitive; derivatives and tessellation decompose to repeated evaluation.

---

### resin-gpu (done)

**True Primitives (1):**
1. `NoiseConfig` - GPU-accelerated noise generation parameters

**Implementation Details (not primitives):**
- `NoiseTextureKernel` / `NoiseTextureNode` - GPU compute implementations
- `ParameterizedNoiseNode` - parameterized noise via GPU
- `MapPixelsKernel` / `RemapUvKernel` (with `image-expr` feature)

**Infrastructure:**
- `GpuContext`, `GpuTexture` - GPU resource management

**Key Insight:** GPU crate provides accelerated implementations of existing primitives (noise, image ops). NoiseConfig is the only domain-specific op struct.

---

## Summary: Minimal Primitive Sets

### Image (14)
**Spatial:** `MapPixels`, `RemapUv`, `Convolve`, `Composite`, `Resize`
**Spectral:** `Fft2d`, `Ifft2d`, `FftShift`, `Dct2d`, `Idct2d` (via resin-spectral)
**Integer:** `ToInt`, `FromInt`, `ExtractBitPlane`, `SetBitPlane`

### Spectral (6) - shared crate
`fft`, `ifft`, `fft2d`, `ifft2d`, `dct2d`, `idct2d` + window functions

### Audio (8)
`DelayLine`, `PhaseOsc`, `Biquad`, `EnvelopeFollower`, `Allpass1`, `AffineNode`, `Smoother` + spectral ops via resin-spectral
*(Mix removed - it's `Zip3 + lerp expression`, provided as ergonomic helper)*

### Field (5)
`Map`, `Zip`, `Zip3`, `FnField`, `{Twist, Bend, Repeat}`
*(Add/Mul/Lerp/Mix removed - all are Zip/Zip3 + expression, provided as ergonomic helpers)* ✅ Implemented: `Add`, `Mul`, `Mix` structs deprecated; `add()`, `mul()`, `sub()`, `div()`, `mix()`, `lerp()` helper functions added

### Mesh (7)
`Poke`, `SplitEdge`, `RipVertex`, `TransformVertex`, `Linear/Loop/CatmullClark Subdivision`

### Vector (8)
`LineTo`, `QuadraticTo`, `CubicTo`, `Close`, `sutherland_hodgman`, `weiler_atherton`, `rdp`, `convex_hull`

### Noise (5)
`Perlin`, `Simplex`, `Value`, `Worley`, `Velvet`

### Particle (8)
`{Position,Direction,Speed,Lifetime}Provider`, `Acceleration`, `DistanceField`, `NoisePerturbation`, `Damping`

### Physics (5)
`RigidBody`, `Force/Impulse`, `Integration`, `CollisionShapes`, `Constraint`

### Curve (4)
`Line`, `QuadBezier`, `CubicBezier`, `Arc`

### Easing (0)
All easing = dew expressions. Ergonomic presets only, optimizer pattern-matches for GPU intrinsics.

### Color (3)
Gamma conversion, HSL/HSV coordinate transform, hue-wrapped lerp

### Spatial (9)
`Aabb2/3`, `Quadtree`, `Octree`, `KdTree2D/3D`, `BallTree2D/3D`, `Bvh`, `SpatialHash`, `Rtree`

### Spline (4)
`CubicBezier`, `CatmullRom`, `BSpline`, `Nurbs` (evaluation algorithms)

### Automata (2)
`StepElementaryCA`, `StepCellularAutomaton2D`

### Spring (2)
`Verlet::integrate`, `SpringConstraint::solve`

### Voxel (8)
`FillSphere`, `FillBox`, `Dilate`, `Erode`, `VoxelsToMesh`, `SparseVoxelsToMesh`, `sdf_to_voxels`, `sdf_to_density`

### L-System (3)
`LSystem.generate`, `Turtle2D`, `Turtle3D`

### Reaction-Diffusion (2)
`Step` (PDE integration), `Laplacian` (finite difference)

### Motion (3)
`Spring`, `Oscillate`, `Wiggle` (time wrappers like Delay/Loop are transforms, not primitives)

### Pointcloud (2)
`Poisson`, `RemoveOutliers`

### Scatter (6)
`ScatterRandom`, `ScatterGrid`, `ScatterSphere`, `ScatterPoissonDisk2D`, `ScatterLine`, `ScatterCircle`

### Procgen (8)
`WfcSolver`, `RecursiveBacktracker`, `Prim`, `Kruskal`, `Eller`, `BinaryTree`, `Sidewinder`, river generation

### Rig (8)
`Skeleton`, `Pose`, `Transform3D`, `Skin`, `solve_ccd`, `solve_fabrik`, `JiggleBone`, `Track`

### Fluid (4)
Stable Fluids (advect/diffuse/project), SPH, Smoke buoyancy, Dissipation

### Crossdomain (2)
`ImageToAudio`, `AudioToImage`

### Space Colonization (2)
`SpaceColonizationStep`, `ComputeRadii`

### Surface (1)
`NurbsSurface::evaluate` (value container like Spline)

### GPU (1)
`NoiseConfig` (GPU acceleration of existing primitives)

---

## Non-Primitives (Intentionally Excluded)

These operations are **not** included as primitives because they're compositions of existing primitives. Users should build them from the underlying ops rather than having them as special cases.

### Image

| Excluded Op | Why | Compose From |
|-------------|-----|--------------|
| LsbEmbed | Byte-level loop over ExtractBitPlane/SetBitPlane | Loop + SetBitPlane per bit |
| Glow | Multi-step composite effect | Threshold → Blur → Colorize → Composite |
| Bloom | Multi-step composite effect | Threshold → PyramidBlur → Composite |
| DropShadow | Multi-step composite effect | ExtractChannel → Displace → Blur → Colorize → Composite |
| Downsample | Trivial wrapper | `Resize { w/2, h/2 }` |
| Upsample | Trivial wrapper | `Resize { w*2, h*2 }` |
| ApplyMask | Just lerp with black/transparent | `Lerp(original, black, mask)` |
| MaskUnion | Just field math | `max(a, b)` or `Zip.map(max)` |
| MaskIntersect | Just field math | `min(a, b)` or `Zip.map(min)` |
| MaskSubtract | Just field math | `a * (1 - b)` |
| MaskInvert | Just field math | `1 - mask` |
| FrequencyMask | FFT composition | `Fft2d → Zip(mask) → Ifft2d` |
| HighPassFreq | FFT + radial mask | `Fft2d → Zip(radial_mask(cutoff, 0→1)) → Ifft2d` |
| LowPassFreq | FFT + radial mask | `Fft2d → Zip(radial_mask(cutoff, 1→0)) → Ifft2d` |
| BandPassFreq | FFT + ring mask | `Fft2d → Zip(ring_mask(lo, hi)) → Ifft2d` |
| SdfToImage | SDF is already a Field | Sample SDF directly as ImageField via Field trait |
| GetBitPlane | Trivial pattern | `(channel >> bit) & 1` via dew expression |

### Audio

| Excluded Op | Why | Compose From |
|-------------|-----|--------------|
| Tremolo | Single multiplication | `Input × PhaseOsc (LFO)` |
| RingModulator | Single multiplication | `Input × Oscillator` |
| Gravity | Constant vector | `Acceleration { (0, -9.81, 0) }` |
| Wind | Constant with target | `Acceleration { target_velocity }` |

### Field

| Excluded Op | Why | Compose From |
|-------------|-----|--------------|
| Add | Binary Zip + Map | `Zip<A, B>.map(\|(a,b)\| a + b)` |
| Mul | Binary Zip + Map | `Zip<A, B>.map(\|(a,b)\| a * b)` |
| Lerp / Mix | Ternary Zip + Map | `Zip3<A, B, T>.map(\|(a,b,t)\| a*(1-t) + b*t)` |
| Blend | Just lerp | `lerp(a, b, mask)` |
| SdfUnion | Binary Zip + Map | `Zip.map(min)` |
| SdfIntersection | Binary Zip + Map | `Zip.map(max)` |
| SdfSubtraction | Binary Zip + Map | `Zip.map(\|(a,b)\| max(a, -b))` |
| SdfSmoothUnion | Binary Zip + Map | `Zip + smooth_min` |
| SdfToMask | Threshold a field | `map(\|d\| if d < 0.0 { 1.0 } else { 0.0 })` |
| SdfToImage | SDF implements Field | Just sample the SDF—no conversion op needed |

**Note on SDFs:** An SDF *is* a Field<Vec2, f32> or Field<Vec3, f32>. There's no need for a dedicated "SdfToField" or "SdfToImage" op—just use the SDF directly where a field is expected, or sample it into an ImageField.

### Mesh

| Excluded Op | Why | Compose From |
|-------------|-----|--------------|
| Extrude | Multi-step mesh edit | Duplicate verts + Transform + Bridge edges |
| Inset | Multi-step mesh edit | Poke + Scale + Bridge |
| Bevel | Multi-step mesh edit | Split + Scale + Create caps |
| Smooth | Iterative neighbor average | Loop: Transform toward centroid |
| Merge | Index manipulation | Index rewiring + Remove degenerates |

### Particle

| Excluded Op | Why | Compose From |
|-------------|-----|--------------|
| Gravity | Constant acceleration | `Acceleration { (0, -9.81, 0) }` |
| Wind | Constant with target | `Acceleration { target_velocity }` |
| Vortex | Tangent of distance field | `DistanceField + TangentTransform` |
| CurlNoise | Curl of noise field | `NoisePerturbation + CurlOperator` |

### Vector

| Excluded Op | Why | Compose From |
|-------------|-----|--------------|
| circle | Bezier approximation | 4× CubicTo + Close |
| ellipse | Scaled circle | circle + scale_xy |
| rect | Line segments | 4× LineTo + Close |
| rounded_rect | Lines + arcs | LineTo + CubicTo + Close |
| star | Interleaved vertices | polygon with computed points |

### General Principle

An operation is **not a primitive** if:
1. It can be expressed as a composition of other primitives (no unique algorithm)
2. It's a trivial wrapper with no added behavior (e.g., downsample = resize/2)
3. It's a specific use case of a general primitive (e.g., gravity = constant acceleration)
4. It duplicates what expressions/fields already provide (e.g., SDF already *is* a field)
5. It can be expressed as a dew expression pattern (e.g., `(channel >> bit) & 1` for bit extraction)

**No "blessed compositions"** - if it can be reduced, it's not a primitive. Period. Ergonomics are handled separately (see below).

---

## Three-Layer Architecture

Primitives, ergonomics, and optimization are **separate concerns**:

### Layer 1: True Primitives
Irreducible operations with unique algorithms:
- `Zip<A, B>` - evaluate two fields at same input
- `Zip3<A, B, C>` - evaluate three fields at same input
- `Map<F, Expr>` - transform output
- `Convolve`, `FFT`, `Resize`, etc.

### Layer 2: Ergonomic Helpers
Functions that return compositions of primitives:
```rust
fn lerp<A, B, T>(a: A, b: B, t: T) -> Zip3<A, B, T, LerpExpr> {
    Zip3::new(a, b, t).map(|(a, b, t)| a * (1.0 - t) + b * t)
}

fn add<A, B>(a: A, b: B) -> Zip<A, B, AddExpr> {
    Zip::new(a, b).map(|(a, b)| a + b)
}

fn blend<A, B, M>(a: A, b: B, mask: M) -> impl Field {
    lerp(a, b, mask)
}
```

Users write `lerp(a, b, t)`. They don't care it's a composition internally.

### Layer 3: Pattern-Matching Optimizer
Recognizes common patterns and emits optimal code:

| Pattern | Expression | Optimized To |
|---------|------------|--------------|
| Lerp | `a * (1-t) + b * t` | GPU `mix` instruction |
| Bit extraction | `(channel >> N) & 1` | Fused bit extract |
| Bit setting | `(channel & ~(1 << N)) \| (bit << N)` | Fused bit set |
| Threshold | `if x > t { 1.0 } else { 0.0 }` | GPU `step` instruction |
| Clamp | `max(min(x, hi), lo)` | GPU `clamp` instruction |

**Result:** Minimal primitive set + ergonomic API + optimal codegen. No compromises.

---

## Action Items

### High Priority
- [x] **Refactor resin-image to ops-as-values** - Convolve, Resize, Composite, RemapUv, MapPixels now structs
- [x] Add `Zip<A, B>` and `Zip3<A, B, C>` combinators to resin-field
- [x] Expose Integrator trait in resin-particle
- [x] Fix code duplication in resin-image colorspace ops (AdjustHsl, brightness/contrast, color_matrix, channel ops now use ColorExpr)

### Medium Priority
- [x] Refactor emitters to CompositeEmitter pattern
- [x] Unify constraint solvers to common pattern
- [x] Remove trivial wrappers - upsample deprecated (same as Resize), downsample kept (distinct box filter algorithm)

### Low Priority (Documentation)
- [x] Document which ops are compositions vs primitives (see `docs/design/primitive-decomposition.md`)
- [x] Add pattern-matching optimizer for common compositions
  - **Expression-level optimization**: dew handles AST optimization (constant folding, identity elimination, algebraic simplification). ColorExpr converts to dew AST via `to_dew_ast()`.
  - **Operation-level optimization**: unshape-audio has `OptimizerPipeline` for audio graphs. Image would need an `ImagePipeline` type first to enable similar optimization.
  - See `docs/design/primitive-decomposition.md` for Layer 3 architecture.

### Remaining Candidates
- [ ] **`adjust_levels`** - Has inline pixel iteration. Could be `ColorExpr::AdjustLevels { input_black, input_white, gamma, output_black, output_white }`. Formula: normalize → gamma → remap output.

---

## Verification Approach

### Ops Reference (Auto-Generated)

Run `cargo run -p extract-ops -- --md > docs/ops-reference.md` to generate the ops reference.

This tool parses all crates and extracts structs with `apply` methods (the ops-as-values pattern), including:
- Struct name and documentation
- Field names, types, and docs
- Input/output types from `apply` signature
- Source file and line number

The generated `docs/ops-reference.md` is the authoritative list of all ops.

### Finding Non-Decomposed Code

To find functions that could use expression-based decomposition but don't:

### Pattern 1: Pixel Iteration Without ColorExpr

Functions with `for y in 0..height { for x in 0..width` that apply per-pixel transforms should use `map_pixels` + `ColorExpr`.

**Grep command:**
```bash
rg "for y in 0\.\.height" --type rust crates/unshape-image/src/ -B5 | grep "pub fn"
```

### Pattern 2: Functions Returning ImageField Without Delegation

Functions that return `ImageField` and have inline pixel manipulation instead of delegating to a primitive:

```bash
rg "pub fn.*ImageField" --type rust crates/unshape-image/src/ -A20 | grep -E "(for.*0\.\.|\.push\()"
```

### Pattern 3: Missing Op Struct

Free functions without corresponding op structs violate ops-as-values:

```bash
# Find public functions that don't have a matching struct
rg "^pub fn \w+" --type rust crates/unshape-image/src/lib.rs -o | sort > /tmp/fns.txt
rg "^pub struct \w+" --type rust crates/unshape-image/src/lib.rs -o | sort > /tmp/structs.txt
comm -23 /tmp/fns.txt /tmp/structs.txt
```

### Exceptions (Not Candidates)

Some inline iteration is legitimate:
- **Convolution** - requires neighbor access, not expressible as per-pixel ColorExpr
- **FFT/spectral ops** - complex algorithms, not per-pixel
- **Dithering** - error diffusion requires state between pixels
- **Inpainting** - iterative diffusion algorithm
- **Multi-pass effects** - blue noise, datamosh, etc. with algorithmic complexity

### CI Integration (Future)

A lint could flag new functions with pixel iteration patterns that don't use `map_pixels`:

```bash
# In CI script
if rg "for y in 0\.\.height" --type rust crates/unshape-image/src/lib.rs | \
   grep -v "// PRIMITIVE:" > /dev/null; then
  echo "Warning: New pixel iteration found. Consider using map_pixels + ColorExpr."
fi
```

Functions that legitimately need pixel iteration should have a `// PRIMITIVE:` comment explaining why.
