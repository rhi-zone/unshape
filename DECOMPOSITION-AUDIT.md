# Primitive Decomposition Audit

Track progress auditing each crate for decomposition opportunities.

## Status

| Crate | Status | Primitives Found | Key Insight |
|-------|--------|------------------|-------------|
| resin-audio | done | 9 | DelayLine, PhaseOsc, Biquad, EnvelopeFollower, Allpass1, FFT/IFFT, AffineNode, Smoother, Mix |
| resin-audio-codegen | skip | — | Codegen, not ops |
| resin-automata | - | | |
| resin-backend | skip | — | Infrastructure |
| resin-bytes | - | | |
| resin-color | - | | |
| resin-core | - | | |
| resin-crossdomain | - | | |
| resin-curve | - | | |
| resin-easing | - | | |
| resin-expr-field | - | | |
| resin-field | done | 3-4 | Map, FnField, Twist/Bend/Repeat. Add/Mul/Mix are Zip+Map |
| resin-fluid | - | | |
| resin-geometry | skip | — | Traits only |
| resin-gltf | skip | — | I/O |
| resin-gpu | - | | |
| resin-history | skip | — | Infrastructure |
| resin-image | done | 5 | map_pixels, remap_uv, convolve, composite, resize |
| resin-jit | skip | — | Codegen |
| resin-lsystem | - | | |
| resin-macros | skip | — | Proc macros |
| resin-mesh | done | 7 | Poke, SplitEdge, RipVertex, Transform, Linear/Loop/CC subdivision |
| resin-motion | - | | |
| resin-motion-fn | - | | |
| resin-noise | done | 5 | Perlin, Simplex(2D/3D), Value, Worley, Velvet |
| resin-op | skip | — | Infrastructure |
| resin-op-macros | skip | — | Proc macros |
| resin-particle | done | ~8 | CompositeEmitter pattern, 4 force primitives, Integrator missing |
| resin-physics | done | 5 | RigidBody, Force/Impulse, Integration, Collision shapes, Constraint |
| resin-pointcloud | - | | |
| resin-procgen | - | | |
| resin-rd | - | | |
| resin-rig | - | | |
| resin-scatter | - | | |
| resin-serde | skip | — | Serialization |
| resin-space-colonization | - | | |
| resin-spatial | - | | |
| resin-spline | - | | |
| resin-spring | - | | |
| resin-surface | - | | |
| resin-transform | skip | — | Traits only |
| resin-vector | done | ~8 | Line/Curve segments, polygon algorithms. Heavy flattening |
| resin-voxel | - | | |

**Legend:** `-` = not started, `partial` = in progress, `done` = complete, `skip` = not applicable

---

## Findings by Crate

### resin-image (done)

**True Primitives (5):**
1. `MapPixels { expr: ColorExpr }` - per-pixel color transform
2. `RemapUv { expr: UvExpr }` - UV coordinate remapping
3. `Convolve { kernel: Kernel }` - 2D spatial convolution
4. `Composite { blend_mode, opacity }` - image blending
5. `Resize { width, height }` - resampling

**WARNING:** Currently implemented as functions, not op structs. Needs refactor to ops-as-values.

**Decompositions Found:**

| Function | Decomposes To |
|----------|---------------|
| blur | `convolve(gaussian_kernel)` in loop |
| sharpen | `convolve(sharpen_kernel)` |
| emboss | `convolve(emboss_kernel) + map_pixels(+0.5)` |
| detect_edges | `convolve(sobel_h) + convolve(sobel_v) + map_pixels(magnitude)` |
| grayscale | `map_pixels(ColorExpr::grayscale())` |
| invert | `map_pixels(ColorExpr::invert())` |
| posterize | `map_pixels(ColorExpr::posterize(levels))` |
| threshold | `map_pixels(ColorExpr::threshold(t))` |
| lens_distortion | `remap_uv(config.to_uv_expr())` |
| wave_distortion | `remap_uv(config.to_uv_expr())` |
| glow | `threshold + blur + colorize + composite` |
| bloom | `threshold + pyramid_blur + composite` |
| drop_shadow | `extract_channel + displace + blur + colorize + composite` |
| chromatic_aberration | `set_channel(R, remap_uv) + set_channel(B, remap_uv)` |
| downsample | `resize(w/2, h/2)` - trivial wrapper |
| upsample | `resize(w*2, h*2)` - trivial wrapper |

**Issues Found:**
- `adjust_hsl()` duplicates ColorExpr colorspace logic
- `adjust_brightness_contrast()` duplicates per-pixel math
- Channel operations duplicate sampling patterns

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

**True Primitives (3-4):**
1. `Map<F, M>` - output transformation
2. `FnField<I, O, F>` - universal closure adapter
3. `Twist/Bend/Repeat` - irreducible domain transforms

**Missing Primitive:**
- `Zip<A, B>` - evaluate both fields at same input, yield tuple

**Redundant (all are Zip+Map):**

| Current | Becomes |
|---------|---------|
| `Add<A, B>` | `Zip<A, B>.map(\|(a,b)\| a + b)` |
| `Mul<A, B>` | `Zip<A, B>.map(\|(a,b)\| a * b)` |
| `Mix<A, B, T>` | `Zip<Zip<A,B>, T>.map(...)` or keep as primitive |
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
- Integrator hardcoded to Euler (should be trait)
- All emitters duplicate attribute initialization

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

---

## Summary: Minimal Primitive Sets

### Image (5)
`MapPixels`, `RemapUv`, `Convolve`, `Composite`, `Resize` *(currently functions, need op struct refactor)*

### Audio (9)
`DelayLine`, `PhaseOsc`, `Biquad`, `EnvelopeFollower`, `Allpass1`, `FFT/IFFT`, `AffineNode`, `Smoother`, `Mix`

### Field (4)
`Map`, `Zip`, `FnField`, `{Twist, Bend, Repeat}`

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

---

## Action Items

### High Priority
- [ ] **Refactor resin-image to ops-as-values** - primitives are functions, should be structs
- [ ] Add `Zip<A, B>` combinator to resin-field
- [ ] Expose Integrator trait in resin-particle
- [ ] Fix code duplication in resin-image colorspace ops

### Medium Priority
- [ ] Refactor emitters to CompositeEmitter pattern
- [ ] Unify constraint solvers to common pattern
- [ ] Remove trivial wrappers (downsample, upsample)

### Low Priority (Documentation)
- [ ] Document which ops are compositions vs primitives
- [ ] Add pattern-matching optimizer for common compositions
