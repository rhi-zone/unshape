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
| resin-field | done | 5 | Map, Zip, Zip3, FnField, Twist/Bend/Repeat. Add/Mul/Mix are Zip+Map |
| resin-fluid | - | | |
| resin-geometry | skip | — | Traits only |
| resin-gltf | skip | — | I/O |
| resin-gpu | - | | |
| resin-history | skip | — | Infrastructure |
| resin-image | done | 5+spectral | MapPixels, RemapUv, Convolve, Composite, Resize + spectral ops from resin-spectral |
| resin-jit | skip | — | Codegen |
| resin-spectral | done | 6 | FFT, IFFT, FFT2D, IFFT2D, DCT, IDCT (shared by audio/image) |
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

**Note:** Some spatial primitives are still snake_case functions. Needs refactor to ops-as-values.

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
- `AdjustHsl` duplicates ColorExpr colorspace logic
- `AdjustBrightnessContrast` duplicates per-pixel math
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

**True Primitives (5):**
1. `Map<F, M>` - output transformation
2. `Zip<A, B>` - evaluate both fields at same input, yield tuple ✅
3. `Zip3<A, B, C>` - evaluate three fields at same input, yield triple ✅
4. `FnField<I, O, F>` - universal closure adapter
5. `Twist/Bend/Repeat` - irreducible domain transforms

**Ergonomic Helpers (Layer 2):**
- `lerp(a, b, t)` - expands to `Zip3 + Map` ✅
- `zip(a, b)`, `zip3(a, b, c)` - standalone functions ✅

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
*(Add/Mul/Lerp/Mix removed - all are Zip/Zip3 + expression, provided as ergonomic helpers)*

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
- [ ] **Refactor resin-image to ops-as-values** - primitives are functions, should be structs
- [x] Add `Zip<A, B>` and `Zip3<A, B, C>` combinators to resin-field
- [ ] Expose Integrator trait in resin-particle
- [ ] Fix code duplication in resin-image colorspace ops

### Medium Priority
- [ ] Refactor emitters to CompositeEmitter pattern
- [ ] Unify constraint solvers to common pattern
- [ ] Remove trivial wrappers (downsample, upsample)

### Low Priority (Documentation)
- [ ] Document which ops are compositions vs primitives
- [ ] Add pattern-matching optimizer for common compositions
