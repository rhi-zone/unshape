# Image Primitives Architecture

Design for low-level, composable image processing primitives.

## Goal

Expose minimal surface area with highly composable primitives. Higher-level effects (drop shadow, glow, lens distortion) become thin sugar over these primitives.

## The Primitive Layer

### True Primitives

Four fundamental operations that all image effects reduce to:

| Primitive | Signature | Purpose |
|-----------|-----------|---------|
| `remap_uv` | `(ImageField, Expr&lt;Vec2, Vec2&gt;) -> ImageField` | UV coordinate remapping |
| `map_pixels` | `(ImageField, Expr&lt;Vec4, Vec4&gt;) -> ImageField` | Per-pixel color transform |
| `convolve` | `(ImageField, Kernel) -> ImageField` | Neighborhood operations |
| `composite` | `(ImageField, ImageField, BlendMode, f32) -> ImageField` | Image blending |

Plus the existing `sample_uv` on `ImageField` for texture sampling.

### Why These Four?

Every image operation falls into one of these categories:

1. **Geometric transforms** - Where to sample from (UV remapping)
2. **Color transforms** - What color to output (pixel mapping)
3. **Neighborhood transforms** - Combine nearby pixels (convolution)
4. **Compositing** - Combine multiple images (blending)

This is analogous to how audio has `DelayLine`, `PhaseOsc`, `Allpass1` as composable primitives - minimal building blocks that combine into complex effects.

## Expressions via Dew

### Why Not Closures?

The obvious API would be:

```rust
// Closure-based (NOT what we're doing)
pub fn remap_uv<F: Fn(Vec2) -> Vec2>(image: &ImageField, f: F) -> ImageField
pub fn map_pixels<F: Fn([f32; 4]) -> [f32; 4]>(image: &ImageField, f: F) -> ImageField
```

Problems:
- **Not serializable** - Can't save/load effect graphs
- **Not compilable** - Can't generate GPU shaders or JIT
- **Opaque** - Can't inspect, optimize, or transform

### Dew Expressions

Instead, use [Dew](https://github.com/rhizome-lab/dew) expressions:

```rust
pub fn remap_uv(image: &ImageField, expr: &Expr<Vec2, Vec2>) -> ImageField
pub fn map_pixels(image: &ImageField, expr: &Expr<Vec4, Vec4>) -> ImageField
```

Dew AST provides:

| Capability | Benefit |
|------------|---------|
| **Serializable** | Save/load effect graphs as JSON |
| **Interpretable** | CPU fallback, always works |
| **Cranelift JIT** | Fast CPU execution |
| **WGSL codegen** | GPU shader generation |
| **GLSL codegen** | WebGL/OpenGL support |
| **Inspectable** | Optimization passes, debugging |

### Expression Examples

```rust
// UV remapping expressions
"vec2(u, v)"                           // identity
"vec2(u + 0.1, v)"                     // translate
"vec2(u * cos(0.5) - v * sin(0.5), u * sin(0.5) + v * cos(0.5))"  // rotate
"vec2(u + sin(v * 6.28) * 0.05, v)"    // wave distortion

// Pixel mapping expressions
"vec4(r, g, b, a)"                     // identity
"vec4(lum, lum, lum, a) where lum = 0.299*r + 0.587*g + 0.114*b"  // grayscale
"vec4(1.0 - r, 1.0 - g, 1.0 - b, a)"  // invert
"vec4(r, g, b, a) if lum > 0.5 else vec4(0, 0, 0, a)"  // threshold
```

### When Closures Might Still Be Needed

Runtime closures could be useful for:
- Capturing exotic external state (sensors, network data)
- Callbacks into host application
- Prototyping before committing to an expression

But these are edge cases. For the 99% case, Dew expressions are sufficient and superior.

If needed, we could provide:
```rust
// Escape hatch for exotic cases (not serializable)
pub fn remap_uv_fn<F: Fn(Vec2) -> Vec2>(image: &ImageField, f: F) -> ImageField
```

But this should be discouraged in favor of expressions.

## Sugar Layer

Config structs remain as ergonomic APIs that generate Dew expressions internally:

```rust
// User writes:
let result = lens_distortion(&image, &LensDistortion::barrel(0.3));

// Internally becomes:
let expr = LensDistortion::barrel(0.3).to_uv_expr();
let result = remap_uv(&image, &expr);
```

### Existing Effects → Primitives

| Effect | Primitive | Expression |
|--------|-----------|------------|
| `lens_distortion` | `remap_uv` | Radial polynomial |
| `wave_distortion` | `remap_uv` | Sinusoidal offset |
| `swirl` | `remap_uv` | Angular displacement |
| `spherize` | `remap_uv` | Spherical projection |
| `transform_image` | `remap_uv` | Affine matrix |
| `grayscale` | `map_pixels` | Luminance weights |
| `invert` | `map_pixels` | `1.0 - channel` |
| `threshold` | `map_pixels` | Conditional |
| `posterize` | `map_pixels` | Quantization |
| `bit_manip` | `map_pixels` | Bitwise ops |
| `blur` | `convolve` | Gaussian kernel |
| `sharpen` | `convolve` | Sharpening kernel |
| `edge_detect` | `convolve` | Sobel kernels |
| `drop_shadow` | Composed | extract → remap_uv → convolve → map_pixels → composite |
| `glow` | Composed | map_pixels (threshold) → convolve → composite |
| `bloom` | Composed | map_pixels → pyramid convolve → composite |

## Execution Backends

The same expression can execute on different backends:

```
                    ┌─────────────────┐
                    │   Dew Expr AST  │
                    │  (serializable) │
                    └────────┬────────┘
                             │
         ┌───────────────────┼───────────────────┐
         ▼                   ▼                   ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│   Interpreter   │ │  Cranelift JIT  │ │   WGSL/GLSL     │
│   (fallback)    │ │   (fast CPU)    │ │   (GPU)         │
└─────────────────┘ └─────────────────┘ └─────────────────┘
```

### Backend Selection

```rust
// Automatic backend selection based on context
let result = remap_uv(&image, &expr);  // Uses best available

// Or explicit:
let result = remap_uv_gpu(&image, &expr);      // Force GPU
let result = remap_uv_jit(&image, &expr);      // Force Cranelift
let result = remap_uv_interpret(&image, &expr); // Force interpreter
```

## Live Coding Implications

This architecture directly supports live coding (Strudel/TidalCycles-style):

1. **Hot reload** - Parse new expression, recompile, swap in
2. **Low latency** - JIT for CPU audio, WGSL for GPU visuals
3. **Serializable state** - Save/restore sessions
4. **Inspectable** - Visualize expression graphs

## Rejected Alternatives

### Alternative 1: Trait-based Polymorphism

```rust
trait UvTransform {
    fn transform(&self, uv: Vec2) -> Vec2;
}

impl UvTransform for LensDistortion { ... }
impl UvTransform for WaveDistortion { ... }
```

**Rejected because:**
- Still requires dynamic dispatch or generics
- Each transform is a separate type (type explosion)
- Can't combine transforms without wrapper types
- Not directly compilable to shaders

### Alternative 2: Enum of Known Transforms

```rust
enum UvRemap {
    LensDistortion { strength: f32, center: Vec2 },
    Wave { amplitude: f32, frequency: f32 },
    Swirl { angle: f32, radius: f32 },
    Transform(Mat3),
    // ...
}
```

**Rejected because:**
- Closed set - can't add new transforms without modifying enum
- Combinators become complex (`Compose(Box<UvRemap>, Box<UvRemap>)`)
- Shader codegen requires matching on every variant

### Alternative 3: Closure with Serializable Hint

```rust
struct SerializableTransform {
    closure: Box<dyn Fn(Vec2) -> Vec2>,
    source: String,  // Original expression for serialization
}
```

**Rejected because:**
- Closure and source can drift out of sync
- Still can't compile closure to GPU
- Complex ownership/lifetime issues

### Alternative 4: Always GPU (wgpu compute shaders)

```rust
// All image ops go through GPU
fn remap_uv(image: &GpuImage, shader: &str) -> GpuImage
```

**Rejected because:**
- Requires GPU availability
- Overhead for small images
- Harder to debug
- Not all environments have GPU (CI, servers)

Dew expressions give us the best of all worlds - they're data (serializable), but can target any backend including GPU.

## Implementation Plan

1. **Add `remap_uv` primitive** with Dew expression input
2. **Add `map_pixels` primitive** with Dew expression input
3. **Refactor existing distortions** to use `remap_uv` internally
4. **Refactor color operations** to use `map_pixels` internally
5. **Add expression builders** to config structs (`LensDistortion::to_expr()`)
6. **Wire up backends** - interpreter first, then JIT, then WGSL

## Implementation Status

**Completed (Phase 1 - Interpreter Backend):**

- [x] `UvExpr` - Typed expression AST for Vec2 → Vec2 UV transforms
- [x] `ColorExpr` - Typed expression AST for Vec4 → Vec4 color transforms
- [x] `remap_uv(image, expr)` - UV remapping primitive
- [x] `map_pixels(image, expr)` - Per-pixel color transform primitive
- [x] `LensDistortion::to_uv_expr()` - Expression builder for lens distortion
- [x] `WaveDistortion::to_uv_expr()` - Expression builder for wave distortion
- [x] Refactored `lens_distortion()` to use `remap_uv` internally
- [x] Refactored `wave_distortion()` to use `remap_uv` internally
- [x] Static builders for common color operations: `ColorExpr::grayscale()`, `invert()`, `threshold()`, `brightness()`, `contrast()`, `posterize()`, `gamma()`, `tint()`
- [x] Static builders for common UV transforms: `UvExpr::identity()`, `translate()`, `scale_centered()`, `rotate_centered()`

**Pending:**

- [ ] Wire up Cranelift JIT backend for UvExpr/ColorExpr
- [ ] Wire up WGSL codegen backend
- [ ] Refactor remaining distortions (chromatic_aberration, swirl, spherize, transform_image)
- [ ] Migrate `dew-linalg` and switch to Dew expressions (currently using custom AST)

## Dew Capabilities

Dew provides the necessary building blocks via its module system:

| Crate | Provides |
|-------|----------|
| `dew-core` | AST, parsing, core types |
| `dew-scalar` | Scalar functions (sin, cos, etc.), interpreter, compilation backends |
| `dew-linalg` | Vec2, Vec3, Vec4, Mat3, Mat4, swizzling, compilation backends |

Each library (scalar, linalg) defines its own compilation to various backends (Cranelift, WGSL, GLSL) via feature flags, rather than having separate backend crates. This keeps compilation logic co-located with the types it operates on.

If anything is missing (e.g., specific functions, optimizations), we extend Dew first rather than working around it here.

## Open Questions

- **Kernel as expression?** - Should `convolve` also take an expression for the kernel weights?
- **Automatic backend selection** - Heuristics for choosing interpreter vs JIT vs GPU?
- **Expression optimization** - Constant folding, dead code elimination before compilation?
