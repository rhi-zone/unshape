# Texture Materialization

When does a procedural texture (lazy field) become actual pixel data?

## The Tension

```rust
// Lazy field: resolution-independent, no memory cost
trait Field {
    fn sample(&self, uv: Vec2) -> Color;
}

// Materialized image: pixels at specific resolution
struct Image {
    pixels: Vec<Color>,  // or GpuTexture
    width: u32,
    height: u32,
}
```

**Lazy is good:**
- No memory until needed
- Resolution-independent (theoretically)
- Composable (chain ops without intermediate allocations)
- GPU-friendly (shader = field evaluation)

**But some ops need pixels:**
- Blur (average neighbors)
- Normal from heightmap (sample dx, dy)
- Edge detection, erosion, dilation
- Any convolution kernel

## Operations Classification

| Op type | Needs neighbors? | Can stay lazy? |
|---------|------------------|----------------|
| Color adjust (brightness, contrast) | No | Yes |
| Per-pixel math (multiply, add) | No | Yes |
| Noise sampling | No | Yes |
| UV transform (scale, rotate) | No | Yes |
| Blur | Yes | No - must materialize |
| Sharpen | Yes | No |
| Normal from height | Yes | No |
| Edge detection | Yes | No |

## Design: Explicit Field/Image Split

Two distinct types, not an enum:

```rust
/// Lazy, resolution-independent
trait Field {
    fn sample(&self, uv: Vec2) -> Color;
}

/// Materialized, has pixels at specific resolution
struct Image {
    data: Vec<Color>,  // or GpuTexture
    width: u32,
    height: u32,
}
```

**Field ops (lazy, composable):**
```rust
impl<F: Field> FieldOps for F {
    /// Transform colors (fuses with previous ops)
    fn map<G: Fn(Color) -> Color>(&self, f: G) -> impl Field;

    /// Transform UVs
    fn uv_transform(&self, matrix: Mat3) -> impl Field;

    /// Materialize at explicit resolution
    fn render(&self, width: u32, height: u32) -> Image;
}
```

**Image ops (materialized, have resolution):**
```rust
impl Image {
    /// Neighbor ops (work on pixels)
    fn blur(&self, radius_uv: f32) -> Image;
    fn sharpen(&self, amount: f32) -> Image;
    fn normals_from_height(&self) -> Image;

    /// Resize
    fn upscale(&self, width: u32, height: u32) -> Image;
    fn downscale(&self, width: u32, height: u32) -> Image;

    /// Back to lazy (samples this image)
    fn as_field(&self) -> impl Field;
}
```

**Example workflow:**
```rust
let noise = Perlin::new()           // Field
    .map(|v| Color::gray(v))         // Field (fused)
    .brightness(0.1)                 // Field (fused)
    .contrast(1.2);                  // Field (still lazy)

// Explicit materialization
let image = noise.render(1024, 1024);  // Now we have pixels
let blurred = image.blur(0.01);         // Image op

// Or: lower res for performance
let cheap = noise.render(512, 512)
    .blur(0.01)
    .upscale(1024, 1024);
```

**Key principle:** Resolution is always explicit at `render()`. No hidden propagation.

## Kernel Fusion

### Field Fusion (all ops)

Multiple lazy Field ops fuse into single evaluation:

```rust
// Conceptually three ops
noise.brightness(0.1).contrast(1.2).saturate(0.5)

// Should compile to single shader/function:
fn fused_sample(uv: Vec2) -> Color {
    let v = noise.sample(uv);
    let c = Color::gray(v);
    let c = c.brightness(0.1);
    let c = c.contrast(1.2);
    let c = c.saturate(0.5);
    c
}
```

### Image Fusion (point-wise ops)

Even materialized Images can defer point-wise ops:

```rust
image              // materialized (has pixels)
  .blur(5)         // neighbor op -> must allocate
  .brightness(0.1) // point-wise -> defer
  .contrast(1.2)   // point-wise -> defer
  .sharpen(2)      // neighbor op -> fuse brightness+contrast into read
```

**Implementation:**

```rust
struct Image {
    /// Actual pixel data
    data: GpuTexture,

    /// Pending point-wise ops (fused on next neighbor op or output)
    pending_ops: Vec<PointWiseOp>,
}

impl Image {
    fn brightness(&self, amount: f32) -> Image {
        // Don't allocate - just queue the op
        Image {
            data: self.data.clone(),
            pending_ops: chain(&self.pending_ops, Brightness(amount)),
        }
    }

    fn blur(&self, radius: f32) -> Image {
        // Must read pixels - fuse pending ops into the read
        let fused = compile_fused_shader(&self.pending_ops, BlurShader(radius));
        let new_data = execute(fused, &self.data);
        Image { data: new_data, pending_ops: vec![] }
    }
}
```

**Memory benefit:**

| Without fusion | With fusion |
|----------------|-------------|
| `blur -> brightness -> contrast -> sharpen` | Same pipeline |
| 4 allocations | 2 allocations |
| 4 shader passes | 2 shader passes |

### Three Levels of Deferral

| Type | Has pixels? | Ops deferred? |
|------|-------------|---------------|
| Field | No | All ops |
| Image (pending) | Yes | Point-wise ops |
| Image (materialized) | Yes | None (just executed neighbor op) |

**API implication:** Both Field and Image support fusion. Field defers everything until `render()`. Image defers point-wise until next neighbor op or final output.

## Resolution: Explicit and Simple

### Resolution is always explicit at `render()`

```rust
// Field has no resolution
let field = Perlin::new().brightness(0.1);

// User chooses resolution when materializing
let image = field.render(1024, 1024);
```

### Neighbor ops work on Images (already have resolution)

```rust
// blur() is an Image method, not Field method
// Resolution comes from the image itself
let blurred = image.blur(0.01);  // image is 1024x1024, blur works at that res
```

### Want different resolution? Explicit render/resize

```rust
// Full res blur
let result = noise.render(1024, 1024).blur(0.01);

// Cheap blur (half res, upscale after)
let result = noise.render(512, 512).blur(0.01).upscale(1024, 1024);

// No magic, no hidden resolution propagation
```

### Resolution Units

Image ops use UV-space units for resolution independence:

```rust
// blur(0.01) = blur by 1% of image width
// At 1024px: 10.24 pixel radius
// At 512px: 5.12 pixel radius
// Visually similar relative to image size

image.blur(0.01)  // UV units (default)
image.blur_px(10) // Pixel units (when you need exact control)
```

## Default Resolution (Convenience)

When you don't want to specify resolution everywhere:

```rust
impl<F: Field> FieldOps for F {
    /// Explicit resolution (preferred)
    fn render(&self, width: u32, height: u32) -> Image;

    /// Use context's resolution (convenience)
    fn render_default(&self, ctx: &EvalContext) -> Image {
        self.render(ctx.default_width, ctx.default_height)
    }
}

struct EvalContext {
    default_width: u32,
    default_height: u32,
    // ...
}
```

Context provides a default. No magic propagation - just a fallback value.

## GPU Execution Model

On GPU:
- **Field** = shader program (runs per-pixel in parallel)
- **Materialize** = render to texture
- **Neighbor op** = shader that samples texture

```
Field (shader) -> render() -> GPU texture -> blur (shader samples texture)
```

Fusion on GPU = compose shader functions before compilation.

## Open Questions

1. **Partial materialization:** Can we materialize only the region needed? (tiled evaluation)

2. **Mipmap integration:** When to generate mipmaps? Affects blur quality at different scales.

3. **Image -> Field -> Image roundtrip:** Any precision/quality concerns?

## Summary

| Aspect | Decision |
|--------|----------|
| Types | Separate `Field` (lazy) and `Image` (materialized) |
| Resolution | Explicit at `render()`, no propagation |
| Neighbor ops | Image methods only (already have resolution) |
| Fusion | Field ops fuse into single evaluation |
| Units | UV-space default, pixel units available |
| Default resolution | Context provides fallback for convenience |
