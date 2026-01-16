# GPU Image Processing

Integration between `resin-image` expressions and `resin-gpu` for GPU-accelerated image processing.

## Architecture

```
UvExpr / ColorExpr
        │
        ▼ to_dew_ast()
    dew::Ast
        │
        ▼ emit_wgsl()
   WGSL expression
        │
        ▼ wrap in shader template
   Complete WGSL shader
        │
        ▼ compile + dispatch
   wgpu compute pipeline
        │
        ▼
   GpuTexture output
```

## API Design

```rust
// In resin-gpu, add image module

/// GPU-accelerated UV remapping.
pub fn remap_uv_gpu(
    ctx: &GpuContext,
    input: &GpuTexture,
    expr: &UvExpr,
) -> GpuResult<GpuTexture>;

/// GPU-accelerated per-pixel color transform.
pub fn map_pixels_gpu(
    ctx: &GpuContext,
    input: &GpuTexture,
    expr: &ColorExpr,
) -> GpuResult<GpuTexture>;
```

## Shader Template

For `remap_uv`:

```wgsl
@group(0) @binding(0) var input_texture: texture_2d<f32>;
@group(0) @binding(1) var input_sampler: sampler;
@group(0) @binding(2) var output_texture: texture_storage_2d<rgba8unorm, write>;

struct Uniforms {
    width: u32,
    height: u32,
}
@group(0) @binding(3) var<uniform> uniforms: Uniforms;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;

    if (x >= uniforms.width || y >= uniforms.height) {
        return;
    }

    // Normalized UV coordinates
    let uv = vec2<f32>(f32(x) + 0.5, f32(y) + 0.5) / vec2<f32>(f32(uniforms.width), f32(uniforms.height));

    // === GENERATED EXPRESSION ===
    let src_uv = {EXPR};  // e.g., uv + vec2(0.1, 0.0)
    // === END GENERATED ===

    let color = textureSampleLevel(input_texture, input_sampler, src_uv, 0.0);
    textureStore(output_texture, vec2<i32>(i32(x), i32(y)), color);
}
```

For `map_pixels`:

```wgsl
@group(0) @binding(0) var input_texture: texture_2d<f32>;
@group(0) @binding(1) var output_texture: texture_storage_2d<rgba8unorm, write>;

struct Uniforms {
    width: u32,
    height: u32,
}
@group(0) @binding(2) var<uniform> uniforms: Uniforms;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;

    if (x >= uniforms.width || y >= uniforms.height) {
        return;
    }

    let rgba = textureLoad(input_texture, vec2<i32>(i32(x), i32(y)), 0);

    // === GENERATED EXPRESSION ===
    let result = {EXPR};  // e.g., vec4(dot(rgba.rgb, vec3(0.299, 0.587, 0.114)), ...)
    // === END GENERATED ===

    textureStore(output_texture, vec2<i32>(i32(x), i32(y)), result);
}
```

## Implementation Steps

1. **Add wgsl feature to resin-gpu** that pulls in resin-image with wgsl
2. **Create shader template strings** for remap_uv and map_pixels
3. **Implement `remap_uv_gpu`**:
   - Convert UvExpr to dew AST
   - Emit WGSL via dew-linalg
   - Substitute into template
   - Compile and dispatch
4. **Implement `map_pixels_gpu`**:
   - Same pattern for ColorExpr
5. **Add texture input support** to GpuTexture (currently only output)

## Dependencies

```toml
# resin-gpu/Cargo.toml
[features]
image-expr = ["dep:rhizome-resin-image", "rhizome-resin-image/wgsl"]

[dependencies]
rhizome-resin-image = { path = "../resin-image", optional = true }
```

## Future: Pipeline Fusion

Multiple operations can be fused into a single shader:

```rust
// Instead of separate passes:
let a = remap_uv_gpu(&ctx, &input, &wave_expr)?;
let b = map_pixels_gpu(&ctx, &a, &grayscale_expr)?;

// Fused into one pass:
let b = gpu_pipeline(&ctx, &input)
    .remap_uv(&wave_expr)
    .map_pixels(&grayscale_expr)
    .execute()?;
```

This would generate a single shader that does both transforms.
