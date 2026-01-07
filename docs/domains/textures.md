# Textures

Procedural texture generation.

## Prior Art

### .kkrieger / Werkzeug (Farbrausch)
- **Operator stacks**: generators → filters → combiners
- **Resolution-independent**: operators work at any size
- **Tiny code**: complex textures from ~50 operators
- **Realtime generation**: textures built at load time

### Substance Designer
- **Node graph**: visual composition of texture operations
- **Atomic nodes**: noise, patterns, blends, transforms
- **PBR outputs**: albedo, normal, roughness, metallic, height
- **Tiling**: seamless texture support built-in
- **Exposed parameters**: knobs for artist control

### Shadertoy / GLSL
- **Per-pixel functions**: f(uv, time) → color
- **Raymarching**: SDFs for 3D procedural geometry
- **Domain warping**: distort coordinates recursively
- **GPU-native**: embarrassingly parallel

### libnoise
- **Noise modules**: Perlin, Simplex, Ridged, Billow
- **Combiners**: add, multiply, min, max, select
- **Modifiers**: scale, turbulence, clamp

## Core Types

```rust
/// A 2D texture
struct Texture {
    width: u32,
    height: u32,
    channels: TextureFormat,
    data: Vec<f32>,  // interleaved channels
}

enum TextureFormat {
    Gray,     // 1 channel
    GrayAlpha, // 2 channels
    Rgb,      // 3 channels
    Rgba,     // 4 channels
}

/// Procedural texture - evaluated lazily
trait TextureNode {
    fn sample(&self, uv: Vec2) → Color;
    // Or: fn evaluate(&self, output: &mut Texture);
}

/// Color (linear RGB + alpha)
struct Color {
    r: f32,
    g: f32,
    b: f32,
    a: f32,
}
```

## Primitives (Generators)

### Noise
| Primitive | Parameters | Notes |
|-----------|------------|-------|
| Perlin | scale, octaves, persistence, lacunarity | Classic gradient noise |
| Simplex | scale, octaves, ... | Less directional artifacts |
| Worley | scale, distance_func | Cellular/Voronoi |
| Value | scale, octaves | Interpolated random values |
| White | | Pure random per pixel |

### Patterns
| Primitive | Parameters | Notes |
|-----------|------------|-------|
| Gradient | direction, colors | Linear, radial, etc. |
| Checkerboard | scale, color_a, color_b | |
| Bricks | size, mortar, offset | |
| Grid | spacing, thickness | |
| Voronoi | scale, randomness | Cell regions |
| Hexagons | scale | Honeycomb |

### Solid
| Primitive | Parameters | Notes |
|-----------|------------|-------|
| Solid | color | Constant color |
| UV | | UV coordinates as color |

## Operations (Filters/Combiners)

### Blend/Composite
- **Mix**: lerp(a, b, factor)
- **Add**: a + b
- **Multiply**: a * b
- **Screen**: 1 - (1-a)(1-b)
- **Overlay**: conditional multiply/screen
- **Max/Min**: per-channel
- **Blend modes**: full Photoshop set

### Transform
- **Scale**: resize UV
- **Offset**: translate UV
- **Rotate**: rotate UV
- **Tile**: repeat UV
- **Mirror**: flip at boundaries
- **Warp**: distort UV by another texture

### Color
- **Levels**: input/output range mapping
- **Curves**: arbitrary transfer function
- **HSV adjust**: hue, saturation, value shift
- **Gradient map**: grayscale → color ramp
- **Invert**: 1 - x
- **Threshold**: binary cutoff

### Blur/Sharpen
- **Gaussian blur**: radius
- **Directional blur**: angle, radius
- **Sharpen**: amount
- **Emboss**: direction, strength

### Convolution
- **Sobel**: edge detection
- **Custom kernel**: arbitrary NxN

### Normal/Height
- **Normal from height**: generate normal map from grayscale
- **Height from normal**: integrate normals (ambiguous)
- **Curvature**: concavity/convexity from normals
- **Ambient occlusion**: from height

## Data Flow Pattern

```
Generator → Filter → Combiner → Filter → ... → Output
                        ↑
Generator → Filter ─────┘
```

Textures naturally form DAGs - multiple generators feed into combiners.

### Resolution Handling

Two approaches:
1. **Rasterized**: evaluate at specific resolution, pass pixel buffers
2. **Continuous**: evaluate at arbitrary UV, resolution-independent

Continuous is more flexible but some operations (blur, convolution) need neighborhood → require resolution.

**Hybrid**: lazy nodes that can be sampled at any UV, materialized to buffer when needed.

## PBR Workflow

Typical outputs for game/render assets:

```
Noise ──→ Height
              │
              ├──→ Normal (from height)
              │
              └──→ Roughness (inverted/curved height)

Pattern ──→ Albedo (gradient mapped)

Mask ──→ Metallic (binary/gradient)
```

## Open Questions

1. **GPU execution**: Substance runs on GPU. Do we target CPU (portable) or GPU (fast)? Or abstract over both?

2. **Tiling**: Automatic seamless tiling? Explicit tile operator? Some noises are naturally tiling.

3. **Resolution**: When to materialize vs keep lazy? Blur needs neighbors. Normal-from-height needs neighbors. But sample() at arbitrary UV is nice.

4. **3D textures**: Volumetric noise for displacement, clouds, etc. Same nodes, Vec3 input?

5. **Texture vs field**: Blender geometry nodes has "fields" - evaluated per-vertex. Same concept? Can we unify texture sampling with attribute evaluation?
