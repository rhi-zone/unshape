# Noise Taxonomy

Comprehensive guide to noise types in resin: what they are, when to use them, and why we have so many.

## Why So Many Noise Types?

Different noise types serve different purposes:

1. **Spectral properties** - How energy is distributed across frequencies (white, pink, brown, blue, violet)
2. **Spatial correlation** - How neighboring samples relate (uncorrelated vs coherent)
3. **Visual/audio character** - Smooth gradients vs sharp cells vs random static
4. **Performance tradeoffs** - Simple hashing vs expensive generation

No single noise type serves all needs, so we provide a comprehensive toolkit.

## The Matrix

| Type | 1D | 2D | 3D | Character | Use Cases |
|------|----|----|----|----|---|
| **White** | `WhiteNoise1D` | `WhiteNoise2D` | `WhiteNoise3D` | Uncorrelated static | Dithering, randomization, audio hiss |
| **Perlin** | `Perlin1D` | `Perlin2D` | `Perlin3D` | Smooth gradients | Terrain, clouds, organic patterns |
| **Simplex** | `Simplex1D` | `Simplex2D` | `Simplex3D` | Smooth, less artifacts | Same as Perlin, better quality |
| **Value** | `Value1D` | `Value2D` | `Value3D` | Grid-aligned smooth | Fast approximation of Perlin |
| **Worley** | `Worley1D` | `Worley2D` | `Worley3D` | Cellular/distance | Event timing, cells, caustics |
| **Blue** | `generate_blue_noise_1d` | `generate_blue_noise` | `generate_blue_noise_3d`* | Well-distributed | Optimal dithering, sampling |
| **Pink** | `PinkNoise1D` | `PinkNoise2D` | - | Natural 1/f | Audio, natural phenomena |
| **Brown** | `BrownNoise1D` | `BrownNoise2D` | - | Random walk | Deep rumble, terrain, drift |
| **Violet** | `VioletNoise1D` | - | - | High-frequency | Audio enhancement |

*Blue noise 3D is expensive (O(n³), clamped to max 32³) - use with caution or pre-generate.

## Spectral "Colors"

Noise color refers to spectral distribution, borrowed from light/audio terminology:

```
Power
  ^
  |██                          Brown (1/f²) - more bass
  |████                        Pink (1/f) - equal per octave
  |████████                    White (flat) - equal per Hz
  |      ████████              Blue (f) - more treble
  |          ████████          Violet (f²) - very high
  +-----------------------> Frequency
```

### White Noise
- **Spectrum**: Flat - equal power at all frequencies
- **Character**: Pure static, uncorrelated samples
- **1D**: Audio hiss, random jitter
- **2D**: TV static, simple dithering (grainy)
- **3D**: Volumetric randomness, temporal dithering

### Blue Noise
- **Spectrum**: High-frequency bias
- **Character**: Well-distributed, no clumping
- **Why optimal for dithering**: Appears smoother to human vision
- **Generation**: Void-and-cluster algorithm (expensive)
- **2D**: Primary use - image dithering
- **3D**: Temporal stability in animations (very expensive to generate)

### Pink Noise (1/f)
- **Spectrum**: Equal energy per octave
- **Character**: Natural, organic variation
- **Why it sounds "natural"**: Many natural phenomena follow 1/f
- **1D**: Audio (waterfalls, wind), heartbeat variation
- **2D**: Natural textures, terrain details
- **3D**: Not typically needed (use 2D + time)

### Brown/Red Noise (1/f²)
- **Spectrum**: Strong low-frequency bias
- **Character**: Slow drift, random walk
- **Named after**: Brownian motion, not the color
- **1D**: Deep rumble, slow parameter drift
- **2D**: Smooth terrain base, fog density
- **3D**: Not typically needed

### Violet Noise (f²)
- **Spectrum**: Very high-frequency
- **Character**: Differentiated white noise
- **1D**: Audio dither for high-freq content
- **2D/3D**: Rarely useful (too high-frequency)

## Coherent vs Uncorrelated

### Coherent Noise (Perlin, Simplex, Value)
- Nearby samples are correlated
- Produces smooth gradients
- Useful for: terrain, clouds, organic textures

### Uncorrelated Noise (White, Blue)
- Each sample is independent
- No gradients, just points
- Useful for: dithering, random placement, audio

## Why Certain Combinations Don't Exist

| Missing | Reason |
|---------|--------|
| Blue 3D | O(n³) generation - possible but expensive (we have it with warning) |
| Pink/Brown 3D | Use 2D + time dimension instead |
| Violet 2D/3D | High-frequency in 2D/3D is just noise |

## Dimension Semantics

- **1D**: Time (audio), scanlines, parameter curves
- **2D**: Images, textures, terrain heightmaps
- **3D**: Volumetric (fog, clouds), or 2D + time (animation)

### 3D for Temporal Stability

For animation, use 3D noise with z = time:
```rust
let noise = WhiteNoise3D::new();
// Same (x,y) at same time = same value
let v = noise.sample(Vec3::new(x, y, frame as f32 * 0.1), &ctx);
```

This gives temporally coherent randomness - no flickering between frames.

## Performance Notes

| Type | Speed | Notes |
|------|-------|-------|
| White | Fastest | Just hashing |
| Value | Fast | Interpolated hash |
| Perlin | Medium | Gradient interpolation |
| Simplex | Medium | Slightly better than Perlin |
| Worley | Slow | Distance calculations |
| Blue (generate) | Very slow | O(n²) or O(n³) generation |

For real-time use:
- Prefer White/Value for simple randomness
- Use Perlin/Simplex for quality coherent noise
- Pre-generate blue noise textures
- Cache expensive noise in textures

## FBM (Fractal Brownian Motion)

Layer multiple octaves of coherent noise for natural detail:

```rust
let terrain = Fbm2D::new(Perlin2D::new())
    .octaves(6)
    .lacunarity(2.0)
    .gain(0.5);
```

Available for all coherent noise types (Perlin, Simplex, Value).

## Crate Organization

- **resin-noise**: Raw noise functions (`perlin2`, `worley3`, etc.)
- **resin-field**: Field wrappers (`Perlin2D`, `Worley3D`, etc.) implementing `Field<I, O>`
- **resin-image**: Image-specific (`BlueNoiseField`, `BayerField` for dithering)

This separation allows:
- Using raw functions without the Field abstraction
- Composing noise fields with other fields
- Domain-specific optimizations (image dithering)
