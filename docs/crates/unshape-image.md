# unshape-image

Image-based fields and processing operations.

## Purpose

Bridges images and the field system. An `ImageField` wraps image data and exposes it as a `Field<Vec2, Rgba>`, enabling images to be sampled, combined with procedural fields, and processed through the unified field pipeline.

Also provides common image processing operations: convolution, color adjustments, distortion effects, and multi-scale processing.

## Related Crates

- **unshape-field** - Core field abstraction that `ImageField` implements
- **unshape-color** - Color types (`Rgba`, `Hsl`) used throughout
- **unshape-noise** - Procedural noise fields that compose with images
- **unshape-texture** - Higher-level texture nodes (checkerboard, voronoi, etc.)

## Use Cases

### Texture Sampling
Load textures and sample them in shaders or procedural pipelines:
```rust
let texture = ImageField::from_file("diffuse.png")?;
let color = texture.sample(uv, &ctx);
```

### Procedural Texture Baking
Render procedural fields to images for export or caching:
```rust
let noise = Perlin2D::new().scale(8.0);
let config = BakeConfig::new(1024, 1024).with_samples(4);
let image = bake_scalar(&noise, &config, &ctx);
export_png(&image, "noise.png")?;
```

### Image Effects Pipeline
Chain image processing operations:
```rust
let processed = blur(&image, 2);
let processed = adjust_levels(&processed, &LevelsConfig::gamma(0.8));
let processed = chromatic_aberration_simple(&processed, 0.01);
```

### Normal Map Generation
Convert heightfields to normal maps for 3D rendering:
```rust
let heightfield = bake_scalar(&noise, &config, &ctx);
let normals = heightfield_to_normal_map(&heightfield, 2.0);
```

### Multi-Scale Processing
Use image pyramids for coarse-to-fine algorithms:
```rust
let pyramid = ImagePyramid::gaussian(&image, 4);
// Process at different scales...
let result = pyramid.reconstruct_laplacian();
```

### Inpainting
Fill masked regions using diffusion or patch-based methods:
```rust
// Simple diffusion for small holes
let mask = create_color_key_mask(&image, magenta, 0.1);
let repaired = inpaint_diffusion(&image, &mask, &InpaintConfig::new(100));

// PatchMatch for texture synthesis
let config = PatchMatchConfig::new(7).with_pyramid_levels(4);
let filled = inpaint_patchmatch(&image, &mask, &config);
```

## Compositions

### With unshape-noise
Combine procedural noise with image textures:
```rust
// Distort texture UVs with noise
let noise = Simplex2D::new().scale(4.0);
// Use noise output as displacement map input
```

### With unshape-mesh
Bake textures for 3D meshes, generate normal maps from heightfields:
```rust
// Heightfield -> mesh via marching cubes
// Same heightfield -> normal map for material
```

### With unshape-vector
Rasterize vector paths to images, or use images as fill patterns:
```rust
// Rasterize SVG path to image
// Use image as texture source for gradient mesh
```

### With unshape-audio (cross-domain)
Interpret images as spectrograms (MetaSynth-style):
```rust
// Image brightness -> frequency amplitude
// Image position -> time
// Creative reinterpretation across domains
```
