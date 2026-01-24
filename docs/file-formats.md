# File Formats

Import and export support for common file formats.

## Supported Formats

| Format | Import | Export | Crate |
|--------|--------|--------|-------|
| OBJ | Yes | Yes | `resin-mesh` |
| glTF/GLB | Yes | Yes | `resin-gltf` |
| SVG | Yes | Yes | `resin-vector` |
| WAV | Yes | Yes | `resin-audio` |

## OBJ (Wavefront)

Simple text-based mesh format. Good for geometry interchange.

### Import

```rust
use rhi_unshape_mesh::import_obj;

let mesh = import_obj("model.obj")?;

// Access data
let positions = mesh.positions();
let normals = mesh.normals();
let uvs = mesh.uvs();
```

### Export

```rust
use rhi_unshape_mesh::export_obj;

export_obj(&mesh, "output.obj")?;

// With options
export_obj_with_options(&mesh, "output.obj", ObjExportOptions {
    write_normals: true,
    write_uvs: true,
})?;
```

### Limitations

- No materials (MTL support planned)
- No animation
- Quads triangulated on import

## glTF / GLB

Modern 3D format with materials, animation, and scenes.

### Import

```rust
use rhi_unshape_gltf::{import_gltf, import_gltf_from_bytes, GltfScene};

// From file
let scene = import_gltf("model.gltf")?;

// From bytes (embedded)
let scene = import_gltf_from_bytes(&gltf_bytes)?;

// Access meshes
for mesh in scene.meshes() {
    let positions = mesh.positions();
    let indices = mesh.indices();
}

// Access materials
for material in scene.materials() {
    let base_color = material.base_color();
    let metallic = material.metallic();
}

// Merge all meshes into one
let combined = scene.merge();
```

### Export

```rust
use rhi_unshape_gltf::{export_gltf, export_glb, GltfExportOptions};

// Binary GLB (recommended)
export_glb(&mesh, "output.glb")?;

// Text glTF with external binary
export_gltf(&mesh, "output.gltf")?;

// With materials
let options = GltfExportOptions {
    material: Some(PbrMaterial {
        base_color: [1.0, 0.5, 0.0, 1.0],
        metallic: 0.0,
        roughness: 0.5,
    }),
    ..Default::default()
};
export_glb_with_options(&mesh, "output.glb", &options)?;
```

### Limitations

- Animation import not yet supported
- Skinning import not yet supported
- Only basic PBR materials on export

## SVG (Scalable Vector Graphics)

2D vector graphics format.

### Import

```rust
use rhi_unshape_vector::svg::{parse_svg, parse_path_data};

// Parse SVG file
let paths = parse_svg(&svg_content)?;

// Parse just path data (d attribute)
let path = parse_path_data("M 0 0 L 100 0 L 100 100 Z")?;
```

### Supported Path Commands

| Command | Parameters | Description |
|---------|------------|-------------|
| `M/m` | x, y | Move to |
| `L/l` | x, y | Line to |
| `H/h` | x | Horizontal line |
| `V/v` | y | Vertical line |
| `C/c` | x1, y1, x2, y2, x, y | Cubic bezier |
| `S/s` | x2, y2, x, y | Smooth cubic |
| `Q/q` | x1, y1, x, y | Quadratic bezier |
| `T/t` | x, y | Smooth quadratic |
| `A/a` | rx, ry, rotation, large, sweep, x, y | Arc |
| `Z/z` | | Close path |

### Export

```rust
use rhi_unshape_vector::svg::{path_to_svg, paths_to_svg_document, SvgExportOptions};

// Single path to d attribute
let d = path_to_svg(&path);

// Full SVG document
let svg = paths_to_svg_document(&paths, SvgExportOptions {
    width: 800.0,
    height: 600.0,
    stroke: Some("#000000"),
    stroke_width: Some(1.0),
    fill: None,
})?;

std::fs::write("output.svg", svg)?;
```

## WAV (Waveform Audio)

Uncompressed audio format.

### Import

```rust
use rhi_unshape_audio::{WavFile, WavFormat};

let wav = WavFile::load("audio.wav")?;

// Access data
let samples = wav.samples();           // Interleaved samples
let sample_rate = wav.sample_rate();   // e.g., 44100
let channels = wav.channels();         // 1 or 2
let format = wav.format();             // PCM16, Float32, etc.

// Convert to mono
let mono = wav.to_mono();

// Resample
let resampled = wav.resample(48000);
```

### Supported Formats

| Format | Bits | Range |
|--------|------|-------|
| `PCM8` | 8 | 0-255 (unsigned) |
| `PCM16` | 16 | -32768 to 32767 |
| `PCM24` | 24 | -8388608 to 8388607 |
| `PCM32` | 32 | Full i32 range |
| `Float32` | 32 | -1.0 to 1.0 |

### Export

```rust
use rhi_unshape_audio::{WavFile, WavFormat};

// Create from samples
let wav = WavFile::new(samples, sample_rate: 44100, WavFormat::PCM16);

// Or with channels
let wav = WavFile::from_channels(
    left: &left_samples,
    right: &right_samples,
    sample_rate: 44100,
    format: WavFormat::Float32,
);

// Save
wav.save("output.wav")?;
```

## Future Formats

These formats are under consideration but not yet implemented:

### High Priority

- **FBX**: Industry standard for animation interchange
  - Complex proprietary format
  - May require external SDK

- **USD**: Universal Scene Description
  - Very large specification
  - Likely needs OpenUSD bindings

### Medium Priority

- **Alembic**: Cached geometry/animation
  - Good for VFX pipelines
  - C++ library with Rust bindings available

- **MP4/WebM**: Video export
  - Useful for animation export
  - Best handled by pipeline tools

### Low Priority

- **MIDI**: Music data (basic parsing already exists)
- **PNG**: Image export (for textures/renders)
- **EXR**: HDR images

## Design Notes

### Why Limited Format Support?

Resin focuses on procedural generation rather than asset loading. File formats are secondary to the generative workflow. Complex format support may be better handled by:

1. **Cambium** (pipeline orchestrator) - external tool for data conversion
2. **Asset conditioning pipelines** - convert to simple formats before loading
3. **External converters** - Blender export scripts, etc.

### Format Guidelines

When adding format support:

1. **Prefer simple formats** - OBJ over FBX
2. **Support round-trip** - if you import, also export
3. **Document limitations** - what features are skipped
4. **Test with real files** - from Blender, Maya, etc.
