# unshape-bytes

Raw byte reinterpretation utilities for glitch-art style domain crossing.

## Purpose

Provides safe byte casting between different numeric representations, enabling creative reinterpretation of data across domains. Load any file as audio samples, interpret audio as vertex positions, visualize executables as images.

This is a minimal crate with only `bytemuck` as a dependency.

## Related Crates

- **unshape-crossdomain** - Re-exports this crate and adds `bytes_to_image` bridge

## Use Cases

### Raw Byte Casting
Reinterpret bytes as different numeric types:
```rust
use rhi_unshape_bytes::*;

// Any file as audio samples
let jpeg_bytes = std::fs::read("photo.jpg")?;
let samples = bytes_as_f32(&jpeg_bytes).unwrap();
let normalized = normalize_f32(samples);
// Play it - hear what a JPEG "sounds like"

// Bytes as vertex positions
let verts = bytes_as_xyz(&model_bytes).unwrap();

// Bytes as RGBA pixels
let pixels = bytes_as_rgba(&data).unwrap();
```

### Sample Format Conversion
Convert between audio sample formats:
```rust
// i16 PCM to float
let i16_samples = bytes_as_i16(&raw_audio)?;
let float_samples = i16_to_f32(i16_samples);

// Float back to i16
let back = f32_to_i16(&float_samples);

// u8 to float (for 8-bit audio)
let u8_samples = &raw_bytes;
let floats = u8_to_f32(u8_samples);
```

### Normalizing Arbitrary Data
After raw casting, values may be arbitrary (NaN, Inf, huge values):
```rust
let raw_floats = bytes_as_f32(&any_file)?;

// Normalize to [-1, 1] for audio
let audio_ready = normalize_f32(raw_floats);

// Normalize to [0, 1] for image brightness
let image_ready = normalize_f32_unsigned(raw_floats);
```

## Available Functions

### Byte Casting
- `bytes_as_f32`, `bytes_as_i16`, `bytes_as_u16`, `bytes_as_i32`, `bytes_as_u32`
- `bytes_as_rgba`, `bytes_as_rgb` - pixel data
- `bytes_as_xy`, `bytes_as_xyz`, `bytes_as_xyzw` - coordinate data

### Format Conversion
- `i16_to_f32`, `u8_to_f32`, `u8_to_f32_unsigned`
- `f32_to_i16`, `f32_to_u8`

### Normalization
- `normalize_f32` - to [-1, 1] range
- `normalize_f32_unsigned` - to [0, 1] range

## Creative Applications

- **Data sonification** - Hear what any file sounds like
- **Glitch visualization** - See executables, audio, or any data as images
- **Raw audio processing** - Work with PCM samples directly
- **Cross-domain experiments** - Treat mesh data as audio, audio as vertices
