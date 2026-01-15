# resin-crossdomain

Cross-domain data interpretation and conversion utilities.

## Purpose

Inspired by MetaSynth and glitch art - the insight that structure is transferable between domains. An image can become audio, audio can become vertices, noise can be anything.

This crate provides utilities for:
- **Buffer reinterpretation** - View `&[f32]` as audio samples, vertices, or pixels
- **Image↔Audio conversion** - Spectral painting, sonification, spectrograms
- **Noise-as-anything** - Sample noise fields to generate audio, images, vertices

## Related Crates

- **resin-bytes** - Raw byte casting (re-exported as `bytes` module)
- **resin-image** - Image fields used for spectral conversions
- **resin-audio** - Audio types and spectral analysis (FFT, STFT)
- **resin-field** - Field trait for sampling noise/procedural content
- **resin-color** - Color types used in conversions

## Use Cases

### Image to Audio (MetaSynth-style)
Convert images to audio using additive synthesis, where each row represents a frequency band:
```rust
let config = ImageToAudioConfig {
    sample_rate: 44100,
    duration: 10.0,
    min_freq: 80.0,
    max_freq: 8000.0,
    log_frequency: true,
};
let audio = image_to_audio(&image, &config);
```

### Audio to Image (Spectrogram)
Convert audio to a spectrogram image:
```rust
let config = AudioToImageConfig {
    sample_rate: 44100,
    width: 512,
    height: 256,
    window_size: 2048,
    hop_size: 512,
    log_magnitude: true,
    gain: 1.5,
};
let spectrogram = audio_to_image(&audio, 44100, &config);

// Or with phase-based coloring
let colored = audio_to_image_colored(&audio, 44100, &config);
```

### Buffer Reinterpretation
View float data as different domain types without copying:
```rust
// View as audio samples
let audio_view = AudioView::new(&data, 44100);
println!("Duration: {}s", audio_view.duration());

// View as 2D vertices
let vertices = Vertices2DView::new(&data)?;
for v in vertices.iter() {
    println!("Vertex: {:?}", v);
}

// View as RGBA pixels
let pixels = PixelView::new(&data)?;
for p in pixels.iter() {
    println!("Pixel: {:?}", p);
}
```

### Noise/Fields as Audio
Sample a field along the time axis to generate audio:
```rust
// 1D noise as audio
let audio = field_to_audio(&noise_field, 44100, 2.0);

// 2D field as stereo (Y axis for panning)
let stereo = field_to_audio_stereo(&noise_2d, 44100, 2.0);
```

### Fields to Images
Bake a 2D field to an image:
```rust
// Scalar field to grayscale
let image = field_to_image(&scalar_field, 512, 512);

// RGBA field to color image
let color_image = field_rgba_to_image(&rgba_field, 512, 512);
```

### Fields to Geometry
Sample fields to generate vertex data:
```rust
// 2D path from field
let vertices_2d = field_to_vertices_2d(&path_field, 100, 1.0);

// 3D path from field
let vertices_3d = field_to_vertices_3d(&path_field_3d, 100, 1.0);

// Displacement map from 2D field
let displacement = field_to_displacement(&heightfield, 256, 256);
```

## Compositions

### With resin-noise
Sample noise fields for any domain:
```rust
// Use same noise as texture, audio modulation, and displacement
let noise = Perlin2D::new().scale(4.0);
let texture = field_to_image(&noise, 256, 256);
let audio_mod = field_to_audio(&noise.map(|v| v * 0.5), 44100, 1.0);
let displacement = field_to_displacement(&noise, 64, 64);
```

### With resin-audio
Create spectrograms for visualization or debugging:
```rust
let synth_output = synth.generate(44100);
let visualization = audio_to_image(&synth_output, 44100, &AudioToImageConfig::new(800, 200));
```

### With resin-image
Paint audio in an image editor, then sonify:
```rust
let painted = ImageField::from_file("audio_painting.png")?;
let sound = image_to_audio(&painted, &ImageToAudioConfig::new(44100, 5.0));
```

## Raw Byte Casting (via resin-bytes)

Raw byte utilities are provided by `resin-bytes`, re-exported as the `bytes` module:

```rust
use rhizome_resin_crossdomain::bytes::*;
use rhizome_resin_crossdomain::bytes_to_image_auto;

// Load any file and interpret as audio
let jpeg_bytes = std::fs::read("photo.jpg")?;
let samples = bytes_as_f32(&jpeg_bytes).unwrap();
let normalized = normalize_f32(samples);
// Now play it - hear what a JPEG "sounds like"

// Or interpret as an image (bridge function in crossdomain)
let exe_bytes = std::fs::read("program.exe")?;
let glitch_image = bytes_to_image_auto(&exe_bytes)?;

// Convert between sample formats
let i16_samples = bytes_as_i16(&raw_audio).unwrap();
let float_samples = i16_to_f32(i16_samples);
```

## Creative Applications

- **Spectral painting** - Draw frequency content visually
- **Glitch art** - Reinterpret audio as image data and vice versa
- **Data sonification** - Convert any numeric data to audio
- **Texture-to-audio** - Use textures as modulation sources
- **Audio visualization** - Generate images from audio for feedback loops
- **Raw byte glitching** - Hear what any file "sounds like"
