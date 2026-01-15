# Conventions

This document describes conventions used across the resin codebase.

## Coordinate Systems

Different domains use different coordinate conventions, matching their traditional usage:

### 2D Image/Raster (resin-image)

**Screen coordinates:**
- Origin `(0, 0)` at **top-left**
- X increases rightward
- Y increases **downward**
- UV coordinates: `(0, 0)` = top-left, `(1, 1)` = bottom-right

This matches standard image formats (PNG, JPEG) and screen rendering conventions.

### 2D Vector (resin-vector)

**Math coordinates:**
- Origin `(0, 0)` at **bottom-left** (or center, depending on context)
- X increases rightward
- Y increases **upward**
- Angles: counterclockwise from positive X-axis

This matches SVG, mathematical conventions, and most vector graphics software.

### 3D Mesh (resin-mesh)

**Right-handed, Y-up:**
- Origin `(0, 0, 0)` at center
- X increases rightward
- Y increases **upward**
- Z increases **toward viewer** (out of screen)

This matches Blender, glTF, and most 3D modeling software.

```
      +Y (up)
       |
       |
       +---- +X (right)
      /
     /
   +Z (toward viewer)
```

### Audio (resin-audio)

- Time is in **seconds** (f32)
- Sample indices are 0-based
- Frequencies in **Hz**
- Amplitudes normalized to **[-1.0, 1.0]**

## Units

Unless otherwise documented, these units are assumed:

| Domain | Unit |
|--------|------|
| 2D coordinates | Arbitrary units (often pixels or normalized [0,1]) |
| 3D coordinates | Meters (but arbitrary in practice) |
| Angles | Radians (use `to_radians()` / `to_degrees()` for conversion) |
| Time | Seconds |
| Color | Linear RGB [0,1], sRGB for display |
| Audio samples | Normalized [-1, 1] |

## Naming Conventions

### Types

- `*Config` - Configuration struct for an operation (e.g., `BlurConfig`, `StftConfig`)
- `*Error` - Error enum for a domain (e.g., `ImageFieldError`, `WavError`)
- `*Result<T>` - Type alias for `Result<T, *Error>`
- `*Id` - Identifier type (usually newtype around `u32`)

### Methods

- `new()` - Primary constructor with required parameters
- `default()` - Default configuration via `Default` trait
- `apply(&self, ...)` - Apply operation to input data
- `sample(&self, ...)` - Sample a value at a coordinate/time
- `with_*()` - Builder method (only for validation/derived values)

### Parameters

- `t` - Interpolation factor [0, 1]
- `u`, `v` - Texture/UV coordinates [0, 1]
- `x`, `y`, `z` - Spatial coordinates
- `time` - Time in seconds
- `ctx` - Evaluation context (`&EvalContext`)
