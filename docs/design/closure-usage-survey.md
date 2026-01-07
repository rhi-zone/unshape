# Closure Usage Survey

Where would users want to pass custom functions/closures across resin domains?

Goal: understand the problem space before designing an expression language.

## Mesh

| Operation | Closure signature | Example use case |
|-----------|-------------------|------------------|
| `map_vertices` | `Fn(Vec3) -> Vec3` | Custom displacement, warping |
| `filter_vertices` | `Fn(Vec3) -> bool` | Select vertices by position |
| `filter_faces` | `Fn(&Face) -> bool` | Select faces by normal, area |
| `vertex_color_from` | `Fn(Vec3) -> Color` | Procedural vertex coloring |
| `custom_subdivision` | `Fn(&Face) -> Vec<Face>` | Non-standard subdivision |

**Common patterns:**
- Position → Position (transforms)
- Position → Scalar (selection, weighting)
- Position → Color (procedural coloring)

## Textures

| Operation | Closure signature | Example use case |
|-----------|-------------------|------------------|
| `map_pixels` | `Fn(Color) -> Color` | Color grading, adjustments |
| `sample_custom` | `Fn(Vec2) -> Color` | Entirely custom texture |
| `warp` | `Fn(Vec2) -> Vec2` | UV distortion |
| `blend_custom` | `Fn(Color, Color) -> Color` | Custom blend mode |
| `threshold_custom` | `Fn(f32) -> f32` | Custom transfer function |

**Common patterns:**
- Color → Color (adjustments)
- UV → UV (warping)
- UV → Color (sampling)
- Scalar → Scalar (curves, transfer functions)

## Audio

| Operation | Closure signature | Example use case |
|-----------|-------------------|------------------|
| `map_samples` | `Fn(f32) -> f32` | Waveshaping, distortion |
| `custom_oscillator` | `Fn(f32) -> f32` | Phase → amplitude |
| `custom_envelope` | `Fn(f32) -> f32` | Time → amplitude |
| `custom_filter` | `Fn(&[f32]) -> f32` | FIR filter with custom kernel |

**Common patterns:**
- Sample → Sample (waveshaping)
- Phase → Sample (oscillators)
- Time → Scalar (envelopes, LFOs)

## Vector 2D

| Operation | Closure signature | Example use case |
|-----------|-------------------|------------------|
| `map_points` | `Fn(Vec2) -> Vec2` | Custom warping |
| `filter_points` | `Fn(Vec2) -> bool` | Select points |
| `vary_stroke` | `Fn(f32) -> f32` | t along path → stroke width |
| `vary_color` | `Fn(f32) -> Color` | t along path → color |

**Common patterns:**
- Position → Position
- t (0-1) → Scalar (varying properties along path)

## Rigging

| Operation | Closure signature | Example use case |
|-----------|-------------------|------------------|
| `custom_constraint` | `Fn(&Pose) -> Transform` | Procedural bone positioning |
| `driver` | `Fn(f32) -> f32` | Parameter → parameter mapping |
| `blend_custom` | `Fn(Pose, Pose, f32) -> Pose` | Custom pose interpolation |
| `physics_force` | `Fn(Vec3, Vec3) -> Vec3` | Position, velocity → force |

**Common patterns:**
- Scalar → Scalar (drivers)
- Pose → Pose (constraints)

---

## Summary: Common Closure Signatures

| Signature | Domains | Frequency |
|-----------|---------|-----------|
| `Fn(Vec3) -> Vec3` | Mesh, Rigging | High |
| `Fn(Vec2) -> Vec2` | Texture, Vector | High |
| `Fn(f32) -> f32` | All | Very High |
| `Fn(Color) -> Color` | Texture | Medium |
| `Fn(Vec3) -> f32` | Mesh (selection) | Medium |
| `Fn(f32) -> Color` | Texture, Vector | Medium |
| `Fn(Vec3) -> Color` | Mesh | Low |
| `Fn(Vec3) -> bool` | Mesh | Medium |
| Complex (multiple inputs) | Various | Low |

## What Could Be Named Ops Instead?

Many "closures" are actually common operations:

| Closure pattern | Named op equivalent |
|----------------|---------------------|
| `\|v\| v * 2.0` | `Scale::uniform(2.0)` |
| `\|v\| v + offset` | `Translate::new(offset)` |
| `\|v\| rotate(v, angle)` | `Rotate::new(axis, angle)` |
| `\|c\| c.brighten(0.1)` | `Brightness::new(0.1)` |
| `\|x\| x.clamp(0.0, 1.0)` | `Clamp::new(0.0, 1.0)` |
| `\|x\| x.powf(2.2)` | `Gamma::new(2.2)` |
| `\|uv\| uv * 2.0` | `TileUV::new(2.0)` |

## What Genuinely Needs Expressions?

Cases where named ops aren't enough:

1. **Composite math**: `|v| v * noise(v * 4.0) + vec3(0, v.y * 0.5, 0)`
2. **Conditionals**: `|v| if v.y > 0.0 { v * 2.0 } else { v }`
3. **Domain-specific formulas**: `|t| sin(t * TAU * 3.0) * exp(-t * 2.0)` (custom envelope)
4. **Artistic curves**: arbitrary remapping functions

## Questions to Answer

1. **How common are "genuinely needs expression" cases?**
   - If rare: named ops + escape hatch might suffice
   - If common: need expression language

2. **What's the expression language complexity?**
   - Just math (+-*/, sin, cos, pow, etc.)?
   - Conditionals (if/else)?
   - Variables/bindings?
   - Loops?
   - Function definitions?

3. **Per-domain or unified?**
   - Same expression language everywhere?
   - Or domain-specific (MeshExpr, AudioExpr, etc.)?

4. **Prior art expression languages?**
   - Blender drivers (Python subset)
   - Houdini VEX
   - Shadertoy GLSL
   - Max/MSP expr object
   - Desmos (math only)

---

## Notes

(Space for investigation notes as we explore each domain)

### Mesh expressions
TODO

### Texture expressions
TODO

### Audio expressions
TODO

### Vector expressions
TODO

### Rigging expressions
TODO
