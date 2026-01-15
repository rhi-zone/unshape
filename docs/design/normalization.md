# Codebase Normalization

This document tracks inconsistencies across the resin codebase and plans for normalization.

## Status Legend

- 🔴 Not started
- 🟡 In progress
- 🟢 Done

---

## High Priority

### 1. Transform Representations 🟢

**Problem:** Three incompatible transform representations across crates.

| Crate | Type | Representation |
|-------|------|----------------|
| resin-mesh | `Mat4` | Raw 4x4 matrix |
| resin-motion | `Transform2D` | Struct with position, rotation, scale, anchor |
| resin-rig | `Transform` | Struct with translation: Vec3, rotation: Quat, scale: Vec3 |

**Solution:**
- Keep domain-specific types (they serve different purposes)
- Added `From` impls for matrix conversions:
  - `impl From<Transform> for Mat4` and `impl From<Mat4> for Transform`
  - `impl From<Transform2D> for Mat3`
  - `impl From<LinearTransform2D> for Mat2` and `impl From<Mat2> for LinearTransform2D`

---

### 2. Interpolation Trait Fragmentation 🟢

**Problem:** `lerp` implemented independently in 6+ crates with no shared trait.

**Solution:**
- Added `Lerp` trait to `resin-easing` (chosen over `resin-core` to respect SRP)
- Implemented for: `f32`, `f64`, `Vec2`, `Vec3`, `Vec4`, `Quat`, `[T; N]`
- `resin-rig` now uses `resin-easing::Lerp` with `Interpolate` as supertrait (`Lerp + Clone + Default`)
- `resin-color` implements `Lerp` for `LinearRgb`, `Hsl`, `Hsv`, `Rgba`

```rust
// In resin-easing/src/lib.rs
pub trait Lerp {
    fn lerp_to(&self, other: &Self, t: f32) -> Self;
}
```

Crates with `Lerp` implementations:
- `resin-easing`: `f32`, `f64`, `Vec2`, `Vec3`, `Vec4`, `Quat`, `[T; N]`
- `resin-rig`: `Transform`
- `resin-color`: `LinearRgb`, `Hsl`, `Hsv`, `Rgba`
- `resin-motion`: `Transform2D`, `LinearTransform2D`

---

### 3. Duplicated Cubic Bezier 🟢

**Problem:** Identical cubic bezier evaluation implemented 3 times in resin-vector.

**Solution:**
- Created `crates/resin-vector/src/bezier.rs` with shared bezier utilities
- Updated `rasterize.rs`, `boolean.rs`, `stroke.rs` to use shared module
- Exported functions: `quadratic_point`, `quadratic_tangent`, `cubic_point`, `cubic_tangent`, `cubic_split`, `quadratic_split`, `cubic_bounds`

---

## Medium Priority

### 4. Color Representation Inconsistency 🟢

**Problem:** Color stored as arrays in some places, structs in others.

| Location | Representation |
|----------|----------------|
| `ImageField.data` | `Vec<[f32; 4]>` |
| `ImageField.sample_uv()` returns | `Rgba` struct |
| `resin-color` types | `Rgba`, `LinearRgb`, `Hsl` structs |

**Solution:**
- Added `From` conversions in `resin-color`:
  - `impl From<[f32; 3]> for LinearRgb` and `impl From<LinearRgb> for [f32; 3]`
  - `impl From<[f32; 4]> for Rgba` and `impl From<Rgba> for [f32; 4]`
- Convention: arrays for bulk storage, structs for API boundaries

---

### 5. Sampling Interface Inconsistency - SKIPPED

**Problem:** `sample()` methods have different signatures across crates.

| Crate | Signature | Notes |
|-------|-----------|-------|
| resin-field | `sample(input: I, ctx: &EvalContext) -> O` | Generic trait |
| resin-color | `sample(&self, t: f32) -> Rgba` | 1D gradient sampling |
| resin-image | `sample_uv(&self, u: f32, v: f32) -> Rgba` | 2D, no context |
| resin-rig | `sample(&self, time: f32) -> T` | Animation tracks |

**Decision: Skip this normalization.**

Adding `Field` trait impls to `Gradient` or `AnimationTrack` would require adding `resin-field` as a dependency, creating coupling for limited benefit:
- The `EvalContext` parameter is unnecessary for simple 1D/time-based sampling
- Domain-specific methods (`gradient.sample(t)`, `track.sample(time)`) are more ergonomic
- `resin-image` already implements `Field` where composability matters

**Convention:** Domain-specific `sample()` methods are the preferred API. Add `Field` impls only when field composition is actually needed (as in resin-image).

---

### 6. Config Struct Builder Patterns 🟢

**Original Problem:** Config structs have inconsistent construction patterns with ~260 builder methods.

**Solution:**
- Removed ~170 redundant builder methods that were just `self.x = x; self`
- Kept builders that validate input, compute derived values, or set multiple fields
- Updated code to use struct literal syntax with `..Default::default()`

For structs with public fields, Rust's struct literal syntax is cleaner:
```rust
let config = ClothConfig {
    damping: 0.5,
    iterations: 8,
    ..Default::default()
};
```

Builders (`with_*` methods) are only justified when they:
- Validate input (`.clamp()`, `.max(1)`)
- Compute derived values (`with_fps` → `frame_duration = 1.0 / fps`)
- Hide private fields

**Action:** Removed boilerplate builders. ~90 useful builders remain.

---

## Low Priority

### 7. Error Handling Patterns 🟢

**Problem:** Inconsistent error handling across crates - some used `thiserror`, others had manual impls.

**Solution:** Standardized all error types to use `thiserror`:
- `resin-image`: `ImageFieldError`
- `resin-audio`: `WavError`
- `resin-mesh`: `ObjError`
- `resin-vector`: `FontError`, `SvgParseError`
- `resin-gltf`: `GltfError`
- `resin-procgen`: `WfcError`

This reduced boilerplate by replacing manual `Display`, `Error`, and `From` impls with derive macros.

---

### 8. Coordinate System Documentation 🟢

**Problem:** Ambiguous coordinate conventions.

| Crate | Convention |
|-------|------------|
| resin-image | Screen coords: (0,0) top-left, Y down |
| resin-vector/gradient_mesh | Math coords: counterclockwise, (0,0) bottom-left implied |
| resin-mesh | Right-handed, Y-up (Blender/glTF convention) |

**Solution:** Created `docs/conventions.md` documenting:
- Coordinate systems for 2D image, 2D vector, 3D mesh, and audio domains
- Unit conventions (meters, seconds, radians, etc.)
- Naming conventions for types and methods

---

## Implementation Order

1. ~~**Cubic bezier dedup**~~ ✅ - Done: `resin-vector/src/bezier.rs`
2. ~~**Interpolate trait**~~ ✅ - Done: `resin-easing::Lerp`, impls in rig and color
3. ~~**Color conversions**~~ ✅ - Done: `From` impls for arrays
4. ~~**Transform conversions**~~ ✅ - Done: `From` impls for matrix types
5. ~~**Sampling interface**~~ ⏭️ - Skipped: domain-specific methods are more ergonomic
6. ~~**Config builders**~~ ✅ - Done: removed ~170 boilerplate builders
7. ~~**Error standardization**~~ ✅ - Done: all error types now use thiserror
8. ~~**Coordinate docs**~~ ✅ - Done: `docs/conventions.md`

---

## Related Documents

- [ops-as-values.md](ops-as-values.md) - Operations as serializable structs
- [general-internal-constrained-api.md](general-internal-constrained-api.md) - API design philosophy
