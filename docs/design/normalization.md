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

### 5. Sampling Interface Inconsistency 🔴

**Problem:** `sample()` methods have different signatures across crates.

| Crate | Signature | Notes |
|-------|-----------|-------|
| resin-field | `sample(input: I, ctx: &EvalContext) -> O` | Generic trait |
| resin-color | `sample(&self, t: f32) -> Rgba` | 1D gradient sampling |
| resin-image | `sample_uv(&self, u: f32, v: f32) -> Rgba` | 2D, no context |
| resin-rig | `sample(&self, time: f32) -> T` | Animation tracks |

**Issue:** `resin-field::Field` trait is powerful and generic but not adopted by other crates.

**Proposal:**
- Keep domain-specific `sample()` methods (they're ergonomic)
- Add `Field` trait implementations where it makes sense:
  - `impl Field<f32, Rgba> for Gradient`
  - `impl Field<f32, T> for AnimationTrack<T>`
- Document when to use `Field` vs direct methods

---

### 6. Config Struct Builder Patterns - SKIPPED

**Original Problem:** Config structs have inconsistent construction patterns.

**Decision: Skip this normalization.**

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

Most existing builders in the codebase are just `self.x = x; self` - pure boilerplate.

**Action:** Don't add new builders for simple configs. Existing ones can stay (removing would be churn).

---

## Low Priority

### 7. Error Handling Patterns 🔴

**Problem:** Inconsistent error handling across crates.

| Crate | Pattern |
|-------|---------|
| resin-core | `thiserror` derive, `GraphError` enum |
| resin-audio | `thiserror`, `WavError`, type alias `WavResult<T>` |
| resin-image | Manual `Display`/`Error` impl (no thiserror) |

**Proposal:**
- Standardize on `thiserror` for all error enums
- Add type aliases: `type XyzResult<T> = Result<T, XyzError>`
- Update `resin-image` to use `thiserror`

---

### 8. Coordinate System Documentation 🔴

**Problem:** Ambiguous coordinate conventions.

| Crate | Convention |
|-------|------------|
| resin-image | Screen coords: (0,0) top-left, Y down |
| resin-vector/gradient_mesh | Math coords: counterclockwise, (0,0) bottom-left implied |
| resin-mesh | Right-handed, Y-up (Blender/glTF convention) |

**Proposal:**
- Document conventions in `CLAUDE.md` or dedicated `docs/conventions.md`
- Add doc comments to key types stating their coordinate system
- No code changes needed, just documentation

---

## Implementation Order

1. ~~**Cubic bezier dedup**~~ ✅ - Done: `resin-vector/src/bezier.rs`
2. ~~**Interpolate trait**~~ ✅ - Done: `resin-easing::Lerp`, impls in rig and color
3. ~~**Color conversions**~~ ✅ - Done: `From` impls for arrays
4. ~~**Transform conversions**~~ ✅ - Done: `From` impls for matrix types
5. ~~**Config builders**~~ ⏭️ - Skipped: struct literals with `..Default::default()` are cleaner
6. **Error standardization** - Low priority, cosmetic
7. **Coordinate docs** - Documentation only

---

## Related Documents

- [ops-as-values.md](ops-as-values.md) - Operations as serializable structs
- [general-internal-constrained-api.md](general-internal-constrained-api.md) - API design philosophy
