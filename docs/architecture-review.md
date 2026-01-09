# Architecture Review

Review of patterns, inconsistencies, and code smells across the workspace.

## Summary

| Category | Issue | Files | Severity |
|----------|-------|-------|----------|
| API Design | Tuple returns violate CLAUDE.md rule | `resin-vector/boolean.rs`, `resin-surface/lib.rs` | HIGH |
| Error Handling | `panic!()` in library code | `resin-spline/lib.rs` (4 instances) | HIGH |
| Code Duplication | Collision response pattern repeats | `resin-physics/lib.rs` | MEDIUM |
| Type Safety | String-based tile IDs instead of enums | `resin-procgen/lib.rs` | MEDIUM |
| Consistency | Trait implementations vary between similar types | `resin-spline/lib.rs` | MEDIUM |
| Complexity | Large functions handle multiple concerns | `resin-physics/lib.rs::step()` | MEDIUM |

## Strengths

1. **Consistent builder pattern** - Most crates use `with_*` methods for configuration
2. **Good public API organization** - Clean re-exports in lib.rs files
3. **Consistent glam usage** - Vec2, Vec3, Quat, Mat4 throughout
4. **Comprehensive test coverage** - 750+ tests across workspace
5. **Clean error handling** - `thiserror` used consistently where errors exist

## HIGH Priority Issues

### 1. Tuple Returns (CLAUDE.md Violation)

Per CLAUDE.md: "Return tuples from functions" is explicitly a negative constraint.

**Locations:**
- `resin-vector/src/boolean.rs:132` - `bounds() -> (Vec2, Vec2)`
- `resin-vector/src/boolean.rs:158` - `split() -> (CurveSegment, CurveSegment)`
- `resin-vector/src/boolean.rs:424` - `closest_point_on_curve() -> (Vec2, f32)`
- `resin-surface/src/lib.rs:147` - `domain() -> ((f32, f32), (f32, f32))`

**Fix:** Create named structs:
```rust
pub struct Bounds { pub min: Vec2, pub max: Vec2 }
pub struct SplitResult { pub before: CurveSegment, pub after: CurveSegment }
pub struct ClosestPoint { pub point: Vec2, pub distance: f32 }
pub struct Domain2D { pub u: (f32, f32), pub v: (f32, f32) }
```

### 2. Panics in Library Code

Library code should return `Result<T, E>` instead of panicking.

**Locations in `resin-spline/src/lib.rs`:**
- Line 126: `panic!("Cannot evaluate empty spline")` in `CatmullRom::evaluate()`
- Line 248: `panic!("Cannot evaluate empty spline")` in `BSpline::evaluate()`
- Line 371: `panic!("Cannot evaluate empty spline")` in `BezierSpline::evaluate()`
- Line 519: `panic!("Cannot evaluate empty NURBS curve")` in `Nurbs::evaluate()`

**Fix:** Return `Option<T>` or `Result<T, SplineError>`.

## MEDIUM Priority Issues

### 3. Collision Response Duplication

In `resin-physics/src/lib.rs` lines 1022-1077, collision pairs are handled with manual normal flipping:

```rust
// Repeated pattern for sphere-plane, box-plane, sphere-box
(ColliderShape::X, ColliderShape::Y) => { ... }
(ColliderShape::Y, ColliderShape::X) => {
    // Same logic with flipped normal
}
```

**Fix:** Create helper that handles symmetry:
```rust
fn collide_symmetric<F>(a: &Collider, b: &Collider, f: F) -> Option<Contact>
where F: Fn(&Collider, &Collider) -> Option<Contact>
```

### 4. String-Based Tile IDs

In `resin-procgen/src/lib.rs`, WFC tiles use string names with HashMap lookups:
```rust
pub fn add_tile(&mut self, name: &str) -> usize
pub fn add_rule(&mut self, from: &str, to: &str, ...)
```

**Fix:** Use newtype `TileId(usize)` for type safety, keep names for display only.

### 5. Inconsistent Trait Implementations

In `resin-spline/src/lib.rs`:
- `BezierSpline<T>` implements `Default`
- `BSpline<T>` does NOT implement `Default`
- `CatmullRom<T>` does NOT implement `Default`

**Fix:** Add missing derives consistently.

### 6. Large Function Complexity

`PhysicsWorld::step()` (55 lines) handles:
- Gravity application
- Velocity integration
- Collision detection
- Constraint solving
- Position integration

**Fix:** Extract into smaller methods:
```rust
fn apply_forces(&mut self, dt: f32)
fn integrate_velocities(&mut self, dt: f32)
fn detect_collisions(&self) -> Vec<Contact>
fn solve_constraints(&mut self, contacts: &[Contact])
fn integrate_positions(&mut self, dt: f32)
```

## LOW Priority / Observations

### Error Type Variation

Different patterns across crates:
- `resin-core`: `GraphError`, `TypeError`
- `resin-gpu`: `GpuError` with `#[from]`
- `resin-vector`: `FontError` type alias
- `resin-procgen`: `WfcError`

Not necessarily a problem if each domain has specific error needs.

### Feature Flags

No conditional compilation used. Consider optional features for:
- `resin-gpu` (heavyweight wgpu dependency)
- `resin-gltf` (external format support)
- `resin-image` (image processing)

## Refactoring Plan

1. **Phase 1 (HIGH):** Fix tuple returns, replace panics with Results
2. **Phase 2 (MEDIUM):** Deduplicate collision code, add missing traits
3. **Phase 3 (MEDIUM):** Refactor large functions, improve type safety
