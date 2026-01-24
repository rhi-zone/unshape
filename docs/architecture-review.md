# Architecture Review

Review of patterns, inconsistencies, and code smells across the workspace.

**Status:** HIGH and MEDIUM priority issues resolved.

## Summary

| Category | Issue | Files | Severity | Status |
|----------|-------|-------|----------|--------|
| API Design | Tuple returns violate CLAUDE.md rule | `unshape-vector/boolean.rs`, `unshape-surface/lib.rs` | HIGH | ✅ Fixed |
| Error Handling | `panic!()` in library code | `unshape-spline/lib.rs` (4 instances) | HIGH | ✅ Fixed |
| Code Duplication | Collision response pattern repeats | `unshape-physics/lib.rs` | MEDIUM | ✅ Fixed |
| Type Safety | String-based tile IDs instead of enums | `unshape-procgen/lib.rs` | MEDIUM | ✅ Fixed |
| Consistency | Trait implementations vary between similar types | `unshape-spline/lib.rs` | MEDIUM | ✅ Fixed |
| Complexity | Large functions handle multiple concerns | `unshape-physics/lib.rs::step()` | MEDIUM | ✅ Fixed |

## Recent Additions

New modules added (Jan 2026):

| Module | Crate | Purpose |
|--------|-------|---------|
| `topology.rs` | unshape-mesh | Euler characteristic, genus, manifold testing, boundary detection |
| `weights.rs` | unshape-mesh | Vertex weight painting, smoothing, heat diffusion for skinning |
| `lod.rs` | unshape-mesh | Automatic LOD generation via mesh decimation |
| `spatial.rs` | unshape-audio | 3D spatial audio, HRTF, distance attenuation, Doppler |
| `pattern.rs` | unshape-audio | TidalCycles-inspired pattern combinators (fast/slow/rev/jux) |

These follow existing patterns and conventions. No architectural concerns.

## Strengths

1. **Consistent builder pattern** - Most crates use `with_*` methods for configuration
2. **Good public API organization** - Clean re-exports in lib.rs files
3. **Consistent glam usage** - Vec2, Vec3, Quat, Mat4 throughout
4. **Comprehensive test coverage** - 750+ tests across workspace
5. **Clean error handling** - `thiserror` used consistently where errors exist

## HIGH Priority Issues (✅ Resolved)

### 1. Tuple Returns (CLAUDE.md Violation) ✅

Per CLAUDE.md: "Return tuples from functions" is explicitly a negative constraint.

**Fixed:** Created named structs:
- `Bounds2D { min: Vec2, max: Vec2 }` - bounds()
- `SplitCurve { before: CurveSegment, after: CurveSegment }` - split()
- `ClosestPoint { point: Vec2, t: f32 }` - closest_point_on_curve()
- `SurfaceDomain { u: ParameterRange, v: ParameterRange }` - domain()
- `ParameterRange { min: f32, max: f32 }` - for surface parameter ranges

### 2. Panics in Library Code ✅

Library code should return `Result<T, E>` instead of panicking.

**Fixed:** Changed all spline evaluate/derivative methods to return `Option<T>`:
- `CatmullRom::evaluate()` → `Option<T>`
- `BSpline::evaluate()` → `Option<T>`
- `BezierSpline::evaluate()` → `Option<T>`
- `BezierSpline::derivative()` → `Option<T>`
- `Nurbs::evaluate()` → `Option<T>`
- `Nurbs::derivative()` → `Option<T>`

## MEDIUM Priority Issues

### 3. Collision Response Duplication ✅

In `unshape-physics/src/lib.rs`, collision pairs were handled with manual normal flipping.

**Fixed:** Added `Contact::flip()` helper method to handle symmetric collision pairs cleanly:
```rust
impl Contact {
    fn flip(mut self) -> Self {
        self.normal = -self.normal;
        std::mem::swap(&mut self.body_a, &mut self.body_b);
        self
    }
}
```

### 4. String-Based Tile IDs ✅

**Fixed:** Implemented layered design:
- `TileId(usize)` newtype for type-safe, zero-overhead tile references
- `TileSet` base type uses `TileId` for rules and weights
- `NamedTileSet` wrapper provides ergonomic string-based API
- Users can start with strings, drop to IDs for hot paths via `into_inner()`

### 5. Inconsistent Trait Implementations ✅

**Fixed:** Added `Default` implementations for all spline types:
- `CatmullRom<T>` - empty points, alpha 0.5
- `BSpline<T>` - empty points, degree 3
- `Nurbs<T>` - empty points, degree 3, empty knots

### 6. Large Function Complexity ✅

**Fixed:** `PhysicsWorld::step()` refactored into focused methods:
```rust
pub fn step(&mut self) {
    self.apply_forces(dt);
    self.integrate_velocities(dt);
    let contacts = self.detect_collisions();
    self.solve_contacts_and_constraints(&contacts);
    self.integrate_positions(dt);
}
```

## LOW Priority / Observations

### Utility Function Duplication

Some helper functions are duplicated across modules. Noted but acceptable trade-offs:

**`build_adjacency`** (3 implementations):
- `weights.rs` → returns `Vec<HashSet<usize>>` (for weight smoothing)
- `ops.rs` → returns `Vec<Vec<usize>>` (for mesh operations)
- `geodesic.rs` → returns `Vec<Vec<usize>>` (for distance calculation)

Different return types serve different use cases. HashSet provides O(1) neighbor lookup, Vec is more cache-friendly for iteration.

**`edge_key`** (2 implementations):
- `topology.rs` - canonical edge ordering for topology analysis
- `subdivision.rs` - same pattern for edge midpoint caching

Simple 2-line function. Creating a shared utility would add more complexity than it saves.

**Recommendation:** Leave as-is. The duplication is minimal and serves slightly different purposes. A shared mesh utilities module could be considered if more common patterns emerge.

### Error Type Variation

Different patterns across crates:
- `unshape-core`: `GraphError`, `TypeError`
- `unshape-gpu`: `GpuError` with `#[from]`
- `unshape-vector`: `FontError` type alias
- `unshape-procgen`: `WfcError`

Not necessarily a problem if each domain has specific error needs.

### Feature Flags

No conditional compilation used. Consider optional features for:
- `unshape-gpu` (heavyweight wgpu dependency)
- `unshape-gltf` (external format support)
- `unshape-image` (image processing)

## Refactoring Plan

1. **Phase 1 (HIGH):** ✅ Fixed tuple returns, replaced panics with Option
2. **Phase 2 (MEDIUM):** ✅ Deduplicated collision code, added missing Default traits
3. **Phase 3 (MEDIUM):** ✅ Refactored PhysicsWorld::step() into smaller methods

All planned refactoring complete. String-based tile IDs deferred as acceptable trade-off.
