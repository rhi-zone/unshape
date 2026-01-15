# Type Unification Analysis

This document analyzes opportunities for type unification across the resin codebase - cases where multiple types represent overlapping or subset concepts that could benefit from a unified abstraction.

## Summary

| Domain | Issue | Priority | Design Doc? |
|--------|-------|----------|-------------|
| **Curves/Paths** | Fragmented 2D/3D, mixed function/struct APIs | HIGH | Yes: `curve-types.md` |
| **Graphs** | "Node"/"Edge" overloaded across domains | MEDIUM | No |
| **Transforms** | ~~Separate 2D/3D types~~ | ~~MEDIUM~~ | ✅ Done - `SpatialTransform` trait |
| **Vertex Data** | Per-subsystem Vertex structs | LOW | No |
| **Mesh** | Two representations | NONE | Already unified correctly |
| **Fields** | Trait + implementations | NONE | Well-designed |

---

## HIGH PRIORITY

### 1. Curve/Path Representations

**Current state (fragmented):**

| Crate | Type | Purpose |
|-------|------|---------|
| resin-vector | `Path` + `PathCommand` | SVG-like 2D paths (MoveTo, LineTo, CubicTo, etc.) |
| resin-vector | `bezier.rs` functions | `quadratic_point()`, `cubic_point()`, etc. |
| resin-spline | `CubicBezier<T>` | Generic typed cubic bezier struct |
| resin-spline | `BezierSpline<T>` | Sequence of beziers |
| resin-spline | `CatmullRom<T>`, `BSpline<T>`, `Nurbs<T>` | Other spline types |
| resin-rig | `Path3D` + `PathCommand3D` | 3D version of Path (nearly identical structure) |

**Problems:**

1. `Path` (2D) and `Path3D` are nearly identical but separate implementations
2. `CubicBezier<T>` in resin-spline is a struct, but resin-vector has function-based API
3. Bezier math is implemented twice (functions vs struct methods)
4. No unified interface between linear paths and curved splines
5. Operations that work on 2D don't automatically work on 3D

**Recommended solution:**

See `docs/design/curve-types.md` for detailed design. Summary:

```rust
// Core trait (in resin-vector or new resin-curve)
pub trait Curve: Clone {
    type Point;  // Vec2 or Vec3

    fn point_at(&self, t: f32) -> Self::Point;
    fn tangent_at(&self, t: f32) -> Self::Point;
    fn bounding_box(&self) -> Bounds<Self::Point>;
    fn to_cubics(&self) -> Vec<CubicBezier<Self::Point>>;

    // Default implementations
    fn length(&self) -> f32 { /* adaptive integration */ }
    fn flatten(&self, tolerance: f32) -> Vec<Self::Point> { /* subdivision */ }
}

// Concrete types implement trait
impl Curve for Line<Vec2> { ... }
impl Curve for CubicBezier<Vec2> { ... }
impl Curve for Arc { ... }

// Enum for mixed paths (single match point per method)
pub enum Segment<V> {
    Line(Line<V>),
    Quad(QuadBezier<V>),
    Cubic(CubicBezier<V>),
    Arc(Arc<V>),  // 2D only, or generalize
}

impl<V> Curve for Segment<V> where ... { ... }

// Path generic over segment type
pub struct Path<C: Curve = Segment<Vec2>> {
    segments: Vec<C>,
    closed: bool,
}
```

**Migration path:**

1. Create `Curve` trait in resin-vector (or new resin-curve crate)
2. Implement for existing types (`CubicBezier`, etc.)
3. Create `Segment<V>` enum with trait impl
4. Make `Path<C>` generic, default to `Segment<Vec2>`
5. Deprecate `Path3D`, replace with `Path<Segment<Vec3>>`
6. Move/consolidate bezier functions into `CubicBezier` impl

---

## MEDIUM PRIORITY

### 2. Graph/Node/Edge Terminology

**Current state (overloaded terms):**

| Crate | "Node" means | "Edge" means |
|-------|--------------|--------------|
| resin-core | Data flow node (has ports) | Port-to-port connection |
| resin-vector | Spatial position in network | Bezier curve between positions |
| resin-procgen | Spatial position (roads/rivers) | Connection between positions |
| resin-mesh | (uses "Vertex") | Pair of vertex indices |
| resin-mesh/halfedge | (uses "Vertex") | Half-edge (directional) |
| resin-space-colonization | Branch node (3D position) | Parent-child connection |

**Problems:**

1. "Node" means completely different things (data processing vs spatial position)
2. "Edge" means different things (port connection vs geometric curve vs index pair)
3. Multiple `NodeId` / `VertexId` / `EdgeId` types that aren't interchangeable
4. Code that says "node" is ambiguous without context

**Recommended solution:**

Establish clear terminology domains:

```
Data Flow Domain (resin-core):
  - GraphNode: processing node with typed inputs/outputs
  - Port: input or output slot on a node
  - Wire: connection from output port to input port

Spatial Domain (resin-vector, resin-procgen):
  - SpatialNode / Waypoint: position in space
  - SpatialEdge / Link: connection between positions (may carry geometry)

Topology Domain (resin-mesh):
  - Vertex: position in mesh
  - HalfEdge: directional edge (half-edge data structure)
  - Face: polygon bounded by edges
```

**Migration approach:**

1. Document terminology in `docs/conventions.md`
2. Consider renaming in resin-vector: `Node` → `Waypoint` or `NetworkNode`
3. Consider renaming in resin-core: `Edge` → `Wire` or `Connection`
4. Add type aliases for backwards compatibility
5. Update documentation to use consistent terms

### 3. Transform Types

**Status: ✅ Complete**

Added `SpatialTransform` trait in `resin-transform` crate with implementations in `resin-rig` (Transform) and `resin-motion` (Transform2D).

```rust
pub trait SpatialTransform {
    type Vector: Copy;   // Vec2 or Vec3
    type Rotation: Copy; // f32 or Quat
    type Matrix: Copy;   // Mat3 or Mat4

    fn translation(&self) -> Self::Vector;
    fn rotation(&self) -> Self::Rotation;
    fn scale(&self) -> Self::Vector;
    fn to_matrix(&self) -> Self::Matrix;
    fn transform_point(&self, point: Self::Vector) -> Self::Vector;
}
```

**Implementation:**
- `Transform3D`: `Vector=Vec3`, `Rotation=Quat`, `Matrix=Mat4`
- `Transform2D`: `Vector=Vec2`, `Rotation=f32`, `Matrix=Mat3`

This enables generic algorithms over transforms while preserving domain-specific features (2D anchor/skew, 3D quaternion rotation).

---

## LOW PRIORITY

### 4. Vertex Attribute Types

**Current state:**

| Crate | Type | Fields |
|-------|------|--------|
| resin-mesh | `Vertex` (in halfedge) | position, normal, uv |
| resin-mesh | `VertexWeights` | bone indices, weights |
| resin-rig | `VertexInfluences` | bone/weight pairs |
| resin-vector | gradient mesh vertex | position, color |
| resin-physics | soft body vertex | position, velocity, mass |

**Problems:**

1. Each subsystem defines its own vertex struct
2. Not using existing `Has*` traits from resin-core
3. Difficult to combine attributes (e.g., skinned + physics vertex)

**Recommended approach:**

Use composition and traits rather than monolithic structs:

```rust
// Core position data
pub struct VertexPosition {
    pub position: Vec3,
}

// Optional attributes as separate structs
pub struct VertexNormal {
    pub normal: Vec3,
}

pub struct VertexUv {
    pub uv: Vec2,
}

pub struct VertexSkin {
    pub bones: [u32; 4],
    pub weights: [f32; 4],
}

// Compose via tuples or wrapper
type SkinnedVertex = (VertexPosition, VertexNormal, VertexUv, VertexSkin);

// Or use traits for duck typing
pub trait HasPosition {
    fn position(&self) -> Vec3;
}
```

This is lower priority because the current approach works, just with some duplication.

---

## ALREADY WELL-DESIGNED

### Mesh Representations

**Status: Good example of unification done right**

| Type | Purpose |
|------|---------|
| `HalfEdgeMesh` | Topology-rich, for editing operations |
| `Mesh` | Indexed arrays, for GPU rendering |

Clear conversions exist:
- `HalfEdgeMesh::from_mesh(&mesh)` - convert for editing
- `halfedge.to_mesh()` - convert for rendering

This follows the "general-internal-constrained-api" pattern documented in `docs/design/general-internal-constrained-api.md`.

### Field System

**Status: Well-designed composition**

| Crate | Purpose |
|-------|---------|
| resin-field | `Field<I, O>` trait, combinators |
| resin-noise | Noise function implementations |
| resin-expr-field | Expression-based field building |

The trait-based design allows composition without type proliferation.

---

## Implementation Priority

1. ~~**Curves** (HIGH)~~ - ✅ Complete - `resin-curve` crate with `Curve` trait
2. **Graph terminology** (MEDIUM) - Documentation + gradual renaming
3. ~~**Transforms** (MEDIUM)~~ - ✅ Complete - `resin-transform` crate with `SpatialTransform` trait
4. **Vertex attributes** (LOW) - Works fine, optimize later

---

## Related Documents

- `curve-types.md` - Detailed curve trait design
- `general-internal-constrained-api.md` - Pattern for internal vs public types
- `normalization.md` - Code style consistency (completed)
- `conventions.md` - Coordinate systems and naming conventions
