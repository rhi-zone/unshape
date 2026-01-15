# Curve Types: Trait-Based Design

Evaluating whether traits can elegantly support multiple curve types (cubic Bézier, quadratic, arcs, NURBS) without "more code paths in every operation."

## Current State

Before designing the future, document what exists:

| Crate | Type | Dimension | Notes |
|-------|------|-----------|-------|
| resin-vector | `Path`, `PathCommand` | 2D only | SVG-like command enum |
| resin-vector | `CurveSegment` (stroke.rs) | 2D only | Line/Quad/Cubic enum |
| resin-vector | `bezier.rs` functions | 2D only | `cubic_point()`, `cubic_tangent()`, etc. |
| resin-spline | `CubicBezier<T>` | Generic | Via `Interpolatable` trait |
| resin-spline | `BSpline<T>`, `Nurbs<T>` | Generic | Full spline support |
| resin-rig | `Path3D`, `PathCommand3D` | 3D only | Arc-length parameterized |

**Existing generics:** resin-spline uses `Interpolatable` trait:

```rust
pub trait Interpolatable: Clone + Copy + Add + Sub + Mul<f32> {}
impl Interpolatable for f32 {}
impl Interpolatable for Vec2 {}
impl Interpolatable for Vec3 {}
```

**Gap:** `Interpolatable` lacks `length()` method needed for arc length calculation. The `Curve` trait may need additional bounds or a separate `VectorSpace` trait.

**Existing segment enum:** stroke.rs has:

```rust
pub enum CurveSegment {
    Line { start: Vec2, end: Vec2 },
    Quadratic { start: Vec2, control: Vec2, end: Vec2 },
    Cubic { start: Vec2, control1: Vec2, control2: Vec2, end: Vec2 },
}
```

**Key insight:** The generic infrastructure exists in resin-spline. The gap is that resin-vector is 2D-only and resin-rig reimplements 3D separately.

## The Concern

Supporting multiple curve types naively:

```rust
enum Segment {
    Line(Vec2, Vec2),
    QuadBezier { start: Vec2, control: Vec2, end: Vec2 },
    CubicBezier { start: Vec2, c1: Vec2, c2: Vec2, end: Vec2 },
    Arc { center: Vec2, radius: Vec2, start_angle: f32, end_angle: f32 },
    // NURBS...
}

fn point_at(seg: &Segment, t: f32) -> Vec2 {
    match seg {
        Segment::Line(..) => { /* impl */ }
        Segment::QuadBezier { .. } => { /* impl */ }
        Segment::CubicBezier { .. } => { /* impl */ }
        Segment::Arc { .. } => { /* impl */ }
    }
}

// Every operation needs this match...
fn tangent_at(seg: &Segment, t: f32) -> Vec2 { /* match... */ }
fn length(seg: &Segment) -> f32 { /* match... */ }
fn bounding_box(seg: &Segment) -> Rect { /* match... */ }
fn subdivide(seg: &Segment, t: f32) -> (Segment, Segment) { /* match... */ }
```

This is the "more code paths" problem.

## Trait-Based Approach

Using an associated type for 2D/3D genericity:

```rust
/// Unified curve interface for any dimension.
pub trait Curve: Clone {
    /// Point type: Vec2, Vec3, or any Interpolatable
    type Point: Interpolatable;

    /// Point at parameter t ∈ [0, 1]
    fn position_at(&self, t: f32) -> Self::Point;

    /// Tangent vector at t (not normalized)
    fn tangent_at(&self, t: f32) -> Self::Point;

    /// Split curve at parameter t, returning (before, after)
    fn split(&self, t: f32) -> (Self, Self) where Self: Sized;

    /// Start point (equivalent to position_at(0.0))
    fn start(&self) -> Self::Point { self.position_at(0.0) }

    /// End point (equivalent to position_at(1.0))
    fn end(&self) -> Self::Point { self.position_at(1.0) }

    /// Approximate arc length (default: numerical integration)
    fn length(&self) -> f32 {
        // Gaussian quadrature or adaptive subdivision
        // Override for closed-form when available (e.g., lines)
    }

    /// Sample points for rendering (adaptive subdivision)
    fn flatten(&self, tolerance: f32) -> Vec<Self::Point> {
        // Default: recursive subdivision until flat enough
    }

    /// Convert to cubic Bézier approximation(s)
    /// Some curves (arcs, NURBS) may produce multiple cubics
    fn to_cubics(&self) -> Vec<CubicBezier<Self::Point>>;
}

// Note: bounding_box() is NOT in trait - it requires dimension-specific
// Bounds2D vs Bounds3D return types. Use extension traits or free functions.
```

Now each curve type implements the trait:

```rust
impl<V: Interpolatable> Curve for CubicBezier<V> {
    type Point = V;

    fn position_at(&self, t: f32) -> V {
        // De Casteljau or direct formula
        let t2 = t * t;
        let t3 = t2 * t;
        let mt = 1.0 - t;
        let mt2 = mt * mt;
        let mt3 = mt2 * mt;
        self.p0 * mt3 + self.p1 * (3.0 * mt2 * t)
            + self.p2 * (3.0 * mt * t2) + self.p3 * t3
    }

    fn tangent_at(&self, t: f32) -> V { /* derivative of above */ }

    fn split(&self, t: f32) -> (Self, Self) {
        // De Casteljau subdivision (already in resin-spline)
    }

    fn to_cubics(&self) -> Vec<CubicBezier<V>> { vec![self.clone()] }
}

impl<V: Interpolatable> Curve for QuadBezier<V> {
    type Point = V;

    fn position_at(&self, t: f32) -> V {
        let mt = 1.0 - t;
        self.p0 * (mt * mt) + self.p1 * (2.0 * mt * t) + self.p2 * (t * t)
    }

    fn to_cubics(&self) -> Vec<CubicBezier<V>> {
        // Degree elevation (exact conversion)
        vec![self.elevate()]
    }
    // ...
}

impl<V: Interpolatable> Curve for Line<V> {
    type Point = V;

    fn position_at(&self, t: f32) -> V {
        self.start.lerp_to(&self.end, t)
    }

    fn length(&self) -> f32 {
        // Override: closed-form for lines
        (self.end - self.start).length()
    }

    fn to_cubics(&self) -> Vec<CubicBezier<V>> {
        // Degenerate cubic with collinear control points
        vec![CubicBezier {
            p0: self.start,
            p1: self.start.lerp_to(&self.end, 1.0/3.0),
            p2: self.start.lerp_to(&self.end, 2.0/3.0),
            p3: self.end,
        }]
    }
}
```

### Arc Representation

Arcs are 2D-specific (3D "arcs" are better represented as NURBS):

```rust
/// 2D elliptical arc
pub struct Arc {
    pub center: Vec2,
    pub radii: Vec2,      // (rx, ry) for ellipse, (r, r) for circle
    pub start_angle: f32,
    pub sweep: f32,       // positive = CCW, negative = CW
    pub rotation: f32,    // x-axis rotation for ellipse
}

impl Curve for Arc {
    type Point = Vec2;

    fn position_at(&self, t: f32) -> Vec2 {
        let angle = self.start_angle + self.sweep * t;
        let p = Vec2::new(self.radii.x * angle.cos(), self.radii.y * angle.sin());
        // Apply rotation and translate to center
        rotate_2d(p, self.rotation) + self.center
    }

    fn to_cubics(&self) -> Vec<CubicBezier<Vec2>> {
        // Approximate with 1-4 cubics depending on sweep angle
        // Each cubic handles up to 90° accurately
        arc_to_cubics(self)
    }
}
```

**Alternative: NURBS for exact arcs.** resin-spline's `Nurbs<V>` can represent circles/arcs exactly via rational weights. For interchange formats (SVG, fonts), cubic approximation is standard. For internal precision, NURBS may be preferred. Design should support both.

## Where Traits Work Well

### 1. Operations that are inherently per-curve

Each curve type has its own math. Trait methods encapsulate this:

```rust
fn flatten_path<C: Curve>(path: &[C], tolerance: f32) -> Vec<C::Point> {
    path.iter().flat_map(|c| c.flatten(tolerance)).collect()
}
```

### 2. Algorithms that only need the trait interface

```rust
fn path_length<C: Curve>(path: &[C]) -> f32 {
    path.iter().map(|c| c.length()).sum()
}

fn position_on_path<C: Curve>(path: &[C], t: f32) -> C::Point {
    // Find which segment based on t, call position_at
    let segment_count = path.len();
    let scaled = t * segment_count as f32;
    let index = (scaled.floor() as usize).min(segment_count - 1);
    let local_t = scaled.fract();
    path[index].position_at(local_t)
}
```

### 3. Mixed curve types via enum

The enum approach with trait impl gives static dispatch with one match per method:

```rust
/// 2D segment types (matches existing CurveSegment, extended)
pub enum Segment2D {
    Line(Line<Vec2>),
    Quad(QuadBezier<Vec2>),
    Cubic(CubicBezier<Vec2>),
    Arc(Arc),
}

/// 3D segment types (no Arc - use NURBS for 3D curves)
pub enum Segment3D {
    Line(Line<Vec3>),
    Quad(QuadBezier<Vec3>),
    Cubic(CubicBezier<Vec3>),
}

impl Curve for Segment2D {
    type Point = Vec2;

    fn position_at(&self, t: f32) -> Vec2 {
        match self {
            Self::Line(l) => l.position_at(t),
            Self::Quad(q) => q.position_at(t),
            Self::Cubic(c) => c.position_at(t),
            Self::Arc(a) => a.position_at(t),
        }
    }
    // ... other methods delegate similarly
}

impl Curve for Segment3D {
    type Point = Vec3;

    fn position_at(&self, t: f32) -> Vec3 {
        match self {
            Self::Line(l) => l.position_at(t),
            Self::Quad(q) => q.position_at(t),
            Self::Cubic(c) => c.position_at(t),
        }
    }
}
```

The enum still has matches, but they're in ONE place (the trait impl), not scattered across every operation.

**Trait objects?** Possible via `Box<dyn Curve<Point = Vec2>>` but associated types make this awkward. Enums are preferred for known, finite curve types.

## Where Traits Have Friction

### 1. Operations between different curve types

Intersection of Arc with CubicBezier:

```rust
fn intersect<C: Curve>(a: &C, b: &C) -> Vec<CurveIntersection> {
    // Generic numerical method works (subdivision + Newton refinement)
    // Could be faster with specialized arc-arc, line-line, etc.
}

// Or with explicit same-point constraint:
fn intersect_curves<A, B, V>(a: &A, b: &B) -> Vec<CurveIntersection>
where
    A: Curve<Point = V>,
    B: Curve<Point = V>,
    V: Interpolatable,
{
    // ...
}
```

Solution: provide generic default, allow specialization via separate functions:

```rust
// Generic (works for any curve pair)
fn curve_intersections<A: Curve, B: Curve>(a: &A, b: &B) -> Vec<CurveIntersection>;

// Specialized (faster for specific pairs, optional)
fn line_line_intersection(a: &Line<Vec2>, b: &Line<Vec2>) -> Option<CurveIntersection>;
fn arc_arc_intersections(a: &Arc, b: &Arc) -> Vec<CurveIntersection>;
```

### 2. Split returns Self

```rust
trait Curve {
    fn split(&self, t: f32) -> (Self, Self) where Self: Sized;
}
```

Works fine for concrete types and enums. The `Sized` bound prevents trait objects from using this method directly, which is acceptable - use enums instead.

### 3. Binary operations need same dimension

Boolean operations on paths (union, difference) require same point type:

```rust
fn boolean_union<C: Curve>(a: &Path<C>, b: &Path<C>) -> Path<C>
where
    C::Point: /* 2D-specific bounds for boolean ops */;

// Or use to_cubics() as escape hatch for mixed input:
fn boolean_union_mixed(a: &[impl Curve<Point = Vec2>], b: &[impl Curve<Point = Vec2>])
    -> Path<CubicBezier<Vec2>>
{
    let a_cubic: Vec<_> = a.iter().flat_map(|c| c.to_cubics()).collect();
    let b_cubic: Vec<_> = b.iter().flat_map(|c| c.to_cubics()).collect();
    boolean_union_cubics(&a_cubic, &b_cubic)
}
```

### 4. Bounding box return type

Different dimensions need different bounds types:

```rust
// Can't be in trait due to different return types
fn bounds_2d(curve: &impl Curve<Point = Vec2>) -> Rect;
fn bounds_3d(curve: &impl Curve<Point = Vec3>) -> Aabb;

// Or use extension traits:
trait Curve2DExt: Curve<Point = Vec2> {
    fn bounding_box(&self) -> Rect;
}

trait Curve3DExt: Curve<Point = Vec3> {
    fn bounding_box(&self) -> Aabb;
}
```

## Recommended Design

```rust
// ═══════════════════════════════════════════════════════════════════════════
// Core Trait (in resin-vector or resin-curve)
// ═══════════════════════════════════════════════════════════════════════════

/// Unified curve interface for any dimension.
pub trait Curve: Clone {
    type Point: Interpolatable;

    fn position_at(&self, t: f32) -> Self::Point;
    fn tangent_at(&self, t: f32) -> Self::Point;
    fn split(&self, t: f32) -> (Self, Self) where Self: Sized;
    fn to_cubics(&self) -> Vec<CubicBezier<Self::Point>>;

    // Default implementations
    fn start(&self) -> Self::Point { self.position_at(0.0) }
    fn end(&self) -> Self::Point { self.position_at(1.0) }
    fn length(&self) -> f32 { /* Gaussian quadrature */ }
    fn flatten(&self, tolerance: f32) -> Vec<Self::Point> { /* adaptive */ }
}

// ═══════════════════════════════════════════════════════════════════════════
// Concrete Types (generic over point type)
// ═══════════════════════════════════════════════════════════════════════════

/// Line segment
pub struct Line<V> { pub start: V, pub end: V }

/// Quadratic Bézier curve
pub struct QuadBezier<V> { pub p0: V, pub p1: V, pub p2: V }

/// Cubic Bézier curve (already exists in resin-spline)
pub struct CubicBezier<V> { pub p0: V, pub p1: V, pub p2: V, pub p3: V }

/// Elliptical arc (2D only)
pub struct Arc {
    pub center: Vec2,
    pub radii: Vec2,
    pub start_angle: f32,
    pub sweep: f32,
    pub rotation: f32,
}

// ═══════════════════════════════════════════════════════════════════════════
// Segment Enums (for mixed-type paths)
// ═══════════════════════════════════════════════════════════════════════════

/// 2D segment (extends existing CurveSegment)
pub enum Segment2D {
    Line(Line<Vec2>),
    Quad(QuadBezier<Vec2>),
    Cubic(CubicBezier<Vec2>),
    Arc(Arc),
}

/// 3D segment (no arc - use NURBS for 3D curves)
pub enum Segment3D {
    Line(Line<Vec3>),
    Quad(QuadBezier<Vec3>),
    Cubic(CubicBezier<Vec3>),
}

impl Curve for Segment2D {
    type Point = Vec2;
    // ... delegate to inner types
}

impl Curve for Segment3D {
    type Point = Vec3;
    // ... delegate to inner types
}

// ═══════════════════════════════════════════════════════════════════════════
// Path (generic over curve type)
// ═══════════════════════════════════════════════════════════════════════════

/// A sequence of connected curves
pub struct Path<C: Curve = Segment2D> {
    pub segments: Vec<C>,
    pub closed: bool,
}

// Type aliases for common cases
pub type Path2D = Path<Segment2D>;
pub type Path3D = Path<Segment3D>;
pub type CubicPath2D = Path<CubicBezier<Vec2>>;
pub type CubicPath3D = Path<CubicBezier<Vec3>>;

// ═══════════════════════════════════════════════════════════════════════════
// Extension Traits (dimension-specific operations)
// ═══════════════════════════════════════════════════════════════════════════

pub trait Curve2DExt: Curve<Point = Vec2> {
    fn bounding_box(&self) -> Rect;
    fn offset(&self, distance: f32) -> Vec<CubicBezier<Vec2>>;
}

pub trait Curve3DExt: Curve<Point = Vec3> {
    fn bounding_box(&self) -> Aabb;
}
```

## Conclusion

**Traits DO solve the "code paths everywhere" problem:**

| Without traits | With traits |
|----------------|-------------|
| Match in every function | Match in one place (trait impl) |
| N functions × M types = N×M matches | M trait impls |
| Hard to add new curve types | Just impl Curve for new type |
| 2D and 3D separate | Generic over `Point` type |

**Recommendation:**
- Use traits for curve operations
- Generic over `Interpolatable` point type (works for Vec2, Vec3, f32)
- Provide concrete types (`Line<V>`, `CubicBezier<V>`, etc.)
- Provide segment enums (`Segment2D`, `Segment3D`) for mixed paths
- Use extension traits for dimension-specific operations (bounding box, offset)
- Use `to_cubics()` as escape hatch for operations that need uniform type

## Migration Path

### Phase 1: Add Trait, Implement for Existing Types

1. **Add `Curve` trait to resin-vector** (or create resin-curve crate)
2. **Implement for resin-spline types:**
   - `CubicBezier<V>` already has `evaluate()` → rename to `position_at()`
   - `BezierSpline<V>`, `CatmullRom<V>`, `BSpline<V>`, `Nurbs<V>`
3. **Add missing concrete types:**
   - `Line<V>` (trivial)
   - `QuadBezier<V>` (extract from bezier.rs)
   - `Arc` (2D only, from existing arc_to_cubic code)
4. **Test:** Existing code unaffected, new trait available

### Phase 2: Add Segment Enums

1. **Create `Segment2D` enum** (similar to existing `CurveSegment`)
2. **Create `Segment3D` enum**
3. **Implement `Curve` for both enums**
4. **Add extension traits** (`Curve2DExt`, `Curve3DExt`)

### Phase 3: Unify Path Types

1. **Make `Path<C>` generic** with default `C = Segment2D`
2. **Delete `resin-rig::Path3D`** → replace with `Path<Segment3D>`
3. **Update path operations** to use trait bounds instead of concrete types

### Phase 4: Migrate Consumers

1. **Update resin-vector operations** (offset, boolean, stroke) to use trait
2. **Update resin-rig path constraint** to use generic path
3. **Consolidate bezier.rs functions** into `CubicBezier<Vec2>` methods
4. **Remove duplicate implementations**
5. **Delete `PathCommand` enum** - replace with direct `Segment2D` construction

## Open Questions

1. **Crate location:** Should `Curve` trait live in:
   - `resin-vector` (current 2D home)?
   - `resin-spline` (already has generic bezier)?
   - New `resin-curve` crate (clean separation)?

2. **NURBS integration:** Should `Nurbs<V>` implement `Curve`? It's already in resin-spline and can represent arcs exactly.

3. **Arc-length parameterization:** Path3D has valuable arc-length caching. Should this be:
   - Part of `Path<C>` (always available)?
   - A separate wrapper `ArcLengthPath<C>`?
   - A method `path.with_arc_length_cache()`?

4. **Method naming:** Current code uses mix of:
   - `evaluate(t)` (resin-spline)
   - `position_at(t)` (resin-rig)
   - `point_at(t)` (proposed)

   Recommendation: `position_at()` - matches resin-rig's arc-length API and is clear.

5. **Interpolatable bounds:** The current `Interpolatable` trait lacks `length()` needed for arc length. Options:
   - Extend `Interpolatable` to include `fn length(&self) -> f32`
   - Create separate `VectorSpace` trait with length/normalize
   - Use concrete bounds like `Curve<Point: HasLength>` where needed
   - Provide default `length()` implementations only for known types via specialization
