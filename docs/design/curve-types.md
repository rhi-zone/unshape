# Curve Types: Trait-Based Design

Evaluating whether traits can elegantly support multiple curve types (cubic Bézier, quadratic, arcs, NURBS) without "more code paths in every operation."

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

```rust
trait Curve {
    /// Point at parameter t ∈ [0, 1]
    fn point_at(&self, t: f32) -> Vec2;

    /// Tangent vector at t
    fn tangent_at(&self, t: f32) -> Vec2;

    /// Approximate arc length
    fn length(&self) -> f32 {
        // Default: numerical integration
        // Override for closed-form when available
        self.length_adaptive(1e-4)
    }

    /// Bounding box
    fn bounding_box(&self) -> Rect;

    /// Split at t
    fn subdivide(&self, t: f32) -> (Self, Self) where Self: Sized;

    /// Convert to cubic Bézier(s)
    fn to_cubics(&self) -> Vec<CubicBezier>;

    /// Sample points for rendering
    fn flatten(&self, tolerance: f32) -> Vec<Vec2> {
        // Default: adaptive subdivision
        // Uses point_at and tangent_at
    }
}
```

Now each curve type implements the trait:

```rust
impl Curve for CubicBezier {
    fn point_at(&self, t: f32) -> Vec2 {
        // De Casteljau or direct formula
    }
    fn tangent_at(&self, t: f32) -> Vec2 { /* ... */ }
    fn bounding_box(&self) -> Rect { /* ... */ }
    fn subdivide(&self, t: f32) -> (Self, Self) { /* ... */ }
    fn to_cubics(&self) -> Vec<CubicBezier> { vec![*self] }
}

impl Curve for Arc {
    fn point_at(&self, t: f32) -> Vec2 {
        // Parametric circle
    }
    fn to_cubics(&self) -> Vec<CubicBezier> {
        // Approximate with 1-4 cubics
    }
    // ...
}

impl Curve for QuadBezier {
    fn to_cubics(&self) -> Vec<CubicBezier> {
        // Degree elevation (exact)
        vec![self.elevate_to_cubic()]
    }
    // ...
}
```

## Where Traits Work Well

### 1. Operations that are inherently per-curve

Each curve type has its own math. Trait methods encapsulate this:

```rust
fn render_path<C: Curve>(path: &[C], tolerance: f32) -> Vec<Vec2> {
    path.iter().flat_map(|c| c.flatten(tolerance)).collect()
}
```

### 2. Algorithms that only need the trait interface

```rust
fn path_length<C: Curve>(path: &[C]) -> f32 {
    path.iter().map(|c| c.length()).sum()
}

fn point_on_path<C: Curve>(path: &[C], t: f32) -> Vec2 {
    // Find which segment, call point_at
}
```

### 3. Mixed curve types via trait objects or enum

```rust
// Trait object (dynamic dispatch)
type DynPath = Vec<Box<dyn Curve>>;

// Or enum with trait impl (static dispatch, one match per method)
enum AnyCurve {
    Cubic(CubicBezier),
    Quad(QuadBezier),
    Arc(Arc),
}

impl Curve for AnyCurve {
    fn point_at(&self, t: f32) -> Vec2 {
        match self {
            Self::Cubic(c) => c.point_at(t),
            Self::Quad(q) => q.point_at(t),
            Self::Arc(a) => a.point_at(t),
        }
    }
    // ...
}
```

The enum still has matches, but they're in ONE place (the trait impl), not scattered across every operation.

## Where Traits Have Friction

### 1. Operations between different curve types

Intersection of Arc with CubicBezier:

```rust
fn intersect<A: Curve, B: Curve>(a: &A, b: &B) -> Vec<(f32, f32)> {
    // Generic numerical method works, but...
    // Could be faster with specialized arc-arc, line-line, etc.
}
```

Solution: provide generic default, allow specialization via separate functions or feature detection.

### 2. Subdivision returns Self

```rust
trait Curve {
    fn subdivide(&self, t: f32) -> (Self, Self) where Self: Sized;
}
```

Works fine for concrete types. For trait objects:

```rust
trait Curve {
    fn subdivide_boxed(&self, t: f32) -> (Box<dyn Curve>, Box<dyn Curve>);
}
```

Or return the same concrete type via associated type.

### 3. Binary operations need same type or conversion

Boolean operations on paths (union, difference) typically require same curve type or conversion to common format:

```rust
fn boolean_union<C: Curve>(a: &Path<C>, b: &Path<C>) -> Path<C>;

// Or convert to cubic first:
fn boolean_union_any(a: &dyn Curve, b: &dyn Curve) -> Path<CubicBezier> {
    let a_cubic = a.to_cubics();
    let b_cubic = b.to_cubics();
    boolean_union(&a_cubic, &b_cubic)
}
```

## Recommended Design

```rust
// Core trait
trait Curve: Clone {
    fn point_at(&self, t: f32) -> Vec2;
    fn tangent_at(&self, t: f32) -> Vec2;
    fn bounding_box(&self) -> Rect;
    fn to_cubics(&self) -> Vec<CubicBezier>;

    // Default impls using above
    fn length(&self) -> f32 { /* adaptive integration */ }
    fn flatten(&self, tolerance: f32) -> Vec<Vec2> { /* adaptive subdivision */ }
    fn subdivide(&self, t: f32) -> (Self, Self) where Self: Sized;
}

// Concrete types
struct Line { start: Vec2, end: Vec2 }
struct QuadBezier { start: Vec2, control: Vec2, end: Vec2 }
struct CubicBezier { start: Vec2, c1: Vec2, c2: Vec2, end: Vec2 }
struct Arc { center: Vec2, radii: Vec2, start: f32, sweep: f32 }

// Enum for mixed paths (single match point per operation)
enum Segment {
    Line(Line),
    Quad(QuadBezier),
    Cubic(CubicBezier),
    Arc(Arc),
}

impl Curve for Segment {
    fn point_at(&self, t: f32) -> Vec2 {
        match self {
            Self::Line(l) => l.point_at(t),
            Self::Quad(q) => q.point_at(t),
            Self::Cubic(c) => c.point_at(t),
            Self::Arc(a) => a.point_at(t),
        }
    }
    // etc.
}

// Path generic over segment type
struct Path<C: Curve = Segment> {
    segments: Vec<C>,
    closed: bool,
}
```

## Conclusion

**Traits DO solve the "code paths everywhere" problem:**

| Without traits | With traits |
|----------------|-------------|
| Match in every function | Match in one place (trait impl) |
| N functions × M types = N×M matches | M trait impls |
| Hard to add new curve types | Just impl Curve for new type |

**Recommendation:**
- Use traits for curve operations
- Provide concrete types (CubicBezier, Arc, etc.)
- Provide enum wrapper (Segment) for mixed paths
- Use `to_cubics()` as escape hatch for operations that need uniform type
