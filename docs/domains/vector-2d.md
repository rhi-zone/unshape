# 2D Vector

2D vector graphics generation and manipulation.

## Prior Art

### SVG / PostScript / PDF
- **Paths**: moveTo, lineTo, curveTo, closePath
- **Cubic Béziers**: standard curve primitive
- **Stroke/fill**: separate outline and interior styling
- **Groups**: hierarchical transforms
- **Clipping**: mask regions

### Paper.js / Fabric.js
- **Path operations**: boolean (unite, subtract, intersect, exclude)
- **Compound paths**: paths with holes
- **Simplify**: reduce point count while preserving shape
- **Smooth**: convert corners to curves

### Figma / Sketch / Illustrator
- **Vector networks**: beyond simple paths (branches, multiple fills)
- **Boolean operations**: union, subtract, intersect, difference
- **Blend/morph**: interpolate between shapes
- **Offset/outline**: expand or contract paths

### Context Free Art
- **Shape grammars**: recursive shape rules
- **Randomness**: probabilistic variations
- **Transforms**: scale, rotate, translate in rules

### Potrace / Autotrace
- **Bitmap -> vector**: convert raster to paths
- **Corner detection**: where to place anchors
- **Curve fitting**: Bézier approximation of edges

## Core Types

```rust
/// A 2D path
struct Path {
    segments: Vec<Segment>,
    closed: bool,
}

enum Segment {
    MoveTo(Vec2),
    LineTo(Vec2),
    QuadTo { control: Vec2, end: Vec2 },
    CubicTo { control1: Vec2, control2: Vec2, end: Vec2 },
    ArcTo { radius: Vec2, rotation: f32, large: bool, sweep: bool, end: Vec2 },
    Close,
}

/// A shape with fill and stroke
struct Shape {
    path: Path,
    fill: Option<Fill>,
    stroke: Option<Stroke>,
}

/// Compound shape (with holes)
struct CompoundPath {
    paths: Vec<Path>,  // first is outer, rest are holes (by winding)
}

/// Fill style
enum Fill {
    Solid(Color),
    LinearGradient { start: Vec2, end: Vec2, stops: Vec<GradientStop> },
    RadialGradient { center: Vec2, radius: f32, stops: Vec<GradientStop> },
    Pattern { /* ... */ },
}

/// Stroke style
struct Stroke {
    color: Color,
    width: f32,
    cap: LineCap,    // Butt, Round, Square
    join: LineJoin,  // Miter, Round, Bevel
    dash: Option<Vec<f32>>,
}
```

## Primitives (Generators)

| Primitive | Parameters | Notes |
|-----------|------------|-------|
| Rectangle | position, size, corner_radius | Rounded corners optional |
| Ellipse | center, radii | Circle if radii equal |
| Polygon | center, radius, sides | Regular polygon |
| Star | center, outer_r, inner_r, points | |
| Line | start, end | Open path |
| Arc | center, radius, start_angle, end_angle | |
| Spiral | center, start_r, end_r, turns | |
| Text | string, font, size | Outline only |

## Operations

### Boolean
- **Union**: combine shapes
- **Difference**: subtract B from A
- **Intersection**: overlap only
- **Exclusion/XOR**: non-overlapping regions

### Path Manipulation
- **Offset**: parallel curve at distance (inward or outward)
- **Outline stroke**: convert stroke to filled path
- **Simplify**: reduce anchor count
- **Smooth**: convert corners to curves
- **Flatten**: convert curves to line segments
- **Reverse**: flip path direction (affects winding)

### Transform
- **Translate**: move
- **Rotate**: around point
- **Scale**: uniform or non-uniform
- **Skew**: shear
- **Transform matrix**: arbitrary 2D affine

### Path Analysis
- **Bounds**: bounding box
- **Length**: arc length
- **Point at**: position at t along path
- **Tangent at**: direction at t
- **Contains**: point-in-path test
- **Intersections**: path-path intersection points

### Generation
- **From points**: fit path through points
- **From bitmap**: trace raster image
- **L-system**: grammar-based generation

## Data Flow Pattern

```
Generator -> Operation -> Operation -> ... -> Output
                ↑
                other path(s) for booleans
```

Similar to meshes: paths are discrete objects transformed by operations.

### Hierarchical Structure

```rust
struct Group {
    children: Vec<Node>,
    transform: Transform2D,
    opacity: f32,
    clip: Option<Path>,
}

enum Node {
    Shape(Shape),
    Group(Group),
    // Text, Image, etc.
}
```

Scene graph / tree structure common for complex illustrations.

## Rasterization

Converting vector to pixels:

```rust
trait Rasterizer {
    fn rasterize(&self, shape: &Shape, width: u32, height: u32) -> Image;
}
```

Options: CPU (tiny-skia), GPU (lyon + wgpu), external (cairo, skia).

## Relationship to Meshes

2D paths can be:
- **Extruded** -> 3D mesh
- **Revolved** -> 3D mesh
- **Triangulated** -> 2D mesh (for rendering, physics)

Shared concepts with meshes:
- Topology (vertices, edges)
- Boolean operations
- Transforms

## Open Questions

1. **Curve representation**: Only cubic Bézier? Or also quadratic, arcs, NURBS? SVG has all, but cubic is most common.

2. **Precision**: f32 sufficient? CAD tools use f64. Games use f32.

3. **Winding rule**: Even-odd vs non-zero? Both? Affects compound paths.

4. **Vector networks**: Figma's model allows branches at points. Much more complex than paths. Worth it?

5. **Animation**: Paths are often animated (morphing). How does this relate to rigging?

6. **Text**: Text outlines are paths, but text layout is complex. Include or exclude from scope?
