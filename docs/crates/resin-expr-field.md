# resin-expr-field

Expression-based spatial fields with typed AST.

## Purpose

Bridges the dew expression language and the resin field system. Provides two ways to define spatial fields from expressions:

1. **ExprField** - Parse dew expressions at runtime, evaluate as `Field<Vec2/Vec3, f32>`
2. **FieldExpr** - Typed AST enum for UI introspection, JSON serialization, and GPU compilation

## Related Crates

- **resin-field** - Core `Field` trait that expression fields implement
- **resin-noise** - Noise functions registered for expression evaluation
- **resin-motion-fn** - Similar typed AST for motion (`MotionExpr`)
- **dew** - Expression parsing and evaluation

## ExprField (Runtime Expressions)

Parse and evaluate dew expressions as spatial fields:

```rust
use rhizome_resin_expr_field::{ExprField, register_noise, scalar_registry};
use rhizome_resin_field::{Field, EvalContext};

// Create registry with math + noise functions
let mut registry = scalar_registry();
register_noise(&mut registry);

// Parse expression
let field = ExprField::parse("sin(x * 3.14) + noise(x * 4, y * 4)", registry)?;

// Sample as a field
let ctx = EvalContext::new();
let value: f32 = field.sample(Vec2::new(0.5, 0.5), &ctx);
```

### Built-in Variables

Expressions automatically have access to:
- `x`, `y`, `z` - Spatial coordinates
- `t`, `time` - Time from EvalContext

### Registered Noise Functions

With `register_noise()`:
- `noise(x, y)` / `perlin(x, y)` - 2D Perlin noise
- `perlin3(x, y, z)` - 3D Perlin noise
- `simplex(x, y)` - 2D Simplex noise
- `simplex3(x, y, z)` - 3D Simplex noise
- `fbm(x, y, octaves)` - 2D fractional Brownian motion

### Variable Introspection

Discover what variables an expression needs:

```rust
let field = ExprField::parse("sin(t * speed) * amplitude + x", registry)?;

// All referenced variables
let free = field.free_vars(); // {"t", "speed", "amplitude", "x"}

// User-defined only (excludes builtins)
let inputs = field.user_inputs(); // {"speed", "amplitude"}
```

## FieldExpr (Typed AST)

For UI editors, serialization, and GPU compilation, use the typed AST:

```rust
use rhizome_resin_expr_field::FieldExpr;

// Build expression: perlin(x * 4, y * 4) + 0.5 * simplex(x * 8, y * 8)
let expr = FieldExpr::Add(
    Box::new(FieldExpr::Perlin2 {
        x: Box::new(FieldExpr::Mul(
            Box::new(FieldExpr::X),
            Box::new(FieldExpr::Constant(4.0)),
        )),
        y: Box::new(FieldExpr::Mul(
            Box::new(FieldExpr::Y),
            Box::new(FieldExpr::Constant(4.0)),
        )),
    }),
    Box::new(FieldExpr::Mul(
        Box::new(FieldExpr::Constant(0.5)),
        Box::new(FieldExpr::Simplex2 {
            x: Box::new(FieldExpr::Mul(Box::new(FieldExpr::X), Box::new(FieldExpr::Constant(8.0)))),
            y: Box::new(FieldExpr::Mul(Box::new(FieldExpr::Y), Box::new(FieldExpr::Constant(8.0)))),
        }),
    )),
);

// Evaluate
let value = expr.eval(0.5, 0.5, 0.0, 0.0, &HashMap::new());

// Use as Field
let ctx = EvalContext::new();
let v: f32 = Field::<Vec2, f32>::sample(&expr, Vec2::new(0.5, 0.5), &ctx);
```

### Available Variants

**Coordinates**: `X`, `Y`, `Z`, `T`

**Noise**:
- `Perlin2 { x, y }`, `Perlin3 { x, y, z }`
- `Simplex2 { x, y }`, `Simplex3 { x, y, z }`
- `Fbm2 { x, y, octaves }`, `Fbm3 { x, y, z, octaves }`

**SDF Primitives**:
- `SdfCircle { x, y, radius }`
- `SdfSphere { x, y, z, radius }`
- `SdfBox2 { x, y, half_width, half_height }`
- `SdfBox3 { x, y, z, half_x, half_y, half_z }`

**SDF Operations**:
- `SdfSmoothUnion { a, b, k }`
- `SdfSmoothIntersection { a, b, k }`
- `SdfSmoothSubtraction { a, b, k }`

**Distance/Length**:
- `Distance2 { x, y, px, py }`
- `Distance3 { x, y, z, px, py, pz }`
- `Length2 { x, y }`, `Length3 { x, y, z }`

**Math**: `Sin`, `Cos`, `Tan`, `Abs`, `Floor`, `Ceil`, `Fract`, `Sqrt`, `Exp`, `Ln`, `Sign`, `Min`, `Max`, `Clamp`, `Lerp`, `SmoothStep`, `Step`

**Conditionals**: `IfThenElse`, `Gt`, `Lt`, `Eq`

### Converting to Dew AST

For GPU compilation via dew's WGSL/Cranelift backends:

```rust
let dew_ast = expr.to_dew_ast();
// Can now compile to WGSL or native code
```

### Serialization

```rust
#[cfg(feature = "serde")]
{
    let json = serde_json::to_string(&expr)?;
    let loaded: FieldExpr = serde_json::from_str(&json)?;
}
```

## Use Cases

### Procedural Texture Definition

Define textures as expressions for artist-friendly editing:

```rust
// User edits expression string in UI
let expr_str = "fbm(x * freq, y * freq, 4) * contrast + 0.5";
let field = ExprField::parse(expr_str, registry)?;

// Discover what parameters to expose
let params = field.user_inputs(); // {"freq", "contrast"}

// Evaluate with user-provided values
let mut vars = HashMap::new();
vars.insert("freq".into(), 8.0);
vars.insert("contrast".into(), 0.8);
let value = field.eval(&vars)?;
```

### SDF Modeling

Build signed distance fields with smooth blending:

```rust
let sphere = FieldExpr::SdfSphere {
    x: Box::new(FieldExpr::X),
    y: Box::new(FieldExpr::Y),
    z: Box::new(FieldExpr::Z),
    radius: 1.0,
};

let box_sdf = FieldExpr::SdfBox3 {
    x: Box::new(FieldExpr::X),
    y: Box::new(FieldExpr::Y),
    z: Box::new(FieldExpr::Z),
    half_x: 0.8,
    half_y: 0.8,
    half_z: 0.8,
};

let blended = FieldExpr::SdfSmoothSubtraction {
    a: Box::new(sphere),
    b: Box::new(box_sdf),
    k: 0.2,
};
```

### Node-Based Editor Backend

Pattern match on FieldExpr to render UI controls:

```rust
fn render_node(expr: &FieldExpr) -> NodeWidget {
    match expr {
        FieldExpr::Perlin2 { .. } => perlin_node_widget(),
        FieldExpr::SdfCircle { radius, .. } => {
            circle_node_widget(*radius)
        }
        FieldExpr::Add(a, b) => {
            binary_op_widget("Add", render_node(a), render_node(b))
        }
        // ... etc
    }
}
```

## Compositions

### With resin-mesh

Generate meshes from SDF expressions via marching cubes:

```rust
// Define SDF as FieldExpr
let sdf = FieldExpr::SdfSmoothUnion { ... };

// Wrap as Field for marching cubes
// (FieldExpr implements Field<Vec3, f32>)
let mesh = marching_cubes(&sdf, bounds, resolution);
```

### With resin-image

Bake expression fields to textures:

```rust
let noise = FieldExpr::Fbm2 {
    x: Box::new(FieldExpr::Mul(Box::new(FieldExpr::X), Box::new(FieldExpr::Constant(8.0)))),
    y: Box::new(FieldExpr::Mul(Box::new(FieldExpr::Y), Box::new(FieldExpr::Constant(8.0)))),
    octaves: 4,
};

let config = BakeConfig::new(512, 512);
let image = bake_scalar(&noise, &config, &ctx);
```

### With resin-gpu

Compile expressions to WGSL for GPU evaluation:

```rust
let ast = expr.to_dew_ast();
// Use dew's WGSL backend to generate shader code
```
