# Operations as Values vs Graph Recording

Two approaches to making operations recordable/replayable.

## Approach A: Operations as Values

Every operation is a struct. Methods are sugar.

```rust
// Core: operation as data
#[derive(Clone, Serialize, Deserialize)]
pub struct Subdivide {
    pub levels: u32,
}

impl Subdivide {
    pub fn apply(&self, mesh: &Mesh) -> Mesh {
        // actual implementation
    }
}

// Sugar: method syntax
impl Mesh {
    pub fn subdivide(&self, levels: u32) -> Mesh {
        Subdivide { levels }.apply(self)
    }
}
```

Recording is just collecting ops:

```rust
let ops: Vec<Box<dyn MeshOp>> = vec![
    Box::new(Cube::new(1.0)),
    Box::new(Subdivide { levels: 2 }),
];

// Replay
let mut mesh = Mesh::empty();
for op in &ops {
    mesh = op.apply(&mesh);
}
```

## Approach B: Separate Graph API

Direct API and graph API are separate.

```rust
// Direct: no recording
let mesh = Mesh::cube(1.0).subdivide(2);

// Graph: built-in recording
let graph = MeshGraph::new()
    .add(Cube::new(1.0))
    .add(Subdivide::new(2));

let mesh = graph.evaluate();
```

## Comparison

| Aspect | Ops as Values | Separate Graph |
|--------|---------------|----------------|
| Recording overhead | None (ops exist anyway) | Only if using graph |
| API consistency | One way to do things | Two parallel APIs |
| Serialization | Automatic (ops are data) | Graph handles it |
| Closures/lambdas | ❌ Can't serialize | ✅ Direct API allows |
| External refs | Needs explicit handling | Same |
| Impl complexity | Lower | Higher (two systems) |

## The Closure Problem

Approach A breaks down with closures:

```rust
// This can't be an "operation as value"
mesh.map_vertices(|v| v * 2.0)

// What would the op struct look like?
struct MapVertices {
    f: ???  // Can't serialize a closure
}
```

### Solutions for closures:

**1. Named transforms instead of closures**

```rust
// Instead of closure:
mesh.map_vertices(|v| v * 2.0)

// Use named op:
mesh.apply(Scale::uniform(2.0))
```

Limitation: loses generality. Can't do arbitrary transforms.

**2. Expression language**

```rust
#[derive(Serialize, Deserialize)]
enum Expr {
    Position,
    Const(Vec3),
    Mul(Box<Expr>, Box<Expr>),
    // ...
}

struct MapVertices {
    expr: Expr,  // Serializable!
}

// Usage
mesh.map_vertices(Expr::Mul(Expr::Position, Expr::Const(Vec3::splat(2.0))))
```

Ugly API, but serializable. Could have builder/macro sugar.

**3. Hybrid: ops when possible, escape hatch for closures**

```rust
// Serializable ops
let op = Subdivide { levels: 2 };

// Non-serializable escape hatch
let custom = CustomOp::new(|mesh| {
    // arbitrary code
});
// Marked as non-serializable, graph warns/errors on save
```

## The External Reference Problem

Both approaches have this:

```rust
struct Displace {
    texture: ???,  // How to serialize a reference to a texture?
    amount: f32,
}
```

### Solutions:

**1. IDs / Paths**

```rust
#[derive(Serialize, Deserialize)]
struct Displace {
    texture: TextureId,  // or String path
    amount: f32,
}

// Resolution at apply time
impl Displace {
    fn apply(&self, mesh: &Mesh, ctx: &Context) -> Mesh {
        let tex = ctx.get_texture(self.texture)?;
        // ...
    }
}
```

**2. Inline the dependency**

```rust
#[derive(Serialize, Deserialize)]
struct Displace {
    texture: TextureGraph,  // Inline the texture's graph
    amount: f32,
}
```

Graphs can reference other graphs.

## How to Serialize a Graph

```rust
#[derive(Serialize, Deserialize)]
struct MeshGraph {
    nodes: Vec<MeshNode>,
    wires: Vec<(NodeId, PortId, NodeId, PortId)>,
}

#[derive(Serialize, Deserialize)]
enum MeshNode {
    Cube { size: Vec3 },
    Subdivide { levels: u32 },
    Transform { matrix: Mat4 },
    Displace { texture: TextureId, amount: f32 },
    // ...
}
```

For extensibility (plugins), use a registry:

```rust
#[derive(Serialize, Deserialize)]
struct MeshNode {
    type_name: String,  // "resin::Subdivide" or "myplugin::CustomOp"
    params: serde_json::Value,  // or similar
}

// Registry resolves type_name -> deserializer
```

## Recommendation

**Hybrid approach:**

1. **Ops as values where possible** - most ops are pure data
2. **Expression system for transforms** - avoids closure problem
3. **External refs via IDs** - resolved at evaluation time
4. **Graph is collection of ops** - not a separate system

```rust
// All ops derive Serialize
#[derive(Clone, Serialize, Deserialize)]
pub struct Subdivide { pub levels: u32 }

// Expression-based for generality
#[derive(Clone, Serialize, Deserialize)]
pub struct MapVertices { pub expr: VertexExpr }

// Graph is just Vec<Op> + wires
#[derive(Serialize, Deserialize)]
pub struct MeshGraph {
    ops: Vec<MeshOp>,
    wires: Vec<Wire>,
}

// Method API is sugar, uses ops internally
impl Mesh {
    pub fn subdivide(&self, levels: u32) -> Mesh {
        Subdivide { levels }.apply(self)
    }
}
```

## Runtime Type Safety

When pipelines are loaded at runtime, we lose compile-time type checking.

### The Problem

```rust
// Compile-time: types checked statically
let result: Image = noise.render(1024, 1024).blur(0.01);  // ✓ compiler verifies

// Runtime: types unknown at compile time
let pipeline = load_pipeline("effect.json")?;
let result = pipeline.execute(input)?;  // what type is result?
```

### Solution: Value Enum + Schema Validation

**Value enum for dynamic typing:**

```rust
enum Value {
    Field(Box<dyn Field>),
    Image(Image),
    Mesh(Mesh),
    Float(f32),
    Vec2(Vec2),
    Vec3(Vec3),
    Color(Color),
    // ...
}

impl Value {
    fn into_image(self) -> Result<Image> {
        match self {
            Value::Image(img) => Ok(img),
            other => Err(TypeError { expected: "Image", got: other.type_name() }),
        }
    }
}
```

**Ops use Value for dynamic execution:**

```rust
trait DynOp: Serialize + Deserialize {
    fn input_type(&self) -> ValueType;
    fn output_type(&self) -> ValueType;
    fn apply(&self, input: Value) -> Result<Value>;
}

#[derive(Serialize, Deserialize)]
struct Blur {
    radius_uv: f32,
}

impl DynOp for Blur {
    fn input_type(&self) -> ValueType { ValueType::Image }
    fn output_type(&self) -> ValueType { ValueType::Image }

    fn apply(&self, input: Value) -> Result<Value> {
        let image = input.into_image()?;
        Ok(Value::Image(self.blur_impl(&image)))
    }
}
```

**Schema validation at load time:**

```rust
fn validate_pipeline(ops: &[Box<dyn DynOp>]) -> Result<(ValueType, ValueType)> {
    if ops.is_empty() {
        return Err(EmptyPipeline);
    }

    let mut current = ops[0].input_type();
    for op in ops {
        if op.input_type() != current {
            return Err(TypeMismatch {
                op: op.type_name(),
                expected: op.input_type(),
                got: current,
            });
        }
        current = op.output_type();
    }

    Ok((ops[0].input_type(), current))
}

// Errors caught at load time, not execution time
let pipeline = load_pipeline("effect.json")?;
let (in_type, out_type) = validate_pipeline(&pipeline.ops)?;
```

### Two APIs

```rust
// Static API: compile-time types, zero overhead
// For code written in Rust
let result: Image = Perlin::new()
    .render(1024, 1024)
    .blur(0.01);

// Dynamic API: runtime types, Value enum
// For loaded/deserialized pipelines
let pipeline = load_pipeline("effect.json")?;
let input = Value::Field(Box::new(Perlin::new()));
let result: Value = pipeline.execute(input)?;
let image: Image = result.into_image()?;
```

### Op Author DX: Concrete Types, Not Value

Op authors should never see Value enum. They write concrete types:

```rust
// What op author writes (good DX)
#[derive(Serialize, Deserialize, Op)]
struct Blur { radius_uv: f32 }

impl Blur {
    fn apply(&self, input: &Image) -> Image {
        // just blur, no wrapping/unwrapping
    }
}

// NOT this (bad DX)
impl DynOp for Blur {
    fn apply(&self, input: Value) -> Result<Value> {
        let img = input.into_image()?;  // boilerplate
        Ok(Value::Image(...))            // boilerplate
    }
}
```

**Solution: Derive macro**

```rust
#[derive(Serialize, Deserialize, Op)]
struct Blur { radius_uv: f32 }

impl Blur {
    // Author writes normal method with concrete types
    fn apply(&self, input: &Image) -> Image {
        // actual blur implementation
    }
}

// Macro generates:
// - DynOp impl with Value wrapping/unwrapping
// - input_type() -> ValueType::Image
// - output_type() -> ValueType::Image
// - Registration with type name "resin::texture::Blur"
```

**Works for any signature:**

```rust
#[derive(Serialize, Deserialize, Op)]
struct Render { width: u32, height: u32 }

impl Render {
    fn apply(&self, input: &dyn Field) -> Image { ... }
}

#[derive(Serialize, Deserialize, Op)]
struct Blend { factor: f32 }

impl Blend {
    fn apply(&self, a: &Image, b: &Image) -> Image { ... }
}

#[derive(Serialize, Deserialize, Op)]
struct Displace { amount: f32 }

impl Displace {
    fn apply(&self, mesh: &Mesh, texture: &Image) -> Mesh { ... }
}
```

**Macro responsibilities:**
1. Find `apply` method on impl block
2. Infer input/output types from signature
3. Generate `DynOp` impl with Value wrapping
4. Generate type name for serialization

**Why derive macro over trait proliferation:**

| Approach | DX | Maintenance |
|----------|-----|-------------|
| Trait per signature | Bad - many traits to know | Bad - grows with type combinations |
| Derive macro | Good - one pattern | Good - macro handles all cases |

Derive macros are standard Rust practice (serde, thiserror, clap). Minor compile-time cost, major DX win.

**Result:** Value enum is internal plumbing. Op authors just write `#[derive(Op)]` and a normal `apply` method.

### Type-Safe Wrappers (Optional)

If you know expected types at load time:

```rust
// Load with expected signature
let pipeline: TypedPipeline<Field, Image> =
    load_typed("effect.json")?;  // validates at load

// Execute is type-safe
let result: Image = pipeline.execute(noise);  // no Value enum needed
```

Implementation wraps dynamic internals:

```rust
struct TypedPipeline<In, Out> {
    inner: DynPipeline,
    _marker: PhantomData<(In, Out)>,
}

impl<In: IntoValue, Out: FromValue> TypedPipeline<In, Out> {
    fn execute(&self, input: In) -> Result<Out> {
        let value = self.inner.execute(input.into_value())?;
        Out::from_value(value)
    }
}
```

## Open Questions

1. **Expression language scope**: How powerful? Just math, or control flow too?
2. ~~**Plugin ops**: How do plugins register serializable op types?~~ -> Resolved, see [plugin-architecture](./plugin-architecture.md)
3. ~~**Graph evaluation caching**: If input changes, re-evaluate only affected nodes?~~ -> Resolved, hash-based caching
4. **Lazy vs eager**: Does method API evaluate immediately, or build implicit graph?
5. **Value enum exhaustiveness**: How many types in Value? Extensible or fixed?

## Summary

| Approach | Use when |
|----------|----------|
| Ops as values | Most operations (pure data) |
| Expression system | Generic transforms (replaces closures) |
| Graph API | When you need recording/replay/serialization |
| Direct method API | Quick scripts, one-off operations |

The direct API uses ops internally, so recording is always *possible* even if not used.
