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
    edges: Vec<(NodeId, PortId, NodeId, PortId)>,
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

// Registry resolves type_name → deserializer
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

// Graph is just Vec<Op> + edges
#[derive(Serialize, Deserialize)]
pub struct MeshGraph {
    ops: Vec<MeshOp>,
    edges: Vec<Edge>,
}

// Method API is sugar, uses ops internally
impl Mesh {
    pub fn subdivide(&self, levels: u32) -> Mesh {
        Subdivide { levels }.apply(self)
    }
}
```

## Open Questions

1. **Expression language scope**: How powerful? Just math, or control flow too?
2. **Plugin ops**: How do plugins register serializable op types?
3. **Graph evaluation caching**: If input changes, re-evaluate only affected nodes?
4. **Lazy vs eager**: Does method API evaluate immediately, or build implicit graph?

## Summary

| Approach | Use when |
|----------|----------|
| Ops as values | Most operations (pure data) |
| Expression system | Generic transforms (replaces closures) |
| Graph API | When you need recording/replay/serialization |
| Direct method API | Quick scripts, one-off operations |

The direct API uses ops internally, so recording is always *possible* even if not used.
