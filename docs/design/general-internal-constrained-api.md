# Design Principle: General Internal, Constrained APIs

Store the general representation internally. Expose constrained APIs for common cases.

## The Pattern

```
General Representation (internal)
         │
         ├── Constrained API A (common case)
         ├── Constrained API B (common case)
         └── General API (full power)
```

The constrained APIs are views/wrappers that enforce invariants. Users pick the mental model that fits their use case.

## Applied Across Domains

### Vector 2D

| General (internal) | Constrained API |
|--------------------|-----------------|
| VectorNetwork (anchors + edges, any degree) | Path (degree ≤ 2, ordered) |

```rust
struct VectorNetwork {
    anchors: Vec<Anchor>,
    edges: Vec<Edge>,
}

// Path is a constrained view
struct Path(VectorNetwork);  // enforces degree ≤ 2

impl Path {
    fn line_to(&mut self, p: Vec2) { /* maintains constraint */ }
    fn point_at(&self, t: f32) -> Vec2 { /* global parameterization */ }
}
```

### Mesh

| General (internal) | Constrained API |
|--------------------|-----------------|
| Half-edge mesh (full topology) | IndexedMesh (GPU-friendly, no adjacency queries) |

```rust
struct HalfEdgeMesh {
    vertices: Vec<Vertex>,
    half_edges: Vec<HalfEdge>,
    faces: Vec<Face>,
}

// IndexedMesh is a constrained/simplified view
struct IndexedMesh {
    positions: Vec<Vec3>,
    indices: Vec<u32>,
    // No topology - just triangles
}

impl HalfEdgeMesh {
    fn to_indexed(&self) -> IndexedMesh { /* flatten */ }
}

impl IndexedMesh {
    fn to_half_edge(&self) -> HalfEdgeMesh { /* rebuild topology */ }
}
```

**Why both?**
- Half-edge: subdivision, extrusion, topology queries
- Indexed: GPU upload, rendering, physics engines

### Audio

| General (internal) | Constrained API |
|--------------------|-----------------|
| AudioGraph (DAG of nodes) | Chain (linear sequence) |

```rust
struct AudioGraph {
    nodes: Vec<Box<dyn AudioNode>>,
    wires: Vec<(NodeId, usize, NodeId, usize)>,  // (from, output, to, input)
}

// Chain is a constrained view - linear, no branching
struct Chain(AudioGraph);  // enforces linear topology

impl Chain {
    fn then<N: AudioNode>(self, node: N) -> Self { /* append */ }
}

// Builder pattern for common case
let synth = Chain::new()
    .then(Oscillator::saw(440.0))
    .then(Filter::lowpass(1000.0))
    .then(Envelope::adsr(0.1, 0.2, 0.7, 0.3));

// Equivalent to graph, but simpler API
```

**Why both?**
- Graph: parallel processing, feedback loops, complex routing
- Chain: simple synth patches, effects chains, common case

### Rigging / Deformers

| General (internal) | Constrained API |
|--------------------|-----------------|
| DeformerGraph (DAG) | DeformerStack (ordered list) |

```rust
struct DeformerGraph {
    deformers: Vec<Box<dyn Deformer>>,
    wires: Vec<(NodeId, NodeId)>,
}

// Stack is a constrained view - linear, no branching
struct DeformerStack(DeformerGraph);  // enforces linear topology

impl DeformerStack {
    fn push(&mut self, d: impl Deformer) { /* append */ }
}

let deformed = mesh
    .apply(Bend::new(axis, angle))
    .apply(Twist::new(axis, amount));
```

**Why both?**
- Graph: parallel deformations, blend between branches
- Stack: 95% of use cases, simpler mental model

### Textures

| General (internal) | Constrained API |
|--------------------|-----------------|
| TextureGraph (DAG of nodes) | TextureExpr (composable expressions) |

```rust
struct TextureGraph {
    nodes: Vec<Box<dyn TextureNode>>,
    wires: Vec<(NodeId, NodeId)>,
}

// Expression-style API for simple cases
let tex = noise(scale: 4.0)
    .fbm(octaves: 6)
    .remap(0.0, 1.0, -1.0, 1.0)
    .blend(gradient(Dir::Y), 0.5);

// Internally builds a graph
```

**Why both?**
- Graph: complex multi-input operations, reusable subgraphs
- Expression: fluent API for linear pipelines

## Benefits

1. **No loss of generality** - full power available when needed
2. **Simpler common case** - constrained APIs are easier to use
3. **Single source of truth** - one internal representation
4. **Interoperability** - convert between views freely
5. **Progressive disclosure** - start simple, go general when needed

## Implementation Notes

### Constraint Enforcement

Constrained types should enforce their invariants:

```rust
impl Path {
    fn add_vertex(&mut self, v: Vertex) -> Result<VertexId, PathError> {
        if self.would_branch(v) {
            return Err(PathError::WouldBranch);
        }
        Ok(self.0.add_vertex(v))
    }
}
```

### Zero-Cost When Possible

If the constraint is structural, it can be compile-time:

```rust
// Path is always linear - some operations don't need runtime checks
impl Path {
    fn reverse(&mut self) {
        // Just reverse the edge order - always valid for paths
        self.0.edges.reverse();
    }
}
```

### Escape Hatch

Always provide access to the general representation:

```rust
impl Path {
    fn into_network(self) -> VectorNetwork { self.0 }
    fn as_network(&self) -> &VectorNetwork { &self.0 }
}
```

## Summary

| Domain | General | Constrained |
|--------|---------|-------------|
| Vector | VectorNetwork | Path |
| Mesh | HalfEdgeMesh | IndexedMesh |
| Audio | AudioGraph | Chain |
| Deformers | DeformerGraph | DeformerStack |
| Textures | TextureGraph | TextureExpr |

This is a core resin pattern: **general storage, constrained interfaces**.
