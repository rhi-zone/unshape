# Deformer Stacking: List vs Graph

Deformers modify geometry (mesh vertices, path points, etc.). When multiple deformers apply, order matters. How should we represent this?

## The Problem

```
Mesh -> Bend -> Twist -> Lattice -> Output
```

Order matters:
- Bend then Twist ≠ Twist then Bend
- Results are visibly different

## Option 1: List (Stack)

```rust
struct DeformerStack {
    deformers: Vec<Box<dyn Deformer>>,
}

impl DeformerStack {
    fn apply(&self, mesh: &mut Mesh) {
        for deformer in &self.deformers {
            deformer.apply(mesh);
        }
    }
}
```

**Pros:**
- Simple mental model: top-to-bottom or bottom-to-top
- Easy to reorder (swap indices)
- Matches Blender's modifier stack UI
- Clear execution order

**Cons:**
- No parallelism (strictly sequential)
- Can't express "these two are independent"
- All deformers see same input (no branching)

**Prior art:**
- Blender modifier stack
- Maya deformer stack
- Most 3D apps

## Option 2: DAG (Graph)

```rust
struct DeformerGraph {
    nodes: Vec<DeformerNode>,
    wires: Vec<(NodeId, NodeId)>,  // from -> to
}

impl DeformerGraph {
    fn apply(&self, mesh: &mut Mesh) {
        // Topological sort, then apply in order
        for node in self.topo_sort() {
            node.apply(mesh);
        }
    }
}
```

**Pros:**
- Can express parallelism (independent branches)
- Can merge multiple deformation streams
- More flexible composition
- Matches node graph paradigm used elsewhere in resin

**Cons:**
- More complex to reason about
- "What order?" is less obvious
- Need topological sort
- UI is more complex than a list

**Prior art:**
- Houdini (everything is nodes)
- Blender Geometry Nodes (but modifiers are still a stack)

## Option 3: Hybrid

Stack by default, with explicit "parallel group":

```rust
enum DeformerEntry {
    Single(Box<dyn Deformer>),
    Parallel(Vec<Box<dyn Deformer>>),  // applied independently, results merged
}

struct DeformerStack {
    entries: Vec<DeformerEntry>,
}
```

**Pros:**
- Simple default case (just a list)
- Parallelism when needed
- Familiar mental model

**Cons:**
- How to "merge" parallel results? Addition? Average? Max?
- Still limited vs full graph

## Key Questions

### 1. Do deformers need to branch?

```
        ┌-> Deformer A ─┐
Mesh ──->│              │──-> Merge ──-> Output
        └-> Deformer B ─┘
```

If yes, need graph. If no, list is fine.

**Analysis:** Most deformer workflows are linear. Branching is rare. When needed, it's usually:
- Blend between two deformation results (morph targets do this better)
- Apply different deformers to different vertex groups (selection-based)

### 2. Is deformer order ever ambiguous?

In a list, order is explicit. In a graph, order is defined by dependencies. If two nodes have no edge between them, order is undefined (or implementation-defined).

For deformers, **order almost always matters**, so we'd end up adding edges to force order anyway -> might as well use a list.

### 3. Consistency with rest of resin

If resin uses graphs everywhere else (mesh ops, texture ops), having deformers be a list is inconsistent.

Counter-argument: deformers are a specific pattern (sequential mutation) that doesn't fit the "DAG of pure operations" model well anyway.

## Recommendation

**Start with List**, add graph later if needed.

Reasoning:
1. Matches mental model of "stack of effects"
2. Simpler implementation
3. Covers 95% of use cases
4. Can always generalize later (list is a degenerate graph)

```rust
// Simple list-based API
let deformed = mesh
    .apply(Bend::new(axis, angle))
    .apply(Twist::new(axis, amount))
    .apply(Lattice::new(cage));

// Internally stored as Vec<Box<dyn Deformer>>
```

If we later need graphs, the list can become a single-chain graph.

## Escape Hatch: Explicit Blend

For the rare "parallel deformation" case:

```rust
let deformed = mesh.apply(Blend::new(
    weight_a, Bend::new(...),
    weight_b, Twist::new(...),
));
```

This keeps the stack linear while allowing weighted combination of two deformations.

## Decision

Following the [General Internal, Constrained APIs](./general-internal-constrained-api.md) principle:

- **Internal**: DeformerGraph (DAG, full flexibility)
- **Constrained API**: DeformerStack (linear, simple)

```rust
// General
struct DeformerGraph { nodes, edges }

// Constrained
struct DeformerStack(DeformerGraph);  // enforces linear topology
```

Most users use the stack API. Power users access the graph when needed.

| Approach | Complexity | Flexibility | Recommended? |
|----------|------------|-------------|--------------|
| Graph internal + Stack API | Medium | High | **Yes** |
