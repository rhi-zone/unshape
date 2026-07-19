# Operation Shapes in the Graph Model

An "op" is a unit of computation in a graph: it has inputs, parameters, and produces outputs. The shape of that computation varies.

## The Five Shapes

**Transform**: Input type → same domain output. Refines or modifies existing data.
```rust
struct Subdivide { levels: u32 }
impl Subdivide { fn apply(&self, mesh: &Mesh) -> Mesh }
```

**Construction**: Multiple inputs → composed structure. Combines or sequences data.
```rust
struct Sequence { clips: Vec<AudioClip>, timings: Vec<f32> }
impl Sequence { fn apply(&self) -> Audio }
```

**Aggregation**: Collection → reduced value. Summarizes or extracts grouped data.
```rust
struct Sum { column: String }
impl Sum { fn apply(&self, table: &Table) -> f32 }
```

**Observation**: Data → extracted information. Queries properties without modification.
```rust
struct BoundingBox { }
impl BoundingBox { fn apply(&self, mesh: &Mesh) -> Bounds }
```

**Routing**: Condition + inputs → selected input. Branches control flow.
```rust
struct Switch { condition: bool }
impl Switch { fn apply(&self, a: &T, b: &T) -> T }
```

## Why Transforms Dominate Today

The implemented domains (mesh, image, audio, noise) are mostly iterative refinement: start with something, transform it. Transforms are the natural fit.

The other shapes exist *structurally* in the graph model — nodes can have any input/output signature — but they're underrepresented because media domains don't exercise them much.

## New Domains Exercise the Other Shapes

- **Tables** need aggregation (GROUP BY, pivot, statistics)
- **Timelines** need construction (sequence clips with timing)
- **Scene editing** needs routing (select which layer is visible, which effect applies)
- **All domains** benefit from observation (measure bounds, query properties, introspect state)

These aren't extensions to the graph model. They exercise parts that were always there.

## The Model is Already General

Mechanically, the graph supports any shape:

```
Input1 ──┐
Input2 ──┤ [Op] ──→ Output
Param ──→│
```

An op is: take typed inputs, consume parameters, return typed output. No restriction on what types or how many. Construction, aggregation, routing, observation all fit this.

Implementing them is domain work, not architectural work.

## Summary

Shapes are not new concepts — they're different *uses* of the same graph mechanism. As unshape grows beyond media refinement into data manipulation, composition, and control flow, the latent shapes naturally emerge.
