# Vector Networks vs Paths

Comparing traditional path-based vector graphics with Figma-style vector networks.

## Definitions

### Path (Traditional)

A path is an **ordered sequence of segments** forming a chain:

```
A ──-> B ──-> C ──-> D
```

- Each point has at most 2 connections (previous, next)
- May be open (A to D) or closed (D connects back to A)
- No branching allowed

```rust
struct Path {
    segments: Vec<Segment>,  // ordered
    closed: bool,
}
```

### Vector Network (Figma)

A vector network is a **graph of anchors and edges**:

```
    A ───── B
   /│\      │
  / │ \     │
 D  │  E    │
  \ │ /     │
   \│/      │
    C ───── F
```

- Points can have any number of connections
- Edges connect any two anchors
- Regions (faces) are implicit from the topology
- More like a 2D mesh than a path

```rust
struct VectorNetwork {
    anchors: Vec<Anchor>,
    edges: Vec<Edge>,
    // faces derived from edges
}

struct Anchor {
    position: Vec2,
    // possibly: handle style, etc.
}

struct Edge {
    start: AnchorId,
    end: AnchorId,
    curve: CurveType,  // line, bezier, etc.
}
```

## Visual Comparison

### Simple shape: both work fine

```
Path:                    Network:
┌───────────┐            ┌───────────┐
│           │            │           │
│           │            │           │
└───────────┘            └───────────┘
(4 segments, closed)     (4 anchors, 4 edges)
```

### 2D projection of cube: network required

```
Paths (need 3 separate paths):     Network (single structure):
    ╱╲                                  A
   ╱  ╲                                ╱│╲
  ╱    ╲       Path 1: outer hex      B │ C
 │      │      Path 2: line A->D       │╲│╱│
 │      │      Path 3: line B->E        D─E
  ╲    ╱       (or draw as strokes)    │╱│╲
   ╲  ╱                                F │ G
    ╲╱                                  ╲│╱
                                        H

With network: single unified structure
Interior lines are edges, not separate paths
```

### Letter 'A': network might help

```
Path approach:              Network approach:
Outer path + inner path     Single network with hole
(compound path)             (hole is a face property)
    ╱╲                          ╱╲
   ╱  ╲                        ╱  ╲
  ╱────╲                      A────B
 ╱ ╱╲   ╲                    ╱│    │╲
╱ ╱  ╲   ╲                  C─D    E─F

2 paths                     6 anchors, multiple edges
```

## Trade-offs

### Paths

**Pros:**
- Simpler mental model
- Easier to iterate (just walk the array)
- Clear ordering (start -> end)
- Maps directly to SVG, PostScript, PDF
- Boolean operations well-understood
- Most algorithms assume path input

**Cons:**
- Can't represent branching structures natively
- Cube projection needs multiple paths or stroke hacks
- Compound paths (holes) are separate concept
- Shared anchors are duplicated

### Vector Networks

**Pros:**
- More general (paths are a subset)
- Natural for branching structures
- Shared anchors are actually shared
- Closer to mesh topology (2D mesh, essentially)
- Regions/fills are topological, not winding-rule based
- Edit any anchor/edge independently

**Cons:**
- More complex implementation
- Need to derive regions from edges (face finding algorithm)
- Fill rules become topology, not just math
- Most file formats don't support (SVG needs conversion)
- Most renderers expect paths
- Boolean ops more complex

## When Networks Win

1. **Technical drawings**: circuit diagrams, floor plans, graphs
2. **Geometric constructions**: projections, wireframes
3. **Connected structures**: flowcharts, node diagrams
4. **Shared anchors**: when you need "move this point, all connected edges follow"

## When Paths Win

1. **Artistic illustration**: most art is non-branching curves
2. **Text/fonts**: glyphs are paths
3. **Interoperability**: SVG, PDF, fonts all use paths
4. **Simple shapes**: rectangles, circles, hand-drawn curves

## Implementation Complexity

| Operation | Path | Network |
|-----------|------|---------|
| Iterate points | O(n) trivial | O(n) trivial |
| Find neighbors | index ± 1 | graph traversal |
| Boolean union | well-studied | research territory |
| Render/fill | direct | find regions first |
| Export to SVG | direct | convert to paths |
| Offset/stroke | well-studied | per-region? |

## Conversion

### Network -> Paths

Always possible by tracing each region boundary:

```rust
fn network_to_paths(net: &VectorNetwork) -> Vec<Path> {
    let regions = find_regions(net);  // face-finding algorithm
    regions.iter().map(|r| trace_boundary(net, r)).collect()
}
```

Loses the "shared anchor" property - anchors get duplicated.

### Paths -> Network

Possible but may not capture intent:

```rust
fn paths_to_network(paths: &[Path]) -> VectorNetwork {
    // Add all anchors and edges
    // Optionally: merge coincident anchors
}
```

Multiple paths through same point become shared anchor.

## Hybrid Approach?

Could support both:

```rust
enum VectorGeometry {
    Path(Path),
    Network(VectorNetwork),
}

// Convert on demand
impl VectorGeometry {
    fn as_paths(&self) -> Vec<Path> { /* ... */ }
    fn as_network(&self) -> VectorNetwork { /* ... */ }
}
```

Or: use network internally, present path-like API for common cases.

## Prior Art

| Tool | Model | Notes |
|------|-------|-------|
| SVG | Paths | Industry standard |
| PostScript/PDF | Paths | |
| Figma | **Networks** | Primary innovation |
| Illustrator | Paths | |
| Inkscape | Paths | |
| Paper.js | Paths | |
| OpenType/TrueType | Paths | Fonts |

Figma is the only major tool using networks natively.

## Recommendation

**Network internally, both APIs as equals.**

Key insight: paths are just networks with a constraint (degree ≤ 2). No need to choose - use network as the internal representation, offer both API surfaces.

```rust
// Internal: always a network
struct VectorNetwork {
    anchors: Vec<Anchor>,
    edges: Vec<Edge>,
}

// Path is a constrained view/wrapper
struct Path(VectorNetwork);  // newtype enforcing degree ≤ 2

impl Path {
    fn line_to(&mut self, p: Vec2) -> &mut Self { ... }
    fn point_at(&self, t: f32) -> Vec2 { ... }  // global parameterization
}

impl VectorNetwork {
    fn as_path(&self) -> Option<Path> { ... }  // None if branching
    fn add_edge(&mut self, a: AnchorId, b: AnchorId) { ... }
}
```

**Different APIs for different mental models:**

| Path API (constrained) | Network API (general) |
|------------------------|----------------------|
| `point_at(t)` global | `point_at(edge, t)` per-edge |
| `iter_segments()` ordered | `iter_edges()` unordered |
| `reverse()` flip direction | edges directed or not |
| `append(other)` concatenate | `merge(other)` union graphs |

**Interoperability is just conversion at boundaries:**
- Import SVG -> parse as paths -> store as network
- Export -> decompose to paths -> write SVG

## Open Questions

1. Is Figma's network model documented anywhere? (Seems proprietary)
2. Are there academic papers on vector network algorithms?
3. What's the actual use case frequency for branching structures?
4. Could we get 80% of the benefit with "paths + shared anchor references"?

```rust
// Lighter-weight alternative to full networks?
struct PathWithSharedAnchors {
    anchors: Vec<Anchor>,
    paths: Vec<Vec<AnchorId>>,  // paths reference shared anchors
}
```

This gets shared-anchor editing without full graph topology.
