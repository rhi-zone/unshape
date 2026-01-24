# Mesh Representation

Internal mesh structure for unshape's generation operations.

## The Question

What data structure for meshes?

| Structure | Good for | Bad for |
|-----------|----------|---------|
| Indexed | GPU upload, export, simple ops | Topology queries |
| Half-edge | Topology ops, adjacency queries | Memory, GPU upload |

## Indexed Mesh

What GPUs and file formats expect:

```rust
struct IndexedMesh {
    positions: Vec<Vec3>,
    normals: Vec<Vec3>,
    uvs: Vec<Vec2>,
    indices: Vec<u32>,  // triangles: [0,1,2, 0,2,3, ...]
}
```

**Pros:**
- Compact memory
- Direct GPU upload
- Simple to understand
- Standard export format

**Cons:**
- "Which faces share this edge?" -> O(n) search
- "Walk around vertex" -> O(n) search
- Topology ops require rebuilding adjacency each time

## Half-Edge Mesh

Stores topology explicitly:

```rust
struct HalfEdgeMesh {
    vertices: Vec<Vertex>,
    half_edges: Vec<HalfEdge>,
    faces: Vec<Face>,
}

struct Vertex {
    position: Vec3,
    half_edge: HalfEdgeId,  // one outgoing half-edge
}

struct HalfEdge {
    origin: VertexId,       // vertex this edge starts from
    twin: HalfEdgeId,       // opposite direction edge
    next: HalfEdgeId,       // next edge in face loop
    face: FaceId,           // face this edge borders
}

struct Face {
    half_edge: HalfEdgeId,  // one edge in face loop
}
```

**Pros:**
- O(1) adjacency queries:
  - `edge.twin` -> opposite edge
  - `edge.next` -> next edge in face
  - `edge.twin.next` -> edges around vertex
- Topology ops are natural
- Supports ngons (face loop can have any length)

**Cons:**
- ~3-4x memory of indexed (important, see below)
- Must convert for GPU
- More complex to construct

## Memory Considerations

**Very important** tradeoff.

Rough comparison for a mesh with V vertices, E edges, F faces:

| Structure | Storage |
|-----------|---------|
| Indexed (tris) | V × (pos + normal + uv) + F × 3 indices |
| Half-edge | V × vertex + 2E × half-edge + F × face |

For a typical manifold mesh: E ≈ 3V, F ≈ 2V (Euler's formula)

**Example: 10k vertex mesh**
- Indexed: ~400 KB
- Half-edge: ~1.2-1.6 MB

**When this matters:**
- Many meshes in memory simultaneously
- Very high-poly meshes
- Memory-constrained environments (WASM, mobile)

**When it doesn't:**
- Single mesh being processed
- Generation pipeline (process then export)
- Modern desktop (16+ GB RAM)

**Decision:** Accept the memory cost for topology ops. Subdivision, bevel, etc. are core operations. Can always convert to indexed for export/GPU.

## Conversion

### Half-edge -> Indexed

Straightforward traversal:

```rust
impl HalfEdgeMesh {
    fn to_indexed(&self) -> IndexedMesh {
        let mut positions = Vec::new();
        let mut indices = Vec::new();

        for face in &self.faces {
            // Walk face loop, triangulate, emit indices
            let verts: Vec<_> = self.face_vertices(face).collect();
            for tri in triangulate(&verts) {
                // ... emit triangle
            }
        }

        IndexedMesh { positions, indices, .. }
    }
}
```

### Indexed -> Half-edge

More complex - must build topology:

```rust
impl HalfEdgeMesh {
    fn from_indexed(indexed: &IndexedMesh) -> Result<Self, NonManifoldError> {
        // 1. Create vertices
        // 2. Create half-edges for each triangle edge
        // 3. Find and link twin edges
        // 4. Detect non-manifold cases (edge with 3+ faces)
    }
}
```

### Incremental Conversion

**Question:** If mesh changes, rebuild entire indexed mesh or update incrementally?

| Change type | Incremental possible? |
|-------------|----------------------|
| Move vertex | Yes - update position buffer |
| Change UV/normal | Yes - update attribute buffer |
| Add face | Harder - may need to grow buffers |
| Remove face | Harder - leaves holes or requires compaction |
| Topology change | Effectively full rebuild |

**Decision:** Full rebuild for now. We're generation-focused, not real-time editing. Revisit if live preview becomes important.

## Ngons and Triangulation

Half-edge naturally supports ngons (any polygon, not just triangles):

```
Triangle:        Quad:           Ngon:
    ●              ●───●           ●───●
   ╱ ╲            │   │          ╱     ╲
  ●───●           ●───●         ●       ●
                                 ╲     ╱
   3 half-edges   4 half-edges    ●───●
   in face loop   in face loop
                                  6 half-edges
```

**Internal:** Keep ngons. Some operations are cleaner:
- Subdivision expects quads
- Extrude preserves face shape
- Less geometry to process

**Output:** Triangulate on export. Any simple polygon can be triangulated:

```rust
fn triangulate(polygon: &[Vec3]) -> Vec<[usize; 3]> {
    if polygon.len() == 3 {
        return vec![[0, 1, 2]];
    }
    if is_convex(polygon) {
        // Fan triangulation
        return (1..polygon.len()-1)
            .map(|i| [0, i, i+1])
            .collect();
    }
    // Ear clipping for concave
    ear_clip(polygon)
}
```

**Ear clipping:** O(n²) but robust for any simple polygon (no self-intersection).

## Non-Manifold Geometry

### What is Manifold?

**Manifold** = every point has a neighborhood that looks like a disk (2D surface).

**Non-manifold** = topology violations:

```
T-junction                    Bowtie
(edge with 3+ faces):         (vertex connects separate surfaces):

    ┌───┬───┐                        ╱╲
    │ A │ B │                       ╱  ╲
    │   │   │                      ╱    ╲
    └───┼───┘                     ◯──────◯
        │ C │                      ╲    ╱
        └───┘                       ╲  ╱
        ↑                            ╲╱
  edge has 3 faces                   ↑
                              vertex shared by
                              2 separate cones
```

Other non-manifold cases:
- Edge with only one face (not at boundary)
- Face inside solid (interior face)
- Self-intersecting surface

### Why Half-Edge Struggles

Half-edge assumes: each edge has exactly 2 half-edges (one per adjacent face, or one + boundary marker).

T-junction breaks this: edge would need 3+ half-edges.

### Normalization Strategies

Non-manifold can **always** be converted to manifold, but requires choices:

```rust
enum NonManifoldFix {
    /// Error on non-manifold input
    Reject,

    /// Split edges/vertices to separate surfaces
    Split,

    /// Delete faces that cause non-manifold topology
    DeleteExtra,

    /// Try to merge/bridge to maintain connectivity
    Bridge,
}
```

**Split strategy (most common):**

```
T-junction before:          After split:
    ┌───┬───┐                  ┌───┐ ┌───┐
    │ A │ B │                  │ A │ │ B │
    └───┼───┘        ->         └───┘ └───┘
        │ C │                      ┌───┐
        └───┘                      │ C │
                                   └───┘
    (1 mesh)                   (2 separate meshes,
                                or 1 mesh with gap)
```

**Bowtie before:**          **After split:**
```
       ╱╲                          ╱╲
      ╱  ╲                        ╱  ╲
     ◯────◯          ->       ◯──◯    ◯──◯
      ╲  ╱                        ╲  ╱
       ╲╱                          ╲╱

  (shared vertex)           (two separate vertices)
```

### API Design

```rust
impl Mesh {
    /// Check if mesh is manifold
    fn is_manifold(&self) -> bool;

    /// Get list of non-manifold elements
    fn find_non_manifold(&self) -> NonManifoldReport {
        NonManifoldReport {
            t_junction_edges: Vec<EdgeId>,
            bowtie_vertices: Vec<VertexId>,
            // ...
        }
    }

    /// Convert to manifold using given strategy
    fn make_manifold(&self, strategy: NonManifoldFix) -> Mesh;
}

/// Import with explicit non-manifold handling
fn import_mesh(data: &[u8], non_manifold: NonManifoldFix) -> Result<Mesh>;
```

### Decision

1. Internal representation: half-edge (assumes manifold)
2. On import: detect non-manifold, apply explicit strategy
3. During operations: maintain manifold invariant
4. If operation would create non-manifold: error or fix with explicit strategy

Most procedural generation produces manifold meshes naturally. Non-manifold mainly appears in imports and boolean operations.

## Summary

| Aspect | Decision |
|--------|----------|
| Internal structure | Half-edge |
| Memory cost | Accept ~3-4x for topology benefits |
| Output format | Convert to indexed on demand |
| Incremental update | Not needed initially |
| Ngons | Keep internally, triangulate on export |
| Non-manifold | Reject or normalize with explicit strategy |

## Open Questions

1. **Memory optimization:** Can we compress half-edge for large meshes? (Index types, optional attributes)

2. **Parallel construction:** Building half-edge from indexed is sequential (twin linking). Parallelizable?

3. **Attribute storage:** Where do UVs, vertex colors, custom attributes live? Per-vertex? Per-face-corner?

4. **Boundary representation:** How to mark mesh boundaries? Null twin? Explicit boundary loops?

## Prior Art

- **OpenMesh:** C++ half-edge, handles non-manifold via extensions
- **CGAL:** Very general, surface mesh + polyhedral surface
- **libigl:** Indexed-focused, computes adjacency on demand
- **Blender BMesh:** Extended half-edge with loops and radial edges (handles non-manifold)
