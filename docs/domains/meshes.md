# Meshes

3D mesh generation and manipulation.

## Prior Art

### Blender Geometry Nodes
- **Fields**: lazy-evaluated expressions over geometry elements
- **Attributes**: named data on vertices/edges/faces, typed (float, vector, color, etc.)
- **Domain transfer**: interpolate attributes between vertex/edge/face/corner
- **Instances**: lightweight copies sharing geometry, with transforms

### .kkrieger / Werkzeug
- **Operators**: mesh generators and modifiers as stackable ops
- **Splines**: cubic splines for extrusion paths, lathe profiles
- **CSG**: boolean operations on meshes

### OpenSubdiv / Catmull-Clark
- **Subdivision surfaces**: coarse cage → smooth limit surface
- **Creases**: edge weights controlling sharpness
- **Face-varying data**: UVs that don't smooth across boundaries

### SDF Libraries (libfive, mTec)
- **Implicit surfaces**: f(x,y,z) → distance
- **CSG via min/max**: union = min(a,b), intersection = max(a,b)
- **Meshing**: marching cubes, dual contouring

## Core Types

```rust
/// Half-edge mesh representation
struct Mesh {
    vertices: Vec<Vertex>,
    edges: Vec<HalfEdge>,
    faces: Vec<Face>,
    attributes: AttributeStorage,
}

struct Vertex {
    position: Vec3,
    edge: EdgeId,  // one outgoing half-edge
}

struct HalfEdge {
    vertex: VertexId,    // vertex at tip
    face: Option<FaceId>, // adjacent face (None = boundary)
    twin: EdgeId,        // opposite half-edge
    next: EdgeId,        // next edge in face loop
}

struct Face {
    edge: EdgeId,  // one edge in face loop
}

/// Attribute storage - data per element
struct AttributeStorage {
    vertex_attrs: HashMap<String, VertexAttribute>,
    edge_attrs: HashMap<String, EdgeAttribute>,
    face_attrs: HashMap<String, FaceAttribute>,
    corner_attrs: HashMap<String, CornerAttribute>,  // per face-vertex
}

enum AttributeData {
    Float(Vec<f32>),
    Vec2(Vec<Vec2>),
    Vec3(Vec<Vec3>),
    Vec4(Vec<Vec4>),
    Int(Vec<i32>),
}
```

## Primitives (Generators)

| Primitive | Parameters | Notes |
|-----------|------------|-------|
| Box | size: Vec3, segments: UVec3 | Axis-aligned |
| Sphere | radius: f32, segments: u32, rings: u32 | UV sphere |
| IcoSphere | radius: f32, subdivisions: u32 | More uniform distribution |
| Cylinder | radius: f32, height: f32, segments: u32 | Open or capped |
| Cone | radius: f32, height: f32, segments: u32 | |
| Torus | major: f32, minor: f32, segments: (u32, u32) | |
| Plane | size: Vec2, segments: UVec2 | |
| Grid | size: Vec2, resolution: UVec2 | Same as plane? |

## Operations (Modifiers)

### Topology-changing
- **Subdivide**: Catmull-Clark, Loop, simple
- **Triangulate**: fan, ear-clip
- **Decimate**: collapse edges to reduce poly count
- **Extrude**: faces, edges, along normals or direction
- **Inset**: scale faces inward, creating border
- **Bevel**: chamfer edges/vertices
- **Boolean**: union, difference, intersection (CSG)

### Attribute-modifying
- **Transform**: translate, rotate, scale
- **Displace**: offset vertices by noise/texture
- **Smooth**: laplacian smoothing
- **Set attribute**: assign values from expression

### Analysis
- **Normals**: compute vertex/face normals
- **Tangents**: compute tangent space for normal mapping
- **Bounds**: AABB, OBB
- **Topology queries**: adjacent faces, edge loops, etc.

## Data Flow Pattern

```
Generator → Modifier → Modifier → ... → Output
   ↑            ↑
   params       params (can be expressions)
```

Attributes flow through: each modifier declares which attributes it preserves, modifies, or creates.

## Open Questions

1. **Half-edge vs index-based**: Half-edge is better for topology operations but more memory. Index-based (pos[], indices[]) is GPU-friendly. Support both? Convert at boundaries?

2. **Instancing**: Blender's instance system is powerful. How do we represent "100 copies of this mesh with different transforms"?

3. **Fields/expressions**: Blender's field system allows `position.z > 0` as a selection. How much of this do we want?

4. **SDF integration**: Should SDFs be a separate representation, or can we unify with mesh ops?
