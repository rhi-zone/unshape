# Meshes

3D mesh generation, manipulation, and related data structures.

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
- **Subdivision surfaces**: coarse cage -> smooth limit surface
- **Creases**: edge weights controlling sharpness
- **Face-varying data**: UVs that don't smooth across boundaries

### SDF Libraries (libfive, mTec)
- **Implicit surfaces**: f(x,y,z) -> distance
- **CSG via min/max**: union = min(a,b), intersection = max(a,b)
- **Meshing**: marching cubes, dual contouring

## Crate Structure

| Crate | Purpose |
|-------|---------|
| `resin-mesh` | Core mesh types, primitives, operations |
| `resin-voxel` | Voxel grids and operations |
| `resin-pointcloud` | Point cloud sampling and processing |
| `resin-surface` | NURBS surfaces |

## Core Types

### Indexed Mesh

```rust
use rhi_unshape_mesh::{Mesh, MeshBuilder};

// Create mesh with builder
let mesh = MeshBuilder::new()
    .add_vertex(Vec3::new(0.0, 0.0, 0.0))
    .add_vertex(Vec3::new(1.0, 0.0, 0.0))
    .add_vertex(Vec3::new(0.5, 1.0, 0.0))
    .add_triangle(0, 1, 2)
    .build();

// Access data
let positions = mesh.positions();
let indices = mesh.indices();
let normals = mesh.normals();
let uvs = mesh.uvs();
```

### Half-Edge Mesh

```rust
use rhi_unshape_mesh::{HalfEdgeMesh, halfedge_from_mesh};

// Convert for topology operations
let he_mesh = halfedge_from_mesh(&mesh);

// Query topology
let vertex_edges = he_mesh.vertex_edges(vertex_id);
let face_vertices = he_mesh.face_vertices(face_id);
let edge_faces = he_mesh.edge_faces(edge_id);

// Convert back
let mesh = he_mesh.to_mesh();
```

## Primitives

```rust
use rhi_unshape_mesh::*;

// Basic shapes
let cube = box_mesh();
let cube_sized = box_mesh_sized(Vec3::new(2.0, 1.0, 3.0));
let sphere = uv_sphere(1.0, 32, 16);
let ico = icosphere(1.0, 3);
let cylinder = cylinder_mesh(1.0, 2.0, 32);
let cone = cone_mesh(1.0, 2.0, 32);
let torus = torus_mesh(1.0, 0.3, 32, 16);
let plane = plane_mesh(10.0, 10.0, 10, 10);

// With options
let cylinder = cylinder_mesh_with_caps(1.0, 2.0, 32, true, true);
```

## Basic Operations

### Transform

```rust
use rhi_unshape_mesh::{transform, translate, rotate, scale};

let mesh = transform(&mesh, Mat4::from_rotation_y(PI / 4.0));
let mesh = translate(&mesh, Vec3::new(0.0, 1.0, 0.0));
let mesh = rotate(&mesh, Quat::from_rotation_x(PI / 2.0));
let mesh = scale(&mesh, Vec3::splat(2.0));
```

### Normals

```rust
use rhi_unshape_mesh::{compute_normals, compute_smooth_normals};

// Flat shading (per-face normals)
let mesh = compute_normals(&mesh);

// Smooth shading (averaged vertex normals)
let mesh = compute_smooth_normals(&mesh);
```

### Merge and Combine

```rust
use rhi_unshape_mesh::{merge_meshes, weld_vertices};

// Combine multiple meshes
let combined = merge_meshes(&[mesh1, mesh2, mesh3]);

// Weld close vertices
let welded = weld_vertices(&mesh, threshold: 0.001);
```

## Topology Operations

### Extrude

```rust
use rhi_unshape_mesh::{extrude_faces, extrude_along_normals};

// Extrude selected faces
let mesh = extrude_faces(&mesh, &face_indices, distance: 1.0);

// Extrude along face normals
let mesh = extrude_along_normals(&mesh, &face_indices, distance: 0.5);
```

### Inset

```rust
use rhi_unshape_mesh::inset_faces;

let mesh = inset_faces(&mesh, &face_indices, amount: 0.2);
```

### Bevel

```rust
use rhi_unshape_mesh::{bevel_edges, bevel_vertices, BevelConfig};

let config = BevelConfig {
    amount: 0.1,
    segments: 2,
};

let mesh = bevel_edges(&mesh, &edge_indices, &config);
let mesh = bevel_vertices(&mesh, &vertex_indices, &config);
```

### Edge Loops

```rust
use rhi_unshape_mesh::{select_edge_loop, select_edge_ring, loop_cut};

// Select connected edges
let loop_edges = select_edge_loop(&mesh, start_edge);
let ring_edges = select_edge_ring(&mesh, start_edge);

// Add edge loop
let mesh = loop_cut(&mesh, &edge_indices, position: 0.5);
```

## Subdivision

```rust
use rhi_unshape_mesh::{subdivide_catmull_clark, subdivide_loop, subdivide_simple};

// Catmull-Clark (quads)
let smooth = subdivide_catmull_clark(&mesh, iterations: 2);

// Loop (triangles)
let smooth = subdivide_loop(&mesh, iterations: 2);

// Simple (no smoothing)
let subdivided = subdivide_simple(&mesh, iterations: 1);
```

## Decimation

```rust
use rhi_unshape_mesh::{decimate, DecimateConfig};

let config = DecimateConfig {
    target_ratio: 0.5,       // Keep 50% of triangles
    max_error: 0.01,         // Maximum geometric error
    preserve_boundary: true, // Keep mesh edges intact
};

let simplified = decimate(&mesh, &config);
```

## Mesh Repair

```rust
use rhi_unshape_mesh::{
    find_boundary_loops, fill_hole_fan, fill_hole_ear_clip,
    fill_hole_minimum_area, remove_degenerate_faces
};

// Find holes
let loops = find_boundary_loops(&mesh);

// Fill holes
for loop_vertices in loops {
    let mesh = fill_hole_fan(&mesh, &loop_vertices);
    // or
    let mesh = fill_hole_ear_clip(&mesh, &loop_vertices);
    // or
    let mesh = fill_hole_minimum_area(&mesh, &loop_vertices);
}

// Clean up
let mesh = remove_degenerate_faces(&mesh, area_threshold: 0.0001);
```

## Boolean Operations

```rust
use rhi_unshape_mesh::{boolean_union, boolean_subtract, boolean_intersect};

let result = boolean_union(&mesh_a, &mesh_b);
let result = boolean_subtract(&mesh_a, &mesh_b);
let result = boolean_intersect(&mesh_a, &mesh_b);
```

## Smoothing

```rust
use rhi_unshape_mesh::{smooth, smooth_taubin, SmoothConfig};

// Laplacian smoothing
let config = SmoothConfig {
    iterations: 10,
    factor: 0.5,
    preserve_boundary: true,
};
let smoothed = smooth(&mesh, &config);

// Taubin smoothing (reduces shrinkage)
let smoothed = smooth_taubin(&mesh, iterations: 10, lambda: 0.5, mu: -0.53);
```

## Remeshing

```rust
use rhi_unshape_mesh::{isotropic_remesh, quadify, RemeshConfig};

// Isotropic remeshing (uniform triangles)
let config = RemeshConfig {
    target_edge_length: 0.1,
    iterations: 5,
};
let remeshed = isotropic_remesh(&mesh, &config);

// Convert to quads
let quad_mesh = quadify(&mesh);
```

## UV Projection

```rust
use rhi_unshape_mesh::{uv_project_planar, uv_project_cylindrical, uv_project_spherical, uv_project_box};

let mesh = uv_project_planar(&mesh, normal: Vec3::Y, scale: 1.0);
let mesh = uv_project_cylindrical(&mesh, axis: Vec3::Y);
let mesh = uv_project_spherical(&mesh, center: Vec3::ZERO);
let mesh = uv_project_box(&mesh, scale: 1.0);
```

## Mesh from Curves

### Extrude Profile

```rust
use rhi_unshape_mesh::{extrude_profile, revolve_profile, sweep_profile};

// Extrude 2D profile along axis
let profile = vec![Vec2::new(0.0, 0.0), Vec2::new(1.0, 0.0), Vec2::new(1.0, 1.0)];
let mesh = extrude_profile(&profile, direction: Vec3::Z, distance: 2.0);

// Revolve around axis (lathe)
let mesh = revolve_profile(&profile, axis: Vec3::Y, segments: 32);

// Sweep along path
let path = vec![Vec3::ZERO, Vec3::new(0.0, 1.0, 0.0), Vec3::new(1.0, 2.0, 0.0)];
let mesh = sweep_profile(&profile, &path, twist: 0.0);
```

### Lofting

```rust
use rhi_unshape_mesh::{loft, loft_along_path};

// Loft between profiles
let profiles = vec![profile1, profile2, profile3];
let mesh = loft(&profiles);

// Loft with path
let mesh = loft_along_path(&profiles, &path);
```

## SDF and Marching Cubes

### Marching Cubes

```rust
use rhi_unshape_mesh::{marching_cubes, MarchingCubesConfig};

let config = MarchingCubesConfig {
    resolution: 64,
    bounds_min: Vec3::splat(-2.0),
    bounds_max: Vec3::splat(2.0),
    iso_value: 0.0,
};

// From SDF function
let mesh = marching_cubes(|p| p.length() - 1.0, &config);

// From Field
let mesh = marching_cubes_field(&sdf_field, &config);
```

### Mesh to SDF

```rust
use rhi_unshape_mesh::{mesh_to_sdf, mesh_to_sdf_fast};

// Accurate but slower
let sdf = mesh_to_sdf(&mesh, resolution: 64);

// Fast approximation
let sdf = mesh_to_sdf_fast(&mesh, resolution: 64);

// Query distance
let distance = sdf.sample(point);
```

## Navigation Meshes

```rust
use rhi_unshape_mesh::{NavMesh, find_path, smooth_path};

// Generate navmesh from walkable geometry
let navmesh = NavMesh::from_mesh(&mesh, agent_radius: 0.5, agent_height: 2.0);

// Pathfinding
let path = find_path(&navmesh, start, goal)?;

// Smooth path (string pulling)
let smooth = smooth_path(&navmesh, &path);
```

## Geodesic Distance

```rust
use rhi_unshape_mesh::{geodesic_distance, geodesic_path, find_mesh_center};

// Distance from source vertex to all others
let distances = geodesic_distance(&mesh, source_vertex);

// Shortest path on surface
let path = geodesic_path(&mesh, start_vertex, end_vertex);

// Find geometric center
let center_vertex = find_mesh_center(&mesh);
```

## Lattice Deformation (FFD)

```rust
use rhi_unshape_mesh::{Lattice, LatticeConfig};

let config = LatticeConfig {
    divisions: UVec3::new(4, 4, 4),
    bounds: mesh.bounds(),
};

let mut lattice = Lattice::new(&config);

// Move control points
lattice.set_control_point(1, 1, 1, new_position);

// Deform mesh
let deformed = lattice.deform(&mesh);
```

## Ambient Occlusion Baking

```rust
use rhi_unshape_mesh::{bake_ao_vertices, bake_ao_texture, AoConfig};

let config = AoConfig {
    rays: 64,
    max_distance: 1.0,
};

// Per-vertex AO
let ao_values = bake_ao_vertices(&mesh, &config);

// Texture baking
let ao_texture = bake_ao_texture(&mesh, resolution: 1024, &config);
```

## Point Clouds

```rust
use rhi_unshape_pointcloud::{PointCloud, sample_mesh_uniform, sample_mesh_weighted};

// Sample from mesh
let cloud = sample_mesh_uniform(&mesh, count: 10000);
let cloud = sample_mesh_weighted(&mesh, count: 10000, |face| face.area());

// Sample from SDF
let cloud = sample_sdf_surface(&sdf, count: 10000, bounds);

// Operations
let cloud = cloud.estimate_normals(k_neighbors: 10);
let cloud = cloud.filter_statistical(k: 10, std_ratio: 2.0);
let cloud = cloud.voxel_downsample(voxel_size: 0.1);

// Query
let nearest = cloud.nearest_neighbor(point);
let neighbors = cloud.k_nearest(point, k: 10);
```

## Voxels

```rust
use rhi_unshape_voxel::{VoxelGrid, SparseVoxels, fill_sphere, fill_box};

// Dense grid
let mut grid = VoxelGrid::new(64, 64, 64, false);

// Edit operations
grid.set(x, y, z, true);
fill_sphere(&mut grid, center, radius, true);
fill_box(&mut grid, min, max, true);

// Morphological operations
let dilated = grid.dilate();
let eroded = grid.erode();

// Convert to mesh
let mesh = grid.to_mesh();

// Sparse storage (for large volumes)
let mut sparse = SparseVoxels::new();
sparse.set(IVec3::new(10, 20, 30), true);

// From SDF
let grid = sdf_to_voxels(&sdf, resolution: 64, bounds);
```

## Terrain

```rust
use rhi_unshape_mesh::{Heightfield, HydraulicErosion, ThermalErosion};

// Create heightfield
let mut hf = Heightfield::new(256, 256);
hf.apply_diamond_square(roughness: 0.5, seed: 12345);

// Erosion
HydraulicErosion::default().erode(&mut hf, 10000);
ThermalErosion::default().erode(&mut hf);

// Convert to mesh
let mesh = hf.to_mesh(scale_xz: 100.0, scale_y: 20.0);
let normal_map = hf.to_normal_map();
```

## File I/O

```rust
use rhi_unshape_mesh::{import_obj, export_obj, import_gltf, export_gltf};

// OBJ
let mesh = import_obj("model.obj")?;
export_obj(&mesh, "output.obj")?;

// glTF
let scene = import_gltf("model.gltf")?;
export_gltf(&mesh, "output.glb")?;  // Binary GLB
```

## NURBS Surfaces

```rust
use rhi_unshape_surface::{NurbsSurface, nurbs_sphere, nurbs_cylinder, nurbs_torus};

// Primitives
let sphere = nurbs_sphere(Vec3::ZERO, radius: 1.0);
let cylinder = nurbs_cylinder(Vec3::ZERO, radius: 1.0, height: 2.0);
let torus = nurbs_torus(Vec3::ZERO, major: 1.0, minor: 0.3);

// Evaluate
let point = sphere.evaluate(u: 0.5, v: 0.5);
let normal = sphere.normal(u: 0.5, v: 0.5);

// Tessellate to mesh
let mesh = sphere.tessellate(divisions_u: 32, divisions_v: 16);
```
