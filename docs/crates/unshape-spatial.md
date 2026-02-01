# unshape-spatial

Spatial data structures for efficient queries and collision detection.

## Purpose

Provides data structures for organizing objects in space to enable fast queries. These structures accelerate operations that would otherwise require O(n) or O(n²) comparisons.

## Structures

- **Quadtree** - 2D point storage with region queries and nearest neighbor search
- **Octree** - 3D point storage with region queries and nearest neighbor search
- **BVH** - Binary tree of bounding boxes for ray intersection and AABB queries
- **SpatialHash** - Hash grid for broad-phase collision detection
- **R-tree** - Balanced tree for rectangle/AABB storage and queries

## Related Crates

- **unshape-physics** - Uses spatial structures for collision detection
- **unshape-mesh** - Uses BVH for ray-mesh intersection (AO baking)
- **unshape-pointcloud** - Point clouds can use octrees for neighbor queries

## Use Cases

### Point Queries
Find points within a region or nearest to a location:
```rust
let mut tree = Quadtree::new(bounds, 8, 4);
tree.insert(Vec2::new(10.0, 20.0), "point A");
let nearby: Vec<_> = tree.query_region(&search_box).collect();
let nearest = tree.nearest(query_point);
```

### Ray Casting
Efficient ray intersection testing against many objects:
```rust
let bvh = Bvh::build(objects_with_bounds);
let hits = bvh.intersect_ray(&ray);
let closest = bvh.intersect_ray_closest(&ray);
```

### Collision Detection
Broad-phase neighbor finding for physics:
```rust
let mut hash = SpatialHash::new(cell_size);
for (pos, obj) in objects {
    hash.insert(pos, obj);
}
// Check only nearby pairs for collision
for (_, obj_a) in hash.query_neighbors(position) {
    // Fine-grained collision test
}
```

### Rectangle Queries
Find overlapping rectangles (UI hit testing, map rendering):
```rust
let mut rtree = Rtree::new(4);
for rect in ui_elements {
    rtree.insert(rect.bounds, rect.id);
}
let under_cursor = rtree.query_point(mouse_pos);
```

## Compositions

### With unshape-physics
Replace linear collision checks with spatial acceleration:
- Use SpatialHash for uniform particle distributions
- Use BVH for complex scene geometry

### With unshape-mesh
Use BVH for operations requiring ray-mesh intersection:
- Ambient occlusion baking
- Ray casting for selection
- Collision detection

### With unshape-scatter
Accelerate Poisson disk sampling with spatial lookups:
- Use Quadtree/Octree to find existing points
- Reject new points based on nearest neighbor distance
