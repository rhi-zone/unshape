# Feature Catalog

Index of unshape's crates organized by domain. See individual crate docs in `docs/crates/` for use cases and compositions, or run `cargo doc` for API details.

## Audio

| Crate | Description |
|-------|-------------|
| **unshape-audio** | Procedural audio synthesis (FM, wavetable, granular, physical modeling), effects (reverb, delay, filters), spectral processing (FFT/STFT), signal routing |
| **unshape-easing** | 31 easing functions (quad, cubic, elastic, bounce, etc.) with in/out/inout variants |

## Mesh & 3D Geometry

| Crate | Description |
|-------|-------------|
| **unshape-mesh** | 3D mesh operations: primitives, booleans, subdivision, remeshing, terrain/erosion, navigation meshes, architecture generation, SDF |
| **unshape-gltf** | glTF 2.0 import/export with PBR materials |
| **unshape-pointcloud** | Point cloud sampling, normal estimation, downsampling |
| **unshape-voxel** | Dense and sparse voxel grids, morphological ops, greedy meshing |
| **unshape-surface** | NURBS tensor product surfaces |
| **unshape-spline** | Curves: cubic Bezier, Catmull-Rom, B-spline, NURBS |
| **unshape-curve** | Unified `Curve` trait, 2D/3D segment types, arc-length paths |

## 2D Vector Graphics

| Crate | Description |
|-------|-------------|
| **unshape-vector** | 2D paths, boolean ops, stroke/offset, path trim, triangulation, vector networks, gradient meshes, text-to-path, hatching, SVG import/export |

## Image & Texture

| Crate | Description | Docs |
|-------|-------------|------|
| **[unshape-image](crates/unshape-image.md)** | Image as field, convolution, channel ops, color adjustments, distortion, image pyramids, normal maps | [docs](crates/unshape-image.md) |

## Color

| Crate | Description |
|-------|-------------|
| **unshape-color** | Color spaces (RGB, HSL, HSV, RGBA), gradients, blend modes |

## Animation & Rigging

| Crate | Description |
|-------|-------------|
| **unshape-rig** | Skeleton/bones, animation clips, blending/layers, IK (FABRIK, CCD), motion matching, procedural walk, secondary motion (jiggle, follow-through), skinning |
| **unshape-easing** | Animation easing functions |
| **unshape-motion-fn** | Motion functions: Spring, Oscillate, Wiggle, Eased, Lerp; typed MotionExpr AST; dew expression integration |
| **unshape-motion** | 2D motion graphics scene graph: Transform2D with anchor point, Layer hierarchy, blend modes, opacity |

## Physics

| Crate | Description |
|-------|-------------|
| **unshape-physics** | Rigid body simulation, colliders, constraints (distance, hinge, spring), soft body FEM, cloth |
| **unshape-spring** | Verlet spring systems, particles, angular constraints |
| **unshape-particle** | Particle systems with emitters and forces |
| **unshape-fluid** | Grid-based stable fluids (2D/3D), SPH particle fluids |

## Procedural Generation

| Crate | Description |
|-------|-------------|
| **unshape-noise** | Perlin, simplex noise (2D/3D), fractional Brownian motion |
| **unshape-automata** | 1D elementary CA (Wolfram rules), 2D automata (Game of Life, etc.) |
| **unshape-procgen** | Maze generation, Wave Function Collapse, road/river networks |
| **unshape-rd** | Gray-Scott reaction-diffusion with presets |
| **unshape-lsystem** | L-systems with turtle interpretation (2D/3D) |
| **unshape-space-colonization** | Space colonization for branching structures (trees, lightning) |

## Fields & Expressions

| Crate | Description |
|-------|-------------|
| **unshape-field** | Core `Field<I, O>` trait for lazy spatial computation, combinators |
| **unshape-expr-field** | Expression language for fields (math, noise functions) |

## Instancing & Scattering

| Crate | Description |
|-------|-------------|
| **unshape-scatter** | Instance placement: random, grid, Poisson disk sampling; stagger timing for animations |

## GPU Acceleration

| Crate | Description |
|-------|-------------|
| **unshape-gpu** | wgpu compute backend for noise/texture generation |

## Spatial Data Structures

| Crate | Description |
|-------|-------------|
| **unshape-spatial** | Quadtree, octree, BVH, spatial hash, R-tree for efficient spatial queries |

## Cross-Domain

| Crate | Description | Docs |
|-------|-------------|------|
| **[unshape-bytes](crates/unshape-bytes.md)** | Raw byte casting between numeric types (bytemuck) | [docs](crates/unshape-bytes.md) |
| **[unshape-crossdomain](crates/unshape-crossdomain.md)** | Image↔audio conversion, noise-as-anything adapters | [docs](crates/unshape-crossdomain.md) |

## Core & Serialization

| Crate | Description | Docs |
|-------|-------------|------|
| **[unshape-core](crates/unshape-core.md)** | Graph container, DynNode trait, Value enum, EvalContext, pluggable evaluation strategies (lazy/eager), caching | [docs](crates/unshape-core.md) |
| **unshape-transform** | `SpatialTransform` trait for unified 2D/3D transform interface | |
| **[unshape-jit](crates/unshape-jit.md)** | Cranelift JIT compilation with SIMD (41x faster than scalar, 6.6x faster than native) | [docs](crates/unshape-jit.md) |
| **unshape-op** | DynOp trait, `#[derive(Op)]` macro, OpRegistry, Pipeline execution | |
| **unshape-serde** | Graph serialization: SerialGraph format, NodeRegistry, JSON/bincode | |
| **unshape-history** | History tracking: snapshots (undo/redo), event sourcing (fine-grained) | |

---

## Summary

| Domain | Crates | Highlights |
|--------|--------|------------|
| Audio | 2 | FM, wavetable, granular, physical modeling, effects, spectral |
| Mesh | 7 | Booleans, subdivision, remeshing, terrain, NURBS, voxels, curves |
| 2D Vector | 1 | Paths, booleans, networks, gradients, text, SVG |
| Image | 1 | Convolution, color adjust, distortion, pyramids |
| Color | 1 | Color spaces, gradients, blend modes |
| Animation | 4 | Skeleton, IK, motion matching, easing, motion functions, 2D scene graph |
| Physics | 4 | Rigid body, soft body, cloth, springs, particles, fluids |
| Procedural | 6 | Noise, automata, WFC, L-systems, reaction-diffusion |
| Spatial | 1 | Quadtree, octree, BVH, spatial hash, R-tree |
| Cross-Domain | 2 | Raw byte casting, image↔audio, noise-as-anything |
| Fields | 2 | Lazy evaluation, expression language |
| GPU | 1 | wgpu compute for noise/textures |
| Core | 5 | Graph system with lazy/eager evaluation, caching, transforms, serialization, history |
