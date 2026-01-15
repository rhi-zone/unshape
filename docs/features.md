# Feature Catalog

Index of resin's crates organized by domain. See individual crate docs in `docs/crates/` for use cases and compositions, or run `cargo doc` for API details.

## Audio

| Crate | Description |
|-------|-------------|
| **resin-audio** | Procedural audio synthesis (FM, wavetable, granular, physical modeling), effects (reverb, delay, filters), spectral processing (FFT/STFT), signal routing |
| **resin-easing** | 31 easing functions (quad, cubic, elastic, bounce, etc.) with in/out/inout variants |

## Mesh & 3D Geometry

| Crate | Description |
|-------|-------------|
| **resin-mesh** | 3D mesh operations: primitives, booleans, subdivision, remeshing, terrain/erosion, navigation meshes, architecture generation, SDF |
| **resin-gltf** | glTF 2.0 import/export with PBR materials |
| **resin-pointcloud** | Point cloud sampling, normal estimation, downsampling |
| **resin-voxel** | Dense and sparse voxel grids, morphological ops, greedy meshing |
| **resin-surface** | NURBS tensor product surfaces |
| **resin-spline** | Curves: cubic Bezier, Catmull-Rom, B-spline, NURBS |
| **resin-curve** | Unified `Curve` trait, 2D/3D segment types, arc-length paths |

## 2D Vector Graphics

| Crate | Description |
|-------|-------------|
| **resin-vector** | 2D paths, boolean ops, stroke/offset, path trim, triangulation, vector networks, gradient meshes, text-to-path, hatching, SVG import/export |

## Image & Texture

| Crate | Description | Docs |
|-------|-------------|------|
| **[resin-image](crates/resin-image.md)** | Image as field, convolution, channel ops, color adjustments, distortion, image pyramids, normal maps | [docs](crates/resin-image.md) |

## Color

| Crate | Description |
|-------|-------------|
| **resin-color** | Color spaces (RGB, HSL, HSV, RGBA), gradients, blend modes |

## Animation & Rigging

| Crate | Description |
|-------|-------------|
| **resin-rig** | Skeleton/bones, animation clips, blending/layers, IK (FABRIK, CCD), motion matching, procedural walk, secondary motion (jiggle, follow-through), skinning |
| **resin-easing** | Animation easing functions |
| **resin-motion-fn** | Motion functions: Spring, Oscillate, Wiggle, Eased, Lerp; typed MotionExpr AST; dew expression integration |
| **resin-motion** | 2D motion graphics scene graph: Transform2D with anchor point, Layer hierarchy, blend modes, opacity |

## Physics

| Crate | Description |
|-------|-------------|
| **resin-physics** | Rigid body simulation, colliders, constraints (distance, hinge, spring), soft body FEM, cloth |
| **resin-spring** | Verlet spring systems, particles, angular constraints |
| **resin-particle** | Particle systems with emitters and forces |
| **resin-fluid** | Grid-based stable fluids (2D/3D), SPH particle fluids |

## Procedural Generation

| Crate | Description |
|-------|-------------|
| **resin-noise** | Perlin, simplex noise (2D/3D), fractional Brownian motion |
| **resin-automata** | 1D elementary CA (Wolfram rules), 2D automata (Game of Life, etc.) |
| **resin-procgen** | Maze generation, Wave Function Collapse, road/river networks |
| **resin-rd** | Gray-Scott reaction-diffusion with presets |
| **resin-lsystem** | L-systems with turtle interpretation (2D/3D) |
| **resin-space-colonization** | Space colonization for branching structures (trees, lightning) |

## Fields & Expressions

| Crate | Description |
|-------|-------------|
| **resin-field** | Core `Field<I, O>` trait for lazy spatial computation, combinators |
| **resin-expr-field** | Expression language for fields (math, noise functions) |

## Instancing & Scattering

| Crate | Description |
|-------|-------------|
| **resin-scatter** | Instance placement: random, grid, Poisson disk sampling; stagger timing for animations |

## GPU Acceleration

| Crate | Description |
|-------|-------------|
| **resin-gpu** | wgpu compute backend for noise/texture generation |

## Spatial Data Structures

| Crate | Description |
|-------|-------------|
| **resin-spatial** | Quadtree, octree, BVH, spatial hash, R-tree for efficient spatial queries |

## Cross-Domain

| Crate | Description | Docs |
|-------|-------------|------|
| **[resin-bytes](crates/resin-bytes.md)** | Raw byte casting between numeric types (bytemuck) | [docs](crates/resin-bytes.md) |
| **[resin-crossdomain](crates/resin-crossdomain.md)** | Image↔audio conversion, noise-as-anything adapters | [docs](crates/resin-crossdomain.md) |

## Core & Serialization

| Crate | Description |
|-------|-------------|
| **resin-core** | Graph container, DynNode trait, Value enum, node derive macro |
| **resin-op** | DynOp trait, `#[derive(Op)]` macro, OpRegistry, Pipeline execution |
| **resin-serde** | Graph serialization: SerialGraph format, NodeRegistry, JSON/bincode |
| **resin-history** | History tracking: snapshots (undo/redo), event sourcing (fine-grained) |

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
| Core | 4 | Graph system, DynOp pipelines, serialization, history tracking |
