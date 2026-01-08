# TODO

## Next Up

- [ ] Wavetable oscillators

## Backlog
- [ ] FM synthesis
- [ ] L-systems
- [ ] MIDI support

## Done

- [x] Instancing/scattering (scatter_random, scatter_grid, scatter_sphere, scatter_poisson_2d, Instance)
- [x] Text to paths (Font, text_to_path, text_to_paths_outlined, TextConfig, measure_text)
- [x] Marching cubes (MarchingCubesConfig, sphere/box/torus SDFs, iso-value support)
- [x] glTF import (import_gltf, import_gltf_from_bytes, GltfScene with merge)
- [x] Mesh from curves (extrude_profile, revolve_profile, sweep_profile with caps and scaling)
- [x] Curve lofting (loft, loft_along_path, interpolation, profile helpers)
- [x] Boolean operations on 2D paths (union, intersect, subtract, xor via Sutherland-Hodgman)
- [x] Mesh decimation (edge collapse, DecimateConfig, boundary preservation, max error threshold)
- [x] Image/texture loading as fields (ImageField, WrapMode, FilterMode, bilinear sampling)
- [x] OBJ import/export (import_obj, export_obj, normals, UVs, quad triangulation)
- [x] Laplacian mesh smoothing (smooth, smooth_taubin, preserve boundary, SmoothConfig)
- [x] Animation blending and layering (layers, crossfade, 1D/2D blend trees, additive blending)
- [x] glTF export for meshes (GLB binary, embedded base64, PBR materials)

- [x] NURBS surfaces (tensor product surfaces, sphere/cylinder/torus/cone primitives, tessellation)
- [x] SVG export for 2D vector graphics (paths, shapes, styles, viewBox)
- [x] GPU compute backend for fields (wgpu compute shaders, noise generation)

- [x] NURBS curves (rational B-splines with weights, circle/ellipse/arc primitives)
- [x] Bevy integration examples (mesh generation, skeletal animation with IK)
- [x] Path offset/stroke operations (offset, stroke-to-path, dash patterns, point/tangent at length)
- [x] Edge loop/ring selection and loop cut
- [x] Bevel operations (edge/vertex bevel with configurable amount)
- [x] Mesh boolean operations (BSP-based union, subtract, intersect)
- [x] Vector network (node-based vector editing, region detection)
- [x] Half-edge mesh representation (Catmull-Clark subdivision)
- [x] Easing functions (quad, cubic, elastic, bounce, etc.)
- [x] Audio effects (reverb, chorus, phaser, flanger, distortion)
- [x] Audio graph / signal chain (Chain, Mixer, nodes)
- [x] Color types and gradients (RGB, HSL, HSV, blend modes)
- [x] Spline curves (Bezier, Catmull-Rom, B-spline)
- [x] UV projection (planar, cylindrical, spherical, box)
- [x] Mesh ops (extrude, inset, solidify, weld, flip)
- [x] Particle system (emitters, forces, turbulence)
- [x] Path constraints (bones follow curves)
- [x] Morph targets / blend shapes
- [x] Animation clips and tracks
- [x] IK solvers (CCD, FABRIK)
- [x] Subdivision surfaces (Loop)
- [x] SDF operations (smooth union, intersection, subtraction)
- [x] Domain modifiers (twist, bend, repeat, mirror)
- [x] Envelope generators (ADSR, AR, LFO)
- [x] Expression language parser (expr::Expr)
- [x] Basic texture nodes (Checkerboard, Voronoi, Stripes, Brick, Dots, SDFs)
- [x] Audio effects/filters (LowPass, HighPass, Biquad, Delay, FeedbackDelay)
- [x] Skeleton/bone types
- [x] Derive macro for DynNode
- [x] Field<I, O> trait for lazy evaluation
- [x] Noise functions (Perlin, Simplex)
- [x] Basic oscillators
- [x] 2D path primitives
- [x] Implement basic mesh primitives (box, sphere)
- [x] Set up graph evaluation system
- [x] Define attribute traits
- [x] Review all design decisions for consistency, extract patterns/philosophy to docs/philosophy.md
