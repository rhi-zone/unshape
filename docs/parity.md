# Parity Goals

Operation coverage targets from reference tools. Not a source of truth — a benchmark for completeness across domains. Compare against [ops-reference.md](./ops-reference.md) (auto-generated from actual code) to find gaps.

---

## NURBS / Solid Modeling — MoI + Elephant

[Elephant](https://github.com/nkallen/elephant) exposes MoI's full geometry kernel as graph nodes. This is the benchmark for an eventual `unshape-nurbs` crate.

### Curve Creation

| Op | Notes |
|----|-------|
| `line` | |
| `circle` | |
| `circlediameter` | |
| `circletangent` | tangent to existing curves |
| `arc` (center / continue / tangent) | 3 variants |
| `ellipse` | |
| `ellipsecorner` | |
| `ellipsediameter` | |
| `helix` | |
| `polygon` | |
| `polygonedge` | |
| `polygonstar` | |
| `rectangle` | |
| `rectcenter` | |

### Solid / Surface Creation

| Op | Notes |
|----|-------|
| `box` / `boxcenter` | |
| `sphere` | |
| `cylinder` | |
| `cone` | |
| `plane` / `planecenter` | |
| `text` | text as geometry |

### Surface Construction

| Op | Notes |
|----|-------|
| `extrude` | |
| `revolve` | |
| `railrevolve` | revolve along a rail curve |
| `sweep` | one rail + profile |
| `loft` | through profile curves |
| `network` | surface from curve network (XNurbs-style) |
| `nsided` | N-sided patch fill |
| `blend` | smooth surface blend between edges |
| `planarsrf` | fill planar boundary curves |
| `offset` | offset surface |
| `shell` | hollow solid with wall thickness |
| `inset` | inset face |
| `flow` | deform along a curve |
| `project` | project curve onto surface |
| `isocurve` | extract isocurve from surface |
| `silhouette` | extract silhouette curves |

### Boolean / Constructive

| Op | Notes |
|----|-------|
| `booleanunion` | |
| `booleandifference` | |
| `booleanintersection` | |
| `booleanmerge` | union + keep interior edges |
| `intersect` | curve/surface intersections |

### Editing

| Op | Notes |
|----|-------|
| `trim` | trim curves/surfaces |
| `join` | join curves/surfaces into one |
| `separate` | separate joined objects |
| `extend` | extend curve/surface |
| `fillet` | rounded edge between surfaces |
| `chamfer` | angled edge between surfaces |
| `addpoint` | add point on curve |
| `addpointsrf` | add point on surface |
| `rebuildcurve` | refit with different point count / degree |
| `shrinktrimmedsrf` | shrink trimmed surface to trim boundary |
| `flip` | flip surface normal |
| `merge` | merge coplanar faces |

### Transform

| Op | Notes |
|----|-------|
| `move` | |
| `rotate` / `rotateaxis` | |
| `scale` | |
| `mirror` | |
| `align` | align to points/curves |
| `orient` | orient object to reference frame |
| `twist` | twist along axis |
| `arraygrid` | rectangular array |
| `arraycircular` | circular array |
| `arraycurve` | array along curve |
| `arraydir` | directional array |

### Query / Extract

| Op | Notes |
|----|-------|
| `subobject` | extract sub-objects by type and index |
| `object_property` | extract named property |
| `point` | construct a point |
| `copy` | |
| `delete` | |

---

## Graph / Control Flow — Elephant

These are graph-level primitives, not geometry ops. Relevant to `unshape-core`.

| Op | Notes |
|----|-------|
| `subgraph` | nested graph as a node (user-defined ops) |
| `input` / `output` | graph boundary nodes |
| `loopbegin` / `loopend` | iterate over object list, accumulate results |
| `branch` | if/else over data flow |
| `gate` | ternary: `if v then A else B` |

---

## Math / Signal — Elephant

Relevant to `unshape-field` / general field ops.

| Op | Notes |
|----|-------|
| `+`, `-`, `*`, `/`, `%`, `^` | arithmetic |
| `max`, `min`, `abs`, `floor`, `frac` | |
| `clamp`, `lerp` | |
| `smoothstep` | |
| `range` | remap value to different range |
| `rand` | random |
| `noise` | Perlin-like |
| `spikes` | periodic spikes |
| `tendTo` | asymptotic interpolation toward target |
| `accumulate` | stateful increment over time |
| `average` | moving average |
| `trigonometry` | sin, cos, tan, asin, acos, atan |
| `formula` | arbitrary expression (escape hatch) |
| `condition` / `compare` | boolean comparisons |
| `vec2`/`vec3`/`vec4` pack/unpack | |
| `time` | current time as signal source |

---

## NURBS / Solid Modeling — Plasticity

[Plasticity](https://www.plasticity.xyz/) — NURBS/solid modeler for artists, built on Parasolid. Complements MoI/Elephant with its own op vocabulary; overlaps noted below only where Plasticity adds something distinct.

### Curve Creation (beyond MoI)

| Op | Notes |
|----|-------|
| `spline` | through-point curve |
| `control-point curve` | explicit CP placement |
| `bezier` | weighted bezier |
| `spiral` | |
| `arc` (center / three-point / tangent) | |

### Surface Construction (beyond MoI)

| Op | Notes |
|----|-------|
| `patch` | fill from closed curves or edge holes; G0/G1/G2/planar continuity |
| `pipe` | curve/edge → tube with polygon or custom profile |
| `xnurbs` | N-sided and quad-sided surface filling (licensed XNurbs solver) |
| `bridge-surface` | G2 or chamfer blend between surfaces |
| `thicken` | sheet → solid with wall thickness |
| `draft-face` | add draft angle to face (isocline / curve / surface modes) |
| `match-face` | positional + continuity matching between surfaces |
| `raise-surface-degree` | |
| `wrap-curve` | wrap curve onto surface |

### Editing (beyond MoI)

| Op | Notes |
|----|-------|
| `fillet-curve` | fillet on curves (conic / chordal / G2 shape options) |
| `fillet-vertex` | |
| `fillet-shell` | fillet on 3D solid edges |
| `chamfer` | angled edge |
| `offset-face-loop` | |
| `shrink-trimmed-surface` | shrink to trim boundary |
| `imprint` | project curves onto solid/sheet (creates edges) |
| `complete-edge` | extend until intersection |
| `unjoin-faces` / `unjoin-shells` | |

### Transform (beyond MoI)

| Op | Notes |
|----|-------|
| `place` | duplicate + position with scale/rotate/flip |
| `create-curve-instance` | parametric instance with transform/color data |
| `deform-solid` | scale/offset in UVN coordinates |

### Analysis

| Op | Notes |
|----|-------|
| `measure` | straight-line distance |
| `dimension` | live dimensions, update on geometry change |
| `section-analysis` | virtual slice with adjustable plane |
| `continuity-analysis` | G0/G1/G2/G3 validation |
| `curvature-comb` | visualize curvature |

### Continuity Vocabulary

Plasticity uses G0/G1/G2/G3 continuity consistently across surface ops. Worth standardizing on this in `unshape-nurbs`.

---

## 2D Vector — Figma

[Figma](https://figma.com/) — design tool with a graph-based vector network representation. Most relevant: the VectorNetwork model and boolean ops.

### Shape Primitives

| Op | Notes |
|----|-------|
| `rectangle` | per-corner radius, corner smoothing |
| `ellipse` | |
| `polygon` | n-sided regular |
| `star` | pointCount + innerRadius for sharpness |
| `line` | 1D |
| `vector` | arbitrary path / vector network |

### Boolean Ops

| Op | Notes |
|----|-------|
| `union` | outer boundary |
| `subtract` | remove bottom from top |
| `intersect` | keep overlap only |
| `exclude` | keep non-overlapping, remove intersection |
| `flatten` | merge nodes into single vector |

### Vector Network

Figma's internal representation: vertices + segments (with cubic bezier tangents) + regions (fill zones). More general than SVG paths — multiple disconnected paths, non-manifold junctions, per-segment handle mirroring. See prior-art.md entry for design notes.

### Path Operations

| Op | Notes |
|----|-------|
| `outline-stroke` | convert stroke geometry to filled vector |
| `create-text-path` | place text on vector path |

### Stroke Properties

| Property | Notes |
|----------|-------|
| `strokeAlign` | CENTER / INSIDE / OUTSIDE |
| `strokeCap` | NONE / ROUND / SQUARE / ARROW / DIAMOND / TRIANGLE / CIRCLE |
| `strokeJoin` | MITER / BEVEL / ROUND |
| `dashPattern` | dash/gap array |

### Fill Types

| Type | Notes |
|------|-------|
| `solid` | RGBA |
| `gradient-linear` | |
| `gradient-radial` | |
| `gradient-angular` | |
| `gradient-diamond` | |
| `image` | with FILL / FIT / CROP / TILE scale modes |
| `pattern` | tile from source node, rectangular or hexagonal |

### Effects

| Effect | Notes |
|--------|-------|
| `drop-shadow` | color, offset, blur, spread, blend mode |
| `inner-shadow` | |
| `blur` | |
| `progressive-blur` | beta; start/end radius with offset |
| `noise` | monotone / duotone / multitone |
| `texture` | noise-based texture |
| `glass` | refraction, dispersion, depth |

### Layout

| Op | Notes |
|----|-------|
| `auto-layout` | flex-style horizontal/vertical with gap, padding, alignment |
| `layout-grid` | rows / columns / grid overlays |
| `constraints` | MIN / MAX / CENTER / STRETCH / SCALE per axis |

### Blend Modes

16 modes: NORMAL, DARKEN, MULTIPLY, LINEAR_BURN, COLOR_BURN, LIGHTEN, SCREEN, LINEAR_DODGE, COLOR_DODGE, OVERLAY, SOFT_LIGHT, HARD_LIGHT, DIFFERENCE, EXCLUSION, HUE, SATURATION, COLOR, LUMINOSITY, PASS_THROUGH.

### Components / Variables

| Concept | Notes |
|---------|-------|
| `component` | reusable node with properties |
| `variant` | component with variant dimensions |
| `instance` | copy with property overrides |
| `variable` | design token: boolean / number / string / color, with modes for theming |

---

## Audio — VCV Rack Fundamental + Max/MSP

[VCV Rack Fundamental](https://github.com/VCVRack/Fundamental) and [Max/MSP](https://cycling74.com/products/max) cover complementary territory: Fundamental is hardware-modular-style, Max/MSP is functional/mathematical.

### Oscillators

| Op | Notes |
|----|-------|
| `VCO` | sine / triangle / saw / square; PWM; hard+soft sync |
| `wavetable-VCO` | wavetable with interpolation |
| `LFO` | modulation-rate oscillator |
| `phasor` | 0–1 ramp (Max) |
| `noise` | white / pink / red / blue / violet / gray / black |

### Filters

| Op | Notes |
|----|-------|
| `VCF` | 4-pole ladder; LP / HP / BP / notch; resonance + drive |
| `biquad` | configurable filter type |
| `svf` | state-variable filter |
| `allpass` | |
| `onepole` | |
| `comb` | |
| `hilbert` | Hilbert transformer |
| `resonant-bandpass` (`reson~`) | |

### Envelopes

| Op | Notes |
|----|-------|
| `ADSR` | attack / decay / sustain / release |
| `line` | linear ramp |
| `curve` | curved breakpoint envelope |
| `trapezoid` | |
| `sah` | sample-and-hold |

### Dynamics

| Op | Notes |
|----|-------|
| `VCA` | linear / exponential; CV-controlled gain |
| `compressor` | |
| `peak-limiter` | |
| `clip` | hard clip |
| `overdrive` | |
| `degrade` | bit-depth + sample-rate reduction |

### Effects

| Op | Notes |
|----|-------|
| `delay` | with feedback + tone |
| `tapin` / `tapout` | explicit delay buffer read/write |
| `freqshift` | frequency shifting |
| `phaseshift` | |

### Mixing / Routing

| Op | Notes |
|----|-------|
| `mix` | N-channel unity mixer |
| `VCA-mix` | N-channel mixer with per-channel VCA |
| `crossfade` | |
| `selector` | route 1→N or N→1 |
| `matrix-mixer` | M×N routing matrix |
| `send` / `receive` | named signal routing (non-cable) |
| `mute` | |
| `mid-side` | encode/decode L/R ↔ M/S |

### Pitch / Scale

| Op | Notes |
|----|-------|
| `quantizer` | 12-note to scale |
| `octave-shift` | |
| `split` | polyphonic → monophonic |
| `merge` | monophonic → polyphonic |

### Sequencing

| Op | Notes |
|----|-------|
| `SEQ-3` | 3-channel 8-step sequencer |
| `sample-and-hold` | |
| `analog-shift-register` | |
| `random-values` | fixed random voltages |

### FFT / Spectral

| Op | Notes |
|----|-------|
| `fft` / `ifft` | |
| `pfft` | polyphonic FFT |
| `cartesian-to-polar` / `polar-to-cartesian` | |
| `fbinshift` | frequency bin shifting |
| `phase-wrap` | |
| `frame-accumulate` | |
| `frame-average` | |
| `frame-delta` | |

### Sampling / Buffer

| Op | Notes |
|----|-------|
| `buffer` | audio sample storage |
| `play` / `groove` | sample playback |
| `record` | write to buffer |
| `peek` / `poke` | direct buffer access |

### Analysis

| Op | Notes |
|----|-------|
| `scope` | oscilloscope |
| `peak-amp` | |
| `zero-crossing` | |
| `edge-detect` | |
| `threshold` | |

### Math (signal-rate)

| Op | Notes |
|----|-------|
| `+`, `-`, `*`, `/`, `%`, `^` | |
| `abs`, `floor`, `frac`, `round` | |
| `sin`, `cos`, `tan`, `atan2` | |
| `exp`, `log`, `sqrt`, `pow` | |
| `clamp`, `lerp`, `smoothstep` | |
| `delta` | sample-to-sample difference |
| `downsamp` | downsampling |
| `rampsmooth` | smooth value changes |

---

## Procedural Geometry — Blender Geometry Nodes

[Geometry Nodes](https://docs.blender.org/manual/en/latest/modeling/geometry_nodes/index.html) (Blender 5.1) — the benchmark for procedural mesh/curve/volume/instance operations.

### Mesh Primitives

| Op | Notes |
|----|-------|
| `grid` | |
| `cube` | |
| `circle` | |
| `cone` | |
| `cylinder` | |
| `icosphere` | |
| `uv-sphere` | |
| `line` | |

### Mesh Operations

| Op | Notes |
|----|-------|
| `extrude-mesh` | |
| `inset-faces` | |
| `bevel-mesh` | |
| `bridge-edge-loops` | |
| `subdivide-mesh` | |
| `merge-by-distance` | |
| `dissolve-edges` | |
| `flip-faces` | |
| `mesh-boolean` | union / difference / intersect |
| `delete-geometry` | |
| `duplicate-elements` | |
| `mesh-to-curve` | |
| `mesh-to-points` | |
| `mesh-to-volume` | |
| `shade-smooth` | |

### Mesh Topology Query

| Op | Notes |
|----|-------|
| `corners-of-edge` | |
| `corners-of-face` | |
| `corners-of-vertex` | |
| `face-area` | |
| `face-neighbors` | |
| `edge-neighbors` | |
| `is-face-planar` | |
| `shortest-path-on-mesh` | |

### Curve Primitives

| Op | Notes |
|----|-------|
| `arc` | |
| `bezier-segment` | |
| `circle` | |
| `line` | |
| `quadrilateral` | |
| `spiral` | |
| `star` | |

### Curve Operations

| Op | Notes |
|----|-------|
| `resample-curve` | |
| `trim-curve` | |
| `fillet-curve` | |
| `curve-to-mesh` | |
| `curve-to-points` | |
| `set-curve-radius` | |
| `set-curve-tilt` | |
| `set-handle-positions` | |
| `set-handle-types` | |
| `deform-curves-on-surface` | |
| `interpolate-curves` | |
| `curve-length` | |

### Volume / Grid

| Op | Notes |
|----|-------|
| `mesh-to-volume` | |
| `points-to-volume` | |
| `volume-to-mesh` | |
| `mesh-to-sdf-grid` | |
| `points-to-sdf-grid` | |
| `voxelize-grid` | |
| `grid-dilate-erode` | |
| `grid-mean` / `grid-median` | |
| `grid-gradient` / `grid-curl` / `grid-divergence` / `grid-laplacian` | |
| `sdf-grid-boolean` | |
| `sample-grid` | |

### Instances

| Op | Notes |
|----|-------|
| `instance-on-points` | |
| `realize-instances` | |
| `instances-to-points` | |

### Points / Point Cloud

| Op | Notes |
|----|-------|
| `distribute-points-on-faces` | |
| `distribute-points-on-edges` | |
| `distribute-points-in-volume` | |
| `points-to-curves` | |

### Fields / Attributes

| Op | Notes |
|----|-------|
| `capture-attribute` | |
| `store-named-attribute` | |
| `named-attribute` | |
| `remove-named-attribute` | |
| `attribute-statistic` | |
| `accumulate-field` | |
| `field-at-index` | |
| `sample-index` | |
| `sample-nearest` | |
| `sample-nearest-surface` | |
| `blur-attribute` | |
| `copy-to-points` | |

### Control Flow

| Op | Notes |
|----|-------|
| `simulation-zone` | stateful iteration (input + output boundary) |
| `repeat-zone` | explicit loop (Blender 5.0+) |
| `switch` | |
| `if-else-branch` | |

### Math

| Op | Notes |
|----|-------|
| `math` | add / sub / mul / div / pow / log / sqrt / abs / floor / ceil / fract / mod / wrap / pingpong / smoothstep / clamp / min / max / sign / compare / snap |
| `map-range` | remap with optional clamp |
| `mix` | for scalars, vectors, colors, rotations |
| `vector-math` | dot / cross / normalize / length / distance / reflect / refract / project / faceforward / scale / wrap |
| `vector-rotate` | |
| `rotation-math` | |
| `boolean-math` | AND / OR / NOT / NAND / NOR / XOR / XNOR |
| `random-value` | float / int / vector / bool with seed |

---

## Procedural Geometry — Houdini SOPs

[Houdini SOP docs](https://www.sidefx.com/docs/houdini/nodes/sop/). The deepest procedural geometry reference. ~800+ nodes total including Labs.

### Geometry Creation

| Op | Notes |
|----|-------|
| `add` | construct points/polygons explicitly |
| `box` | |
| `circle` | |
| `curve` | |
| `font` | text as geometry |
| `grid` | |
| `l-system` | L-system grammar to geometry |
| `sphere` | |
| `torus` | |

### Topology / Boolean

| Op | Notes |
|----|-------|
| `boolean` | union / difference / intersect |
| `boolean-fracture` | fracture along a cutter |
| `blast` | delete by group |
| `clean` | remove degenerates |
| `clip` | clip by plane |
| `delete` | |
| `dissolve` | remove edges/points |
| `divide` | triangulate / convex |
| `extrude` | |
| `facet` | faceting, cusp edges |
| `fuse` | weld nearby points |
| `hole` | punch holes in polygons |
| `join` | join edges |
| `assemble` | pack pieces into packed prims |

### Edge Operations

| Op | Notes |
|----|-------|
| `edge-collapse` | |
| `edge-divide` | subdivide edges |
| `edge-equalize` | equalize edge lengths |
| `edge-flip` | flip shared edge |
| `edge-fracture` | fracture along edges |
| `edge-relax` | relax edge positions |
| `edge-straighten` | |
| `edge-cusp` | crease edges |

### Deformation

| Op | Notes |
|----|-------|
| `bend` | |
| `blend-shapes` | |
| `bone-deform` | skeleton deformation |
| `bulge` | |
| `delta-mush` | smooth deformation preserving detail |
| `lattice` | cage deformation |
| `magnet` | attract/repel deformation |
| `wire-deform` | deform by wires |
| `cloth-deform` | cloth capture result |
| `fem-deform` | finite element deformation |
| `vellum-*` | constraint-based soft body |

### UV / Texture

| Op | Notes |
|----|-------|
| `uvunwrap` | automatic UV unwrapping |
| `uvproject` | planar/cylindrical/spherical projection |
| `uvlayout` | pack UV islands |
| `uvpelt` | pelt (seam-based) unwrap |
| `uvtransform` | transform UVs |
| `uvfuse` | weld UV seams |

### Attribute System

| Op | Notes |
|----|-------|
| `attribute-create` | |
| `attribute-delete` | |
| `attribute-rename` | |
| `attribute-copy` | |
| `attribute-transfer` | transfer between geometries |
| `attribute-blur` | |
| `attribute-noise` | |
| `attribute-randomize` | |
| `attribute-remap` | remap values via ramp |
| `attribute-promote` | change attribute class (point→prim, etc.) |
| `attribute-vop` | VEX shader over attributes |
| `attribute-wrangle` | raw VEX code |
| `point-wrangle` | VEX per-point |
| `prim-wrangle` | VEX per-primitive |

### Groups / Selection

| Op | Notes |
|----|-------|
| `group-create` | by expression, range, bounding |
| `group-combine` | boolean ops on groups |
| `group-expand` | grow by adjacency |
| `group-invert` | |
| `group-paint` | paint-based selection |
| `group-promote` | change group class |
| `group-transfer` | transfer groups between geos |

### Volume / VDB

| Op | Notes |
|----|-------|
| `convert-vdb` | mesh↔VDB |
| `vdb-combine` | union / intersection / difference |
| `vdb-smooth` | |
| `vdb-reshape` | dilate / erode / close / open |
| `vdb-analysis` | gradient / curl / divergence / laplacian |
| `vdb-advect` | advect points/VDB through velocity field |
| `volume-blur` | |
| `volume-vop` | VEX shader over volumes |
| `volume-wrangle` | |
| `isooffset` | create SDF from geometry |
| `cloud-noise` | |

### Heightfield / Terrain

| Op | Notes |
|----|-------|
| `heightfield` | create from image/function |
| `heightfield-erode` | hydraulic erosion |
| `heightfield-erode-hydro` | |
| `heightfield-erode-thermal` | |
| `heightfield-erode-precipitation` | |
| `heightfield-noise` | |
| `heightfield-blur` | |
| `heightfield-mask-by-feature` | mask by slope/occlusion/etc |
| `heightfield-scatter` | scatter points by density |
| `heightfield-terrace` | |
| `heightfield-slump` | |
| `heightfield-remap` | |
| `heightfield-resample` | |
| `heightfield-project` | project geo onto height |

### Copy / Instancing

| Op | Notes |
|----|-------|
| `copy-to-points` | instance geometry on points |
| `copy-and-transform` | array with transform |
| `instance` | packed prim instancing |
| `replicate` | |

### Analysis / Measure

| Op | Notes |
|----|-------|
| `distance-from-geometry` | |
| `distance-along-geometry` | |
| `measure` | area / perimeter / curvature |
| `connectivity` | label connected components |
| `cluster` | cluster points |
| `graph-color` | color graph vertices |

### File / Cache

| Op | Notes |
|----|-------|
| `file` | read/write geometry |
| `file-cache` | disk cache |
| `alembic` | Alembic I/O |
| `fetch` | pull from other network |

### Simulation Coupling

| Op | Notes |
|----|-------|
| `dop-import` | import DOP fields/geo |
| `dop-network` | embed dynamics |
| `collision-source` | prep geo for simulation |
| `vellum-*` | cloth, hair, softbody |
| `particle-fluid-surface` | FLIP surface |

---

## Shaders / Fields / Signals — Houdini VOPs + CHOPs + COPs

### VOPs (VEX OPerators — shader/field nodes)

| Category | Ops |
|----------|-----|
| Math | add / sub / mul / div / mod / pow / abs / floor / ceil / fract / clamp / fit / smooth / sign |
| Trig | sin / cos / tan / asin / acos / atan2 |
| Vector | dot / cross / normalize / length / distance / reflect / refract |
| Matrix | build / invert / transpose / multiply / extract |
| Noise | Perlin / flow / cellular / Worley / anti-aliased variants |
| Patterns | checkerboard / gingham / burlap / bricker |
| Shading | Lambert / Phong / BSDF / Fresnel / displacement / environment |
| Geometry | ray-cast / point-in-mesh / attribute import/export |
| Control flow | if / for / while / foreach |
| Binding | bind / import / export / global |

### CHOPs (Channel OPerators — signal/audio/animation)

| Category | Ops |
|----------|-----|
| Generation | oscillator / noise / wave / pulse / LFO / constant |
| Audio | audio-in / band-EQ / MIDI-in/out / pitch / beat / spectrum / spatial-audio |
| Filter | filter / lag / spring / jiggle / limit / parametric-EQ / envelope |
| Animation | blend / composite / layer / sequence / keyframe / IK-solver / pose |
| Analysis | slope / area / count / footplant / voice-split / dynamic-warp |
| Timing | resample / stretch / delay / time-shift / time-range |
| Constraint | look-at / parent / path / surface / object / blend / sequence |
| Data I/O | file / FBX / fetch-channels / fetch-parameters / export |
| Math | math / logic / function / expression / channel-wrangle / channel-VOP |
| Routing | switch / null / rename / reorder / shuffle / merge / copy / cycle |

### COPs (Composite OPerators — image; legacy, Copernicus in 20.5+)

| Category | Ops |
|----------|-----|
| Generation | color / file / geometry / noise / ramp / shape / font |
| Composite | over / under / add / multiply / screen / atop / subtract |
| Color | brightness / color-correct / gamma / HSV / levels / lookup / contrast |
| Blur | blur / defocus / median / radial-blur / streak-blur / velocity-blur |
| Effects | emboss / edge-detect / sharpen / grain / glow / god-rays |
| Distort | transform / warp / flip / corner-pin / deform |
| Keying | chroma-key / luma-key / luma-matte |
| Channel | channel-copy / convert / extract / merge / rename |
| Morphology | dilate-erode / expand |
| Sequence | blend / extend / tile / reverse / time-warp / trim / wipe |

---

## Parametric Geometry — Grasshopper

[grasshopperdocs.com](https://grasshopperdocs.com/). ~500 built-in components; 10,000+ across community plugins.

### Curve

| Op | Notes |
|----|-------|
| `line` / `circle` / `arc` / `ellipse` / `polygon` / `rectangle` | primitives |
| `interpolate` | through-point NURBS |
| `nurbs-curve` | control-point NURBS |
| `polyline` | |
| `blend-curve` | |
| `offset-curve` | |
| `extend-curve` | |
| `fillet-curve` | |
| `project-curve` | |
| `rebuild-curve` | |
| `divide-curve` | by count |
| `divide-length` | by length |
| `shatter` | split at parameters |
| `end-points` | |
| `evaluate-curve` | sample at parameter |
| `curve-curvature` | |

### Surface

| Op | Notes |
|----|-------|
| `loft` | |
| `extrude` | |
| `sweep` | |
| `revolution` | |
| `pipe` | |
| `ruled-surface` | |
| `edge-surface` | from 2-4 edge curves |
| `plane-surface` | |
| `bounding-box` | |
| `offset-surface` | |
| `fillet-edge` | |
| `cap-holes` | |
| `extend-surface` | |
| `rebuild-surface` | |
| `divide-surface` | by UV count |
| `evaluate-surface` | sample at UV |
| `surface-curvature` | |
| `deconstruct-brep` | extract faces/edges/vertices |

### Boolean / Intersect

| Op | Notes |
|----|-------|
| `solid-union` | |
| `solid-difference` | |
| `solid-intersection` | |
| `solid-split` | |
| `region-union` / `region-difference` / `region-intersection` | 2D |
| `curve-curve` | intersection |
| `surface-curve` | |
| `brep-brep` | |
| `surface-split` | |

### Mesh

| Op | Notes |
|----|-------|
| `mesh-plane` / `mesh-box` / `mesh-sphere` | |
| `delaunay-mesh` | from points |
| `voronoi` | |
| `quad-remesh` | |
| `triangulate` | |
| `mesh-join` / `mesh-weld` | |
| `flip-normals` | |
| `deconstruct-mesh` | |

### Transform

| Op | Notes |
|----|-------|
| `move` / `rotate` / `mirror` / `scale` | |
| `orient` | from frame to frame |
| `linear-array` / `rectangular-array` / `polar-array` | |
| `surface-morph` | morph geometry to surface |
| `box-morph` | |
| `flow-along-surface` | |

### Sets / Data

| Op | Notes |
|----|-------|
| `list-item` / `list-length` / `reverse` / `sort` / `unique` | |
| `dispatch` | split by predicate |
| `weave` | interleave lists |
| `series` / `range` / `random` | |
| `flatten` / `graft` | tree structure ops |
| `tree-item` / `tree-paths` / `tree-branch` | |
| `cull-duplicates` | |

---

## Procedural Textures — Substance Designer

[Substance Designer node library](https://helpx.adobe.com/substance-3d-designer/substance-compositing-graphs/nodes-reference-for-substance-compositing-graphs/node-library.html).

### Atomic Nodes (core primitives)

| Op | Notes |
|----|-------|
| `blend` | with blend modes + mask |
| `curve` | remap via spline |
| `levels` | tone remap |
| `warp` | displace by gradient |
| `directional-warp` | displace in a direction |
| `blur` | box blur |
| `sharpen` | |
| `normal` | height→normal |
| `distance` | distance field |
| `transformation-2d` | move/rotate/scale with tiling |
| `channel-shuffle` | rearrange RGBA |
| `uniform-color` | solid color/value |
| `gradient` | interpolated gradient |
| `gradient-map` | grayscale→color via gradient |
| `fx-map` | Markov chain procedural subdivision |
| `pixel-processor` | custom per-pixel op |
| `bitmap` / `svg` / `text` | external image sources |

### Noise Generators

| Op | Notes |
|----|-------|
| `perlin-noise` | |
| `simplex-noise` | |
| `worley-noise` / `voronoi` | cellular |
| `voronoi-fractal` | |
| `cells-1` through `cells-4` | |
| `clouds-1` through `clouds-3` | |
| `3d-perlin` / `3d-simplex` / `3d-worley` / `3d-voronoi` | 3D variants |

### Pattern Generators

| Op | Notes |
|----|-------|
| `shape` | square / disc / bell / gaussian / pyramid / brick / waves / crescent / capsule / cone / hemisphere / etc. |
| `tile-generator` | configurable grid with per-tile variation |
| `tile-sampler` | arrange input as tiles |
| `triangle-grid` | |
| `splatter-circular` | ring-based arrangement |
| `weave-generator` | fabric weave |
| `stripes` | |
| `starburst` | |

### Filters — Adjustments

| Op | Notes |
|----|-------|
| `auto-contrast` / `auto-levels` | |
| `brightness-contrast` | |
| `hue-saturation` | |
| `channel-mixer` | |
| `color-balance` | shadows/mids/highlights |
| `invert` | |
| `posterize` | |
| `quantize-color` | |
| `threshold` | |
| `histogram-equalize` | |
| `histogram-range` | |
| `remap` | |
| `clamp` | |

### Filters — Blurs

| Op | Notes |
|----|-------|
| `blur-hq` | Gaussian |
| `directional-blur-hq` | |
| `anisotropic-blur` | |
| `median-blur` | |
| `motion-blur` | |
| `kuwahara` | painterly |
| `pixelize` | mosaic |

### Filters — Effects

| Op | Notes |
|----|-------|
| `bevel` | height-based bevel |
| `curvature` | curvature map from normal |
| `edge-detect-hq` | |
| `swirl` | |
| `vector-warp` | warp by vector/color map |
| `non-uniform-directional-warp` | |

### Filters — Transforms

| Op | Notes |
|----|-------|
| `polar-coordinates` | cartesian ↔ polar |
| `radial-symmetry` | |
| `tile-and-offset` | |
| `trapezoid` | |
| `triplanar-projection` | |
| `mirrored-tiles` | |

### Normal Map

| Op | Notes |
|----|-------|
| `height-to-normal-hq` | |
| `normal-blend` | |
| `normal-combine` | overlay |
| `normal-to-height-hq` | |

### Tiling

| Op | Notes |
|----|-------|
| `make-it-tile` | seamless from any input |
| `make-it-tile-patch` | grid-based |
| `tile-break` | break visible repetition |

### Blending Modes (on Blend node)

Normal, Add, Subtract, Multiply, Screen, Overlay, Soft Light, Hard Light, Divide, Max, Min, Color Dodge, Color Burn, Linear Dodge, Linear Burn, Vivid Light, Exclusion, Lighten, Darken, Difference, Dissolve.

---

## Realtime AV — TouchDesigner

[docs.derivative.ca](https://docs.derivative.ca/). Organized by operator family.

### TOPs (Texture OPs — 2D image)

| Category | Ops |
|----------|-----|
| Generation | constant / circle / noise / ramp / SVG / function |
| Processing | blur / sharpen / emboss / edge / bloom / threshold / monochrome |
| Color | channel-mix / HSV-adjust / level / lookup / limit |
| Composite | over / under / add / multiply / screen / difference / cross |
| Keying | chroma-key / luma-key / RGB-key / matte |
| Transform | transform / crop / fit / flip / mirror / corner-pin / lens-distort / tile / layout |
| Analysis | analyze / optical-flow / blob-track / spectrum |
| Feedback | feedback / cache / time-machine |
| I/O | movie-file-in/out / video-device-in/out / NDI-in/out / screen-grab |
| Custom | GLSL / script |

### CHOPs (Channel OPs — signal/audio/animation)

| Category | Ops |
|----------|-----|
| Generation | constant / pattern / LFO / noise / wave / audio-oscillator |
| Audio | audio-device-in / audio-file-in / audio-filter / audio-dynamics / audio-band-EQ / beat / spectrum |
| Filter | math / logic / filter / lag / spring / limit / slope |
| Animation | keyframe / lookup / trail / inverse-curve / S-curve / spring |
| Timing | clock / count / delay / hold / resample / stretch / trigger / timer |
| Input | MIDI-in / OSC-in / keyboard / mouse / joystick / gamepad |
| Routing | select / switch / merge / shuffle / join / reorder / delete |
| Analysis | analyze / beat |

### SOPs (Surface OPs — geometry)

| Category | Ops |
|----------|-----|
| Primitives | box / sphere / torus / circle / grid / cone / tube |
| Manipulation | merge / copy / extrude / boolean / subdivide / weld / clip / facet |
| Deformation | transform / magnet / noise / lattice / trail |
| Conversion | trace / font / L-system / DAT-to / CHOP-to |
| Particles | particle / sweep / skin / loft |

### DATs (Data OPs — tables/text/network)

| Category | Ops |
|----------|-----|
| I/O | OSC-in/out / TCP/IP / UDP / WebSocket / serial / NDI / Art-Net |
| Manipulation | table / text / convert / merge / select / sort / transpose / insert / lookup |
| Scripting | script / execute / DAT-execute / CHOP-execute / panel-execute |
| Web | web-client / web-server / WebRTC / JSON / XML |

### COMPs (Component OPs — containers/UI)

| Category | Ops |
|----------|-----|
| 3D Objects | camera / light / geometry / bone / ambient-light / environment-light |
| UI Panels | button / container / field / slider / list / table / OP-viewer |
| Control | engine / time / animation / replicator / window |

---

## Image Compositing — Nuke / Natron

[Nuke reference](https://learn.foundry.com/nuke/content/reference_guide.html).

### Color Correction

| Op | Notes |
|----|-------|
| `grade` | lift / gain / multiply / offset |
| `color-correct` | master + shadow/mid/highlight ranges |
| `exposure` | |
| `gamma` | |
| `saturation` | |
| `hue-shift` | |
| `hue-correct` | per-hue curve correction |
| `color-lookup` | via spline curves |
| `color-matrix` | matrix color transform |
| `log2lin` / `lin2log` | log encoding |
| `OCIO-*` | OpenColorIO transforms |
| `hist-EQ` | histogram equalization |
| `posterize` | |
| `clamp` | |
| `invert` | |
| `soft-clip` | |

### Filter

| Op | Notes |
|----|-------|
| `blur` | |
| `defocus` | lens defocus |
| `z-defocus` | depth-of-field from Z pass |
| `dir-blur` | directional |
| `edge-blur` | |
| `motion-blur` | |
| `vector-blur` | from motion vector pass |
| `sharpen` / `soften` | |
| `median` | |
| `laplacian` | |
| `bilateral` | |
| `convolve` | custom kernel |
| `denoise` / `denoise-AI` | |
| `erode` | |
| `glow` | |
| `god-rays` | |
| `emboss` | |

### Merge / Composite

| Op | Notes |
|----|-------|
| `merge` | all blend modes: over, under, plus, multiply, screen, max, min, in, out, atop, etc. |
| `keymix` | merge with key/mask input |
| `premult` / `unpremult` | |
| `dissolve` | timed crossfade |
| `z-merge` | merge using Z depth |
| `copy` | copy channels |
| `contact-sheet` | tile multiple inputs |

### Transform

| Op | Notes |
|----|-------|
| `transform` | |
| `crop` | |
| `reformat` | resize/change format |
| `corner-pin` | 4-point warp |
| `lens-distortion` | |
| `st-map` | warp by UV map |
| `spline-warp` | warp by splines |
| `grid-warp` | |
| `mirror` | |
| `tile` | |
| `stabilize` | tracking-based |
| `camera-shake` | |

### Keying

| Op | Notes |
|----|-------|
| `keylight` | |
| `IBK-color` + `IBK-gizmo` | IBK keyer |
| `primatte` | |
| `chroma-keyer` | |
| `luma-keyer` | |
| `hue-keyer` | |
| `difference` | diff matte |

### Channel Ops

| Op | Notes |
|----|-------|
| `shuffle` / `shuffle-copy` | reroute channels |
| `channel-merge` | combine layers |
| `copy` | copy specific channels |

### Draw / Roto

| Op | Notes |
|----|-------|
| `roto` | shape-based mattes |
| `roto-paint` | paint + roto |
| `text` | |
| `sparkles` | particle sparkles |
| `lens-flare` | |
| `grain` | film grain |

### Time

| Op | Notes |
|----|-------|
| `time-offset` | |
| `frame-hold` | |
| `time-blur` | temporal blur |
| `time-clip` | retime / reverse / loop |

### Deep Compositing

| Op | Notes |
|----|-------|
| `deep-color-correct` | |
| `deep-crop` | |
| `deep-holdout` | |
| `deep-merge` | |
| `deep-to-image` | flatten deep |
| `deep-from-image` | |
| `deep-expression` | |

---

## Animation — Rive + Cavalry

### Rive

[rive.app/docs](https://rive.app/docs/)

| Op / Concept | Notes |
|--------------|-------|
| `timeline-animation` | keyframe with easing curves |
| `state-machine` | animation graph with inputs and transitions |
| `bone` | skeletal hierarchy |
| `mesh-deform` | vertex weighting to bones |
| `blend-state` | mix multiple animations |
| `trim-path` | animate stroke draw-on |
| `clipping-path` | mask by shape |
| `constraint: IK` | inverse kinematics chain |
| `constraint: distance` | |
| `constraint: transform` | copy position/rotation/scale |
| `constraint: follow-path` | orient along curve |
| `text-modifier` | per-glyph position/rotation/scale/opacity with range + falloff |
| `draw-order-animation` | animate layer depth |
| `data-binding` | drive properties from external data |

### Cavalry

[cavalry.studio/docs](https://cavalry.studio/docs/)

| Op / Concept | Notes |
|--------------|-------|
| `timeline-animation` | keyframe |
| `procedural-animation` | single value drives whole system |
| **Shape types** | |
| `basic-shape` | ellipse / rectangle / polygon / star |
| `path` | bezier with contours + holes |
| `mesh-shape` | custom mesh |
| `text-shape` | per-char / per-word / per-line hierarchy |
| `duplicator` | procedural duplication |
| `layout-shape` | procedural positioning |
| **Behaviours** | |
| `oscillator` | cyclic value generation |
| `spring` | physics-based animation |
| `stagger` | sequential timing |
| `motion-stretch` | dynamic blur/deformation |
| `travel-deformer` | path-based motion |
| `bend-deformer` | circular arc bend |
| `lattice-deformer` | grid-based deformation |
| `four-point-warp` | corner-based perspective |
| `pinch` | |
| `noise` | procedural noise behaviour |
| `falloff` | smooth weight distribution |
| **Path ops** | |
| `pathfinder` | boolean on paths |
| `resample-path` | redistribute points |
| `chop-path` | segment into sections |
| `path-offset` | parallel path |
| `reverse-path` | |

---

## Audio Synthesis — SuperCollider UGens + FAUST

### Oscillators

| Op | Notes |
|----|-------|
| `sine` | band-limited |
| `saw` | band-limited |
| `pulse` / `square` | band-limited |
| `triangle` | |
| `phasor` | 0–1 ramp |
| `impulse` | impulse train |
| `LF-saw` / `LF-pulse` / `LF-triangle` | non-band-limited (SC: LFSaw/LFPulse/LFTri) |
| `var-saw` | variable saw/triangle morph |
| `wavetable` | table lookup |
| `VOsc` | variable-waveform wavetable |
| `polyBLEP-saw/square/triangle` | polyBLEP anti-aliasing |
| `DPW-saw` | differentiated polynomial |
| `quadrature` | sine+cosine pair |
| `FM` | frequency modulation |
| `white-noise` | |
| `pink-noise` | |
| `brown-noise` | |
| `dust` | sparse random impulses |
| `LF-noise-0/1/2` | stepped / linear / smooth random |

### Filters

| Op | Notes |
|----|-------|
| `LPF` / `HPF` / `BPF` / `notch` | Butterworth |
| `resonz` | resonant bandpass |
| `moog-ladder` | |
| `SVF` | state variable (LP/HP/BP/notch in one) |
| `biquad` | direct coefficient control |
| `allpass` | |
| `one-pole` | |
| `comb` | |
| `formlet` | FOF grain-like filter |
| `hilbert` | Hilbert transform |
| `linkwitz-riley-4` | crossover |
| `peak-EQ` / `low-shelf` / `high-shelf` | parametric EQ |
| `spectral-tilt` | arbitrary spectral slope |

### Envelopes

| Op | Notes |
|----|-------|
| `AR` | attack/release |
| `ASR` | |
| `ADSR` | |
| `AHDSR` | with hold |
| `line` | linear ramp |
| `curve` | curved breakpoint |
| `trapezoid` | |
| `DX7-envelope` | 4-point breakpoint |
| `smooth-envelope` | exponential attack/release |

### Dynamics

| Op | Notes |
|----|-------|
| `compressor` | feed-forward / feedback / hybrid |
| `limiter` | 1176-style |
| `lookahead-limiter` | |
| `expander` | |
| `gate` | threshold/attack/hold/release |
| `clip` | hard clip |
| `soft-clip-quadratic` | |
| `wavefold` | wavefolding distortion |
| `overdrive` | |
| `bit-crusher` | |
| `degrade` | sample-rate reduction |

### Effects

| Op | Notes |
|----|-------|
| `delay-N/L/C` | no-interp / linear / cubic |
| `comb-filter` | feedback delay |
| `allpass-delay` | |
| `reverb: freeverb` | |
| `reverb: zita` | 8×8 FDN |
| `reverb: dattorro` | |
| `reverb: jc-reverb` | Chowning 1972 |
| `reverb: spring` | |
| `chorus` / `flanger` / `phaser` | |
| `pitch-shift` | |
| `freq-shift` | |
| `doppler` | |
| `tape-stop` | |

### Spectral / FFT

| Op | Notes |
|----|-------|
| `FFT` / `IFFT` | |
| `PV-mag-smooth` | spectral smoothing |
| `PV-mag-shift` | shift frequency bins |
| `PV-phase-shift` | |
| `PV-rect-comb` | spectral comb |
| `goertzel` | single-frequency DFT |
| `octave-analyzer` | filterbank |

### Analysis

| Op | Notes |
|----|-------|
| `amplitude-follower` | |
| `RMS-envelope` | |
| `zero-crossing-rate` | |
| `pitch-tracker` | |
| `spectral-centroid` | |
| `peak-hold` | |
| `edge-detect` | zero-crossing edge |

### Utilities

| Op | Notes |
|----|-------|
| `sample-and-hold` | |
| `latch` | |
| `downsample` | |
| `ramp-smooth` | |
| `midi-to-hz` / `db-to-linear` | conversions |
| `panning` | uniform / constant-power |
| `mid-side` | encode/decode |
| `dry-wet-mixer` | |
| `sliding-mean/RMS/max` | windowed statistics |
