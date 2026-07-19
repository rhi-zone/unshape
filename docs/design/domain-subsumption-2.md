# Domain Subsumption: Three More Tools

Continuation of `domain-subsumption.md` (Excel, ClickHouse, Flash/Animate, DaVinci Resolve, OBS,
Plasticity). Same method: separate *branded as* from *actually used for*, derive the computation
model, check against crate source (not assumed), name what's missing. This round covers
Procreate, Toon Boom Harmony, and Blender — chosen because they're painting, rigged 2D animation,
and "everything" 3D respectively, and because two of the three original doc's proposed synthesis
primitives (`Table`, `Timeline`, `LiveSource`) have since shipped as real crates
(`unshape-table`, `unshape-timeline`, `unshape-live`), which changes what counts as a gap here —
noted inline where it matters.

## Baseline delta since the first doc

`unshape-table`, `unshape-timeline`, and `unshape-live` now exist
(`crates/unshape-table/src/lib.rs`, `crates/unshape-timeline/src/lib.rs`,
`crates/unshape-live/src/lib.rs`). `unshape-live::LiveSource` is pull-polled, not push
(`fn poll(&mut self) -> Option<Self::Output>`, `crates/unshape-live/src/source.rs`) — the host
calls `poll` on its own schedule; this is exactly the shape Synthesis #3 specified. `Timeline` /
`Track` / `ClipInstance` / `TimeMap` / `Transition` exist per `crates/unshape-timeline/src/*.rs`
matching Synthesis #2's design. `unshape-motion-fn::Keyframes<T>` (`crates/unshape-motion-fn/src/
lib.rs:320`) implements `Field<f32, T>` for `f32`/`Vec2`/`Vec3` via `impl_field_for_motion!` — the
interpolation math Synthesis's `AnimatedProperty<T>` needed already exists generically. What's
still missing is the *wiring*: nothing binds a `Keyframes<T>` to a specific `Scene`/`Layer` field
by `(LayerId, PropertyId)` so a static layer pays nothing and an animated one doesn't need a
bespoke struct per property. `StateMachine<S>` (Synthesis #4) and `unshape-scene-state` were not
found in the crate list — still a gap, unless it now lives under a different name (not checked
exhaustively here; flagged as open).

Also relevant: `unshape-editor` (`crates/unshape-editor/src/lib.rs`) exists — "a single live
texture is produced by an ordered list of image modifiers... shown in two co-equal projections: an
editable op-stack row... and an editable formula view." This is a working instance of the
projection model these three tools' UI patterns get compared against below, not a hypothetical.

## Procreate

**Branded as:** the professional digital painting app for iPad — "it feels like real media."

**Actually used for:** raster painting with a pressure/tilt/velocity-sensitive brush engine
(stamping, texture, grain, dynamics, custom brush authoring in Brush Studio); a layer stack
(blend modes, opacity, clipping masks, alpha lock, groups, reference layer); non-destructive
adjustment operations (Hue/Saturation, Curves, Gaussian Blur — applied in place, not as a layer
stack of adjustment nodes the way Photoshop's adjustment layers work); frame-by-frame raster
animation with onion skinning ("Animation Assist"); QuickShape (stroke → geometric primitive
snap); selections with feather.

**Computation model:** the canvas is a stack of raster buffers (`Layer` = pixel grid + blend mode
+ opacity + mask + clip-to-below flag + group membership), composited top-down — the same
per-pixel blend algebra as OBS's scene composite or Resolve's track stack, but operating on
whole-canvas pixel buffers rather than time-sampled clips. A brush stroke is not a single
transform applied to the canvas; it's a discrete polyline of input samples
`(position, pressure, tilt, azimuth, velocity, timestamp)` from the tablet driver, and the brush
engine stamps a small procedural or bitmap "nib" image repeatedly along that polyline at a
spacing interval, with each stamp's size/opacity/rotation/color/scatter computed from the
sample's pressure/tilt/velocity/random-jitter via per-brush dynamics curves — i.e., a function
`StrokeSample -> StampParams`, applied at accretion time, each stamp mutating the layer's pixel
buffer in place. This is fundamentally different from a filter (`Blur { radius }` reads a whole
image and produces a whole image); a stroke is *stateful accretion* onto a persistent buffer,
order-dependent, and its "undo" is a log entry, not a recomputation from parameters. Onion
skinning composites the current frame with neighbors at reduced opacity — an instance of the
`Timeline`/`ClipInstance` compositing model from the first doc, at per-discrete-raster-frame
granularity rather than continuous-field-sample granularity.

**Maps to unshape subsystems:** `unshape-image`'s `ImageField` + `Composite`/`BlendMode`
(`crates/unshape-image/src/composite.rs`) is exactly the per-pixel blend algebra the layer stack
needs — `composite(base, overlay, mode, opacity)` is Procreate's layer-merge math, one call per
layer in the stack. `unshape-timeline`'s `ClipInstance`/`Track` can drive onion-skinning and
frame sequencing once each animation frame is modeled as a clip source. `unshape-history`'s
snapshot + event sourcing is the right shape for undo (Procreate's undo is a linear stroke log,
not a graph edit log, but the crate is domain-agnostic over what an "event" is).
`unshape-live::LiveSource` is the right *trait shape* for polling stylus input as an external
time-varying source, though no concrete stylus/pointer source exists yet.

**Gaps:**
- **No `Canvas`/`LayerStack` value type.** `unshape-motion::Layer` covers hierarchy, opacity,
  blend mode, z-order for *transform*-based scenes, but has no raster content, no mask, and no
  clip-to-below flag — it's the wrong shape for a stack of pixel buffers with per-layer masking.
  `unshape-image::ImageField` is the right content type per layer; what's missing is the stack
  structure around it (ordered `Vec<Layer>` where `Layer { content: ImageField, mask: Option<ImageField>,
  clip_to_below: bool, blend: BlendMode, opacity: f32, group: Option<GroupId> }`) plus a
  `CompositeStack` op that folds it top-down through `unshape-image::composite`.
- **No brush/stamp engine.** This is the actual gap, not a smaller version of an existing one:
  nothing today models a discrete input-sample stream driving repeated, order-dependent,
  accretive mutation of a persistent buffer. `Field<I, O>` is pure — same input always produces
  the same output — which is the wrong contract for "the 400th stamp depends on the buffer state
  left by the previous 399." Needs a `Stamp { nib: ImageField, dynamics: BrushDynamics }` op and
  a `StrokeSample` sequence type, plus an explicit accretive-apply operation distinct from
  `Field::eval`. This same gap reappears under Blender's sculpting below — see Shared Gaps.
- **No pointer/stylus input abstraction.** Feeding pressure/tilt/velocity samples into the brush
  engine as they arrive is the same "external, time-varying, arrives on its own schedule" shape
  `LiveSource` already generalizes, but no concrete `StylusSource: LiveSource<Output =
  StrokeSample>` exists, and more importantly there's no mechanism for a `LiveSource`'s incoming
  value to *drive a graph mutation* rather than just refresh a cached input value — this is the
  "interactive artifacts" gap the first doc explicitly deferred (Synthesis #5). It recurs a third
  time here; see Shared Gaps.
- **No selection/mask primitive with feathering** exposed as a first-class value distinct from
  an ordinary image channel — a marquee/lasso selection that clips subsequent brush strokes needs
  a `Mask` (effectively a single-channel `ImageField`) that composites *into* the stamp-apply step
  rather than being a downstream filter. Likely a thin wrapper over `ImageField`'s existing
  channel handling (`crates/unshape-image/src/channel.rs`) rather than new math — flagged as an
  open question on the exact API shape, not asserted as a large gap.

## Toon Boom Harmony

**Branded as:** professional 2D animation software for film/TV/games — "everything from classic
frame-by-frame to cutout rigging."

**Actually used for:** cutout character animation via deformers (bone deformers, curve/envelope
deformers) applied to vector or bitmap art; a peg hierarchy (parent transform nodes, independent
of what's visible, used purely for organizing motion — a peg can have no art of its own); a
node-based compositing view ("Node View") for the render graph, alongside the more common
timeline/exposure-sheet view for the same project; drawing-to-drawing morphing; a multiplane
camera producing parallax from z-depth-ordered layers; lip-sync automation from audio phoneme
detection; frame-exact exposure sheet timing (each timeline cell references a specific drawing
index, distinct from continuous keyframe interpolation).

**Computation model:** the Node View is already a compositing DAG — same story as Resolve's
Fusion, a direct match to `unshape-core::Graph`, not a modeling gap. The peg hierarchy is a
parent-relative `Transform2D` chain, structurally identical to `unshape_motion::Scene`/`Layer`.
Deformation is the interesting part: a bone or curve deformer blends the *positions of vector
anchor points* (or, for bitmap art, pixel positions via a warp field) according to per-point
bone weights and the current `Pose` — the same linear-blend-skinning math `unshape-rig::skin.rs`
already implements (`SkinningMethod::LinearBlend`/`DualQuaternion`), except the thing being
deformed is a 2D `VectorNetwork`'s anchors (or a raster warp grid), not a 3D `Mesh`'s vertices.
Morphing between two drawings requires a *point correspondence* between the two vector networks'
anchor sets before any interpolation makes sense — this is not a `Curve`/keyframe problem (which
assumes a fixed set of properties varying smoothly), it's a matching problem between two
possibly-differently-structured graphs, then interpolation of matched anchor positions. The
exposure sheet is a `Timeline` at frame-index granularity — each `ClipInstance`'s `source_range`
is a single discrete drawing index rather than a continuous local-time window, but the type is the
same. The multiplane camera is z-ordered `Layer`s scaled by a camera-projection factor derived
from their z-depth — a variant of the existing z-order/hierarchy model with a projection step
added.

**Maps to unshape subsystems:** `unshape-core::Graph` covers the Node View directly, same as
Resolve's Fusion in the original doc — the best-covered part of Harmony's model.
`unshape_motion::{Scene, Layer, Transform2D}` covers the peg hierarchy. `unshape-timeline`
(now shipped) covers the exposure sheet — `ClipInstance.source_range` can already point at a
single-frame span; `TimeMap`'s `LoopMode::Hold` covers "exposure held on one drawing for N
frames," which is exactly how traditional animation exposure works. `unshape-vector::VectorNetwork`
(`crates/unshape-vector/src/network.rs`) is the underlying drawing representation deformers would
act on. `unshape-rig::skin.rs`'s LBS/DQS math is directly reusable in principle — 2D is a
codimension-1 subset of the same blend math, the same way `unshape-transform::SpatialTransform`
already unifies 2D/3D transforms elsewhere in the codebase.

**Gaps:**
- **No 2D skeletal deformation of vector/raster art.** `unshape-rig::skin.rs` is hard-typed to
  `glam::Vec3`/`Mat4`/`Quat` (`crates/unshape-rig/src/skin.rs:8`) and `unshape-mesh::VertexWeights`
  (`crates/unshape-mesh/src/weights.rs`) is keyed to `Mesh` vertices, not `VectorNetwork` anchors.
  Nothing wires a `Skeleton`/`Pose` to a `VectorNetwork`'s anchors or to an `ImageField` warp grid.
  The underlying math generalizes (skinning a 2D point set is a strict subset of skinning a 3D
  one), but the wiring — weights keyed by `AnchorId` instead of mesh vertex index, and a
  `DeformVectorNetwork`/`DeformImage` op producing a deformed network/image from
  `(Skeleton, Pose, Weights)` — doesn't exist.
- **Vector morphing / point correspondence.** No op establishes correspondence between two
  `VectorNetwork`s' anchor sets (nearest-neighbor, or a more principled shape-matching approach)
  as a precondition for interpolating between them. This is a genuinely new algorithmic piece —
  `unshape-curve`/`unshape-spline` provide the interpolation once correspondence is known, but
  nothing solves the correspondence problem itself. Domain depth specific to this tool; doesn't
  recur elsewhere in this batch.
- **`AnimatedProperty` wiring**, same gap flagged in the original doc's Flash section and not yet
  closed: `unshape-motion-fn::Keyframes<T>` is a working `Field<f32, T>`, but nothing associates
  one with a specific `(LayerId/PegId, PropertyId)` on a `Scene`/`Layer` so peg animation curves
  can be authored and sampled without a bespoke struct per property. Third recurrence across the
  two docs (Flash, Harmony, and — see below — Blender's F-curves); this is now the
  highest-recurrence unclosed gap in either doc.

## Blender

**Branded as:** the free, open-source 3D creation suite — modeling, sculpting, animation, rigging,
simulation, rendering, compositing, video editing, 2D animation (Grease Pencil), all in one.

**Actually used for:** non-destructive polygon modeling via a linear modifier stack (Subdivision
Surface, Mirror, Array, Boolean, Bevel — each a parameterized, reorderable, non-destructive
operation); sculpting via brush-driven displacement on adaptively-retessellated topology (dynamic
topology / "Dyntopo") or a fixed-topology displacement layer on top of a coarse cage
("Multiresolution"); rigging (armatures = bone hierarchies, vertex/weight-painted skinning, IK
constraints, drivers); animation via F-curves (one keyframe curve per animatable property) and
the NLA editor (nonlinear arrangement of reusable Action clips — a timeline of clip instances,
structurally identical to Flash/Resolve's timeline); Geometry Nodes — a visible, user-editable
node graph that procedurally builds or modifies mesh/curve/point-cloud/volume data, evaluated per
object per frame; a material/shading node graph (BSDF graph) evaluated per shading point by the
renderer; rendering — Cycles (unidirectional path tracing) or EEVEE (rasterization with
approximated global illumination) turning the scene graph + materials + lights + camera into
pixels; a 2D compositor — a *separate* node graph operating on rendered image passes
(depth, normal, diffuse, etc.), same DAG shape as Geometry Nodes but a different value domain;
a Video Sequence Editor — a multi-track clip timeline, structurally identical to Resolve's;
physics simulation (rigid body, cloth, fluid via FLIP, soft body, particles); Grease Pencil — 2D
vector strokes drawn and rigged inside a 3D scene, sharing the armature/rigging system with 3D
meshes.

**Computation model:** this is the widest tool surveyed across both docs, and it's also the one
whose core authoring model — a node graph — is already unshape's own architecture, just applied
per-domain instead of exposed uniformly. Where Blender needs a separately-designed Geometry Nodes
system layered on top of an otherwise-imperative mesh editor (because its base modeling operations
predate the node system and aren't natively graph-shaped), unshape's mesh operations
(`Subdivide`, `Bevel`, `BooleanUnion`, etc. — all ops-as-values per house style, see
`crates/unshape-mesh/src/{subdivision,bevel,boolean}.rs`) are Geometry Nodes *by construction*:
every op is already a serializable struct with an `apply`, which is exactly what a Geometry Nodes
node is. The material shading graph and the 2D compositor are the same DAG pattern again, over
`Material`/BSDF values and `Image` values respectively — `unshape-core::Graph` is domain-agnostic
over what flows on wires, so this is not a new execution model, just two more value domains for
existing infrastructure to carry, *modulo* the renderer needed to actually evaluate a shading
graph into pixels (see gaps). F-curves are per-property `Curve`s, the same `AnimatedProperty`
shape flagged above. The NLA editor and VSE are both `Timeline`/`ClipInstance` arrangements.
Sculpting is the 3D sibling of Procreate's brush-stamp gap: a brush stroke displaces a persistent
surface (a position/displacement field over the mesh, or a re-tessellating topology) via
accretive per-stamp mutation, not a pure `Field` evaluation.

**Maps to unshape subsystems:** `unshape-mesh`'s op structs cover Geometry Nodes' modeling
vocabulary about as directly as any mapping in either doc — this is *better* covered than Flash's
timeline was in the first doc, because the underlying architecture is already graph-shaped, not
just individually portable pieces. `unshape-rig` (`skeleton.rs`, `ik.rs`, `skin.rs`,
`constraint.rs`, `weights.rs`) covers armatures, IK, skinning, weight painting close to 1:1 —
`crates/unshape-rig/src/weights.rs`'s heat-diffusion/Laplacian-smoothing weight tools
(`crates/unshape-mesh/src/weights.rs`, confirmed by source) match Blender's automatic weight
generation algorithms. `unshape-motion-fn::Keyframes<T>` covers F-curve interpolation math (same
wiring gap as Harmony/Flash above). `unshape-timeline` covers the NLA and VSE. `unshape-physics`,
`unshape-spring`, `unshape-particle`, `unshape-fluid` cover the simulation domains at a crate-per-
domain granularity roughly matching Blender's own physics tab structure. `unshape-mesh::
subdivision.rs` implements Catmull-Clark with edge creases (confirmed:
`crates/unshape-mesh/src/subdivision.rs`), matching Blender's Subdivision Surface modifier
directly, creases included.

**Gaps:**
- **No renderer.** Nothing in the codebase turns a 3D scene (mesh + material + lights + camera)
  into a pixel image via path tracing or rasterization — confirmed by search: no `PathTracer`,
  `Rasterizer`, or renderer-shaped type exists anywhere in `crates/*/src`. `unshape-gpu` is a wgpu
  *compute* backend for noise/texture generation (`crates/unshape-gpu/src/{kernels,image_ops,
  noise}.rs`), not a rendering-equation integrator. This is a large, genuinely new subsystem —
  the same class of gap as Resolve's color science or Plasticity's B-rep kernel: single-tool
  depth, not a shared primitive, but by far the biggest missing piece surfaced in either doc.
- **No material/shading value type or graph.** A BSDF graph needs its own value domain
  (`Material`/`Bsdf`/spectral-or-RGB `Radiance`) the way `Image` and `Table` are their own `Value`
  variants — `unshape-color` (`crates/unshape-color/src/lib.rs`) covers color spaces and blend
  modes for compositing, not physically-based shading. This gap only matters once a renderer
  exists to consume it — counted as one gap with the renderer above, not two, since a shading
  graph with no evaluator is inert.
- **No 3D scene graph / `Camera` type.** Confirmed by search: no `struct Camera` exists anywhere
  in the codebase (the only "camera" hits are `unshape-vector::rasterize`'s 2D viewport and
  `unshape-vector::text`, unrelated). `unshape_motion::Scene` is explicitly 2D
  (`docs/features.md`: "2D motion graphics scene graph"). A 3D equivalent — object hierarchy,
  camera (projection, lens), lights — is the structural prerequisite for both the renderer gap
  above and for Grease Pencil (2D content composited as a layer *inside* a 3D scene).
- **Sculpting: no dynamic-topology or multiresolution-displacement representation.** Confirmed by
  search: no dyntopo/multires/displacement-layer type exists in `unshape-mesh`
  (`decimate.rs`/`remesh.rs`/`subdivision.rs` exist but model whole-mesh batch operations, not
  "brush stroke displaces a persistent surface, locally re-tessellating as needed"). This is the
  same brush-stamp-engine gap flagged for Procreate, applied to a 3D displacement/position field
  instead of a 2D raster buffer — see Shared Gaps.
- **`AnimatedProperty` wiring** — third/fourth recurrence (Flash, Harmony, Blender F-curves); see
  Harmony section above.
- **2D vector deformation inside a 3D scene (Grease Pencil)** is not a separate gap from Harmony's
  2D skeletal deformation gap — it's the same missing wiring (`Skeleton`/`Pose` → `VectorNetwork`
  anchors), positioned inside a to-be-built 3D scene graph rather than a 2D one.

## Shared gaps

Cross-referencing this trio against each other and against the original six:

### Brush/stamp engine (Procreate, Blender sculpting — 2 of 3 here; new class of gap, not in the original six)

Both are the same shape: a discrete stream of input samples (pointer position + pressure/tilt/
velocity for painting; the same plus a target surface point for sculpting) drives repeated,
order-dependent, *accretive* mutation of a persistent buffer (a raster `Layer` for painting, a
mesh displacement/position field for sculpting). This doesn't fit `Field<I, O>`'s pure-function
contract — a `Field` gives the same output for the same input every time; a brush stroke's 400th
stamp depends on the mutated state the previous 399 left behind. This is a distinct execution
mode from both the pull/lazy graph eval and `LiveSource`'s poll-and-cache: it needs an explicit
`Accretive`/`Canvas` abstraction — a mutable substrate plus a `Stamp` op consumed in sequence
along a `StrokeSample` stream, with the mutation itself (not just its final result) as what
`unshape-history` records for undo. Two tools in this batch hit it; zero in the original six did
— it's a new gap class this round surfaces, worth a `unshape-brush` (or similarly named) crate if
pursued, parameterized over what's being stamped into (`ImageField` for painting, a mesh
displacement field for sculpting) the same way `LiveSource` is parameterized over `Output`.

### Interactive/live input driving graph state (Procreate stylus, Harmony rig manipulation, Blender viewport/sculpt interaction — all 3 here; recurrence with OBS/Flash from the original doc)

The original doc's Synthesis #5 named this ("interactive artifacts... require ground-up design
treatment") and explicitly deferred it rather than designing it. It recurs in all three tools this
round — Procreate's stylus input, live rig manipulation in Harmony (dragging a bone updates the
deformed art in real time, distinct from playing back a baked animation), and Blender's viewport
navigation/sculpt-brush interaction — on top of its two prior recurrences (OBS hotkeys/stream
events, Flash frame scripts/button events). That's five recurrences across nine tools surveyed,
the highest of any deferred gap in either doc. `unshape-live::LiveSource` supplies half of it (an
external, time-varying input, polled on the host's schedule) but the other half — a polled value
*driving a graph mutation or triggering a discrete transition* rather than just refreshing a
cached `GraphInput` value — still has no design. Worth promoting from "deferred, revisit later"
to "next design doc," given the recurrence count.

### Layer stack with masks and grouping (Procreate raster layers, Harmony drawing layers under pegs — 2 of 3 here)

`unshape_motion::Layer` covers hierarchy, parent-relative `Transform2D`, opacity, and blend mode,
but has no mask (a per-layer alpha field that clips its content), no clip-to-below flag (mask by
the layer immediately beneath in the stack, Photoshop/Procreate-style), and no group nesting with
its own composite-then-blend semantics (a group composites its children first, then blends the
result as a unit — not the same as flattening the group's children into the parent stack).
Extending `Layer` with `mask: Option<ImageField>`, `clip_to_below: bool`, and `group: Option<
GroupId>` (plus a `CompositeGroup` op which folds a group's children before blending the group as
a whole) would close this for both tools without a new crate — it's additive to the existing
`unshape-motion::Layer`, the same way the original doc's `AnimatedProperty<T>` was additive rather
than a replacement.

### `AnimatedProperty` wiring (Flash [original doc], Harmony, Blender F-curves — 3 recurrences, still unclosed)

Flagged as missing in the original doc's Flash section and still missing after this round's
check: `unshape-motion-fn::Keyframes<T>` already does the interpolation math as a working
`Field<f32, T>`, so this is purely a wiring gap now, not a math gap — associate a `Keyframes<T>`
(or any `Field<f32, T>`) with a `(LayerId, PropertyId)` on a `Scene`/`Layer`, sampled at the
timeline's resolved local time each time that field is read, so a static layer's `Transform2D`
stays a plain value and only animated layers pay for a keyframe lookup. Highest-recurrence
unclosed gap between the two docs after interactive input.

### 2D skeletal deformation of vector/raster content (Harmony bone/curve deformers, Blender Grease Pencil rigging — 2 of 3 here)

`unshape-rig::skin.rs`'s LBS/DQS math is dimension-agnostic in principle (2D is a subset of the
3D case) but hard-typed to `Vec3`/`Mat4`/`Quat`, and `VertexWeights` is keyed to `Mesh` vertex
indices, not `VectorNetwork` anchor IDs or raster warp-grid points. Closing this needs weights
keyed by a generic point-id (anchor or grid cell) and a `DeformVectorNetwork`/`DeformImage` op
taking `(Skeleton, Pose, Weights)`, not a rewrite of the skinning math itself.

### What doesn't generalize

Blender's renderer (path tracing/rasterization) and shading-graph value domain is the single
largest gap surfaced across both docs, but it's Blender-only in this trio — the same "domain
depth, not shared primitive" verdict as Resolve's color science or Plasticity's B-rep kernel in
the original doc. Harmony's vector-drawing morphing (point correspondence between two topologies)
is domain depth specific to that tool. Blender's 3D scene graph/`Camera` type is a structural
prerequisite for the renderer gap, not a separate shared primitive — it doesn't recur in Procreate
or Harmony, both of which are 2D-only.

## UI patterns worth noting

Not to copy these tools' UI, but to name the interactions their computation models are built to
support — what unshape's eventual projection model (`docs/design/projection-model.md`) needs to
be capable of, even if the concrete UI looks nothing like any of these three.

- **Procreate's Brush Studio exposes brush *parameters* directly as the authoring surface**, not
  a picker over presets — shape, grain, dynamics-per-input-channel are each independently tunable
  and the result previews live. This is the "parameters, not presets" design principle already in
  `CLAUDE.md` playing out concretely in a shipped tool: the brush-stamp gap above should be
  designed the same way (a `BrushDynamics` struct with named curves per input channel), not as an
  opaque "brush preset" blob.
- **Harmony's Node View and its timeline/exposure-sheet view are two projections of the same
  underlying graph+timeline data**, switchable per-user-preference, not two different documents.
  This is the dual-projection pattern `unshape-editor` already implements for op-stack-vs-formula
  (`crates/unshape-editor/src/lib.rs`) — same principle, different domain: a `Timeline` value and
  the `Graph` its `ClipInstance` sources belong to are both real data, and a UI can project either
  one as "the" view without one being derived-and-thrown-away from the other.
- **Blender's modifier stack shows applied-but-not-yet-baked operations as an ordered, reorderable,
  individually-toggleable list with live parameter widgets per entry** — the direct UI expression
  of ops-as-values (`docs/design/ops-as-values.md`): every modifier is a struct with named fields,
  the stack is a `Vec` of them, "apply" bakes it into the base mesh (destroying replayability for
  that one op) and "toggle visibility" just skips one entry during eval. unshape's op-as-values
  discipline already gives this for free structurally; the missing piece is only the UI affordance
  (reorder, toggle, tweak-in-place), not new backing data.
- **Live rig manipulation (Harmony bones, Blender armature in pose mode) requires interaction
  latency low enough that dragging a control feels like touching the deformed result directly**,
  which is a stronger real-time constraint than "render eventually re-triggers" — it's the same
  requirement OBS's continuously-running composite pipeline has, generalized from "live source
  changes" to "live user input changes a parameter deep in the graph and everything downstream
  needs to re-evaluate before the next frame is due." This is a performance/scheduling
  requirement on top of the interactive-input *design* gap named above, not a separate feature.
- **Procreate's gesture-based unlimited undo and Blender's F-curve graph editor both expose the
  *history of a value over time* as directly manipulable** (drag a point on the undo timeline;
  drag a point on an F-curve) rather than as a hidden implementation detail. `unshape-history`'s
  event sourcing already stores this; the UI pattern worth carrying forward is that the history
  itself — not just its current-state result — is something the projection model should be able
  to render and let a user grab.
