# Domain Subsumption: Six Tools

Unshape's pitch is "a substrate for constructive generation and manipulation of media" — a claim
that a node graph, `Value` types, and eval/compute infrastructure are general enough to absorb
entire categories of tool, not just file formats. This doc stress-tests that against six widely
used tools that *aren't* obviously media-graph shaped — a spreadsheet, an OLAP database, a
timeline animation tool, a color-grading NLE, a live streaming mixer, a B-rep solid modeling
CAD tool — and asks what it would take to model each as unshape graphs, and what's missing to
do it honestly (not by hiding it all behind `Opaque`).

Method: separate what each tool is *branded as* from what it's *actually used for*, derive the
computation model that implies, map it against what exists today (checked against crate source,
not assumed), and name the gap. The synthesis looks for primitives that close gaps in more than
one tool — that's where the leverage is.

## Current substrate (baseline)

- **Graph**: DAG with cycles permitted only through a `Latch` node (seeded unit-delay; see
  `docs/design/recurrent-graphs.md`). Lazy evaluation + caching (`docs/design/evaluation-strategy.md`).
- **Value**: `F32 | F64 | I32 | Bool | Vec2 | Vec3 | Vec4 | Opaque(Arc<dyn GraphValue>)`
  (`crates/unshape-core/src/value.rs`). Domain-specific types (Image, Mesh, AudioBuffer) are
  `Opaque`, downcast by `TypeId`.
- **EvalContext**: `time: f64, frame: u64, dt: f64, preview_mode: bool,
  target_resolution: Option<(u32,u32)>, seed: u64`, plus named `GraphInput` values
  (`crates/unshape-core/src/eval.rs`). No `sample_rate`/`quality` field exists on the shipped
  struct — that's a `time-models.md` proposal, not current code — audio rate and quality are
  handled at the node/backend level instead.
- **Backends**: CPU, GPU (wgpu), JIT (Cranelift), selected per-node via `unshape-backend`.
- **History**: snapshot + event sourcing (`unshape-history`), independent of graph structure.
- **Motion graphics**: `unshape-motion` has `Scene`/`Layer` (hierarchy, parent-relative
  `Transform2D`, opacity, `BlendMode`) but no timeline, keyframes, or tweening — transforms are
  static, not time-varying.
- No tabular value type, streaming/live-source abstraction, timeline/cut-list type, scene state
  machine, or event/interaction model — the actual gaps this doc is about.

## Excel

**What it's for:** not a cell-dependency DAG in practice, though that's the implementation. An
editable table that's also a display surface, SUMIF/pivot-style aggregation without a query
language, a schema you can see (columns as loosely-typed fields), and a scratch pad for small
"params → formula → result" models.

**Computation model:** a 2D grid of typed columns; per-cell formulas that are graph nodes
addressed by (row, col) instead of NodeId — Excel *is* a spatial editor for a table-shaped graph;
aggregation (SUM, COUNTIF, pivot) = group-by + reduce over columns; lookup (VLOOKUP) = a
single-key join; recalculation is incremental, only a cell's dependents re-evaluate.

**Already in unshape:** the incremental-recompute half is architecturally identical to
`unshape-core::Graph` + `EvalCache` (cell formula = node, cell reference = wire). `unshape-op`'s
`DynOp`/registry covers "named function with typed params" once there's a value type to operate
over.

**Missing:** a columnar `Value` variant and an algebra over it — see Synthesis #1
(`Table`/`Column`) and the `GroupByAggregate`/`Join`/`FilterRows` ops it implies. Excel's
cell-as-spatial-node UI is a *projection* of `Table` + a formula graph
(`docs/design/projection-model.md`), not a new execution model.

## ClickHouse

**What it's for:** bulk columnar analytics — data larger than RAM, vectorized column-at-a-time
execution, materialized views that keep an aggregate continuously current as rows land instead
of recomputing from scratch.

**Computation model:** the same relational algebra as Excel, but at a scale where the whole
`Table` can't be materialized in memory — evaluation must stream in chunks; a materialized view
is a `Table`-producing subgraph that updates incrementally per incoming batch, i.e. a
continuously-running node rather than a pull-once one; vectorized execution is the same
SIMD/batch concern `unshape-jit` and the GPU backend already serve, applied to columns instead
of pixels/vertices.

**Already in unshape:** `unshape-backend`'s CPU/GPU/JIT dispatch is agnostic to *what's* being
computed — columnar batch ops fit the same story as image filters. Lazy eval + caching is the
right shape for "recompute only what changed," but caching granularity is per-node-output, not
per-row.

**Missing:**
- A streamed `TableSource` (batch iterator), co-equal with in-memory `Table` per the
  general/constrained pattern's "co-equal primitives" exception — converting a billion-row source
  into one in-memory `Table` is exactly the O(N²)-class non-starter that exception exists for.
- Incremental aggregation state: a materialized view is a `Latch`-shaped problem — running
  aggregate state (sum/count/min/max, anything with an incremental update rule) seeded empty,
  fed by batches as the `signal` input to a stateful reducer. No new recurrence mechanism needed.
- Push-driven evaluation for "new data arrives, propagates forward" — this is the same gap OBS
  hits from the live-source side; see Synthesis #3.

## Flash / Animate

**What it's for:** timeline-driven vector animation — nested symbols each with an independent
timeline/playhead, keyframe interpolation as the primary authoring tool, a z-ordered display
list with blend modes and masks, scripted interactivity (frame scripts, button events) on top.

**Computation model:** a timeline is an ordered sequence of frames/spans per layer, not a single
scalar `time` — symbols nest timelines, a movie clip's playhead runs independently of its
parent's; keyframes are sparse time→value samples with an interpolation rule, i.e. a `Curve`
(`docs/design/curve-types.md`) evaluated at *local* time, addressed per symbol instance; the
display list is `unshape_motion::Scene`/`Layer`; scripted interactivity is discrete events
mutating graph state or jumping a playhead.

**Already in unshape:** `unshape-curve` covers keyframe interpolation numerically.
`unshape_motion::{Scene, Layer, Transform2D, BlendMode}` covers hierarchy, opacity, blend mode,
z-order — the best-covered domain of the five. `unshape-vector`'s `VectorNetwork`/`Path` covers
fills/strokes.

**Missing:**
- Time-varying layer properties: nothing today associates a `Curve` with a `Layer` field and
  samples it at a given time (`Transform2D` is a static value). Needs an `AnimatedProperty<T>`
  wrapping keyframes + interpolation, either inline on `Layer` or a side-table keyed by
  `(LayerId, PropertyId)` so static layers pay nothing.
- Nested independent playheads: a symbol instance needs local time derived from parent time by
  offset/rate/loop-mode, not by directly inheriting `EvalContext::time` — see Synthesis #2
  (`Timeline`/`TimeMap`), which generalizes to Resolve's tracks and OBS's source clocks too.
- Discrete event dispatch — see Synthesis #5.

## DaVinci Resolve

**What it's for:** Fusion (the compositor) is already a node graph, so that half is a non-issue.
Distinct: a temporal timeline with cuts/transitions across tracks, color science as a
first-class typed pipeline (color spaces, ACES, LUTs, wheels, qualifiers — not "apply a filter"),
video as a frame sequence with defined in/out semantics, and proxy/multi-res workflows.

**Computation model:** ordered, possibly transition-overlapping clip spans per track, each
referencing a media source + in/out points + speed ramp; color operations need to know *which
space* they operate in (scene-linear ACEScg vs. display-referred Rec.709) because the same math
is wrong across spaces without an explicit transform; multi-res is the same graph evaluated at
different resolutions for proxy vs. final.

**Already in unshape:** the compositor-as-graph mapping is direct — Fusion nodes are unshape
nodes. `EvalContext::target_resolution` already covers the proxy/LOD case. `unshape-color`
exists; whether it tracks color space *in the type* (vs. by convention) needs to be checked
against its actual API — flagged here as an open question, not asserted either way.

**Missing:**
- Timeline/clip-list type — the same structural gap as Flash's timeline, but multi-track with
  transitions rather than nested-symbol. See Synthesis #2. Evaluating a `Timeline` at time `t`
  means finding the active clip(s) per track, remapping `t` into local clip time via
  `source_range`/`speed` (the same `TimeMap` concept as Flash), and compositing tracks top-down
  through `Layer`-style blend modes — a scheduler over subgraphs, not a new eval model.
- Color space as a type: `Fill { space: ACEScg }` and `Fill { space: Rec709 }` should not be
  silently mixable — an op needing scene-linear input should refuse (or explicitly convert) a
  display-referred one at the type level, not by convention.

## OBS

**What it's for:** live composition — sources (camera, capture, window, browser) mixed into
scenes, each source with its own filter chain, scene switching as a discrete state machine (not
a global crossfade), a real-time encode/output pipeline running continuously.

**Computation model:** a live source is a node whose output changes on its own schedule rather
than being pulled by downstream demand — the inverse of the current pull/lazy eval; a scene is a
named composition of sources + filter chains + layout; scene switching is a discrete transition
between named scenes, optionally with an effect (fade, cut, stinger), structurally identical to
Resolve's `Transition`; a filter chain per source is a linear op stack (`unshape-op`'s
`Pipeline`/`Chain`); routing is source → filter chain → scene composite → encoder, continuously,
at a fixed cadence regardless of whether anything "asked" for a frame.

**Already in unshape:** `unshape_motion::Scene`/`Layer` covers composition, z-order, blend
modes. The `Chain` constrained-API pattern covers filter chains. The CPU/GPU backend split
covers "encode is a different backend than compose."

**Missing:**
- Push/live source abstraction — the same gap ClickHouse's materialized views hit from the other
  direction. See Synthesis #3.
- Scene state machine: no type today represents "one of N named scenes is active, switch on
  discrete trigger, optionally transition." See Synthesis #4 — same shape as Flash's scene list
  and Resolve's `Transition`.
- Event/interaction model to drive scene switches from hotkeys/timers/stream events — same gap
  as Flash's scripted interactivity. See Synthesis #5.

## Plasticity

**Status: Deferred.** B-rep/CAD modeling is too expensive to pursue now. Noted here for reference. If revisited, opencascade-rs (Rust bindings to OpenCASCADE) is the likely path rather than implementing a kernel from scratch.

**What it's for:** exact solid modeling via boundary representation (B-rep) — constructive
operations (extrude, revolve, sweep, loft, solid boolean union/subtract/intersect) that build
and combine watertight solids exactly rather than approximating them as polygon soup; edge
operations (fillet, chamfer) that modify topology while staying exact; a sketch-to-solid
workflow (2D constrained profile → 3D solid); direct B-rep modeling — Plasticity deliberately
does *not* use a history/feature tree (unlike most parametric CAD), because keeping a feature
tree robust under edits forces disabling advanced solid/surfacing operations; it exposes the
full power of its kernels (Parasolid + xNURBS) by manipulating the B-rep directly instead;
NURBS surface evaluation with trim curves bounding each face to an arbitrary (non-rectangular)
region of its parameter domain.

**Computation model:** B-rep topology is a strict hierarchy — `Solid` → `Shell` → `Face` →
`Loop` → `Edge` → `Vertex` — where each `Face` is backed by an analytical surface (a NURBS patch,
or a plane/cylinder/sphere/cone/torus as a closed-form special case) and bounded by one or more
`Loop`s of `Edge`s that are themselves curves living in the face's 2D parameter space (trim
curves), not just 3D space; booleans operate on this topology directly — intersecting two
B-reps means finding curve-surface intersections between the operand faces, splitting loops at
those intersections, and reclassifying resulting face fragments as inside/outside/on-boundary,
which is a fundamentally different (and exact) operation from BSP-tree mesh boolean; because
there's no feature tree to keep valid, each operation just mutates (or produces a new) `Solid`
directly — there's no dependency graph of "this fillet depends on that extrude's face id"
that later edits have to keep resolvable; sketch constraints are a 2D geometric constraint
solver (coincident, tangent, perpendicular, parallel, equal, dimensional) that resolves a
profile's points before it's fed to a constructive op.

**Already in unshape:**
- `unshape-surface::NurbsSurface` (`crates/unshape-surface/src/lib.rs`) — tensor-product NURBS
  surface evaluation (`evaluate`, `derivative_u`/`derivative_v`, `normal`, `tessellate`) plus
  closed-form quadric constructors (`nurbs_sphere`, `nurbs_cylinder`, `nurbs_cone`,
  `nurbs_torus`, `nurbs_bilinear_patch`). This is real NURBS surface math, not an approximation —
  but the surface is always the *whole* rectangular parameter domain; there is no way to bound a
  face to a sub-region, so it cannot represent a trimmed face (e.g. a cylinder with a hole cut in
  its side).
- `unshape-spline::Nurbs<T>` (`crates/unshape-spline/src/curve_impl.rs`, generic over `Vec2`/
  `Vec3`) — NURBS curve evaluation, alongside `BSpline`, `CatmullRom`, `BezierSpline`. Usable as
  the curve type a trim curve would need, but nothing currently associates a curve with a
  surface's parameter space to bound it.
- `unshape-mesh::HalfEdgeMesh` (`crates/unshape-mesh/src/halfedge.rs`) — a real half-edge
  topology (`HalfEdge`/`Vertex`/`Face` with next/twin/vertex/face adjacency), but it indexes a
  polygon mesh, not a B-rep: faces are flat polygons, not references to analytical surfaces, and
  there's no `Solid`/`Shell` grouping above `Face`.
- `unshape-mesh::boolean` (`BooleanUnion`/`BooleanSubtract`/`BooleanIntersect`,
  `crates/unshape-mesh/src/boolean.rs`) — mesh CSG via BSP tree splitting/classification. Correct
  op-as-value shape (op struct + `apply`) but operates on triangulated `Mesh`, so booleans are an
  approximation of the operand geometry, not exact — a NURBS cylinder boolean'd against a NURBS
  sphere this way loses the exact surfaces and produces a triangle soup, not a solid whose faces
  are still cylinder/sphere patches.
- `unshape-mesh::Bevel` (`crates/unshape-mesh/src/bevel.rs`) — edge/vertex bevel and chamfer, but
  on `HalfEdgeMesh` polygon topology, not on B-rep edges with adjacent analytical faces; a
  polygon-mesh bevel can't guarantee the result stays on the true fillet surface (a cylindrical
  patch of the fillet radius) the way a B-rep fillet does.
- `unshape-mesh::Loft` (`crates/unshape-mesh/src/loft.rs`) — interpolates between profile point
  lists directly into a triangulated `Mesh`; this is Plasticity's "loft" by name but not by
  construction — it never produces a NURBS surface, so there's no exact face to later trim,
  fillet, or boolean against.
- `unshape-mesh::primitives` (`Cuboid`, `UvSphere`, `Cylinder`, `Cone`, `Torus`, `Plane`,
  `Icosphere`, `crates/unshape-mesh/src/primitives.rs`) — the same primitive shapes Plasticity
  offers as solid primitives, but generated directly as polygon meshes, not as B-rep solids with
  analytical faces.
- Replayability without a feature tree needs no new primitive: `unshape-op`'s `DynOp`/`Pipeline`
  and the ops-as-values convention already give "ordered list of serializable operations, undo/
  redo, replay" for free (`unshape-history`'s snapshot + event sourcing, independent of graph
  structure) — the same way it does for every other domain in this codebase. This is a genuine
  edge over Plasticity's own no-history-tree design, not just parity with it: Plasticity avoids a
  feature tree specifically because keeping one *valid* under edits is what forces disabling
  advanced kernel operations, whereas a flat op log has no cross-op validity constraint to
  maintain — the history is just a record of what happened, not a dependency graph an edit has to
  keep resolvable. Recording history and refusing to constrain the kernel to keep it tidy aren't
  in tension here, so unshape's B-rep crates (below) don't need to choose between them the way a
  feature-tree CAD tool would.

**Missing primitives:**
- B-rep topology structure: `Solid`/`Shell`/`Face`/`Loop`/`Edge`/`Vertex` with adjacency, where
  `Face` references an analytical surface (reusing `NurbsSurface` plus closed-form primitives)
  instead of storing flat polygon geometry. This is the structural gap underneath everything
  else in this section — `HalfEdgeMesh` is the closest existing shape but is one topological
  layer too shallow (no `Solid`/`Shell`, `Face` isn't surface-backed).
- Trim curves: a curve (reusing `unshape-spline::Nurbs<Vec2>` or `unshape-curve::Path`) living in
  a face's `(u, v)` parameter space that bounds the face's `Loop`s to less than the full
  rectangular domain `NurbsSurface` currently always covers.
- Exact solid booleans: intersection/union/subtract operating on B-rep topology via
  curve-surface intersection and loop splitting/reclassification, not BSP-tree mesh splitting —
  a different algorithm from `unshape-mesh::boolean`, not a parameter on it, because the inputs
  and outputs are a different representation (B-rep faces stay analytical surfaces, not
  triangles).
- Fillet/chamfer as B-rep edge operations: given a `Solid` and a set of its `Edge`s, replace each
  edge with a new face on the true rolling-ball/constant-radius fillet surface (or a flat chamfer
  plane), re-stitching adjacent faces' loops — the B-rep-topology analogue of
  `unshape-mesh::Bevel`, not reusable from it because the output faces need to carry real
  surfaces, not polygon fans.
- Constructive ops as `Solid`-producing op structs: `Extrude { profile, direction, distance }`,
  `Revolve { profile, axis, angle }`, `Sweep { profile, path }`, and a `Solid`-producing `Loft`
  (distinct from `unshape-mesh::Loft`, which stays as the polygon-mesh version) — each builds
  B-rep topology with analytical side/cap faces from a 2D profile, per the ops-as-values pattern
  used everywhere else in this codebase.
- 2D sketch constraint solver: a system over 2D points/curves with constraints (coincident,
  tangent, perpendicular, parallel, equal-length, dimensional) that resolves a profile's degrees
  of freedom before it's handed to `Extrude`/`Revolve`/`Sweep`/`Loft` — no existing crate has a
  constraint-solving component; `unshape-curve`/`unshape-spline` provide the curve types a solved
  sketch would be made of but nothing resolves constraints over them.

## Synthesis — shared primitives across the other five

Plasticity's gap list is intentionally kept separate from the synthesis below: none of its
missing primitives (B-rep topology, trim curves, exact booleans, B-rep fillet, sketch
constraints) recur in Excel, ClickHouse, Flash, Resolve, or OBS, so by the same
co-equal-primitives test used throughout this doc, they don't earn a place in the shared
synthesis — they're domain depth for solid modeling, the same way Resolve's color science is
domain depth for grading. Replayability is the one piece of Plasticity's model that *does* map
onto existing infrastructure (ops-as-values plus `unshape-history`) without needing anything new,
which is why it isn't listed as missing above — note this is deliberately not "parametric
history": Plasticity itself avoids a feature tree, and the ops-as-values log unshape would use
gets replay/undo without imposing one either.

Five gaps recur across the other five tools; each closes more than one, so the minimal addition set is
five primitives, not five tools' worth of bespoke types.

### 1. `Table` as a first-class `Value` variant (Excel, ClickHouse)

```rust
pub enum Value {
    F32(f32), F64(f64), I32(i32), Bool(bool),
    Vec2(Vec2), Vec3(Vec3), Vec4(Vec4),
    Table(Arc<Table>),     // NEW
    Opaque(Arc<dyn GraphValue>),
}

pub struct Column { pub name: String, pub data: ColumnData } // F32Vec | I32Vec | BoolVec | StringVec | ...
pub struct Table { pub columns: Vec<Column> }  // same len enforced on construction

// Ops as values, per house style:
pub struct GroupByAggregate { pub group_by: Vec<String>, pub aggregations: Vec<(String, AggregateFn)> }
pub struct Join { pub left_key: String, pub right_key: String, pub kind: JoinKind }
pub struct FilterRows { pub predicate: Expr } // reuses the dew expression language

pub trait TableSource {                        // co-equal with Table, not a wrapper
    fn schema(&self) -> &[ColumnSchema];
    fn next_batch(&mut self, max_rows: usize) -> Option<Table>;
}
```

Why a real variant, not `Opaque<Table>`: group-by/join/filter need to see column names and types
to validate wiring — the way `Vec3` gets swizzle-aware ports today. Opaque would push that into
runtime `TypeId` downcasts and lose the static-shape benefit `Value` exists to provide.
Streaming/oversized tables use `TableSource`, unified with `Table` by a trait per the existing
co-equal-primitives exception rather than one wrapping the other.

### 2. `Timeline` (Flash, Resolve, OBS)

```rust
pub struct TimeMap { pub rate: f64, pub offset: f64, pub mode: LoopMode }
pub enum LoopMode { Once, Loop, PingPong, Hold }

pub struct Timeline { pub tracks: Vec<Track>, pub transitions: Vec<Transition> }
pub struct Track { pub clips: Vec<ClipInstance> }
pub struct ClipInstance {
    pub source: NodeId,
    pub timeline_range: TimeRange,
    pub source_range: TimeRange,
    pub time_map: TimeMap,
}
pub struct Transition { pub at: f64, pub duration: f64, pub kind: TransitionKind } // Cut | CrossDissolve | Wipe(..)
```

Flash's nested symbol timelines, Resolve's multi-track edit, and OBS's "source running since
scene-load, independent of scene-switch time" are the same structure: a tree/list of `Timeline`s,
each remapping a local clock from its parent via `TimeMap`, evaluated by walking the active
`ClipInstance`(s) at a given time and recursing. Additive to `EvalContext` — global `time: f64`
becomes the root clock; `Timeline` evaluation is a node family that resolves local time and feeds
it to a subgraph, not a change to the core eval loop.

### 3. `LiveSource` / push scheduling (OBS, ClickHouse materialized views, Resolve capture)

```rust
pub trait LiveSource {
    type Output;
    fn poll(&mut self) -> Option<Self::Output>; // non-blocking; None if nothing new
}
```

A `GraphInput` node whose value arrives on its own schedule instead of being computed on pull.
Downstream stays pull/lazy — this only changes *when* a `GraphInput` node's cached value is
stale, an addition to `evaluation-strategy.md`'s cache-invalidation policy, not a parallel
evaluator. Resolve capture cards, OBS cameras, and ClickHouse's "rows landed" are the same trait
with different `Output` (`Frame`, `AudioBlock`, `Table` batch).

### 4. `StateMachine` scene switching (OBS scenes, Flash scene list)

```rust
pub struct StateMachine<S> {
    pub states: HashMap<StateId, S>,
    pub active: StateId,
    pub transitions: Vec<(StateId, StateId, Option<TransitionKind>)>,
}
pub struct SwitchState { pub to: StateId, pub transition: Option<TransitionKind> }
```

Generic over `S` so it applies to `unshape_motion::Scene` without a scene-specific type.
`TransitionKind` is shared with `Timeline`'s `Transition` above, not duplicated.

### 5. Interactive artifacts (user input at runtime)

Interactive artifacts (responding to user input at runtime) may be in scope for unshape eventually,
but require ground-up design treatment — not an afterthought bolted onto the media computation model.
This is a separate future design problem, not addressed here.

### What doesn't generalize

Excel's cell-as-spatial-node UI is a projection concern, not an execution primitive. Resolve's
color science (ACES, LUTs) is domain depth for `unshape-color` — it doesn't recur elsewhere, so
by the co-equal-primitives test it doesn't earn a shared abstraction. ClickHouse's query
optimizer (predicate pushdown, join reordering) is a backend/scheduling concern for evaluating
`Join`/`FilterRows` over a `TableSource` efficiently, not a primitive. Plasticity's B-rep
topology, trim curves, exact booleans, B-rep fillet/chamfer, and sketch constraint solver are the
same story from the sixth tool: real gaps, but domain depth for solid modeling rather than
primitives any of the other five tools would use.

### Net new surface

| Primitive | Closes gap in |
|---|---|
| `Table` / `Value::Table` + `TableSource` | Excel, ClickHouse |
| `Timeline` / `Track` / `ClipInstance` / `TimeMap` | Flash, Resolve, OBS |
| `LiveSource` trait | OBS, ClickHouse, Resolve |
| `StateMachine<S>` / `TransitionKind` | OBS, Flash, Resolve (shared with Timeline) |

Four primitives, each shared by two or more of the other five tools, is the actual scope of
"subsume Excel/ClickHouse/Flash/Resolve/OBS" — not five independent subsystems. All are additive
to the existing `Graph`/`Value`/`EvalContext` model: none requires replacing the pull/lazy
evaluator, the `Opaque` escape hatch, or the `Latch`-based recurrence primitive. Plasticity's
primitives don't join this table — see the Plasticity section above — but are carried forward
into the Roadmap below as their own single-tool depth, the same way Resolve's color science is.

## Roadmap — new crates

The Synthesis section identifies *primitives*; this section maps those (plus everything that
doesn't generalize — color science, video frames, encoding, B-rep solid modeling) onto concrete
crates, so "subsume these six tools" has a buildable shape rather than staying six paragraphs of
gap analysis. Domain-specific work belongs in the tool's crate even where it doesn't share a
primitive with anything else — the point of the table below is which crates carry *shared* load
and which are single-tool depth.

| Crate | Serves | Core types / ops |
|---|---|---|
| `unshape-table` | Excel, ClickHouse | `Value::Table`, `Column`/`ColumnData`, `GroupByAggregate`, `Join`, `FilterRows`, `Pivot` — the in-memory relational algebra from Synthesis #1. Excel's cell-as-spatial-node UI is a projection layer on top of this, not part of the crate. |
| `unshape-table-store` | ClickHouse (Excel indirectly, for large sheets) | `TableSource` trait impls backed by bigger-than-RAM columnar storage, chunked/batch iteration, materialized-view state (the `Latch`-shaped incremental aggregate from the ClickHouse section). Depends on `unshape-table` for the `Table`/op vocabulary; adds the storage and streaming-execution layer, not new relational ops. |
| `unshape-timeline` | Flash, Resolve, OBS | `Timeline`, `Track`, `ClipInstance`, `TimeMap`, `Transition`/`TransitionKind` from Synthesis #2 — nested local clocks, clip spans, transitions. Extends `unshape-motion` rather than replacing it: `Timeline` evaluation resolves local time and hands off to `Scene`/`Layer` for what gets drawn at that time. |
| `unshape-live` | OBS, ClickHouse, Resolve (capture) | `LiveSource` trait (Synthesis #3) and concrete sources: capture devices, stream inputs, screen capture, hotkey/timer event sources (Synthesis #5's `KeyDown`/`Timer`/`FrameReached`). Push-to-pull bridging (cache invalidation on new data) lives here, not in `unshape-core`, since it's a policy over `GraphInput` staleness rather than a core eval-loop change. |
| `unshape-scene-state` | OBS, Flash, Resolve | `StateMachine<S>` (Synthesis #4), generic over the thing being switched — `unshape_motion::Scene` for OBS/Flash, a `Timeline` selection for Resolve multi-cam. Shares `TransitionKind` with `unshape-timeline` rather than redefining it. |
| `unshape-motion` *(existing, extended)* | Flash, Resolve, OBS | Adds `AnimatedProperty<T>` (keyframes + interpolation via `unshape-curve`, sampled at local time from `unshape-timeline`) to the existing `Scene`/`Layer`/`Transform2D`/`BlendMode`. This is the missing link between "timeline knows what time it is" and "layer property has a value at that time." |
| `unshape-video` | Resolve | Frame-sequence value type (in/out points, frame rate, defined seek semantics), multi-track compositing over `unshape-timeline`, proxy/multi-res evaluation via existing `EvalContext::target_resolution`. Depends on `unshape-timeline` for track/clip structure; the video-specific part is what a "clip" *is* and how frames composite, not the scheduling. |
| `unshape-color-science` (or extend `unshape-color`) | Resolve | Color space as a type (`ACEScg`, `Rec709`, ...) tracked through the graph so ops can refuse or explicitly convert mismatched spaces, LUTs, color wheels, qualifiers, ACES pipeline stages. Purely domain depth — per the "what doesn't generalize" note above, this doesn't recur in the other tools, so it stays scoped to Resolve rather than becoming a shared primitive. |
| `unshape-encode` | OBS (Resolve export, indirectly) | Takes a composed `Scene`/`Timeline` output and encodes to a stream or file — the continuously-running tail of OBS's `source → filter chain → scene composite → encoder` pipeline. Likely an integration crate wrapping a codec backend rather than new core abstraction; kept separate so `unshape-live`/`unshape-scene-state` don't gain an encoder dependency. |
| `unshape-brep` (deferred) | Plasticity | `Solid`/`Shell`/`Face`/`Loop`/`Edge`/`Vertex` B-rep topology with `Face` backed by `unshape-surface::NurbsSurface` (or a closed-form quadric); trim curves (reusing `unshape-spline::Nurbs<Vec2>`) bounding a `Face`'s parameter domain. This is the structural crate everything else in the Plasticity section depends on — the B-rep analogue of `unshape-mesh::HalfEdgeMesh`, one topological layer deeper (`Solid`/`Shell` above `Face`, `Face` surface-backed instead of flat). |
| `unshape-brep-ops` (deferred) | Plasticity | `Extrude`, `Revolve`, `Sweep`, a `Solid`-producing `Loft` (op structs, per house style), exact solid `Union`/`Subtract`/`Intersect` via curve-surface intersection and loop reclassification, and `Fillet`/`Chamfer` as B-rep edge operations. Depends on `unshape-brep` for topology and `unshape-curve`/`unshape-spline` for the profile/trim curve types; kept separate from `unshape-brep` so the topology crate doesn't carry every constructive algorithm's dependencies. |
| `unshape-sketch` (deferred) | Plasticity | 2D geometric constraint solver (coincident, tangent, perpendicular, parallel, equal, dimensional) over `unshape-curve` profile geometry, resolving a sketch's degrees of freedom before it's handed to `unshape-brep-ops`. No existing crate has a constraint-solving component, so this is new algorithmic surface, not a wrapper over something that exists. |

Shared-load crates in order of how many tools they touch: `unshape-timeline` (Flash, Resolve,
OBS) and `unshape-table` (Excel, ClickHouse) carry the most weight — they're the two primitives
from the Synthesis that are genuinely load-bearing across more than one tool's core model, not
just an incidental reuse. `unshape-live` and `unshape-scene-state` are three-tool and three-tool
respectively but each is a thinner trait/type than a full relational algebra or timeline model.
`unshape-video`, `unshape-color-science`, `unshape-encode`, and the three Plasticity crates
(`unshape-brep`, `unshape-brep-ops`, `unshape-sketch`) are single-tool depth for Resolve, OBS,
and Plasticity respectively — expected, since those are the tools whose "what it's actually for"
section names capabilities (color science, real-time encode, exact B-rep booleans) that don't
show up anywhere else in the six.
