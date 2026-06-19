# Projectional Editor

The primary UI for unshape is not a single surface. It is an **arbitrary projectional editor**: one typed op-graph is the source of truth, and the editor renders it into multiple co-equal, editable *lens-projections* (op-stack, timeline, structure/formula inspector, literate document, node-graph, direct-manipulation canvas). You edit through whichever projection fits the task; every edit round-trips to the shared model. No projection is "primary."

## Why not a node editor (and why not any single surface)

A conventional node editor leaks the implementation as the interface — it promotes the *general internal representation* (the graph) to the editing surface, violating [general-internal-constrained-api](./general-internal-constrained-api.md). But the deeper finding is that **no single surface is correct**. A design exploration generated five decorrelated interface paradigms (artifact-records-history / signal-performance-instrument / literate-document / reactive-formula-inspector / better-structured-graph). Adversarial judging showed each was strongest in a *different* regime and each fell back to "the graph returns as view-source." They are not rival paradigms — they are **projections** of one model. The escape-hatch graph is a defeat only if it is the *sole* place a workload is authorable; it is honest when it is a lossless peer projection.

This matches the ecosystem's interaction-graph thesis ("the paradigm is the graph, not the pixels; frontends are projections") and dusklight's projectional-viewer model ("everything is data and functions over data; there is no read/write asymmetry").

## The source of truth: the typed op-graph

unshape already has the substrate, per [ops-as-values](./ops-as-values.md):

- Every operation is a serializable struct implementing `DynOp` (`input_type`/`output_type`/`apply`), authored via `#[derive(Op)]`, registered in an `OpRegistry`.
- "Graph is just `Vec<Op>` + wires" — the graph is a collection of ops, not a separate system.
- `Value`/`ValueType` give the dynamically-typed dataflow.

This already satisfies the hard prerequisite most tools cannot meet: **actions are data**. Every capability is a queryable, introspectable, serializable entity — not behavior buried in UI code. The op-graph is the general internal representation; each projection is a *constrained API view* over it.

## Projections are lenses

Each surface is a renderer handed a **composable reactive lens** focused on its slice of the graph — the model ported (in design, not code) from dusklight: `ReactiveLens { signal, set, modify, focus }`, built by composing optics root→leaf, with **no read/write asymmetry** (reads and writes flow through the same optic). A surface is not a separate editor; it is a typed rendering of edges over the common graph, and an edit in any projection is the same model mutation expressed through a different affordance.

Co-equal projections (none primary):

| Projection | Strong for | Ancestor in the bar |
|---|---|---|
| **Op-stack** (linear, Blender-modifier-like) | single-artifact procedural lineage; insert/reorder | Blender modifier stack |
| **Timeline / signal-lanes** | temporal & audio-reactive work; live performance | After Effects, DaVinci, Pure Data |
| **Structure inspector** (formula/typed) | precise parameter & expression authoring | spreadsheets (done right) |
| **Literate document** | archival, explanation, exploration | notebooks, Bret Victor |
| **Node-graph** (view-source) | dense DAG audit, navigation, debugging | Houdini, Pd |
| **Direct-manipulation canvas** | the gestural inner loop (sculpt/draw/model) | Plasticity, ZBrush |

Which projection is *foregrounded* is chosen by predicted intent (recency, specificity, workflow position, selection shape) — **not** by object-type or mode, both of which the affordance docs identify as too-coarse proxies.

## One op, many affordance types

An op does not have one appearance. Per the affordance-type taxonomy (command / gestural / ambient / navigational / directional / data-entry), the *same* op renders as the affordance type its surface supports: a `Displace` is a **gestural** handle on the canvas, a **data-entry** field in the inspector, a **command** in a palette. Plasticity proves the key discipline — the gizmo and the numeric field are **the same parameter, two affordances, simultaneously live** (drag while typing `x12` constrains axis and sets the exact value through one update path, not two).

## The gestural inner loop

The hardest unsolved problem in creative-tool UIs — and the one all five paradigms missed — is the **deep gestural inner loop**: sculpt, draw, model, paint as *first-class editable ops*, not baked data. Plasticity (NURBS modeler) is best-in-class here, and its mechanism transfers directly:

- **Factory state machine.** Each gesture drives a live *preview evaluation* of a pending op, distinct from commit: it renders a temporary artifact, drops stale frames when the user drags faster than evaluation, and tolerates transient-invalid parameters mid-drag without breaking. This async-coalescing state machine *is* the fluidity. Generalize it as a `preview`/`commit` distinction on every op.
- **Gesture-first, op-recorded.** A drag does not record N tiny ops — it continuously rewrites the parameters of *one* pending op; only the final committed parameter set lands in the graph. The gesture authors an op-as-value whose fields are filled by direct manipulation, then commits atomically.
- **Command lifecycle = atomicity + undo boundary.** Each gesture is a command-scoped transaction; a half-finished drag never pollutes the graph or history. Undo granularity is the command, not the keystroke.
- **Quasimodes** (held, spring-loaded sub-modes) layer snapping/constraints onto a live gesture without modal lock-in.
- **Snapping / construction-plane as a decoupled input-transform layer** between raw pointer and op-parameter, reused across domains.

**Adopt Plasticity's front-of-loop; reject its back-of-loop.** Plasticity commits to baked B-rep with a linear memento-snapshot undo and *no* recompute graph — the deliberate inverse of unshape's premise. We keep the committed gesture **parametric and re-evaluable** in the op-graph. This is also the generalization of the MoI/Elephant finalization/staging model already noted in [prior-art](../prior-art.md): stage a live preview, commit to a recorded op-log.

Domain caveat: a spatial gizmo generalizes to mesh and vector; audio/motion have no spatial gizmo — there, direct manipulation is timeline scrubbing, breakpoint dragging, envelope editing. The unifying abstraction is *gesture → live op-parameter binding → commit*, not the gizmo itself.

## Unified time (and space)

Five independent designers blindly converged on the same primitive, which makes it the most load-bearing settled decision: every input is

```
Value = Const(v) | Curve(sample over t) | Signal(live)
```

with one `sample(t) -> Value`. A constant is a flat curve; a curve is a precomputed signal; a live signal is a curve not yet frozen — one sum type, three fills, promoted between states with no mode switch. This unifies the three time models the bar models incompatibly (keyframe-timeline / state+modal / realtime-signal) and is built on the existing `Field<I, O>` abstraction.

The generalization past SOTA: `sample(u)` where `u` is **time, or arc-length, or UV** — "over time" becomes "over a parameter," so a static mesh gradient and an audio envelope are the same kind of value. This is consistent with [philosophy](../philosophy.md)'s Realtime Editing section, the `SeekBehavior`/checkpoint-replay semantics in [time-models](./time-models.md), the `EvalContext { time, sample_rate, resolution }` host model, and "the graph is the artwork; its expression is ephemeral."

## Typed holes and synthesis-grade completion (model-level)

Typed holes belong to the **model**, not to any one surface — they are valuable in every projection. A hole is a *typed gap in the op-graph* with an expected type; each projection renders the same hole natively (an empty op-stack slot, an unfilled formula argument, a dangling node input, an empty lane) and offers the same fills.

This is what rescues the structure-inspector projection from being Excel's formula bar (the canonical failure; Notion/Sheets are no better; Desmos is decent but not for code). Excel does token-level autocomplete over an open string grammar. unshape has a **finite, typed, registered op algebra** (`OpRegistry`): every candidate fill is enumerable, type-checkable, and rankable. **Type-directed program synthesis over a registered op algebra is far more tractable than general LSP** — holes are typed, candidates are bounded, and `Expr::free_vars()` (dew/wick-core, `introspect` enabled) supplies dependency structure for free. Rank fills by predicted intent (recency / specificity / workflow position / selection shape; no ML required). The result is structural completion, not textual — the "beyond-LSP" bar, reachable *because* of ops-as-values.

## Live preview everywhere

The deepest reason node editors fail grokkability is *abstraction-blindness*: opaque boxes, unlabeled wires. The universal graft, applicable to every projection: **every op-slot and every edge shows its live output thumbnail**, with semantic zoom (gestalt at distance, detail on approach). This makes even the view-source node-graph legible, converting the "graph returns" escape hatch from a defeat into an honest, readable peer projection.

## Surface discipline

Each projection obeys the affordance-surface rules: at most ~7 scannable items at every hierarchy level, achieved by **removal, not prioritization** (irrelevant affordances are absent, not demoted); **pinning, not sorting** (stability is earned per-item, so muscle memory survives switching projections — pinned items hold position, unstable items flow); the command palette is an **escape hatch, not primary navigation** (palette-as-primary signals the contextual model has failed). If a projection needs heroic filtering, the op vocabulary is too multiplied — generalize the primitives rather than filter harder.

## Framework and stack

**Hybrid: Rust-native, dusklight's design.**

- **Shell:** egui (immediate-mode; the inspector/stack/timeline surfaces are dense, field-heavy panels — egui's strength). Native-first, wasm-capable later.
- **Render core:** wgpu, via the existing `unshape-gpu` (`image_expr`) — dew→WGSL compilation is the *only* live-preview engine that already ships, and it is the cheapest validated path.
- **Heavy multi-preview viewport:** optional bevy backend behind the egui shell (ECS is a genuine fit for N live preview render-targets), swappable, not the app frame.
- **Expression substrate:** dew (wick-core) is the single ecosystem expression language. We port dusklight's optics/`ReactiveLens` *design* clean into Rust; we do **not** reuse dusklight's TypeScript code or its Marinada language (that divergence — one expr language ecosystem-wide — is the deliberate trade for staying native).

## Reality checks (the critical path)

Three facts from the actual codebase gate everything:

- **`IncrementalEvaluator`, `apply_into`, `apply_gpu` are design-stage and unimplemented.** They are specified in [editor-integration.md](./editor-integration.md) (dirty-tracking, cached GPU buffers, "undo/redo is free" via buffer swaps, the Interactive-Editing and History-Replay scenarios) but exist only as prose. **Every** realtime/live-preview claim in this document depends on building them. This is the true critical path. The repo's current undo is coarse graph-edit event-sourcing, not free buffer-swap undo.
- **No realtime audio I/O exists** (no cpal/rodio). The timeline/signal projection's *audio* half needs a greenfield realtime audio engine — defer it; build the visual projections first.
- **dew→WGSL is real and shipping** — lead with it.

## Relationship to prior art

- **MoI / Elephant** (already in [prior-art](../prior-art.md)) — the closest existing projectional editor: lazy eval + dirty tracking, history-via-op-log, finalization/staging. We generalize it to multi-domain and multi-projection.
- **Plasticity** — gestural front-of-loop (adopt: Factory preview state machine, gizmo↔numeric dual-affordance, quasimodes); baked back-of-loop (reject: keep ops parametric).
- **Pure Data** — dataflow/signal model; ancestor of the timeline/signal projection.
- **After Effects / DaVinci / Houdini** — the capability+UX bar; each is essentially one of our projections, unified here rather than chosen.
- **dusklight** — the projection mechanics: `ReactiveLens` over composed optics, no read/write asymmetry, layout-as-data, ranked switchable renderers.

## MVP (first validated slice)

Build the smallest slice that leans entirely on shipping code and resolves the critical-path risk:

> A single egui + wgpu window showing one live texture, driven by an ordered list of dew/`UvExpr`/`ColorExpr` modifiers, where editing any parameter recompiles the WGSL and repaints in realtime, `Expr::free_vars()` highlights which inputs each modifier reads, and the **same op is shown in two co-equal projections** (the op-stack row and a structure-edited formula) that round-trip through one lens.

This validates the three load-bearing claims at once — lens round-trip across projections, live preview, and whether interactive re-evaluation needs `IncrementalEvaluator` *before* committing to scale — in days, on the image domain where every dependency already exists. Explicitly out of MVP scope: a playhead (no time model implemented yet), realtime audio (no I/O layer), and any custom retained canvas.

## Open questions

- **Branching vs linear history.** philosophy wants a fully-rewindable project file (`.psd`-but-rewindable); Plasticity is linear. The projectional model should support branching exploration — needs a concrete history representation beyond the current event-sourced undo.
- **Where the gesture commits.** The preview→commit boundary is per-domain; mesh sculpt, vector draw, and audio envelope each need a committed-op granularity that stays re-editable without exploding history.
- **Audio realtime engine.** Greenfield (device I/O, callback thread, lock-free param ring); schedule it after the visual projections prove the model.
