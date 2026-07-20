> **STATUS: working document — frames the open design space, answers accumulate here as they're resolved. Nothing here is blessed until explicitly marked.**

# UI Architecture

The computation substrate is sketched (graph, fields, ops, timeline, table, live). This document designs how to project it for humans.

## Settled ground (pointers only)

- **Interaction principles**: `editor-interaction.md` — plural modality, direct-manip as one among many, approximate/exact co-equality, invariants-owned-by-tool, variants/sets first-class, promote-in-place, continuous audition, gesture-as-data, relevance-via-frecency+tags+context, config-as-data.
- **Projection ontology (candidate)**: `projection-model.md` — projection = (query + render-spec + gesture→rewrite) as data; graph-as-value ontology; lossy boundaries made visible.
- **Platform**: Rust / egui / wgpu.
- **Backend perf**: `editor-integration.md` — apply_into/apply_gpu, buffer caching, undo-as-buffer-swap.
- **Anti-slop**: determinism (resolved-delta-only events), CI-gated render-primitive count, ban on per-op inverses, UI-state firewall.

## Thread 1: Projection Types

Make the projection model implementable. The surviving synthesis from `projection-model.md`:

> projection = (query over model + render-spec + gesture→rewrite map), authored as data

### Open questions

1. **What is a query?** Selects which part of the graph a projection shows. Must stay a Rust builder/combinator (no DSL). Risk: creeping into a full query language.

2. **What is a render-spec?** Maps selected data to visual primitives. Must stay idiom-bounded (~6–10 render primitives, CI-gated). What are the primitives?

3. **What is a gesture→rewrite map?** Maps user interactions back to graph edits. Must record only resolved structural deltas (determinism rule). How does disambiguation work before recording?

4. **What is the Projection trait?** The unifying Rust type that composes these three into something the editor can mount, layout, switch between.

5. **Lossy boundary UX** — when a projection can't show everything (e.g., a linear view of a branched graph), how is the boundary made visible? Named "let-binding / refusal / boundary-token" but not designed.

### Dependencies

- Hits the **carving problem** when defining what queries select over.
- Hits the **rewrite substrate** when defining gesture→rewrite.
- Render-spec depends on knowing what **layout containers** exist.

## Thread 2: The Carving

~300 primitives across 6 implementation domains. How should the editor organize them for humans?

### The tension

- **Implementation domain**: mesh ops, audio ops, image ops, vector ops, field ops, physics ops. Clean crate boundaries. But a user thinking "blur" doesn't care whether it's image-blur or mesh-smooth or audio-lowpass.
- **Human cross-cutting concepts**: noise, field, warp/deform, blend/mix, filter/convolve, oscillator/LFO, scatter/instance, repeat/tile. These recur across domains with colliding names (already disambiguated in code: `SurfaceScatter`/`VolumeScatter`, `DomainWarp`/`TimeWarp`).
- **Relevance**: the mechanism (tags + frecency + context, candidate pool, pinning, disuse-only eviction) is agreed. But relevance *cannot be computed* without answering what the candidate pool is organized by.

### Open questions

1. **Is the carving a taxonomy or a tag cloud?** A taxonomy implies hierarchy and single-parent assignment. A tag cloud implies flat, multi-label, queryable. The codebase already has machine-extractable structural facets (optype, arity, domain-dim, pure-vs-recurrent) — are human concepts layered on top as tags, or do they replace the structural facets as the organizing axis?

2. **Who maintains the carving?** Machine-derived from code structure (stable, no maintenance burden, possibly unintuitive)? Human-curated (intuitive but rots)? Hybrid (structural base + human synonym layer, per S5)?

3. **Does the user ever SEE the carving?** Or is it purely an internal relevance-computation input, invisible behind ranked results? The "Excel without the sin" framing suggests spatial organization is user-authored, not system-imposed.

4. **The image/texture acute case**: 107 `Field<Vec2,f32>` impls, all type-identical. Named the "WORST slop risk." What distinguishes them in a relevance context? (Structural facets don't help — they're all `Field<Vec2,f32>`.)

### Dependencies

- Gates **relevance computation** for all domain surfaces.
- Feeds into Thread 1's query vocabulary (what can queries select on?).

## Thread 3: Editor Architecture

How graph editor, freeform canvas, tiled panes, and floating windows share a window.

### Settled constraints

- Layout is **editor config, not graph data** — layout state is UI-side, firewalled from the deterministic graph.
- Multiple projections coexist — "there is no single correct representation."
- The graph/node editor is **not primary** — it's one projection, openable ("view source"), never mandatory.

### Open questions

1. **What is the layout primitive?** A pane? A split? A tab? A floating window? All of the above with a unifying container type?

2. **How do projections bind to layout slots?** Is a "pane" just a mounting point for a projection? Does the pane know what projection it holds, or does something external manage the binding?

3. **Freeform canvas semantics**: "Excel without the sin = freeform canvas with independent structures." What does this mean concretely? A 2D infinite canvas where the user places projection instances spatially? How does this differ from a tiled layout?

4. **Navigation between projections**: When a user "opens" a node in a graph view, does it spawn a new pane? Replace the current one? Float? Is this configurable per-user, or does the system decide?

5. **Focus/selection propagation**: When user selects something in one projection, do other projections of the same data highlight it? (Cross-projection selection sync.) What about selections that don't have meaning in other projections?

### Dependencies

- Depends on Thread 1 (what IS a projection, what does "mount one in a slot" mean).
- The freeform-canvas question connects to Thread 2 (if the canvas IS the organizing principle, then "the carving" might be spatial rather than taxonomic).

## Coupling map

```
Thread 1 (Projection Types)
    ├── needs Thread 2 (what do queries select over?)
    ├── needs Thread 3 (what containers do render-specs target?)
    └── needs rewrite substrate (gesture→rewrite)

Thread 2 (The Carving)
    ├── feeds Thread 1 (query vocabulary)
    └── feeds Thread 3 (if canvas = organizing principle)

Thread 3 (Editor Architecture)
    ├── needs Thread 1 (what is a projection?)
    └── informed by Thread 2 (spatial vs taxonomic organization)
```

All three are coupled. Progress on any one will force partial answers on the others.

## Resolved decisions

(Empty — decisions accumulate here as threads are worked.)

## Design Principle

**The job of a good UI is minimizing distance (number of steps) between intent and artifact.**

Not "each step moves you closer" (that's convergence — a low bar). The distance itself — the total number of actions between having an intent and holding the result — should be as small as possible.

## Grounding: 2D Animation / Drawing / Rigging

Abstract architecture divorced from real use cases produces beautiful designs that don't fit half of reality. Ground the design in concrete creative work.

### The full arc (currently split across separate apps)

1. **Drawing/sketching** — creating art from scratch (Procreate, Clip Studio, ToonSquid)
2. **Vector art** — clean vector drawing with SVG export (surprisingly few good options exist)
3. **Image editing** — compositing, masking, color (Photoshop)
4. **Rigging** — skeletal structure, deformers, weight painting (Live2D, Toon Boom, Spine)
5. **Animation** — posing, keyframing, timing (Live2D, Toon Boom, After Effects + DUIK)

**Tool boundary collapse is a major distance reduction.** The round-trip between apps (draw → export → import → rig → realize arm needs fixing → back to drawing app → re-export → re-import → re-rig affected region) is where huge distance accumulates. In a unified graph, fixing the arm automatically propagates downstream — the rig is already looking at the updated art.

### Input modalities for posing (closest to DID)

The Dinosaur Input Device (Jurassic Park, 1993): a physical armature with joint sensors that drove the CG rig directly. The animator's hands shaped the motion with zero abstraction. The historical trend in rig interaction has been toward removing indirection: numeric entry → FK gizmos → IK handles → control rigs → direct surface grab → physical input → body-as-controller.

**Collision over mapping.** Interaction via physical collision in the virtual workspace (hand pushes character's limb, it moves) beats abstract parameter mapping (hand curl → parameter X). The collision approach has zero learning curve — the mental model is just physics.

Modalities, ordered by accessibility:

| Input | DOF | Requires | Best for |
|-------|-----|----------|----------|
| **Hand pose via CV + collision** | ~25 | Webcam (universal) | Quick posing, non-humanoid shapes, improvised topology |
| **Face detection** | ~50 (ARKit blendshapes) | Webcam | Sketch-level expression/head keyframes |
| **Body tracking** | ~30+ | Webcam + space | Full-body performance, motion |
| **Physical armature + CV** | Per-armature joints | Armature + camera | Sustained precise posing, holds pose, tactile feedback |
| **Full mocap** | Highest | Suit/markers/studio | Highest fidelity full-body capture |

**Practical gate:** Whether to implement any of these depends on whether high-quality open source solutions exist to build on (MediaPipe, OpenPose, ARKit, etc.), not on building pose estimation from scratch.

**Key properties of hand tracking for posing:**
- Universal — everyone has hands, most devices have cameras
- Show the camera when done (no need for continuous tracking if awkward)
- Can represent arbitrary topology with improvised gestures (curl for mouth shape, spread for wing, etc.)
- The armature is the "enthusiast upgrade" — tactile, holds pose without effort, can look at it directly

### Prior art

- **DID** (Jurassic Park) — physical armature with sensors driving CG rig
- **Live2D** — 2D deformation via parameter sliders / face tracking
- **Toon Boom Harmony** — 2D bone rigging, deformer chains
- **Spine** — 2D skeletal animation with IK
- **Procreate / ToonSquid** — drawing/frame-by-frame on tablet
- **Houdini KineFX** — direct surface-grab posing (geometry-based, not handle-based)
- **VTuber pipeline** — face/body tracking → Live2D/3D avatar in real-time
- **Corridor Crew DID video** — popularized awareness of physical-input-for-digital-animation

### Mode switches and input channels

A mode switch is an action that produces no creative output — it only changes what future actions mean (e.g., stylus switches from "draw" to "move bone"). Activating a parallel input channel (e.g., turning on the camera) is NOT a mode switch — it adds an input stream without changing what existing inputs do.

Of the posing input modalities:
- **Body tracking / hand+armature CV**: no mode switch. The camera is a parallel channel — activating it doesn't change what the stylus does. You can draw and have CV input running simultaneously.
- **Touch + IK**: requires mode switch on devices without finger/stylus hardware disambiguation. On devices that distinguish finger from stylus, finger = pose is viable without switching.
- **Traditional keyframing**: requires switching to parameter editing context.

### CV capture patterns

CV pose input is NOT always-on (waste of battery/compute). The capture pattern depends on the use:

- **Keyframe pose** (snapshot): strike a pose (body or armature), capture a single frame, apply to the rig at this point in the timeline. Armature naturally fits this — pose it, capture, pose again, capture. Stop-motion workflow.
- **Animation clip** (bounded recording): perform a motion (body), system records the movement over a duration. Start → perform → stop, like audio recording. Better suited to body tracking than armature (armatures come from stop-motion, not fluid continuous motion).
- **Iterative exploration**: rapid capture-and-compare — try poses, keep the good one.

### Collision over mapping (elaborated)

When CV-tracked input (hand, body) is projected into the virtual workspace, interaction should use **collision** — the projected hand/body pushes/grabs virtual objects by touching them — rather than abstract parameter mapping (hand curl → value X, finger spread → value Y).

Collision requires zero learned abstraction. The mental model is physical: push the thing, it moves. Mapping schemes require the user to learn and remember arbitrary correspondences.

Tradeoff: collision requires the virtual workspace to have physics/collision detection. Mapping is computationally simpler but cognitively harder.

## Backlog (not yet threaded)

- **Editing-through-a-definition** — bidirectional editing of formula-defined values. Repeatedly flagged as the hardest unsolved interaction problem. Scoped out pending constraint-solver substrate.
- **Events/interactivity** — deferred, needs ground-up design, not patch-fitting.
- **Rewrite substrate** — matcher, application strategy, boundary primitive, DAG gluing. Needed for collapse/expand and view-source. No design exists.
- **Save/history implementation** — requirements recorded (append-only event log, atomic fsync, auto-restore), implementation open.
- **Determinism/UI-state fencing** — the crate boundary that keeps relevance/frecency out of op-crate deps. Rule agreed, fence unbuilt.
