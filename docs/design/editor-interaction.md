# Editor Interaction Model

The concrete interaction design for the unshape editor, layered over the [projectional editor](./projectional-editor.md) model. Synthesized from a five-way design exploration (five decorrelated candidate interaction models) and three adversarial judging passes against an explicit UX rubric.

## The rubric (the scoring function)

The user's definition of good UX, used as the judging rubric throughout:

1. **Simple mental model.**
2. **Not 10,000 things to learn.**
3. **Surface the most useful affordances naturally.**
4. **Escape hatches** (offer a way deeper / a way out).

Plus the affordance lenses (see the `affordance-*` notes in the github-io docs): **affordance-types** (one op renders as the affordance *type* its surface supports — gestural drag-handle vs command vs data-entry; gestural primary, command a co-equal twin, data-entry the escape hatch); **affordance-surfaces** (≤7 scannable items, achieved by *removal* not prioritization; palette is an escape hatch, not primary navigation; stability is *pinned per-item*, not globally sorted); **interaction-graph** (foreground by *predicted intent* — recency / selection / workflow — not fixed always-on panels); **affordance-opacity** (actions are queryable data).

## What's settled (five blind designers agreed)

These were independently reinvented by all five candidates, so they are treated as correct, not optional:

- **Full-bleed live preview is the default.** Kill the always-on side panels.
- **The op-stack collapses to a minimal, self-describing always-on representation** (live thumbnails, not rows of sliders+buttons).
- **Add-a-modifier is a fuzzy search/palette, never a flat menu** — the unanimous fix for "add doesn't scale to many op types."
- **Reorder is drag (gestural); up/down/× are a faded *fallback*.**
- **Units/precision are a display lens on the parameter descriptor** — stored canonical (radians, f32), shown human (degrees, `1.7` not `1.7000000476837158`). The lens is read by *every* projection (slider and formula alike), so the representation-leak is structurally impossible to surface.
- **The formula/structure is summoned per-op, never shown unconditionally.**
- **Projections are foregrounded by intent; none are pinned to a fixed location.**
- **Delete the debug clutter** — "parsed ok", type signatures, "editable" labels, descriptions, and select-all-text-by-default were debug output leaking into the UI.

## The synthesis: Reading Stack + Command Twin

The winning design is the union of each judge's winner. No single candidate; a hybrid whose parts each survived adversarial attack.

### Skin — the Reading Stack (from candidate E)

The workspace is a **vertical column of live outcome-tiles**: each tile shows *the image as of that step* (not an abstract node), labeled in human terms. Read top-to-bottom like a comic strip of the edit; the bottom tile is the final result, rendered largest. This wins because **legibility is intrinsic to the representation** — a tile *shows* what its op did, so op identity is visible at rest with no click, and survey/compare ("which op darkens the shadows?") is native. It matches how artists narrate their own work.

### Op selection — the live-preview-chip palette (from E; unanimous best mechanism)

Adding an op opens a palette rendered as a **grid of live preview chips: the user's own image under each candidate op**. You pick by *recognizing the outcome*, not recalling a name — recognition over recall, zero learned vocabulary. Type-to-filter handles the long tail; chips are ranked by predicted intent (recency / selection / workflow) but **visible and stable** (frequently-used ops earn a pinned slot; prediction only fills empty slots, never reshuffles earned ones, and never moves anything while you are looking at it).

### Organizing rule — name the verb, manipulate the spatial (from candidate D)

The rule that tells you where any action lives: **named operations** are summoned (palette / command); **spatial manipulations** happen directly on the large preview (gizmos). One predictive rule instead of memorized locations — the essence of a simple mental model.

### The Command Twin — co-equal, never hidden or faded (from D; required by the accessibility finding)

A first-class, always-reachable command/keyboard surface that is the **full-parity twin of every gesture** — same target value, same precision, same discoverability. It is *not* a faded fallback (that is a defect — it rots and fails keyboard / screen-reader / motor-impaired users, who have only that path). The hard bar: **disable the pointer and every operation, including exact values and bulk edits, must be reachable at full precision.** The command twin also carries the power features the gestural skin is bad at: exact values, bulk/set editing, and scripting/animation authoring.

### Parameters

- **Units as a display lens** (degrees, `1.7`) — stored canonical, shown human.
- **Spatial params → on-canvas gizmos** (direct manipulation on the preview).
- **Non-spatial params → a *named, visible* scrub affordance on the preview** — candidate B's co-located drag-on-the-thing feedback loop, made discoverable by candidate E's named target (a labeled axis/handle, not an invisible "the whole image is secretly a slider"). Plus a type / wick-expression escape hatch.
- **Never bind a value-changing or destructive action to an unlabeled gesture** (no invisible image-slider, no fling-to-delete-without-undo).

### Projections, foregrounded by intent

One typed op-graph; projections appear when intent calls them, none pinned:

| Projection | Foregrounded when |
|---|---|
| Reading Stack (op-stack) | always — it *is* the canvas |
| Per-op param controls | selecting / zooming into a tile |
| Formula / structure (wick) | semantic-zoom deeper into a tile, on demand |
| Command twin | always reachable (a keystroke), full-parity |
| Timeline | a parameter is animated / signal-driven |
| Node-graph (view-source) | summoned to *verify* topology — never the authoring surface |
| Literate-doc | summoned for export / narrative |

### Depth — collapsible sections + a structural filter

A 30–50 op stack is never 50 peers; it is "color grade (8), retouch (12), background (10), output (6)." **Collapsible named sections** (nameable, reorderable as a unit, parameter-addressable as a unit) turn 50 rows into a few legible chunks. Complemented by a **stack-wide structural filter/query** (`type:curves`, `param:radius>4`, `affects:shadows`) for the survey case and inherited/unorganized stacks. Both are required — sections give legibility *when you organized*, filter gives findability *when you didn't*.

### Branching — named-reference input slots, not wires, not embedded sub-trees

A multi-input op (blend, mask-composite) declares **named ports** (`base`, `over`, `mask`). Each port is filled by a **reference, by name, to a first-class top-level stack** — rendered inline as a one-line chip (`over ← glow-pass`), not by nesting that stack's tiles inside the slot, and not by drawing a wire. The topology becomes a *flat list of named stacks that cite each other* — readable as an outline (like a spreadsheet's named ranges), not a 2D graph needing layout. The common case stays a single linear column; branching is opt-in and appears only when a 2-input op is added. The node-graph is a **derived view summoned to verify topology**, never the primitive you author in — which is how this honors the "no node editor as primary surface" mandate (data-over-code at the seam: serializable named references, not embedded sub-trees or hidden closures).

### The blind spot we design *for* — variants and sets

The single most common real creative task is the one **no candidate (and few real tools) handle**: every design treats an edit as a single `(op, value)` cursor, but production work is *sets and variants*.

- **Variants / forks:** "show this op at N parameter values side by side", "compare blur radius 4 vs 8", "try a few and keep one." There is no native fork/compare affordance in any candidate.
- **Bulk set-addressing:** "set exactly 0.5 across 40 layers", "the same hue on the whole color-grade section" — the stack must be addressable *as a set* (multi-select, or a query → edit-once), which the command twin's query surface is the natural home for.

v1 may defer these, but **the data model must not preclude them** — a parameter is potentially a *set of variant values*, and an edit potentially targets *a selection*, from the start.

### Performance — lazy, downscaled tiles

Live-outcome-tiles risk N GPU passes per edit (change op #3 → tiles #3..#40 invalidate). Mitigation: render the focused tile + its neighbors at full resolution, off-screen / distant tiles at low resolution and lazily, and lean on the incremental-evaluation / cached-buffer machinery specified in [editor-integration.md](./editor-integration.md) so an edit only re-runs the pipeline *from the changed op forward*.

## The non-negotiable floor

Below all three of these, "minimal" becomes "blank confusing screen" — the design has moved the missing UI into the user's head and called the result clean:

1. **A visible, no-prior-knowledge way to add the first op** (not a hidden keypress).
2. **Op identity visible at rest, without a click** (a labeled live tile passes; an anonymous dot fails).
3. **No destructive or value-changing action bound to an unlabeled gesture.**

## What was rejected, and why

- **Dot-strip minimalism (candidate A):** dots carry no identity — a memory test masquerading as simplicity; fails survey, deep stacks, and floor rules 2–3.
- **Command-spine as the *primary skin* (candidate D as a whole):** the add-surface hidden behind a keypress fails floor rule 1. D's command surface is adopted as the *co-equal twin*, not the skin.
- **Invisible "image-as-slider" as the primary param affordance (pure candidate B):** undiscoverable and an accessibility regression; adopted only as a *named, visible* scrub.
- **Embedded sub-pipeline input-wells (candidate E's branching mechanism):** a real node graph with no layout — worse than the rejected node editor; replaced by named references.
- **Unbounded adaptive content (candidate C in full):** content that moves under you defeats spatial memory and violates rubric rule 1; only the *fixed-geography + pinning-earns-stability* subset is kept.

## Relationship to the projectional model

This is the concrete interaction layer over [projectional-editor.md](./projectional-editor.md): the tile column is the op-stack projection; semantic-zoom-into-a-tile reveals the structure-inspector projection; the command twin is the affordance-opacity "actions as data" surface made manifest; live-preview-everywhere is what the tiles render; units-as-display-lens and summoned-per-op formula are the projection layer doing its job. The five projections round-trip through the same reactive lenses described there.

## Implementation path (incremental, on the current MVP)

Each step is independently shippable and moves the MVP toward this design:

1. **Units as a display lens** — give each parameter a descriptor (unit + display precision + canonical↔human transform); render degrees, strip trailing zeros. Fixes the radians / `1.7000000476837158` / ugly-formula class at the model level. (resolves c, g)
2. **Declutter + summon the formula** — remove the always-on formula panel, "parsed ok", signatures, labels, descriptions; text not selectable by default; formula reached on demand per-op. (resolves d, e, h)
3. **Op-stack → live outcome-tile column** — self-describing tiles; drag-to-reorder; remove via labeled affordance + undo (not unlabeled fling). (resolves a, b, and the floor)
4. **Add-modifier → live-preview-chip palette** with type-to-filter and intent ranking. (resolves f)
5. **The command twin** — a full-parity keyboard/command surface for every gesture (accessibility gate) that also unlocks exact values and bulk editing.
6. **Later:** collapsible sections + structural filter (depth); named-reference branching; the variant/set primitives; lazy/downscaled tile rendering.
