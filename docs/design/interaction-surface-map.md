> **STATUS: navigational map — provisional, crowns nothing, everything here is reopenable.** Consolidates recorded direction (principles + substrate facts) and flags open gaps; it is NOT designed UI and NOT final.

# unshape — Interaction Surface Map

## PART A — THE CROSS-CUTTING SPINE

Shared invariants every surface must obey; violating one = accreted slop. Status: R recorded/agreed · P partial · O open/net-new.

- S1 Projection model — one deterministic model; co-equal projections (node/stack/list/literate/formula/timeline); no projection canonical; node-editor-as-primary rejected. R principle / O mechanism (no projection concept in code).
- S2 Plural modality, evoked not picked — direct-manip/radial/palette/shortcut/formula simultaneous; input evokes the modality; shortcuts shown inline, graduate to eyes-free. R / O (which subset shows when).
- S3 Direct-manip is ONE approximate modality, NOT primary — nothing is primary (don't bless one modality). Often insufficient; exact (type/formula/numeric, unit/tempo-snap) co-equal on the same field; formula/code is a fallback. You manipulate generative parameters and fields, not baked outputs (low-dim manip over high-DOF baked output = frustration; output→definition inversion is backlogged). See projection-model.md. R.
- S4 Relevance = pinning + disuse-only eviction — used items are pins (hard positional constraints); unpinned slots flow by frecency+context; eviction by disuse only, never displacement; cold-start optimizes freely; stability earned per-item. R rule / O signal-set+weighting, recurrent-bucket ordering.
- S5 Tags/aliases = stable query-expansion — system+authored tags/aliases widen what a query matches; deterministic given query; distinct from frecency; human layer may only add synonyms, never be an entry. R.
- S6 Membership = machine-extracted hard filter — typed/structural (optype, arity, domain-dim, pure-vs-recurrent, facet); CI-enforceable; legality layer; stable position. R direction / P carving OPEN.
- S7 Time model: constant|curve|signal — any value is constant, authored curve, or live signal, one promotable continuum; gesture=authored data; EvalContext carries time. R.
- S8 Promote-in-place / named first-class signals — constant→signal/variation/formula is one in-place gesture; graph forms behind you, openable, never primary; late-bound named signals. R.
- S9 Per-connection remap — one source→many targets; each edge carries its own range/curve/retiming. R.
- S10 Variants & sets first-class — compare-variants / vary-per-X / shared-tokens-with-local-override; config-enum families collapse INTO this. R.
- S11 Graph-rewrites; recipes = rule-sides; collapse/expand = projection — a rewrite is a value; recipe is a rewrite rule (collapse=backward, expand=forward); boundary = which side shown; re-collapse structural, zero provenance tags. R direction / O substrate net-new.
- S12 Recurrence rides feedback edges — 9/9 sims ported to pure &self Step + Init seed; state explicit & serializable; seek = Resimulate|Discontinuity|Error. R.
- S13 Determinism / UI-state firewall (anti-slop) — same graph+context=same output; relevance/frecency is UI state fenced in a crate node/op crates can't depend on; fingerprints build-time-pinned; stable tiebreaks. R rule / O fences unbuilt.
- S14 Invariants owned by the tool not the user — tiling, loop-seamlessness, rig re-solve, UV preservation: requested-and-maintained, never babysat. R.
- S15 Continuous audition — edit during playback; scrub updates live; no bake/render/play cycle; realtime default; two regimes (replay vs editing). R.
- S16 Never lose work — append-only op event log = autosave + replay-recovery + persistent undo; atomic fsync writes the one non-free piece; redundancy; fault isolation; auto-restore. R requirements / O implementation (event log manual today).
- S17 Config is data / overlays — zero-config defaults + deep config (tags/aliases/weights/keybinds/layout) as portable, temporarily-applicable overlay. R.
- S18 Projection-purity / no per-feature escape hatch (anti-slop) — features surface through the projection/modality/relevance spine, not bolt-on panels; editing-through-a-definition (inverse/bidirectional) is the hard latent case. P / O bidirectional unsolved.

## PART B — THE SURFACE MAP

For each: (a) what/what-user-does; (b) SOTA tool to beat + strength to exceed; (c) slop/tech-debt prone where; (d) spine constraints; (e) status.

### (1) DOMAIN surfaces

- Mesh/3D — (a) direct viewport manip of procedural geometry, edit-earlier-without-redo, select named features; (b) Blender (modeling+modifier stack) / Houdini (procedural); exceed: topology change w/o breaking UVs/rig as maintained invariant; (c) node-graph escape hatch as "the" 3D surface; UV/rig babysat; gizmo hidden mode; (d) S1,S3,S11,S14,S2; (e) representations recorded, surface O.
- 2D vector/rigging — (a) direct path/network manip, stroke/corner tokens, path-follow brush, 2D puppet; (b) CSP/Krita (vector+brush) / AE (puppet); exceed: shared-token global edit + per-instance override w/o severing link; (c) overrides becoming forks; corner/winding as special-case code; (d) S3,S10,S14,S9; (e) representations recorded, surface O.
- Audio — (a) synthesis+fx as openable archetypes, drive params by env/LFO, tempo-sync, audition by ear, macro knob; (b) Reaper (DAW+routing) / Pure Data (signal-flow); exceed: archetypes-as-recipes; seamless recurrence; (c) stateful seek glitching; rate boundaries as hidden conversions; accidental recurrence wires; (d) S7,S12,S15,S8,S3; (e) primitives+recurrence recorded, surface O.
- Image/texture/noise — (a) empty-texture surfaces only relevant generators (not 107-impl wall), vary-per-cell, edge-wear, seamless tiling invariant, LUT scrub-compare; (b) Krita/Paint.NET (paint) / Houdini COPs; exceed: relevance over 107 type-identical Field<Vec2,f32> (Perlin vs Worley opposite relevance); (c) WORST slop risk: naive "show all" = flat wall of 107; tiling babysat; per-cell hand-wired; (d) S4,S5,S6,S10,S14; (e) primitives recorded, relevance carving OPEN (load-bearing gap).
- Motion graphics/rigging — (a) spring-stagger, seamless particle loops, data-driven charts, secondary-motion-as-relationship, record-live-wiggle-as-curve; (b) After Effects (motion-gfx + keyframe UX); exceed: secondary motion as relationship not keyframes; gesture=authored-data; loop-seamless free; (c) keyframe model creeping back as primary (violates S7); recorded gesture as separate imported type; (d) S7,S8,S9,S14,S10; (e) crates exist, surface largely O.
- Physics/fluid/particle — (a) scatter-500-rocks (noise yaw, painted density, seed reshuffle), fluid/RD/automata sims, painted-mask OR slope-rule OR number on one property; (b) Houdini (DOPs); exceed: sim as rewindable+seedable+serializable feedback recurrence; painted+rule+numeric co-equal; (c) seek/scrub cost hidden; intra-bin ordering of 9 sims is named open frontier; baking as hidden auto-op; (d) S12,S7,S15,S3,S13; (e) migration complete, surface O, intra-bin ordering OPEN.

### (2) CROSS-CUTTING surfaces

- Relevance/quick-actions — (a) "what can I do now": relevant set from selection+features+frecency+tags, pins hold position, full search behind; (b) VS Code palette (multi-modal+inline shortcuts) — its flat fuzzy+recency is the cautionary case to beat; (c) HIGHEST-leverage; vestigial suggested sidebar; reorder under finger; ranker leaking into determinism; (d) S4,S5,S6,S2,S13; (e) rule R, carving OPEN.
- Variant/compare — (a) audition N versions side by side live, pick one, any granularity; vary-per-X; shared-token+override; (b) DaVinci Resolve (gallery/stills, node compare); exceed: compare at any granularity incl token-set, live, vary-per-X dual; (c) compare per-domain instead of once over model; discarded branches lost; (d) S10,S15,S7,S9; (e) principle R, surface O.
- Timeline/time — (a) scrub playhead (live update), authored curves, recorded gestures, per-edge delay/retime, mixed rates, per-domain tracks; (b) AE timeline / Reaper transport; exceed: gesture=authored-data on one track; constant|curve|signal unified; per-connection retiming; (c) mixed-rate hidden conversions; seek glitching; (d) S7,S15,S12,S9; (e) decisions R, timeline UI O.
- Collapse/expand (recipe/abstraction boundary) — (a) show/hide body, collapse-many-into-one + spread-one-across-many, name a recipe, re-collapse free; (b) Houdini (subnets/HDAs); exceed: boundary projection-neutral (node/stack/transclusion/function/track), structural re-collapse zero tags; (c) "collapsed group node only" blesses node projection (S1); inlining destroys abstraction → forbidden provenance string-match; (d) S11,S1,S10; (e) direction R, substrate net-new O.
- Save/history/rewind — (a) continuous autosave, crash recovery, persistent undo, rewind anywhere, auto-restore; (b) Reaper (gold standard) / Krita (failure catalog); exceed: op-granular append-only log makes undo/recovery near-free IF atomic writes added; (c) non-atomic copy-over-original; recovery in volatile temp; accumulators w/o undo/branch semantics → replay divergence; (d) S16,S13,S12,S11; (e) requirements R, auto-record+atomic writes O.
- Inspector/parameter — (a) drag AND type same field (unit/tempo-snap), promote-in-place to signal, surfaced relevant params not all; (b) Blender N-panel / AE effect controls; exceed: drag-XOR-type co-equal; promote one gesture; relevant params surfaced; (c) flat dump of every struct field; drag/type separate widgets; promote as separate dialog (escape hatch); (d) S3,S8,S4,S7; (e) principle R, surface O.
- Formula/expression fallback — (a) exact authoring via wick/dew when direct-manip insufficient; edit existing formula; direct-manip a formula-defined value (back through definition); (b) Houdini VEX / AE expressions; exceed: stays a fallback, one co-present modality, bidirectional; (c) making formula the centerpiece (rejected); editing-through-a-definition/inverse UNSOLVED — where a per-feature hack lands; (d) S3,S18,S2,S1; (e) language recorded, editing-through-a-definition OPEN.
- View-source/graph projection — (a) open the graph promote-in-place built; node/stack/literate/formula views of one model; tree-rewrite view of simplifications; (b) Pure Data / Houdini network; exceed: a projection never the primary; rewrites shown before→after (more grokkable than static wiring); (c) blessing node view as canonical (founding rejection); rewrite order leaking nondeterminism; DAG boundary-gluing unspecified; (d) S1,S11,S13; (e) direction R, no projection concept in code O.

## NET-NEW / undesigned (the real work ahead)

All six domain direct-manip surfaces (representations recorded, interaction O); the projection mechanism (no view concept in code); the rewrite substrate (matcher, application strategy, boundary primitive, DAG gluing); the relevance carving (rule agreed, carving OPEN); intra-bin ordering of 9 recurrent sims; editing-through-a-definition/bidirectional; the latent-vs-shown modality selector; save-system implementation; determinism fences.

## HIGHEST-LEVERAGE — design + adversarially attack FIRST (because everything hangs off them, NOT because only these get built)

1. Projection model (S1) + view-source surface — every surface is a projection of one model; get it wrong → node view blessed or each surface forks the model → systemic tech debt. Spine of the spine.
2. Relevance/quick-actions + primitive carving (S4/S5/S6) — without it every domain surface is a flat wall of hundreds (107 type-identical fields make it acute).
3. Value/time model: constant|curve|signal + promote-in-place + per-connection remap (S7/S8/S9) — most recurring interaction; if incoherent, timeline/inspector/audio/motion each reinvent it.
4. Collapse/expand as rewrite-rule-sides (S11) — grokkability at scale + honors node-editor rejection; substrate net-new; other surfaces depend on its shape.
5. Determinism/UI-state firewall (S13) + save (S16) — invisible poison-the-whole; cheap to fence early, catastrophic to retrofit.
