> **STATUS: provisional — outcome of a decorrelated design + adversarial-attack round. Crowns nothing; the surviving synthesis is a candidate, open questions marked. Everything reopenable.**

# unshape — Projection Model

## Premise correction (important)

An earlier framing treated "direct manipulation is PRIMARY" as a constraint. That is WRONG and contradicts the blessed docs: `editor-interaction.md` records direct manipulation as ONE modality — approximate, often insufficient — and lists "direct manipulation as THE thesis" under **Rejected**. "The node editor is not primary" (S1) does NOT make direct-manip primary; NOTHING is primary (don't bless one modality). The projection model must not be designed around direct-manip-inversion as the main path.

## Principle: manipulate generative parameters and fields, not baked outputs

"Low-dimension manipulation over high degrees of freedom is a recipe for frustration." (user) Dragging one low-DOF handle (a 2D screen point) to control a high-DOF procedural output (e.g. one rock among 500 placed by a hash; a vertex at the end of extrude→bevel→twist) is inherently ambiguous and frustrating. The direct manipulation that WORKS is manipulation of generative parameters and fields — paint a density mask, set a slope rule, reshuffle a seed, vary-per-instance, edit a parameter — all of which are tractable/invertible. Editing an arbitrary baked output by flowing it back through a many-to-one function is scoped OUT (see Backlog: the constraint-solver substrate).

## The inverse problem (why ambitious direct-manip was scoped out)

Concrete kill case (verified in code): `VolumeScatter` places rock #347 at LCG^347(seed) — a hash, no inverse. Dragging it forces every design into one of {refuse, pollute the graph with override nodes, reshuffle all 500, offer no handle}. "Nearest free parameter" is a black box (drag a position, an unrelated upstream knob twitches because it had the smallest delta). The ONLY thing that could make output→definition inversion work is a first-class shared constraint/optimization solver (express the drag as a constraint; a deterministic SEEDED solver finds the minimal free-param change or inserts a least-surprise override; the solver is itself a serializable node). Per-op inverses cannot compose through many-to-one. This is backlogged low-priority, not pursued now.

## The four candidate frames (decorrelated; all provisional)

- **A — LENS:** projection = get/put lens; put emits rewrites; round-trip only "up to e-graph equivalence". Adversarial verdict: silently RE-SORTS the user's stack under them on edit (worst grokkability outcome); rests on NP-hard e-graph extraction; alignment heuristic is a shared fuzzy determinism liability.
- **B — DECLARATIVE TRIPLE:** projection = (query + render-spec + gesture→rewrite) as DATA from closed vocabularies. Verdict: most grokkable to an artist; the only frame whose bespoke code is fenced, countable, compiler-checked. Leak: nearly every interesting projection (timeline, waveform, piano-roll, node, canvas, formula) is a NEW IDIOM = ~1 render primitive each.
- **C — REACTIVE:** reactive views reusing the eval cache. Verdict: the "reuse the cache for free" claim is FALSE — `LazyEvaluator::invalidate` does NOT propagate to dependents (`eval.rs:882`, "for that we'd need dependency tracking"); each non-trivial projection grows its own incremental runtime.
- **D — NO-LAYER:** a view = a render op over graph-as-value (`Value::Opaque(Arc<Graph>)` already exists); editing = upstream rewrite. Verdict: cleanest ontology (4 concepts) but "one-edit delay" framing breaks continuous-audition (S15), and "forward/inverse pair per op" = up to 155 hand-authored inverses with no CI = silent rot (violates the repo's own rot-proof standard).

## Surviving synthesis (provisional candidate)

- **USER-FACING MODEL = B:** a projection is (query over the model + render-spec + gesture→rewrite map), authored as DATA.
- **ONTOLOGY = D's** "a view is a render op over the graph-as-value; edits are upstream rewrites" — BUT drop the "one-edit delay" framing (present as "your edit changes upstream; the view re-derives", not a felt lag).
- **DROP A's** silent e-graph re-sort on edit; fence the "≈ / simplify" rewrite to an EXPLICIT user action, never applied silently while editing.
- Lossy projections must make the boundary VISIBLE (named let-binding for a shared node; an explicit "open in another view" boundary token; an honest refusal) — never silent re-sort or silent guess.
- Co-equality is only as real as edit-tractability: make the lossy boundary visible so the user KNOWS when a view can't faithfully edit something, rather than being silently pushed to the formula/node view (which would secretly bless it — the founding rejection).

## Note: promote-in-place folds in here

**Promote-in-place / "drive this input"** — making a connection from an input to a source from within a non-node projection — folds in here; it is the projection-model gesture that the (now-dissolved) value/time model reduces to. (See interaction-surface-map.md S7.)

## Two hard rules (grounded in the actual substrate)

1. **DETERMINISM:** an edit is recorded ONLY as its resolved structural delta (final params/wires), NEVER as a raw gesture plus a pointer to mutable disambiguation/preference state. `GraphEvent` already stores `new_params` and `replay()` folds resolved deltas, so replay is deterministic — PROVIDED disambiguation is resolved to a concrete choice BEFORE the event is appended and baked into its params. Relational "edit-earlier-without-redoing" re-flow must be expressed as recorded rewrite RULES in the log, not gestures re-resolved against live preference.
2. **ANTI-SLOP (S18):** the "every surface is a projection of one model" promise is real ONLY with two enforcement points — (a) a CI gate on render-primitive COUNT so the idiom set stays bounded (~6-10), not one-per-surface; (b) a HARD BAN on hand-authored per-op inverses — invertibility comes from the shared rewrite layer, not 155 rotting per-op inverses.

## Open questions

- Whether B's render-primitive set actually stays idiom-bounded under real growth (the CI-gate is the bet).
- How visible-boundary UX feels in practice (the let-binding / refusal / boundary-token need real surfaces to judge).
- The exact selector/query vocabulary (risk of creeping toward a query language; must stay Rust builder/combinator, no DSL).
- DAG boundary-gluing for rewrites (DPO/SPO) — shared open item with `graph-rewrites.md`.
