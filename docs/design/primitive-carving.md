# Primitive Carving (open candidates — generated material)

> **STATUS: OPEN. Nothing here is decided, blessed, or recommended.**
>
> This doc records four candidate "carvings" (ways of organizing unshape's
> primitives) and the adversarial attacks generated against them in an earlier
> design session. They are **inputs to an in-progress decorrelated design
> exploration, not its output.** None is crowned. No recommendation is made
> here on purpose. Treat every "C1..C4" below as a live candidate carrying its
> own self-attack — read the cracks as seriously as the principle.

## Framing

The load-bearing open question for the projectional editor's relevance and
discovery surface: **how do we organize unshape's ~few-hundred primitives**
(spanning mesh, vector, audio, image/texture/noise, motion, physics/fluid, and
rig) **so the editor can surface the *relevant few* at any moment** — the small
set of things you'd plausibly want next given what you're holding — **rather
than presenting the whole library?**

This is *the* discovery/relevance question. The editor's "what can I do now"
behavior (see `interaction-trajectories.md`, need #7; `editor-interaction.md`)
depends on having an organizing structure over primitives that is both
*machine-usable for relevance* and *learnable by a human*. A carving is a
proposed answer to that.

### Verified substrate facts

These were checked against current source *this session* and are stated as
established, not conjectured:

- **Statefulness-as-feedback-edge is implemented for a SUBSET of recurrent
  sims, not all.** For *ported* nodes, `DynNode::execute(&self, ...)` is pure
  and simulation state rides a *feedback edge* (`Wire { feedback: bool }`): an
  `*Init` source node seeds the state, and a pure `*Step` node evolves it by
  clone-and-advance. But the feedback-node migration is **incomplete**. Audit
  (source-verified):
  - **PORTED** to pure `&self` Step + Init nodes on a feedback edge —
    `unshape-rd`, `unshape-fluid`, `unshape-particle`, `unshape-audio`
    (vocoder): **4 crates.**
  - **NATIVE-ONLY** (still `step(&mut self)`, no feedback feature) —
    `unshape-automata`, `unshape-physics`, `unshape-spring`,
    `unshape-space-colonization`, `unshape-procgen`: **5 crates.**
  - **NOT recurrent** (correctly no feedback) — `unshape-rig`/IK, a pure
    functional solver `solve(&self, chain, skeleton, pose: &mut Pose)`.

  So: **4 of 9 genuinely-recurrent sims are ported; the migration is
  incomplete.** The accurate claim is: for ported sims, statefulness rides a
  feedback edge with a pure `&self` Step node; native-only sims still mutate
  internal state via `&mut self`. The **pure-vs-recurrent** carving axis and any
  state-TypeId bucketing / behavioral probing therefore **only apply to PORTED
  sims today.** (An earlier audit's claim that "no carving models state" was an
  artifact of reading the native `step(&mut self)` kernels instead of the pure
  feedback nodes — but the inverse overclaim, that *all* sim state rides a
  feedback edge, is equally wrong: only 4 of 9 do.)

- **`Field<I, O>` is a pure `coord → value` function.** Stateful grid
  simulations correctly do *not* implement it, because their value depends on
  accumulated history, not on the coordinate alone. This is correct by design,
  not a gap to be patched.

- **The real discriminating problem: typed membership is non-discriminating.**
  ~107 primitives share the signature `Field<Vec2, f32>` (`coord → f32`).
  Perlin and Worley are *type-identical* yet need *opposite* relevance. Arity
  also hides under the type: `Composite: (Image, Image, BlendMode) → Image`
  types as `Image → Image`. So any carving that leans only on the type
  signature fails to separate the largest primitive family.

- **Median steelman.** For an expert who already knows the name, plain search +
  domain tabs beats all four carvings. A carving must therefore justify being
  *more than a secondary discovery aid* — it has to earn its place against the
  null hypothesis of "just search."

## The four candidate carvings

Each is recorded with its principle, relevance mechanism, what it decomposes
well, and its own self-attack (the cracks). The cracks are not optional reading.

### C1 — Verb / intent (8-verb spine)

**Principle.** Organize by *intent* along a "where does structure come from"
spine of eight verbs:

> SUMMON → SHAPE → REFINE → RELATE → MULTIPLY → DRIVE → CONSTRAIN → DISRUPT

**Relevance mechanism.** A last-verb → next-verb transition prior (an 8×8
matrix) × `(verb × focus-type)` tags. A single primitive carries multiple
intent-tags — e.g. noise is *Summon* early in a session and *Disrupt* late.

**Cracks (self-attack).**
- DRIVE is a *binding-mode*, not a node — it doesn't sit cleanly alongside the
  other seven as a kind of primitive.
- Compound primitives (reaction-diffusion, L-systems) are *mini-sessions*
  spanning several verbs, not single points on the spine.
- The transition matrix is *asserted, not measured* — there's no usage data
  behind the priors yet.

### C2 — Field-operation basis (7, hidden IR)

**Principle.** Carve by *which structural part of a `domain → value` field* each
primitive touches:

> GENERATE, REMAP-DOMAIN, MAP-RANGE, GATHER, COMBINE, INTEGRATE, RESTRUCTURE

(existence / input-side / output-side / neighborhood / two-fields /
ordered-history / cardinality, respectively).

**Relevance mechanism.** Type-match operation slots against the field's
signature — e.g. GATHER is hidden on metric-less domains; INTEGRATE only
appears on ordered domains.

**Decomposes well.** blur = GATHER; warp = REMAP-DOMAIN; ADSR = INTEGRATE;
scatter = RESTRUCTURE; vocoder = FFT → COMBINE → IFFT.

**Cracks (self-attack).**
- This is a *compiler-IR / dedup engine*, and **wrong as a human browse
  vocabulary** — a maker thinks "blur," not "neighborhood gather."
- INTEGRATE and RESTRUCTURE are *god-buckets* that absorb too much.

### C3 — Object-kind / lens (6 + 1 meta)

**Principle.** Carve by object kind:

> Form / Field / Flow / Population / Topology / Look + Relation (meta)

**Relevance mechanism.** A primitive declares a `reads → writes` kind-signature.
A selection has an *active lens set*; relevance = primitives whose reads-kind ∩
active lens. One lens is active at a time (a multi-kind object becomes
multi-mode). This **dissolves the duplicate Scatter/Warp structurally** —
they're disambiguated by which kind they read.

**Cracks (self-attack).**
- Invisible lens-state → *mode-errors* (the user doesn't see which lens is
  active).
- Topology and Relation *degrade toward domains* rather than staying clean
  kinds.
- Flagship multi-kind objects (a rigged vector character) become the *worst
  case* under a one-lens-at-a-time model.

### C4 — Anti-taxonomy / property manifold

**Principle.** Refuse a single tree. Each primitive is a *feature vector* over
orthogonal, code-extractable axes:

> in_kind, out_kind, arity, domain_dim, locality, determinism, temporality,
> scale_of_effect, invertibility

This collapses the Perlin1D / 2D / 3D triple onto a single `domain_dim` axis,
and renders the two Scatters as *visible proximity* rather than accidental
duplicates.

**Relevance mechanism.** A **hard** typed constraint for set *membership* +
a **soft** behavioral/usage *distance* for ordering. That hard/soft split is
the one discipline that keeps it learnable. The working-set "magnet" is a
recency-weighted centroid in the manifold.

**Cracks (self-attack).**
- No *spatial muscle memory* — there's no fixed place where a primitive "lives."
- The 2D projection is *lossy* — do not sell the visual layout as the model;
  the model is the vector, not the picture.
- *Cold-start* buries the rare-but-powerful primitive (e.g. erosion) under the
  common ones.
- Feature extraction was *claimed* to fail for stateful sims — per the
  verified facts above, the feedback-edge axis (recurrent vs. pure) rescues
  that case **only for the 4 ported sims**; the 5 native-only sims still have
  no graph-level state node to extract from, so this crack is weaker than
  originally stated **but not closed**.

## Adversarial findings

Recorded faithfully from the attack pass:

- **Type-degeneracy, cited.** Perlin (`unshape-noise/src/lib.rs:200`) and Worley
  (`unshape-noise/src/lib.rs:751`) share *identical* `Field<Vec2, f32>`
  signatures; ~107 such impls exist. Typed membership is non-discriminating for
  the largest primitive family.
- **Hidden arity, cited.** `Composite: (Image, Image, BlendMode) → Image`
  (`unshape-image/src/kernel.rs:320`) types as `Image → Image` — a *binary* op
  surfaces as *unary*.
- **Mental-model cost ranking** of the four (most costly first):
  **C2 > C3 > C1 > C4.** C2 (the field-operation IR) is the most costly to hold
  in a human head.
- **Median steelman verdict.** Plain search + domain-tabs beats all four for
  name-knowing experts. The clever carvings should be **secondary discovery
  aids, not the primary surface.**

## Status

**OPEN. None selected.** A fresh decorrelated design exploration is underway.
The four carvings above are *inputs* to that exploration — generated candidate
material with their own attacks attached — **not the answer.**

## Candidate synthesis (unblessed): organize by job, in layers

> **UNBLESSED CANDIDATE. Not a decision, not a recommendation, not settled.**
> This is the *output* of a decorrelated design exploration + adversarial
> judging — a candidate recorded **under** the `STATUS: OPEN` above, **not** a
> resolution of it. The four carvings, their cracks, and the open framing all
> remain live.

This emerged from five decorrelated candidate designs (a learned multi-signal
ranker; intent/verb transitions; an enriched machine-extractable signature;
selection×lens affordance; example/output retrieval) put through three
adversarial judges (determinism, expert-recall, coverage/honesty). The
candidates disagreed on the surface but **converged at the root**: the organizer
is not one mechanism — it splits into layers by **job**, and most candidate
failures came from making one mechanism do two jobs. Recorded as a candidate;
not blessed.

### The three layers

1. **Membership = hard, machine-extracted typed/structural filter.** OpType
   match + arity + domain-dimensionality + the pure-vs-recurrent axis (which
   discriminates only the **4 ported** sims today — native-only sims carry no
   graph-level state node) + facet. Deterministic by construction; CI-enforceable
   (machine-extracted axes
   can be asserted at build time; subjective tags cannot). This is the legality
   layer — "a sharper domain tab." Position within it is stable. It is the only
   layer that organizes the structural/behavioral third of the catalogue by real
   signal.
2. **Recall surface = canonical list sorted by query alone.** The decisive rule
   from the expert-recall judge: context/frecency/usage signals may only **add**
   to a separate, visually distinct "suggested" zone — they must **never**
   reorder the canonical list. The moment a soft signal reorders the memorized
   list, expert muscle-memory recall is forfeit. This is the VS Code palette
   pattern already cited in `editor-interaction.md` (fuzzy + light recency as a
   top stratum over otherwise-stable order). For recall, the plain-search
   baseline **is** the right answer — preserve it, don't try to beat it.
3. **Discovery = additive, spatially-separate channels** that never touch the
   canonical order: (a) soft context/frecency/corpus-transition suggestions in a
   distinct "suggested" strip; (b) output-demonstration retrieval
   (point/sketch/example) — found to be the **only** mechanism that beats plain
   search at no-name discovery. Every other candidate's "discovery win" reduced
   to "better tags or a better filter" — so harvest the tags/filter, don't build
   a ranker around them.

### The reproducibility crux (resolved)

Per-user personalization is **not** a determinism violation: ranking/relevance
is UI/view state, not project state. Fence it: ranker code lives in a crate that
node/op crates must not depend on, so a relevance score can never become a node
input. Membership is the only reproducible-by-contract layer. Per-user frecency
is legitimate exactly like window layout.

### Determinism fences (left unstated by the candidates; required if adopted)

- (a) The crate firewall above (ranker not a dependency of node/op crates).
- (b) Build-time fingerprints/embeddings pinned as data, never recomputed at
  runtime (FFT spectra are platform-dependent — fine if frozen, fatal if live).
- (c) cooc/frecency accumulators need defined undo/branch semantics or event-log
  replay diverges; plus stable total-order tiebreaks (Rust `HashMap` iteration
  is randomized — a real leak the candidates did not address).

### Two gaps NONE of the five candidates solved (the open frontier)

1. **The recurrent/stateful third needs its own organizing story.** "Recurrent
   vs pure" can **bin** the ported steppers (rd / fluid / particle / vocoder),
   but only those — the 5 native-only sims (automata / physics / spring /
   space-colonization / procgen) have no graph-level state node to bin on. Even
   for ported sims it gives no intra-bin ordering, and the fingerprint/preview
   mechanisms structurally cannot sample the native ones (they are not pure
   Fields; e.g. native `step()` is `&mut self`, IK `solve` takes a `&mut Pose`).
   The Init+Step recurrence pair needs a first-class organizing treatment the
   morphism-centric carvings do not provide.
2. **Config-enum variants: a primitive is often a family, not a point.**
   `Worley2D` is 9 behaviors (3 distance functions × 3 return types — verified in
   source), visually ranging from cells to cracks, yet all five candidates
   collapse it to one node. The enriched-signature candidate is worst here (it
   fingerprints only the default config). The organizer must surface family
   members, not just the family.

STATUS of this synthesis: candidate, unblessed. A further decorrelated pass on
the two open gaps is underway.

## Refinement after second adversarial round (unblessed)

> **UNBLESSED.** This subsection refines — and in places **corrects** — the
> candidate synthesis above. It is the result of three decorrelated gap-designs
> (probe / structure / template) put through three adversarial judges
> (determinism, curation, projectional-coherence). Nothing here is blessed; the
> `STATUS: OPEN` at the top of the doc still governs.

1. **PRECONDITION (blocking).** Per *finish migrations before building on top*:
   the recurrent-axis carving — state-TypeId bucketing (read off
   `Step.outputs()` `ValueType::Custom`) and behavioral probing — works only for
   the **4 ported** sims. The **5 native-only** sims (automata, physics, spring,
   space-colonization, procgen) have no graph-level state node and nothing pure
   to sample. Before the carving can claim to cover sims, those 5 must be
   **ported to feedback nodes OR explicitly fenced as legacy native-step
   kernels.** Note: `ValueType::Custom` carries a **non-serializable `TypeId`**,
   so the *persisted* bucket key must be the **stable type-name string**, not the
   TypeId.

2. **The "two gaps are one gap" collapse is REJECTED.** A config-enum family
   (e.g. Worley's 9 = `DistanceFunction` × `WorleyReturn`) is a pure selection
   over a **stateless morphism** — no tick/seed/seek. A sim is a **recurrence**
   needing `connect_recurrence` / `run_to_tick` / seek / `Init`. They share no
   machinery; merging them papers over an asymmetry that resurfaces at preview
   (a Worley variant previews by pure re-eval; a sim only by `run_to_tick` from a
   seed) and at the feedback wire. They stay **DISTINCT.** Furthermore: a
   config-enum family is the **enumerable case of the already-committed
   first-class variant-set** (compare-variants / vary-per-X in
   `editor-interaction.md`) — it should **collapse into that existing
   primitive** (the enum *is* the variant axis), not spawn a parallel
   "config-family" notion. (*collapse-asymmetries-to-primitives.*)

3. **Templates-as-unit via INLINING is REJECTED.** Inlining a recipe into the
   project graph as anonymous nodes either destroys the abstraction (the
   node-editor abstraction-blindness we reject) or forces reconstructing identity
   from provenance tags — which is the forbidden *string-matching-when-structure-
   exists* anti-pattern. The coherence-preserving form of the legitimate need (a
   reusable, nameable, abstraction-preserving sim/recipe unit): a template
   instantiates as a **collapsed group node that STAYS in the project graph.**
   Identity is structural (the group node), show/hide-body is a projection, holes
   are real input ports (so painted-density / live-signal inputs enter as
   faithful wires rather than escaping a frozen subgraph), it unifies with
   first-class variants instead of duplicating them, and the sim/morphism
   asymmetry stays visible (a recurrent group carries its `connect_recurrence` +
   seed inside the boundary; a Worley-family node does not).

4. **Layer assignment after the round.**
   - **Membership / recall backbone = STRUCTURE (machine-derived):**
     state-type-name buckets for ported sims; family variant axes read off the
     existing serde/schemars enum schema. `worley` → **ONE entry + enum
     dropdowns** (NOT 9 rows). Lowest marginal cost per new op; no
     silent-omission failure; rot-proof because derived from types the compiler
     keeps consistent.
   - **Discovery soft signal = PROBE (behavioral fingerprint),** build-time-
     pinned, and STRICTLY: never the canonical list, never an op input.
     Determinism conditions it must meet: (a) no op-input path; (b) hermetic
     build OR tolerance-banded fuzzy fingerprints (raw settle-time /
     oscillation-count / DCT reductions flip ±1 at cross-platform float
     boundaries and reintroduce the FFT platform-dependence already fenced —
     "pin as build-time data" only relocates this to non-hermetic builds); (c) no
     runtime / author-time recomputation; (d) it structurally cannot sample
     native-only sims, so it covers only ported ones.
   - **Curated layer discipline:** the human layer may only **ADD** synonyms /
     labels / demotions to machine-derived entries — it may **never BE** an
     entry. Build must **FAIL** if a hand-written synonym/template references an
     op or enum variant that no longer resolves. (Outcome-naming alone is
     strictly worse than tagging for recall — one primary string must coincide
     with the query — so names are a display+synonym overlay on the machine
     index, not the recall key.)

STATUS: still OPEN. Precondition (port-or-fence the 5 native-only sims) is the
gating next step before sim coverage is real.
