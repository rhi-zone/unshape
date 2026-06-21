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

- **Statefulness is NOT an unsolved problem for carving.**
  `DynNode::execute(&self, ...)` is pure. Simulation state rides a *feedback
  edge* (`Wire { feedback: bool }`): an `*Init` source node seeds the state,
  and a pure `*Step` node evolves it by clone-and-advance. So there is already
  a clean *typed* axis available to any carving: **pure morphism vs. recurrent
  (Init + Step pair).** (An earlier audit's claim that "no carving models
  state" was an artifact of reading the native `step(&mut self)` kernels
  instead of the pure feedback nodes — it does not hold.)

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
- Feature extraction was *claimed* to fail for stateful sims — but per the
  verified facts above, the feedback-edge axis (recurrent vs. pure) actually
  rescues that case, so **this particular crack is weaker than originally
  stated.**

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
   match + arity + domain-dimensionality + the verified pure-vs-recurrent axis +
   facet. Deterministic by construction; CI-enforceable (machine-extracted axes
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
   vs pure" correctly **bins** GrayScott / fluid / cellular-automata / particle
   steppers / IK, but gives no intra-bin ordering, and the fingerprint/preview
   mechanisms structurally cannot sample them (they are not pure Fields; e.g. rd
   `Step` is `&mut self step()`, IK `SolveCcd`/`SolveFabrik` take
   `(&Skeleton, &mut Pose)`). The Init+Step recurrence pair needs a first-class
   organizing treatment the morphism-centric carvings do not provide.
2. **Config-enum variants: a primitive is often a family, not a point.**
   `Worley2D` is 9 behaviors (3 distance functions × 3 return types — verified in
   source), visually ranging from cells to cracks, yet all five candidates
   collapse it to one node. The enriched-signature candidate is worst here (it
   fingerprints only the default config). The organizer must surface family
   members, not just the family.

STATUS of this synthesis: candidate, unblessed. A further decorrelated pass on
the two open gaps is underway.
