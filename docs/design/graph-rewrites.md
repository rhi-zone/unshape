# Graph Rewrites (design direction — endorsed this session)

> **STATUS: DIRECTION endorsed this session; substrate is entirely net-new.**
>
> This records a design thread developed and **endorsed by the user this
> session**. The agreed *direction* is stated as such; genuinely-open parts are
> marked **OPEN**. The substrate this would build on is **entirely net-new** —
> see "Substrate today (verified)"; nothing here is an accretion on existing
> infrastructure. The verified substrate facts were source-checked this session
> and are stated as established.

## Motivation (the throughline)

The founding complaint: a conventional node editor is a **copout for
grokkability**. A *rewrite* is more grokkable than static wiring because it
shows the **transformation** (before → after), not just the end state. And a
rewrite is itself a **value** — serializable, inspectable, replayable,
nameable — consistent with ops-as-values.

## Three things kept distinct (do not conflate)

Source-verified this session:

- **An op** (`DynOp`, `unshape-op/src/lib.rs` — `apply_dyn(input: OpValue) ->
  OpValue`) is a *value→value* transform performed *inside* a node. It is **NOT**
  a graph rewrite.
- **A graph edit** (`GraphEvent::{AddNode, RemoveNode, Connect, Disconnect,
  UpdateParams, Batch}`, `unshape-history/src/event.rs`) is a primitive
  *structural* mutation, recorded for undo. A primitive edit, **not** a
  rule-based rewrite.
- **A graph rewrite** = match a subgraph *pattern* → replace with a *template*.
  Rule-based, structure→structure. This does **NOT** exist in the codebase yet.

(Earlier in the session these were conflated — "an op is a rewrite" — which is
wrong; corrected here.)

## The mechanism (agreed direction)

- A rewrite rule is `LHS pattern (with holes/variables) → RHS replacement`.
- **The matcher is a state machine.** Regex corresponds to a finite automaton;
  the graph analog is a **tree/graph automaton**. Compile the WHOLE rule set
  into one automaton and do a single bottom-up pass that labels each node with
  the rules it matches — so all rules match in **ONE pass**, not rule-by-rule.
  (This is the bottom-up-rewrite-system / instruction-selection approach,
  BURS/iburg.)
- **Trivial in the tree/expression projection.** The dew/wick `Expr` AST is a
  tree, so this is plain term rewriting + a tree automaton (e.g. `add(x,0) → x`,
  `mul(x,1) → x`) — maximally grokkable, no exotic machinery.
- **DAG projection is harder.** Nodes have named ports and fan-out, so a capture
  variable matches a subgraph-region-with-ports, and replacement must re-glue the
  boundary (the formal name is graph rewriting; DPO/SPO defines the gluing).
  **OPEN.**

## Application strategy (a SEPARATE dial)

The matcher and the *strategy for applying* matches are distinct dials.

- **Objective: the simplest final graph.**
- **Ideal = global minimum**: equality saturation / e-graphs — apply rewrites
  non-destructively into a structure holding *all* equivalent forms, then
  **EXTRACT** the minimum-cost form. Confluence-free by construction;
  deterministic because the result is defined by (rule set + cost function +
  tie-break), **NOT** by application order.
- **Practical approximation (user hedged "may")**: greedy — "rewrites that
  reduce node count the most take precedence." Local; can stick in a local
  minimum (a rewrite that temporarily *grows* the graph to enable a larger later
  collapse is exactly what greedy refuses and saturation allows).
- **Cost function:** node/DAG count is the natural first choice.
- **Honest cost:** e-graphs can blow up in memory; optimal extraction is
  NP-hard under some cost functions → heuristic/ILP in practice. So "simplest
  final graph" is the right *definition*; reaching the true global min vs a good
  approximation is an engineering dial. **OPEN.**

## Recipes & abstraction boundaries = rewrite-rule sides (projection-neutral)

This collapses the earlier "is a recipe a boundary node or a rewrite rule?"
fork — they are the **same thing**:

- A **recipe is a rewrite rule** `R : compact-form(holes) ↔ expanded-form`.
- **Collapse** = apply `R` backward; **expand/open** = apply `R` forward.
- The **abstraction boundary is not a node type** — it is *which side of `R` the
  current projection is showing*. "Show/hide body" = view the region in
  collapsed (LHS) vs expanded (RHS) form.
- Renders **co-equally per projection** (none privileged): node view → a
  collapsed node; stack/list → a collapsed section / macro step;
  literate/document → a named transclusion (`@spring-follower`); formula → a
  named function call `springFollower(target, stiffness)`; timeline → a grouped
  track. (Do **NOT** bless the node projection as canonical.)
- **Re-collapse is structural, not tag-based.** Because the matcher is a state
  machine, the recipe's pattern can be re-recognized at any time → re-collapse is
  free and identity is the *rule* (structural), with **ZERO provenance tags**.
  This resolves the coherence-judge objection to template-inlining (Frame 3 in
  primitive-carving.md) without the forbidden
  string-matching-when-structure-exists anti-pattern. If the user edits the
  expansion so it no longer matches `R`, it correctly will **NOT** re-collapse —
  it genuinely isn't that recipe anymore. (Honest behavior, falls straight out
  of the automaton.)
- This **supersedes/refines the "collapsed group node" framing** recorded in
  primitive-carving.md, which was only the *node-projection rendering* of this
  projection-neutral boundary.

## Substrate today (verified — all net-new)

Source-checked this session: the graph is **FLAT** (`Graph { nodes: HashMap,
wires: Vec }`, `unshape-core/src/graph.rs`); **NO** nesting/subgraph mechanism;
**NO** group/boundary node type (only `ConstantNode` / `GraphInput` /
`GraphOutput`); **NO** pattern-matching; **NO** rewrite engine (only an
imperative `Optimizer<G>::apply()` trait, `unshape-core/src/optimize.rs`, no
rules); `GraphEvent` history exists but is manually constructed (not
auto-recorded); `SerialGraph` is flat (`unshape-serde/src/serial.rs`); **NO**
projection/view concept.

So the rewrite engine, the automaton matcher, the abstraction-boundary
primitive, and projections are **ALL net-new** — not accretions on existing
infrastructure.

## Open questions

- **DAG boundary-gluing (DPO/SPO):** which surrounding wires reconnect to the
  replacement's ports.
- **Greedy node-count vs e-graph saturation+extraction:** fidelity vs
  memory/NP-hardness.
- **Parameterized recipes:** how holes/variables bind in the automaton.
- **Expressiveness limit:** a finite/tree automaton recognizes only the
  tree-regular fragment; fan-out/sharing/context-sensitive patterns exceed a pure
  FSM and need more.
- **Determinism:** integrate the "result = rules + cost + tie-break" story with
  the existing determinism guard.
- **Relation to ops-as-values pipeline and the `GraphEvent` log:** are recorded
  edits a degenerate rewrite log?
