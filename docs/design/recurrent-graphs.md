# Recurrent Graphs

Graphs with cycles (feedback loops). Not special to audio - a general computation pattern.

## The Latch (implemented — sole recurrence primitive)

> **STATUS:** IMPLEMENTED. The Latch is the sole memory primitive; the
> feedback-edge model (`feedback: bool` + `connect_recurrence`) described below
> has been REMOVED. All 9 recurrent sims (rd, particle, fluid, audio/vocoder,
> automata, spring, physics, space-colonization, procgen) drive recurrence
> through an explicit `Latch` node via `tick_latched` / `run_to_tick_latched` /
> `seek_latched` over a `LatchSnapshot`. The one residue still genuinely open is
> *incommensurate* multi-rate (see below); integer sub-rates are solved via
> `Rate::Every(n)` (hold/decimate).
>
> Code: `Latch` / `Rate` in `unshape_core::nodes::latch`; the driver
> (`tick_latched` etc.) and `validate_latches` in `unshape_core::graph`;
> `LatchSnapshot` in `unshape_core::eval`; serde registration as `core::Latch`
> in `unshape_serde::feedback_nodes`.

### The primitive

A **Latch** node: a seeded unit-delay (1-tick memory). It is definitionally the same primitive as Lustre `fby`, Pd `[z~]`, Faust `~`, a hardware register — "the visible delay element on a cycle" has one shape.

Ports:
- input 0 `init` — **REQUIRED WIRED** seed (not a default field; this is load-bearing, see below)
- input 1 `signal` — value captured for next tick
- output 0 `out` — the stored value: `init` at tick 0, previously-captured `signal` thereafter

`rate` is a per-node param (default: per-tick).

### Why it beats the alternatives (decorrelated verdict)

- **vs `feedback: bool` (current):** identical semantics, but the delay becomes a VISIBLE node instead of an invisible wire flag; cycle-validation becomes structural ("every cycle must contain a latch"); and it REMOVES machinery (`connect_recurrence`, `Wire.feedback`, the two-edge seed/evolve resolution in `tick`).
- **vs a Read/Write memory-cell (shared `cell_id`):** rejected — the recurrence dependency would be a NAME match, not a wire = the "no string-matching when structure exists" anti-pattern; loses topo-validatable cycles and wire-level state dup/fork.
- **vs pure step-function / boundary-I/O** (`(state,input)→(state',output)`, iterate externally): rejected — either the loop lives OUTSIDE the graph (breaks graph-as-source-of-truth for the projectional editor) or it collapses back into needing an in-graph seeded delay = the Latch.
- **vs bare Lustre `pre` (un-seeded):** already excluded — un-seeded recurrence is an error here because opaque sim state (fluid/RD grids) has no zero value; the seed must be a wired source. So the surviving form is `fby` = the Latch with seed-as-port-0.

### The load-bearing detail the audit caught

`init` MUST be a required wired input PORT, not a default value field on the node. If it were a field defaulting to zero/`Default`, it would silently reintroduce the implicit-zero-seed fallback that the recurrence work deliberately removed (opaque grids have no zero value; a missing seed must be an error). With init-as-a-port, the latch is "the existing two-edge recurrence with a visible face" — and the swappable in-graph seed property (Init/checkpoint/baked-state) is preserved as port 0.

### The model

- Latch `out` is a WITHIN-TICK SOURCE (emits stored value, no within-tick dependency on its input); latch `signal` input is a WITHIN-TICK SINK (captured at tick end). So the within-tick graph is a pure DAG — no cycles, no edge flags, no cut.
- **Loop legality:** a cycle is legal IFF it contains a latch. Validation = run the Kahn topo pass with edges-into-`latch.signal` excluded; any leftover (a latch-free cycle) is an instantaneous loop → error. O(V+E), same cost as today, but stricter/more honest than the current per-edge `connect_recurrence` opt-out.
- **State = the set of latches.** The snapshot is `{latch_node_id → Value}` (replacing `FeedbackState`'s `(from_node, from_port)` key). Seek (`Resimulate`/`Discontinuity`/`Error`) is unchanged in meaning — `Resimulate` clears the snapshot and replays, re-seeding each latch from its `init` wire (works for opaque grids).
- "Recurrence" stops being a subsystem: there is a Latch (memory) node and one validation rule; a sim is a Step op with a latch in its loop, structurally identical to any other graph.

### Implementation (done)

- **Scheduler:** the Latch is a driver-recognized node (downcast like `GraphInput`); `tick_latched` produces its `out` from the snapshot (or cold-seeds from `init`), and edges whose destination is a `latch.signal` port are excluded from the within-tick DAG (`is_latch_signal_sink`). The within-tick graph is therefore a pure DAG.
- **Serialization:** new graphs carry no feedback wire flag; latches serialize as ordinary `core::Latch` nodes with params `{ty, rate}` (the opaque `ty` stored by name, resolved at load against the enabled feedback features). The runtime stored value is NOT serialized (graph = program; reproduce by replay from `init`); opaque snapshots are replay-only (TypeId can't serde). Old graphs whose wires carry `feedback: true` are rejected with a migration hint (`SerdeError::LegacyFeedbackWire`), not silently loaded. (The `SerialWire.feedback` field is retained as an `Option<bool>` solely for this legacy detection and to keep bincode's positional layout stable.)
- **Migration:** all 9 sims were the identical `connect(Init→Step.state)` + `connect_recurrence(Step→Step.state)` self-loop → `Init→Latch.init`, `Latch.out→Step.state`, `Step.out→Latch.signal`. Mechanical; the `*Step` execute bodies and `*Init` nodes are unchanged. The vocoder additionally has non-state I/O ports (carrier/modulator in, audio out), wired conventionally; the latch is only on the state port.
- **Deletions (done):** `Wire.feedback` + `Wire::feedback` ctor, `Graph::connect_recurrence`, `Graph::has_feedback`, the feedback/!feedback edge filters, the old `FeedbackState`-based `tick`/`run_to_tick`/`seek` drivers, `FeedbackState`, `EvalContext`'s feedback-state slot, and `ValueType::zero_value` (its only caller was tick-0 feedback seeding).

### Multi-rate (re-scoped: integer sub-rates solved; incommensurate deferred — mostly resolved)

- There is ONE base tick axis = the FASTEST rate present in the graph. Everything else is a sub-rate of it. There is no rate "above" base (base is the ceiling), so there is NO nested sub-ticking — which dissolves the original worry ("does a frame node sub-tick a sample latch N times inside it?"). One tick axis.
- A below-base-rate loop = a latch with `rate = Every(n)`: it captures/advances once every n base-ticks and HOLDS its output between advances (zero-order hold). The loop's nodes re-evaluate only on the ticks its latch advances; otherwise their outputs hold. `Every(1)` is today's base-rate case (the only one currently exercised).
- Cross-rate reads are DEFINED by default: a slow→fast read (fast consumer reads a slow latch) returns the held value (ZOH); a fast→slow read (slow consumer reads a fast signal) POINT-SAMPLES — it reads the current value at the consumer's own tick (decimation). So "which of the N samples does the consumer see?" = the one at its own tick. Anti-aliasing on decimation, if wanted, is an explicit filter node — not latch semantics.
- => Integer sub-rates are SOLVED by hold + decimate. No ambiguity, no sub-ticking.
- The genuine RESIDUE — INCOMMENSURATE rates (non-integer ratio, e.g. 44.1kHz alongside 48kHz): `Every(n)` can only express integer sub-divisions of the base, so a non-integer ratio cannot be a sub-rate of one base clock. These cross an explicit RESAMPLER node at a clock-domain boundary, which implies MULTI-CLOCK-DOMAIN scheduling (two independent tick axes mediated by the resampler). That is a separate, larger architectural question and is DEFERRED — but its SHAPE is known (resampler node + multi-domain scheduling), so it's a bounded future item, not a void. It arises mainly at external device/file boundaries; internal procedural work picks one rate, so it is rare. The resampler's own interpolation/filter state is just more latches.
- Smaller residues (real, deferrable): PHASE — `tick % n == 0` is relative to the absolute tick origin, so a latch added mid-run needs a per-latch phase offset; ANTI-ALIASING on decimation — an optional explicit filter, not a latch concern.

Net status of multi-rate: integer sub-rates RESOLVED (hold/decimate, one base tick); incommensurate rates DEFERRED with known shape (resampler + multi-domain scheduling); phase + anti-aliasing are small deferrable residues. This is no longer an open void — it's a bounded, shaped future item.

---

*(The feedback-edge model described below has been **REMOVED**. It is retained
here only as historical context for the Latch design above, which superseded and
replaced it. `feedback: bool`, `connect_recurrence`, `FeedbackState`, and the
seed/evolve two-edge resolution no longer exist in the codebase.)*

## The Problem

DAGs (directed acyclic graphs) can't express feedback:

```
       ┌──────────────┐
       │              ↓
In ──-> Add ──-> Delay ──-> Out
       ↑              │
       └──────────────┘  ← cycle!
```

But feedback is fundamental to many domains:
- Audio: reverb, comb filters, physical modeling
- Physics: iterative solvers, constraint resolution
- Simulation: cellular automata, reaction-diffusion
- Control systems: PID controllers
- ML: RNNs, LSTMs

## Prior Art

### Pure Data / Max/MSP

Cycles are allowed with explicit single-sample delay:

```
[osc~ 440]
    |
[*~ 0.5]──────┐
    |         │
[+~ ]←────────┘  ← feedback adds one sample delay
    |
[dac~]
```

Rule: every cycle must contain at least one delay element. This ensures evaluation order is well-defined.

### Dataflow languages

- Signal processing: feedback = z⁻¹ (unit delay)
- Lustre/Lucid: `pre` operator for previous value
- Faust: `~` operator for feedback with implicit delay

### Iteration vs Streaming

Two models for cycles:

**Streaming**: Each "tick" produces one output, feedback arrives next tick
```
tick 0: out = f(in, 0)           // no feedback yet
tick 1: out = f(in, out[0])      // feedback from tick 0
tick 2: out = f(in, out[1])      // feedback from tick 1
```

**Iteration**: Run until convergence
```
x = initial
while not converged:
    x = f(x)
return x
```

Physics solvers often use iteration. Audio uses streaming.

## Semantics

### Feedback = Delayed Wire

Every back-edge (wire that creates a cycle) has implicit or explicit delay:

```rust
enum Wire {
    /// Normal wire - value available immediately
    Direct { from: NodeId, to: NodeId },

    /// Feedback wire - value from previous iteration
    Feedback { from: NodeId, to: NodeId, delay: Delay },
}

enum Delay {
    OneSample,           // audio: z⁻¹
    OneFrame,            // animation: previous frame
    OneTick,             // simulation: previous step
    Explicit(Duration),  // explicit time delay
}
```

### Evaluation Order

With delays on back-edges, the graph becomes a DAG per-iteration:

```
Iteration N:
  - Read feedback values from iteration N-1
  - Evaluate nodes in topological order
  - Write new feedback values for iteration N+1
```

This is deterministic and reproducible.

### Recurrence: seed input + evolve output

A **recurrence** is a first-class, declared relationship, not an emergent
"feedback wire that happens to win a precedence tiebreak". Its definition:

> A recurrent state input is **SEEDED** by its incoming (non-feedback) edge at
> tick 0, and **EVOLVED** by a looped output thereafter.

It is declared with one call, `graph.connect_recurrence(output, state_input)`,
which marks `state_input` as recurrent and names the `output` that evolves it.
The recurrence is made of **two edges into the same input**:

- a **seed** — an ordinary `connect` (direct) edge from an in-graph source node
  (the `Init` node): produces the initial state from config, pure and
  deterministic, no inputs. This is the tick-0 value.
- an **evolve** — the `connect_recurrence` (back-)edge from the step's own
  output: carries the previous-tick output back into the state input on ticks
  ≥ 1.

The seed is an **ordinary, swappable input**. `dup`-ing the step forks the state
(each fork can share or re-point its seed); pinning swaps the `Init` seed source
for a baked-state source. Nothing about the recurrence is special-cased on the
wire — the seed is just an input edge, the evolve is just a marked input edge.

At evaluation of tick N, a recurrent input resolves as:

1. **Evolve (carried value)** — if `FeedbackState` holds the evolving source's
   previous-tick output (ticks ≥ 1), use it.
2. **Seed (direct edge)** — otherwise (tick 0) evaluate the direct seed edge.

There is **no zero-value fallback**. A recurrence declared *without* a seed edge
is a declaration error, reported by `tick`, uniformly for scalar and opaque
state alike. Seeding is uniform: every recurrence — an `f32` accumulator loop and
an opaque reaction-diffusion grid — is seeded by an explicit source node. (This
removed an earlier inconsistency where scalar loops silently fell back to a
`zero_value` while opaque sims errored.)

Because the seed is an in-graph node, `run_to_tick` / `seek(Resimulate)` work for
*opaque* (`Custom`) state types — reaction-diffusion grids, particle systems,
fluid grids, vocoder state — which have no `zero_value`. `run_to_tick` clears
`FeedbackState` and replays from tick 0; tick 0 re-seeds via the seed node, so no
manual pre-seed is needed and the sim is fully rewindable.

Per-domain `Init` seed-source nodes (behind each crate's `feedback` feature):

All 9 genuinely-recurrent sims are ported (`unshape-rig`/IK is **not** recurrent —
it solves a pose in place, no feedback edge — so the migration is complete at 9/9).

| Crate                        | Init node                                  | Produces                                  |
|------------------------------|--------------------------------------------|-------------------------------------------|
| `unshape-rd`                 | `GrayScottInit`                            | seeded `ReactionDiffusion` grid           |
| `unshape-particle`           | `ParticleInit`                             | emitted `ParticleSystem` (seeded RNG)     |
| `unshape-fluid`              | `FluidInit` (+ smoke/SPH/3D variants)      | sourced `FluidGrid2D`                      |
| `unshape-audio`              | `VocoderInit`                              | zeroed (defined) `VocoderState`           |
| `unshape-automata`           | `ElementaryInit` / `LifeInit` / `SmoothLifeInit` | seeded CA grid (1D / 2D / continuous) |
| `unshape-spring`             | `SpringInit`                               | built soft body (`SpringSystem`)          |
| `unshape-physics`            | `PhysicsInit`                              | built `PhysicsWorld` (bodies + constraints) |
| `unshape-space-colonization` | `GrowInit`                                 | seeded `SpaceColonization` (attractors + roots) |
| `unshape-procgen`            | `WfcInit`                                  | seeded `WfcSolver<TileSet>` (RNG on edge) |

Options considered and rejected:

- **Zero / default** — opaque state has no zero value, and a silent default makes
  the recurrence implicit rather than declared.
- **Explicit initial value stored on the wire** — a `Value` is not `Copy` (so
  `Wire` would lose `Copy`), and a baked initial value is not generative.

A seed *node* is parameterized, serializable, inspectable, swappable, and
composable — consistent with the generative-mindset and operations-as-values
principles.

### Policy-free core

The recurrence primitive bakes in **no** checkpoint / record / preview policy.
The zero-cost base is plain recompute: `run_to_tick` replays from tick 0, which
is the dependency-free default. Checkpointing and recording are not implemented
here and are not forced on consumers — the seam stays bare so callers compose
their own policy (and unused machinery tree-shakes away).

## State Model

Feedback wires ARE the state. No hidden state in nodes.

```rust
struct GraphState {
    /// Values on feedback wires, keyed by wire ID
    feedback_values: HashMap<WireId, Value>,
}

fn evaluate(graph: &Graph, inputs: &Inputs, state: &mut GraphState) -> Outputs {
    // 1. Collect feedback values from state
    let feedback = collect_feedback(state);

    // 2. Evaluate DAG (treating feedback as inputs)
    let outputs = evaluate_dag(graph, inputs, &feedback);

    // 3. Update state with new feedback values
    update_feedback(state, &outputs);

    outputs
}
```

**Benefits:**
- State is explicit and inspectable
- Easy to serialize/restore (save game, undo)
- Nodes are stateless (pure functions)
- Clear what depends on history

## Per-Domain Applications

### Audio

Classic feedback patterns:

```
// Comb filter (creates resonance)
In ──-> [+] ──-> [Delay N samples] ──-> Out
        ↑                        │
        └──── [* feedback] ←─────┘

// Karplus-Strong (plucked string)
Noise burst ──-> [+] ──-> [Delay] ──-> [LowPass] ──-> Out
                ↑                              │
                └──────────────────────────────┘
```

Delay = sample count. Feedback coefficient < 1 for stability.

### Physics / Simulation

Iterative constraint solving:

```
// Verlet integration
positions ──-> [Apply forces] ──-> [Integrate] ──-> [Solve constraints] ──-> new positions
    ↑                                                                        │
    └────────────────────────────────────────────────────────────────────────┘
```

Each frame: read previous positions, compute new positions.

### Procedural Animation

Secondary motion (jiggle, cloth):

```
// Simple spring simulation
target ──-> [Spring force] ──-> [Integrate] ──-> position
                ↑                               │
                └───────────────────────────────┘
```

### Reaction-Diffusion (Textures)

```
concentration ──-> [Diffuse] ──-> [React] ──-> new concentration
      ↑                                          │
      └──────────────────────────────────────────┘
```

Run for N iterations to generate pattern.

## Graph Analysis

Need to detect and handle cycles:

```rust
impl Graph {
    /// Find all strongly connected components (cycles)
    fn find_cycles(&self) -> Vec<Vec<NodeId>>;

    /// Check if graph has any cycles
    fn is_dag(&self) -> bool;

    /// Get wires that would need to be feedback wires
    fn find_back_wires(&self) -> Vec<WireId>;

    /// Validate: every cycle has at least one delay
    fn validate_feedback(&self) -> Result<(), CycleWithoutDelay>;
}
```

## Implications

### For Time Models

Recurrence IS statefulness. A recurrent graph:
- Cannot seek (without replaying from start, or caching)
- Must evaluate in order
- Has implicit state (feedback wire values)

But it's still deterministic - same inputs + same initial state = same outputs.

### For Caching

DAG portions can still be cached. Only feedback wires carry state between iterations.

```
[Noise] ──-> [Process] ──-> [+] ──-> Out
     cacheable           ↑  │
                         └──┘ stateful
```

### For Parallelization

Within one iteration, the DAG can be parallelized. Across iterations, must be sequential.

### For Serialization

Graph structure + feedback wire values = complete state.

```rust
#[derive(Serialize, Deserialize)]
struct GraphSnapshot {
    graph: Graph,
    feedback_state: HashMap<WireId, Value>,
}
```

## Open Questions

1. **Delay granularity**: One sample? One frame? Configurable per-wire?

2. **Stability**: Feedback coefficient > 1 = explosion. Detect/warn?

3. **Warm-up**: How many iterations before "stable"? Domain-dependent.

4. **Mixed rates**: What if audio (48kHz) feeds back into control rate (60Hz)?

5. **Nested iteration**: Iterative solver inside streaming audio?

## Summary

| Concept | DAG | Recurrent |
|---------|-----|-----------|
| Cycles | Not allowed | Allowed with delay |
| State | None | Feedback wire values |
| Seekable | Yes | No (without replay) |
| Deterministic | Yes | Yes |
| Parallelizable | Fully | Per-iteration |

Recurrent graphs unify:
- Audio feedback (delay lines, filters)
- Physics simulation (iterative solvers)
- Procedural animation (springs, jiggle)
- Generative textures (reaction-diffusion)

Not "audio is special" - feedback is a general pattern.
