# Recurrent Graphs

Graphs with cycles (feedback loops). Not special to audio - a general computation pattern.

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

| Crate              | Init node       | Produces                          |
|--------------------|-----------------|-----------------------------------|
| `unshape-rd`       | `GrayScottInit` | seeded `ReactionDiffusion` grid   |
| `unshape-particle` | `ParticleInit`  | emitted `ParticleSystem` (seeded RNG) |
| `unshape-fluid`    | `FluidInit`     | sourced `FluidGrid2D`             |
| `unshape-audio`    | `VocoderInit`   | zeroed (defined) `VocoderState`   |

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
