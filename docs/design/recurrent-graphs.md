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

### Initial Values

What's the feedback value on first iteration?

Options:
1. Zero / default
2. Explicit initial value on wire
3. "Undefined" - let it warm up

```rust
struct FeedbackWire {
    from: NodeId,
    to: NodeId,
    initial: Value,  // value for iteration 0
}
```

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
