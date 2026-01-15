# Graph Pattern Matching for Effect Optimization

This document describes the design for recognizing and optimizing subgraph patterns in audio (and potentially other) graphs.

## Problem Statement

Users build graphs from primitives. We want to:
1. Serialize as primitives (portable, flexible)
2. Deserialize and optimize recognized patterns to efficient implementations
3. Support subgraph matching (not just whole-graph)

Example:
```
User's graph:
  input → [LFO→Gain] → [LFO→Delay→Mix] → [Reverb] → output
              ↑              ↑              ↑
           tremolo        chorus      (no pattern)

After optimization:
  input → TremoloOpt → ChorusOpt → [Reverb nodes...] → output
```

## Design Decisions

### 1. Subgraph matching, not whole-graph

**Decision:** Match patterns as subgraphs within larger graphs.

**Reasoning:** Real-world graphs combine multiple effects. Users won't build "just a tremolo" - they'll build chains with tremolo as one part.

**Implication:** Need to handle pattern boundaries - how optimized nodes connect to the rest of the graph.

### 2. Fingerprint pre-filter + structural match

**Decision:** Two-phase matching:
1. Fast fingerprint check (node type counts)
2. Detailed structural match only on candidates

**Reasoning:**
- Graph sizes are small (typically 5-100 nodes)
- Pattern count is small (10-50 patterns)
- Matching runs once at setup, not per-sample
- Fingerprint check is O(N) linear scan, cache-friendly, SIMDable
- Most patterns rejected by fingerprint alone (tremolo needs LFO+Gain, if graph has no LFO, skip)

**Alternatives considered:**
- **Sequential pattern scan:** O(N×P), simple but doesn't share work
- **Rete network:** Single pass, but complex and overkill for small graphs
- **Index + relational queries:** Flexible but HashMap pointer-chasing is cache-unfriendly

**Why fingerprint wins:**
- At our scale, constants dominate asymptotic complexity
- Linear memory scan beats pointer chasing
- Simple to implement and debug
- Naturally extends to SIMD (compare byte arrays)

### 3. Patterns define connection points explicitly

**Decision:** Patterns declare their boundary ports:
```rust
struct Pattern {
    // Internal structure
    nodes: Vec<PatternNode>,
    internal_wires: Vec<Wire>,

    // Boundary (how this connects to rest of graph)
    audio_inputs: Vec<PortSpec>,   // External audio enters here
    audio_outputs: Vec<PortSpec>,  // Audio leaves here
    param_inputs: Vec<PortSpec>,   // External modulation accepted here
}
```

**Reasoning:** When we replace a subgraph with an optimized node, we need to rewire connections. Explicit ports make this unambiguous.

### 4. Non-overlapping matches applied greedily

**Decision:** When multiple patterns match overlapping subgraphs, apply greedily (largest/first wins). Don't try to find globally optimal covering.

**Reasoning:**
- Optimal subgraph covering is NP-hard
- Greedy is simple and predictable
- Users can understand what happened
- In practice, effect patterns rarely overlap ambiguously

**Future consideration:** Could add pattern priorities if needed.

### 5. Optimized nodes implement same trait

**Decision:** Optimized replacements implement `AudioNode`, same as graph nodes.

**Reasoning:**
- Uniform interface - rest of system doesn't care if node is primitive or optimized
- Can nest optimizations (optimized node inside a graph that gets further optimized)
- Matches existing architecture

### 6. Optimization is optional and explicit

**Decision:** Optimization is a separate pass, not automatic on every deserialization.

```rust
let graph = AudioGraph::deserialize(data);      // Just loads
let optimized = graph.optimize(&PATTERN_SET);   // Explicit optimization
```

**Reasoning:**
- Predictable behavior
- Users can debug unoptimized graph first
- Can compare optimized vs unoptimized
- No magic

## Architecture

### Fingerprint

```rust
/// Compact representation of node type counts.
/// Index = node type enum, value = count (saturates at 255).
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct GraphFingerprint {
    counts: [u8; NodeType::COUNT],
}

impl GraphFingerprint {
    /// O(N) linear scan over nodes.
    pub fn from_graph(graph: &AudioGraph) -> Self { ... }

    /// True if self has at least as many of each node type as required.
    pub fn contains(&self, required: &GraphFingerprint) -> bool {
        // SIMDable: self.counts >= required.counts element-wise
    }
}
```

### Pattern Definition

```rust
pub struct Pattern {
    /// Human-readable name for debugging.
    pub name: &'static str,

    /// Minimum node types required (for fingerprint filtering).
    pub required_fingerprint: GraphFingerprint,

    /// Pattern graph structure to match.
    pub structure: PatternStructure,

    /// Factory to create optimized replacement.
    pub build_optimized: fn(&MatchResult) -> Box<dyn AudioNode>,
}

pub struct PatternStructure {
    /// Nodes in pattern (type + optional constraints).
    pub nodes: Vec<PatternNode>,

    /// Required internal wiring.
    pub audio_wires: Vec<(usize, usize)>,        // (from_idx, to_idx)
    pub param_wires: Vec<(usize, usize, &'static str)>,  // (from_idx, to_idx, param)

    /// Boundary ports.
    pub inputs: Vec<usize>,    // Pattern node indices that receive external audio
    pub outputs: Vec<usize>,   // Pattern node indices that output to external
}

pub struct PatternNode {
    pub node_type: NodeType,
    pub constraints: Vec<ParamConstraint>,  // Optional: "rate > 0.1", etc.
}
```

### Matching Algorithm

```rust
pub fn optimize_graph(graph: &mut AudioGraph, patterns: &[Pattern]) {
    let fingerprint = GraphFingerprint::from_graph(graph);

    loop {
        let mut best_match: Option<(Match, &Pattern)> = None;

        // Phase 1: Filter by fingerprint
        let candidates: Vec<&Pattern> = patterns.iter()
            .filter(|p| fingerprint.contains(&p.required_fingerprint))
            .collect();

        // Phase 2: Structural match on candidates
        for pattern in candidates {
            if let Some(m) = structural_match(graph, pattern) {
                // Prefer larger matches (more nodes covered)
                if best_match.as_ref().map_or(true, |(b, _)| m.size() > b.size()) {
                    best_match = Some((m, pattern));
                }
            }
        }

        // Phase 3: Apply best match or terminate
        match best_match {
            Some((m, pattern)) => {
                let optimized = (pattern.build_optimized)(&m);
                replace_subgraph(graph, &m, optimized);
                // fingerprint changes, but we recompute next iteration
            }
            None => break,  // No more matches
        }
    }
}
```

### Subgraph Replacement

```rust
fn replace_subgraph(
    graph: &mut AudioGraph,
    match_: &Match,
    replacement: Box<dyn AudioNode>,
) {
    // 1. Add replacement node
    let new_id = graph.add(replacement);

    // 2. Rewire external inputs to replacement
    for (ext_node, matched_input) in match_.external_inputs() {
        // ext_node was connected to matched_input, now connect to new_id
        graph.reconnect(ext_node, matched_input, new_id);
    }

    // 3. Rewire replacement outputs to external nodes
    for (matched_output, ext_node) in match_.external_outputs() {
        graph.reconnect(matched_output, ext_node, new_id);
    }

    // 4. Remove matched nodes
    for node_id in match_.matched_nodes() {
        graph.remove(node_id);
    }
}
```

## Performance Characteristics

| Operation | Complexity | Notes |
|-----------|------------|-------|
| Fingerprint computation | O(N) | Linear scan, cache-friendly |
| Fingerprint comparison | O(T) | T = node type count (~20), SIMDable |
| Candidate filtering | O(P) | P = pattern count (~50) |
| Structural match | O(N × C) | C = candidates after filter (usually small) |
| Subgraph replacement | O(E) | E = edges touching subgraph |

For typical graphs (N=50, P=30, C=5): total ~250-500 operations, <1ms.

## Example Pattern: Tremolo

```rust
Pattern {
    name: "tremolo",
    required_fingerprint: fingerprint![LfoNode: 1, GainNode: 1],
    structure: PatternStructure {
        nodes: vec![
            PatternNode { node_type: NodeType::Lfo, constraints: vec![] },
            PatternNode { node_type: NodeType::Gain, constraints: vec![] },
        ],
        audio_wires: vec![],  // LFO doesn't send audio to Gain
        param_wires: vec![(0, 1, "gain")],  // LFO modulates Gain's gain param
        inputs: vec![1],   // External audio enters at Gain
        outputs: vec![1],  // Audio leaves from Gain
    },
    build_optimized: |m| {
        let lfo_rate = m.get_param(0, "rate");
        let depth = m.get_modulation_scale(0, 1);
        Box::new(TremoloOptimized::new(lfo_rate, depth))
    },
}
```

## Future Extensions

### 1. Pattern priorities
If patterns can overlap ambiguously, add priority field to prefer certain matches.

### 2. Parameterized patterns
Patterns that match structural variations:
```rust
// Match 2, 4, or 6 allpass stages (all valid phaser configs)
PatternNode { node_type: NodeType::Allpass, repeat: 2..=6 }
```

### 3. Cross-domain patterns
Same architecture could optimize `resin-core` Graph, mesh processing pipelines, etc.

### 4. Learned patterns
Profile graph execution, automatically identify hot subgraphs worth optimizing.

## Related Documents

- `audio-graph-primitives.md` - Performance tiers overview
- `ops-as-values.md` - Serialization philosophy
- `../domains/audio.md` - Audio domain overview
