# Pattern System Primitives

Design decisions for the rhythmic pattern system in `resin-audio`.

## Goals

1. **Minimal primitive set** - fewer primitives = easier for backends to implement
2. **Ops as values** - all operations are serializable structs
3. **Composable** - complex patterns built from simple parts
4. **Musically useful** - primitives should map to real musical concepts

## Current Primitives

### Event Generation
- `Pattern::pure(value)` - single event per cycle
- `Pattern::from_events(events)` - explicit event list

### Combination
- `cat(patterns)` - concatenate sequentially
- `stack(patterns)` - layer simultaneously

### Time Transformation
- `fast(factor, pattern)` / `slow(factor, pattern)` - time scaling
- `shift(amount, pattern)` - phase offset

### Conditional
- `every(n, f, pattern)` - apply transformation every N cycles
- `degrade(probability, seed, pattern)` - probabilistic event removal

### Rhythm Generation
- `euclid(k, n, value)` - Euclidean rhythm (Bjorklund's algorithm)

## Proposed Addition: `warp`

```rust
#[derive(Clone, Serialize, Deserialize)]
pub struct Warp {
    pub time_expr: FieldExpr,
}
```

Remaps event timing via a Dew expression. Covers multiple use cases:

| Use Case | Expression |
|----------|------------|
| Swing | `x + (floor(x * 2) % 2) * amount` |
| Humanize | `x + rand(x) * amount` |
| Quantize (floor) | `floor(x * grid) / grid` |
| Quantize (round) | `round(x * grid) / grid` |
| Quantize (ceil) | `ceil(x * grid) / grid` |

**Why a primitive**: One operation covers swing, humanize, and quantize with different rounding modes. High value-to-complexity ratio.

## Rejected: `range` + `filter`

We considered decomposing `euclid` into:
```rust
range(n, value).filter(|i| euclidean_predicate(i, k, n))
```

### Trade-offs

**Pros:**
- More composable
- `range` is useful on its own (metronome, grid)
- Consistent with "general internal, constrained API"

**Cons:**
- `filter` has limited use cases beyond Euclidean:
  - Drop every Nth event (rare)
  - Probability filtering (`degrade` already exists)
- Adds two primitives to save one
- Predicate integration complexity (binding x to onset vs index)

### Decision

Keep `euclid` as a primitive. The decomposition doesn't pay for itself - `filter` isn't useful enough elsewhere to justify its existence.

## Why `euclid` Can't Use Dew Directly

Dew is designed for point evaluation: `(x, y, z, t) → value`. It doesn't support:
- Array construction (building a sequence)
- Iteration state (Bjorklund's bucket algorithm)

While the Euclidean formula CAN be expressed per-point:
```
floor(i * k / n) - floor((i - 1) * k / n) > 0.5
```

This requires a `filter` primitive to apply it, which isn't justified (see above).

**Pattern generation is a different computational model than point evaluation.** Dew stays simple, patterns have their own primitives.

## Polyrhythms

Already composable via existing primitives:
```rust
stack(vec![
    fast(3.0, pattern),  // 3 beats per cycle
    fast(4.0, pattern),  // 4 beats per cycle
])
```

No new primitive needed.

## Implementation Notes

### Ops as Values

Pattern operations should be serializable structs:

```rust
#[derive(Clone, Serialize, Deserialize)]
pub struct Euclid<T> {
    pub hits: usize,
    pub steps: usize,
    pub value: T,
}

impl<T: Clone + Send + Sync + 'static> Euclid<T> {
    pub fn apply(&self) -> Pattern<T> {
        euclid(self.hits, self.steps, self.value.clone())
    }
}
```

Note: `Pattern<T>` itself contains `Arc<dyn Fn>` and is NOT serializable. The ops describe how to build patterns, not the patterns themselves.

### Warp Implementation

```rust
#[derive(Clone, Serialize, Deserialize)]
pub struct Warp {
    pub time_expr: FieldExpr,
}

impl Warp {
    pub fn apply<T: Clone + Send + Sync + 'static>(&self, pattern: Pattern<T>) -> Pattern<T> {
        let query = pattern.query;
        let expr = self.time_expr.clone();
        Pattern::from_query(move |arc| {
            query(arc)
                .into_iter()
                .map(|e| Event {
                    onset: expr.eval(e.onset as f32, 0.0, 0.0, 0.0, &HashMap::new()) as f64,
                    duration: e.duration,
                    value: e.value,
                })
                .collect()
        })
    }
}
```

~15 lines. Requires `resin-expr-field` dependency.

## Summary

| Primitive | Justification |
|-----------|---------------|
| `pure`, `from_events` | Event generation fundamentals |
| `cat`, `stack` | Combination fundamentals |
| `fast`, `slow`, `shift` | Time transformation fundamentals |
| `every`, `degrade` | Conditional/probabilistic |
| `euclid` | Unique algorithm, decomposition not worth it |
| `warp` (proposed) | Covers swing/humanize/quantize |

Rejected primitives:
- `range` - only useful with `filter`
- `filter` - not enough use cases beyond Euclidean
- `swing`, `humanize`, `quantize` - sugar over `warp`, avoid sugar

## Future Considerations

- **Integer support in Dew**: Would make predicates cleaner but doesn't change the architectural decision about `filter`
- **JIT for patterns**: `warp` expressions could be JIT-compiled for performance
- **More rhythm algorithms**: If added, evaluate whether they decompose to existing primitives first
