# Dead Code Patterns

Analysis of dead code discovered during warning cleanup (2025-01-16). Documents patterns that led to unused code and how to avoid them.

## Pattern: Storing Constructor Parameters

**Problem:** Storing original constructor parameters alongside their derived/precomputed values.

**Example from `percussion.rs`:**

```rust
struct Mode {
    freq: f32,           // Never read after construction
    decay: f32,          // Never read after construction
    phase_inc: f32,      // Actually used (derived from freq)
    decay_coeff: f32,    // Actually used (derived from decay)
}
```

Similarly, `Membrane`, `Bar`, and `Plate` stored both `config` and `sample_rate` even though these were only needed to compute the `modes` vector during construction.

**Why it happens:**
- "Might need it later" thinking
- Debugging convenience (can inspect original values)
- Serialization concerns (need to reconstruct)

**Solution:** Only store what you actually use. If you need the original values:
1. For debugging: derive them back from the computed values, or add a `debug_info()` method
2. For serialization: the *config* struct should be serializable, not the runtime state
3. For "might need later": YAGNI - add it when you need it

**Applied fix:** Removed `freq`/`decay` from `Mode`, removed `sample_rate`/`config` from `Membrane`/`Bar`/`Plate`.

## Pattern: Redundant State in Search Algorithms

**Problem:** Duplicating state that's authoritatively stored elsewhere.

**Example from `navmesh.rs`:**

```rust
struct PathNode {
    poly_idx: usize,
    g_cost: f32,    // Redundant - authoritative value is in g_score HashMap
    f_cost: f32,    // Used for priority queue ordering
}
```

The A* implementation maintains `g_score: HashMap<usize, f32>` as the source of truth. The `g_cost` in `PathNode` was just a snapshot at insertion time, never read back.

**Why it happens:**
- Copy-paste from reference implementations that use a different structure
- Premature optimization ("avoid HashMap lookup")
- Incomplete refactoring

**Solution:** Identify the authoritative source and remove duplicates. If you need the value in the struct for Ord/priority, that's fine - but if it's also in a HashMap, one is redundant.

**Applied fix:** Removed `g_cost` from `PathNode`.

## Pattern: Loop Variables Overwritten Before Use

**Problem:** Computing a value in a loop that gets recomputed after the loop anyway.

**Example from `boolean.rs`:**

```rust
let mut best_point = curve.evaluate(0.0);  // Initial value never used

for i in 0..=samples {
    let p = curve.evaluate(t);
    if dist < best_dist {
        best_t = t;
        best_point = p;  // Overwritten each iteration, only best_t matters
    }
}

// Newton refinement changes best_t, then:
best_point = curve.evaluate(best_t);  // Recomputed anyway
```

The loop only needs to track `best_t` and `best_dist`. The final `best_point` is computed from the refined `best_t`.

**Why it happens:**
- Premature optimization ("avoid recomputing the point")
- Evolving algorithm where Newton refinement was added later

**Solution:** Think about what the loop's *output* is. If a variable is always overwritten after the loop, don't track it in the loop.

**Applied fix:** Removed `best_point` from the loop, only track `best_t` and `best_dist`.

## Pattern: Planned But Unimplemented Variants

**Problem:** Adding enum variants or struct fields for features that don't exist yet.

**Example from `svg.rs`:**

```rust
enum SvgElement {
    Path { ... },
    Circle { ... },
    Rect { ... },
    Line { ... },
    Group { elements: Vec<SvgElement>, transform: Option<String> },  // Never constructed
}
```

The `Group` variant was added anticipating SVG group support, but no code ever creates it.

**Why it happens:**
- "Design for the future"
- Making the enum "complete" according to the SVG spec
- Started implementing, got distracted

**Solutions:**
1. **YAGNI approach:** Remove it entirely. Add when needed.
2. **Explicit deferral:** Keep it with `#[allow(dead_code)]` and a comment explaining it's planned but unimplemented. This is appropriate when:
   - The rendering/handling code already exists (as with SvgElement::Group)
   - The feature is clearly useful and will be added
   - Removing would lose non-trivial work

**Applied fix:** Kept `Group` with `#[allow(dead_code)]` since the write_element handler already exists.

## Pattern: Unused Imports After Refactoring

**Problem:** Imports that were used before a refactoring but are now orphaned.

**Examples:**
- `Quat` imported in `animation.rs` but quaternion operations moved elsewhere
- `Interpolate` imported in `blend.rs` but trait bounds simplified

**Why it happens:**
- Manual refactoring misses cleanup
- IDE doesn't always catch unused imports in Rust (especially with re-exports)

**Solution:** Run `cargo check` after refactoring. Use `cargo fix` for mechanical fixes. Some IDEs have "optimize imports" features.

**Applied fix:** Removed unused imports.

## Pattern: Enumerate Without Using Index

**Problem:** Using `.enumerate()` when you don't need the index.

**Example from `delaunay.rs`:**

```rust
for (i, &(e1, e2)) in tri_edges.iter().enumerate() {
    // i is never used, only e1 and e2
}
```

**Why it happens:**
- Copy-paste from similar loop that did use the index
- Index was used during development/debugging, then removed

**Solution:** Use `for &(e1, e2) in tri_edges.iter()` if you don't need the index. If you might need it for debugging, use `_i` to suppress the warning while keeping the enumerate.

**Applied fix:** Used `_` for unused index.

## General Principles

1. **Store computed values, not inputs + computed values.** Choose one.
2. **One source of truth.** If data exists in two places, one is wrong (or redundant).
3. **YAGNI for structure.** Don't add fields/variants for hypothetical future use.
4. **Clean up after refactoring.** Warnings are your friend.
5. **Loop outputs matter.** Only track what survives past the loop.
