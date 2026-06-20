# Determinism guard

Nodes (`impl DynNode`) and ops (`#[derive(unshape_op::Op)]` / `impl DynOp`)
**must be deterministic**: given the same inputs and the same `EvalContext`
(including its explicit `seed`), they must produce the same outputs. This is what
lets the graph be a project file — replayable, diffable, cacheable — *and* a live
signal substrate at the same time. Memoization, undo/redo, snapshot+replay, and
GPU/CPU backend equivalence all rest on it.

`DynNode::execute` is already `&self` + `Send + Sync`, which removes *some* ways
to be non-deterministic, but nothing stops a node from reaching for ambient
non-determinism: the wall clock, an unseeded RNG, process environment, or hidden
mutable global/thread-local state. The determinism guard mechanizes the ban so
those APIs can't quietly enter the node/op crates.

## What is banned, and where

Banned in the crates that define graph nodes / serializable ops (those containing
`impl DynNode`, `impl DynOp`, or `#[derive(unshape_op::Op)]`):

| API | Why |
| --- | --- |
| `std::time::SystemTime::now`, `std::time::Instant::now` | reading the clock; drive time via the explicit `EvalContext` time/`dt` inputs instead |
| `rand::random`, `rand::thread_rng`, `rand::rng` | unseeded randomness; construct a seeded `Rng` from an explicit seed (e.g. `EvalContext::seed` or an op `seed` field) |
| `std::env::var`, `std::env::vars` | reading process environment; pass configuration as explicit inputs |
| `thread_local!`, `static mut` | hidden ambient/mutable state |
| `thread::spawn` (inside ops) | non-deterministic interleaving |

**Seeded RNG is explicitly fine.** The ban targets only the *unseeded* free
functions. Constructing an `Rng` from an explicit seed and calling
`rng.random()` (a trait method, a different path) is the encouraged pattern.

## How it is enforced

Two halves, because no single tool covers everything:

1. **Function calls → clippy `disallowed-methods`** (`clippy.toml` at the repo
   root). Each entry carries a reason string explaining the deterministic
   alternative. `clippy.toml` is workspace-global (clippy has no per-crate
   scoping), so it bans these APIs everywhere; legitimate *runtime/optimizer*
   uses inside node/op crates carry an explicit
   `#[allow(clippy::disallowed_methods)]` at the use site with a documented
   reason (e.g. the evaluator's perf timing and LRU cache bookkeeping in
   `unshape-core::eval`, and `build.rs`/test fixtures).

2. **Macros & language items → `tooling/check-determinism.sh`**. clippy can't see
   `thread_local!`, `static mut`, or `thread::spawn`, so a small grep-based check
   scans the `src/` of the node/op crates and fails on matches. A genuinely
   deterministic use (e.g. an immutable const lookup table) may opt out with a
   marker comment on the same line or the line immediately above:

   ```rust
   // determinism-guard: allow -- immutable read-only const lookup table
   thread_local! { /* ... */ }
   ```

Both halves run in CI (`.github/workflows/ci.yml`) and in the pre-commit hook
(`.githooks/pre-commit`).

## Keeping the crate list in sync

`tooling/check-determinism.sh` carries an explicit `NODEOP_CRATES` list. A crate
belongs there if it contains a node or op. Regenerate the candidate list with:

```bash
grep -rl 'impl DynNode\|impl .*DynOp\|unshape_op::Op' crates/*/src \
  | sed -E 's|/src/.*||' | sort -u
```

The clippy half needs no per-crate list — it is workspace-global by construction.
