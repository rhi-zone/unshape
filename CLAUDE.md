# CLAUDE.md

Behavioral rules for Claude Code working in this repository.

**Unshape goal:** Constructive generation and manipulation of media - 3D meshes/rigging, 2D vector art/rigging, audio, textures/noise. Designed as a substrate for both archival/reproducible work (graph as project file, fully rewindable) and live signal-driven experiences (any node input can be a live signal). Realtime is a first-class concern, not an afterthought. See `docs/philosophy.md` for design philosophy and `docs/prior-art.md` for references.

**Bevy compatibility:** Compatible with bevy ecosystem but no hard dependency. Use individual bevy crates (e.g., `bevy_math`, `bevy_reflect`) where useful. Core types should be convertible to/from bevy equivalents.

Design docs: `docs/` (VitePress). Architecture decisions should live there.

## Behavioral Patterns

From ecosystem-wide session analysis:

- **Question scope early:** Before implementing, ask whether it belongs in this crate/module
- **Check consistency:** Look at how similar things are done elsewhere in the codebase
- **Implement fully:** No silent arbitrary caps, incomplete pagination, or unexposed trait methods
- **Name for purpose:** Avoid names that describe one consumer
- **Verify before stating:** Don't assert API behavior or codebase facts without checking

## Workflow

**Batch cargo commands** to minimize round-trips:
```bash
cargo clippy --all-targets --all-features -- -D warnings && cargo test -q
```
After editing multiple files, run the full check once — not after each edit. Formatting is handled automatically by the pre-commit hook (`cargo fmt`).

**Prefer `cargo test -q`** over `cargo test` — quiet mode only prints failures, significantly reducing output noise and context usage.

**When making the same change across multiple crates**, edit all files first, then build once.

**Minimize file churn.** When editing a file, read it once, plan all changes, and apply them in one pass. Avoid read-edit-build-fail-read-fix cycles by thinking through the complete change before starting.

**Use `normalize view` for structural exploration:**
```bash
~/git/rhizone/normalize/target/debug/normalize view <file>    # outline with line numbers
~/git/rhizone/normalize/target/debug/normalize view <dir>     # directory structure
```

## Commit Convention

Use conventional commits: `type(scope): message`

Types:
- `feat` - New feature
- `fix` - Bug fix
- `refactor` - Code change that neither fixes a bug nor adds a feature
- `docs` - Documentation only
- `chore` - Maintenance (deps, CI, etc.)
- `test` - Adding or updating tests

Scope is optional but recommended for multi-crate repos.

## Hard Constraints

- No `--no-verify`. Fix the issue or fix the hook.
- No path dependencies in `Cargo.toml` — they couple repos and break independent publishing.
- No interactive git (`git add -p`, `git add -i`, `git rebase -i`) — these block on stdin and hang.
- No assuming a tool is missing without checking `nix develop`.
- No returning tuples from functions — use structs with named fields.
- No string-matching when structure exists — use proper typed representations.
- No DSLs — custom syntax is hard to maintain and creates learning burden. Use Rust APIs instead (builders, combinators, method chaining).
- No generic error catches — catch specific error types.

## Design Principles

**General internal, constrained APIs.** Store the general representation, expose simpler APIs for common cases:
- VectorNetwork internally, Path API for linear curves
- HalfEdgeMesh internally, IndexedMesh for GPU
- AudioGraph internally, Chain for linear pipelines
- See `docs/design/general-internal-constrained-api.md`

**Exception: Multiple co-equal primitives.** When conversion between representations is not viable (O(N²) explosion, lossy, fundamentally different trade-offs), multiple concrete types can be co-equal primitives unified by a trait:
- `TileSet` (explicit adjacency) vs `WangTileSet` (edge-color indexed) - both implement `AdjacencySource`
- Converting 1000 Wang tiles → TileSet = 1M rules, not viable
- The *trait* is the abstraction; concrete types are interchangeable primitives
- This is NOT the same as "convenience wrappers" - these are genuinely different representations for different use cases

**Generative mindset.** Everything in unshape should be describable procedurally:
- Prefer node graphs / expression trees over baked data
- Parameters > presets
- Composition > inheritance

**Operations as values.** THIS IS CRITICAL. Every new piece of functionality MUST be an op struct first, method second.

```rust
// CORRECT: Op struct with all parameters
#[derive(Clone, Serialize, Deserialize)]
pub struct Subdivide { pub levels: u32 }

impl Subdivide {
    pub fn apply(&self, mesh: &Mesh) -> Mesh { ... }
}

// Method is SUGAR for the op - just delegates
impl Mesh {
    pub fn subdivide(&self, levels: u32) -> Mesh {
        Subdivide { levels }.apply(self)
    }
}
```

**Why this matters:**
- Serialization: ops can be saved/loaded as JSON, enabling project files
- History/undo: collect ops into a Vec, replay or reverse them
- Node graphs: ops become nodes trivially
- Inspection: users can see what parameters were used
- Reproducibility: same ops = same output

**Apply to ALL domains:**
```rust
// Image
pub struct GaussianBlur { pub radius: f32, pub sigma: f32 }
pub struct ExtractBitPlane { pub channel: Channel, pub bit: u8 }
pub struct Fft2d { pub inverse: bool }

// Audio
pub struct LowPass { pub cutoff_hz: f32, pub resonance: f32 }
pub struct Reverb { pub room_size: f32, pub damping: f32 }

// Mesh
pub struct Extrude { pub distance: f32, pub segments: u32 }
pub struct Bevel { pub width: f32, pub segments: u32 }
```

**Anti-patterns to AVOID:**
```rust
// BAD: Function with many parameters, no struct
pub fn blur(image: &Image, radius: f32, sigma: f32, edge_mode: EdgeMode) -> Image

// BAD: Method that doesn't delegate to an op
impl Image {
    pub fn blur(&self, radius: f32) -> Image {
        // implementation directly here - NOT serializable!
    }
}

// BAD: Proposing a "primitive" as just a function
// "we could add an extract_bit_plane() function" - NO! Make it a struct first
```

**When proposing new functionality, ALWAYS structure as:**
1. Define the op struct with all parameters
2. Implement `apply(&self, input) -> output`
3. Optionally add method sugar on the input type
4. Derive Serialize/Deserialize

See `docs/design/ops-as-values.md` for full rationale.

## Conventions

### Rust

- Edition 2024
- Workspace with sub-crates by domain (e.g., `crates/rhi-unshape-mesh/`, `crates/rhi-unshape-audio/`)
- Implementation goes in sub-crates, not all in one monolith

### Core Crates

**unshape-core** - Node graph system:
- `Graph`, `NodeId`, `Wire` - node graph container and execution
- `DynNode` trait - dynamic node execution with type-erased inputs/outputs
- `Value` - runtime value type for graph data flow

**unshape-geometry** - Geometry attribute traits:
- `HasPositions`, `HasPositions2D` - vertex positions (3D/2D)
- `HasNormals`, `HasUVs`, `HasColors`, `HasIndices` - other vertex attributes
- `Geometry`, `FullGeometry` - composite trait bounds

**unshape-op** - Operations as values (dynop system):
- `DynOp` trait, `OpRegistry`, `Pipeline` - for serializable operations
- `#[derive(Op)]` macro - derive for domain ops

**unshape-serde** - Graph serialization:
- `SerialGraph`, `NodeRegistry` - JSON/bincode graph format

**unshape-field** - Lazy evaluation:
- `Field<I, O>` trait - composable function abstraction for noise, SDFs, textures
- Re-exports from `unshape-field-ops` (where the `Field` trait actually lives)

**Expression language naming:** The "dew" expression language is `wick-core` on crates.io. `Expr::free_vars()` is enabled by default via the `introspect` feature.

### Updating CLAUDE.md

Add: workflow patterns, conventions, project-specific knowledge.
Don't add: temporary notes (TODO.md), implementation details (docs/), one-off decisions (commit messages).

### Updating TODO.md

Proactively add features, ideas, patterns, technical debt.
- Next Up: 3-5 concrete tasks for immediate work
- Backlog: pending items
- When completing items: mark as `[x]`, don't delete
- TODO.md drifts stale — items marked `[ ]` may already be implemented; verify against code before assuming work is needed

### Documenting New Features

When adding a new feature or module:
1. **Document immediately** - write doc comments as you implement (rustdoc handles API details)
2. **Update `docs/features.md`** - add/update the crate's one-line summary in the index
3. **Update `docs/crates/<crate>.md`** - add conceptual docs:
   - What the crate is for (not API listings)
   - Related crates
   - Example use cases
   - Example compositions with other crates

### Reference Documents (Keep in Sync)

These documents are authoritative references - **update them when implementation changes**:

- **`docs/archive/decomposition-audit.md`** - Primitive decomposition audit. Update when:
  - Adding/removing primitives in any domain
  - Finding new decomposition opportunities
  - Changing the three-layer architecture (primitives → helpers → optimizer)

- **`docs/spec/graph-format.md`** (when created) - Graph JSON serialization spec. Update when:
  - `SerialGraph` structure changes (nodes, edges, metadata)
  - Node/edge format changes
  - Expression (dew) AST serialization changes

### Working Style

Agentic by default - continue through tasks unless:
- Genuinely blocked and need clarification
- Decision has significant irreversible consequences
- User explicitly asked to be consulted

Commit consistently. Each commit = one logical change.

### Invariant Tests

For modules with statistical or mathematical properties, add feature-gated invariant tests that verify correctness beyond simple unit tests. Gate behind `invariant-tests` feature to keep normal test runs fast.

**Good candidates for invariant tests:**
- **Noise**: spectral slopes (white=0, pink=-1, brown=-2, violet=+2), autocorrelation, distribution uniformity
- **Image**: blue noise distribution (negative autocorrelation, even spacing), blur kernel sums to 1, dithering preserves average brightness
- **Audio**: filter frequency response via FFT, oscillator frequency accuracy, envelope smoothness
- **Mesh**: Euler characteristic preservation (V - E + F), subdivision count relationships, normal unit length
- **Spatial**: range queries return all/only points in bounds, k-nearest returns exactly k correctly ordered
- **Easing**: ease(0)≈0, ease(1)≈1, monotonicity where expected
- **Curve**: arc length accuracy, continuity at knots

**Pattern:**
```rust
// In Cargo.toml
[features]
invariant-tests = []

// In lib.rs
#[cfg(all(test, feature = "invariant-tests"))]
mod invariant_tests {
    // Statistical/mathematical property tests here
}
```

Run with: `cargo test -p crate-name --features invariant-tests`

<!-- BEGIN ECOSYSTEM RULES -->

## Delegation & relay

The main session is an orchestrator, not an implementer. It never answers world/codebase
questions from its own priors and never ingests raw foreign content (file/command output,
fetched text): that anti-signal anchors it to the state being left, dilutes the user's
direction, and can carry injection that then poisons every subagent it later spawns. Its
only epistemic act is route → reason over the returned, attenuated digest. Exploration and
implementation happen in subagents; the orchestrator ingests only the user's input and its
subagents' digests. Guessing is not an available move. When delegating, name the explicit agent type the work calls for rather than a generic subagent — a custom default can't be forced onto every subagent, so specialized disposition only applies when you ask for it by name. Delegation names the cheapest tier adequate to the task, and frontier-tier subagents or fan-outs happen only after the user approves a stated cost estimate — spend is the user's decision, never a silent default.

Relay/blackboard is the mechanism — reach for it when it earns its keep. When a payload is
large or evidence-heavy enough that passing it through the orchestrator's context would
poison it, or when a downstream critic must read by path so the orchestrator routes on a
verdict without ingesting the evidence, the subagent writes its raw output to a file the
orchestrator never opens and returns a path + short, provenance-marked digest. That is what
stops conclusions being laundered in place of evidence. Otherwise the subagent just returns
its digest; don't write a file by default. Persist to a tracked path only when the output is
durable (docs-shaped repos: `docs/artifacts/<session>/`); ephemeral relay scratch stays out
of the tracked tree.

## Hard Constraints

- No `--no-verify`. Fix the issue or fix the hook.
- No path dependencies in `Cargo.toml` — they couple repos and break independent publishing.
- No interactive git (no `git rebase -i`, no `git add -i`, no `--no-edit` on rebase).
- No suggesting project names. LLMs are bad at this; refine the conceptual space only.
- No tracking cross-project issues in conversation — they go in TODO.md in the affected repo.
- No assuming a tool is missing without checking `nix develop`.
- No entering plan mode except to present the handoff itself, and only when that is the
  ONLY remaining step. Subagents spawned from inside plan mode can only write their own
  plan files — not the files the work needs — so every delegated write and commit must
  be complete before EnterPlanMode.
- Commit completed work in the same turn it finishes. Uncommitted work is lost work.

## Disposition

How the agent thinks — embodied, not rules to check against:

- Something unexpected is a signal. Stop and find out why; never accept the anomaly and
  proceed.
- **The agent does not guess — it is clear and it proceeds, or it is unclear and it asks.**
  This is a bright line, not a preference: never submit a guess, never ship a design you are
  not clear is right. The move is binary — when the path is clear, act; when it is unclear,
  clarify — and there is no third mode where the agent floats a tentative wrong thing to see
  if it sticks. When it is uncertain which mode applies, that uncertainty is itself
  unclarity: ask. Crucially, inventing options and laying them out as a menu is still guessing;
  a fabricated set of choices is not clarification, it is a guess wearing more hats. What IS
  clarification is surfacing a divergence that genuinely exists in the problem — a real
  branch point, including a legitimately-open tradeoff whose call is the user's — put as a
  question. The discriminator is provenance: a branch the problem actually contains,
  surfaced, is clarification; a branch the agent fabricated and dressed as choices is a
  guess. So don't pronounce conclusions and don't cling to them: on any rejection reset the
  footing — return to the last thing the user certified and re-derive from there, never patch
  forward from the rejected thing. The user decides; only certified items count as settled; a
  guess recorded as fact poisons every loop built on it. (This wording is newly installed and
  under live evaluation — the *formulation* is provisional and awaiting testing in the wild;
  the injunction against guessing is not. Supersedes the earlier "offer attempts, not
  verdicts" framing, whose "attempt" was a poisoned name that licensed exactly this guessing.)
- **The agent suggests, the user decides — and to speak a thing as settled it must have
  earned the standing.** A candidate stays a candidate until earned standing closes it (the
  user asked for the opinion; it can cite a file read, a command run, a source quoted);
  voiced as fact without that, an unsolicited evidence-free judgment is the live failure.
  Standing scales to the cost of being wrong: a wrong direction can burn weeks and may never
  be recovered, while hedging-when-right costs a breath, and in the moment the two look
  identical — so the more a reversal would cost, the more a claim must earn before it
  hardens. (root failure: confabulation.)
- **Act from the live source, read fresh — before acting on context, and again when
  challenged.** Let the evidence place the answer: hold if you were right, correct
  specifically if you were wrong; the new position comes from re-reading, never from the
  pressure. (failures: stale-context action; backpedaling.)
- **Finish migrations before building on top; fence what you can't finish.** A partial
  refactor poisons context — old patterns that dominate by count get read as canonical and
  copied forward. Complete the migration, or explicitly mark old code as legacy, before
  adding new code on top.

<!-- END ECOSYSTEM RULES -->
