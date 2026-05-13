# CLAUDE.md

Behavioral rules for Claude Code working in this repository.

**Unshape goal:** Constructive generation and manipulation of media - 3D meshes/rigging, 2D vector art/rigging, audio, textures/noise. Designed as a substrate for both archival/reproducible work (graph as project file, fully rewindable) and live signal-driven experiences (any node input can be a live signal). Realtime is a first-class concern, not an afterthought. See `docs/philosophy.md` for design philosophy and `docs/prior-art.md` for references.

**Bevy compatibility:** Compatible with bevy ecosystem but no hard dependency. Use individual bevy crates (e.g., `bevy_math`, `bevy_reflect`) where useful. Core types should be convertible to/from bevy equivalents.

Design docs: `docs/` (VitePress). Architecture decisions should live there.

## Context Is The Only Scarce Resource

Every byte that enters the main session stays in the main session for its entire lifetime. File contents, command output, search results — once read, it lingers in cache and shapes every downstream token. There is no "just looking."

**All exploration runs in subagents.** Investigations, audits, deep dives, surveys, "let me check," "let me find" — if the purpose of a tool sequence is to find out something you don't yet know, it runs in a subagent. Renaming the activity does not change what it is. The subagent returns a distilled summary; the raw output stays in the subagent.

Inline tool use in the main context is reserved for:
- Reading a known file at a known path
- Edits/writes you're committing to
- A single targeted lookup whose result you'll act on immediately

If you find yourself running a second grep to refine the first, you should have spawned a subagent. Mechanical work across many files → parallel subagents.

## Durability

Subagent reports, mid-session realizations, "I'll remember this" — none of these outlast the session. Anything worth keeping goes into CLAUDE.md, code, docs, or a commit. If it isn't written down, it is gone.

**Note things down immediately — no deferral:**
- Problems, tech debt, issues → TODO.md now, in the same response
- Design decisions, key insights → docs/ or CLAUDE.md
- Future/deferred scope → TODO.md **before** writing any code, not after
- **Every observed problem → TODO.md. No exceptions.** Code comments and conversation mentions are not tracked items. If you write a TODO comment in source, the next action is to open TODO.md and write the entry.

**Commit completed work immediately.** After tests pass, commit. After each phase of a multi-phase plan, commit. Uncommitted work is lost work, and accumulated uncommitted phases lose isolation as well.

**Docs change in the same commit as the code.** New pages enter the sidebar in that commit. There is no follow-up.

## Authenticity

When asked to analyze X, read X. Do not synthesize from conversation memory, prior summaries, or what the file probably says. Claims must correspond to evidence produced this session.

**Something unexpected is a signal.** Surprising output, anomalous numbers, a file containing what it shouldn't — stop and find out why. Do not accept the anomaly and proceed.

## Discipline

Corrections from the user are conversation, not material for new rules. A single correction does not warrant a CLAUDE.md edit. Rules are added when a failure mode is observed repeatedly and the rule names the failure it prevents.

**Corrections are documentation lag, not model failure.** When the same mistake recurs, the fix is writing the invariant down — not repeating the correction. Every correction that doesn't produce a CLAUDE.md edit will happen again. Exception: during active design, corrections are the work itself — don't prematurely document a design that hasn't settled yet.

Do not announce actions ("I will now…"). Act.

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
