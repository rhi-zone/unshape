# CLAUDE.md

Behavioral rules for Claude Code working in this repository.

**Resin goal:** Constructive generation and manipulation of media - 3D meshes/rigging, 2D vector art/rigging, audio, textures/noise. See `docs/philosophy.md` for design philosophy and `docs/prior-art.md` for references.

**Bevy compatibility:** Compatible with bevy ecosystem but no hard dependency. Use individual bevy crates (e.g., `bevy_math`, `bevy_reflect`) where useful. Core types should be convertible to/from bevy equivalents.

Design docs: `docs/` (VitePress). Architecture decisions should live there.

## Core Rule

ALWAYS NOTE THINGS DOWN. When you discover something important, write it immediately:
- Bugs/issues → fix them or add to TODO.md
- Design decisions → docs/ or code comments
- Future work → TODO.md
- Conventions → this file
- Key insights → THIS FILE, immediately

**Triggers to document immediately:**
- User corrects you → write down what you learned before fixing
- Trial-and-error (2+ failed attempts) → document what actually works
- Framework/library quirk discovered → add to relevant docs/ file
- "I'll remember this" thought → you won't, write it down now

## Behavioral Patterns

From ecosystem-wide session analysis:

- **Question scope early:** Before implementing, ask whether it belongs in this crate/module
- **Check consistency:** Look at how similar things are done elsewhere in the codebase
- **Implement fully:** No silent arbitrary caps, incomplete pagination, or unexposed trait methods
- **Name for purpose:** Avoid names that describe one consumer
- **Verify before stating:** Don't assert API behavior or codebase facts without checking

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

## Negative Constraints

Do not:
- Announce actions with "I will now..." - just do them
- Write preamble or summary in generated content
- Catch generic errors - catch specific error types
- Leave work uncommitted
- Create special cases - design to avoid them
- **Return tuples from functions** - use structs with named fields
- **String-match when structure exists** - use proper typed representations
- **Create DSLs** - custom syntax is subjective, hard to maintain, and creates learning burden. Use Rust APIs instead (builders, combinators, method chaining)
- Use path dependencies in Cargo.toml - causes clippy to stash changes across repos
- Use `--no-verify` - fix the issue or fix the hook
- Assume tools are missing - check if `nix develop` is available for the right environment

## Design Principles

**Unify, don't multiply.** Fewer concepts = less mental load.
- One interface that handles multiple cases > separate interfaces per case
- Plugin/trait systems > hardcoded switches
- Extend existing abstractions > create parallel ones

**Simplicity over cleverness.**
- If proposing a new dependency, ask: can stdlib/existing code do this?
- HashMap > inventory crate. OnceLock > lazy_static. Functions > traits (until you need the trait).

**Explicit over implicit.**
- Convenience = zero-config. Hiding information = pretending everything is okay.
- Log when skipping something - user should know why.

**General internal, constrained APIs.** Store the general representation, expose simpler APIs for common cases:
- VectorNetwork internally, Path API for linear curves
- HalfEdgeMesh internally, IndexedMesh for GPU
- AudioGraph internally, Chain for linear pipelines
- See `docs/design/general-internal-constrained-api.md`

**Generative mindset.** Everything in resin should be describable procedurally:
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
- Workspace with sub-crates by domain (e.g., `crates/rhizome-resin-mesh/`, `crates/rhizome-resin-audio/`)
- Implementation goes in sub-crates, not all in one monolith

### Core Crates

**resin-core** - Node graph system:
- `Graph`, `NodeId`, `Wire` - node graph container and execution
- `DynNode` trait - dynamic node execution with type-erased inputs/outputs
- `Value` - runtime value type for graph data flow

**resin-geometry** - Geometry attribute traits:
- `HasPositions`, `HasPositions2D` - vertex positions (3D/2D)
- `HasNormals`, `HasUVs`, `HasColors`, `HasIndices` - other vertex attributes
- `Geometry`, `FullGeometry` - composite trait bounds

**resin-op** - Operations as values (dynop system):
- `DynOp` trait, `OpRegistry`, `Pipeline` - for serializable operations
- `#[derive(Op)]` macro - derive for domain ops

**resin-serde** - Graph serialization:
- `SerialGraph`, `NodeRegistry` - JSON/bincode graph format

**resin-field** - Lazy evaluation:
- `Field<I, O>` trait - composable function abstraction for noise, SDFs, textures

### Updating CLAUDE.md

Add: workflow patterns, conventions, project-specific knowledge.
Don't add: temporary notes (TODO.md), implementation details (docs/), one-off decisions (commit messages).

### Updating TODO.md

Proactively add features, ideas, patterns, technical debt.
- Next Up: 3-5 concrete tasks for immediate work
- Backlog: pending items
- When completing items: mark as `[x]`, don't delete

### Documenting New Features

When adding a new feature or module:
1. **Document immediately** - write doc comments as you implement (rustdoc handles API details)
2. **Update `docs/features.md`** - add/update the crate's one-line summary in the index
3. **Update `docs/crates/<crate>.md`** - add conceptual docs:
   - What the crate is for (not API listings)
   - Related crates
   - Example use cases
   - Example compositions with other crates

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
