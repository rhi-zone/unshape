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

## Negative Constraints

Do not:
- Announce actions with "I will now..." - just do them
- Write preamble or summary in generated content
- Catch generic errors - catch specific error types
- Leave work uncommitted
- Create special cases - design to avoid them
- **Return tuples from functions** - use structs with named fields
- **String-match when structure exists** - use proper typed representations

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

## Conventions

### Rust

- Edition 2024
- Workspace with sub-crates by domain (e.g., `crates/rhizome-resin-mesh/`, `crates/rhizome-resin-audio/`)
- Implementation goes in sub-crates, not all in one monolith

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
1. **Document immediately** - write doc comments as you implement
2. **Update `docs/features.md`** - add the feature to the catalog under the appropriate crate section
3. **Include API signatures** - list public types and functions with brief descriptions

### Working Style

Agentic by default - continue through tasks unless:
- Genuinely blocked and need clarification
- Decision has significant irreversible consequences
- User explicitly asked to be consulted

Commit consistently. Each commit = one logical change.
