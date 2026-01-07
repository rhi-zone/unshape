# Philosophy

Core design principles for Resin.

## Generative Mindset

Everything in Resin should be describable procedurally:

- **Parameters over presets** - expose knobs, don't bake decisions
- **Expressions over constants** - values can be computed, animated, or data-driven
- **Node graphs over imperative code** - composition of operations, not sequences of mutations
- **Lazy evaluation** - build descriptions, evaluate on demand

## Unify, Don't Multiply

Fewer concepts = less mental load.

- One interface that handles multiple cases > separate interfaces per case
- Plugin/trait systems > hardcoded switches
- Extend existing abstractions > create parallel ones

When adding a new feature, first ask: can an existing concept handle this?

## Simplicity Over Cleverness

- Prefer stdlib over dependencies
- Functions over traits (until you need the trait)
- Explicit over implicit

If proposing a new dependency, ask: can existing code do this?

## Bevy Compatibility

Resin is designed to work with the bevy ecosystem without requiring it:

- Core types use `glam` for math (same as bevy)
- Types should implement `From`/`Into` for bevy equivalents where sensible
- Individual bevy crates (e.g., `bevy_reflect`) can be used where valuable
- No hard dependency on `bevy` itself

## Workspace Structure

Implementation is split by domain:

```
crates/
  resin/           # umbrella crate, re-exports
  resin-mesh/      # 3D mesh generation
  resin-audio/     # audio synthesis
  resin-texture/   # procedural textures
  resin-vector/    # 2D vector art
  resin-rig/       # rigging/animation
  resin-core/      # shared primitives
```

Each crate should be usable independently.
