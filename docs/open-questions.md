# Open Questions

Single source of truth for design decisions. Updated as we resolve questions.

## Resolution Key

- ✅ **Resolved** - Decision made, documented
- 🔶 **Leaning** - Direction chosen, may revisit
- ❓ **Open** - Needs investigation/decision

---

## Core Architecture

| Question | Status | Notes |
|----------|--------|-------|
| Type system for slots | 🔶 Leaning | Simpler than maki (opaque types). Generics maybe unnecessary |
| Unified graph container | ❓ Open | Typed slots may enable mixed-domain graphs |
| Portable workflows | ❓ Open | Should graphs be serializable/shareable artifacts? |
| Parameter system | 🔶 Leaning | Yes, first-class across all domains |
| Modularity | 🔶 Leaning | Very modular, bevy philosophy |
| Bevy integration | 🔶 Leaning | Low priority. Separate adapter crate if needed. Must not affect core design - resin is standalone first |
| Evaluation strategy | ❓ Open | Lazy vs eager? Pull vs push? |
| Time models | ❓ Open | Stateless vs stateful vs streaming. Recurrence = stateful. See [time-models](./design/time-models.md), [recurrent-graphs](./design/recurrent-graphs.md) |

## Expression Language

Backend strategy resolved. Language design still open.

| Question | Status | Notes |
|----------|--------|-------|
| Backend selection | ✅ Resolved | Cranelift JIT (CPU hot paths), WGSL (GPU), Interpreted (fallback). See [closure-usage-survey](./design/closure-usage-survey.md) |
| Expression AST scope | ❓ Open | Just math? Conditionals? Variables? Loops? |
| Per-domain or unified | ❓ Open | Same Expr everywhere, or MeshExpr/AudioExpr/etc? |
| Expr → WGSL codegen | ❓ Open | How to translate Expr AST to WGSL shader code? |
| Expr → Cranelift codegen | ❓ Open | IR generation details, libm calls, etc. |
| Built-in functions | ❓ Open | sin/cos/pow obvious. noise()? smoothstep()? domain-specific? |
| How common are custom expressions? | ❓ Open | If rare, named ops suffice. If common, need rich language |

## Ops & Serialization

| Question | Status | Notes |
|----------|--------|-------|
| Ops as values | 🔶 Leaning | Yes, ops are serializable structs. See [ops-as-values](./design/ops-as-values.md) |
| Plugin op registration | ✅ Resolved | Core defines trait + serialization contract. Plugin *loading* is host's responsibility. Optional adapters (resin-wasm-plugins, etc.) for common cases. See [plugin-architecture](./design/plugin-architecture.md) |
| Graph evaluation caching | ✅ Resolved | Hash-based caching at node boundaries (salsa-style). Inputs unchanged → return cached output |
| External references | 🔶 Leaning | IDs + resolution context (ComfyUI pattern). Maybe support embedding small assets for reproducibility? |

## Meshes

| Question | Status | Notes |
|----------|--------|-------|
| Half-edge vs index-based | ❓ Open | Half-edge better for topology, index-based GPU-friendly. Both? Convert at boundaries? |
| Instancing | ❓ Open | How to represent "100 copies with different transforms"? |
| SDF integration | ❓ Open | Separate representation or unify with mesh ops? |
| Fields for selection | ❓ Open | Blender's `position.z > 0` as selection. How much do we want? |

## Audio

| Question | Status | Notes |
|----------|--------|-------|
| Sample rate | ❓ Open | Fixed at graph creation or runtime-configurable? |
| Block size | ❓ Open | Fixed or variable? Trade-off: efficiency vs latency |
| Modulation depth | ❓ Open | Every param modulatable (VCV)? Or explicit mod inputs (Pd)? |
| Polyphony model | ❓ Open | Per-node (VCV poly cables)? Per-graph (Pd clone)? Explicit voice management? |
| Control vs audio rate | ❓ Open | Automatic promotion? Explicit types like SuperCollider .kr/.ar? |
| State management | 🔶 Leaning | Recurrent graphs: feedback edges carry state, nodes stay pure. Open: delay granularity, stability, mixed rates. See [recurrent-graphs](./design/recurrent-graphs.md) |

## Textures

| Question | Status | Notes |
|----------|--------|-------|
| GPU vs CPU | ✅ Resolved | Abstract over both via burn/CubeCL. See [prior-art](./prior-art.md#burn--cubecl) |
| Tiling | ✅ Resolved | Explicit operators. Tiling isn't fundamental to most generators. `MakeSeamless`, `Tile`, etc. |
| Resolution/materialization | ❓ Open | When to materialize vs keep lazy? Blur/normal-from-height need neighbors |
| 3D textures | ✅ Resolved | Same nodes, Vec3 input. Vec4 for looping animation (time as 4th dim). Memory/preview are host concerns. |
| Texture vs field | ❓ Open | In shaders they're the same (sample at coord). Needs exploration. Textures need materialization, fields are lazy. |

## Vector 2D

| Question | Status | Notes |
|----------|--------|-------|
| Curve types | ✅ Resolved | All via traits. See [curve-types](./design/curve-types.md) |
| Precision f32/f64 | ✅ Resolved | Support both via generic `T: Float` |
| Winding rule | ✅ Resolved | Both, default non-zero. See [winding-rules](./design/winding-rules.md) |
| Vector networks | ✅ Resolved | Network internally, both APIs as equals. See [vector-networks](./design/vector-networks.md) |
| Text | 🔶 Leaning | Include outline extraction, exclude layout (harfbuzz territory) |
| Path ↔ rigging | ❓ Open | How does path animation/morphing relate to rigging system? |

## Rigging

| Question | Status | Notes |
|----------|--------|-------|
| Unified 2D/3D rig | 🔶 Leaning | Yes, some abstractions shareable (bones, constraints, skinning) |
| Deformer stacking | ✅ Resolved | Graph internal, Stack API. See [deformer-stacking](./design/deformer-stacking.md) |
| Animation blending | 🔶 Leaning | Separate crate, bevy-style modularity |
| Procedural rigging | ❓ Open | Auto-rig from mesh topology? |
| Real-time vs offline | ❓ Open | Games need <16ms. Different constraints? |

---

## Summary by Status

### ✅ Resolved (13)
- GPU vs CPU abstraction (burn/CubeCL)
- Precision f32/f64 (generic `T: Float`)
- Winding rule (both, default non-zero)
- Curve types (trait-based)
- Vector networks (network internal, both APIs)
- Deformer stacking (graph internal, stack API)
- Expression backend (Cranelift/WGSL/Interpreted)
- General internal, constrained APIs pattern
- Plugin architecture (core = contract, host = loading)
- Graph caching (hash-based at node boundaries)
- 3D textures (same nodes, Vec3/Vec4 input)
- Tiling (explicit operators)

### 🔶 Leaning (12)
- Type system for slots (simpler than maki)
- Parameter system (yes, first-class)
- Modularity (very modular)
- Ops as values (yes)
- Text (outlines yes, layout no)
- Unified 2D/3D rig (yes)
- Animation blending (separate crate)
- Expression language direction (Cranelift for CPU)
- Bevy integration (low priority, standalone first)
- External references (IDs + context, maybe embed small assets)
- Audio state management (recurrent graphs, feedback edges)

### ❓ Open (19+)
- **High impact**: Half-edge vs index mesh, Evaluation strategy, Time models
- **Expression language**: AST scope, codegen details, built-in functions
- **Cross-cutting**: Texture vs field unification
- **Domain-specific**: Many audio questions, texture materialization, instancing
