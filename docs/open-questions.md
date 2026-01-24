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
| Type system for slots | ✅ Resolved | Simpler than maki (opaque types). Generics unnecessary given typed slots. |
| Unified graph container | ✅ Resolved | Yes. Single `Graph` type, typed slots (`Output<T>`/`Input<T>`), compile-time safety for Rust, runtime TypeId validation for loaded graphs. Value enum at execution. |
| Portable workflows | ✅ Resolved | Yes. JSON (human-readable) + optional binary. Versioned files, ops declare compatibility. External refs via IDs, optional asset embedding for full portability. |
| Parameter system | ✅ Resolved | Yes, first-class across all domains |
| Modularity | ✅ Resolved | Very modular, bevy philosophy. Plugin crates for optional features. |
| Bevy integration | ✅ Resolved | Low priority. Separate adapter crate if needed. Must not affect core design - unshape is standalone first. |
| Evaluation strategy | ✅ Resolved | Evaluator trait. Lazy default, others as needed. See [evaluation-strategy](./design/evaluation-strategy.md) |
| Time models | ✅ Resolved | EvalContext for time, explicit baking, numeric rates + explicit conversion, block = audio iteration, hybrid nodes = feedback edges, determinism = best effort + strict mode. See [time-models](./design/time-models.md) |

## Expression Language

See [expression-language](./design/expression-language.md) for full design.

| Question | Status | Notes |
|----------|--------|-------|
| Backend selection | ✅ Resolved | Cranelift JIT (CPU hot paths), WGSL (GPU), Interpreted (fallback) |
| Expression AST scope | ✅ Resolved | Math + conditionals + let bindings. No loops (use graph recurrence) |
| Per-domain or unified | ✅ Resolved | Unified Expr, domains bind different variables (position, uv, time, etc.) |
| Built-in functions | ✅ Resolved | WGSL built-ins as reference set. Plugin functions decompose or backend traits |
| Value representation | ✅ Resolved | Enum (not dyn trait). Vectors/matrices feature-gated. Square matrices only |
| Matrix operations | ✅ Resolved | `*` works on matrices (WGSL-style). Type inference dispatches |
| Crate structure | ✅ Resolved | core + macros + parse + wgsl + cranelift. Interpreter in core |
| Expr -> WGSL codegen | ✅ Resolved | String generation from AST. Decomposition-first for plugins. |
| Expr -> Cranelift codegen | ✅ Resolved | IR generation, external calls for complex functions. |
| Plugin function API | ✅ Resolved | Core trait + backend extension traits in backend crates. |
| Constant folding | ✅ Resolved | AST -> AST transform (could be in dew or wrapper). |

## Ops & Serialization

| Question | Status | Notes |
|----------|--------|-------|
| Ops as values | ✅ Resolved | Yes, ops are serializable structs. Derive macro for DynOp impl. See [ops-as-values](./design/ops-as-values.md) |
| Plugin op registration | ✅ Resolved | Core defines trait + serialization contract. Plugin *loading* is host's responsibility. Optional adapters (rhi-unshape-wasm-plugins, etc.) for common cases. See [plugin-architecture](./design/plugin-architecture.md) |
| Graph evaluation caching | ✅ Resolved | Hash-based caching at node boundaries (salsa-style). Inputs unchanged -> return cached output |
| External references | ✅ Resolved | IDs + resolution context (ComfyUI pattern). Optional asset embedding for portability (any size). |

## Meshes

| Question | Status | Notes |
|----------|--------|-------|
| Half-edge vs index-based | ✅ Resolved | Half-edge internal, indexed on export. Accept memory cost for topology ops. See [mesh-representation](./design/mesh-representation.md) |
| Instancing | ✅ Resolved | Plugin crate (`rhi-unshape-instances`), not core. Provides `Instances` type + scatter ops + merge to `Mesh`. |
| SDF integration | ✅ Resolved | SDF is `Field<Vec3, f32>`. SDF ops are field combinators. Meshing via explicit operators (MarchingCubes, etc.). Optional `Sdf` marker trait for type-level guarantees. |
| Fields for selection | ✅ Resolved | Yes. `Field<VertexData, bool>` for inline/lazy selection, `SelectionSet` for materialized/named/manual picks. Same Field vs materialized pattern. |

## Audio

| Question | Status | Notes |
|----------|--------|-------|
| Sample rate | ✅ Resolved | Runtime from EvalContext. Nodes query `ctx.sample_rate`. Lazy buffer init (one-time cost per rate). Same graph works at any host rate. |
| Block size | ✅ Resolved | Host-controlled, variable. Nodes handle any size. Block size from EvalContext. Feedback edges update per-block. |
| Modulation depth | ✅ Resolved | Hybrid. Node author decides which params are modulatable via `#[modulatable]`. Explicit mod inputs, not implicit on everything. Graph resolves per-block (not per-sample), zero cost when unconnected. |
| Polyphony model | ✅ Resolved | Per-graph cloning (Pd pattern). Plugin crate (`rhi-unshape-poly`), not core. Voice allocator clones graph instances, mixes output. Matches instancing pattern. |
| Control vs audio rate | ✅ Resolved | No special types. Control rate = lower numeric rate. Explicit conversion nodes if needed (from mixed rates decision). |
| State management | ✅ Resolved | Recurrent graphs: feedback edges carry state, nodes stay pure. Delay granularity per-edge, mixed rates via explicit conversion. See [recurrent-graphs](./design/recurrent-graphs.md) |

## Textures

| Question | Status | Notes |
|----------|--------|-------|
| GPU vs CPU | ✅ Resolved | Abstract over both via burn/CubeCL. See [prior-art](./prior-art.md#burn--cubecl) |
| Tiling | ✅ Resolved | Explicit operators. Tiling isn't fundamental to most generators. `MakeSeamless`, `Tile`, etc. |
| Resolution/materialization | ✅ Resolved | Separate Field (lazy) / Image (materialized). Resolution explicit at render(). No propagation. See [texture-materialization](./design/texture-materialization.md) |
| 3D textures | ✅ Resolved | Same nodes, Vec3 input. Vec4 for looping animation (time as 4th dim). Memory/preview are host concerns. |
| Texture vs field | ✅ Resolved | Unified via generic `Field<I, O>` trait. Same concept, different input types. Time handling -> see Time models. |

## Vector 2D

| Question | Status | Notes |
|----------|--------|-------|
| Curve types | ✅ Resolved | All via traits. See [curve-types](./design/curve-types.md) |
| Precision f32/f64 | ✅ Resolved | Support both via generic `T: Float` |
| Winding rule | ✅ Resolved | Both, default non-zero. See [winding-rules](./design/winding-rules.md) |
| Vector networks | ✅ Resolved | Network internally, both APIs as equals. See [vector-networks](./design/vector-networks.md) |
| Text | ✅ Resolved | Include outline extraction, exclude layout (harfbuzz territory). |
| Path ↔ rigging | ✅ Resolved | Rigging (bones+skinning) unified 2D/3D. Morphing is separate `Morph<G>` trait. Paths can drive rig params (spline IK, curve deformers). |

## Rigging

| Question | Status | Notes |
|----------|--------|-------|
| Unified 2D/3D rig | ✅ Resolved | Yes. Generic `Rig<G: HasPositions>` trait. Same bones/skinning concepts, just different dimensionality. Live2D validates 2D case. |
| Deformer stacking | ✅ Resolved | Graph internal, Stack API. See [deformer-stacking](./design/deformer-stacking.md) |
| Animation blending | ✅ Resolved | Separate crate (`rhi-unshape-anim`), bevy-style modularity. |
| Procedural rigging | ✅ Resolved | Plugin crate (`rhi-unshape-autorig`). Core provides rig primitives, auto-generation is domain-specific. |
| Real-time vs offline | ✅ Resolved | Same API, user manages budget. Solver params (iterations, tolerance) for quality/speed tradeoff. Bone count is authoring choice. |

---

## Summary by Status

### ✅ Resolved (50)

**Core Architecture:**
- Type system for slots (simpler than maki, typed slots)
- Unified graph container (typed slots, Value enum execution)
- Portable workflows (JSON + binary, versioned, asset embedding)
- Parameter system (first-class across all domains)
- Modularity (very modular, plugin crates for optional features)
- Bevy integration (low priority, standalone first)
- Evaluation strategy (Evaluator trait, lazy default)
- Time models (EvalContext, explicit baking, numeric rates, feedback = state)

**Expression Language:**
- Backend selection (Cranelift/WGSL/Interpreted)
- AST scope (math + conditionals + let, no loops)
- Built-in functions (WGSL as reference, plugin extensions)
- Value enum (not dyn trait, vectors/matrices feature-gated)
- Matrix ops (`*` works on matrices, type inference dispatches)
- Crate structure (core + macros + parse + wgsl + cranelift)
- Unified vs per-domain (unified, domains bind different vars)
- WGSL codegen (string generation, decomposition-first)
- Cranelift codegen (IR generation, external calls)
- Plugin function API (core trait + backend extension traits)
- Constant folding (AST transform in dew or wrapper)

**Ops & Serialization:**
- Ops as values (derive macro for DynOp impl)
- Plugin op registration (core = contract, host = loading)
- Graph caching (hash-based at node boundaries)
- External references (IDs + context, asset embedding)

**Meshes:**
- Half-edge vs index-based (half-edge internal, indexed export)
- Instancing (plugin crate)
- SDF integration (Field<Vec3, f32>, meshing via explicit ops)
- Fields for selection (Field for lazy, SelectionSet for materialized)

**Audio:**
- Sample rate (runtime from EvalContext, lazy buffer init)
- Block size (host-controlled, variable)
- Modulation depth (hybrid, per-block resolution)
- Polyphony (per-graph cloning, plugin crate)
- Control vs audio rate (no special types, explicit conversion)
- State management (recurrent graphs, feedback edges)

**Textures:**
- GPU vs CPU (burn/CubeCL abstraction)
- Tiling (explicit operators)
- Resolution/materialization (Field/Image split, explicit resolution)
- 3D textures (same nodes, Vec3/Vec4 input)
- Texture vs field (Field<I,O> trait)

**Vector 2D:**
- Curve types (trait-based)
- Precision f32/f64 (generic `T: Float`)
- Winding rule (both, default non-zero)
- Vector networks (network internal, both APIs)
- Text (outline extraction, no layout)
- Path ↔ rigging (paths can drive rigs)

**Rigging:**
- Unified 2D/3D rig (generic Rig<G: HasPositions>)
- Deformer stacking (graph internal, stack API)
- Animation blending (separate crate)
- Procedural rigging (plugin crate)
- Real-time vs offline (same API, solver params)

### 🔶 Leaning (0)
All promoted to resolved!

### ❓ Open (0)
All questions resolved!
