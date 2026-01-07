# Open Questions

Collected from domain design docs. To be resolved through design discussion and prototyping.

## Core Architecture

1. **Type system for slots**: Schema-based with generics (like maki) or simpler interned strings?
2. **Unified graph container?**: Typed slots may enable mixed-domain graphs
3. **Portable workflows**: Should graphs be serializable/shareable artifacts?
4. **Fields/expressions**: Blender's field system is powerful. How much do we want? Shared across mesh, texture, rigging?
5. **Parameter system**: First-class across all domains? (Live2D model is elegant, audio already does modulation)

## Meshes

1. **Half-edge vs index-based**: Half-edge is better for topology ops but more memory. Index-based is GPU-friendly. Support both? Convert at boundaries?
2. **Instancing**: How to represent "100 copies with different transforms"?
3. **SDF integration**: Separate representation or unify with mesh ops?

## Audio

1. **Sample rate**: Fixed at graph creation or runtime-configurable?
2. **Block size**: Fixed or variable? Trade-off: efficiency vs latency.
3. **Modulation depth**: Every param modulatable? Or explicit mod inputs? (VCV: everything is a cable. Pd: explicit inlets)
4. **Polyphony model**: Per-node (VCV poly cables)? Per-graph (Pd [clone])? Explicit voice management?
5. **Control vs audio rate**: Automatic promotion? Explicit types? (SuperCollider: `.kr` vs `.ar`)
6. **State management**: Filters have state. How does this fit pure graph model?

## Textures

1. **GPU vs CPU**: Abstract over both?
2. **Tiling**: Automatic seamless tiling? Explicit tile operator? Some noises naturally tile.
3. **Resolution**: When to materialize vs keep lazy? Blur/normal-from-height need neighbors.
4. **3D textures**: Volumetric noise for displacement, clouds. Same nodes with Vec3 input?
5. **Texture vs field**: Unify texture sampling with mesh attribute evaluation?

## Vector 2D

1. **Curve types**: Only cubic Bézier? Or also quadratic, arcs, NURBS? (SVG has all, cubic most common)
2. **Precision**: f32 or f64? (CAD uses f64, games use f32)
3. **Winding rule**: Even-odd vs non-zero? Both?
4. **Vector networks**: Figma allows branches at points. Much more complex than paths. Worth it?
5. **Text**: Text outlines are paths, but layout is complex. Include or exclude?
6. **Animation/morphing**: How does path animation relate to rigging?

## Rigging

1. **Unified 2D/3D rig**: Can they share abstractions? Bones, constraints, skinning work in both.
2. **Deformer stacking**: Order matters. List or graph?
3. **Animation blending**: Blend trees, state machines, layers. Include or separate crate?
4. **Procedural rigging**: Auto-rig from mesh topology?
5. **Real-time vs offline**: Different constraints. Games need <16ms.

## Cross-Cutting

1. **Modularity**: How granular should crates be? (Bevy philosophy: very modular)
2. **Bevy integration**: Optional feature flags? Separate adapter crates?
3. **Evaluation strategies**: Lazy vs eager? Pull vs push?
4. **Ops as values**: Should operations be serializable structs? See [design/ops-as-values](./design/ops-as-values.md)
5. **Expression language**: For serializable transforms (replaces closures). How powerful?
6. **Plugin op registration**: How do plugins register new serializable operations?

---

## Resolution Status

| Question | Status | Decision |
|----------|--------|----------|
| GPU vs CPU | ✅ Resolved | Abstract over both via burn/CubeCL. See [prior-art](./prior-art.md#burn--cubecl) |
| Precision f32/f64 | ✅ Resolved | Support both via generic `T: Float` |
| Winding rule | ✅ Resolved | Both, default non-zero. See [design/winding-rules](./design/winding-rules.md) |
| Curve types | ✅ Resolved | All via traits. See [design/curve-types](./design/curve-types.md) |
| Unified 2D/3D rig | 🔶 Leaning | Yes, some abstractions shareable (bones, constraints, skinning) |
| Parameter system | 🔶 Leaning | Yes, first-class across all domains |
| Deformer stacking | ✅ Resolved | List first, graph later if needed. See [design/deformer-stacking](./design/deformer-stacking.md) |
| Animation blending | 🔶 Leaning | Separate crate, bevy-style modularity |
| Type system for slots | 🔶 Leaning | Simpler than maki (types can be opaque). Generics maybe unnecessary for resin's use case |
| Modularity | 🔶 Leaning | Very modular, bevy philosophy |
| Vector networks | ✅ Resolved | Network internally, both APIs as equals. See [design/vector-networks](./design/vector-networks.md) |
| Text | 🔶 Leaning | Include outline extraction (`font.glyph_outline('A') -> Path`), exclude layout (harfbuzz territory) |
| Expression language | 🔶 Leaning | Cranelift JIT for runtime, WGSL for GPU textures. See [design/closure-usage-survey](./design/closure-usage-survey.md) |
