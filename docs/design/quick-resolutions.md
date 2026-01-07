# Quick Resolutions

Likely answers to simpler open questions. To be validated individually.

---

## External References

**Question**: How to serialize refs to textures/meshes? IDs? Inline graphs?

**Likely answer**: IDs with resolution context.

```rust
#[derive(Serialize, Deserialize)]
struct Displace {
    texture: AssetRef,  // "textures/noise.png" or uuid
    amount: f32,
}

// Resolution at evaluation time
impl MeshOp for Displace {
    fn apply(&self, mesh: &Mesh, ctx: &EvalContext) -> Mesh {
        let tex = ctx.resolve_texture(&self.texture)?;
        // ...
    }
}
```

Graphs reference external assets by ID/path. Context resolves them. Allows:
- Same graph, different assets
- Assets loaded from different sources (disk, network, generated)
- Clear error when asset missing

**Caveat**: Need to decide ID format (string path? UUID? both?).

---

## Bevy Integration

**Question**: Optional feature flags? Separate adapter crates?

**Likely answer**: Separate adapter crate(s), minimal feature flags in core.

```
resin-core       # no bevy dependency
resin-bevy       # From/Into impls, bevy asset loader, etc.
```

Pattern:
- `resin::Mesh` ↔ `bevy::Mesh` conversions
- `resin::Image` ↔ `bevy::Image` conversions
- Bevy asset loader for `.resin` graph files
- Maybe bevy systems for live graph evaluation

**Caveat**: How deep does integration go? Just type conversions, or full ECS integration?

---

## 3D Textures

**Question**: Volumetric noise for displacement, clouds. Same nodes with Vec3 input?

**Likely answer**: Yes, same nodes generalized over dimension.

```rust
// Generic over input dimension
trait NoiseGenerator<const D: usize> {
    fn sample(&self, point: [f32; D]) -> f32;
}

// Or simpler: 2D and 3D variants
impl Perlin {
    fn sample_2d(&self, uv: Vec2) -> f32;
    fn sample_3d(&self, pos: Vec3) -> f32;
    fn sample_4d(&self, pos: Vec4) -> f32;  // for looping animation
}
```

**Caveats**:
- 4D noise for seamlessly looping animated textures (time as 4th dimension)
- 3D-specific issues? (Can't think of any that aren't just "more dimensions")
- UVs are 2D texture thing, not relevant to 3D volumetric

---

## Tiling

**Question**: Automatic seamless tiling? Explicit tile operator?

**Likely answer**: Explicit operator, some generators tile naturally.

```rust
// Some noise already tiles
let noise = Perlin::new().tileable(true);  // generates tileable output

// Explicit tiling operator for non-tileable inputs
let tiled = texture.apply(Tile::new(2, 2));  // repeat 2x2

// Make any texture seamless (blend edges)
let seamless = texture.apply(MakeSeamless::new());
```

**Caveats**:
- "Make seamless" is lossy (blurs edges)
- Some ops break tileability (non-uniform transforms)
- Should track tileability as metadata? Probably overkill.

---

## Texture vs Field

**Question**: Unify texture sampling with mesh attribute evaluation?

**Likely answer**: Unified concept, different evaluation contexts.

Both are: `position → value`. Difference is where positions come from.

```rust
trait Field<In, Out> {
    fn sample(&self, pos: In) -> Out;
}

// Texture: sample at UV coordinates
impl Field<Vec2, Color> for Texture { ... }

// Mesh field: sample at vertex positions
impl Field<Vec3, f32> for NoiseField { ... }

// Same noise, different contexts:
let noise = Perlin::new();
let texture = noise.eval_grid(1024, 1024);           // materialize to image
let weights = mesh.sample_field(&noise);              // eval at vertices
```

**Caveats**:
- Textures often need materialization (neighbor access for blur, normal maps)
- Fields are naturally lazy
- Unifying might be overcomplication? Maybe just "both exist, interoperate"

---

## Animation / Time

**Question**: How do animated/time-dependent computations work?

**Initial answer**: Graph context provides time.

**Actual answer**: It's complicated. See [time-models.md](./time-models.md).

Multiple time models exist:
- **Stateless**: `f(inputs, time)` - can seek, parallelize (textures, synth, rigging)
- **Stateful**: depends on history - must process sequentially (filters, physics)
- **Streaming**: time = position in stream (audio)
- **Baked**: pre-computed stateful → stateless lookup (caches)

**Open questions**:
- How to represent statefulness in type system (or not)?
- Seeking stateful graphs?
- State serialization for save/restore?
- Audio block boundaries vs graph model?

---

## Summary

| Question | Likely Answer | Confidence |
|----------|---------------|------------|
| External refs | IDs + resolution context | High |
| Bevy integration | Separate adapter crate | High |
| 3D textures | Same nodes, Vec3/Vec4 input | High |
| Tiling | Explicit ops, some natural tiling | Medium |
| Texture vs field | Unified concept, maybe overkill | Medium |
| Time models | Complicated - multiple models coexist | Low |
