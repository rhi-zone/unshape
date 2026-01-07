# Closure Usage Survey

Where would users want to pass custom functions/closures across resin domains?

Goal: understand the problem space before designing an expression language.

## Mesh

| Operation | Closure signature | Example use case |
|-----------|-------------------|------------------|
| `map_vertices` | `Fn(Vec3) -> Vec3` | Custom displacement, warping |
| `filter_vertices` | `Fn(Vec3) -> bool` | Select vertices by position |
| `filter_faces` | `Fn(&Face) -> bool` | Select faces by normal, area |
| `vertex_color_from` | `Fn(Vec3) -> Color` | Procedural vertex coloring |
| `custom_subdivision` | `Fn(&Face) -> Vec<Face>` | Non-standard subdivision |

**Common patterns:**
- Position → Position (transforms)
- Position → Scalar (selection, weighting)
- Position → Color (procedural coloring)

## Textures

| Operation | Closure signature | Example use case |
|-----------|-------------------|------------------|
| `map_pixels` | `Fn(Color) -> Color` | Color grading, adjustments |
| `sample_custom` | `Fn(Vec2) -> Color` | Entirely custom texture |
| `warp` | `Fn(Vec2) -> Vec2` | UV distortion |
| `blend_custom` | `Fn(Color, Color) -> Color` | Custom blend mode |
| `threshold_custom` | `Fn(f32) -> f32` | Custom transfer function |

**Common patterns:**
- Color → Color (adjustments)
- UV → UV (warping)
- UV → Color (sampling)
- Scalar → Scalar (curves, transfer functions)

## Audio

| Operation | Closure signature | Example use case |
|-----------|-------------------|------------------|
| `map_samples` | `Fn(f32) -> f32` | Waveshaping, distortion |
| `custom_oscillator` | `Fn(f32) -> f32` | Phase → amplitude |
| `custom_envelope` | `Fn(f32) -> f32` | Time → amplitude |
| `custom_filter` | `Fn(&[f32]) -> f32` | FIR filter with custom kernel |

**Common patterns:**
- Sample → Sample (waveshaping)
- Phase → Sample (oscillators)
- Time → Scalar (envelopes, LFOs)

## Vector 2D

| Operation | Closure signature | Example use case |
|-----------|-------------------|------------------|
| `map_points` | `Fn(Vec2) -> Vec2` | Custom warping |
| `filter_points` | `Fn(Vec2) -> bool` | Select points |
| `vary_stroke` | `Fn(f32) -> f32` | t along path → stroke width |
| `vary_color` | `Fn(f32) -> Color` | t along path → color |

**Common patterns:**
- Position → Position
- t (0-1) → Scalar (varying properties along path)

## Rigging

| Operation | Closure signature | Example use case |
|-----------|-------------------|------------------|
| `custom_constraint` | `Fn(&Pose) -> Transform` | Procedural bone positioning |
| `driver` | `Fn(f32) -> f32` | Parameter → parameter mapping |
| `blend_custom` | `Fn(Pose, Pose, f32) -> Pose` | Custom pose interpolation |
| `physics_force` | `Fn(Vec3, Vec3) -> Vec3` | Position, velocity → force |

**Common patterns:**
- Scalar → Scalar (drivers)
- Pose → Pose (constraints)

---

## Summary: Common Closure Signatures

| Signature | Domains | Frequency |
|-----------|---------|-----------|
| `Fn(Vec3) -> Vec3` | Mesh, Rigging | High |
| `Fn(Vec2) -> Vec2` | Texture, Vector | High |
| `Fn(f32) -> f32` | All | Very High |
| `Fn(Color) -> Color` | Texture | Medium |
| `Fn(Vec3) -> f32` | Mesh (selection) | Medium |
| `Fn(f32) -> Color` | Texture, Vector | Medium |
| `Fn(Vec3) -> Color` | Mesh | Low |
| `Fn(Vec3) -> bool` | Mesh | Medium |
| Complex (multiple inputs) | Various | Low |

## What Could Be Named Ops Instead?

Many "closures" are actually common operations:

| Closure pattern | Named op equivalent |
|----------------|---------------------|
| `\|v\| v * 2.0` | `Scale::uniform(2.0)` |
| `\|v\| v + offset` | `Translate::new(offset)` |
| `\|v\| rotate(v, angle)` | `Rotate::new(axis, angle)` |
| `\|c\| c.brighten(0.1)` | `Brightness::new(0.1)` |
| `\|x\| x.clamp(0.0, 1.0)` | `Clamp::new(0.0, 1.0)` |
| `\|x\| x.powf(2.2)` | `Gamma::new(2.2)` |
| `\|uv\| uv * 2.0` | `TileUV::new(2.0)` |

## What Genuinely Needs Expressions?

Cases where named ops aren't enough:

1. **Composite math**: `|v| v * noise(v * 4.0) + vec3(0, v.y * 0.5, 0)`
2. **Conditionals**: `|v| if v.y > 0.0 { v * 2.0 } else { v }`
3. **Domain-specific formulas**: `|t| sin(t * TAU * 3.0) * exp(-t * 2.0)` (custom envelope)
4. **Artistic curves**: arbitrary remapping functions

## Questions to Answer

1. **How common are "genuinely needs expression" cases?**
   - If rare: named ops + escape hatch might suffice
   - If common: need expression language

2. **What's the expression language complexity?**
   - Just math (+-*/, sin, cos, pow, etc.)?
   - Conditionals (if/else)?
   - Variables/bindings?
   - Loops?
   - Function definitions?

3. **Per-domain or unified?**
   - Same expression language everywhere?
   - Or domain-specific (MeshExpr, AudioExpr, etc.)?

4. **Prior art expression languages?**
   - Blender drivers (Python subset)
   - Houdini VEX
   - Shadertoy GLSL
   - Max/MSP expr object
   - Desmos (math only)

---

## Use Case: Generative Art

For generative art, **arbitrary code IS the point**. The artist's custom formula is the art.

Examples:
- Shadertoy: entire shader is custom GLSL
- Processing: `draw()` is arbitrary code
- Nannou: Rust closures everywhere
- Context Free Art: custom rules

**Implication:** Resin can't just offer "named ops" for this use case. Need real expressiveness.

**Use case spectrum:**

| Use case | Needs expressions? | Serialization? | Example |
|----------|-------------------|----------------|---------|
| Conventional modeling | No - named ops | Nice to have | "Make a box, bevel edges" |
| Asset pipeline | Minimal | Required | Game assets, VFX |
| Procedural content | Some | Required | Dungeon generator |
| Generative art | Full | Less critical | Shadertoy, demos |

**Implication:** Support all tiers. Named ops for simple cases, full code for artists.

Or: the graph is the serializable part, nodes can contain arbitrary code for artists.

```rust
// Serializable graph structure
let graph = TextureGraph::new()
    .add(Noise::perlin(4.0))
    .add(CustomShader::from_wgsl("..."))  // inline WGSL string
    .add(Blend::multiply());

// The WGSL string IS the serialized form of the custom part
```

## Alternative: Pure Composition

Can we avoid expressions entirely via op composition?

```rust
// Expression
|v| v * noise(v * 4.0) + vec3(0, v.y * 0.5, 0)

// Composition (verbose but serializable)
Compose::new([
    Mul::new(Position, Noise::new(Scale::new(Position, 4.0))),
    Add::new(Vec3::new(0.0, Mul::new(GetY::new(Position), 0.5), 0.0)),
])
```

**Question: when does composition fail?**
- Recursive definitions?
- State/accumulation?
- Complex control flow?

If composition always works → no expression language needed, just a rich op library.

## Performance Spectrum

| Approach | Performance | Use case | Serializable |
|----------|-------------|----------|--------------|
| Native Rust closure | Best | Compile-time known | No |
| Cranelift JIT | Near-native | Runtime hot paths | Yes |
| WGSL (GPU) | Best for parallel | Textures, per-pixel | Yes |
| Interpreted | Slow | One-shot, debugging | Yes |
| Op composition | Good | Simple transforms | Yes |

**Removed from consideration:**
- LuaJIT: Cranelift gives native speed without another language
- Generated Rust (build.rs): Build-time codegen is complex, Cranelift simpler

## Key Insight: Static vs Dynamic

Dynamic expressions only needed when constructed at runtime:

```rust
// STATIC (compile-time): just use Rust closures
mesh.map_vertices(|v| v * 2.0)  // Zero overhead, monomorphized

// DYNAMIC (runtime): need interpreter or JIT
let expr = load_expr_from_file("transform.expr")?;
mesh.map_vertices_dyn(&expr)
```

**Layered approach:**

| Source | Backend | When to use |
|--------|---------|-------------|
| Rust source | Native closure | Compile-time known |
| Serialized graph | Cranelift JIT | Runtime hot paths |
| Debugging/one-shot | Interpreted | When compile cost > benefit |
| Textures | WGSL | GPU parallel ops |

This means most users never need dynamic expressions - they write Rust. Only graph deserialization / live coding needs the dynamic path.

**But runtime should still be fast.** When expressions ARE constructed at runtime, we should still achieve native performance via JIT compilation. This is what Cranelift provides.

## Cranelift JIT for Runtime Expressions

[Cranelift](https://github.com/bytecodealliance/wasmtime/tree/main/cranelift) is a code generator designed for JIT use cases. It's what Wasmtime uses internally.

**Why Cranelift:**
- Compiles to native code (x86-64, AArch64)
- Fast compilation (~10ms for small functions)
- No external dependencies (pure Rust)
- Designed for embedding

**How it works:**

```rust
use cranelift::prelude::*;
use cranelift_jit::{JITBuilder, JITModule};

// 1. Define expression AST
enum Expr {
    Var(usize),           // input variable by index
    Const(f32),
    Add(Box<Expr>, Box<Expr>),
    Mul(Box<Expr>, Box<Expr>),
    Sin(Box<Expr>),
    // ...
}

// 2. Compile to Cranelift IR
fn compile_expr(expr: &Expr, builder: &mut FunctionBuilder, vars: &[Value]) -> Value {
    match expr {
        Expr::Var(i) => vars[*i],
        Expr::Const(c) => builder.ins().f32const(*c),
        Expr::Add(a, b) => {
            let a = compile_expr(a, builder, vars);
            let b = compile_expr(b, builder, vars);
            builder.ins().fadd(a, b)
        }
        Expr::Sin(x) => {
            let x = compile_expr(x, builder, vars);
            // Call libm sin
            builder.ins().call(sin_func, &[x])
        }
        // ...
    }
}

// 3. Get native function pointer
let code_ptr = module.get_finalized_function(func_id);
let f: fn(f32, f32, f32) -> f32 = unsafe { std::mem::transmute(code_ptr) };

// 4. Call at native speed
let result = f(x, y, z);
```

**Compilation cost:**
- ~1-10ms per function (depending on complexity)
- Amortized over many calls
- For mesh with 100k vertices, compile once, call 100k times

**When to JIT vs interpret:**

| Scenario | Strategy |
|----------|----------|
| Expression used once | Interpret (compile cost > benefit) |
| Expression used in hot loop | JIT compile |
| Expression used across frames | JIT compile, cache |
| Texture shader (millions of pixels) | WGSL (GPU) |

**Caching compiled expressions:**

```rust
struct ExprCache {
    compiled: HashMap<ExprHash, CompiledFn>,
}

impl ExprCache {
    fn get_or_compile(&mut self, expr: &Expr) -> &CompiledFn {
        let hash = expr.hash();
        self.compiled.entry(hash).or_insert_with(|| {
            jit_compile(expr)
        })
    }
}
```

**Compile-time option (zero overhead):**

For users who know their expressions at compile time, provide a proc macro:

```rust
// Compile-time: expands to native Rust closure
let f = resin_expr!(|v: Vec3| v * 2.0 + vec3(0.0, v.y, 0.0));

// Runtime: JIT compiled
let f = Expr::parse("v * 2.0 + vec3(0, v.y, 0)")?.compile()?;

// Both have same performance when called
```

**Domain-specific considerations:**

| Domain | Volume | Backend |
|--------|--------|---------|
| Textures | Millions of pixels | WGSL (GPU) |
| Mesh | Thousands of vertices | Cranelift |
| Audio | ~86 blocks/sec (512 samples) | Cranelift |
| Rigging | Per-frame | Cranelift |
| Vector | Thousands of points | Cranelift |

All hot paths get native performance via Cranelift. GPU only for massively parallel (textures).

## Backend Selection

Unified `Expr` type, automatic backend selection based on context:

```rust
impl Expr {
    /// Compile for best available backend
    fn compile(&self) -> CompiledExpr {
        // Auto-select based on expression characteristics
    }

    /// Force specific backend
    fn compile_with(&self, backend: Backend) -> CompiledExpr;
}

enum Backend {
    Cranelift,     // Native JIT - default for hot paths
    Wgsl,          // GPU - for parallel pixel/vertex ops
    Interpreted,   // Fallback - works everywhere
}
```

**Automatic selection heuristics:**

| Context | Default Backend | Reason |
|---------|-----------------|--------|
| `mesh.map_vertices_dyn(expr)` | Cranelift | Called per-vertex, benefits from native |
| `texture.eval_dyn(expr)` | WGSL | Massively parallel, GPU wins |
| `audio.process_block(expr)` | Cranelift | Real-time, needs predictable latency |
| `rig.driver(expr)` | Cranelift | Per-frame, needs speed |
| One-shot evaluation | Interpreted | Compile cost not worth it |

**Same Expr, different backends:**

```rust
let expr = Expr::parse("position * 2.0 + noise(position * 4.0)")?;

// Same expression, different compilation targets
let cpu_fn = expr.compile_with(Backend::Cranelift)?;
let gpu_shader = expr.compile_with(Backend::Wgsl)?;

// Use CPU version for mesh
mesh.map_vertices_dyn(&cpu_fn);

// Use GPU version for texture
texture.eval_dyn(&gpu_shader);
```

## Pure Data Model

Pd uses a minimal expression language inside `[expr]` object:

```
[expr $f1 * sin($f2 * 6.28)]
```

- `$f1`, `$f2` = float inlets
- Basic math ops: + - * / sin cos pow etc.
- No variables, no loops, no conditionals
- Everything else: use objects and patch cords

**Key insight:** Pd keeps expressions minimal. Complex logic = more objects, not bigger expressions.

Could resin do the same? Expressions only for math, composition for logic.

## Notes

(Space for investigation notes as we explore each domain)

### Mesh expressions
TODO: What can't be done with named ops?

### Texture expressions
TODO: Shadertoy patterns, what needs custom code?

### Audio expressions
TODO: Look at Pure Data, SuperCollider, FAUST

**TidalCycles / Strudel**

[TidalCycles](https://tidalcycles.org/) (Haskell) / [Strudel](https://strudel.cc/) (JS port)

Pattern-based live coding for music:

```javascript
// Strudel example
s("bd sd [~ bd] sd").speed("1 2 1.5")
```

- **Mini-notation**: `"bd sd [~ bd] sd"` = kick, snare, [rest, kick], snare
- **Pattern transformations**: `fast()`, `slow()`, `rev()`, `jux()`
- **Composable**: patterns are values, combine with operators
- **Time-aware**: patterns are functions of time

Key insight: **patterns as composable values**, not the mini-notation syntax. The notation is TidalCycles-specific; resin would use Rust API instead.

**Relevance to resin:**
- Patterns are values with composable transformations
- `fast()`, `slow()`, `rev()` = ops
- Same model we're already planning

### Vector expressions
TODO: SVG filters? Path effects?

### Rigging expressions
TODO: Blender drivers, Maya expressions
