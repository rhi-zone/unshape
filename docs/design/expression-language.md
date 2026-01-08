# Expression Language

Serializable expressions that replace closures for transforms.

## The Problem

Closures can't be serialized:

```rust
// This works but can't be saved/loaded
mesh.map_vertices(|v| v * 2.0 + offset)

// What would the serialized form look like?
struct MapVertices {
    f: ???  // Can't serialize a closure
}
```

## Solution: Expression AST

Expressions are data structures that represent computations:

```rust
// Instead of closure:
mesh.map_vertices(|v| v * 2.0)

// Expression:
mesh.map_vertices(Expr::Mul(Expr::Var(Position), Expr::Const(2.0)))

// With builder sugar:
mesh.map_vertices(var::position() * 2.0)
```

## AST Design

### Core Types

```rust
/// Scalar or vector value
#[derive(Clone, Serialize, Deserialize)]
pub enum Scalar {
    F32(f32),
    F64(f64),
    I32(i32),
    Bool(bool),
}

/// Expression AST node
#[derive(Clone, Serialize, Deserialize)]
pub enum Expr {
    // === Literals ===
    Const(Scalar),
    Vec2([Box<Expr>; 2]),
    Vec3([Box<Expr>; 3]),
    Vec4([Box<Expr>; 4]),

    // === Variables (bound by evaluation context) ===
    Var(VarId),

    // === Arithmetic (scalar and vector) ===
    Add(Box<Expr>, Box<Expr>),
    Sub(Box<Expr>, Box<Expr>),
    Mul(Box<Expr>, Box<Expr>),
    Div(Box<Expr>, Box<Expr>),
    Neg(Box<Expr>),
    Mod(Box<Expr>, Box<Expr>),

    // === Comparison ===
    Eq(Box<Expr>, Box<Expr>),
    Ne(Box<Expr>, Box<Expr>),
    Lt(Box<Expr>, Box<Expr>),
    Le(Box<Expr>, Box<Expr>),
    Gt(Box<Expr>, Box<Expr>),
    Ge(Box<Expr>, Box<Expr>),

    // === Boolean ===
    And(Box<Expr>, Box<Expr>),
    Or(Box<Expr>, Box<Expr>),
    Not(Box<Expr>),

    // === Control flow ===
    If(Box<Expr>, Box<Expr>, Box<Expr>),  // cond, then, else

    // === Let binding ===
    Let(VarId, Box<Expr>, Box<Expr>),  // let var = value in body

    // === Built-in functions ===
    Builtin(BuiltinFn, Vec<Expr>),

    // === Plugin functions (resolved by name at eval time) ===
    Plugin(String, Vec<Expr>),

    // === Vector ops ===
    Swizzle(Box<Expr>, Swizzle),  // v.xy, v.zyx, etc.
    Index(Box<Expr>, usize),      // v[0]
}

/// Swizzle pattern (up to 4 components)
#[derive(Clone, Serialize, Deserialize)]
pub struct Swizzle(pub Vec<Component>);

#[derive(Clone, Copy, Serialize, Deserialize)]
pub enum Component { X, Y, Z, W }
```

### Built-in Functions

Primitives that all backends must implement. Based on WGSL built-ins for concrete spec:

```rust
#[derive(Clone, Copy, Serialize, Deserialize)]
pub enum BuiltinFn {
    // Trigonometric
    Sin, Cos, Tan,
    Asin, Acos, Atan, Atan2,
    Sinh, Cosh, Tanh,

    // Exponential
    Exp, Exp2, Log, Log2,
    Pow, Sqrt, InverseSqrt,

    // Common
    Abs, Sign,
    Floor, Ceil, Round, Trunc, Fract,
    Min, Max, Clamp,
    Mix,      // lerp(a, b, t)
    Step,     // step(edge, x)
    Smoothstep,

    // Geometric (vector)
    Length, Distance,
    Dot, Cross,
    Normalize,
    Reflect, Refract,
    FaceForward,

    // Component-wise
    Saturate,  // clamp(x, 0, 1)

    // Derivative (GPU only, returns 0 on CPU)
    Dfdx, Dfdy, Fwidth,
}
```

**Why WGSL as reference:**
- Concrete spec with defined behavior
- Cranelift/CPU can match WGSL semantics
- No ambiguity about edge cases

## Function Tiers

### Tier 1: Primitives (BuiltinFn)

All backends implement these directly. Finite, stable set.

### Tier 2: Standard Library

Functions that decompose to primitives. No backend-specific code needed:

```rust
/// Standard library functions (decompose to primitives)
pub enum StdFn {
    Remap,      // remap(x, in_lo, in_hi, out_lo, out_hi)
    InverseLerp, // inverse_lerp(a, b, v) = (v - a) / (b - a)
    SmoothMin,  // smooth_min(a, b, k)
    // ... etc
}

impl StdFn {
    /// Express this function using only primitives
    pub fn decompose(&self, args: &[Expr]) -> Expr {
        match self {
            StdFn::Remap => {
                // remap(x, in_lo, in_hi, out_lo, out_hi) =
                // mix(out_lo, out_hi, (x - in_lo) / (in_hi - in_lo))
                let [x, in_lo, in_hi, out_lo, out_hi] = args else { panic!() };
                let t = (x.clone() - in_lo.clone()) / (in_hi.clone() - in_lo.clone());
                Expr::Builtin(BuiltinFn::Mix, vec![out_lo.clone(), out_hi.clone(), t])
            }
            // ...
        }
    }
}
```

### Tier 3: Plugin Functions

Functions provided by plugins.

**Core trait** (all plugins implement):

```rust
/// Core plugin function - interpret + optional decompose
pub trait PluginFn: Send + Sync {
    /// Unique function name (e.g., "resin_noise::perlin")
    fn name(&self) -> &str;

    /// Function signature for type checking
    fn signature(&self) -> FnSignature;

    /// Express as simpler expressions (works with all backends automatically)
    fn decompose(&self, args: &[Expr]) -> Option<Expr> { None }

    /// CPU interpretation (required - fallback for all backends)
    fn interpret(&self, args: &[Value]) -> Result<Value>;
}
```

**Backend extension traits** (optional, defined by backend crates):

```rust
// In resin-expr-wgsl crate
pub trait WgslPluginFn: PluginFn {
    /// Generate WGSL code for this function call
    fn compile_wgsl(&self, args: &[&str]) -> String;
}

// In resin-expr-cranelift crate
pub trait CraneliftPluginFn: PluginFn {
    /// Generate Cranelift IR for this function call
    fn compile_cranelift(&self, builder: &mut FunctionBuilder, args: &[Value]) -> Value;
}
```

**Registry structure:**

```rust
pub struct FunctionRegistry {
    /// Core functions (interpret + decompose)
    core: HashMap<String, Arc<dyn PluginFn>>,
}

impl FunctionRegistry {
    pub fn register<F: PluginFn + 'static>(&mut self, func: F) {
        self.core.insert(func.name().to_string(), Arc::new(func));
    }

    pub fn get(&self, name: &str) -> Option<&Arc<dyn PluginFn>> {
        self.core.get(name)
    }
}

// Backend crates extend with their own registries
// In resin-expr-wgsl:
pub struct WgslFunctionRegistry {
    funcs: HashMap<String, Arc<dyn WgslPluginFn>>,
}
```

**Backend compilation flow:**

```rust
// WGSL compiler
impl WgslCompiler {
    fn compile_plugin(&self, name: &str, args: &[Expr]) -> Result<String> {
        let func = self.core_registry.get(name)
            .ok_or(UnknownFunction(name))?;

        // 1. Try decomposition (works for all backends)
        if let Some(decomposed) = func.decompose(args) {
            return self.compile(&decomposed);
        }

        // 2. Try WGSL-specific implementation
        if let Some(wgsl_fn) = self.wgsl_registry.get(name) {
            let arg_strs: Vec<_> = args.iter()
                .map(|a| self.compile(a))
                .collect::<Result<_>>()?;
            return Ok(wgsl_fn.compile_wgsl(&arg_strs));
        }

        // 3. No WGSL impl - error (or fall back to interpret + readback?)
        Err(NoBackendImpl { func: name, backend: "wgsl" })
    }
}
```

**Why this design:**

| Concern | Solution |
|---------|----------|
| Core doesn't know about backends | Backend traits in separate crates |
| Plugins don't need all backends | decompose() works everywhere |
| Complex functions (noise) | Implement backend extension traits |
| String → function lookup | Registry per scope |

**Example: Simple function (decomposition only)**

```rust
struct InverseLerp;

impl PluginFn for InverseLerp {
    fn name(&self) -> &str { "inverse_lerp" }

    fn signature(&self) -> FnSignature {
        FnSignature::new(&[Type::F32; 3], Type::F32)
    }

    fn decompose(&self, args: &[Expr]) -> Option<Expr> {
        // inverse_lerp(a, b, v) = (v - a) / (b - a)
        let [a, b, v] = args else { return None };
        Some((v.clone() - a.clone()) / (b.clone() - a.clone()))
    }

    fn interpret(&self, args: &[Value]) -> Result<Value> {
        // Fallback if decomposition somehow fails
        let [a, b, v] = args else { return Err(ArgCount) };
        Ok(Value::F32((v.as_f32()? - a.as_f32()?) / (b.as_f32()? - a.as_f32()?)))
    }
}
// No WgslPluginFn or CraneliftPluginFn needed - decomposition works everywhere
```

**Example: Complex function (backend-specific impls)**

```rust
struct PerlinNoise {
    octaves: u32,
    lacunarity: f32,
    persistence: f32,
}

// Core trait - required
impl PluginFn for PerlinNoise {
    fn name(&self) -> &str { "perlin_noise" }

    fn signature(&self) -> FnSignature {
        FnSignature::new(&[Type::Vec3], Type::F32)
    }

    fn decompose(&self, _: &[Expr]) -> Option<Expr> {
        None  // Can't express Perlin as primitives
    }

    fn interpret(&self, args: &[Value]) -> Result<Value> {
        let pos = args[0].as_vec3()?;
        Ok(Value::F32(self.sample_cpu(pos)))
    }
}

// WGSL extension - optional, in resin-expr-wgsl
impl WgslPluginFn for PerlinNoise {
    fn compile_wgsl(&self, args: &[&str]) -> String {
        format!("perlin_fbm_{}({}, {}, {})",
            self.octaves, args[0], self.lacunarity, self.persistence)
    }
}

// Cranelift extension - optional, in resin-expr-cranelift
impl CraneliftPluginFn for PerlinNoise {
    fn compile_cranelift(&self, builder: &mut FunctionBuilder, args: &[Value]) -> Value {
        // Call external C function or inline IR
        builder.call_extern("resin_perlin_fbm", args)
    }
}
```

## Type System

Expressions are typed. Type checking happens before compilation.

```rust
#[derive(Clone, Copy, PartialEq)]
pub enum Type {
    F32, F64, I32, Bool,
    Vec2, Vec3, Vec4,
    BVec2, BVec3, BVec4,  // boolean vectors
    Mat2, Mat3, Mat4,
}

/// Infer type of expression given variable types
pub fn infer_type(expr: &Expr, vars: &HashMap<VarId, Type>) -> Result<Type, TypeError> {
    match expr {
        Expr::Const(Scalar::F32(_)) => Ok(Type::F32),
        Expr::Var(id) => vars.get(id).copied().ok_or(TypeError::UnboundVar(*id)),
        Expr::Add(a, b) => {
            let ta = infer_type(a, vars)?;
            let tb = infer_type(b, vars)?;
            // Same types, or scalar broadcast
            match (ta, tb) {
                (t, t2) if t == t2 => Ok(t),
                (Type::F32, Type::Vec3) | (Type::Vec3, Type::F32) => Ok(Type::Vec3),
                // ... etc
                _ => Err(TypeError::BinaryMismatch(ta, tb)),
            }
        }
        // ...
    }
}
```

## Evaluation Context

Expressions evaluate in a context that binds variables:

```rust
pub struct EvalContext {
    /// Variable bindings
    pub vars: HashMap<VarId, Value>,

    /// Plugin function registry
    pub functions: FunctionRegistry,
}

/// Well-known variable IDs
pub mod var {
    pub const POSITION: VarId = VarId(0);
    pub const NORMAL: VarId = VarId(1);
    pub const UV: VarId = VarId(2);
    pub const TIME: VarId = VarId(3);
    pub const COLOR: VarId = VarId(4);
    // ... etc
}
```

Different domains bind different variables:

| Domain | Available variables |
|--------|---------------------|
| Mesh vertex | position, normal, uv, color, index |
| Texture field | uv, time |
| Audio | time, sample_rate, phase |

## Expression Construction

Three ways to build expressions:

### 1. Builder Functions (always available)

Operator overloading + named constructors:

```rust
use resin_expr::prelude::*;

// Builds Expr AST via operators
let e = (position() + 1.0) * sin(time());

// Implementation
impl Add<f32> for Expr {
    type Output = Expr;
    fn add(self, rhs: f32) -> Expr {
        Expr::Add(Box::new(self), Box::new(Expr::Const(Scalar::F32(rhs))))
    }
}

pub fn position() -> Expr { Expr::Var(var::POSITION) }
pub fn sin(x: impl Into<Expr>) -> Expr {
    Expr::Builtin(BuiltinFn::Sin, vec![x.into()])
}
```

**Pros:** Type-safe, IDE autocomplete, refactoring works
**Cons:** Slightly verbose

### 2. Proc Macro (compile-time parsing)

```rust
use resin_expr::expr;

// Parsed at compile time, produces Expr AST
let e = expr!(sin(position * 2.0) + time);

// Let bindings
let e = expr! {
    let scaled = position * 2.0;
    sin(scaled) + cos(scaled)
};

// Conditionals
let e = expr!(if uv.x > 0.5 { 1.0 } else { 0.0 });
```

**Pros:** Readable, familiar syntax
**Cons:** Proc macro compile time, custom syntax to learn

**Implementation sketch:**

```rust
// resin-expr-macros crate
#[proc_macro]
pub fn expr(input: TokenStream) -> TokenStream {
    let parsed = parse_expr_syntax(input);
    let ast = to_expr_ast(parsed);
    quote! { #ast }
}
```

### 3. Runtime Parser (for loaded files, user input)

```rust
use resin_expr::parse;

// Parsed at runtime
let source = "sin(position * 2.0) + time";
let e: Expr = parse::expr(source)?;

// With custom variables
let e = parse::expr_with_vars(source, &["position", "time"])?;
```

**Pros:** Dynamic, works with loaded pipelines, user-editable
**Cons:** Runtime errors, parsing overhead

**When to use each:**

| Method | Use case |
|--------|----------|
| Builder functions | Library code, performance-critical |
| Proc macro | Application code, readability matters |
| Runtime parser | Config files, UI input, loaded pipelines |

### Serialization Format

Expressions serialize to JSON/MessagePack via serde:

```json
{
  "Add": [
    { "Builtin": ["Sin", [{ "Var": 0 }]] },
    { "Var": 3 }
  ]
}
```

For human-readable configs, store the source string and parse at load:

```json
{
  "vertex_offset": "sin(position.x * 10.0) * 0.1"
}
```

Parse on deserialize:

```rust
#[derive(Deserialize)]
struct Config {
    #[serde(deserialize_with = "parse_expr")]
    vertex_offset: Expr,
}
```

## Compilation Pipeline

```
   Expr (AST)
       │
       ▼
   Type Check
       │
       ▼
   ┌───┴───┐
   │       │
   ▼       ▼
 WGSL   Cranelift   Interpreter
(GPU)    (CPU JIT)    (fallback)
```

### Interpreter (always available)

```rust
pub fn interpret(expr: &Expr, ctx: &EvalContext) -> Result<Value> {
    match expr {
        Expr::Const(s) => Ok(s.into()),
        Expr::Var(id) => ctx.vars.get(id).cloned().ok_or(UnboundVar(*id)),
        Expr::Add(a, b) => {
            let va = interpret(a, ctx)?;
            let vb = interpret(b, ctx)?;
            Ok(va.add(&vb)?)
        }
        Expr::Builtin(f, args) => {
            let values: Vec<_> = args.iter().map(|a| interpret(a, ctx)).collect::<Result<_>>()?;
            evaluate_builtin(*f, &values)
        }
        Expr::Plugin(name, args) => {
            let func = ctx.functions.get(name)?;
            let values: Vec<_> = args.iter().map(|a| interpret(a, ctx)).collect::<Result<_>>()?;

            // Try decomposition first
            if let Some(decomposed) = func.decompose(args) {
                return interpret(&decomposed, ctx);
            }

            // Fall back to native interpret
            func.interpret(&values)
        }
        // ...
    }
}
```

### WGSL Codegen

```rust
pub fn to_wgsl(expr: &Expr, ctx: &WgslContext) -> String {
    match expr {
        Expr::Const(Scalar::F32(v)) => format!("{:.8}", v),
        Expr::Var(id) => ctx.var_name(*id),
        Expr::Add(a, b) => format!("({} + {})", to_wgsl(a, ctx), to_wgsl(b, ctx)),
        Expr::Builtin(BuiltinFn::Sin, args) => {
            format!("sin({})", to_wgsl(&args[0], ctx))
        }
        Expr::Plugin(name, args) => {
            let func = ctx.functions.get(name)?;

            // Try decomposition first
            if let Some(decomposed) = func.decompose(args) {
                return to_wgsl(&decomposed, ctx);
            }

            // Use native WGSL impl
            match func.wgsl_impl() {
                Some(WgslImpl::Inline(template)) => {
                    // Replace $0, $1, ... with args
                    let mut result = template.clone();
                    for (i, arg) in args.iter().enumerate() {
                        result = result.replace(&format!("${}", i), &to_wgsl(arg, ctx));
                    }
                    result
                }
                Some(WgslImpl::Function { name, code }) => {
                    ctx.add_function(code);
                    let args_str = args.iter().map(|a| to_wgsl(a, ctx)).join(", ");
                    format!("{}({})", name, args_str)
                }
                None => panic!("No WGSL impl for {}", name),
            }
        }
        // ...
    }
}
```

### Cranelift Codegen

Similar structure, generating IR instead of strings.

## What's NOT in Expressions

### Loops

No explicit loops. Reasons:
- Most transforms are per-element (embarrassingly parallel)
- Iteration is graph-level (evaluate node N times)
- Loops complicate GPU compilation

If you need iteration, use graph recurrence (see [recurrent-graphs](./recurrent-graphs.md)).

### Texture Sampling

Not a primitive expression. Why:
- Requires texture binding (external resource)
- Different from pure math

Instead: ops that take textures are separate nodes:

```rust
// Not: expr with texture.sample(uv)
// But: separate op
#[derive(Op)]
struct DisplaceByTexture {
    texture: TextureId,
    amount: f32,
}
```

### Mutable State

Expressions are pure. No assignment, no side effects.

## Crate Structure

```
resin-expr/           # Core: AST, types, builders, interpreter
resin-expr-macros/    # Proc macro: expr!()
resin-expr-parse/     # Runtime parser
resin-expr-wgsl/      # WGSL codegen
resin-expr-cranelift/ # Cranelift JIT codegen
```

**Why this split:**

| Crate | Reason for separation |
|-------|----------------------|
| `resin-expr-macros` | Required - proc macros must be own crate |
| `resin-expr-parse` | Optional - not needed for hardcoded expressions |
| `resin-expr-wgsl` | Optional - heavy dep (naga), not needed for CPU-only |
| `resin-expr-cranelift` | Optional - very heavy dep (~50 crates) |

**Core crate includes interpreter:**

The interpreter lives in core, not a separate crate. Reasons:

1. **Universal fallback** - Any code using expressions can always evaluate them, even without JIT or GPU
2. **Plugin development** - Plugin authors need `interpret()` for testing without setting up backends
3. **Small footprint** - ~200 LOC, no heavy dependencies
4. **Required by trait** - `PluginFn::interpret()` is mandatory, so interpreter must be available wherever plugins are defined
5. **Validation** - Type checking and basic evaluation happen before backend compilation

If interpreter were separate, every crate defining plugin functions would need to depend on it anyway.

**Dependency flow:**

```
resin-expr-macros ──┐
                    │
resin-expr-parse ───┼──▶ resin-expr (core)
                    │
resin-expr-wgsl ────┤
                    │
resin-expr-cranelift┘
```

All optional crates depend on core. Core has no heavy deps.

## Open Questions

1. **Matrix types** - Do we need `Mat4` as a primitive type, or just functions like `mat_mul(m, v)`? WGSL has matrix primitives with operator overloading (`mat4 * vec4`), but we could keep AST simpler with explicit functions. Trade-off: ergonomics vs simplicity.

2. **Array/buffer access** - `buffer[i]` syntax for accessing arrays/buffers. Important for algorithms that need neighbor access (convolutions, sorting). Complicates GPU compilation (bounds checking, memory layout).

3. **Constant folding** - Should live in separate `resin-expr-opt` crate as AST transformation. Not part of core.

4. **Error recovery** - Parser should support partial parsing for IDE integration. Source spans optional, stored separately from AST.

5. **Debugging** - Step-through interpreter for development. Low priority, leave open.

## Summary

| Aspect | Decision |
|--------|----------|
| AST scope | Math + conditionals + let bindings |
| Loops | No (use graph recurrence) |
| Built-in functions | WGSL built-ins as reference set |
| Plugin functions | Decompose to primitives, or backend extension traits |
| Type system | Inference from operations |
| Construction | Builders, proc macro, runtime parser |
| Crate structure | Core + macros + parse + wgsl + cranelift |
