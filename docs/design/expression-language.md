# Expression Language

Serializable expressions that replace closures for transforms.

> **Implementation:** Expressions are provided by [dew](https://github.com/rhizome-lab/dew), a separate expression library.
> The `rhi-unshape-expr-field` crate bridges dew to the Field system.

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

## Solution: Expression AST (dew)

Expressions are data structures that represent computations. We use the `dew` crate:

```rust
use rhizome_dew_core::Expr;
use rhizome_dew_scalar::{scalar_registry, eval};

// Parse an expression
let expr = Expr::parse("x * 2.0 + y").unwrap();

// Evaluate with variable bindings
let vars = [("x".to_string(), 3.0), ("y".to_string(), 1.0)].into();
let result = eval(expr.ast(), &vars, &scalar_registry()).unwrap();
// result = 7.0
```

### Using with Fields

The `rhi-unshape-expr-field` crate provides `ExprField` which bridges dew to the Field system:

```rust
use rhi_unshape_expr_field::{ExprField, register_noise, scalar_registry};
use rhi_unshape_field::{Field, EvalContext};
use glam::Vec2;

// Create registry with standard math + noise functions
let mut registry = scalar_registry();
register_noise(&mut registry);

// Parse and use as a Field
let field = ExprField::parse("sin(x * 3.14159) + noise(x, y)", registry).unwrap();
let ctx = EvalContext::new();
let value: f32 = field.sample(Vec2::new(0.5, 0.5), &ctx);
```

## AST Design

> **Note:** The dew library provides the actual AST implementation. See the [dew repository](https://github.com/rhizome-lab/dew) for details.

### dew-core AST

```rust
/// Expression AST node (from dew-core)
pub enum Ast {
    Num(f32),                          // Numeric literal
    Var(String),                       // Variable reference
    BinOp(BinOp, Box<Ast>, Box<Ast>), // Binary operation (+, -, *, /, ^)
    UnaryOp(UnaryOp, Box<Ast>),        // Unary operation (-)
    Call(String, Vec<Ast>),            // Function call
}
```

### Conceptual Extensions

The design below describes potential extensions that could be added to dew or a wrapper layer. These are not currently implemented but inform future direction.

```rust
/// Extended AST (conceptual - not in dew)
pub enum ExtendedExpr {
    // === Vector literals ===
    Vec2([Box<ExtendedExpr>; 2]),
    Vec3([Box<ExtendedExpr>; 3]),
    Vec4([Box<ExtendedExpr>; 4]),

    // === Comparison ===
    Eq(Box<ExtendedExpr>, Box<ExtendedExpr>),
    Lt(Box<ExtendedExpr>, Box<ExtendedExpr>),
    // ...

    // === Control flow ===
    If(Box<ExtendedExpr>, Box<ExtendedExpr>, Box<ExtendedExpr>),

    // === Let binding ===
    Let(String, Box<ExtendedExpr>, Box<ExtendedExpr>),

    // === Vector ops ===
    Swizzle(Box<ExtendedExpr>, Swizzle),
    Index(Box<ExtendedExpr>, usize),
}
```

## Functions

All functions are registered via a function registry. In dew, this is `ScalarFn<T>`:

### ScalarFn Trait (from dew-scalar)

```rust
/// Scalar function trait (from dew-scalar)
pub trait ScalarFn<T>: Send + Sync {
    fn name(&self) -> &str;
    fn arg_count(&self) -> usize;
    fn call(&self, args: &[T]) -> T;
}
```

### Unshape Noise Functions

The `rhi-unshape-expr-field` crate provides noise functions:

```rust
use rhi_unshape_expr_field::{register_noise, scalar_registry};

let mut registry = scalar_registry();  // Standard math (sin, cos, etc.)
register_noise(&mut registry);         // Adds: noise, perlin, perlin3, simplex, simplex3, fbm

// Available functions:
// - noise(x, y)      - 2D Perlin noise
// - perlin(x, y)     - 2D Perlin noise (alias)
// - perlin3(x, y, z) - 3D Perlin noise
// - simplex(x, y)    - 2D Simplex noise
// - simplex3(x, y, z)- 3D Simplex noise
// - fbm(x, y, octaves) - 2D FBM noise
```

### Custom Functions

```rust
use rhizome_dew_scalar::{ScalarFn, FunctionRegistry};

struct MyFunction;
impl ScalarFn<f32> for MyFunction {
    fn name(&self) -> &str { "my_func" }
    fn arg_count(&self) -> usize { 2 }
    fn call(&self, args: &[f32]) -> f32 {
        args[0] * args[1] + 1.0
    }
}

let mut registry = FunctionRegistry::new();
registry.register(MyFunction);
```

---

## Future: Backend Extensions

> **Note:** The sections below describe potential future architecture for multi-backend compilation (WGSL, Cranelift, Lua). dew already has some of this infrastructure - see its `cranelift`, `wgsl`, and `lua` modules.

### Backend Extension Traits

Backend crates define extension traits for native compilation. Functions that don't implement an extension trait fall back to `decompose()` or `interpret()`.

```rust
// In rhi-unshape-expr-wgsl crate
pub trait WgslExprFn: ExprFn {
    /// Generate WGSL code for this function call
    fn compile_wgsl(&self, args: &[&str]) -> String;
}

// In rhi-unshape-expr-cranelift crate
pub trait CraneliftExprFn: ExprFn {
    /// Generate Cranelift IR for this function call
    fn compile_cranelift(&self, builder: &mut FunctionBuilder, args: &[cranelift::Value]) -> cranelift::Value;
}

// In rhi-unshape-expr-lua crate (potential future backend)
pub trait LuaExprFn: ExprFn {
    /// Generate Lua code for this function call
    fn compile_lua(&self, args: &[&str]) -> String;
}
```

### Function Registry

```rust
pub struct FunctionRegistry {
    funcs: HashMap<String, Arc<dyn ExprFn>>,
}

impl FunctionRegistry {
    pub fn new() -> Self {
        Self { funcs: HashMap::new() }
    }

    pub fn register<F: ExprFn + 'static>(&mut self, func: F) {
        self.funcs.insert(func.name().to_string(), Arc::new(func));
    }

    pub fn get(&self, name: &str) -> Option<&Arc<dyn ExprFn>> {
        self.funcs.get(name)
    }
}
```

Backend crates wrap the registry with their extension trait downcasting:

```rust
// In rhi-unshape-expr-wgsl:
impl WgslCompiler {
    fn compile_call(&self, name: &str, args: &[Expr]) -> Result<String> {
        let func = self.registry.get(name).ok_or(UnknownFunction(name))?;

        // 1. Try decomposition (works for all backends)
        if let Some(decomposed) = func.decompose(args) {
            return self.compile(&decomposed);
        }

        // 2. Try WGSL-specific implementation via downcast
        if let Some(wgsl_fn) = func.as_any().downcast_ref::<dyn WgslExprFn>() {
            let arg_strs = args.iter().map(|a| self.compile(a)).collect::<Result<Vec<_>>>()?;
            return Ok(wgsl_fn.compile_wgsl(&arg_strs));
        }

        // 3. No native impl - could interpret + inline result, or error
        Err(NoBackendImpl { func: name, backend: "wgsl" })
    }
}
```

**Why this design:**

| Concern | Solution |
|---------|----------|
| Core doesn't know about backends | Backend traits in separate crates |
| Functions don't need all backends | `decompose()` works everywhere |
| Complex functions (noise) | Implement backend extension traits |
| String -> function lookup | Single registry, backends downcast |

### Standard Library (`rhi-unshape-expr-std`)

Standard math functions live in a separate crate. This keeps core minimal and allows users to customize or replace the stdlib.

```rust
// In rhi-unshape-expr-std crate

/// Sine function
pub struct Sin;

impl ExprFn for Sin {
    fn name(&self) -> &str { "sin" }

    fn signature(&self) -> FnSignature {
        FnSignature::new(&[Type::F32], Type::F32)
    }

    fn interpret(&self, args: &[Value]) -> Result<Value> {
        Ok(Value::F32(args[0].as_f32()?.sin()))
    }
}

// WGSL backend: sin is a native WGSL function
impl WgslExprFn for Sin {
    fn compile_wgsl(&self, args: &[&str]) -> String {
        format!("sin({})", args[0])
    }
}

// Cranelift backend: call libm
impl CraneliftExprFn for Sin {
    fn compile_cranelift(&self, builder: &mut FunctionBuilder, args: &[cranelift::Value]) -> cranelift::Value {
        builder.call_libm("sinf", args)
    }
}

// Lua backend: native Lua function
impl LuaExprFn for Sin {
    fn compile_lua(&self, args: &[&str]) -> String {
        format!("math.sin({})", args[0])
    }
}

/// Register all standard functions
pub fn register_std(registry: &mut FunctionRegistry) {
    registry.register(Sin);
    registry.register(Cos);
    registry.register(Tan);
    registry.register(Sqrt);
    registry.register(Abs);
    registry.register(Floor);
    registry.register(Ceil);
    registry.register(Min);
    registry.register(Max);
    registry.register(Clamp);
    registry.register(Mix);  // lerp
    // ... etc
}
```

**Example: Decomposition-only function**

Functions that can express themselves using other functions don't need backend traits:

```rust
/// inverse_lerp(a, b, v) = (v - a) / (b - a)
pub struct InverseLerp;

impl ExprFn for InverseLerp {
    fn name(&self) -> &str { "inverse_lerp" }

    fn signature(&self) -> FnSignature {
        FnSignature::new(&[Type::F32; 3], Type::F32)
    }

    fn decompose(&self, args: &[Expr]) -> Option<Expr> {
        let [a, b, v] = args else { return None };
        Some((v.clone() - a.clone()) / (b.clone() - a.clone()))
    }

    fn interpret(&self, args: &[Value]) -> Result<Value> {
        let [a, b, v] = &args[..] else { return Err(ArgCount) };
        Ok(Value::F32((v.as_f32()? - a.as_f32()?) / (b.as_f32()? - a.as_f32()?)))
    }
}
// Works on ALL backends automatically via decomposition!
```

**Example: Complex function (noise)**

Functions that can't decompose need backend-specific implementations:

```rust
// In rhi-unshape-noise crate
pub struct Perlin2D;

impl ExprFn for Perlin2D {
    fn name(&self) -> &str { "perlin" }

    fn signature(&self) -> FnSignature {
        FnSignature::new(&[Type::F32, Type::F32], Type::F32)
    }

    fn decompose(&self, _: &[Expr]) -> Option<Expr> {
        None  // Can't express as simpler math
    }

    fn interpret(&self, args: &[Value]) -> Result<Value> {
        let x = args[0].as_f32()?;
        let y = args[1].as_f32()?;
        Ok(Value::F32(rhi_unshape_core::noise::perlin2(x, y)))
    }
}

// WGSL: emit shader helper function
impl WgslExprFn for Perlin2D {
    fn compile_wgsl(&self, args: &[&str]) -> String {
        format!("perlin2d({}, {})", args[0], args[1])
    }

    fn wgsl_helpers(&self) -> Option<&str> {
        Some(include_str!("perlin2d.wgsl"))
    }
}

// Cranelift: call into native code
impl CraneliftExprFn for Perlin2D {
    fn compile_cranelift(&self, builder: &mut FunctionBuilder, args: &[cranelift::Value]) -> cranelift::Value {
        builder.call_extern("unshape_perlin2d", args)
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
```

### Value Representation

Runtime values use an enum, not `dyn Trait`:

```rust
#[derive(Clone, Copy, Debug)]
pub enum Value {
    // Always available
    F32(f32),
    F64(f64),
    I32(i32),
    Bool(bool),

    // feature = "vectors"
    #[cfg(feature = "vectors")]
    Vec2([f32; 2]),
    #[cfg(feature = "vectors")]
    Vec3([f32; 3]),
    #[cfg(feature = "vectors")]
    Vec4([f32; 4]),
    #[cfg(feature = "vectors")]
    BVec2([bool; 2]),
    #[cfg(feature = "vectors")]
    BVec3([bool; 3]),
    #[cfg(feature = "vectors")]
    BVec4([bool; 4]),

    // feature = "matrices" (implies vectors)
    #[cfg(feature = "matrices")]
    Mat2([[f32; 2]; 2]),
    #[cfg(feature = "matrices")]
    Mat3([[f32; 3]; 3]),
    #[cfg(feature = "matrices")]
    Mat4([[f32; 4]; 4]),
}
```

Feature gates in `Cargo.toml`:
```toml
[features]
default = ["vectors"]
vectors = []
matrices = ["vectors"]  # matrices implies vectors
```

**Why enum over `dyn Trait`:**

| Concern | Enum | dyn Trait |
|---------|------|-----------|
| Allocation | Stack, no heap | Box per value |
| Exhaustiveness | Compiler enforces | Runtime checks |
| Extensibility | Fixed set | Open |
| Serialization | Trivial | Complex |
| Performance | Fast match | Virtual dispatch + allocation |

Expression primitives are a fixed, finite set (unlike graph `Value` which wraps domain types like `Mesh`, `Image`). Enum is the right choice here.

### Type Inference

```rust
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
pub struct ExprContext {
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
    pub const RESOLUTION: VarId = VarId(5);
    pub const SAMPLE_RATE: VarId = VarId(6);
    // ... etc
}
```

### Ops Bind Variables

Expressions don't access `EvalContext` (graph context) directly. Instead, **ops bind variables when invoking expressions**:

```rust
impl TextureOp {
    fn apply(&self, uv: Vec2, ctx: &EvalContext) -> Color {
        // Op decides what variables to expose
        let expr_ctx = ExprContext::new()
            .bind(var::UV, uv)
            .bind(var::TIME, ctx.time)
            .bind(var::RESOLUTION, ctx.resolution);

        self.expr.eval(&expr_ctx)
    }
}

impl MeshVertexOp {
    fn apply(&self, vertex: &Vertex, ctx: &EvalContext) -> Vec3 {
        let expr_ctx = ExprContext::new()
            .bind(var::POSITION, vertex.position)
            .bind(var::NORMAL, vertex.normal)
            .bind(var::UV, vertex.uv)
            .bind(var::TIME, ctx.time);

        self.expr.eval(&expr_ctx)
    }
}
```

**Why ops bind, not expressions access:**
- No implicit coupling between expressions and graph context
- Ops explicitly control what's available
- Same expression type works in different contexts
- Clear documentation of what variables exist where

### Per-Domain Variables

| Domain | Variables bound by ops |
|--------|------------------------|
| Mesh vertex | position, normal, uv, color, index, time |
| Texture field | uv, time, resolution |
| Audio | time, sample_rate, sample_index, phase |

## Expression Construction

Three ways to build expressions:

### 1. Builder Functions (always available)

Operator overloading + named constructors:

```rust
use rhi_unshape_expr::prelude::*;

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
use rhi_unshape_expr::expr;

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
// rhi-unshape-expr-macros crate
#[proc_macro]
pub fn expr(input: TokenStream) -> TokenStream {
    let parsed = parse_expr_syntax(input);
    let ast = to_expr_ast(parsed);
    quote! { #ast }
}
```

### 3. Runtime Parser (for loaded files, user input)

```rust
use rhi_unshape_expr::parse;

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

### Current Implementation

Expressions are now provided by the `dew` library:

```
dew (external git dependency):
├── dew-core/      # AST, parsing
├── dew-scalar/    # Scalar functions, registry, eval
├── dew-linalg/    # Linear algebra (future)
├── dew-complex/   # Complex numbers (future)
├── dew-quaternion/# Quaternions (future)
├── cranelift.rs   # Cranelift JIT codegen
├── wgsl.rs        # WGSL shader codegen
└── lua.rs         # Lua codegen

unshape:
└── rhi-unshape-expr-field/  # Bridge: dew + Field + noise functions
```

### Unshape Expression Crate

**rhi-unshape-expr-field** provides:
- `ExprField`: evaluates expressions as spatial fields (implements `Field<Vec2, f32>` and `Field<Vec3, f32>`)
- Noise functions: `noise`, `perlin`, `perlin3`, `simplex`, `simplex3`, `fbm`
- Re-exports of key dew types for convenience

**Dependencies:**

```
rhi-unshape-expr-field
├── rhizome-dew-core      # Parsing, AST
├── rhizome-dew-scalar    # FunctionRegistry, eval, scalar_registry
├── rhi-unshape-field   # Field trait, EvalContext
└── rhi-unshape-noise   # Noise implementations
```

## Decisions

1. **Matrix operations** - `*` operator works on matrices (like WGSL). Type inference dispatches: scalar×scalar, vec×vec (component-wise), mat×vec, mat×mat. No AST change needed.

2. **Constant folding** - Separate `rhi-unshape-expr-opt` crate. AST -> AST transformation, not part of core.

3. **Square matrices only** - Mat2/3/4, no Mat3x4. Convert at domain boundaries.

## Open Questions

1. **Array/buffer access** - Do expressions need `buffer[i]` syntax? Most use cases are per-element. For textures, neighbor access is a materialized image op. For audio, recurrence handles simple delays but not: variable-tap reverb, pitch shifting (variable read rate), convolution, granular synthesis. Options: (a) dedicated ops for these patterns, (b) buffer access as power-user escape hatch. Probably ops first, revisit if patterns emerge that need expression-level access.

2. **Error recovery** - Parser should support partial parsing for IDE integration. Source spans optional, stored separately from AST. Low priority.

3. **Debugging** - Step-through interpreter for development. Low priority.

## Summary

| Aspect | Decision |
|--------|----------|
| Implementation | [dew](https://github.com/rhizome-lab/dew) library (external) |
| AST scope | Math + function calls (via dew-core) |
| Loops | No (use graph recurrence) |
| Functions | All functions are `ScalarFn<T>` plugins |
| Standard library | `dew-scalar::scalar_registry()` provides math functions |
| Noise functions | `rhi-unshape-expr-field::register_noise()` |
| Backends | WGSL, Cranelift, Lua (in dew crate) |
| Field integration | `ExprField` in `rhi-unshape-expr-field` |
| Type system | Generic over float type via `ScalarFn<T>` |
