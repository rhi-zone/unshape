//! JIT compilation for FieldExpr.
//!
//! Compiles FieldExpr AST to native code using Cranelift.
//! Noise functions are called via external symbols.

#![cfg(feature = "cranelift")]

use cranelift::Context;
use cranelift::ir::{AbiParam, InstBuilder, Value, types};
use cranelift::settings::{self, Configurable};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{Linkage, Module};

use crate::FieldExpr;
use std::collections::HashMap;

/// Errors during field expression JIT compilation.
#[derive(Debug, thiserror::Error)]
pub enum FieldJitError {
    #[error("compilation error: {0}")]
    Compilation(String),

    #[error("module error: {0}")]
    Module(#[from] cranelift_module::ModuleError),

    #[error("unsupported expression: {0}")]
    Unsupported(String),
}

/// Result type for field JIT operations.
pub type FieldJitResult<T> = Result<T, FieldJitError>;

/// A JIT-compiled field expression.
///
/// Evaluates `(x, y, z, t) -> f32` using native code.
pub struct CompiledFieldExpr {
    func: fn(f32, f32, f32, f32) -> f32,
}

impl CompiledFieldExpr {
    /// Evaluate at position (x, y, z) and time t.
    #[inline]
    pub fn eval(&self, x: f32, y: f32, z: f32, t: f32) -> f32 {
        (self.func)(x, y, z, t)
    }

    /// Evaluate at 2D position.
    #[inline]
    pub fn eval_2d(&self, x: f32, y: f32, t: f32) -> f32 {
        (self.func)(x, y, 0.0, t)
    }
}

/// Compiler for field expressions.
pub struct FieldExprCompiler {
    module: JITModule,
    builder_ctx: FunctionBuilderContext,
    ctx: Context,
    func_counter: usize,
    // External function IDs for noise
    perlin2_id: Option<cranelift_module::FuncId>,
    perlin3_id: Option<cranelift_module::FuncId>,
    simplex2_id: Option<cranelift_module::FuncId>,
    simplex3_id: Option<cranelift_module::FuncId>,
    fbm2_id: Option<cranelift_module::FuncId>,
    fbm3_id: Option<cranelift_module::FuncId>,
}

// External noise function wrappers (need extern "C" ABI)
extern "C" fn perlin2_wrapper(x: f32, y: f32) -> f32 {
    rhizome_resin_noise::perlin2(x, y)
}

extern "C" fn perlin3_wrapper(x: f32, y: f32, z: f32) -> f32 {
    rhizome_resin_noise::perlin3(x, y, z)
}

extern "C" fn simplex2_wrapper(x: f32, y: f32) -> f32 {
    rhizome_resin_noise::simplex2(x, y)
}

extern "C" fn simplex3_wrapper(x: f32, y: f32, z: f32) -> f32 {
    rhizome_resin_noise::simplex3(x, y, z)
}

extern "C" fn fbm2_wrapper(x: f32, y: f32, octaves: f32) -> f32 {
    rhizome_resin_noise::fbm_perlin2(x, y, octaves as u32)
}

extern "C" fn fbm3_wrapper(x: f32, y: f32, z: f32, octaves: f32) -> f32 {
    rhizome_resin_noise::fbm_perlin3(x, y, z, octaves as u32)
}

impl FieldExprCompiler {
    /// Create a new field expression compiler.
    pub fn new() -> FieldJitResult<Self> {
        let mut flag_builder = settings::builder();
        flag_builder.set("opt_level", "speed").unwrap();

        let isa_builder =
            cranelift_native::builder().map_err(|e| FieldJitError::Compilation(e.to_string()))?;
        let isa = isa_builder
            .finish(settings::Flags::new(flag_builder))
            .map_err(|e| FieldJitError::Compilation(e.to_string()))?;

        let mut builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());

        // Register noise function symbols
        builder.symbol("perlin2", perlin2_wrapper as *const u8);
        builder.symbol("perlin3", perlin3_wrapper as *const u8);
        builder.symbol("simplex2", simplex2_wrapper as *const u8);
        builder.symbol("simplex3", simplex3_wrapper as *const u8);
        builder.symbol("fbm2", fbm2_wrapper as *const u8);
        builder.symbol("fbm3", fbm3_wrapper as *const u8);

        let module = JITModule::new(builder);
        let ctx = module.make_context();

        Ok(Self {
            module,
            builder_ctx: FunctionBuilderContext::new(),
            ctx,
            func_counter: 0,
            perlin2_id: None,
            perlin3_id: None,
            simplex2_id: None,
            simplex3_id: None,
            fbm2_id: None,
            fbm3_id: None,
        })
    }

    fn next_func_name(&mut self) -> String {
        let name = format!("field_expr_{}", self.func_counter);
        self.func_counter += 1;
        name
    }

    /// Declare external noise functions.
    fn declare_noise_functions(&mut self) -> FieldJitResult<()> {
        // perlin2(f32, f32) -> f32
        if self.perlin2_id.is_none() {
            let mut sig = self.module.make_signature();
            sig.params.push(AbiParam::new(types::F32));
            sig.params.push(AbiParam::new(types::F32));
            sig.returns.push(AbiParam::new(types::F32));
            self.perlin2_id = Some(self.module.declare_function(
                "perlin2",
                Linkage::Import,
                &sig,
            )?);
        }

        // perlin3(f32, f32, f32) -> f32
        if self.perlin3_id.is_none() {
            let mut sig = self.module.make_signature();
            sig.params.push(AbiParam::new(types::F32));
            sig.params.push(AbiParam::new(types::F32));
            sig.params.push(AbiParam::new(types::F32));
            sig.returns.push(AbiParam::new(types::F32));
            self.perlin3_id = Some(self.module.declare_function(
                "perlin3",
                Linkage::Import,
                &sig,
            )?);
        }

        // simplex2(f32, f32) -> f32
        if self.simplex2_id.is_none() {
            let mut sig = self.module.make_signature();
            sig.params.push(AbiParam::new(types::F32));
            sig.params.push(AbiParam::new(types::F32));
            sig.returns.push(AbiParam::new(types::F32));
            self.simplex2_id = Some(self.module.declare_function(
                "simplex2",
                Linkage::Import,
                &sig,
            )?);
        }

        // simplex3(f32, f32, f32) -> f32
        if self.simplex3_id.is_none() {
            let mut sig = self.module.make_signature();
            sig.params.push(AbiParam::new(types::F32));
            sig.params.push(AbiParam::new(types::F32));
            sig.params.push(AbiParam::new(types::F32));
            sig.returns.push(AbiParam::new(types::F32));
            self.simplex3_id = Some(self.module.declare_function(
                "simplex3",
                Linkage::Import,
                &sig,
            )?);
        }

        // fbm2(f32, f32, f32) -> f32
        if self.fbm2_id.is_none() {
            let mut sig = self.module.make_signature();
            sig.params.push(AbiParam::new(types::F32));
            sig.params.push(AbiParam::new(types::F32));
            sig.params.push(AbiParam::new(types::F32));
            sig.returns.push(AbiParam::new(types::F32));
            self.fbm2_id = Some(
                self.module
                    .declare_function("fbm2", Linkage::Import, &sig)?,
            );
        }

        // fbm3(f32, f32, f32, f32) -> f32
        if self.fbm3_id.is_none() {
            let mut sig = self.module.make_signature();
            sig.params.push(AbiParam::new(types::F32));
            sig.params.push(AbiParam::new(types::F32));
            sig.params.push(AbiParam::new(types::F32));
            sig.params.push(AbiParam::new(types::F32));
            sig.returns.push(AbiParam::new(types::F32));
            self.fbm3_id = Some(
                self.module
                    .declare_function("fbm3", Linkage::Import, &sig)?,
            );
        }

        Ok(())
    }

    /// Compile a field expression to native code.
    pub fn compile(&mut self, expr: &FieldExpr) -> FieldJitResult<CompiledFieldExpr> {
        // Declare external functions we might need
        self.declare_noise_functions()?;

        // Signature: fn(x: f32, y: f32, z: f32, t: f32) -> f32
        let mut sig = self.module.make_signature();
        sig.params.push(AbiParam::new(types::F32)); // x
        sig.params.push(AbiParam::new(types::F32)); // y
        sig.params.push(AbiParam::new(types::F32)); // z
        sig.params.push(AbiParam::new(types::F32)); // t
        sig.returns.push(AbiParam::new(types::F32));

        let func_name = self.next_func_name();
        let func_id = self
            .module
            .declare_function(&func_name, Linkage::Export, &sig)?;

        self.ctx.func.signature = sig;

        {
            let mut builder = FunctionBuilder::new(&mut self.ctx.func, &mut self.builder_ctx);
            let entry = builder.create_block();
            builder.append_block_params_for_function_params(entry);
            builder.switch_to_block(entry);
            builder.seal_block(entry);

            let x = builder.block_params(entry)[0];
            let y = builder.block_params(entry)[1];
            let z = builder.block_params(entry)[2];
            let t = builder.block_params(entry)[3];

            // Build context for compilation
            let mut compile_ctx = CompileContext {
                builder: &mut builder,
                module: &mut self.module,
                x,
                y,
                z,
                t,
                vars: HashMap::new(),
                perlin2_id: self.perlin2_id,
                perlin3_id: self.perlin3_id,
                simplex2_id: self.simplex2_id,
                simplex3_id: self.simplex3_id,
                fbm2_id: self.fbm2_id,
                fbm3_id: self.fbm3_id,
            };

            let result = compile_expr(expr, &mut compile_ctx)?;
            builder.ins().return_(&[result]);
            builder.finalize();
        }

        self.module.define_function(func_id, &mut self.ctx)?;
        self.module.clear_context(&mut self.ctx);
        self.module.finalize_definitions()?;

        let code_ptr = self.module.get_finalized_function(func_id);

        Ok(CompiledFieldExpr {
            func: unsafe { std::mem::transmute(code_ptr) },
        })
    }
}

/// Context for recursive expression compilation.
struct CompileContext<'a, 'b> {
    builder: &'a mut FunctionBuilder<'b>,
    module: &'a mut JITModule,
    x: Value,
    y: Value,
    z: Value,
    t: Value,
    vars: HashMap<String, Value>,
    perlin2_id: Option<cranelift_module::FuncId>,
    perlin3_id: Option<cranelift_module::FuncId>,
    simplex2_id: Option<cranelift_module::FuncId>,
    simplex3_id: Option<cranelift_module::FuncId>,
    fbm2_id: Option<cranelift_module::FuncId>,
    fbm3_id: Option<cranelift_module::FuncId>,
}

/// Recursively compile a FieldExpr to Cranelift IR.
fn compile_expr(expr: &FieldExpr, ctx: &mut CompileContext) -> FieldJitResult<Value> {
    match expr {
        // Coordinates
        FieldExpr::X => Ok(ctx.x),
        FieldExpr::Y => Ok(ctx.y),
        FieldExpr::Z => Ok(ctx.z),
        FieldExpr::T => Ok(ctx.t),

        // Literals
        FieldExpr::Constant(v) => Ok(ctx.builder.ins().f32const(*v)),
        FieldExpr::Var(name) => ctx
            .vars
            .get(name)
            .copied()
            .ok_or_else(|| FieldJitError::Unsupported(format!("unbound variable: {}", name))),

        // Binary operations
        FieldExpr::Add(a, b) => {
            let va = compile_expr(a, ctx)?;
            let vb = compile_expr(b, ctx)?;
            Ok(ctx.builder.ins().fadd(va, vb))
        }
        FieldExpr::Sub(a, b) => {
            let va = compile_expr(a, ctx)?;
            let vb = compile_expr(b, ctx)?;
            Ok(ctx.builder.ins().fsub(va, vb))
        }
        FieldExpr::Mul(a, b) => {
            let va = compile_expr(a, ctx)?;
            let vb = compile_expr(b, ctx)?;
            Ok(ctx.builder.ins().fmul(va, vb))
        }
        FieldExpr::Div(a, b) => {
            let va = compile_expr(a, ctx)?;
            let vb = compile_expr(b, ctx)?;
            Ok(ctx.builder.ins().fdiv(va, vb))
        }
        FieldExpr::Mod(a, b) => {
            // a % b = a - floor(a / b) * b
            let va = compile_expr(a, ctx)?;
            let vb = compile_expr(b, ctx)?;
            let div = ctx.builder.ins().fdiv(va, vb);
            let floored = ctx.builder.ins().floor(div);
            let mul = ctx.builder.ins().fmul(floored, vb);
            Ok(ctx.builder.ins().fsub(va, mul))
        }
        FieldExpr::Pow(_a, _b) => {
            // Cranelift doesn't have built-in pow, would need libm
            Err(FieldJitError::Unsupported(
                "pow requires libm - not yet implemented".into(),
            ))
        }

        // Unary operations
        FieldExpr::Neg(a) => {
            let va = compile_expr(a, ctx)?;
            Ok(ctx.builder.ins().fneg(va))
        }

        // Noise functions - call external
        FieldExpr::Perlin2 { x, y } => {
            let vx = compile_expr(x, ctx)?;
            let vy = compile_expr(y, ctx)?;
            let func_id = ctx
                .perlin2_id
                .ok_or_else(|| FieldJitError::Compilation("perlin2 not declared".into()))?;
            let func_ref = ctx.module.declare_func_in_func(func_id, ctx.builder.func);
            let call = ctx.builder.ins().call(func_ref, &[vx, vy]);
            Ok(ctx.builder.inst_results(call)[0])
        }
        FieldExpr::Perlin3 { x, y, z } => {
            let vx = compile_expr(x, ctx)?;
            let vy = compile_expr(y, ctx)?;
            let vz = compile_expr(z, ctx)?;
            let func_id = ctx
                .perlin3_id
                .ok_or_else(|| FieldJitError::Compilation("perlin3 not declared".into()))?;
            let func_ref = ctx.module.declare_func_in_func(func_id, ctx.builder.func);
            let call = ctx.builder.ins().call(func_ref, &[vx, vy, vz]);
            Ok(ctx.builder.inst_results(call)[0])
        }
        FieldExpr::Simplex2 { x, y } => {
            let vx = compile_expr(x, ctx)?;
            let vy = compile_expr(y, ctx)?;
            let func_id = ctx
                .simplex2_id
                .ok_or_else(|| FieldJitError::Compilation("simplex2 not declared".into()))?;
            let func_ref = ctx.module.declare_func_in_func(func_id, ctx.builder.func);
            let call = ctx.builder.ins().call(func_ref, &[vx, vy]);
            Ok(ctx.builder.inst_results(call)[0])
        }
        FieldExpr::Simplex3 { x, y, z } => {
            let vx = compile_expr(x, ctx)?;
            let vy = compile_expr(y, ctx)?;
            let vz = compile_expr(z, ctx)?;
            let func_id = ctx
                .simplex3_id
                .ok_or_else(|| FieldJitError::Compilation("simplex3 not declared".into()))?;
            let func_ref = ctx.module.declare_func_in_func(func_id, ctx.builder.func);
            let call = ctx.builder.ins().call(func_ref, &[vx, vy, vz]);
            Ok(ctx.builder.inst_results(call)[0])
        }
        FieldExpr::Fbm2 { x, y, octaves } => {
            let vx = compile_expr(x, ctx)?;
            let vy = compile_expr(y, ctx)?;
            let voct = ctx.builder.ins().f32const(*octaves as f32);
            let func_id = ctx
                .fbm2_id
                .ok_or_else(|| FieldJitError::Compilation("fbm2 not declared".into()))?;
            let func_ref = ctx.module.declare_func_in_func(func_id, ctx.builder.func);
            let call = ctx.builder.ins().call(func_ref, &[vx, vy, voct]);
            Ok(ctx.builder.inst_results(call)[0])
        }
        FieldExpr::Fbm3 { x, y, z, octaves } => {
            let vx = compile_expr(x, ctx)?;
            let vy = compile_expr(y, ctx)?;
            let vz = compile_expr(z, ctx)?;
            let voct = ctx.builder.ins().f32const(*octaves as f32);
            let func_id = ctx
                .fbm3_id
                .ok_or_else(|| FieldJitError::Compilation("fbm3 not declared".into()))?;
            let func_ref = ctx.module.declare_func_in_func(func_id, ctx.builder.func);
            let call = ctx.builder.ins().call(func_ref, &[vx, vy, vz, voct]);
            Ok(ctx.builder.inst_results(call)[0])
        }

        // Distance functions - inline
        FieldExpr::Length2 { x, y } => {
            let vx = compile_expr(x, ctx)?;
            let vy = compile_expr(y, ctx)?;
            let x2 = ctx.builder.ins().fmul(vx, vx);
            let y2 = ctx.builder.ins().fmul(vy, vy);
            let sum = ctx.builder.ins().fadd(x2, y2);
            Ok(ctx.builder.ins().sqrt(sum))
        }
        FieldExpr::Length3 { x, y, z } => {
            let vx = compile_expr(x, ctx)?;
            let vy = compile_expr(y, ctx)?;
            let vz = compile_expr(z, ctx)?;
            let x2 = ctx.builder.ins().fmul(vx, vx);
            let y2 = ctx.builder.ins().fmul(vy, vy);
            let z2 = ctx.builder.ins().fmul(vz, vz);
            let sum_xy = ctx.builder.ins().fadd(x2, y2);
            let sum = ctx.builder.ins().fadd(sum_xy, z2);
            Ok(ctx.builder.ins().sqrt(sum))
        }
        FieldExpr::Distance2 { x, y, px, py } => {
            let vx = compile_expr(x, ctx)?;
            let vy = compile_expr(y, ctx)?;
            let cpx = ctx.builder.ins().f32const(*px);
            let cpy = ctx.builder.ins().f32const(*py);
            let dx = ctx.builder.ins().fsub(vx, cpx);
            let dy = ctx.builder.ins().fsub(vy, cpy);
            let dx2 = ctx.builder.ins().fmul(dx, dx);
            let dy2 = ctx.builder.ins().fmul(dy, dy);
            let sum = ctx.builder.ins().fadd(dx2, dy2);
            Ok(ctx.builder.ins().sqrt(sum))
        }
        FieldExpr::Distance3 {
            x,
            y,
            z,
            px,
            py,
            pz,
        } => {
            let vx = compile_expr(x, ctx)?;
            let vy = compile_expr(y, ctx)?;
            let vz = compile_expr(z, ctx)?;
            let cpx = ctx.builder.ins().f32const(*px);
            let cpy = ctx.builder.ins().f32const(*py);
            let cpz = ctx.builder.ins().f32const(*pz);
            let dx = ctx.builder.ins().fsub(vx, cpx);
            let dy = ctx.builder.ins().fsub(vy, cpy);
            let dz = ctx.builder.ins().fsub(vz, cpz);
            let dx2 = ctx.builder.ins().fmul(dx, dx);
            let dy2 = ctx.builder.ins().fmul(dy, dy);
            let dz2 = ctx.builder.ins().fmul(dz, dz);
            let sum_xy = ctx.builder.ins().fadd(dx2, dy2);
            let sum = ctx.builder.ins().fadd(sum_xy, dz2);
            Ok(ctx.builder.ins().sqrt(sum))
        }

        // SDF operations - inline
        FieldExpr::SdfCircle { x, y, radius } => {
            let vx = compile_expr(x, ctx)?;
            let vy = compile_expr(y, ctx)?;
            let x2 = ctx.builder.ins().fmul(vx, vx);
            let y2 = ctx.builder.ins().fmul(vy, vy);
            let sum = ctx.builder.ins().fadd(x2, y2);
            let len = ctx.builder.ins().sqrt(sum);
            let r = ctx.builder.ins().f32const(*radius);
            Ok(ctx.builder.ins().fsub(len, r))
        }
        FieldExpr::SdfSphere { x, y, z, radius } => {
            let vx = compile_expr(x, ctx)?;
            let vy = compile_expr(y, ctx)?;
            let vz = compile_expr(z, ctx)?;
            let x2 = ctx.builder.ins().fmul(vx, vx);
            let y2 = ctx.builder.ins().fmul(vy, vy);
            let z2 = ctx.builder.ins().fmul(vz, vz);
            let sum_xy = ctx.builder.ins().fadd(x2, y2);
            let sum = ctx.builder.ins().fadd(sum_xy, z2);
            let len = ctx.builder.ins().sqrt(sum);
            let r = ctx.builder.ins().f32const(*radius);
            Ok(ctx.builder.ins().fsub(len, r))
        }

        // Math functions - use Cranelift intrinsics where available
        FieldExpr::Abs(a) => {
            let va = compile_expr(a, ctx)?;
            Ok(ctx.builder.ins().fabs(va))
        }
        FieldExpr::Floor(a) => {
            let va = compile_expr(a, ctx)?;
            Ok(ctx.builder.ins().floor(va))
        }
        FieldExpr::Ceil(a) => {
            let va = compile_expr(a, ctx)?;
            Ok(ctx.builder.ins().ceil(va))
        }
        FieldExpr::Sqrt(a) => {
            let va = compile_expr(a, ctx)?;
            Ok(ctx.builder.ins().sqrt(va))
        }
        FieldExpr::Fract(a) => {
            // fract(x) = x - floor(x)
            let va = compile_expr(a, ctx)?;
            let floored = ctx.builder.ins().floor(va);
            Ok(ctx.builder.ins().fsub(va, floored))
        }
        FieldExpr::Min(a, b) => {
            let va = compile_expr(a, ctx)?;
            let vb = compile_expr(b, ctx)?;
            Ok(ctx.builder.ins().fmin(va, vb))
        }
        FieldExpr::Max(a, b) => {
            let va = compile_expr(a, ctx)?;
            let vb = compile_expr(b, ctx)?;
            Ok(ctx.builder.ins().fmax(va, vb))
        }
        FieldExpr::Clamp { value, min, max } => {
            let vval = compile_expr(value, ctx)?;
            let vmin = compile_expr(min, ctx)?;
            let vmax = compile_expr(max, ctx)?;
            let clamped_low = ctx.builder.ins().fmax(vval, vmin);
            Ok(ctx.builder.ins().fmin(clamped_low, vmax))
        }
        FieldExpr::Lerp { a, b, t } => {
            // lerp(a, b, t) = a + (b - a) * t
            let va = compile_expr(a, ctx)?;
            let vb = compile_expr(b, ctx)?;
            let vt = compile_expr(t, ctx)?;
            let diff = ctx.builder.ins().fsub(vb, va);
            let scaled = ctx.builder.ins().fmul(diff, vt);
            Ok(ctx.builder.ins().fadd(va, scaled))
        }
        FieldExpr::Sign(a) => {
            // sign(x) = x > 0 ? 1 : (x < 0 ? -1 : 0)
            // Approximation: x / (|x| + epsilon) but that's not exact
            // For now, use conditional
            let va = compile_expr(a, ctx)?;
            let zero = ctx.builder.ins().f32const(0.0);
            let one = ctx.builder.ins().f32const(1.0);
            let neg_one = ctx.builder.ins().f32const(-1.0);

            // Use select: sign = (x > 0) ? 1 : ((x < 0) ? -1 : 0)
            let is_pos =
                ctx.builder
                    .ins()
                    .fcmp(cranelift::ir::condcodes::FloatCC::GreaterThan, va, zero);
            let is_neg =
                ctx.builder
                    .ins()
                    .fcmp(cranelift::ir::condcodes::FloatCC::LessThan, va, zero);

            let neg_or_zero = ctx.builder.ins().select(is_neg, neg_one, zero);
            Ok(ctx.builder.ins().select(is_pos, one, neg_or_zero))
        }

        // Transcendentals - not directly supported in Cranelift, return error
        FieldExpr::Sin(_)
        | FieldExpr::Cos(_)
        | FieldExpr::Tan(_)
        | FieldExpr::Exp(_)
        | FieldExpr::Ln(_) => Err(FieldJitError::Unsupported(
            "transcendental functions (sin, cos, tan, exp, ln) require libm - not yet implemented"
                .into(),
        )),

        // SDF smooth operations
        FieldExpr::SdfSmoothUnion { a, b, k } => {
            // h = clamp(0.5 + 0.5 * (d2 - d1) / k, 0, 1)
            // result = d2 * (1 - h) + d1 * h - k * h * (1 - h)
            let d1 = compile_expr(a, ctx)?;
            let d2 = compile_expr(b, ctx)?;
            let ck = ctx.builder.ins().f32const(*k);
            let half = ctx.builder.ins().f32const(0.5);
            let zero = ctx.builder.ins().f32const(0.0);
            let one = ctx.builder.ins().f32const(1.0);

            let diff = ctx.builder.ins().fsub(d2, d1);
            let ratio = ctx.builder.ins().fdiv(diff, ck);
            let scaled = ctx.builder.ins().fmul(half, ratio);
            let h_unclamped = ctx.builder.ins().fadd(half, scaled);
            let h_low = ctx.builder.ins().fmax(h_unclamped, zero);
            let h = ctx.builder.ins().fmin(h_low, one);

            let one_minus_h = ctx.builder.ins().fsub(one, h);
            let term1 = ctx.builder.ins().fmul(d2, one_minus_h);
            let term2 = ctx.builder.ins().fmul(d1, h);
            let term3_a = ctx.builder.ins().fmul(ck, h);
            let term3 = ctx.builder.ins().fmul(term3_a, one_minus_h);

            let sum = ctx.builder.ins().fadd(term1, term2);
            Ok(ctx.builder.ins().fsub(sum, term3))
        }

        // SmoothStep, Step
        FieldExpr::SmoothStep { edge0, edge1, x } => {
            // t = clamp((x - edge0) / (edge1 - edge0), 0, 1)
            // result = t * t * (3 - 2 * t)
            let ve0 = compile_expr(edge0, ctx)?;
            let ve1 = compile_expr(edge1, ctx)?;
            let vx = compile_expr(x, ctx)?;
            let zero = ctx.builder.ins().f32const(0.0);
            let one = ctx.builder.ins().f32const(1.0);
            let two = ctx.builder.ins().f32const(2.0);
            let three = ctx.builder.ins().f32const(3.0);

            let x_minus_e0 = ctx.builder.ins().fsub(vx, ve0);
            let e1_minus_e0 = ctx.builder.ins().fsub(ve1, ve0);
            let ratio = ctx.builder.ins().fdiv(x_minus_e0, e1_minus_e0);
            let t_low = ctx.builder.ins().fmax(ratio, zero);
            let t = ctx.builder.ins().fmin(t_low, one);

            let two_t = ctx.builder.ins().fmul(two, t);
            let three_minus_2t = ctx.builder.ins().fsub(three, two_t);
            let t_squared = ctx.builder.ins().fmul(t, t);
            Ok(ctx.builder.ins().fmul(t_squared, three_minus_2t))
        }
        FieldExpr::Step { edge, x } => {
            // step(edge, x) = x < edge ? 0 : 1
            let ve = compile_expr(edge, ctx)?;
            let vx = compile_expr(x, ctx)?;
            let zero = ctx.builder.ins().f32const(0.0);
            let one = ctx.builder.ins().f32const(1.0);

            let is_less =
                ctx.builder
                    .ins()
                    .fcmp(cranelift::ir::condcodes::FloatCC::LessThan, vx, ve);
            Ok(ctx.builder.ins().select(is_less, zero, one))
        }

        // Conditionals
        FieldExpr::Gt(a, b) => {
            let va = compile_expr(a, ctx)?;
            let vb = compile_expr(b, ctx)?;
            let zero = ctx.builder.ins().f32const(0.0);
            let one = ctx.builder.ins().f32const(1.0);
            let cmp =
                ctx.builder
                    .ins()
                    .fcmp(cranelift::ir::condcodes::FloatCC::GreaterThan, va, vb);
            Ok(ctx.builder.ins().select(cmp, one, zero))
        }
        FieldExpr::Lt(a, b) => {
            let va = compile_expr(a, ctx)?;
            let vb = compile_expr(b, ctx)?;
            let zero = ctx.builder.ins().f32const(0.0);
            let one = ctx.builder.ins().f32const(1.0);
            let cmp = ctx
                .builder
                .ins()
                .fcmp(cranelift::ir::condcodes::FloatCC::LessThan, va, vb);
            Ok(ctx.builder.ins().select(cmp, one, zero))
        }
        FieldExpr::Eq(a, b) => {
            // Approximate equality within epsilon
            let va = compile_expr(a, ctx)?;
            let vb = compile_expr(b, ctx)?;
            let zero = ctx.builder.ins().f32const(0.0);
            let one = ctx.builder.ins().f32const(1.0);
            let eps = ctx.builder.ins().f32const(1e-6);

            let diff = ctx.builder.ins().fsub(va, vb);
            let abs_diff = ctx.builder.ins().fabs(diff);
            let is_eq =
                ctx.builder
                    .ins()
                    .fcmp(cranelift::ir::condcodes::FloatCC::LessThan, abs_diff, eps);
            Ok(ctx.builder.ins().select(is_eq, one, zero))
        }
        FieldExpr::IfThenElse {
            condition,
            then_expr,
            else_expr,
        } => {
            let vcond = compile_expr(condition, ctx)?;
            let vthen = compile_expr(then_expr, ctx)?;
            let velse = compile_expr(else_expr, ctx)?;
            let half = ctx.builder.ins().f32const(0.5);
            let cmp =
                ctx.builder
                    .ins()
                    .fcmp(cranelift::ir::condcodes::FloatCC::GreaterThan, vcond, half);
            Ok(ctx.builder.ins().select(cmp, vthen, velse))
        }

        // Not yet implemented
        FieldExpr::SdfBox2 { .. }
        | FieldExpr::SdfBox3 { .. }
        | FieldExpr::SdfSmoothIntersection { .. }
        | FieldExpr::SdfSmoothSubtraction { .. } => Err(FieldJitError::Unsupported(
            "complex SDF operations not yet implemented".into(),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compile_constant() {
        let mut compiler = FieldExprCompiler::new().unwrap();
        let expr = FieldExpr::Constant(42.0);
        let compiled = compiler.compile(&expr).unwrap();
        assert_eq!(compiled.eval(0.0, 0.0, 0.0, 0.0), 42.0);
    }

    #[test]
    fn test_compile_coordinates() {
        let mut compiler = FieldExprCompiler::new().unwrap();

        let expr_x = FieldExpr::X;
        let compiled = compiler.compile(&expr_x).unwrap();
        assert_eq!(compiled.eval(1.0, 2.0, 3.0, 4.0), 1.0);

        let expr_y = FieldExpr::Y;
        let compiled = compiler.compile(&expr_y).unwrap();
        assert_eq!(compiled.eval(1.0, 2.0, 3.0, 4.0), 2.0);

        let expr_z = FieldExpr::Z;
        let compiled = compiler.compile(&expr_z).unwrap();
        assert_eq!(compiled.eval(1.0, 2.0, 3.0, 4.0), 3.0);

        let expr_t = FieldExpr::T;
        let compiled = compiler.compile(&expr_t).unwrap();
        assert_eq!(compiled.eval(1.0, 2.0, 3.0, 4.0), 4.0);
    }

    #[test]
    fn test_compile_arithmetic() {
        let mut compiler = FieldExprCompiler::new().unwrap();

        // x + y
        let expr = FieldExpr::Add(Box::new(FieldExpr::X), Box::new(FieldExpr::Y));
        let compiled = compiler.compile(&expr).unwrap();
        assert_eq!(compiled.eval(3.0, 4.0, 0.0, 0.0), 7.0);

        // x * 2
        let expr = FieldExpr::Mul(Box::new(FieldExpr::X), Box::new(FieldExpr::Constant(2.0)));
        let compiled = compiler.compile(&expr).unwrap();
        assert_eq!(compiled.eval(5.0, 0.0, 0.0, 0.0), 10.0);

        // (x + y) * z
        let expr = FieldExpr::Mul(
            Box::new(FieldExpr::Add(
                Box::new(FieldExpr::X),
                Box::new(FieldExpr::Y),
            )),
            Box::new(FieldExpr::Z),
        );
        let compiled = compiler.compile(&expr).unwrap();
        assert_eq!(compiled.eval(2.0, 3.0, 4.0, 0.0), 20.0);
    }

    #[test]
    fn test_compile_noise() {
        let mut compiler = FieldExprCompiler::new().unwrap();

        let expr = FieldExpr::Perlin2 {
            x: Box::new(FieldExpr::X),
            y: Box::new(FieldExpr::Y),
        };
        let compiled = compiler.compile(&expr).unwrap();
        let v = compiled.eval(0.5, 0.5, 0.0, 0.0);
        assert!((0.0..=1.0).contains(&v), "Perlin out of range: {}", v);
    }

    #[test]
    fn test_compile_sdf_circle() {
        let mut compiler = FieldExprCompiler::new().unwrap();

        let expr = FieldExpr::SdfCircle {
            x: Box::new(FieldExpr::X),
            y: Box::new(FieldExpr::Y),
            radius: 1.0,
        };
        let compiled = compiler.compile(&expr).unwrap();

        // At origin, distance = -1 (inside)
        assert!((compiled.eval(0.0, 0.0, 0.0, 0.0) - (-1.0)).abs() < 0.01);
        // At (1, 0), distance = 0 (on surface)
        assert!((compiled.eval(1.0, 0.0, 0.0, 0.0) - 0.0).abs() < 0.01);
        // At (2, 0), distance = 1 (outside)
        assert!((compiled.eval(2.0, 0.0, 0.0, 0.0) - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_parity_with_interpreted() {
        let mut compiler = FieldExprCompiler::new().unwrap();

        // Complex expression: perlin(x * 4, y * 4) + 0.5
        let expr = FieldExpr::Add(
            Box::new(FieldExpr::Perlin2 {
                x: Box::new(FieldExpr::Mul(
                    Box::new(FieldExpr::X),
                    Box::new(FieldExpr::Constant(4.0)),
                )),
                y: Box::new(FieldExpr::Mul(
                    Box::new(FieldExpr::Y),
                    Box::new(FieldExpr::Constant(4.0)),
                )),
            }),
            Box::new(FieldExpr::Constant(0.5)),
        );

        let compiled = compiler.compile(&expr).unwrap();

        // Test at several points
        for i in 0..10 {
            for j in 0..10 {
                let x = i as f32 * 0.1;
                let y = j as f32 * 0.1;

                let interpreted = expr.eval(x, y, 0.0, 0.0, &Default::default());
                let jit = compiled.eval(x, y, 0.0, 0.0);

                assert!(
                    (interpreted - jit).abs() < 1e-5,
                    "Parity mismatch at ({}, {}): interpreted={}, jit={}",
                    x,
                    y,
                    interpreted,
                    jit
                );
            }
        }
    }
}
