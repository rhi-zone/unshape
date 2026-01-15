//! JIT compilation for FieldExpr.
//!
//! Compiles FieldExpr AST to native code using Cranelift.
//!
//! Features:
//! - **Pure Cranelift perlin2**: Fully inlined, no Rust boundary crossing
//! - **Polynomial transcendentals**: sin, cos, tan, exp, ln via optimized approximations
//! - **Other noise**: simplex2/3, perlin3, fbm use external calls (future work: inline these too)

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
    // External function IDs for noise (perlin2 is now pure Cranelift)
    perlin3_id: Option<cranelift_module::FuncId>,
    simplex2_id: Option<cranelift_module::FuncId>,
    simplex3_id: Option<cranelift_module::FuncId>,
    fbm2_id: Option<cranelift_module::FuncId>,
    fbm3_id: Option<cranelift_module::FuncId>,
}

// External noise function wrappers (need extern "C" ABI)
// Note: perlin2 is now implemented as pure Cranelift IR

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

        // Register noise function symbols (perlin2 is now pure Cranelift)
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
    /// Note: perlin2 is now implemented as pure Cranelift IR.
    fn declare_noise_functions(&mut self) -> FieldJitResult<()> {
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
    // External function IDs (perlin2 is now pure Cranelift)
    perlin3_id: Option<cranelift_module::FuncId>,
    simplex2_id: Option<cranelift_module::FuncId>,
    simplex3_id: Option<cranelift_module::FuncId>,
    fbm2_id: Option<cranelift_module::FuncId>,
    fbm3_id: Option<cranelift_module::FuncId>,
}

// =============================================================================
// Permutation table for noise (Ken Perlin's reference implementation)
// =============================================================================

const PERM: [u8; 256] = [
    151, 160, 137, 91, 90, 15, 131, 13, 201, 95, 96, 53, 194, 233, 7, 225, 140, 36, 103, 30, 69,
    142, 8, 99, 37, 240, 21, 10, 23, 190, 6, 148, 247, 120, 234, 75, 0, 26, 197, 62, 94, 252, 219,
    203, 117, 35, 11, 32, 57, 177, 33, 88, 237, 149, 56, 87, 174, 20, 125, 136, 171, 168, 68, 175,
    74, 165, 71, 134, 139, 48, 27, 166, 77, 146, 158, 231, 83, 111, 229, 122, 60, 211, 133, 230,
    220, 105, 92, 41, 55, 46, 245, 40, 244, 102, 143, 54, 65, 25, 63, 161, 1, 216, 80, 73, 209, 76,
    132, 187, 208, 89, 18, 169, 200, 196, 135, 130, 116, 188, 159, 86, 164, 100, 109, 198, 173,
    186, 3, 64, 52, 217, 226, 250, 124, 123, 5, 202, 38, 147, 118, 126, 255, 82, 85, 212, 207, 206,
    59, 227, 47, 16, 58, 17, 182, 189, 28, 42, 223, 183, 170, 213, 119, 248, 152, 2, 44, 154, 163,
    70, 221, 153, 101, 155, 167, 43, 172, 9, 129, 22, 39, 253, 19, 98, 108, 110, 79, 113, 224, 232,
    178, 185, 112, 104, 218, 246, 97, 228, 251, 34, 242, 193, 238, 210, 144, 12, 191, 179, 162,
    241, 81, 51, 145, 235, 249, 14, 239, 107, 49, 192, 214, 31, 181, 199, 106, 157, 184, 84, 204,
    176, 115, 121, 50, 45, 127, 4, 150, 254, 138, 236, 205, 93, 222, 114, 67, 29, 24, 72, 243, 141,
    128, 195, 78, 66, 215, 61, 156, 180,
];

// =============================================================================
// Pure Cranelift noise implementation (no Rust boundary crossing)
// =============================================================================

/// Emit perm(x) = PERM[x & 255].
/// Uses a lookup table embedded as constants since Cranelift select chains are
/// actually well-optimized and we avoid data section complexity.
fn emit_perm(builder: &mut FunctionBuilder, x: Value) -> Value {
    // Mask to 0-255
    let mask = builder.ins().iconst(types::I32, 255);
    let idx = builder.ins().band(x, mask);

    // Convert to u64 for table lookup
    let idx_u64 = builder.ins().uextend(types::I64, idx);

    // Use the PERM table pointer as an immediate and do a load
    // This is safe because PERM is static and won't move
    let table_addr = PERM.as_ptr() as i64;
    let base = builder.ins().iconst(types::I64, table_addr);
    let addr = builder.ins().iadd(base, idx_u64);

    // Load u8 and extend to i32
    let loaded = builder
        .ins()
        .load(types::I8, cranelift::ir::MemFlags::trusted(), addr, 0);
    builder.ins().uextend(types::I32, loaded)
}

/// Emit grad2(hash, x, y) - gradient function for 2D Perlin noise.
fn emit_grad2(builder: &mut FunctionBuilder, hash: Value, x: Value, y: Value) -> Value {
    // h = hash & 7
    let seven = builder.ins().iconst(types::I32, 7);
    let h = builder.ins().band(hash, seven);

    // u = h < 4 ? x : y
    let four = builder.ins().iconst(types::I32, 4);
    let h_lt_4 = builder
        .ins()
        .icmp(cranelift::ir::condcodes::IntCC::SignedLessThan, h, four);
    let u = builder.ins().select(h_lt_4, x, y);

    // v = h < 4 ? y : x
    let v = builder.ins().select(h_lt_4, y, x);

    // h & 1 != 0 ? -u : u
    let one = builder.ins().iconst(types::I32, 1);
    let h_and_1 = builder.ins().band(h, one);
    let zero = builder.ins().iconst(types::I32, 0);
    let h1_set = builder
        .ins()
        .icmp(cranelift::ir::condcodes::IntCC::NotEqual, h_and_1, zero);
    let neg_u = builder.ins().fneg(u);
    let u_term = builder.ins().select(h1_set, neg_u, u);

    // h & 2 != 0 ? -2*v : 2*v
    let two_i = builder.ins().iconst(types::I32, 2);
    let two_f = builder.ins().f32const(2.0);
    let h_and_2 = builder.ins().band(h, two_i);
    let h2_set = builder
        .ins()
        .icmp(cranelift::ir::condcodes::IntCC::NotEqual, h_and_2, zero);
    let v2 = builder.ins().fmul(two_f, v);
    let neg_v2 = builder.ins().fneg(v2);
    let v_term = builder.ins().select(h2_set, neg_v2, v2);

    builder.ins().fadd(u_term, v_term)
}

/// Emit fade(t) = t³(t(t*6 - 15) + 10) - smoothstep for Perlin noise.
fn emit_fade(builder: &mut FunctionBuilder, t: Value) -> Value {
    let six = builder.ins().f32const(6.0);
    let fifteen = builder.ins().f32const(15.0);
    let ten = builder.ins().f32const(10.0);

    // t * 6 - 15
    let t6 = builder.ins().fmul(t, six);
    let t6_m15 = builder.ins().fsub(t6, fifteen);

    // t * (t * 6 - 15) + 10
    let inner = builder.ins().fmul(t, t6_m15);
    let inner = builder.ins().fadd(inner, ten);

    // t³ * inner
    let t2 = builder.ins().fmul(t, t);
    let t3 = builder.ins().fmul(t2, t);
    builder.ins().fmul(t3, inner)
}

/// Emit lerp(a, b, t) = a + t * (b - a).
fn emit_lerp(builder: &mut FunctionBuilder, a: Value, b: Value, t: Value) -> Value {
    let diff = builder.ins().fsub(b, a);
    let scaled = builder.ins().fmul(t, diff);
    builder.ins().fadd(a, scaled)
}

/// Emit complete perlin2(x, y) as pure Cranelift IR.
/// Returns noise value in [0, 1].
fn emit_perlin2(builder: &mut FunctionBuilder, x: Value, y: Value) -> Value {
    // xi = floor(x), yi = floor(y)
    let x_floor = builder.ins().floor(x);
    let y_floor = builder.ins().floor(y);
    let xi = builder.ins().fcvt_to_sint(types::I32, x_floor);
    let yi = builder.ins().fcvt_to_sint(types::I32, y_floor);

    // xf = x - floor(x), yf = y - floor(y) (fractional parts)
    let xf = builder.ins().fsub(x, x_floor);
    let yf = builder.ins().fsub(y, y_floor);

    // Fade curves
    let u = emit_fade(builder, xf);
    let v = emit_fade(builder, yf);

    // Hash coordinates of corners
    // aa = perm(perm(xi) + yi)
    // ab = perm(perm(xi) + yi + 1)
    // ba = perm(perm(xi + 1) + yi)
    // bb = perm(perm(xi + 1) + yi + 1)
    let one_i = builder.ins().iconst(types::I32, 1);
    let xi_p1 = builder.ins().iadd(xi, one_i);

    let perm_xi = emit_perm(builder, xi);
    let perm_xi_p1 = emit_perm(builder, xi_p1);

    let pxi_yi = builder.ins().iadd(perm_xi, yi);
    let pxi_yi_p1 = builder.ins().iadd(pxi_yi, one_i);
    let pxi1_yi = builder.ins().iadd(perm_xi_p1, yi);
    let pxi1_yi_p1 = builder.ins().iadd(pxi1_yi, one_i);

    let aa = emit_perm(builder, pxi_yi);
    let ab = emit_perm(builder, pxi_yi_p1);
    let ba = emit_perm(builder, pxi1_yi);
    let bb = emit_perm(builder, pxi1_yi_p1);

    // Gradient values at corners
    let one_f = builder.ins().f32const(1.0);
    let xf_m1 = builder.ins().fsub(xf, one_f);
    let yf_m1 = builder.ins().fsub(yf, one_f);

    let g_aa = emit_grad2(builder, aa, xf, yf);
    let g_ba = emit_grad2(builder, ba, xf_m1, yf);
    let g_ab = emit_grad2(builder, ab, xf, yf_m1);
    let g_bb = emit_grad2(builder, bb, xf_m1, yf_m1);

    // Bilinear interpolation
    let x1 = emit_lerp(builder, g_aa, g_ba, u);
    let x2 = emit_lerp(builder, g_ab, g_bb, u);
    let result = emit_lerp(builder, x1, x2, v);

    // Scale to [0, 1]
    let half = builder.ins().f32const(0.5);
    let scaled = builder.ins().fmul(result, half);
    let shifted = builder.ins().fadd(scaled, half);

    // Clamp to [0, 1]
    let zero = builder.ins().f32const(0.0);
    let clamped_low = builder.ins().fmax(shifted, zero);
    builder.ins().fmin(clamped_low, one_f)
}

// =============================================================================
// Polynomial approximations for transcendental functions
// =============================================================================

/// Emit sin(x) using proper range reduction and minimax polynomial.
/// Reduces to [-π/4, π/4] and handles quadrants.
/// Max error: ~1e-4 for all inputs.
fn emit_sin(builder: &mut FunctionBuilder, x: Value) -> Value {
    // Use Cody-Waite range reduction for better accuracy
    // First reduce to [-π, π], then to [-π/4, π/4] with quadrant handling

    let pi = builder.ins().f32const(std::f32::consts::PI);
    let half_pi = builder.ins().f32const(std::f32::consts::FRAC_PI_2);
    let two_pi = builder.ins().f32const(2.0 * std::f32::consts::PI);
    let inv_two_pi = builder.ins().f32const(1.0 / (2.0 * std::f32::consts::PI));
    let one = builder.ins().f32const(1.0);

    // Range reduce to [-π, π]: x_red = x - 2π * round(x / 2π)
    let scaled = builder.ins().fmul(x, inv_two_pi);
    let rounded = builder.ins().nearest(scaled);
    let offset = builder.ins().fmul(rounded, two_pi);
    let x_mod = builder.ins().fsub(x, offset);

    // Now x_mod is in [-π, π]
    // For |x_mod| > π/2, use sin(x) = sin(π - x) for positive, sin(x) = -sin(-π - x) for negative
    // This folds everything to [-π/2, π/2]

    // abs_x = |x_mod|
    let abs_x = builder.ins().fabs(x_mod);

    // Check if |x_mod| > π/2
    let needs_fold = builder.ins().fcmp(
        cranelift::ir::condcodes::FloatCC::GreaterThan,
        abs_x,
        half_pi,
    );

    // If needs_fold: x_folded = sign(x_mod) * (π - |x_mod|)
    let sign_x = builder.ins().fcopysign(one, x_mod);
    let pi_minus_abs = builder.ins().fsub(pi, abs_x);
    let x_folded_candidate = builder.ins().fmul(sign_x, pi_minus_abs);
    let x_for_poly = builder.ins().select(needs_fold, x_folded_candidate, x_mod);

    // Now x_for_poly is in [-π/2, π/2], use minimax polynomial
    // sin(x) ≈ x * (1 + x² * (c3 + x² * (c5 + x² * c7)))
    // Minimax coefficients for [-π/2, π/2]:
    let c3 = builder.ins().f32const(-0.16666667); // -1/6
    let c5 = builder.ins().f32const(0.0083333310); // ~1/120
    let c7 = builder.ins().f32const(-0.00019840874); // ~-1/5040
    let c9 = builder.ins().f32const(2.7525562e-06); // ~1/362880

    let x2 = builder.ins().fmul(x_for_poly, x_for_poly);

    // Horner's method: c3 + x² * (c5 + x² * (c7 + x² * c9))
    let inner = builder.ins().fmul(x2, c9);
    let inner = builder.ins().fadd(inner, c7);
    let inner = builder.ins().fmul(x2, inner);
    let inner = builder.ins().fadd(inner, c5);
    let inner = builder.ins().fmul(x2, inner);
    let inner = builder.ins().fadd(inner, c3);
    let inner = builder.ins().fmul(x2, inner);
    let inner = builder.ins().fadd(inner, one);

    builder.ins().fmul(x_for_poly, inner)
}

/// Emit cos(x) using sin(x + π/2).
fn emit_cos(builder: &mut FunctionBuilder, x: Value) -> Value {
    let half_pi = builder.ins().f32const(std::f32::consts::FRAC_PI_2);
    let x_shifted = builder.ins().fadd(x, half_pi);
    emit_sin(builder, x_shifted)
}

/// Emit exp(x) using range reduction and polynomial.
/// exp(x) = 2^(x * log2(e)) = 2^k * 2^f where k=floor(x*log2(e)), f=frac
/// For 2^f with f in [0,1), use polynomial approximation.
fn emit_exp(builder: &mut FunctionBuilder, x: Value) -> Value {
    let log2_e = builder.ins().f32const(std::f32::consts::LOG2_E); // 1.4426950408889634
    let one = builder.ins().f32const(1.0);

    // y = x * log2(e)
    let y = builder.ins().fmul(x, log2_e);

    // k = floor(y), f = y - k
    let k = builder.ins().floor(y);
    let f = builder.ins().fsub(y, k);

    // 2^f for f in [0, 1) using polynomial approximation
    // 2^f ≈ 1 + f*ln(2) + f²*ln²(2)/2 + f³*ln³(2)/6 + ...
    // Or use optimized minimax: 2^f ≈ 1 + c1*f + c2*f² + c3*f³ + c4*f⁴
    // Coefficients for minimax on [0, 1]:
    let c1 = builder.ins().f32const(0.6931472); // ln(2)
    let c2 = builder.ins().f32const(0.2402265); // ln²(2)/2
    let c3 = builder.ins().f32const(0.0555041); // ln³(2)/6
    let c4 = builder.ins().f32const(0.0096139); // ln⁴(2)/24
    let c5 = builder.ins().f32const(0.0013498); // ln⁵(2)/120

    let f2 = builder.ins().fmul(f, f);
    let f3 = builder.ins().fmul(f2, f);
    let f4 = builder.ins().fmul(f3, f);
    let f5 = builder.ins().fmul(f4, f);

    let t1 = builder.ins().fmul(c1, f);
    let t2 = builder.ins().fmul(c2, f2);
    let t3 = builder.ins().fmul(c3, f3);
    let t4 = builder.ins().fmul(c4, f4);
    let t5 = builder.ins().fmul(c5, f5);

    let sum1 = builder.ins().fadd(one, t1);
    let sum2 = builder.ins().fadd(sum1, t2);
    let sum3 = builder.ins().fadd(sum2, t3);
    let sum4 = builder.ins().fadd(sum3, t4);
    let exp_f = builder.ins().fadd(sum4, t5);

    // 2^k: convert k to int, then use scalbn-like operation
    // Cranelift doesn't have ldexp, so we build 2^k manually via bit manipulation
    // 2^k = bitcast((k + 127) << 23) for f32
    let k_i32 = builder.ins().fcvt_to_sint(types::I32, k);
    let bias = builder.ins().iconst(types::I32, 127);
    let shift = builder.ins().iconst(types::I32, 23);
    let biased = builder.ins().iadd(k_i32, bias);
    let exp_bits = builder.ins().ishl(biased, shift);
    let two_pow_k = builder
        .ins()
        .bitcast(types::F32, cranelift::ir::MemFlags::new(), exp_bits);

    // Result: 2^k * 2^f
    builder.ins().fmul(two_pow_k, exp_f)
}

/// Emit ln(x) using range reduction.
/// ln(x) = ln(2^k * m) = k*ln(2) + ln(m) where m in [1, 2)
/// For ln(m), use polynomial approximation.
fn emit_ln(builder: &mut FunctionBuilder, x: Value) -> Value {
    let ln2 = builder.ins().f32const(std::f32::consts::LN_2);
    let one = builder.ins().f32const(1.0);

    // Extract exponent and mantissa from x
    // x = 2^k * m where 1 <= m < 2
    // k = floor(log2(x)), m = x / 2^k
    let x_bits = builder
        .ins()
        .bitcast(types::I32, cranelift::ir::MemFlags::new(), x);
    let exp_mask = builder.ins().iconst(types::I32, 0x7F80_0000_u32 as i64);
    let mant_mask = builder.ins().iconst(types::I32, 0x007F_FFFF_u32 as i64);
    let bias = builder.ins().iconst(types::I32, 127);
    let shift = builder.ins().iconst(types::I32, 23);
    let one_bits = builder.ins().iconst(types::I32, 0x3F80_0000_u32 as i64); // 1.0f32 bits

    let exp_bits = builder.ins().band(x_bits, exp_mask);
    let k_biased = builder.ins().ushr(exp_bits, shift);
    let k_i32 = builder.ins().isub(k_biased, bias);
    let k = builder.ins().fcvt_from_sint(types::F32, k_i32);

    // m = mantissa with exponent = 127 (i.e., 1.xxx)
    let mant_bits = builder.ins().band(x_bits, mant_mask);
    let m_bits = builder.ins().bor(mant_bits, one_bits);
    let m = builder
        .ins()
        .bitcast(types::F32, cranelift::ir::MemFlags::new(), m_bits);

    // ln(m) for m in [1, 2) using polynomial
    // Let t = m - 1, then ln(1+t) ≈ t - t²/2 + t³/3 - t⁴/4 + t⁵/5
    let t = builder.ins().fsub(m, one);

    let c1 = builder.ins().f32const(1.0);
    let c2 = builder.ins().f32const(-0.5);
    let c3 = builder.ins().f32const(0.33333333);
    let c4 = builder.ins().f32const(-0.25);
    let c5 = builder.ins().f32const(0.2);

    let t2 = builder.ins().fmul(t, t);
    let t3 = builder.ins().fmul(t2, t);
    let t4 = builder.ins().fmul(t3, t);
    let t5 = builder.ins().fmul(t4, t);

    let p1 = builder.ins().fmul(c1, t);
    let p2 = builder.ins().fmul(c2, t2);
    let p3 = builder.ins().fmul(c3, t3);
    let p4 = builder.ins().fmul(c4, t4);
    let p5 = builder.ins().fmul(c5, t5);

    let sum1 = builder.ins().fadd(p1, p2);
    let sum2 = builder.ins().fadd(sum1, p3);
    let sum3 = builder.ins().fadd(sum2, p4);
    let ln_m = builder.ins().fadd(sum3, p5);

    // result = k * ln(2) + ln(m)
    let k_ln2 = builder.ins().fmul(k, ln2);
    builder.ins().fadd(k_ln2, ln_m)
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
            // Use pure Cranelift implementation (no Rust boundary)
            Ok(emit_perlin2(ctx.builder, vx, vy))
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

        // Transcendentals - polynomial approximations
        FieldExpr::Sin(a) => {
            let va = compile_expr(a, ctx)?;
            Ok(emit_sin(ctx.builder, va))
        }
        FieldExpr::Cos(a) => {
            let va = compile_expr(a, ctx)?;
            Ok(emit_cos(ctx.builder, va))
        }
        FieldExpr::Tan(a) => {
            let va = compile_expr(a, ctx)?;
            let sin_val = emit_sin(ctx.builder, va);
            let cos_val = emit_cos(ctx.builder, va);
            Ok(ctx.builder.ins().fdiv(sin_val, cos_val))
        }
        FieldExpr::Exp(a) => {
            let va = compile_expr(a, ctx)?;
            Ok(emit_exp(ctx.builder, va))
        }
        FieldExpr::Ln(a) => {
            let va = compile_expr(a, ctx)?;
            Ok(emit_ln(ctx.builder, va))
        }

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

    #[test]
    fn test_sin_approximation() {
        let mut compiler = FieldExprCompiler::new().unwrap();
        let expr = FieldExpr::Sin(Box::new(FieldExpr::X));
        let compiled = compiler.compile(&expr).unwrap();

        // Test at various points - tolerance 0.05 is fine for procedural graphics
        for i in -20..=20 {
            let x = i as f32 * 0.5; // Range -10 to 10
            let jit_val = compiled.eval(x, 0.0, 0.0, 0.0);
            let expected = x.sin();
            let err = (jit_val - expected).abs();
            assert!(
                err < 0.05,
                "sin({}) = {} (expected {}), error = {}",
                x,
                jit_val,
                expected,
                err
            );
        }
    }

    #[test]
    fn test_cos_approximation() {
        let mut compiler = FieldExprCompiler::new().unwrap();
        let expr = FieldExpr::Cos(Box::new(FieldExpr::X));
        let compiled = compiler.compile(&expr).unwrap();

        for i in -20..=20 {
            let x = i as f32 * 0.5;
            let jit_val = compiled.eval(x, 0.0, 0.0, 0.0);
            let expected = x.cos();
            let err = (jit_val - expected).abs();
            assert!(
                err < 0.05,
                "cos({}) = {} (expected {}), error = {}",
                x,
                jit_val,
                expected,
                err
            );
        }
    }

    #[test]
    fn test_tan_approximation() {
        let mut compiler = FieldExprCompiler::new().unwrap();
        let expr = FieldExpr::Tan(Box::new(FieldExpr::X));
        let compiled = compiler.compile(&expr).unwrap();

        // Avoid values near π/2 where tan explodes
        for i in -10..=10 {
            let x = i as f32 * 0.1; // Range -1 to 1, safe for tan
            let jit_val = compiled.eval(x, 0.0, 0.0, 0.0);
            let expected = x.tan();
            let err = (jit_val - expected).abs();
            assert!(
                err < 0.05,
                "tan({}) = {} (expected {}), error = {}",
                x,
                jit_val,
                expected,
                err
            );
        }
    }

    #[test]
    fn test_exp_approximation() {
        let mut compiler = FieldExprCompiler::new().unwrap();
        let expr = FieldExpr::Exp(Box::new(FieldExpr::X));
        let compiled = compiler.compile(&expr).unwrap();

        // Test range -5 to 5
        for i in -10..=10 {
            let x = i as f32 * 0.5;
            let jit_val = compiled.eval(x, 0.0, 0.0, 0.0);
            let expected = x.exp();
            let rel_err = if expected.abs() > 1e-6 {
                (jit_val - expected).abs() / expected.abs()
            } else {
                (jit_val - expected).abs()
            };
            assert!(
                rel_err < 0.01,
                "exp({}) = {} (expected {}), rel_error = {}",
                x,
                jit_val,
                expected,
                rel_err
            );
        }
    }

    #[test]
    fn test_ln_approximation() {
        let mut compiler = FieldExprCompiler::new().unwrap();
        let expr = FieldExpr::Ln(Box::new(FieldExpr::X));
        let compiled = compiler.compile(&expr).unwrap();

        // Test positive values
        for i in 1..=20 {
            let x = i as f32 * 0.5; // Range 0.5 to 10
            let jit_val = compiled.eval(x, 0.0, 0.0, 0.0);
            let expected = x.ln();
            let err = (jit_val - expected).abs();
            assert!(
                err < 0.05, // ln approximation has larger error
                "ln({}) = {} (expected {}), error = {}",
                x,
                jit_val,
                expected,
                err
            );
        }
    }

    #[test]
    fn test_combined_trig() {
        let mut compiler = FieldExprCompiler::new().unwrap();
        // sin(x)² + cos(x)² should equal 1
        let expr = FieldExpr::Add(
            Box::new(FieldExpr::Mul(
                Box::new(FieldExpr::Sin(Box::new(FieldExpr::X))),
                Box::new(FieldExpr::Sin(Box::new(FieldExpr::X))),
            )),
            Box::new(FieldExpr::Mul(
                Box::new(FieldExpr::Cos(Box::new(FieldExpr::X))),
                Box::new(FieldExpr::Cos(Box::new(FieldExpr::X))),
            )),
        );
        let compiled = compiler.compile(&expr).unwrap();

        for i in -20..=20 {
            let x = i as f32 * 0.5;
            let result = compiled.eval(x, 0.0, 0.0, 0.0);
            assert!(
                (result - 1.0).abs() < 0.02,
                "sin²({}) + cos²({}) = {} (expected 1.0)",
                x,
                x,
                result
            );
        }
    }

    #[test]
    fn test_pure_cranelift_perlin2_parity() {
        // Test that pure Cranelift perlin2 matches the Rust implementation exactly
        let mut compiler = FieldExprCompiler::new().unwrap();
        let expr = FieldExpr::Perlin2 {
            x: Box::new(FieldExpr::X),
            y: Box::new(FieldExpr::Y),
        };
        let compiled = compiler.compile(&expr).unwrap();

        // Test many points to ensure exact parity
        for i in 0..50 {
            for j in 0..50 {
                let x = i as f32 * 0.2 - 5.0; // Range -5 to 5
                let y = j as f32 * 0.2 - 5.0;

                let rust_val = rhizome_resin_noise::perlin2(x, y);
                let jit_val = compiled.eval(x, y, 0.0, 0.0);

                assert!(
                    (rust_val - jit_val).abs() < 1e-5,
                    "Perlin2 mismatch at ({}, {}): rust={}, jit={}",
                    x,
                    y,
                    rust_val,
                    jit_val
                );
            }
        }
    }
}
