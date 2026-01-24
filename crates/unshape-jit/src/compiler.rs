//! JIT compiler for graph types.

#[cfg(feature = "cranelift")]
use cranelift::Context;
#[cfg(feature = "cranelift")]
use cranelift::settings::{self, Configurable};
#[cfg(feature = "cranelift")]
use cranelift_frontend::FunctionBuilderContext;
#[cfg(feature = "cranelift")]
use cranelift_jit::{JITBuilder, JITModule};
#[cfg(feature = "cranelift")]
use cranelift_module::{Linkage, Module};

#[cfg(feature = "cranelift")]
use crate::compiled::CompiledScalar;
use crate::error::{JitError, JitResult};
use crate::traits::SimdWidth;

/// Optimization level for Cranelift.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OptLevel {
    /// No optimization (fastest compilation).
    None,
    /// Balanced optimization.
    #[default]
    Speed,
    /// Maximum optimization (slowest compilation).
    SpeedAndSize,
}

/// Configuration for JIT compilation.
#[derive(Debug, Clone)]
pub struct JitConfig {
    /// Optimization level for Cranelift.
    pub opt_level: OptLevel,
    /// Whether to enable SIMD code generation.
    pub enable_simd: bool,
    /// Preferred SIMD width when enabled.
    pub simd_width: SimdWidth,
}

impl Default for JitConfig {
    fn default() -> Self {
        Self {
            opt_level: OptLevel::Speed,
            enable_simd: true,
            simd_width: SimdWidth::X4,
        }
    }
}

/// JIT compiler for graph types.
///
/// Creates native code from graph structures using Cranelift.
///
/// # Example
///
/// ```ignore
/// use unshape_jit::{JitCompiler, JitConfig};
///
/// let mut compiler = JitCompiler::new(JitConfig::default())?;
/// let compiled = compiler.compile(&graph)?;
/// let output = compiled.call_f32(input);
/// ```
#[cfg(feature = "cranelift")]
pub struct JitCompiler {
    /// Cranelift JIT module.
    module: JITModule,
    /// Function builder context (reused across compilations).
    builder_ctx: FunctionBuilderContext,
    /// Cranelift context (reused across compilations).
    ctx: Context,
    /// Compiler configuration.
    config: JitConfig,
    /// Counter for unique function names.
    func_counter: usize,
}

#[cfg(feature = "cranelift")]
impl JitCompiler {
    /// Creates a new JIT compiler with the given configuration.
    pub fn new(config: JitConfig) -> JitResult<Self> {
        let mut flag_builder = settings::builder();

        let opt_str = match config.opt_level {
            OptLevel::None => "none",
            OptLevel::Speed => "speed",
            OptLevel::SpeedAndSize => "speed_and_size",
        };
        flag_builder.set("opt_level", opt_str).unwrap();

        let isa_builder =
            cranelift_native::builder().map_err(|e| JitError::Compilation(e.to_string()))?;
        let isa = isa_builder
            .finish(settings::Flags::new(flag_builder))
            .map_err(|e| JitError::Compilation(e.to_string()))?;

        let builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());
        let module = JITModule::new(builder);
        let ctx = module.make_context();

        Ok(Self {
            module,
            builder_ctx: FunctionBuilderContext::new(),
            ctx,
            config,
            func_counter: 0,
        })
    }

    /// Creates a new JIT compiler with default configuration.
    pub fn with_defaults() -> JitResult<Self> {
        Self::new(JitConfig::default())
    }

    /// Returns the compiler configuration.
    pub fn config(&self) -> &JitConfig {
        &self.config
    }

    /// Generates a unique function name.
    fn next_func_name(&mut self, prefix: &str) -> String {
        let name = format!("{}_{}", prefix, self.func_counter);
        self.func_counter += 1;
        name
    }

    /// Compiles a simple scalar function: `output = input * gain + offset`.
    ///
    /// This is a convenience method for the common affine transform case.
    pub fn compile_affine(&mut self, gain: f32, offset: f32) -> JitResult<CompiledScalar> {
        use cranelift::ir::{AbiParam, InstBuilder, types};
        use cranelift_frontend::FunctionBuilder;

        // Signature: fn(f32) -> f32
        let mut sig = self.module.make_signature();
        sig.params.push(AbiParam::new(types::F32));
        sig.returns.push(AbiParam::new(types::F32));

        let func_name = self.next_func_name("jit_affine");
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

            let input = builder.block_params(entry)[0];

            // Optimize based on values
            let is_identity_gain = (gain - 1.0).abs() < 1e-10;
            let is_zero_offset = offset.abs() < 1e-10;

            let output = match (is_identity_gain, is_zero_offset) {
                (true, true) => input,
                (true, false) => {
                    let o = builder.ins().f32const(offset);
                    builder.ins().fadd(input, o)
                }
                (false, true) => {
                    let g = builder.ins().f32const(gain);
                    builder.ins().fmul(input, g)
                }
                (false, false) => {
                    let g = builder.ins().f32const(gain);
                    let o = builder.ins().f32const(offset);
                    let mul = builder.ins().fmul(input, g);
                    builder.ins().fadd(mul, o)
                }
            };

            builder.ins().return_(&[output]);
            builder.finalize();
        }

        self.module.define_function(func_id, &mut self.ctx)?;
        self.module.clear_context(&mut self.ctx);
        self.module.finalize_definitions()?;

        let code_ptr = self.module.get_finalized_function(func_id);

        // Safety: The code lives in self.module. Caller must ensure compiler outlives result.
        Ok(unsafe { CompiledScalar::new(code_ptr, 1, 1) })
    }

    /// Compiles a SIMD block processing function for an affine transform.
    ///
    /// Processes 4 samples at a time using f32x4 SIMD operations.
    /// Signature: `fn(input: *const f32, output: *mut f32, len: usize)`
    ///
    /// # Returns
    ///
    /// A `CompiledSimdBlock` that processes buffers of samples.
    pub fn compile_affine_simd(&mut self, gain: f32, offset: f32) -> JitResult<CompiledSimdBlock> {
        use cranelift::ir::{AbiParam, InstBuilder, MemFlags, condcodes::IntCC, types};
        use cranelift_frontend::FunctionBuilder;

        // Signature: fn(input: *const f32, output: *mut f32, len: usize)
        let ptr_type = types::I64;
        let mut sig = self.module.make_signature();
        sig.params.push(AbiParam::new(ptr_type)); // input ptr
        sig.params.push(AbiParam::new(ptr_type)); // output ptr
        sig.params.push(AbiParam::new(ptr_type)); // length

        let func_name = self.next_func_name("jit_affine_simd");
        let func_id = self
            .module
            .declare_function(&func_name, Linkage::Export, &sig)?;

        self.ctx.func.signature = sig;

        {
            let mut builder = FunctionBuilder::new(&mut self.ctx.func, &mut self.builder_ctx);

            // Create blocks
            let entry = builder.create_block();
            let simd_loop = builder.create_block();
            let simd_body = builder.create_block();
            let scalar_loop = builder.create_block();
            let scalar_body = builder.create_block();
            let exit = builder.create_block();

            builder.append_block_params_for_function_params(entry);
            builder.switch_to_block(entry);

            let input_ptr = builder.block_params(entry)[0];
            let output_ptr = builder.block_params(entry)[1];
            let len = builder.block_params(entry)[2];

            // Constants
            let zero = builder.ins().iconst(ptr_type, 0);
            let four = builder.ins().iconst(ptr_type, 4);
            let sixteen = builder.ins().iconst(ptr_type, 16); // 4 * sizeof(f32)

            // SIMD constants - splat gain and offset to f32x4
            let is_identity_gain = (gain - 1.0).abs() < 1e-10;
            let is_zero_offset = offset.abs() < 1e-10;

            let gain_scalar = builder.ins().f32const(gain);
            let gain_vec = builder.ins().splat(types::F32X4, gain_scalar);
            let offset_scalar = builder.ins().f32const(offset);
            let offset_vec = builder.ins().splat(types::F32X4, offset_scalar);

            // Calculate SIMD iterations: len / 4
            let simd_count = builder.ins().udiv(len, four);
            let remainder_start = builder.ins().imul(simd_count, four);

            // Jump to SIMD loop
            builder.ins().jump(simd_loop, &[zero]);

            // SIMD loop header
            builder.switch_to_block(simd_loop);
            builder.append_block_param(simd_loop, ptr_type); // i (in units of 4)
            let simd_i = builder.block_params(simd_loop)[0];

            let simd_done =
                builder
                    .ins()
                    .icmp(IntCC::UnsignedGreaterThanOrEqual, simd_i, simd_count);
            builder
                .ins()
                .brif(simd_done, scalar_loop, &[remainder_start], simd_body, &[]);

            // SIMD loop body
            builder.switch_to_block(simd_body);

            // Calculate byte offset: i * 4 * 4 = i * 16
            let byte_offset = builder.ins().imul(simd_i, sixteen);
            let in_addr = builder.ins().iadd(input_ptr, byte_offset);
            let out_addr = builder.ins().iadd(output_ptr, byte_offset);

            // Load 4 floats as f32x4
            let mem = MemFlags::new();
            let input_vec = builder.ins().load(types::F32X4, mem, in_addr, 0);

            // Apply affine transform
            let output_vec = match (is_identity_gain, is_zero_offset) {
                (true, true) => input_vec,
                (true, false) => builder.ins().fadd(input_vec, offset_vec),
                (false, true) => builder.ins().fmul(input_vec, gain_vec),
                (false, false) => {
                    let mul = builder.ins().fmul(input_vec, gain_vec);
                    builder.ins().fadd(mul, offset_vec)
                }
            };

            // Store result
            builder.ins().store(mem, output_vec, out_addr, 0);

            // Increment and loop
            let one = builder.ins().iconst(ptr_type, 1);
            let next_i = builder.ins().iadd(simd_i, one);
            builder.ins().jump(simd_loop, &[next_i]);

            // Scalar loop header (for remainder)
            builder.switch_to_block(scalar_loop);
            builder.append_block_param(scalar_loop, ptr_type); // scalar index
            let scalar_i = builder.block_params(scalar_loop)[0];

            let scalar_done = builder
                .ins()
                .icmp(IntCC::UnsignedGreaterThanOrEqual, scalar_i, len);
            builder.ins().brif(scalar_done, exit, &[], scalar_body, &[]);

            // Scalar loop body
            builder.switch_to_block(scalar_body);

            let scalar_byte_offset = builder.ins().ishl_imm(scalar_i, 2); // * 4
            let scalar_in_addr = builder.ins().iadd(input_ptr, scalar_byte_offset);
            let scalar_out_addr = builder.ins().iadd(output_ptr, scalar_byte_offset);

            let scalar_input = builder.ins().load(types::F32, mem, scalar_in_addr, 0);

            let scalar_output = match (is_identity_gain, is_zero_offset) {
                (true, true) => scalar_input,
                (true, false) => builder.ins().fadd(scalar_input, offset_scalar),
                (false, true) => builder.ins().fmul(scalar_input, gain_scalar),
                (false, false) => {
                    let mul = builder.ins().fmul(scalar_input, gain_scalar);
                    builder.ins().fadd(mul, offset_scalar)
                }
            };

            builder.ins().store(mem, scalar_output, scalar_out_addr, 0);

            let next_scalar_i = builder.ins().iadd_imm(scalar_i, 1);
            builder.ins().jump(scalar_loop, &[next_scalar_i]);

            // Exit
            builder.switch_to_block(exit);
            builder.ins().return_(&[]);

            // Seal blocks
            builder.seal_block(entry);
            builder.seal_block(simd_loop);
            builder.seal_block(simd_body);
            builder.seal_block(scalar_loop);
            builder.seal_block(scalar_body);
            builder.seal_block(exit);

            builder.finalize();
        }

        self.module.define_function(func_id, &mut self.ctx)?;
        self.module.clear_context(&mut self.ctx);
        self.module.finalize_definitions()?;

        let code_ptr = self.module.get_finalized_function(func_id);

        Ok(unsafe { CompiledSimdBlock::new(code_ptr) })
    }
}

/// A JIT-compiled SIMD block processing function.
///
/// Processes arrays of f32 values using SIMD operations.
#[cfg(feature = "cranelift")]
pub struct CompiledSimdBlock {
    func: fn(*const f32, *mut f32, usize),
}

#[cfg(feature = "cranelift")]
impl CompiledSimdBlock {
    /// Creates a new compiled SIMD block function.
    ///
    /// # Safety
    ///
    /// The function pointer must be valid.
    pub(crate) unsafe fn new(func: *const u8) -> Self {
        Self {
            func: unsafe { std::mem::transmute(func) },
        }
    }

    /// Processes a block of samples.
    ///
    /// # Panics
    ///
    /// Panics if input and output lengths don't match.
    pub fn process(&self, input: &[f32], output: &mut [f32]) {
        assert_eq!(input.len(), output.len());
        (self.func)(input.as_ptr(), output.as_mut_ptr(), input.len());
    }
}

/// Stub compiler when cranelift feature is disabled.
#[cfg(not(feature = "cranelift"))]
pub struct JitCompiler {
    config: JitConfig,
}

#[cfg(not(feature = "cranelift"))]
impl JitCompiler {
    /// Creates a new JIT compiler (stub - cranelift feature required).
    pub fn new(config: JitConfig) -> JitResult<Self> {
        Ok(Self { config })
    }

    /// Returns the compiler configuration.
    pub fn config(&self) -> &JitConfig {
        &self.config
    }
}
