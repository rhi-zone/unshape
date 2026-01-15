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
/// use rhizome_resin_jit::{JitCompiler, JitConfig};
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
