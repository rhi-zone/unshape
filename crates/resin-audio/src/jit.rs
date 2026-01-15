//! JIT compilation for audio graphs using Cranelift.
//!
//! This module provides experimental JIT compilation for `AudioGraph` instances,
//! eliminating dynamic dispatch and wire iteration overhead at runtime.
//!
//! # Status
//!
//! **EXPERIMENTAL** - This is a proof-of-concept exploring Cranelift JIT for audio.
//! The current implementation has significant limitations:
//!
//! - Only supports a subset of node types (gain, simple math)
//! - Doesn't handle stateful nodes (delay lines, filters) yet
//! - Requires unsafe code to call JIT-compiled functions
//!
//! # Example
//!
//! ```ignore
//! use rhizome_resin_audio::jit::JitCompiler;
//! use rhizome_resin_audio::graph::AudioGraph;
//!
//! let graph = /* build your graph */;
//! let mut compiler = JitCompiler::new()?;
//! let compiled = compiler.compile(&graph)?;
//!
//! // Process samples using JIT code
//! let output = unsafe { compiled.process(input) };
//! ```
//!
//! # When to use
//!
//! JIT compilation is beneficial when:
//! - The graph structure is fixed during processing
//! - Processing many samples (compilation has ~1-10ms latency)
//! - Dynamic dispatch overhead is measurable in your workload
//!
//! For most use cases, the optimized `AudioGraph` (control-rate, cached wire lookups)
//! is sufficient. Consider JIT only after profiling shows graph overhead.

#![cfg(feature = "cranelift")]

use cranelift::Context;
use cranelift::ir::types;
use cranelift::ir::{AbiParam, InstBuilder};
use cranelift::settings::{self, Configurable};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{Linkage, Module};
use std::mem;

/// Errors that can occur during JIT compilation.
#[derive(Debug, thiserror::Error)]
pub enum JitError {
    /// Cranelift module creation failed.
    #[error("module error: {0}")]
    Module(#[from] cranelift_module::ModuleError),

    /// Unsupported node type for JIT compilation.
    #[error("unsupported node type: {0}")]
    UnsupportedNode(String),

    /// Graph structure not suitable for JIT.
    #[error("graph error: {0}")]
    Graph(String),
}

/// Result type for JIT operations.
pub type JitResult<T> = Result<T, JitError>;

/// JIT compiler for audio graphs.
///
/// Creates native code from graph structures using Cranelift.
pub struct JitCompiler {
    /// Cranelift JIT module.
    module: JITModule,
    /// Function builder context (reused across compilations).
    builder_ctx: FunctionBuilderContext,
    /// Cranelift context (reused across compilations).
    ctx: Context,
}

impl JitCompiler {
    /// Creates a new JIT compiler.
    pub fn new() -> JitResult<Self> {
        let mut flag_builder = settings::builder();
        flag_builder.set("opt_level", "speed").unwrap();
        let isa_builder =
            cranelift_native::builder().map_err(|e| JitError::Graph(e.to_string()))?;
        let isa = isa_builder
            .finish(settings::Flags::new(flag_builder))
            .map_err(|e| JitError::Graph(e.to_string()))?;

        let builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());
        let module = JITModule::new(builder);
        let ctx = module.make_context();

        Ok(Self {
            module,
            builder_ctx: FunctionBuilderContext::new(),
            ctx,
        })
    }

    /// Compiles a simple gain graph to native code.
    ///
    /// This is a proof-of-concept that compiles: `output = input * gain`
    ///
    /// # Returns
    ///
    /// A `CompiledGain` that can process samples without dynamic dispatch.
    pub fn compile_gain(&mut self, gain: f32) -> JitResult<CompiledGain> {
        // Signature: fn(f32) -> f32
        let mut sig = self.module.make_signature();
        sig.params.push(AbiParam::new(types::F32));
        sig.returns.push(AbiParam::new(types::F32));

        let func_id = self
            .module
            .declare_function("jit_gain", Linkage::Export, &sig)?;

        self.ctx.func.signature = sig;

        {
            let mut builder = FunctionBuilder::new(&mut self.ctx.func, &mut self.builder_ctx);
            let entry = builder.create_block();
            builder.append_block_params_for_function_params(entry);
            builder.switch_to_block(entry);
            builder.seal_block(entry);

            // Get input parameter
            let input = builder.block_params(entry)[0];

            // Create constant for gain
            let gain_val = builder.ins().f32const(gain);

            // Multiply: output = input * gain
            let output = builder.ins().fmul(input, gain_val);

            // Return
            builder.ins().return_(&[output]);
            builder.finalize();
        }

        // Compile
        self.module.define_function(func_id, &mut self.ctx)?;
        self.module.clear_context(&mut self.ctx);
        self.module.finalize_definitions()?;

        // Get function pointer
        let code_ptr = self.module.get_finalized_function(func_id);
        let func: fn(f32) -> f32 = unsafe { mem::transmute(code_ptr) };

        Ok(CompiledGain { func })
    }

    /// Compiles a simple tremolo: `output = input * (base + lfo * scale)`
    ///
    /// The LFO value must be passed in each call (stateless from JIT perspective).
    pub fn compile_tremolo(&mut self, base: f32, scale: f32) -> JitResult<CompiledTremolo> {
        // Signature: fn(input: f32, lfo: f32) -> f32
        let mut sig = self.module.make_signature();
        sig.params.push(AbiParam::new(types::F32)); // input
        sig.params.push(AbiParam::new(types::F32)); // lfo
        sig.returns.push(AbiParam::new(types::F32));

        let func_id = self
            .module
            .declare_function("jit_tremolo", Linkage::Export, &sig)?;

        self.ctx.func.signature = sig;

        {
            let mut builder = FunctionBuilder::new(&mut self.ctx.func, &mut self.builder_ctx);
            let entry = builder.create_block();
            builder.append_block_params_for_function_params(entry);
            builder.switch_to_block(entry);
            builder.seal_block(entry);

            let input = builder.block_params(entry)[0];
            let lfo = builder.block_params(entry)[1];

            // Constants
            let base_val = builder.ins().f32const(base);
            let scale_val = builder.ins().f32const(scale);

            // gain = base + lfo * scale
            let scaled = builder.ins().fmul(lfo, scale_val);
            let gain = builder.ins().fadd(base_val, scaled);

            // output = input * gain
            let output = builder.ins().fmul(input, gain);

            builder.ins().return_(&[output]);
            builder.finalize();
        }

        self.module.define_function(func_id, &mut self.ctx)?;
        self.module.clear_context(&mut self.ctx);
        self.module.finalize_definitions()?;

        let code_ptr = self.module.get_finalized_function(func_id);
        let func: fn(f32, f32) -> f32 = unsafe { mem::transmute(code_ptr) };

        Ok(CompiledTremolo { func, base, scale })
    }
}

/// JIT-compiled gain function.
pub struct CompiledGain {
    func: fn(f32) -> f32,
}

impl CompiledGain {
    /// Processes a sample through the JIT-compiled gain.
    #[inline]
    pub fn process(&self, input: f32) -> f32 {
        (self.func)(input)
    }
}

/// JIT-compiled tremolo function.
///
/// Requires LFO value to be computed externally and passed in.
pub struct CompiledTremolo {
    func: fn(f32, f32) -> f32,
    /// Base gain value.
    pub base: f32,
    /// Scale for LFO modulation.
    pub scale: f32,
}

impl CompiledTremolo {
    /// Processes a sample with the given LFO value.
    #[inline]
    pub fn process(&self, input: f32, lfo: f32) -> f32 {
        (self.func)(input, lfo)
    }
}

// ============================================================================
// Feasibility Notes
// ============================================================================

/// # Cranelift JIT Feasibility Analysis
///
/// ## What works well
///
/// - Pure math operations (add, mul, div) compile trivially
/// - Function signatures are straightforward
/// - Generated code quality is good (Cranelift targets native ISA)
/// - Compilation latency is ~1-5ms for simple functions
///
/// ## Challenges
///
/// 1. **Stateful nodes** - Delay lines, filters need persistent buffers.
///    Options:
///    - Pass buffer pointers as function arguments (complex signatures)
///    - Embed buffers in JIT data section (requires careful memory management)
///    - Keep stateful nodes as Rust code, JIT only the math
///
/// 2. **Complex control flow** - Modulation routing creates variable data flow.
///    The graph's wire structure needs to be "baked" into generated code.
///
/// 3. **Node diversity** - Each AudioNode implementation would need a
///    Cranelift code generator. This is significant effort for ~20 node types.
///
/// 4. **Debugging** - JIT code is harder to debug than Rust code.
///    Stack traces don't work normally through JIT boundaries.
///
/// ## Recommendation
///
/// Cranelift JIT is **feasible but high effort** for full AudioGraph support.
/// Consider instead:
///
/// 1. **Proc macro compilation** - Generate Rust code from graph at compile time.
///    Benefits: normal debugging, LTO optimization, type safety.
///    Drawback: graphs must be known at compile time.
///
/// 2. **Pre-monomorphized compositions** - Feature-gated concrete types.
///    Benefits: zero overhead, normal Rust code, easy to debug.
///    Drawback: only covers "blessed" effect patterns.
///
/// 3. **Hybrid approach** - JIT only the innermost loop (param calculation),
///    keep node processing in Rust. Reduces complexity while eliminating
///    the most significant overhead (set_param calls in inner loop).
///
/// ## Performance Estimate
///
/// If fully implemented, JIT should achieve ~0-5% overhead vs hardcoded Rust.
/// Current graph overhead is ~30-200% depending on effect complexity.
/// Whether the implementation effort is worth ~30-195% improvement depends
/// on use case (real-time synthesis = yes, offline rendering = probably not).

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_jit_gain() {
        let mut compiler = JitCompiler::new().unwrap();
        let compiled = compiler.compile_gain(0.5).unwrap();

        let output = compiled.process(1.0);
        assert!((output - 0.5).abs() < 0.0001);

        let output2 = compiled.process(2.0);
        assert!((output2 - 1.0).abs() < 0.0001);
    }

    #[test]
    fn test_jit_tremolo() {
        let mut compiler = JitCompiler::new().unwrap();
        // base=0.5, scale=0.5 means gain varies 0.0-1.0 as LFO goes -1 to 1
        let compiled = compiler.compile_tremolo(0.5, 0.5).unwrap();

        // LFO at 0: gain = 0.5
        let out1 = compiled.process(1.0, 0.0);
        assert!((out1 - 0.5).abs() < 0.0001);

        // LFO at 1: gain = 1.0
        let out2 = compiled.process(1.0, 1.0);
        assert!((out2 - 1.0).abs() < 0.0001);

        // LFO at -1: gain = 0.0
        let out3 = compiled.process(1.0, -1.0);
        assert!(out3.abs() < 0.0001);
    }
}
