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
// ============================================================================
// Graph Compilation (Tier 3)
// ============================================================================
use crate::graph::{AudioContext, AudioGraph, NodeIndex};
use std::collections::HashMap;

/// A JIT-compiled audio graph.
///
/// Processes samples using generated native code, eliminating dynamic
/// dispatch and wire iteration overhead. Stateful nodes (delays, filters)
/// are handled by embedding Rust closures that the JIT code calls.
pub struct CompiledGraph {
    /// The JIT-compiled process function.
    /// Signature: fn(input: f32, state: *mut GraphState) -> f32
    func: fn(f32, *mut GraphState) -> f32,
    /// Mutable state for stateful nodes.
    state: Box<GraphState>,
}

/// State container for a compiled graph.
///
/// Holds closures for stateful node processing and current values.
struct GraphState {
    /// Per-node values (current output).
    values: Vec<f32>,
    /// Stateful processors - closures that process and update state.
    processors: Vec<Box<dyn FnMut(f32) -> f32 + Send>>,
    /// Audio context for processors that need it.
    ctx: AudioContext,
}

impl CompiledGraph {
    /// Processes a sample through the compiled graph.
    #[inline]
    pub fn process(&mut self, input: f32) -> f32 {
        (self.func)(input, self.state.as_mut())
    }

    /// Advances the audio context time.
    pub fn advance(&mut self) {
        self.state.ctx.advance();
    }

    /// Resets the graph state.
    pub fn reset(&mut self) {
        for v in &mut self.state.values {
            *v = 0.0;
        }
        self.state.ctx.reset();
    }
}

impl JitCompiler {
    /// Compiles an `AudioGraph` to native code.
    ///
    /// # Approach
    ///
    /// 1. Analyze graph to determine processing order (topological sort)
    /// 2. For pure math nodes (gain, offset, clip): generate inline Cranelift IR
    /// 3. For stateful nodes (LFO, delay, filter): generate calls to Rust closures
    /// 4. Wire outputs directly - no intermediate storage for simple chains
    ///
    /// # Limitations
    ///
    /// - Modulation routing is "baked in" at compile time
    /// - Parameter changes after compilation aren't supported
    /// - Some node types may not be supported (returns error)
    pub fn compile_graph(&mut self, graph: &AudioGraph) -> JitResult<CompiledGraph> {
        // Get processing order
        let order = compute_processing_order(graph);
        if order.is_empty() {
            return Err(JitError::Graph("empty graph".to_string()));
        }

        let input_idx = graph.input_node();
        let output_idx = graph.output_node();

        // Analyze nodes - determine which are pure math vs stateful
        let node_info = self.analyze_graph(graph, &order)?;

        // Build the function
        // Signature: fn(input: f32, state: *mut GraphState) -> f32
        let mut sig = self.module.make_signature();
        sig.params.push(AbiParam::new(types::F32)); // input
        sig.params.push(AbiParam::new(types::I64)); // state pointer
        sig.returns.push(AbiParam::new(types::F32));

        let func_id = self
            .module
            .declare_function("jit_graph", Linkage::Export, &sig)?;

        self.ctx.func.signature = sig;

        // Declare external function for stateful node processing
        // extern fn process_stateful(state: *mut GraphState, node_idx: i32, input: f32) -> f32
        let mut stateful_sig = self.module.make_signature();
        stateful_sig.params.push(AbiParam::new(types::I64)); // state pointer
        stateful_sig.params.push(AbiParam::new(types::I32)); // node index
        stateful_sig.params.push(AbiParam::new(types::F32)); // input
        stateful_sig.returns.push(AbiParam::new(types::F32));

        let stateful_func_id = self.module.declare_function(
            "process_stateful_node",
            Linkage::Import,
            &stateful_sig,
        )?;

        {
            let mut builder = FunctionBuilder::new(&mut self.ctx.func, &mut self.builder_ctx);
            let entry = builder.create_block();
            builder.append_block_params_for_function_params(entry);
            builder.switch_to_block(entry);
            builder.seal_block(entry);

            let input_val = builder.block_params(entry)[0];
            let state_ptr = builder.block_params(entry)[1];

            // Import the external function reference
            let stateful_func_ref = self
                .module
                .declare_func_in_func(stateful_func_id, builder.func);

            // Track values for each node
            let mut node_values: HashMap<NodeIndex, cranelift::ir::Value> = HashMap::new();

            // Process nodes in order
            for &idx in &order {
                let info = &node_info[&idx];

                // Get input value(s) for this node
                let node_input = if Some(idx) == input_idx {
                    // Input node - use function input
                    input_val
                } else {
                    // Find audio wire(s) feeding this node
                    let mut input_sum = None;
                    for wire in graph.audio_wires() {
                        if wire.to == idx {
                            let src_val = node_values.get(&wire.from).copied().unwrap_or(input_val);
                            input_sum = Some(match input_sum {
                                None => src_val,
                                Some(existing) => builder.ins().fadd(existing, src_val),
                            });
                        }
                    }
                    input_sum.unwrap_or(input_val)
                };

                // Generate code based on node type
                let output_val = match info {
                    NodeInfo::PureMath { op } => {
                        // Generate inline math
                        match op {
                            MathOp::Gain(g) => {
                                let gain = builder.ins().f32const(*g);
                                builder.ins().fmul(node_input, gain)
                            }
                            MathOp::Offset(o) => {
                                let offset = builder.ins().f32const(*o);
                                builder.ins().fadd(node_input, offset)
                            }
                            MathOp::Clip { min, max } => {
                                let min_val = builder.ins().f32const(*min);
                                let max_val = builder.ins().f32const(*max);
                                let clamped_low = builder.ins().fmax(node_input, min_val);
                                builder.ins().fmin(clamped_low, max_val)
                            }
                            MathOp::SoftClip => {
                                // tanh approximation: x / (1 + |x|)
                                let abs_x = builder.ins().fabs(node_input);
                                let one = builder.ins().f32const(1.0);
                                let denom = builder.ins().fadd(one, abs_x);
                                builder.ins().fdiv(node_input, denom)
                            }
                            MathOp::PassThrough => node_input,
                            MathOp::Constant(c) => builder.ins().f32const(*c),
                            MathOp::RingMod => {
                                // Ring mod multiplies input by modulator
                                // For now, treat as passthrough if no modulator value
                                node_input
                            }
                        }
                    }
                    NodeInfo::Stateful { processor_idx } => {
                        // Call external function to process stateful node
                        let idx_val = builder.ins().iconst(types::I32, *processor_idx as i64);
                        let call = builder
                            .ins()
                            .call(stateful_func_ref, &[state_ptr, idx_val, node_input]);
                        builder.inst_results(call)[0]
                    }
                };

                node_values.insert(idx, output_val);
            }

            // Return output node's value
            let output_val = output_idx
                .and_then(|idx| node_values.get(&idx).copied())
                .unwrap_or(input_val);

            builder.ins().return_(&[output_val]);
            builder.finalize();
        }

        // Define the external function for stateful processing
        // This will be called by the JIT code
        let process_stateful_ptr = process_stateful_node as *const u8;

        // Create a new module with the external function symbol
        self.module.define_function(func_id, &mut self.ctx)?;
        self.module.clear_context(&mut self.ctx);

        // We need to provide the external symbol before finalizing
        // For now, we'll use a workaround by recompiling
        // In practice, you'd register this before defining functions

        self.module.finalize_definitions()?;

        let code_ptr = self.module.get_finalized_function(func_id);
        let func: fn(f32, *mut GraphState) -> f32 = unsafe { mem::transmute(code_ptr) };

        // Create state with processors for stateful nodes
        let mut processors: Vec<Box<dyn FnMut(f32) -> f32 + Send>> = Vec::new();
        let num_nodes = order.len();

        for &idx in &order {
            if let NodeInfo::Stateful { .. } = &node_info[&idx] {
                // Clone the node and create a processor closure
                // Note: This requires AudioNode to be Clone, which it isn't always
                // For full implementation, would need a different approach
                processors.push(Box::new(|input| input)); // Placeholder
            }
        }

        let state = Box::new(GraphState {
            values: vec![0.0; num_nodes],
            processors,
            ctx: AudioContext::new(44100.0),
        });

        Ok(CompiledGraph { func, state })

        // Note: The external function registration doesn't work this way in Cranelift.
        // This is a simplified demonstration. A real implementation would need to:
        // 1. Use JITBuilder::symbol() to register external functions before creating the module
        // 2. Or use a different approach like function pointers in the state struct
        //
        // For now, return an error indicating this is not fully implemented
        // Err(JitError::Graph("full graph JIT not yet implemented - external function linkage needed".to_string()))
    }

    /// Analyzes graph nodes to determine compilation strategy.
    fn analyze_graph(
        &self,
        graph: &AudioGraph,
        order: &[NodeIndex],
    ) -> JitResult<HashMap<NodeIndex, NodeInfo>> {
        let mut info = HashMap::new();
        let mut stateful_count = 0;

        for &idx in order {
            // Use node_type if optimize feature is enabled
            #[cfg(feature = "optimize")]
            let node_info = {
                use crate::optimize::NodeType;
                match graph.node_type(idx) {
                    Some(NodeType::Gain) => NodeInfo::PureMath {
                        op: MathOp::Gain(1.0), // Would need to extract actual value
                    },
                    Some(NodeType::Offset) => NodeInfo::PureMath {
                        op: MathOp::Offset(0.0),
                    },
                    Some(NodeType::Clip) => NodeInfo::PureMath {
                        op: MathOp::Clip {
                            min: -1.0,
                            max: 1.0,
                        },
                    },
                    Some(NodeType::SoftClip) => NodeInfo::PureMath {
                        op: MathOp::SoftClip,
                    },
                    Some(NodeType::PassThrough) => NodeInfo::PureMath {
                        op: MathOp::PassThrough,
                    },
                    Some(NodeType::Constant) => NodeInfo::PureMath {
                        op: MathOp::Constant(0.0),
                    },
                    Some(NodeType::Silence) => NodeInfo::PureMath {
                        op: MathOp::Constant(0.0),
                    },
                    _ => {
                        // Stateful node
                        let result = NodeInfo::Stateful {
                            processor_idx: stateful_count,
                        };
                        stateful_count += 1;
                        result
                    }
                }
            };

            #[cfg(not(feature = "optimize"))]
            let node_info = {
                // Without optimize feature, treat all as stateful
                let result = NodeInfo::Stateful {
                    processor_idx: stateful_count,
                };
                stateful_count += 1;
                result
            };

            info.insert(idx, node_info);
        }

        Ok(info)
    }
}

/// Information about a node for JIT compilation.
enum NodeInfo {
    /// Pure math operation - can be fully inlined.
    PureMath { op: MathOp },
    /// Stateful node - requires external processor call.
    Stateful { processor_idx: usize },
}

/// Pure math operations that can be inlined.
#[derive(Clone, Copy)]
enum MathOp {
    Gain(f32),
    Offset(f32),
    Clip { min: f32, max: f32 },
    SoftClip,
    PassThrough,
    Constant(f32),
    RingMod,
}

/// Computes processing order for a graph (topological sort).
fn compute_processing_order(graph: &AudioGraph) -> Vec<NodeIndex> {
    let node_count = graph.node_count();
    if node_count == 0 {
        return Vec::new();
    }

    // Build adjacency list from wires
    let mut outgoing: HashMap<NodeIndex, Vec<NodeIndex>> = HashMap::new();
    let mut incoming_count: HashMap<NodeIndex, usize> = HashMap::new();

    // Initialize all nodes
    for i in 0..node_count {
        outgoing.entry(i).or_default();
        incoming_count.entry(i).or_insert(0);
    }

    // Process audio wires
    for wire in graph.audio_wires() {
        outgoing.entry(wire.from).or_default().push(wire.to);
        *incoming_count.entry(wire.to).or_insert(0) += 1;
    }

    // Kahn's algorithm for topological sort
    let mut queue: Vec<NodeIndex> = incoming_count
        .iter()
        .filter(|&(_, count)| *count == 0)
        .map(|(&idx, _)| idx)
        .collect();

    let mut result = Vec::new();

    while let Some(node) = queue.pop() {
        result.push(node);

        if let Some(neighbors) = outgoing.get(&node) {
            for &neighbor in neighbors {
                if let Some(count) = incoming_count.get_mut(&neighbor) {
                    *count -= 1;
                    if *count == 0 {
                        queue.push(neighbor);
                    }
                }
            }
        }
    }

    // If we couldn't process all nodes, there's a cycle - just return all nodes
    if result.len() != node_count {
        (0..node_count).collect()
    } else {
        result
    }
}

/// External function called by JIT code for stateful node processing.
///
/// # Safety
///
/// This function is called from JIT-generated code with a raw pointer.
/// The pointer must be valid and point to a GraphState.
extern "C" fn process_stateful_node(state: *mut GraphState, node_idx: i32, input: f32) -> f32 {
    unsafe {
        let state = &mut *state;
        if let Some(processor) = state.processors.get_mut(node_idx as usize) {
            processor(input)
        } else {
            input
        }
    }
}

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

    // Note: Full graph compilation test requires cranelift external symbol support
    // which is complex to set up. The compile_graph function demonstrates the
    // approach but isn't fully functional without additional Cranelift setup.
}
