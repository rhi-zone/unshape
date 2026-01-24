//! Traits for JIT-compilable nodes and graphs.
//!
//! These traits define the interface that domain-specific graphs must implement
//! to enable JIT compilation. The separation of `JitCompilable` and `SimdCompilable`
//! allows gradual SIMD adoption.

#[cfg(feature = "cranelift")]
use cranelift::ir::Value;
#[cfg(feature = "cranelift")]
use cranelift_frontend::FunctionBuilder;

#[cfg(feature = "cranelift")]
use crate::context::JitContext;

// Re-export SimdWidth unconditionally since it's used in JitConfig
pub use self::simd_width::SimdWidth;

mod simd_width {
    /// SIMD width for vectorized operations.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub enum SimdWidth {
        /// 4-wide SIMD (f32x4) - SSE/NEON compatible
        X4 = 4,
        /// 8-wide SIMD (f32x8) - AVX compatible
        X8 = 8,
    }

    impl SimdWidth {
        /// Returns the number of lanes in this SIMD width.
        pub fn lanes(self) -> usize {
            self as usize
        }
    }
}

/// Classification of nodes for JIT compilation strategy.
///
/// This determines how the compiler handles each node:
/// - `PureMath` nodes are fully inlined and can be SIMD-vectorized
/// - `Stateful` nodes require callbacks to Rust for state management
/// - `External` nodes call external functions (transcendentals, noise, etc.)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum JitCategory {
    /// Pure math operation - can be fully inlined with SIMD.
    ///
    /// Examples: gain, offset, clip, add, multiply
    PureMath,

    /// Stateful operation - requires callback to Rust.
    ///
    /// Examples: delay lines, filters, oscillators with phase
    Stateful,

    /// External function call - cannot be inlined.
    ///
    /// Examples: noise functions, transcendentals (sin, cos, exp)
    External,
}

/// A node that can be JIT-compiled to native code.
///
/// Implementors describe how to generate Cranelift IR for their operation.
/// Domain crates (audio, fields, etc.) implement this for their node types.
///
/// # Example
///
/// ```ignore
/// impl JitCompilable for GainNode {
///     fn jit_category(&self) -> JitCategory {
///         JitCategory::PureMath
///     }
///
///     fn emit_ir(
///         &self,
///         inputs: &[Value],
///         builder: &mut FunctionBuilder<'_>,
///         _ctx: &mut JitContext,
///     ) -> Vec<Value> {
///         let input = inputs[0];
///         let gain = builder.ins().f32const(self.gain);
///         vec![builder.ins().fmul(input, gain)]
///     }
/// }
/// ```
#[cfg(feature = "cranelift")]
pub trait JitCompilable {
    /// Returns the node's compilation category.
    ///
    /// This determines how the compiler handles this node:
    /// - `PureMath`: Inline the operation, enable SIMD
    /// - `Stateful`: Generate callback to Rust closure
    /// - `External`: Generate call to external function
    fn jit_category(&self) -> JitCategory;

    /// Emits Cranelift IR for this node's operation.
    ///
    /// # Arguments
    ///
    /// * `inputs` - Cranelift IR values for each input (from predecessor nodes)
    /// * `builder` - Cranelift function builder for emitting instructions
    /// * `ctx` - JIT compilation context with helpers and state
    ///
    /// # Returns
    ///
    /// Cranelift IR values for each output port.
    fn emit_ir(
        &self,
        inputs: &[Value],
        builder: &mut FunctionBuilder<'_>,
        ctx: &mut JitContext,
    ) -> Vec<Value>;
}

/// Stub trait when cranelift feature is disabled.
#[cfg(not(feature = "cranelift"))]
pub trait JitCompilable {
    /// Returns the node's compilation category.
    fn jit_category(&self) -> JitCategory;
}

/// Extension trait for nodes that support SIMD batched execution.
///
/// Nodes implementing this can process 4 or 8 values simultaneously,
/// significantly improving throughput for pure-math operations.
#[cfg(feature = "cranelift")]
pub trait SimdCompilable: JitCompilable {
    /// Returns the preferred SIMD width for this node.
    ///
    /// Default is `X4` (f32x4) which is widely supported.
    fn preferred_simd_width(&self) -> SimdWidth {
        SimdWidth::X4
    }

    /// Emits SIMD IR for batched processing.
    ///
    /// # Arguments
    ///
    /// * `inputs` - Vector values (f32x4 or f32x8) for each input
    /// * `builder` - Cranelift function builder
    /// * `ctx` - JIT compilation context
    /// * `width` - SIMD width to use
    ///
    /// # Returns
    ///
    /// Vector values for each output.
    fn emit_simd_ir(
        &self,
        inputs: &[Value],
        builder: &mut FunctionBuilder<'_>,
        ctx: &mut JitContext,
        width: SimdWidth,
    ) -> Vec<Value>;
}

/// A graph that can be compiled to native code.
///
/// This trait abstracts over different graph representations (audio graphs,
/// field expressions, operation pipelines) to enable generic JIT compilation.
pub trait JitGraph {
    /// The node type in this graph.
    #[cfg(feature = "cranelift")]
    type Node: JitCompilable;

    #[cfg(not(feature = "cranelift"))]
    type Node;

    /// Node identifier type.
    type NodeId: Copy + Eq + std::hash::Hash;

    /// Returns the number of nodes in the graph.
    fn node_count(&self) -> usize;

    /// Returns a node by its identifier.
    fn node(&self, id: Self::NodeId) -> Option<&Self::Node>;

    /// Returns nodes in topological order for compilation.
    ///
    /// The returned order ensures that when processing a node, all its
    /// inputs have already been computed.
    fn nodes_in_order(&self) -> Vec<Self::NodeId>;

    /// Returns input node IDs that feed into the given node.
    fn node_inputs(&self, id: Self::NodeId) -> Vec<Self::NodeId>;

    /// Returns the graph's input nodes (entry points).
    ///
    /// These receive external input values.
    fn input_nodes(&self) -> Vec<Self::NodeId>;

    /// Returns the graph's output nodes.
    ///
    /// These produce the final output values.
    fn output_nodes(&self) -> Vec<Self::NodeId>;
}
