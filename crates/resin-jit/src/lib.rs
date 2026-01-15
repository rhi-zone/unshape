//! JIT compilation infrastructure for resin graphs.
//!
//! This crate provides generic JIT compilation for graph-based data flow structures
//! using Cranelift. It supports both scalar and SIMD compilation modes.
//!
//! # Features
//!
//! - `cranelift` - Enables JIT compilation via Cranelift (required for actual compilation)
//!
//! # Architecture
//!
//! The JIT system uses a trait-based approach:
//!
//! - [`JitCompilable`] - Nodes that can be compiled to native code
//! - [`SimdCompilable`] - Extension for SIMD-capable nodes
//! - [`JitGraph`] - Graphs that can be compiled as a unit
//!
//! Nodes are classified by [`JitCategory`]:
//!
//! - `PureMath` - Fully inlined, SIMD-able operations
//! - `Stateful` - Operations requiring Rust callbacks (delay lines, filters)
//! - `External` - External function calls (noise, transcendentals)
//!
//! # Example
//!
//! ```ignore
//! use rhizome_resin_jit::{JitCompiler, JitConfig};
//!
//! // Create compiler
//! let mut compiler = JitCompiler::new(JitConfig::default())?;
//!
//! // Compile a simple affine transform
//! let affine = compiler.compile_affine(0.5, 1.0)?;
//!
//! // Use the compiled function
//! let output = affine.call_f32(2.0);  // 2.0 * 0.5 + 1.0 = 2.0
//! ```
//!
//! # Domain Integration
//!
//! Domain crates (audio, fields, etc.) implement the JIT traits for their node types:
//!
//! ```ignore
//! // In resin-audio
//! impl JitCompilable for AffineNode {
//!     fn jit_category(&self) -> JitCategory {
//!         JitCategory::PureMath
//!     }
//!
//!     fn emit_ir(&self, inputs: &[Value], builder: &mut FunctionBuilder, ctx: &mut JitContext) -> Vec<Value> {
//!         // ... emit Cranelift IR
//!     }
//! }
//! ```

#![cfg_attr(not(feature = "cranelift"), allow(unused_imports))]

mod compiled;
mod compiler;
mod context;
mod error;
mod traits;

pub use compiled::*;
pub use compiler::*;
pub use context::*;
pub use error::*;
pub use traits::*;

#[cfg(all(test, feature = "cranelift"))]
mod tests {
    use super::*;

    #[test]
    fn test_compile_affine_identity() {
        let mut compiler = JitCompiler::new(JitConfig::default()).unwrap();
        let identity = compiler.compile_affine(1.0, 0.0).unwrap();

        assert!((identity.call_f32(0.0) - 0.0).abs() < 1e-6);
        assert!((identity.call_f32(1.0) - 1.0).abs() < 1e-6);
        assert!((identity.call_f32(-5.0) - -5.0).abs() < 1e-6);
    }

    #[test]
    fn test_compile_affine_gain() {
        let mut compiler = JitCompiler::new(JitConfig::default()).unwrap();
        let gain = compiler.compile_affine(0.5, 0.0).unwrap();

        assert!((gain.call_f32(2.0) - 1.0).abs() < 1e-6);
        assert!((gain.call_f32(10.0) - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_compile_affine_offset() {
        let mut compiler = JitCompiler::new(JitConfig::default()).unwrap();
        let offset = compiler.compile_affine(1.0, 3.0).unwrap();

        assert!((offset.call_f32(0.0) - 3.0).abs() < 1e-6);
        assert!((offset.call_f32(2.0) - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_compile_affine_full() {
        let mut compiler = JitCompiler::new(JitConfig::default()).unwrap();
        // y = 2x + 1
        let affine = compiler.compile_affine(2.0, 1.0).unwrap();

        assert!((affine.call_f32(0.0) - 1.0).abs() < 1e-6);
        assert!((affine.call_f32(1.0) - 3.0).abs() < 1e-6);
        assert!((affine.call_f32(3.0) - 7.0).abs() < 1e-6);
    }
}
