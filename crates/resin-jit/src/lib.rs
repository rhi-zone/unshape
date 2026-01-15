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

    // ========================================================================
    // SIMD Tests
    // ========================================================================

    #[test]
    fn test_simd_affine_identity() {
        let mut compiler = JitCompiler::new(JitConfig::default()).unwrap();
        let simd = compiler.compile_affine_simd(1.0, 0.0).unwrap();

        let input: Vec<f32> = (0..8).map(|i| i as f32).collect();
        let mut output = vec![0.0; 8];

        simd.process(&input, &mut output);

        for i in 0..8 {
            assert!(
                (output[i] - input[i]).abs() < 1e-6,
                "mismatch at {}: {} vs {}",
                i,
                output[i],
                input[i]
            );
        }
    }

    #[test]
    fn test_simd_affine_gain() {
        let mut compiler = JitCompiler::new(JitConfig::default()).unwrap();
        let simd = compiler.compile_affine_simd(2.0, 0.0).unwrap();

        let input: Vec<f32> = (0..8).map(|i| i as f32).collect();
        let mut output = vec![0.0; 8];

        simd.process(&input, &mut output);

        for i in 0..8 {
            let expected = (i as f32) * 2.0;
            assert!(
                (output[i] - expected).abs() < 1e-6,
                "mismatch at {}: {} vs {}",
                i,
                output[i],
                expected
            );
        }
    }

    #[test]
    fn test_simd_affine_offset() {
        let mut compiler = JitCompiler::new(JitConfig::default()).unwrap();
        let simd = compiler.compile_affine_simd(1.0, 5.0).unwrap();

        let input: Vec<f32> = (0..8).map(|i| i as f32).collect();
        let mut output = vec![0.0; 8];

        simd.process(&input, &mut output);

        for i in 0..8 {
            let expected = (i as f32) + 5.0;
            assert!(
                (output[i] - expected).abs() < 1e-6,
                "mismatch at {}: {} vs {}",
                i,
                output[i],
                expected
            );
        }
    }

    #[test]
    fn test_simd_affine_full() {
        let mut compiler = JitCompiler::new(JitConfig::default()).unwrap();
        // y = 2x + 1
        let simd = compiler.compile_affine_simd(2.0, 1.0).unwrap();

        let input: Vec<f32> = (0..8).map(|i| i as f32).collect();
        let mut output = vec![0.0; 8];

        simd.process(&input, &mut output);

        for i in 0..8 {
            let expected = (i as f32) * 2.0 + 1.0;
            assert!(
                (output[i] - expected).abs() < 1e-6,
                "mismatch at {}: {} vs {}",
                i,
                output[i],
                expected
            );
        }
    }

    #[test]
    fn test_simd_affine_non_aligned() {
        // Test with non-multiple-of-4 length to exercise scalar tail
        let mut compiler = JitCompiler::new(JitConfig::default()).unwrap();
        let simd = compiler.compile_affine_simd(2.0, 1.0).unwrap();

        let input: Vec<f32> = (0..7).map(|i| i as f32).collect(); // 7 elements
        let mut output = vec![0.0; 7];

        simd.process(&input, &mut output);

        for i in 0..7 {
            let expected = (i as f32) * 2.0 + 1.0;
            assert!(
                (output[i] - expected).abs() < 1e-6,
                "mismatch at {}: {} vs {}",
                i,
                output[i],
                expected
            );
        }
    }

    #[test]
    fn test_simd_large_buffer() {
        // Test with large buffer (1 second of audio at 44100 Hz)
        let mut compiler = JitCompiler::new(JitConfig::default()).unwrap();
        let simd = compiler.compile_affine_simd(0.5, 0.0).unwrap();

        let input: Vec<f32> = (0..44100).map(|i| (i as f32) * 0.001).collect();
        let mut output = vec![0.0; 44100];

        simd.process(&input, &mut output);

        // Spot check a few values
        assert!((output[0] - 0.0).abs() < 1e-6);
        assert!((output[1000] - 0.5).abs() < 1e-4);
        assert!((output[44099] - (44099.0 * 0.001 * 0.5)).abs() < 1e-3);
    }

    // ========================================================================
    // Parity Tests: Scalar JIT == SIMD JIT == Native
    // ========================================================================

    /// Native Rust implementation for parity comparison
    fn native_affine(input: &[f32], output: &mut [f32], gain: f32, offset: f32) {
        for i in 0..input.len() {
            output[i] = input[i] * gain + offset;
        }
    }

    #[test]
    fn test_parity_identity() {
        let mut compiler = JitCompiler::new(JitConfig::default()).unwrap();
        let scalar = compiler.compile_affine(1.0, 0.0).unwrap();
        let simd = compiler.compile_affine_simd(1.0, 0.0).unwrap();

        let input: Vec<f32> = (0..1000).map(|i| (i as f32) * 0.1 - 50.0).collect();
        let mut output_simd = vec![0.0; 1000];
        let mut output_native = vec![0.0; 1000];

        simd.process(&input, &mut output_simd);
        native_affine(&input, &mut output_native, 1.0, 0.0);

        for i in 0..1000 {
            let scalar_val = scalar.call_f32(input[i]);
            assert!(
                (scalar_val - output_simd[i]).abs() < 1e-6,
                "scalar vs simd mismatch at {}: {} vs {}",
                i,
                scalar_val,
                output_simd[i]
            );
            assert!(
                (scalar_val - output_native[i]).abs() < 1e-6,
                "scalar vs native mismatch at {}: {} vs {}",
                i,
                scalar_val,
                output_native[i]
            );
        }
    }

    #[test]
    fn test_parity_gain_and_offset() {
        let gain = 0.7;
        let offset = -3.5;

        let mut compiler = JitCompiler::new(JitConfig::default()).unwrap();
        let scalar = compiler.compile_affine(gain, offset).unwrap();
        let simd = compiler.compile_affine_simd(gain, offset).unwrap();

        // Test various buffer sizes including non-aligned
        for size in [1, 3, 4, 7, 8, 15, 16, 100, 1023, 1024] {
            let input: Vec<f32> = (0..size).map(|i| (i as f32) * 0.01 - 5.0).collect();
            let mut output_simd = vec![0.0; size];
            let mut output_native = vec![0.0; size];

            simd.process(&input, &mut output_simd);
            native_affine(&input, &mut output_native, gain, offset);

            for i in 0..size {
                let scalar_val = scalar.call_f32(input[i]);
                assert!(
                    (scalar_val - output_simd[i]).abs() < 1e-5,
                    "size={} scalar vs simd mismatch at {}: {} vs {}",
                    size,
                    i,
                    scalar_val,
                    output_simd[i]
                );
                assert!(
                    (scalar_val - output_native[i]).abs() < 1e-5,
                    "size={} scalar vs native mismatch at {}: {} vs {}",
                    size,
                    i,
                    scalar_val,
                    output_native[i]
                );
            }
        }
    }

    #[test]
    fn test_parity_edge_values() {
        // Test with edge case values: zero, negative, large, small
        let mut compiler = JitCompiler::new(JitConfig::default()).unwrap();
        let scalar = compiler.compile_affine(2.5, -1.0).unwrap();
        let simd = compiler.compile_affine_simd(2.5, -1.0).unwrap();

        let input = vec![
            0.0,
            -0.0,
            1.0,
            -1.0,
            f32::MIN_POSITIVE,
            -f32::MIN_POSITIVE,
            1e10,
            -1e10,
            1e-10,
            -1e-10,
            std::f32::consts::PI,
            std::f32::consts::E,
        ];
        let mut output_simd = vec![0.0; input.len()];
        let mut output_native = vec![0.0; input.len()];

        simd.process(&input, &mut output_simd);
        native_affine(&input, &mut output_native, 2.5, -1.0);

        for i in 0..input.len() {
            let scalar_val = scalar.call_f32(input[i]);
            // Use relative error for large values
            let tolerance = (scalar_val.abs() * 1e-5).max(1e-6);
            assert!(
                (scalar_val - output_simd[i]).abs() < tolerance,
                "edge scalar vs simd mismatch at {}: {} vs {} (input={})",
                i,
                scalar_val,
                output_simd[i],
                input[i]
            );
            assert!(
                (scalar_val - output_native[i]).abs() < tolerance,
                "edge scalar vs native mismatch at {}: {} vs {} (input={})",
                i,
                scalar_val,
                output_native[i],
                input[i]
            );
        }
    }
}
