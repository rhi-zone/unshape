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

    /// Simple LCG for deterministic pseudo-random test data
    fn pseudo_random(seed: u32, count: usize) -> Vec<f32> {
        let mut state = seed;
        (0..count)
            .map(|_| {
                state = state.wrapping_mul(1103515245).wrapping_add(12345);
                // Map to [-10.0, 10.0] range
                ((state >> 16) as f32 / 32768.0) * 10.0 - 10.0
            })
            .collect()
    }

    /// Helper to verify parity between scalar, SIMD, and native
    fn assert_parity(gain: f32, offset: f32, input: &[f32]) {
        let mut compiler = JitCompiler::new(JitConfig::default()).unwrap();
        let scalar = compiler.compile_affine(gain, offset).unwrap();
        let simd = compiler.compile_affine_simd(gain, offset).unwrap();

        let mut output_simd = vec![0.0; input.len()];
        let mut output_native = vec![0.0; input.len()];

        simd.process(input, &mut output_simd);
        native_affine(input, &mut output_native, gain, offset);

        for i in 0..input.len() {
            let scalar_val = scalar.call_f32(input[i]);
            let tolerance = (scalar_val.abs() * 1e-5).max(1e-6);

            assert!(
                (scalar_val - output_simd[i]).abs() < tolerance,
                "scalar vs simd mismatch at {}: {} vs {} (input={}, gain={}, offset={})",
                i,
                scalar_val,
                output_simd[i],
                input[i],
                gain,
                offset
            );
            assert!(
                (scalar_val - output_native[i]).abs() < tolerance,
                "scalar vs native mismatch at {}: {} vs {} (input={}, gain={}, offset={})",
                i,
                scalar_val,
                output_native[i],
                input[i],
                gain,
                offset
            );
        }
    }

    #[test]
    fn test_parity_typical_audio() {
        // Typical audio processing: -1.0 to 1.0 range
        let input: Vec<f32> = (0..4096)
            .map(|i| ((i as f32 / 4096.0) * 2.0 * std::f32::consts::PI).sin())
            .collect();

        // Various gain/offset combos used in audio
        assert_parity(1.0, 0.0, &input); // identity
        assert_parity(0.5, 0.0, &input); // half gain
        assert_parity(2.0, 0.0, &input); // double gain
        assert_parity(1.0, 0.5, &input); // DC offset
        assert_parity(0.7, 0.3, &input); // typical mix
        assert_parity(-1.0, 0.0, &input); // phase invert
        assert_parity(0.0, 0.5, &input); // silence + offset
    }

    #[test]
    fn test_parity_random_data() {
        // Pseudo-random data with various seeds
        for seed in [42u32, 12345, 98765, 0xDEADBEEF] {
            let input = pseudo_random(seed, 1024);
            assert_parity(0.8, -0.2, &input);
            assert_parity(1.5, 0.5, &input);
            assert_parity(0.1, 10.0, &input);
        }
    }

    #[test]
    fn test_parity_varied_buffer_sizes() {
        let input_base = pseudo_random(777, 2048);

        // Test alignment edge cases
        for size in [
            1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 255, 256, 1000,
            1023, 1024, 2048,
        ] {
            let input = &input_base[..size];
            assert_parity(0.75, -1.25, input);
        }
    }

    #[test]
    fn test_parity_varied_parameters() {
        let input = pseudo_random(999, 512);

        // Test many gain/offset combinations
        let gains = [
            0.0, 0.001, 0.1, 0.5, 1.0, 2.0, 10.0, 100.0, -0.5, -1.0, -2.0,
        ];
        let offsets = [0.0, 0.001, 0.1, 1.0, 10.0, -0.5, -1.0, -10.0];

        for &gain in &gains {
            for &offset in &offsets {
                assert_parity(gain, offset, &input);
            }
        }
    }

    #[test]
    fn test_parity_edge_values() {
        // Edge case values
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
            f32::MAX / 2.0,
            f32::MIN / 2.0,
        ];

        assert_parity(1.0, 0.0, &input);
        assert_parity(0.5, 0.0, &input);
        assert_parity(1.0, 1.0, &input);
    }

    #[test]
    fn test_parity_large_buffer() {
        // Simulate 1 second of audio at 44.1kHz
        let input: Vec<f32> = (0..44100)
            .map(|i| ((i as f32 / 44100.0) * 440.0 * 2.0 * std::f32::consts::PI).sin() * 0.8)
            .collect();

        assert_parity(0.5, 0.1, &input);
    }
}
