//! JitCompilable implementations for audio nodes.
//!
//! This module implements the generic `JitCompilable` trait from `unshape-jit`
//! for audio-specific node types. Both scalar and SIMD implementations are provided.

#![cfg(feature = "cranelift")]

use cranelift::ir::{InstBuilder, Value, types};
use cranelift_frontend::FunctionBuilder;
use unshape_jit::{JitCategory, JitCompilable, JitContext, SimdCompilable, SimdWidth};

use crate::graph::{AffineNode, Clip, Constant, SoftClip};

// ============================================================================
// Pure Math Nodes
// ============================================================================

impl JitCompilable for AffineNode {
    fn jit_category(&self) -> JitCategory {
        JitCategory::PureMath
    }

    fn emit_ir(
        &self,
        inputs: &[Value],
        builder: &mut FunctionBuilder<'_>,
        _ctx: &mut JitContext,
    ) -> Vec<Value> {
        let input = inputs[0];

        // Optimize based on values
        let is_identity_gain = (self.gain - 1.0).abs() < 1e-10;
        let is_zero_offset = self.offset.abs() < 1e-10;

        let output = match (is_identity_gain, is_zero_offset) {
            (true, true) => input,
            (true, false) => {
                let o = builder.ins().f32const(self.offset);
                builder.ins().fadd(input, o)
            }
            (false, true) => {
                let g = builder.ins().f32const(self.gain);
                builder.ins().fmul(input, g)
            }
            (false, false) => {
                let g = builder.ins().f32const(self.gain);
                let o = builder.ins().f32const(self.offset);
                let mul = builder.ins().fmul(input, g);
                builder.ins().fadd(mul, o)
            }
        };

        vec![output]
    }
}

impl JitCompilable for Clip {
    fn jit_category(&self) -> JitCategory {
        JitCategory::PureMath
    }

    fn emit_ir(
        &self,
        inputs: &[Value],
        builder: &mut FunctionBuilder<'_>,
        _ctx: &mut JitContext,
    ) -> Vec<Value> {
        let input = inputs[0];
        let min_val = builder.ins().f32const(self.min);
        let max_val = builder.ins().f32const(self.max);

        // clamp(input, min, max)
        let clamped_low = builder.ins().fmax(input, min_val);
        let output = builder.ins().fmin(clamped_low, max_val);

        vec![output]
    }
}

impl JitCompilable for SoftClip {
    fn jit_category(&self) -> JitCategory {
        JitCategory::PureMath
    }

    fn emit_ir(
        &self,
        inputs: &[Value],
        builder: &mut FunctionBuilder<'_>,
        _ctx: &mut JitContext,
    ) -> Vec<Value> {
        let input = inputs[0];

        // Apply drive first: driven = input * drive
        let drive = builder.ins().f32const(self.drive);
        let driven = builder.ins().fmul(input, drive);

        // Soft clip approximation: x / (1 + |x|)
        let abs_x = builder.ins().fabs(driven);
        let one = builder.ins().f32const(1.0);
        let denom = builder.ins().fadd(one, abs_x);
        let output = builder.ins().fdiv(driven, denom);

        vec![output]
    }
}

impl JitCompilable for Constant {
    fn jit_category(&self) -> JitCategory {
        JitCategory::PureMath
    }

    fn emit_ir(
        &self,
        _inputs: &[Value],
        builder: &mut FunctionBuilder<'_>,
        _ctx: &mut JitContext,
    ) -> Vec<Value> {
        vec![builder.ins().f32const(self.0)]
    }
}

// ============================================================================
// SIMD Implementations
// ============================================================================

impl SimdCompilable for AffineNode {
    fn emit_simd_ir(
        &self,
        inputs: &[Value],
        builder: &mut FunctionBuilder<'_>,
        ctx: &mut JitContext,
        width: SimdWidth,
    ) -> Vec<Value> {
        let input_vec = inputs[0];
        let simd_type = ctx.simd_type(width);

        // Optimize based on values
        let is_identity_gain = (self.gain - 1.0).abs() < 1e-10;
        let is_zero_offset = self.offset.abs() < 1e-10;

        let output = match (is_identity_gain, is_zero_offset) {
            (true, true) => input_vec,
            (true, false) => {
                // Splat offset to vector and add
                let o_scalar = builder.ins().f32const(self.offset);
                let o_vec = builder.ins().splat(simd_type, o_scalar);
                builder.ins().fadd(input_vec, o_vec)
            }
            (false, true) => {
                // Splat gain to vector and multiply
                let g_scalar = builder.ins().f32const(self.gain);
                let g_vec = builder.ins().splat(simd_type, g_scalar);
                builder.ins().fmul(input_vec, g_vec)
            }
            (false, false) => {
                // Full affine: output = input * gain + offset
                let g_scalar = builder.ins().f32const(self.gain);
                let g_vec = builder.ins().splat(simd_type, g_scalar);
                let o_scalar = builder.ins().f32const(self.offset);
                let o_vec = builder.ins().splat(simd_type, o_scalar);

                let mul = builder.ins().fmul(input_vec, g_vec);
                builder.ins().fadd(mul, o_vec)
            }
        };

        vec![output]
    }
}

impl SimdCompilable for Clip {
    fn emit_simd_ir(
        &self,
        inputs: &[Value],
        builder: &mut FunctionBuilder<'_>,
        ctx: &mut JitContext,
        width: SimdWidth,
    ) -> Vec<Value> {
        let input_vec = inputs[0];
        let simd_type = ctx.simd_type(width);

        // Splat min/max to vectors
        let min_scalar = builder.ins().f32const(self.min);
        let min_vec = builder.ins().splat(simd_type, min_scalar);
        let max_scalar = builder.ins().f32const(self.max);
        let max_vec = builder.ins().splat(simd_type, max_scalar);

        // Vector clamp: fmax(fmin(input, max), min)
        let clamped_high = builder.ins().fmin(input_vec, max_vec);
        let output = builder.ins().fmax(clamped_high, min_vec);

        vec![output]
    }
}

impl SimdCompilable for SoftClip {
    fn emit_simd_ir(
        &self,
        inputs: &[Value],
        builder: &mut FunctionBuilder<'_>,
        ctx: &mut JitContext,
        width: SimdWidth,
    ) -> Vec<Value> {
        let input_vec = inputs[0];
        let simd_type = ctx.simd_type(width);

        // Apply drive: driven = input * drive
        let drive_scalar = builder.ins().f32const(self.drive);
        let drive_vec = builder.ins().splat(simd_type, drive_scalar);
        let driven = builder.ins().fmul(input_vec, drive_vec);

        // Soft clip: x / (1 + |x|)
        let abs_x = builder.ins().fabs(driven);
        let one_scalar = builder.ins().f32const(1.0);
        let one_vec = builder.ins().splat(simd_type, one_scalar);
        let denom = builder.ins().fadd(one_vec, abs_x);
        let output = builder.ins().fdiv(driven, denom);

        vec![output]
    }
}

impl SimdCompilable for Constant {
    fn emit_simd_ir(
        &self,
        _inputs: &[Value],
        builder: &mut FunctionBuilder<'_>,
        ctx: &mut JitContext,
        width: SimdWidth,
    ) -> Vec<Value> {
        let simd_type = ctx.simd_type(width);
        let scalar = builder.ins().f32const(self.0);
        vec![builder.ins().splat(simd_type, scalar)]
    }
}

// ============================================================================
// Stateful Nodes (placeholder - these return Stateful category)
// ============================================================================

// For stateful nodes like Oscillator, Delay, Filter, etc., we return
// JitCategory::Stateful. The actual processing is handled by callbacks
// to Rust code, which is managed by the audio-specific JIT compiler.

#[cfg(test)]
mod tests {
    use super::*;
    use unshape_jit::{JitCompiler, JitConfig};

    #[test]
    fn test_affine_category() {
        let node = AffineNode::gain(2.0);
        assert_eq!(node.jit_category(), JitCategory::PureMath);
    }

    #[test]
    fn test_clip_category() {
        let node = Clip::new(-1.0, 1.0);
        assert_eq!(node.jit_category(), JitCategory::PureMath);
    }

    #[test]
    fn test_softclip_category() {
        let node = SoftClip::new(1.0);
        assert_eq!(node.jit_category(), JitCategory::PureMath);
    }

    #[test]
    fn test_constant_category() {
        let node = Constant(1.0);
        assert_eq!(node.jit_category(), JitCategory::PureMath);
    }
}
