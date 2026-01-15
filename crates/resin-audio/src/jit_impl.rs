//! JitCompilable implementations for audio nodes.
//!
//! This module implements the generic `JitCompilable` trait from `resin-jit`
//! for audio-specific node types.

#![cfg(feature = "cranelift")]

use cranelift::ir::{InstBuilder, Value};
use cranelift_frontend::FunctionBuilder;
use rhizome_resin_jit::{JitCategory, JitCompilable, JitContext};

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
// Stateful Nodes (placeholder - these return Stateful category)
// ============================================================================

// For stateful nodes like Oscillator, Delay, Filter, etc., we return
// JitCategory::Stateful. The actual processing is handled by callbacks
// to Rust code, which is managed by the audio-specific JIT compiler.

#[cfg(test)]
mod tests {
    use super::*;
    use rhizome_resin_jit::{JitCompiler, JitConfig};

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
