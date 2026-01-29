use super::{AudioContext, ParamDescriptor};

/// Trait for audio processing nodes.
pub trait AudioNode: Send {
    /// Processes a single sample.
    fn process(&mut self, input: f32, ctx: &AudioContext) -> f32;

    /// Resets the node's internal state.
    fn reset(&mut self) {}

    /// Returns descriptors for modulatable parameters.
    ///
    /// Override this to expose parameters that can be modulated by other nodes
    /// in a graph. Parameters are set via [`set_param`] before each [`process`] call.
    fn params(&self) -> &'static [ParamDescriptor] {
        &[]
    }

    /// Sets a parameter value by index.
    ///
    /// Called by the graph executor to apply modulation before processing.
    /// Index corresponds to the parameter's position in [`params()`].
    fn set_param(&mut self, _index: usize, _value: f32) {}

    /// Gets a parameter's current value by index.
    ///
    /// Returns `None` if the index is out of bounds.
    /// Used by JIT compilation to extract parameter values at compile time.
    fn get_param(&self, _index: usize) -> Option<f32> {
        None
    }
}

/// Trait for block-based audio processing.
///
/// This is the unified interface for all audio processing tiers:
/// - Tier 1/2/4: Default impl loops over `AudioNode::process()` (already efficient)
/// - Tier 3 JIT: Native block impl (amortizes function call overhead)
///
/// Use this trait when you want code that works with any tier.
///
/// # Example
///
/// ```
/// use unshape_audio::graph::{BlockProcessor, AudioContext, Chain, AffineNode};
///
/// fn apply_effect<P: BlockProcessor>(effect: &mut P, audio: &mut [f32], sample_rate: f32) {
///     let mut output = vec![0.0; audio.len()];
///     let mut ctx = AudioContext::new(sample_rate);
///     effect.process_block(audio, &mut output, &mut ctx);
///     audio.copy_from_slice(&output);
/// }
/// ```
pub trait BlockProcessor: Send {
    /// Processes a block of samples.
    ///
    /// # Arguments
    /// * `input` - Input sample buffer
    /// * `output` - Output sample buffer (must be same length as input)
    /// * `ctx` - Audio context (will be advanced for each sample)
    fn process_block(&mut self, input: &[f32], output: &mut [f32], ctx: &mut AudioContext);

    /// Resets the processor's internal state.
    fn reset(&mut self);
}

/// Blanket implementation of BlockProcessor for all AudioNode types.
///
/// This provides efficient per-sample processing for Tier 1/2/4 where
/// the compiler can inline the process() calls.
impl<T: AudioNode> BlockProcessor for T {
    fn process_block(&mut self, input: &[f32], output: &mut [f32], ctx: &mut AudioContext) {
        debug_assert_eq!(
            input.len(),
            output.len(),
            "input and output buffers must be same length"
        );
        for (inp, out) in input.iter().zip(output.iter_mut()) {
            *out = self.process(*inp, ctx);
            ctx.advance();
        }
    }

    fn reset(&mut self) {
        AudioNode::reset(self);
    }
}
