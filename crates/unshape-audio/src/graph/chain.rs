use super::{AudioContext, AudioNode, BlockProcessor};

/// A linear chain of audio processors.
#[derive(Default)]
pub struct Chain {
    nodes: Vec<Box<dyn AudioNode>>,
}

impl Chain {
    /// Creates an empty chain.
    pub fn new() -> Self {
        Self { nodes: Vec::new() }
    }

    /// Adds a node to the end of the chain.
    pub fn push<N: AudioNode + 'static>(&mut self, node: N) {
        self.nodes.push(Box::new(node));
    }

    /// Adds a node and returns self (for builder pattern).
    pub fn with<N: AudioNode + 'static>(mut self, node: N) -> Self {
        self.push(node);
        self
    }

    /// Returns the number of nodes.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Returns true if the chain is empty.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Processes a single sample through the chain.
    pub fn process(&mut self, input: f32, ctx: &AudioContext) -> f32 {
        let mut signal = input;
        for node in &mut self.nodes {
            signal = node.process(signal, ctx);
        }
        signal
    }

    /// Processes a buffer of samples.
    pub fn process_buffer(&mut self, buffer: &mut [f32], ctx: &mut AudioContext) {
        for sample in buffer {
            *sample = self.process(*sample, ctx);
            ctx.advance();
        }
    }

    /// Generates samples into a buffer (no input).
    pub fn generate(&mut self, buffer: &mut [f32], ctx: &mut AudioContext) {
        for sample in buffer {
            *sample = self.process(0.0, ctx);
            ctx.advance();
        }
    }

    /// Resets all nodes in the chain.
    pub fn reset(&mut self) {
        for node in &mut self.nodes {
            node.reset();
        }
    }
}

impl BlockProcessor for Chain {
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
        Chain::reset(self);
    }
}
