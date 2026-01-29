use super::{AudioContext, AudioNode};

/// Mixes multiple audio sources together.
pub struct Mixer {
    sources: Vec<Box<dyn AudioNode>>,
    gains: Vec<f32>,
}

impl Mixer {
    /// Creates an empty mixer.
    pub fn new() -> Self {
        Self {
            sources: Vec::new(),
            gains: Vec::new(),
        }
    }

    /// Adds a source with the given gain.
    pub fn add<N: AudioNode + 'static>(&mut self, node: N, gain: f32) {
        self.sources.push(Box::new(node));
        self.gains.push(gain);
    }

    /// Adds a source and returns self.
    pub fn with<N: AudioNode + 'static>(mut self, node: N, gain: f32) -> Self {
        self.add(node, gain);
        self
    }

    /// Sets the gain for a source by index.
    pub fn set_gain(&mut self, index: usize, gain: f32) {
        if index < self.gains.len() {
            self.gains[index] = gain;
        }
    }

    /// Processes and mixes all sources.
    pub fn process(&mut self, input: f32, ctx: &AudioContext) -> f32 {
        let mut output = 0.0;
        for (source, &gain) in self.sources.iter_mut().zip(self.gains.iter()) {
            output += source.process(input, ctx) * gain;
        }
        output
    }

    /// Resets all sources.
    pub fn reset(&mut self) {
        for source in &mut self.sources {
            source.reset();
        }
    }
}

impl Default for Mixer {
    fn default() -> Self {
        Self::new()
    }
}

impl AudioNode for Mixer {
    fn process(&mut self, input: f32, ctx: &AudioContext) -> f32 {
        Mixer::process(self, input, ctx)
    }

    fn reset(&mut self) {
        Mixer::reset(self);
    }
}
