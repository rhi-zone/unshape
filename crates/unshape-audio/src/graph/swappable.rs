use super::{AudioContext, AudioGraph};

/// A graph wrapper that supports glitch-free hot-swapping.
///
/// `SwappableGraph` allows replacing an audio graph at runtime while audio
/// is playing, using crossfading to avoid clicks and pops.
///
/// # Example
///
/// ```ignore
/// use unshape_audio::graph::{SwappableGraph, AudioGraph, AudioContext};
///
/// // Create initial graph
/// let graph1 = AudioGraph::new();
/// let mut swappable = SwappableGraph::new(graph1);
///
/// // In audio callback
/// fn process_audio(swappable: &mut SwappableGraph, buffer: &mut [f32], ctx: &mut AudioContext) {
///     swappable.process_buffer(buffer, ctx);
/// }
///
/// // From UI thread - queue a new graph
/// let graph2 = AudioGraph::new();
/// swappable.swap(graph2, 1024); // 1024 sample crossfade
/// ```
pub struct SwappableGraph {
    /// Current active graph.
    current: AudioGraph,
    /// Pending graph being crossfaded in.
    pending: Option<AudioGraph>,
    /// Crossfade progress (samples remaining).
    crossfade_remaining: u32,
    /// Total crossfade duration (samples).
    crossfade_total: u32,
}

impl SwappableGraph {
    /// Creates a new swappable graph wrapper.
    pub fn new(graph: AudioGraph) -> Self {
        Self {
            current: graph,
            pending: None,
            crossfade_remaining: 0,
            crossfade_total: 0,
        }
    }

    /// Queues a new graph to replace the current one.
    ///
    /// The new graph will be crossfaded in over `crossfade_samples` samples.
    /// Typical values: 256-2048 samples (5-45ms at 44.1kHz).
    ///
    /// If a crossfade is already in progress, the pending graph is replaced.
    pub fn swap(&mut self, new_graph: AudioGraph, crossfade_samples: u32) {
        // If we're mid-crossfade, instantly complete it
        if self.pending.is_some() {
            self.complete_crossfade();
        }

        self.pending = Some(new_graph);
        self.crossfade_remaining = crossfade_samples;
        self.crossfade_total = crossfade_samples;
    }

    /// Returns true if a crossfade is currently in progress.
    pub fn is_crossfading(&self) -> bool {
        self.pending.is_some()
    }

    /// Returns the crossfade progress (0.0 = just started, 1.0 = complete).
    pub fn crossfade_progress(&self) -> f32 {
        if self.crossfade_total == 0 {
            1.0
        } else {
            1.0 - (self.crossfade_remaining as f32 / self.crossfade_total as f32)
        }
    }

    /// Processes a single sample through the graph(s).
    pub fn process(&mut self, input: f32, ctx: &AudioContext) -> f32 {
        if self.pending.is_some() {
            // Calculate crossfade position before borrowing
            let t = if self.crossfade_total == 0 {
                1.0
            } else {
                1.0 - (self.crossfade_remaining as f32 / self.crossfade_total as f32)
            };

            // Use equal-power crossfade for smoother transition
            let old_gain = ((1.0 - t) * std::f32::consts::FRAC_PI_2).sin();
            let new_gain = (t * std::f32::consts::FRAC_PI_2).sin();

            let old_out = self.current.process(input, ctx);
            let new_out = self.pending.as_mut().unwrap().process(input, ctx);

            let output = old_out * old_gain + new_out * new_gain;

            // Advance crossfade
            self.crossfade_remaining = self.crossfade_remaining.saturating_sub(1);
            if self.crossfade_remaining == 0 {
                self.complete_crossfade();
            }

            output
        } else {
            self.current.process(input, ctx)
        }
    }

    /// Processes a buffer of samples.
    pub fn process_buffer(&mut self, buffer: &mut [f32], ctx: &mut AudioContext) {
        for sample in buffer {
            *sample = self.process(*sample, ctx);
            ctx.advance();
        }
    }

    /// Generates samples (no input).
    pub fn generate(&mut self, buffer: &mut [f32], ctx: &mut AudioContext) {
        for sample in buffer {
            *sample = self.process(0.0, ctx);
            ctx.advance();
        }
    }

    /// Completes any pending crossfade immediately.
    fn complete_crossfade(&mut self) {
        if let Some(pending) = self.pending.take() {
            self.current = pending;
        }
        self.crossfade_remaining = 0;
        self.crossfade_total = 0;
    }

    /// Cancels any pending crossfade and keeps the current graph.
    pub fn cancel_swap(&mut self) {
        self.pending = None;
        self.crossfade_remaining = 0;
        self.crossfade_total = 0;
    }

    /// Returns a reference to the current (active) graph.
    pub fn current(&self) -> &AudioGraph {
        &self.current
    }

    /// Returns a mutable reference to the current graph.
    ///
    /// Use this to modify parameters on the active graph.
    pub fn current_mut(&mut self) -> &mut AudioGraph {
        &mut self.current
    }

    /// Resets all graphs.
    pub fn reset(&mut self) {
        self.current.reset();
        if let Some(ref mut pending) = self.pending {
            pending.reset();
        }
    }
}
