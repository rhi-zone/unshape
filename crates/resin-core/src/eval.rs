//! Evaluation context and utilities for graph execution.
//!
//! This module provides the execution context for node evaluation,
//! including cancellation, progress reporting, and evaluation parameters.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use crate::graph::NodeId;
use crate::value::Value;

/// Token for cooperative cancellation of graph evaluation.
///
/// Clone this token to share it between threads. Call `cancel()` from one
/// thread, and check `is_cancelled()` from the evaluation thread.
#[derive(Clone, Default)]
pub struct CancellationToken {
    cancelled: Arc<AtomicBool>,
}

impl CancellationToken {
    /// Create a new cancellation token.
    pub fn new() -> Self {
        Self::default()
    }

    /// Signal cancellation.
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::SeqCst);
    }

    /// Check if cancellation has been signaled.
    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::SeqCst)
    }

    /// Reset the cancellation state (for reuse).
    pub fn reset(&self) {
        self.cancelled.store(false, Ordering::SeqCst);
    }
}

/// How cancellation should be handled during evaluation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CancellationMode {
    /// Check only between nodes (zero overhead, coarse granularity).
    NodeBoundary,
    /// Pass token to nodes via EvalContext, nodes check periodically.
    #[default]
    Cooperative,
    /// Spawn nodes as abortable tasks (has overhead, true preemption).
    Preemptive,
}

/// Context passed to node execution.
///
/// Provides environment information beyond just input values: time, cancellation,
/// progress reporting, quality hints, and feedback state for recurrent graphs.
pub struct EvalContext {
    // === Control ===
    cancel: Option<CancellationToken>,
    progress_callback: Option<Box<dyn Fn(EvalProgress) + Send>>,

    // === Time ===
    /// Current time in seconds.
    pub time: f64,
    /// Current frame number.
    pub frame: u64,
    /// Delta time since last evaluation.
    pub dt: f64,

    // === Quality hints ===
    /// True if this is a preview render (nodes can reduce quality).
    pub preview_mode: bool,
    /// Target resolution hint for LOD decisions.
    pub target_resolution: Option<(u32, u32)>,

    // === Recurrent graphs ===
    feedback_state: Option<FeedbackState>,

    // === Determinism ===
    /// Random seed for reproducible procedural generation.
    pub seed: u64,
}

impl Default for EvalContext {
    fn default() -> Self {
        Self {
            cancel: None,
            progress_callback: None,
            time: 0.0,
            frame: 0,
            dt: 1.0 / 60.0,
            preview_mode: false,
            target_resolution: None,
            feedback_state: None,
            seed: 0,
        }
    }
}

impl EvalContext {
    /// Create a new evaluation context with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the cancellation token.
    pub fn with_cancel(mut self, token: CancellationToken) -> Self {
        self.cancel = Some(token);
        self
    }

    /// Set the progress callback.
    pub fn with_progress(mut self, callback: impl Fn(EvalProgress) + Send + 'static) -> Self {
        self.progress_callback = Some(Box::new(callback));
        self
    }

    /// Set time parameters.
    pub fn with_time(mut self, time: f64, frame: u64, dt: f64) -> Self {
        self.time = time;
        self.frame = frame;
        self.dt = dt;
        self
    }

    /// Set preview mode.
    pub fn with_preview_mode(mut self, preview: bool) -> Self {
        self.preview_mode = preview;
        self
    }

    /// Set target resolution hint.
    pub fn with_resolution(mut self, width: u32, height: u32) -> Self {
        self.target_resolution = Some((width, height));
        self
    }

    /// Set random seed.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Check if cancellation has been signaled.
    pub fn is_cancelled(&self) -> bool {
        self.cancel.as_ref().is_some_and(|t| t.is_cancelled())
    }

    /// Report progress from within a node.
    pub fn report_progress(&self, completed: usize, total: usize) {
        if let Some(ref callback) = self.progress_callback {
            callback(EvalProgress {
                completed_nodes: completed,
                total_nodes: total,
                current_node: None,
                elapsed: Duration::ZERO,
            });
        }
    }

    /// Get feedback state for recurrent graphs (if available).
    pub fn feedback_state(&self) -> Option<&FeedbackState> {
        self.feedback_state.as_ref()
    }

    /// Get mutable feedback state for recurrent graphs (if available).
    pub fn feedback_state_mut(&mut self) -> Option<&mut FeedbackState> {
        self.feedback_state.as_mut()
    }

    /// Set feedback state for recurrent graphs.
    pub fn with_feedback_state(mut self, state: FeedbackState) -> Self {
        self.feedback_state = Some(state);
        self
    }
}

/// Progress information for evaluation.
#[derive(Debug, Clone)]
pub struct EvalProgress {
    /// Number of nodes that have been evaluated.
    pub completed_nodes: usize,
    /// Total number of nodes to evaluate.
    pub total_nodes: usize,
    /// Currently executing node (if known).
    pub current_node: Option<NodeId>,
    /// Time elapsed since evaluation started.
    pub elapsed: Duration,
}

/// State for feedback wires in recurrent graphs.
///
/// Stores values that carry across iterations/frames for feedback loops.
#[derive(Debug, Clone, Default)]
pub struct FeedbackState {
    /// Feedback wire values, keyed by (from_node, from_port).
    values: HashMap<(NodeId, usize), Value>,
}

impl FeedbackState {
    /// Create empty feedback state.
    pub fn new() -> Self {
        Self::default()
    }

    /// Get a feedback value.
    pub fn get(&self, node: NodeId, port: usize) -> Option<&Value> {
        self.values.get(&(node, port))
    }

    /// Set a feedback value.
    pub fn set(&mut self, node: NodeId, port: usize, value: Value) {
        self.values.insert((node, port), value);
    }

    /// Clear all feedback values.
    pub fn clear(&mut self) {
        self.values.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cancellation_token() {
        let token = CancellationToken::new();
        assert!(!token.is_cancelled());

        token.cancel();
        assert!(token.is_cancelled());

        token.reset();
        assert!(!token.is_cancelled());
    }

    #[test]
    fn test_cancellation_token_clone() {
        let token1 = CancellationToken::new();
        let token2 = token1.clone();

        token1.cancel();
        assert!(token2.is_cancelled());
    }

    #[test]
    fn test_eval_context_defaults() {
        let ctx = EvalContext::new();
        assert!(!ctx.is_cancelled());
        assert_eq!(ctx.time, 0.0);
        assert_eq!(ctx.frame, 0);
        assert!(!ctx.preview_mode);
        assert_eq!(ctx.seed, 0);
    }

    #[test]
    fn test_eval_context_builder() {
        let token = CancellationToken::new();
        let ctx = EvalContext::new()
            .with_cancel(token.clone())
            .with_time(1.5, 90, 1.0 / 60.0)
            .with_preview_mode(true)
            .with_resolution(1920, 1080)
            .with_seed(42);

        assert!(!ctx.is_cancelled());
        assert_eq!(ctx.time, 1.5);
        assert_eq!(ctx.frame, 90);
        assert!(ctx.preview_mode);
        assert_eq!(ctx.target_resolution, Some((1920, 1080)));
        assert_eq!(ctx.seed, 42);

        token.cancel();
        assert!(ctx.is_cancelled());
    }

    #[test]
    fn test_feedback_state() {
        let mut state = FeedbackState::new();
        assert!(state.get(0, 0).is_none());

        state.set(0, 0, Value::F32(1.0));
        assert_eq!(state.get(0, 0), Some(&Value::F32(1.0)));

        state.clear();
        assert!(state.get(0, 0).is_none());
    }
}
