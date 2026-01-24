//! Evaluation context and utilities for graph execution.
//!
//! This module provides the execution context for node evaluation,
//! including cancellation, progress reporting, and evaluation parameters.
//!
//! # Evaluation Strategies
//!
//! The [`Evaluator`] trait allows different evaluation strategies:
//! - [`LazyEvaluator`]: Only computes nodes needed for requested outputs, with caching
//! - The eager evaluator in [`Graph::execute`](crate::Graph::execute) computes all nodes
//!
//! # Lazy Evaluation with Caching
//!
//! [`LazyEvaluator`] only computes nodes required for requested outputs,
//! and caches results for subsequent evaluations:
//!
//! ```ignore
//! use unshape_core::{Graph, LazyEvaluator, Evaluator, EvalContext};
//!
//! // Assume graph is built with nodes connected
//! let graph: Graph = /* ... */;
//! let output_node = /* node ID */;
//!
//! let mut evaluator = LazyEvaluator::new();
//! let ctx = EvalContext::new();
//!
//! // First call computes all upstream nodes
//! let result = evaluator.evaluate(&graph, &[output_node], &ctx)?;
//! println!("Computed {} nodes", result.computed_nodes.len());
//!
//! // Second call uses cache
//! let result = evaluator.evaluate(&graph, &[output_node], &ctx)?;
//! println!("Cache hits: {}", result.cached_nodes.len());
//! ```
//!
//! # EvalContext
//!
//! The evaluation context provides environment information for node execution:
//!
//! ```
//! use unshape_core::{EvalContext, CancellationToken};
//!
//! // Basic context
//! let ctx = EvalContext::new();
//!
//! // Context with time (for animation)
//! let ctx = EvalContext::new()
//!     .with_time(1.5, 90, 1.0 / 60.0)  // time=1.5s, frame=90, dt=16.6ms
//!     .with_seed(42);                   // deterministic randomness
//!
//! // Context with cancellation
//! let token = CancellationToken::new();
//! let ctx = EvalContext::new().with_cancel(token.clone());
//!
//! // Check cancellation in long-running code
//! assert!(!ctx.is_cancelled());
//! token.cancel();
//! assert!(ctx.is_cancelled());
//! ```
//!
//! # Cancellation
//!
//! Long-running evaluations can be cancelled cooperatively using [`CancellationToken`]:
//!
//! ```
//! use unshape_core::CancellationToken;
//!
//! let token = CancellationToken::new();
//! let token_clone = token.clone();
//!
//! // In one thread: signal cancellation
//! token.cancel();
//!
//! // In another thread: check for cancellation
//! assert!(token_clone.is_cancelled());
//! ```

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use crate::error::GraphError;
use crate::graph::{Graph, NodeId};
use crate::node::DynNode;
use crate::value::Value;

// ============================================================================
// Node Executor Trait
// ============================================================================

/// Strategy for executing a single node.
///
/// This trait allows different execution backends to be plugged into evaluators
/// without duplicating the traversal and caching logic. Implementations can:
/// - Route execution to CPU/GPU backends
/// - Apply transformations to inputs/outputs
/// - Collect execution metrics
///
/// # Example
///
/// ```ignore
/// use unshape_core::{NodeExecutor, DynNode, Value, EvalContext, GraphError};
///
/// struct LoggingExecutor;
///
/// impl NodeExecutor for LoggingExecutor {
///     fn execute(
///         &self,
///         node: &dyn DynNode,
///         inputs: &[Value],
///         ctx: &EvalContext,
///     ) -> Result<Vec<Value>, GraphError> {
///         println!("Executing node: {}", node.type_name());
///         node.execute(inputs, ctx)
///     }
/// }
/// ```
pub trait NodeExecutor: Send + Sync {
    /// Execute a node with the given inputs.
    fn execute(
        &self,
        node: &dyn DynNode,
        inputs: &[Value],
        ctx: &EvalContext,
    ) -> Result<Vec<Value>, GraphError>;
}

/// Default node executor that calls `node.execute()` directly.
///
/// This is the standard executor used by [`LazyEvaluator`] when no
/// custom executor is provided.
#[derive(Debug, Clone, Copy, Default)]
pub struct DefaultNodeExecutor;

impl NodeExecutor for DefaultNodeExecutor {
    fn execute(
        &self,
        node: &dyn DynNode,
        inputs: &[Value],
        ctx: &EvalContext,
    ) -> Result<Vec<Value>, GraphError> {
        node.execute(inputs, ctx)
    }
}

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

// ============================================================================
// Error Handling
// ============================================================================

/// How errors should be handled during evaluation.
///
/// This is a per-request configuration - different contexts want different behavior.
pub enum ErrorHandling {
    /// Stop on first error, return Err (default).
    FailFast,

    /// Propagate errors as `Value::Error`, continue evaluation.
    /// Downstream nodes receive the error and can propagate or handle it.
    Propagate,

    /// Use fallback values for failed nodes.
    /// The function receives (node_id, error) and returns Some(value) to use,
    /// or None to propagate the error.
    Fallback {
        default_fn: Box<dyn Fn(NodeId, &GraphError) -> Option<Value> + Send>,
    },
}

impl Default for ErrorHandling {
    fn default() -> Self {
        Self::FailFast
    }
}

impl ErrorHandling {
    /// Create a fallback handler using a static map of defaults.
    pub fn fallback_map(defaults: HashMap<NodeId, Value>) -> Self {
        Self::Fallback {
            default_fn: Box::new(move |id, _| defaults.get(&id).cloned()),
        }
    }
}

// ============================================================================
// Cache Policy
// ============================================================================

/// Key for cache lookup: (node_id, hash of input values).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CacheKey {
    pub node_id: NodeId,
    pub input_hash: u64,
}

impl CacheKey {
    /// Create a cache key from node ID and input values.
    pub fn new(node_id: NodeId, inputs: &[Value]) -> Self {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        for input in inputs {
            input.hash(&mut hasher);
        }
        Self {
            node_id,
            input_hash: hasher.finish(),
        }
    }
}

/// Entry in the evaluation cache.
#[derive(Debug, Clone)]
pub struct CacheEntry {
    /// Cached output values.
    pub outputs: Vec<Value>,
    /// When this entry was created.
    pub created_at: Instant,
    /// When this entry was last accessed.
    pub last_accessed: Instant,
    /// Size estimate in bytes (for memory-bounded caches).
    pub size_bytes: usize,
}

/// Policy for cache retention and eviction.
pub trait CachePolicy: Send {
    /// Called after node evaluation - should we cache this result?
    fn should_cache(&self, node: NodeId, outputs: &[Value]) -> bool;

    /// Called before lookup - is this entry still valid?
    fn is_valid(&self, key: &CacheKey, entry: &CacheEntry) -> bool;

    /// Called on memory pressure or explicit clear. Returns bytes freed.
    fn evict(&mut self, cache: &mut EvalCache) -> usize;
}

/// Cache that keeps all entries (never evicts).
#[derive(Debug, Default)]
pub struct KeepAllPolicy;

impl CachePolicy for KeepAllPolicy {
    fn should_cache(&self, _node: NodeId, _outputs: &[Value]) -> bool {
        true
    }

    fn is_valid(&self, _key: &CacheKey, _entry: &CacheEntry) -> bool {
        true
    }

    fn evict(&mut self, _cache: &mut EvalCache) -> usize {
        0 // Never evicts
    }
}

/// Evaluation cache for memoizing node outputs.
#[derive(Debug, Default)]
pub struct EvalCache {
    entries: HashMap<CacheKey, CacheEntry>,
}

impl EvalCache {
    /// Create an empty cache.
    pub fn new() -> Self {
        Self::default()
    }

    /// Get a cached result if available.
    pub fn get(&mut self, key: &CacheKey) -> Option<&CacheEntry> {
        if let Some(entry) = self.entries.get_mut(key) {
            entry.last_accessed = Instant::now();
            Some(entry)
        } else {
            None
        }
    }

    /// Store a result in the cache.
    pub fn insert(&mut self, key: CacheKey, outputs: Vec<Value>) {
        let size_bytes = outputs.iter().map(|_| 64).sum(); // rough estimate
        let now = Instant::now();
        self.entries.insert(
            key,
            CacheEntry {
                outputs,
                created_at: now,
                last_accessed: now,
                size_bytes,
            },
        );
    }

    /// Clear all entries.
    pub fn clear(&mut self) {
        self.entries.clear();
    }

    /// Number of cached entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if cache is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Remove a specific entry.
    pub fn remove(&mut self, key: &CacheKey) -> Option<CacheEntry> {
        self.entries.remove(key)
    }

    /// Iterate over all keys (for eviction policies).
    pub fn keys(&self) -> impl Iterator<Item = &CacheKey> {
        self.entries.keys()
    }
}

// ============================================================================
// Evaluator Trait
// ============================================================================

/// Result of graph evaluation.
#[derive(Debug)]
pub struct EvalResult {
    /// Output values for each requested node, in order.
    /// Each inner Vec contains the outputs for one node.
    pub outputs: Vec<Vec<Value>>,

    /// Nodes that were computed (for debugging/profiling).
    pub computed_nodes: Vec<NodeId>,

    /// Nodes that were served from cache.
    pub cached_nodes: Vec<NodeId>,

    /// Total evaluation time.
    pub elapsed: Duration,
}

/// Trait for graph evaluation strategies.
///
/// Different evaluators can implement different strategies:
/// - Lazy evaluation (only compute what's needed)
/// - Eager evaluation (compute everything)
/// - Incremental evaluation (recompute only dirty nodes)
/// - Parallel evaluation (compute independent nodes concurrently)
pub trait Evaluator {
    /// Evaluate the graph and return outputs for the requested nodes.
    ///
    /// # Arguments
    /// * `graph` - The graph to evaluate
    /// * `outputs` - Node IDs whose outputs to return
    /// * `ctx` - Evaluation context (time, cancellation, etc.)
    ///
    /// # Returns
    /// Output values for each requested node, or error if evaluation fails.
    fn evaluate(
        &mut self,
        graph: &Graph,
        outputs: &[NodeId],
        ctx: &EvalContext,
    ) -> Result<EvalResult, GraphError>;

    /// Invalidate cached results for a node and its dependents.
    fn invalidate(&mut self, node: NodeId);

    /// Clear all cached results.
    fn clear_cache(&mut self);
}

// ============================================================================
// Lazy Evaluator
// ============================================================================

/// Lazy evaluator that only computes nodes needed for requested outputs.
///
/// Features:
/// - Recursive pull-based evaluation
/// - Memoization of computed values
/// - Caching with pluggable policy
/// - Pluggable node execution via [`NodeExecutor`]
///
/// # Type Parameter
///
/// The executor `E` determines how nodes are executed. The default
/// [`DefaultNodeExecutor`] simply calls `node.execute()`. Custom executors
/// can route execution to different backends (CPU, GPU) or add instrumentation.
///
/// # Example: Custom Executor
///
/// ```ignore
/// use unshape_core::{LazyEvaluator, NodeExecutor, DynNode, Value, EvalContext, GraphError};
///
/// struct CountingExecutor { count: std::cell::Cell<usize> }
///
/// impl NodeExecutor for CountingExecutor {
///     fn execute(&self, node: &dyn DynNode, inputs: &[Value], ctx: &EvalContext) -> Result<Vec<Value>, GraphError> {
///         self.count.set(self.count.get() + 1);
///         node.execute(inputs, ctx)
///     }
/// }
///
/// let executor = CountingExecutor { count: std::cell::Cell::new(0) };
/// let evaluator = LazyEvaluator::with_executor(executor);
/// ```
pub struct LazyEvaluator<E: NodeExecutor = DefaultNodeExecutor> {
    executor: E,
    cache: EvalCache,
    policy: Box<dyn CachePolicy>,
}

impl Default for LazyEvaluator<DefaultNodeExecutor> {
    fn default() -> Self {
        Self::new()
    }
}

impl LazyEvaluator<DefaultNodeExecutor> {
    /// Create a new lazy evaluator with default executor and cache policy.
    pub fn new() -> Self {
        Self {
            executor: DefaultNodeExecutor,
            cache: EvalCache::new(),
            policy: Box::new(KeepAllPolicy),
        }
    }

    /// Create a lazy evaluator with a custom cache policy.
    pub fn with_policy(policy: impl CachePolicy + 'static) -> Self {
        Self {
            executor: DefaultNodeExecutor,
            cache: EvalCache::new(),
            policy: Box::new(policy),
        }
    }
}

impl<E: NodeExecutor> LazyEvaluator<E> {
    /// Create a lazy evaluator with a custom node executor.
    pub fn with_executor(executor: E) -> Self {
        Self {
            executor,
            cache: EvalCache::new(),
            policy: Box::new(KeepAllPolicy),
        }
    }

    /// Create a lazy evaluator with both custom executor and cache policy.
    pub fn with_executor_and_policy(executor: E, policy: impl CachePolicy + 'static) -> Self {
        Self {
            executor,
            cache: EvalCache::new(),
            policy: Box::new(policy),
        }
    }

    /// Returns a reference to the node executor.
    pub fn executor(&self) -> &E {
        &self.executor
    }

    /// Returns a mutable reference to the node executor.
    pub fn executor_mut(&mut self) -> &mut E {
        &mut self.executor
    }

    /// Recursively evaluate a node, using cache when possible.
    fn evaluate_node(
        &mut self,
        graph: &Graph,
        node_id: NodeId,
        ctx: &EvalContext,
        computed: &mut Vec<NodeId>,
        cached: &mut Vec<NodeId>,
    ) -> Result<Vec<Value>, GraphError> {
        // Check for cancellation
        if ctx.is_cancelled() {
            return Err(GraphError::Cancelled);
        }

        let node = graph
            .get_node(node_id)
            .ok_or(GraphError::NodeNotFound(node_id))?;

        let inputs_desc = node.inputs();
        let num_inputs = inputs_desc.len();

        // Gather inputs by recursively evaluating upstream nodes
        let mut inputs = Vec::with_capacity(num_inputs);
        for port in 0..num_inputs {
            // Find wire that feeds this input
            let wire = graph
                .wires()
                .iter()
                .find(|w| w.to_node == node_id && w.to_port == port);

            match wire {
                Some(w) => {
                    // Recursively evaluate the upstream node
                    let upstream_outputs =
                        self.evaluate_node(graph, w.from_node, ctx, computed, cached)?;
                    let value = upstream_outputs.get(w.from_port).cloned().ok_or_else(|| {
                        GraphError::ExecutionError(format!(
                            "missing output port {} on node {}",
                            w.from_port, w.from_node
                        ))
                    })?;
                    inputs.push(value);
                }
                None => {
                    return Err(GraphError::UnconnectedInput {
                        node: node_id,
                        port,
                    });
                }
            }
        }

        // Check cache
        let cache_key = CacheKey::new(node_id, &inputs);
        if let Some(entry) = self.cache.get(&cache_key) {
            if self.policy.is_valid(&cache_key, entry) {
                cached.push(node_id);
                return Ok(entry.outputs.clone());
            }
        }

        // Execute node through the executor
        let outputs = self.executor.execute(node.as_ref(), &inputs, ctx)?;

        // Cache result if policy allows
        if self.policy.should_cache(node_id, &outputs) {
            self.cache.insert(cache_key, outputs.clone());
        }

        computed.push(node_id);
        Ok(outputs)
    }
}

impl<E: NodeExecutor> Evaluator for LazyEvaluator<E> {
    fn evaluate(
        &mut self,
        graph: &Graph,
        outputs: &[NodeId],
        ctx: &EvalContext,
    ) -> Result<EvalResult, GraphError> {
        let start = Instant::now();
        let mut computed = Vec::new();
        let mut cached = Vec::new();
        let mut results = Vec::with_capacity(outputs.len());

        for &node_id in outputs {
            let node_outputs =
                self.evaluate_node(graph, node_id, ctx, &mut computed, &mut cached)?;
            results.push(node_outputs);
        }

        Ok(EvalResult {
            outputs: results,
            computed_nodes: computed,
            cached_nodes: cached,
            elapsed: start.elapsed(),
        })
    }

    fn invalidate(&mut self, node: NodeId) {
        // Remove all cache entries for this node
        // Note: This doesn't invalidate dependents - for that we'd need dependency tracking
        let keys_to_remove: Vec<_> = self
            .cache
            .keys()
            .filter(|k| k.node_id == node)
            .cloned()
            .collect();
        for key in keys_to_remove {
            self.cache.remove(&key);
        }
    }

    fn clear_cache(&mut self) {
        self.cache.clear();
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

    #[test]
    fn test_cache_key() {
        let key1 = CacheKey::new(0, &[Value::F32(1.0), Value::F32(2.0)]);
        let key2 = CacheKey::new(0, &[Value::F32(1.0), Value::F32(2.0)]);
        let key3 = CacheKey::new(0, &[Value::F32(1.0), Value::F32(3.0)]);
        let key4 = CacheKey::new(1, &[Value::F32(1.0), Value::F32(2.0)]);

        // Same inputs should produce same key
        assert_eq!(key1, key2);
        assert_eq!(key1.input_hash, key2.input_hash);

        // Different inputs should produce different hash
        assert_ne!(key1.input_hash, key3.input_hash);

        // Different node should produce different key
        assert_ne!(key1, key4);
    }

    #[test]
    fn test_eval_cache() {
        let mut cache = EvalCache::new();
        assert!(cache.is_empty());

        let key = CacheKey::new(0, &[Value::F32(1.0)]);
        cache.insert(key, vec![Value::F32(2.0)]);

        assert_eq!(cache.len(), 1);
        assert!(!cache.is_empty());

        let entry = cache.get(&key).unwrap();
        assert_eq!(entry.outputs, vec![Value::F32(2.0)]);

        cache.clear();
        assert!(cache.is_empty());
    }

    #[test]
    fn test_keep_all_policy() {
        let policy = KeepAllPolicy;
        let key = CacheKey::new(0, &[]);
        let entry = CacheEntry {
            outputs: vec![Value::F32(1.0)],
            created_at: Instant::now(),
            last_accessed: Instant::now(),
            size_bytes: 64,
        };

        assert!(policy.should_cache(0, &[Value::F32(1.0)]));
        assert!(policy.is_valid(&key, &entry));
    }
}
