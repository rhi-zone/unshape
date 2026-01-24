//! Compiled function types.

use crate::traits::SimdWidth;

/// A JIT-compiled function for scalar processing.
///
/// # Lifetime
///
/// The compiled code lives in the `JitCompiler`'s internal module.
/// This struct must not outlive the compiler that created it.
#[cfg(feature = "cranelift")]
pub struct CompiledScalar {
    /// Raw function pointer.
    func: fn(f32) -> f32,
    /// Number of inputs expected.
    num_inputs: usize,
    /// Number of outputs produced.
    num_outputs: usize,
}

#[cfg(feature = "cranelift")]
impl CompiledScalar {
    /// Creates a new compiled scalar function.
    ///
    /// # Safety
    ///
    /// The function pointer must be valid and have the expected signature.
    /// The caller must ensure the JitCompiler that created this outlives it.
    pub(crate) unsafe fn new(func: *const u8, num_inputs: usize, num_outputs: usize) -> Self {
        Self {
            func: unsafe { std::mem::transmute(func) },
            num_inputs,
            num_outputs,
        }
    }

    /// Returns the number of inputs this function expects.
    pub fn num_inputs(&self) -> usize {
        self.num_inputs
    }

    /// Returns the number of outputs this function produces.
    pub fn num_outputs(&self) -> usize {
        self.num_outputs
    }

    /// Calls the compiled function with a single f32 input.
    ///
    /// # Panics
    ///
    /// Panics if `num_inputs != 1` or `num_outputs != 1`.
    pub fn call_f32(&self, input: f32) -> f32 {
        assert_eq!(self.num_inputs, 1);
        assert_eq!(self.num_outputs, 1);
        (self.func)(input)
    }
}

// Safety: The compiled code doesn't capture any thread-local state.
#[cfg(feature = "cranelift")]
unsafe impl Send for CompiledScalar {}
#[cfg(feature = "cranelift")]
unsafe impl Sync for CompiledScalar {}

/// State container for stateful node processors.
///
/// Used by block processing to maintain state across samples.
#[cfg(feature = "cranelift")]
pub struct BatchState<Ctx> {
    /// Domain-specific context (e.g., AudioContext).
    pub ctx: Ctx,
    /// Per-node output values (for graph evaluation).
    pub values: Vec<f32>,
    /// Stateful processor closures.
    pub processors: Vec<Box<dyn FnMut(f32, &Ctx) -> f32 + Send>>,
}

#[cfg(feature = "cranelift")]
impl<Ctx: Default> BatchState<Ctx> {
    /// Creates a new batch state with default context.
    pub fn new(node_count: usize) -> Self {
        Self {
            ctx: Ctx::default(),
            values: vec![0.0; node_count],
            processors: Vec::new(),
        }
    }
}

#[cfg(feature = "cranelift")]
impl<Ctx> BatchState<Ctx> {
    /// Creates batch state with the given context.
    pub fn with_ctx(ctx: Ctx, node_count: usize) -> Self {
        Self {
            ctx,
            values: vec![0.0; node_count],
            processors: Vec::new(),
        }
    }

    /// Resets all cached values to zero.
    pub fn reset_values(&mut self) {
        for v in &mut self.values {
            *v = 0.0;
        }
    }
}

/// A JIT-compiled function for SIMD batched processing.
///
/// Processes multiple samples at once using vector operations.
///
/// # Lifetime
///
/// The compiled code lives in the `JitCompiler`'s internal module.
/// This struct must not outlive the compiler that created it.
#[cfg(feature = "cranelift")]
pub struct CompiledBatch<Ctx> {
    /// Block processing function.
    /// Signature: fn(input: *const f32, output: *mut f32, len: usize, state: *mut BatchState<Ctx>)
    func: fn(*const f32, *mut f32, usize, *mut BatchState<Ctx>),
    /// SIMD width used.
    width: SimdWidth,
    /// Stateful node processors.
    pub(crate) state: Box<BatchState<Ctx>>,
}

#[cfg(feature = "cranelift")]
impl<Ctx> CompiledBatch<Ctx> {
    /// Creates a new compiled batch function.
    ///
    /// # Safety
    ///
    /// The function pointer must be valid and have the expected signature.
    /// The caller must ensure the JitCompiler that created this outlives it.
    pub(crate) unsafe fn new(
        func: *const u8,
        width: SimdWidth,
        state: Box<BatchState<Ctx>>,
    ) -> Self {
        Self {
            func: unsafe { std::mem::transmute(func) },
            width,
            state,
        }
    }

    /// Returns the SIMD width used for this compiled function.
    pub fn simd_width(&self) -> SimdWidth {
        self.width
    }

    /// Processes a batch of inputs through the compiled function.
    ///
    /// # Arguments
    ///
    /// * `input` - Input buffer
    /// * `output` - Output buffer (must be same length as input)
    ///
    /// # Panics
    ///
    /// Panics if input and output lengths don't match.
    pub fn process_batch(&mut self, input: &[f32], output: &mut [f32]) {
        assert_eq!(
            input.len(),
            output.len(),
            "input and output must be same length"
        );

        let func: fn(*const f32, *mut f32, usize, *mut BatchState<Ctx>) =
            unsafe { std::mem::transmute(self.func) };

        func(
            input.as_ptr(),
            output.as_mut_ptr(),
            input.len(),
            self.state.as_mut(),
        );
    }

    /// Resets the state for all stateful processors.
    pub fn reset(&mut self) {
        self.state.reset_values();
    }

    /// Returns a reference to the batch state.
    pub fn state(&self) -> &BatchState<Ctx> {
        &self.state
    }

    /// Returns a mutable reference to the batch state.
    pub fn state_mut(&mut self) -> &mut BatchState<Ctx> {
        &mut self.state
    }
}

// Safety: BatchState requires Ctx: Send and processors are Send.
#[cfg(feature = "cranelift")]
unsafe impl<Ctx: Send> Send for CompiledBatch<Ctx> {}
