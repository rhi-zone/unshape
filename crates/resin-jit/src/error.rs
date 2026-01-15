//! Error types for JIT compilation.

/// Errors that can occur during JIT compilation.
#[derive(Debug, thiserror::Error)]
pub enum JitError {
    /// Cranelift module creation failed.
    #[cfg(feature = "cranelift")]
    #[error("module error: {0}")]
    Module(#[from] cranelift_module::ModuleError),

    /// Unsupported node type for JIT compilation.
    #[error("unsupported node: {0}")]
    UnsupportedNode(String),

    /// Graph structure not suitable for JIT.
    #[error("graph error: {0}")]
    Graph(String),

    /// Compilation failed.
    #[error("compilation error: {0}")]
    Compilation(String),
}

/// Result type for JIT operations.
pub type JitResult<T> = Result<T, JitError>;
