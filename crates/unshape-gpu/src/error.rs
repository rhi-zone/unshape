//! GPU error types.

use thiserror::Error;

/// Errors that can occur during GPU operations.
#[derive(Error, Debug)]
pub enum GpuError {
    /// Failed to request GPU adapter.
    #[error("failed to request GPU adapter")]
    AdapterNotFound,

    /// Failed to request GPU device.
    #[error("failed to request GPU device: {0}")]
    DeviceRequestFailed(#[from] wgpu::RequestDeviceError),

    /// Shader compilation error.
    #[error("shader compilation error: {0}")]
    ShaderError(String),

    /// Buffer operation error.
    #[error("buffer operation failed: {0}")]
    BufferError(String),

    /// Invalid dimensions.
    #[error("invalid dimensions: {0}")]
    InvalidDimensions(String),

    /// Invalid input to a kernel.
    #[error("invalid input: {0}")]
    InvalidInput(String),
}

/// Result type for GPU operations.
pub type GpuResult<T> = Result<T, GpuError>;
