//! GPU compute backend for resin fields.
//!
//! Provides GPU-accelerated evaluation of field functions using wgpu compute shaders.
//!
//! # Features
//!
//! - Batch field sampling: evaluate fields at thousands of points in parallel
//! - Texture generation: render fields to 2D textures
//! - Built-in noise shaders: Perlin, Simplex, FBM
//! - Compute backend for heterogeneous execution
//!
//! # Compute Backend
//!
//! The [`GpuComputeBackend`] implements the [`ComputeBackend`](unshape_backend::ComputeBackend)
//! trait, allowing GPU execution to be selected via [`ExecutionPolicy`](unshape_backend::ExecutionPolicy).
//!
//! ```ignore
//! use unshape_gpu::{GpuComputeBackend, GpuKernel};
//! use unshape_backend::BackendRegistry;
//!
//! let mut registry = BackendRegistry::with_cpu();
//! if let Ok(gpu) = GpuComputeBackend::new() {
//!     registry.register(Arc::new(gpu));
//! }
//! ```
//!
//! # Example
//!
//! ```ignore
//! use unshape_gpu::{GpuContext, NoiseType, noise_texture_gpu};
//!
//! let ctx = GpuContext::new()?;
//! let texture = noise_texture_gpu(&ctx, 512, 512, NoiseType::Perlin, 4.0)?;
//! ```

mod backend;
mod context;
mod error;
mod kernels;
mod noise;
mod texture;

#[cfg(feature = "image-expr")]
mod image_expr;

pub use backend::{GpuComputeBackend, GpuKernel};
pub use context::GpuContext;
pub use error::GpuError;
pub use kernels::{
    CpuNoiseData, NoiseTextureKernel, NoiseTextureNode, ParameterizedNoiseNode, register_kernels,
};
#[cfg(feature = "image-expr")]
pub use kernels::{MapPixelsKernel, MapPixelsNode, RemapUvKernel, RemapUvNode};
pub use noise::{NoiseConfig, NoiseType, generate_noise_texture_gpu, noise_texture_gpu};
pub use texture::{GpuTexture, TextureFormat};

#[cfg(feature = "image-expr")]
pub use image_expr::{map_pixels_gpu, remap_uv_gpu};

/// Registers all GPU operations with an [`OpRegistry`].
///
/// Call this to enable deserialization of GPU ops from saved pipelines.
#[cfg(feature = "dynop")]
pub fn register_ops(registry: &mut unshape_op::OpRegistry) {
    registry.register_type::<NoiseConfig>("resin::NoiseConfig");
}
