//! GPU kernel implementations for graph nodes.
//!
//! This module provides GPU kernels for nodes that can benefit from
//! GPU acceleration. Each kernel wraps existing GPU functions and
//! adapts them to the [`GpuKernel`] trait.

use crate::backend::GpuKernel;
use crate::noise::{NoiseConfig, NoiseType, generate_noise_texture_gpu};
use crate::texture::GpuTexture;
use crate::{GpuContext, GpuError};
use rhizome_resin_core::{DynNode, EvalContext, GraphError, PortDescriptor, Value, ValueType};
use std::any::Any;
use std::sync::Arc;

// ============================================================================
// Noise Texture Node
// ============================================================================

/// Node that generates a noise texture on the GPU.
///
/// This node produces a [`GpuTexture`] containing procedural noise.
/// When executed through a GPU backend with the registered kernel,
/// it runs entirely on the GPU. Otherwise, it falls back to CPU.
///
/// # Inputs
///
/// None - all parameters are stored in the node.
///
/// # Outputs
///
/// - `texture`: The generated noise texture as an opaque value
///
/// # Example
///
/// ```ignore
/// use rhizome_resin_gpu::{NoiseTextureNode, NoiseConfig, NoiseType};
///
/// let node = NoiseTextureNode::new(512, 512, NoiseConfig::new(NoiseType::Perlin, 4.0));
/// ```
#[derive(Debug, Clone)]
pub struct NoiseTextureNode {
    /// Width of the texture in pixels.
    pub width: u32,
    /// Height of the texture in pixels.
    pub height: u32,
    /// Noise configuration.
    pub config: NoiseConfig,
}

impl NoiseTextureNode {
    /// Creates a new noise texture node.
    pub fn new(width: u32, height: u32, config: NoiseConfig) -> Self {
        Self {
            width,
            height,
            config,
        }
    }

    /// Creates a noise texture node with default Perlin noise.
    pub fn perlin(width: u32, height: u32, scale: f32) -> Self {
        Self::new(width, height, NoiseConfig::new(NoiseType::Perlin, scale))
    }

    /// Creates a noise texture node with Simplex noise.
    pub fn simplex(width: u32, height: u32, scale: f32) -> Self {
        Self::new(width, height, NoiseConfig::new(NoiseType::Simplex, scale))
    }
}

impl DynNode for NoiseTextureNode {
    fn type_name(&self) -> &'static str {
        "NoiseTextureNode"
    }

    fn inputs(&self) -> Vec<PortDescriptor> {
        vec![] // No inputs - parameters are stored in the node
    }

    fn outputs(&self) -> Vec<PortDescriptor> {
        vec![PortDescriptor::new(
            "texture",
            ValueType::Custom {
                type_id: std::any::TypeId::of::<GpuTexture>(),
                name: "GpuTexture",
            },
        )]
    }

    fn execute(&self, _inputs: &[Value], _ctx: &EvalContext) -> Result<Vec<Value>, GraphError> {
        // CPU fallback: generate noise data on CPU
        // This is a simplified version - real implementation would use CPU noise
        let pixel_count = (self.width * self.height) as usize;
        let mut data = Vec::with_capacity(pixel_count);

        // Simple CPU noise fallback (not as sophisticated as GPU version)
        for y in 0..self.height {
            for x in 0..self.width {
                let fx = x as f32 / self.width as f32 * self.config.scale;
                let fy = y as f32 / self.height as f32 * self.config.scale;
                // Simple hash-based noise for CPU fallback
                let noise = simple_noise(fx, fy, self.config.seed);
                data.push(noise);
            }
        }

        // Return as a CPU-side noise data wrapper
        Ok(vec![Value::Opaque(Arc::new(CpuNoiseData {
            width: self.width,
            height: self.height,
            data,
        }))])
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// CPU-side noise data (fallback when no GPU available).
#[derive(Debug, Clone)]
pub struct CpuNoiseData {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// Noise values in row-major order.
    pub data: Vec<f32>,
}

impl rhizome_resin_core::GraphValue for CpuNoiseData {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn type_name(&self) -> &'static str {
        "CpuNoiseData"
    }
}

/// Simple hash-based noise for CPU fallback.
fn simple_noise(x: f32, y: f32, seed: u32) -> f32 {
    let n = (x * 12.9898 + y * 78.233 + seed as f32 * 0.1) * 43758.5453;
    n.fract()
}

// ============================================================================
// Noise Texture Kernel (GPU)
// ============================================================================

/// GPU kernel for noise texture generation.
///
/// This kernel executes [`NoiseTextureNode`] on the GPU using compute shaders.
/// Register it with [`GpuComputeBackend`](crate::GpuComputeBackend) to enable
/// GPU-accelerated noise generation.
///
/// # Example
///
/// ```ignore
/// use rhizome_resin_gpu::{GpuComputeBackend, NoiseTextureNode, NoiseTextureKernel};
/// use std::sync::Arc;
///
/// let backend = GpuComputeBackend::new()?;
/// backend.register_kernel::<NoiseTextureNode>(Arc::new(NoiseTextureKernel));
/// ```
pub struct NoiseTextureKernel;

impl GpuKernel for NoiseTextureKernel {
    fn execute(
        &self,
        ctx: &GpuContext,
        inputs: &[Value],
        _eval_ctx: &EvalContext,
    ) -> Result<Vec<Value>, GpuError> {
        // The node parameters come from the node itself, not inputs
        // We need to get them from the eval context or reconstruct
        // For now, we'll expect the kernel to be called with node info embedded

        // This is a limitation of the current design - the kernel doesn't
        // have direct access to the node. We'll need to pass parameters
        // through inputs or use a different approach.

        // For demonstration, we'll accept parameters as inputs:
        // inputs[0] = width (I32)
        // inputs[1] = height (I32)
        // inputs[2] = scale (F32)
        // inputs[3] = noise_type (I32: 0=Perlin, 1=Simplex, 2=Value, 3=Worley)
        // inputs[4] = seed (I32)

        if inputs.len() < 5 {
            return Err(GpuError::InvalidInput(
                "NoiseTextureKernel requires 5 inputs: width, height, scale, noise_type, seed"
                    .into(),
            ));
        }

        let width = inputs[0]
            .as_i32()
            .map_err(|e| GpuError::InvalidInput(e.to_string()))? as u32;
        let height = inputs[1]
            .as_i32()
            .map_err(|e| GpuError::InvalidInput(e.to_string()))? as u32;
        let scale = inputs[2]
            .as_f32()
            .map_err(|e| GpuError::InvalidInput(e.to_string()))?;
        let noise_type_int = inputs[3]
            .as_i32()
            .map_err(|e| GpuError::InvalidInput(e.to_string()))?;
        let seed = inputs[4]
            .as_i32()
            .map_err(|e| GpuError::InvalidInput(e.to_string()))? as u32;

        let noise_type = match noise_type_int {
            0 => NoiseType::Perlin,
            1 => NoiseType::Simplex,
            2 => NoiseType::Value,
            3 => NoiseType::Worley,
            _ => NoiseType::Perlin,
        };

        let config = NoiseConfig {
            noise_type,
            scale,
            seed,
            ..Default::default()
        };

        let texture = generate_noise_texture_gpu(ctx, width, height, &config)?;

        Ok(vec![Value::Opaque(Arc::new(texture))])
    }
}

// ============================================================================
// Parameterized Noise Node (for kernel compatibility)
// ============================================================================

/// Noise texture node with inputs for kernel compatibility.
///
/// Unlike [`NoiseTextureNode`] which stores parameters internally,
/// this node accepts parameters as inputs, making it compatible with
/// the GPU kernel execution model.
#[derive(Debug, Clone, Default)]
pub struct ParameterizedNoiseNode;

impl DynNode for ParameterizedNoiseNode {
    fn type_name(&self) -> &'static str {
        "ParameterizedNoiseNode"
    }

    fn inputs(&self) -> Vec<PortDescriptor> {
        vec![
            PortDescriptor::new("width", ValueType::I32),
            PortDescriptor::new("height", ValueType::I32),
            PortDescriptor::new("scale", ValueType::F32),
            PortDescriptor::new("noise_type", ValueType::I32),
            PortDescriptor::new("seed", ValueType::I32),
        ]
    }

    fn outputs(&self) -> Vec<PortDescriptor> {
        vec![PortDescriptor::new(
            "texture",
            ValueType::Custom {
                type_id: std::any::TypeId::of::<GpuTexture>(),
                name: "GpuTexture",
            },
        )]
    }

    fn execute(&self, inputs: &[Value], _ctx: &EvalContext) -> Result<Vec<Value>, GraphError> {
        // CPU fallback
        let width = inputs[0]
            .as_i32()
            .map_err(|e| GraphError::ExecutionError(e.to_string()))? as u32;
        let height = inputs[1]
            .as_i32()
            .map_err(|e| GraphError::ExecutionError(e.to_string()))? as u32;
        let scale = inputs[2]
            .as_f32()
            .map_err(|e| GraphError::ExecutionError(e.to_string()))?;
        let seed = inputs[4]
            .as_i32()
            .map_err(|e| GraphError::ExecutionError(e.to_string()))? as u32;

        let pixel_count = (width * height) as usize;
        let mut data = Vec::with_capacity(pixel_count);

        for y in 0..height {
            for x in 0..width {
                let fx = x as f32 / width as f32 * scale;
                let fy = y as f32 / height as f32 * scale;
                let noise = simple_noise(fx, fy, seed);
                data.push(noise);
            }
        }

        Ok(vec![Value::Opaque(Arc::new(CpuNoiseData {
            width,
            height,
            data,
        }))])
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Registers all GPU kernels with a [`GpuComputeBackend`](crate::GpuComputeBackend).
///
/// Call this to enable GPU acceleration for built-in node types.
///
/// # Example
///
/// ```ignore
/// use rhizome_resin_gpu::{GpuComputeBackend, register_kernels};
///
/// let backend = GpuComputeBackend::new()?;
/// register_kernels(&backend);
/// ```
pub fn register_kernels(backend: &crate::GpuComputeBackend) {
    backend.register_kernel::<ParameterizedNoiseNode>(Arc::new(NoiseTextureKernel));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::GpuComputeBackend;
    use rhizome_resin_backend::ComputeBackend;

    #[test]
    fn test_noise_texture_node_cpu_fallback() {
        let node = NoiseTextureNode::perlin(64, 64, 4.0);
        let ctx = EvalContext::new();
        let result = node.execute(&[], &ctx).unwrap();

        assert_eq!(result.len(), 1);

        // Should be CpuNoiseData
        let noise_data = result[0].downcast_ref::<CpuNoiseData>().unwrap();
        assert_eq!(noise_data.width, 64);
        assert_eq!(noise_data.height, 64);
        assert_eq!(noise_data.data.len(), 64 * 64);
    }

    #[test]
    fn test_parameterized_noise_node_cpu() {
        let node = ParameterizedNoiseNode;
        let ctx = EvalContext::new();
        let inputs = vec![
            Value::I32(32),  // width
            Value::I32(32),  // height
            Value::F32(4.0), // scale
            Value::I32(0),   // noise_type (Perlin)
            Value::I32(42),  // seed
        ];

        let result = node.execute(&inputs, &ctx).unwrap();
        assert_eq!(result.len(), 1);

        let noise_data = result[0].downcast_ref::<CpuNoiseData>().unwrap();
        assert_eq!(noise_data.width, 32);
        assert_eq!(noise_data.height, 32);
    }

    #[test]
    fn test_gpu_kernel_registration() {
        let backend = match GpuComputeBackend::new() {
            Ok(b) => b,
            Err(_) => return, // No GPU available
        };

        register_kernels(&backend);

        // Should now support ParameterizedNoiseNode
        let node = ParameterizedNoiseNode;
        assert!(backend.supports_node(&node));
    }

    #[test]
    fn test_gpu_kernel_execution() {
        let backend = match GpuComputeBackend::new() {
            Ok(b) => b,
            Err(_) => return, // No GPU available
        };

        register_kernels(&backend);

        let node = ParameterizedNoiseNode;
        let ctx = EvalContext::new();
        let inputs = vec![
            Value::I32(64),  // width
            Value::I32(64),  // height
            Value::F32(4.0), // scale
            Value::I32(0),   // noise_type
            Value::I32(0),   // seed
        ];

        let result = backend.execute(&node, &inputs, &ctx).unwrap();
        assert_eq!(result.len(), 1);

        // Should be GpuTexture when executed on GPU
        let texture = result[0].downcast_ref::<GpuTexture>().unwrap();
        assert_eq!(texture.width(), 64);
        assert_eq!(texture.height(), 64);
    }
}
