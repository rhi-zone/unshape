//! GPU context and resource management.

use crate::error::{GpuError, GpuResult};
use wgpu::{Adapter, Device, Instance, Queue};

/// GPU context for compute operations.
///
/// Manages the wgpu instance, adapter, device, and queue.
/// Create one context and reuse it for multiple operations.
pub struct GpuContext {
    #[allow(dead_code)]
    instance: Instance,
    #[allow(dead_code)]
    adapter: Adapter,
    pub(crate) device: Device,
    pub(crate) queue: Queue,
}

impl GpuContext {
    /// Creates a new GPU context.
    ///
    /// This will request a GPU adapter and device. Prefers high-performance
    /// discrete GPUs when available.
    pub fn new() -> GpuResult<Self> {
        pollster::block_on(Self::new_async())
    }

    /// Creates a new GPU context asynchronously.
    pub async fn new_async() -> GpuResult<Self> {
        let instance = Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
            })
            .await
            .ok_or(GpuError::AdapterNotFound)?;

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("resin-gpu"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    memory_hints: wgpu::MemoryHints::Performance,
                },
                None,
            )
            .await?;

        Ok(Self {
            instance,
            adapter,
            device,
            queue,
        })
    }

    /// Returns the device name/info.
    pub fn device_info(&self) -> String {
        format!("{:?}", self.adapter.get_info())
    }
}
