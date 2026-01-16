//! GPU texture types for field output.

use crate::GpuContext;
use crate::error::{GpuError, GpuResult};
use rhizome_resin_core::{DataLocation, GraphValue};
use std::any::Any;

/// Texture format for GPU output.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextureFormat {
    /// Single channel grayscale (f32).
    R32Float,
    /// RGBA with 8 bits per channel.
    Rgba8Unorm,
    /// RGBA with 32-bit floats.
    Rgba32Float,
}

impl TextureFormat {
    /// Returns the wgpu texture format.
    pub fn to_wgpu(self) -> wgpu::TextureFormat {
        match self {
            TextureFormat::R32Float => wgpu::TextureFormat::R32Float,
            TextureFormat::Rgba8Unorm => wgpu::TextureFormat::Rgba8Unorm,
            TextureFormat::Rgba32Float => wgpu::TextureFormat::Rgba32Float,
        }
    }

    /// Returns bytes per pixel.
    pub fn bytes_per_pixel(self) -> u32 {
        match self {
            TextureFormat::R32Float => 4,
            TextureFormat::Rgba8Unorm => 4,
            TextureFormat::Rgba32Float => 16,
        }
    }
}

/// A GPU texture that can be read back to CPU.
pub struct GpuTexture {
    pub(crate) texture: wgpu::Texture,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) format: TextureFormat,
}

impl GpuTexture {
    /// Creates a new GPU texture for compute output.
    pub fn new(
        ctx: &GpuContext,
        width: u32,
        height: u32,
        format: TextureFormat,
    ) -> GpuResult<Self> {
        Self::with_usage(
            ctx,
            width,
            height,
            format,
            wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC,
        )
    }

    /// Creates a new GPU texture that can be used as both input and output.
    ///
    /// This texture can be sampled from in shaders and also written to as storage.
    pub fn new_input_output(
        ctx: &GpuContext,
        width: u32,
        height: u32,
        format: TextureFormat,
    ) -> GpuResult<Self> {
        Self::with_usage(
            ctx,
            width,
            height,
            format,
            wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::COPY_DST,
        )
    }

    /// Creates a GPU texture with custom usage flags.
    pub fn with_usage(
        ctx: &GpuContext,
        width: u32,
        height: u32,
        format: TextureFormat,
        usage: wgpu::TextureUsages,
    ) -> GpuResult<Self> {
        if width == 0 || height == 0 {
            return Err(GpuError::InvalidDimensions(format!(
                "texture dimensions must be > 0, got {}x{}",
                width, height
            )));
        }

        let texture = ctx.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("resin_gpu_texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: format.to_wgpu(),
            usage,
            view_formats: &[],
        });

        Ok(Self {
            texture,
            width,
            height,
            format,
        })
    }

    /// Returns the texture width.
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Returns the texture height.
    pub fn height(&self) -> u32 {
        self.height
    }

    /// Returns the texture format.
    pub fn format(&self) -> TextureFormat {
        self.format
    }

    /// Creates a view for binding to shaders.
    pub fn create_view(&self) -> wgpu::TextureView {
        self.texture
            .create_view(&wgpu::TextureViewDescriptor::default())
    }

    /// Reads the texture data back to CPU as raw bytes.
    pub fn read_to_bytes(&self, ctx: &GpuContext) -> Vec<u8> {
        let bytes_per_pixel = self.format.bytes_per_pixel();
        let unpadded_row_bytes = self.width * bytes_per_pixel;
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let padded_row_bytes = (unpadded_row_bytes + align - 1) / align * align;

        let buffer_size = (padded_row_bytes * self.height) as u64;

        let staging_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging_buffer"),
            size: buffer_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("read_texture_encoder"),
            });

        encoder.copy_texture_to_buffer(
            wgpu::ImageCopyTexture {
                texture: &self.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::ImageCopyBuffer {
                buffer: &staging_buffer,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_row_bytes),
                    rows_per_image: Some(self.height),
                },
            },
            wgpu::Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
        );

        ctx.queue.submit(Some(encoder.finish()));

        let buffer_slice = staging_buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });

        ctx.device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();

        let data = buffer_slice.get_mapped_range();

        // Remove row padding if present
        let mut result = Vec::with_capacity((self.width * self.height * bytes_per_pixel) as usize);
        for row in 0..self.height {
            let start = (row * padded_row_bytes) as usize;
            let end = start + unpadded_row_bytes as usize;
            result.extend_from_slice(&data[start..end]);
        }

        result
    }

    /// Reads the texture as f32 values (for R32Float format).
    pub fn read_to_f32(&self, ctx: &GpuContext) -> Vec<f32> {
        let bytes = self.read_to_bytes(ctx);
        bytemuck::cast_slice(&bytes).to_vec()
    }

    /// Reads the texture as RGBA u8 values (for Rgba8Unorm format).
    pub fn read_to_rgba8(&self, ctx: &GpuContext) -> Vec<u8> {
        self.read_to_bytes(ctx)
    }
}

impl std::fmt::Debug for GpuTexture {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuTexture")
            .field("width", &self.width)
            .field("height", &self.height)
            .field("format", &self.format)
            .finish_non_exhaustive()
    }
}

impl GraphValue for GpuTexture {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn type_name(&self) -> &'static str {
        "GpuTexture"
    }

    fn location(&self) -> DataLocation {
        // TODO: Support multiple GPU devices
        DataLocation::Gpu { device_id: 0 }
    }
}
