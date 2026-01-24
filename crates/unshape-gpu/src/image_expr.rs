//! GPU-accelerated image expression evaluation.
//!
//! Provides GPU compute shader execution for image primitives defined in `unshape-image`.
//! Uses dew's WGSL code generation to compile expressions to shaders.

use crate::GpuContext;
use crate::error::{GpuError, GpuResult};
use crate::texture::{GpuTexture, TextureFormat};
use bytemuck::{Pod, Zeroable};
use rhizome_dew_linalg::{Type, wgsl::emit_wgsl};
use unshape_image::{ColorExpr, UvExpr};
use std::collections::HashMap;
use wgpu::util::DeviceExt;

/// Uniform buffer for image transforms.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct ImageUniforms {
    width: u32,
    height: u32,
    _padding: [u32; 2],
}

/// Remaps UV coordinates using an expression, executed on the GPU.
///
/// This is the GPU-accelerated version of `resin_image::remap_uv`. The expression
/// is compiled to a WGSL compute shader and executed in parallel across all pixels.
///
/// # Arguments
///
/// * `ctx` - GPU context
/// * `input` - Source texture to sample from
/// * `expr` - UV remapping expression (Vec2 → Vec2)
///
/// # Example
///
/// ```ignore
/// use unshape_gpu::{GpuContext, remap_uv_gpu};
/// use unshape_image::UvExpr;
///
/// let ctx = GpuContext::new()?;
/// let input = load_texture(&ctx, "image.png")?;
///
/// // Apply wave distortion on GPU
/// let wave = UvExpr::Add(
///     Box::new(UvExpr::Uv),
///     Box::new(UvExpr::Vec2 {
///         x: Box::new(UvExpr::Mul(
///             Box::new(UvExpr::Constant(0.02)),
///             Box::new(UvExpr::Sin(Box::new(UvExpr::Mul(
///                 Box::new(UvExpr::V),
///                 Box::new(UvExpr::Constant(20.0)),
///             )))),
///         )),
///         y: Box::new(UvExpr::Constant(0.0)),
///     }),
/// );
///
/// let output = remap_uv_gpu(&ctx, &input, &wave)?;
/// ```
pub fn remap_uv_gpu(ctx: &GpuContext, input: &GpuTexture, expr: &UvExpr) -> GpuResult<GpuTexture> {
    // Convert expression to dew AST, then to WGSL
    let ast = expr.to_dew_ast();

    let mut var_types = HashMap::new();
    var_types.insert("uv".to_string(), Type::Vec2);

    let wgsl_expr = emit_wgsl(&ast, &var_types)
        .map_err(|e| GpuError::ShaderError(format!("WGSL codegen failed: {e:?}")))?;

    // Generate the complete shader
    let shader_source = generate_remap_uv_shader(&wgsl_expr.code);

    // Create output texture (same dimensions as input)
    let output = GpuTexture::new(
        ctx,
        input.width(),
        input.height(),
        TextureFormat::Rgba8Unorm,
    )?;

    // Create shader module
    let shader = ctx
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("remap_uv_shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

    // Create uniforms
    let uniforms = ImageUniforms {
        width: input.width(),
        height: input.height(),
        _padding: [0; 2],
    };

    let uniform_buffer = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("remap_uv_uniforms"),
            contents: bytemuck::cast_slice(&[uniforms]),
            usage: wgpu::BufferUsages::UNIFORM,
        });

    // Create sampler for input texture
    let sampler = ctx.device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("remap_uv_sampler"),
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::FilterMode::Nearest,
        ..Default::default()
    });

    // Create bind group layout
    let bind_group_layout = ctx
        .device
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("remap_uv_bind_group_layout"),
            entries: &[
                // Input texture
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // Sampler
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                // Output texture
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                // Uniforms
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

    let input_view = input.create_view();
    let output_view = output.create_view();

    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("remap_uv_bind_group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&input_view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(&sampler),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(&output_view),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: uniform_buffer.as_entire_binding(),
            },
        ],
    });

    let pipeline_layout = ctx
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("remap_uv_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

    let pipeline = ctx
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("remap_uv_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

    // Dispatch compute shader
    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("remap_uv_encoder"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("remap_uv_pass"),
            timestamp_writes: None,
        });

        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);

        // Dispatch workgroups (8x8 threads per workgroup)
        let workgroups_x = (input.width() + 7) / 8;
        let workgroups_y = (input.height() + 7) / 8;
        pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
    }

    ctx.queue.submit(Some(encoder.finish()));

    Ok(output)
}

/// Generates the WGSL shader source for UV remapping.
fn generate_remap_uv_shader(uv_expr: &str) -> String {
    format!(
        r#"
@group(0) @binding(0) var input_texture: texture_2d<f32>;
@group(0) @binding(1) var input_sampler: sampler;
@group(0) @binding(2) var output_texture: texture_storage_2d<rgba8unorm, write>;

struct Uniforms {{
    width: u32,
    height: u32,
}}
@group(0) @binding(3) var<uniform> uniforms: Uniforms;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
    let x = global_id.x;
    let y = global_id.y;

    if (x >= uniforms.width || y >= uniforms.height) {{
        return;
    }}

    // Normalized UV coordinates [0, 1]
    let uv = vec2<f32>(f32(x) + 0.5, f32(y) + 0.5) / vec2<f32>(f32(uniforms.width), f32(uniforms.height));

    // Apply UV transformation
    let src_uv = {uv_expr};

    // Sample and write
    let color = textureSampleLevel(input_texture, input_sampler, src_uv, 0.0);
    textureStore(output_texture, vec2<i32>(i32(x), i32(y)), color);
}}
"#
    )
}

/// Applies a per-pixel color transform using an expression, executed on the GPU.
///
/// This is the GPU-accelerated version of `resin_image::map_pixels`. The expression
/// is compiled to a WGSL compute shader and executed in parallel across all pixels.
///
/// # Arguments
///
/// * `ctx` - GPU context
/// * `input` - Source texture to transform
/// * `expr` - Color transform expression (Vec4 → Vec4)
///
/// # Example
///
/// ```ignore
/// use unshape_gpu::{GpuContext, map_pixels_gpu};
/// use unshape_image::ColorExpr;
///
/// let ctx = GpuContext::new()?;
/// let input = load_texture(&ctx, "image.png")?;
///
/// // Apply grayscale on GPU
/// let grayscale = ColorExpr::grayscale();
/// let output = map_pixels_gpu(&ctx, &input, &grayscale)?;
/// ```
pub fn map_pixels_gpu(
    ctx: &GpuContext,
    input: &GpuTexture,
    expr: &ColorExpr,
) -> GpuResult<GpuTexture> {
    // Convert expression to dew AST, then to WGSL
    let ast = expr.to_dew_ast();

    let mut var_types = HashMap::new();
    var_types.insert("rgba".to_string(), Type::Vec4);

    let wgsl_expr = emit_wgsl(&ast, &var_types)
        .map_err(|e| GpuError::ShaderError(format!("WGSL codegen failed: {e:?}")))?;

    // Generate the complete shader
    let shader_source = generate_map_pixels_shader(&wgsl_expr.code);

    // Create output texture (same dimensions as input)
    let output = GpuTexture::new(
        ctx,
        input.width(),
        input.height(),
        TextureFormat::Rgba8Unorm,
    )?;

    // Create shader module
    let shader = ctx
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("map_pixels_shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

    // Create uniforms
    let uniforms = ImageUniforms {
        width: input.width(),
        height: input.height(),
        _padding: [0; 2],
    };

    let uniform_buffer = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("map_pixels_uniforms"),
            contents: bytemuck::cast_slice(&[uniforms]),
            usage: wgpu::BufferUsages::UNIFORM,
        });

    // Create bind group layout (simpler than remap_uv - no sampler needed)
    let bind_group_layout = ctx
        .device
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("map_pixels_bind_group_layout"),
            entries: &[
                // Input texture
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // Output texture
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                // Uniforms
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

    let input_view = input.create_view();
    let output_view = output.create_view();

    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("map_pixels_bind_group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&input_view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&output_view),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: uniform_buffer.as_entire_binding(),
            },
        ],
    });

    let pipeline_layout = ctx
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("map_pixels_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

    let pipeline = ctx
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("map_pixels_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

    // Dispatch compute shader
    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("map_pixels_encoder"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("map_pixels_pass"),
            timestamp_writes: None,
        });

        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);

        // Dispatch workgroups (8x8 threads per workgroup)
        let workgroups_x = (input.width() + 7) / 8;
        let workgroups_y = (input.height() + 7) / 8;
        pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
    }

    ctx.queue.submit(Some(encoder.finish()));

    Ok(output)
}

/// Generates the WGSL shader source for per-pixel color transforms.
fn generate_map_pixels_shader(color_expr: &str) -> String {
    format!(
        r#"
@group(0) @binding(0) var input_texture: texture_2d<f32>;
@group(0) @binding(1) var output_texture: texture_storage_2d<rgba8unorm, write>;

struct Uniforms {{
    width: u32,
    height: u32,
}}
@group(0) @binding(2) var<uniform> uniforms: Uniforms;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
    let x = global_id.x;
    let y = global_id.y;

    if (x >= uniforms.width || y >= uniforms.height) {{
        return;
    }}

    // Load input pixel
    let rgba = textureLoad(input_texture, vec2<i32>(i32(x), i32(y)), 0);

    // Apply color transformation
    let result = {color_expr};

    // Write output
    textureStore(output_texture, vec2<i32>(i32(x), i32(y)), result);
}}
"#
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_shader_identity() {
        let shader = generate_remap_uv_shader("uv");
        assert!(shader.contains("let src_uv = uv;"));
        assert!(shader.contains("@compute @workgroup_size(8, 8)"));
    }

    #[test]
    fn test_generate_shader_translation() {
        let shader = generate_remap_uv_shader("(uv + vec2(0.1, 0.0))");
        assert!(shader.contains("let src_uv = (uv + vec2(0.1, 0.0));"));
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_remap_uv_gpu_identity() {
        let ctx = GpuContext::new().unwrap();

        // Create a simple test texture
        let input = GpuTexture::new(&ctx, 64, 64, TextureFormat::Rgba8Unorm).unwrap();

        // Identity transform
        let expr = UvExpr::Uv;
        let _output = remap_uv_gpu(&ctx, &input, &expr).unwrap();
    }

    #[test]
    fn test_generate_map_pixels_shader_identity() {
        let shader = generate_map_pixels_shader("rgba");
        assert!(shader.contains("let result = rgba;"));
        assert!(shader.contains("textureLoad"));
    }

    #[test]
    fn test_generate_map_pixels_shader_grayscale() {
        // Simplified grayscale expression
        let shader = generate_map_pixels_shader("vec4(dot(rgba.rgb, vec3(0.299, 0.587, 0.114)))");
        assert!(shader.contains("let result = vec4(dot(rgba.rgb, vec3(0.299, 0.587, 0.114)));"));
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_map_pixels_gpu_identity() {
        let ctx = GpuContext::new().unwrap();

        // Create a simple test texture
        let input = GpuTexture::new(&ctx, 64, 64, TextureFormat::Rgba8Unorm).unwrap();

        // Identity transform
        let expr = ColorExpr::Rgba;
        let _output = map_pixels_gpu(&ctx, &input, &expr).unwrap();
    }
}
