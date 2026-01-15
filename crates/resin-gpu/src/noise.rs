//! GPU-accelerated noise generation.

use crate::GpuContext;
use crate::error::GpuResult;
use crate::texture::{GpuTexture, TextureFormat};
use bytemuck::{Pod, Zeroable};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Type of noise to generate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum NoiseType {
    /// Classic Perlin noise.
    Perlin,
    /// Simplex noise (faster, fewer artifacts).
    Simplex,
    /// Value noise (simple interpolated random).
    Value,
    /// Worley/cellular noise.
    Worley,
}

/// Configuration for noise generation.
///
/// See `docs/design/ops-as-values.md`.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dynop", derive(rhizome_resin_op::Op))]
#[cfg_attr(feature = "dynop", op(input = (), output = NoiseConfig))]
pub struct NoiseConfig {
    /// Type of noise.
    pub noise_type: NoiseType,
    /// Frequency/scale of the noise.
    pub scale: f32,
    /// Number of octaves for FBM.
    pub octaves: u32,
    /// Persistence for FBM (amplitude multiplier per octave).
    pub persistence: f32,
    /// Lacunarity for FBM (frequency multiplier per octave).
    pub lacunarity: f32,
    /// Seed for randomization.
    pub seed: u32,
}

impl Default for NoiseConfig {
    fn default() -> Self {
        Self {
            noise_type: NoiseType::Perlin,
            scale: 4.0,
            octaves: 1,
            persistence: 0.5,
            lacunarity: 2.0,
            seed: 0,
        }
    }
}

impl NoiseConfig {
    /// Creates a new noise config with the given type and scale.
    pub fn new(noise_type: NoiseType, scale: f32) -> Self {
        Self {
            noise_type,
            scale,
            ..Default::default()
        }
    }

    /// Applies this configuration, returning a clone of self.
    ///
    /// This is the identity operation for config structs - the config
    /// is the value itself. Useful for serialization pipelines.
    pub fn apply(&self) -> NoiseConfig {
        self.clone()
    }
}

/// Uniform buffer data for the noise shader.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct NoiseUniforms {
    width: u32,
    height: u32,
    scale: f32,
    octaves: u32,
    persistence: f32,
    lacunarity: f32,
    seed: u32,
    noise_type: u32,
}

/// Generates a noise texture on the GPU.
pub fn generate_noise_texture(
    ctx: &GpuContext,
    width: u32,
    height: u32,
    config: &NoiseConfig,
) -> GpuResult<GpuTexture> {
    let texture = GpuTexture::new(ctx, width, height, TextureFormat::R32Float)?;

    let shader = ctx
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("noise_shader"),
            source: wgpu::ShaderSource::Wgsl(NOISE_SHADER.into()),
        });

    let uniforms = NoiseUniforms {
        width,
        height,
        scale: config.scale,
        octaves: config.octaves,
        persistence: config.persistence,
        lacunarity: config.lacunarity,
        seed: config.seed,
        noise_type: match config.noise_type {
            NoiseType::Perlin => 0,
            NoiseType::Simplex => 1,
            NoiseType::Value => 2,
            NoiseType::Worley => 3,
        },
    };

    let uniform_buffer = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("noise_uniforms"),
            contents: bytemuck::cast_slice(&[uniforms]),
            usage: wgpu::BufferUsages::UNIFORM,
        });

    let bind_group_layout = ctx
        .device
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("noise_bind_group_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::R32Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });

    let texture_view = texture.create_view();

    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("noise_bind_group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&texture_view),
            },
        ],
    });

    let pipeline_layout = ctx
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("noise_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

    let pipeline = ctx
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("noise_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("noise_encoder"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("noise_pass"),
            timestamp_writes: None,
        });

        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);

        // Dispatch workgroups (8x8 threads per workgroup)
        let workgroups_x = (width + 7) / 8;
        let workgroups_y = (height + 7) / 8;
        pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
    }

    ctx.queue.submit(Some(encoder.finish()));

    Ok(texture)
}

/// Convenience function to generate noise texture with simple parameters.
pub fn noise_texture(
    ctx: &GpuContext,
    width: u32,
    height: u32,
    noise_type: NoiseType,
    scale: f32,
) -> GpuResult<GpuTexture> {
    generate_noise_texture(ctx, width, height, &NoiseConfig::new(noise_type, scale))
}

// WGSL compute shader for noise generation
const NOISE_SHADER: &str = r#"
struct Uniforms {
    width: u32,
    height: u32,
    scale: f32,
    octaves: u32,
    persistence: f32,
    lacunarity: f32,
    seed: u32,
    noise_type: u32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var output: texture_storage_2d<r32float, write>;

// Hash function for noise
fn hash(p: vec2<f32>) -> f32 {
    var p3 = fract(vec3<f32>(p.x, p.y, p.x) * 0.1031);
    p3 = p3 + dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

fn hash2(p: vec2<f32>) -> vec2<f32> {
    let k = vec2<f32>(0.3183099, 0.3678794);
    var q = p * k + k.yx;
    return fract(16.0 * k * fract(q.x * q.y * (q.x + q.y)));
}

// Gradient for Perlin noise
fn grad(hash: f32, x: f32, y: f32) -> f32 {
    let h = u32(hash * 16.0) & 3u;
    let u = select(y, x, h < 2u);
    let v = select(x, y, h < 2u);
    return select(-u, u, (h & 1u) == 0u) + select(-v, v, (h & 2u) == 0u);
}

// Smoothstep interpolation
fn fade(t: f32) -> f32 {
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
}

// 2D Perlin noise
fn perlin(p: vec2<f32>) -> f32 {
    let pi = floor(p);
    let pf = fract(p);

    let u = fade(pf.x);
    let v = fade(pf.y);

    let n00 = grad(hash(pi), pf.x, pf.y);
    let n10 = grad(hash(pi + vec2<f32>(1.0, 0.0)), pf.x - 1.0, pf.y);
    let n01 = grad(hash(pi + vec2<f32>(0.0, 1.0)), pf.x, pf.y - 1.0);
    let n11 = grad(hash(pi + vec2<f32>(1.0, 1.0)), pf.x - 1.0, pf.y - 1.0);

    let nx0 = mix(n00, n10, u);
    let nx1 = mix(n01, n11, u);

    return mix(nx0, nx1, v) * 0.5 + 0.5;
}

// 2D Simplex noise
fn simplex(p: vec2<f32>) -> f32 {
    let C = vec4<f32>(0.211324865405187, 0.366025403784439, -0.577350269189626, 0.024390243902439);

    var i = floor(p + dot(p, C.yy));
    let x0 = p - i + dot(i, C.xx);

    let i1 = select(vec2<f32>(0.0, 1.0), vec2<f32>(1.0, 0.0), x0.x > x0.y);

    var x12 = x0.xyxy + C.xxzz;
    x12 = vec4<f32>(x12.xy - i1, x12.zw);

    i = i - floor(i / 289.0) * 289.0;
    let p_perm = ((i.y + vec3<f32>(0.0, i1.y, 1.0)) * 34.0 + 1.0) * (i.y + vec3<f32>(0.0, i1.y, 1.0));
    let p_perm2 = ((p_perm + i.x + vec3<f32>(0.0, i1.x, 1.0)) * 34.0 + 1.0) * (p_perm + i.x + vec3<f32>(0.0, i1.x, 1.0));
    let p_final = p_perm2 - floor(p_perm2 / 289.0) * 289.0;

    var m = max(0.5 - vec3<f32>(dot(x0, x0), dot(x12.xy, x12.xy), dot(x12.zw, x12.zw)), vec3<f32>(0.0));
    m = m * m;
    m = m * m;

    let x_vec = 2.0 * fract(p_final * C.www) - 1.0;
    let h = abs(x_vec) - 0.5;
    let ox = floor(x_vec + 0.5);
    let a0 = x_vec - ox;

    m = m * (1.79284291400159 - 0.85373472095314 * (a0 * a0 + h * h));

    let g = vec3<f32>(
        a0.x * x0.x + h.x * x0.y,
        a0.y * x12.x + h.y * x12.y,
        a0.z * x12.z + h.z * x12.w
    );

    return (130.0 * dot(m, g)) * 0.5 + 0.5;
}

// Value noise
fn value(p: vec2<f32>) -> f32 {
    let pi = floor(p);
    let pf = fract(p);

    let u = fade(pf.x);
    let v = fade(pf.y);

    let n00 = hash(pi);
    let n10 = hash(pi + vec2<f32>(1.0, 0.0));
    let n01 = hash(pi + vec2<f32>(0.0, 1.0));
    let n11 = hash(pi + vec2<f32>(1.0, 1.0));

    let nx0 = mix(n00, n10, u);
    let nx1 = mix(n01, n11, u);

    return mix(nx0, nx1, v);
}

// Worley/cellular noise
fn worley(p: vec2<f32>) -> f32 {
    let pi = floor(p);
    let pf = fract(p);

    var min_dist = 1.0;

    for (var y = -1; y <= 1; y++) {
        for (var x = -1; x <= 1; x++) {
            let neighbor = vec2<f32>(f32(x), f32(y));
            let point = hash2(pi + neighbor);
            let diff = neighbor + point - pf;
            let dist = length(diff);
            min_dist = min(min_dist, dist);
        }
    }

    return min_dist;
}

// FBM (fractal Brownian motion)
fn fbm(p: vec2<f32>, noise_type: u32, octaves: u32, persistence: f32, lacunarity: f32) -> f32 {
    var sum = 0.0;
    var amplitude = 1.0;
    var frequency = 1.0;
    var max_value = 0.0;

    for (var i = 0u; i < octaves; i++) {
        let pos = p * frequency;
        var noise_val: f32;

        switch noise_type {
            case 0u: { noise_val = perlin(pos); }
            case 1u: { noise_val = simplex(pos); }
            case 2u: { noise_val = value(pos); }
            case 3u: { noise_val = worley(pos); }
            default: { noise_val = perlin(pos); }
        }

        sum += noise_val * amplitude;
        max_value += amplitude;
        amplitude *= persistence;
        frequency *= lacunarity;
    }

    return sum / max_value;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;

    if (x >= uniforms.width || y >= uniforms.height) {
        return;
    }

    let uv = vec2<f32>(f32(x), f32(y)) / vec2<f32>(f32(uniforms.width), f32(uniforms.height));
    let p = uv * uniforms.scale + vec2<f32>(f32(uniforms.seed) * 0.1, f32(uniforms.seed) * 0.17);

    let value = fbm(p, uniforms.noise_type, uniforms.octaves, uniforms.persistence, uniforms.lacunarity);

    textureStore(output, vec2<i32>(i32(x), i32(y)), vec4<f32>(value, 0.0, 0.0, 1.0));
}
"#;

// Re-export for convenience
use wgpu::util::DeviceExt;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore] // Requires GPU
    fn test_generate_noise_texture() {
        let ctx = GpuContext::new().unwrap();
        let config = NoiseConfig::new(NoiseType::Perlin, 4.0);
        let texture = generate_noise_texture(&ctx, 256, 256, &config).unwrap();

        assert_eq!(texture.width(), 256);
        assert_eq!(texture.height(), 256);

        let data = texture.read_to_f32(&ctx);
        assert_eq!(data.len(), 256 * 256);

        // Values should be in [0, 1] range
        for &v in &data {
            assert!(v >= 0.0 && v <= 1.0, "noise value {} out of range", v);
        }
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_noise_types() {
        let ctx = GpuContext::new().unwrap();

        for noise_type in [
            NoiseType::Perlin,
            NoiseType::Simplex,
            NoiseType::Value,
            NoiseType::Worley,
        ] {
            let config = NoiseConfig::new(noise_type, 4.0);
            let texture = generate_noise_texture(&ctx, 64, 64, &config).unwrap();
            let data = texture.read_to_f32(&ctx);

            // All noise types should produce valid output
            assert_eq!(data.len(), 64 * 64);
        }
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_fbm_octaves() {
        let ctx = GpuContext::new().unwrap();

        let config = NoiseConfig {
            noise_type: NoiseType::Perlin,
            scale: 4.0,
            octaves: 4,
            ..Default::default()
        };

        let texture = generate_noise_texture(&ctx, 128, 128, &config).unwrap();
        let data = texture.read_to_f32(&ctx);

        assert_eq!(data.len(), 128 * 128);
    }
}
