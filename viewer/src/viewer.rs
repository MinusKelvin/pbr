use bytemuck::{Pod, Zeroable};
use glam::{EulerRot, Mat3, Vec3, Vec4};

use crate::Image;

pub struct Viewer {
    layout: wgpu::PipelineLayout,
    shader: wgpu::ShaderModule,
    pipeline: wgpu::RenderPipeline,
    bg: wgpu::BindGroup,
    uniform_buffer: wgpu::Buffer,

    image: wgpu::Texture,

    pub yaw: f32,
    pub pitch: f32,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct Uniforms {
    rot: [Vec4; 3],
}

impl Viewer {
    pub fn new(
        device: &wgpu::Device,
        target_format: wgpu::TextureFormat,
        width: u32,
        height: u32,
    ) -> Self {
        let spherical = width == height;

        let shader = match spherical {
            true => device.create_shader_module(wgpu::include_wgsl!("viewer_sphere.wgsl")),
            false => device.create_shader_module(wgpu::include_wgsl!("viewer_blit.wgsl")),
        };

        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("view uniform"),
            size: std::mem::size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: None,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            lod_min_clamp: 0.0,
            lod_max_clamp: 0.0,
            compare: None,
            anisotropy_clamp: 1,
            border_color: None,
        });

        let bg_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&bg_layout],
            push_constant_ranges: &[],
        });

        let pipeline = Self::create_pipeline(device, &layout, &shader, target_format);

        let image = device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        let image_view = image.create_view(&Default::default());

        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bg_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&image_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: uniform_buffer.as_entire_binding(),
                },
            ],
        });

        Viewer {
            layout,
            shader,
            pipeline,
            bg,
            uniform_buffer,
            image,
            yaw: 0.0,
            pitch: 0.0,
        }
    }

    fn create_pipeline(
        device: &wgpu::Device,
        layout: &wgpu::PipelineLayout,
        shader: &wgpu::ShaderModule,
        target_format: wgpu::TextureFormat,
    ) -> wgpu::RenderPipeline {
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: None,
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: None,
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: None,
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview: None,
            cache: None,
        })
    }

    pub fn target_format_changed(
        &mut self,
        device: &wgpu::Device,
        target_format: wgpu::TextureFormat,
    ) {
        self.pipeline = Self::create_pipeline(device, &self.layout, &self.shader, target_format);
    }

    pub fn render(&mut self, queue: &wgpu::Queue, rp: &mut wgpu::RenderPass) {
        let matrix = Mat3::from_euler(EulerRot::YXZ, self.yaw, self.pitch, 0.0);
        queue.write_buffer(
            &self.uniform_buffer,
            0,
            bytemuck::bytes_of(&Uniforms {
                rot: matrix
                    .to_cols_array_2d()
                    .map(|c| Vec3::from_array(c).extend(0.0)),
            }),
        );

        rp.set_pipeline(&self.pipeline);
        rp.set_bind_group(0, &self.bg, &[]);
        rp.draw(0..4, 0..1);
    }

    pub fn update_image(&mut self, queue: &wgpu::Queue, updated: Image<Vec4>) {
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.image,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(&updated.data),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some((updated.width * std::mem::size_of::<Vec4>()) as u32),
                rows_per_image: Some(updated.height as u32),
            },
            wgpu::Extent3d {
                width: updated.width as u32,
                height: updated.height as u32,
                depth_or_array_layers: 1,
            },
        );
    }
}
