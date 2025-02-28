use std::sync::Arc;

use egui::{Ui, ViewportId};
use egui_wgpu::{Renderer, ScreenDescriptor};
use egui_winit::{EventResponse, State};
use winit::dpi::PhysicalSize;
use winit::event::WindowEvent;
use winit::window::Window;

pub struct EguiSetup {
    window: Arc<Window>,
    state: State,
    renderer: Renderer,

    output_config: wgpu::TextureDescriptor<'static>,
    output: wgpu::Texture,
    pipeline_layout: wgpu::PipelineLayout,
    composite_shader: wgpu::ShaderModule,
    pipeline: wgpu::RenderPipeline,
    bg_layout: wgpu::BindGroupLayout,
    bg: wgpu::BindGroup,
}

impl EguiSetup {
    pub fn new(
        device: &wgpu::Device,
        window: Arc<Window>,
        size: PhysicalSize<u32>,
        target_format: wgpu::TextureFormat,
    ) -> Self {
        let output_config = wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d {
                width: size.width,
                height: size.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        };
        let output = device.create_texture(&output_config);

        let state = State::new(
            Default::default(),
            ViewportId::ROOT,
            &window,
            None,
            None,
            None,
        );

        let renderer = Renderer::new(device, wgpu::TextureFormat::Rgba8UnormSrgb, None, 1, false);

        let composite_shader =
            device.create_shader_module(wgpu::include_wgsl!("egui_composite.wgsl"));

        let bg_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            }],
        });

        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bg_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(
                    &output.create_view(&Default::default()),
                ),
            }],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&bg_layout],
            push_constant_ranges: &[],
        });

        let pipeline = Self::create_composite_pipeline(
            device,
            &pipeline_layout,
            &composite_shader,
            target_format,
        );

        EguiSetup {
            output_config,
            output,
            window,
            state,
            renderer,
            pipeline_layout,
            composite_shader,
            pipeline,
            bg_layout,
            bg,
        }
    }

    fn create_composite_pipeline(
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
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent::OVER,
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview: None,
            cache: None,
        })
    }

    pub fn resize(&mut self, device: &wgpu::Device, new_size: PhysicalSize<u32>) {
        self.output_config.size.width = new_size.width;
        self.output_config.size.height = new_size.height;
        self.output = device.create_texture(&self.output_config);

        self.bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.bg_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(
                    &self.output.create_view(&Default::default()),
                ),
            }],
        });
    }

    pub fn target_format_changed(
        &mut self,
        device: &wgpu::Device,
        target_format: wgpu::TextureFormat,
    ) {
        self.pipeline = Self::create_composite_pipeline(
            device,
            &self.pipeline_layout,
            &self.composite_shader,
            target_format,
        );
    }

    pub fn event(&mut self, event: &WindowEvent) -> EventResponse {
        self.state.on_window_event(&self.window, event)
    }

    pub fn render(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        mut f: impl FnMut(&mut Ui),
    ) {
        let input = self.state.take_egui_input(&self.window);

        let output = self.state.egui_ctx().run(input, |ctx| {
            egui::Window::new("Viewer")
                .default_width(400.0)
                .resizable(false)
                .show(ctx, &mut f);
        });

        self.state
            .handle_platform_output(&self.window, output.platform_output);

        let clipped_primitives = self
            .state
            .egui_ctx()
            .tessellate(output.shapes, output.pixels_per_point);

        for free in output.textures_delta.free {
            self.renderer.free_texture(&free);
        }

        for (id, delta) in output.textures_delta.set {
            self.renderer.update_texture(device, queue, id, &delta);
        }

        let screen_descriptor = ScreenDescriptor {
            size_in_pixels: self.window.inner_size().into(),
            pixels_per_point: output.pixels_per_point,
        };

        let buffers = self.renderer.update_buffers(
            device,
            queue,
            encoder,
            &clipped_primitives,
            &screen_descriptor,
        );

        assert!(buffers.is_empty());

        let output = self.output.create_view(&Default::default());

        self.renderer.render(
            &mut encoder
                .begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: None,
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &output,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                })
                .forget_lifetime(),
            &clipped_primitives,
            &screen_descriptor,
        );
    }

    pub fn composite(&mut self, rp: &mut wgpu::RenderPass) {
        rp.set_pipeline(&self.pipeline);
        rp.set_bind_group(0, &self.bg, &[]);
        rp.draw(0..4, 0..1);
    }
}
