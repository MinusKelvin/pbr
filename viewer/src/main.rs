mod egui_setup;
mod tonemap;
mod viewer;

use std::error::Error;
use std::ops::{Index, IndexMut};
use std::path::Path;
use std::sync::Arc;

use egui::{Slider, Ui, Widget};
use egui_setup::EguiSetup;
use glam::{Vec3, Vec4};
use tonemap::TonemapOptions;
use viewer::Viewer;
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalSize;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop, EventLoopProxy};
use winit::window::{Window, WindowAttributes, WindowId};

struct App {
    queue: wgpu::Queue,
    device: wgpu::Device,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,

    hdr_format: Option<wgpu::TextureFormat>,
    sdr_format: wgpu::TextureFormat,

    egui: EguiSetup,
    viewer: Viewer,

    tonemapper: TonemapOptions,
    image: Arc<Image<Vec3>>,
    scale: f32,

    mpos: Option<(usize, usize)>,

    window: Arc<Window>,
}

type InitArgs = (EventLoopProxy<Image<Vec4>>, String, Image<Vec3>);

impl App {
    async fn new(el: &ActiveEventLoop, (proxy, filename, image): InitArgs) -> Self {
        let image = Arc::new(image);
        let tonemapper = TonemapOptions::new(image.clone(), proxy);

        let window = el
            .create_window(
                WindowAttributes::default()
                    .with_title(format!("MinusKelvin PBR Viewer - {filename}"))
                    .with_inner_size(PhysicalSize::new(image.width as u32, image.height as u32))
                    .with_resizable(false),
            )
            .unwrap();
        let window = Arc::new(window);

        let instance = wgpu::Instance::new(&Default::default());

        let surface = instance.create_surface(window.clone()).unwrap();

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                compatible_surface: Some(&surface),
                ..Default::default()
            })
            .await
            .unwrap();

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    required_features: wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES
                        | wgpu::Features::FLOAT32_FILTERABLE,
                    required_limits: wgpu::Limits::default().using_resolution(adapter.limits()),
                    ..Default::default()
                },
                None,
            )
            .await
            .unwrap();

        let formats = surface.get_capabilities(&adapter).formats;
        dbg!(&formats);

        let hdr_format = formats
            .iter()
            .copied()
            .find(|f| matches!(f, wgpu::TextureFormat::Rgba16Float));
        dbg!(hdr_format);

        let sdr_format = formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or_else(|| panic!("no sdr format available?"));
        dbg!(sdr_format);

        let size = window.inner_size();
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: sdr_format,
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::AutoVsync,
            desired_maximum_frame_latency: 3,
            alpha_mode: wgpu::CompositeAlphaMode::Opaque,
            view_formats: vec![],
        };

        surface.configure(&device, &config);

        let viewer = Viewer::new(
            &device,
            config.format,
            image.width as u32,
            image.height as u32,
        );
        let egui = EguiSetup::new(&device, window.clone(), size, config.format);

        App {
            queue,
            device,
            surface,
            config,

            hdr_format,
            sdr_format,

            egui,
            viewer,

            tonemapper,
            image,
            scale: 1.0,
            mpos: None,

            window,
        }
    }

    fn user_event(&mut self, updated: Image<Vec4>) {
        self.viewer.update_image(&self.queue, updated);
        self.tonemapper.updated();
    }

    fn event(&mut self, event: WindowEvent, el: &ActiveEventLoop) {
        let response = self.egui.event(&event);
        if response.consumed {
            return;
        }

        match event {
            WindowEvent::CloseRequested => el.exit(),
            WindowEvent::Resized(new_size) => {
                if new_size.width != 0 && new_size.height != 0 {
                    self.config.width = new_size.width;
                    self.config.height = new_size.height;
                    self.surface.configure(&self.device, &self.config);
                    self.egui.resize(&self.device, new_size);
                }
            }
            WindowEvent::ScaleFactorChanged {
                mut inner_size_writer,
                ..
            } => {
                let _ = inner_size_writer
                    .request_inner_size(PhysicalSize::new(self.config.width, self.config.height));
            }
            WindowEvent::CursorMoved { position, .. } => {
                let x = position.x / self.config.width as f64 * self.image.width as f64;
                let y = position.y / self.config.height as f64 * self.image.height as f64;
                let in_bounds = x >= 0.0
                    && x < self.image.width as f64
                    && y >= 0.0
                    && y < self.image.height as f64;
                self.mpos = in_bounds.then(|| (x as usize, y as usize));
            }
            WindowEvent::RedrawRequested => match self.surface.get_current_texture() {
                Ok(frame) => {
                    let output = frame.texture.create_view(&Default::default());

                    let mut encoder = self.device.create_command_encoder(&Default::default());

                    let mut recreate_surface = false;
                    let mut resize = false;

                    let render = |ui: &mut Ui| {
                        match self.mpos {
                            Some(p) => ui.label(format!("XYZ: {:.4?}", self.image[p])),
                            None => ui.label("XYZ: -"),
                        };

                        resize = Slider::new(&mut self.scale, 0.5..=4.0)
                            .step_by(0.5)
                            .ui(ui)
                            .changed();

                        if let Some(hdr_format) = self.hdr_format {
                            let mut hdr = self.config.format == hdr_format;
                            if ui.checkbox(&mut hdr, "HDR").changed() {
                                if hdr {
                                    self.config.format = hdr_format;
                                } else {
                                    self.config.format = self.sdr_format;
                                }
                                recreate_surface = true;
                            }
                        }

                        ui.separator();

                        self.tonemapper.ui(ui);
                    };

                    self.egui
                        .render(&self.device, &self.queue, &mut encoder, render);

                    {
                        let mut rp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                            label: None,
                            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                view: &output,
                                resolve_target: None,
                                ops: wgpu::Operations {
                                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                                    store: wgpu::StoreOp::Store,
                                },
                            })],
                            depth_stencil_attachment: None,
                            timestamp_writes: None,
                            occlusion_query_set: None,
                        });

                        self.viewer.render(&mut rp);

                        self.egui.composite(&mut rp);
                    }

                    self.queue.submit([encoder.finish()]);

                    self.window.pre_present_notify();
                    frame.present();

                    if resize {
                        let width = (self.scale * self.image.width as f32).round() as u32;
                        let height = (self.scale * self.image.height as f32).round() as u32;
                        if let Some(size) = self
                            .window
                            .request_inner_size(PhysicalSize::new(width, height))
                        {
                            self.config.width = size.width;
                            self.config.height = size.height;
                            self.egui.resize(&self.device, size);
                            recreate_surface = true;
                        }
                    }

                    if recreate_surface {
                        self.surface.configure(&self.device, &self.config);
                        self.viewer
                            .target_format_changed(&self.device, self.config.format);
                        self.egui
                            .target_format_changed(&self.device, self.config.format);
                    }

                    self.window.request_redraw();
                }
                Err(e) => eprintln!("Surface Error: {e}"),
            },
            _ => {}
        }
    }
}

struct Image<T> {
    data: Box<[T]>,
    width: usize,
    height: usize,
}

impl<T> Image<T> {
    fn new(width: usize, height: usize, init: impl Fn(usize, usize) -> T) -> Self {
        let init = &init;
        let data = (0..height)
            .flat_map(|y| (0..width).map(move |x| init(x, y)))
            .collect();
        Image {
            data,
            width,
            height,
        }
    }

    fn from_pixel(width: usize, height: usize, pixel: T) -> Self
    where
        T: Clone,
    {
        Self::new(width, height, |_, _| pixel.clone())
    }
}

impl<T> Index<(usize, usize)> for Image<T> {
    type Output = T;

    fn index(&self, (x, y): (usize, usize)) -> &Self::Output {
        assert!(x < self.width);
        assert!(y < self.height);
        &self.data[x + y * self.width]
    }
}

impl<T> IndexMut<(usize, usize)> for Image<T> {
    fn index_mut(&mut self, (x, y): (usize, usize)) -> &mut Self::Output {
        assert!(x < self.width);
        assert!(y < self.height);
        &mut self.data[x + y * self.width]
    }
}

fn main() {
    let result = (|| -> Result<_, Box<dyn Error>> {
        let path = std::env::args_os()
            .nth(1)
            .ok_or("Must specify raw image file as argument")?;
        let path = Path::new(&path);

        use exr::prelude::*;
        let image = read()
            .no_deep_data()
            .largest_resolution_level()
            .rgb_channels(
                |size, _| crate::Image::from_pixel(size.0, size.1, Vec3::ZERO),
                |pixels, xy, p| pixels[(xy.0, xy.1)] = Vec3::from(p),
            )
            .first_valid_layer()
            .all_attributes()
            .from_file(path)?;
        let image = image.layer_data.channel_data.pixels;

        Ok((
            path.file_stem().unwrap().to_string_lossy().into_owned(),
            image,
        ))
    })();

    let (filename, image) = result.unwrap_or_else(|e| {
        eprintln!("Could not open image: {e}");
        std::process::exit(1)
    });

    let el = EventLoop::with_user_event().build().unwrap();
    let proxy = el.create_proxy();

    el.run_app(&mut LateinitApp {
        app: None,
        args: Some((proxy, filename, image)),
    })
    .unwrap();
}

struct LateinitApp {
    app: Option<App>,
    args: Option<InitArgs>,
}

impl ApplicationHandler<Image<Vec4>> for LateinitApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        self.app = self
            .args
            .take()
            .map(|args| pollster::block_on(App::new(event_loop, args)));
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _: WindowId, event: WindowEvent) {
        if let Some(app) = self.app.as_mut() {
            app.event(event, event_loop);
        }
    }

    fn user_event(&mut self, _: &ActiveEventLoop, event: Image<Vec4>) {
        if let Some(app) = self.app.as_mut() {
            app.user_event(event);
        }
    }
}
