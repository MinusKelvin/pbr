mod egui_setup;
mod tonemap;
mod viewer;

use std::error::Error;
use std::ops::{Index, IndexMut};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use egui::{Slider, Ui, Widget};
use egui_setup::EguiSetup;
use glam::{Vec2, Vec3, Vec4};
use tonemap::TonemapOptions;
use viewer::Viewer;
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalSize;
use winit::event::{MouseButton, WindowEvent};
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

    tonemappers: Vec<TonemapOptions>,
    images: Vec<(String, Arc<Image<Vec3>>, PathBuf)>,
    selected: usize,
    scale: f32,

    mpos: Option<(usize, usize)>,

    window: Arc<Window>,

    panning: bool,
    last_pos: Vec2,
}

type InitArgs = (
    EventLoopProxy<Image<Vec4>>,
    Vec<(String, Arc<Image<Vec3>>, PathBuf)>,
);

impl App {
    async fn new(el: &ActiveEventLoop, (proxy, images): InitArgs) -> Self {
        let tonemappers = images
            .iter()
            .map(|(_, img, _)| TonemapOptions::new(img.clone(), proxy.clone()))
            .collect();

        let window = el
            .create_window(
                WindowAttributes::default()
                    .with_title(format!("MinusKelvin PBR Viewer - {}", images[0].0))
                    .with_inner_size(PhysicalSize::new(
                        images[0].1.width as u32 * 3,
                        images[0].1.height as u32 * 3,
                    ))
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
            images[0].1.width as u32,
            images[0].1.height as u32,
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

            tonemappers,
            images,
            selected: 0,
            scale: 3.0,
            mpos: None,

            window,

            panning: false,
            last_pos: Vec2::ZERO,
        }
    }

    fn user_event(&mut self, updated: Image<Vec4>) {
        self.viewer.update_image(&self.queue, updated);
        self.tonemappers[self.selected].updated();
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
                let p = Vec2::new(position.x as f32, position.y as f32);
                if self.panning {
                    let d = p - self.last_pos;
                    self.viewer.yaw -= d.x * 0.004 / self.scale;
                    self.viewer.pitch -= d.y * 0.004 / self.scale;
                }
                self.last_pos = p;

                let x = position.x / self.config.width as f64
                    * self.images[self.selected].1.width as f64;
                let y = position.y / self.config.height as f64
                    * self.images[self.selected].1.height as f64;
                let in_bounds = x >= 0.0
                    && x < self.images[self.selected].1.width as f64
                    && y >= 0.0
                    && y < self.images[self.selected].1.height as f64;
                self.mpos = in_bounds.then(|| (x as usize, y as usize));
            }
            WindowEvent::MouseInput {
                state,
                button: MouseButton::Left,
                ..
            } => self.panning = state.is_pressed(),
            WindowEvent::RedrawRequested => match self.surface.get_current_texture() {
                Ok(frame) => {
                    let output = frame.texture.create_view(&Default::default());

                    let mut encoder = self.device.create_command_encoder(&Default::default());

                    let mut recreate_surface = false;
                    let mut resize = false;

                    let render = |ui: &mut Ui| {
                        match self.mpos {
                            Some(p) => {
                                ui.label(format!("XYZ: {:.4?}", self.images[self.selected].1[p]))
                            }
                            None => ui.label("XYZ: -"),
                        };

                        if self.images.len() > 1 {
                            let changed =
                                Slider::new(&mut self.selected, 0..=self.images.len() - 1)
                                    .drag_value_speed(0.1)
                                    .ui(ui)
                                    .changed();
                            if changed {
                                resize |= true;
                                self.tonemappers[self.selected].refresh();
                                self.window.set_title(&self.images[self.selected].0);
                            }
                        }

                        resize |= Slider::new(&mut self.scale, 0.5..=8.0)
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

                        self.tonemappers[self.selected].ui(ui);
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

                        self.viewer.render(&self.queue, &mut rp);

                        self.egui.composite(&mut rp);
                    }

                    self.queue.submit([encoder.finish()]);

                    self.window.pre_present_notify();
                    frame.present();

                    if resize {
                        let width =
                            (self.scale * self.images[self.selected].1.width as f32).round() as u32;
                        let height = (self.scale * self.images[self.selected].1.height as f32)
                            .round() as u32;
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
        let mut iter = std::env::args_os().skip(1).peekable();
        let mode = iter
            .next_if(|s| s.to_str().is_some_and(|s| s.starts_with("--")))
            .and_then(|s| s.to_str().map(|s| s.to_string()));

        let mut images = vec![];
        for path in iter {
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

            images.push((
                path.file_stem().unwrap().to_string_lossy().into_owned(),
                Arc::new(image),
                path.to_owned(),
            ));
        }
        if images.is_empty() {
            eprintln!("At least one image must be provided");
            std::process::exit(1);
        }
        Ok((mode, images))
    })();

    let (mode, images) = result.unwrap();

    if let Some(mapper) = mode {
        for (_, image, path) in images {
            let result = match &*mapper {
                "--krawczyk2005" | "--tonemap" => {
                    tonemap::krawczyk_2005::Options::new(&image).process(&image)
                }
                "--none" => tonemap::none::Options::new(&image).process(&image),
                _ => {
                    eprintln!("unrecognized tonemapper: {mapper}");
                    std::process::exit(1);
                }
            };

            let img =
                image::RgbaImage::from_fn(result.width as u32, result.height as u32, |x, y| {
                    image::Rgba(
                        result[(x as usize, y as usize)]
                            .map(|v| (egui::ecolor::gamma_from_linear(v) * 255.0).round())
                            .as_u8vec4()
                            .to_array(),
                    )
                });
            img.save(path.with_extension("png")).unwrap();
        }
        std::process::exit(0);
    }

    let el = EventLoop::with_user_event().build().unwrap();
    let proxy = el.create_proxy();

    el.run_app(&mut LateinitApp {
        app: None,
        args: Some((proxy, images)),
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
