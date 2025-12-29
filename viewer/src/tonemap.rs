use std::ops::RangeInclusive;
use std::sync::{Arc, LazyLock};

use egui::{ComboBox, Slider, Ui, Widget};
use glam::{Mat3, Vec3, Vec4};
use winit::event_loop::EventLoopProxy;

pub mod krawczyk_2005;
pub mod none;

use crate::Image;

pub struct TonemapOptions {
    none: none::Options,
    krawczyk_2005: krawczyk_2005::Options,

    selected: usize,
    image: Arc<Image<Vec3>>,
    needs_update: bool,
    proxy: EventLoopProxy<Image<Vec4>>,
}

#[derive(Debug)]
enum Tonemapper {
    None,
    Krawczyk2005,
}

const TONEMAPPERS: &[Tonemapper] = &[Tonemapper::None, Tonemapper::Krawczyk2005];

impl TonemapOptions {
    pub fn new(image: Arc<Image<Vec3>>, proxy: EventLoopProxy<Image<Vec4>>) -> TonemapOptions {
        TonemapOptions {
            none: none::Options::new(&image),
            krawczyk_2005: krawczyk_2005::Options::new(&image),

            selected: 1,
            image,
            needs_update: true,
            proxy,
        }
    }

    pub fn ui(&mut self, ui: &mut Ui) {
        let waiting = self.needs_update;

        self.needs_update |= ComboBox::from_label("Algorithm")
            .show_index(ui, &mut self.selected, TONEMAPPERS.len(), |id| {
                format!("{:?}", TONEMAPPERS[id])
            })
            .changed();

        match TONEMAPPERS[self.selected] {
            Tonemapper::None => self.none.ui(ui, &mut self.needs_update),
            Tonemapper::Krawczyk2005 => self.krawczyk_2005.ui(ui, &mut self.needs_update),
        }

        if !waiting && self.needs_update {
            self.process();
        }
    }

    pub fn updated(&mut self) {
        if self.needs_update {
            self.process();
        }
    }

    pub fn refresh(&mut self) {
        self.process();
    }

    pub fn set_adapting_luminance(&mut self, adapting_luminance: f32) {
        self.krawczyk_2005.set_adapting_luminance(adapting_luminance);
        self.none.set_adapting_luminance(adapting_luminance);
    }

    fn process(&mut self) {
        let proxy = self.proxy.clone();
        let image = self.image.clone();
        match TONEMAPPERS[self.selected] {
            Tonemapper::None => {
                let mapper = self.none.clone();
                std::thread::spawn(move || proxy.send_event(mapper.process(&image)));
            }
            Tonemapper::Krawczyk2005 => {
                let mapper = self.krawczyk_2005.clone();
                std::thread::spawn(move || proxy.send_event(mapper.process(&image)));
            }
        };
        self.needs_update = false;
    }
}

fn xyz_to_srgb_linear(xyz: Vec3) -> Vec3 {
    static XYZ_TO_SRGB_MATRIX: LazyLock<Mat3> = LazyLock::new(|| {
        Mat3::from_cols_array_2d(&[
            [0.4124, 0.3576, 0.1805],
            [0.2126, 0.7152, 0.0722],
            [0.0193, 0.1192, 0.9505],
        ])
        .transpose()
        .inverse()
    });

    *XYZ_TO_SRGB_MATRIX * xyz
}

#[derive(Clone)]
struct DefaultValueSlider {
    value: f32,
    default: f32,
    range: RangeInclusive<f32>,
    log: bool,
}

impl DefaultValueSlider {
    fn new(default: f32, range: RangeInclusive<f32>, log: bool) -> Self {
        DefaultValueSlider {
            value: default,
            default,
            range,
            log,
        }
    }

    fn show(&mut self, ui: &mut Ui, name: &str) -> bool {
        let mut needs_update = false;

        ui.horizontal(|ui| {
            ui.label(name);

            needs_update |= Slider::new(&mut self.value, self.range.clone())
                .logarithmic(self.log)
                .show_value(false)
                .smart_aim(false)
                .ui(ui)
                .changed();

            ui.label(format!("{}", self.value));

            if self.value != self.default {
                if ui.small_button("Reset").clicked() {
                    self.value = self.default;
                    needs_update = true;
                }
            }
        });

        needs_update
    }
}
