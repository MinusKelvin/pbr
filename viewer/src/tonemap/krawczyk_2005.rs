use egui::Ui;
use glam::{Vec3, Vec4};

use crate::Image;

use super::DefaultValueSlider;

#[derive(Clone)]
pub struct Options {
    adapting_luminance: DefaultValueSlider,
    key_value: DefaultValueSlider,
    scotopic: bool,
}

impl Options {
    pub fn new(img: &Image<Vec3>) -> Self {
        let mut total_log_lum = 1e-4f32.ln();
        let mut count = 1.0;
        for xyz in &img.data {
            if xyz.y > 1e-4 {
                total_log_lum += xyz.y.ln();
                count += 1.0;
            }
        }
        let avg_luminance = (total_log_lum / count).exp();

        let key_value = 1.03 - 2.0 / (2.0 + (avg_luminance + 1.0).log10());

        Options {
            adapting_luminance: DefaultValueSlider::new(avg_luminance, 1e-4..=1e8, true),
            key_value: DefaultValueSlider::new(key_value, 0.0..=1.0, false),
            scotopic: true,
        }
    }

    pub fn ui(&mut self, ui: &mut Ui, needs_update: &mut bool) {
        *needs_update |= self.adapting_luminance.show(ui, "Adapting Luminance");
        *needs_update |= self.key_value.show(ui, "Key Value");
        *needs_update |= ui
            .checkbox(&mut self.scotopic, "Scotopic Simulation")
            .changed();
    }

    pub fn process(self, image: &Image<Vec3>) -> Image<Vec4> {
        Image::new(image.width, image.height, |x, y| {
            let xyz = image[(x, y)];
            let y = xyz.y.max(1e-6);

            let y_r = self.key_value.value * y / self.adapting_luminance.value;
            let l = y_r / (1.0 + y_r);

            let scotopic = match self.scotopic {
                true => 0.04 / (0.04 + y),
                false => 0.0,
            };

            let rgb = super::xyz_to_srgb_linear(xyz);
            let rgb_l =
                (l / y) * (1.0 - scotopic) * rgb + scotopic * l * Vec3::new(1.05, 0.97, 1.27);

            rgb_l.extend(1.0)
        })
    }

    pub fn set_adapting_luminance(&mut self, adapting_luminance: f32) {
        self.adapting_luminance.value = adapting_luminance;
        self.key_value.value = 1.03 - 2.0 / (2.0 + (adapting_luminance + 1.0).log10());
    }
}
