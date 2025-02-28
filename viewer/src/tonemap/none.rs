use egui::{Slider, Ui, Widget};
use glam::{Vec3, Vec4};

use crate::Image;

use super::DefaultValueSlider;

#[derive(Clone)]
pub struct Options {
    avg_luminance: DefaultValueSlider,
}

impl Options {
    pub fn new(img: &Image<Vec3>) -> Self {
        let total_log_lum = img.data.iter().map(|v| v.y.max(1e-4).ln()).sum::<f32>();
        let avg_luminance = (total_log_lum / img.data.len() as f32).exp();

        Options {
            avg_luminance: DefaultValueSlider::new(avg_luminance, 1e-4..=1e8, true),
        }
    }

    pub fn ui(&mut self, ui: &mut Ui, needs_update: &mut bool) {
        *needs_update |= self.avg_luminance.show(ui, "Adapting Luminance");
    }

    pub fn process(self, image: &Image<Vec3>) -> Image<Vec4> {
        Image::new(image.width, image.height, |x, y| {
            let xyz = image[(x, y)] / self.avg_luminance.value;

            super::xyz_to_srgb_linear(xyz).extend(1.0)
        })
    }
}
