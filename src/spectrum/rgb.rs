use glam::{DVec3, FloatExt};
use rand::{thread_rng, Rng};

use crate::spectrum::{lambda_to_xyz, spectrum_to_xyz, srgb_to_xyz, xyz_to_srgb, VISIBLE};

use super::physical::cie_d65_1nit;
use super::{ConstantSpectrum, Spectrum};

#[derive(Debug)]
pub struct RgbAlbedo {
    pub a: f64,
    pub b: f64,
    pub c: f64,
}

pub struct RgbIlluminant<S> {
    raw: RgbAlbedo,
    brightness: f64,
    whitepoint: S,
}

impl RgbAlbedo {
    pub fn new(srgb: DVec3) -> Self {
        todo!()
    }
}

impl<S> RgbIlluminant<S> {
    pub fn new(srgb: DVec3, brightness: f64, whitepoint: S) -> Self {
        todo!()
    }

    pub fn new_d65(srgb: DVec3, brightness: f64) -> RgbIlluminant<impl Spectrum> {
        RgbIlluminant::new(srgb, brightness, cie_d65_1nit())
    }
}

impl Spectrum for RgbAlbedo {
    fn sample(&self, lambda: f64) -> f64 {
        let l = (lambda - VISIBLE.start) / (VISIBLE.end - VISIBLE.start);
        let q = self.a * l * l + self.b * l + self.c;
        let sigmoid = 0.5 + 0.5 * q / (1.0 + q * q).sqrt();
        sigmoid
    }
}

impl<S: Spectrum> Spectrum for RgbIlluminant<S> {
    fn sample(&self, lambda: f64) -> f64 {
        self.raw.sample(lambda) * self.brightness * self.whitepoint.sample(lambda)
    }
}
