use std::f64::consts::PI;

use glam::{DVec3, DVec4, FloatExt};

use crate::spectrum::{AmplifiedSpectrum, Spectrum};

pub struct LightSample {
    pub dir: DVec3,
    pub dist: f64,
    pub pdf: f64,
    pub emission: DVec4,
}

pub trait Light: Send + Sync {
    fn emission(&self, pos: DVec3, dir: DVec3, lambdas: DVec4, max_t: f64) -> DVec4;

    fn sample(&self, pos: DVec3, lambdas: DVec4, random: DVec3) -> LightSample;

    fn pdf(&self, pos: DVec3, dir: DVec3, lambdas: DVec4) -> f64;
}

pub struct DistantDiskLight<S> {
    pub emission: S,
    pub dir: DVec3,
    pub cos_radius: f64,
}

impl<S> DistantDiskLight<S> {
    pub fn from_irradiance(
        dir: DVec3,
        cos_radius: f64,
        irradiance: S,
    ) -> DistantDiskLight<AmplifiedSpectrum<S>> {
        let size_steradians = 2.0 * PI * (1.0 - cos_radius);
        let emission = AmplifiedSpectrum {
            factor: 1.0 / size_steradians,
            s: irradiance,
        };
        DistantDiskLight {
            emission,
            dir,
            cos_radius,
        }
    }
}

impl<S: Spectrum + Send + Sync> Light for DistantDiskLight<S> {
    fn emission(&self, pos: DVec3, dir: DVec3, lambdas: DVec4, max_t: f64) -> DVec4 {
        _ = pos;
        if max_t == f64::INFINITY && dir.dot(self.dir) >= self.cos_radius {
            self.emission.sample_multi(lambdas)
        } else {
            DVec4::ZERO
        }
    }

    fn sample(&self, pos: DVec3, lambdas: DVec4, random: DVec3) -> LightSample {
        let z = self.cos_radius.lerp(1.0, random.x);
        let (x, y) = (random.y * PI * 2.0).sin_cos();
        let r = (1.0 - z * z).sqrt();

        let (tangent, bitangent) = self.dir.any_orthonormal_pair();
        let dir = x * r * tangent + y * r * bitangent + z * self.dir;

        LightSample {
            dir,
            dist: f64::INFINITY,
            pdf: self.pdf(pos, dir, lambdas),
            emission: self.emission(pos, dir, lambdas, f64::INFINITY),
        }
    }

    fn pdf(&self, pos: DVec3, dir: DVec3, lambdas: DVec4) -> f64 {
        _ = pos;
        _ = lambdas;
        if dir.dot(self.dir) >= self.cos_radius {
            1.0 / ((1.0 - self.cos_radius) * 2.0 * PI)
        } else {
            0.0
        }
    }
}
