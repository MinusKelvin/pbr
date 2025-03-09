use std::f64::consts::PI;

use glam::{DVec2, DVec3, DVec4};

use crate::spectrum::Spectrum;

use super::{Brdf, BrdfSample};

#[derive(Clone)]
pub struct PhongSpecularBrdf<S> {
    pub albedo: S,
    pub power: f64,
}

impl<S: Spectrum> Brdf for PhongSpecularBrdf<S> {
    fn f(&self, incoming: DVec3, outgoing: DVec3, normal: DVec3, lambdas: DVec4) -> DVec4 {
        if outgoing.dot(normal) > 0.0 || incoming.dot(normal) < 0.0 {
            return DVec4::ZERO;
        }
        let reflect = outgoing.reflect(normal);
        self.albedo.sample_multi(lambdas) * (self.power + 2.0) / (2.0 * PI)
            * incoming.dot(reflect).max(0.0).powf(self.power)
    }

    fn sample(&self, outgoing: DVec3, normal: DVec3, lambdas: DVec4, random: DVec3) -> BrdfSample {
        let reflect = outgoing.reflect(normal);

        let z = random.x.powf(1.0 / (self.power + 1.0));
        let angle = 2.0 * PI * random.y;
        let (y, x) = angle.sin_cos();
        let r = (1.0 - z * z).sqrt();
        let d = DVec2::new(x, y) * r;

        let tangent = reflect.cross(normal).normalize();
        let bitangent = reflect.cross(tangent);
        let incoming = d.x * tangent + d.y * bitangent + z * reflect;

        BrdfSample {
            dir: incoming,
            pdf: self.pdf(incoming, outgoing, normal, lambdas.x),
            f: self.f(incoming, outgoing, normal, lambdas),
            terminate_secondary: false,
            singular: false,
        }
    }

    fn pdf(&self, incoming: DVec3, outgoing: DVec3, normal: DVec3, lambda: f64) -> f64 {
        _ = lambda;
        let reflect = outgoing.reflect(normal);
        (self.power + 1.0) / (2.0 * PI) * incoming.dot(reflect).max(0.0).powf(self.power)
    }
}

#[derive(Clone)]
pub struct PhongRetroBrdf<S> {
    pub power: f64,
    pub albedo: S,
}

impl<S: Spectrum> Brdf for PhongRetroBrdf<S> {
    fn f(&self, incoming: DVec3, outgoing: DVec3, normal: DVec3, lambdas: DVec4) -> DVec4 {
        if outgoing.dot(normal) > 0.0 || incoming.dot(normal) < 0.0 {
            return DVec4::ZERO;
        }
        let retro = -outgoing;
        self.albedo.sample_multi(lambdas) * (self.power + 2.0) / (2.0 * PI)
            * incoming.dot(retro).max(0.0).powf(self.power)
    }

    fn sample(&self, outgoing: DVec3, normal: DVec3, lambdas: DVec4, random: DVec3) -> BrdfSample {
        let retro = -outgoing;

        let z = random.x.powf(1.0 / (self.power + 1.0));
        let angle = 2.0 * PI * random.y;
        let (y, x) = angle.sin_cos();
        let r = (1.0 - z * z).sqrt();
        let d = DVec2::new(x, y) * r;

        let tangent = retro.cross(normal).normalize();
        let bitangent = retro.cross(tangent);
        let incoming = d.x * tangent + d.y * bitangent + z * retro;

        BrdfSample {
            dir: incoming,
            pdf: self.pdf(incoming, outgoing, normal, lambdas.x),
            f: self.f(incoming, outgoing, normal, lambdas),
            terminate_secondary: false,
            singular: false,
        }
    }

    fn pdf(&self, incoming: DVec3, outgoing: DVec3, normal: DVec3, lambda: f64) -> f64 {
        _ = normal;
        _ = lambda;
        let retro = -outgoing;
        (self.power + 1.0) / (2.0 * PI) * incoming.dot(retro).max(0.0).powf(self.power)
    }
}