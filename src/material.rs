use glam::DVec3;

use crate::brdf::{Brdf, BrdfSample};
use crate::spectrum::Spectrum;

pub mod physical;

#[derive(Clone)]
pub struct Material<E, B> {
    pub emission: E,
    pub brdf: B,
}

pub trait MaterialErased: Send + Sync {
    fn emission_sample(&self, lambda: f64) -> f64;
    fn brdf_f(&self, incoming: DVec3, outgoing: DVec3, normal: DVec3, lambda: f64) -> f64;
    fn brdf_sample(&self, outgoing: DVec3, normal: DVec3, lambda: f64, random: DVec3)
        -> BrdfSample;
    fn brdf_pdf(&self, incoming: DVec3, outgoing: DVec3, normal: DVec3, lambda: f64) -> f64;
}

impl<E: Spectrum + Send + Sync, B: Brdf + Send + Sync> MaterialErased for Material<E, B> {
    fn emission_sample(&self, lambda: f64) -> f64 {
        self.emission.sample(lambda)
    }

    fn brdf_f(&self, incoming: DVec3, outgoing: DVec3, normal: DVec3, lambda: f64) -> f64 {
        self.brdf.f(incoming, outgoing, normal, lambda)
    }

    fn brdf_sample(
        &self,
        outgoing: DVec3,
        normal: DVec3,
        lambda: f64,
        random: DVec3,
    ) -> BrdfSample {
        self.brdf.sample(outgoing, normal, lambda, random)
    }

    fn brdf_pdf(&self, incoming: DVec3, outgoing: DVec3, normal: DVec3, lambda: f64) -> f64 {
        self.brdf.pdf(incoming, outgoing, normal, lambda)
    }
}
