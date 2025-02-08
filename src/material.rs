use glam::{DVec3, DVec4};

use crate::brdf::{Brdf, BrdfSample};
use crate::spectrum::Spectrum;

pub mod physical;

#[derive(Clone)]
pub struct Material<E, B> {
    pub emission: E,
    pub brdf: B,
}

pub trait MaterialErased: Send + Sync {
    fn emission_sample(&self, lambdas: DVec4) -> DVec4;
    fn brdf_f(&self, incoming: DVec3, outgoing: DVec3, normal: DVec3, lambdas: DVec4) -> DVec4;
    fn brdf_sample(
        &self,
        outgoing: DVec3,
        normal: DVec3,
        lambdas: DVec4,
        random: DVec3,
    ) -> BrdfSample;
    fn brdf_pdf(&self, incoming: DVec3, outgoing: DVec3, normal: DVec3, lambda: f64) -> f64;

    fn name(&self) -> &'static str;
}

impl<E: Spectrum + Send + Sync, B: Brdf + Send + Sync> MaterialErased for Material<E, B> {
    fn emission_sample(&self, lambdas: DVec4) -> DVec4 {
        self.emission.sample_multi(lambdas)
    }

    fn brdf_f(&self, incoming: DVec3, outgoing: DVec3, normal: DVec3, lambdas: DVec4) -> DVec4 {
        self.brdf.f(incoming, outgoing, normal, lambdas)
    }

    fn brdf_sample(
        &self,
        outgoing: DVec3,
        normal: DVec3,
        lambdas: DVec4,
        random: DVec3,
    ) -> BrdfSample {
        self.brdf.sample(outgoing, normal, lambdas, random)
    }

    fn brdf_pdf(&self, incoming: DVec3, outgoing: DVec3, normal: DVec3, lambda: f64) -> f64 {
        self.brdf.pdf(incoming, outgoing, normal, lambda)
    }

    fn name(&self) -> &'static str {
        self.brdf.name()
    }
}
