use glam::{DVec3, DVec4};

use crate::brdf::{Brdf, BrdfSample};
use crate::medium::Medium;
use crate::spectrum::Spectrum;

pub mod physical;

#[derive(Clone)]
pub struct Material<E, B, Mi, Mo> {
    pub emission: E,
    pub brdf: B,
    pub enter_medium: Mi,
    pub exit_medium: Mo,
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

    fn enter_medium(&self) -> &dyn Medium;
    fn exit_medium(&self) -> &dyn Medium;

    fn name(&self) -> &'static str;
}

impl<E: Spectrum, B: Brdf, Mi: Medium, Mo: Medium> MaterialErased for Material<E, B, Mi, Mo> {
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

    fn enter_medium(&self) -> &dyn Medium {
        &self.enter_medium
    }

    fn exit_medium(&self) -> &dyn Medium {
        &self.exit_medium
    }

    fn name(&self) -> &'static str {
        self.brdf.name()
    }
}
