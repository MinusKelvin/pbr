use std::f64::consts::PI;

use glam::{DVec3, DVec4};

use crate::spectrum::Spectrum;

pub trait Medium: Send + Sync {
    fn majorant(&self, lambdas: DVec4) -> f64;

    fn absorption(&self, pos: DVec3, outgoing: DVec3, lambdas: DVec4) -> DVec4;

    fn emission(&self, pos: DVec3, outgoing: DVec3, lambdas: DVec4) -> DVec4;

    fn scattering(&self, pos: DVec3, outgoing: DVec3, lambdas: DVec4) -> DVec4;

    fn attenuation(&self, pos: DVec3, outgoing: DVec3, lambdas: DVec4) -> DVec4 {
        self.absorption(pos, outgoing, lambdas) + self.scattering(pos, outgoing, lambdas)
    }

    fn single_scattering_albedo(&self, pos: DVec3, outgoing: DVec3, lambdas: DVec4) -> DVec4 {
        self.scattering(pos, outgoing, lambdas) / self.attenuation(pos, outgoing, lambdas)
    }

    fn null_scattering(&self, pos: DVec3, outgoing: DVec3, lambdas: DVec4) -> DVec4 {
        self.majorant(lambdas) - self.attenuation(pos, outgoing, lambdas)
    }

    fn phase(&self, pos: DVec3, outgoing: DVec3, incoming: DVec3, lambdas: DVec4) -> DVec4;

    fn participating(&self) -> bool {
        true
    }
}

#[derive(Copy, Clone)]
pub struct Vacuum;

impl Medium for Vacuum {
    fn majorant(&self, lambdas: DVec4) -> f64 {
        _ = lambdas;
        0.0
    }

    fn absorption(&self, pos: DVec3, outgoing: DVec3, lambdas: DVec4) -> DVec4 {
        _ = (pos, outgoing, lambdas);
        DVec4::ZERO
    }

    fn emission(&self, pos: DVec3, outgoing: DVec3, lambdas: DVec4) -> DVec4 {
        _ = (pos, outgoing, lambdas);
        DVec4::ZERO
    }

    fn scattering(&self, pos: DVec3, outgoing: DVec3, lambdas: DVec4) -> DVec4 {
        _ = (pos, outgoing, lambdas);
        DVec4::ZERO
    }

    fn phase(&self, pos: DVec3, outgoing: DVec3, incoming: DVec3, lambdas: DVec4) -> DVec4 {
        _ = (pos, outgoing, incoming, lambdas);
        DVec4::ZERO
    }

    fn participating(&self) -> bool {
        false
    }
}

pub struct SimpleUniformMedium<Sa, Se, Ss> {
    pub absorption: Sa,
    pub emission: Se,
    pub scattering: Ss,
}

impl<Sa: Spectrum, Se: Spectrum, Ss: Spectrum> Medium for SimpleUniformMedium<Sa, Se, Ss> {
    fn majorant(&self, lambdas: DVec4) -> f64 {
        (self.absorption.sample_multi(lambdas) + self.scattering.sample_multi(lambdas))
            .max_element()
    }

    fn absorption(&self, pos: DVec3, outgoing: DVec3, lambdas: DVec4) -> DVec4 {
        _ = (pos, outgoing);
        self.absorption.sample_multi(lambdas)
    }

    fn emission(&self, pos: DVec3, outgoing: DVec3, lambdas: DVec4) -> DVec4 {
        _ = (pos, outgoing);
        self.emission.sample_multi(lambdas)
    }

    fn scattering(&self, pos: DVec3, outgoing: DVec3, lambdas: DVec4) -> DVec4 {
        _ = (pos, outgoing);
        self.scattering.sample_multi(lambdas)
    }

    fn phase(&self, pos: DVec3, outgoing: DVec3, incoming: DVec3, lambdas: DVec4) -> DVec4 {
        _ = (pos, outgoing, incoming, lambdas);
        DVec4::splat(1.0 / (4.0 * PI))
    }
}
