use std::f64::consts::PI;

use glam::{DVec3, DVec4, Vec3Swizzles};

use crate::phase::{Draine, Phase};
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

    fn phase(&self, pos: DVec3, incoming: DVec3, outgoing: DVec3, lambdas: DVec4) -> DVec4;

    fn sample_phase(&self, pos: DVec3, outgoing: DVec3, lambdas: DVec4, random: DVec3) -> DVec3 {
        _ = (pos, outgoing, lambdas);
        crate::random::sphere(random.xy())
    }

    fn pdf_phase(&self, pos: DVec3, incoming: DVec3, outgoing: DVec3, lambdas: DVec4) -> f64 {
        _ = (pos, incoming, outgoing, lambdas);
        1.0 / (4.0 * PI)
    }

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

    fn phase(&self, pos: DVec3, incoming: DVec3, outgoing: DVec3, lambdas: DVec4) -> DVec4 {
        _ = (pos, outgoing, incoming, lambdas);
        DVec4::ZERO
    }

    fn participating(&self) -> bool {
        false
    }
}

#[derive(Clone)]
pub struct CombinedMedium<M1, M2> {
    pub m1: M1,
    pub m2: M2,
}

impl<M1: Medium, M2: Medium> Medium for CombinedMedium<M1, M2> {
    fn majorant(&self, lambdas: DVec4) -> f64 {
        self.m1.majorant(lambdas) + self.m2.majorant(lambdas)
    }

    fn absorption(&self, pos: DVec3, outgoing: DVec3, lambdas: DVec4) -> DVec4 {
        self.m1.absorption(pos, outgoing, lambdas) + self.m2.absorption(pos, outgoing, lambdas)
    }

    fn emission(&self, pos: DVec3, outgoing: DVec3, lambdas: DVec4) -> DVec4 {
        self.m1.emission(pos, outgoing, lambdas) + self.m2.emission(pos, outgoing, lambdas)
    }

    fn scattering(&self, pos: DVec3, outgoing: DVec3, lambdas: DVec4) -> DVec4 {
        self.m1.scattering(pos, outgoing, lambdas) + self.m2.scattering(pos, outgoing, lambdas)
    }

    fn phase(&self, pos: DVec3, incoming: DVec3, outgoing: DVec3, lambdas: DVec4) -> DVec4 {
        let s1 = self.m1.scattering(pos, outgoing, lambdas);
        let s2 = self.m2.scattering(pos, outgoing, lambdas);
        let t = s1 / (s1 + s2);
        let p1 = self.m1.phase(pos, outgoing, incoming, lambdas);
        let p2 = self.m2.phase(pos, outgoing, incoming, lambdas);
        p1 * t + p2 * (1.0 - t)
    }

    fn sample_phase(&self, pos: DVec3, outgoing: DVec3, lambdas: DVec4, random: DVec3) -> DVec3 {
        let s1 = self.m1.scattering(pos, outgoing, lambdas).x;
        let s2 = self.m2.scattering(pos, outgoing, lambdas).x;
        let t = s1 / (s1 + s2);
        if random.z < t {
            let random = random.with_z(random.z / t);
            self.m1.sample_phase(pos, outgoing, lambdas, random)
        } else {
            let random = random.with_z((random.z - t) / (1.0 - t));
            self.m2.sample_phase(pos, outgoing, lambdas, random)
        }
    }

    fn pdf_phase(&self, pos: DVec3, incoming: DVec3, outgoing: DVec3, lambdas: DVec4) -> f64 {
        let s1 = self.m1.scattering(pos, outgoing, lambdas).x;
        let s2 = self.m2.scattering(pos, outgoing, lambdas).x;
        let t = s1 / (s1 + s2);
        let pdf1 = self.m1.pdf_phase(pos, incoming, outgoing, lambdas);
        let pdf2 = self.m2.pdf_phase(pos, incoming, outgoing, lambdas);
        pdf1 * t + pdf2 * (1.0 - t)
    }
}

#[derive(Clone)]
pub struct TestMedium<Sa, Se, Ss> {
    pub absorption: Sa,
    pub emission: Se,
    pub scattering: Ss,
}

impl<Sa: Spectrum, Se: Spectrum, Ss: Spectrum> Medium for TestMedium<Sa, Se, Ss> {
    fn majorant(&self, lambdas: DVec4) -> f64 {
        (self.absorption.sample_multi(lambdas) + self.scattering.sample_multi(lambdas))
            .max_element()
    }

    fn absorption(&self, pos: DVec3, outgoing: DVec3, lambdas: DVec4) -> DVec4 {
        _ = (pos, outgoing);
        self.absorption.sample_multi(lambdas) * (1.0 - pos.length())
    }

    fn emission(&self, pos: DVec3, outgoing: DVec3, lambdas: DVec4) -> DVec4 {
        _ = (pos, outgoing);
        self.emission.sample_multi(lambdas)
    }

    fn scattering(&self, pos: DVec3, outgoing: DVec3, lambdas: DVec4) -> DVec4 {
        _ = (pos, outgoing);
        self.scattering.sample_multi(lambdas) * (1.0 - pos.length())
    }

    fn phase(&self, pos: DVec3, incoming: DVec3, outgoing: DVec3, lambdas: DVec4) -> DVec4 {
        _ = (pos, outgoing, incoming, lambdas);
        DVec4::splat(1.0 / (4.0 * PI))
    }
}

#[derive(Clone)]
pub struct AtmosphereRayleigh {
    pub origin: DVec3,
    pub sea_level: f64,
    pub height_scale: f64,
}

impl AtmosphereRayleigh {
    fn density_coefficient(&self, lambdas: DVec4) -> DVec4 {
        const NSQ_M1: f64 = 1.00029 * 1.00029 - 1.0;
        const COEFFICIENT: f64 = 8.0 * PI * PI * PI * NSQ_M1 * NSQ_M1 / (3.0 * 2.504e25);

        let lm = lambdas * 1e-9;
        COEFFICIENT / (lm * lm * lm * lm)
    }
}

impl Medium for AtmosphereRayleigh {
    fn majorant(&self, lambdas: DVec4) -> f64 {
        self.density_coefficient(lambdas).max_element()
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
        _ = outgoing;
        let altitude = (pos - self.origin).length() - self.sea_level;
        self.density_coefficient(lambdas) * (-altitude / self.height_scale).exp()
    }

    fn phase(&self, pos: DVec3, incoming: DVec3, outgoing: DVec3, lambdas: DVec4) -> DVec4 {
        _ = (pos, lambdas);
        let cos_theta = outgoing.dot(incoming);
        DVec4::splat(3.0 / (16.0 * PI) * (1.0 + cos_theta * cos_theta))
    }
}

#[derive(Clone)]
pub struct AtmosphereMie {
    pub origin: DVec3,
    pub sea_level: f64,
    pub sea_level_density: f64,
    pub height_scale: f64,
}

impl AtmosphereMie {
    const PHASE: Draine = Draine { alpha: 1.0, g: 0.76 };
}

impl Medium for AtmosphereMie {
    fn majorant(&self, lambdas: DVec4) -> f64 {
        _ = lambdas;
        self.sea_level_density
    }

    fn absorption(&self, pos: DVec3, outgoing: DVec3, lambdas: DVec4) -> DVec4 {
        _ = (pos, outgoing, lambdas);
        self.scattering(pos, outgoing, lambdas) * 0.1
    }

    fn emission(&self, pos: DVec3, outgoing: DVec3, lambdas: DVec4) -> DVec4 {
        _ = (pos, outgoing, lambdas);
        DVec4::ZERO
    }

    fn scattering(&self, pos: DVec3, outgoing: DVec3, lambdas: DVec4) -> DVec4 {
        _ = (outgoing, lambdas);
        let altitude = (pos - self.origin).length() - self.sea_level;
        DVec4::splat(self.sea_level_density * (-altitude / self.height_scale).exp())
    }

    fn phase(&self, pos: DVec3, incoming: DVec3, outgoing: DVec3, lambdas: DVec4) -> DVec4 {
        _ = pos;
        Self::PHASE.f(incoming, outgoing, lambdas)
    }

    fn sample_phase(&self, pos: DVec3, outgoing: DVec3, lambdas: DVec4, random: DVec3) -> DVec3 {
        _ = pos;
        Self::PHASE.sample(outgoing, lambdas, random)
    }

    fn pdf_phase(&self, pos: DVec3, incoming: DVec3, outgoing: DVec3, lambdas: DVec4) -> f64 {
        _ = pos;
        Self::PHASE.pdf(incoming, outgoing, lambdas)
    }
}
