use std::f64::consts::PI;

use glam::{DVec3, DVec4, Vec3Swizzles};

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
        crate::random::sphere(random.xy())
    }

    fn pdf_phase(&self, pos: DVec3, incoming: DVec3, outgoing: DVec3, lambdas: DVec4) -> f64 {
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
    pub g: f64,
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
        DVec4::splat(self.pdf_phase(pos, incoming, outgoing, lambdas))
    }

    fn sample_phase(&self, pos: DVec3, outgoing: DVec3, lambdas: DVec4, random: DVec3) -> DVec3 {
        _ = (pos, lambdas);
        // Draine function sampling, see paper:
        // Supplemental: An Approximate Mie Scattering Function for Fog and Cloud Rendering
        // by Johannes Jendersie and Eugene d'Eon, of NVIDIA
        // when alpha = 1.0, Draine's function is Cornette-Shanks
        let alpha = 1.0;
        let g = self.g;
        let g2 = g * g;
        let g4 = g2 * g2;
        let t0 = alpha - alpha * g2;
        let t1 = alpha * g4 - alpha;
        let t2 = -3.0 * (4.0 * (g4 - g2) + t1 * (1.0 + g2));
        let t3 = g * (2.0 * random.x - 1.0);
        let t4 = 3.0 * g2 * (1.0 + t3) + alpha * (2.0 + g2 * (1.0 + (1.0 + 2.0 * g2) * t3));
        let t5 = t0 * (t1 * t2 + t4 * t4) + t1 * t1 * t1;
        let t6 = t0 * 4.0 * (g4 - g2);
        let t7 = (t5 + (t5 * t5 - t6 * t6 * t6).sqrt()).cbrt();
        let t8 = 2.0 * (t1 + t6 / t7 + t7) / t0;
        let t9 = (6.0 * (1.0 + g2) + t8).sqrt();
        let t10 = (6.0 * (1.0 + g2) - t8 + 8.0 * t4 / (t0 * t9)).sqrt() - t9;
        let cos_theta = g / 2.0 + 1.0 / (2.0 * g) - 1.0 / (8.0 * g) * t10 * t10;
        // thank goodness for that paper o_O

        let sin_theta = (1.0 - cos_theta * cos_theta).sqrt();
        let (y, x) = (random.y * 2.0 * PI).sin_cos();
        let (tangent, bitangent) = outgoing.any_orthonormal_pair();

        cos_theta * outgoing + sin_theta * (y * tangent + x * bitangent)
    }

    fn pdf_phase(&self, pos: DVec3, incoming: DVec3, outgoing: DVec3, lambdas: DVec4) -> f64 {
        _ = (pos, lambdas);
        // Cornette-Shanks phase function
        let u = outgoing.dot(incoming);
        let g = self.g;
        let term = (1.0 + g * g - 2.0 * g * u).sqrt();
        let denom = 8.0 * PI * (2.0 + g * g) * term * term * term;
        3.0 * (1.0 - g * g) * (1.0 + u * u) / denom
    }
}
