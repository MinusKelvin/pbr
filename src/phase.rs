use std::f64::consts::PI;

use glam::{DVec3, DVec4, Vec3Swizzles};

pub trait Phase: Send + Sync {
    fn f(&self, incoming: DVec3, outgoing: DVec3, lambdas: DVec4) -> DVec4;

    fn sample(&self, outgoing: DVec3, lambdas: DVec4, random: DVec3) -> DVec3 {
        _ = (outgoing, lambdas);
        crate::random::sphere(random.xy())
    }

    fn pdf(&self, incoming: DVec3, outgoing: DVec3, lambdas: DVec4) -> f64 {
        _ = (incoming, outgoing, lambdas);
        1.0 / (4.0 * PI)
    }
}

#[derive(Clone)]
pub struct Draine {
    pub alpha: f64,
    pub g: f64,
}

impl Phase for Draine {
    fn f(&self, incoming: DVec3, outgoing: DVec3, lambdas: DVec4) -> DVec4 {
        DVec4::splat(self.pdf(incoming, outgoing, lambdas))
    }

    fn sample(&self, outgoing: DVec3, lambdas: DVec4, random: DVec3) -> DVec3 {
        _ = lambdas;
        // Draine function sampling, see paper:
        // Supplemental: An Approximate Mie Scattering Function for Fog and Cloud Rendering
        // by Johannes Jendersie and Eugene d'Eon, of NVIDIA
        // when alpha = 1.0, Draine's function is Cornette-Shanks
        let Self { alpha, g } = *self;
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

    fn pdf(&self, incoming: DVec3, outgoing: DVec3, lambdas: DVec4) -> f64 {
        _ = lambdas;
        let Self { alpha, g } = *self;
        let cos = incoming.dot(outgoing);
        let numerator = (1.0 - g * g) * (1.0 + alpha * cos * cos);
        let t0 = (1.0 + g * g - 2.0 * g * cos).sqrt();
        let denominator = 4.0 * PI * t0 * t0 * t0 * (1.0 + alpha * (1.0 + 2.0 * g * g) / 3.0);
        numerator / denominator
    }
}
