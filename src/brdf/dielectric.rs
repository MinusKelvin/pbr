use glam::{DVec3, DVec4};

use crate::spectrum::Spectrum;

use super::{Brdf, BrdfSample};

#[derive(Clone)]
pub struct DielectricBrdf<S> {
    pub ior: S,
}

impl<S: Spectrum> Brdf for DielectricBrdf<S> {
    fn f(&self, incoming: DVec3, outgoing: DVec3, normal: DVec3, lambdas: DVec4) -> DVec4 {
        _ = incoming;
        _ = outgoing;
        _ = normal;
        _ = lambdas;
        DVec4::ZERO
    }

    fn sample(&self, outgoing: DVec3, normal: DVec3, lambdas: DVec4, random: DVec3) -> BrdfSample {
        _ = random;
        let ior = self.ior.sample_multi(lambdas);
        let (ior, normal) = match outgoing.dot(normal) < 0.0 {
            true => (ior, normal),
            false => (1.0 / ior, -normal),
        };
        let reflected = outgoing.reflect(normal);
        let cos_i = reflected.dot(normal);
        let fresnel_reflect = ior.map(|ior| fresnel_reflectance_real(cos_i, ior));

        if random.z < fresnel_reflect.x {
            BrdfSample {
                dir: reflected,
                pdf: fresnel_reflect.x,
                f: fresnel_reflect / cos_i,
                terminate_secondary: false,
                singular: true,
            }
        } else {
            let refracted = outgoing.refract(normal, 1.0 / ior.x);
            BrdfSample {
                dir: refracted,
                pdf: 1.0 - fresnel_reflect.x,
                f: DVec4::splat(
                    (1.0 - fresnel_reflect.x) / refracted.dot(normal).abs() / (ior.x * ior.x),
                ),
                terminate_secondary: true,
                singular: true,
            }
        }
    }

    fn pdf(&self, incoming: DVec3, outgoing: DVec3, normal: DVec3, lambda: f64) -> f64 {
        _ = incoming;
        _ = outgoing;
        _ = normal;
        _ = lambda;
        0.0
    }
}

#[derive(Clone)]
pub struct ThinDielectricBrdf<S> {
    pub ior: S,
}

impl<S: Spectrum> Brdf for ThinDielectricBrdf<S> {
    fn f(&self, incoming: DVec3, outgoing: DVec3, normal: DVec3, lambdas: DVec4) -> DVec4 {
        _ = incoming;
        _ = outgoing;
        _ = normal;
        _ = lambdas;
        DVec4::ZERO
    }

    fn sample(
        &self,
        outgoing: DVec3,
        mut normal: DVec3,
        lambdas: DVec4,
        random: DVec3,
    ) -> BrdfSample {
        _ = random;
        let ior = self.ior.sample_multi(lambdas);
        if outgoing.dot(normal) > 0.0 {
            normal = -normal;
        }

        let reflected = outgoing.reflect(normal);
        let cos_i = reflected.dot(normal);
        let mut fresnel_reflect = ior.map(|ior| fresnel_reflectance_real(cos_i, ior));
        let t = 1.0 - fresnel_reflect;
        let adjust = t * t * fresnel_reflect / (1.0 - fresnel_reflect * fresnel_reflect);
        fresnel_reflect += DVec4::select(fresnel_reflect.cmplt(DVec4::ONE), adjust, DVec4::ZERO);

        if random.z < fresnel_reflect.x {
            BrdfSample {
                dir: reflected,
                pdf: fresnel_reflect.x,
                f: fresnel_reflect / cos_i,
                terminate_secondary: false,
                singular: true,
            }
        } else {
            BrdfSample {
                dir: outgoing,
                pdf: 1.0 - fresnel_reflect.x,
                f: (1.0 - fresnel_reflect) / cos_i,
                terminate_secondary: false,
                singular: true,
            }
        }
    }

    fn pdf(&self, incoming: DVec3, outgoing: DVec3, normal: DVec3, lambda: f64) -> f64 {
        _ = incoming;
        _ = outgoing;
        _ = normal;
        _ = lambda;
        0.0
    }
}

fn fresnel_reflectance_real(cos_i: f64, rel_ior: f64) -> f64 {
    let sin2_i = 1.0 - cos_i * cos_i;
    let sin2_t = sin2_i / (rel_ior * rel_ior);
    if sin2_t >= 1.0 {
        return 1.0;
    }
    let cos_t = (1.0 - sin2_t).sqrt();

    let r_par = (rel_ior * cos_i - cos_t) / (rel_ior * cos_i + cos_t);
    let r_perp = (cos_i - rel_ior * cos_t) / (cos_i + rel_ior * cos_t);

    (r_par * r_par + r_perp * r_perp) / 2.0
}
