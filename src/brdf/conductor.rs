use glam::{DVec3, DVec4};
use num::complex::Complex64;

use crate::spectrum::Spectrum;

use super::{Brdf, BrdfSample, TrowbridgeReitzDistribution};

#[derive(Clone)]
pub struct SmoothConductorBrdf<Sr, Si> {
    pub ior_re: Sr,
    pub ior_im: Si,
}

impl<'a, S> SmoothConductorBrdf<&'a S, &'a S> {
    pub fn new(ior: &'a [S; 2]) -> Self {
        SmoothConductorBrdf {
            ior_re: &ior[0],
            ior_im: &ior[1],
        }
    }
}

impl<Sr: Spectrum, Si: Spectrum> Brdf for SmoothConductorBrdf<Sr, Si> {
    fn f(&self, incoming: DVec3, outgoing: DVec3, normal: DVec3, lambdas: DVec4) -> DVec4 {
        _ = incoming;
        _ = outgoing;
        _ = normal;
        _ = lambdas;
        DVec4::ZERO
    }

    fn sample(&self, outgoing: DVec3, normal: DVec3, lambdas: DVec4, random: DVec3) -> BrdfSample {
        _ = random;
        let cos_i = -outgoing.dot(normal);
        if cos_i < 0.0 {
            return BrdfSample {
                dir: DVec3::ZERO,
                pdf: 0.0,
                f: DVec4::ZERO,
                terminate_secondary: false,
                singular: true,
            };
        }
        let incoming = outgoing.reflect(normal);
        let ior_re = self.ior_re.sample_multi(lambdas);
        let ior_im = self.ior_im.sample_multi(lambdas);
        let mut fresnel = DVec4::ZERO;
        for i in 0..4 {
            fresnel[i] = fresnel_reflectance_complex(cos_i, Complex64::new(ior_re[i], ior_im[i]));
        }
        BrdfSample {
            dir: incoming,
            pdf: 1.0,
            f: fresnel / cos_i,
            terminate_secondary: false,
            singular: true,
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
pub struct RoughConductorBrdf<Sr, Si> {
    pub ior_re: Sr,
    pub ior_im: Si,
    pub microfacets: TrowbridgeReitzDistribution,
}

impl<'a, S> RoughConductorBrdf<&'a S, &'a S> {
    pub fn new(ior: &'a [S; 2], alpha: f64) -> Self {
        RoughConductorBrdf {
            ior_re: &ior[0],
            ior_im: &ior[1],
            microfacets: TrowbridgeReitzDistribution { alpha },
        }
    }
}

impl<Sr: Spectrum, Si: Spectrum> Brdf for RoughConductorBrdf<Sr, Si> {
    fn f(&self, incoming: DVec3, outgoing: DVec3, normal: DVec3, lambdas: DVec4) -> DVec4 {
        if incoming.dot(normal) * outgoing.dot(normal) > 0.0 {
            return DVec4::ZERO;
        }
        let cos_out = outgoing.dot(normal).abs();
        let cos_in = incoming.dot(normal).abs();
        if cos_out == 0.0 || cos_in == 0.0 {
            return DVec4::ZERO;
        }
        let Some(micro_normal) = (incoming - outgoing).try_normalize() else {
            return DVec4::ZERO;
        };

        let ior_re = self.ior_re.sample_multi(lambdas);
        let ior_im = self.ior_im.sample_multi(lambdas);
        let mut fresnel = DVec4::ZERO;
        for i in 0..4 {
            fresnel[i] = fresnel_reflectance_complex(
                outgoing.dot(micro_normal).abs(),
                Complex64::new(ior_re[i], ior_im[i]),
            );
        }

        let factor = self.microfacets.d(micro_normal, normal)
            * self.microfacets.g(incoming, outgoing, normal)
            / (4.0 * cos_in * cos_out);
        fresnel * factor
    }

    fn sample(&self, outgoing: DVec3, normal: DVec3, lambdas: DVec4, random: DVec3) -> BrdfSample {
        let cos_out = -outgoing.dot(normal);
        if cos_out < 0.0 {
            return BrdfSample {
                dir: DVec3::ZERO,
                pdf: 0.0,
                f: DVec4::ZERO,
                terminate_secondary: false,
                singular: true,
            };
        }

        let micro_normal = self
            .microfacets
            .sample_micro_normal(outgoing, normal, random);
        let incoming = outgoing.reflect(micro_normal);
        if outgoing.dot(normal) * incoming.dot(normal) > 0.0 {
            return BrdfSample {
                dir: DVec3::ZERO,
                pdf: 0.0,
                f: DVec4::ZERO,
                terminate_secondary: false,
                singular: true,
            };
        }

        let pdf = self
            .microfacets
            .micro_normal_pdf(outgoing, micro_normal, normal)
            / (4.0 * outgoing.dot(micro_normal).abs());

        let cos_in = incoming.dot(normal).abs();

        let ior_re = self.ior_re.sample_multi(lambdas);
        let ior_im = self.ior_im.sample_multi(lambdas);
        let mut fresnel = DVec4::ZERO;
        for i in 0..4 {
            fresnel[i] = fresnel_reflectance_complex(
                outgoing.dot(micro_normal).abs(),
                Complex64::new(ior_re[i], ior_im[i]),
            );
        }

        let factor = self.microfacets.d(micro_normal, normal)
            * self.microfacets.g(incoming, outgoing, normal)
            / (4.0 * cos_in * cos_out);

        BrdfSample {
            dir: incoming,
            pdf,
            f: fresnel * factor,
            terminate_secondary: false,
            singular: false,
        }
    }

    fn pdf(&self, incoming: DVec3, outgoing: DVec3, normal: DVec3, lambda: f64) -> f64 {
        _ = lambda;
        let Some(mut micro_normal) = (incoming - outgoing).try_normalize() else {
            return 0.0;
        };
        if micro_normal.dot(normal) < 0.0 {
            micro_normal = -micro_normal;
        }
        self.microfacets
            .micro_normal_pdf(outgoing, micro_normal, normal)
            / (4.0 * outgoing.dot(micro_normal).abs())
    }
}

fn fresnel_reflectance_complex(cos_i: f64, rel_ior: Complex64) -> f64 {
    let sin2_i = 1.0 - cos_i * cos_i;
    let sin2_t = sin2_i / (rel_ior * rel_ior);
    let cos_t = (1.0 - sin2_t).sqrt();

    let r_par = (rel_ior * cos_i - cos_t) / (rel_ior * cos_i + cos_t);
    let r_perp = (cos_i - rel_ior * cos_t) / (cos_i + rel_ior * cos_t);

    (r_par.norm_sqr() + r_perp.norm_sqr()) / 2.0
}
