use std::f64::consts::PI;

use glam::{DVec2, DVec3, DVec4, FloatExt, Vec3Swizzles};
use num::complex::Complex64;

use crate::random;
use crate::spectrum::Spectrum;

pub struct BrdfSample {
    pub dir: DVec3,
    pub pdf: f64,
    pub f: DVec4,
    pub terminate_secondary: bool,
    pub singular: bool,
}

pub trait Brdf: Send + Sync {
    /// Returns the amount of light reflected from the incoming direction to the outgoing
    /// direction.
    ///
    /// The incoming and outgoing directions are backwards, pointing in the opposite direction the
    /// light is going, i.e. incoming points away from the surface and outgoing points towards.
    ///
    /// This function should be *energy conserving*: for all `outgoing`, the integral of
    /// `f(incoming, outgoing) * cos(theta)` wrt `incoming` over the sphere should be <= 1.
    fn f(&self, incoming: DVec3, outgoing: DVec3, normal: DVec3, lambdas: DVec4) -> DVec4;

    /// Samples an incoming light direction from a distribution approximating [`Bsdf::f`], given
    /// canonical random variables on `[0, 1)`.
    ///
    /// The incoming and outgoing directions are backwards, pointing in the opposite direction the
    /// light is going, i.e. incoming points away from the surface and outgoing points towards.
    ///
    /// The PDF of the sampled distribution is provided by [`Bsdf::pdf`].
    ///
    /// The default implementation samples the hemisphere with a cosine-weighted distribution,
    /// which effectively importance samples the `cos(theta)` term in the rendering equation.
    fn sample(&self, outgoing: DVec3, normal: DVec3, lambdas: DVec4, random: DVec3) -> BrdfSample {
        _ = outgoing;
        let d = random::disk(random.xy());
        let z = (1.0 - d.length_squared()).sqrt();
        let tangent = normal.cross(outgoing).normalize();
        let bitangent = normal.cross(tangent);
        let incoming = d.x * tangent + d.y * bitangent + z * normal;

        BrdfSample {
            dir: incoming,
            pdf: self.pdf(incoming, outgoing, normal, lambdas.x),
            f: self.f(incoming, outgoing, normal, lambdas),
            terminate_secondary: false,
            singular: false,
        }
    }

    /// Returns the probability density of the distribution sampled by [`Bsdf::sample`].
    ///
    /// The incoming and outgoing directions are backwards, pointing in the opposite direction the
    /// light is going, i.e. incoming points away from the surface and outgoing points towards.
    ///
    /// Since this is a PDF, for all `outgoing`, the integral of `pdf(incoming, outgoing)` wrt
    /// `incoming` over the sphere should be exactly 1.
    fn pdf(&self, incoming: DVec3, outgoing: DVec3, normal: DVec3, lambda: f64) -> f64 {
        _ = outgoing;
        _ = lambda;
        incoming.dot(normal).max(0.0) / PI
    }

    fn name(&self) -> &'static str {
        std::any::type_name::<Self>()
    }
}

#[derive(Clone)]
pub struct LambertianBrdf<S> {
    pub albedo: S,
}

impl<S: Spectrum> Brdf for LambertianBrdf<S> {
    fn f(&self, incoming: DVec3, outgoing: DVec3, normal: DVec3, lambdas: DVec4) -> DVec4 {
        _ = incoming;
        _ = outgoing;
        if incoming.dot(normal) < 0.0 {
            return DVec4::ZERO;
        }
        self.albedo.sample_multi(lambdas) / PI
    }
}

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

#[derive(Clone)]
pub struct CompositeBrdf<A, B> {
    pub a_weight: f64,
    pub a: A,
    pub b: B,
}

impl<A: Brdf, B: Brdf> Brdf for CompositeBrdf<A, B> {
    fn f(&self, incoming: DVec3, outgoing: DVec3, normal: DVec3, lambdas: DVec4) -> DVec4 {
        let a = self.a.f(incoming, outgoing, normal, lambdas);
        let b = self.b.f(incoming, outgoing, normal, lambdas);
        a.lerp(b, 1.0 - self.a_weight)
    }

    fn sample(&self, outgoing: DVec3, normal: DVec3, lambdas: DVec4, random: DVec3) -> BrdfSample {
        if random.z < self.a_weight {
            let mut sample = self.a.sample(
                outgoing,
                normal,
                lambdas,
                random.with_z(random.z / self.a_weight),
            );
            sample.pdf = sample.pdf.lerp(
                self.b.pdf(sample.dir, outgoing, normal, lambdas.x),
                1.0 - self.a_weight,
            );
            sample.f = sample.f.lerp(
                self.b.f(sample.dir, outgoing, normal, lambdas),
                1.0 - self.a_weight,
            );
            sample
        } else {
            let mut sample = self.b.sample(
                outgoing,
                normal,
                lambdas,
                random.with_z((random.z - self.a_weight) / (1.0 - self.a_weight)),
            );
            sample.pdf = sample.pdf.lerp(
                self.a.pdf(sample.dir, outgoing, normal, lambdas.x),
                self.a_weight,
            );
            sample.f = sample.f.lerp(
                self.a.f(sample.dir, outgoing, normal, lambdas),
                self.a_weight,
            );
            sample
        }
    }

    fn pdf(&self, incoming: DVec3, outgoing: DVec3, normal: DVec3, lambda: f64) -> f64 {
        let a = self.a.pdf(incoming, outgoing, normal, lambda);
        let b = self.b.pdf(incoming, outgoing, normal, lambda);
        a.lerp(b, 1.0 - self.a_weight)
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

fn fresnel_reflectance_complex(cos_i: f64, rel_ior: Complex64) -> f64 {
    let sin2_i = 1.0 - cos_i * cos_i;
    let sin2_t = sin2_i / (rel_ior * rel_ior);
    let cos_t = (1.0 - sin2_t).sqrt();

    let r_par = (rel_ior * cos_i - cos_t) / (rel_ior * cos_i + cos_t);
    let r_perp = (cos_i - rel_ior * cos_t) / (cos_i + rel_ior * cos_t);

    (r_par.norm_sqr() + r_perp.norm_sqr()) / 2.0
}
