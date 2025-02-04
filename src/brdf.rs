use std::f64::consts::PI;

use glam::{DVec2, DVec3, FloatExt, Vec3Swizzles};
use num::complex::Complex64;

use crate::random;
use crate::spectrum::Spectrum;

pub struct BrdfSample {
    pub dir: DVec3,
    pub pdf: f64,
    pub f: f64,
}

pub trait Brdf {
    /// Returns the amount of light reflected from the incoming direction to the outgoing
    /// direction.
    ///
    /// The incoming and outgoing directions are backwards, pointing in the opposite direction the
    /// light is going, i.e. incoming points away from the surface and outgoing points towards.
    ///
    /// This function should be *energy conserving*: for all `outgoing`, the integral of
    /// `f(incoming, outgoing) * cos(theta)` wrt `incoming` over the sphere should be <= 1.
    fn f(&self, incoming: DVec3, outgoing: DVec3, normal: DVec3, lambda: f64) -> f64;

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
    fn sample(&self, outgoing: DVec3, normal: DVec3, lambda: f64, random: DVec3) -> BrdfSample {
        _ = outgoing;
        let d = random::disk(random.xy());
        let z = (1.0 - d.length_squared()).sqrt();
        let tangent = normal.cross(outgoing).normalize();
        let bitangent = normal.cross(tangent);
        let incoming = d.x * tangent + d.y * bitangent + z * normal;

        BrdfSample {
            dir: incoming,
            pdf: self.pdf(incoming, outgoing, normal, lambda),
            f: self.f(incoming, outgoing, normal, lambda),
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
        incoming.dot(normal) / PI
    }
}

#[derive(Clone)]
pub struct LambertianBrdf<S> {
    pub albedo: S,
}

impl<S: Spectrum> Brdf for LambertianBrdf<S> {
    fn f(&self, incoming: DVec3, outgoing: DVec3, normal: DVec3, lambda: f64) -> f64 {
        _ = normal;
        _ = outgoing;
        _ = incoming;
        self.albedo.sample(lambda) / PI
    }
}

#[derive(Clone)]
pub struct PhongSpecularBrdf<S> {
    pub albedo: S,
    pub power: f64,
}

impl<S: Spectrum> Brdf for PhongSpecularBrdf<S> {
    fn f(&self, incoming: DVec3, outgoing: DVec3, normal: DVec3, lambda: f64) -> f64 {
        let reflect = outgoing.reflect(normal);
        self.albedo.sample(lambda) * (self.power + 2.0) / (2.0 * PI)
            * incoming.dot(reflect).max(0.0).powf(self.power)
    }

    fn sample(&self, outgoing: DVec3, normal: DVec3, lambda: f64, random: DVec3) -> BrdfSample {
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
            pdf: self.pdf(incoming, outgoing, normal, lambda),
            f: self.f(incoming, outgoing, normal, lambda),
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
    fn f(&self, incoming: DVec3, outgoing: DVec3, normal: DVec3, lambda: f64) -> f64 {
        _ = normal;
        let retro = -outgoing;
        self.albedo.sample(lambda) * (self.power + 2.0) / (2.0 * PI)
            * incoming.dot(retro).max(0.0).powf(self.power)
    }

    fn sample(&self, outgoing: DVec3, normal: DVec3, lambda: f64, random: DVec3) -> BrdfSample {
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
            pdf: self.pdf(incoming, outgoing, normal, lambda),
            f: self.f(incoming, outgoing, normal, lambda),
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
    fn f(&self, incoming: DVec3, outgoing: DVec3, normal: DVec3, lambda: f64) -> f64 {
        _ = incoming;
        _ = outgoing;
        _ = normal;
        _ = lambda;
        0.0
    }

    fn sample(&self, outgoing: DVec3, normal: DVec3, lambda: f64, random: DVec3) -> BrdfSample {
        _ = random;
        let incoming = outgoing.reflect(normal);
        let cos_i = incoming.dot(normal);
        let ior = Complex64::new(self.ior_re.sample(lambda), self.ior_im.sample(lambda));
        let fresnel = fresnel_reflectance_complex(cos_i, ior);
        BrdfSample {
            dir: incoming,
            pdf: 1.0,
            f: fresnel / cos_i,
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
    fn f(&self, incoming: DVec3, outgoing: DVec3, normal: DVec3, lambda: f64) -> f64 {
        let a = self.a.f(incoming, outgoing, normal, lambda);
        let b = self.b.f(incoming, outgoing, normal, lambda);
        a.lerp(b, 1.0 - self.a_weight)
    }

    fn sample(&self, outgoing: DVec3, normal: DVec3, lambda: f64, random: DVec3) -> BrdfSample {
        if random.z < self.a_weight {
            let mut sample = self.a.sample(
                outgoing,
                normal,
                lambda,
                random.with_z(random.z / self.a_weight),
            );
            sample.pdf = sample.pdf.lerp(
                self.b.pdf(sample.dir, outgoing, normal, lambda),
                1.0 - self.a_weight,
            );
            sample.f = sample.f.lerp(
                self.b.f(sample.dir, outgoing, normal, lambda),
                1.0 - self.a_weight,
            );
            sample
        } else {
            let mut sample = self.b.sample(
                outgoing,
                normal,
                lambda,
                random.with_z((random.z - self.a_weight) / (1.0 - self.a_weight)),
            );
            sample.pdf = sample.pdf.lerp(
                self.a.pdf(sample.dir, outgoing, normal, lambda),
                self.a_weight,
            );
            sample.f = sample.f.lerp(
                self.a.f(sample.dir, outgoing, normal, lambda),
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
