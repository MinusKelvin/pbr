use std::f64::consts::PI;

use glam::{DMat3, DVec3, DVec4, FloatExt, Vec3Swizzles};

use crate::random;
use crate::spectrum::Spectrum;

mod phong;
pub use phong::*;

mod conductor;
pub use conductor::*;

mod dielectric;
pub use dielectric::*;

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
        let (tangent, bitangent) = normal.any_orthonormal_pair();
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

#[derive(Clone)]
pub struct TrowbridgeReitzDistribution {
    pub alpha: f64,
}

impl TrowbridgeReitzDistribution {
    pub fn d(&self, micro_normal: DVec3, macro_normal: DVec3) -> f64 {
        let alpha2 = self.alpha * self.alpha;
        let cos_theta = micro_normal.dot(macro_normal);
        let cos2_theta = cos_theta * cos_theta;
        let cos4_theta = cos2_theta * cos2_theta;
        let sin2_theta = 1.0 - cos2_theta;
        let tan2_theta = sin2_theta / cos2_theta;
        if tan2_theta.is_infinite() {
            return 0.0;
        }
        let t = 1.0 + tan2_theta / alpha2;
        1.0 / (PI * alpha2 * cos4_theta * t * t)
    }

    pub fn effectively_smooth(&self) -> bool {
        self.alpha < 0.001
    }

    pub fn g1(&self, d: DVec3, macro_normal: DVec3) -> f64 {
        1.0 / (1.0 + self.lambda(d, macro_normal))
    }

    pub fn lambda(&self, d: DVec3, macro_normal: DVec3) -> f64 {
        let cos_theta = d.dot(macro_normal);
        let cos2_theta = cos_theta * cos_theta;
        let sin2_theta = 1.0 - cos2_theta;
        let tan2_theta = sin2_theta / cos2_theta;
        if tan2_theta.is_infinite() {
            return 0.0;
        }
        ((1.0 + self.alpha * self.alpha * tan2_theta).sqrt() - 1.0) / 2.0
    }

    pub fn g(&self, incoming: DVec3, outgoing: DVec3, macro_normal: DVec3) -> f64 {
        1.0 / (1.0 + self.lambda(incoming, macro_normal) + self.lambda(-outgoing, macro_normal))
    }

    pub fn density(&self, outgoing: DVec3, micro_normal: DVec3, macro_normal: DVec3) -> f64 {
        let outgoing = -outgoing;
        self.g1(outgoing, macro_normal) / outgoing.dot(macro_normal).abs()
            * self.d(micro_normal, macro_normal)
            * outgoing.dot(micro_normal).abs()
    }

    pub fn micro_normal_pdf(&self, d: DVec3, micro_normal: DVec3, macro_normal: DVec3) -> f64 {
        self.density(d, micro_normal, macro_normal)
    }

    pub fn sample_micro_normal(&self, outgoing: DVec3, macro_normal: DVec3, random: DVec3) -> DVec3 {
        let (macro_x, macro_y) = macro_normal.any_orthonormal_pair();
        let to_global = DMat3::from_cols(macro_x, macro_y, macro_normal);
        let to_local = to_global.transpose();
        let outgoing = to_local * outgoing;

        let mut wh = (outgoing.xy() * self.alpha).extend(outgoing.z).normalize();
        if wh.z < 0.0 {
            wh = -wh;
        }

        let (t1, t2) = wh.any_orthonormal_pair();

        let mut p = crate::random::disk(random.xy());

        let h = (1.0 - p.x * p.x).sqrt();
        p.y = h.lerp(p.y, (1.0 + wh.z) / 2.0);

        let pz = 0.0f64.max(1.0 - p.length_squared()).sqrt();
        let nh = p.x * t1 + p.y * t2 + pz * wh;
        let r = (nh.xy() * self.alpha).extend(nh.z.max(1e-6)).normalize();

        to_global * r
    }
}
