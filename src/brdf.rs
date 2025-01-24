use std::f64::consts::PI;

use glam::{DVec2, DVec3, FloatExt, Vec3Swizzles};

use crate::{random, Spectrum};

pub struct BrdfSample {
    pub dir: DVec3,
    pub pdf: f64,
    pub f: Spectrum,
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
    fn f(&self, incoming: DVec3, outgoing: DVec3, normal: DVec3) -> Spectrum;

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
    fn sample(&self, outgoing: DVec3, normal: DVec3, random: DVec3) -> BrdfSample {
        _ = outgoing;
        let d = random::disk(random.xy());
        let z = (1.0 - d.length_squared()).sqrt();
        let tangent = normal.cross(outgoing).normalize();
        let bitangent = normal.cross(tangent);
        let incoming = d.x * tangent + d.y * bitangent + z * normal;

        BrdfSample {
            dir: incoming,
            pdf: self.pdf(incoming, outgoing, normal),
            f: self.f(incoming, outgoing, normal),
        }
    }

    /// Returns the probability density of the distribution sampled by [`Bsdf::sample`].
    ///
    /// The incoming and outgoing directions are backwards, pointing in the opposite direction the
    /// light is going, i.e. incoming points away from the surface and outgoing points towards.
    ///
    /// Since this is a PDF, for all `outgoing`, the integral of `pdf(incoming, outgoing)` wrt
    /// `incoming` over the sphere should be exactly 1.
    fn pdf(&self, incoming: DVec3, outgoing: DVec3, normal: DVec3) -> f64 {
        _ = outgoing;
        incoming.dot(normal) / PI
    }
}

pub struct LambertianBrdf {
    pub albedo: Spectrum,
}

impl Brdf for LambertianBrdf {
    fn f(&self, incoming: DVec3, outgoing: DVec3, normal: DVec3) -> Spectrum {
        _ = normal;
        _ = outgoing;
        _ = incoming;
        self.albedo / PI
    }
}

pub struct PhongSpecularBrdf {
    pub albedo: Spectrum,
    pub power: f64,
}

impl Brdf for PhongSpecularBrdf {
    fn f(&self, incoming: DVec3, outgoing: DVec3, normal: DVec3) -> Spectrum {
        let reflect = outgoing.reflect(normal);
        self.albedo * (self.power + 2.0) / (2.0 * PI)
            * incoming.dot(reflect).max(0.0).powf(self.power)
    }

    fn sample(&self, outgoing: DVec3, normal: DVec3, random: DVec3) -> BrdfSample {
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
            pdf: self.pdf(incoming, outgoing, normal),
            f: self.f(incoming, outgoing, normal),
        }
    }

    fn pdf(&self, incoming: DVec3, outgoing: DVec3, normal: DVec3) -> f64 {
        let reflect = outgoing.reflect(normal);
        (self.power + 1.0) / (2.0 * PI) * incoming.dot(reflect).max(0.0).powf(self.power)
    }
}

pub struct PhongRetroBrdf {
    pub power: f64,
    pub albedo: Spectrum,
}

impl Brdf for PhongRetroBrdf {
    fn f(&self, incoming: DVec3, outgoing: DVec3, normal: DVec3) -> Spectrum {
        _ = normal;
        let retro = -outgoing;
        self.albedo * (self.power + 2.0) / (2.0 * PI)
            * incoming.dot(retro).max(0.0).powf(self.power)
    }

    fn sample(&self, outgoing: DVec3, normal: DVec3, random: DVec3) -> BrdfSample {
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
            pdf: self.pdf(incoming, outgoing, normal),
            f: self.f(incoming, outgoing, normal),
        }
    }

    fn pdf(&self, incoming: DVec3, outgoing: DVec3, normal: DVec3) -> f64 {
        _ = normal;
        let retro = -outgoing;
        (self.power + 1.0) / (2.0 * PI) * incoming.dot(retro).max(0.0).powf(self.power)
    }
}

pub struct PerfectReflectionBrdf {
    pub albedo: Spectrum,
}

impl Brdf for PerfectReflectionBrdf {
    fn f(&self, incoming: DVec3, outgoing: DVec3, normal: DVec3) -> Spectrum {
        _ = incoming;
        _ = outgoing;
        _ = normal;
        Spectrum::ZERO
    }

    fn sample(&self, outgoing: DVec3, normal: DVec3, random: DVec3) -> BrdfSample {
        _ = random;
        let incoming = outgoing.reflect(normal);
        BrdfSample {
            dir: incoming,
            pdf: 1.0,
            f: self.albedo / incoming.dot(normal),
        }
    }

    fn pdf(&self, incoming: DVec3, outgoing: DVec3, normal: DVec3) -> f64 {
        _ = incoming;
        _ = outgoing;
        _ = normal;
        0.0
    }
}

pub struct CompositeBrdf<A, B> {
    pub a_weight: f64,
    pub a: A,
    pub b: B,
}

impl<A: Brdf, B: Brdf> Brdf for CompositeBrdf<A, B> {
    fn f(&self, incoming: DVec3, outgoing: DVec3, normal: DVec3) -> Spectrum {
        let a = self.a.f(incoming, outgoing, normal);
        let b = self.b.f(incoming, outgoing, normal);
        a.lerp(b, self.a_weight)
    }

    fn sample(&self, outgoing: DVec3, normal: DVec3, random: DVec3) -> BrdfSample {
        if random.z < self.a_weight {
            let mut sample =
                self.a
                    .sample(outgoing, normal, random.with_z(random.z / self.a_weight));
            sample.pdf = sample
                .pdf
                .lerp(self.b.pdf(sample.dir, outgoing, normal), self.a_weight);
            sample.f = sample
                .f
                .lerp(self.b.f(sample.dir, outgoing, normal), self.a_weight);
            sample
        } else {
            let mut sample = self.b.sample(
                outgoing,
                normal,
                random.with_z((random.z - self.a_weight) / (1.0 - self.a_weight)),
            );
            sample.pdf = self
                .a
                .pdf(sample.dir, outgoing, normal)
                .lerp(sample.pdf, self.a_weight);
            sample.f = self
                .a
                .f(sample.dir, outgoing, normal)
                .lerp(sample.f, self.a_weight);
            sample
        }
    }

    fn pdf(&self, incoming: DVec3, outgoing: DVec3, normal: DVec3) -> f64 {
        let a = self.a.pdf(incoming, outgoing, normal);
        let b = self.b.pdf(incoming, outgoing, normal);
        a.lerp(b, self.a_weight)
    }
}
