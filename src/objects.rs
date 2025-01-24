use std::sync::Arc;

use glam::DVec3;

use crate::brdf::Brdf;
use crate::Spectrum;

pub struct RayHit<'a> {
    pub t: f64,
    pub normal: DVec3,
    pub material: &'a Material,
}

#[derive(Clone)]
pub struct Material {
    pub emission: Spectrum,
    pub brdf: Arc<dyn Brdf + Send + Sync>,
}

pub trait Object {
    fn raycast(&self, origin: DVec3, direction: DVec3) -> Option<RayHit>;
}

pub struct Sphere {
    pub origin: DVec3,
    pub radius: f64,
    pub material: Material,
}

impl Object for Sphere {
    fn raycast(&self, origin: DVec3, direction: DVec3) -> Option<RayHit> {
        let origin = origin - self.origin;
        // radius = sqrt(lengthsq(o + t*d))
        // radius^2 = sum_i (o[i] + t*d[i])^2
        // radius^2 = sum_i (o[i]^2 + 2*t*o[i]*d[i] + d[i]^2*t^2)
        // radius^2 = dot(o, o) + 2*dot(o, d)*t + dot(d, d)*t^2
        // 0 = 1*t^2 + 2*sum(d)*t + dot(o, o)-radius^2
        let a = 1.0;
        let b = 2.0 * origin.dot(direction);
        let c = origin.dot(origin) - self.radius * self.radius;
        let det = b * b - 4.0 * a * c;

        if det < 0.0 {
            return None;
        }

        let sqrt = det.sqrt();

        let t0 = (-b - sqrt) / 2.0;
        let t1 = (-b + sqrt) / 2.0;

        if t1 < 0.0001 {
            return None;
        }

        let t = match t0 > 0.0 {
            true => t0,
            false => t1,
        };

        Some(RayHit {
            t,
            normal: (origin + t * direction).normalize(),
            material: &self.material,
        })
    }
}

pub struct Plane {
    pub point: DVec3,
    pub normal: DVec3,
    pub material: Material,
}

impl Object for Plane {
    fn raycast(&self, origin: DVec3, direction: DVec3) -> Option<RayHit> {
        let o = origin.dot(self.normal);
        let d = direction.dot(self.normal);
        let p = self.point.dot(self.normal);

        let t = (p - o) / d;

        (t > 0.0).then_some(RayHit {
            t,
            normal: self.normal,
            material: &self.material,
        })
    }
}
