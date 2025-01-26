use std::sync::Arc;

use glam::{DVec3, Vec3Swizzles};

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
        // 0 = 1*t^2 + 2*dot(o, d)*t + dot(o, o)-radius^2
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

        if t1 < 0.0 {
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

pub struct Triangle {
    pub a: DVec3,
    pub b: DVec3,
    pub c: DVec3,
    pub material: Material,
}

impl Object for Triangle {
    fn raycast(&self, origin: DVec3, direction: DVec3) -> Option<RayHit> {
        let n = (self.c - self.b).cross(self.a - self.b);

        if n.length_squared() == 0.0 {
            return None;
        }

        let a = self.a - origin;
        let b = self.b - origin;
        let c = self.c - origin;

        let d = direction.abs();
        let (i_x, i_y, i_z) = match () {
            _ if d.x >= d.y && d.x >= d.z => (1, 2, 0),
            _ if d.y >= d.z => (2, 0, 1),
            _ => (0, 1, 2),
        };

        let d = DVec3::new(direction[i_x], direction[i_y], direction[i_z]);
        let a = DVec3::new(a[i_x], a[i_y], a[i_z]);
        let b = DVec3::new(b[i_x], b[i_y], b[i_z]);
        let c = DVec3::new(c[i_x], c[i_y], c[i_z]);

        let shear = d.xy() / d.z;
        let a_xy = a.xy() - a.z * shear;
        let b_xy = b.xy() - b.z * shear;
        let c_xy = c.xy() - c.z * shear;

        let e_a = b_xy.perp_dot(c_xy);
        let e_b = c_xy.perp_dot(a_xy);
        let e_c = a_xy.perp_dot(b_xy);

        let e = DVec3::new(e_a, e_b, e_c);

        if e.cmplt(DVec3::ZERO).any() && e.cmpgt(DVec3::ZERO).any() {
            return None;
        }

        let det = e.element_sum();
        if det == 0.0 {
            return None;
        }

        let t = (a.z * e_a + b.z * e_b + c.z * e_c) / det / d.z;

        if t < 0.0 {
            return None;
        }

        Some(RayHit {
            t,
            normal: n.normalize(),
            material: &self.material,
        })
    }
}
