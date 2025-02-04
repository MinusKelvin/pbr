use std::sync::Arc;

use glam::{BVec3, DMat4, DVec3, Vec3Swizzles};

use crate::material::{Material, MaterialErased};
use crate::Bounds;

pub struct RayHit<'a> {
    pub t: f64,
    pub normal: DVec3,
    pub geo_normal: DVec3,
    pub material: &'a dyn MaterialErased,
}

pub trait Object: Send + Sync {
    fn bounds(&self) -> Bounds;
    fn raycast(&self, origin: DVec3, direction: DVec3) -> Option<RayHit>;
}

pub struct Sphere<E, B> {
    pub origin: DVec3,
    pub radius: f64,
    pub material: Material<E, B>,
}

impl<E, B> Object for Sphere<E, B>
where
    Material<E, B>: MaterialErased,
{
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

        let normal = (origin + t * direction).normalize();
        Some(RayHit {
            t,
            normal,
            geo_normal: normal,
            material: &self.material,
        })
    }

    fn bounds(&self) -> Bounds {
        Bounds {
            min: self.origin - self.radius,
            max: self.origin + self.radius,
        }
    }
}

pub struct Triangle<E, B> {
    pub a: DVec3,
    pub b: DVec3,
    pub c: DVec3,
    pub a_n: DVec3,
    pub b_n: DVec3,
    pub c_n: DVec3,
    pub material: Material<E, B>,
}

impl<E, B> Object for Triangle<E, B>
where
    Material<E, B>: MaterialErased,
{
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
        let scale = 1.0 / det;

        let t = (a.z * e_a + b.z * e_b + c.z * e_c) * scale / d.z;

        if t < 0.0 {
            return None;
        }

        let dots = DVec3::new(
            self.a_n.dot(direction),
            self.b_n.dot(direction),
            self.c_n.dot(direction),
        );
        let n_dot = n.dot(direction);
        let normal = if dots.signum() == DVec3::splat(n_dot.signum()) {
            ((self.a_n * e_a + self.b_n * e_b + self.c_n * e_c) * scale).normalize()
        } else {
            n
        };

        Some(RayHit {
            t,
            normal,
            geo_normal: n,
            material: &self.material,
        })
    }

    fn bounds(&self) -> Bounds {
        Bounds {
            min: self.a.min(self.b).min(self.c),
            max: self.a.max(self.b).max(self.c),
        }
    }
}

pub struct Transform {
    transform: DMat4,
    inverse: DMat4,
    obj: Arc<dyn Object>,
}

impl Transform {
    pub fn new(transform: DMat4, obj: Arc<dyn Object>) -> Self {
        Transform {
            inverse: transform.inverse(),
            transform,
            obj,
        }
    }
}

impl Object for Transform {
    fn bounds(&self) -> Bounds {
        let obj_bounds = self.obj.bounds();
        (0..8)
            .map(|corner| {
                let bvec = BVec3::new(corner & 1 != 0, corner & 2 != 0, corner & 4 != 0);
                let p = DVec3::select(bvec, obj_bounds.min, obj_bounds.max);
                Bounds::point(self.transform.transform_point3(p))
            })
            .reduce(Bounds::union)
            .unwrap()
    }

    fn raycast(&self, origin: DVec3, direction: DVec3) -> Option<RayHit> {
        let orig_transformed = self.inverse.transform_point3(origin);
        let dir_transformed = self.inverse.transform_vector3(direction);
        self.obj
            .raycast(orig_transformed, dir_transformed)
            .map(|mut hit| {
                hit.normal = self.transform.transform_vector3(hit.normal).normalize();
                hit.geo_normal = self.transform.transform_vector3(hit.geo_normal).normalize();
                hit
            })
    }
}

pub struct SetMaterial<O, E, B> {
    pub material: Material<E, B>,
    pub obj: O,
}

impl<O: Object, E, B> Object for SetMaterial<O, E, B>
where
    Material<E, B>: MaterialErased,
{
    fn bounds(&self) -> Bounds {
        self.obj.bounds()
    }

    fn raycast(&self, origin: DVec3, direction: DVec3) -> Option<RayHit> {
        self.obj.raycast(origin, direction).map(|hit| RayHit {
            material: &self.material,
            ..hit
        })
    }
}
