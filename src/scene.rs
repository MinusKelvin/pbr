use std::sync::Arc;

use glam::{DVec3, DVec4};

use crate::light::{Light, LightSample};
use crate::objects::{Object, RayHit};

pub struct Scene {
    objects: Vec<Arc<dyn Object>>,
    lights: Vec<Arc<dyn Light>>,
}

impl Scene {
    pub fn new() -> Self {
        Scene {
            objects: vec![],
            lights: vec![],
        }
    }

    pub fn add<O: Object + 'static>(&mut self, obj: impl Into<Arc<O>>) {
        self.objects.push(obj.into());
    }

    pub fn add_light<L: Light + 'static>(&mut self, light: impl Into<Arc<L>>) {
        self.lights.push(light.into());
    }

    pub fn raycast(&self, origin: DVec3, direction: DVec3, mut max_t: f64) -> Option<RayHit> {
        let mut closest = None;
        for obj in &self.objects {
            if let Some(hit) = obj.raycast(origin, direction, max_t) {
                if hit.t < max_t - hit.normal.dot(direction) * 1.0e-12 {
                    max_t = hit.t;
                    closest = Some(hit);
                }
            }
        }
        closest
    }

    pub fn sample_light(
        &self,
        pos: DVec3,
        lambdas: DVec4,
        random: f64,
    ) -> Option<(&dyn Light, f64)> {
        if self.lights.is_empty() {
            return None;
        }
        let i = random * self.lights.len() as f64;
        Some((&*self.lights[i as usize], 1.0 / self.lights.len() as f64))
    }

    pub fn light_emission(
        &self,
        pos: DVec3,
        direction: DVec3,
        lambdas: DVec4,
        max_t: f64,
    ) -> DVec4 {
        self.lights
            .iter()
            .map(|l| l.emission(pos, direction, lambdas, max_t))
            .sum()
    }
}
