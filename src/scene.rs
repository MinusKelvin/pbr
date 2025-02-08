use std::sync::Arc;

use glam::DVec3;

use crate::objects::{Object, RayHit};

pub struct Scene {
    objects: Vec<Arc<dyn Object>>,
}

impl Scene {
    pub fn new() -> Self {
        Scene { objects: vec![] }
    }

    pub fn add<O: Object + 'static>(&mut self, obj: impl Into<Arc<O>>) {
        self.objects.push(obj.into());
    }

    pub fn raycast(&self, origin: DVec3, direction: DVec3) -> Option<RayHit> {
        let mut closest = None;
        for obj in &self.objects {
            if let Some(hit) = obj.raycast(origin, direction) {
                if closest
                    .as_ref()
                    .is_none_or(|old: &RayHit| hit.t < old.t - hit.normal.dot(direction) * 1.0e-12)
                {
                    closest = Some(hit);
                }
            }
        }
        closest
    }
}
