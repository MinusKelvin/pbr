use std::f64::consts::PI;

use glam::{DVec2, DVec3};

pub fn sphere(random: DVec2) -> DVec3 {
    let z = 2.0 * random.x - 1.0;
    let r = (1.0 - z*z).sqrt();
    let angle = 2.0 * PI * random.y;
    let (y, x) = angle.sin_cos();
    DVec3::new(x * r, y * r, z)
}

pub fn disk(random: DVec2) -> DVec2 {
    let r = random.x.sqrt();
    let angle = 2.0 * PI * random.y;
    let (y, x) = angle.sin_cos();
    DVec2::new(x, y) * r
}
