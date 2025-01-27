use core::f64;
use std::f64::consts::PI;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use brdf::{CompositeBrdf, LambertianBrdf, PhongSpecularBrdf, SmoothConductorBrdf};
use bvh::Bvh;
use glam::{DMat3, DMat4, DQuat, DVec3, EulerRot};
use image::{Rgb32FImage, RgbImage};
use objects::{Material, Object, RayHit, SetMaterial, Sphere, Transform, Triangle};
use rand::{thread_rng, Rng};
use rayon::prelude::*;

mod brdf;
mod bvh;
mod objects;
mod plymesh;
mod random;

type Spectrum = DVec3;

fn main() {
    let t = Instant::now();
    let (dragon, dragon_bounds) = plymesh::load_plymesh(
        std::fs::File::open("models/dragon_vrip.ply").unwrap(),
        &Material {
            emission: Spectrum::ZERO,
            brdf: Arc::new(LambertianBrdf {
                albedo: DVec3::new(1.0, 0.25, 0.25),
            }),
        },
    )
    .unwrap();
    let (bunny, bunny_bounds) = plymesh::load_plymesh(
        std::fs::File::open("models/bun_zipper.ply").unwrap(),
        &Material {
            emission: Spectrum::ZERO,
            brdf: Arc::new(SmoothConductorBrdf {
                albedo: Spectrum::ONE,
                ior_re: DVec3::new(0.22568, 0.40325, 1.3319),
                ior_im: DVec3::new(3.1919, 2.5329, 1.8693),
            }),
        },
    )
    .unwrap();
    println!("Took {:.2?} to load models", t.elapsed());

    let cb_dragon = dragon_bounds.centroid().with_y(dragon_bounds.min.y);
    let cb_bunny = bunny_bounds.centroid().with_y(bunny_bounds.min.y);

    let mut objects = vec![
        Arc::new(Triangle {
            a: DVec3::new(-10.0, 0.0, -10.0),
            b: DVec3::new(10.0, 0.0, 10.0),
            c: DVec3::new(10.0, 0.0, -10.0),
            a_n: DVec3::Y,
            b_n: DVec3::Y,
            c_n: DVec3::Y,
            material: Material {
                emission: Spectrum::ZERO,
                brdf: Arc::new(LambertianBrdf {
                    albedo: DVec3::new(0.5, 0.5, 0.5),
                }),
            },
        }) as Arc<dyn Object>,
        Arc::new(Triangle {
            a: DVec3::new(10.0, 0.0, 10.0),
            b: DVec3::new(-10.0, 0.0, -10.0),
            c: DVec3::new(-10.0, 0.0, 10.0),
            a_n: DVec3::Y,
            b_n: DVec3::Y,
            c_n: DVec3::Y,
            material: Material {
                emission: Spectrum::ZERO,
                brdf: Arc::new(LambertianBrdf {
                    albedo: DVec3::new(0.5, 0.5, 0.5),
                }),
            },
        }),
    ];

    let t = Instant::now();
    let dragon = Arc::new(Bvh::build(dragon));
    let bunny = Arc::new(Bvh::build(bunny));
    println!("Took {:.2?} to build BVH", t.elapsed());

    objects.push(Arc::new(SetMaterial {
        material: Material {
            emission: DVec3::ZERO,
            brdf: Arc::new(LambertianBrdf {
                albedo: DVec3::new(0.25, 1.0, 0.25),
            }),
        },
        obj: Transform::new(
            DMat4::from_scale_rotation_translation(
                DVec3::splat(1.5),
                DQuat::from_axis_angle(DVec3::Y, -1.0),
                DVec3::new(dragon_bounds.max.x - dragon_bounds.min.x, 0.0, 0.0) - 1.5 * cb_dragon,
            ),
            dragon.clone(),
        ),
    }));
    objects.push(Arc::new(SetMaterial {
        material: Material {
            emission: DVec3::ZERO,
            brdf: Arc::new(LambertianBrdf {
                albedo: DVec3::new(0.25, 0.25, 1.0),
            }),
        },
        obj: Transform::new(
            DMat4::from_scale_rotation_translation(
                DVec3::splat(0.5),
                DQuat::from_axis_angle(DVec3::Y, 1.0),
                DVec3::new((dragon_bounds.min.x - dragon_bounds.max.x) * 0.75, 0.0, 0.0)
                    - 0.5 * cb_dragon,
            ),
            dragon.clone(),
        ),
    }));
    objects.push(Arc::new(Transform::new(
        DMat4::from_translation(-cb_dragon),
        dragon,
    )));
    objects.push(Arc::new(Transform::new(
        DMat4::from_scale_rotation_translation(
            DVec3::splat(2.0),
            DQuat::IDENTITY,
            DVec3::new(0.0, 0.0, dragon_bounds.min.x - dragon_bounds.max.x) - cb_bunny * 2.0,
        ),
        bunny,
    )));
    objects.push(Arc::new(Sphere {
        origin: DVec3::new(
            -(dragon_bounds.max.x - dragon_bounds.min.x) * 0.7,
            (dragon_bounds.max.z - dragon_bounds.min.z) * 0.3,
            -(dragon_bounds.max.z - dragon_bounds.min.z),
        ),
        radius: (dragon_bounds.max.z - dragon_bounds.min.z) * 0.3,
        material: Material {
            emission: DVec3::ZERO,
            brdf: Arc::new(SmoothConductorBrdf {
                albedo: DVec3::ONE,
                ior_re: DVec3::new(0.058040, 0.058090, 0.046878),
                ior_im: DVec3::new(4.0944, 3.6380, 2.8028),
            }),
        },
    }));
    objects.push(Arc::new(Sphere {
        origin: DVec3::new(
            (dragon_bounds.max.x - dragon_bounds.min.x) * 0.5,
            (dragon_bounds.max.z - dragon_bounds.min.z) * 0.3,
            (dragon_bounds.max.z - dragon_bounds.min.z) * 0.8,
        ),
        radius: (dragon_bounds.max.z - dragon_bounds.min.z) * 0.3,
        material: Material {
            emission: DVec3::ZERO,
            brdf: Arc::new(SmoothConductorBrdf {
                albedo: DVec3::ONE,
                ior_re: DVec3::new(0.058040, 0.058090, 0.046878),
                ior_im: DVec3::new(4.0944, 3.6380, 2.8028),
            }),
        },
    }));

    let approx_model_size = (dragon_bounds.max - dragon_bounds.min).length() * 0.8;

    const N: usize = 100;
    const S: u32 = 100;
    for i in 0..1 {
        let t = Instant::now();
        let yaw = i as f64 / N as f64 * PI * 2.0;
        let looking = DMat3::from_euler(EulerRot::YXZ, yaw - 0.3, -0.4, 0.0);
        let camera = approx_model_size * (looking * DVec3::Z + DVec3::new(0.0, 0.5, 0.0));
        let (img, conf, var) = render(853, 480, S, &objects, camera, looking);
        let d = t.elapsed();
        let efficiency = 1.0 / (var * d.as_secs_f64());
        // save_final(&img, format!("i/{i}.png"));
        save_final(&img, "img.png");
        save_final(&conf, "conf.png");
        println!(
            "rendered frame {i} in {:.2?} ({:.2} samples/sec) with efficiency {efficiency}",
            d,
            S as f64 / d.as_secs_f64()
        );
    }
}

fn save_final(img: &Rgb32FImage, path: impl AsRef<Path>) {
    let mut out = RgbImage::new(img.width(), img.height());
    for (o, i) in out.pixels_mut().zip(img.pixels()) {
        o.0 = i.0.map(|v| (to_srgb(v.clamp(0.0, 1.0)) * 255.0) as u8);
    }
    out.save(path).unwrap();
}

fn render(
    width: u32,
    height: u32,
    samples: u32,
    objects: &[Arc<dyn Object>],
    camera: DVec3,
    looking: DMat3,
) -> (Rgb32FImage, Rgb32FImage, f64) {
    let mut image = Rgb32FImage::new(width, height);
    let mut conf_image = Rgb32FImage::new(width, height);
    let average_conf = image
        .par_enumerate_pixels_mut()
        .zip(conf_image.par_enumerate_pixels_mut())
        .map(|((x, y, out), (_, _, conf))| {
            let mut mean = DVec3::ZERO;
            let mut m2 = DVec3::ZERO;
            let mut count = 0.0;
            for _ in 0..samples {
                let x = x as f64 + thread_rng().gen::<f64>() - width as f64 / 2.0;
                let y = y as f64 + thread_rng().gen::<f64>() - height as f64 / 2.0;
                let d = DVec3::new(x / height as f64 * 2.0, -y / height as f64 * 2.0, -1.0);
                let d = looking * d.normalize();

                // let value = raycast_scene(objects, camera, d)
                //     .map_or(DVec3::ZERO, |hit| hit.normal / 2.0 + 0.5);

                let value = path_trace(objects, camera, d);
                let delta = value - mean;
                count += 1.0;
                mean += delta / count;
                let delta2 = value - mean;
                m2 += delta * delta2;
            }

            let sample_var_conf = m2 / (count - 1.0) / count;
            out.0 = mean.as_vec3().to_array();
            conf.0 = sample_var_conf.map(f64::sqrt).as_vec3().to_array();

            sample_var_conf.element_sum() / 3.0
        })
        .sum::<f64>()
        / (width * height) as f64;
    (image, conf_image, average_conf)
}

fn path_trace(objs: &[Arc<dyn Object>], pos: DVec3, dir: DVec3) -> DVec3 {
    let mut light_color = DVec3::ONE;
    let mut color = DVec3::ZERO;

    let mut pos = pos;
    let mut dir = dir;

    while light_color != DVec3::ZERO {
        let Some(hit) = raycast_scene(objs, pos, dir) else {
            color += light_color * DVec3::splat(0.5);
            break;
        };

        if hit.normal.dot(dir) > 0.0 {
            break;
        }

        color += light_color * hit.material.emission;

        let sample = hit
            .material
            .brdf
            .sample(dir, hit.normal, thread_rng().gen());
        light_color *= sample.f * sample.dir.dot(hit.normal).max(0.0) / sample.pdf;

        pos += dir * hit.t + hit.geo_normal * (f64::EPSILON * pos.abs().max_element() * 32.0);
        dir = sample.dir;

        if light_color.max_element() < 0.5 {
            if thread_rng().gen_bool(0.5) {
                break;
            } else {
                light_color *= 2.0;
            }
        }
    }

    color
}

fn raycast_scene(objs: &[Arc<dyn Object>], origin: DVec3, direction: DVec3) -> Option<RayHit> {
    let mut closest = None;
    for obj in objs {
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

fn to_srgb(v: f32) -> f32 {
    if v < 0.0031308 {
        v * 12.92
    } else {
        v.powf(1.0 / 2.4) * 1.055 - 0.055
    }
}

#[derive(Clone, Copy, Debug)]
struct Bounds {
    min: DVec3,
    max: DVec3,
}

impl Bounds {
    fn point(p: DVec3) -> Self {
        Bounds {
            min: p - 1.0e-9,
            max: p + 1.0e-9,
        }
    }

    fn ray_intersect(self, origin: DVec3, dir: DVec3, t_max: f64) -> Option<(f64, f64)> {
        let t_l = (self.min - origin) / dir;
        let t_u = (self.max - origin) / dir;
        let t_near = t_l.min(t_u);
        let t_far = t_l.max(t_u);
        let t_0 = t_near.max_element().max(0.0);
        let t_1 = t_far.min_element().min(t_max);
        (t_0 <= t_1).then_some((t_0, t_1))
    }

    fn centroid(self) -> DVec3 {
        (self.min + self.max) / 2.0
    }

    fn union(self, other: Bounds) -> Self {
        Bounds {
            min: self.min.min(other.min),
            max: self.max.max(other.max),
        }
    }
}

impl FromIterator<DVec3> for Bounds {
    fn from_iter<T: IntoIterator<Item = DVec3>>(iter: T) -> Self {
        iter.into_iter()
            .map(Bounds::point)
            .reduce(Bounds::union)
            .unwrap()
    }
}
