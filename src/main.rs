use core::f64;
use std::f64::consts::PI;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use brdf::{DielectricBrdf, LambertianBrdf, SmoothConductorBrdf};
use bvh::Bvh;
use glam::{DMat3, DMat4, DQuat, DVec3, EulerRot, Vec3};
use image::{Rgb32FImage, RgbImage};
use material::physical::ior_glass;
use material::Material;
use objects::{Object, RayHit, SetMaterial, Sphere, Transform, Triangle};
use rand::{thread_rng, Rng};
use rayon::prelude::*;
use spectrum::physical::cie_d65;
use spectrum::{ConstantSpectrum, Spectrum};

mod brdf;
mod bvh;
mod material;
mod objects;
mod plymesh;
mod random;
mod spectrum;

fn main() {
    let t = Instant::now();
    let (dragon, dragon_bounds) = plymesh::load_plymesh(
        std::fs::File::open("models/dragon_vrip.ply").unwrap(),
        &Material {
            emission: spectrum::ZERO,
            // brdf: LambertianBrdf {
            //     albedo: DVec3::new(1.0, 0.25, 0.25),
            // },
            // brdf: SmoothConductorBrdf::new(material::physical::ior_gold()),
            brdf: DielectricBrdf { ior: ior_glass() },
        },
    )
    .unwrap();
    let (bunny, bunny_bounds) = plymesh::load_plymesh(
        std::fs::File::open("models/bun_zipper.ply").unwrap(),
        &Material {
            emission: spectrum::ZERO,
            brdf: SmoothConductorBrdf::new(material::physical::ior_gold()),
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
                emission: spectrum::ZERO,
                brdf: LambertianBrdf {
                    albedo: ConstantSpectrum(0.5),
                },
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
                emission: spectrum::ZERO,
                brdf: LambertianBrdf {
                    albedo: ConstantSpectrum(0.5),
                },
            },
        }),
    ];

    let t = Instant::now();
    let dragon = Arc::new(Bvh::build(dragon));
    let bunny = Arc::new(Bvh::build(bunny));
    println!("Took {:.2?} to build BVH", t.elapsed());

    objects.push(Arc::new(SetMaterial {
        material: Material {
            emission: spectrum::ZERO,
            // brdf: Arc::new(LambertianBrdf {
            //     albedo: DVec3::new(0.25, 1.0, 0.25),
            // }),
            brdf: SmoothConductorBrdf::new(material::physical::ior_gold()),
            // brdf: DielectricBrdf {
            //     ior: ConstantSpectrum(1.5),
            // },
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
            emission: spectrum::ZERO,
            // brdf: Arc::new(LambertianBrdf {
            //     albedo: DVec3::new(0.25, 0.25, 1.0),
            // }),
            brdf: SmoothConductorBrdf::new(material::physical::ior_copper()),
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
            emission: spectrum::ZERO,
            brdf: SmoothConductorBrdf::new(material::physical::ior_silver()),
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
            emission: spectrum::ZERO,
            brdf: SmoothConductorBrdf::new(material::physical::ior_silver()),
            // brdf: DielectricBrdf {
            //     ior: ConstantSpectrum(1.5),
            // },
        },
    }));

    let approx_model_size = (dragon_bounds.max - dragon_bounds.min).length() * 0.8;

    const N: usize = 100;
    const S: u32 = 100000;
    for i in 0..1 {
        let t = Instant::now();
        let yaw = i as f64 / N as f64 * PI * 2.0;
        let looking = DMat3::from_euler(EulerRot::YXZ, yaw - 0.3, -0.4, 0.0);
        let camera = approx_model_size * (looking * DVec3::Z + DVec3::new(0.0, 0.5, 0.0));
        let (img, conf, var) = render(853/10, 480/10, S, &objects, camera, looking);
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
        o.0 = spectrum::xyz_to_srgb(Vec3::from_array(i.0).as_dvec3())
            .to_array()
            .map(|v| (v * 255.0).round() as u8);
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
                let lambda = thread_rng().gen_range(spectrum::VISIBLE);
                let pdf = 1.0 / (spectrum::VISIBLE.end - spectrum::VISIBLE.start);

                // let value = raycast_scene(objects, camera, d)
                //     .map_or(DVec3::ZERO, |hit| hit.normal / 2.0 + 0.5);

                let radiance = path_trace(objects, camera, d, lambda);
                let value = radiance / pdf * spectrum::lambda_to_xyz(lambda);
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

fn path_trace(objs: &[Arc<dyn Object>], pos: DVec3, dir: DVec3, lambda: f64) -> f64 {
    let mut throughput = 1.0;
    let mut radiance = 0.0;

    let mut pos = pos;
    let mut dir = dir;

    while throughput != 0.0 {
        let Some(hit) = raycast_scene(objs, pos, dir) else {
            let strength = if dir.dot(DVec3::new(0.2, 1.0, -0.3).normalize()) > 0.99 {
                50.0
            } else {
                0.1
            };
            radiance += throughput * cie_d65().sample(lambda) * strength;
            break;
        };

        radiance += throughput * hit.material.emission_sample(lambda);

        let sample = hit
            .material
            .brdf_sample(dir, hit.normal, lambda, thread_rng().gen());

        if sample.dir == DVec3::ZERO {
            break;
        }

        let cos_theta = sample.dir.dot(hit.normal).abs();
        throughput *= sample.f * cos_theta / sample.pdf;

        let offset_dir = hit.geo_normal.dot(sample.dir).signum();
        pos += dir * hit.t
            + hit.geo_normal * (f64::EPSILON * pos.abs().max_element() * 32.0 * offset_dir);
        dir = sample.dir;

        if throughput < 0.5 {
            if thread_rng().gen_bool(0.5) {
                break;
            } else {
                throughput *= 2.0;
            }
        }
    }

    radiance
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
