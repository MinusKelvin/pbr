#![allow(unused)]

use std::f64::consts::PI;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use brdf::{DielectricBrdf, LambertianBrdf, SmoothConductorBrdf, ThinDielectricBrdf};
use bvh::Bvh;
use glam::{DMat3, DMat4, DQuat, DVec3, DVec4, EulerRot};
use image::RgbImage;
use light::DistantDiskLight;
use material::physical::ior_glass;
use material::Material;
use objects::{Object, RayHit, SetMaterial, Sphere, Transform, Triangle};
use ordered_float::OrderedFloat;
use rand::{thread_rng, Rng};
use rayon::prelude::*;
use scene::Scene;
use spectrum::physical::cie_d65;
use spectrum::{AmplifiedSpectrum, ConstantSpectrum, PiecewiseLinearSpectrum, Spectrum};

mod brdf;
mod bvh;
mod light;
mod material;
mod objects;
mod plymesh;
mod random;
mod scene;
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

    let mut scene = Scene::new();

    scene.add(Triangle {
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
    });
    scene.add(Triangle {
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
    });

    let t = Instant::now();
    let dragon = Arc::new(Bvh::build(dragon));
    let bunny = Arc::new(Bvh::build(bunny));
    println!("Took {:.2?} to build BVH", t.elapsed());

    scene.add(SetMaterial {
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
    });
    scene.add(SetMaterial {
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
    });
    scene.add(Transform::new(DMat4::from_translation(-cb_dragon), dragon));
    scene.add(Transform::new(
        DMat4::from_scale_rotation_translation(
            DVec3::splat(2.0),
            DQuat::IDENTITY,
            DVec3::new(0.0, 0.0, dragon_bounds.min.x - dragon_bounds.max.x) - cb_bunny * 2.0,
        ),
        bunny,
    ));
    scene.add(Sphere {
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
    });
    scene.add(Sphere {
        origin: DVec3::new(
            (dragon_bounds.max.x - dragon_bounds.min.x) * 0.5,
            (dragon_bounds.max.z - dragon_bounds.min.z) * 0.5,
            (dragon_bounds.max.z - dragon_bounds.min.z) * 0.8,
        ),
        radius: (dragon_bounds.max.z - dragon_bounds.min.z) * 0.5,
        material: Material {
            emission: spectrum::ZERO,
            // brdf: SmoothConductorBrdf::new(material::physical::ior_silver()),
            brdf: DielectricBrdf { ior: ior_glass() },
        },
    });

    scene.add_light(DistantDiskLight {
        emission: AmplifiedSpectrum {
            factor: 25.0,
            s: cie_d65(),
        },
        dir: DVec3::new(-1.0, 0.5, -0.3).normalize(),
        cos_radius: 10f64.to_radians().cos(),
    });

    let approx_model_size = (dragon_bounds.max - dragon_bounds.min).length() * 0.8;

    const N: usize = 100;
    const S: u32 = 1000;
    for i in 0..1 {
        let mut film = Film::new(853, 480);

        let yaw = i as f64 / N as f64 * PI * 2.0;
        let looking = DMat3::from_euler(EulerRot::YXZ, yaw - 0.3, -0.4, 0.0);
        let camera = approx_model_size * (looking * DVec3::Z + DVec3::new(0.0, 0.5, 0.0));

        let t = Instant::now();
        let mut last = 0;
        for j in 1.. {
            let to_render = S.min(2.0f64.powf(j as f64 / 2.0).round() as u32);
            if to_render == last {
                break;
            }
            render(&mut film, to_render - last, &scene, camera, looking);
            last = to_render;

            film.save(format!("partial/{to_render}.png"));

            let d = t.elapsed();
            println!(
                "{:>6}/{S} in {d:>8.2?} {:>12.2} paths/sec   {:>8.5} avg   {:>8.5} max",
                to_render,
                film.num_paths() / d.as_secs_f64(),
                film.average_sterr_sq().sqrt(),
                film.max_sterr_sq().sqrt()
            );
        }

        film.save("img.png");
        film.save_conf("conf.png");

        let d = t.elapsed();
        let efficiency = 1.0 / (film.average_sterr_sq() * d.as_secs_f64());
        println!(
            "rendered frame {i} in {:.2?} ({:.2} paths/sec) with efficiency {efficiency}",
            d,
            film.num_paths() / d.as_secs_f64()
        );
    }
}

struct Film {
    width: usize,
    height: usize,
    data: Box<[Pixel]>,
}

#[derive(Default)]
struct Pixel {
    mean: DVec3,
    m2: DVec3,
    count: f64,
}

impl Film {
    fn new(width: usize, height: usize) -> Self {
        Film {
            width,
            height,
            data: std::iter::repeat_with(Default::default)
                .take(width * height)
                .collect(),
        }
    }

    fn save(&self, path: impl AsRef<Path>) {
        let image = RgbImage::from_fn(self.width as u32, self.height as u32, |x, y| {
            self.data[x as usize + y as usize * self.width]
                .srgb()
                .to_array()
                .map(|v| (v * 255.0).round() as u8)
                .into()
        });
        image.save(path).unwrap();
    }

    fn save_conf(&self, path: impl AsRef<Path>) {
        let image = RgbImage::from_fn(self.width as u32, self.height as u32, |x, y| {
            spectrum::xyz_to_srgb(
                self.data[x as usize + y as usize * self.width]
                    .sterr_sq()
                    .map(f64::sqrt),
            )
            .to_array()
            .map(|v| (v * 255.0).round() as u8)
            .into()
        });
        image.save(path).unwrap();
    }

    fn par_iter_mut(&mut self) -> impl IndexedParallelIterator<Item = (usize, usize, &mut Pixel)> {
        self.data.par_iter_mut().enumerate().map(|(i, p)| {
            let x = i % self.width;
            let y = i / self.width;
            (x, y, p)
        })
    }

    fn average_sterr_sq(&self) -> f64 {
        self.data
            .iter()
            .map(|p| p.sterr_sq().element_sum())
            .sum::<f64>()
            / self.data.len() as f64
            / 3.0
    }

    fn max_sterr_sq(&self) -> f64 {
        self.data
            .iter()
            .map(|p| OrderedFloat(p.sterr_sq().element_sum()))
            .max()
            .unwrap()
            .0
            / 3.0
    }

    fn num_paths(&self) -> f64 {
        self.data.iter().map(|p| p.count).sum()
    }
}

impl Pixel {
    fn accumulate_sample(&mut self, value: DVec3) {
        let delta = value - self.mean;
        self.count += 1.0;
        self.mean += delta / self.count;
        let delta2 = value - self.mean;
        self.m2 += delta * delta2;
    }

    fn srgb(&self) -> DVec3 {
        spectrum::xyz_to_srgb(self.mean)
    }

    fn sterr_sq(&self) -> DVec3 {
        self.m2 / (self.count - 1.0) / self.count
    }
}

fn render(film: &mut Film, samples: u32, scene: &Scene, camera: DVec3, looking: DMat3) {
    let width = film.width;
    let height = film.height;
    let fov = 2.0;
    film.par_iter_mut().for_each(|(x, y, pixel)| {
        for _ in 0..samples {
            let x = x as f64 + thread_rng().gen::<f64>() - width as f64 / 2.0;
            let y = y as f64 + thread_rng().gen::<f64>() - height as f64 / 2.0;
            let d = DVec3::new(x / height as f64 * fov, -y / height as f64 * fov, -1.0);
            let d = looking * d.normalize();

            let lambda = thread_rng().gen_range(spectrum::VISIBLE);
            let r = spectrum::VISIBLE.end - spectrum::VISIBLE.start;
            let pdf = 1.0 / r;
            let lambdas = DVec4::new(
                lambda,
                (lambda + r / 4.0 - spectrum::VISIBLE.start) % r + spectrum::VISIBLE.start,
                (lambda + 2.0 * r / 4.0 - spectrum::VISIBLE.start) % r + spectrum::VISIBLE.start,
                (lambda + 3.0 * r / 4.0 - spectrum::VISIBLE.start) % r + spectrum::VISIBLE.start,
            );

            let radiance = path_trace(scene, camera, d, lambdas);
            let mut value = DVec3::ZERO;
            for i in 0..4 {
                value += (radiance[i] / pdf / 4.0) * spectrum::lambda_to_xyz(lambdas[i]);
            }

            pixel.accumulate_sample(value);
        }
    });
}

fn path_trace(scene: &Scene, pos: DVec3, dir: DVec3, lambdas: DVec4) -> DVec4 {
    let mut throughput = DVec4::ONE;
    let mut radiance = DVec4::ZERO;
    let mut secondary_terminated = false;
    let mut pos = pos;
    let mut dir = dir;

    let mut bounces = 0;

    let mut singular_bounce = true;

    while throughput != DVec4::ZERO {
        let hit = scene.raycast(pos, dir, f64::INFINITY);

        if singular_bounce {
            let max_t = hit.as_ref().map_or(f64::INFINITY, |h| h.t);
            radiance += throughput * scene.light_emission(pos, dir, lambdas, max_t);
        }

        let Some(hit) = hit else { break };

        let hit_pos = pos + dir * hit.t;

        radiance += throughput * hit.material.emission_sample(lambdas);

        if let Some((light, pdf)) = scene.sample_light(hit_pos, lambdas, thread_rng().gen()) {
            let sample = light.sample(pos, lambdas, thread_rng().gen());

            let f = hit.material.brdf_f(sample.dir, dir, hit.normal, lambdas)
                * sample.emission
                * sample.dir.dot(hit.normal).abs();

            if f != DVec4::ZERO {
                let offset = hit.geo_normal * (1e-10 * hit.geo_normal.dot(sample.dir).signum());
                let hit2 = scene.raycast(hit_pos + offset, sample.dir, sample.dist);
                if hit2.is_none() {
                    radiance += throughput * f / pdf / sample.pdf;
                }
            }
        }

        let sample = hit
            .material
            .brdf_sample(dir, hit.normal, lambdas, thread_rng().gen());

        if sample.dir == DVec3::ZERO {
            break;
        }

        if sample.terminate_secondary && !secondary_terminated {
            throughput.x *= 4.0;
            throughput.y = 0.0;
            throughput.z = 0.0;
            throughput.w = 0.0;
            secondary_terminated = true;
        }

        let cos_theta = sample.dir.dot(hit.normal).abs();
        throughput *= sample.f * cos_theta / sample.pdf;

        let offset = hit.geo_normal * (1e-10 * hit.geo_normal.dot(sample.dir).signum());
        pos = hit_pos + offset;
        dir = sample.dir;
        singular_bounce = sample.singular;

        if throughput.max_element() < 0.5 || bounces > 20 {
            if bounces > 20 {
                bounces = 0;
            }
            if thread_rng().gen_bool(0.5) {
                break;
            } else {
                throughput *= 2.0;
            }
        }

        bounces += 1;
    }

    radiance
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
