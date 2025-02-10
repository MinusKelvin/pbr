use std::f64::consts::PI;
use std::path::Path;
use std::time::{Duration, Instant};

use clap::Parser;
use glam::{DMat3, DVec3, DVec4, Vec4Swizzles};
use image::RgbImage;
use medium::Medium;
use ordered_float::OrderedFloat;
use rand::{thread_rng, Rng};
use rayon::prelude::*;
use scene::Scene;
use spectrum::Spectrum;

mod brdf;
mod bvh;
mod light;
mod material;
mod medium;
mod objects;
mod plymesh;
mod random;
mod scene;
mod scene_description;
mod spectrum;

#[derive(clap::Parser)]
struct Options {
    #[arg(short = 'W', default_value_t = 853)]
    width: usize,
    #[arg(short = 'H', default_value_t = 480)]
    height: usize,
    #[arg(short, default_value_t = 128)]
    samples: u32,
}

fn main() {
    let opt = Options::parse();

    let (scene, camera, looking, camera_medium) = scene_description::load();

    let mut film = Film::new(opt.width, opt.height);

    let t = Instant::now();
    let mut last = 0;
    for j in 1.. {
        let to_render = opt.samples.min(2.0f64.powf(j as f64 / 2.0).round() as u32);
        if to_render == last {
            break;
        }
        render(
            &mut film,
            to_render - last,
            &scene,
            camera,
            looking,
            &camera_medium,
        );
        last = to_render;

        film.save(format!("partial/{to_render}.png"));

        let d = t.elapsed();
        println!(
            "{:>8}/{} in {:>8.2} {:>12.2} paths/sec   {:>8.5} avg   {:>8.5} max",
            to_render,
            opt.samples,
            Time(d),
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
        "rendered in {:.2?} ({:.2} paths/sec) with efficiency {efficiency}",
        d,
        film.num_paths() / d.as_secs_f64()
    );
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
                .clamp(DVec3::ZERO, DVec3::ONE)
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

fn render(
    film: &mut Film,
    samples: u32,
    scene: &Scene,
    camera: DVec3,
    looking: DMat3,
    camera_medium: &dyn Medium,
) {
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

            let radiance = path_trace(scene, camera, d, lambdas, camera_medium);
            let mut value = DVec3::ZERO;
            for i in 0..4 {
                value += (radiance[i] / pdf / 4.0) * spectrum::lambda_to_xyz(lambdas[i]);
            }

            pixel.accumulate_sample(value);
        }
    });
}

fn path_trace(
    scene: &Scene,
    pos: DVec3,
    dir: DVec3,
    lambdas: DVec4,
    camera_medium: &dyn Medium,
) -> DVec4 {
    let mut throughput = DVec4::ONE;
    let mut radiance = DVec4::ZERO;
    let mut secondary_terminated = false;
    let mut pos = pos;
    let mut dir = dir;
    let mut medium = camera_medium;

    let mut bounces = 0;

    let mut specular_bounce = true;

    'mainloop: while throughput != DVec4::ZERO {
        let hit = scene.raycast(pos, dir, f64::INFINITY);
        let d = hit.as_ref().map_or(f64::INFINITY, |hit| hit.t);

        if medium.participating() {
            let majorant = match secondary_terminated {
                true => medium.majorant(lambdas.xxxx()),
                false => medium.majorant(lambdas),
            };
            let mut t = 0.0;
            loop {
                let dt = -(1.0 - thread_rng().gen::<f64>()).ln() / majorant;
                t += dt;
                if t >= d {
                    break;
                }

                let p = pos + t * dir;
                let pr_absorption = medium.absorption(p, dir, lambdas) / majorant;
                let pr_scattering = medium.scattering(p, dir, lambdas) / majorant;
                let pr_null = 1.0 - pr_absorption - pr_scattering;

                let rng: f64 = thread_rng().gen();
                if rng < pr_absorption.x {
                    throughput *= pr_absorption / pr_absorption.x;
                    radiance += throughput * medium.emission(p, dir, lambdas);
                    break 'mainloop;
                } else if rng < pr_absorption.x + pr_scattering.x {
                    throughput *= pr_scattering / pr_scattering.x;

                    let new_dir = random::sphere(thread_rng().gen());
                    let new_dir_pdf = 1.0 / (4.0 * PI);

                    throughput *= medium.phase(pos, dir, new_dir, lambdas) / new_dir_pdf;
                    pos = p;
                    dir = new_dir;
                    // since we don't light sample at scattering events
                    specular_bounce = true;

                    continue 'mainloop;
                } else {
                    throughput *= pr_null / pr_null.x;
                }
            }
        }

        if specular_bounce {
            radiance += throughput * scene.light_emission(pos, dir, lambdas, d);
        }

        let Some(hit) = hit else {
            radiance += throughput * spectrum::physical::cie_d65().sample_multi(lambdas) * 0.03;
            break;
        };

        let hit_pos = pos + dir * hit.t;

        radiance += throughput * hit.material.emission_sample(lambdas);

        // if let Some((light, pdf)) = scene.sample_light(hit_pos, lambdas, thread_rng().gen()) {
        //     let sample = light.sample(pos, lambdas, thread_rng().gen());

        //     let f = hit.material.brdf_f(sample.dir, dir, hit.normal, lambdas)
        //         * sample.emission
        //         * sample.dir.dot(hit.normal).abs();

        //     if f != DVec4::ZERO {
        //         let offset = hit.geo_normal * (1e-10 * hit.geo_normal.dot(sample.dir).signum());
        //         let hit2 = scene.raycast(hit_pos + offset, sample.dir, sample.dist);
        //         // TODO: medium transmittance
        //         // TODO: change of medium if light sample transmits
        //         if hit2.is_none() {
        //             radiance += throughput * f / pdf / sample.pdf;
        //         }
        //     }
        // }

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

        if dir.dot(sample.dir) > 0.0 {
            medium = match sample.dir.dot(hit.geo_normal) > 0.0 {
                true => hit.material.exit_medium(),
                false => hit.material.enter_medium(),
            };
        }

        let cos_theta = sample.dir.dot(hit.normal).abs();
        throughput *= sample.f * cos_theta / sample.pdf;

        let offset = hit.geo_normal * (1e-10 * hit.geo_normal.dot(sample.dir).signum());
        pos = hit_pos + offset;
        dir = sample.dir;
        specular_bounce = true; //sample.singular;

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

struct Time(Duration);

impl std::fmt::Display for Time {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let secs = self.0.as_secs_f64();
        let width = f.width().unwrap_or(0);
        if secs < 1.0 {
            match f.precision() {
                Some(prec) => write!(
                    f,
                    "{:w$.prec$}ms",
                    secs * 1000.0,
                    w = width.saturating_sub(2)
                ),
                None => write!(f, "{:w$}ms", secs * 1000.0, w = width.saturating_sub(2)),
            }
        } else if secs < 60.0 {
            match f.precision() {
                Some(prec) => write!(f, "{:w$.prec$}s", secs, w = width.saturating_sub(1)),
                None => write!(f, "{:w$}s", secs, w = width.saturating_sub(1)),
            }
        } else if secs < 3600.0 {
            match f.precision() {
                Some(prec) => write!(f, "{:w$.prec$}m", secs / 60.0, w = width.saturating_sub(1)),
                None => write!(f, "{:w$}m", secs / 60.0, w = width.saturating_sub(1)),
            }
        } else {
            match f.precision() {
                Some(prec) => write!(
                    f,
                    "{:w$.prec$}hr",
                    secs / 3600.0,
                    w = width.saturating_sub(2)
                ),
                None => write!(f, "{:w$}hr", secs / 3600.0, w = width.saturating_sub(2)),
            }
        }
    }
}
