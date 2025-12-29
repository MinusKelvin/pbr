use std::f64::consts::PI;
use std::path::Path;
use std::sync::LazyLock;
use std::time::{Duration, Instant};

use clap::Parser;
use glam::{DMat3, DVec2, DVec3, DVec4};
use medium::Medium;
use ordered_float::OrderedFloat;
use rand::{thread_rng, Rng};
use random::Tabulated1DFunction;
use rayon::prelude::*;
use scene::Scene;
use spectrum::physical::cie_xyz_absolute;
use spectrum::{Spectrum, VISIBLE};

mod brdf;
mod bvh;
mod light;
mod material;
mod medium;
mod objects;
mod path_trace;
mod phase;
mod plymesh;
mod random;
mod scene;
mod scene_description;
mod spectrum;

#[derive(clap::Parser)]
struct Options {
    #[arg(short = 'W', default_value_t = 960)]
    width: usize,
    #[arg(short = 'H', default_value_t = 480)]
    height: usize,
    #[arg(short, default_value_t = 128)]
    samples: u32,
    #[arg(long, default_value_t = 6.0, allow_negative_numbers(true))]
    time: f64,
    #[arg(long, default_value_t = 10.0)]
    altitude: f64,
}

fn main() {
    let opt = Options::parse();

    let (scene, camera, looking, camera_medium) =
        scene_description::atmosphere_scene(opt.time, opt.altitude);
        // scene_description::load();
        // scene_description::simple_volume_scene();

    // let mut pixel = Pixel::default();
    // for _ in 0..100000 {
    //     let random = thread_rng().gen_range(0.0..1.0);
    //     let stratified = (DVec4::splat(random) + DVec4::new(0.0, 0.25, 0.5, 0.75)) % 1.0;
    //     let stratified = DVec4::splat(stratified.x);
    //     let lambdas = stratified.map(sample_wavelengths);
    //     let pdf = lambdas.map(wavelength_pdf);

    //     let Some((light, l_pdf)) = scene.sample_light(camera, lambdas, thread_rng().gen()) else {
    //         unreachable!()
    //     };
    //     let sample = light.sample(camera, lambdas, thread_rng().gen());

    //     let transmittance = 4.0 * DVec4::X * path_trace::transmittance(
    //         &scene,
    //         camera,
    //         sample.dir,
    //         lambdas,
    //         true,
    //         &camera_medium,
    //         sample.dist,
    //     );
    //     let radiance = sample.emission * transmittance / l_pdf;

    //     let mut value = DVec3::ZERO;
    //     for i in 0..4 {
    //         value += (radiance[i] / pdf[i] / 4.0) * spectrum::lambda_to_xyz_absolute(lambdas[i]);
    //     }

    //     pixel.accumulate_sample(value);
    // }
    // dbg!(pixel.mean);
    // dbg!((pixel.sterr_sq().element_sum() / 3.0).sqrt());
    // return;

    // let mut film = Film::new(1000, 1);
    // film.par_iter_mut().for_each(|(x, _, p)| {
    //     let spectrum = spectrum::physical::Blackbody {
    //         temperature: (x as f64 + 0.5) * 10.0,
    //     };

    //     for _ in 0..10000 {
    //         let lambda = sample_wavelengths(thread_rng().gen_range(0.0..1.0));
    //         let pdf = wavelength_pdf(lambda);
    //         p.accumulate_sample(
    //             spectrum::lambda_to_xyz_absolute(lambda) * spectrum.sample(lambda) / pdf,
    //         );
    //     }
    // });
    // film.save_raw("blackbody.exr");
    // return;

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

        film.save_raw(format!("partial/{to_render}.exr"));

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

    film.save_raw("raw.exr");

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

    fn save_raw(&self, path: impl AsRef<Path>) {
        use exr::prelude::*;

        let attributes = ImageAttributes {
            display_window: IntegerBounds {
                position: Vec2(0, 0),
                size: Vec2(self.width, self.height),
            },
            pixel_aspect: 1.0,
            chromaticities: Some(attribute::Chromaticities {
                red: Vec2(1.0, 0.0),
                green: Vec2(0.0, 1.0),
                blue: Vec2(0.0, 0.0),
                white: Vec2(1.0 / 3.0, 1.0 / 3.0),
            }),
            time_code: None,
            other: Default::default(),
        };

        Image::empty(attributes)
            .with_layer(Layer::new(
                (self.width, self.height),
                LayerAttributes::default(),
                Encoding::FAST_LOSSLESS,
                SpecificChannels::rgb(|Vec2(x, y): Vec2<usize>| {
                    self.data[x + y * self.width].mean.as_vec3().into()
                }),
            ))
            .write()
            .to_file(path)
            .unwrap();
    }

    fn par_iter_mut(&mut self) -> impl IndexedParallelIterator<Item = (usize, usize, &mut Pixel)> {
        self.data.par_iter_mut().enumerate().map(|(i, p)| {
            let x = i % self.width;
            let y = i / self.width;
            (x, y, p)
        })
    }

    #[allow(unused)]
    fn iter_mut(&mut self) -> impl Iterator<Item = (usize, usize, &mut Pixel)> {
        self.data.iter_mut().enumerate().map(|(i, p)| {
            let x = i % self.width;
            let y = i / self.width;
            (x, y, p)
        })
    }

    fn l_avg(&self) -> f64 {
        (self
            .data
            .iter()
            .map(|p| p.mean.y.max(0.001).ln())
            .sum::<f64>()
            / (self.data.len() as f64))
            .exp()
    }

    fn average_sterr_sq(&self) -> f64 {
        let l_avg = self.l_avg();
        self.data
            .iter()
            .map(|p| p.sterr_sq().element_sum())
            .sum::<f64>()
            / self.data.len() as f64
            / 3.0
            / l_avg
            / l_avg
    }

    fn max_sterr_sq(&self) -> f64 {
        let l_avg = self.l_avg();
        self.data
            .iter()
            .map(|p| OrderedFloat(p.sterr_sq().element_sum()))
            .max()
            .unwrap()
            .0
            / 3.0
            / l_avg
            / l_avg
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
    let fov = 1.0;
    film.par_iter_mut().for_each(|(x, y, pixel)| {
        for _ in 0..samples {
            let d;
            if width == height {
                let p = (DVec2::new(x as f64, y as f64) + thread_rng().gen::<DVec2>())
                    / DVec2::new(width as f64, height as f64);
                d = equal_area_square_to_sphere(p);
            } else {
                let x = x as f64 + thread_rng().gen::<f64>() - width as f64 / 2.0;
                let y = y as f64 + thread_rng().gen::<f64>() - height as f64 / 2.0;
                let v = DVec3::new(x / height as f64 * fov, -y / height as f64 * fov, 1.0);
                d = looking * v.normalize();
            }

            let random = thread_rng().gen_range(0.0..1.0);
            let stratified = (DVec4::splat(random) + DVec4::new(0.0, 0.25, 0.5, 0.75)) % 1.0;
            let lambdas = stratified.map(sample_wavelengths);
            let pdf = lambdas.map(wavelength_pdf);

            let radiance = path_trace::path_trace(scene, camera, d, lambdas, camera_medium);
            let mut value = DVec3::ZERO;
            for i in 0..4 {
                value +=
                    (radiance[i] / pdf[i] / 4.0) * spectrum::lambda_to_xyz_absolute(lambdas[i]);
            }

            pixel.accumulate_sample(value);
        }
    });
}

static XYZ_SUM: LazyLock<Tabulated1DFunction> = LazyLock::new(|| {
    let [x, y, z] = cie_xyz_absolute();
    let mut data = vec![0.0; x.raw().raw().len()];
    for i in 0..data.len() {
        data[i] += x.raw().raw()[i].abs();
        data[i] += y.raw().raw()[i].abs();
        data[i] += z.raw().raw()[i].abs();
    }
    Tabulated1DFunction::new(&data, VISIBLE.start, VISIBLE.end)
});

fn sample_wavelengths(random: f64) -> f64 {
    XYZ_SUM.sample(random)
}

fn wavelength_pdf(lambda: f64) -> f64 {
    XYZ_SUM.pdf(lambda)
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

fn equal_area_square_to_sphere(p: DVec2) -> DVec3 {
    let uv = 2.0 * p - 1.0;
    let uvp = uv.abs();

    let signed_distance = 1.0 - (uvp.x + uvp.y);
    let d = signed_distance.abs();
    let r = 1.0 - d;

    let phi = PI / 4.0
        * match r == 0.0 {
            true => 1.0,
            false => (uvp.y - uvp.x) / r + 1.0,
        };

    let z = (1.0 - r * r).copysign(signed_distance);

    let cos_phi = phi.cos().copysign(uv.x);
    let sin_phi = phi.sin().copysign(uv.y);

    let factor = r * (2.0 - r * r).sqrt();
    DVec3::new(cos_phi * factor, z, sin_phi * factor)
}
