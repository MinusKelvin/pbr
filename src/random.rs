use std::f64::consts::PI;

use glam::{DVec2, DVec3};
use ordered_float::OrderedFloat;

pub fn sphere(random: DVec2) -> DVec3 {
    let z = 2.0 * random.x - 1.0;
    let r = (1.0 - z * z).sqrt();
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

#[derive(Debug)]
pub struct Tabulated1DFunction {
    data: Box<[f64]>,
    min_x: f64,
    max_x: f64,
    cdf: Box<[f64]>,
}

impl Tabulated1DFunction {
    pub fn new(data: &[f64], min_x: f64, max_x: f64) -> Tabulated1DFunction {
        let mut cdf = Vec::with_capacity(data.len() + 1);
        cdf.push(0.0);
        for &v in data {
            cdf.push(cdf.last().unwrap() + v.abs() / data.len() as f64);
        }
        Tabulated1DFunction {
            data: data.to_owned().into_boxed_slice(),
            cdf: cdf.into_boxed_slice(),
            min_x,
            max_x,
        }
    }

    pub fn raw(&self) -> &[f64] {
        &self.data
    }

    pub fn f(&self, x: f64) -> f64 {
        if x < self.min_x || x >= self.max_x {
            return 0.0;
        }
        let x = (x - self.min_x) / (self.max_x - self.min_x);
        let i = (x * self.data.len() as f64) as usize;
        self.data[i]
    }

    pub fn pdf(&self, x: f64) -> f64 {
        self.f(x).abs() / self.cdf.last().unwrap() / (self.max_x - self.min_x)
    }

    pub fn sample(&self, random: f64) -> f64 {
        let random = random * self.cdf.last().unwrap();
        let x = match self
            .cdf
            .binary_search_by_key(&OrderedFloat(random), |&v| OrderedFloat(v))
        {
            Ok(i) => i as f64 / self.data.len() as f64,
            Err(i) => {
                let y_low = self.cdf[i - 1];
                let y_high = self.cdf[i];
                let t = (random - y_low) / (y_high - y_low);
                ((i - 1) as f64 + t) / self.data.len() as f64
            }
        };

        x * (self.max_x - self.min_x) + self.min_x
    }
}
