use std::cmp::Ordering;
use std::ops::Range;
use std::sync::LazyLock;

use glam::{DMat3, DVec3, FloatExt};
use ordered_float::OrderedFloat;

pub mod physical;

pub trait Spectrum {
    fn sample(&self, lambda: f64) -> f64;
}

impl<S: Spectrum> Spectrum for &S {
    fn sample(&self, lambda: f64) -> f64 {
        S::sample(*self, lambda)
    }
}

pub const VISIBLE: Range<f64> = 360.0..830.0;

pub const ZERO: ConstantSpectrum = ConstantSpectrum(0.0);
pub const ONE: ConstantSpectrum = ConstantSpectrum(1.0);

#[derive(Clone, Copy)]
pub struct ConstantSpectrum(pub f64);

impl Spectrum for ConstantSpectrum {
    fn sample(&self, lambda: f64) -> f64 {
        _ = lambda;
        self.0
    }
}

pub struct PiecewiseLinearSpectrum {
    data: Box<[(f64, f64)]>,
}

impl PiecewiseLinearSpectrum {
    pub fn from_csv(csv: &str) -> Self {
        let [this] = Self::from_csv_multi(csv);
        this
    }

    pub fn from_csv_multi<const N: usize>(csv: &str) -> [Self ; N] {
        let mut result = [const { vec![] }; N];
        for line in csv.lines() {
            let mut fields = line.split(",");
            let lambda: f64 = fields.next().unwrap().parse().unwrap();
            for (j, word) in fields.enumerate() {
                result[j].push((lambda, word.parse().unwrap()));
            }
        }
        result.map(|mut v| {
            v.sort_unstable_by_key(|&(l, _)| OrderedFloat(l));
            PiecewiseLinearSpectrum {
                data: v.into_boxed_slice(),
            }
        })
    }
}

impl Spectrum for PiecewiseLinearSpectrum {
    fn sample(&self, lambda: f64) -> f64 {
        let i = self
            .data
            .binary_search_by(|&(l, _)| match l <= lambda {
                true => Ordering::Less,
                false => Ordering::Greater,
            })
            .unwrap_err();
        let (low_lambda, low_value) = self.data[i - 1];
        let (high_lambda, high_value) = self.data[i];
        low_value.lerp(
            high_value,
            (lambda - low_lambda) / (high_lambda - low_lambda),
        )
    }
}

pub fn xyz_to_srgb(xyz: DVec3) -> DVec3 {
    static MAT: LazyLock<DMat3> = LazyLock::new(|| {
        DMat3::from_cols_array_2d(&[
            [0.4124, 0.3576, 0.1805],
            [0.2126, 0.7152, 0.0722],
            [0.0193, 0.1192, 0.9505],
        ])
        .transpose()
        .inverse()
    });

    let srgb_linear = (*MAT * xyz).clamp(DVec3::ZERO, DVec3::ONE);
    let gamma_low = srgb_linear * 12.92;
    let gamma_high = srgb_linear.powf(1.0 / 2.4) * 1.055 - 0.055;
    DVec3::select(
        srgb_linear.cmplt(DVec3::splat(0.0031308)),
        gamma_low,
        gamma_high,
    )
}

pub fn integrate_product(a: &impl Spectrum, b: &impl Spectrum) -> f64 {
    let mut result = 0.0;
    const N: usize = 1000;
    for i in 0..N {
        let lambda = VISIBLE.start.lerp(VISIBLE.end, i as f64 / N as f64);
        result += a.sample(lambda as f64) * b.sample(lambda as f64);
    }
    result
}

pub fn spectrum_to_xyz(spectrum: &impl Spectrum) -> DVec3 {
    physical::cie_xyz()
        .each_ref()
        .map(|matcher| integrate_product(spectrum, matcher))
        .into()
}

pub fn lambda_to_xyz(lambda: f64) -> DVec3 {
    physical::cie_xyz()
        .each_ref()
        .map(|matcher| matcher.sample(lambda))
        .into()
}
