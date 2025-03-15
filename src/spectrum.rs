use std::cmp::Ordering;
use std::ops::Range;
use std::sync::LazyLock;

use glam::{DMat3, DVec3, DVec4, FloatExt};
use ordered_float::OrderedFloat;

use crate::random::Tabulated1DFunction;

pub mod physical;
#[allow(unused)]
pub mod rgb;

pub trait Spectrum: Send + Sync {
    fn sample(&self, lambda: f64) -> f64;

    fn sample_multi(&self, lambdas: DVec4) -> DVec4 {
        lambdas.map(|l| self.sample(l))
    }
}

impl<S: Spectrum> Spectrum for &S {
    fn sample(&self, lambda: f64) -> f64 {
        S::sample(*self, lambda)
    }
}

pub const VISIBLE: Range<f64> = 360.0..830.0;

pub const ZERO: ConstantSpectrum = ConstantSpectrum(0.0);

#[derive(Clone, Copy)]
pub struct ConstantSpectrum(pub f64);

impl Spectrum for ConstantSpectrum {
    fn sample(&self, lambda: f64) -> f64 {
        _ = lambda;
        self.0
    }
}

#[derive(Clone)]
pub struct PiecewiseLinearSpectrum {
    data: Box<[(f64, f64)]>,
}

impl PiecewiseLinearSpectrum {
    pub fn from_points(data: &[(f64, f64)]) -> Self {
        let mut data = data.to_vec().into_boxed_slice();
        data.sort_by_key(|&(l, _)| OrderedFloat(l));
        PiecewiseLinearSpectrum { data }
    }

    pub fn from_csv(csv: &str) -> Self {
        let [this] = Self::from_csv_multi(csv);
        this
    }

    pub fn from_csv_multi<const N: usize>(csv: &str) -> [Self; N] {
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

pub struct TabulatedSpectrum {
    f: Tabulated1DFunction,
}

impl TabulatedSpectrum {
    pub fn from_spectrum(s: impl Spectrum) -> Self {
        let mut data: Vec<_> = (VISIBLE.start as usize..VISIBLE.end as usize)
            .map(|wl| s.sample(wl as f64))
            .collect();
        TabulatedSpectrum {
            f: Tabulated1DFunction::new(&data, VISIBLE.start, VISIBLE.end),
        }
    }

    pub fn raw(&self) -> &Tabulated1DFunction {
        &self.f
    }
}

impl Spectrum for TabulatedSpectrum {
    fn sample(&self, lambda: f64) -> f64 {
        self.f.f(lambda)
    }
}

#[derive(Clone)]
pub struct AmplifiedSpectrum<S> {
    pub factor: f64,
    pub s: S,
}

impl<S: Spectrum> Spectrum for AmplifiedSpectrum<S> {
    fn sample(&self, lambda: f64) -> f64 {
        self.s.sample(lambda) * self.factor
    }
}

const SRGB_TO_XYZ_T: DMat3 = DMat3::from_cols_array_2d(&[
    [0.4124, 0.3576, 0.1805],
    [0.2126, 0.7152, 0.0722],
    [0.0193, 0.1192, 0.9505],
]);

pub fn xyz_to_srgb(xyz: DVec3) -> DVec3 {
    static XYZ_TO_SRGB_MATRIX: LazyLock<DMat3> =
        LazyLock::new(|| SRGB_TO_XYZ_T.transpose().inverse());

    let srgb_linear = *XYZ_TO_SRGB_MATRIX * xyz;
    let low = srgb_linear * 12.92;
    let high = srgb_linear.powf(1.0 / 2.4) * 1.055 - 0.055;
    DVec3::select(srgb_linear.cmplt(DVec3::splat(0.0031308)), low, high)
}

pub fn srgb_to_xyz(srgb: DVec3) -> DVec3 {
    let low = srgb / 12.92;
    let high = ((srgb + 0.055) / 1.055).powf(2.4);
    let srgb_linear = DVec3::select(srgb.cmplt(DVec3::splat(0.04045)), low, high);
    SRGB_TO_XYZ_T.transpose() * srgb_linear
}

pub fn integrate_product(a: &impl Spectrum, b: &impl Spectrum) -> f64 {
    let mut result = 0.0;
    const N: usize = 1000;
    for i in 0..N {
        let lambda = VISIBLE.start.lerp(VISIBLE.end, i as f64 / N as f64);
        result += a.sample(lambda) * b.sample(lambda) * (VISIBLE.end - VISIBLE.start);
    }
    result / N as f64
}

pub fn lambda_to_xyz_absolute(lambda: f64) -> DVec3 {
    physical::cie_xyz_absolute()
        .each_ref()
        .map(|matcher| matcher.sample(lambda))
        .into()
}
