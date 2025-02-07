use std::sync::LazyLock;

use crate::spectrum::{PiecewiseLinearSpectrum, Spectrum};

pub fn ior_gold() -> &'static [impl Spectrum; 2] {
    static IOR: LazyLock<[PiecewiseLinearSpectrum; 2]> = LazyLock::new(|| {
        PiecewiseLinearSpectrum::from_csv_multi(include_str!("ior-gold_Johnson.csv"))
    });
    &IOR
}

pub fn ior_silver() -> &'static [impl Spectrum; 2] {
    static IOR: LazyLock<[PiecewiseLinearSpectrum; 2]> = LazyLock::new(|| {
        PiecewiseLinearSpectrum::from_csv_multi(include_str!("ior-silver_Johnson.csv"))
    });
    &IOR
}

pub fn ior_copper() -> &'static [impl Spectrum; 2] {
    static IOR: LazyLock<[PiecewiseLinearSpectrum; 2]> = LazyLock::new(|| {
        PiecewiseLinearSpectrum::from_csv_multi(include_str!("ior-copper_Johnson.csv"))
    });
    &IOR
}

pub fn ior_glass() -> &'static impl Spectrum {
    static IOR: LazyLock<PiecewiseLinearSpectrum> = LazyLock::new(|| {
        PiecewiseLinearSpectrum::from_csv(include_str!("ior-glass_Rubin.csv"))
    });
    &*IOR
}
