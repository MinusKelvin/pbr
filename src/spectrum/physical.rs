use std::sync::LazyLock;

use crate::spectrum::{PiecewiseLinearSpectrum, VISIBLE};

use super::Spectrum;

pub fn cie_d65() -> &'static impl Spectrum {
    static CIE_D65: LazyLock<PiecewiseLinearSpectrum> = LazyLock::new(|| {
        let mut d65 = PiecewiseLinearSpectrum::from_csv(include_str!("CIE_std_illum_D65.csv"));
        let d65_y = super::integrate_product(&d65, &cie_xyz()[1]);
        let d65_y = d65_y / (VISIBLE.end - VISIBLE.start);
        for v in &mut d65.data {
            v.1 /= d65_y;
        }
        d65
    });
    &*CIE_D65
}

pub fn cie_xyz() -> &'static [impl Spectrum; 3] {
    static CIE_XYZ: LazyLock<[PiecewiseLinearSpectrum; 3]> = LazyLock::new(|| {
        PiecewiseLinearSpectrum::from_csv_multi(include_str!("CIE_xyz_1931_2deg.csv"))
    });
    &*CIE_XYZ
}
