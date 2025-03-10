use std::sync::LazyLock;

use crate::spectrum::PiecewiseLinearSpectrum;

use super::Spectrum;

pub fn cie_d65_1nit() -> &'static impl Spectrum {
    static CIE_D65: LazyLock<PiecewiseLinearSpectrum> = LazyLock::new(|| {
        let mut d65 = PiecewiseLinearSpectrum::from_csv(include_str!("CIE_std_illum_D65.csv"));
        let d65_y = super::integrate_product(&d65, &cie_xyz_absolute()[1]);
        for v in &mut d65.data {
            v.1 /= d65_y;
        }
        d65
    });
    &*CIE_D65
}

/// Normalized to give Y in cd/m^2
pub fn cie_xyz_absolute() -> &'static [impl Spectrum; 3] {
    static CIE_XYZ: LazyLock<[PiecewiseLinearSpectrum; 3]> = LazyLock::new(|| {
        let mut xyz =
            PiecewiseLinearSpectrum::from_csv_multi(include_str!("CIE_xyz_1931_2deg.csv"));
        for component in &mut xyz {
            for v in &mut component.data {
                v.1 *= 683.002;
            }
        }
        xyz
    });
    &CIE_XYZ
}

pub fn extraterrestrial_solar_irradiance() -> &'static impl Spectrum {
    static SPECTRUM: LazyLock<PiecewiseLinearSpectrum> = LazyLock::new(|| {
        PiecewiseLinearSpectrum::from_csv(include_str!("gueymard_1995_extraterrestrial_solar.csv"))
    });
    &*SPECTRUM
}

pub fn ozone_absorption_coeff_sea_level() -> &'static impl Spectrum {
    static SPECTRUM: LazyLock<PiecewiseLinearSpectrum> = LazyLock::new(|| {
        PiecewiseLinearSpectrum::from_csv(include_str!(
            "pure-ozone-absorption-coeff-sea-level-serdyuchenko.csv"
        ))
    });
    &*SPECTRUM
}

pub struct Blackbody {
    pub temperature: f64,
}

impl Spectrum for Blackbody {
    fn sample(&self, lambda: f64) -> f64 {
        const C: f64 = 299_792_458.0;
        const H: f64 = 6.62606957e-34;
        const K_B: f64 = 1.3806488e-23;
        let l = lambda * 1e-9;
        let l2 = l * l;
        let l5 = l2 * l2 * l;
        let exp = H * C / (l * K_B * self.temperature);
        let radiance = 2.0 * H * C * C / (l5 * (exp.exp() - 1.0));
        // Planck's law gives radiance per meter, but we use nanometers
        radiance * 1e-9
    }
}
