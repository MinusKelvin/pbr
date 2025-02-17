use glam::DVec4;

use crate::brdf::Brdf;
use crate::medium::Medium;
use crate::spectrum::Spectrum;

pub mod physical;

#[derive(Clone)]
pub struct Material<E, B, Mi, Mo> {
    pub emission: E,
    pub brdf: B,
    pub enter_medium: Mi,
    pub exit_medium: Mo,
}

pub trait MaterialErased: Send + Sync {
    fn emission_sample(&self, lambdas: DVec4) -> DVec4;

    fn brdf(&self) -> Option<&dyn Brdf>;

    fn enter_medium(&self) -> &dyn Medium;
    fn exit_medium(&self) -> &dyn Medium;
}

trait Maybe<T: ?Sized>: Send + Sync {
    fn get(&self) -> Option<&T>;
}

impl<T: ?Sized> Maybe<T> for () {
    fn get(&self) -> Option<&T> {
        None
    }
}

impl<'a, B: Brdf + 'a> Maybe<dyn Brdf + 'a> for B {
    fn get(&self) -> Option<&(dyn Brdf + 'a)> {
        Some(self)
    }
}

impl<'a, M: Medium + 'a> Maybe<dyn Medium + 'a> for M {
    fn get(&self) -> Option<&(dyn Medium + 'a)> {
        Some(self)
    }
}

impl<E: Spectrum, B: Maybe<dyn Brdf>, Mi: Maybe<dyn Medium>, Mo: Maybe<dyn Medium>> MaterialErased
    for Material<E, B, Mi, Mo>
{
    fn emission_sample(&self, lambdas: DVec4) -> DVec4 {
        self.emission.sample_multi(lambdas)
    }

    fn brdf(&self) -> Option<&dyn Brdf> {
        self.brdf.get()
    }

    fn enter_medium(&self) -> &dyn Medium {
        self.enter_medium
            .get()
            .expect("transmitting non-transmissive surface")
    }

    fn exit_medium(&self) -> &dyn Medium {
        self.exit_medium
            .get()
            .expect("transmitting non-transmissive surface")
    }
}
