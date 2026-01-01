use std::ops::ControlFlow::{self, Break, Continue};

use glam::{DVec3, DVec4};
use rand::prelude::*;

use crate::medium::{Medium, MediumProperties};
use crate::scene::Scene;

pub fn path_trace(
    scene: &Scene,
    pos: DVec3,
    dir: DVec3,
    lambdas: DVec4,
    camera_medium: &dyn Medium,
) -> DVec4 {
    // beta = throughput / p_{u, lambda[0]}
    let mut beta = DVec4::ONE;
    // r_u = p_u / p_path, r_l = p_l / p_path
    let mut r_u = DVec4::ONE;
    let mut r_l = DVec4::ONE;

    let mut radiance = DVec4::ZERO;
    let mut pos = pos;
    let mut dir = dir;
    let mut medium = camera_medium;

    let mut specular_bounce = true;

    let mut prev_interaction = (DVec3::ZERO, DVec3::ZERO);

    loop {
        let hit = scene.raycast(pos, dir, f64::INFINITY);
        let d = hit.as_ref().map_or(f64::INFINITY, |hit| hit.t);

        if medium.participating() {
            if hit.is_none() {
                panic!("can't exit participating medium?");
            }

            let mut scattered = false;
            let mut terminated = false;
            let t_maj = sample_tmaj(pos, dir, medium, d, lambdas, |p, mp, majorant, t_maj| {
                // compute emission from the medium
                if mp.emission != DVec4::ZERO {
                    let pdf = majorant.x * t_maj.x;
                    let beta_prime = beta * t_maj / pdf;
                    let r_e = r_u * majorant * t_maj / pdf;
                    if r_e != DVec4::ZERO {
                        radiance += beta_prime * mp.absorption * mp.emission / average(r_e);
                    }
                }

                let pr_absorption = mp.absorption / majorant;
                let pr_scattering = mp.scattering / majorant;

                let rng: f64 = thread_rng().gen();
                if rng < pr_absorption.x {
                    // absorption
                    terminated = true;
                    Break(())
                } else if rng < pr_absorption.x + pr_scattering.x {
                    // scattering
                    let pdf = t_maj.x * mp.scattering.x;
                    beta *= t_maj * mp.scattering / pdf;
                    r_u *= t_maj * mp.scattering / pdf;

                    if let Some((light, pdf)) = scene.sample_light(pos, lambdas, thread_rng().gen())
                    {
                        let sample = light.sample(pos, lambdas, thread_rng().gen());

                        let light_pdf = pdf * sample.pdf;
                        let scatter_pdf = medium.pdf_phase(p, sample.dir, dir, lambdas);

                        let tp_f =
                            beta * medium.phase(p, sample.dir, dir, lambdas) * sample.emission;

                        if tp_f != DVec4::ZERO {
                            let (transmittance, tr_u, tr_l) = transmittance_with_path_pr(
                                scene,
                                p,
                                sample.dir,
                                lambdas,
                                medium,
                                sample.dist,
                            );
                            let tr_u = tr_u * r_u * scatter_pdf;
                            let tr_l = tr_l * r_u * light_pdf;

                            radiance += tp_f * transmittance / average(tr_u + tr_l);
                        }
                    }

                    let new_dir = medium.sample_phase(p, dir, lambdas, thread_rng().gen());
                    let new_dir_pdf = medium.pdf_phase(p, new_dir, dir, lambdas);

                    beta *= medium.phase(p, new_dir, dir, lambdas) / new_dir_pdf;
                    r_l = r_u / new_dir_pdf;
                    prev_interaction = (p, DVec3::ZERO);
                    scattered = true;
                    pos = p;
                    dir = new_dir;
                    specular_bounce = false;

                    Break(())
                } else {
                    // null scattering
                    let null = majorant - mp.absorption - mp.scattering;
                    let pdf = t_maj.x * null.x;
                    beta *= t_maj * null / pdf;
                    if pdf == 0.0 {
                        beta = DVec4::ZERO;
                    }
                    r_u *= t_maj * null / pdf;
                    r_l *= t_maj * majorant / pdf;
                    if beta != DVec4::ZERO && r_u != DVec4::ZERO {
                        Continue(())
                    } else {
                        Break(())
                    }
                }
            });

            if terminated || beta == DVec4::ZERO || r_u == DVec4::ZERO {
                break;
            }
            if scattered {
                continue;
            }

            beta *= t_maj / t_maj.x;
            r_u *= t_maj / t_maj.x;
            r_l *= t_maj / t_maj.x;
        }

        for light in scene.lights() {
            let light_emission = light.emission(pos, dir, lambdas, d);
            if light_emission == DVec4::ZERO {
                continue;
            }
            if specular_bounce {
                radiance += beta * light_emission / average(r_u);
            } else {
                let light_pdf = scene.light_pmf(prev_interaction.0, lambdas, light)
                    * light.pdf(prev_interaction.0, dir, lambdas);
                // pbrt-v4 accumulates into r_l here, but that seems really, really wrong to me?
                let r_l = r_l * light_pdf;
                radiance += beta * light_emission / average(r_u + r_l);
            }
        }

        let Some(hit) = hit else {
            break;
        };

        // i don't have the infrastructure for this - can't figure out light sampling probability and stuff
        // let light_emission = hit.material.emission_sample(lambdas);
        // if light_emission == DVec4::ZERO {
        //     continue;
        // }
        // if specular_bounce {
        //     radiance += beta * light_emission / average(r_u);
        // } else {
        //     let light_pdf = scene.light_pmf(prev_interaction.0, lambdas, light)
        //         * light.pdf(prev_interaction.0, dir, lambdas);
        //     r_l *= light_pdf;
        // }

        let hit_pos = pos + dir * hit.t;
        let old_dir = dir;

        if let Some(brdf) = hit.material.brdf() {
            if let Some((light, pdf)) = scene.sample_light(pos, lambdas, thread_rng().gen()) {
                let sample = light.sample(pos, lambdas, thread_rng().gen());

                let light_pdf = pdf * sample.pdf;
                let scatter_pdf = brdf.pdf(sample.dir, dir, hit.normal, lambdas.x);

                let tp_f = beta
                    * brdf.f(sample.dir, dir, hit.normal, lambdas)
                    * sample.dir.dot(hit.normal).abs()
                    * sample.emission;

                if tp_f != DVec4::ZERO {
                    let offset = hit.geo_normal * (1e-6 * hit.geo_normal.dot(sample.dir).signum());
                    let (transmittance, tr_u, tr_l) = transmittance_with_path_pr(
                        scene,
                        hit_pos + offset,
                        sample.dir,
                        lambdas,
                        medium,
                        sample.dist,
                    );
                    let tr_u = tr_u * r_u * scatter_pdf;
                    let tr_l = tr_l * r_u * light_pdf;

                    radiance += tp_f * transmittance / average(tr_u + tr_l);
                }
            }
            prev_interaction = (hit_pos, hit.normal);

            let sample = brdf.sample(dir, hit.normal, lambdas, thread_rng().gen());

            if sample.dir == DVec3::ZERO {
                break;
            }

            let cos_theta = sample.dir.dot(hit.normal).abs();
            beta *= sample.f * cos_theta / sample.pdf;
            r_l = r_u / sample.pdf;

            dir = sample.dir;
            specular_bounce = sample.singular;
        }

        let offset = hit.geo_normal * (1e-6 * hit.geo_normal.dot(dir).signum());
        pos = hit_pos + offset;

        if old_dir.dot(hit.geo_normal).signum() == dir.dot(hit.geo_normal).signum() {
            medium = match dir.dot(hit.geo_normal) > 0.0 {
                true => hit.material.exit_medium(),
                false => hit.material.enter_medium(),
            };
        }

        let rr_beta = beta / average(r_u);
        if rr_beta.max_element() < 1.0 {
            let q = 1.0 - rr_beta.max_element();
            if thread_rng().gen_bool(q) {
                break;
            } else {
                beta /= 1.0 - q;
            }
        }
    }

    radiance
}

pub fn transmittance_with_path_pr<'a>(
    scene: &'a Scene,
    mut pos: DVec3,
    dir: DVec3,
    lambdas: DVec4,
    mut medium: &'a dyn Medium,
    mut d: f64,
) -> (DVec4, DVec4, DVec4) {
    let mut t_ray = DVec4::ONE;
    let mut r_l = DVec4::ONE;
    let mut r_u = DVec4::ONE;

    loop {
        let Some(hit) = scene.raycast(pos, dir, d) else {
            if medium.participating() {
                panic!("can't exit participating medium?");
            }
            break;
        };

        if hit.t < d && hit.material.brdf().is_some() {
            return (DVec4::ZERO, DVec4::ONE, DVec4::ONE);
        }

        if medium.participating() {
            let t_max = d.min(hit.t);
            let t_maj = sample_tmaj(pos, dir, medium, t_max, lambdas, |p, mp, majorant, t_maj| {
                let null = majorant - mp.absorption - mp.scattering;
                let pdf = t_maj.x * majorant.x;
                t_ray *= t_maj * null / pdf;
                r_l *= t_maj * majorant / pdf;
                r_u *= t_maj * null / pdf;

                Continue(())
            });
            t_ray *= t_maj / t_maj.x;
            r_u *= t_maj / t_maj.x;
            r_l *= t_maj / t_maj.x;
        }

        d -= hit.t;
        let offset = hit.geo_normal * (1e-6 * hit.geo_normal.dot(dir).signum());
        pos += hit.t * dir + offset;

        medium = match dir.dot(hit.geo_normal) > 0.0 {
            true => hit.material.exit_medium(),
            false => hit.material.enter_medium(),
        };
    }

    (t_ray, r_u, r_l)
}

fn sample_tmaj(
    pos: DVec3,
    mut dir: DVec3,
    medium: &dyn Medium,
    mut t_max: f64,
    lambdas: DVec4,
    mut cb: impl FnMut(DVec3, MediumProperties, DVec4, DVec4) -> ControlFlow<()>,
) -> DVec4 {
    t_max *= dir.length();
    dir = dir.normalize();

    let majorant = medium.majorant(lambdas);
    if majorant.x == 0.0 {
        if t_max.is_infinite() {
            t_max = f64::MAX;
        }
        return (-t_max * majorant).exp();
    }

    let mut t_min = 0.0;
    loop {
        let t = t_min - (1.0 - thread_rng().gen::<f64>()).ln() / majorant.x;
        if t >= t_max {
            return (-(t_max - t_min) * majorant).exp();
        }

        let p = pos + dir * t;
        let mp = medium.properties(p, dir, lambdas);
        if cb(p, mp, majorant, (-(t - t_min) * majorant).exp()).is_break() {
            return DVec4::ONE;
        }

        t_min = t;
    }
}

fn average(v: DVec4) -> f64 {
    v.element_sum() / 4.0
}
