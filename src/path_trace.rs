use glam::{DVec3, DVec4, Vec4Swizzles};
use rand::prelude::*;

use crate::medium::Medium;
use crate::scene::Scene;

pub fn path_trace(
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
            if hit.is_none() {
                panic!("can't exit participating medium?");
            }
            if !secondary_terminated {
                throughput.x *= 4.0;
                throughput.y = 0.0;
                throughput.z = 0.0;
                throughput.w = 0.0;
                secondary_terminated = true;
            }

            let majorant = match secondary_terminated {
                true => medium.majorant(lambdas.xxxx()),
                false => medium.majorant(lambdas),
            }
            .max_element();
            let mut t = 0.0;
            loop {
                let dt = -(1.0 - thread_rng().gen::<f64>()).ln() / majorant;
                t += dt;
                if t >= d {
                    break;
                }

                let p = pos + t * dir;
                let mp = medium.properties(pos, dir, lambdas);
                let pr_absorption = mp.absorption / majorant;
                let pr_scattering = mp.scattering / majorant;
                let pr_null = 1.0 - pr_absorption - pr_scattering;

                let rng: f64 = thread_rng().gen();
                if rng < pr_absorption.x {
                    if specular_bounce {
                        radiance += throughput * scene.light_emission(pos, dir, lambdas, t);
                    }

                    throughput *= pr_absorption / pr_absorption.x;
                    radiance += throughput * mp.emission;

                    break 'mainloop;
                } else if rng < pr_absorption.x + pr_scattering.x {
                    if specular_bounce {
                        radiance += throughput * scene.light_emission(pos, dir, lambdas, t);
                    }

                    throughput *= pr_scattering / pr_scattering.x;

                    if let Some((light, pdf)) = scene.sample_light(p, lambdas, thread_rng().gen()) {
                        let sample = light.sample(p, lambdas, thread_rng().gen());

                        let tp_f = throughput
                            * medium.phase(p, sample.dir, dir, lambdas)
                            * sample.emission;

                        if tp_f != DVec4::ZERO {
                            let transmittance = transmittance(
                                scene,
                                p,
                                sample.dir,
                                lambdas,
                                secondary_terminated,
                                medium,
                                sample.dist,
                            );
                            radiance += tp_f * transmittance / (pdf * sample.pdf);
                        }
                    }

                    let new_dir = medium.sample_phase(p, dir, lambdas, thread_rng().gen());
                    let new_dir_pdf = medium.pdf_phase(p, new_dir, dir, lambdas);

                    throughput *= medium.phase(p, new_dir, dir, lambdas) / new_dir_pdf;
                    pos = p;
                    dir = new_dir;
                    specular_bounce = false;

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
            // radiance += throughput * spectrum::physical::cie_d65().sample_multi(lambdas) * 0.03;
            break;
        };

        let hit_pos = pos + dir * hit.t;

        radiance += throughput * hit.material.emission_sample(lambdas);

        let old_dir = dir;

        if let Some(brdf) = hit.material.brdf() {
            if let Some((light, pdf)) = scene.sample_light(hit_pos, lambdas, thread_rng().gen()) {
                let sample = light.sample(pos, lambdas, thread_rng().gen());

                let tp_f = throughput
                    * brdf.f(sample.dir, dir, hit.normal, lambdas)
                    * sample.emission
                    * sample.dir.dot(hit.normal).abs();

                if tp_f != DVec4::ZERO {
                    let offset = hit.geo_normal * (1e-6 * hit.geo_normal.dot(sample.dir).signum());
                    let transmittance = transmittance(
                        scene,
                        hit_pos + offset,
                        sample.dir,
                        lambdas,
                        secondary_terminated,
                        medium,
                        sample.dist,
                    );
                    radiance += tp_f * transmittance / (pdf * sample.pdf);
                }
            }

            let sample = brdf.sample(dir, hit.normal, lambdas, thread_rng().gen());

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

            let cos_theta = sample.dir.dot(hit.normal).abs();
            throughput *= sample.f * cos_theta / sample.pdf;

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

        if throughput.element_sum() < 0.5 || bounces > 20 {
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

pub fn transmittance<'a>(
    scene: &'a Scene,
    mut pos: DVec3,
    dir: DVec3,
    lambdas: DVec4,
    secondary_terminated: bool,
    mut medium: &'a dyn Medium,
    mut d: f64,
) -> DVec4 {
    let mut transmittance = DVec4::ONE;
    while d > 0.0 {
        let Some(hit) = scene.raycast(pos, dir, d) else {
            if medium.participating() {
                panic!("can't exit participating medium?");
            }
            break;
        };

        if d > hit.t && hit.material.brdf().is_some() {
            return DVec4::ZERO;
        }

        if medium.participating() {
            let majorant = match secondary_terminated {
                true => medium.majorant(lambdas.xxxx()),
                false => medium.majorant(lambdas),
            }
            .max_element();
            let mut t = 0.0;
            loop {
                let dt = -(1.0 - thread_rng().gen::<f64>()).ln() / majorant;
                t += dt;
                if t >= hit.t {
                    break;
                }

                let p = pos + t * dir;
                let mp = medium.properties(p, dir, lambdas);
                let pr_attenuation = (mp.absorption + mp.scattering) / majorant;
                let pr_null = 1.0 - pr_attenuation;
                transmittance *= pr_null;
            }
        }

        d -= hit.t;
        let offset = hit.geo_normal * (1e-6 * hit.geo_normal.dot(dir).signum());
        pos += hit.t * dir + offset;

        medium = match dir.dot(hit.geo_normal) > 0.0 {
            true => hit.material.exit_medium(),
            false => hit.material.enter_medium(),
        };
    }
    transmittance
}
