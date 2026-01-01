use std::f64::consts::PI;
use std::sync::Arc;
use std::time::Instant;

use glam::{DMat3, DMat4, DQuat, DVec3, EulerRot};

use crate::brdf::*;
use crate::bvh::Bvh;
use crate::light::DistantDiskLight;
use crate::material::Material;
use crate::medium::{
    AtmosphereAerosols, AtmosphereDryAir, CombinedMedium, Medium, TestMedium, Vacuum,
};
use crate::objects::{SetMaterial, Sphere, Transform, Triangle, VoxelOctree};
use crate::phase::Draine;
use crate::scene::Scene;
use crate::spectrum::physical::extraterrestrial_solar_irradiance;
use crate::spectrum::{AmplifiedSpectrum, ConstantSpectrum, PiecewiseLinearSpectrum, ReciprocalSpectrum};
use crate::{material, plymesh, spectrum};

#[allow(unused)]
pub fn load() -> (Scene, DVec3, DMat3, impl Medium) {
    let atmosphere = TestMedium {
        absorption: spectrum::ZERO,
        emission: spectrum::ZERO,
        scattering: PiecewiseLinearSpectrum::from_points(&[(360.0, 0.0), (830.0, 1.0)]),
    };
    let atmosphere = Vacuum;

    let t = Instant::now();
    let (dragon, dragon_bounds) = plymesh::load_plymesh(
        std::fs::File::open("models/dragon_vrip.ply").unwrap(),
        &Material {
            emission: spectrum::ZERO,
            // brdf: LambertianBrdf {
            //     albedo: DVec3::new(1.0, 0.25, 0.25),
            // },
            // brdf: SmoothConductorBrdf::new(material::physical::ior_gold()),
            brdf: DielectricBrdf {
                ior: material::physical::ior_glass(),
            },
            enter_medium: Vacuum,
            exit_medium: atmosphere.clone(),
        },
    )
    .unwrap();
    let (bunny, bunny_bounds) = plymesh::load_plymesh(
        std::fs::File::open("models/bun_zipper.ply").unwrap(),
        &Material {
            emission: spectrum::ZERO,
            // brdf: SmoothConductorBrdf::new(material::physical::ior_gold()),
            brdf: RoughConductorBrdf::new(material::physical::ior_gold(), 0.05),
            enter_medium: Vacuum,
            exit_medium: Vacuum,
        },
    )
    .unwrap();
    // let voxel_world = VoxelOctree::load(
    //     "world.dat",
    //     vec![
    //         Arc::new(Material {
    //             emission: spectrum::ZERO,
    //             brdf: LambertianBrdf {
    //                 albedo: ConstantSpectrum(0.5),
    //             },
    //             enter_medium: (),
    //             exit_medium: (),
    //         }),
    //         Arc::new(Material {
    //             emission: spectrum::ZERO,
    //             brdf: LambertianBrdf {
    //                 albedo: ConstantSpectrum(1.0),
    //             },
    //             enter_medium: (),
    //             exit_medium: (),
    //         }),
    //         Arc::new(Material {
    //             emission: spectrum::ZERO,
    //             brdf: LambertianBrdf {
    //                 albedo: ConstantSpectrum(0.1),
    //             },
    //             enter_medium: (),
    //             exit_medium: (),
    //         }),
    //     ],
    // );
    println!("Took {:.2?} to load models", t.elapsed());

    let cb_dragon = dragon_bounds.centroid().with_y(dragon_bounds.min.y);
    let cb_bunny = bunny_bounds.centroid().with_y(bunny_bounds.min.y);

    let mut scene = Scene::new();

    scene.add(Triangle {
        a: DVec3::new(-10.0, 0.0, -10.0),
        b: DVec3::new(10.0, 0.0, 10.0),
        c: DVec3::new(10.0, 0.0, -10.0),
        a_n: DVec3::Y,
        b_n: DVec3::Y,
        c_n: DVec3::Y,
        material: Material {
            emission: spectrum::ZERO,
            // brdf: RoughConductorBrdf::new(material::physical::ior_silver(), 0.1),
            brdf: LambertianBrdf {
                albedo: ConstantSpectrum(0.3)
            },
            enter_medium: Vacuum,
            exit_medium: Vacuum,
        },
    });
    scene.add(Triangle {
        a: DVec3::new(10.0, 0.0, 10.0),
        b: DVec3::new(-10.0, 0.0, -10.0),
        c: DVec3::new(-10.0, 0.0, 10.0),
        a_n: DVec3::Y,
        b_n: DVec3::Y,
        c_n: DVec3::Y,
        material: Material {
            emission: spectrum::ZERO,
            // brdf: RoughConductorBrdf::new(material::physical::ior_silver(), 0.1),
            brdf: LambertianBrdf {
                albedo: ConstantSpectrum(0.3)
            },
            enter_medium: Vacuum,
            exit_medium: Vacuum,
        },
    });

    let t = Instant::now();
    let dragon = Arc::new(Bvh::build(dragon));
    let bunny = Arc::new(Bvh::build(bunny));
    println!("Took {:.2?} to build BVH", t.elapsed());

    scene.add(SetMaterial {
        material: Material {
            emission: spectrum::ZERO,
            // brdf: Arc::new(LambertianBrdf {
            //     albedo: DVec3::new(0.25, 1.0, 0.25),
            // }),
            // brdf: SmoothConductorBrdf::new(material::physical::ior_gold()),
            brdf: RoughConductorBrdf::new(material::physical::ior_gold(), 0.05),
            // brdf: DielectricBrdf {
            //     ior: ConstantSpectrum(1.5),
            // },
            enter_medium: Vacuum,
            exit_medium: Vacuum,
        },
        obj: Transform::new(
            DMat4::from_scale_rotation_translation(
                DVec3::splat(1.5),
                DQuat::from_axis_angle(DVec3::Y, -1.0),
                DVec3::new(dragon_bounds.max.x - dragon_bounds.min.x, 0.0, 0.0) - 1.5 * cb_dragon,
            ),
            dragon.clone(),
        ),
    });
    scene.add(SetMaterial {
        material: Material {
            emission: spectrum::ZERO,
            // brdf: Arc::new(LambertianBrdf {
            //     albedo: DVec3::new(0.25, 0.25, 1.0),
            // }),
            brdf: SmoothConductorBrdf::new(material::physical::ior_copper()),
            // brdf: RoughConductorBrdf::new(material::physical::ior_copper(), 0.05),
            enter_medium: Vacuum,
            exit_medium: Vacuum,
        },
        obj: Transform::new(
            DMat4::from_scale_rotation_translation(
                DVec3::splat(0.5),
                DQuat::from_axis_angle(DVec3::Y, 1.0),
                DVec3::new((dragon_bounds.min.x - dragon_bounds.max.x) * 0.75, 0.0, 0.0)
                    - 0.5 * cb_dragon,
            ),
            dragon.clone(),
        ),
    });
    // scene.add(Transform::new(DMat4::from_translation(-cb_dragon), dragon));
    scene.add(Transform::new(
        DMat4::from_scale_rotation_translation(
            DVec3::splat(2.0),
            DQuat::IDENTITY,
            DVec3::new(0.0, 0.0, dragon_bounds.min.x - dragon_bounds.max.x) - cb_bunny * 2.0,
        ),
        bunny,
    ));
    scene.add(Sphere {
        origin: DVec3::new(
            -(dragon_bounds.max.x - dragon_bounds.min.x) * 0.7,
            (dragon_bounds.max.z - dragon_bounds.min.z) * 0.3,
            -(dragon_bounds.max.z - dragon_bounds.min.z),
        ),
        radius: (dragon_bounds.max.z - dragon_bounds.min.z) * 0.3,
        material: Material {
            emission: spectrum::ZERO,
            brdf: SmoothConductorBrdf::new(material::physical::ior_silver()),
            // brdf: RoughConductorBrdf::new(material::physical::ior_silver(), 0.1),
            enter_medium: Vacuum,
            exit_medium: Vacuum,
        },
    });
    scene.add(Sphere {
        origin: DVec3::new(
            (dragon_bounds.max.x - dragon_bounds.min.x) * 0.5,
            (dragon_bounds.max.z - dragon_bounds.min.z) * 0.5,
            (dragon_bounds.max.z - dragon_bounds.min.z) * 0.8,
        ),
        radius: (dragon_bounds.max.z - dragon_bounds.min.z) * 0.5,
        material: Material {
            emission: spectrum::ZERO,
            brdf: DielectricBrdf { ior: material::physical::ior_glass() },
            enter_medium: Vacuum,
            // brdf: ThinDielectricBrdf {
            //     ior: material::physical::ior_glass(),
            // },
            // enter_medium: TestMedium {
            //     absorption: spectrum::ZERO,
            //     emission: spectrum::ZERO,
            //     scattering: spectrum::ConstantSpectrum(10.0),
            // },
            exit_medium: atmosphere.clone(),
        },
    });
    // scene.add(Sphere {
    //     origin: DVec3::ZERO,
    //     radius: 1.0,
    //     material: Material {
    //         emission: spectrum::ZERO,
    //         brdf: (),
    //         enter_medium: atmosphere.clone(),
    //         exit_medium: Vacuum,
    //     },
    // });

    // scene.add(Sphere {
    //     origin: (dragon_bounds.max.z - dragon_bounds.min.z) * 0.5 * DVec3::Y,
    //     radius: (dragon_bounds.max.z - dragon_bounds.min.z) * 0.5,
    //     material: Material {
    //         emission: spectrum::ZERO,
    //         brdf: RoughConductorBrdf::new(material::physical::ior_silver(), 0.1),
    //         // brdf: SmoothConductorBrdf::new(material::physical::ior_silver()),
    //         // brdf: LambertianBrdf {
    //         //     albedo: ConstantSpectrum(1.0),
    //         // },
    //         enter_medium: (),
    //         exit_medium: (),
    //     },
    // });

    // scene.add(Transform::new(
    //     DMat4::from_scale(DVec3::splat(
    //         (dragon_bounds.max.z - dragon_bounds.min.z) * 1.5,
    //     )),
    //     voxel_world,
    // ));

    scene.add(SetMaterial {
        material: Material {
            emission: spectrum::ZERO,
            // // brdf: Arc::new(LambertianBrdf {
            // //     albedo: DVec3::new(0.25, 1.0, 0.25),
            // // }),
            // brdf: SmoothConductorBrdf::new(material::physical::ior_gold()),
            brdf: RoughConductorBrdf::new(material::physical::ior_copper(), 0.05),
            // brdf: DielectricBrdf {
            //     ior: material::physical::ior_glass(),
            // },
            enter_medium: Vacuum,
            exit_medium: Vacuum,
        },
        obj: Transform::new(
            DMat4::from_scale_rotation_translation(
                DVec3::splat(1.0),
                DQuat::from_axis_angle(DVec3::Y, 0.3),
                (dragon_bounds.max.z - dragon_bounds.min.z) * DVec3::new(-0.3, 0.0, 0.0) - 1.0 * cb_dragon,
            ),
            dragon.clone(),
        ),
    });

    scene.add_light(DistantDiskLight::from_irradiance(
        DVec3::new(-1.0, 0.5, -0.3).normalize(),
        10.0f64.to_radians().cos(),
        AmplifiedSpectrum {
            factor: 1000.0,
            s: spectrum::physical::cie_d65_1nit(),
        },
    ));

    let scale = (dragon_bounds.max - dragon_bounds.min).length() * 0.8;

    let looking = DMat3::from_euler(EulerRot::YXZ, PI-0.4, 0.4, 0.0) * DMat3::from_diagonal(DVec3::new(-1.0, 1.0, 1.0));
    let camera = scale * (looking * DVec3::new(0.0, 0.0, -2.0) + DVec3::new(0.0, 0.5, 0.0));

    (scene, camera, looking, atmosphere)
}

#[allow(unused)]
pub fn simple_volume_scene() -> (Scene, DVec3, DMat3, impl Medium) {
    let mut scene = Scene::new();

    scene.add(Triangle {
        a: DVec3::new(-100.0, -1.5, -100.0),
        b: DVec3::new(100.0, -1.5, 100.0),
        c: DVec3::new(100.0, -1.5, -100.0),
        a_n: DVec3::Y,
        b_n: DVec3::Y,
        c_n: DVec3::Y,
        material: Material {
            emission: spectrum::ZERO,
            brdf: LambertianBrdf {
                albedo: ConstantSpectrum(0.5),
            },
            enter_medium: Vacuum,
            exit_medium: Vacuum,
        },
    });
    scene.add(Triangle {
        a: DVec3::new(100.0, -1.5, 100.0),
        b: DVec3::new(-100.0, -1.5, -100.0),
        c: DVec3::new(-100.0, -1.5, 100.0),
        a_n: DVec3::Y,
        b_n: DVec3::Y,
        c_n: DVec3::Y,
        material: Material {
            emission: spectrum::ZERO,
            brdf: LambertianBrdf {
                albedo: ConstantSpectrum(0.5),
            },
            enter_medium: Vacuum,
            exit_medium: Vacuum,
        },
    });
    const STRENGTH: f64 = 500.0;
    scene.add(Sphere {
        origin: DVec3::ZERO,
        radius: 1.0,
        material: Material {
            emission: spectrum::ZERO,
            // brdf: (),
            brdf: LambertianBrdf {
                albedo: PiecewiseLinearSpectrum::from_points(&[(360.0, 0.0), (830.0, 1.0)]),
            },
            // brdf: ThinDielectricBrdf {
            //     ior: ConstantSpectrum(1.3),
            // },
            enter_medium: TestMedium {
                absorption: PiecewiseLinearSpectrum::from_points(&[
                    (360.0, STRENGTH),
                    (830.0, 0.0),
                ]),
                emission: spectrum::ZERO,
                scattering: PiecewiseLinearSpectrum::from_points(&[
                    (360.0, 0.0),
                    (830.0, STRENGTH),
                ]),
                // scattering: spectrum::ZERO,
            },
            exit_medium: Vacuum,
        },
    });

    scene.add_light(DistantDiskLight::from_irradiance(
        DVec3::new(-1.0, 0.5, -0.3).normalize(),
        0.268f64.to_radians().cos(),
        AmplifiedSpectrum {
            // factor: 1.0,
            factor: 10000.0,
            // factor: 50000.0,
            s: spectrum::physical::cie_d65_1nit(),
        },
    ));
    // scene.add_light(DistantDiskLight {
    //     emission: AmplifiedSpectrum {
    //         // factor: 1.0,
    //         factor: 25.0,
    //         // factor: 50000.0,
    //         s: spectrum::physical::cie_d65_1nit(),
    //     },
    //     dir: DVec3::new(-1.0, 0.5, -0.3).normalize(),
    //     // cos_radius: -1.0,
    //     cos_radius: 10.0f64.to_radians().cos(),
    //     // cos_radius: 0.268f64.to_radians().cos(),
    // });

    let looking = DMat3::from_euler(EulerRot::YXZ, 0.3, 0.5, 0.0);
    let camera = looking * DVec3::new(0.0, 0.0, -3.0);

    (scene, camera, looking, Vacuum)
}

pub fn atmosphere_scene(time: f64, altitude: f64) -> (Scene, DVec3, DMat3, impl Medium) {
    let mut scene = Scene::new();

    const PLANET_RADIUS: f64 = 6371000.0;
    const ATMOSPHERE_HEIGHT: f64 = 50_000.0;

    let atmosphere = CombinedMedium {
        m1: AtmosphereDryAir {
            origin: DVec3::new(0.0, -PLANET_RADIUS, 0.0),
            sea_level: PLANET_RADIUS,
            height_scale: 8000.0,
            sea_level_air_density: 2.504e25,
            ozone_start_altitude: 12_000.0,
            ozone_peak_altitude: 32_000.0,
            ozone_peak_concentration: 5e-6,
            ozone_height_scale: 15_000.0,
            min_ozone_concentration: 10e-9,
        },
        m2: AtmosphereAerosols {
            origin: DVec3::new(0.0, -PLANET_RADIUS, 0.0),
            sea_level: PLANET_RADIUS,
            height_scale: 1_200.0,
            sea_level_density: 1e-5,
            max_height: 20_000.0,
            phase: Draine {
                alpha: 1.0,
                g: 0.85,
            },
        },
    };

    scene.add(Sphere {
        origin: DVec3::new(0.0, -PLANET_RADIUS, 0.0),
        radius: PLANET_RADIUS,
        material: Material {
            emission: spectrum::ZERO,
            brdf: LambertianBrdf {
                albedo: ConstantSpectrum(0.3),
            },
            enter_medium: (),
            exit_medium: (),
        },
    });

    scene.add(Sphere {
        origin: DVec3::new(0.0, -PLANET_RADIUS, 0.0),
        radius: PLANET_RADIUS + ATMOSPHERE_HEIGHT,
        material: Material {
            emission: spectrum::ZERO,
            brdf: (),
            enter_medium: atmosphere.clone(),
            exit_medium: Vacuum,
        },
    });

    scene.add(Transform::new(
        DMat4::from_scale_rotation_translation(
            DVec3::new(8192.0, 8192.0, -8192.0),
            DQuat::IDENTITY,
            DVec3::new(-4096.0, 0.0, 4096.0),
        ),
        VoxelOctree::load(
            "world.dat",
            vec![Arc::new(Material {
                emission: spectrum::ZERO,
                brdf: LambertianBrdf {
                    albedo: ConstantSpectrum(0.3),
                },
                enter_medium: (),
                exit_medium: (),
            })],
        ),
    ));

    let axis_tilt = 0.40909;
    let time_of_year: f64 = 1.1;//PI/2.0; //0.086; //0.3068;
    let latitude: f64 = -37.8f64.to_radians();

    let celestial_pole = DVec3::new(0.0, latitude.sin(), latitude.cos());
    let celestial_sun_angle = PI / 2.0 - time_of_year.cos() * axis_tilt;
    let local_sun_angle = celestial_sun_angle + latitude;
    let sun_noon_pos = DVec3::new(0.0, local_sun_angle.sin(), local_sun_angle.cos());

    scene.add_light(DistantDiskLight::from_irradiance(
        dbg!(DQuat::from_axis_angle(celestial_pole, (time - 12.0) / 12.0 * PI) * sun_noon_pos),
        0.268f64.to_radians().cos(),
        extraterrestrial_solar_irradiance(),
    ));

    // scene.add_light(DistantDiskLight {
    //     emission: AmplifiedSpectrum {
    //         factor: 0.06,
    //         s: extraterrestrial_solar_irradiance(),
    //     },
    //     dir: DQuat::from_axis_angle(celestial_pole, time / 12.0 * PI) * sun_noon_pos,
    //     cos_radius: 0.268f64.to_radians().cos(),
    // });

    let looking = DMat3::from_euler(EulerRot::YXZ, 1.2, 0.0, 0.0);
    let camera = DVec3::new(-3701.4, 455.5+altitude, 1999.2);
    // let camera = DVec3::new(0.0, altitude, 0.0);

    // let looking = DMat3::from_euler(EulerRot::YXZ, -10.914795, 0.08500001, 0.0);
    // let camera = DVec3::new(0.23460676, 0.01720072, 0.32080263);

    // let looking = DMat3::from_euler(EulerRot::YXZ, -4.639654, 0.3849998, 0.0);
    // let camera = DVec3::new(0.87864524, 0.27254108, 0.158398);

    let atmosphere = match camera.y < ATMOSPHERE_HEIGHT {
        true => Box::new(atmosphere) as Box<dyn Medium>,
        false => Box::new(Vacuum),
    };

    (scene, camera, looking, atmosphere)
}
