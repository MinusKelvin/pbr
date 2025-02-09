use std::sync::Arc;
use std::time::Instant;

use glam::{DMat3, DMat4, DQuat, DVec3, EulerRot};

use crate::brdf::{DielectricBrdf, LambertianBrdf, SmoothConductorBrdf};
use crate::bvh::Bvh;
use crate::light::DistantDiskLight;
use crate::material::Material;
use crate::scene::Scene;
use crate::{material, plymesh, spectrum};
use crate::objects::{SetMaterial, Sphere, Transform, Triangle};
use crate::spectrum::{AmplifiedSpectrum, ConstantSpectrum};

pub fn load() -> (Scene, DVec3, DMat3) {
    let t = Instant::now();
    let (dragon, dragon_bounds) = plymesh::load_plymesh(
        std::fs::File::open("models/dragon_vrip.ply").unwrap(),
        &Material {
            emission: spectrum::ZERO,
            // brdf: LambertianBrdf {
            //     albedo: DVec3::new(1.0, 0.25, 0.25),
            // },
            // brdf: SmoothConductorBrdf::new(material::physical::ior_gold()),
            brdf: DielectricBrdf { ior: material::physical::ior_glass() },
        },
    )
    .unwrap();
    let (bunny, bunny_bounds) = plymesh::load_plymesh(
        std::fs::File::open("models/bun_zipper.ply").unwrap(),
        &Material {
            emission: spectrum::ZERO,
            brdf: SmoothConductorBrdf::new(material::physical::ior_gold()),
        },
    )
    .unwrap();
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
            brdf: LambertianBrdf {
                albedo: ConstantSpectrum(0.5),
            },
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
            brdf: LambertianBrdf {
                albedo: ConstantSpectrum(0.5),
            },
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
            brdf: SmoothConductorBrdf::new(material::physical::ior_gold()),
            // brdf: DielectricBrdf {
            //     ior: ConstantSpectrum(1.5),
            // },
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
    scene.add(Transform::new(DMat4::from_translation(-cb_dragon), dragon));
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
            // brdf: SmoothConductorBrdf::new(material::physical::ior_silver()),
            brdf: DielectricBrdf { ior: material::physical::ior_glass() },
        },
    });

    scene.add_light(DistantDiskLight {
        emission: AmplifiedSpectrum {
            factor: 25.0,
            // factor: 50000.0,
            s: spectrum::physical::cie_d65(),
        },
        dir: DVec3::new(-1.0, 0.5, -0.3).normalize(),
        cos_radius: 10.0f64.to_radians().cos(),
        // cos_radius: 0.268f64.to_radians().cos(),
    });

    let scale = (dragon_bounds.max - dragon_bounds.min).length() * 0.8;

    let looking = DMat3::from_euler(EulerRot::YXZ, -0.3, -0.4, 0.0);
    let camera = scale * (looking * DVec3::Z + DVec3::new(0.0, 0.5, 0.0));

    (scene, camera, looking)
}
