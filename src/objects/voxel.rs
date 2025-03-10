use std::fs::File;
use std::io::{BufReader, ErrorKind, Read};
use std::path::Path;
use std::sync::Arc;

use glam::{BVec3, DVec3};

use crate::material::MaterialErased;
use crate::Bounds;

use super::{Object, RayHit};

pub struct VoxelOctree {
    materials: Vec<Arc<dyn MaterialErased>>,
    root: Node,
    nodes: Vec<[Node; 8]>,
}

#[derive(Copy, Clone)]
struct Node(u32);

#[derive(Debug)]
enum NodeKind {
    Empty,
    Material(usize),
    Internal(usize),
}

impl Node {
    fn get(self) -> NodeKind {
        if self.0 & (1 << 31) == 0 {
            NodeKind::Internal(self.0 as usize)
        } else if self.0 == !0 {
            NodeKind::Empty
        } else {
            NodeKind::Material(self.0 as usize & !(1 << 31))
        }
    }
}

impl VoxelOctree {
    pub fn test(materials: Vec<Arc<dyn MaterialErased>>) -> Self {
        VoxelOctree {
            materials,
            root: Node(0),
            nodes: vec![[
                Node(1 << 31),
                Node(1 << 31),
                Node(1 << 31),
                Node(1 << 31),
                Node(1 << 31),
                Node(1 << 31),
                Node(1 << 31),
                Node(!0),
            ]],
        }
    }

    pub fn load(path: impl AsRef<Path>, materials: Vec<Arc<dyn MaterialErased>>) -> Self {
        let mut f = BufReader::new(File::open(path).unwrap());
        let mut read_u32 = || -> Option<u32> {
            let mut u32_buf = [0; 4];
            match f.read_exact(&mut u32_buf) {
                Ok(()) => Some(u32::from_le_bytes(u32_buf)),
                Err(e) => match e.kind() {
                    ErrorKind::UnexpectedEof => None,
                    _ => Err(e).unwrap(),
                },
            }
        };

        let num_materials = read_u32().unwrap();
        if materials.len() < num_materials as usize {
            panic!(
                "need {num_materials} materials, but only {} were specified",
                materials.len()
            );
        }

        let root = Node(read_u32().unwrap());

        let mut nodes = vec![];
        while let Some(c1) = read_u32() {
            let mut children = [Node(c1); 8];
            for i in 1..8 {
                children[i] = Node(read_u32().unwrap());
            }
            nodes.push(children);
        }

        VoxelOctree {
            materials,
            root,
            nodes,
        }
    }
}

impl Object for VoxelOctree {
    fn bounds(&self) -> Bounds {
        Bounds {
            min: DVec3::ZERO,
            max: DVec3::ONE,
        }
    }

    fn raycast(&self, origin: DVec3, direction: DVec3, max_t: f64) -> Option<RayHit> {
        let flip = direction.cmplt(DVec3::ZERO);
        let d_sign = direction.signum();
        let direction = DVec3::select(flip, -direction, direction);
        let direction = direction.max(DVec3::splat(f64::EPSILON));
        let origin = DVec3::select(flip, 1.0 - origin, origin);

        let octree_enter = -origin / direction;
        let mut t = octree_enter.max_element().max(0.0);
        let octree_exit = ((1.0 - origin) / direction).min_element();
        if octree_exit < t {
            return None;
        }
        let mut enter_dir = octree_enter.cmpeq(DVec3::splat(t));

        let mut height = 1;
        let mut two_exp_minus_height = 0.5;
        let mut node_stack = [Node(0); 32];
        let mut offset_stack = [DVec3::ZERO; 32];

        node_stack[1] = self.root;

        while height > 0 && t < max_t {
            match node_stack[height].get() {
                NodeKind::Empty => {
                    let exit_coord = offset_stack[height] + 2.0 * two_exp_minus_height - origin;
                    let t_exit = (exit_coord / direction).min_element();

                    t = t_exit;
                    height -= 1;
                    two_exp_minus_height *= 2.0;
                }
                NodeKind::Material(idx) => {
                    assert_ne!(enter_dir, BVec3::FALSE, "{origin:?} {octree_enter:?}");
                    return Some(RayHit {
                        t,
                        normal: DVec3::select(enter_dir, -d_sign, DVec3::ZERO),
                        geo_normal: DVec3::select(enter_dir, -d_sign, DVec3::ZERO),
                        material: &*self.materials[idx],
                    });
                }
                NodeKind::Internal(idx) => {
                    let enter_coord = offset_stack[height] - origin;
                    let exit_coord = offset_stack[height] + 2.0 * two_exp_minus_height - origin;
                    let middle_coord = offset_stack[height] + two_exp_minus_height - origin;

                    let t_exit = (exit_coord / direction).min_element();

                    if t == t_exit {
                        height -= 1;
                        two_exp_minus_height *= 2.0;
                    } else {
                        let t_enter = (enter_coord / direction).max_element();
                        let t_midplanes = middle_coord / direction;
                        let child = t_midplanes.cmple(DVec3::splat(t));

                        if t != t_enter {
                            enter_dir = t_midplanes.cmpeq(DVec3::splat(t));
                        }

                        offset_stack[height + 1] = offset_stack[height]
                            + DVec3::select(child, DVec3::splat(two_exp_minus_height), DVec3::ZERO);
                        node_stack[height + 1] = self.nodes[idx][(child ^ flip).bitmask() as usize];

                        height += 1;
                        two_exp_minus_height *= 0.5;
                    }
                }
            }
        }

        None
    }
}
