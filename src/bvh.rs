use core::f64;

use glam::DVec3;
use ordered_float::OrderedFloat;

use crate::objects::{Object, RayHit};
use crate::Bounds;

pub struct Bvh {
    objs: Vec<Box<dyn Object + Sync>>,
    root: BvhNode,
}

enum BvhChildren {
    Leaf(usize),
    Node(Box<[BvhNode; 2]>),
}

struct BvhNode {
    bounds: Bounds,
    children: BvhChildren,
}

impl Bvh {
    pub fn build(objects: Vec<Box<dyn Object + Sync>>) -> Self {
        let root =
            build_bvh_node(&mut objects.iter().map(|b| &**b).enumerate().collect::<Vec<_>>());
        Bvh {
            objs: objects,
            root,
        }
    }
}

fn build_bvh_node(objs: &mut [(usize, &(dyn Object + Sync))]) -> BvhNode {
    let (bounds, centroid_bounds) = objs
        .iter()
        .map(|(_, obj)| obj.bounds())
        .map(|b| (b, Bounds::point(b.centroid())))
        .reduce(|(a1, a2), (b1, b2)| (a1.union(b1), a2.union(b2)))
        .unwrap();

    if objs.len() == 1 {
        return BvhNode {
            bounds,
            children: BvhChildren::Leaf(objs[0].0),
        };
    }

    let size = centroid_bounds.max - centroid_bounds.min;
    let dim = match () {
        _ if size.x >= size.y && size.x >= size.z => 0,
        _ if size.y >= size.z => 1,
        _ => 2,
    };

    let split = objs.len() / 2;
    objs.select_nth_unstable_by_key(split, |(_, o)| OrderedFloat(o.bounds().centroid()[dim]));

    let (left, right) = objs.split_at_mut(split);

    let left = build_bvh_node(left);
    let right = build_bvh_node(right);

    BvhNode {
        bounds,
        children: BvhChildren::Node(Box::new([left, right])),
    }
}

impl Object for Bvh {
    fn bounds(&self) -> Bounds {
        self.root.bounds
    }

    fn raycast(&self, origin: DVec3, direction: DVec3) -> Option<RayHit> {
        let mut stack = vec![&self.root];
        let mut closest = None;
        let mut t_hit = f64::INFINITY;

        while let Some(node) = stack.pop() {
            let Some((_, _)) = node.bounds.ray_intersect(origin, direction, t_hit) else {
                continue;
            };

            match node.children {
                BvhChildren::Leaf(index) => {
                    if let Some(hit) = self.objs[index].raycast(origin, direction) {
                        if hit.t < t_hit - hit.normal.dot(direction) * 1.0e-12 {
                            t_hit = hit.t;
                            closest = Some(hit);
                        }
                    }
                }
                BvhChildren::Node(ref children) => {
                    let hits = children
                        .each_ref()
                        .map(|n| n.bounds.ray_intersect(origin, direction, t_hit));
                    match hits {
                        [None, None] => continue,
                        [Some(_), None] => stack.push(&children[0]),
                        [None, Some(_)] => stack.push(&children[1]),
                        [Some(_), Some(_)] => {
                            stack.push(&children[0]);
                            stack.push(&children[1]);
                        }
                    }
                }
            }
        }

        closest
    }
}
