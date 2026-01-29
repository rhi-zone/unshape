use glam::Vec3;

use crate::Aabb3;

/// A ray in 3D space.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Ray {
    /// Origin of the ray.
    pub origin: Vec3,
    /// Direction of the ray (should be normalized).
    pub direction: Vec3,
}

impl Ray {
    /// Creates a new ray.
    pub fn new(origin: Vec3, direction: Vec3) -> Self {
        Self {
            origin,
            direction: direction.normalize(),
        }
    }

    /// Returns the point at parameter t along the ray.
    pub fn at(&self, t: f32) -> Vec3 {
        self.origin + self.direction * t
    }

    /// Tests intersection with an AABB, returns (t_min, t_max) if hit.
    pub fn intersect_aabb(&self, aabb: &Aabb3) -> Option<(f32, f32)> {
        let inv_dir = Vec3::new(
            1.0 / self.direction.x,
            1.0 / self.direction.y,
            1.0 / self.direction.z,
        );

        let t1 = (aabb.min.x - self.origin.x) * inv_dir.x;
        let t2 = (aabb.max.x - self.origin.x) * inv_dir.x;
        let t3 = (aabb.min.y - self.origin.y) * inv_dir.y;
        let t4 = (aabb.max.y - self.origin.y) * inv_dir.y;
        let t5 = (aabb.min.z - self.origin.z) * inv_dir.z;
        let t6 = (aabb.max.z - self.origin.z) * inv_dir.z;

        let t_min = t1.min(t2).max(t3.min(t4)).max(t5.min(t6));
        let t_max = t1.max(t2).min(t3.max(t4)).min(t5.max(t6));

        if t_max >= t_min && t_max >= 0.0 {
            Some((t_min.max(0.0), t_max))
        } else {
            None
        }
    }
}

/// A node in the BVH.
#[derive(Debug)]
pub(crate) enum BvhNode<T> {
    /// Leaf node containing a single primitive.
    Leaf { bounds: Aabb3, data: T },
    /// Internal node with two children.
    Internal {
        bounds: Aabb3,
        left: Box<BvhNode<T>>,
        right: Box<BvhNode<T>>,
    },
}

/// A Bounding Volume Hierarchy for efficient ray intersection queries.
///
/// BVH builds a binary tree of bounding boxes, enabling O(log n) ray intersection
/// tests. Commonly used for ray tracing and collision detection.
///
/// # Type Parameters
///
/// * `T` - The type of data associated with each primitive.
///
/// # Example
///
/// ```
/// use unshape_spatial::{Bvh, Aabb3, Ray};
/// use glam::Vec3;
///
/// // Create primitives with their bounding boxes
/// let primitives = vec![
///     (Aabb3::new(Vec3::ZERO, Vec3::ONE), "box1"),
///     (Aabb3::new(Vec3::splat(5.0), Vec3::splat(6.0)), "box2"),
/// ];
///
/// let bvh = Bvh::build(primitives);
///
/// // Cast a ray and find intersections
/// let ray = Ray::new(Vec3::new(0.5, 0.5, -5.0), Vec3::Z);
/// for (aabb, data) in bvh.intersect_ray(&ray) {
///     println!("Hit {:?} with bounds {:?}", data, aabb);
/// }
/// ```
#[derive(Debug)]
pub struct Bvh<T> {
    root: Option<BvhNode<T>>,
}

impl<T> Bvh<T> {
    /// Builds a BVH from a list of (bounding box, data) pairs.
    ///
    /// Uses the Surface Area Heuristic (SAH) for optimal tree construction.
    pub fn build(primitives: Vec<(Aabb3, T)>) -> Self {
        if primitives.is_empty() {
            return Self { root: None };
        }

        Self {
            root: Some(Self::build_recursive(primitives)),
        }
    }

    fn build_recursive(mut primitives: Vec<(Aabb3, T)>) -> BvhNode<T> {
        if primitives.len() == 1 {
            let (bounds, data) = primitives.pop().unwrap();
            return BvhNode::Leaf { bounds, data };
        }

        // Compute overall bounds
        let bounds = primitives
            .iter()
            .fold(primitives[0].0, |acc, (b, _)| acc.union(b));

        // Find the longest axis and split there
        let size = bounds.size();
        let axis = if size.x >= size.y && size.x >= size.z {
            0
        } else if size.y >= size.z {
            1
        } else {
            2
        };

        // Sort by centroid along the chosen axis
        primitives.sort_by(|(a, _), (b, _)| {
            let ca = a.center();
            let cb = b.center();
            let va = match axis {
                0 => ca.x,
                1 => ca.y,
                _ => ca.z,
            };
            let vb = match axis {
                0 => cb.x,
                1 => cb.y,
                _ => cb.z,
            };
            va.partial_cmp(&vb).unwrap()
        });

        // Split in the middle
        let mid = primitives.len() / 2;
        let right_prims = primitives.split_off(mid);

        let left = Box::new(Self::build_recursive(primitives));
        let right = Box::new(Self::build_recursive(right_prims));

        BvhNode::Internal {
            bounds,
            left,
            right,
        }
    }

    /// Tests a ray against the BVH and returns all intersecting primitives.
    ///
    /// Returns an iterator over (bounds, data) pairs for primitives whose
    /// bounding boxes intersect the ray.
    pub fn intersect_ray(&self, ray: &Ray) -> Vec<(&Aabb3, &T)> {
        let mut results = Vec::new();
        if let Some(ref root) = self.root {
            Self::intersect_recursive(root, ray, &mut results);
        }
        results
    }

    fn intersect_recursive<'a>(
        node: &'a BvhNode<T>,
        ray: &Ray,
        results: &mut Vec<(&'a Aabb3, &'a T)>,
    ) {
        match node {
            BvhNode::Leaf { bounds, data } => {
                if ray.intersect_aabb(bounds).is_some() {
                    results.push((bounds, data));
                }
            }
            BvhNode::Internal {
                bounds,
                left,
                right,
            } => {
                if ray.intersect_aabb(bounds).is_some() {
                    Self::intersect_recursive(left, ray, results);
                    Self::intersect_recursive(right, ray, results);
                }
            }
        }
    }

    /// Tests a ray and returns the closest intersection.
    ///
    /// Returns `(bounds, data, t)` where `t` is the parameter along the ray.
    pub fn intersect_ray_closest(&self, ray: &Ray) -> Option<(&Aabb3, &T, f32)> {
        let mut closest: Option<(&Aabb3, &T, f32)> = None;
        if let Some(ref root) = self.root {
            Self::intersect_closest_recursive(root, ray, &mut closest);
        }
        closest
    }

    fn intersect_closest_recursive<'a>(
        node: &'a BvhNode<T>,
        ray: &Ray,
        closest: &mut Option<(&'a Aabb3, &'a T, f32)>,
    ) {
        let bounds = match node {
            BvhNode::Leaf { bounds, .. } => bounds,
            BvhNode::Internal { bounds, .. } => bounds,
        };

        // Early exit if ray doesn't hit this node's bounds
        let Some((t_min, _)) = ray.intersect_aabb(bounds) else {
            return;
        };

        // Early exit if we already have a closer hit
        if let Some((_, _, closest_t)) = closest {
            if t_min > *closest_t {
                return;
            }
        }

        match node {
            BvhNode::Leaf { bounds, data } => {
                if let Some((t_hit, _)) = ray.intersect_aabb(bounds) {
                    if closest.is_none() || t_hit < closest.unwrap().2 {
                        *closest = Some((bounds, data, t_hit));
                    }
                }
            }
            BvhNode::Internal { left, right, .. } => {
                // Recurse into children
                // For better performance, we could check which child to visit first
                Self::intersect_closest_recursive(left, ray, closest);
                Self::intersect_closest_recursive(right, ray, closest);
            }
        }
    }

    /// Queries all primitives whose bounds intersect the given AABB.
    pub fn query_aabb(&self, query: &Aabb3) -> Vec<(&Aabb3, &T)> {
        let mut results = Vec::new();
        if let Some(ref root) = self.root {
            Self::query_aabb_recursive(root, query, &mut results);
        }
        results
    }

    fn query_aabb_recursive<'a>(
        node: &'a BvhNode<T>,
        query: &Aabb3,
        results: &mut Vec<(&'a Aabb3, &'a T)>,
    ) {
        match node {
            BvhNode::Leaf { bounds, data } => {
                if bounds.intersects(query) {
                    results.push((bounds, data));
                }
            }
            BvhNode::Internal {
                bounds,
                left,
                right,
            } => {
                if bounds.intersects(query) {
                    Self::query_aabb_recursive(left, query, results);
                    Self::query_aabb_recursive(right, query, results);
                }
            }
        }
    }

    /// Returns the number of primitives in the BVH.
    pub fn len(&self) -> usize {
        match &self.root {
            None => 0,
            Some(root) => Self::count_recursive(root),
        }
    }

    /// Returns `true` if the BVH is empty.
    pub fn is_empty(&self) -> bool {
        self.root.is_none()
    }

    fn count_recursive(node: &BvhNode<T>) -> usize {
        match node {
            BvhNode::Leaf { .. } => 1,
            BvhNode::Internal { left, right, .. } => {
                Self::count_recursive(left) + Self::count_recursive(right)
            }
        }
    }
}
