//! Collision detection functions.
//!
//! Narrow-phase collision tests between pairs of primitive shapes
//! (sphere, box, plane), returning `Contact` points with normals and depths.

use glam::{Mat3, Quat, Vec3};

/// A contact point between two bodies.
#[derive(Clone, Debug)]
pub struct Contact {
    /// Index of first body.
    pub body_a: usize,
    /// Index of second body.
    pub body_b: usize,
    /// Contact point in world space.
    pub point: Vec3,
    /// Contact normal (from A to B).
    pub normal: Vec3,
    /// Penetration depth.
    pub depth: f32,
}

impl Contact {
    /// Flip a contact to swap bodies and invert normal.
    ///
    /// Used to handle symmetric collision pairs (e.g., sphere-plane vs plane-sphere).
    #[inline]
    pub fn flip(mut self) -> Self {
        self.normal = -self.normal;
        std::mem::swap(&mut self.body_a, &mut self.body_b);
        self
    }
}

/// Test sphere-sphere collision.
pub fn sphere_sphere(
    a: usize,
    b: usize,
    pos_a: Vec3,
    radius_a: f32,
    pos_b: Vec3,
    radius_b: f32,
) -> Option<Contact> {
    let d = pos_b - pos_a;
    let dist_sq = d.length_squared();
    let radius_sum = radius_a + radius_b;

    if dist_sq < radius_sum * radius_sum {
        let dist = dist_sq.sqrt();
        // Normal points from A toward B
        let normal = if dist > 0.0 { d / dist } else { Vec3::Y };
        let depth = radius_sum - dist;
        let point = pos_a + normal * radius_a;

        Some(Contact {
            body_a: a,
            body_b: b,
            point,
            normal,
            depth,
        })
    } else {
        None
    }
}

/// Test sphere-plane collision.
pub fn sphere_plane(
    sphere_idx: usize,
    plane_idx: usize,
    sphere_pos: Vec3,
    radius: f32,
    plane_normal: Vec3,
    plane_dist: f32,
) -> Option<Contact> {
    let dist = sphere_pos.dot(plane_normal) - plane_dist;

    if dist < radius {
        let depth = radius - dist;
        let point = sphere_pos - plane_normal * dist;

        Some(Contact {
            body_a: sphere_idx,
            body_b: plane_idx,
            point,
            normal: plane_normal,
            depth,
        })
    } else {
        None
    }
}

/// Test box-plane collision.
pub fn box_plane(
    box_idx: usize,
    plane_idx: usize,
    box_pos: Vec3,
    box_rot: Quat,
    half_extents: Vec3,
    plane_normal: Vec3,
    plane_dist: f32,
) -> Option<Contact> {
    // Get box axes
    let rot = Mat3::from_quat(box_rot);
    let axes = [
        rot.col(0) * half_extents.x,
        rot.col(1) * half_extents.y,
        rot.col(2) * half_extents.z,
    ];

    // Find the vertex furthest in the direction of -normal
    let mut min_dist = f32::MAX;
    let mut min_point = Vec3::ZERO;

    for sx in [-1.0_f32, 1.0] {
        for sy in [-1.0_f32, 1.0] {
            for sz in [-1.0_f32, 1.0] {
                let vertex = box_pos + axes[0] * sx + axes[1] * sy + axes[2] * sz;
                let dist = vertex.dot(plane_normal) - plane_dist;
                if dist < min_dist {
                    min_dist = dist;
                    min_point = vertex;
                }
            }
        }
    }

    if min_dist < 0.0 {
        Some(Contact {
            body_a: box_idx,
            body_b: plane_idx,
            point: min_point,
            normal: plane_normal,
            depth: -min_dist,
        })
    } else {
        None
    }
}

/// Test sphere-box collision.
pub fn sphere_box(
    sphere_idx: usize,
    box_idx: usize,
    sphere_pos: Vec3,
    radius: f32,
    box_pos: Vec3,
    box_rot: Quat,
    half_extents: Vec3,
) -> Option<Contact> {
    // Transform sphere to box local space
    let inv_rot = box_rot.inverse();
    let local_pos = inv_rot * (sphere_pos - box_pos);

    // Find closest point on box
    let clamped = Vec3::new(
        local_pos.x.clamp(-half_extents.x, half_extents.x),
        local_pos.y.clamp(-half_extents.y, half_extents.y),
        local_pos.z.clamp(-half_extents.z, half_extents.z),
    );

    let diff = local_pos - clamped;
    let dist_sq = diff.length_squared();

    if dist_sq < radius * radius {
        let dist = dist_sq.sqrt();
        let local_normal = if dist > 0.0 {
            diff / dist
        } else {
            // Sphere center inside box - push out along shortest axis
            let penetrations = half_extents - local_pos.abs();
            if penetrations.x < penetrations.y && penetrations.x < penetrations.z {
                Vec3::X * local_pos.x.signum()
            } else if penetrations.y < penetrations.z {
                Vec3::Y * local_pos.y.signum()
            } else {
                Vec3::Z * local_pos.z.signum()
            }
        };

        let normal = box_rot * local_normal;
        let point = box_pos + box_rot * clamped;
        let depth = radius - dist;

        Some(Contact {
            body_a: sphere_idx,
            body_b: box_idx,
            point,
            normal,
            depth,
        })
    } else {
        None
    }
}

/// Test AABB box-box collision (ignores rotation).
pub fn box_box_aabb(
    a: usize,
    b: usize,
    pos_a: Vec3,
    he_a: Vec3,
    pos_b: Vec3,
    he_b: Vec3,
) -> Option<Contact> {
    // Simple AABB vs AABB (ignores rotation)
    let overlap = Vec3::new(
        (he_a.x + he_b.x) - (pos_b.x - pos_a.x).abs(),
        (he_a.y + he_b.y) - (pos_b.y - pos_a.y).abs(),
        (he_a.z + he_b.z) - (pos_b.z - pos_a.z).abs(),
    );

    if overlap.x > 0.0 && overlap.y > 0.0 && overlap.z > 0.0 {
        // Find axis of minimum penetration
        let min_overlap = overlap.x.min(overlap.y).min(overlap.z);
        let normal;
        let depth;

        if min_overlap == overlap.x {
            normal = Vec3::X * (pos_b.x - pos_a.x).signum();
            depth = overlap.x;
        } else if min_overlap == overlap.y {
            normal = Vec3::Y * (pos_b.y - pos_a.y).signum();
            depth = overlap.y;
        } else {
            normal = Vec3::Z * (pos_b.z - pos_a.z).signum();
            depth = overlap.z;
        }

        let point = (pos_a + pos_b) * 0.5;

        Some(Contact {
            body_a: a,
            body_b: b,
            point,
            normal,
            depth,
        })
    } else {
        None
    }
}
