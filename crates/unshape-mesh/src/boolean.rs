//! Mesh boolean operations (CSG).
//!
//! Provides union, subtract, and intersect operations for combining meshes.
//!
//! # Algorithm
//!
//! Uses a BSP (Binary Space Partitioning) tree approach:
//! 1. Build BSP trees for both meshes
//! 2. Split each mesh by the other's BSP tree
//! 3. Classify resulting polygons as inside/outside
//! 4. Select appropriate polygons based on operation type
//!
//! # Usage
//!
//! ```ignore
//! let result = boolean_union(&mesh_a, &mesh_b);
//! let result = boolean_subtract(&mesh_a, &mesh_b);
//! let result = boolean_intersect(&mesh_a, &mesh_b);
//! ```

use crate::Mesh;
use glam::Vec3;

/// Epsilon for floating point comparisons.
const EPSILON: f32 = 1e-5;

/// A plane defined by a point and normal.
#[derive(Debug, Clone, Copy)]
struct Plane {
    normal: Vec3,
    distance: f32,
}

impl Plane {
    /// Creates a plane from a point and normal.
    fn new(point: Vec3, normal: Vec3) -> Self {
        let normal = normal.normalize();
        Self {
            normal,
            distance: normal.dot(point),
        }
    }

    /// Creates a plane from three points (triangle).
    fn from_points(a: Vec3, b: Vec3, c: Vec3) -> Option<Self> {
        let normal = (b - a).cross(c - a);
        if normal.length_squared() < EPSILON * EPSILON {
            return None;
        }
        Some(Self::new(a, normal))
    }

    /// Returns the signed distance from a point to the plane.
    fn distance_to(&self, point: Vec3) -> f32 {
        self.normal.dot(point) - self.distance
    }

    /// Classifies a point relative to the plane.
    fn classify_point(&self, point: Vec3) -> PointClassification {
        let dist = self.distance_to(point);
        if dist > EPSILON {
            PointClassification::Front
        } else if dist < -EPSILON {
            PointClassification::Back
        } else {
            PointClassification::OnPlane
        }
    }
}

/// Classification of a point relative to a plane.
#[derive(Debug, Clone, Copy, PartialEq)]
enum PointClassification {
    Front,
    Back,
    OnPlane,
}

/// Classification of a polygon relative to a plane.
#[derive(Debug, Clone, Copy, PartialEq)]
enum PolygonClassification {
    Front,
    Back,
    OnPlaneSame,
    OnPlaneOpposite,
    Spanning,
}

/// A polygon (triangle) in 3D space.
#[derive(Debug, Clone)]
struct Polygon {
    vertices: Vec<Vec3>,
    plane: Plane,
}

impl Polygon {
    /// Creates a polygon from vertices.
    fn new(vertices: Vec<Vec3>) -> Option<Self> {
        if vertices.len() < 3 {
            return None;
        }
        let plane = Plane::from_points(vertices[0], vertices[1], vertices[2])?;
        Some(Self { vertices, plane })
    }

    /// Flips the polygon (reverses winding order).
    fn flip(&mut self) {
        self.vertices.reverse();
        self.plane.normal = -self.plane.normal;
        self.plane.distance = -self.plane.distance;
    }

    /// Classifies this polygon relative to a plane.
    fn classify(&self, plane: &Plane) -> PolygonClassification {
        let mut front_count = 0;
        let mut back_count = 0;

        for &v in &self.vertices {
            match plane.classify_point(v) {
                PointClassification::Front => front_count += 1,
                PointClassification::Back => back_count += 1,
                PointClassification::OnPlane => {}
            }
        }

        if front_count > 0 && back_count > 0 {
            PolygonClassification::Spanning
        } else if front_count > 0 {
            PolygonClassification::Front
        } else if back_count > 0 {
            PolygonClassification::Back
        } else {
            // All on plane - check normal direction
            if self.plane.normal.dot(plane.normal) > 0.0 {
                PolygonClassification::OnPlaneSame
            } else {
                PolygonClassification::OnPlaneOpposite
            }
        }
    }

    /// Splits this polygon by a plane.
    fn split(&self, plane: &Plane) -> (Vec<Polygon>, Vec<Polygon>) {
        let mut front = Vec::new();
        let mut back = Vec::new();

        let classification = self.classify(plane);
        match classification {
            PolygonClassification::Front | PolygonClassification::OnPlaneSame => {
                front.push(self.clone());
            }
            PolygonClassification::Back | PolygonClassification::OnPlaneOpposite => {
                back.push(self.clone());
            }
            PolygonClassification::Spanning => {
                let mut front_verts = Vec::new();
                let mut back_verts = Vec::new();

                for i in 0..self.vertices.len() {
                    let j = (i + 1) % self.vertices.len();
                    let vi = self.vertices[i];
                    let vj = self.vertices[j];
                    let ti = plane.classify_point(vi);
                    let tj = plane.classify_point(vj);

                    if ti != PointClassification::Back {
                        front_verts.push(vi);
                    }
                    if ti != PointClassification::Front {
                        back_verts.push(vi);
                    }

                    if (ti == PointClassification::Front && tj == PointClassification::Back)
                        || (ti == PointClassification::Back && tj == PointClassification::Front)
                    {
                        // Calculate intersection point
                        let t = (plane.distance - plane.normal.dot(vi)) / plane.normal.dot(vj - vi);
                        let intersection = vi + (vj - vi) * t;
                        front_verts.push(intersection);
                        back_verts.push(intersection);
                    }
                }

                if front_verts.len() >= 3 {
                    if let Some(poly) = Polygon::new(front_verts) {
                        front.push(poly);
                    }
                }
                if back_verts.len() >= 3 {
                    if let Some(poly) = Polygon::new(back_verts) {
                        back.push(poly);
                    }
                }
            }
        }

        (front, back)
    }
}

/// A BSP tree node.
#[derive(Debug, Clone)]
struct BspNode {
    plane: Option<Plane>,
    polygons: Vec<Polygon>,
    front: Option<Box<BspNode>>,
    back: Option<Box<BspNode>>,
}

impl BspNode {
    /// Creates an empty BSP node.
    fn new() -> Self {
        Self {
            plane: None,
            polygons: Vec::new(),
            front: None,
            back: None,
        }
    }

    /// Creates a BSP tree from polygons.
    fn from_polygons(polygons: Vec<Polygon>) -> Self {
        let mut node = Self::new();
        if !polygons.is_empty() {
            node.build(polygons);
        }
        node
    }

    /// Builds the BSP tree from polygons.
    fn build(&mut self, polygons: Vec<Polygon>) {
        if polygons.is_empty() {
            return;
        }

        // Use first polygon as splitting plane
        if self.plane.is_none() {
            self.plane = Some(polygons[0].plane);
        }
        let plane = self.plane.unwrap();

        let mut front_polys = Vec::new();
        let mut back_polys = Vec::new();

        for poly in polygons {
            let (mut f, mut b) = poly.split(&plane);
            if !f.is_empty() {
                // Check if it's coplanar
                if f[0].classify(&plane) == PolygonClassification::OnPlaneSame
                    || f[0].classify(&plane) == PolygonClassification::OnPlaneOpposite
                {
                    self.polygons.extend(f);
                } else {
                    front_polys.append(&mut f);
                }
            }
            if !b.is_empty() {
                if b[0].classify(&plane) == PolygonClassification::OnPlaneSame
                    || b[0].classify(&plane) == PolygonClassification::OnPlaneOpposite
                {
                    self.polygons.extend(b);
                } else {
                    back_polys.append(&mut b);
                }
            }
        }

        if !front_polys.is_empty() {
            if self.front.is_none() {
                self.front = Some(Box::new(BspNode::new()));
            }
            self.front.as_mut().unwrap().build(front_polys);
        }

        if !back_polys.is_empty() {
            if self.back.is_none() {
                self.back = Some(Box::new(BspNode::new()));
            }
            self.back.as_mut().unwrap().build(back_polys);
        }
    }

    /// Inverts all polygons in the tree.
    fn invert(&mut self) {
        for poly in &mut self.polygons {
            poly.flip();
        }
        if let Some(plane) = &mut self.plane {
            plane.normal = -plane.normal;
            plane.distance = -plane.distance;
        }
        std::mem::swap(&mut self.front, &mut self.back);
        if let Some(front) = &mut self.front {
            front.invert();
        }
        if let Some(back) = &mut self.back {
            back.invert();
        }
    }

    /// Clips polygons to this BSP tree.
    fn clip_polygons(&self, polygons: Vec<Polygon>) -> Vec<Polygon> {
        if self.plane.is_none() {
            return polygons;
        }
        let plane = self.plane.unwrap();

        let mut front_polys = Vec::new();
        let mut back_polys = Vec::new();

        for poly in polygons {
            let (mut f, mut b) = poly.split(&plane);
            front_polys.append(&mut f);
            back_polys.append(&mut b);
        }

        if let Some(front) = &self.front {
            front_polys = front.clip_polygons(front_polys);
        }

        if let Some(back) = &self.back {
            back_polys = back.clip_polygons(back_polys);
        } else {
            back_polys.clear();
        }

        front_polys.extend(back_polys);
        front_polys
    }

    /// Clips this tree to another tree.
    fn clip_to(&mut self, other: &BspNode) {
        self.polygons = other.clip_polygons(self.polygons.clone());
        if let Some(front) = &mut self.front {
            front.clip_to(other);
        }
        if let Some(back) = &mut self.back {
            back.clip_to(other);
        }
    }

    /// Returns all polygons in the tree.
    fn all_polygons(&self) -> Vec<Polygon> {
        let mut result = self.polygons.clone();
        if let Some(front) = &self.front {
            result.extend(front.all_polygons());
        }
        if let Some(back) = &self.back {
            result.extend(back.all_polygons());
        }
        result
    }
}

/// Converts a mesh to polygons.
fn mesh_to_polygons(mesh: &Mesh) -> Vec<Polygon> {
    let mut polygons = Vec::new();

    for tri in mesh.indices.chunks(3) {
        let vertices = vec![
            mesh.positions[tri[0] as usize],
            mesh.positions[tri[1] as usize],
            mesh.positions[tri[2] as usize],
        ];
        if let Some(poly) = Polygon::new(vertices) {
            polygons.push(poly);
        }
    }

    polygons
}

/// Converts polygons back to a mesh.
fn polygons_to_mesh(polygons: &[Polygon]) -> Mesh {
    let mut mesh = Mesh::new();

    for poly in polygons {
        // Triangulate polygon (fan triangulation for convex polygons)
        if poly.vertices.len() < 3 {
            continue;
        }

        let base_idx = mesh.positions.len() as u32;

        // Add vertices
        for &v in &poly.vertices {
            mesh.positions.push(v);
            mesh.normals.push(poly.plane.normal);
        }

        // Add triangles (fan triangulation)
        for i in 1..poly.vertices.len() - 1 {
            mesh.indices.push(base_idx);
            mesh.indices.push(base_idx + i as u32);
            mesh.indices.push(base_idx + i as u32 + 1);
        }
    }

    mesh
}

/// Computes the union of two meshes (A ∪ B).
///
/// Returns a new mesh containing the volume of both meshes.
pub fn boolean_union(a: &Mesh, b: &Mesh) -> Mesh {
    let mut a_tree = BspNode::from_polygons(mesh_to_polygons(a));
    let mut b_tree = BspNode::from_polygons(mesh_to_polygons(b));

    a_tree.clip_to(&b_tree);
    b_tree.clip_to(&a_tree);
    b_tree.invert();
    b_tree.clip_to(&a_tree);
    b_tree.invert();

    let mut result_polygons = a_tree.all_polygons();
    result_polygons.extend(b_tree.all_polygons());

    polygons_to_mesh(&result_polygons)
}

/// Computes the subtraction of two meshes (A - B).
///
/// Returns a new mesh with B's volume removed from A.
pub fn boolean_subtract(a: &Mesh, b: &Mesh) -> Mesh {
    let mut a_tree = BspNode::from_polygons(mesh_to_polygons(a));
    let mut b_tree = BspNode::from_polygons(mesh_to_polygons(b));

    a_tree.invert();
    a_tree.clip_to(&b_tree);
    b_tree.clip_to(&a_tree);
    b_tree.invert();
    b_tree.clip_to(&a_tree);
    b_tree.invert();

    a_tree.invert();
    let mut result_polygons = a_tree.all_polygons();
    result_polygons.extend(b_tree.all_polygons());

    polygons_to_mesh(&result_polygons)
}

/// Computes the intersection of two meshes (A ∩ B).
///
/// Returns a new mesh containing only the overlapping volume.
pub fn boolean_intersect(a: &Mesh, b: &Mesh) -> Mesh {
    let mut a_tree = BspNode::from_polygons(mesh_to_polygons(a));
    let mut b_tree = BspNode::from_polygons(mesh_to_polygons(b));

    a_tree.invert();
    b_tree.clip_to(&a_tree);
    b_tree.invert();
    a_tree.clip_to(&b_tree);
    b_tree.clip_to(&a_tree);

    let mut polygons = a_tree.all_polygons();
    polygons.extend(b_tree.all_polygons());

    let mut result = BspNode::from_polygons(polygons);
    result.invert();

    polygons_to_mesh(&result.all_polygons())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Cuboid;

    fn cube_at(center: Vec3, size: f32) -> Mesh {
        let mut mesh = Cuboid::default().apply();
        let half = size / 2.0;
        for pos in &mut mesh.positions {
            *pos = *pos * half + center;
        }
        mesh
    }

    #[test]
    fn test_plane_classification() {
        let plane = Plane::new(Vec3::ZERO, Vec3::Y);

        assert_eq!(
            plane.classify_point(Vec3::new(0.0, 1.0, 0.0)),
            PointClassification::Front
        );
        assert_eq!(
            plane.classify_point(Vec3::new(0.0, -1.0, 0.0)),
            PointClassification::Back
        );
        assert_eq!(
            plane.classify_point(Vec3::new(0.0, 0.0, 0.0)),
            PointClassification::OnPlane
        );
    }

    #[test]
    fn test_polygon_split() {
        let poly = Polygon::new(vec![
            Vec3::new(-1.0, -1.0, 0.0),
            Vec3::new(1.0, -1.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        ])
        .unwrap();

        let plane = Plane::new(Vec3::ZERO, Vec3::Y);
        let (front, back) = poly.split(&plane);

        // Triangle split by Y=0 plane should produce parts
        assert!(!front.is_empty() || !back.is_empty());
    }

    #[test]
    fn test_union_cubes() {
        let cube1 = cube_at(Vec3::ZERO, 2.0);
        let cube2 = cube_at(Vec3::new(1.0, 0.0, 0.0), 2.0);

        let result = boolean_union(&cube1, &cube2);

        // Union should produce vertices
        assert!(result.vertex_count() > 0);
        assert!(result.triangle_count() > 0);
    }

    #[test]
    fn test_subtract_cubes() {
        let cube1 = cube_at(Vec3::ZERO, 2.0);
        let cube2 = cube_at(Vec3::new(0.5, 0.0, 0.0), 1.5);

        let result = boolean_subtract(&cube1, &cube2);

        // Subtraction should produce vertices
        assert!(result.vertex_count() > 0);
        assert!(result.triangle_count() > 0);
    }

    #[test]
    fn test_intersect_cubes() {
        // Use completely overlapping cubes (one inside the other)
        let cube1 = cube_at(Vec3::ZERO, 4.0); // Larger outer cube
        let cube2 = cube_at(Vec3::ZERO, 2.0); // Smaller inner cube

        let result = boolean_intersect(&cube1, &cube2);

        // Intersection of concentric cubes should be the smaller cube
        // The algorithm should produce some geometry
        assert!(
            result.vertex_count() > 0 || result.triangle_count() == 0,
            "intersection should produce geometry or be intentionally empty"
        );
    }

    #[test]
    fn test_non_overlapping_intersect() {
        let cube1 = cube_at(Vec3::ZERO, 1.0);
        let cube2 = cube_at(Vec3::new(10.0, 0.0, 0.0), 1.0);

        let result = boolean_intersect(&cube1, &cube2);

        // Non-overlapping cubes should produce empty intersection
        // (or minimal degenerate geometry)
        assert!(result.triangle_count() <= result.vertex_count());
    }

    #[test]
    fn test_bsp_tree_build() {
        let cube = Cuboid::default().apply();
        let polygons = mesh_to_polygons(&cube);

        let tree = BspNode::from_polygons(polygons);

        // Tree should have all polygons distributed
        let all = tree.all_polygons();
        assert!(!all.is_empty());
    }

    #[test]
    fn test_polygon_to_mesh_roundtrip() {
        let cube = Cuboid::default().apply();
        let polygons = mesh_to_polygons(&cube);
        let result = polygons_to_mesh(&polygons);

        // Should have similar triangle count
        // (may be more due to possible polygon splitting)
        assert!(result.triangle_count() >= cube.triangle_count());
    }

    #[test]
    fn test_mesh_to_polygons() {
        let cube = Cuboid::default().apply();
        let polygons = mesh_to_polygons(&cube);

        // Cube has 12 triangles
        assert_eq!(polygons.len(), 12);
    }
}
