//! glTF import implementation.

use glam::{Vec2, Vec3};
use gltf::{Document, Gltf};
use unshape_mesh::Mesh;
use std::path::Path;

use crate::export::{GltfError, GltfMaterial, GltfMesh, GltfResult};

/// Result of importing a glTF file.
#[derive(Debug, Default)]
pub struct GltfScene {
    /// All meshes in the scene.
    pub meshes: Vec<GltfMesh>,
}

impl GltfScene {
    /// Returns the first mesh, if any.
    pub fn first_mesh(&self) -> Option<&Mesh> {
        self.meshes.first().map(|m| &m.mesh)
    }

    /// Returns all meshes as a flat list.
    pub fn all_meshes(&self) -> impl Iterator<Item = &Mesh> {
        self.meshes.iter().map(|m| &m.mesh)
    }

    /// Merges all meshes into a single mesh.
    pub fn merge_meshes(&self) -> Mesh {
        let mut merged = Mesh::new();

        for gltf_mesh in &self.meshes {
            let mesh = &gltf_mesh.mesh;
            let base_idx = merged.positions.len() as u32;

            merged.positions.extend(&mesh.positions);
            merged.normals.extend(&mesh.normals);
            merged.uvs.extend(&mesh.uvs);

            for idx in &mesh.indices {
                merged.indices.push(idx + base_idx);
            }
        }

        merged
    }
}

/// Imports a glTF file from a path.
pub fn import_gltf(path: impl AsRef<Path>) -> GltfResult<GltfScene> {
    let path = path.as_ref();
    let (document, buffers, _images) = gltf::import(path).map_err(|e| {
        GltfError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            e.to_string(),
        ))
    })?;

    import_from_document(&document, &buffers)
}

/// Imports glTF from raw bytes (GLB format).
pub fn import_gltf_from_bytes(data: &[u8]) -> GltfResult<GltfScene> {
    let glb = Gltf::from_slice(data).map_err(|e| {
        GltfError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            e.to_string(),
        ))
    })?;

    // Extract buffer data from GLB
    let blob = glb.blob.as_ref();
    let buffers: Vec<gltf::buffer::Data> = glb
        .buffers()
        .map(|buffer| {
            let data = match buffer.source() {
                gltf::buffer::Source::Bin => blob.map(|b| b.to_vec()).unwrap_or_default(),
                gltf::buffer::Source::Uri(_) => Vec::new(),
            };
            gltf::buffer::Data(data)
        })
        .collect();

    import_from_document(&glb.document, &buffers)
}

/// Imports meshes from a glTF document.
fn import_from_document(
    document: &Document,
    buffers: &[gltf::buffer::Data],
) -> GltfResult<GltfScene> {
    let mut scene = GltfScene::default();

    for mesh in document.meshes() {
        for primitive in mesh.primitives() {
            let imported_mesh = import_primitive(&primitive, buffers)?;
            let material = import_material(primitive.material());

            let name = mesh
                .name()
                .map(String::from)
                .unwrap_or_else(|| format!("mesh_{}", mesh.index()));

            scene.meshes.push(GltfMesh {
                name,
                mesh: imported_mesh,
                material,
            });
        }
    }

    Ok(scene)
}

/// Imports a single primitive into a Mesh.
fn import_primitive(
    primitive: &gltf::Primitive<'_>,
    buffers: &[gltf::buffer::Data],
) -> GltfResult<Mesh> {
    let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));

    let mut mesh = Mesh::new();

    // Read positions (required)
    if let Some(positions) = reader.read_positions() {
        mesh.positions = positions.map(|p| Vec3::from(p)).collect();
    } else {
        return Err(GltfError::InvalidMesh(
            "Primitive has no position data".to_string(),
        ));
    }

    // Read normals (optional)
    if let Some(normals) = reader.read_normals() {
        mesh.normals = normals.map(|n| Vec3::from(n)).collect();
    }

    // Read UVs (optional, first set)
    if let Some(uvs) = reader.read_tex_coords(0) {
        mesh.uvs = uvs.into_f32().map(|uv| Vec2::from(uv)).collect();
    }

    // Read indices
    if let Some(indices) = reader.read_indices() {
        mesh.indices = indices.into_u32().collect();
    } else {
        // Generate sequential indices if none provided
        mesh.indices = (0..mesh.positions.len() as u32).collect();
    }

    Ok(mesh)
}

/// Imports material data.
fn import_material(material: gltf::Material<'_>) -> Option<GltfMaterial> {
    let pbr = material.pbr_metallic_roughness();
    let base_color = pbr.base_color_factor();

    Some(GltfMaterial {
        name: material
            .name()
            .map(String::from)
            .unwrap_or_else(|| "default".to_string()),
        base_color,
        metallic: pbr.metallic_factor(),
        roughness: pbr.roughness_factor(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::export::{ExportFormat, GltfExporter};

    fn create_test_mesh() -> Mesh {
        let mut mesh = Mesh::new();
        mesh.positions = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.5, 1.0, 0.0),
        ];
        mesh.normals = vec![Vec3::Z, Vec3::Z, Vec3::Z];
        mesh.uvs = vec![
            Vec2::new(0.0, 0.0),
            Vec2::new(1.0, 0.0),
            Vec2::new(0.5, 1.0),
        ];
        mesh.indices = vec![0, 1, 2];
        mesh
    }

    #[test]
    fn test_roundtrip_simple() {
        let original = create_test_mesh();
        let exporter = GltfExporter::new().with_mesh("triangle", original.clone());
        let glb_data = exporter.build(ExportFormat::Glb).unwrap();

        let scene = import_gltf_from_bytes(&glb_data).unwrap();

        assert_eq!(scene.meshes.len(), 1);
        let imported = &scene.meshes[0].mesh;

        assert_eq!(imported.positions.len(), 3);
        assert_eq!(imported.normals.len(), 3);
        assert_eq!(imported.uvs.len(), 3);
        assert_eq!(imported.indices.len(), 3);

        // Check positions match
        for (orig, imp) in original.positions.iter().zip(imported.positions.iter()) {
            assert!((orig.x - imp.x).abs() < 0.001);
            assert!((orig.y - imp.y).abs() < 0.001);
            assert!((orig.z - imp.z).abs() < 0.001);
        }
    }

    #[test]
    fn test_roundtrip_with_material() {
        let mesh = create_test_mesh();
        let material = GltfMaterial::metallic("gold", [1.0, 0.843, 0.0, 1.0], 1.0, 0.3);

        let exporter = GltfExporter::new().with_mesh_and_material("triangle", mesh, material);
        let glb_data = exporter.build(ExportFormat::Glb).unwrap();

        let scene = import_gltf_from_bytes(&glb_data).unwrap();

        let mat = scene.meshes[0].material.as_ref().unwrap();
        assert!((mat.base_color[0] - 1.0).abs() < 0.001);
        assert!((mat.metallic - 1.0).abs() < 0.001);
        assert!((mat.roughness - 0.3).abs() < 0.001);
    }

    #[test]
    fn test_roundtrip_multiple_meshes() {
        let mut mesh1 = Mesh::new();
        mesh1.positions = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.5, 1.0, 0.0),
        ];
        mesh1.indices = vec![0, 1, 2];

        let mut mesh2 = Mesh::new();
        mesh2.positions = vec![
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::new(3.0, 0.0, 0.0),
            Vec3::new(2.5, 1.0, 0.0),
        ];
        mesh2.indices = vec![0, 1, 2];

        let exporter = GltfExporter::new()
            .with_mesh("triangle1", mesh1)
            .with_mesh("triangle2", mesh2);
        let glb_data = exporter.build(ExportFormat::Glb).unwrap();

        let scene = import_gltf_from_bytes(&glb_data).unwrap();

        assert_eq!(scene.meshes.len(), 2);
    }

    #[test]
    fn test_merge_meshes() {
        let mut mesh1 = Mesh::new();
        mesh1.positions = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.5, 1.0, 0.0),
        ];
        mesh1.indices = vec![0, 1, 2];

        let mut mesh2 = Mesh::new();
        mesh2.positions = vec![
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::new(3.0, 0.0, 0.0),
            Vec3::new(2.5, 1.0, 0.0),
        ];
        mesh2.indices = vec![0, 1, 2];

        let exporter = GltfExporter::new()
            .with_mesh("triangle1", mesh1)
            .with_mesh("triangle2", mesh2);
        let glb_data = exporter.build(ExportFormat::Glb).unwrap();

        let scene = import_gltf_from_bytes(&glb_data).unwrap();
        let merged = scene.merge_meshes();

        assert_eq!(merged.positions.len(), 6);
        assert_eq!(merged.indices.len(), 6);
        // Second triangle's indices should be offset by 3
        assert_eq!(merged.indices[3..6], [3, 4, 5]);
    }

    #[test]
    fn test_first_mesh() {
        let mesh = create_test_mesh();
        let exporter = GltfExporter::new().with_mesh("triangle", mesh);
        let glb_data = exporter.build(ExportFormat::Glb).unwrap();

        let scene = import_gltf_from_bytes(&glb_data).unwrap();

        assert!(scene.first_mesh().is_some());
        assert_eq!(scene.first_mesh().unwrap().positions.len(), 3);
    }

    #[test]
    fn test_gltf_scene_empty() {
        let scene = GltfScene::default();
        assert!(scene.first_mesh().is_none());
        assert_eq!(scene.all_meshes().count(), 0);
    }
}
