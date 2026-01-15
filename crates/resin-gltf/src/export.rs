//! glTF export implementation.

use glam::Vec3;
use gltf_json as json;
use rhizome_resin_mesh::Mesh;
use std::path::Path;

/// Result type for glTF operations.
pub type GltfResult<T> = Result<T, GltfError>;

/// Errors that can occur during glTF export.
#[derive(Debug)]
pub enum GltfError {
    /// I/O error.
    Io(std::io::Error),
    /// JSON serialization error.
    Json(serde_json::Error),
    /// Invalid mesh data.
    InvalidMesh(String),
}

impl std::fmt::Display for GltfError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GltfError::Io(e) => write!(f, "I/O error: {}", e),
            GltfError::Json(e) => write!(f, "JSON error: {}", e),
            GltfError::InvalidMesh(msg) => write!(f, "Invalid mesh: {}", msg),
        }
    }
}

impl std::error::Error for GltfError {}

impl From<std::io::Error> for GltfError {
    fn from(e: std::io::Error) -> Self {
        GltfError::Io(e)
    }
}

impl From<serde_json::Error> for GltfError {
    fn from(e: serde_json::Error) -> Self {
        GltfError::Json(e)
    }
}

/// Export format for glTF files.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ExportFormat {
    /// Binary glTF (.glb) - single file with embedded binary data.
    #[default]
    Glb,
    /// JSON glTF with base64-encoded binary data embedded in the JSON.
    GltfEmbedded,
}

/// Material definition for glTF export.
#[derive(Debug, Clone)]
pub struct GltfMaterial {
    /// Material name.
    pub name: String,
    /// Base color factor (RGBA).
    pub base_color: [f32; 4],
    /// Metallic factor (0.0 = dielectric, 1.0 = metal).
    pub metallic: f32,
    /// Roughness factor (0.0 = smooth, 1.0 = rough).
    pub roughness: f32,
}

impl Default for GltfMaterial {
    fn default() -> Self {
        Self {
            name: "default".to_string(),
            base_color: [0.8, 0.8, 0.8, 1.0],
            metallic: 0.0,
            roughness: 0.5,
        }
    }
}

impl GltfMaterial {
    /// Creates a new material with the given name and color.
    pub fn new(name: impl Into<String>, color: [f32; 4]) -> Self {
        Self {
            name: name.into(),
            base_color: color,
            ..Default::default()
        }
    }

    /// Creates a metallic material.
    pub fn metallic(
        name: impl Into<String>,
        color: [f32; 4],
        metallic: f32,
        roughness: f32,
    ) -> Self {
        Self {
            name: name.into(),
            base_color: color,
            metallic,
            roughness,
        }
    }
}

/// A mesh with associated material for glTF export.
#[derive(Debug, Clone)]
pub struct GltfMesh {
    /// Mesh name.
    pub name: String,
    /// The mesh data.
    pub mesh: Mesh,
    /// Optional material.
    pub material: Option<GltfMaterial>,
}

impl GltfMesh {
    /// Creates a new glTF mesh.
    pub fn new(name: impl Into<String>, mesh: Mesh) -> Self {
        Self {
            name: name.into(),
            mesh,
            material: None,
        }
    }
}

/// Builder for exporting meshes to glTF format.
#[derive(Debug, Default)]
pub struct GltfExporter {
    meshes: Vec<GltfMesh>,
}

impl GltfExporter {
    /// Creates a new glTF exporter.
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a mesh with the given name.
    pub fn with_mesh(mut self, name: impl Into<String>, mesh: Mesh) -> Self {
        self.meshes.push(GltfMesh::new(name, mesh));
        self
    }

    /// Adds a mesh with material.
    pub fn with_mesh_and_material(
        mut self,
        name: impl Into<String>,
        mesh: Mesh,
        material: GltfMaterial,
    ) -> Self {
        self.meshes.push(GltfMesh {
            name: name.into(),
            mesh,
            material: Some(material),
        });
        self
    }

    /// Adds a pre-configured GltfMesh.
    pub fn with_gltf_mesh(mut self, mesh: GltfMesh) -> Self {
        self.meshes.push(mesh);
        self
    }

    /// Exports to a binary GLB file.
    pub fn export_glb(&self, path: impl AsRef<Path>) -> GltfResult<()> {
        let data = self.build(ExportFormat::Glb)?;
        std::fs::write(path, data)?;
        Ok(())
    }

    /// Exports to a glTF JSON file with embedded base64 data.
    pub fn export_gltf(&self, path: impl AsRef<Path>) -> GltfResult<()> {
        let data = self.build(ExportFormat::GltfEmbedded)?;
        std::fs::write(path, data)?;
        Ok(())
    }

    /// Builds the glTF data in the specified format.
    pub fn build(&self, format: ExportFormat) -> GltfResult<Vec<u8>> {
        if self.meshes.is_empty() {
            return Err(GltfError::InvalidMesh("No meshes to export".to_string()));
        }

        // Build binary buffer with all mesh data
        let mut buffer_data = Vec::new();
        let mut buffer_views = Vec::new();
        let mut accessors = Vec::new();
        let mut meshes = Vec::new();
        let mut nodes = Vec::new();
        let mut materials = Vec::new();
        let mut material_map = std::collections::HashMap::new();

        for gltf_mesh in &self.meshes {
            let mesh = &gltf_mesh.mesh;

            if mesh.positions.is_empty() {
                return Err(GltfError::InvalidMesh(format!(
                    "Mesh '{}' has no vertices",
                    gltf_mesh.name
                )));
            }

            // Calculate bounds for positions
            let (min_pos, max_pos) = calculate_bounds(&mesh.positions);

            // Align buffer to 4 bytes
            while buffer_data.len() % 4 != 0 {
                buffer_data.push(0);
            }

            // Positions
            let positions_offset = buffer_data.len();
            for pos in &mesh.positions {
                buffer_data.extend_from_slice(bytemuck_cast_slice(&[pos.x, pos.y, pos.z]));
            }
            let positions_len = buffer_data.len() - positions_offset;

            let positions_view_idx = buffer_views.len();
            buffer_views.push(json::buffer::View {
                buffer: json::Index::new(0),
                byte_length: json::validation::USize64(positions_len as u64),
                byte_offset: Some(json::validation::USize64(positions_offset as u64)),
                byte_stride: None,
                extensions: None,
                extras: Default::default(),
                name: None,
                target: Some(json::validation::Checked::Valid(
                    json::buffer::Target::ArrayBuffer,
                )),
            });

            let positions_accessor_idx = accessors.len();
            accessors.push(json::Accessor {
                buffer_view: Some(json::Index::new(positions_view_idx as u32)),
                byte_offset: Some(json::validation::USize64(0)),
                count: json::validation::USize64(mesh.positions.len() as u64),
                component_type: json::validation::Checked::Valid(
                    json::accessor::GenericComponentType(json::accessor::ComponentType::F32),
                ),
                extensions: None,
                extras: Default::default(),
                name: None,
                type_: json::validation::Checked::Valid(json::accessor::Type::Vec3),
                min: Some(json::Value::from(vec![min_pos.x, min_pos.y, min_pos.z])),
                max: Some(json::Value::from(vec![max_pos.x, max_pos.y, max_pos.z])),
                normalized: false,
                sparse: None,
            });

            // Normals (if present)
            let normals_accessor_idx = if mesh.has_normals() {
                while buffer_data.len() % 4 != 0 {
                    buffer_data.push(0);
                }

                let normals_offset = buffer_data.len();
                for normal in &mesh.normals {
                    buffer_data
                        .extend_from_slice(bytemuck_cast_slice(&[normal.x, normal.y, normal.z]));
                }
                let normals_len = buffer_data.len() - normals_offset;

                let normals_view_idx = buffer_views.len();
                buffer_views.push(json::buffer::View {
                    buffer: json::Index::new(0),
                    byte_length: json::validation::USize64(normals_len as u64),
                    byte_offset: Some(json::validation::USize64(normals_offset as u64)),
                    byte_stride: None,
                    extensions: None,
                    extras: Default::default(),
                    name: None,
                    target: Some(json::validation::Checked::Valid(
                        json::buffer::Target::ArrayBuffer,
                    )),
                });

                let idx = accessors.len();
                accessors.push(json::Accessor {
                    buffer_view: Some(json::Index::new(normals_view_idx as u32)),
                    byte_offset: Some(json::validation::USize64(0)),
                    count: json::validation::USize64(mesh.normals.len() as u64),
                    component_type: json::validation::Checked::Valid(
                        json::accessor::GenericComponentType(json::accessor::ComponentType::F32),
                    ),
                    extensions: None,
                    extras: Default::default(),
                    name: None,
                    type_: json::validation::Checked::Valid(json::accessor::Type::Vec3),
                    min: None,
                    max: None,
                    normalized: false,
                    sparse: None,
                });
                Some(idx)
            } else {
                None
            };

            // UVs (if present)
            let uvs_accessor_idx = if mesh.has_uvs() {
                while buffer_data.len() % 4 != 0 {
                    buffer_data.push(0);
                }

                let uvs_offset = buffer_data.len();
                for uv in &mesh.uvs {
                    buffer_data.extend_from_slice(bytemuck_cast_slice(&[uv.x, uv.y]));
                }
                let uvs_len = buffer_data.len() - uvs_offset;

                let uvs_view_idx = buffer_views.len();
                buffer_views.push(json::buffer::View {
                    buffer: json::Index::new(0),
                    byte_length: json::validation::USize64(uvs_len as u64),
                    byte_offset: Some(json::validation::USize64(uvs_offset as u64)),
                    byte_stride: None,
                    extensions: None,
                    extras: Default::default(),
                    name: None,
                    target: Some(json::validation::Checked::Valid(
                        json::buffer::Target::ArrayBuffer,
                    )),
                });

                let idx = accessors.len();
                accessors.push(json::Accessor {
                    buffer_view: Some(json::Index::new(uvs_view_idx as u32)),
                    byte_offset: Some(json::validation::USize64(0)),
                    count: json::validation::USize64(mesh.uvs.len() as u64),
                    component_type: json::validation::Checked::Valid(
                        json::accessor::GenericComponentType(json::accessor::ComponentType::F32),
                    ),
                    extensions: None,
                    extras: Default::default(),
                    name: None,
                    type_: json::validation::Checked::Valid(json::accessor::Type::Vec2),
                    min: None,
                    max: None,
                    normalized: false,
                    sparse: None,
                });
                Some(idx)
            } else {
                None
            };

            // Indices
            while buffer_data.len() % 4 != 0 {
                buffer_data.push(0);
            }

            let indices_offset = buffer_data.len();
            for idx in &mesh.indices {
                buffer_data.extend_from_slice(bytemuck_cast_slice(&[*idx]));
            }
            let indices_len = buffer_data.len() - indices_offset;

            let indices_view_idx = buffer_views.len();
            buffer_views.push(json::buffer::View {
                buffer: json::Index::new(0),
                byte_length: json::validation::USize64(indices_len as u64),
                byte_offset: Some(json::validation::USize64(indices_offset as u64)),
                byte_stride: None,
                extensions: None,
                extras: Default::default(),
                name: None,
                target: Some(json::validation::Checked::Valid(
                    json::buffer::Target::ElementArrayBuffer,
                )),
            });

            let indices_accessor_idx = accessors.len();
            accessors.push(json::Accessor {
                buffer_view: Some(json::Index::new(indices_view_idx as u32)),
                byte_offset: Some(json::validation::USize64(0)),
                count: json::validation::USize64(mesh.indices.len() as u64),
                component_type: json::validation::Checked::Valid(
                    json::accessor::GenericComponentType(json::accessor::ComponentType::U32),
                ),
                extensions: None,
                extras: Default::default(),
                name: None,
                type_: json::validation::Checked::Valid(json::accessor::Type::Scalar),
                min: None,
                max: None,
                normalized: false,
                sparse: None,
            });

            // Build primitive attributes
            let mut attributes = std::collections::BTreeMap::new();
            attributes.insert(
                json::validation::Checked::Valid(json::mesh::Semantic::Positions),
                json::Index::new(positions_accessor_idx as u32),
            );

            if let Some(idx) = normals_accessor_idx {
                attributes.insert(
                    json::validation::Checked::Valid(json::mesh::Semantic::Normals),
                    json::Index::new(idx as u32),
                );
            }

            if let Some(idx) = uvs_accessor_idx {
                attributes.insert(
                    json::validation::Checked::Valid(json::mesh::Semantic::TexCoords(0)),
                    json::Index::new(idx as u32),
                );
            }

            // Handle material
            let material_idx = if let Some(mat) = &gltf_mesh.material {
                if let Some(&idx) = material_map.get(&mat.name) {
                    Some(idx)
                } else {
                    let idx = materials.len();
                    materials.push(json::Material {
                        alpha_cutoff: None,
                        alpha_mode: json::validation::Checked::Valid(
                            json::material::AlphaMode::Opaque,
                        ),
                        double_sided: false,
                        name: Some(mat.name.clone()),
                        pbr_metallic_roughness: json::material::PbrMetallicRoughness {
                            base_color_factor: json::material::PbrBaseColorFactor(mat.base_color),
                            base_color_texture: None,
                            metallic_factor: json::material::StrengthFactor(mat.metallic),
                            roughness_factor: json::material::StrengthFactor(mat.roughness),
                            metallic_roughness_texture: None,
                            extensions: None,
                            extras: Default::default(),
                        },
                        normal_texture: None,
                        occlusion_texture: None,
                        emissive_texture: None,
                        emissive_factor: json::material::EmissiveFactor([0.0, 0.0, 0.0]),
                        extensions: None,
                        extras: Default::default(),
                    });
                    material_map.insert(mat.name.clone(), idx);
                    Some(idx)
                }
            } else {
                None
            };

            // Create mesh
            let mesh_idx = meshes.len();
            meshes.push(json::Mesh {
                extensions: None,
                extras: Default::default(),
                name: Some(gltf_mesh.name.clone()),
                primitives: vec![json::mesh::Primitive {
                    attributes,
                    extensions: None,
                    extras: Default::default(),
                    indices: Some(json::Index::new(indices_accessor_idx as u32)),
                    material: material_idx.map(|i| json::Index::new(i as u32)),
                    mode: json::validation::Checked::Valid(json::mesh::Mode::Triangles),
                    targets: None,
                }],
                weights: None,
            });

            // Create node for this mesh
            nodes.push(json::Node {
                camera: None,
                children: None,
                extensions: None,
                extras: Default::default(),
                matrix: None,
                mesh: Some(json::Index::new(mesh_idx as u32)),
                name: Some(gltf_mesh.name.clone()),
                rotation: None,
                scale: None,
                translation: None,
                skin: None,
                weights: None,
            });
        }

        // Create scene with all nodes
        let node_indices: Vec<_> = (0..nodes.len() as u32).map(json::Index::new).collect();

        let scene = json::Scene {
            extensions: None,
            extras: Default::default(),
            name: None,
            nodes: node_indices,
        };

        // Create buffer
        let buffer = match format {
            ExportFormat::Glb => json::Buffer {
                byte_length: json::validation::USize64(buffer_data.len() as u64),
                extensions: None,
                extras: Default::default(),
                name: None,
                uri: None,
            },
            ExportFormat::GltfEmbedded => {
                let base64_data = base64::Engine::encode(
                    &base64::engine::general_purpose::STANDARD,
                    &buffer_data,
                );
                json::Buffer {
                    byte_length: json::validation::USize64(buffer_data.len() as u64),
                    extensions: None,
                    extras: Default::default(),
                    name: None,
                    uri: Some(format!(
                        "data:application/octet-stream;base64,{}",
                        base64_data
                    )),
                }
            }
        };

        // Build root
        let root = json::Root {
            accessors,
            animations: Vec::new(),
            asset: json::Asset {
                copyright: None,
                extensions: None,
                extras: Default::default(),
                generator: Some("resin-gltf".to_string()),
                min_version: None,
                version: "2.0".to_string(),
            },
            buffers: vec![buffer],
            buffer_views,
            scene: Some(json::Index::new(0)),
            extensions: None,
            extras: Default::default(),
            extensions_used: Vec::new(),
            extensions_required: Vec::new(),
            cameras: Vec::new(),
            images: Vec::new(),
            materials,
            meshes,
            nodes,
            samplers: Vec::new(),
            scenes: vec![scene],
            skins: Vec::new(),
            textures: Vec::new(),
        };

        match format {
            ExportFormat::Glb => build_glb(&root, &buffer_data),
            ExportFormat::GltfEmbedded => {
                let json = serde_json::to_vec_pretty(&root)?;
                Ok(json)
            }
        }
    }
}

/// Builds a GLB binary file.
fn build_glb(root: &json::Root, buffer_data: &[u8]) -> GltfResult<Vec<u8>> {
    let json_bytes = serde_json::to_vec(root)?;

    // Pad JSON to 4-byte alignment
    let json_padding = (4 - (json_bytes.len() % 4)) % 4;
    let json_chunk_len = json_bytes.len() + json_padding;

    // Pad buffer to 4-byte alignment
    let buffer_padding = (4 - (buffer_data.len() % 4)) % 4;
    let buffer_chunk_len = buffer_data.len() + buffer_padding;

    // GLB header (12 bytes) + JSON chunk header (8 bytes) + JSON + BIN chunk header (8 bytes) + BIN
    let total_len = 12 + 8 + json_chunk_len + 8 + buffer_chunk_len;

    let mut output = Vec::with_capacity(total_len);

    // GLB Header
    output.extend_from_slice(b"glTF"); // magic
    output.extend_from_slice(&2u32.to_le_bytes()); // version
    output.extend_from_slice(&(total_len as u32).to_le_bytes()); // length

    // JSON chunk
    output.extend_from_slice(&(json_chunk_len as u32).to_le_bytes()); // chunk length
    output.extend_from_slice(&0x4E4F534Au32.to_le_bytes()); // chunk type "JSON"
    output.extend_from_slice(&json_bytes);
    for _ in 0..json_padding {
        output.push(b' '); // JSON padding uses spaces
    }

    // BIN chunk
    output.extend_from_slice(&(buffer_chunk_len as u32).to_le_bytes()); // chunk length
    output.extend_from_slice(&0x004E4942u32.to_le_bytes()); // chunk type "BIN\0"
    output.extend_from_slice(buffer_data);
    for _ in 0..buffer_padding {
        output.push(0); // BIN padding uses zeros
    }

    Ok(output)
}

/// Calculates min/max bounds for a set of positions.
fn calculate_bounds(positions: &[Vec3]) -> (Vec3, Vec3) {
    if positions.is_empty() {
        return (Vec3::ZERO, Vec3::ZERO);
    }

    let mut min = positions[0];
    let mut max = positions[0];

    for pos in positions.iter().skip(1) {
        min = min.min(*pos);
        max = max.max(*pos);
    }

    (min, max)
}

/// Casts a slice of f32 or u32 to bytes.
fn bytemuck_cast_slice<T: bytemuck::Pod>(slice: &[T]) -> &[u8] {
    bytemuck::cast_slice(slice)
}

// Need bytemuck for safe casting
mod bytemuck {
    pub unsafe trait Pod: Copy + 'static {}
    unsafe impl Pod for f32 {}
    unsafe impl Pod for u32 {}

    pub fn cast_slice<T: Pod>(slice: &[T]) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(slice.as_ptr() as *const u8, std::mem::size_of_val(slice))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec2;

    #[test]
    fn test_export_simple_triangle() {
        let mut mesh = Mesh::new();
        mesh.positions = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.5, 1.0, 0.0),
        ];
        mesh.normals = vec![Vec3::Z, Vec3::Z, Vec3::Z];
        mesh.indices = vec![0, 1, 2];

        let exporter = GltfExporter::new().with_mesh("triangle", mesh);
        let glb = exporter.build(ExportFormat::Glb).unwrap();

        // Check GLB magic
        assert_eq!(&glb[0..4], b"glTF");
        // Check version
        assert_eq!(u32::from_le_bytes([glb[4], glb[5], glb[6], glb[7]]), 2);
    }

    #[test]
    fn test_export_with_uvs() {
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

        let exporter = GltfExporter::new().with_mesh("triangle", mesh);
        let glb = exporter.build(ExportFormat::Glb).unwrap();

        assert!(!glb.is_empty());
    }

    #[test]
    fn test_export_with_material() {
        let mut mesh = Mesh::new();
        mesh.positions = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.5, 1.0, 0.0),
        ];
        mesh.indices = vec![0, 1, 2];

        let material = GltfMaterial::new("red", [1.0, 0.0, 0.0, 1.0]);
        let exporter = GltfExporter::new().with_mesh_and_material("triangle", mesh, material);
        let glb = exporter.build(ExportFormat::Glb).unwrap();

        assert!(!glb.is_empty());
    }

    #[test]
    fn test_export_multiple_meshes() {
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
        let glb = exporter.build(ExportFormat::Glb).unwrap();

        assert!(!glb.is_empty());
    }

    #[test]
    fn test_export_gltf_embedded() {
        let mut mesh = Mesh::new();
        mesh.positions = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.5, 1.0, 0.0),
        ];
        mesh.indices = vec![0, 1, 2];

        let exporter = GltfExporter::new().with_mesh("triangle", mesh);
        let gltf = exporter.build(ExportFormat::GltfEmbedded).unwrap();

        // Should be valid JSON
        let json_str = std::str::from_utf8(&gltf).unwrap();
        assert!(json_str.contains("\"asset\""));
        // Version might be serialized with or without spaces
        assert!(json_str.contains("2.0"), "Should contain version 2.0");
        assert!(json_str.contains("data:application/octet-stream;base64,"));
    }

    #[test]
    fn test_calculate_bounds() {
        let positions = vec![
            Vec3::new(-1.0, 0.0, 2.0),
            Vec3::new(1.0, -2.0, 0.0),
            Vec3::new(0.0, 3.0, -1.0),
        ];

        let (min, max) = calculate_bounds(&positions);

        assert_eq!(min, Vec3::new(-1.0, -2.0, -1.0));
        assert_eq!(max, Vec3::new(1.0, 3.0, 2.0));
    }

    #[test]
    fn test_empty_mesh_error() {
        let exporter = GltfExporter::new();
        let result = exporter.build(ExportFormat::Glb);
        assert!(result.is_err());
    }
}
