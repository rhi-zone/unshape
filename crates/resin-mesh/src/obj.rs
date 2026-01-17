//! OBJ file format import and export.
//!
//! Supports:
//! - Vertex positions (v)
//! - Vertex normals (vn)
//! - Texture coordinates (vt)
//! - Triangle and quad faces (f)
//!
//! # Example
//!
//! ```ignore
//! use rhizome_resin_mesh::{Cuboid, export_obj, import_obj};
//!
//! let cube = Cuboid::default().apply();
//! let obj_string = export_obj(&cube);
//!
//! let imported = import_obj(&obj_string).unwrap();
//! ```

use std::fmt::Write as FmtWrite;
use std::io::{BufRead, BufReader, Read};

use glam::{Vec2, Vec3};

use crate::Mesh;

/// Errors that can occur during OBJ import.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ObjError {
    /// Invalid or malformed line in the OBJ file.
    #[error("Invalid line: {0}")]
    InvalidLine(String),
    /// Face references a vertex index that doesn't exist.
    #[error("Invalid vertex index: {0}")]
    InvalidVertexIndex(usize),
    /// Face references a normal index that doesn't exist.
    #[error("Invalid normal index: {0}")]
    InvalidNormalIndex(usize),
    /// Face references a texture coordinate index that doesn't exist.
    #[error("Invalid texture coordinate index: {0}")]
    InvalidTexCoordIndex(usize),
    /// Failed to parse a number.
    #[error("Parse error: {0}")]
    ParseError(String),
    /// Face has fewer than 3 vertices.
    #[error("Invalid face: {0}")]
    InvalidFace(String),
}

/// Imports a mesh from OBJ format string.
pub fn import_obj(obj_str: &str) -> Result<Mesh, ObjError> {
    import_obj_from_reader(obj_str.as_bytes())
}

/// Imports a mesh from any reader containing OBJ data.
pub fn import_obj_from_reader<R: Read>(reader: R) -> Result<Mesh, ObjError> {
    let reader = BufReader::new(reader);

    // Temporary storage for OBJ data (1-indexed in the file)
    let mut positions: Vec<Vec3> = Vec::new();
    let mut normals: Vec<Vec3> = Vec::new();
    let mut tex_coords: Vec<Vec2> = Vec::new();

    // Final mesh data
    let mut mesh_positions: Vec<Vec3> = Vec::new();
    let mut mesh_normals: Vec<Vec3> = Vec::new();
    let mut mesh_uvs: Vec<Vec2> = Vec::new();
    let mut mesh_indices: Vec<u32> = Vec::new();

    // Map from (pos_idx, tex_idx, norm_idx) to final vertex index
    let mut vertex_map: std::collections::HashMap<(usize, Option<usize>, Option<usize>), u32> =
        std::collections::HashMap::new();

    for line in reader.lines() {
        let line = line.map_err(|e| ObjError::ParseError(e.to_string()))?;
        let line = line.trim();

        // Skip empty lines and comments
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.is_empty() {
            continue;
        }

        match parts[0] {
            "v" => {
                // Vertex position
                if parts.len() < 4 {
                    return Err(ObjError::InvalidLine(line.to_string()));
                }
                let x = parts[1]
                    .parse::<f32>()
                    .map_err(|e| ObjError::ParseError(e.to_string()))?;
                let y = parts[2]
                    .parse::<f32>()
                    .map_err(|e| ObjError::ParseError(e.to_string()))?;
                let z = parts[3]
                    .parse::<f32>()
                    .map_err(|e| ObjError::ParseError(e.to_string()))?;
                positions.push(Vec3::new(x, y, z));
            }
            "vn" => {
                // Vertex normal
                if parts.len() < 4 {
                    return Err(ObjError::InvalidLine(line.to_string()));
                }
                let x = parts[1]
                    .parse::<f32>()
                    .map_err(|e| ObjError::ParseError(e.to_string()))?;
                let y = parts[2]
                    .parse::<f32>()
                    .map_err(|e| ObjError::ParseError(e.to_string()))?;
                let z = parts[3]
                    .parse::<f32>()
                    .map_err(|e| ObjError::ParseError(e.to_string()))?;
                normals.push(Vec3::new(x, y, z).normalize_or_zero());
            }
            "vt" => {
                // Texture coordinate
                if parts.len() < 3 {
                    return Err(ObjError::InvalidLine(line.to_string()));
                }
                let u = parts[1]
                    .parse::<f32>()
                    .map_err(|e| ObjError::ParseError(e.to_string()))?;
                let v = parts[2]
                    .parse::<f32>()
                    .map_err(|e| ObjError::ParseError(e.to_string()))?;
                tex_coords.push(Vec2::new(u, v));
            }
            "f" => {
                // Face
                if parts.len() < 4 {
                    return Err(ObjError::InvalidFace(
                        "Face has fewer than 3 vertices".to_string(),
                    ));
                }

                // Parse face vertex indices
                let mut face_vertices: Vec<u32> = Vec::new();

                for part in &parts[1..] {
                    let indices = parse_face_vertex(part)?;
                    let pos_idx = indices.0;
                    let tex_idx = indices.1;
                    let norm_idx = indices.2;

                    // Validate indices (OBJ uses 1-based indexing)
                    if pos_idx == 0 || pos_idx > positions.len() {
                        return Err(ObjError::InvalidVertexIndex(pos_idx));
                    }
                    if let Some(ti) = tex_idx {
                        if ti == 0 || ti > tex_coords.len() {
                            return Err(ObjError::InvalidTexCoordIndex(ti));
                        }
                    }
                    if let Some(ni) = norm_idx {
                        if ni == 0 || ni > normals.len() {
                            return Err(ObjError::InvalidNormalIndex(ni));
                        }
                    }

                    // Get or create vertex
                    let key = (pos_idx, tex_idx, norm_idx);
                    let vertex_index = if let Some(&idx) = vertex_map.get(&key) {
                        idx
                    } else {
                        let idx = mesh_positions.len() as u32;
                        mesh_positions.push(positions[pos_idx - 1]);
                        if let Some(ti) = tex_idx {
                            mesh_uvs.push(tex_coords[ti - 1]);
                        }
                        if let Some(ni) = norm_idx {
                            mesh_normals.push(normals[ni - 1]);
                        }
                        vertex_map.insert(key, idx);
                        idx
                    };

                    face_vertices.push(vertex_index);
                }

                // Triangulate the face (fan triangulation)
                for i in 1..face_vertices.len() - 1 {
                    mesh_indices.push(face_vertices[0]);
                    mesh_indices.push(face_vertices[i]);
                    mesh_indices.push(face_vertices[i + 1]);
                }
            }
            // Ignore other commands (mtllib, usemtl, g, o, s, etc.)
            _ => {}
        }
    }

    // Build the mesh
    let mut mesh = Mesh::new();
    mesh.positions = mesh_positions;
    mesh.indices = mesh_indices;

    // Only add normals/uvs if we have them for all vertices
    if mesh_normals.len() == mesh.positions.len() {
        mesh.normals = mesh_normals;
    } else if !mesh.positions.is_empty() {
        // Compute normals if not provided
        mesh.compute_smooth_normals();
    }

    if mesh_uvs.len() == mesh.positions.len() {
        mesh.uvs = mesh_uvs;
    }

    Ok(mesh)
}

/// Parses a face vertex specification (e.g., "1/2/3" or "1//3" or "1").
fn parse_face_vertex(s: &str) -> Result<(usize, Option<usize>, Option<usize>), ObjError> {
    let parts: Vec<&str> = s.split('/').collect();

    let pos_idx = parts[0]
        .parse::<usize>()
        .map_err(|e| ObjError::ParseError(e.to_string()))?;

    let tex_idx = if parts.len() > 1 && !parts[1].is_empty() {
        Some(
            parts[1]
                .parse::<usize>()
                .map_err(|e| ObjError::ParseError(e.to_string()))?,
        )
    } else {
        None
    };

    let norm_idx = if parts.len() > 2 && !parts[2].is_empty() {
        Some(
            parts[2]
                .parse::<usize>()
                .map_err(|e| ObjError::ParseError(e.to_string()))?,
        )
    } else {
        None
    };

    Ok((pos_idx, tex_idx, norm_idx))
}

/// Exports a mesh to OBJ format string.
pub fn export_obj(mesh: &Mesh) -> String {
    export_obj_with_name(mesh, None)
}

/// Exports a mesh to OBJ format string with an optional object name.
pub fn export_obj_with_name(mesh: &Mesh, name: Option<&str>) -> String {
    let mut output = String::new();

    // Header
    writeln!(output, "# OBJ exported by resin").unwrap();
    if let Some(name) = name {
        writeln!(output, "o {}", name).unwrap();
    }
    writeln!(output).unwrap();

    // Vertex positions
    for pos in &mesh.positions {
        writeln!(output, "v {} {} {}", pos.x, pos.y, pos.z).unwrap();
    }
    writeln!(output).unwrap();

    // Texture coordinates
    let has_uvs = mesh.uvs.len() == mesh.positions.len();
    if has_uvs {
        for uv in &mesh.uvs {
            writeln!(output, "vt {} {}", uv.x, uv.y).unwrap();
        }
        writeln!(output).unwrap();
    }

    // Vertex normals
    let has_normals = mesh.normals.len() == mesh.positions.len();
    if has_normals {
        for normal in &mesh.normals {
            writeln!(output, "vn {} {} {}", normal.x, normal.y, normal.z).unwrap();
        }
        writeln!(output).unwrap();
    }

    // Faces (triangles)
    for tri in mesh.indices.chunks(3) {
        if tri.len() != 3 {
            continue;
        }

        // OBJ uses 1-based indexing
        let i0 = tri[0] + 1;
        let i1 = tri[1] + 1;
        let i2 = tri[2] + 1;

        match (has_uvs, has_normals) {
            (true, true) => {
                writeln!(
                    output,
                    "f {}/{}/{} {}/{}/{} {}/{}/{}",
                    i0, i0, i0, i1, i1, i1, i2, i2, i2
                )
                .unwrap();
            }
            (true, false) => {
                writeln!(output, "f {}/{} {}/{} {}/{}", i0, i0, i1, i1, i2, i2).unwrap();
            }
            (false, true) => {
                writeln!(output, "f {}//{} {}//{} {}//{}", i0, i0, i1, i1, i2, i2).unwrap();
            }
            (false, false) => {
                writeln!(output, "f {} {} {}", i0, i1, i2).unwrap();
            }
        }
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Cuboid;

    #[test]
    fn test_export_simple_triangle() {
        let mut mesh = Mesh::new();
        mesh.positions = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.5, 1.0, 0.0),
        ];
        mesh.indices = vec![0, 1, 2];

        let obj = export_obj(&mesh);
        assert!(obj.contains("v 0 0 0"));
        assert!(obj.contains("v 1 0 0"));
        assert!(obj.contains("v 0.5 1 0"));
        assert!(obj.contains("f 1 2 3"));
    }

    #[test]
    fn test_import_simple_triangle() {
        let obj = r#"
            # Simple triangle
            v 0 0 0
            v 1 0 0
            v 0.5 1 0
            f 1 2 3
        "#;

        let mesh = import_obj(obj).unwrap();
        assert_eq!(mesh.positions.len(), 3);
        assert_eq!(mesh.indices.len(), 3);
        assert_eq!(mesh.positions[0], Vec3::new(0.0, 0.0, 0.0));
        assert_eq!(mesh.positions[1], Vec3::new(1.0, 0.0, 0.0));
        assert_eq!(mesh.positions[2], Vec3::new(0.5, 1.0, 0.0));
    }

    #[test]
    fn test_import_with_normals() {
        let obj = r#"
            v 0 0 0
            v 1 0 0
            v 0.5 1 0
            vn 0 0 1
            f 1//1 2//1 3//1
        "#;

        let mesh = import_obj(obj).unwrap();
        assert_eq!(mesh.positions.len(), 3);
        assert_eq!(mesh.normals.len(), 3);
        // All normals should be (0, 0, 1)
        for normal in &mesh.normals {
            assert!((normal.z - 1.0).abs() < 0.001);
        }
    }

    #[test]
    fn test_import_with_uvs() {
        let obj = r#"
            v 0 0 0
            v 1 0 0
            v 0.5 1 0
            vt 0 0
            vt 1 0
            vt 0.5 1
            f 1/1 2/2 3/3
        "#;

        let mesh = import_obj(obj).unwrap();
        assert_eq!(mesh.positions.len(), 3);
        assert_eq!(mesh.uvs.len(), 3);
        assert_eq!(mesh.uvs[0], Vec2::new(0.0, 0.0));
        assert_eq!(mesh.uvs[1], Vec2::new(1.0, 0.0));
        assert_eq!(mesh.uvs[2], Vec2::new(0.5, 1.0));
    }

    #[test]
    fn test_import_quad_triangulation() {
        let obj = r#"
            v 0 0 0
            v 1 0 0
            v 1 1 0
            v 0 1 0
            f 1 2 3 4
        "#;

        let mesh = import_obj(obj).unwrap();
        assert_eq!(mesh.positions.len(), 4);
        // Quad should be triangulated into 2 triangles
        assert_eq!(mesh.indices.len(), 6);
    }

    #[test]
    fn test_roundtrip() {
        let original = Cuboid::default().apply();
        let obj = export_obj(&original);
        let imported = import_obj(&obj).unwrap();

        // Same number of triangles
        assert_eq!(imported.triangle_count(), original.triangle_count());
        // Positions should match (approximately, due to floating point)
        assert_eq!(imported.positions.len(), original.positions.len());
    }

    #[test]
    fn test_export_with_uvs_and_normals() {
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

        let obj = export_obj(&mesh);
        assert!(obj.contains("vt "));
        assert!(obj.contains("vn "));
        assert!(obj.contains("f 1/1/1 2/2/2 3/3/3"));
    }

    #[test]
    fn test_import_invalid_vertex_index() {
        let obj = r#"
            v 0 0 0
            f 1 2 3
        "#;

        let result = import_obj(obj);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            ObjError::InvalidVertexIndex(_)
        ));
    }

    #[test]
    fn test_import_comments_and_empty_lines() {
        let obj = r#"
            # This is a comment

            v 0 0 0
            # Another comment
            v 1 0 0

            v 0.5 1 0

            f 1 2 3
        "#;

        let mesh = import_obj(obj).unwrap();
        assert_eq!(mesh.positions.len(), 3);
        assert_eq!(mesh.triangle_count(), 1);
    }

    #[test]
    fn test_export_with_name() {
        let mesh = Cuboid::default().apply();
        let obj = export_obj_with_name(&mesh, Some("MyCube"));
        assert!(obj.contains("o MyCube"));
    }
}
