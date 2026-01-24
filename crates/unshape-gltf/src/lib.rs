//! glTF 2.0 import and export for resin meshes.
//!
//! Provides import and export functionality for glTF format (`.gltf` with embedded data or `.glb` binary).
//!
//! # Export Example
//!
//! ```ignore
//! use unshape_mesh::{Mesh, uv_sphere};
//! use unshape_gltf::{GltfExporter, ExportFormat};
//!
//! let mesh = uv_sphere(32, 16);
//! let exporter = GltfExporter::new().with_mesh("sphere", mesh);
//! exporter.export_glb("sphere.glb").unwrap();
//! ```
//!
//! # Import Example
//!
//! ```ignore
//! use unshape_gltf::import_gltf;
//!
//! let scene = import_gltf("model.glb").unwrap();
//! let mesh = scene.first_mesh().unwrap();
//! println!("Loaded {} vertices", mesh.positions.len());
//! ```

mod export;
mod import;

pub use export::{ExportFormat, GltfError, GltfExporter, GltfMaterial, GltfMesh, GltfResult};
pub use import::{GltfScene, import_gltf, import_gltf_from_bytes};
