//! 3D mesh generation for resin.
//!
//! Provides mesh primitives and operations for procedural 3D geometry.

mod mesh;
mod primitives;

pub use mesh::{Mesh, MeshBuilder};
pub use primitives::{box_mesh, sphere, uv_sphere};
