//! Domain-specific typed constant nodes that support serialization.
//!
//! [`ConstantNode`] cannot serialize `Value::Opaque` values because opaque
//! payloads are type-erased. This module provides domain-specific constant
//! nodes where the concrete type is known, so full serialization is possible.

use std::any::Any;

use serde::{Deserialize, Serialize};
use unshape_core::{
    DynNode, EvalContext, GraphError, GraphValue, PortDescriptor, Value, ValueType,
};
use unshape_image::ImageField;
use unshape_mesh::Mesh;

use crate::registry::SerializableNode;

// ---------------------------------------------------------------------------
// MeshValue — a GraphValue wrapper for Mesh
//
// The orphan rule prevents `impl GraphValue for Mesh` here (Mesh is from
// unshape-mesh, GraphValue from unshape-core). We use a transparent newtype
// so Mesh can flow through the graph as an opaque value.
// ---------------------------------------------------------------------------

/// A transparent newtype that lets [`Mesh`] flow through the graph as an opaque value.
///
/// Implements [`GraphValue`] so it can be wrapped in [`Value::Opaque`].
/// Use `.0` or [`MeshValue::into_inner`] to recover the original mesh.
#[derive(Debug, Clone)]
pub struct MeshValue(pub Mesh);

impl MeshValue {
    /// Unwrap into the inner [`Mesh`].
    pub fn into_inner(self) -> Mesh {
        self.0
    }
}

impl GraphValue for MeshValue {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn type_name(&self) -> &'static str {
        "Mesh"
    }
}

// ---------------------------------------------------------------------------
// ConstantMesh
// ---------------------------------------------------------------------------

/// A source node that outputs a constant `Mesh` value wrapped in [`MeshValue`].
///
/// Unlike [`ConstantNode`], this node stores the concrete mesh type so it can be
/// fully serialized and deserialized. The output is a `Value::Opaque(MeshValue)`.
///
/// # Ports
/// - Inputs: none
/// - Outputs: `"mesh"` — `ValueType::Custom { name: "Mesh", … }`
///
/// # Serialization
///
/// Params are the full JSON representation of the mesh (positions, normals,
/// UVs, indices). Requires the `serde` feature on `unshape-mesh`.
///
/// # Downcasting
///
/// To extract the mesh from the output value:
/// ```ignore
/// let mesh_val = output.downcast_ref::<MeshValue>().unwrap();
/// let mesh = mesh_val.0.clone();
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstantMesh {
    /// The mesh value this node outputs.
    pub mesh: Mesh,
}

impl ConstantMesh {
    /// Create a new constant mesh node.
    pub fn new(mesh: Mesh) -> Self {
        Self { mesh }
    }
}

impl DynNode for ConstantMesh {
    fn type_name(&self) -> &'static str {
        "mesh::ConstantMesh"
    }

    fn inputs(&self) -> Vec<PortDescriptor> {
        vec![]
    }

    fn outputs(&self) -> Vec<PortDescriptor> {
        vec![PortDescriptor::new(
            "mesh",
            ValueType::Custom {
                type_id: std::any::TypeId::of::<MeshValue>(),
                name: "Mesh",
            },
        )]
    }

    fn execute(&self, _inputs: &[Value], _ctx: &EvalContext) -> Result<Vec<Value>, GraphError> {
        Ok(vec![Value::opaque(MeshValue(self.mesh.clone()))])
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl SerializableNode for ConstantMesh {
    fn params(&self) -> serde_json::Value {
        serde_json::to_value(self)
            .unwrap_or_else(|e| serde_json::json!({ "__error": e.to_string() }))
    }
}

// ---------------------------------------------------------------------------
// ImageValue — a GraphValue wrapper for ImageField
// ---------------------------------------------------------------------------

/// A transparent newtype that lets [`ImageField`] flow through the graph as an opaque value.
///
/// Implements [`GraphValue`] so it can be wrapped in [`Value::Opaque`].
/// Use `.0` or [`ImageValue::into_inner`] to recover the original image field.
#[derive(Debug, Clone)]
pub struct ImageValue(pub ImageField);

impl ImageValue {
    /// Unwrap into the inner [`ImageField`].
    pub fn into_inner(self) -> ImageField {
        self.0
    }
}

impl GraphValue for ImageValue {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn type_name(&self) -> &'static str {
        "ImageField"
    }
}

// ---------------------------------------------------------------------------
// ConstantImage
// ---------------------------------------------------------------------------

/// A source node that outputs a constant [`ImageField`] value wrapped in [`ImageValue`].
///
/// Unlike [`ConstantNode`], this node stores the concrete image type so it can be
/// fully serialized and deserialized. The output is a `Value::Opaque(ImageValue)`.
///
/// # Ports
/// - Inputs: none
/// - Outputs: `"image"` — `ValueType::Custom { name: "ImageField", … }`
///
/// # Serialization
///
/// Params are the full JSON representation of the image (pixel data, width, height,
/// wrap mode, filter mode). Requires the `serde` feature on `unshape-image`.
///
/// # Downcasting
///
/// To extract the image from the output value:
/// ```ignore
/// let image_val = output.downcast_ref::<ImageValue>().unwrap();
/// let image = image_val.0.clone();
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstantImage {
    /// The image field value this node outputs.
    pub image: ImageField,
}

impl ConstantImage {
    /// Create a new constant image node.
    pub fn new(image: ImageField) -> Self {
        Self { image }
    }
}

impl DynNode for ConstantImage {
    fn type_name(&self) -> &'static str {
        "image::ConstantImage"
    }

    fn inputs(&self) -> Vec<PortDescriptor> {
        vec![]
    }

    fn outputs(&self) -> Vec<PortDescriptor> {
        vec![PortDescriptor::new(
            "image",
            ValueType::Custom {
                type_id: std::any::TypeId::of::<ImageValue>(),
                name: "ImageField",
            },
        )]
    }

    fn execute(&self, _inputs: &[Value], _ctx: &EvalContext) -> Result<Vec<Value>, GraphError> {
        Ok(vec![Value::opaque(ImageValue(self.image.clone()))])
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl SerializableNode for ConstantImage {
    fn params(&self) -> serde_json::Value {
        serde_json::to_value(self)
            .unwrap_or_else(|e| serde_json::json!({ "__error": e.to_string() }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use unshape_mesh::UvSphere;

    #[test]
    fn test_constant_mesh_execute() {
        let mesh = UvSphere::new(1.0, 8, 8).apply();
        let vertex_count = mesh.positions.len();
        let node = ConstantMesh::new(mesh);

        let ctx = EvalContext::new();
        let outputs = node.execute(&[], &ctx).unwrap();
        assert_eq!(outputs.len(), 1);

        let extracted = outputs[0]
            .downcast_ref::<MeshValue>()
            .expect("should be a MeshValue");
        assert_eq!(extracted.0.positions.len(), vertex_count);
    }

    #[test]
    fn test_constant_mesh_ports() {
        let node = ConstantMesh::new(Mesh::default());
        assert!(node.inputs().is_empty());
        assert_eq!(node.outputs().len(), 1);
        assert_eq!(node.outputs()[0].name, "mesh");
    }

    #[test]
    fn test_constant_mesh_params_roundtrip() {
        let mesh = UvSphere::new(1.0, 4, 4).apply();
        let node = ConstantMesh::new(mesh.clone());
        let params = node.params();

        // Deserialize back
        let restored: ConstantMesh = serde_json::from_value(params).unwrap();
        assert_eq!(restored.mesh.positions.len(), mesh.positions.len());
    }

    #[test]
    fn test_constant_image_execute() {
        let image = ImageField::from_raw(vec![[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0]], 2, 1);
        let pixel_count = image.data.len();
        let node = ConstantImage::new(image);

        let ctx = EvalContext::new();
        let outputs = node.execute(&[], &ctx).unwrap();
        assert_eq!(outputs.len(), 1);

        let extracted = outputs[0]
            .downcast_ref::<ImageValue>()
            .expect("should be an ImageValue");
        assert_eq!(extracted.0.data.len(), pixel_count);
    }

    #[test]
    fn test_constant_image_ports() {
        let image = ImageField::from_raw(vec![[0.0, 0.0, 0.0, 1.0]], 1, 1);
        let node = ConstantImage::new(image);
        assert!(node.inputs().is_empty());
        assert_eq!(node.outputs().len(), 1);
        assert_eq!(node.outputs()[0].name, "image");
    }

    #[test]
    fn test_constant_image_params_roundtrip() {
        let image = ImageField::from_raw(vec![[0.5, 0.5, 0.5, 1.0], [0.1, 0.2, 0.3, 0.4]], 2, 1);
        let original_len = image.data.len();
        let node = ConstantImage::new(image);
        let params = node.params();

        // Deserialize back
        let restored: ConstantImage = serde_json::from_value(params).unwrap();
        assert_eq!(restored.image.data.len(), original_len);
        assert_eq!(restored.image.width, 2);
        assert_eq!(restored.image.height, 1);
    }
}
