//! Node registry for type-based deserialization.

use crate::error::SerdeError;
use unshape_core::{BoxedNode, DynNode};
use serde::de::DeserializeOwned;
use serde_json::Value as JsonValue;
use std::collections::HashMap;

/// Trait for nodes that can be serialized.
///
/// Extends `DynNode` with the ability to extract parameters as JSON.
/// Implement this trait to enable serialization for your node types.
pub trait SerializableNode: DynNode {
    /// Extract node parameters as a JSON value.
    ///
    /// The returned JSON should contain all state needed to reconstruct
    /// the node via deserialization.
    fn params(&self) -> JsonValue;
}

/// Type alias for node factory functions.
type NodeFactory = Box<dyn Fn(JsonValue) -> Result<BoxedNode, SerdeError> + Send + Sync>;

/// Registry mapping type names to node deserializers.
///
/// Used during deserialization to reconstruct nodes from their type names
/// and JSON parameters.
pub struct NodeRegistry {
    factories: HashMap<String, NodeFactory>,
}

impl NodeRegistry {
    /// Creates an empty registry.
    pub fn new() -> Self {
        Self {
            factories: HashMap::new(),
        }
    }

    /// Registers a node type.
    ///
    /// The type's `type_name()` is used as the key.
    ///
    /// # Type Parameters
    /// * `N` - Node type that implements `SerializableNode`, `DeserializeOwned`, and `'static`
    pub fn register<N>(&mut self)
    where
        N: SerializableNode + DeserializeOwned + 'static,
    {
        // Create a temporary instance to get the type name
        // This requires Default, but we can work around it
        let type_name = std::any::type_name::<N>().to_string();
        self.register_with_name::<N>(&type_name);
    }

    /// Registers a node type with an explicit type name.
    ///
    /// Use this when you want to control the serialized type name,
    /// or for plugin nodes.
    pub fn register_with_name<N>(&mut self, type_name: &str)
    where
        N: SerializableNode + DeserializeOwned + 'static,
    {
        let name = type_name.to_string();
        self.factories.insert(
            name,
            Box::new(|params| {
                let node: N = serde_json::from_value(params)?;
                Ok(Box::new(node) as BoxedNode)
            }),
        );
    }

    /// Registers a node type using a custom factory function.
    ///
    /// Use this for complex deserialization logic.
    pub fn register_factory<F>(&mut self, type_name: &str, factory: F)
    where
        F: Fn(JsonValue) -> Result<BoxedNode, SerdeError> + Send + Sync + 'static,
    {
        self.factories
            .insert(type_name.to_string(), Box::new(factory));
    }

    /// Deserializes a node by type name and parameters.
    ///
    /// Returns an error if the type name is not registered.
    pub fn deserialize(&self, type_name: &str, params: JsonValue) -> Result<BoxedNode, SerdeError> {
        let factory = self
            .factories
            .get(type_name)
            .ok_or_else(|| SerdeError::UnknownNodeType(type_name.to_string()))?;
        factory(params)
    }

    /// Checks if a type name is registered.
    pub fn contains(&self, type_name: &str) -> bool {
        self.factories.contains_key(type_name)
    }

    /// Returns an iterator over all registered type names.
    pub fn registered_types(&self) -> impl Iterator<Item = &str> {
        self.factories.keys().map(String::as_str)
    }

    /// Returns the number of registered types.
    pub fn len(&self) -> usize {
        self.factories.len()
    }

    /// Returns true if no types are registered.
    pub fn is_empty(&self) -> bool {
        self.factories.is_empty()
    }
}

impl Default for NodeRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use unshape_core::{EvalContext, GraphError, PortDescriptor, Value, ValueType};
    use serde::{Deserialize, Serialize};

    #[derive(Debug, Clone, Serialize, Deserialize)]
    struct TestNode {
        value: f32,
    }

    impl DynNode for TestNode {
        fn type_name(&self) -> &'static str {
            "test::TestNode"
        }

        fn inputs(&self) -> Vec<PortDescriptor> {
            vec![]
        }

        fn outputs(&self) -> Vec<PortDescriptor> {
            vec![PortDescriptor::new("out", ValueType::F32)]
        }

        fn execute(&self, _inputs: &[Value], _ctx: &EvalContext) -> Result<Vec<Value>, GraphError> {
            Ok(vec![Value::F32(self.value)])
        }

        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
    }

    impl SerializableNode for TestNode {
        fn params(&self) -> JsonValue {
            serde_json::to_value(self).unwrap()
        }
    }

    #[test]
    fn test_registry_register_and_deserialize() {
        let mut registry = NodeRegistry::new();
        registry.register_with_name::<TestNode>("test::TestNode");

        let params = serde_json::json!({"value": 42.0});
        let node = registry.deserialize("test::TestNode", params).unwrap();

        assert_eq!(node.type_name(), "test::TestNode");

        let ctx = EvalContext::new();
        let outputs = node.execute(&[], &ctx).unwrap();
        assert_eq!(outputs[0].as_f32().unwrap(), 42.0);
    }

    #[test]
    fn test_registry_unknown_type() {
        let registry = NodeRegistry::new();
        let result = registry.deserialize("unknown::Type", serde_json::json!({}));
        assert!(matches!(result, Err(SerdeError::UnknownNodeType(_))));
    }

    #[test]
    fn test_registry_contains() {
        let mut registry = NodeRegistry::new();
        assert!(!registry.contains("test::TestNode"));

        registry.register_with_name::<TestNode>("test::TestNode");
        assert!(registry.contains("test::TestNode"));
    }

    #[test]
    fn test_registry_registered_types() {
        let mut registry = NodeRegistry::new();
        registry.register_with_name::<TestNode>("test::A");
        registry.register_with_name::<TestNode>("test::B");

        let types: Vec<_> = registry.registered_types().collect();
        assert_eq!(types.len(), 2);
        assert!(types.contains(&"test::A"));
        assert!(types.contains(&"test::B"));
    }

    #[test]
    fn test_registry_custom_factory() {
        let mut registry = NodeRegistry::new();
        registry.register_factory("custom::Node", |params| {
            let value = params["value"].as_f64().unwrap_or(0.0) as f32;
            Ok(Box::new(TestNode { value }))
        });

        let node = registry
            .deserialize("custom::Node", serde_json::json!({"value": 123.0}))
            .unwrap();
        let ctx = EvalContext::new();
        let outputs = node.execute(&[], &ctx).unwrap();
        assert_eq!(outputs[0].as_f32().unwrap(), 123.0);
    }
}
