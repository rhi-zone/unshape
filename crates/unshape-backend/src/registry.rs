//! Backend registry for managing available backends.

use crate::backend::{BackendKind, ComputeBackend};
use unshape_core::DynNode;
use std::sync::Arc;

/// Registry of available compute backends.
///
/// Backends register themselves at startup. The scheduler queries
/// the registry to find backends that can execute each node.
///
/// # Example
///
/// ```
/// use unshape_backend::{BackendRegistry, CpuBackend};
/// use std::sync::Arc;
///
/// let mut registry = BackendRegistry::new();
///
/// // CPU backend is always available
/// registry.register(Arc::new(CpuBackend));
///
/// // Register GPU backend if available
/// // if let Ok(gpu) = GpuBackend::new() {
/// //     registry.register(Arc::new(gpu));
/// // }
///
/// assert_eq!(registry.len(), 1);
/// assert!(registry.get("cpu").is_some());
/// ```
#[derive(Default)]
pub struct BackendRegistry {
    backends: Vec<Arc<dyn ComputeBackend>>,
}

impl BackendRegistry {
    /// Creates an empty registry.
    pub fn new() -> Self {
        Self {
            backends: Vec::new(),
        }
    }

    /// Creates a registry with only the CPU backend.
    pub fn with_cpu() -> Self {
        let mut registry = Self::new();
        registry.register(Arc::new(crate::CpuBackend));
        registry
    }

    /// Registers a backend.
    ///
    /// Backends are tried in registration order for automatic selection.
    pub fn register(&mut self, backend: Arc<dyn ComputeBackend>) {
        self.backends.push(backend);
    }

    /// Returns the backend with the given name.
    pub fn get(&self, name: &str) -> Option<&Arc<dyn ComputeBackend>> {
        self.backends.iter().find(|b| b.name() == name)
    }

    /// Returns an iterator over all registered backends.
    pub fn iter(&self) -> impl Iterator<Item = &Arc<dyn ComputeBackend>> {
        self.backends.iter()
    }

    /// Returns the number of registered backends.
    pub fn len(&self) -> usize {
        self.backends.len()
    }

    /// Returns `true` if no backends are registered.
    pub fn is_empty(&self) -> bool {
        self.backends.is_empty()
    }

    /// Returns backends that can execute the given node.
    pub fn backends_for_node(&self, node: &dyn DynNode) -> Vec<&Arc<dyn ComputeBackend>> {
        self.backends
            .iter()
            .filter(|b| b.supports_node(node))
            .collect()
    }

    /// Returns backends of the given kind.
    pub fn backends_of_kind(&self, kind: &BackendKind) -> Vec<&Arc<dyn ComputeBackend>> {
        self.backends
            .iter()
            .filter(|b| &b.capabilities().kind == kind)
            .collect()
    }

    /// Returns the first backend that supports the given node, if any.
    pub fn first_supporting(&self, node: &dyn DynNode) -> Option<&Arc<dyn ComputeBackend>> {
        self.backends.iter().find(|b| b.supports_node(node))
    }

    /// Returns backend names for debugging.
    pub fn backend_names(&self) -> Vec<&str> {
        self.backends.iter().map(|b| b.name()).collect()
    }
}

impl std::fmt::Debug for BackendRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BackendRegistry")
            .field("backends", &self.backend_names())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CpuBackend;

    #[test]
    fn test_registry_new() {
        let registry = BackendRegistry::new();
        assert!(registry.is_empty());
        assert_eq!(registry.len(), 0);
    }

    #[test]
    fn test_registry_with_cpu() {
        let registry = BackendRegistry::with_cpu();
        assert_eq!(registry.len(), 1);
        assert!(registry.get("cpu").is_some());
    }

    #[test]
    fn test_registry_register() {
        let mut registry = BackendRegistry::new();
        registry.register(Arc::new(CpuBackend));
        assert_eq!(registry.len(), 1);
    }

    #[test]
    fn test_registry_get() {
        let mut registry = BackendRegistry::new();
        registry.register(Arc::new(CpuBackend));

        assert!(registry.get("cpu").is_some());
        assert!(registry.get("nonexistent").is_none());
    }

    #[test]
    fn test_registry_iter() {
        let mut registry = BackendRegistry::new();
        registry.register(Arc::new(CpuBackend));

        let names: Vec<_> = registry.iter().map(|b| b.name()).collect();
        assert_eq!(names, vec!["cpu"]);
    }

    #[test]
    fn test_registry_backends_of_kind() {
        let mut registry = BackendRegistry::new();
        registry.register(Arc::new(CpuBackend));

        let cpu_backends = registry.backends_of_kind(&BackendKind::Cpu);
        assert_eq!(cpu_backends.len(), 1);

        let gpu_backends = registry.backends_of_kind(&BackendKind::Gpu);
        assert!(gpu_backends.is_empty());
    }

    #[test]
    fn test_registry_debug() {
        let registry = BackendRegistry::with_cpu();
        let debug_str = format!("{:?}", registry);
        assert!(debug_str.contains("cpu"));
    }
}
