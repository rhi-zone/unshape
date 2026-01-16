//! Execution policy for backend selection.

use crate::backend::BackendKind;

/// Policy for choosing execution backends.
///
/// Policies express intent, not specific backends. The scheduler
/// interprets the policy given available backends and node requirements.
///
/// # Examples
///
/// ```
/// use rhizome_resin_backend::{ExecutionPolicy, BackendKind};
///
/// // Let the scheduler decide
/// let auto = ExecutionPolicy::Auto;
///
/// // Prefer GPU when available
/// let gpu_preferred = ExecutionPolicy::PreferKind(BackendKind::Gpu);
///
/// // Use a specific backend
/// let explicit = ExecutionPolicy::Named("my-cuda-backend".into());
///
/// // Minimize data movement
/// let local = ExecutionPolicy::LocalFirst;
/// ```
#[derive(Clone, Debug, Default)]
pub enum ExecutionPolicy {
    /// Let the scheduler choose based on workload and capabilities.
    ///
    /// The scheduler uses heuristics:
    /// - Small workloads → CPU (avoid GPU overhead)
    /// - Large bulk workloads → GPU (if available)
    /// - Streaming/low-latency → CPU
    #[default]
    Auto,

    /// Prefer backends of the given kind.
    ///
    /// Falls back to other backends if preferred kind is unavailable
    /// or doesn't support the node.
    PreferKind(BackendKind),

    /// Use a specific backend by name.
    ///
    /// Fails if the named backend doesn't exist or doesn't support the node.
    /// Use for explicit control when you know what you want.
    Named(String),

    /// Minimize data transfers by keeping data where it is.
    ///
    /// If input data is on GPU, prefer GPU execution.
    /// If input data is on CPU, prefer CPU execution.
    /// Good for pipelines where the same data flows through many nodes.
    LocalFirst,

    /// Choose the backend with lowest estimated cost.
    ///
    /// Uses [`Cost`](crate::Cost) estimates from each backend.
    /// Most accurate but requires good cost models.
    MinimizeCost,
}

impl ExecutionPolicy {
    /// Returns `true` if this policy allows automatic backend selection.
    pub fn is_automatic(&self) -> bool {
        matches!(
            self,
            ExecutionPolicy::Auto | ExecutionPolicy::LocalFirst | ExecutionPolicy::MinimizeCost
        )
    }

    /// Returns the preferred backend kind, if any.
    pub fn preferred_kind(&self) -> Option<&BackendKind> {
        match self {
            ExecutionPolicy::PreferKind(kind) => Some(kind),
            _ => None,
        }
    }

    /// Returns the required backend name, if any.
    pub fn required_name(&self) -> Option<&str> {
        match self {
            ExecutionPolicy::Named(name) => Some(name),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_policy_default() {
        let policy: ExecutionPolicy = Default::default();
        assert!(matches!(policy, ExecutionPolicy::Auto));
    }

    #[test]
    fn test_policy_is_automatic() {
        assert!(ExecutionPolicy::Auto.is_automatic());
        assert!(ExecutionPolicy::LocalFirst.is_automatic());
        assert!(ExecutionPolicy::MinimizeCost.is_automatic());
        assert!(!ExecutionPolicy::Named("foo".into()).is_automatic());
        assert!(!ExecutionPolicy::PreferKind(BackendKind::Gpu).is_automatic());
    }

    #[test]
    fn test_policy_preferred_kind() {
        assert_eq!(
            ExecutionPolicy::PreferKind(BackendKind::Gpu).preferred_kind(),
            Some(&BackendKind::Gpu)
        );
        assert_eq!(ExecutionPolicy::Auto.preferred_kind(), None);
    }

    #[test]
    fn test_policy_required_name() {
        assert_eq!(
            ExecutionPolicy::Named("cuda".into()).required_name(),
            Some("cuda")
        );
        assert_eq!(ExecutionPolicy::Auto.required_name(), None);
    }
}
