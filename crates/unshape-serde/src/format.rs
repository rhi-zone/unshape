//! Format abstraction for graph serialization.

use crate::error::SerdeError;
use crate::serial::SerialGraph;

/// Trait for serialization formats.
///
/// Implementations convert between `SerialGraph` and bytes.
/// Built-in implementations: `JsonFormat`, `BincodeFormat`.
pub trait GraphFormat: Send + Sync {
    /// Serialize a graph to bytes.
    fn serialize(&self, graph: &SerialGraph) -> Result<Vec<u8>, SerdeError>;

    /// Deserialize bytes to a graph.
    fn deserialize(&self, bytes: &[u8]) -> Result<SerialGraph, SerdeError>;

    /// Human-readable format name (for logging/debugging).
    fn name(&self) -> &'static str;

    /// Suggested file extension (e.g., "json", "bin").
    fn extension(&self) -> &'static str;
}
