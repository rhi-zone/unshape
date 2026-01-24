//! History tracking for resin graphs.
//!
//! This crate provides two approaches to history management:
//!
//! - **Snapshots**: Store full graph state at each checkpoint. Simple and reliable.
//! - **Event sourcing**: Record individual modifications. More flexible, smaller storage.
//!
//! # Choosing an Approach
//!
//! | Aspect | Snapshots | Events |
//! |--------|-----------|--------|
//! | Storage | Larger (full graph per snapshot) | Smaller (only changes) |
//! | Complexity | Simple | More complex |
//! | Undo/Redo | Instant (swap graphs) | Apply inverse events |
//! | Audit trail | No | Yes (every action recorded) |
//! | Replay | No | Yes |
//!
//! For most applications, start with snapshots. Use event sourcing when you need
//! detailed history, audit trails, or collaborative editing.
//!
//! # Example: Snapshots
//!
//! ```ignore
//! use resin_history::{SnapshotHistory, SnapshotConfig};
//!
//! let mut history = SnapshotHistory::new(SnapshotConfig::default());
//!
//! // After each user action
//! history.record(&graph, extract_params)?;
//!
//! // Undo
//! if let Some(prev) = history.undo(&registry)? {
//!     graph = prev;
//! }
//! ```
//!
//! # Example: Event Sourcing
//!
//! ```ignore
//! use resin_history::{EventHistory, GraphEvent};
//!
//! let mut history = EventHistory::new();
//!
//! // Record events
//! history.record(GraphEvent::add_node(0, "MyNode", params), &mut graph, &registry)?;
//!
//! // Undo
//! history.undo(&mut graph, &registry)?;
//! ```

mod error;
mod event;
mod snapshot;

pub use error::HistoryError;
pub use event::{EventHistory, GraphEvent, StampedEvent};
pub use snapshot::{SnapshotConfig, SnapshotHistory};
