//! Temporal arrangement of clips: tracks, clip instances, and transitions.
//!
//! `Timeline` is a *value* — serializable arrangement data describing which clip
//! sources play when, on which tracks, with what transitions — not an op and not
//! itself evaluable. It plays the same role for time that `unshape-scatter`'s
//! `Instance` list plays for space: a [`ClipInstance`] references a source plus a
//! temporal transform (when it's active, what span of the source it draws from,
//! playback rate/loop mode) rather than copying the source's data, so one source
//! can appear at multiple points on the timeline.
//!
//! Clip sources are expected to be `Field<f32, T>` (`unshape-field-ops`): lazy,
//! evaluable at any time. With the `field` feature enabled, [`TimelineComposite`]
//! is the construction op that wires a `Timeline` together with its sources and
//! produces a composed `Field<f32, T>` — domain-agnostic over `T` (an `f32`
//! property, an `Image`, an `AudioBuffer`, ...) via the [`Compositable`] trait,
//! which each domain implements to say what "layer this clip over that one"
//! means for its own value type.
//!
//! # Example
//!
//! ```
//! use unshape_timeline::{ClipInstance, Compositable, SourceId, TimeRange, Timeline, Track};
//! use std::collections::HashMap;
//!
//! let timeline = Timeline::from_tracks(vec![Track::from_clips(vec![
//!     ClipInstance::new("intro", TimeRange::new(0.0, 2.0)),
//!     ClipInstance::new("main", TimeRange::new(2.0, 10.0)),
//! ])]);
//!
//! let sources = HashMap::from([
//!     (SourceId::new("intro"), 1.0_f32),
//!     (SourceId::new("main"), 2.0_f32),
//! ]);
//! let mut lookup = |id: &SourceId, _source_t: f32| sources.get(id).copied();
//!
//! assert_eq!(timeline.sample(1.0, &mut lookup), 1.0);
//! assert_eq!(timeline.sample(5.0, &mut lookup), 2.0);
//! ```

mod clip;
mod compositable;
#[cfg(feature = "field")]
mod composite;
mod time_map;
mod time_range;
mod timeline;
mod track;
mod transition;

pub use clip::{ClipInstance, SourceId};
pub use compositable::Compositable;
#[cfg(feature = "field")]
pub use composite::TimelineComposite;
pub use time_map::{LoopMode, TimeMap};
pub use time_range::TimeRange;
pub use timeline::Timeline;
pub use track::Track;
pub use transition::{Transition, TransitionKind};
