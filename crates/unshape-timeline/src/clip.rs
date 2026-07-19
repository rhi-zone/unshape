//! Temporal instances of clip sources.

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::{TimeMap, TimeRange};

/// Identifies a clip source (a `Field<f32, T>` input) by name.
///
/// Named rather than indexed so one source can be referenced by multiple
/// [`ClipInstance`]s without duplicating data — the same pattern as
/// `unshape-scatter`'s instances referencing shared geometry by id.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct SourceId(pub String);

impl SourceId {
    /// Creates a new source id from anything convertible to a `String`.
    pub fn new(name: impl Into<String>) -> Self {
        Self(name.into())
    }
}

impl From<&str> for SourceId {
    fn from(value: &str) -> Self {
        Self(value.to_string())
    }
}

impl From<String> for SourceId {
    fn from(value: String) -> Self {
        Self(value)
    }
}

/// A temporal instance of a clip source: a reference to a source plus a temporal
/// transform (when it's active, what span of the source it draws from, and how
/// source time is derived from timeline time).
///
/// This is the temporal analogue of `unshape-scatter`'s spatial `Instance` — a
/// clip doesn't copy its source's data, it references it plus a transform.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ClipInstance {
    /// Which source this instance samples.
    pub source: SourceId,
    /// When this clip is active on the timeline.
    pub timeline_range: TimeRange,
    /// The span of the source's own local time to draw from.
    pub source_range: TimeRange,
    /// How timeline-local time (time since `timeline_range.start`) maps into
    /// source-local time within `source_range`.
    pub time_map: TimeMap,
}

impl ClipInstance {
    /// Creates a clip instance that plays `source` at unit speed for the full
    /// duration of `timeline_range`, drawing from the same span of source time.
    pub fn new(source: impl Into<SourceId>, timeline_range: TimeRange) -> Self {
        Self {
            source: source.into(),
            timeline_range,
            source_range: TimeRange::new(0.0, timeline_range.duration()),
            time_map: TimeMap::default(),
        }
    }

    /// Returns `true` if this clip is active at timeline time `t`.
    pub fn is_active_at(&self, t: f32) -> bool {
        self.timeline_range.contains(t)
    }

    /// Resolves timeline time `t` into a time within the source's own local time,
    /// or `None` if the clip isn't active at `t`.
    pub fn source_time_at(&self, t: f32) -> Option<f32> {
        if !self.is_active_at(t) {
            return None;
        }
        let local_t = t - self.timeline_range.start;
        let resolved = self.time_map.resolve(local_t, self.source_range.duration());
        Some(self.source_range.start + resolved)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_time_none_outside_range() {
        let clip = ClipInstance::new("a", TimeRange::new(1.0, 3.0));
        assert_eq!(clip.source_time_at(0.0), None);
        assert_eq!(clip.source_time_at(3.0), None);
    }

    #[test]
    fn source_time_maps_local_time() {
        let clip = ClipInstance::new("a", TimeRange::new(1.0, 3.0));
        assert_eq!(clip.source_time_at(1.0), Some(0.0));
        assert_eq!(clip.source_time_at(2.0), Some(1.0));
    }

    #[test]
    fn source_range_offsets_into_source() {
        let clip = ClipInstance {
            source: SourceId::new("a"),
            timeline_range: TimeRange::new(0.0, 2.0),
            source_range: TimeRange::new(5.0, 7.0),
            time_map: TimeMap::default(),
        };
        assert_eq!(clip.source_time_at(0.0), Some(5.0));
        assert_eq!(clip.source_time_at(1.5), Some(6.5));
    }
}
