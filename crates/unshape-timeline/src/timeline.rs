//! The temporal arrangement value: an ordered list of tracks.

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::{Compositable, SourceId, Track};

/// A temporal arrangement of clips across tracks — the value edited, serialized,
/// and passed around as data (e.g. as a construction op's parameter), the same
/// way a `Subdivide { levels }` op's `levels` is a plain value rather than an op
/// in its own right.
///
/// `Timeline` only ever says which clip is active when; it stays domain-agnostic
/// over what the clips actually contain.
#[derive(Debug, Clone, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Timeline {
    /// Tracks in bottom-to-top compositing order: later tracks draw over earlier ones.
    pub tracks: Vec<Track>,
}

impl Timeline {
    /// Creates an empty timeline.
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a timeline from a list of tracks.
    pub fn from_tracks(tracks: Vec<Track>) -> Self {
        Self { tracks }
    }

    /// The timeline's total duration: the latest `timeline_range.end` across every
    /// clip on every track. `0.0` if there are no clips.
    pub fn duration(&self) -> f32 {
        self.tracks
            .iter()
            .flat_map(|track| track.clips.iter())
            .map(|clip| clip.timeline_range.end)
            .fold(0.0, f32::max)
    }

    /// Samples the composed timeline at time `t`, resolving clip sources through
    /// `sample_source` and layering each track's result over the previous one
    /// (later tracks on top). Tracks with no active clip at `t` contribute nothing.
    ///
    /// Returns [`Compositable::empty`] if no track has an active clip at `t`.
    pub fn sample<T: Compositable>(
        &self,
        t: f32,
        sample_source: &mut dyn FnMut(&SourceId, f32) -> Option<T>,
    ) -> T {
        let mut accum = T::empty();
        for track in &self.tracks {
            if let Some(value) = track.sample(t, sample_source) {
                accum = T::composite(&accum, &value, 1.0);
            }
        }
        accum
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ClipInstance, TimeRange};
    use std::collections::HashMap;

    #[test]
    fn duration_spans_all_clips() {
        let timeline = Timeline::from_tracks(vec![
            Track::from_clips(vec![ClipInstance::new("a", TimeRange::new(0.0, 2.0))]),
            Track::from_clips(vec![ClipInstance::new("b", TimeRange::new(1.0, 5.0))]),
        ]);
        assert_eq!(timeline.duration(), 5.0);
    }

    #[test]
    fn empty_timeline_samples_empty() {
        let timeline = Timeline::new();
        let mut lookup = |_id: &SourceId, _t: f32| None;
        assert_eq!(timeline.sample::<f32>(0.0, &mut lookup), 0.0);
    }

    #[test]
    fn later_track_composites_over_earlier() {
        let timeline = Timeline::from_tracks(vec![
            Track::from_clips(vec![ClipInstance::new("bg", TimeRange::new(0.0, 5.0))]),
            Track::from_clips(vec![ClipInstance::new("fg", TimeRange::new(1.0, 2.0))]),
        ]);
        let src = HashMap::from([(SourceId::new("bg"), 1.0), (SourceId::new("fg"), 9.0)]);
        let mut lookup = |id: &SourceId, _t: f32| src.get(id).copied();

        // Only the background track is active.
        assert_eq!(timeline.sample::<f32>(0.5, &mut lookup), 1.0);
        // Foreground track composites over background.
        assert_eq!(timeline.sample::<f32>(1.5, &mut lookup), 9.0);
    }
}
