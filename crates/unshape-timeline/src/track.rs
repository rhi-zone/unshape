//! An ordered layer of clips, plus the transitions between them.

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::{ClipInstance, Compositable, SourceId, Transition, TransitionKind};

/// An ordered layer of [`ClipInstance`]s, like a video/audio track in an NLE.
/// Tracks composite top-down: later tracks in a [`crate::Timeline`] are drawn
/// over earlier ones.
#[derive(Debug, Clone, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Track {
    /// The clips on this track. May overlap in time to describe a transition.
    pub clips: Vec<ClipInstance>,
    /// Transitions between overlapping clips on this track.
    pub transitions: Vec<Transition>,
}

impl Track {
    /// Creates an empty track.
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a track from a list of non-overlapping clips with no transitions.
    pub fn from_clips(clips: Vec<ClipInstance>) -> Self {
        Self {
            clips,
            transitions: Vec::new(),
        }
    }

    /// The clips active at timeline time `t`, in `clips` order.
    pub fn active_clips_at(&self, t: f32) -> impl Iterator<Item = &ClipInstance> {
        self.clips.iter().filter(move |clip| clip.is_active_at(t))
    }

    /// The transition (if any) covering timeline time `t`.
    pub fn transition_at(&self, t: f32) -> Option<&Transition> {
        self.transitions.iter().find(|tr| tr.contains(t))
    }

    /// Samples this track at timeline time `t`, resolving each active clip's
    /// source time through `sample_source` (which looks up a [`SourceId`] and
    /// samples its `Field` at the given source-local time) and blending
    /// overlapping clips per this track's [`Transition`]s.
    ///
    /// Returns `None` if no clip is active at `t`, or if `sample_source` returns
    /// `None` for the only active clip.
    pub fn sample<T: Compositable>(
        &self,
        t: f32,
        sample_source: &mut dyn FnMut(&SourceId, f32) -> Option<T>,
    ) -> Option<T> {
        let mut active: Vec<&ClipInstance> = self.active_clips_at(t).collect();
        match active.len() {
            0 => None,
            1 => {
                let clip = active[0];
                let source_t = clip.source_time_at(t)?;
                sample_source(&clip.source, source_t)
            }
            _ => {
                // Overlap: order by start time, oldest (outgoing) first.
                active.sort_by(|a, b| a.timeline_range.start.total_cmp(&b.timeline_range.start));
                let outgoing = active[active.len() - 2];
                let incoming = active[active.len() - 1];

                let transition = self.transition_at(t);
                let kind = transition.map(|tr| tr.kind).unwrap_or(TransitionKind::Cut);

                match kind {
                    TransitionKind::Cut => {
                        let source_t = incoming.source_time_at(t)?;
                        sample_source(&incoming.source, source_t)
                    }
                    TransitionKind::CrossDissolve => {
                        let weight = transition.map(|tr| tr.weight_at(t)).unwrap_or(1.0);
                        let out_t = outgoing.source_time_at(t);
                        let in_t = incoming.source_time_at(t);
                        match (out_t, in_t) {
                            (Some(out_t), Some(in_t)) => {
                                let out_val = sample_source(&outgoing.source, out_t);
                                let in_val = sample_source(&incoming.source, in_t);
                                match (out_val, in_val) {
                                    (Some(out_val), Some(in_val)) => {
                                        Some(T::composite(&out_val, &in_val, weight))
                                    }
                                    (Some(out_val), None) => Some(out_val),
                                    (None, Some(in_val)) => Some(in_val),
                                    (None, None) => None,
                                }
                            }
                            (Some(out_t), None) => sample_source(&outgoing.source, out_t),
                            (None, Some(in_t)) => sample_source(&incoming.source, in_t),
                            (None, None) => None,
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TimeRange;
    use std::collections::HashMap;

    fn sources() -> HashMap<SourceId, f32> {
        HashMap::from([(SourceId::new("a"), 1.0), (SourceId::new("b"), 2.0)])
    }

    #[test]
    fn empty_track_samples_none() {
        let track = Track::new();
        let src = sources();
        let mut lookup = |id: &SourceId, _t: f32| src.get(id).copied();
        assert_eq!(track.sample::<f32>(0.0, &mut lookup), None);
    }

    #[test]
    fn single_clip_samples_constant_source() {
        let track = Track::from_clips(vec![ClipInstance::new("a", TimeRange::new(0.0, 2.0))]);
        let src = sources();
        let mut lookup = |id: &SourceId, _t: f32| src.get(id).copied();
        assert_eq!(track.sample::<f32>(1.0, &mut lookup), Some(1.0));
        assert_eq!(track.sample::<f32>(2.0, &mut lookup), None);
    }

    #[test]
    fn cross_dissolve_blends_overlap() {
        let track = Track {
            clips: vec![
                ClipInstance::new("a", TimeRange::new(0.0, 2.0)),
                ClipInstance::new("b", TimeRange::new(1.0, 3.0)),
            ],
            transitions: vec![Transition::cross_dissolve(1.0, 1.0)],
        };
        let src = sources();
        let mut lookup = |id: &SourceId, _t: f32| src.get(id).copied();
        assert_eq!(track.sample::<f32>(1.0, &mut lookup), Some(1.0));
        assert_eq!(track.sample::<f32>(1.5, &mut lookup), Some(1.5));
        assert_eq!(track.sample::<f32>(2.0, &mut lookup), Some(2.0));
    }

    #[test]
    fn cut_without_transition_prefers_incoming() {
        let track = Track {
            clips: vec![
                ClipInstance::new("a", TimeRange::new(0.0, 2.0)),
                ClipInstance::new("b", TimeRange::new(1.0, 3.0)),
            ],
            transitions: Vec::new(),
        };
        let src = sources();
        let mut lookup = |id: &SourceId, _t: f32| src.get(id).copied();
        assert_eq!(track.sample::<f32>(1.5, &mut lookup), Some(2.0));
    }
}
