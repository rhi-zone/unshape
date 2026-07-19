//! Blending between adjacent or overlapping clips on a track.

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// How two overlapping clips on a track blend during a [`Transition`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum TransitionKind {
    /// No blending: the later clip replaces the earlier one instantly.
    Cut,
    /// Linearly cross-fades from the outgoing clip to the incoming clip.
    CrossDissolve,
}

/// Describes how a track blends between two clips whose `timeline_range`s overlap.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Transition {
    /// When the transition begins, in timeline time.
    pub at: f32,
    /// How long the transition lasts, in seconds.
    pub duration: f32,
    /// The blend behavior to apply.
    pub kind: TransitionKind,
}

impl Transition {
    /// Creates a cross-dissolve transition starting at `at` and lasting `duration` seconds.
    pub fn cross_dissolve(at: f32, duration: f32) -> Self {
        Self {
            at,
            duration,
            kind: TransitionKind::CrossDissolve,
        }
    }

    /// Returns `true` if timeline time `t` falls within this transition's span.
    pub fn contains(&self, t: f32) -> bool {
        t >= self.at && t < self.at + self.duration
    }

    /// The blend weight at time `t`, in `[0, 1]`, where `0` is fully the outgoing
    /// clip and `1` is fully the incoming clip. Clamped outside the transition's span.
    pub fn weight_at(&self, t: f32) -> f32 {
        if self.duration <= 0.0 {
            return 1.0;
        }
        ((t - self.at) / self.duration).clamp(0.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn weight_ramps_across_span() {
        let tr = Transition::cross_dissolve(1.0, 2.0);
        assert_eq!(tr.weight_at(1.0), 0.0);
        assert_eq!(tr.weight_at(2.0), 0.5);
        assert_eq!(tr.weight_at(3.0), 1.0);
        assert_eq!(tr.weight_at(0.0), 0.0);
        assert_eq!(tr.weight_at(10.0), 1.0);
    }

    #[test]
    fn contains_is_half_open() {
        let tr = Transition::cross_dissolve(1.0, 2.0);
        assert!(tr.contains(1.0));
        assert!(!tr.contains(3.0));
    }
}
