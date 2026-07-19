//! Mapping from a clip's local timeline-relative time into its source's own time.

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// How a clip's source time behaves once the mapped time runs past the source's
/// available duration (or before its start, for negative rates).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum LoopMode {
    /// Play through once; time past the end freezes at the last sample.
    #[default]
    Once,
    /// Wrap back to the start, repeating indefinitely.
    Loop,
    /// Bounce back and forth between start and end.
    PingPong,
    /// Ignore elapsed time entirely; always sample at `offset` (a held still frame).
    Hold,
}

/// Maps a clip's local timeline-relative time (0 at the start of the clip's
/// [`TimeRange`](crate::TimeRange)) into a time within its source's own local time,
/// honoring playback rate, a time offset, and a loop mode over the source's available
/// duration.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct TimeMap {
    /// Playback speed multiplier. `1.0` is real time, `2.0` is double speed,
    /// negative values play in reverse.
    pub rate: f32,
    /// Offset added to the source time, in source-local seconds.
    pub offset: f32,
    /// Behavior once the mapped time runs past `source_duration`.
    pub mode: LoopMode,
}

impl Default for TimeMap {
    fn default() -> Self {
        Self {
            rate: 1.0,
            offset: 0.0,
            mode: LoopMode::Once,
        }
    }
}

impl TimeMap {
    /// Creates a `TimeMap` with the given rate, zero offset, and `Once` looping.
    pub fn with_rate(rate: f32) -> Self {
        Self {
            rate,
            ..Default::default()
        }
    }

    /// Maps `local_t` (seconds since the clip became active on the timeline) into
    /// a time within `[0, source_duration]` in the source's own local time,
    /// applying `rate`, `offset`, and `mode`.
    ///
    /// `source_duration` must be positive; `Hold` mode ignores it.
    pub fn resolve(&self, local_t: f32, source_duration: f32) -> f32 {
        let raw = self.offset + local_t * self.rate;
        match self.mode {
            LoopMode::Hold => self.offset,
            LoopMode::Once => raw.clamp(0.0, source_duration.max(0.0)),
            LoopMode::Loop => {
                if source_duration <= 0.0 {
                    0.0
                } else {
                    raw.rem_euclid(source_duration)
                }
            }
            LoopMode::PingPong => {
                if source_duration <= 0.0 {
                    0.0
                } else {
                    let period = source_duration * 2.0;
                    let phase = raw.rem_euclid(period);
                    if phase <= source_duration {
                        phase
                    } else {
                        period - phase
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn once_clamps_at_end() {
        let map = TimeMap::default();
        assert_eq!(map.resolve(0.0, 5.0), 0.0);
        assert_eq!(map.resolve(3.0, 5.0), 3.0);
        assert_eq!(map.resolve(10.0, 5.0), 5.0);
    }

    #[test]
    fn loop_wraps() {
        let map = TimeMap {
            mode: LoopMode::Loop,
            ..Default::default()
        };
        assert_eq!(map.resolve(7.0, 5.0), 2.0);
        assert_eq!(map.resolve(12.0, 5.0), 2.0);
    }

    #[test]
    fn ping_pong_bounces() {
        let map = TimeMap {
            mode: LoopMode::PingPong,
            ..Default::default()
        };
        assert_eq!(map.resolve(0.0, 5.0), 0.0);
        assert_eq!(map.resolve(5.0, 5.0), 5.0);
        assert_eq!(map.resolve(7.0, 5.0), 3.0);
        assert_eq!(map.resolve(10.0, 5.0), 0.0);
    }

    #[test]
    fn hold_ignores_elapsed_time() {
        let map = TimeMap {
            offset: 2.0,
            mode: LoopMode::Hold,
            ..Default::default()
        };
        assert_eq!(map.resolve(0.0, 5.0), 2.0);
        assert_eq!(map.resolve(100.0, 5.0), 2.0);
    }

    #[test]
    fn rate_scales_time() {
        let map = TimeMap::with_rate(2.0);
        assert_eq!(map.resolve(1.0, 10.0), 2.0);
        let reverse = TimeMap::with_rate(-1.0);
        assert_eq!(reverse.resolve(1.0, 10.0), 0.0);
    }
}
