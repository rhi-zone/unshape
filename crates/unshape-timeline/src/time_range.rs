//! A half-open span of time, `[start, end)`.

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// A half-open span of time in seconds: `start <= t < end`.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct TimeRange {
    /// Start time, inclusive.
    pub start: f32,
    /// End time, exclusive.
    pub end: f32,
}

impl TimeRange {
    /// Creates a new time range from `start` to `end`.
    ///
    /// # Panics
    ///
    /// Panics if `end < start`.
    pub fn new(start: f32, end: f32) -> Self {
        assert!(end >= start, "TimeRange end must be >= start");
        Self { start, end }
    }

    /// Creates a time range starting at `start` and lasting `duration` seconds.
    pub fn with_duration(start: f32, duration: f32) -> Self {
        Self::new(start, start + duration)
    }

    /// The length of this range in seconds.
    pub fn duration(&self) -> f32 {
        self.end - self.start
    }

    /// Returns `true` if `t` falls within `[start, end)`.
    pub fn contains(&self, t: f32) -> bool {
        t >= self.start && t < self.end
    }

    /// Returns `true` if this range overlaps `other` at all.
    pub fn overlaps(&self, other: &TimeRange) -> bool {
        self.start < other.end && other.start < self.end
    }

    /// The overlapping span between `self` and `other`, if any.
    pub fn intersection(&self, other: &TimeRange) -> Option<TimeRange> {
        let start = self.start.max(other.start);
        let end = self.end.min(other.end);
        (start < end).then_some(TimeRange { start, end })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn contains_is_half_open() {
        let r = TimeRange::new(1.0, 2.0);
        assert!(r.contains(1.0));
        assert!(!r.contains(2.0));
        assert!(r.contains(1.999));
    }

    #[test]
    fn duration_matches_span() {
        let r = TimeRange::with_duration(2.0, 3.0);
        assert_eq!(r.end, 5.0);
        assert_eq!(r.duration(), 3.0);
    }

    #[test]
    fn overlap_detection() {
        let a = TimeRange::new(0.0, 2.0);
        let b = TimeRange::new(1.0, 3.0);
        let c = TimeRange::new(2.0, 3.0);
        assert!(a.overlaps(&b));
        assert!(
            !a.overlaps(&c),
            "half-open ranges touching at a point don't overlap"
        );
        assert_eq!(a.intersection(&b), Some(TimeRange::new(1.0, 2.0)));
        assert_eq!(a.intersection(&c), None);
    }
}
