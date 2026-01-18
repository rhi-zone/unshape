//! Pattern combinators for audio sequencing.
//!
//! TidalCycles-inspired pattern transformations for rhythmic composition:
//! - `fast()` / `slow()` - speed up or slow down patterns
//! - `rev()` - reverse a pattern
//! - `jux()` - apply a function differently to left and right channels
//! - `every()` - apply a transformation every N cycles
//!
//! # Example
//!
//! ```
//! use rhizome_resin_audio::pattern::{Pattern, fast, slow, rev, cat};
//!
//! // Create a simple pattern
//! let kicks = Pattern::from_events(vec![
//!     (0.0, 0.5, "kick"),
//!     (0.5, 0.5, "kick"),
//! ]);
//!
//! // Double the speed
//! let fast_kicks = fast(2.0, kicks.clone());
//!
//! // Reverse the pattern
//! let reversed = rev(kicks.clone());
//!
//! // Concatenate patterns
//! let both = cat(vec![kicks, reversed]);
//! ```

use rhizome_resin_expr_field::FieldExpr;
use std::collections::HashMap;
use std::sync::Arc as StdArc;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// A time value in cycles (0.0 to 1.0 per cycle).
pub type Time = f64;

/// An event in a pattern.
#[derive(Debug, Clone, PartialEq)]
pub struct Event<T> {
    /// Start time in cycles.
    pub onset: Time,
    /// Duration in cycles.
    pub duration: Time,
    /// The event value.
    pub value: T,
}

impl<T> Event<T> {
    /// Creates a new event.
    pub fn new(onset: Time, duration: Time, value: T) -> Self {
        Self {
            onset,
            duration,
            value,
        }
    }

    /// Returns the end time of the event.
    pub fn offset(&self) -> Time {
        self.onset + self.duration
    }

    /// Maps the value of this event.
    pub fn map<U, F: FnOnce(T) -> U>(self, f: F) -> Event<U> {
        Event {
            onset: self.onset,
            duration: self.duration,
            value: f(self.value),
        }
    }
}

/// A time span (arc) in a pattern.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TimeArc {
    /// Start time.
    pub start: Time,
    /// End time.
    pub end: Time,
}

impl TimeArc {
    /// Creates a new arc.
    pub fn new(start: Time, end: Time) -> Self {
        Self { start, end }
    }

    /// Returns the duration of the arc.
    pub fn duration(&self) -> Time {
        self.end - self.start
    }

    /// Checks if a time is within this arc.
    pub fn contains(&self, t: Time) -> bool {
        t >= self.start && t < self.end
    }

    /// Returns the intersection of two arcs.
    pub fn intersect(&self, other: &TimeArc) -> Option<TimeArc> {
        let start = self.start.max(other.start);
        let end = self.end.min(other.end);
        if start < end {
            Some(TimeArc::new(start, end))
        } else {
            None
        }
    }
}

/// A pattern that generates events over time.
#[derive(Clone)]
pub struct Pattern<T: Clone + 'static> {
    /// Function that queries events for a given time arc.
    query: StdArc<dyn Fn(TimeArc) -> Vec<Event<T>> + Send + Sync>,
}

impl<T: Clone + Send + Sync + 'static> Pattern<T> {
    /// Creates a pattern from a query function.
    pub fn from_query<F>(f: F) -> Self
    where
        F: Fn(TimeArc) -> Vec<Event<T>> + Send + Sync + 'static,
    {
        Self {
            query: StdArc::new(f),
        }
    }

    /// Creates an empty pattern.
    pub fn silence() -> Self {
        Self::from_query(|_| vec![])
    }

    /// Creates a pattern from a list of events (within cycle 0-1).
    pub fn from_events(events: Vec<(Time, Time, T)>) -> Self {
        let events: Vec<Event<T>> = events
            .into_iter()
            .map(|(onset, dur, val)| Event::new(onset, dur, val))
            .collect();

        Self::from_query(move |arc| {
            let mut result = Vec::new();
            // Handle multiple cycles
            let start_cycle = arc.start.floor() as i64;
            let end_cycle = arc.end.ceil() as i64;

            for cycle in start_cycle..end_cycle {
                let cycle_offset = cycle as f64;
                for event in &events {
                    let shifted = Event {
                        onset: event.onset + cycle_offset,
                        duration: event.duration,
                        value: event.value.clone(),
                    };
                    // Check if event overlaps with query arc
                    if shifted.offset() > arc.start && shifted.onset < arc.end {
                        result.push(shifted);
                    }
                }
            }
            result
        })
    }

    /// Creates a pattern with a single event per cycle.
    pub fn pure(value: T) -> Self {
        Self::from_events(vec![(0.0, 1.0, value)])
    }

    /// Queries events for a given arc.
    pub fn query(&self, arc: TimeArc) -> Vec<Event<T>> {
        (self.query)(arc)
    }

    /// Queries events for a given cycle.
    pub fn query_cycle(&self, cycle: i64) -> Vec<Event<T>> {
        let start = cycle as f64;
        self.query(TimeArc::new(start, start + 1.0))
    }

    /// Maps values in the pattern.
    pub fn fmap<U, F>(self, f: F) -> Pattern<U>
    where
        U: Clone + Send + Sync + 'static,
        F: Fn(T) -> U + Send + Sync + 'static,
    {
        let query = self.query;
        Pattern::from_query(move |arc| query(arc).into_iter().map(|e| e.map(|v| f(v))).collect())
    }

    /// Filters events by a predicate.
    pub fn filter<F>(self, predicate: F) -> Self
    where
        F: Fn(&T) -> bool + Send + Sync + 'static,
    {
        let query = self.query;
        Self::from_query(move |arc| {
            query(arc)
                .into_iter()
                .filter(|e| predicate(&e.value))
                .collect()
        })
    }
}

// ============================================================================
// Pattern Combinators
// ============================================================================

/// Speeds up a pattern by the given factor.
pub fn fast<T: Clone + Send + Sync + 'static>(factor: f64, pattern: Pattern<T>) -> Pattern<T> {
    if factor <= 0.0 {
        return Pattern::silence();
    }

    let query = pattern.query;
    Pattern::from_query(move |arc| {
        // Stretch the query arc
        let stretched = TimeArc::new(arc.start * factor, arc.end * factor);
        query(stretched)
            .into_iter()
            .map(|e| Event {
                onset: e.onset / factor,
                duration: e.duration / factor,
                value: e.value,
            })
            .collect()
    })
}

/// Slows down a pattern by the given factor.
pub fn slow<T: Clone + Send + Sync + 'static>(factor: f64, pattern: Pattern<T>) -> Pattern<T> {
    fast(1.0 / factor, pattern)
}

/// Reverses a pattern within each cycle.
pub fn rev<T: Clone + Send + Sync + 'static>(pattern: Pattern<T>) -> Pattern<T> {
    let query = pattern.query;
    Pattern::from_query(move |arc| {
        query(arc)
            .into_iter()
            .map(|e| {
                let cycle = e.onset.floor();
                let offset = e.onset - cycle;
                Event {
                    onset: cycle + (1.0 - offset - e.duration),
                    duration: e.duration,
                    value: e.value,
                }
            })
            .collect()
    })
}

/// Concatenates patterns sequentially.
pub fn cat<T: Clone + Send + Sync + 'static>(patterns: Vec<Pattern<T>>) -> Pattern<T> {
    if patterns.is_empty() {
        return Pattern::silence();
    }

    let patterns = StdArc::new(patterns);

    Pattern::from_query(move |arc| {
        let mut result = Vec::new();
        let start_cycle = arc.start.floor() as i64;
        let end_cycle = arc.end.ceil() as i64;

        for cycle in start_cycle..end_cycle {
            let pattern_idx = (cycle as usize) % patterns.len();
            let pattern = &patterns[pattern_idx];

            // Query this cycle
            let cycle_arc = TimeArc::new(cycle as f64, cycle as f64 + 1.0);
            if let Some(intersection) = arc.intersect(&cycle_arc) {
                let events = pattern.query(TimeArc::new(
                    intersection.start - cycle as f64,
                    intersection.end - cycle as f64,
                ));

                for e in events {
                    result.push(Event {
                        onset: e.onset + cycle as f64,
                        duration: e.duration,
                        value: e.value.clone(),
                    });
                }
            }
        }
        result
    })
}

/// Stacks patterns, playing them simultaneously.
pub fn stack<T: Clone + Send + Sync + 'static>(patterns: Vec<Pattern<T>>) -> Pattern<T> {
    let patterns = StdArc::new(patterns);
    Pattern::from_query(move |arc| patterns.iter().flat_map(|p| p.query(arc)).collect())
}

/// Applies a function every N cycles.
pub fn every<T: Clone + Send + Sync + 'static, F>(n: usize, f: F, pattern: Pattern<T>) -> Pattern<T>
where
    F: Fn(Pattern<T>) -> Pattern<T> + Send + Sync + 'static,
{
    if n == 0 {
        return pattern;
    }

    let query = pattern.query.clone();
    let pattern_clone = Pattern {
        query: pattern.query,
    };
    let transformed = f(pattern_clone);

    Pattern::from_query(move |arc| {
        let start_cycle = arc.start.floor() as i64;
        let end_cycle = arc.end.ceil() as i64;

        let mut result = Vec::new();

        for cycle in start_cycle..end_cycle {
            let cycle_arc = TimeArc::new(cycle as f64, cycle as f64 + 1.0);
            if let Some(intersection) = arc.intersect(&cycle_arc) {
                let local_arc = TimeArc::new(
                    intersection.start - cycle as f64,
                    intersection.end - cycle as f64,
                );

                let events = if (cycle as usize) % n == 0 {
                    transformed.query(local_arc)
                } else {
                    query(local_arc)
                };

                for e in events {
                    result.push(Event {
                        onset: e.onset + cycle as f64,
                        duration: e.duration,
                        value: e.value.clone(),
                    });
                }
            }
        }
        result
    })
}

/// Shifts a pattern in time (positive = later).
pub fn shift<T: Clone + Send + Sync + 'static>(amount: f64, pattern: Pattern<T>) -> Pattern<T> {
    let query = pattern.query;
    Pattern::from_query(move |arc| {
        let shifted_arc = TimeArc::new(arc.start - amount, arc.end - amount);
        query(shifted_arc)
            .into_iter()
            .map(|e| Event {
                onset: e.onset + amount,
                duration: e.duration,
                value: e.value,
            })
            .collect()
    })
}

/// Applies a function to stereo channels differently.
/// Left channel gets the original, right gets the transformed.
pub fn jux<T: Clone + Send + Sync + 'static, F>(
    f: F,
    pattern: Pattern<T>,
) -> (Pattern<T>, Pattern<T>)
where
    F: Fn(Pattern<T>) -> Pattern<T> + Send + Sync + 'static,
{
    let left = Pattern {
        query: pattern.query.clone(),
    };
    let right = f(Pattern {
        query: pattern.query,
    });
    (left, right)
}

/// Degrades events randomly (removes some events).
pub fn degrade<T: Clone + Send + Sync + 'static>(
    probability: f64,
    seed: u64,
    pattern: Pattern<T>,
) -> Pattern<T> {
    let query = pattern.query;
    Pattern::from_query(move |arc| {
        query(arc)
            .into_iter()
            .filter(|e| {
                // Use f64 bit representation for better hash distribution
                let onset_bits = e.onset.to_bits();
                let hash = onset_bits
                    .wrapping_mul(seed.wrapping_add(0x517cc1b727220a95))
                    .wrapping_add(0x9e3779b97f4a7c15);
                let rand = (hash as f64) / (u64::MAX as f64);
                rand >= probability
            })
            .collect()
    })
}

// Helper: deterministic hash from onset + seed (used by rand, choose)
fn onset_hash(onset: f64, seed: u64) -> u64 {
    let onset_bits = onset.to_bits();
    // Combine onset and seed through multiple mixing stages
    let h = onset_bits.wrapping_add(seed);
    let h = h ^ (h >> 33);
    let h = h.wrapping_mul(0xff51afd7ed558ccd);
    let h = h ^ (h >> 33);
    let h = h.wrapping_mul(0xc4ceb9fe1a85ec53);
    let h = h ^ (h >> 33);
    h
}

// Helper: deterministic hash from cycle, index, and seed (used by shuffle)
fn shuffle_hash(cycle: i64, index: usize, seed: u64) -> u64 {
    // Combine all inputs into a single u64
    let h = (cycle as u64)
        .wrapping_mul(0x9e3779b97f4a7c15)
        .wrapping_add(index as u64)
        .wrapping_mul(0xff51afd7ed558ccd)
        .wrapping_add(seed);
    // Mix bits thoroughly
    let h = h ^ (h >> 33);
    let h = h.wrapping_mul(0xff51afd7ed558ccd);
    let h = h ^ (h >> 33);
    let h = h.wrapping_mul(0xc4ceb9fe1a85ec53);
    let h = h ^ (h >> 33);
    h
}

// Helper: convert hash to f64 in [0, 1)
fn hash_to_f64(hash: u64) -> f64 {
    (hash as f64) / (u64::MAX as f64)
}

/// Generates random f64 values for each event.
///
/// Each event gets a deterministic random value in [0, 1) based on its onset
/// and the provided seed. Querying the same pattern twice with the same seed
/// produces identical results.
///
/// # Example
///
/// ```ignore
/// use rhizome_resin_audio::pattern::{rand, Pattern, pure};
///
/// let random_values = rand(42); // Pattern of random f64 values
/// ```
pub fn rand(seed: u64) -> Pattern<f64> {
    Pattern::from_query(move |arc| {
        // Generate one event per integer cycle in the arc
        let start_cycle = arc.start.floor() as i64;
        let end_cycle = arc.end.ceil() as i64;

        (start_cycle..end_cycle)
            .filter_map(|cycle| {
                let onset = cycle as f64;
                if onset >= arc.start && onset < arc.end {
                    let hash = onset_hash(onset, seed);
                    Some(Event {
                        onset,
                        duration: 1.0,
                        value: hash_to_f64(hash),
                    })
                } else {
                    None
                }
            })
            .collect()
    })
}

/// Randomly chooses from multiple patterns per cycle.
///
/// For each cycle, deterministically selects one of the input patterns
/// based on the seed. This allows random pattern variations while
/// maintaining reproducibility.
///
/// # Example
///
/// ```ignore
/// use rhizome_resin_audio::pattern::{choose, pure, Pattern};
///
/// let kick = pure("kick");
/// let snare = pure("snare");
/// let hihat = pure("hihat");
///
/// // Randomly picks kick, snare, or hihat each cycle
/// let drums = choose(&[kick, snare, hihat], 42);
/// ```
pub fn choose<T: Clone + Send + Sync + 'static>(patterns: &[Pattern<T>], seed: u64) -> Pattern<T> {
    if patterns.is_empty() {
        return Pattern::silence();
    }

    let patterns: Vec<_> = patterns.iter().map(|p| p.query.clone()).collect();
    let count = patterns.len();

    Pattern::from_query(move |arc| {
        let mut result = Vec::new();
        let start_cycle = arc.start.floor() as i64;
        let end_cycle = arc.end.ceil() as i64;

        for cycle in start_cycle..end_cycle {
            let cycle_f = cycle as f64;
            let cycle_arc = TimeArc::new(cycle_f, cycle_f + 1.0);

            if let Some(intersection) = arc.intersect(&cycle_arc) {
                // Pick pattern based on cycle number
                let hash = onset_hash(cycle_f, seed);
                let index = (hash as usize) % count;

                let events = patterns[index](intersection);
                result.extend(events);
            }
        }
        result
    })
}

/// Shuffles event order within each cycle.
///
/// Events within each cycle get their onsets permuted based on the seed,
/// while preserving the set of onset positions and durations.
///
/// # Example
///
/// ```ignore
/// use rhizome_resin_audio::pattern::{shuffle, cat, pure, Pattern};
///
/// let sequence = cat(vec![Pattern::pure("a"), Pattern::pure("b"), Pattern::pure("c"), Pattern::pure("d")]);
/// let shuffled = shuffle(42, sequence); // Random permutation each cycle
/// ```
pub fn shuffle<T: Clone + Send + Sync + 'static>(seed: u64, pattern: Pattern<T>) -> Pattern<T> {
    let query = pattern.query;

    Pattern::from_query(move |arc| {
        let start_cycle = arc.start.floor() as i64;
        let end_cycle = arc.end.ceil() as i64;
        let mut result = Vec::new();

        for cycle in start_cycle..end_cycle {
            let cycle_f = cycle as f64;
            let cycle_arc = TimeArc::new(cycle_f, cycle_f + 1.0);

            // Get all events for this cycle
            let events = query(cycle_arc.clone());
            if events.is_empty() {
                continue;
            }

            // Extract onsets and values
            let mut onsets: Vec<f64> = events.iter().map(|e| e.onset).collect();
            let values: Vec<_> = events
                .iter()
                .map(|e| (e.duration, e.value.clone()))
                .collect();

            // Fisher-Yates shuffle on onsets using seeded hash
            let n = onsets.len();
            for i in (1..n).rev() {
                let hash = shuffle_hash(cycle, i, seed);
                let j = (hash as usize) % (i + 1);
                onsets.swap(i, j);
            }

            // Reassign values to shuffled onsets
            for (onset, (duration, value)) in onsets.into_iter().zip(values) {
                if let Some(intersection) = arc.intersect(&TimeArc::new(onset, onset + duration)) {
                    if intersection.start >= arc.start && intersection.start < arc.end {
                        result.push(Event {
                            onset,
                            duration,
                            value,
                        });
                    }
                }
            }
        }
        result
    })
}

/// Repeats a pattern a number of times within each cycle.
pub fn ply<T: Clone + Send + Sync + 'static>(times: usize, pattern: Pattern<T>) -> Pattern<T> {
    if times == 0 {
        return Pattern::silence();
    }
    fast(times as f64, pattern)
}

/// Chops a pattern into N equal parts.
pub fn chop<T: Clone + Send + Sync + 'static>(n: usize, pattern: Pattern<T>) -> Pattern<T> {
    if n == 0 {
        return Pattern::silence();
    }

    let query = pattern.query;
    let n_f = n as f64;

    Pattern::from_query(move |arc| {
        let mut result = Vec::new();
        let start_cycle = arc.start.floor() as i64;
        let end_cycle = arc.end.ceil() as i64;

        for cycle in start_cycle..end_cycle {
            let cycle_f = cycle as f64;
            for i in 0..n {
                let slice_start = cycle_f + (i as f64) / n_f;
                let slice_end = cycle_f + ((i + 1) as f64) / n_f;
                let slice_arc = TimeArc::new(slice_start, slice_end);

                if let Some(intersection) = arc.intersect(&slice_arc) {
                    // Query the full pattern and scale down
                    let full_events = query(TimeArc::new(
                        (intersection.start - slice_start) * n_f,
                        (intersection.end - slice_start) * n_f,
                    ));

                    for e in full_events {
                        result.push(Event {
                            onset: slice_start + e.onset / n_f,
                            duration: e.duration / n_f,
                            value: e.value.clone(),
                        });
                    }
                }
            }
        }
        result
    })
}

/// Euclidean rhythm generator (Bjorklund's algorithm).
///
/// Distributes `hits` events evenly across `steps` time slots.
/// Classic patterns like tresillo (3,8) and clave (5,16) emerge naturally.
///
/// # Design Note
///
/// This could theoretically decompose to `range(n).filter(euclidean_predicate)`,
/// but `filter` doesn't have enough other use cases to justify its existence
/// as a primitive. See `docs/design/pattern-primitives.md` for the full
/// analysis of pattern primitive decisions.
pub fn euclid<T: Clone + Send + Sync + 'static>(hits: usize, steps: usize, value: T) -> Pattern<T> {
    if steps == 0 || hits == 0 {
        return Pattern::silence();
    }

    let hits = hits.min(steps);
    let mut pattern = vec![false; steps];

    // Bjorklund's algorithm
    let mut bucket = 0;
    for i in 0..steps {
        bucket += hits;
        if bucket >= steps {
            bucket -= steps;
            pattern[i] = true;
        }
    }

    let events: Vec<(Time, Time, T)> = pattern
        .iter()
        .enumerate()
        .filter(|&(_, hit)| *hit)
        .map(|(i, _)| {
            let onset = i as f64 / steps as f64;
            let duration = 1.0 / steps as f64;
            (onset, duration, value.clone())
        })
        .collect();

    Pattern::from_events(events)
}

// ============================================================================
// Warp (time remapping)
// ============================================================================

/// Time remapping operation via Dew expression.
///
/// Transforms event timing by evaluating a [`FieldExpr`] at each event's onset.
/// The expression receives the onset time as `x` and should return the new time.
///
/// This single primitive covers multiple use cases:
/// - **Swing**: `x + (floor(x * 2.0) % 2.0) * amount`
/// - **Humanize**: `x + rand(x) * amount` (requires rand in expr)
/// - **Quantize**: `floor(x * grid) / grid` or `round(x * grid) / grid`
///
/// See `docs/design/pattern-primitives.md` for design rationale.
///
/// # Example
///
/// ```
/// use rhizome_resin_audio::pattern::{Pattern, Warp};
/// use rhizome_resin_expr_field::FieldExpr;
///
/// let pattern = Pattern::from_events(vec![(0.25, 0.5, "hit")]);
///
/// // Quantize to 0.5 grid (floor)
/// let quantize = Warp {
///     time_expr: FieldExpr::Mul(
///         Box::new(FieldExpr::Floor(Box::new(FieldExpr::Mul(
///             Box::new(FieldExpr::X),
///             Box::new(FieldExpr::Constant(2.0)),
///         )))),
///         Box::new(FieldExpr::Constant(0.5)),
///     ),
/// };
/// let quantized = quantize.apply(pattern);
/// ```
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Warp {
    /// Expression that maps old onset time (x) to new onset time.
    pub time_expr: FieldExpr,
}

impl Warp {
    /// Creates a new warp operation with the given time expression.
    pub fn new(time_expr: FieldExpr) -> Self {
        Self { time_expr }
    }

    /// Applies the time warp to a pattern.
    pub fn apply<T: Clone + Send + Sync + 'static>(&self, pattern: Pattern<T>) -> Pattern<T> {
        let query = pattern.query;
        let expr = self.time_expr.clone();

        Pattern::from_query(move |arc| {
            // Query a wider arc to catch events that might warp into our range
            // This is a heuristic - extreme warps might need wider margins
            let margin = 1.0;
            let wide_arc = TimeArc::new(arc.start - margin, arc.end + margin);

            query(wide_arc)
                .into_iter()
                .map(|e| {
                    let new_onset =
                        expr.eval(e.onset as f32, 0.0, 0.0, 0.0, &HashMap::new()) as f64;
                    Event {
                        onset: new_onset,
                        duration: e.duration,
                        value: e.value,
                    }
                })
                .filter(|e| e.onset >= arc.start && e.onset < arc.end)
                .collect()
        })
    }
}

/// Convenience function for warping pattern timing.
pub fn warp<T: Clone + Send + Sync + 'static>(
    time_expr: FieldExpr,
    pattern: Pattern<T>,
) -> Pattern<T> {
    Warp::new(time_expr).apply(pattern)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pattern_from_events() {
        let pattern = Pattern::from_events(vec![(0.0, 0.25, "kick"), (0.5, 0.25, "snare")]);

        let events = pattern.query_cycle(0);
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].value, "kick");
        assert_eq!(events[1].value, "snare");
    }

    #[test]
    fn test_fast() {
        let pattern = Pattern::from_events(vec![(0.0, 0.5, "a"), (0.5, 0.5, "b")]);
        let faster = fast(2.0, pattern);

        // At 2x speed, should get 4 events per cycle
        let events = faster.query_cycle(0);
        assert_eq!(events.len(), 4);
    }

    #[test]
    fn test_slow() {
        let pattern = Pattern::from_events(vec![(0.0, 0.5, "a"), (0.5, 0.5, "b")]);
        let slower = slow(2.0, pattern);

        // At 0.5x speed, events should span 2 cycles
        let events = slower.query(TimeArc::new(0.0, 2.0));
        assert_eq!(events.len(), 2);
    }

    #[test]
    fn test_rev() {
        let pattern = Pattern::from_events(vec![(0.0, 0.25, "a"), (0.75, 0.25, "b")]);
        let reversed = rev(pattern);

        let events = reversed.query_cycle(0);
        assert_eq!(events.len(), 2);
        // First event should now be at 0.75, second at 0.0
        assert!(
            events
                .iter()
                .any(|e| e.value == "a" && (e.onset - 0.75).abs() < 0.01)
        );
        assert!(
            events
                .iter()
                .any(|e| e.value == "b" && e.onset.abs() < 0.01)
        );
    }

    #[test]
    fn test_cat() {
        let a = Pattern::from_events(vec![(0.0, 1.0, "a")]);
        let b = Pattern::from_events(vec![(0.0, 1.0, "b")]);

        let combined = cat(vec![a, b]);

        let events0 = combined.query_cycle(0);
        assert_eq!(events0.len(), 1);
        assert_eq!(events0[0].value, "a");

        let events1 = combined.query_cycle(1);
        assert_eq!(events1.len(), 1);
        assert_eq!(events1[0].value, "b");
    }

    #[test]
    fn test_stack() {
        let a = Pattern::from_events(vec![(0.0, 0.5, "a")]);
        let b = Pattern::from_events(vec![(0.5, 0.5, "b")]);

        let stacked = stack(vec![a, b]);

        let events = stacked.query_cycle(0);
        assert_eq!(events.len(), 2);
    }

    #[test]
    fn test_shift() {
        let pattern = Pattern::from_events(vec![(0.0, 0.5, "a")]);
        let shifted = shift(0.25, pattern);

        let events = shifted.query_cycle(0);
        assert_eq!(events.len(), 1);
        assert!((events[0].onset - 0.25).abs() < 0.01);
    }

    #[test]
    fn test_euclid() {
        // E(3, 8) should give a tresillo pattern
        let pattern = euclid(3, 8, "x");

        let events = pattern.query_cycle(0);
        assert_eq!(events.len(), 3);
    }

    #[test]
    fn test_every() {
        let pattern = Pattern::from_events(vec![(0.0, 1.0, "a")]);
        let transformed = every(2, |p| fast(2.0, p), pattern);

        // Cycle 0: should be fast (2x)
        let events0 = transformed.query_cycle(0);
        assert_eq!(events0.len(), 2);

        // Cycle 1: should be normal
        let events1 = transformed.query_cycle(1);
        assert_eq!(events1.len(), 1);
    }

    #[test]
    fn test_jux() {
        let pattern = Pattern::from_events(vec![(0.0, 1.0, 1.0)]);
        let (left, right) = jux(|p| fast(2.0, p), pattern);

        let left_events = left.query_cycle(0);
        let right_events = right.query_cycle(0);

        assert_eq!(left_events.len(), 1);
        assert_eq!(right_events.len(), 2);
    }

    #[test]
    fn test_degrade() {
        let pattern = Pattern::from_events(vec![
            (0.0, 0.1, "a"),
            (0.1, 0.1, "b"),
            (0.2, 0.1, "c"),
            (0.3, 0.1, "d"),
            (0.4, 0.1, "e"),
        ]);

        let degraded = degrade(0.5, 12345, pattern.clone());
        let events = degraded.query_cycle(0);

        // Some events should be removed
        assert!(events.len() < 5);
        assert!(!events.is_empty()); // But not all (statistically unlikely)
    }

    #[test]
    fn test_fmap() {
        let pattern = Pattern::from_events(vec![(0.0, 1.0, 1)]);
        let doubled = pattern.fmap(|x| x * 2);

        let events = doubled.query_cycle(0);
        assert_eq!(events[0].value, 2);
    }

    #[test]
    fn test_pure() {
        let pattern = Pattern::pure(42);

        let events = pattern.query_cycle(0);
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].value, 42);
        assert_eq!(events[0].duration, 1.0);
    }

    #[test]
    fn test_arc_intersect() {
        let a = TimeArc::new(0.0, 1.0);
        let b = TimeArc::new(0.5, 1.5);

        let intersection = a.intersect(&b);
        assert!(intersection.is_some());

        let i = intersection.unwrap();
        assert!((i.start - 0.5).abs() < 0.001);
        assert!((i.end - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_warp_identity() {
        // x -> x should leave pattern unchanged
        let pattern = Pattern::from_events(vec![(0.25, 0.5, "hit")]);
        let warped = Warp::new(FieldExpr::X).apply(pattern);

        let events = warped.query_cycle(0);
        assert_eq!(events.len(), 1);
        assert!((events[0].onset - 0.25).abs() < 0.001);
    }

    #[test]
    fn test_warp_shift() {
        // x -> x + 0.1 should shift events forward
        let pattern = Pattern::from_events(vec![(0.2, 0.5, "hit")]);
        let shift_expr = FieldExpr::Add(Box::new(FieldExpr::X), Box::new(FieldExpr::Constant(0.1)));
        let warped = Warp::new(shift_expr).apply(pattern);

        let events = warped.query_cycle(0);
        assert_eq!(events.len(), 1);
        assert!((events[0].onset - 0.3).abs() < 0.001);
    }

    #[test]
    fn test_warp_quantize_floor() {
        // floor(x * 4) / 4 should quantize to 0.25 grid
        let pattern = Pattern::from_events(vec![(0.3, 0.1, "hit")]);
        let quantize_expr = FieldExpr::Div(
            Box::new(FieldExpr::Floor(Box::new(FieldExpr::Mul(
                Box::new(FieldExpr::X),
                Box::new(FieldExpr::Constant(4.0)),
            )))),
            Box::new(FieldExpr::Constant(4.0)),
        );
        let warped = Warp::new(quantize_expr).apply(pattern);

        let events = warped.query_cycle(0);
        assert_eq!(events.len(), 1);
        // 0.3 * 4 = 1.2, floor = 1, / 4 = 0.25
        assert!((events[0].onset - 0.25).abs() < 0.001);
    }

    #[test]
    fn test_warp_convenience_fn() {
        let pattern = Pattern::from_events(vec![(0.5, 0.5, "x")]);
        let warped = warp(FieldExpr::X, pattern);

        let events = warped.query_cycle(0);
        assert_eq!(events.len(), 1);
    }

    #[test]
    fn test_warp_multiple_events() {
        // Shift all events by 0.05
        let pattern =
            Pattern::from_events(vec![(0.0, 0.1, "a"), (0.25, 0.1, "b"), (0.5, 0.1, "c")]);
        let shift = FieldExpr::Add(Box::new(FieldExpr::X), Box::new(FieldExpr::Constant(0.05)));
        let warped = Warp::new(shift).apply(pattern);

        let events = warped.query_cycle(0);
        assert_eq!(events.len(), 3);
        assert!((events[0].onset - 0.05).abs() < 0.001);
        assert!((events[1].onset - 0.30).abs() < 0.001);
        assert!((events[2].onset - 0.55).abs() < 0.001);
    }

    #[test]
    fn test_rand_produces_values() {
        let pattern = rand(42);
        let events = pattern.query_cycle(0);

        assert_eq!(events.len(), 1);
        assert!(events[0].value >= 0.0 && events[0].value < 1.0);
    }

    #[test]
    fn test_rand_reproducible() {
        let pattern1 = rand(42);
        let pattern2 = rand(42);

        let events1 = pattern1.query_cycle(0);
        let events2 = pattern2.query_cycle(0);

        assert_eq!(events1.len(), events2.len());
        assert!((events1[0].value - events2[0].value).abs() < 1e-10);
    }

    #[test]
    fn test_rand_different_seeds() {
        let pattern1 = rand(42);
        let pattern2 = rand(999);

        // Check multiple cycles - at least one should differ
        let mut any_different = false;
        for cycle in 0..10 {
            let events1 = pattern1.query_cycle(cycle);
            let events2 = pattern2.query_cycle(cycle);
            if (events1[0].value - events2[0].value).abs() > 0.001 {
                any_different = true;
                break;
            }
        }
        assert!(
            any_different,
            "Different seeds should produce different values across cycles"
        );
    }

    #[test]
    fn test_choose_picks_patterns() {
        let a = Pattern::pure("a");
        let b = Pattern::pure("b");
        let c = Pattern::pure("c");

        let chosen = choose(&[a, b, c], 42);

        // Query multiple cycles
        let events0 = chosen.query_cycle(0);
        let events1 = chosen.query_cycle(1);
        let events2 = chosen.query_cycle(2);

        // Each should have one event
        assert_eq!(events0.len(), 1);
        assert_eq!(events1.len(), 1);
        assert_eq!(events2.len(), 1);

        // Values should be from the input patterns
        for e in [&events0[0], &events1[0], &events2[0]] {
            assert!(e.value == "a" || e.value == "b" || e.value == "c");
        }
    }

    #[test]
    fn test_choose_reproducible() {
        let a = Pattern::pure("a");
        let b = Pattern::pure("b");

        let chosen1 = choose(&[a.clone(), b.clone()], 42);
        let chosen2 = choose(&[a, b], 42);

        for cycle in 0..5 {
            let events1 = chosen1.query_cycle(cycle);
            let events2 = chosen2.query_cycle(cycle);
            assert_eq!(events1[0].value, events2[0].value);
        }
    }

    #[test]
    fn test_choose_empty() {
        let chosen: Pattern<&str> = choose(&[], 42);
        let events = chosen.query_cycle(0);
        assert!(events.is_empty());
    }

    #[test]
    fn test_shuffle_preserves_values() {
        // Use fast to create 4 events per cycle (shuffle requires multiple events)
        let pattern = fast(
            4.0,
            cat(vec![
                Pattern::pure("a"),
                Pattern::pure("b"),
                Pattern::pure("c"),
                Pattern::pure("d"),
            ]),
        );
        let shuffled = shuffle(42, pattern.clone());

        let original = pattern.query_cycle(0);
        let shuffled_events = shuffled.query_cycle(0);

        // Same number of events
        assert_eq!(original.len(), shuffled_events.len());

        // Same set of values (possibly different order)
        let mut orig_values: Vec<_> = original.iter().map(|e| e.value).collect();
        let mut shuf_values: Vec<_> = shuffled_events.iter().map(|e| e.value).collect();
        orig_values.sort();
        shuf_values.sort();
        assert_eq!(orig_values, shuf_values);
    }

    #[test]
    fn test_shuffle_reproducible() {
        // Use fast to create 4 events per cycle
        let pattern = fast(
            4.0,
            cat(vec![
                Pattern::pure("a"),
                Pattern::pure("b"),
                Pattern::pure("c"),
                Pattern::pure("d"),
            ]),
        );
        let shuffled1 = shuffle(42, pattern.clone());
        let shuffled2 = shuffle(42, pattern);

        let events1 = shuffled1.query_cycle(0);
        let events2 = shuffled2.query_cycle(0);

        for (e1, e2) in events1.iter().zip(events2.iter()) {
            assert_eq!(e1.value, e2.value);
            assert!((e1.onset - e2.onset).abs() < 1e-10);
        }
    }

    #[test]
    fn test_shuffle_different_seeds() {
        // Use fast to create 4 events per cycle
        let pattern = fast(
            4.0,
            cat(vec![
                Pattern::pure("a"),
                Pattern::pure("b"),
                Pattern::pure("c"),
                Pattern::pure("d"),
            ]),
        );

        let shuffled1 = shuffle(42, pattern.clone());
        let shuffled2 = shuffle(999, pattern);

        // Check multiple cycles - at least one should have different orderings
        let mut any_different = false;
        for cycle in 0..10 {
            let events1 = shuffled1.query_cycle(cycle);
            let events2 = shuffled2.query_cycle(cycle);

            let different = events1
                .iter()
                .zip(events2.iter())
                .any(|(e1, e2)| (e1.onset - e2.onset).abs() > 0.001);
            if different {
                any_different = true;
                break;
            }
        }
        assert!(
            any_different,
            "Different seeds should produce different shuffles across cycles"
        );
    }
}
