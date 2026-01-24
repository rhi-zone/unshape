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
//! use unshape_audio::pattern::{Pattern, fast, slow, rev, cat};
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

use unshape_expr_field::FieldExpr;
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
// Continuous Patterns
// ============================================================================

/// A continuous pattern that can be evaluated at any time point.
///
/// Unlike [`Pattern`] which generates discrete events, `Continuous` represents
/// a smooth function of time. Use this for LFOs, envelopes, control signals,
/// and any value that varies continuously.
///
/// # Example
///
/// ```
/// use unshape_audio::pattern::Continuous;
/// use std::f64::consts::PI;
///
/// // Sine wave LFO (1 cycle per beat)
/// let lfo = Continuous::from_fn(|t| (t * 2.0 * PI).sin());
///
/// // Sample at various points
/// assert!((lfo.sample(0.0) - 0.0).abs() < 0.001);
/// assert!((lfo.sample(0.25) - 1.0).abs() < 0.001);
/// assert!((lfo.sample(0.5) - 0.0).abs() < 0.001);
/// ```
#[derive(Clone)]
pub struct Continuous<T: Clone + 'static> {
    /// Function that evaluates the pattern at a given time.
    eval: StdArc<dyn Fn(Time) -> T + Send + Sync>,
}

impl<T: Clone + Send + Sync + 'static> Continuous<T> {
    /// Creates a continuous pattern from an evaluation function.
    pub fn from_fn<F>(f: F) -> Self
    where
        F: Fn(Time) -> T + Send + Sync + 'static,
    {
        Self {
            eval: StdArc::new(f),
        }
    }

    /// Samples the pattern at a given time.
    #[inline]
    pub fn sample(&self, time: Time) -> T {
        (self.eval)(time)
    }

    /// Samples the pattern at multiple time points.
    pub fn sample_range(&self, start: Time, end: Time, num_samples: usize) -> Vec<T> {
        if num_samples == 0 {
            return vec![];
        }
        if num_samples == 1 {
            return vec![self.sample(start)];
        }
        let step = (end - start) / (num_samples - 1) as f64;
        (0..num_samples)
            .map(|i| self.sample(start + i as f64 * step))
            .collect()
    }

    /// Maps values in the pattern.
    pub fn map<U, F>(self, f: F) -> Continuous<U>
    where
        U: Clone + Send + Sync + 'static,
        F: Fn(T) -> U + Send + Sync + 'static,
    {
        let eval = self.eval;
        Continuous::from_fn(move |t| f(eval(t)))
    }

    /// Scales the time (makes pattern faster).
    pub fn fast(self, factor: f64) -> Self {
        let eval = self.eval;
        Self::from_fn(move |t| eval(t * factor))
    }

    /// Scales the time (makes pattern slower).
    pub fn slow(self, factor: f64) -> Self {
        if factor == 0.0 {
            return self;
        }
        self.fast(1.0 / factor)
    }

    /// Shifts the pattern in time.
    pub fn shift(self, amount: Time) -> Self {
        let eval = self.eval;
        Self::from_fn(move |t| eval(t - amount))
    }
}

impl Continuous<f64> {
    /// Creates a constant pattern.
    pub fn constant(value: f64) -> Self {
        Self::from_fn(move |_| value)
    }

    /// Creates a sine wave pattern (0 to 1 range, 1 cycle per unit time).
    pub fn sine() -> Self {
        Self::from_fn(|t| ((t * 2.0 * std::f64::consts::PI).sin() + 1.0) / 2.0)
    }

    /// Creates a cosine wave pattern (0 to 1 range, 1 cycle per unit time).
    pub fn cosine() -> Self {
        Self::from_fn(|t| ((t * 2.0 * std::f64::consts::PI).cos() + 1.0) / 2.0)
    }

    /// Creates a triangle wave pattern (0 to 1 range, 1 cycle per unit time).
    pub fn triangle() -> Self {
        Self::from_fn(|t| {
            let t = t.rem_euclid(1.0);
            if t < 0.5 { t * 2.0 } else { 2.0 - t * 2.0 }
        })
    }

    /// Creates a sawtooth wave pattern (0 to 1 range, 1 cycle per unit time).
    pub fn saw() -> Self {
        Self::from_fn(|t| t.rem_euclid(1.0))
    }

    /// Creates a square wave pattern (0 or 1, 1 cycle per unit time).
    pub fn square() -> Self {
        Self::from_fn(|t| if t.rem_euclid(1.0) < 0.5 { 0.0 } else { 1.0 })
    }

    /// Creates a pulse wave with given duty cycle (0 to 1).
    pub fn pulse(duty: f64) -> Self {
        let duty = duty.clamp(0.0, 1.0);
        Self::from_fn(move |t| if t.rem_euclid(1.0) < duty { 1.0 } else { 0.0 })
    }

    /// Creates a linear ramp from 0 to 1 over one cycle.
    pub fn ramp() -> Self {
        Self::saw()
    }

    /// Adds two continuous patterns.
    pub fn add(self, other: Self) -> Self {
        let eval_a = self.eval;
        let eval_b = other.eval;
        Self::from_fn(move |t| eval_a(t) + eval_b(t))
    }

    /// Multiplies two continuous patterns.
    pub fn mul(self, other: Self) -> Self {
        let eval_a = self.eval;
        let eval_b = other.eval;
        Self::from_fn(move |t| eval_a(t) * eval_b(t))
    }

    /// Scales the output by a factor.
    pub fn scale(self, factor: f64) -> Self {
        let eval = self.eval;
        Self::from_fn(move |t| eval(t) * factor)
    }

    /// Adds an offset to the output.
    pub fn offset(self, amount: f64) -> Self {
        let eval = self.eval;
        Self::from_fn(move |t| eval(t) + amount)
    }

    /// Converts to bipolar range (-1 to 1).
    pub fn bipolar(self) -> Self {
        let eval = self.eval;
        Self::from_fn(move |t| eval(t) * 2.0 - 1.0)
    }

    /// Converts to unipolar range (0 to 1) from bipolar input.
    pub fn unipolar(self) -> Self {
        let eval = self.eval;
        Self::from_fn(move |t| (eval(t) + 1.0) / 2.0)
    }

    /// Converts to discrete events by sampling at regular intervals.
    pub fn to_events(&self, samples_per_cycle: usize) -> Pattern<f64> {
        if samples_per_cycle == 0 {
            return Pattern::silence();
        }

        let eval = self.eval.clone();
        let step = 1.0 / samples_per_cycle as f64;

        Pattern::from_query(move |arc| {
            let mut result = Vec::new();
            let start_cycle = arc.start.floor() as i64;
            let end_cycle = arc.end.ceil() as i64;

            for cycle in start_cycle..end_cycle {
                for i in 0..samples_per_cycle {
                    let onset = cycle as f64 + i as f64 * step;
                    if onset >= arc.start && onset < arc.end {
                        let value = eval(onset);
                        result.push(Event::new(onset, step, value));
                    }
                }
            }
            result
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
/// use unshape_audio::pattern::{rand, Pattern, pure};
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
/// use unshape_audio::pattern::{choose, pure, Pattern};
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
/// use unshape_audio::pattern::{shuffle, cat, pure, Pattern};
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
// Polymetric patterns
// ============================================================================

/// Computes the greatest common divisor of two numbers.
fn gcd(mut a: u32, mut b: u32) -> u32 {
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}

/// Computes the least common multiple of two numbers.
fn lcm(a: u32, b: u32) -> u32 {
    if a == 0 || b == 0 {
        0
    } else {
        (a / gcd(a, b)) * b
    }
}

/// Creates a polymetric stack of patterns with different cycle lengths.
///
/// Each pattern is paired with its cycle length (in beats). The patterns
/// run simultaneously but at different rates, creating polyrhythmic phasing.
/// The combined pattern's cycle length is the LCM of all input lengths.
///
/// # Example
///
/// ```
/// use unshape_audio::pattern::{polymeter, Pattern};
///
/// // 3-over-4 polyrhythm
/// let threes = Pattern::from_events(vec![(0.0, 0.33, "x"), (0.33, 0.33, "x"), (0.66, 0.34, "x")]);
/// let fours = Pattern::from_events(vec![(0.0, 0.25, "o"), (0.25, 0.25, "o"), (0.5, 0.25, "o"), (0.75, 0.25, "o")]);
///
/// // Threes cycle every 3 beats, fours cycle every 4 beats
/// // Combined cycle is LCM(3,4) = 12 beats
/// let poly = polymeter(vec![(threes, 3), (fours, 4)]);
///
/// // Query the first 12 beats (one full polymetric cycle)
/// let events = poly.query(unshape_audio::pattern::TimeArc::new(0.0, 12.0));
/// // Contains 4 repetitions of threes (4*3=12) and 3 repetitions of fours (3*4=12)
/// ```
pub fn polymeter<T: Clone + Send + Sync + 'static>(patterns: Vec<(Pattern<T>, u32)>) -> Pattern<T> {
    if patterns.is_empty() {
        return Pattern::silence();
    }

    // Compute the LCM of all cycle lengths (the super-cycle length)
    let super_cycle = patterns
        .iter()
        .map(|(_, len)| *len)
        .fold(1u32, |acc, len| if len == 0 { acc } else { lcm(acc, len) });

    if super_cycle == 0 {
        return Pattern::silence();
    }

    let patterns = StdArc::new(patterns);
    let super_cycle_f = super_cycle as f64;

    Pattern::from_query(move |arc| {
        let mut result = Vec::new();

        for (pattern, cycle_len) in patterns.iter() {
            if *cycle_len == 0 {
                continue;
            }

            let cycle_len_f = *cycle_len as f64;
            // How many times does this pattern repeat in the super-cycle?
            let repetitions = super_cycle / cycle_len;

            // For each event from the pattern, we need to place it at multiple
            // positions within the super-cycle
            let start_super = arc.start;
            let end_super = arc.end;

            // Query the pattern for one cycle (0 to 1)
            let base_events = pattern.query_cycle(0);

            // For each super-cycle that overlaps our arc
            let start_super_cycle = (start_super / super_cycle_f).floor() as i64;
            let end_super_cycle = (end_super / super_cycle_f).ceil() as i64;

            for super_cycle_idx in start_super_cycle..end_super_cycle {
                let super_cycle_start = super_cycle_idx as f64 * super_cycle_f;

                // Place pattern events at each repetition point
                for rep in 0..repetitions {
                    let rep_offset = rep as f64 * cycle_len_f;

                    for event in &base_events {
                        // Scale event onset to absolute time
                        let abs_onset = super_cycle_start + rep_offset + event.onset * cycle_len_f;
                        let abs_duration = event.duration * cycle_len_f;

                        // Check if this event falls within our query arc
                        if abs_onset + abs_duration > arc.start && abs_onset < arc.end {
                            result.push(Event {
                                onset: abs_onset,
                                duration: abs_duration,
                                value: event.value.clone(),
                            });
                        }
                    }
                }
            }
        }

        // Sort by onset time
        result.sort_by(|a, b| a.onset.partial_cmp(&b.onset).unwrap());
        result
    })
}

/// Creates a simple polymetric pattern from a value repeated at different rates.
///
/// This is a convenience function for creating polyrhythms like 3-over-4.
///
/// # Example
///
/// ```
/// use unshape_audio::pattern::{polyrhythm, TimeArc};
///
/// // Classic 3-over-4 polyrhythm
/// let poly = polyrhythm(3, 4, ("x", "o"));
///
/// // Query first 12 beats
/// let events = poly.query(TimeArc::new(0.0, 12.0));
/// ```
pub fn polyrhythm<T: Clone + Send + Sync + 'static, U: Clone + Send + Sync + 'static>(
    beats_a: u32,
    beats_b: u32,
    values: (T, U),
) -> Pattern<(Option<T>, Option<U>)> {
    if beats_a == 0 || beats_b == 0 {
        return Pattern::silence();
    }

    let super_cycle = lcm(beats_a, beats_b);
    let super_cycle_f = super_cycle as f64;

    let value_a = values.0;
    let value_b = values.1;

    Pattern::from_query(move |arc| {
        let mut result = Vec::new();

        let start_super_cycle = (arc.start / super_cycle_f).floor() as i64;
        let end_super_cycle = (arc.end / super_cycle_f).ceil() as i64;

        for super_cycle_idx in start_super_cycle..end_super_cycle {
            let super_cycle_start = super_cycle_idx as f64 * super_cycle_f;

            // Add pattern A events
            let reps_a = super_cycle / beats_a;
            for rep in 0..reps_a {
                for beat in 0..beats_a {
                    let onset = super_cycle_start + rep as f64 * beats_a as f64 + beat as f64;
                    let duration = 1.0;
                    if onset + duration > arc.start && onset < arc.end {
                        result.push(Event {
                            onset,
                            duration,
                            value: (Some(value_a.clone()), None),
                        });
                    }
                }
            }

            // Add pattern B events
            let reps_b = super_cycle / beats_b;
            for rep in 0..reps_b {
                for beat in 0..beats_b {
                    let onset = super_cycle_start + rep as f64 * beats_b as f64 + beat as f64;
                    let duration = 1.0;
                    if onset + duration > arc.start && onset < arc.end {
                        result.push(Event {
                            onset,
                            duration,
                            value: (None, Some(value_b.clone())),
                        });
                    }
                }
            }
        }

        result.sort_by(|a, b| a.onset.partial_cmp(&b.onset).unwrap());
        result
    })
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
/// use unshape_audio::pattern::{Pattern, Warp};
/// use unshape_expr_field::FieldExpr;
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

    #[test]
    fn test_gcd_lcm() {
        assert_eq!(gcd(12, 8), 4);
        assert_eq!(gcd(8, 12), 4);
        assert_eq!(gcd(3, 4), 1);
        assert_eq!(gcd(6, 9), 3);

        assert_eq!(lcm(3, 4), 12);
        assert_eq!(lcm(4, 6), 12);
        assert_eq!(lcm(2, 5), 10);
    }

    #[test]
    fn test_polymeter_basic() {
        // Create two simple patterns
        let threes =
            Pattern::from_events(vec![(0.0, 0.33, "a"), (0.33, 0.33, "b"), (0.67, 0.33, "c")]);
        let fours = Pattern::from_events(vec![
            (0.0, 0.25, "1"),
            (0.25, 0.25, "2"),
            (0.5, 0.25, "3"),
            (0.75, 0.25, "4"),
        ]);

        // 3-over-4 polyrhythm: LCM = 12
        let poly = polymeter(vec![(threes, 3), (fours, 4)]);

        // Query the full super-cycle (12 beats)
        let events = poly.query(TimeArc::new(0.0, 12.0));

        // Should have:
        // - 4 repetitions of 3-beat pattern = 12 "a/b/c" events
        // - 3 repetitions of 4-beat pattern = 12 "1/2/3/4" events
        // Total: 24 events
        assert_eq!(events.len(), 24);

        // Count each pattern's contribution
        let threes_count = events
            .iter()
            .filter(|e| ["a", "b", "c"].contains(&e.value))
            .count();
        let fours_count = events
            .iter()
            .filter(|e| ["1", "2", "3", "4"].contains(&e.value))
            .count();

        assert_eq!(threes_count, 12); // 4 reps * 3 events
        assert_eq!(fours_count, 12); // 3 reps * 4 events
    }

    #[test]
    fn test_polymeter_empty() {
        let poly: Pattern<&str> = polymeter(vec![]);
        let events = poly.query(TimeArc::new(0.0, 10.0));
        assert!(events.is_empty());
    }

    #[test]
    fn test_polyrhythm_3_over_4() {
        let poly = polyrhythm(3, 4, ("x", "o"));

        // Query full super-cycle (12 beats)
        let events = poly.query(TimeArc::new(0.0, 12.0));

        // Count x and o events
        let x_count = events.iter().filter(|e| e.value.0.is_some()).count();
        let o_count = events.iter().filter(|e| e.value.1.is_some()).count();

        // 3-beat pattern repeats 4 times = 12 x events
        // 4-beat pattern repeats 3 times = 12 o events
        assert_eq!(x_count, 12);
        assert_eq!(o_count, 12);
    }

    #[test]
    fn test_continuous_constant() {
        let c = Continuous::constant(0.5);
        assert!((c.sample(0.0) - 0.5).abs() < 1e-10);
        assert!((c.sample(1.0) - 0.5).abs() < 1e-10);
        assert!((c.sample(100.0) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_continuous_sine() {
        let sine = Continuous::sine();
        // At t=0, sine is 0.5 (normalized from sin(0) = 0)
        assert!((sine.sample(0.0) - 0.5).abs() < 1e-10);
        // At t=0.25, sine is 1.0 (normalized from sin(π/2) = 1)
        assert!((sine.sample(0.25) - 1.0).abs() < 1e-10);
        // At t=0.5, sine is 0.5 (normalized from sin(π) = 0)
        assert!((sine.sample(0.5) - 0.5).abs() < 1e-10);
        // At t=0.75, sine is 0.0 (normalized from sin(3π/2) = -1)
        assert!((sine.sample(0.75) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_continuous_triangle() {
        let tri = Continuous::triangle();
        assert!((tri.sample(0.0) - 0.0).abs() < 1e-10);
        assert!((tri.sample(0.25) - 0.5).abs() < 1e-10);
        assert!((tri.sample(0.5) - 1.0).abs() < 1e-10);
        assert!((tri.sample(0.75) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_continuous_saw() {
        let saw = Continuous::saw();
        assert!((saw.sample(0.0) - 0.0).abs() < 1e-10);
        assert!((saw.sample(0.5) - 0.5).abs() < 1e-10);
        assert!((saw.sample(0.99) - 0.99).abs() < 1e-10);
        // Wraps at 1.0
        assert!((saw.sample(1.0) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_continuous_square() {
        let sq = Continuous::square();
        assert!((sq.sample(0.0) - 0.0).abs() < 1e-10);
        assert!((sq.sample(0.25) - 0.0).abs() < 1e-10);
        assert!((sq.sample(0.5) - 1.0).abs() < 1e-10);
        assert!((sq.sample(0.75) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_continuous_fast_slow() {
        let sine = Continuous::sine();
        let fast_sine = sine.clone().fast(2.0);

        // Fast doubles the frequency
        // At t=0.125, fast_sine should be at the same point as sine at t=0.25
        assert!((fast_sine.sample(0.125) - 1.0).abs() < 1e-10);

        let slow_sine = Continuous::sine().slow(2.0);
        // Slow halves the frequency
        // At t=0.5, slow_sine should be at the same point as sine at t=0.25
        assert!((slow_sine.sample(0.5) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_continuous_shift() {
        let sine = Continuous::sine();
        let shifted = sine.clone().shift(0.25);

        // Shifted by 0.25, so at t=0 we get what was at t=-0.25 = t=0.75
        // sine at 0.75 is 0.0
        assert!((shifted.sample(0.0) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_continuous_arithmetic() {
        let a = Continuous::constant(0.3);
        let b = Continuous::constant(0.2);

        let sum = a.clone().add(b.clone());
        assert!((sum.sample(0.0) - 0.5).abs() < 1e-10);

        let product = Continuous::constant(0.3).mul(Continuous::constant(0.2));
        assert!((product.sample(0.0) - 0.06).abs() < 1e-10);

        let scaled = Continuous::constant(0.5).scale(2.0);
        assert!((scaled.sample(0.0) - 1.0).abs() < 1e-10);

        let offset = Continuous::constant(0.3).offset(0.2);
        assert!((offset.sample(0.0) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_continuous_bipolar_unipolar() {
        let uni = Continuous::constant(0.5);
        let bi = uni.bipolar();
        // 0.5 * 2 - 1 = 0.0
        assert!((bi.sample(0.0) - 0.0).abs() < 1e-10);

        let back = Continuous::constant(0.0).unipolar();
        // (0.0 + 1) / 2 = 0.5
        assert!((back.sample(0.0) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_continuous_sample_range() {
        let saw = Continuous::saw();
        let samples = saw.sample_range(0.0, 1.0, 5);

        assert_eq!(samples.len(), 5);
        assert!((samples[0] - 0.0).abs() < 1e-10);
        assert!((samples[1] - 0.25).abs() < 1e-10);
        assert!((samples[2] - 0.5).abs() < 1e-10);
        assert!((samples[3] - 0.75).abs() < 1e-10);
        // Last sample is at t=1.0, which wraps to 0.0
        assert!((samples[4] - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_continuous_to_events() {
        let saw = Continuous::saw();
        let pattern = saw.to_events(4);

        let events = pattern.query_cycle(0);
        assert_eq!(events.len(), 4);

        // Events at 0.0, 0.25, 0.5, 0.75
        assert!((events[0].onset - 0.0).abs() < 1e-10);
        assert!((events[0].value - 0.0).abs() < 1e-10);
        assert!((events[1].onset - 0.25).abs() < 1e-10);
        assert!((events[1].value - 0.25).abs() < 1e-10);
    }

    #[test]
    fn test_continuous_map() {
        let saw = Continuous::saw();
        let doubled = saw.map(|x| x * 2.0);

        assert!((doubled.sample(0.25) - 0.5).abs() < 1e-10);
        assert!((doubled.sample(0.5) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_continuous_from_fn() {
        let custom = Continuous::from_fn(|t| t * t);
        assert!((custom.sample(2.0) - 4.0).abs() < 1e-10);
        assert!((custom.sample(3.0) - 9.0).abs() < 1e-10);
    }

    #[test]
    fn test_continuous_pulse() {
        let pulse = Continuous::pulse(0.25);
        assert!((pulse.sample(0.0) - 1.0).abs() < 1e-10);
        assert!((pulse.sample(0.1) - 1.0).abs() < 1e-10);
        assert!((pulse.sample(0.25) - 0.0).abs() < 1e-10);
        assert!((pulse.sample(0.5) - 0.0).abs() < 1e-10);
    }
}
