//! Wall-clock [`LiveSource`] — elapsed time and tick count since it was started.

use std::time::Instant;

use unshape_core::{GraphValue, Value, ValueType};

use crate::source::LiveSource;

/// A single reading from a [`ClockSource`]: elapsed wall-clock time and how
/// many times the source has been polled.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ClockSample {
    /// Seconds elapsed since the [`ClockSource`] was created.
    pub elapsed_secs: f64,
    /// Number of times the source has been polled (including this one).
    pub tick: u64,
}

impl GraphValue for ClockSample {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn type_name(&self) -> &'static str {
        "live::ClockSample"
    }
}

impl From<ClockSample> for Value {
    fn from(sample: ClockSample) -> Self {
        Value::opaque(sample)
    }
}

impl ClockSample {
    /// The [`ValueType`] a [`crate::LiveSourceNode<ClockSource>`] should
    /// declare for its output port.
    pub fn value_type() -> ValueType {
        ValueType::of::<ClockSample>("live::ClockSample")
    }
}

/// A [`LiveSource`] that reports the current wall-clock time and a running
/// tick count. Always has a fresh reading — every [`poll`](LiveSource::poll)
/// call returns `Some`.
pub struct ClockSource {
    started_at: Instant,
    tick: u64,
}

impl ClockSource {
    /// Starts a new clock, with elapsed time measured from now.
    #[allow(
        clippy::disallowed_methods,
        reason = "ClockSource is the deliberate wall-clock entry point for LiveSource; \
                  graph nodes/ops stay deterministic by reading its polled output, not \
                  the clock directly"
    )]
    pub fn new() -> Self {
        Self {
            started_at: Instant::now(),
            tick: 0,
        }
    }
}

impl Default for ClockSource {
    fn default() -> Self {
        Self::new()
    }
}

impl LiveSource for ClockSource {
    type Output = ClockSample;

    fn poll(&mut self) -> Option<ClockSample> {
        self.tick += 1;
        Some(ClockSample {
            elapsed_secs: self.started_at.elapsed().as_secs_f64(),
            tick: self.tick,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use unshape_core::{DynNode, EvalContext};

    use crate::node::LiveSourceNode;

    #[test]
    fn clock_always_has_pending() {
        let clock = ClockSource::new();
        assert!(clock.has_pending());
    }

    #[test]
    fn clock_ticks_advance_and_time_is_monotonic() {
        let mut clock = ClockSource::new();
        let first = clock.poll().unwrap();
        let second = clock.poll().unwrap();
        assert_eq!(first.tick, 1);
        assert_eq!(second.tick, 2);
        assert!(second.elapsed_secs >= first.elapsed_secs);
    }

    #[test]
    fn clock_sample_round_trips_through_value() {
        let sample = ClockSample {
            elapsed_secs: 1.5,
            tick: 3,
        };
        let value: Value = sample.into();
        let recovered = value.downcast_ref::<ClockSample>().unwrap();
        assert_eq!(*recovered, sample);
    }

    #[test]
    fn clock_source_wires_into_a_live_source_node() {
        let node = LiveSourceNode::new("clock", ClockSample::value_type(), ClockSource::new());
        let outputs = node.execute(&[], &EvalContext::new()).unwrap();
        let sample = outputs[0].downcast_ref::<ClockSample>().unwrap();
        assert_eq!(sample.tick, 1);
    }
}
