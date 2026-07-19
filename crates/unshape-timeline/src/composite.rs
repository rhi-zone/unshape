//! Construction op that composes a [`Timeline`] and its clip sources into a
//! single `Field<f32, T>`.

use std::collections::HashMap;

use unshape_field_ops::{EvalContext, Field};

use crate::{Compositable, SourceId, Timeline};

/// Construction op: parameters are the [`Timeline`] arrangement (a serializable
/// value), inputs are the `Field<f32, T>` sources each `ClipInstance::source`
/// refers to, output is the composed field.
///
/// Sampling `TimelineComposite` at time `t` finds each track's active clip(s) at
/// `t`, remaps `t` into the clip's source-local time, samples that source's field
/// at the resolved time, and composites tracks top-down via [`Compositable`].
pub struct TimelineComposite<T> {
    /// The clip arrangement.
    pub timeline: Timeline,
    /// The field sources each clip instance's [`SourceId`] resolves against.
    pub sources: HashMap<SourceId, Box<dyn Field<f32, T>>>,
}

impl<T> TimelineComposite<T> {
    /// Creates a `TimelineComposite` for `timeline` with no sources wired up yet.
    pub fn new(timeline: Timeline) -> Self {
        Self {
            timeline,
            sources: HashMap::new(),
        }
    }

    /// Registers a field as the source for `id`, consuming and returning `self`
    /// for chaining.
    pub fn with_source(
        mut self,
        id: impl Into<SourceId>,
        field: impl Field<f32, T> + 'static,
    ) -> Self {
        self.sources.insert(id.into(), Box::new(field));
        self
    }
}

impl<T: Compositable> Field<f32, T> for TimelineComposite<T> {
    fn sample(&self, input: f32, ctx: &EvalContext) -> T {
        let sources = &self.sources;
        let mut lookup = |id: &SourceId, source_t: f32| -> Option<T> {
            let field = sources.get(id)?;
            let mut local_ctx = ctx.clone();
            local_ctx.time = source_t;
            Some(field.sample(source_t, &local_ctx))
        };
        self.timeline.sample(input, &mut lookup)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ClipInstance, TimeRange, Track};
    use unshape_field_ops::Constant;

    #[test]
    fn composes_field_sources_over_time() {
        let timeline = Timeline::from_tracks(vec![Track::from_clips(vec![
            ClipInstance::new("a", TimeRange::new(0.0, 1.0)),
            ClipInstance::new("b", TimeRange::new(1.0, 2.0)),
        ])]);

        let composite = TimelineComposite::new(timeline)
            .with_source("a", Constant::new(1.0_f32))
            .with_source("b", Constant::new(2.0_f32));

        let ctx = EvalContext::new();
        assert_eq!(composite.sample(0.5, &ctx), 1.0);
        assert_eq!(composite.sample(1.5, &ctx), 2.0);
        assert_eq!(composite.sample(5.0, &ctx), 0.0); // nothing active -> empty()
    }
}
