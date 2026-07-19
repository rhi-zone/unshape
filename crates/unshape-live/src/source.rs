//! [`LiveSource`] — the core abstraction for external, time-varying data.

/// Anything that produces values that change over time from outside the graph:
/// audio capture, video capture, MIDI input, a network stream, sensor data,
/// file-watch events, wall-clock time.
///
/// The graph is pull/lazy (see `docs/design/domain-subsumption.md`, Synthesis #3);
/// a `LiveSource` does not push evaluation itself. Instead it answers "has anything
/// changed since I last asked?" via [`poll`](LiveSource::poll), which the host calls
/// on its own schedule (once per frame, once per graph evaluation, from a background
/// thread that then wakes the graph, etc). [`LiveCache`] bridges this into the
/// cache-invalidation policy a [`unshape_core::nodes::GraphInput`]-shaped node needs:
/// hold the last value, refresh it when `poll` yields something new, and report
/// staleness in between.
///
/// Implementations are domain-specific (audio capture is nothing like a network
/// stream) and are expected to live in their own crates; this trait is only the
/// shared shape they poll through.
pub trait LiveSource: Send {
    /// The value type this source produces.
    type Output: Send;

    /// Non-blocking poll for the latest value.
    ///
    /// Returns `Some` when new data is available (a push source draining its
    /// internal queue, or a pull source that always has a fresh reading, e.g.
    /// wall-clock time). Returns `None` when nothing has changed since the last
    /// call — the caller should keep using its cached value.
    fn poll(&mut self) -> Option<Self::Output>;

    /// Whether calling [`poll`](LiveSource::poll) is expected to yield new data
    /// right now, without actually consuming it.
    ///
    /// Push sources backed by a queue can answer this cheaply (queue non-empty).
    /// Pull sources that always have a fresh reading (clocks, live sensors) should
    /// leave the default (`true`) — every poll is "new" for them by construction.
    fn has_pending(&self) -> bool {
        true
    }
}

/// Bridges a [`LiveSource`]'s push/pull data into a cached "latest value" a graph
/// node can read on every evaluation without necessarily having fresh data each time.
///
/// This is the cache-invalidation policy referenced in
/// `docs/design/domain-subsumption.md`'s `unshape-live` entry: a `GraphInput`-shaped
/// node's cached value goes stale when the source has new data, not on every pull.
pub struct LiveCache<S: LiveSource> {
    source: S,
    latest: Option<S::Output>,
}

impl<S: LiveSource> LiveCache<S> {
    /// Wraps a source with an empty cache (no value observed yet).
    pub fn new(source: S) -> Self {
        Self {
            source,
            latest: None,
        }
    }

    /// Polls the source and updates the cache if new data arrived.
    ///
    /// Returns `true` if the cached value changed as a result of this call.
    pub fn refresh(&mut self) -> bool {
        match self.source.poll() {
            Some(value) => {
                self.latest = Some(value);
                true
            }
            None => false,
        }
    }

    /// The most recently cached value, if any has ever arrived.
    pub fn latest(&self) -> Option<&S::Output> {
        self.latest.as_ref()
    }

    /// Whether the underlying source currently has no pending data (i.e. the
    /// cached value would not change on the next [`refresh`](Self::refresh)).
    pub fn is_stale(&self) -> bool {
        !self.source.has_pending()
    }

    /// Shared access to the underlying source.
    pub fn source(&self) -> &S {
        &self.source
    }

    /// Exclusive access to the underlying source.
    pub fn source_mut(&mut self) -> &mut S {
        &mut self.source
    }

    /// Unwraps back into the underlying source, discarding the cache.
    pub fn into_source(self) -> S {
        self.source
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A source that yields queued values, then `None` once drained — the
    /// minimal push-shaped test double.
    struct QueueSource {
        queued: Vec<i32>,
    }

    impl LiveSource for QueueSource {
        type Output = i32;

        fn poll(&mut self) -> Option<i32> {
            if self.queued.is_empty() {
                None
            } else {
                Some(self.queued.remove(0))
            }
        }

        fn has_pending(&self) -> bool {
            !self.queued.is_empty()
        }
    }

    #[test]
    fn cache_starts_empty() {
        let cache = LiveCache::new(QueueSource { queued: vec![] });
        assert_eq!(cache.latest(), None);
    }

    #[test]
    fn refresh_updates_cache_on_new_data() {
        let mut cache = LiveCache::new(QueueSource { queued: vec![1, 2] });
        assert!(cache.refresh());
        assert_eq!(cache.latest(), Some(&1));
        assert!(cache.refresh());
        assert_eq!(cache.latest(), Some(&2));
    }

    #[test]
    fn refresh_keeps_last_value_when_no_new_data() {
        let mut cache = LiveCache::new(QueueSource { queued: vec![7] });
        assert!(cache.refresh());
        assert!(!cache.refresh());
        assert_eq!(cache.latest(), Some(&7));
    }

    #[test]
    fn is_stale_reflects_pending_state() {
        let mut cache = LiveCache::new(QueueSource { queued: vec![1] });
        assert!(!cache.is_stale());
        cache.refresh();
        assert!(cache.is_stale());
    }
}
