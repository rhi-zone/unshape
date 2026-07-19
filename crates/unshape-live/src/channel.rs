//! [`ChannelSource`] — a generic [`LiveSource`] fed by an `mpsc` channel.
//!
//! This is the escape hatch for feeding arbitrary external data into the graph
//! without writing a dedicated [`LiveSource`] impl: hand the [`Sender`] half to
//! whatever produces the data (a capture thread, a network client, a test), and
//! wrap the [`Receiver`] half in a node.

use std::sync::mpsc::{self, Receiver, Sender, TryRecvError};

use crate::source::LiveSource;

/// A [`LiveSource`] backed by an `mpsc::Receiver`. [`poll`](LiveSource::poll)
/// drains the channel non-blocking and returns the most recently sent value
/// (older queued values are discarded — this is a "latest wins" source, not a
/// queue the graph consumes item-by-item).
pub struct ChannelSource<T> {
    receiver: Receiver<T>,
    /// Set once the sender has been dropped, so `has_pending` doesn't keep
    /// claiming data might still arrive.
    disconnected: bool,
}

impl<T> ChannelSource<T> {
    /// Creates a linked `(Sender, ChannelSource)` pair — send values on the
    /// sender from anywhere (another thread, a callback), and wrap the
    /// returned source in a node.
    pub fn channel() -> (Sender<T>, ChannelSource<T>) {
        let (tx, rx) = mpsc::channel();
        (
            tx,
            ChannelSource {
                receiver: rx,
                disconnected: false,
            },
        )
    }

    /// Wraps an existing receiver as a live source.
    pub fn new(receiver: Receiver<T>) -> Self {
        Self {
            receiver,
            disconnected: false,
        }
    }
}

impl<T: Send> LiveSource for ChannelSource<T> {
    type Output = T;

    fn poll(&mut self) -> Option<T> {
        let mut latest = None;
        loop {
            match self.receiver.try_recv() {
                Ok(value) => latest = Some(value),
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    self.disconnected = true;
                    break;
                }
            }
        }
        latest
    }

    fn has_pending(&self) -> bool {
        // `mpsc::Receiver` has no peek/len, so this is a conservative "maybe" —
        // true until the sender disconnects, at which point no more data can
        // ever arrive.
        !self.disconnected
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn poll_returns_none_when_empty() {
        let (_tx, mut source) = ChannelSource::<i32>::channel();
        assert_eq!(source.poll(), None);
    }

    #[test]
    fn poll_drains_to_latest_sent_value() {
        let (tx, mut source) = ChannelSource::<i32>::channel();
        tx.send(1).unwrap();
        tx.send(2).unwrap();
        tx.send(3).unwrap();
        assert_eq!(source.poll(), Some(3));
        assert_eq!(source.poll(), None);
    }

    #[test]
    fn has_pending_false_after_sender_dropped() {
        let (tx, mut source) = ChannelSource::<i32>::channel();
        assert!(source.has_pending());
        drop(tx);
        source.poll();
        assert!(!source.has_pending());
    }
}
