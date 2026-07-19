//! Trait for types that can be composited (crossfaded, layered) by [`crate::TimelineComposite`].
//!
//! What "composite" means is domain-specific — alpha-over for images, mixing for
//! audio, replacement for a scalar property — so `Timeline` itself stays agnostic
//! and defers to this trait.

/// A value that a [`crate::Track`] or [`crate::Timeline`] can blend or layer.
pub trait Compositable: Sized {
    /// The value representing "nothing here" — the base a track or timeline
    /// starts from before any clip is composited in.
    fn empty() -> Self;

    /// Composites `over` on top of `under` at the given `opacity` in `[0, 1]`.
    ///
    /// Used both for track layering (`opacity = 1.0`, `over` fully replaces/blends
    /// over `under`) and for transition cross-fades (`opacity` ramping `0.0..1.0`
    /// from the outgoing clip to the incoming clip).
    fn composite(under: &Self, over: &Self, opacity: f32) -> Self;
}

impl Compositable for f32 {
    fn empty() -> Self {
        0.0
    }

    fn composite(under: &Self, over: &Self, opacity: f32) -> Self {
        under + (over - under) * opacity.clamp(0.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f32_composite_lerps() {
        assert_eq!(f32::composite(&0.0, &10.0, 0.5), 5.0);
        assert_eq!(f32::composite(&0.0, &10.0, 0.0), 0.0);
        assert_eq!(f32::composite(&0.0, &10.0, 1.0), 10.0);
    }
}
