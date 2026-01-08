//! Audio synthesis for resin.
//!
//! Provides oscillators, filters, and audio utilities for procedural sound generation.

pub mod filter;
pub mod osc;

pub use filter::{
    Biquad, BiquadCoeffs, Delay, FeedbackDelay, HighPass, LowPass, highpass_coeff, highpass_sample,
    lowpass_coeff, lowpass_sample,
};
pub use osc::{
    freq_to_phase, pulse, sample_to_phase, saw, saw_blep, saw_rev, sine, square, square_blep,
    triangle,
};
