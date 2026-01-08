//! Audio synthesis for resin.
//!
//! Provides oscillators and audio utilities for procedural sound generation.

pub mod osc;

pub use osc::{
    freq_to_phase, pulse, sample_to_phase, saw, saw_blep, saw_rev, sine, square, square_blep,
    triangle,
};
