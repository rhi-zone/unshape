//! Audio synthesis for resin.
//!
//! Provides oscillators, filters, envelopes, and audio utilities for procedural sound generation.

pub mod effects;
pub mod envelope;
pub mod filter;
pub mod graph;
pub mod osc;

pub use effects::{Chorus, Distortion, DistortionMode, Flanger, Phaser, Reverb, Tremolo};
pub use envelope::{Adsr, AdsrState, Ar, Lfo, LfoWaveform};
pub use filter::{
    Biquad, BiquadCoeffs, Delay, FeedbackDelay, HighPass, LowPass, highpass_coeff, highpass_sample,
    lowpass_coeff, lowpass_sample,
};
pub use graph::{
    AdsrNode, ArNode, AudioContext, AudioNode, BiquadNode, Chain, Clip, Constant, DelayNode,
    FeedbackDelayNode, Gain, HighPassNode, LfoNode, LowPassNode, Mixer, Offset, Oscillator,
    PassThrough, RingMod, Silence, SoftClip, Waveform,
};
pub use osc::{
    Wavetable, WavetableBank, WavetableOsc, additive_wavetable, freq_to_phase, pulse,
    sample_to_phase, saw, saw_blep, saw_rev, sine, square, square_blep, supersaw_wavetable,
    triangle,
};
