//! Audio synthesis for resin.
//!
//! Provides oscillators, filters, envelopes, and audio utilities for procedural sound generation.

pub mod effects;
pub mod envelope;
pub mod filter;
pub mod graph;
pub mod midi;
pub mod osc;
pub mod physical;

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
pub use midi::{
    Channel, ControlValue, Controller, MidiMessage, Note, Program, Velocity, amplitude_to_velocity,
    cc, cc_to_normalized, freq_to_note, freq_to_note_tuned, normalized_to_cc, note_name,
    note_to_freq, note_to_freq_tuned, notes, parse_note_name, pitch_bend_to_ratio,
    velocity_to_amplitude,
};
pub use osc::{
    FmAlgorithm, FmOperator, FmOsc, FmSynth, Wavetable, WavetableBank, WavetableOsc,
    additive_wavetable, fm_presets, freq_to_phase, pulse, sample_to_phase, saw, saw_blep, saw_rev,
    sine, square, square_blep, supersaw_wavetable, triangle,
};
pub use physical::{ExtendedKarplusStrong, KarplusStrong, PluckConfig, PolyStrings};
