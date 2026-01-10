//! Audio synthesis for resin.
//!
//! Provides oscillators, filters, envelopes, and audio utilities for procedural sound generation.

pub mod effects;
pub mod envelope;
pub mod filter;
pub mod granular;
pub mod graph;
pub mod midi;
pub mod osc;
pub mod patch;
pub mod pattern;
pub mod percussion;
pub mod physical;
pub mod room;
pub mod spatial;
pub mod spectral;
pub mod vocoder;
pub mod wav;

pub use effects::{
    Chorus, ConvolutionConfig, ConvolutionReverb, Distortion, DistortionMode, Flanger, Phaser,
    Reverb, Tremolo, convolution_reverb, generate_room_ir,
};
pub use envelope::{Adsr, AdsrState, Ar, Lfo, LfoWaveform};
pub use filter::{
    Biquad, BiquadCoeffs, Delay, FeedbackDelay, HighPass, LowPass, highpass_coeff, highpass_sample,
    lowpass_coeff, lowpass_sample,
};
pub use granular::{
    GrainCloud, GrainConfig, GrainScheduler, noise_grain_buffer, sine_grain_buffer,
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
pub use patch::{
    ModRouting, ModSource, PatchBank, PatchParameter, SynthPatch, interpolate_patches,
    randomize_patch,
};
pub use pattern::{
    Event, Pattern, TimeArc as PatternArc, cat, chop, degrade, euclid, every, fast, jux, ply, rev,
    shift, slow, stack,
};
pub use percussion::{Bar, BarConfig, Membrane, MembraneConfig, Plate, PlateConfig, noise_burst};
pub use physical::{ExtendedKarplusStrong, KarplusStrong, PluckConfig, PolyStrings};
pub use room::{
    EarlyReflection, RoomAcoustics, RoomGeometry, RoomMaterial, RoomMode, RoomModes, RoomSurfaces,
    calculate_early_reflections, calculate_room_modes, calculate_rt60_eyring,
    calculate_rt60_sabine,
};
pub use spatial::{
    DistanceModel, HrtfMode, SpatialListener, SpatialSource, Spatializer, SpatializerConfig,
    StereoMix, Vec3 as SpatialVec3, doppler_factor,
};
pub use spectral::{
    Complex, StftConfig, StftResult, TimeStretchConfig, blackman_window, estimate_pitch, fft,
    fft_complex, find_peak_frequency, hamming_window, hann_window, ifft, ifft_complex, istft,
    pitch_shift, rect_window, spectral_centroid, spectral_flatness, stft, stft_with_sample_rate,
    time_stretch, time_stretch_granular,
};
pub use vocoder::{FilterbankVocoder, Vocoder, VocoderConfig};
pub use wav::{
    WavError, WavFile, WavFormat, WavResult, from_bytes as wav_from_bytes, to_bytes as wav_to_bytes,
};
