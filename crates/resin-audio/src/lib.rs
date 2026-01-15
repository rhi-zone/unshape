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
pub mod primitive;
pub mod room;
pub mod spatial;
pub mod spectral;
pub mod vocoder;
pub mod wav;

pub use effects::{
    AllpassBank, AmplitudeMod, Bitcrusher, Compressor, Convolution, ConvolutionConfig,
    ConvolutionReverb, Distortion, DistortionMode, Limiter, ModulatedDelay, NoiseGate, Reverb,
    chorus, convolution_reverb, flanger, generate_room_ir, phaser, tremolo,
};
pub use envelope::{Adsr, AdsrState, Ar, Lfo, LfoWaveform};
pub use filter::{
    Biquad, BiquadCoeffs, Delay, FeedbackDelay, HighPass, LowPass, highpass_coeff, highpass_sample,
    lowpass_coeff, lowpass_sample,
};
pub use granular::{
    GrainCloud, GrainConfig, GrainScheduler, GranularSynth, noise_grain_buffer, sine_grain_buffer,
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
pub use percussion::{
    Bar, BarConfig, BarSynth, Membrane, MembraneConfig, MembraneSynth, Plate, PlateConfig,
    PlateSynth, noise_burst,
};
pub use physical::{ExtendedKarplusStrong, KarplusStrong, PluckConfig, PluckSynth, PolyStrings};
pub use room::{
    EarlyReflection, RoomAcoustics, RoomGeometry, RoomMaterial, RoomMode, RoomModes, RoomSurfaces,
    calculate_early_reflections, calculate_room_modes, calculate_rt60_eyring,
    calculate_rt60_sabine,
};
pub use spatial::{
    DistanceModel, HrtfMode, SpatialListener, SpatialSource, Spatialize, Spatializer,
    SpatializerConfig, StereoMix, Vec3 as SpatialVec3, doppler_factor,
};
pub use spectral::{
    Complex, Stft, StftConfig, StftResult, TimeStretch, TimeStretchConfig, blackman_window,
    estimate_pitch, fft, fft_complex, find_peak_frequency, hamming_window, hann_window, ifft,
    ifft_complex, istft, pitch_shift, rect_window, spectral_centroid, spectral_flatness, stft,
    stft_with_sample_rate, time_stretch, time_stretch_granular,
};
pub use vocoder::{FilterbankVocoder, VocodeSynth, Vocoder, VocoderConfig};
pub use wav::{
    WavError, WavFile, WavFormat, WavResult, from_bytes as wav_from_bytes, to_bytes as wav_to_bytes,
};

/// Registers all audio operations with an [`OpRegistry`].
///
/// Call this to enable deserialization of audio ops from saved pipelines.
#[cfg(feature = "dynop")]
pub fn register_ops(registry: &mut rhizome_resin_op::OpRegistry) {
    registry.register_type::<Convolution>("resin::Convolution");
    registry.register_type::<GranularSynth>("resin::GranularSynth");
    registry.register_type::<PluckSynth>("resin::PluckSynth");
    registry.register_type::<MembraneSynth>("resin::MembraneSynth");
    registry.register_type::<BarSynth>("resin::BarSynth");
    registry.register_type::<PlateSynth>("resin::PlateSynth");
    registry.register_type::<Stft>("resin::Stft");
    registry.register_type::<TimeStretch>("resin::TimeStretch");
    registry.register_type::<VocodeSynth>("resin::VocodeSynth");
    registry.register_type::<Spatialize>("resin::Spatialize");
}
