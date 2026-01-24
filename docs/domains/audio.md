# Audio

Audio synthesis and processing for procedural sound generation.

## Prior Art

### Pure Data (Pd) / Max/MSP
- **Dataflow**: objects connected by patch cords
- **Two rates**: audio rate (~44100 Hz) vs control rate (~64 samples)
- **Hot/cold inlets**: leftmost inlet triggers computation
- **Abstractions**: patches as reusable objects

### SuperCollider
- **SynthDef**: define synth as UGen graph, instantiate as Synth
- **UGens**: unit generators (oscillators, filters, etc.)
- **Demand rate**: generates values on demand (sequencers)
- **Buffers**: sample data, wavetables

### FAUST
- **Functional DSP**: audio as pure functions on streams
- **Block diagram algebra**: sequential `:`, parallel `,`, split `<:`, merge `:>`
- **Automatic differentiation**: for physical modeling

### VCV Rack / Modular Synths
- **Modules**: self-contained units with inputs/outputs/knobs
- **Polyphony**: multiple voices per cable
- **CV/Gate**: control voltage for parameters, gates for triggers

## Module Structure

| Module | Purpose |
|--------|---------|
| `osc` | Oscillators (sine, saw, square, triangle, noise) |
| `filter` | Filters (lowpass, highpass, bandpass, biquad) |
| `envelope` | Envelopes (ADSR, AR, LFO) |
| `effects` | Effects (reverb, delay, chorus, phaser, flanger, distortion) |
| `granular` | Granular synthesis |
| `percussion` | Physical modeling percussion |
| `physical` | Karplus-Strong strings |
| `room` | Room acoustics and convolution reverb |
| `spectral` | FFT, STFT, spectral processing |
| `vocoder` | Vocoder and filterbank analysis |
| `graph` | Audio graph / signal chain |
| `patch` | Synthesizer patch system |
| `wav` | WAV file import/export |
| `midi` | MIDI parsing and utilities |

## Basic Oscillators

```rust
use rhi_unshape_audio::{SineOsc, SawOsc, SquareOsc, TriangleOsc, NoiseOsc};

// Create oscillators at sample rate
let sample_rate = 44100.0;

let mut sine = SineOsc::new(440.0, sample_rate);
let mut saw = SawOsc::new(440.0, sample_rate);
let mut square = SquareOsc::new(440.0, sample_rate);
let mut triangle = TriangleOsc::new(440.0, sample_rate);
let mut noise = NoiseOsc::white();

// Generate samples
let sample = sine.tick();
sine.set_frequency(880.0);  // Change frequency

// Generate buffer
let buffer: Vec<f32> = (0..44100).map(|_| sine.tick()).collect();
```

## Filters

```rust
use rhi_unshape_audio::{LowPassFilter, HighPassFilter, BiquadFilter, BiquadType};

// Simple one-pole filters
let mut lpf = LowPassFilter::new(1000.0, sample_rate);
let mut hpf = HighPassFilter::new(100.0, sample_rate);

let output = lpf.process(input);

// Biquad filters (more flexible)
let mut biquad = BiquadFilter::new(BiquadType::LowPass, 1000.0, 0.707, sample_rate);
biquad.set_frequency(2000.0);
biquad.set_q(2.0);  // Resonance

let output = biquad.process(input);
```

### Biquad Types

| Type | Parameters | Use |
|------|------------|-----|
| `LowPass` | cutoff, Q | Remove highs |
| `HighPass` | cutoff, Q | Remove lows |
| `BandPass` | center, Q | Isolate band |
| `Notch` | center, Q | Remove band |
| `Peak` | center, Q, gain | EQ boost/cut |
| `LowShelf` | cutoff, gain | Bass boost/cut |
| `HighShelf` | cutoff, gain | Treble boost/cut |

## Envelopes

### ADSR

```rust
use rhi_unshape_audio::{Adsr, AdsrConfig};

let config = AdsrConfig {
    attack: 0.01,   // 10ms attack
    decay: 0.1,     // 100ms decay
    sustain: 0.7,   // 70% sustain level
    release: 0.3,   // 300ms release
};

let mut env = Adsr::new(config, sample_rate);

// Trigger envelope
env.gate(true);   // Note on
// ... process samples ...
env.gate(false);  // Note off

// Get envelope value
let amplitude = env.tick();
let output = oscillator.tick() * amplitude;
```

### LFO

```rust
use rhi_unshape_audio::{Lfo, LfoShape};

let mut lfo = Lfo::new(5.0, sample_rate);  // 5 Hz
lfo.set_shape(LfoShape::Sine);
lfo.set_shape(LfoShape::Triangle);
lfo.set_shape(LfoShape::Square);
lfo.set_shape(LfoShape::SawUp);
lfo.set_shape(LfoShape::SawDown);
lfo.set_shape(LfoShape::Random);

// Use for modulation
let mod_value = lfo.tick();  // -1.0 to 1.0
filter.set_frequency(base_freq + mod_value * mod_depth);
```

## Effects

### Delay

```rust
use rhi_unshape_audio::{Delay, FeedbackDelay};

// Simple delay
let mut delay = Delay::new(0.5, sample_rate);  // 500ms delay
let output = delay.process(input);

// Feedback delay
let mut fb_delay = FeedbackDelay::new(0.25, 0.6, sample_rate);  // 250ms, 60% feedback
let output = fb_delay.process(input);
```

### Reverb

```rust
use rhi_unshape_audio::{Reverb, ReverbConfig};

let config = ReverbConfig {
    room_size: 0.8,
    damping: 0.5,
    wet: 0.3,
    dry: 0.7,
    width: 1.0,
};

let mut reverb = Reverb::new(config, sample_rate);
let (left, right) = reverb.process_stereo(input_left, input_right);
```

### Modulation Effects

```rust
use rhi_unshape_audio::{Chorus, Phaser, Flanger};

// Chorus
let mut chorus = Chorus::new(sample_rate);
chorus.set_rate(1.5);   // LFO rate
chorus.set_depth(0.5);  // Modulation depth
chorus.set_mix(0.5);

// Phaser
let mut phaser = Phaser::new(sample_rate);
phaser.set_rate(0.5);
phaser.set_depth(0.8);
phaser.set_stages(4);  // Number of allpass stages

// Flanger
let mut flanger = Flanger::new(sample_rate);
flanger.set_rate(0.3);
flanger.set_depth(0.7);
flanger.set_feedback(0.5);
```

### Distortion

```rust
use rhi_unshape_audio::{Distortion, DistortionType};

let mut dist = Distortion::new(DistortionType::SoftClip);
dist.set_drive(2.0);

// Types: SoftClip, HardClip, Fold, Bitcrush, Tube
let output = dist.process(input);
```

### Compressor

```rust
use rhi_unshape_audio::{Compressor, CompressorConfig};

let config = CompressorConfig {
    threshold: -20.0,  // dB
    ratio: 4.0,        // 4:1 compression
    attack: 0.01,      // 10ms
    release: 0.1,      // 100ms
    makeup_gain: 6.0,  // dB
};

let mut comp = Compressor::new(config, sample_rate);
let output = comp.process(input);
```

## FM Synthesis

```rust
use rhi_unshape_audio::{FmOsc, FmSynth, FmAlgorithm};

// Simple 2-operator FM
let mut fm = FmOsc::new(sample_rate);
fm.set_carrier_freq(440.0);
fm.set_mod_freq(880.0);      // 2:1 ratio
fm.set_mod_index(2.0);       // Modulation depth

let sample = fm.tick();

// Multi-operator FM synth
let mut synth = FmSynth::new(sample_rate);
synth.set_algorithm(FmAlgorithm::Stack4);  // 4 operators in series
synth.set_ratios(&[1.0, 2.0, 3.0, 4.0]);
synth.set_levels(&[1.0, 0.5, 0.3, 0.2]);

// Presets
let electric_piano = FmSynth::electric_piano(sample_rate);
let bass = FmSynth::bass(sample_rate);
let brass = FmSynth::brass(sample_rate);
```

## Wavetable Synthesis

```rust
use rhi_unshape_audio::{Wavetable, WavetableOsc, WavetableBank};

// Create wavetable from function
let sine_table = Wavetable::from_fn(2048, |phase| (phase * 2.0 * PI).sin());

// Or use generators
let saw_table = Wavetable::saw(2048);
let square_table = Wavetable::square(2048);
let supersaw_table = Wavetable::supersaw(2048, 7, 0.1);  // 7 saws, 10% detune

// Oscillator
let mut osc = WavetableOsc::new(&sine_table, sample_rate);
osc.set_frequency(440.0);
let sample = osc.tick();

// Wavetable bank (morphing between tables)
let mut bank = WavetableBank::new(sample_rate);
bank.add_table(sine_table);
bank.add_table(saw_table);
bank.add_table(square_table);
bank.set_position(0.5);  // Morph between tables

let sample = bank.tick();
```

## Granular Synthesis

```rust
use rhi_unshape_audio::{GrainCloud, GrainConfig, GrainScheduler};

// Load or create source buffer
let buffer: Vec<f32> = load_sample("voice.wav");

let mut cloud = GrainCloud::new(buffer, sample_rate);

// Configure grains
cloud.set_grain_size(50.0);      // 50ms grains
cloud.set_density(20.0);         // 20 grains/second
cloud.set_position(0.5);         // Middle of buffer
cloud.set_position_spread(0.1);  // Random variation
cloud.set_pitch(1.0);            // Original pitch
cloud.set_pitch_spread(0.1);     // Random pitch variation

// Generate audio
let sample = cloud.tick();

// Scheduled grains
let mut scheduler = GrainScheduler::new(sample_rate);
scheduler.schedule_grain(start_time, config);
```

## Physical Modeling

### Karplus-Strong Strings

```rust
use rhi_unshape_audio::{KarplusStrong, PluckConfig, PolyStrings};

// Single string
let config = PluckConfig {
    frequency: 440.0,
    decay: 0.996,
    brightness: 0.5,
    pluck_position: 0.25,
};

let mut string = KarplusStrong::new(config, sample_rate);
string.pluck(0.8);  // Pluck with velocity 0.8

let sample = string.tick();

// Polyphonic strings
let mut strings = PolyStrings::new(8, sample_rate);  // 8 voices
strings.note_on(60, 0.8);  // MIDI note, velocity
strings.note_off(60);

let sample = strings.tick();
```

### Percussion (Modal Synthesis)

```rust
use rhi_unshape_audio::percussion::{Membrane, Bar, Plate, MembraneConfig};

// Drum membrane
let config = MembraneConfig::snare();  // Preset
let mut drum = Membrane::new(config, sample_rate);
drum.strike(0.8);

// Bar (xylophone, marimba)
let mut bar = Bar::xylophone(440.0, sample_rate);
bar.strike(0.7);

// Plate (cymbal, bell)
let mut cymbal = Plate::cymbal(sample_rate);
let mut bell = Plate::bell(440.0, sample_rate);
cymbal.strike(0.9);

// Generate samples
let sample = drum.process() + cymbal.process();
```

## Room Acoustics

### Convolution Reverb

```rust
use rhi_unshape_audio::{ConvolutionReverb, generate_room_ir};

// Load impulse response
let ir = load_wav("hall.wav");
let mut reverb = ConvolutionReverb::new(&ir, sample_rate);

// Or generate synthetic IR
let ir = generate_room_ir(room_size, rt60, sample_rate);

// Process
let output = reverb.process(input);
```

### Room Simulation

```rust
use rhi_unshape_audio::{RoomAcoustics, RoomConfig};

let config = RoomConfig {
    dimensions: Vec3::new(10.0, 3.0, 8.0),  // Room size in meters
    absorption: 0.3,                         // Wall absorption
    source: Vec3::new(2.0, 1.5, 4.0),       // Sound source
    listener: Vec3::new(8.0, 1.5, 4.0),     // Listener
};

let room = RoomAcoustics::new(config);

// Calculate early reflections
let reflections = room.calculate_early_reflections(order: 3);

// Generate impulse response
let ir = room.generate_ir(sample_rate, length_seconds: 2.0);
```

## Spectral Processing

### FFT

```rust
use rhi_unshape_audio::{fft, ifft, Complex, window};

// Apply window
let windowed: Vec<f32> = window::hann(&samples);

// Forward FFT
let spectrum: Vec<Complex> = fft(&windowed);

// Manipulate spectrum...
// spectrum[i].re, spectrum[i].im

// Inverse FFT
let reconstructed: Vec<f32> = ifft(&spectrum);
```

### STFT (Short-Time Fourier Transform)

```rust
use rhi_unshape_audio::{stft, istft, StftConfig};

let config = StftConfig {
    fft_size: 2048,
    hop_size: 512,
    window: WindowType::Hann,
};

// Analyze
let result = stft(&samples, &config);

// result.frames[frame_index][bin_index] is Complex

// Resynthesize
let output = istft(&result);
```

## Vocoder

```rust
use rhi_unshape_audio::{Vocoder, FilterbankVocoder};

// FFT-based vocoder
let mut vocoder = Vocoder::new(2048, sample_rate);

// carrier = synth sound, modulator = voice
let output = vocoder.process(carrier, modulator);

// Filterbank vocoder (classic analog style)
let mut fb_vocoder = FilterbankVocoder::new(16, sample_rate);  // 16 bands
let output = fb_vocoder.process(carrier, modulator);
```

## Audio Graph

```rust
use rhi_unshape_audio::{Chain, Mixer, AudioGraph};

// Simple chain
let mut chain = Chain::new(sample_rate);
chain.add(SineOsc::new(440.0, sample_rate));
chain.add(LowPassFilter::new(2000.0, sample_rate));
chain.add(Reverb::default());

let output = chain.tick();

// Mixer
let mut mixer = Mixer::new(sample_rate);
mixer.add_channel(osc1, 0.5);  // 50% volume
mixer.add_channel(osc2, 0.3);  // 30% volume

let output = mixer.tick();
```

## Synthesizer Patches

```rust
use rhi_unshape_audio::{SynthPatch, PatchBank, ModRouting};

// Create patch
let mut patch = SynthPatch::new("Lead");
patch.set_osc1(OscType::Saw);
patch.set_osc2(OscType::Square);
patch.set_filter_cutoff(2000.0);
patch.set_filter_resonance(0.5);
patch.set_envelope(AdsrConfig { ... });

// Modulation routing
patch.add_mod_routing(ModRouting {
    source: ModSource::Lfo1,
    destination: ModDest::FilterCutoff,
    amount: 0.3,
});

// Save/load patches
let bank = PatchBank::new();
bank.add(patch);
bank.save("my_patches.json");
let bank = PatchBank::load("my_patches.json");
```

## MIDI

```rust
use rhi_unshape_audio::midi::{MidiMessage, note_to_freq, freq_to_note};

// Parse MIDI message
let msg = MidiMessage::parse(&[0x90, 60, 100]);
match msg {
    MidiMessage::NoteOn { channel, note, velocity } => {
        let freq = note_to_freq(note);
        synth.note_on(freq, velocity as f32 / 127.0);
    }
    MidiMessage::NoteOff { channel, note, .. } => {
        synth.note_off(note);
    }
    MidiMessage::ControlChange { channel, controller, value } => {
        // Handle CC
    }
    _ => {}
}

// Utilities
let freq = note_to_freq(69);  // A4 = 440 Hz
let note = freq_to_note(440.0);  // 69
```

## WAV Files

```rust
use rhi_unshape_audio::{WavFile, WavFormat};

// Load
let wav = WavFile::load("sample.wav")?;
let samples = wav.samples();
let sample_rate = wav.sample_rate();

// Save
let wav = WavFile::new(samples, sample_rate, WavFormat::PCM16);
wav.save("output.wav")?;

// Supported formats
WavFormat::PCM8
WavFormat::PCM16
WavFormat::PCM24
WavFormat::PCM32
WavFormat::Float32
```

## Example: Complete Synth

```rust
use rhi_unshape_audio::*;

// Build a simple subtractive synth
let sample_rate = 44100.0;

let mut osc1 = SawOsc::new(440.0, sample_rate);
let mut osc2 = SawOsc::new(440.0 * 1.01, sample_rate);  // Slight detune
let mut filter = BiquadFilter::new(BiquadType::LowPass, 2000.0, 2.0, sample_rate);
let mut env = Adsr::new(AdsrConfig::default(), sample_rate);
let mut lfo = Lfo::new(5.0, sample_rate);

// Process
env.gate(true);
let mut output = Vec::new();

for _ in 0..44100 {
    // Oscillators
    let osc_out = osc1.tick() + osc2.tick() * 0.5;

    // LFO modulates filter
    let lfo_val = lfo.tick();
    filter.set_frequency(2000.0 + lfo_val * 500.0);

    // Filter
    let filtered = filter.process(osc_out);

    // Envelope
    let amp = env.tick();

    output.push(filtered * amp);
}
```
