//! Audio synthesis demo: FM synthesis and effects.
//!
//! Generates a short audio clip using FM synthesis with effects,
//! and exports it as a WAV file.
//!
//! Run with: `cargo run --example audio_synthesis`

use unshape_audio::{FmOsc, Reverb, WavFile, chorus};

fn main() {
    let sample_rate = 44100u32;
    let duration_secs = 3.0;
    let num_samples = (sample_rate as f32 * duration_secs) as usize;

    println!("Generating FM synthesis audio...");

    // Generate a simple melody using 2-op FM
    let notes = [
        (261.63f32, 0.0, 0.5), // C4
        (329.63, 0.5, 0.5),    // E4
        (392.00, 1.0, 0.5),    // G4
        (523.25, 1.5, 1.0),    // C5 (held longer)
    ];

    let mut samples = vec![0.0f32; num_samples];

    for (freq, start_time, note_duration) in notes {
        let start_sample = (start_time * sample_rate as f32) as usize;
        let note_samples = (note_duration * sample_rate as f32) as usize;

        // Create FM oscillator with modulator:carrier ratio and mod index
        let mut fm = FmOsc::new(2.0, 3.0); // Bell-like ratio

        for i in 0..note_samples {
            if start_sample + i < num_samples {
                let t = i as f32 / sample_rate as f32;
                // Simple amplitude envelope
                let env = if t < 0.01 { t / 0.01 } else { (-t * 3.0).exp() };
                samples[start_sample + i] += fm.tick(freq, sample_rate as f32) * env * 0.5;
            }
        }
    }

    // Normalize
    let max = samples
        .iter()
        .map(|s| s.abs())
        .fold(0.0f32, |a, b| a.max(b));
    if max > 0.0 {
        for s in &mut samples {
            *s /= max;
        }
    }

    println!("Applying effects...");

    // Apply chorus
    let mut chorus = chorus(sample_rate as f32);
    for sample in &mut samples {
        *sample = chorus.process(*sample);
    }

    // Apply reverb
    let mut reverb = Reverb::new(sample_rate as f32);
    for sample in &mut samples {
        *sample = reverb.process(*sample);
    }

    // Final normalization
    let max = samples
        .iter()
        .map(|s| s.abs())
        .fold(0.0f32, |a, b| a.max(b));
    if max > 0.0 {
        for s in &mut samples {
            *s = (*s / max) * 0.9;
        }
    }

    // Export to WAV
    let wav = WavFile::mono(samples, sample_rate);
    let output_path = "audio_synthesis_output.wav";

    match wav.save(output_path) {
        Ok(_) => println!("Wrote {}", output_path),
        Err(e) => eprintln!("Failed to write WAV: {}", e),
    }
}
