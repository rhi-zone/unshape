//! MIDI message types and utilities.
//!
//! Provides MIDI message parsing, note/frequency conversion, and basic MIDI utilities.
//!
//! # Example
//!
//! ```
//! use unshape_audio::midi::{MidiMessage, note_to_freq, note_name, Channel};
//!
//! // Parse a MIDI message
//! let msg = MidiMessage::from_bytes(&[0x90, 60, 100]);
//! assert!(matches!(msg, Some(MidiMessage::NoteOn { .. })));
//!
//! // Convert note to frequency
//! let freq = note_to_freq(69); // A4
//! assert!((freq - 440.0).abs() < 0.01);
//!
//! // Get note name
//! assert_eq!(note_name(60), "C4");
//! ```

/// MIDI channel (0-15).
pub type Channel = u8;

/// MIDI note number (0-127).
pub type Note = u8;

/// MIDI velocity (0-127).
pub type Velocity = u8;

/// MIDI control change number (0-127).
pub type Controller = u8;

/// MIDI control change value (0-127).
pub type ControlValue = u8;

/// MIDI program number (0-127).
pub type Program = u8;

/// Common MIDI control change numbers.
pub mod cc {
    use super::Controller;

    /// Modulation wheel (CC 1).
    pub const MOD_WHEEL: Controller = 1;
    /// Breath controller (CC 2).
    pub const BREATH: Controller = 2;
    /// Foot controller (CC 4).
    pub const FOOT: Controller = 4;
    /// Portamento time (CC 5).
    pub const PORTAMENTO_TIME: Controller = 5;
    /// Channel volume (CC 7).
    pub const VOLUME: Controller = 7;
    /// Balance (CC 8).
    pub const BALANCE: Controller = 8;
    /// Pan (CC 10).
    pub const PAN: Controller = 10;
    /// Expression (CC 11).
    pub const EXPRESSION: Controller = 11;
    /// Sustain pedal (CC 64).
    pub const SUSTAIN: Controller = 64;
    /// Portamento on/off (CC 65).
    pub const PORTAMENTO: Controller = 65;
    /// Sostenuto pedal (CC 66).
    pub const SOSTENUTO: Controller = 66;
    /// Soft pedal (CC 67).
    pub const SOFT_PEDAL: Controller = 67;
    /// Legato footswitch (CC 68).
    pub const LEGATO: Controller = 68;
    /// Hold 2 (CC 69).
    pub const HOLD_2: Controller = 69;
    /// Filter resonance (CC 71).
    pub const RESONANCE: Controller = 71;
    /// Release time (CC 72).
    pub const RELEASE_TIME: Controller = 72;
    /// Attack time (CC 73).
    pub const ATTACK_TIME: Controller = 73;
    /// Filter cutoff (CC 74).
    pub const CUTOFF: Controller = 74;
    /// Decay time (CC 75).
    pub const DECAY_TIME: Controller = 75;
    /// Vibrato rate (CC 76).
    pub const VIBRATO_RATE: Controller = 76;
    /// Vibrato depth (CC 77).
    pub const VIBRATO_DEPTH: Controller = 77;
    /// Vibrato delay (CC 78).
    pub const VIBRATO_DELAY: Controller = 78;
    /// Reverb send (CC 91).
    pub const REVERB: Controller = 91;
    /// Tremolo depth (CC 92).
    pub const TREMOLO: Controller = 92;
    /// Chorus send (CC 93).
    pub const CHORUS: Controller = 93;
    /// Detune (CC 94).
    pub const DETUNE: Controller = 94;
    /// Phaser depth (CC 95).
    pub const PHASER: Controller = 95;
    /// All sound off (CC 120).
    pub const ALL_SOUND_OFF: Controller = 120;
    /// Reset all controllers (CC 121).
    pub const RESET_ALL: Controller = 121;
    /// All notes off (CC 123).
    pub const ALL_NOTES_OFF: Controller = 123;
}

/// A parsed MIDI message.
#[derive(Debug, Clone, PartialEq)]
pub enum MidiMessage {
    /// Note Off event.
    NoteOff {
        /// MIDI channel (0-15).
        channel: Channel,
        /// Note number (0-127).
        note: Note,
        /// Release velocity (0-127).
        velocity: Velocity,
    },
    /// Note On event.
    NoteOn {
        /// MIDI channel (0-15).
        channel: Channel,
        /// Note number (0-127).
        note: Note,
        /// Attack velocity (0-127).
        velocity: Velocity,
    },
    /// Polyphonic aftertouch (pressure per note).
    PolyPressure {
        /// MIDI channel (0-15).
        channel: Channel,
        /// Note number (0-127).
        note: Note,
        /// Pressure value (0-127).
        pressure: u8,
    },
    /// Control change.
    ControlChange {
        /// MIDI channel (0-15).
        channel: Channel,
        /// Controller number (0-127).
        controller: Controller,
        /// Controller value (0-127).
        value: ControlValue,
    },
    /// Program change.
    ProgramChange {
        /// MIDI channel (0-15).
        channel: Channel,
        /// Program number (0-127).
        program: Program,
    },
    /// Channel aftertouch (pressure for whole channel).
    ChannelPressure {
        /// MIDI channel (0-15).
        channel: Channel,
        /// Pressure value (0-127).
        pressure: u8,
    },
    /// Pitch bend.
    PitchBend {
        /// MIDI channel (0-15).
        channel: Channel,
        /// Pitch bend value (-8192 to 8191).
        value: i16,
    },
    /// System exclusive (contains data without start/end markers).
    SysEx(Vec<u8>),
    /// MIDI Time Code quarter frame.
    MtcQuarterFrame(u8),
    /// Song position pointer.
    SongPosition(u16),
    /// Song select.
    SongSelect(u8),
    /// Tune request.
    TuneRequest,
    /// Timing clock.
    TimingClock,
    /// Start.
    Start,
    /// Continue.
    Continue,
    /// Stop.
    Stop,
    /// Active sensing.
    ActiveSensing,
    /// System reset.
    SystemReset,
}

impl MidiMessage {
    /// Parses a MIDI message from raw bytes.
    ///
    /// Returns `None` if the bytes don't form a valid message.
    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.is_empty() {
            return None;
        }

        let status = bytes[0];

        // System messages (0xF0-0xFF)
        if status >= 0xF0 {
            return Self::parse_system_message(status, bytes);
        }

        // Channel messages
        let channel = status & 0x0F;
        let message_type = status & 0xF0;

        match message_type {
            0x80 => {
                // Note Off
                if bytes.len() < 3 {
                    return None;
                }
                Some(MidiMessage::NoteOff {
                    channel,
                    note: bytes[1] & 0x7F,
                    velocity: bytes[2] & 0x7F,
                })
            }
            0x90 => {
                // Note On (velocity 0 = Note Off)
                if bytes.len() < 3 {
                    return None;
                }
                let velocity = bytes[2] & 0x7F;
                if velocity == 0 {
                    Some(MidiMessage::NoteOff {
                        channel,
                        note: bytes[1] & 0x7F,
                        velocity: 0,
                    })
                } else {
                    Some(MidiMessage::NoteOn {
                        channel,
                        note: bytes[1] & 0x7F,
                        velocity,
                    })
                }
            }
            0xA0 => {
                // Poly Pressure
                if bytes.len() < 3 {
                    return None;
                }
                Some(MidiMessage::PolyPressure {
                    channel,
                    note: bytes[1] & 0x7F,
                    pressure: bytes[2] & 0x7F,
                })
            }
            0xB0 => {
                // Control Change
                if bytes.len() < 3 {
                    return None;
                }
                Some(MidiMessage::ControlChange {
                    channel,
                    controller: bytes[1] & 0x7F,
                    value: bytes[2] & 0x7F,
                })
            }
            0xC0 => {
                // Program Change
                if bytes.len() < 2 {
                    return None;
                }
                Some(MidiMessage::ProgramChange {
                    channel,
                    program: bytes[1] & 0x7F,
                })
            }
            0xD0 => {
                // Channel Pressure
                if bytes.len() < 2 {
                    return None;
                }
                Some(MidiMessage::ChannelPressure {
                    channel,
                    pressure: bytes[1] & 0x7F,
                })
            }
            0xE0 => {
                // Pitch Bend
                if bytes.len() < 3 {
                    return None;
                }
                let lsb = (bytes[1] & 0x7F) as i16;
                let msb = (bytes[2] & 0x7F) as i16;
                let value = ((msb << 7) | lsb) - 8192;
                Some(MidiMessage::PitchBend { channel, value })
            }
            _ => None,
        }
    }

    fn parse_system_message(status: u8, bytes: &[u8]) -> Option<Self> {
        match status {
            0xF0 => {
                // System Exclusive
                // Find end of sysex (0xF7)
                let end = bytes.iter().position(|&b| b == 0xF7)?;
                Some(MidiMessage::SysEx(bytes[1..end].to_vec()))
            }
            0xF1 => {
                // MTC Quarter Frame
                if bytes.len() < 2 {
                    return None;
                }
                Some(MidiMessage::MtcQuarterFrame(bytes[1]))
            }
            0xF2 => {
                // Song Position
                if bytes.len() < 3 {
                    return None;
                }
                let lsb = (bytes[1] & 0x7F) as u16;
                let msb = (bytes[2] & 0x7F) as u16;
                Some(MidiMessage::SongPosition((msb << 7) | lsb))
            }
            0xF3 => {
                // Song Select
                if bytes.len() < 2 {
                    return None;
                }
                Some(MidiMessage::SongSelect(bytes[1] & 0x7F))
            }
            0xF6 => Some(MidiMessage::TuneRequest),
            0xF8 => Some(MidiMessage::TimingClock),
            0xFA => Some(MidiMessage::Start),
            0xFB => Some(MidiMessage::Continue),
            0xFC => Some(MidiMessage::Stop),
            0xFE => Some(MidiMessage::ActiveSensing),
            0xFF => Some(MidiMessage::SystemReset),
            _ => None,
        }
    }

    /// Converts the message to raw bytes.
    pub fn to_bytes(&self) -> Vec<u8> {
        match self {
            MidiMessage::NoteOff {
                channel,
                note,
                velocity,
            } => vec![0x80 | (channel & 0x0F), note & 0x7F, velocity & 0x7F],
            MidiMessage::NoteOn {
                channel,
                note,
                velocity,
            } => vec![0x90 | (channel & 0x0F), note & 0x7F, velocity & 0x7F],
            MidiMessage::PolyPressure {
                channel,
                note,
                pressure,
            } => vec![0xA0 | (channel & 0x0F), note & 0x7F, pressure & 0x7F],
            MidiMessage::ControlChange {
                channel,
                controller,
                value,
            } => vec![0xB0 | (channel & 0x0F), controller & 0x7F, value & 0x7F],
            MidiMessage::ProgramChange { channel, program } => {
                vec![0xC0 | (channel & 0x0F), program & 0x7F]
            }
            MidiMessage::ChannelPressure { channel, pressure } => {
                vec![0xD0 | (channel & 0x0F), pressure & 0x7F]
            }
            MidiMessage::PitchBend { channel, value } => {
                let centered = (value + 8192) as u16;
                let lsb = (centered & 0x7F) as u8;
                let msb = ((centered >> 7) & 0x7F) as u8;
                vec![0xE0 | (channel & 0x0F), lsb, msb]
            }
            MidiMessage::SysEx(data) => {
                let mut bytes = vec![0xF0];
                bytes.extend_from_slice(data);
                bytes.push(0xF7);
                bytes
            }
            MidiMessage::MtcQuarterFrame(value) => vec![0xF1, value & 0x7F],
            MidiMessage::SongPosition(position) => {
                let lsb = (position & 0x7F) as u8;
                let msb = ((position >> 7) & 0x7F) as u8;
                vec![0xF2, lsb, msb]
            }
            MidiMessage::SongSelect(song) => vec![0xF3, song & 0x7F],
            MidiMessage::TuneRequest => vec![0xF6],
            MidiMessage::TimingClock => vec![0xF8],
            MidiMessage::Start => vec![0xFA],
            MidiMessage::Continue => vec![0xFB],
            MidiMessage::Stop => vec![0xFC],
            MidiMessage::ActiveSensing => vec![0xFE],
            MidiMessage::SystemReset => vec![0xFF],
        }
    }

    /// Returns the channel for channel messages, None for system messages.
    pub fn channel(&self) -> Option<Channel> {
        match self {
            MidiMessage::NoteOff { channel, .. }
            | MidiMessage::NoteOn { channel, .. }
            | MidiMessage::PolyPressure { channel, .. }
            | MidiMessage::ControlChange { channel, .. }
            | MidiMessage::ProgramChange { channel, .. }
            | MidiMessage::ChannelPressure { channel, .. }
            | MidiMessage::PitchBend { channel, .. } => Some(*channel),
            _ => None,
        }
    }
}

/// Converts a MIDI note number to frequency in Hz.
///
/// Uses A4 = 440 Hz as the reference pitch.
#[inline]
pub fn note_to_freq(note: Note) -> f32 {
    note_to_freq_tuned(note, 440.0)
}

/// Converts a MIDI note number to frequency with custom A4 tuning.
#[inline]
pub fn note_to_freq_tuned(note: Note, a4_freq: f32) -> f32 {
    a4_freq * 2.0f32.powf((note as f32 - 69.0) / 12.0)
}

/// Converts a frequency to the nearest MIDI note number.
#[inline]
pub fn freq_to_note(freq: f32) -> Note {
    freq_to_note_tuned(freq, 440.0)
}

/// Converts a frequency to the nearest MIDI note number with custom A4 tuning.
#[inline]
pub fn freq_to_note_tuned(freq: f32, a4_freq: f32) -> Note {
    let note = 69.0 + 12.0 * (freq / a4_freq).log2();
    note.round().clamp(0.0, 127.0) as Note
}

/// Returns the note name (e.g., "C4", "F#3") for a MIDI note number.
pub fn note_name(note: Note) -> String {
    const NOTES: [&str; 12] = [
        "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
    ];
    let octave = (note / 12) as i8 - 1;
    let note_idx = (note % 12) as usize;
    format!("{}{}", NOTES[note_idx], octave)
}

/// Parses a note name (e.g., "C4", "F#3", "Bb5") to a MIDI note number.
pub fn parse_note_name(name: &str) -> Option<Note> {
    let name = name.trim();
    if name.is_empty() {
        return None;
    }

    let (note_part, octave_part) = if name.len() >= 2 {
        let chars: Vec<char> = name.chars().collect();
        if chars.len() >= 3 && (chars[1] == '#' || chars[1] == 'b') {
            // Note with accidental (e.g., "C#4", "Bb3")
            let note_str: String = chars[..2].iter().collect();
            let octave_str: String = chars[2..].iter().collect();
            (note_str, octave_str)
        } else {
            // Note without accidental (e.g., "C4")
            let note_str: String = chars[..1].iter().collect();
            let octave_str: String = chars[1..].iter().collect();
            (note_str, octave_str)
        }
    } else {
        return None;
    };

    let base_note = match note_part.to_uppercase().as_str() {
        "C" | "B#" => 0,
        "C#" | "DB" => 1,
        "D" => 2,
        "D#" | "EB" => 3,
        "E" | "FB" => 4,
        "F" | "E#" => 5,
        "F#" | "GB" => 6,
        "G" => 7,
        "G#" | "AB" => 8,
        "A" => 9,
        "A#" | "BB" => 10,
        "B" | "CB" => 11,
        _ => return None,
    };

    let octave: i8 = octave_part.parse().ok()?;
    let note = ((octave + 1) as i16 * 12 + base_note as i16) as i16;

    if (0..=127).contains(&note) {
        Some(note as Note)
    } else {
        None
    }
}

/// Converts a velocity value (0-127) to a linear amplitude (0.0-1.0).
#[inline]
pub fn velocity_to_amplitude(velocity: Velocity) -> f32 {
    (velocity as f32 / 127.0).powi(2)
}

/// Converts a linear amplitude (0.0-1.0) to a velocity value (0-127).
#[inline]
pub fn amplitude_to_velocity(amplitude: f32) -> Velocity {
    (amplitude.sqrt() * 127.0).round().clamp(0.0, 127.0) as Velocity
}

/// Converts a pitch bend value (-8192 to 8191) to a frequency ratio.
///
/// A standard pitch bend range is 2 semitones (200 cents).
#[inline]
pub fn pitch_bend_to_ratio(value: i16, semitone_range: f32) -> f32 {
    let normalized = value as f32 / 8192.0;
    2.0f32.powf(normalized * semitone_range / 12.0)
}

/// Converts a control change value (0-127) to a normalized value (0.0-1.0).
#[inline]
pub fn cc_to_normalized(value: ControlValue) -> f32 {
    value as f32 / 127.0
}

/// Converts a normalized value (0.0-1.0) to a control change value (0-127).
#[inline]
pub fn normalized_to_cc(value: f32) -> ControlValue {
    (value * 127.0).round().clamp(0.0, 127.0) as ControlValue
}

/// MIDI note constants.
pub mod notes {
    use super::Note;

    /// C in octave -1 (MIDI note 0).
    pub const C_NEG1: Note = 0;
    /// C in octave 0 (MIDI note 12).
    pub const C0: Note = 12;
    /// C in octave 1 (MIDI note 24).
    pub const C1: Note = 24;
    /// C in octave 2 (MIDI note 36).
    pub const C2: Note = 36;
    /// C in octave 3 (MIDI note 48).
    pub const C3: Note = 48;
    /// Middle C (MIDI note 60).
    pub const C4: Note = 60;
    /// C in octave 5 (MIDI note 72).
    pub const C5: Note = 72;
    /// C in octave 6 (MIDI note 84).
    pub const C6: Note = 84;
    /// C in octave 7 (MIDI note 96).
    pub const C7: Note = 96;
    /// C in octave 8 (MIDI note 108).
    pub const C8: Note = 108;

    /// A0, lowest piano key (MIDI note 21).
    pub const A0: Note = 21;
    /// A4, concert pitch 440 Hz (MIDI note 69).
    pub const A4: Note = 69;
    /// C8, highest piano key (MIDI note 108).
    pub const C8_TOP: Note = 108;

    /// D4 (MIDI note 62).
    pub const D4: Note = 62;
    /// E4 (MIDI note 64).
    pub const E4: Note = 64;
    /// F4 (MIDI note 65).
    pub const F4: Note = 65;
    /// G4 (MIDI note 67).
    pub const G4: Note = 67;
    /// B4 (MIDI note 71).
    pub const B4: Note = 71;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_note_to_freq() {
        // A4 = 440 Hz
        assert!((note_to_freq(69) - 440.0).abs() < 0.01);

        // A3 = 220 Hz
        assert!((note_to_freq(57) - 220.0).abs() < 0.01);

        // A5 = 880 Hz
        assert!((note_to_freq(81) - 880.0).abs() < 0.01);

        // Middle C (C4) ≈ 261.63 Hz
        assert!((note_to_freq(60) - 261.63).abs() < 0.1);
    }

    #[test]
    fn test_freq_to_note() {
        assert_eq!(freq_to_note(440.0), 69);
        assert_eq!(freq_to_note(220.0), 57);
        assert_eq!(freq_to_note(880.0), 81);
        assert_eq!(freq_to_note(261.63), 60);
    }

    #[test]
    fn test_note_name() {
        assert_eq!(note_name(60), "C4");
        assert_eq!(note_name(69), "A4");
        assert_eq!(note_name(61), "C#4");
        assert_eq!(note_name(0), "C-1");
        assert_eq!(note_name(127), "G9");
    }

    #[test]
    fn test_parse_note_name() {
        assert_eq!(parse_note_name("C4"), Some(60));
        assert_eq!(parse_note_name("A4"), Some(69));
        assert_eq!(parse_note_name("C#4"), Some(61));
        assert_eq!(parse_note_name("Db4"), Some(61));
        assert_eq!(parse_note_name("C-1"), Some(0));
        assert_eq!(parse_note_name("G9"), Some(127));
        assert_eq!(parse_note_name(""), None);
        assert_eq!(parse_note_name("X4"), None);
    }

    #[test]
    fn test_note_on_message() {
        let bytes = [0x90, 60, 100];
        let msg = MidiMessage::from_bytes(&bytes).unwrap();

        assert_eq!(
            msg,
            MidiMessage::NoteOn {
                channel: 0,
                note: 60,
                velocity: 100
            }
        );

        assert_eq!(msg.to_bytes(), bytes);
    }

    #[test]
    fn test_note_off_message() {
        let bytes = [0x80, 60, 64];
        let msg = MidiMessage::from_bytes(&bytes).unwrap();

        assert_eq!(
            msg,
            MidiMessage::NoteOff {
                channel: 0,
                note: 60,
                velocity: 64
            }
        );
    }

    #[test]
    fn test_note_on_with_zero_velocity() {
        // Note On with velocity 0 should be treated as Note Off
        let bytes = [0x90, 60, 0];
        let msg = MidiMessage::from_bytes(&bytes).unwrap();

        assert_eq!(
            msg,
            MidiMessage::NoteOff {
                channel: 0,
                note: 60,
                velocity: 0
            }
        );
    }

    #[test]
    fn test_control_change() {
        let bytes = [0xB0, cc::MOD_WHEEL, 64];
        let msg = MidiMessage::from_bytes(&bytes).unwrap();

        assert_eq!(
            msg,
            MidiMessage::ControlChange {
                channel: 0,
                controller: cc::MOD_WHEEL,
                value: 64
            }
        );
    }

    #[test]
    fn test_pitch_bend() {
        // Center position (no bend)
        let bytes = [0xE0, 0x00, 0x40];
        let msg = MidiMessage::from_bytes(&bytes).unwrap();

        assert_eq!(
            msg,
            MidiMessage::PitchBend {
                channel: 0,
                value: 0
            }
        );

        // Full bend up
        let bytes = [0xE0, 0x7F, 0x7F];
        let msg = MidiMessage::from_bytes(&bytes).unwrap();

        assert_eq!(
            msg,
            MidiMessage::PitchBend {
                channel: 0,
                value: 8191
            }
        );
    }

    #[test]
    fn test_pitch_bend_to_ratio() {
        // No bend = ratio of 1.0
        let ratio = pitch_bend_to_ratio(0, 2.0);
        assert!((ratio - 1.0).abs() < 0.001);

        // Full bend up with 2 semitone range = major second up
        let ratio = pitch_bend_to_ratio(8191, 2.0);
        let expected = 2.0f32.powf(2.0 / 12.0);
        assert!((ratio - expected).abs() < 0.01);
    }

    #[test]
    fn test_velocity_amplitude_conversion() {
        // Max velocity = max amplitude
        assert!((velocity_to_amplitude(127) - 1.0).abs() < 0.01);

        // Zero velocity = zero amplitude
        assert_eq!(velocity_to_amplitude(0), 0.0);

        // Roundtrip
        let vel = 100;
        let amp = velocity_to_amplitude(vel);
        let vel2 = amplitude_to_velocity(amp);
        assert_eq!(vel, vel2);
    }

    #[test]
    fn test_channel() {
        let msg = MidiMessage::NoteOn {
            channel: 5,
            note: 60,
            velocity: 100,
        };
        assert_eq!(msg.channel(), Some(5));

        let msg = MidiMessage::TimingClock;
        assert_eq!(msg.channel(), None);
    }

    #[test]
    fn test_system_messages() {
        assert_eq!(
            MidiMessage::from_bytes(&[0xF8]),
            Some(MidiMessage::TimingClock)
        );
        assert_eq!(MidiMessage::from_bytes(&[0xFA]), Some(MidiMessage::Start));
        assert_eq!(MidiMessage::from_bytes(&[0xFC]), Some(MidiMessage::Stop));
    }

    #[test]
    fn test_cc_normalized() {
        assert_eq!(cc_to_normalized(0), 0.0);
        assert_eq!(cc_to_normalized(127), 1.0);
        assert!((cc_to_normalized(64) - 0.504).abs() < 0.01);

        assert_eq!(normalized_to_cc(0.0), 0);
        assert_eq!(normalized_to_cc(1.0), 127);
        assert_eq!(normalized_to_cc(0.5), 64);
    }
}
