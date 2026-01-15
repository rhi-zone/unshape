//! WAV file import and export.
//!
//! Supports reading and writing uncompressed PCM WAV files.
//!
//! # Example
//!
//! ```no_run
//! use rhizome_resin_audio::wav::{WavFile, WavFormat};
//!
//! // Read a WAV file
//! let wav = WavFile::load("input.wav").unwrap();
//! println!("Sample rate: {}", wav.sample_rate);
//! println!("Channels: {}", wav.channels);
//!
//! // Create and save a WAV file
//! let samples = vec![0.0f32; 44100]; // 1 second of silence
//! let wav = WavFile::from_samples(samples, 44100, 1);
//! wav.save("output.wav").unwrap();
//! ```

use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::Path;

/// Error type for WAV operations.
#[derive(Debug, thiserror::Error)]
pub enum WavError {
    /// IO error.
    #[error("IO error: {0}")]
    Io(#[from] io::Error),
    /// Invalid WAV format.
    #[error("Invalid WAV format: {0}")]
    InvalidFormat(String),
    /// Unsupported format.
    #[error("Unsupported format: {0}")]
    Unsupported(String),
}

/// Result type for WAV operations.
pub type WavResult<T> = Result<T, WavError>;

/// WAV audio format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WavFormat {
    /// 8-bit unsigned PCM.
    Pcm8,
    /// 16-bit signed PCM.
    Pcm16,
    /// 24-bit signed PCM.
    Pcm24,
    /// 32-bit signed PCM.
    Pcm32,
    /// 32-bit float.
    Float32,
}

impl WavFormat {
    /// Returns the bits per sample for this format.
    pub fn bits_per_sample(&self) -> u16 {
        match self {
            WavFormat::Pcm8 => 8,
            WavFormat::Pcm16 => 16,
            WavFormat::Pcm24 => 24,
            WavFormat::Pcm32 => 32,
            WavFormat::Float32 => 32,
        }
    }

    /// Returns the audio format code.
    fn format_code(&self) -> u16 {
        match self {
            WavFormat::Float32 => 3, // IEEE float
            _ => 1,                  // PCM
        }
    }
}

/// A WAV audio file.
#[derive(Debug, Clone)]
pub struct WavFile {
    /// Sample rate in Hz.
    pub sample_rate: u32,
    /// Number of channels.
    pub channels: u16,
    /// Audio format.
    pub format: WavFormat,
    /// Interleaved sample data (normalized to -1.0 to 1.0).
    pub samples: Vec<f32>,
}

impl WavFile {
    /// Creates a new WAV file from sample data.
    ///
    /// Samples should be interleaved for multi-channel audio.
    pub fn from_samples(samples: Vec<f32>, sample_rate: u32, channels: u16) -> Self {
        Self {
            sample_rate,
            channels,
            format: WavFormat::Float32,
            samples,
        }
    }

    /// Creates a mono WAV file from sample data.
    pub fn mono(samples: Vec<f32>, sample_rate: u32) -> Self {
        Self::from_samples(samples, sample_rate, 1)
    }

    /// Creates a stereo WAV file from interleaved sample data.
    pub fn stereo(samples: Vec<f32>, sample_rate: u32) -> Self {
        Self::from_samples(samples, sample_rate, 2)
    }

    /// Creates a stereo WAV file from separate left and right channels.
    pub fn stereo_from_channels(left: &[f32], right: &[f32], sample_rate: u32) -> Self {
        let len = left.len().min(right.len());
        let mut samples = Vec::with_capacity(len * 2);

        for i in 0..len {
            samples.push(left[i]);
            samples.push(right[i]);
        }

        Self::from_samples(samples, sample_rate, 2)
    }

    /// Returns the duration in seconds.
    pub fn duration(&self) -> f32 {
        self.samples.len() as f32 / (self.sample_rate as f32 * self.channels as f32)
    }

    /// Returns the number of sample frames.
    pub fn frame_count(&self) -> usize {
        self.samples.len() / self.channels as usize
    }

    /// Returns samples for a specific channel.
    pub fn channel(&self, channel: usize) -> Vec<f32> {
        if channel >= self.channels as usize {
            return Vec::new();
        }

        self.samples
            .iter()
            .skip(channel)
            .step_by(self.channels as usize)
            .copied()
            .collect()
    }

    /// Converts to mono by averaging channels.
    pub fn to_mono(&self) -> Vec<f32> {
        if self.channels == 1 {
            return self.samples.clone();
        }

        let frame_count = self.frame_count();
        let mut mono = Vec::with_capacity(frame_count);

        for i in 0..frame_count {
            let mut sum = 0.0;
            for c in 0..self.channels as usize {
                sum += self.samples[i * self.channels as usize + c];
            }
            mono.push(sum / self.channels as f32);
        }

        mono
    }

    /// Loads a WAV file from disk.
    pub fn load<P: AsRef<Path>>(path: P) -> WavResult<Self> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);
        Self::read(&mut reader)
    }

    /// Reads a WAV file from a reader.
    pub fn read<R: Read>(reader: &mut R) -> WavResult<Self> {
        // Read RIFF header
        let mut riff = [0u8; 4];
        reader.read_exact(&mut riff)?;
        if &riff != b"RIFF" {
            return Err(WavError::InvalidFormat("Not a RIFF file".to_string()));
        }

        // Read file size (ignored)
        let mut size_buf = [0u8; 4];
        reader.read_exact(&mut size_buf)?;

        // Read WAVE identifier
        let mut wave = [0u8; 4];
        reader.read_exact(&mut wave)?;
        if &wave != b"WAVE" {
            return Err(WavError::InvalidFormat("Not a WAVE file".to_string()));
        }

        let mut format: Option<(u16, u16, u32, u16)> = None; // (format_code, channels, sample_rate, bits)
        let mut samples: Vec<f32> = Vec::new();

        // Read chunks
        loop {
            let mut chunk_id = [0u8; 4];
            if reader.read_exact(&mut chunk_id).is_err() {
                break;
            }

            let mut chunk_size = [0u8; 4];
            reader.read_exact(&mut chunk_size)?;
            let chunk_size = u32::from_le_bytes(chunk_size);

            match &chunk_id {
                b"fmt " => {
                    let mut fmt_data = vec![0u8; chunk_size as usize];
                    reader.read_exact(&mut fmt_data)?;

                    if fmt_data.len() < 16 {
                        return Err(WavError::InvalidFormat("fmt chunk too small".to_string()));
                    }

                    let format_code = u16::from_le_bytes([fmt_data[0], fmt_data[1]]);
                    let channels = u16::from_le_bytes([fmt_data[2], fmt_data[3]]);
                    let sample_rate =
                        u32::from_le_bytes([fmt_data[4], fmt_data[5], fmt_data[6], fmt_data[7]]);
                    let bits = u16::from_le_bytes([fmt_data[14], fmt_data[15]]);

                    format = Some((format_code, channels, sample_rate, bits));
                }
                b"data" => {
                    let (format_code, channels, sample_rate, bits) = format.ok_or_else(|| {
                        WavError::InvalidFormat("data chunk before fmt chunk".to_string())
                    })?;

                    let mut data = vec![0u8; chunk_size as usize];
                    reader.read_exact(&mut data)?;

                    // Convert samples based on format
                    samples = match (format_code, bits) {
                        (1, 8) => {
                            // 8-bit unsigned PCM
                            data.iter().map(|&b| (b as f32 - 128.0) / 128.0).collect()
                        }
                        (1, 16) => {
                            // 16-bit signed PCM
                            data.chunks_exact(2)
                                .map(|chunk| {
                                    let sample = i16::from_le_bytes([chunk[0], chunk[1]]);
                                    sample as f32 / 32768.0
                                })
                                .collect()
                        }
                        (1, 24) => {
                            // 24-bit signed PCM
                            data.chunks_exact(3)
                                .map(|chunk| {
                                    let sample =
                                        i32::from_le_bytes([chunk[0], chunk[1], chunk[2], 0]) >> 8;
                                    sample as f32 / 8388608.0
                                })
                                .collect()
                        }
                        (1, 32) => {
                            // 32-bit signed PCM
                            data.chunks_exact(4)
                                .map(|chunk| {
                                    let sample = i32::from_le_bytes([
                                        chunk[0], chunk[1], chunk[2], chunk[3],
                                    ]);
                                    sample as f32 / 2147483648.0
                                })
                                .collect()
                        }
                        (3, 32) => {
                            // 32-bit float
                            data.chunks_exact(4)
                                .map(|chunk| {
                                    f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])
                                })
                                .collect()
                        }
                        _ => {
                            return Err(WavError::Unsupported(format!(
                                "Unsupported format: code={}, bits={}",
                                format_code, bits
                            )));
                        }
                    };

                    let wav_format = match (format_code, bits) {
                        (1, 8) => WavFormat::Pcm8,
                        (1, 16) => WavFormat::Pcm16,
                        (1, 24) => WavFormat::Pcm24,
                        (1, 32) => WavFormat::Pcm32,
                        (3, 32) => WavFormat::Float32,
                        _ => unreachable!(),
                    };

                    return Ok(WavFile {
                        sample_rate,
                        channels,
                        format: wav_format,
                        samples,
                    });
                }
                _ => {
                    // Skip unknown chunks
                    let mut skip = vec![0u8; chunk_size as usize];
                    reader.read_exact(&mut skip)?;
                }
            }
        }

        Err(WavError::InvalidFormat("No data chunk found".to_string()))
    }

    /// Saves the WAV file to disk.
    pub fn save<P: AsRef<Path>>(&self, path: P) -> WavResult<()> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);
        self.write(&mut writer)
    }

    /// Writes the WAV file to a writer.
    pub fn write<W: Write>(&self, writer: &mut W) -> WavResult<()> {
        let bits_per_sample = self.format.bits_per_sample();
        let bytes_per_sample = bits_per_sample as u32 / 8;
        let data_size = self.samples.len() as u32 * bytes_per_sample;
        let fmt_size = 16u32;
        let file_size = 4 + (8 + fmt_size) + (8 + data_size);

        // RIFF header
        writer.write_all(b"RIFF")?;
        writer.write_all(&file_size.to_le_bytes())?;
        writer.write_all(b"WAVE")?;

        // fmt chunk
        writer.write_all(b"fmt ")?;
        writer.write_all(&fmt_size.to_le_bytes())?;
        writer.write_all(&self.format.format_code().to_le_bytes())?;
        writer.write_all(&self.channels.to_le_bytes())?;
        writer.write_all(&self.sample_rate.to_le_bytes())?;

        let byte_rate = self.sample_rate * self.channels as u32 * bytes_per_sample;
        writer.write_all(&byte_rate.to_le_bytes())?;

        let block_align = self.channels * bytes_per_sample as u16;
        writer.write_all(&block_align.to_le_bytes())?;
        writer.write_all(&bits_per_sample.to_le_bytes())?;

        // data chunk
        writer.write_all(b"data")?;
        writer.write_all(&data_size.to_le_bytes())?;

        // Write samples
        match self.format {
            WavFormat::Pcm8 => {
                for &sample in &self.samples {
                    let byte = ((sample.clamp(-1.0, 1.0) * 128.0) + 128.0) as u8;
                    writer.write_all(&[byte])?;
                }
            }
            WavFormat::Pcm16 => {
                for &sample in &self.samples {
                    let value = (sample.clamp(-1.0, 1.0) * 32767.0) as i16;
                    writer.write_all(&value.to_le_bytes())?;
                }
            }
            WavFormat::Pcm24 => {
                for &sample in &self.samples {
                    let value = (sample.clamp(-1.0, 1.0) * 8388607.0) as i32;
                    let bytes = value.to_le_bytes();
                    writer.write_all(&[bytes[0], bytes[1], bytes[2]])?;
                }
            }
            WavFormat::Pcm32 => {
                for &sample in &self.samples {
                    let value = (sample.clamp(-1.0, 1.0) * 2147483647.0) as i32;
                    writer.write_all(&value.to_le_bytes())?;
                }
            }
            WavFormat::Float32 => {
                for &sample in &self.samples {
                    writer.write_all(&sample.to_le_bytes())?;
                }
            }
        }

        writer.flush()?;
        Ok(())
    }

    /// Converts to a different format.
    pub fn convert(&self, format: WavFormat) -> Self {
        Self {
            sample_rate: self.sample_rate,
            channels: self.channels,
            format,
            samples: self.samples.clone(),
        }
    }

    /// Resamples to a different sample rate using linear interpolation.
    pub fn resample(&self, target_sample_rate: u32) -> Self {
        if target_sample_rate == self.sample_rate {
            return self.clone();
        }

        let ratio = target_sample_rate as f32 / self.sample_rate as f32;
        let new_frame_count = (self.frame_count() as f32 * ratio) as usize;
        let mut new_samples = Vec::with_capacity(new_frame_count * self.channels as usize);

        for frame in 0..new_frame_count {
            let src_pos = frame as f32 / ratio;
            let src_frame = src_pos as usize;
            let frac = src_pos - src_frame as f32;

            for ch in 0..self.channels as usize {
                let idx1 = src_frame * self.channels as usize + ch;
                let idx2 =
                    (src_frame + 1).min(self.frame_count() - 1) * self.channels as usize + ch;

                let s1 = self.samples.get(idx1).copied().unwrap_or(0.0);
                let s2 = self.samples.get(idx2).copied().unwrap_or(0.0);

                new_samples.push(s1 + (s2 - s1) * frac);
            }
        }

        Self {
            sample_rate: target_sample_rate,
            channels: self.channels,
            format: self.format,
            samples: new_samples,
        }
    }
}

/// Creates a WAV file from raw bytes (for embedded audio).
pub fn from_bytes(data: &[u8]) -> WavResult<WavFile> {
    let mut cursor = std::io::Cursor::new(data);
    WavFile::read(&mut cursor)
}

/// Writes a WAV file to a byte vector.
pub fn to_bytes(wav: &WavFile) -> WavResult<Vec<u8>> {
    let mut buffer = Vec::new();
    wav.write(&mut buffer)?;
    Ok(buffer)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wav_creation() {
        let samples = vec![0.0f32; 44100];
        let wav = WavFile::mono(samples, 44100);

        assert_eq!(wav.sample_rate, 44100);
        assert_eq!(wav.channels, 1);
        assert_eq!(wav.frame_count(), 44100);
        assert!((wav.duration() - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_wav_stereo() {
        let left = vec![1.0f32; 100];
        let right = vec![-1.0f32; 100];
        let wav = WavFile::stereo_from_channels(&left, &right, 44100);

        assert_eq!(wav.channels, 2);
        assert_eq!(wav.samples.len(), 200);

        let left_ch = wav.channel(0);
        let right_ch = wav.channel(1);

        assert_eq!(left_ch.len(), 100);
        assert_eq!(right_ch.len(), 100);
        assert_eq!(left_ch[0], 1.0);
        assert_eq!(right_ch[0], -1.0);
    }

    #[test]
    fn test_wav_to_mono() {
        let samples = vec![1.0, -1.0, 0.5, 0.5]; // 2 stereo frames
        let wav = WavFile::stereo(samples, 44100);

        let mono = wav.to_mono();
        assert_eq!(mono.len(), 2);
        assert_eq!(mono[0], 0.0); // (1 + -1) / 2
        assert_eq!(mono[1], 0.5); // (0.5 + 0.5) / 2
    }

    #[test]
    fn test_wav_roundtrip() {
        let original = vec![0.5, -0.5, 0.25, -0.25];
        let wav = WavFile::from_samples(original.clone(), 44100, 1);

        // Write to bytes
        let bytes = to_bytes(&wav).unwrap();

        // Read back
        let loaded = from_bytes(&bytes).unwrap();

        assert_eq!(loaded.sample_rate, wav.sample_rate);
        assert_eq!(loaded.channels, wav.channels);
        assert_eq!(loaded.samples.len(), wav.samples.len());

        for (a, b) in loaded.samples.iter().zip(original.iter()) {
            assert!((a - b).abs() < 0.0001);
        }
    }

    #[test]
    fn test_wav_pcm16_roundtrip() {
        let original = vec![0.5, -0.5, 0.25, -0.25];
        let wav = WavFile::from_samples(original.clone(), 44100, 1).convert(WavFormat::Pcm16);

        let bytes = to_bytes(&wav).unwrap();
        let loaded = from_bytes(&bytes).unwrap();

        // 16-bit has less precision
        for (a, b) in loaded.samples.iter().zip(original.iter()) {
            assert!((a - b).abs() < 0.001);
        }
    }

    #[test]
    fn test_wav_resample() {
        let samples: Vec<f32> = (0..44100).map(|i| (i as f32 / 100.0).sin()).collect();
        let wav = WavFile::mono(samples, 44100);

        let resampled = wav.resample(22050);

        assert_eq!(resampled.sample_rate, 22050);
        assert!((resampled.duration() - wav.duration()).abs() < 0.1);
    }

    #[test]
    fn test_wav_format_bits() {
        assert_eq!(WavFormat::Pcm8.bits_per_sample(), 8);
        assert_eq!(WavFormat::Pcm16.bits_per_sample(), 16);
        assert_eq!(WavFormat::Pcm24.bits_per_sample(), 24);
        assert_eq!(WavFormat::Pcm32.bits_per_sample(), 32);
        assert_eq!(WavFormat::Float32.bits_per_sample(), 32);
    }

    #[test]
    fn test_invalid_wav() {
        let bad_data = b"not a wav file";
        let result = from_bytes(bad_data);
        assert!(result.is_err());
    }
}
