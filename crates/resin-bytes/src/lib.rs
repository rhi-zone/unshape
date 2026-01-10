//! Raw byte reinterpretation utilities for glitch-art style domain crossing.
//!
//! This crate provides safe byte casting between different numeric representations,
//! enabling creative reinterpretation of data across domains. Load any file as audio
//! samples, interpret audio as vertex positions, etc.
//!
//! # Example
//!
//! ```
//! use rhizome_resin_bytes::*;
//!
//! // Interpret raw bytes as f32 audio samples
//! let bytes: Vec<u8> = vec![0, 0, 128, 63, 0, 0, 0, 64]; // 1.0, 2.0 in little-endian
//! let samples = bytes_as_f32(&bytes).unwrap();
//! assert_eq!(samples.len(), 2);
//!
//! // Normalize arbitrary float values to [-1, 1]
//! let normalized = normalize_f32(samples);
//! ```

/// Reinterprets raw bytes as f32 values.
///
/// The bytes are cast directly to f32 values. Length must be divisible by 4.
/// Values are not normalized - they may be outside [-1, 1] or even NaN/Inf.
///
/// # Example
/// ```
/// use rhizome_resin_bytes::bytes_as_f32;
///
/// let jpeg_bytes = std::fs::read("Cargo.toml").unwrap(); // any file works
/// if let Some(samples) = bytes_as_f32(&jpeg_bytes) {
///     println!("Got {} float samples", samples.len());
/// }
/// ```
pub fn bytes_as_f32(bytes: &[u8]) -> Option<&[f32]> {
    if bytes.len() % 4 != 0 {
        return None;
    }
    Some(bytemuck::cast_slice(bytes))
}

/// Reinterprets raw bytes as i16 values.
///
/// Common format for raw PCM audio. Length must be divisible by 2.
pub fn bytes_as_i16(bytes: &[u8]) -> Option<&[i16]> {
    if bytes.len() % 2 != 0 {
        return None;
    }
    Some(bytemuck::cast_slice(bytes))
}

/// Reinterprets raw bytes as u16 values.
///
/// Length must be divisible by 2.
pub fn bytes_as_u16(bytes: &[u8]) -> Option<&[u16]> {
    if bytes.len() % 2 != 0 {
        return None;
    }
    Some(bytemuck::cast_slice(bytes))
}

/// Reinterprets raw bytes as i32 values.
///
/// Length must be divisible by 4.
pub fn bytes_as_i32(bytes: &[u8]) -> Option<&[i32]> {
    if bytes.len() % 4 != 0 {
        return None;
    }
    Some(bytemuck::cast_slice(bytes))
}

/// Reinterprets raw bytes as u32 values.
///
/// Length must be divisible by 4.
pub fn bytes_as_u32(bytes: &[u8]) -> Option<&[u32]> {
    if bytes.len() % 4 != 0 {
        return None;
    }
    Some(bytemuck::cast_slice(bytes))
}

/// Reinterprets raw bytes as RGBA pixels (4 bytes per pixel).
///
/// Returns pixel data as [u8; 4] arrays.
pub fn bytes_as_rgba(bytes: &[u8]) -> Option<&[[u8; 4]]> {
    if bytes.len() % 4 != 0 {
        return None;
    }
    Some(bytemuck::cast_slice(bytes))
}

/// Reinterprets raw bytes as RGB pixels (3 bytes per pixel).
///
/// Returns pixel data as [u8; 3] arrays.
pub fn bytes_as_rgb(bytes: &[u8]) -> Option<&[[u8; 3]]> {
    if bytes.len() % 3 != 0 {
        return None;
    }
    Some(bytemuck::cast_slice(bytes))
}

/// Reinterprets raw bytes as 2D float coordinates (8 bytes per point).
pub fn bytes_as_xy(bytes: &[u8]) -> Option<&[[f32; 2]]> {
    if bytes.len() % 8 != 0 {
        return None;
    }
    Some(bytemuck::cast_slice(bytes))
}

/// Reinterprets raw bytes as 3D float coordinates (12 bytes per point).
pub fn bytes_as_xyz(bytes: &[u8]) -> Option<&[[f32; 3]]> {
    if bytes.len() % 12 != 0 {
        return None;
    }
    Some(bytemuck::cast_slice(bytes))
}

/// Reinterprets raw bytes as 4D float values (16 bytes per element).
pub fn bytes_as_xyzw(bytes: &[u8]) -> Option<&[[f32; 4]]> {
    if bytes.len() % 16 != 0 {
        return None;
    }
    Some(bytemuck::cast_slice(bytes))
}

/// Converts i16 samples to f32 in [-1, 1] range.
pub fn i16_to_f32(samples: &[i16]) -> Vec<f32> {
    samples
        .iter()
        .map(|&s| s as f32 / i16::MAX as f32)
        .collect()
}

/// Converts u8 samples to f32 in [-1, 1] range.
///
/// 0 maps to -1, 128 maps to 0, 255 maps to ~1.
pub fn u8_to_f32(samples: &[u8]) -> Vec<f32> {
    samples.iter().map(|&s| (s as f32 / 128.0) - 1.0).collect()
}

/// Converts u8 values to f32 in [0, 1] range.
///
/// 0 maps to 0, 255 maps to 1.
pub fn u8_to_f32_unsigned(samples: &[u8]) -> Vec<f32> {
    samples.iter().map(|&s| s as f32 / 255.0).collect()
}

/// Converts f32 samples to i16, clamping to valid range.
pub fn f32_to_i16(samples: &[f32]) -> Vec<i16> {
    samples
        .iter()
        .map(|&s| (s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16)
        .collect()
}

/// Converts f32 samples to u8 in [0, 255] range.
pub fn f32_to_u8(samples: &[f32]) -> Vec<u8> {
    samples
        .iter()
        .map(|&s| (s.clamp(0.0, 1.0) * 255.0) as u8)
        .collect()
}

/// Normalizes f32 values to [-1, 1] range.
///
/// Useful after raw byte casting since values may be arbitrary floats.
/// NaN and Inf values are replaced with 0.
pub fn normalize_f32(samples: &[f32]) -> Vec<f32> {
    let max = samples
        .iter()
        .filter(|s| s.is_finite())
        .map(|s| s.abs())
        .fold(0.0f32, |a, b| a.max(b));

    if max == 0.0 {
        return samples
            .iter()
            .map(|&s| if s.is_finite() { s } else { 0.0 })
            .collect();
    }

    samples
        .iter()
        .map(|&s| if s.is_finite() { s / max } else { 0.0 })
        .collect()
}

/// Normalizes f32 values to [0, 1] range.
///
/// NaN and Inf values are replaced with 0.
pub fn normalize_f32_unsigned(samples: &[f32]) -> Vec<f32> {
    let (min, max) = samples
        .iter()
        .filter(|s| s.is_finite())
        .fold((f32::MAX, f32::MIN), |(min, max), &s| {
            (min.min(s), max.max(s))
        });

    let range = max - min;
    if range == 0.0 {
        return samples
            .iter()
            .map(|&s| if s.is_finite() { 0.5 } else { 0.0 })
            .collect();
    }

    samples
        .iter()
        .map(|&s| {
            if s.is_finite() {
                (s - min) / range
            } else {
                0.0
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bytes_as_f32() {
        let bytes: Vec<u8> = vec![0, 0, 128, 63, 0, 0, 0, 64]; // 1.0f32, 2.0f32 in little-endian
        let floats = bytes_as_f32(&bytes).unwrap();
        assert_eq!(floats.len(), 2);
        assert!((floats[0] - 1.0).abs() < 0.0001);
        assert!((floats[1] - 2.0).abs() < 0.0001);
    }

    #[test]
    fn test_bytes_as_f32_invalid_len() {
        let bytes = vec![0, 0, 128]; // 3 bytes, not divisible by 4
        assert!(bytes_as_f32(&bytes).is_none());
    }

    #[test]
    fn test_bytes_as_i16() {
        let bytes: Vec<u8> = vec![0, 0, 255, 127]; // 0i16, 32767i16 in little-endian
        let samples = bytes_as_i16(&bytes).unwrap();
        assert_eq!(samples.len(), 2);
        assert_eq!(samples[0], 0);
        assert_eq!(samples[1], i16::MAX);
    }

    #[test]
    fn test_bytes_as_rgba() {
        let bytes = vec![255, 0, 0, 255, 0, 255, 0, 128];
        let pixels = bytes_as_rgba(&bytes).unwrap();
        assert_eq!(pixels.len(), 2);
        assert_eq!(pixels[0], [255, 0, 0, 255]);
        assert_eq!(pixels[1], [0, 255, 0, 128]);
    }

    #[test]
    fn test_bytes_as_xyz() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&1.0f32.to_le_bytes());
        bytes.extend_from_slice(&2.0f32.to_le_bytes());
        bytes.extend_from_slice(&3.0f32.to_le_bytes());

        let verts = bytes_as_xyz(&bytes).unwrap();
        assert_eq!(verts.len(), 1);
        assert!((verts[0][0] - 1.0).abs() < 0.0001);
        assert!((verts[0][1] - 2.0).abs() < 0.0001);
        assert!((verts[0][2] - 3.0).abs() < 0.0001);
    }

    #[test]
    fn test_i16_to_f32() {
        let samples = vec![0i16, i16::MAX, i16::MIN];
        let floats = i16_to_f32(&samples);
        assert!((floats[0] - 0.0).abs() < 0.0001);
        assert!((floats[1] - 1.0).abs() < 0.0001);
        assert!((floats[2] - (-1.0)).abs() < 0.01);
    }

    #[test]
    fn test_u8_to_f32() {
        let samples = vec![0u8, 128, 255];
        let floats = u8_to_f32(&samples);
        assert!((floats[0] - (-1.0)).abs() < 0.01);
        assert!((floats[1] - 0.0).abs() < 0.01);
        assert!((floats[2] - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_u8_to_f32_unsigned() {
        let samples = vec![0u8, 128, 255];
        let floats = u8_to_f32_unsigned(&samples);
        assert!((floats[0] - 0.0).abs() < 0.01);
        assert!((floats[1] - 0.5).abs() < 0.01);
        assert!((floats[2] - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_f32_to_i16() {
        let samples = vec![0.0, 1.0, -1.0, 0.5];
        let ints = f32_to_i16(&samples);
        assert_eq!(ints[0], 0);
        assert_eq!(ints[1], i16::MAX);
        assert_eq!(ints[2], -i16::MAX);
        assert!((ints[3] - i16::MAX / 2).abs() < 2);
    }

    #[test]
    fn test_normalize_f32() {
        let samples = vec![0.0, 5.0, -10.0, 2.5];
        let normalized = normalize_f32(&samples);
        assert!((normalized[0] - 0.0).abs() < 0.0001);
        assert!((normalized[1] - 0.5).abs() < 0.0001);
        assert!((normalized[2] - (-1.0)).abs() < 0.0001);
        assert!((normalized[3] - 0.25).abs() < 0.0001);
    }

    #[test]
    fn test_normalize_with_nan() {
        let samples = vec![1.0, f32::NAN, -2.0];
        let normalized = normalize_f32(&samples);
        assert!((normalized[0] - 0.5).abs() < 0.0001);
        assert_eq!(normalized[1], 0.0);
        assert!((normalized[2] - (-1.0)).abs() < 0.0001);
    }

    #[test]
    fn test_normalize_f32_unsigned() {
        let samples = vec![0.0, 5.0, 10.0];
        let normalized = normalize_f32_unsigned(&samples);
        assert!((normalized[0] - 0.0).abs() < 0.0001);
        assert!((normalized[1] - 0.5).abs() < 0.0001);
        assert!((normalized[2] - 1.0).abs() < 0.0001);
    }

    #[test]
    fn test_roundtrip_i16() {
        let original = vec![0.0f32, 0.5, -0.5, 1.0, -1.0];
        let as_i16 = f32_to_i16(&original);
        let back = i16_to_f32(&as_i16);
        for (a, b) in original.iter().zip(back.iter()) {
            assert!((a - b).abs() < 0.001);
        }
    }
}
