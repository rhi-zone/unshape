//! Image operation pipeline pattern-matching optimizer.
//!
//! Recognizes common sequences of image operations and replaces them with
//! fused implementations that avoid intermediate allocations.
//!
//! # Overview
//!
//! Image pipelines built from primitives can be automatically optimized by
//! recognizing common patterns and replacing them with monomorphized
//! implementations.
//!
//! # Example
//!
//! ```ignore
//! use unshape_image::optimizer::ImageOptimizer;
//! use unshape_image::{Fft2d, FreqRadialMul, Ifft2d};
//! use unshape_op::DynOp;
//!
//! let optimizer = ImageOptimizer::new();
//! let pipeline: Vec<Box<dyn DynOp>> = vec![
//!     Box::new(Fft2d),
//!     Box::new(FreqRadialMul { cutoff: 0.3, low_pass: true }),
//!     Box::new(Ifft2d),
//! ];
//! let optimized = optimizer.optimize(pipeline);
//! // Now a single LowPassFreqOptimized op
//! ```

use unshape_op::DynOp;

use crate::ImageField;
use crate::freq::{Fft2d, FreqRadialMul, Ifft2d};
use crate::kernel::{GaussianBlur, SeparableConvolve};

/// Result of a successful pattern match.
pub struct PatternMatch {
    /// How many ops from the match start this pattern consumed.
    pub consumed: usize,
    /// Replacement ops (semantically equivalent to the consumed ops).
    pub replacements: Vec<Box<dyn DynOp>>,
}

/// Recognizes a pattern in a suffix of image ops and replaces it with an optimized equivalent.
pub trait ImagePattern: Send + Sync {
    /// Try to match at the start of `ops` (index 0 of the provided slice).
    ///
    /// Returns `Some(PatternMatch)` if matched, `None` otherwise.
    fn try_match(&self, ops: &[&dyn DynOp]) -> Option<PatternMatch>;

    /// Human-readable name for debugging.
    fn name(&self) -> &'static str;
}

/// Image operation pipeline optimizer.
///
/// Applies registered [`ImagePattern`]s in a single left-to-right pass over
/// the pipeline, replacing matched sequences with optimized equivalents.
pub struct ImageOptimizer {
    patterns: Vec<Box<dyn ImagePattern>>,
}

impl Default for ImageOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

impl ImageOptimizer {
    /// Creates an optimizer with all built-in patterns registered.
    pub fn new() -> Self {
        Self {
            patterns: vec![
                Box::new(LowPassFreqPattern),
                Box::new(HighPassFreqPattern),
                Box::new(GaussianBlurCombinePattern),
                Box::new(SeparableKernelPattern),
            ],
        }
    }

    /// Creates an optimizer with no patterns.
    pub fn empty() -> Self {
        Self {
            patterns: Vec::new(),
        }
    }

    /// Adds a custom pattern to this optimizer.
    pub fn with_pattern(mut self, pattern: impl ImagePattern + 'static) -> Self {
        self.patterns.push(Box::new(pattern));
        self
    }

    /// Optimizes a pipeline of image operations.
    ///
    /// Scans the pipeline left-to-right. When a pattern matches at position `i`,
    /// the matched ops are replaced with the pattern's replacements and scanning
    /// resumes after the replacement.
    pub fn optimize(&self, pipeline: Vec<Box<dyn DynOp>>) -> Vec<Box<dyn DynOp>> {
        let mut result: Vec<Box<dyn DynOp>> = Vec::with_capacity(pipeline.len());
        // Convert to a deque-like structure for consuming from the front.
        let mut remaining: std::collections::VecDeque<Box<dyn DynOp>> = pipeline.into();

        while !remaining.is_empty() {
            // Build a slice of refs for pattern matching.
            let refs: Vec<&dyn DynOp> = remaining.iter().map(|b| b.as_ref()).collect();

            let matched = self.patterns.iter().find_map(|p| p.try_match(&refs));

            match matched {
                Some(m) => {
                    // Drop consumed ops from the front.
                    for _ in 0..m.consumed {
                        remaining.pop_front();
                    }
                    result.extend(m.replacements);
                }
                None => {
                    // No pattern matched; pass through the first op.
                    if let Some(op) = remaining.pop_front() {
                        result.push(op);
                    }
                }
            }
        }

        result
    }
}

// ============================================================================
// Helper
// ============================================================================

/// Returns the `type_name()` of a `DynOp` reference.
fn op_name(op: &dyn DynOp) -> &'static str {
    op.type_name()
}

// ============================================================================
// Pattern: FFT → RadialMul(low_pass=true) → IFFT → LowPassFreqOptimized
// ============================================================================

/// Matches `Fft2d → FreqRadialMul { low_pass: true } → Ifft2d` and replaces
/// with a single [`LowPassFreqOptimized`] op.
pub struct LowPassFreqPattern;

impl ImagePattern for LowPassFreqPattern {
    fn name(&self) -> &'static str {
        "LowPassFreqPattern"
    }

    fn try_match(&self, ops: &[&dyn DynOp]) -> Option<PatternMatch> {
        if ops.len() < 3 {
            return None;
        }
        if op_name(ops[0]) != "resin::Fft2d" {
            return None;
        }
        if op_name(ops[1]) != "resin::FreqRadialMul" {
            return None;
        }
        if op_name(ops[2]) != "resin::Ifft2d" {
            return None;
        }

        let params = ops[1].params();
        let cutoff = params["cutoff"].as_f64().unwrap_or(0.5) as f32;
        let low_pass = params["low_pass"].as_bool().unwrap_or(true);

        if !low_pass {
            return None;
        }

        Some(PatternMatch {
            consumed: 3,
            replacements: vec![Box::new(LowPassFreqOptimized { cutoff })],
        })
    }
}

// ============================================================================
// Pattern: FFT → RadialMul(low_pass=false) → IFFT → HighPassFreqOptimized
// ============================================================================

/// Matches `Fft2d → FreqRadialMul { low_pass: false } → Ifft2d` and replaces
/// with a single [`HighPassFreqOptimized`] op.
pub struct HighPassFreqPattern;

impl ImagePattern for HighPassFreqPattern {
    fn name(&self) -> &'static str {
        "HighPassFreqPattern"
    }

    fn try_match(&self, ops: &[&dyn DynOp]) -> Option<PatternMatch> {
        if ops.len() < 3 {
            return None;
        }
        if op_name(ops[0]) != "resin::Fft2d" {
            return None;
        }
        if op_name(ops[1]) != "resin::FreqRadialMul" {
            return None;
        }
        if op_name(ops[2]) != "resin::Ifft2d" {
            return None;
        }

        let params = ops[1].params();
        let cutoff = params["cutoff"].as_f64().unwrap_or(0.5) as f32;
        let low_pass = params["low_pass"].as_bool().unwrap_or(true);

        if low_pass {
            return None;
        }

        Some(PatternMatch {
            consumed: 3,
            replacements: vec![Box::new(HighPassFreqOptimized { cutoff })],
        })
    }
}

// ============================================================================
// Pattern: GaussianBlur → GaussianBlur → single GaussianBlur with combined sigma
// ============================================================================

/// Matches `GaussianBlur { sigma: s1 } → GaussianBlur { sigma: s2 }` and replaces
/// with `GaussianBlur { sigma: sqrt(s1² + s2²) }`.
///
/// This is valid because the Gaussian function is closed under convolution:
/// `G(σ₁) * G(σ₂) = G(√(σ₁² + σ₂²))`.
pub struct GaussianBlurCombinePattern;

impl ImagePattern for GaussianBlurCombinePattern {
    fn name(&self) -> &'static str {
        "GaussianBlurCombinePattern"
    }

    fn try_match(&self, ops: &[&dyn DynOp]) -> Option<PatternMatch> {
        if ops.len() < 2 {
            return None;
        }
        if op_name(ops[0]) != "resin::GaussianBlur" {
            return None;
        }
        if op_name(ops[1]) != "resin::GaussianBlur" {
            return None;
        }

        let p0 = ops[0].params();
        let p1 = ops[1].params();
        let s1 = p0["sigma"].as_f64().unwrap_or(1.0) as f32;
        let s2 = p1["sigma"].as_f64().unwrap_or(1.0) as f32;
        let combined_sigma = (s1 * s1 + s2 * s2).sqrt();

        Some(PatternMatch {
            consumed: 2,
            replacements: vec![Box::new(GaussianBlur {
                sigma: combined_sigma,
            })],
        })
    }
}

// ============================================================================
// Pattern: Convolve with separable kernel → SeparableConvolve
// ============================================================================

/// Matches a `Convolve` whose kernel factors as an outer product of two 1D vectors
/// and replaces it with a [`SeparableConvolve`] (two 1D passes instead of one 2D pass).
pub struct SeparableKernelPattern;

impl ImagePattern for SeparableKernelPattern {
    fn name(&self) -> &'static str {
        "SeparableKernelPattern"
    }

    fn try_match(&self, ops: &[&dyn DynOp]) -> Option<PatternMatch> {
        if ops.is_empty() {
            return None;
        }
        if op_name(ops[0]) != "resin::Convolve" {
            return None;
        }

        let params = ops[0].params();
        let kernel = params.get("kernel")?;
        let weights_json = kernel.get("weights")?;
        let size_json = kernel.get("size")?;

        let size = size_json.as_u64()? as usize;
        let weights: Vec<f32> = weights_json
            .as_array()?
            .iter()
            .filter_map(|v| v.as_f64().map(|f| f as f32))
            .collect();

        if weights.len() != size * size {
            return None;
        }

        let (row, col) = extract_separable_factors(&weights, size)?;

        Some(PatternMatch {
            consumed: 1,
            replacements: vec![Box::new(SeparableConvolve { row, col })],
        })
    }
}

/// Attempts to factor a 2D kernel into row ⊗ col vectors.
///
/// Tests whether `kernel[r][c] ≈ col[r] * row[c]` for all r, c.
/// Returns `None` if the relative reconstruction error exceeds 1%.
fn extract_separable_factors(weights: &[f32], size: usize) -> Option<(Vec<f32>, Vec<f32>)> {
    // Find the row with maximum absolute sum (carries most signal).
    let mut best_row_idx = 0;
    let mut best_row_sum = 0.0f32;
    for r in 0..size {
        let s: f32 = weights[r * size..(r + 1) * size]
            .iter()
            .map(|x| x.abs())
            .sum();
        if s > best_row_sum {
            best_row_sum = s;
            best_row_idx = r;
        }
    }

    if best_row_sum < 1e-10 {
        return None; // Zero kernel
    }

    let row: Vec<f32> = weights[best_row_idx * size..(best_row_idx + 1) * size].to_vec();

    // Determine col[r] from the center column: col[r] = w[r][center] / row[center]
    let center = size / 2;
    let scale = row[center];
    if scale.abs() < 1e-10 {
        return None;
    }

    let col: Vec<f32> = (0..size)
        .map(|r| weights[r * size + center] / scale)
        .collect();

    // Verify: outer(col, row) ≈ weights.
    let mut max_error: f32 = 0.0;
    let mut max_val: f32 = 0.0;
    for r in 0..size {
        for c in 0..size {
            let approx = col[r] * row[c];
            let actual = weights[r * size + c];
            max_error = max_error.max((approx - actual).abs());
            max_val = max_val.max(actual.abs());
        }
    }

    let relative_error = if max_val > 1e-10 {
        max_error / max_val
    } else {
        0.0
    };

    if relative_error > 0.01 {
        return None;
    }

    Some((row, col))
}

// ============================================================================
// Optimized op: LowPassFreqOptimized
// ============================================================================

/// Fused low-pass frequency filter: FFT + radial mask + IFFT in one operation.
///
/// Equivalent to `Fft2d → FreqRadialMul { cutoff, low_pass: true } → Ifft2d`
/// but avoids creating separate intermediate frequency-domain image allocations.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct LowPassFreqOptimized {
    /// Cutoff frequency as a fraction of Nyquist (0.0–1.0).
    /// Frequencies below this are passed; above are attenuated.
    pub cutoff: f32,
}

impl LowPassFreqOptimized {
    /// Creates a low-pass filter with the given cutoff.
    pub fn new(cutoff: f32) -> Self {
        Self { cutoff }
    }

    /// Applies low-pass frequency filtering in a single fused pass.
    pub fn apply(&self, image: &ImageField) -> ImageField {
        let (mut real, mut imag) = Fft2d.apply(image);
        FreqRadialMul {
            cutoff: self.cutoff,
            low_pass: true,
        }
        .apply_inplace(&mut real, &mut imag);
        Ifft2d.apply(&real, &imag)
    }
}

impl DynOp for LowPassFreqOptimized {
    fn type_name(&self) -> &'static str {
        "resin::LowPassFreqOptimized"
    }

    fn input_type(&self) -> unshape_op::OpType {
        unshape_op::OpType::of::<ImageField>("ImageField")
    }

    fn output_type(&self) -> unshape_op::OpType {
        unshape_op::OpType::of::<ImageField>("ImageField")
    }

    fn apply_dyn(
        &self,
        input: unshape_op::OpValue,
    ) -> Result<unshape_op::OpValue, unshape_op::OpError> {
        let img: ImageField = input.downcast()?;
        let result = self.apply(&img);
        Ok(unshape_op::OpValue::new(
            unshape_op::OpType::of::<ImageField>("ImageField"),
            result,
        ))
    }

    fn params(&self) -> serde_json::Value {
        serde_json::json!({ "cutoff": self.cutoff })
    }
}

// ============================================================================
// Optimized op: HighPassFreqOptimized
// ============================================================================

/// Fused high-pass frequency filter: FFT + radial mask + IFFT in one operation.
///
/// Equivalent to `Fft2d → FreqRadialMul { cutoff, low_pass: false } → Ifft2d`
/// but avoids creating separate intermediate frequency-domain image allocations.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct HighPassFreqOptimized {
    /// Cutoff frequency as a fraction of Nyquist (0.0–1.0).
    /// Frequencies above this are passed; below are attenuated.
    pub cutoff: f32,
}

impl HighPassFreqOptimized {
    /// Creates a high-pass filter with the given cutoff.
    pub fn new(cutoff: f32) -> Self {
        Self { cutoff }
    }

    /// Applies high-pass frequency filtering in a single fused pass.
    pub fn apply(&self, image: &ImageField) -> ImageField {
        let (mut real, mut imag) = Fft2d.apply(image);
        FreqRadialMul {
            cutoff: self.cutoff,
            low_pass: false,
        }
        .apply_inplace(&mut real, &mut imag);
        Ifft2d.apply(&real, &imag)
    }
}

impl DynOp for HighPassFreqOptimized {
    fn type_name(&self) -> &'static str {
        "resin::HighPassFreqOptimized"
    }

    fn input_type(&self) -> unshape_op::OpType {
        unshape_op::OpType::of::<ImageField>("ImageField")
    }

    fn output_type(&self) -> unshape_op::OpType {
        unshape_op::OpType::of::<ImageField>("ImageField")
    }

    fn apply_dyn(
        &self,
        input: unshape_op::OpValue,
    ) -> Result<unshape_op::OpValue, unshape_op::OpError> {
        let img: ImageField = input.downcast()?;
        let result = self.apply(&img);
        Ok(unshape_op::OpValue::new(
            unshape_op::OpType::of::<ImageField>("ImageField"),
            result,
        ))
    }

    fn params(&self) -> serde_json::Value {
        serde_json::json!({ "cutoff": self.cutoff })
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ImageField;
    use crate::freq::{Fft2d, FreqRadialMul, Ifft2d};
    use crate::kernel::{Convolve, GaussianBlur, Kernel};

    fn test_image() -> ImageField {
        ImageField::from_raw(vec![[0.5f32, 0.5, 0.5, 1.0]; 64], 8, 8)
    }

    #[test]
    fn test_low_pass_pattern_match() {
        let ops: Vec<Box<dyn DynOp>> = vec![
            Box::new(Fft2d),
            Box::new(FreqRadialMul {
                cutoff: 0.3,
                low_pass: true,
            }),
            Box::new(Ifft2d),
        ];

        let optimizer = ImageOptimizer::new();
        let result = optimizer.optimize(ops);

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].type_name(), "resin::LowPassFreqOptimized");
    }

    #[test]
    fn test_high_pass_pattern_match() {
        let ops: Vec<Box<dyn DynOp>> = vec![
            Box::new(Fft2d),
            Box::new(FreqRadialMul {
                cutoff: 0.7,
                low_pass: false,
            }),
            Box::new(Ifft2d),
        ];

        let optimizer = ImageOptimizer::new();
        let result = optimizer.optimize(ops);

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].type_name(), "resin::HighPassFreqOptimized");
    }

    #[test]
    fn test_gaussian_blur_combine_pattern() {
        let ops: Vec<Box<dyn DynOp>> = vec![
            Box::new(GaussianBlur { sigma: 3.0 }),
            Box::new(GaussianBlur { sigma: 4.0 }),
        ];

        let optimizer = ImageOptimizer::new();
        let result = optimizer.optimize(ops);

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].type_name(), "resin::GaussianBlur");

        let params = result[0].params();
        let sigma = params["sigma"].as_f64().unwrap() as f32;
        // sqrt(3² + 4²) = sqrt(9 + 16) = sqrt(25) = 5
        assert!(
            (sigma - 5.0).abs() < 1e-5,
            "expected sigma=5.0, got {sigma}"
        );
    }

    #[test]
    fn test_separable_kernel_detection() {
        // 3x3 Gaussian blur kernel is separable.
        let kernel = Kernel::gaussian_blur_3x3();
        let ops: Vec<Box<dyn DynOp>> = vec![Box::new(Convolve::new(kernel))];

        let optimizer = ImageOptimizer::new();
        let result = optimizer.optimize(ops);

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].type_name(), "resin::SeparableConvolve");
    }

    #[test]
    fn test_non_separable_kernel_not_replaced() {
        // Identity matrix is NOT rank-1, so it's not separable.
        let weights = vec![1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let kernel = Kernel::new(weights, 3);
        let ops: Vec<Box<dyn DynOp>> = vec![Box::new(Convolve::new(kernel))];

        let optimizer = ImageOptimizer::new();
        let result = optimizer.optimize(ops);

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].type_name(), "resin::Convolve");
    }

    #[test]
    fn test_no_partial_match() {
        // Only two ops of a three-op pattern → no optimization.
        let ops: Vec<Box<dyn DynOp>> = vec![
            Box::new(Fft2d),
            Box::new(FreqRadialMul {
                cutoff: 0.3,
                low_pass: true,
            }),
            // No Ifft2d
        ];

        let optimizer = ImageOptimizer::new();
        let result = optimizer.optimize(ops);

        assert_eq!(result.len(), 2);
        assert_eq!(result[0].type_name(), "resin::Fft2d");
        assert_eq!(result[1].type_name(), "resin::FreqRadialMul");
    }

    #[test]
    fn test_low_pass_optimized_same_result_as_manual() {
        let image = test_image();

        // Manual FFT → mask → IFFT.
        let (mut real, mut imag) = Fft2d.apply(&image);
        FreqRadialMul {
            cutoff: 0.4,
            low_pass: true,
        }
        .apply_inplace(&mut real, &mut imag);
        let manual_result = Ifft2d.apply(&real, &imag);

        // Fused.
        let fused_result = LowPassFreqOptimized { cutoff: 0.4 }.apply(&image);

        assert_eq!(manual_result.dimensions(), fused_result.dimensions());
        for (a, b) in manual_result.data.iter().zip(fused_result.data.iter()) {
            for ch in 0..4 {
                assert!(
                    (a[ch] - b[ch]).abs() < 1e-5,
                    "channel {ch}: {} vs {}",
                    a[ch],
                    b[ch]
                );
            }
        }
    }

    #[test]
    fn test_mixed_pipeline() {
        use crate::adjust::ChromaticAberration;

        let ops: Vec<Box<dyn DynOp>> = vec![
            Box::new(ChromaticAberration::new(0.003)),
            Box::new(Fft2d),
            Box::new(FreqRadialMul {
                cutoff: 0.5,
                low_pass: true,
            }),
            Box::new(Ifft2d),
            Box::new(GaussianBlur { sigma: 1.0 }),
        ];

        let optimizer = ImageOptimizer::new();
        let result = optimizer.optimize(ops);

        // ChromaticAberration + LowPassFreqOptimized + GaussianBlur
        assert_eq!(result.len(), 3);
        assert_eq!(result[0].type_name(), "resin::ChromaticAberration");
        assert_eq!(result[1].type_name(), "resin::LowPassFreqOptimized");
        assert_eq!(result[2].type_name(), "resin::GaussianBlur");
    }
}
