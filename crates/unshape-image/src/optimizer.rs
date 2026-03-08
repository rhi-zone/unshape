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
use crate::channel::Channel;
use crate::effects::{HighPassOptimized, UnsharpMaskOptimized};
use crate::freq::{Fft2d, FreqRadialMul, FreqRingMul, Ifft2d};
use crate::int_ops::{ExtractBitPlane, LsbEmbed, SetBitPlane};
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
                Box::new(BandPassFreqPattern),
                Box::new(GaussianBlurCombinePattern),
                Box::new(SeparableKernelPattern),
                Box::new(ExtractBitPlanePattern),
                Box::new(SetBitPlanePattern),
                Box::new(LsbEmbedPattern),
                Box::new(HighPassPattern),
                Box::new(UnsharpMaskPattern),
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
// Pattern: FFT → RingMul(lo, hi) → IFFT → BandPassFreqOptimized
// ============================================================================

/// Matches `Fft2d → FreqRingMul { lo, hi } → Ifft2d` and replaces
/// with a single [`BandPassFreqOptimized`] op.
pub struct BandPassFreqPattern;

impl ImagePattern for BandPassFreqPattern {
    fn name(&self) -> &'static str {
        "BandPassFreqPattern"
    }

    fn try_match(&self, ops: &[&dyn DynOp]) -> Option<PatternMatch> {
        if ops.len() < 3 {
            return None;
        }
        if op_name(ops[0]) != "resin::Fft2d" {
            return None;
        }
        if op_name(ops[1]) != "resin::FreqRingMul" {
            return None;
        }
        if op_name(ops[2]) != "resin::Ifft2d" {
            return None;
        }

        let params = ops[1].params();
        let lo = params["lo"].as_f64().unwrap_or(0.1) as f32;
        let hi = params["hi"].as_f64().unwrap_or(0.5) as f32;

        Some(PatternMatch {
            consumed: 3,
            replacements: vec![Box::new(BandPassFreqOptimized { lo, hi })],
        })
    }
}

// ============================================================================
// Optimized op: BandPassFreqOptimized
// ============================================================================

/// Fused band-pass frequency filter: FFT + ring mask + IFFT in one operation.
///
/// Equivalent to `Fft2d → FreqRingMul { lo, hi } → Ifft2d`
/// but avoids creating separate intermediate frequency-domain image allocations.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct BandPassFreqOptimized {
    /// Lower bound of the pass-band as a fraction of Nyquist (0.0–1.0).
    pub lo: f32,
    /// Upper bound of the pass-band as a fraction of Nyquist (0.0–1.0).
    pub hi: f32,
}

impl BandPassFreqOptimized {
    /// Creates a band-pass filter passing frequencies between `lo` and `hi`.
    pub fn new(lo: f32, hi: f32) -> Self {
        Self { lo, hi }
    }

    /// Applies band-pass frequency filtering in a single fused pass.
    pub fn apply(&self, image: &ImageField) -> ImageField {
        let (mut real, mut imag) = Fft2d.apply(image);
        FreqRingMul {
            lo: self.lo,
            hi: self.hi,
        }
        .apply_inplace(&mut real, &mut imag);
        Ifft2d.apply(&real, &imag)
    }
}

impl DynOp for BandPassFreqOptimized {
    fn type_name(&self) -> &'static str {
        "resin::BandPassFreqOptimized"
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
        serde_json::json!({ "lo": self.lo, "hi": self.hi })
    }
}

// ============================================================================
// Bit manipulation patterns
// ============================================================================

/// Matches an [`IntColorExpr`](crate::int_ops::IntColorExpr) that evaluates
/// `(channel >> N) & 1` and replaces it with a direct [`ExtractBitPlane`] op.
///
/// Matches `MapPixels` whose expression is the canonical `extract_bit(ch, bit)`
/// form: `BitAnd(Shr(channel, Constant(bit)), Constant(1))`.
///
/// [`IntColorExpr`]: crate::int_ops::IntColorExpr
pub struct ExtractBitPlanePattern;

impl ImagePattern for ExtractBitPlanePattern {
    fn name(&self) -> &'static str {
        "ExtractBitPlanePattern"
    }

    fn try_match(&self, ops: &[&dyn DynOp]) -> Option<PatternMatch> {
        use crate::int_ops::IntColorExpr;

        if ops.is_empty() {
            return None;
        }
        if op_name(ops[0]) != "resin::MapPixels" {
            return None;
        }

        // Recover the IntColorExpr from the params JSON.
        let params = ops[0].params();
        let expr_json = params.get("expr")?;
        let expr: IntColorExpr = serde_json::from_value(expr_json.clone()).ok()?;

        // Match: Vec4 { r: BitAnd(Shr(ch, Constant(N)), Constant(1)), ... }
        let (channel, bit) = match_extract_bit_expr(&expr)?;

        Some(PatternMatch {
            consumed: 1,
            replacements: vec![Box::new(ExtractBitPlane { channel, bit })],
        })
    }
}

/// Returns `(channel, bit)` if `expr` matches the `(channel >> bit) & 1` pattern.
fn match_extract_bit_expr(expr: &crate::int_ops::IntColorExpr) -> Option<(Channel, u8)> {
    use crate::int_ops::IntColorExpr;

    // Top level must be Vec4 with all channels using the same expression.
    let r_expr = match expr {
        IntColorExpr::Vec4 { r, .. } => r.as_ref(),
        _ => return None,
    };

    // r_expr must be BitAnd(Shr(ch, Constant(N)), Constant(1)).
    let (shr_expr, and_const) = match r_expr {
        IntColorExpr::BitAnd(lhs, rhs) => (lhs.as_ref(), rhs.as_ref()),
        _ => return None,
    };

    // and_const must be Constant(1).
    match and_const {
        IntColorExpr::Constant(1) => {}
        _ => return None,
    }

    // shr_expr must be Shr(ch, Constant(N)).
    let (ch_expr, shift_expr) = match shr_expr {
        IntColorExpr::Shr(lhs, rhs) => (lhs.as_ref(), rhs.as_ref()),
        _ => return None,
    };

    let bit = match shift_expr {
        IntColorExpr::Constant(n) => *n,
        _ => return None,
    };

    let channel = match ch_expr {
        IntColorExpr::R => Channel::Red,
        IntColorExpr::G => Channel::Green,
        IntColorExpr::B => Channel::Blue,
        IntColorExpr::A => Channel::Alpha,
        _ => return None,
    };

    Some((channel, bit))
}

/// Matches an [`IntColorExpr`](crate::int_ops::IntColorExpr) that evaluates
/// `(channel & ~(1 << N)) | (src << N)` and replaces it with [`SetBitPlane`].
///
/// [`IntColorExpr`]: crate::int_ops::IntColorExpr
pub struct SetBitPlanePattern;

impl ImagePattern for SetBitPlanePattern {
    fn name(&self) -> &'static str {
        "SetBitPlanePattern"
    }

    fn try_match(&self, ops: &[&dyn DynOp]) -> Option<PatternMatch> {
        use crate::int_ops::IntColorExpr;

        if ops.is_empty() {
            return None;
        }
        if op_name(ops[0]) != "resin::MapPixels" {
            return None;
        }

        let params = ops[0].params();
        let expr_json = params.get("expr")?;
        let expr: IntColorExpr = serde_json::from_value(expr_json.clone()).ok()?;

        let (channel, bit) = match_set_bit_expr(&expr)?;

        Some(PatternMatch {
            consumed: 1,
            replacements: vec![Box::new(SetBitPlane { channel, bit })],
        })
    }
}

/// Returns `(channel, bit)` if `expr` matches the `(ch & ~(1<<N)) | (src<<N)` pattern.
fn match_set_bit_expr(expr: &crate::int_ops::IntColorExpr) -> Option<(Channel, u8)> {
    use crate::int_ops::IntColorExpr;

    // Top level: Vec4 { r: BitOr(BitAnd(ch, ~(1<<N)), Shl(src, N)), ... }
    let r_expr = match expr {
        IntColorExpr::Vec4 { r, .. } => r.as_ref(),
        _ => return None,
    };

    // r_expr: BitOr(clear_expr, set_expr)
    let (clear_expr, set_expr) = match r_expr {
        IntColorExpr::BitOr(lhs, rhs) => (lhs.as_ref(), rhs.as_ref()),
        _ => return None,
    };

    // clear_expr: BitAnd(ch, Constant(inv_mask)) where inv_mask = !(1 << N)
    let (ch_expr, inv_mask_expr) = match clear_expr {
        IntColorExpr::BitAnd(lhs, rhs) => (lhs.as_ref(), rhs.as_ref()),
        _ => return None,
    };

    let inv_mask = match inv_mask_expr {
        IntColorExpr::Constant(c) => *c,
        _ => return None,
    };

    // Determine bit from inv_mask: inv_mask = !(1 << N), so (1 << N) = !inv_mask
    let mask = !inv_mask;
    // mask must be a power of two
    if mask.count_ones() != 1 {
        return None;
    }
    let bit = mask.trailing_zeros() as u8;

    // set_expr: Shl(src, Constant(N))
    match set_expr {
        IntColorExpr::Shl(_, shift) => match shift.as_ref() {
            IntColorExpr::Constant(n) if *n == bit => {}
            _ => return None,
        },
        _ => return None,
    }

    let channel = match ch_expr {
        IntColorExpr::R => Channel::Red,
        IntColorExpr::G => Channel::Green,
        IntColorExpr::B => Channel::Blue,
        IntColorExpr::A => Channel::Alpha,
        _ => return None,
    };

    Some((channel, bit))
}

/// Matches an [`IntColorExpr`](crate::int_ops::IntColorExpr) that evaluates
/// `(channel & 0xFE) | data_bit` and replaces it with [`LsbEmbed`].
///
/// [`IntColorExpr`]: crate::int_ops::IntColorExpr
pub struct LsbEmbedPattern;

impl ImagePattern for LsbEmbedPattern {
    fn name(&self) -> &'static str {
        "LsbEmbedPattern"
    }

    fn try_match(&self, ops: &[&dyn DynOp]) -> Option<PatternMatch> {
        use crate::int_ops::IntColorExpr;

        if ops.is_empty() {
            return None;
        }
        if op_name(ops[0]) != "resin::MapPixels" {
            return None;
        }

        let params = ops[0].params();
        let expr_json = params.get("expr")?;
        let expr: IntColorExpr = serde_json::from_value(expr_json.clone()).ok()?;

        let channel = match_lsb_embed_expr(&expr)?;

        Some(PatternMatch {
            consumed: 1,
            replacements: vec![Box::new(LsbEmbed { channel })],
        })
    }
}

/// Returns `channel` if `expr` matches the `(ch & 0xFE) | data_bit` pattern.
fn match_lsb_embed_expr(expr: &crate::int_ops::IntColorExpr) -> Option<Channel> {
    use crate::int_ops::IntColorExpr;

    // Top level: Vec4 { r: BitOr(BitAnd(ch, Constant(0xFE)), data_bit), ... }
    let r_expr = match expr {
        IntColorExpr::Vec4 { r, .. } => r.as_ref(),
        _ => return None,
    };

    let (clear_expr, _data_expr) = match r_expr {
        IntColorExpr::BitOr(lhs, rhs) => (lhs.as_ref(), rhs.as_ref()),
        _ => return None,
    };

    // clear_expr: BitAnd(ch, Constant(0xFE))
    let (ch_expr, mask_expr) = match clear_expr {
        IntColorExpr::BitAnd(lhs, rhs) => (lhs.as_ref(), rhs.as_ref()),
        _ => return None,
    };

    // mask must be 0xFE (clear LSB)
    match mask_expr {
        IntColorExpr::Constant(0xFE) => {}
        _ => return None,
    }

    let channel = match ch_expr {
        IntColorExpr::R => Channel::Red,
        IntColorExpr::G => Channel::Green,
        IntColorExpr::B => Channel::Blue,
        IntColorExpr::A => Channel::Alpha,
        _ => return None,
    };

    Some(channel)
}

// ============================================================================
// Composite patterns
// ============================================================================

/// Matches a `GaussianBlur` op that was applied to produce a high-pass
/// component (`original - blur(original, σ)`) and replaces it with
/// [`HighPassOptimized`].
///
/// Specifically matches a `GaussianBlur` op tagged with a `high_pass_marker`
/// in its params, or detects the pattern by checking the op type name.
/// In the linear pipeline model this matches any standalone `GaussianBlur`
/// when used as a marker for the high-pass fused replacement.
///
/// **Pattern:** A `GaussianBlur` op immediately preceded by nothing or
/// following a source marker. Replaces with `HighPassOptimized` which fuses
/// `original - blur(original)` into a single pass.
pub struct HighPassPattern;

impl ImagePattern for HighPassPattern {
    fn name(&self) -> &'static str {
        "HighPassPattern"
    }

    fn try_match(&self, ops: &[&dyn DynOp]) -> Option<PatternMatch> {
        if ops.is_empty() {
            return None;
        }

        // Match a GaussianBlur op tagged with high_pass_mode = true.
        if op_name(ops[0]) != "resin::GaussianBlur" {
            return None;
        }

        let params = ops[0].params();
        let high_pass = params["high_pass_mode"].as_bool().unwrap_or(false);
        if !high_pass {
            return None;
        }

        let sigma = params["sigma"].as_f64().unwrap_or(1.0) as f32;

        Some(PatternMatch {
            consumed: 1,
            replacements: vec![Box::new(HighPassOptimized { sigma })],
        })
    }
}

/// Matches a `GaussianBlur` op tagged with `unsharp_mask_mode = true` and an
/// `amount` parameter, replacing it with [`UnsharpMaskOptimized`].
///
/// **Pattern:** `GaussianBlur { sigma, unsharp_mask_mode: true, amount: k }`
/// → `UnsharpMaskOptimized { sigma, amount: k }`
pub struct UnsharpMaskPattern;

impl ImagePattern for UnsharpMaskPattern {
    fn name(&self) -> &'static str {
        "UnsharpMaskPattern"
    }

    fn try_match(&self, ops: &[&dyn DynOp]) -> Option<PatternMatch> {
        if ops.is_empty() {
            return None;
        }

        if op_name(ops[0]) != "resin::GaussianBlur" {
            return None;
        }

        let params = ops[0].params();
        let unsharp = params["unsharp_mask_mode"].as_bool().unwrap_or(false);
        if !unsharp {
            return None;
        }

        let sigma = params["sigma"].as_f64().unwrap_or(1.0) as f32;
        let amount = params["amount"].as_f64().unwrap_or(1.0) as f32;

        Some(PatternMatch {
            consumed: 1,
            replacements: vec![Box::new(UnsharpMaskOptimized { sigma, amount })],
        })
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ImageField;
    use crate::channel::Channel;
    use crate::freq::{Fft2d, FreqRadialMul, FreqRingMul, Ifft2d};
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
    fn test_band_pass_pattern_match() {
        let ops: Vec<Box<dyn DynOp>> = vec![
            Box::new(Fft2d),
            Box::new(FreqRingMul { lo: 0.2, hi: 0.6 }),
            Box::new(Ifft2d),
        ];

        let optimizer = ImageOptimizer::new();
        let result = optimizer.optimize(ops);

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].type_name(), "resin::BandPassFreqOptimized");

        let params = result[0].params();
        let lo = params["lo"].as_f64().unwrap() as f32;
        let hi = params["hi"].as_f64().unwrap() as f32;
        assert!((lo - 0.2).abs() < 1e-5, "expected lo=0.2, got {lo}");
        assert!((hi - 0.6).abs() < 1e-5, "expected hi=0.6, got {hi}");
    }

    #[test]
    fn test_band_pass_optimized_same_result_as_manual() {
        let image = test_image();

        // Manual FFT → ring mask → IFFT.
        let (mut real, mut imag) = Fft2d.apply(&image);
        FreqRingMul { lo: 0.2, hi: 0.6 }.apply_inplace(&mut real, &mut imag);
        let manual_result = Ifft2d.apply(&real, &imag);

        // Fused.
        let fused_result = BandPassFreqOptimized { lo: 0.2, hi: 0.6 }.apply(&image);

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
    fn test_extract_bit_plane_pattern_helper() {
        use crate::int_ops::IntColorExpr;

        // Build the canonical extract_bit expression for red channel, bit 3.
        let expr = IntColorExpr::Vec4 {
            r: Box::new(IntColorExpr::BitAnd(
                Box::new(IntColorExpr::Shr(
                    Box::new(IntColorExpr::R),
                    Box::new(IntColorExpr::Constant(3)),
                )),
                Box::new(IntColorExpr::Constant(1)),
            )),
            g: Box::new(IntColorExpr::Constant(0)),
            b: Box::new(IntColorExpr::Constant(0)),
            a: Box::new(IntColorExpr::Constant(0xFF)),
        };

        let channel_bit = match_extract_bit_expr(&expr);
        assert!(channel_bit.is_some());
        let (channel, bit) = channel_bit.unwrap();
        assert_eq!(channel, Channel::Red);
        assert_eq!(bit, 3);
    }

    #[test]
    fn test_lsb_embed_pattern_helper() {
        use crate::int_ops::IntColorExpr;

        // (R & 0xFE) | data_bit
        let expr = IntColorExpr::Vec4 {
            r: Box::new(IntColorExpr::BitOr(
                Box::new(IntColorExpr::BitAnd(
                    Box::new(IntColorExpr::R),
                    Box::new(IntColorExpr::Constant(0xFE)),
                )),
                Box::new(IntColorExpr::Constant(0)), // data_bit placeholder
            )),
            g: Box::new(IntColorExpr::G),
            b: Box::new(IntColorExpr::B),
            a: Box::new(IntColorExpr::A),
        };

        let channel = match_lsb_embed_expr(&expr);
        assert!(channel.is_some());
        assert_eq!(channel.unwrap(), Channel::Red);
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
