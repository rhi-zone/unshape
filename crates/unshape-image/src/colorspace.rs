#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::ImageField;

/// Colorspace for decomposition/reconstruction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum Colorspace {
    /// RGB (Red, Green, Blue) - the native colorspace.
    Rgb,
    /// HSL (Hue, Saturation, Lightness).
    Hsl,
    /// HSV (Hue, Saturation, Value).
    Hsv,
    /// HWB (Hue, Whiteness, Blackness) - CSS Color Level 4.
    Hwb,
    /// YCbCr (Luma, Blue-difference, Red-difference chroma).
    YCbCr,
    /// LAB (CIE L*a*b* perceptual colorspace).
    Lab,
    /// LCH (Lightness, Chroma, Hue) - cylindrical LAB.
    Lch,
    /// OkLab (perceptually uniform colorspace).
    OkLab,
    /// OkLCH (cylindrical OkLab) - CSS Color Level 4.
    OkLch,
}

/// Decomposed colorspace channels.
///
/// Contains three channels representing the colorspace components.
/// Channel values are normalized to [0, 1] for storage.
pub struct ColorspaceChannels {
    /// First channel (H/Y/L depending on colorspace).
    pub c0: ImageField,
    /// Second channel (S/Cb/a depending on colorspace).
    pub c1: ImageField,
    /// Third channel (L/V/Cr/b depending on colorspace).
    pub c2: ImageField,
    /// Alpha channel (preserved from original).
    pub alpha: ImageField,
    /// Which colorspace these channels represent.
    pub colorspace: Colorspace,
}

/// Decomposes an RGB image into the specified colorspace.
///
/// Returns separate grayscale images for each channel, normalized to [0, 1].
///
/// # Example
///
/// ```
/// use unshape_image::{ImageField, Colorspace, decompose_colorspace, reconstruct_colorspace};
///
/// let image = ImageField::solid_sized(64, 64, [0.8, 0.4, 0.2, 1.0]);
///
/// // Decompose to HSL
/// let channels = decompose_colorspace(&image, Colorspace::Hsl);
///
/// // Modify saturation channel...
///
/// // Reconstruct back to RGB
/// let result = reconstruct_colorspace(&channels);
/// ```
pub fn decompose_colorspace(image: &ImageField, colorspace: Colorspace) -> ColorspaceChannels {
    let (width, height) = image.dimensions();
    let size = (width * height) as usize;

    let mut c0_data = Vec::with_capacity(size);
    let mut c1_data = Vec::with_capacity(size);
    let mut c2_data = Vec::with_capacity(size);
    let mut alpha_data = Vec::with_capacity(size);

    for i in 0..size {
        let pixel = image.data[i];
        let r = pixel[0];
        let g = pixel[1];
        let b = pixel[2];
        let a = pixel[3];

        let (v0, v1, v2) = match colorspace {
            Colorspace::Rgb => (r, g, b),
            Colorspace::Hsl => rgb_to_hsl(r, g, b),
            Colorspace::Hsv => rgb_to_hsv(r, g, b),
            Colorspace::Hwb => rgb_to_hwb(r, g, b),
            Colorspace::YCbCr => rgb_to_ycbcr(r, g, b),
            Colorspace::Lab => rgb_to_lab(r, g, b),
            Colorspace::Lch => rgb_to_lch(r, g, b),
            Colorspace::OkLab => rgb_to_oklab(r, g, b),
            Colorspace::OkLch => rgb_to_oklch(r, g, b),
        };

        c0_data.push([v0, v0, v0, 1.0]);
        c1_data.push([v1, v1, v1, 1.0]);
        c2_data.push([v2, v2, v2, 1.0]);
        alpha_data.push([a, a, a, 1.0]);
    }

    ColorspaceChannels {
        c0: ImageField::from_raw(c0_data, width, height),
        c1: ImageField::from_raw(c1_data, width, height),
        c2: ImageField::from_raw(c2_data, width, height),
        alpha: ImageField::from_raw(alpha_data, width, height),
        colorspace,
    }
}

/// Reconstructs an RGB image from colorspace channels.
pub fn reconstruct_colorspace(channels: &ColorspaceChannels) -> ImageField {
    let (width, height) = channels.c0.dimensions();
    let size = (width * height) as usize;

    let mut data = Vec::with_capacity(size);

    for i in 0..size {
        let v0 = channels.c0.data[i][0];
        let v1 = channels.c1.data[i][0];
        let v2 = channels.c2.data[i][0];
        let a = channels.alpha.data[i][0];

        let (r, g, b) = match channels.colorspace {
            Colorspace::Rgb => (v0, v1, v2),
            Colorspace::Hsl => hsl_to_rgb(v0, v1, v2),
            Colorspace::Hsv => hsv_to_rgb(v0, v1, v2),
            Colorspace::Hwb => hwb_to_rgb(v0, v1, v2),
            Colorspace::YCbCr => ycbcr_to_rgb(v0, v1, v2),
            Colorspace::Lab => lab_to_rgb(v0, v1, v2),
            Colorspace::Lch => lch_to_rgb(v0, v1, v2),
            Colorspace::OkLab => oklab_to_rgb(v0, v1, v2),
            Colorspace::OkLch => oklch_to_rgb(v0, v1, v2),
        };

        data.push([r, g, b, a]);
    }

    ImageField::from_raw(data, width, height)
}

// --- Colorspace conversion helpers ---

pub(crate) fn rgb_to_hsl(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let l = (max + min) / 2.0;

    if (max - min).abs() < 1e-6 {
        return (0.0, 0.0, l);
    }

    let d = max - min;
    let s = if l > 0.5 {
        d / (2.0 - max - min)
    } else {
        d / (max + min)
    };

    let h = if (max - r).abs() < 1e-6 {
        ((g - b) / d + if g < b { 6.0 } else { 0.0 }) / 6.0
    } else if (max - g).abs() < 1e-6 {
        ((b - r) / d + 2.0) / 6.0
    } else {
        ((r - g) / d + 4.0) / 6.0
    };

    (h, s, l)
}

pub(crate) fn hsl_to_rgb(h: f32, s: f32, l: f32) -> (f32, f32, f32) {
    if s.abs() < 1e-6 {
        return (l, l, l);
    }

    let q = if l < 0.5 {
        l * (1.0 + s)
    } else {
        l + s - l * s
    };
    let p = 2.0 * l - q;

    let hue_to_rgb = |t: f32| -> f32 {
        let t = t.rem_euclid(1.0);
        if t < 1.0 / 6.0 {
            p + (q - p) * 6.0 * t
        } else if t < 0.5 {
            q
        } else if t < 2.0 / 3.0 {
            p + (q - p) * (2.0 / 3.0 - t) * 6.0
        } else {
            p
        }
    };

    (
        hue_to_rgb(h + 1.0 / 3.0),
        hue_to_rgb(h),
        hue_to_rgb(h - 1.0 / 3.0),
    )
}

pub(crate) fn rgb_to_hsv(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let v = max;

    if (max - min).abs() < 1e-6 {
        return (0.0, 0.0, v);
    }

    let d = max - min;
    let s = d / max;

    let h = if (max - r).abs() < 1e-6 {
        ((g - b) / d + if g < b { 6.0 } else { 0.0 }) / 6.0
    } else if (max - g).abs() < 1e-6 {
        ((b - r) / d + 2.0) / 6.0
    } else {
        ((r - g) / d + 4.0) / 6.0
    };

    (h, s, v)
}

pub(crate) fn hsv_to_rgb(h: f32, s: f32, v: f32) -> (f32, f32, f32) {
    if s.abs() < 1e-6 {
        return (v, v, v);
    }

    let h = h * 6.0;
    let i = h.floor() as i32;
    let f = h - h.floor();
    let p = v * (1.0 - s);
    let q = v * (1.0 - s * f);
    let t = v * (1.0 - s * (1.0 - f));

    match i % 6 {
        0 => (v, t, p),
        1 => (q, v, p),
        2 => (p, v, t),
        3 => (p, q, v),
        4 => (t, p, v),
        _ => (v, p, q),
    }
}

pub(crate) fn rgb_to_ycbcr(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    let y = 0.299 * r + 0.587 * g + 0.114 * b;
    let cb = (-0.169 * r - 0.331 * g + 0.500 * b) + 0.5;
    let cr = (0.500 * r - 0.419 * g - 0.081 * b) + 0.5;
    (y, cb, cr)
}

pub(crate) fn ycbcr_to_rgb(y: f32, cb: f32, cr: f32) -> (f32, f32, f32) {
    let cb = cb - 0.5;
    let cr = cr - 0.5;
    let r = (y + 1.402 * cr).clamp(0.0, 1.0);
    let g = (y - 0.344 * cb - 0.714 * cr).clamp(0.0, 1.0);
    let b = (y + 1.772 * cb).clamp(0.0, 1.0);
    (r, g, b)
}

pub(crate) fn rgb_to_lab(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    // sRGB to linear RGB
    let to_linear = |v: f32| -> f32 {
        if v <= 0.04045 {
            v / 12.92
        } else {
            ((v + 0.055) / 1.055).powf(2.4)
        }
    };

    let r = to_linear(r);
    let g = to_linear(g);
    let b = to_linear(b);

    // Linear RGB to XYZ (D65 illuminant)
    let x = r * 0.4124564 + g * 0.3575761 + b * 0.1804375;
    let y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750;
    let z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041;

    // XYZ to Lab (D65 reference white)
    let xn = 0.95047;
    let yn = 1.0;
    let zn = 1.08883;

    let f = |t: f32| -> f32 {
        if t > 0.008856 {
            t.powf(1.0 / 3.0)
        } else {
            7.787 * t + 16.0 / 116.0
        }
    };

    let fx = f(x / xn);
    let fy = f(y / yn);
    let fz = f(z / zn);

    let l = (116.0 * fy - 16.0) / 100.0; // Normalize L* to [0, 1]
    let a = ((500.0 * (fx - fy)) + 128.0) / 255.0; // Normalize a* to [0, 1]
    let lab_b = ((200.0 * (fy - fz)) + 128.0) / 255.0; // Normalize b* to [0, 1]

    (l.clamp(0.0, 1.0), a.clamp(0.0, 1.0), lab_b.clamp(0.0, 1.0))
}

pub(crate) fn lab_to_rgb(l: f32, a: f32, lab_b: f32) -> (f32, f32, f32) {
    // Denormalize
    let l = l * 100.0;
    let a = a * 255.0 - 128.0;
    let b = lab_b * 255.0 - 128.0;

    // Lab to XYZ
    let xn = 0.95047;
    let yn = 1.0;
    let zn = 1.08883;

    let fy = (l + 16.0) / 116.0;
    let fx = a / 500.0 + fy;
    let fz = fy - b / 200.0;

    let f_inv = |t: f32| -> f32 {
        if t > 0.206893 {
            t * t * t
        } else {
            (t - 16.0 / 116.0) / 7.787
        }
    };

    let x = xn * f_inv(fx);
    let y = yn * f_inv(fy);
    let z = zn * f_inv(fz);

    // XYZ to linear RGB
    let r = x * 3.2404542 - y * 1.5371385 - z * 0.4985314;
    let g = -x * 0.9692660 + y * 1.8760108 + z * 0.0415560;
    let b = x * 0.0556434 - y * 0.2040259 + z * 1.0572252;

    // Linear RGB to sRGB
    let from_linear = |v: f32| -> f32 {
        if v <= 0.0031308 {
            v * 12.92
        } else {
            1.055 * v.powf(1.0 / 2.4) - 0.055
        }
    };

    (
        from_linear(r).clamp(0.0, 1.0),
        from_linear(g).clamp(0.0, 1.0),
        from_linear(b).clamp(0.0, 1.0),
    )
}

pub(crate) fn rgb_to_hwb(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    let (h, _, _) = rgb_to_hsl(r, g, b);
    let w = r.min(g).min(b);
    let b_val = 1.0 - r.max(g).max(b);
    (h, w, b_val)
}

pub(crate) fn hwb_to_rgb(h: f32, w: f32, b: f32) -> (f32, f32, f32) {
    // If w + b >= 1, result is gray
    if w + b >= 1.0 {
        let gray = w / (w + b);
        return (gray, gray, gray);
    }

    // Convert via HSV: HWB(h, w, b) = HSV(h, 1 - w/(1-b), 1-b)
    let v = 1.0 - b;
    let s = 1.0 - w / v;
    hsv_to_rgb(h, s, v)
}

pub(crate) fn rgb_to_lch(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    let (l, a, b_val) = rgb_to_lab(r, g, b);
    // Denormalize a and b from [0,1] to [-128, 127] range for math
    let a_real = a * 255.0 - 128.0;
    let b_real = b_val * 255.0 - 128.0;

    let c = (a_real * a_real + b_real * b_real).sqrt();
    let h = b_real.atan2(a_real);
    // Normalize: L is already [0,1], C to [0,1] (max ~181), H to [0,1]
    let c_norm = (c / 181.0).clamp(0.0, 1.0);
    let h_norm = (h / std::f32::consts::TAU).rem_euclid(1.0);
    (l, c_norm, h_norm)
}

pub(crate) fn lch_to_rgb(l: f32, c: f32, h: f32) -> (f32, f32, f32) {
    // Denormalize
    let c_real = c * 181.0;
    let h_real = h * std::f32::consts::TAU;

    let a_real = c_real * h_real.cos();
    let b_real = c_real * h_real.sin();

    // Normalize back to [0,1] for lab_to_rgb
    let a = (a_real + 128.0) / 255.0;
    let b_val = (b_real + 128.0) / 255.0;

    lab_to_rgb(l, a, b_val)
}

pub(crate) fn rgb_to_oklab(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    // sRGB to linear
    let to_linear = |v: f32| -> f32 {
        if v <= 0.04045 {
            v / 12.92
        } else {
            ((v + 0.055) / 1.055).powf(2.4)
        }
    };

    let r = to_linear(r);
    let g = to_linear(g);
    let b = to_linear(b);

    // Linear RGB to OkLab via LMS
    let l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b;
    let m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b;
    let s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b;

    let l_ = l.cbrt();
    let m_ = m.cbrt();
    let s_ = s.cbrt();

    let lab_l = 0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_;
    let lab_a = 1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_;
    let lab_b = 0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_;

    // Normalize: L is [0,1], a/b are roughly [-0.4, 0.4], normalize to [0,1]
    (
        lab_l.clamp(0.0, 1.0),
        ((lab_a + 0.4) / 0.8).clamp(0.0, 1.0),
        ((lab_b + 0.4) / 0.8).clamp(0.0, 1.0),
    )
}

pub(crate) fn oklab_to_rgb(l: f32, a: f32, b_val: f32) -> (f32, f32, f32) {
    // Denormalize a and b
    let lab_a = a * 0.8 - 0.4;
    let lab_b = b_val * 0.8 - 0.4;

    let l_ = l + 0.3963377774 * lab_a + 0.2158037573 * lab_b;
    let m_ = l - 0.1055613458 * lab_a - 0.0638541728 * lab_b;
    let s_ = l - 0.0894841775 * lab_a - 1.2914855480 * lab_b;

    let l_cubed = l_ * l_ * l_;
    let m_cubed = m_ * m_ * m_;
    let s_cubed = s_ * s_ * s_;

    let r = 4.0767416621 * l_cubed - 3.3077115913 * m_cubed + 0.2309699292 * s_cubed;
    let g = -1.2684380046 * l_cubed + 2.6097574011 * m_cubed - 0.3413193965 * s_cubed;
    let b = -0.0041960863 * l_cubed - 0.7034186147 * m_cubed + 1.7076147010 * s_cubed;

    // Linear to sRGB
    let from_linear = |v: f32| -> f32 {
        if v <= 0.0031308 {
            v * 12.92
        } else {
            1.055 * v.powf(1.0 / 2.4) - 0.055
        }
    };

    (
        from_linear(r).clamp(0.0, 1.0),
        from_linear(g).clamp(0.0, 1.0),
        from_linear(b).clamp(0.0, 1.0),
    )
}

pub(crate) fn rgb_to_oklch(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    let (l, a, b_val) = rgb_to_oklab(r, g, b);
    // Denormalize a and b
    let a_real = a * 0.8 - 0.4;
    let b_real = b_val * 0.8 - 0.4;

    let c = (a_real * a_real + b_real * b_real).sqrt();
    let h = b_real.atan2(a_real);
    // Normalize: C max is about 0.4, H to [0,1]
    let c_norm = (c / 0.4).clamp(0.0, 1.0);
    let h_norm = (h / std::f32::consts::TAU).rem_euclid(1.0);
    (l, c_norm, h_norm)
}

pub(crate) fn oklch_to_rgb(l: f32, c: f32, h: f32) -> (f32, f32, f32) {
    let c_real = c * 0.4;
    let h_real = h * std::f32::consts::TAU;

    let a_real = c_real * h_real.cos();
    let b_real = c_real * h_real.sin();

    // Normalize back to [0,1] for oklab_to_rgb
    let a = (a_real + 0.4) / 0.8;
    let b_val = (b_real + 0.4) / 0.8;

    oklab_to_rgb(l, a, b_val)
}
