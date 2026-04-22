use glam::Vec3;

use crate::{EvalContext, Field};

// ============================================================================
// FresnelField
// ============================================================================

/// Computes Schlick's Fresnel approximation from a normal and view direction.
///
/// The output is a scalar in [0, 1] where 0 is head-on (normal parallel to view)
/// and 1 is grazing (normal perpendicular to view). Useful for edge highlights,
/// glass, and physically-based material blending.
///
/// Uses Schlick's approximation:
/// `F(θ) = f0 + (1 - f0) * (1 - dot(N,V))^5`
/// where `f0 = ((1 - ior) / (1 + ior))²`.
///
/// # Example
///
/// ```
/// use unshape_field::{Constant, EvalContext, Field, FresnelField};
/// use glam::{Vec2, Vec3};
///
/// let fresnel = FresnelField {
///     normal_field: Constant::new(Vec3::Z),
///     view_field: Constant::new(Vec3::Z),
///     ior: 1.45,
/// };
/// let ctx = EvalContext::new();
/// // Normal == view → near-zero Fresnel (head-on)
/// let value = fresnel.sample(Vec2::ZERO, &ctx);
/// assert!(value < 0.1);
/// ```
pub struct FresnelField<N, V> {
    /// Field providing the surface normal (should return unit vectors).
    pub normal_field: N,
    /// Field providing the view direction (should return unit vectors).
    pub view_field: V,
    /// Index of refraction. Default 1.45 (glass). Must be positive.
    pub ior: f32,
}

impl<N, V> FresnelField<N, V> {
    /// Creates a new `FresnelField` with the given normal/view fields and IOR.
    pub fn new(normal_field: N, view_field: V, ior: f32) -> Self {
        Self {
            normal_field,
            view_field,
            ior,
        }
    }

    /// Creates a `FresnelField` with the default IOR of 1.45 (glass).
    pub fn glass(normal_field: N, view_field: V) -> Self {
        Self::new(normal_field, view_field, 1.45)
    }
}

impl<I: Clone, N, V> Field<I, f32> for FresnelField<N, V>
where
    N: Field<I, Vec3>,
    V: Field<I, Vec3>,
{
    fn sample(&self, input: I, ctx: &EvalContext) -> f32 {
        let normal = self.normal_field.sample(input.clone(), ctx);
        let view = self.view_field.sample(input, ctx);

        let cos_theta = normal.dot(view).abs().clamp(0.0, 1.0);
        let f0 = {
            let r = (1.0 - self.ior) / (1.0 + self.ior);
            r * r
        };
        f0 + (1.0 - f0) * (1.0 - cos_theta).powi(5)
    }
}

// ============================================================================
// LayerWeight
// ============================================================================

/// Output mode for [`LayerWeight`].
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum LayerWeightMode {
    /// Fresnel-based falloff with `blend`-controlled IOR bias.
    ///
    /// Equivalent to Blender's "Layer Weight > Fresnel" socket.
    Fresnel,
    /// Facing ratio: how directly the surface faces the camera.
    ///
    /// `dot(normal, view)` remapped by `blend`. Fully facing → 0.0, edge-on → 1.0.
    Facing,
}

/// Blender's Layer Weight node: smooth edge detection based on view-dependent angle.
///
/// Produces a scalar in [0, 1]:
/// - [`LayerWeightMode::Fresnel`]: physical Schlick falloff, `blend` controls IOR
/// - [`LayerWeightMode::Facing`]: dot-product facing ratio, `blend` biases the result
///
/// # Example
///
/// ```
/// use unshape_field::{Constant, EvalContext, Field, LayerWeight, LayerWeightMode};
/// use glam::{Vec2, Vec3};
///
/// let lw = LayerWeight {
///     normal_field: Constant::new(Vec3::Z),
///     view_field: Constant::new(Vec3::Z),
///     blend: 0.0,
///     mode: LayerWeightMode::Facing,
/// };
/// let ctx = EvalContext::new();
/// // Normal == view (fully facing) → close to 0
/// let value = lw.sample(Vec2::ZERO, &ctx);
/// assert!(value < 0.1);
/// ```
pub struct LayerWeight<N, V> {
    /// Field providing surface normals (should return unit vectors).
    pub normal_field: N,
    /// Field providing view directions (should return unit vectors).
    pub view_field: V,
    /// Blend parameter in [0, 1]. Controls IOR in Fresnel mode and bias in Facing mode.
    pub blend: f32,
    /// Which output to compute.
    pub mode: LayerWeightMode,
}

impl<N, V> LayerWeight<N, V> {
    /// Creates a new `LayerWeight`.
    pub fn new(normal_field: N, view_field: V, blend: f32, mode: LayerWeightMode) -> Self {
        Self {
            normal_field,
            view_field,
            blend,
            mode,
        }
    }

    /// Creates a Fresnel-mode `LayerWeight`.
    pub fn fresnel(normal_field: N, view_field: V, blend: f32) -> Self {
        Self::new(normal_field, view_field, blend, LayerWeightMode::Fresnel)
    }

    /// Creates a Facing-mode `LayerWeight`.
    pub fn facing(normal_field: N, view_field: V, blend: f32) -> Self {
        Self::new(normal_field, view_field, blend, LayerWeightMode::Facing)
    }
}

impl<I: Clone, N, V> Field<I, f32> for LayerWeight<N, V>
where
    N: Field<I, Vec3>,
    V: Field<I, Vec3>,
{
    fn sample(&self, input: I, ctx: &EvalContext) -> f32 {
        let normal = self.normal_field.sample(input.clone(), ctx);
        let view = self.view_field.sample(input, ctx);

        let cos_theta = normal.dot(view).abs().clamp(0.0, 1.0);

        match self.mode {
            LayerWeightMode::Fresnel => {
                // Blend maps to IOR: blend=0 → ior=1 (no Fresnel), blend=1 → ior=∞.
                // Use Blender's mapping: ior = 1 / max(1 - blend, 1e-5).
                let ior = 1.0 / (1.0 - self.blend.clamp(0.0, 1.0 - 1e-5));
                let f0 = {
                    let r = (1.0 - ior) / (1.0 + ior);
                    r * r
                };
                f0 + (1.0 - f0) * (1.0 - cos_theta).powi(5)
            }
            LayerWeightMode::Facing => {
                // facing = 1 - dot(N, V) remapped by blend.
                // blend=0 → pure dot; blend=1 → always edge.
                let facing = 1.0 - cos_theta;
                let bias = self.blend.clamp(0.0, 1.0);
                // Remap: apply blend as a power curve so that blend=0 → linear, blend→1 → step.
                // Simple Blender-style: result = facing * (1 - bias) + bias * facing^0.5 (approx).
                // We use the clean formulation: clamp(facing + bias * (1 - facing), 0, 1).
                (facing + bias * (1.0 - facing)).clamp(0.0, 1.0)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use glam::{Vec2, Vec3};

    use crate::{Constant, EvalContext, Field, FresnelField, LayerWeight, LayerWeightMode};

    #[test]
    fn fresnel_head_on_is_near_zero() {
        // Normal == view direction → minimum Fresnel (only f0 contribution)
        let fresnel = FresnelField {
            normal_field: Constant::new(Vec3::Z),
            view_field: Constant::new(Vec3::Z),
            ior: 1.45,
        };
        let ctx = EvalContext::new();
        let v = fresnel.sample(Vec2::ZERO, &ctx);
        // f0 for ior=1.45 is ((1-1.45)/(1+1.45))^2 ≈ 0.0338
        assert!(v < 0.05, "expected near f0 ≈ 0.034, got {v}");
    }

    #[test]
    fn fresnel_grazing_is_near_one() {
        // Normal ⊥ view direction → cos_theta ≈ 0 → Fresnel ≈ 1
        let fresnel = FresnelField {
            normal_field: Constant::new(Vec3::Z),
            view_field: Constant::new(Vec3::X),
            ior: 1.45,
        };
        let ctx = EvalContext::new();
        let v = fresnel.sample(Vec2::ZERO, &ctx);
        assert!(v > 0.99, "expected near 1.0, got {v}");
    }

    #[test]
    fn layer_weight_facing_head_on_is_near_zero() {
        // dot(N, V) = 1 → facing = 0
        let lw = LayerWeight {
            normal_field: Constant::new(Vec3::Z),
            view_field: Constant::new(Vec3::Z),
            blend: 0.0,
            mode: LayerWeightMode::Facing,
        };
        let ctx = EvalContext::new();
        let v = lw.sample(Vec2::ZERO, &ctx);
        assert!(v < 0.05, "expected ~0.0, got {v}");
    }

    #[test]
    fn layer_weight_facing_edge_on_is_near_one() {
        // dot(N, V) = 0 → facing = 1
        let lw = LayerWeight {
            normal_field: Constant::new(Vec3::Z),
            view_field: Constant::new(Vec3::X),
            blend: 0.0,
            mode: LayerWeightMode::Facing,
        };
        let ctx = EvalContext::new();
        let v = lw.sample(Vec2::ZERO, &ctx);
        assert!(v > 0.95, "expected ~1.0, got {v}");
    }

    #[test]
    fn layer_weight_fresnel_mode_matches_fresnel_field() {
        // With blend=0 → ior=1/(1-0)=1 → f0=0, result = (1-cos)^5
        // Normal ⊥ view → result ≈ 1
        let lw = LayerWeight {
            normal_field: Constant::new(Vec3::Z),
            view_field: Constant::new(Vec3::X),
            blend: 0.0,
            mode: LayerWeightMode::Fresnel,
        };
        let ctx = EvalContext::new();
        let v = lw.sample(Vec2::ZERO, &ctx);
        assert!(v > 0.99, "expected ~1.0 at grazing, got {v}");
    }
}
