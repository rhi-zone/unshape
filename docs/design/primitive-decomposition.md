# Primitive Decomposition

Identifying irreducible operations vs. compositions across all domains.

## Three-Layer Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Layer 3: Pattern-Matching Optimizer                    │
│  - Recognizes compositions, applies algebraic rules     │
│  - Constant folding, dead code elimination              │
│  - Fuses operations for performance                     │
└─────────────────────────────────────────────────────────┘
                          ▲
                          │ input graphs
┌─────────────────────────────────────────────────────────┐
│  Layer 2: Ergonomic Helpers                             │
│  - Convenience functions that compose primitives        │
│  - blur(), grayscale(), Gravity, QuadIn                 │
│  - User-facing, domain-specific vocabulary              │
└─────────────────────────────────────────────────────────┘
                          │ expands to
                          ▼
┌─────────────────────────────────────────────────────────┐
│  Layer 1: True Primitives                               │
│  - Irreducible operations that cannot decompose further │
│  - Convolve, MapPixels, Integrate, RgbToHsl             │
│  - Minimal orthogonal basis for each domain             │
└─────────────────────────────────────────────────────────┘
```

## What Makes a Primitive?

A true primitive is an operation that:

1. **Cannot be expressed** as a composition of other operations in the domain
2. **Is not a trivial wrapper** around another operation with default parameters
3. **Adds unique capability** - removing it removes expressiveness

### Examples

**Primitive:**
```rust
// Convolve is primitive - it's the fundamental spatial operation
pub struct Convolve { pub kernel: Kernel }
```

**Not primitive (composition):**
```rust
// Blur is Convolve with Gaussian kernel, applied multiple times
pub fn blur(radius: f32) -> impl Op {
    let kernel = Kernel::gaussian(radius);
    Convolve { kernel }.repeat(passes_for_radius(radius))
}
```

**Not primitive (trivial wrapper):**
```rust
// Upsample is just Resize with 2x dimensions - deprecated
pub fn upsample(img: &Image) -> Image {
    Resize::new(img.width() * 2, img.height() * 2).apply(img)
}
```

## Decomposition Patterns by Domain

### Image Processing

| Layer | Operations |
|-------|------------|
| Primitives | Convolve, MapPixels, RemapUv, Composite, Resize, FFT, IFFT |
| Helpers | blur, sharpen, grayscale, invert, glow, bloom, vignette |

```
Blur = Convolve { Gaussian } repeated
Sharpen = Convolve { sharpen_kernel }
Grayscale = MapPixels { luminance }
Glow = threshold + blur + composite
```

### Particle Systems

| Layer | Operations |
|-------|------------|
| Primitives | PositionProvider, VelocityProvider, Acceleration, Damping, NoisePerturbation |
| Helpers | PointEmitter, SphereEmitter, Gravity, Wind, Turbulence |

```
PointEmitter = FixedPosition + ConeVelocity + LifetimeRange
Gravity = Acceleration { (0, -9.81, 0) }
Turbulence = NoisePerturbation { curl_noise }
```

### Physics Constraints

| Layer | Operations |
|-------|------------|
| Primitives | PointConstraint, AxisAlignment, spring force, damping |
| Helpers | DistanceConstraint, HingeConstraint, BallJoint |

All constraints follow the same solver pattern:
1. `compute_error()` - position/angular deviation
2. `apply_correction()` - adjust bodies based on inverse mass

### Easing Functions

| Layer | Operations |
|-------|------------|
| Primitives | (none - all are dew expressions) |
| Helpers | quad_in, cubic_out, elastic_in_out, smoothstep |

```
quad_in(t) = t * t           // Mul(T, T)
cubic_in(t) = t * t * t      // Mul(Mul(T, T), T)
smoothstep(t) = t² * (3 - 2t)
```

Easing functions are not primitives - they're expression builders that return AST nodes the optimizer can fold.

### IK Solvers

| Layer | Operations |
|-------|------------|
| Primitives | SolveCcd, SolveFabrik |
| Helpers | (none - these are already primitive) |

IK algorithms are irreducible - CCD and FABRIK are different approaches to the same problem, not compositions.

## Benefits of Decomposition

### 1. Smaller API Surface

Instead of 50 blur variants, expose one `Convolve` and let users compose.

### 2. Better Optimization

The optimizer works with primitives:
```
MapPixels(grayscale) + MapPixels(invert)
  → MapPixels(grayscale ∘ invert)  // Fused
```

### 3. Serialization Consistency

Primitives are the canonical form in saved graphs. Helpers expand at construction time.

### 4. Testing Coverage

Testing N primitives covers all their compositions. N helpers would require N² interaction tests.

## Implementing Decomposition

### Step 1: Identify Primitives

Look for operations that:
- Introduce unique state transformations
- Cannot be expressed via existing operations
- Are required for domain completeness

### Step 2: Add Op Structs

Every primitive needs an op struct:
```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "dynop", derive(Op))]
#[cfg_attr(feature = "dynop", op(input = Image, output = Image))]
pub struct Convolve {
    pub kernel: Kernel,
}
```

### Step 3: Helpers Delegate

Helpers construct and apply primitives:
```rust
pub fn blur(image: &Image, radius: f32) -> Image {
    let kernel = Kernel::gaussian(radius);
    Convolve { kernel }.apply(image)
}
```

### Step 4: Document the Mapping

Add to `DECOMPOSITION-AUDIT.md`:
```markdown
| Helper | Decomposes To |
|--------|---------------|
| blur | `Convolve { Gaussian }` repeated |
```

## Trait-Based Unification

Where primitives share patterns, use traits:

```rust
// All constraints follow same pattern
pub trait ConstraintSolver {
    fn compute_error(&self, bodies: &[RigidBody]) -> Option<ConstraintError>;
    fn apply_correction(&self, bodies: &mut [RigidBody], stiffness: f32);
}

// All emitters compose from providers
pub trait PositionProvider { fn sample(&self, rng: &mut Rng) -> Vec3; }
pub trait VelocityProvider { fn sample(&self, pos: Vec3, rng: &mut Rng) -> Vec3; }
```

This makes the primitive pattern explicit and enables custom implementations.

## See Also

- [ops-as-values.md](./ops-as-values.md) - Op struct patterns
- [DECOMPOSITION-AUDIT.md](../../DECOMPOSITION-AUDIT.md) - Full audit by crate
- [expression-language.md](./expression-language.md) - dew expression system
