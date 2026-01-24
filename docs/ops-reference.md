# Ops Reference

Auto-generated list of all ops-as-values structs in unshape.

Regenerate with: `cargo run -p extract-ops -- --md > docs/ops-reference.md`

## Contents

**155 ops across 20 crates**

- [unshape-audio](#unshape_audio): 11 ops
- [unshape-automata](#unshape_automata): 5 ops
- [unshape-crossdomain](#unshape_crossdomain): 2 ops
- [unshape-field](#unshape_field): 4 ops
- [unshape-fluid](#unshape_fluid): 4 ops
- [unshape-gpu](#unshape_gpu): 1 ops
- [unshape-image](#unshape_image): 35 ops
- [unshape-lsystem](#unshape_lsystem): 2 ops
- [unshape-mesh](#unshape_mesh): 40 ops
- [unshape-particle](#unshape_particle): 10 ops
- [unshape-physics](#unshape_physics): 3 ops
- [unshape-pointcloud](#unshape_pointcloud): 6 ops
- [unshape-procgen](#unshape_procgen): 4 ops
- [unshape-rd](#unshape_rd): 9 ops
- [unshape-rig](#unshape_rig): 6 ops
- [unshape-scatter](#unshape_scatter): 2 ops
- [unshape-space-colonization](#unshape_space_colonization): 1 ops
- [unshape-spring](#unshape_spring): 1 ops
- [unshape-vector](#unshape_vector): 3 ops
- [unshape-voxel](#unshape_voxel): 6 ops

---

## unshape-audio

### `BarSynth`

Configuration for bar synthesis.

`apply(&PercussionInput) -> Vec<f32>`

| Field | Type | Description |
|-------|------|-------------|
| `fundamental` | `f32` | Fundamental frequency in Hz. |
| `num_modes` | `usize` | Number of modes to simulate. |
| `stiffness` | `f32` | Material stiffness (affects inharmonicity). |
| `decay_time` | `f32` | Decay time for fundamental in seconds. |
| `brightness` | `f32` | Brightness (high frequency emphasis). |

*Source: [crates/unshape-audio/src/percussion.rs:276](crates/unshape-audio/src/percussion.rs#L276)*

### `Convolution`

Creates a convolution reverb from an impulse response.

`apply(&[f32]) -> ConvolutionReverb`

| Field | Type | Description |
|-------|------|-------------|
| `block_size` | `usize` | Processing block size (power of 2). |
| `mix` | `f32` | Dry/wet mix. |
| `gain` | `f32` | Output gain. |

*Source: [crates/unshape-audio/src/effects.rs:821](crates/unshape-audio/src/effects.rs#L821)*

### `GranularSynth`

Configuration for grain generation.

`apply(&GranularInput) -> Vec<f32>`

| Field | Type | Description |
|-------|------|-------------|
| `size_ms` | `f32` | Grain size in milliseconds. |
| `size_jitter` | `f32` | Randomization of grain size (0.0 to 1.0). |
| `position` | `f32` | Position in the source buffer (0.0 to 1.0). |
| `position_jitter` | `f32` | Randomization of position. |
| `pitch` | `f32` | Pitch multiplier (1.0 = original pitch). |
| `pitch_jitter` | `f32` | Randomization of pitch. |
| `density` | `f32` | Grain density (grains per second). |
| `pan` | `f32` | Pan position (-1.0 to 1.0). |
| `pan_jitter` | `f32` | Randomization of pan. |

*Source: [crates/unshape-audio/src/granular.rs:49](crates/unshape-audio/src/granular.rs#L49)*

### `MembraneSynth`

Configuration for membrane synthesis.

`apply(&PercussionInput) -> Vec<f32>`

| Field | Type | Description |
|-------|------|-------------|
| `fundamental` | `f32` | Fundamental frequency in Hz. |
| `num_modes` | `usize` | Number of modes to simulate. |
| `tension` | `f32` | Tension parameter (affects frequency ratios). |
| `damping` | `f32` | Damping factor (higher = faster decay). |
| `decay_time` | `f32` | Decay time for fundamental in seconds. |

*Source: [crates/unshape-audio/src/percussion.rs:102](crates/unshape-audio/src/percussion.rs#L102)*

### `PlateSynth`

Configuration for plate synthesis.

`apply(&PercussionInput) -> Vec<f32>`

| Field | Type | Description |
|-------|------|-------------|
| `fundamental` | `f32` | Fundamental frequency in Hz. |
| `num_modes` | `usize` | Number of modes to simulate. |
| `thickness` | `f32` | Thickness parameter (affects frequency spread). |
| `decay_time` | `f32` | Decay time for fundamental in seconds. |
| `density` | `f32` | High frequency density. |

*Source: [crates/unshape-audio/src/percussion.rs:472](crates/unshape-audio/src/percussion.rs#L472)*

### `PluckSynth`

Configuration for plucking a string.

`apply(&PluckInput) -> Vec<f32>`

| Field | Type | Description |
|-------|------|-------------|
| `amplitude` | `f32` | Initial amplitude (0.0 to 1.0). |
| `damping` | `f32` | Damping factor (0.0 to 1.0). Higher = more damping = faster decay. |
| `brightness` | `f32` | Brightness (0.0 to 1.0). Lower = duller sound. |
| `noise_blend` | `f32` | Blend between noise (0.0) and sawtooth (1.0) for initial excitation. |

*Source: [crates/unshape-audio/src/physical.rs:42](crates/unshape-audio/src/physical.rs#L42)*

### `Spatialize`

Configuration for the spatializer.

`apply(&SpatializeInput) -> SpatializeOutput`

| Field | Type | Description |
|-------|------|-------------|
| `distance_model` | `DistanceModel` | Distance attenuation model. |
| `enable_doppler` | `bool` | Whether to enable Doppler effect. |
| `speed_of_sound` | `f32` | Speed of sound in units per second (default: 343 m/s). |
| `hrtf_mode` | `HrtfMode` | HRTF mode. |
| `max_itd_samples` | `usize` | Maximum delay for ITD in samples. |

*Source: [crates/unshape-audio/src/spatial.rs:289](crates/unshape-audio/src/spatial.rs#L289)*

### `Stft`

Configuration for STFT analysis.

`apply(&[f32]) -> StftResult`

| Field | Type | Description |
|-------|------|-------------|
| `window_size` | `usize` | FFT window size (must be power of 2). |
| `hop_size` | `usize` | Hop size between consecutive frames. |
| `window` | `Vec<f32>` | Window function. |

*Source: [crates/unshape-audio/src/spectral.rs:65](crates/unshape-audio/src/spectral.rs#L65)*

### `TimeStretch`

Configuration for time-stretching.

`apply(&[f32]) -> Vec<f32>`

| Field | Type | Description |
|-------|------|-------------|
| `window_size` | `usize` | FFT window size (must be power of 2). |
| `analysis_hop` | `usize` | Analysis hop size. |
| `stretch_factor` | `f32` | Time stretch factor (< 1.0 = faster, > 1.0 = slower). |
| `preserve_transients` | `bool` | Whether to preserve transients. |
| `transient_threshold` | `f32` | Transient detection threshold. |

*Source: [crates/unshape-audio/src/spectral.rs:405](crates/unshape-audio/src/spectral.rs#L405)*

### `VocodeSynth`

Configuration for the vocoder.

`apply(&VocodeInput) -> Vec<f32>`

| Field | Type | Description |
|-------|------|-------------|
| `window_size` | `usize` | FFT window size (must be power of 2). |
| `hop_size` | `usize` | Hop size between consecutive frames. |
| `num_bands` | `usize` | Number of frequency bands for the filterbank. |
| `envelope_smoothing` | `f32` | Envelope follower smoothing (0-1, higher = smoother). |

*Source: [crates/unshape-audio/src/vocoder.rs:44](crates/unshape-audio/src/vocoder.rs#L44)*

### `Warp`

Time remapping operation via Dew expression.

`apply(Pattern<T>) -> Pattern<T>`

| Field | Type | Description |
|-------|------|-------------|
| `time_expr` | `FieldExpr` | Expression that maps old onset time (x) to new onset time. |

*Source: [crates/unshape-audio/src/pattern.rs:1119](crates/unshape-audio/src/pattern.rs#L1119)*

---

## unshape-automata

### `CellularAutomaton2DConfig`

Configuration operation for creating a 2D cellular automaton.

`apply(()) -> CellularAutomaton2D`

| Field | Type | Description |
|-------|------|-------------|
| `width` | `usize` | Width of the grid. |
| `height` | `usize` | Height of the grid. |
| `birth` | `Vec<u8>` | Birth rule (number of neighbors that cause birth). |
| `survive` | `Vec<u8>` | Survival rule (number of neighbors that allow survival). |
| `wrap` | `bool` | Whether to wrap at edges (toroidal topology). |
| `seed` | `Option<u64>` | Seed for random initialization (None = start empty). |
| `density` | `f32` | Density for random initialization (0.0 - 1.0). |

*Source: [crates/unshape-automata/src/lib.rs:477](crates/unshape-automata/src/lib.rs#L477)*

### `ElementaryCAConfig`

Configuration operation for creating a 1D elementary cellular automaton.

`apply(()) -> ElementaryCA`

| Field | Type | Description |
|-------|------|-------------|
| `width` | `usize` | Width of the automaton (number of cells). |
| `rule` | `u8` | Rule number (0-255). |
| `wrap` | `bool` | Whether to wrap at edges (toroidal topology). |
| `seed` | `Option<u64>` | Seed for random initialization (None = start with center cell only). |

*Source: [crates/unshape-automata/src/lib.rs:428](crates/unshape-automata/src/lib.rs#L428)*

### `GeneratePattern`

Operation to generate a 2D pattern from a 1D elementary CA.

`apply(&ElementaryCA) -> Vec<Vec<bool>>`

| Field | Type | Description |
|-------|------|-------------|
| `generations` | `usize` | Number of generations to produce. |

*Source: [crates/unshape-automata/src/lib.rs:614](crates/unshape-automata/src/lib.rs#L614)*

### `StepCellularAutomaton2D`

Operation to step a 2D CA forward.

`apply(&CellularAutomaton2D) -> CellularAutomaton2D`

| Field | Type | Description |
|-------|------|-------------|
| `steps` | `usize` | Number of steps to advance. |

*Source: [crates/unshape-automata/src/lib.rs:582](crates/unshape-automata/src/lib.rs#L582)*

### `StepElementaryCA`

Operation to step an elementary CA forward.

`apply(&ElementaryCA) -> ElementaryCA`

| Field | Type | Description |
|-------|------|-------------|
| `steps` | `usize` | Number of steps to advance. |

*Source: [crates/unshape-automata/src/lib.rs:550](crates/unshape-automata/src/lib.rs#L550)*

---

## unshape-crossdomain

### `AudioToImage`

Converts audio to a spectrogram image.

`apply(&[f32]) -> ImageField`

| Field | Type | Description |
|-------|------|-------------|
| `sample_rate` | `u32` | Sample rate of the input audio in Hz. |
| `width` | `u32` | Output image width in pixels. |
| `height` | `u32` | Output image height in pixels (number of frequency bins). |
| `window_size` | `usize` | FFT window size. |
| `hop_size` | `usize` | Hop size between windows. |
| `log_magnitude` | `bool` | Whether to use logarithmic magnitude scaling. |
| `gain` | `f32` | Gain applied to magnitude values. |

*Source: [crates/unshape-crossdomain/src/lib.rs:348](crates/unshape-crossdomain/src/lib.rs#L348)*

### `ImageToAudio`

Converts an image to audio using additive synthesis.

`apply(&ImageField) -> Vec<f32>`

| Field | Type | Description |
|-------|------|-------------|
| `sample_rate` | `u32` | Sample rate in Hz. |
| `duration` | `f32` | Duration of the output audio in seconds. |
| `min_freq` | `f32` | Minimum frequency (Hz) for the bottom of the image. |
| `max_freq` | `f32` | Maximum frequency (Hz) for the top of the image. |
| `log_frequency` | `bool` | Whether to use logarithmic frequency scaling. |

*Source: [crates/unshape-crossdomain/src/lib.rs:246](crates/unshape-crossdomain/src/lib.rs#L246)*

---

## unshape-field

### `HydraulicErosion`

Hydraulic erosion simulation operation.

`apply(&Heightmap) -> Heightmap`

| Field | Type | Description |
|-------|------|-------------|
| `iterations` | `usize` | Number of water droplets to simulate. |
| `max_lifetime` | `usize` | Maximum steps per droplet lifetime. |
| `initial_water` | `f32` | Initial water volume per droplet. |
| `initial_speed` | `f32` | Initial speed of droplets. |
| `inertia` | `f32` | Inertia factor (0 = follow gradient exactly, 1 = maintain direction). |
| `min_slope` | `f32` | Minimum slope for erosion to occur. |
| `capacity_factor` | `f32` | Sediment capacity multiplier. |
| `erosion_rate` | `f32` | Rate of sediment pickup. |
| `deposition_rate` | `f32` | Rate of sediment deposition. |
| `evaporation_rate` | `f32` | Water evaporation rate per step. |
| `gravity` | `f32` | Gravity strength. |
| `brush_radius` | `usize` | Erosion brush radius. |
| `seed` | `u64` | Random seed for simulation. |

*Source: [crates/unshape-field/src/lib.rs:3571](crates/unshape-field/src/lib.rs#L3571)*

### `RiverNetwork`

River network generation operation.

`apply(&Heightmap) -> Network`

| Field | Type | Description |
|-------|------|-------------|
| `num_sources` | `usize` | Number of river sources. |
| `source_min_height` | `f32` | Minimum height for sources. |
| `max_steps` | `usize` | Number of steps per river. |
| `step_size` | `f32` | Step size for gradient descent. |
| `merge_rivers` | `bool` | Whether to merge rivers that meet. |
| `seed` | `u64` | Random seed for generation. |

*Source: [crates/unshape-field/src/lib.rs:4351](crates/unshape-field/src/lib.rs#L4351)*

### `RoadNetwork`

Road network generation operation.

`apply(()) -> Network`

| Field | Type | Description |
|-------|------|-------------|
| `num_nodes` | `usize` | Number of cities/intersections. |
| `bounds` | `(f32,f32,f32,f32)` | Area bounds (min_x, min_y, max_x, max_y). |
| `use_mst` | `bool` | Whether to generate minimum spanning tree first. |
| `extra_connectivity` | `f32` | Extra connections beyond MST (0.0 = none, 1.0 = full). |
| `relaxation_iterations` | `usize` | Number of path relaxation iterations. |
| `curvature` | `f32` | Path curvature amount. |
| `seed` | `u64` | Random seed for generation. |

*Source: [crates/unshape-field/src/lib.rs:4121](crates/unshape-field/src/lib.rs#L4121)*

### `ThermalErosion`

Thermal erosion simulation operation.

`apply(&Heightmap) -> Heightmap`

| Field | Type | Description |
|-------|------|-------------|
| `iterations` | `usize` | Number of iterations. |
| `talus_angle` | `f32` | Maximum slope angle (as tangent) before material slides. |
| `transfer_rate` | `f32` | Rate of material transfer per iteration. |

*Source: [crates/unshape-field/src/lib.rs:3790](crates/unshape-field/src/lib.rs#L3790)*

---

## unshape-fluid

### `Fluid`

Configuration for grid-based fluid simulation.

`apply(()) -> Fluid`

| Field | Type | Description |
|-------|------|-------------|
| `diffusion` | `f32` | Diffusion rate (viscosity). |
| `iterations` | `u32` | Number of iterations for Gauss-Seidel solver. |
| `dt` | `f32` | Time step for simulation. |

*Source: [crates/unshape-fluid/src/lib.rs:23](crates/unshape-fluid/src/lib.rs#L23)*

### `Smoke`

Configuration for smoke simulation.

`apply(()) -> Smoke`

| Field | Type | Description |
|-------|------|-------------|
| `diffusion` | `f32` | Diffusion rate for velocity. |
| `iterations` | `u32` | Number of iterations for solver. |
| `dt` | `f32` | Time step. |
| `buoyancy` | `f32` | Buoyancy coefficient (how much hot gas rises). |
| `ambient_temperature` | `f32` | Ambient temperature. |
| `temperature_dissipation` | `f32` | Temperature dissipation rate (cooling). |
| `density_dissipation` | `f32` | Density dissipation rate. |

*Source: [crates/unshape-fluid/src/lib.rs:1342](crates/unshape-fluid/src/lib.rs#L1342)*

### `Sph`

Configuration for SPH simulation.

`apply(()) -> Sph`

| Field | Type | Description |
|-------|------|-------------|
| `rest_density` | `f32` | Rest density of the fluid. |
| `gas_constant` | `f32` | Gas constant for pressure calculation. |
| `viscosity` | `f32` | Viscosity coefficient. |
| `h` | `f32` | Smoothing radius (kernel size). |
| `dt` | `f32` | Time step. |
| `gravity` | `Vec2` | Gravity. |
| `boundary_damping` | `f32` | Boundary damping. |

*Source: [crates/unshape-fluid/src/lib.rs:844](crates/unshape-fluid/src/lib.rs#L844)*

### `SphParams3D`

Configuration for 3D SPH simulation.

`apply(()) -> SphParams3D`

| Field | Type | Description |
|-------|------|-------------|
| `rest_density` | `f32` | Rest density of the fluid. |
| `gas_constant` | `f32` | Gas constant for pressure calculation. |
| `viscosity` | `f32` | Viscosity coefficient. |
| `h` | `f32` | Smoothing radius (kernel size). |
| `dt` | `f32` | Time step. |
| `gravity` | `Vec3` | Gravity. |
| `boundary_damping` | `f32` | Boundary damping. |

*Source: [crates/unshape-fluid/src/lib.rs:1123](crates/unshape-fluid/src/lib.rs#L1123)*

---

## unshape-gpu

### `NoiseConfig`

Configuration for noise generation.

`apply(()) -> NoiseConfig`

| Field | Type | Description |
|-------|------|-------------|
| `noise_type` | `NoiseType` | Type of noise. |
| `scale` | `f32` | Frequency/scale of the noise. |
| `octaves` | `u32` | Number of octaves for FBM. |
| `persistence` | `f32` | Persistence for FBM (amplitude multiplier per octave). |
| `lacunarity` | `f32` | Lacunarity for FBM (frequency multiplier per octave). |
| `seed` | `u32` | Seed for randomization. |

*Source: [crates/unshape-gpu/src/noise.rs:32](crates/unshape-gpu/src/noise.rs#L32)*

---

## unshape-image

### `AnimationConfig`

Configuration for animation rendering.

`apply(()) -> AnimationConfig`

| Field | Type | Description |
|-------|------|-------------|
| `width` | `u32` | Output width in pixels. |
| `height` | `u32` | Output height in pixels. |
| `num_frames` | `usize` | Number of frames. |
| `frame_duration` | `f32` | Frame duration in seconds. |
| `samples` | `u32` | Anti-aliasing samples (1 = no AA). |

*Source: [crates/unshape-image/src/lib.rs:565](crates/unshape-image/src/lib.rs#L565)*

### `BakeConfig`

Configuration for texture baking.

`apply(()) -> BakeConfig`

| Field | Type | Description |
|-------|------|-------------|
| `width` | `u32` | Output width in pixels. |
| `height` | `u32` | Output height in pixels. |
| `samples` | `u32` | Number of samples per pixel for anti-aliasing (1 = no AA). |

*Source: [crates/unshape-image/src/lib.rs:312](crates/unshape-image/src/lib.rs#L312)*

### `ChromaticAberration`

Applies chromatic aberration effect to an image.

`apply(&ImageField) -> ImageField`

| Field | Type | Description |
|-------|------|-------------|
| `red_offset` | `f32` | Offset amount for red channel (negative = inward, positive = outward). |
| `green_offset` | `f32` | Offset amount for green channel. |
| `blue_offset` | `f32` | Offset amount for blue channel. |
| `center` | `(f32,f32)` | Center point for radial offset (normalized coordinates, default: (0.5, 0.5)). |

*Source: [crates/unshape-image/src/lib.rs:1991](crates/unshape-image/src/lib.rs#L1991)*

### `Composite`

Image compositing operation.

`apply(&ImageField) -> ImageField`

| Field | Type | Description |
|-------|------|-------------|
| `mode` | `BlendMode` | The blend mode to use. |
| `opacity` | `f32` | Opacity of the overlay (0.0 = transparent, 1.0 = opaque). |

*Source: [crates/unshape-image/src/lib.rs:1064](crates/unshape-image/src/lib.rs#L1064)*

### `Convolve`

2D spatial convolution operation.

`apply(&ImageField) -> ImageField`

| Field | Type | Description |
|-------|------|-------------|
| `kernel` | `Kernel` | The convolution kernel. |

*Source: [crates/unshape-image/src/lib.rs:954](crates/unshape-image/src/lib.rs#L954)*

### `CurveDiffuse`

Curve-based error diffusion (Riemersma dithering).

`apply(&ImageField) -> ImageField`

| Field | Type | Description |
|-------|------|-------------|
| `curve` | `TraversalCurve` | The traversal curve to use. |
| `history_size` | `usize` | Size of the error history buffer. |
| `decay` | `f32` | Decay ratio for error weights (0-1, smaller = faster decay). |
| `levels` | `u32` | Number of quantization levels. |

*Source: [crates/unshape-image/src/lib.rs:2911](crates/unshape-image/src/lib.rs#L2911)*

### `Datamosh`

Configuration for datamosh glitch effect.

`apply(&ImageField) -> ImageField`

| Field | Type | Description |
|-------|------|-------------|
| `block_size` | `u32` | Block size for motion compensation (typical values: 8, 16, 32). |
| `iterations` | `u32` | Number of "frames" to accumulate artifacts (more = more corruption). |
| `motion_intensity` | `f32` | Motion intensity - how much blocks shift between iterations (0-1). |
| `decay` | `f32` | Decay factor - how much previous frame influences result (0-1). |
| `seed` | `u32` | Random seed for reproducible motion vectors. |
| `motion` | `MotionPattern` | Motion pattern to use. |
| `freeze_probability` | `f32` | Probability of a block "sticking" and not updating (0-1). |

*Source: [crates/unshape-image/src/lib.rs:6154](crates/unshape-image/src/lib.rs#L6154)*

### `Dct2d`

2D Discrete Cosine Transform.

`apply(&ImageField) -> ImageField`

| Field | Type | Description |
|-------|------|-------------|
| `block_size` | `Option<u32>` | Block size for block-based DCT. None = whole image. |

*Source: [crates/unshape-image/src/lib.rs:7590](crates/unshape-image/src/lib.rs#L7590)*

### `ErrorDiffuse`

Error diffusion dithering operation.

`apply(&ImageField) -> ImageField`

| Field | Type | Description |
|-------|------|-------------|
| `kernel` | `DiffusionKernel` | The diffusion kernel to use. |
| `levels` | `u32` | Number of quantization levels. |

*Source: [crates/unshape-image/src/lib.rs:2803](crates/unshape-image/src/lib.rs#L2803)*

### `ExtractBitPlane`

Extracts a single bit plane from an image channel.

`apply(&ImageField) -> ImageField`

| Field | Type | Description |
|-------|------|-------------|
| `channel` | `Channel` | Which channel to extract from. |
| `bit` | `u8` | Which bit to extract (0 = LSB, 7 = MSB for 8-bit). |

*Source: [crates/unshape-image/src/lib.rs:7322](crates/unshape-image/src/lib.rs#L7322)*

### `Fft2d`

2D Fast Fourier Transform.

`apply(&ImageField) -> (ImageField,ImageField)`

*Source: [crates/unshape-image/src/lib.rs:7463](crates/unshape-image/src/lib.rs#L7463)*

### `FftShift`

Shifts zero frequency to center of spectrum.

`apply(&ImageField) -> ImageField`

*Source: [crates/unshape-image/src/lib.rs:7556](crates/unshape-image/src/lib.rs#L7556)*

### `FromInt`

Converts integer pixel data back to floating-point image (0-1).

`apply(&[[i32;4]]) -> ImageField`

| Field | Type | Description |
|-------|------|-------------|
| `range` | `IntRange` | The integer range to convert from. |

*Source: [crates/unshape-image/src/lib.rs:7252](crates/unshape-image/src/lib.rs#L7252)*

### `Idct2d`

2D Inverse Discrete Cosine Transform.

`apply(&ImageField) -> ImageField`

| Field | Type | Description |
|-------|------|-------------|
| `block_size` | `Option<u32>` | Block size for block-based IDCT. None = whole image. |

*Source: [crates/unshape-image/src/lib.rs:7643](crates/unshape-image/src/lib.rs#L7643)*

### `Ifft2d`

2D Inverse Fast Fourier Transform.

`apply(&ImageField) -> ImageField`

*Source: [crates/unshape-image/src/lib.rs:7516](crates/unshape-image/src/lib.rs#L7516)*

### `Inpaint`

Configuration for diffusion-based inpainting operations.

`apply(&ImageField) -> ImageField`

| Field | Type | Description |
|-------|------|-------------|
| `iterations` | `u32` | Number of iterations for diffusion-based inpainting. |
| `diffusion_rate` | `f32` | Diffusion rate (0.0-1.0). Higher values spread color faster. |

*Source: [crates/unshape-image/src/lib.rs:5169](crates/unshape-image/src/lib.rs#L5169)*

### `LensDistortion`

Applies radial lens distortion (barrel or pincushion).

`apply(&ImageField) -> ImageField`

| Field | Type | Description |
|-------|------|-------------|
| `strength` | `f32` | Distortion strength. Positive = barrel, negative = pincushion. |
| `center` | `(f32,f32)` | Center point for distortion (normalized coordinates). |

*Source: [crates/unshape-image/src/lib.rs:4281](crates/unshape-image/src/lib.rs#L4281)*

### `Levels`

Applies levels adjustment to an image.

`apply(&ImageField) -> ImageField`

| Field | Type | Description |
|-------|------|-------------|
| `input_black` | `f32` | Input black point (values below this become 0). Range: 0-1. |
| `input_white` | `f32` | Input white point (values above this become 1). Range: 0-1. |
| `gamma` | `f32` | Gamma correction (1.0 = linear, <1 = brighten, >1 = darken). |
| `output_black` | `f32` | Output black point. Range: 0-1. |
| `output_white` | `f32` | Output white point. Range: 0-1. |

*Source: [crates/unshape-image/src/lib.rs:2114](crates/unshape-image/src/lib.rs#L2114)*

### `Lut1D`

1D lookup table for color grading.

`apply([f32;4]) -> [f32;4]`

| Field | Type | Description |
|-------|------|-------------|
| `red` | `Vec<f32>` | Red channel LUT entries. |
| `green` | `Vec<f32>` | Green channel LUT entries. |
| `blue` | `Vec<f32>` | Blue channel LUT entries. |

*Source: [crates/unshape-image/src/lib.rs:8055](crates/unshape-image/src/lib.rs#L8055)*

### `Lut3D`

3D lookup table for color grading.

`apply([f32;4]) -> [f32;4]`

| Field | Type | Description |
|-------|------|-------------|
| `data` | `Vec<[f32;3]>` | LUT data as [R][G][B] -> [r, g, b]. |
| `size` | `usize` | Size of each dimension. |

*Source: [crates/unshape-image/src/lib.rs:8161](crates/unshape-image/src/lib.rs#L8161)*

### `MapPixels`

Per-pixel color transform operation.

`apply(&ImageField) -> ImageField`

| Field | Type | Description |
|-------|------|-------------|
| `expr` | `ColorExpr` | The color transform expression. |

*Source: [crates/unshape-image/src/lib.rs:1138](crates/unshape-image/src/lib.rs#L1138)*

### `PatchMatch`

Configuration for PatchMatch-based inpainting.

`apply(&ImageField) -> ImageField`

| Field | Type | Description |
|-------|------|-------------|
| `patch_size` | `u32` | Size of patches to match (must be odd). |
| `pyramid_levels` | `u32` | Number of pyramid levels for multi-scale processing. |
| `iterations` | `u32` | Number of iterations per pyramid level. |

*Source: [crates/unshape-image/src/lib.rs:5316](crates/unshape-image/src/lib.rs#L5316)*

### `Quantize`

Quantize a value to discrete levels.

`apply(f32) -> f32`

| Field | Type | Description |
|-------|------|-------------|
| `levels` | `u32` | Number of discrete levels (2-256). |

*Source: [crates/unshape-image/src/lib.rs:2352](crates/unshape-image/src/lib.rs#L2352)*

### `QuantizeWithBias`

Quantizes pixel values with a bias toward specific values.

`apply(&ImageField) -> ImageField`

| Field | Type | Description |
|-------|------|-------------|
| `levels` | `u32` | Number of quantization levels. |

*Source: [crates/unshape-image/src/lib.rs:7825](crates/unshape-image/src/lib.rs#L7825)*

### `RemapUv`

UV coordinate remapping operation.

`apply(&ImageField) -> ImageField`

| Field | Type | Description |
|-------|------|-------------|
| `expr` | `UvExpr` | The UV remapping expression. |

*Source: [crates/unshape-image/src/lib.rs:1102](crates/unshape-image/src/lib.rs#L1102)*

### `Resize`

Image resizing operation.

`apply(&ImageField) -> ImageField`

| Field | Type | Description |
|-------|------|-------------|
| `width` | `u32` | Target width in pixels. |
| `height` | `u32` | Target height in pixels. |

*Source: [crates/unshape-image/src/lib.rs:1026](crates/unshape-image/src/lib.rs#L1026)*

### `SetBitPlane`

Sets a single bit plane in an image channel from a source image.

`apply(&ImageField) -> ImageField`

| Field | Type | Description |
|-------|------|-------------|
| `channel` | `Channel` | Which channel to modify. |
| `bit` | `u8` | Which bit to set (0 = LSB, 7 = MSB for 8-bit). |

*Source: [crates/unshape-image/src/lib.rs:7380](crates/unshape-image/src/lib.rs#L7380)*

### `Spherize`

Configuration for spherize/bulge effect.

`apply(&ImageField) -> ImageField`

| Field | Type | Description |
|-------|------|-------------|
| `strength` | `f32` | Bulge strength (positive = bulge out, negative = pinch in). |
| `center` | `(f32,f32)` | Center point (normalized coordinates). |

*Source: [crates/unshape-image/src/lib.rs:4664](crates/unshape-image/src/lib.rs#L4664)*

### `SpreadSpectrum`

Spreads image data using a pseudorandom sequence.

`apply(&ImageField) -> ImageField`

| Field | Type | Description |
|-------|------|-------------|
| `seed` | `u64` | Seed for the pseudorandom sequence. |
| `factor` | `f32` | Spreading factor (higher = more robust, lower visual quality). |

*Source: [crates/unshape-image/src/lib.rs:7703](crates/unshape-image/src/lib.rs#L7703)*

### `Swirl`

Configuration for swirl/twist distortion.

`apply(&ImageField) -> ImageField`

| Field | Type | Description |
|-------|------|-------------|
| `angle` | `f32` | Maximum rotation in radians at center. |
| `radius` | `f32` | Radius of effect (normalized, 1.0 = half image size). |
| `center` | `(f32,f32)` | Center point (normalized coordinates). |

*Source: [crates/unshape-image/src/lib.rs:4571](crates/unshape-image/src/lib.rs#L4571)*

### `ToInt`

Converts a floating-point image (0-1) to integer representation.

`apply(&ImageField) -> Vec<[i32;4]>`

| Field | Type | Description |
|-------|------|-------------|
| `range` | `IntRange` | The integer range to convert to. |
| `channel` | `Option<Channel>` | Which channel to convert (None = all channels). |

*Source: [crates/unshape-image/src/lib.rs:7176](crates/unshape-image/src/lib.rs#L7176)*

### `TransformConfig`

Configuration for image position transformation.

`apply(&ImageField) -> ImageField`

| Field | Type | Description |
|-------|------|-------------|
| `matrix` | `[[f32;3];3]` | 3x3 transformation matrix for UV coordinates. |
| `filter` | `bool` | Whether to use bilinear filtering regardless of image setting. |

*Source: [crates/unshape-image/src/lib.rs:7923](crates/unshape-image/src/lib.rs#L7923)*

### `UnspreadSpectrum`

Reverses spread spectrum operation.

`apply(&ImageField) -> ImageField`

| Field | Type | Description |
|-------|------|-------------|
| `seed` | `u64` | Seed for the pseudorandom sequence (must match SpreadSpectrum). |
| `factor` | `f32` | Factor used in original spread (must match). |

*Source: [crates/unshape-image/src/lib.rs:7772](crates/unshape-image/src/lib.rs#L7772)*

### `WaveDistortion`

Applies wave distortion to an image.

`apply(&ImageField) -> ImageField`

| Field | Type | Description |
|-------|------|-------------|
| `amplitude_x` | `f32` | Amplitude in X direction (as fraction of image size). |
| `amplitude_y` | `f32` | Amplitude in Y direction. |
| `frequency_x` | `f32` | Frequency of waves in X direction. |
| `frequency_y` | `f32` | Frequency of waves in Y direction. |
| `phase` | `f32` | Phase offset in radians. |

*Source: [crates/unshape-image/src/lib.rs:4391](crates/unshape-image/src/lib.rs#L4391)*

### `WernessDither`

Werness dithering - hybrid noise-threshold + error absorption.

`apply(&ImageField) -> ImageField`

| Field | Type | Description |
|-------|------|-------------|
| `levels` | `u32` | Number of quantization levels. |
| `iterations` | `u32` | Number of iterations. |

*Source: [crates/unshape-image/src/lib.rs:3076](crates/unshape-image/src/lib.rs#L3076)*

---

## unshape-lsystem

### `Turtle2D`

Interprets an L-system string using 2D turtle graphics.

`apply(&str) -> Vec<TurtleSegment2D>`

| Field | Type | Description |
|-------|------|-------------|
| `angle` | `f32` | Rotation angle in degrees for + and - commands. |
| `step` | `f32` | Step distance for F and f commands. |
| `scale_factor` | `f32` | Scale factor for push/pop. |

*Source: [crates/unshape-lsystem/src/lib.rs:163](crates/unshape-lsystem/src/lib.rs#L163)*

### `Turtle3D`

Interprets an L-system string using 3D turtle graphics.

`apply(&str) -> Vec<TurtleSegment3D>`

| Field | Type | Description |
|-------|------|-------------|
| `angle` | `f32` | Rotation angle in degrees for + and - commands. |
| `step` | `f32` | Step distance for F and f commands. |
| `scale_factor` | `f32` | Scale factor for push/pop. |

*Source: [crates/unshape-lsystem/src/lib.rs:197](crates/unshape-lsystem/src/lib.rs#L197)*

---

## unshape-mesh

### `BakeAo`

Bakes ambient occlusion to per-vertex values.

`apply(&Mesh) -> Vec<f32>`

| Field | Type | Description |
|-------|------|-------------|
| `ray_count` | `u32` | Number of rays to cast per sample. |
| `max_distance` | `f32` | Maximum ray distance. |
| `bias` | `f32` | Bias offset along normal to avoid self-intersection. |
| `cosine_weighted` | `bool` | Whether to use cosine-weighted hemisphere sampling. |
| `falloff_power` | `f32` | Power for AO falloff (higher = softer shadows). |

*Source: [crates/unshape-mesh/src/ao.rs:22](crates/unshape-mesh/src/ao.rs#L22)*

### `Bevel`

Bevels edges of a half-edge mesh.

`apply(&HalfEdgeMesh) -> HalfEdgeMesh`

| Field | Type | Description |
|-------|------|-------------|
| `amount` | `f32` | The amount to bevel (distance from original edge/vertex). |
| `segments` | `u32` | Number of segments for smooth bevels (1 = flat chamfer). |
| `smooth` | `bool` | Whether to use a smooth profile (arc) or flat (linear). |

*Source: [crates/unshape-mesh/src/bevel.rs:30](crates/unshape-mesh/src/bevel.rs#L30)*

### `BridgeEdgeLoops`

Bridges two edge loops by creating connecting faces.

`apply(&Mesh) -> Mesh`

| Field | Type | Description |
|-------|------|-------------|
| `segments` | `u32` | Number of segments in the bridge. |
| `twist` | `i32` | Twist amount (in edge loop positions). |

*Source: [crates/unshape-mesh/src/edit.rs:1620](crates/unshape-mesh/src/edit.rs#L1620)*

### `CatmullClark`

Catmull-Clark subdivision operation.

`apply(&Mesh) -> Mesh`

| Field | Type | Description |
|-------|------|-------------|
| `levels` | `u32` | Number of subdivision levels. |
| `creases` | `Option<EdgeCreases>` | Optional edge creases for controlling sharpness. |

*Source: [crates/unshape-mesh/src/subdivision.rs:50](crates/unshape-mesh/src/subdivision.rs#L50)*

### `Cone`

Generates a cone mesh centered at the origin.

`apply(()) -> Mesh`

| Field | Type | Description |
|-------|------|-------------|
| `radius` | `f32` | Radius of the base. |
| `height` | `f32` | Height of the cone. |
| `segments` | `u32` | Number of divisions around the circumference. Minimum 3. |

*Source: [crates/unshape-mesh/src/primitives.rs:443](crates/unshape-mesh/src/primitives.rs#L443)*

### `CreaseEdges`

Marks selected edges with crease weights.

`apply(&mutEdgeCreases) -> ()`

| Field | Type | Description |
|-------|------|-------------|
| `weight` | `f32` | Crease weight to apply (0.0 = smooth, 1.0 = sharp). |

*Source: [crates/unshape-mesh/src/edit.rs:851](crates/unshape-mesh/src/edit.rs#L851)*

### `Cuboid`

Generates a box/cuboid mesh centered at the origin.

`apply(()) -> Mesh`

| Field | Type | Description |
|-------|------|-------------|
| `width` | `f32` | Size along the X axis. |
| `height` | `f32` | Size along the Y axis. |
| `depth` | `f32` | Size along the Z axis. |

*Source: [crates/unshape-mesh/src/primitives.rs:46](crates/unshape-mesh/src/primitives.rs#L46)*

### `Cylinder`

Generates a cylinder mesh centered at the origin.

`apply(()) -> Mesh`

| Field | Type | Description |
|-------|------|-------------|
| `radius` | `f32` | Radius of the cylinder. |
| `height` | `f32` | Height of the cylinder. |
| `segments` | `u32` | Number of divisions around the circumference. Minimum 3. |

*Source: [crates/unshape-mesh/src/primitives.rs:295](crates/unshape-mesh/src/primitives.rs#L295)*

### `Decimate`

Decimates a mesh using edge collapse.

`apply(&Mesh) -> Mesh`

| Field | Type | Description |
|-------|------|-------------|
| `target_triangles` | `Option<usize>` | Target number of triangles. Decimation stops when reached. |
| `target_ratio` | `Option<f32>` | Target reduction ratio (0.0 - 1.0). 0.5 = reduce to half the triangles. |
| `max_error` | `f32` | Maximum error threshold. Edges with higher error won't be collapsed. |
| `preserve_boundary` | `bool` | Whether to preserve boundary edges (edges with only one face). |

*Source: [crates/unshape-mesh/src/decimate.rs:33](crates/unshape-mesh/src/decimate.rs#L33)*

### `DeleteFaces`

Deletes selected faces from the mesh.

`apply(&Mesh) -> Mesh`

| Field | Type | Description |
|-------|------|-------------|
| `remove_orphaned_vertices` | `bool` | Whether to remove vertices that are no longer used by any face. |

*Source: [crates/unshape-mesh/src/edit.rs:53](crates/unshape-mesh/src/edit.rs#L53)*

### `Extrude`

Extrudes a mesh along vertex normals.

`apply(&Mesh) -> Mesh`

| Field | Type | Description |
|-------|------|-------------|
| `amount` | `f32` | Distance to extrude (positive = outward, negative = inward). |
| `create_sides` | `bool` | Whether to create side faces connecting old and new vertices. |
| `keep_original` | `bool` | Whether to keep the original faces. |

*Source: [crates/unshape-mesh/src/ops.rs:22](crates/unshape-mesh/src/ops.rs#L22)*

### `ExtrudeFaces`

Extrudes selected faces along their normals.

`apply(&Mesh) -> Mesh`

| Field | Type | Description |
|-------|------|-------------|
| `amount` | `f32` | Distance to extrude (positive = outward). |

*Source: [crates/unshape-mesh/src/edit.rs:885](crates/unshape-mesh/src/edit.rs#L885)*

### `ExtrudeProfile`

Extrudes a 2D profile along a direction to create a 3D mesh.

`apply(()) -> Mesh`

| Field | Type | Description |
|-------|------|-------------|
| `profile` | `Vec<Vec2>` | The 2D profile points to extrude. |
| `direction` | `Vec3` | Direction and distance to extrude. |
| `cap_start` | `bool` | Whether to cap the start of the extrusion. |
| `cap_end` | `bool` | Whether to cap the end of the extrusion. |
| `segments` | `usize` | Number of segments along the extrusion direction. |

*Source: [crates/unshape-mesh/src/curve_mesh.rs:41](crates/unshape-mesh/src/curve_mesh.rs#L41)*

### `GenerateLodChain`

Generates a LOD chain from a high-poly mesh.

`apply(&Mesh) -> LodChain`

| Field | Type | Description |
|-------|------|-------------|
| `levels` | `usize` | Number of LOD levels to generate (including original). |
| `reduction_ratio` | `f32` | Reduction ratio between consecutive levels (0.0 - 1.0). |
| `min_triangles` | `usize` | Minimum triangles to keep in the lowest LOD. |
| `preserve_boundary` | `bool` | Whether to preserve mesh boundaries during decimation. |
| `max_error` | `f32` | Maximum geometric error allowed during decimation. |
| `screen_thresholds` | `Option<Vec<f32>>` | Screen size thresholds for each LOD (as fraction of screen height). |

*Source: [crates/unshape-mesh/src/lod.rs:33](crates/unshape-mesh/src/lod.rs#L33)*

### `GenerateNavMesh`

Generates a navigation mesh from a floor mesh.

`apply(&Mesh) -> NavMesh`

| Field | Type | Description |
|-------|------|-------------|
| `cell_size` | `f32` | Cell size for rasterization. |
| `cell_height` | `f32` | Cell height for vertical sampling. |
| `agent_height` | `f32` | Minimum walkable height (agent height). |
| `max_slope` | `f32` | Maximum walkable slope in degrees. |
| `agent_radius` | `f32` | Agent radius for obstacle margin. |
| `max_step_height` | `f32` | Maximum step height. |

*Source: [crates/unshape-mesh/src/navmesh.rs:22](crates/unshape-mesh/src/navmesh.rs#L22)*

### `GenerateSdf`

Generates a signed distance field from a mesh.

`apply(&Mesh) -> SdfGrid`

| Field | Type | Description |
|-------|------|-------------|
| `resolution` | `(usize,usize,usize)` | Grid resolution in each dimension. |
| `padding` | `f32` | Bounding box padding (multiplier, 1.1 = 10% padding). |
| `exact` | `bool` | Whether to compute exact distances (slower) or approximate. |

*Source: [crates/unshape-mesh/src/sdf.rs:21](crates/unshape-mesh/src/sdf.rs#L21)*

### `Icosphere`

Generates an icosphere (geodesic sphere) mesh centered at the origin.

`apply(()) -> Mesh`

| Field | Type | Description |
|-------|------|-------------|
| `radius` | `f32` | Radius of the sphere. |
| `subdivisions` | `u32` | Number of subdivision iterations. 0 = icosahedron (20 faces). |

*Source: [crates/unshape-mesh/src/primitives.rs:741](crates/unshape-mesh/src/primitives.rs#L741)*

### `Inset`

Insets all faces toward their centers.

`apply(&Mesh) -> Mesh`

| Field | Type | Description |
|-------|------|-------------|
| `amount` | `f32` | Inset amount (0.0 = no change, 1.0 = shrink to center). |
| `depth` | `f32` | Optional depth (positive = extrude inward after inset). |
| `create_bridge` | `bool` | Whether to create the connecting faces. |

*Source: [crates/unshape-mesh/src/ops.rs:170](crates/unshape-mesh/src/ops.rs#L170)*

### `InsetFaces`

Insets selected faces toward their centers.

`apply(&Mesh) -> Mesh`

| Field | Type | Description |
|-------|------|-------------|
| `amount` | `f32` | Inset amount (0.0 = no change, 1.0 = shrink to center). |
| `individual` | `bool` | Whether to inset faces individually or as a region. |

*Source: [crates/unshape-mesh/src/edit.rs:1046](crates/unshape-mesh/src/edit.rs#L1046)*

### `KnifeCut`

Cuts the mesh along a path.

`apply(&Mesh) -> Mesh`

| Field | Type | Description |
|-------|------|-------------|
| `points` | `Vec<KnifePoint>` | Points defining the cut path. |

*Source: [crates/unshape-mesh/src/edit.rs:1783](crates/unshape-mesh/src/edit.rs#L1783)*

### `Loft`

Lofts between profile curves to create a mesh surface.

`apply(&[Vec<Vec3>]) -> Mesh`

| Field | Type | Description |
|-------|------|-------------|
| `cap_start` | `bool` | Whether to close the surface at the start profile. |
| `cap_end` | `bool` | Whether to close the surface at the end profile. |
| `interpolation_steps` | `usize` | Number of interpolated profiles between each input profile (0 = no interpolation). |
| `closed_profiles` | `bool` | Whether the profiles should be closed loops. |

*Source: [crates/unshape-mesh/src/loft.rs:38](crates/unshape-mesh/src/loft.rs#L38)*

### `MarchingCubes`

Extracts a mesh from a signed distance field using marching cubes.

`apply(&SdfGrid) -> Mesh`

| Field | Type | Description |
|-------|------|-------------|
| `min` | `Vec3` | Bounds of the sampling volume (min corner). |
| `max` | `Vec3` | Bounds of the sampling volume (max corner). |
| `resolution` | `usize` | Resolution (number of cells) in each dimension. |
| `iso_value` | `f32` | Iso-value at which to extract the surface (default: 0.0). |

*Source: [crates/unshape-mesh/src/marching_cubes.rs:37](crates/unshape-mesh/src/marching_cubes.rs#L37)*

### `MergeVertices`

Merges selected vertices into a single vertex.

`apply(&Mesh) -> Mesh`

| Field | Type | Description |
|-------|------|-------------|
| `mode` | `MergeMode` | How to determine the merge position. |
| `position` | `Vec3` | Target position when mode is AtPosition. |

*Source: [crates/unshape-mesh/src/edit.rs:579](crates/unshape-mesh/src/edit.rs#L579)*

### `Plane`

Generates a flat plane mesh in the XZ plane, centered at the origin.

`apply(()) -> Mesh`

| Field | Type | Description |
|-------|------|-------------|
| `width` | `f32` | Size along the X axis. |
| `depth` | `f32` | Size along the Z axis. |
| `subdivisions_x` | `u32` | Number of divisions along X. Minimum 1. |
| `subdivisions_z` | `u32` | Number of divisions along Z. Minimum 1. |

*Source: [crates/unshape-mesh/src/primitives.rs:659](crates/unshape-mesh/src/primitives.rs#L659)*

### `PokeFaces`

Pokes selected faces by adding a vertex at their center.

`apply(&Mesh) -> Mesh`

| Field | Type | Description |
|-------|------|-------------|
| `offset` | `f32` | Offset along face normal for the poked vertex (0 = at face center). |

*Source: [crates/unshape-mesh/src/edit.rs:295](crates/unshape-mesh/src/edit.rs#L295)*

### `Quadify`

Converts triangle pairs to quads where possible.

`apply(&Mesh) -> QuadMesh`

| Field | Type | Description |
|-------|------|-------------|
| `max_angle` | `f32` | Maximum angle difference for merging triangles (degrees). |
| `preserve_sharp` | `bool` | Whether to preserve sharp edges. |
| `sharp_angle` | `f32` | Sharp edge angle threshold (degrees). |

*Source: [crates/unshape-mesh/src/remesh.rs:340](crates/unshape-mesh/src/remesh.rs#L340)*

### `Remesh`

Performs isotropic remeshing to achieve uniform edge lengths.

`apply(&Mesh) -> Mesh`

| Field | Type | Description |
|-------|------|-------------|
| `target_edge_length` | `f32` | Target edge length. |
| `iterations` | `u32` | Number of iterations. |
| `smoothing` | `f32` | Smoothing factor (0-1). |
| `preserve_boundary` | `bool` | Whether to preserve boundary edges. |

*Source: [crates/unshape-mesh/src/remesh.rs:32](crates/unshape-mesh/src/remesh.rs#L32)*

### `Revolve`

Revolves a 2D profile around an axis to create a surface of revolution.

`apply(()) -> Mesh`

| Field | Type | Description |
|-------|------|-------------|
| `profile` | `Vec<Vec2>` | The 2D profile points to revolve. |
| `segments` | `usize` | Number of segments around the revolution. |
| `axis` | `Vec3` | Axis to revolve around (default: Y axis). |
| `angle` | `f32` | Angle to revolve in radians (default: full rotation TAU). |
| `close` | `bool` | Whether to close the revolve (only if angle < TAU). |
| `cap_ends` | `bool` | Whether to cap ends if not a full rotation. |

*Source: [crates/unshape-mesh/src/curve_mesh.rs:93](crates/unshape-mesh/src/curve_mesh.rs#L93)*

### `RipVertices`

Rips selected vertices, disconnecting them from adjacent faces.

`apply(&Mesh) -> Mesh`

*Source: [crates/unshape-mesh/src/edit.rs:1548](crates/unshape-mesh/src/edit.rs#L1548)*

### `ScaleFaces`

Scales selected faces around their individual centers.

`apply(&Mesh) -> Mesh`

| Field | Type | Description |
|-------|------|-------------|
| `factor` | `f32` | Scale factor (1.0 = no change, 0.5 = half size, 2.0 = double). |

*Source: [crates/unshape-mesh/src/edit.rs:394](crates/unshape-mesh/src/edit.rs#L394)*

### `SlideEdges`

Slides selected edges along their adjacent faces.

`apply(&Mesh) -> Mesh`

| Field | Type | Description |
|-------|------|-------------|
| `factor` | `f32` | Slide factor (-1.0 to 1.0, direction along adjacent edges). |

*Source: [crates/unshape-mesh/src/edit.rs:1459](crates/unshape-mesh/src/edit.rs#L1459)*

### `Smooth`

Applies Laplacian smoothing to a mesh.

`apply(&Mesh) -> Mesh`

| Field | Type | Description |
|-------|------|-------------|
| `lambda` | `f32` | Smoothing factor per iteration (0.0 = no change, 1.0 = move to average). |
| `iterations` | `usize` | Number of smoothing iterations. |
| `preserve_boundary` | `bool` | Whether to preserve boundary vertices (don't move them). |

*Source: [crates/unshape-mesh/src/ops.rs:492](crates/unshape-mesh/src/ops.rs#L492)*

### `SmoothVertices`

Applies Laplacian smoothing to selected vertices.

`apply(&Mesh) -> Mesh`

| Field | Type | Description |
|-------|------|-------------|
| `lambda` | `f32` | Smoothing factor per iteration (0.0 = no change, 1.0 = move to average). |
| `iterations` | `usize` | Number of smoothing iterations. |

*Source: [crates/unshape-mesh/src/edit.rs:477](crates/unshape-mesh/src/edit.rs#L477)*

### `SplitEdges`

Splits selected edges to create hard edges.

`apply(&Mesh) -> Mesh`

*Source: [crates/unshape-mesh/src/edit.rs:744](crates/unshape-mesh/src/edit.rs#L744)*

### `SubdivideFaces`

Subdivides selected faces.

`apply(&Mesh) -> Mesh`

| Field | Type | Description |
|-------|------|-------------|
| `levels` | `u32` | Number of subdivision levels. |

*Source: [crates/unshape-mesh/src/edit.rs:1328](crates/unshape-mesh/src/edit.rs#L1328)*

### `Sweep`

Sweeps a 2D profile along a 3D path.

`apply(()) -> Mesh`

| Field | Type | Description |
|-------|------|-------------|
| `profile` | `Vec<Vec2>` | The 2D profile points to sweep. |
| `path` | `Vec<Vec3>` | The 3D path to sweep along. |
| `segments_per_unit` | `usize` | Segments per unit length (currently unused, reserved for future interpolation). |
| `cap_start` | `bool` | Whether to cap the start. |
| `cap_end` | `bool` | Whether to cap the end. |
| `scale_along_path` | `f32` | Scale factor along the path (1.0 = uniform). |

*Source: [crates/unshape-mesh/src/curve_mesh.rs:328](crates/unshape-mesh/src/curve_mesh.rs#L328)*

### `Torus`

Generates a torus (donut shape) mesh centered at the origin.

`apply(()) -> Mesh`

| Field | Type | Description |
|-------|------|-------------|
| `major_radius` | `f32` | Distance from center of torus to center of tube. |
| `minor_radius` | `f32` | Radius of the tube. |
| `major_segments` | `u32` | Divisions around the main ring. Minimum 3. |
| `minor_segments` | `u32` | Divisions around the tube cross-section. Minimum 3. |

*Source: [crates/unshape-mesh/src/primitives.rs:560](crates/unshape-mesh/src/primitives.rs#L560)*

### `TransformVertices`

Transforms selected vertices by a matrix.

`apply(&Mesh) -> Mesh`

| Field | Type | Description |
|-------|------|-------------|
| `matrix` | `Mat4` | The transformation matrix to apply. |

*Source: [crates/unshape-mesh/src/edit.rs:153](crates/unshape-mesh/src/edit.rs#L153)*

### `TriangulateFaces`

Triangulates selected faces (for meshes with quads/ngons).

`apply(&Mesh) -> Mesh`

*Source: [crates/unshape-mesh/src/edit.rs:272](crates/unshape-mesh/src/edit.rs#L272)*

### `UvSphere`

Generates a UV sphere mesh centered at the origin.

`apply(()) -> Mesh`

| Field | Type | Description |
|-------|------|-------------|
| `radius` | `f32` | Radius of the sphere. |
| `segments` | `u32` | Number of horizontal divisions (longitude). Minimum 3. |
| `rings` | `u32` | Number of vertical divisions (latitude). Minimum 2. |

*Source: [crates/unshape-mesh/src/primitives.rs:196](crates/unshape-mesh/src/primitives.rs#L196)*

---

## unshape-particle

### `Attractor`

Attractor/repulsor force.

`apply(()) -> Attractor`

| Field | Type | Description |
|-------|------|-------------|
| `position` | `Vec3` | Attractor position. |
| `strength` | `f32` | Strength (positive = attract, negative = repel). |
| `min_distance` | `f32` | Minimum distance (to prevent extreme forces). |

*Source: [crates/unshape-particle/src/lib.rs:1096](crates/unshape-particle/src/lib.rs#L1096)*

### `ConeEmitter`

Emits particles in a cone shape.

`apply(()) -> ConeEmitter`

| Field | Type | Description |
|-------|------|-------------|
| `position` | `Vec3` | Cone apex position. |
| `direction` | `Vec3` | Cone direction (axis). |
| `angle` | `f32` | Cone angle in radians (half-angle). |
| `speed_min` | `f32` | Minimum initial speed. |
| `speed_max` | `f32` | Maximum initial speed. |
| `lifetime_min` | `f32` | Minimum lifetime. |
| `lifetime_max` | `f32` | Maximum lifetime. |
| `size` | `f32` | Initial size. |
| `color` | `[f32;4]` | Initial color. |

*Source: [crates/unshape-particle/src/lib.rs:896](crates/unshape-particle/src/lib.rs#L896)*

### `CurlNoise`

Curl noise force for divergence-free turbulence.

`apply(()) -> CurlNoise`

| Field | Type | Description |
|-------|------|-------------|
| `strength` | `f32` | Strength of the force. |
| `frequency` | `f32` | Frequency (scale of noise). |
| `epsilon` | `f32` | Small offset for gradient computation. |

*Source: [crates/unshape-particle/src/lib.rs:1261](crates/unshape-particle/src/lib.rs#L1261)*

### `Drag`

Drag force that slows particles.

`apply(()) -> Drag`

| Field | Type | Description |
|-------|------|-------------|
| `coefficient` | `f32` | Drag coefficient (0 = no drag, higher = more drag). |

*Source: [crates/unshape-particle/src/lib.rs:1065](crates/unshape-particle/src/lib.rs#L1065)*

### `Gravity`

Constant directional force (like gravity).

`apply(()) -> Gravity`

| Field | Type | Description |
|-------|------|-------------|
| `acceleration` | `Vec3` | Acceleration vector (units per second squared). |

*Source: [crates/unshape-particle/src/lib.rs:997](crates/unshape-particle/src/lib.rs#L997)*

### `PointEmitter`

Emits particles from a single point.

`apply(()) -> PointEmitter`

| Field | Type | Description |
|-------|------|-------------|
| `position` | `Vec3` | Emission position. |
| `direction` | `Vec3` | Initial velocity direction. |
| `spread` | `f32` | Velocity spread angle in radians. |
| `speed_min` | `f32` | Minimum initial speed. |
| `speed_max` | `f32` | Maximum initial speed. |
| `lifetime_min` | `f32` | Minimum lifetime. |
| `lifetime_max` | `f32` | Maximum lifetime. |
| `size` | `f32` | Initial size. |
| `color` | `[f32;4]` | Initial color. |

*Source: [crates/unshape-particle/src/lib.rs:745](crates/unshape-particle/src/lib.rs#L745)*

### `SphereEmitter`

Emits particles from a sphere surface or volume.

`apply(()) -> SphereEmitter`

| Field | Type | Description |
|-------|------|-------------|
| `center` | `Vec3` | Center position. |
| `radius` | `f32` | Sphere radius. |
| `volume` | `bool` | If true, emit from volume; if false, emit from surface. |
| `speed_min` | `f32` | Initial speed (outward from center). |
| `speed_max` | `f32` | Maximum initial speed. |
| `lifetime_min` | `f32` | Minimum lifetime. |
| `lifetime_max` | `f32` | Maximum lifetime. |
| `size` | `f32` | Initial size. |
| `color` | `[f32;4]` | Initial color. |

*Source: [crates/unshape-particle/src/lib.rs:823](crates/unshape-particle/src/lib.rs#L823)*

### `Turbulence`

Turbulence force using noise.

`apply(()) -> Turbulence`

| Field | Type | Description |
|-------|------|-------------|
| `strength` | `f32` | Strength of the turbulence. |
| `frequency` | `f32` | Frequency (scale of noise). |
| `speed` | `f32` | Animation speed. |
| `time` | `f32` | Current time offset. |

*Source: [crates/unshape-particle/src/lib.rs:1196](crates/unshape-particle/src/lib.rs#L1196)*

### `Vortex`

Vortex force that creates spinning motion.

`apply(()) -> Vortex`

| Field | Type | Description |
|-------|------|-------------|
| `position` | `Vec3` | Vortex axis origin. |
| `axis` | `Vec3` | Vortex axis direction. |
| `strength` | `f32` | Rotational strength. |
| `falloff` | `f32` | How quickly force falls off with distance. |

*Source: [crates/unshape-particle/src/lib.rs:1140](crates/unshape-particle/src/lib.rs#L1140)*

### `Wind`

Constant wind force.

`apply(()) -> Wind`

| Field | Type | Description |
|-------|------|-------------|
| `velocity` | `Vec3` | Wind velocity (target velocity particles are pushed toward). |
| `strength` | `f32` | How strongly particles are pushed (0 = no effect, 1 = instant). |

*Source: [crates/unshape-particle/src/lib.rs:1029](crates/unshape-particle/src/lib.rs#L1029)*

---

## unshape-physics

### `ClothConfig`

Configuration for cloth simulation.

`apply(()) -> ClothConfig`

| Field | Type | Description |
|-------|------|-------------|
| `iterations` | `u32` | Number of constraint solver iterations. |
| `gravity` | `Vec3` | Gravity vector. |
| `damping` | `f32` | Global damping factor (0-1). |
| `stretch_stiffness` | `f32` | Stretch stiffness (0-1). |
| `bend_stiffness` | `f32` | Bend stiffness (0-1). |
| `collision_margin` | `f32` | Collision margin (added to collider radii). |
| `friction` | `f32` | Friction coefficient for collisions. |

*Source: [crates/unshape-physics/src/cloth.rs:15](crates/unshape-physics/src/cloth.rs#L15)*

### `Physics`

Configuration for physics simulation.

`apply(()) -> Physics`

| Field | Type | Description |
|-------|------|-------------|
| `gravity` | `Vec3` | Gravity acceleration. |
| `solver_iterations` | `u32` | Number of constraint solver iterations. |
| `dt` | `f32` | Time step. |

*Source: [crates/unshape-physics/src/lib.rs:879](crates/unshape-physics/src/lib.rs#L879)*

### `SoftBodyConfig`

Configuration for soft body simulation.

`apply(()) -> SoftBodyConfig`

| Field | Type | Description |
|-------|------|-------------|
| `youngs_modulus` | `f32` | Young's modulus (stiffness). |
| `poisson_ratio` | `f32` | Poisson's ratio (0-0.5, incompressibility). |
| `density` | `f32` | Mass density. |
| `damping` | `f32` | Global damping factor. |
| `gravity` | `Vec3` | Gravity vector. |
| `iterations` | `u32` | Number of solver iterations. |

*Source: [crates/unshape-physics/src/softbody.rs:15](crates/unshape-physics/src/softbody.rs#L15)*

---

## unshape-pointcloud

### `CropBounds`

Crop operation for point clouds.

`apply(&PointCloud) -> PointCloud`

| Field | Type | Description |
|-------|------|-------------|
| `min` | `Vec3` | Minimum corner of the bounding box. |
| `max` | `Vec3` | Maximum corner of the bounding box. |

*Source: [crates/unshape-pointcloud/src/lib.rs:652](crates/unshape-pointcloud/src/lib.rs#L652)*

### `EstimateNormals`

Normal estimation operation for point clouds.

`apply(&PointCloud) -> PointCloud`

| Field | Type | Description |
|-------|------|-------------|
| `k` | `usize` | Number of neighbors for local PCA. |

*Source: [crates/unshape-pointcloud/src/lib.rs:687](crates/unshape-pointcloud/src/lib.rs#L687)*

### `Poisson`

Poisson disk sampling operation for mesh surfaces.

`apply(&Mesh) -> PointCloud`

| Field | Type | Description |
|-------|------|-------------|
| `min_distance` | `f32` | Minimum distance between points. |
| `max_attempts` | `u32` | Maximum attempts to place each point. |

*Source: [crates/unshape-pointcloud/src/lib.rs:370](crates/unshape-pointcloud/src/lib.rs#L370)*

### `RemoveOutliers`

Statistical outlier removal operation for point clouds.

`apply(&PointCloud) -> PointCloud`

| Field | Type | Description |
|-------|------|-------------|
| `k` | `usize` | Number of neighbors to consider. |
| `std_ratio` | `f32` | Standard deviation multiplier for outlier threshold. |

*Source: [crates/unshape-pointcloud/src/lib.rs:580](crates/unshape-pointcloud/src/lib.rs#L580)*

### `UniformSampling`

Uniform sampling operation for mesh surfaces.

`apply(&Mesh) -> PointCloud`

| Field | Type | Description |
|-------|------|-------------|
| `count` | `usize` | Number of points to sample. |

*Source: [crates/unshape-pointcloud/src/lib.rs:717](crates/unshape-pointcloud/src/lib.rs#L717)*

### `VoxelDownsample`

Voxel grid downsampling operation for point clouds.

`apply(&PointCloud) -> PointCloud`

| Field | Type | Description |
|-------|------|-------------|
| `voxel_size` | `f32` | Size of voxel grid cells. |

*Source: [crates/unshape-pointcloud/src/lib.rs:622](crates/unshape-pointcloud/src/lib.rs#L622)*

---

## unshape-procgen

### `GenerateMaze`

Operation for maze generation.

`apply(&u64) -> Maze`

| Field | Type | Description |
|-------|------|-------------|
| `width` | `usize` | Width in cells. |
| `height` | `usize` | Height in cells. |
| `algorithm` | `MazeAlgorithm` | Algorithm to use. |
| `add_entrance` | `bool` | Whether to add entrance at top-left. |
| `add_exit` | `bool` | Whether to add exit at bottom-right. |

*Source: [crates/unshape-procgen/src/maze.rs:574](crates/unshape-procgen/src/maze.rs#L574)*

### `GenerateRiver`

Operation to generate a simple procedural river from source to sink.

`apply(&u64) -> RiverNetwork`

| Field | Type | Description |
|-------|------|-------------|
| `source` | `Vec2` | Source position (upstream). |
| `sink` | `Vec2` | Sink position (downstream). |
| `config` | `RiverConfig` | River configuration parameters. |

*Source: [crates/unshape-procgen/src/network.rs:480](crates/unshape-procgen/src/network.rs#L480)*

### `GenerateRoadNetworkGrid`

Operation to generate a grid-based road network.

`apply(&u64) -> RoadNetwork`

| Field | Type | Description |
|-------|------|-------------|
| `bounds_min` | `Vec2` | Minimum corner of the bounds. |
| `bounds_max` | `Vec2` | Maximum corner of the bounds. |
| `spacing` | `f32` | Grid spacing between roads. |

*Source: [crates/unshape-procgen/src/network.rs:768](crates/unshape-procgen/src/network.rs#L768)*

### `GenerateRoadNetworkHierarchical`

Operation to generate a hierarchical road network with main roads and side streets.

`apply(&u64) -> RoadNetwork`

| Field | Type | Description |
|-------|------|-------------|
| `bounds_min` | `Vec2` | Minimum corner of the bounds. |
| `bounds_max` | `Vec2` | Maximum corner of the bounds. |
| `density` | `f32` | Road density factor. |

*Source: [crates/unshape-procgen/src/network.rs:800](crates/unshape-procgen/src/network.rs#L800)*

---

## unshape-rd

### `ApplyPreset`

Applies a preset to the simulation.

`apply(&ReactionDiffusion) -> ReactionDiffusion`

| Field | Type | Description |
|-------|------|-------------|
| `preset` | `GrayScottPreset` | The preset to apply. |

*Source: [crates/unshape-rd/src/lib.rs:655](crates/unshape-rd/src/lib.rs#L655)*

### `Clear`

Clears the simulation to initial state (U=1, V=0).

`apply(&ReactionDiffusion) -> ReactionDiffusion`

*Source: [crates/unshape-rd/src/lib.rs:586](crates/unshape-rd/src/lib.rs#L586)*

### `GrayScottParams`

Gray-Scott simulation parameters.

`apply(&mutReactionDiffusion) -> ()`

| Field | Type | Description |
|-------|------|-------------|
| `du` | `f32` | Diffusion rate of chemical U. |
| `dv` | `f32` | Diffusion rate of chemical V. |
| `feed` | `f32` | Feed rate. |
| `kill` | `f32` | Kill rate. |
| `dt` | `f32` | Time step. |

*Source: [crates/unshape-rd/src/lib.rs:415](crates/unshape-rd/src/lib.rs#L415)*

### `SeedCircle`

Adds a circular seed of chemical V.

`apply(&ReactionDiffusion) -> ReactionDiffusion`

| Field | Type | Description |
|-------|------|-------------|
| `cx` | `usize` | Center X coordinate. |
| `cy` | `usize` | Center Y coordinate. |
| `radius` | `usize` | Radius of the seed circle. |

*Source: [crates/unshape-rd/src/lib.rs:496](crates/unshape-rd/src/lib.rs#L496)*

### `SeedRandom`

Adds random seeds across the grid.

`apply(&ReactionDiffusion) -> ReactionDiffusion`

| Field | Type | Description |
|-------|------|-------------|
| `count` | `usize` | Number of random seeds to add. |
| `radius` | `usize` | Radius of each seed circle. |
| `seed` | `u64` | Random seed for reproducibility. |

*Source: [crates/unshape-rd/src/lib.rs:554](crates/unshape-rd/src/lib.rs#L554)*

### `SeedRect`

Adds a rectangular seed of chemical V.

`apply(&ReactionDiffusion) -> ReactionDiffusion`

| Field | Type | Description |
|-------|------|-------------|
| `x1` | `usize` | Left X coordinate. |
| `y1` | `usize` | Top Y coordinate. |
| `x2` | `usize` | Right X coordinate. |
| `y2` | `usize` | Bottom Y coordinate. |

*Source: [crates/unshape-rd/src/lib.rs:524](crates/unshape-rd/src/lib.rs#L524)*

### `SetFeed`

Sets the feed rate parameter.

`apply(&ReactionDiffusion) -> ReactionDiffusion`

| Field | Type | Description |
|-------|------|-------------|
| `feed` | `f32` | The feed rate to set. |

*Source: [crates/unshape-rd/src/lib.rs:607](crates/unshape-rd/src/lib.rs#L607)*

### `SetKill`

Sets the kill rate parameter.

`apply(&ReactionDiffusion) -> ReactionDiffusion`

| Field | Type | Description |
|-------|------|-------------|
| `kill` | `f32` | The kill rate to set. |

*Source: [crates/unshape-rd/src/lib.rs:631](crates/unshape-rd/src/lib.rs#L631)*

### `Step`

Advances the simulation by a number of steps.

`apply(&ReactionDiffusion) -> ReactionDiffusion`

| Field | Type | Description |
|-------|------|-------------|
| `count` | `usize` | Number of simulation steps to advance. |

*Source: [crates/unshape-rd/src/lib.rs:466](crates/unshape-rd/src/lib.rs#L466)*

---

## unshape-rig

### `Gait`

Configuration for a walking gait.

`apply(()) -> Gait`

| Field | Type | Description |
|-------|------|-------------|
| `stride_length` | `f32` | Length of a single stride. |
| `step_height` | `f32` | Maximum height of foot during step. |
| `cycle_duration` | `f32` | Duration of a full gait cycle in seconds. |
| `stance_ratio` | `f32` | Fraction of cycle spent with foot planted (0.0-1.0). |
| `body_bob` | `f32` | Vertical bob amount for body. |
| `body_sway` | `f32` | Lateral sway amount for body. |
| `lean_amount` | `f32` | Forward lean when moving. |

*Source: [crates/unshape-rig/src/locomotion.rs:18](crates/unshape-rig/src/locomotion.rs#L18)*

### `Ik`

Configuration for IK solving.

`apply(()) -> Ik`

| Field | Type | Description |
|-------|------|-------------|
| `max_iterations` | `u32` | Maximum iterations. |
| `tolerance` | `f32` | Distance threshold for success. |

*Source: [crates/unshape-rig/src/ik.rs:35](crates/unshape-rig/src/ik.rs#L35)*

### `MotionMatching`

Configuration for motion matching.

`apply(()) -> MotionMatching`

| Field | Type | Description |
|-------|------|-------------|
| `position_weight` | `f32` | Weight for bone position matching. |
| `velocity_weight` | `f32` | Weight for bone velocity matching. |
| `trajectory_weight` | `f32` | Weight for trajectory matching. |
| `facing_weight` | `f32` | Weight for facing direction matching. |
| `trajectory_samples` | `usize` | Number of future trajectory samples. |
| `trajectory_sample_time` | `f32` | Time between trajectory samples. |
| `min_transition_time` | `f32` | Minimum time between transitions. |
| `blend_time` | `f32` | Blend time for transitions. |

*Source: [crates/unshape-rig/src/motion_matching.rs:19](crates/unshape-rig/src/motion_matching.rs#L19)*

### `Secondary`

Configuration for secondary motion effects.

`apply(()) -> Secondary`

| Field | Type | Description |
|-------|------|-------------|
| `stiffness` | `f32` | Spring stiffness (higher = snappier return to rest). |
| `damping` | `f32` | Damping coefficient (higher = less oscillation). |
| `mass` | `f32` | Mass of the simulated point. |
| `gravity` | `Vec3` | Gravity vector. |
| `max_displacement` | `f32` | Maximum displacement from rest position. |
| `enable_collision` | `bool` | Whether to enable collision with parent bone. |

*Source: [crates/unshape-rig/src/secondary.rs:45](crates/unshape-rig/src/secondary.rs#L45)*

### `SolveCcd`

CCD (Cyclic Coordinate Descent) IK solver as an op struct.

`apply(&Skeleton) -> IkResult`

| Field | Type | Description |
|-------|------|-------------|
| `config` | `IkConfig` | IK solver configuration. |

*Source: [crates/unshape-rig/src/ik.rs:396](crates/unshape-rig/src/ik.rs#L396)*

### `SolveFabrik`

FABRIK (Forward And Backward Reaching Inverse Kinematics) solver as an op struct.

`apply(&Skeleton) -> IkResult`

| Field | Type | Description |
|-------|------|-------------|
| `config` | `IkConfig` | IK solver configuration. |

*Source: [crates/unshape-rig/src/ik.rs:450](crates/unshape-rig/src/ik.rs#L450)*

---

## unshape-scatter

### `Scatter`

Scatters instances randomly within a box volume.

`apply(()) -> Vec<Instance>`

| Field | Type | Description |
|-------|------|-------------|
| `min` | `Vec3` | Minimum bounds of the scatter volume. |
| `max` | `Vec3` | Maximum bounds of the scatter volume. |
| `count` | `usize` | Number of instances to generate. |
| `seed` | `u64` | Random seed. |
| `min_scale` | `f32` | Minimum scale (for random scaling). |
| `max_scale` | `f32` | Maximum scale (for random scaling). |
| `random_rotation` | `bool` | Whether to apply random rotation. |
| `align_axis` | `Option<Vec3>` | Alignment axis for oriented scatter (e.g., up vector). |

*Source: [crates/unshape-scatter/src/lib.rs:97](crates/unshape-scatter/src/lib.rs#L97)*

### `Stagger`

Configuration for stagger timing generation.

`apply(&[Instance]) -> Vec<f32>`

| Field | Type | Description |
|-------|------|-------------|
| `delay` | `f32` | Delay between each instance (in seconds or frames). |
| `total_duration` | `Option<f32>` | Total duration to spread instances over (overrides delay if set). |
| `pattern` | `StaggerPattern` | Pattern for distributing delays. |
| `easing` | `f32` | Easing function for delay distribution (0 = linear, positive = ease-in, negative = ease-out). |

*Source: [crates/unshape-scatter/src/lib.rs:517](crates/unshape-scatter/src/lib.rs#L517)*

---

## unshape-space-colonization

### `SpaceColonizationParams`

Configuration for the space colonization algorithm.

`apply(()) -> SpaceColonization`

| Field | Type | Description |
|-------|------|-------------|
| `attraction_distance` | `f32` | Distance within which an attraction point influences a node. |
| `kill_distance` | `f32` | Distance at which an attraction point is removed (colonized). |
| `segment_length` | `f32` | Length of new segments when branching. |
| `tropism` | `Vec3` | Tropism direction and strength (e.g., gravity, light). |
| `tropism_strength` | `f32` | Tropism strength (0.0 = none, 1.0 = strong). |
| `smoothing` | `f32` | Smoothing factor for growth direction (0.0 = sharp, 1.0 = smooth). |
| `max_iterations` | `usize` | Maximum number of iterations. |

*Source: [crates/unshape-space-colonization/src/lib.rs:59](crates/unshape-space-colonization/src/lib.rs#L59)*

---

## unshape-spring

### `SpringConfig`

Configuration for a spring constraint.

`apply(()) -> SpringConfig`

| Field | Type | Description |
|-------|------|-------------|
| `rest_length` | `f32` | Rest length of the spring. |
| `stiffness` | `f32` | Stiffness (0-1, higher = stiffer). |
| `damping` | `f32` | Damping (0-1, reduces oscillation). |

*Source: [crates/unshape-spring/src/lib.rs:91](crates/unshape-spring/src/lib.rs#L91)*

---

## unshape-vector

### `PressureStrokeRender`

Operation for pressure-sensitive stroke rendering.

`apply(&PressureStroke) -> Path`

| Field | Type | Description |
|-------|------|-------------|
| `min_width` | `f32` | Minimum stroke width (at pressure 0). |
| `max_width` | `f32` | Maximum stroke width (at pressure 1). |
| `cap` | `CapStyle` | Cap style for stroke ends. |
| `join` | `JoinStyle` | Join style for corners. |
| `miter_limit` | `f32` | Miter limit. |

*Source: [crates/unshape-vector/src/stroke.rs:973](crates/unshape-vector/src/stroke.rs#L973)*

### `Stroke`

Operation for converting strokes to filled path outlines.

`apply(&Path) -> Path`

| Field | Type | Description |
|-------|------|-------------|
| `width` | `f32` | Width of the stroke. |
| `cap` | `CapStyle` | Cap style for line ends. |
| `join` | `JoinStyle` | Join style for corners. |
| `miter_limit` | `f32` | Miter limit (for miter joins). |

*Source: [crates/unshape-vector/src/stroke.rs:62](crates/unshape-vector/src/stroke.rs#L62)*

### `Trim`

Operation for trimming a path to a portion of its length.

`apply(&Path) -> Path`

| Field | Type | Description |
|-------|------|-------------|
| `start` | `f32` | Start position (0.0 to 1.0). |
| `end` | `f32` | End position (0.0 to 1.0). |

*Source: [crates/unshape-vector/src/stroke.rs:1203](crates/unshape-vector/src/stroke.rs#L1203)*

---

## unshape-voxel

### `Dilate`

Operation to dilate a binary voxel grid (grows solid regions).

`apply(&BinaryVoxelGrid) -> BinaryVoxelGrid`

| Field | Type | Description |
|-------|------|-------------|
| `iterations` | `u32` | Number of dilation iterations. |

*Source: [crates/unshape-voxel/src/lib.rs:448](crates/unshape-voxel/src/lib.rs#L448)*

### `Erode`

Operation to erode a binary voxel grid (shrinks solid regions).

`apply(&BinaryVoxelGrid) -> BinaryVoxelGrid`

| Field | Type | Description |
|-------|------|-------------|
| `iterations` | `u32` | Number of erosion iterations. |

*Source: [crates/unshape-voxel/src/lib.rs:512](crates/unshape-voxel/src/lib.rs#L512)*

### `FillBox`

Operation to fill a box region in a binary voxel grid.

`apply(&BinaryVoxelGrid) -> BinaryVoxelGrid`

| Field | Type | Description |
|-------|------|-------------|
| `min` | `UVec3` | Minimum corner of the box (inclusive). |
| `max` | `UVec3` | Maximum corner of the box (exclusive). |
| `value` | `bool` | Value to fill with (true = solid, false = empty). |

*Source: [crates/unshape-voxel/src/lib.rs:376](crates/unshape-voxel/src/lib.rs#L376)*

### `FillSphere`

Operation to fill a sphere in a binary voxel grid.

`apply(&BinaryVoxelGrid) -> BinaryVoxelGrid`

| Field | Type | Description |
|-------|------|-------------|
| `center` | `Vec3` | Center of the sphere in voxel coordinates. |
| `radius` | `f32` | Radius of the sphere in voxel units. |
| `value` | `bool` | Value to fill with (true = solid, false = empty). |

*Source: [crates/unshape-voxel/src/lib.rs:324](crates/unshape-voxel/src/lib.rs#L324)*

### `SparseVoxelsToMesh`

Operation to generate a mesh from sparse voxels.

`apply(&SparseBinaryVoxels) -> Mesh`

| Field | Type | Description |
|-------|------|-------------|
| `voxel_size` | `f32` | Size of each voxel in world units. |

*Source: [crates/unshape-voxel/src/lib.rs:730](crates/unshape-voxel/src/lib.rs#L730)*

### `VoxelsToMesh`

Operation to generate a simple blocky mesh from a binary voxel grid.

`apply(&BinaryVoxelGrid) -> Mesh`

| Field | Type | Description |
|-------|------|-------------|
| `voxel_size` | `f32` | Size of each voxel in world units. |

*Source: [crates/unshape-voxel/src/lib.rs:596](crates/unshape-voxel/src/lib.rs#L596)*

