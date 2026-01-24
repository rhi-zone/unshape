//! Room acoustics simulation.
//!
//! Provides physical room modeling with RT60 calculation, room modes,
//! early reflections via image-source method, and ray tracing.

use unshape_core::glam::Vec3;

/// Room geometry (rectangular box).
#[derive(Debug, Clone)]
pub struct RoomGeometry {
    /// Width (X dimension) in meters.
    pub width: f32,
    /// Height (Y dimension) in meters.
    pub height: f32,
    /// Depth (Z dimension) in meters.
    pub depth: f32,
}

impl RoomGeometry {
    /// Creates a new room.
    pub fn new(width: f32, height: f32, depth: f32) -> Self {
        Self {
            width,
            height,
            depth,
        }
    }

    /// Creates a cubic room.
    pub fn cube(size: f32) -> Self {
        Self::new(size, size, size)
    }

    /// Returns the room volume in cubic meters.
    pub fn volume(&self) -> f32 {
        self.width * self.height * self.depth
    }

    /// Returns the total surface area in square meters.
    pub fn surface_area(&self) -> f32 {
        2.0 * (self.width * self.height + self.height * self.depth + self.width * self.depth)
    }

    /// Returns the mean free path (average distance between reflections).
    pub fn mean_free_path(&self) -> f32 {
        4.0 * self.volume() / self.surface_area()
    }

    /// Preset: small room (bedroom).
    pub fn small_room() -> Self {
        Self::new(4.0, 2.5, 3.0)
    }

    /// Preset: medium room (living room).
    pub fn medium_room() -> Self {
        Self::new(6.0, 2.8, 5.0)
    }

    /// Preset: large room (concert hall).
    pub fn large_hall() -> Self {
        Self::new(30.0, 15.0, 50.0)
    }

    /// Preset: studio control room.
    pub fn studio() -> Self {
        Self::new(5.0, 3.0, 4.0)
    }
}

impl Default for RoomGeometry {
    fn default() -> Self {
        Self::medium_room()
    }
}

/// Acoustic material with frequency-dependent absorption.
#[derive(Debug, Clone)]
pub struct RoomMaterial {
    /// Material name.
    pub name: String,
    /// Absorption coefficients at [125, 250, 500, 1000, 2000, 4000] Hz.
    pub absorption: [f32; 6],
}

impl RoomMaterial {
    /// Creates a new material.
    pub fn new(name: &str, absorption: [f32; 6]) -> Self {
        Self {
            name: name.to_string(),
            absorption,
        }
    }

    /// Returns the average absorption coefficient.
    pub fn average_absorption(&self) -> f32 {
        self.absorption.iter().sum::<f32>() / 6.0
    }

    /// Preset: concrete (very reflective).
    pub fn concrete() -> Self {
        Self::new("concrete", [0.01, 0.01, 0.02, 0.02, 0.02, 0.03])
    }

    /// Preset: plaster on brick.
    pub fn plaster() -> Self {
        Self::new("plaster", [0.01, 0.02, 0.02, 0.03, 0.04, 0.05])
    }

    /// Preset: wood panel.
    pub fn wood() -> Self {
        Self::new("wood", [0.10, 0.08, 0.08, 0.08, 0.08, 0.08])
    }

    /// Preset: glass window.
    pub fn glass() -> Self {
        Self::new("glass", [0.18, 0.06, 0.04, 0.03, 0.02, 0.02])
    }

    /// Preset: heavy carpet.
    pub fn carpet() -> Self {
        Self::new("carpet", [0.02, 0.06, 0.15, 0.40, 0.60, 0.60])
    }

    /// Preset: acoustic foam.
    pub fn acoustic_foam() -> Self {
        Self::new("acoustic_foam", [0.08, 0.25, 0.65, 0.90, 0.95, 0.95])
    }

    /// Preset: heavy curtain.
    pub fn curtain() -> Self {
        Self::new("curtain", [0.07, 0.30, 0.50, 0.70, 0.70, 0.65])
    }

    /// Preset: audience (occupied seats).
    pub fn audience() -> Self {
        Self::new("audience", [0.25, 0.35, 0.42, 0.46, 0.50, 0.50])
    }
}

impl Default for RoomMaterial {
    fn default() -> Self {
        Self::plaster()
    }
}

/// Surface configuration for a room.
#[derive(Debug, Clone)]
pub struct RoomSurfaces {
    /// Floor material and area fraction.
    pub floor: (RoomMaterial, f32),
    /// Ceiling material.
    pub ceiling: RoomMaterial,
    /// Wall material.
    pub walls: RoomMaterial,
}

impl RoomSurfaces {
    /// Creates uniform surfaces.
    pub fn uniform(material: RoomMaterial) -> Self {
        Self {
            floor: (material.clone(), 1.0),
            ceiling: material.clone(),
            walls: material,
        }
    }

    /// Preset: typical living room.
    pub fn living_room() -> Self {
        Self {
            floor: (RoomMaterial::carpet(), 0.8),
            ceiling: RoomMaterial::plaster(),
            walls: RoomMaterial::plaster(),
        }
    }

    /// Preset: studio with treatment.
    pub fn treated_studio() -> Self {
        Self {
            floor: (RoomMaterial::carpet(), 1.0),
            ceiling: RoomMaterial::acoustic_foam(),
            walls: RoomMaterial::acoustic_foam(),
        }
    }

    /// Preset: concert hall.
    pub fn concert_hall() -> Self {
        Self {
            floor: (RoomMaterial::audience(), 0.7),
            ceiling: RoomMaterial::wood(),
            walls: RoomMaterial::plaster(),
        }
    }

    /// Returns average absorption for the room.
    pub fn average_absorption(&self, geometry: &RoomGeometry) -> f32 {
        let floor_area = geometry.width * geometry.depth;
        let ceiling_area = floor_area;
        let wall_area = 2.0 * (geometry.width * geometry.height + geometry.depth * geometry.height);
        let total_area = floor_area + ceiling_area + wall_area;

        let floor_abs = self.floor.0.average_absorption() * self.floor.1;
        let ceiling_abs = self.ceiling.average_absorption();
        let wall_abs = self.walls.average_absorption();

        (floor_area * floor_abs + ceiling_area * ceiling_abs + wall_area * wall_abs) / total_area
    }
}

impl Default for RoomSurfaces {
    fn default() -> Self {
        Self::living_room()
    }
}

/// Calculates RT60 (reverberation time to decay by 60dB) using Sabine's formula.
///
/// # Arguments
/// * `geometry` - Room dimensions
/// * `surfaces` - Surface materials
///
/// # Returns
/// RT60 in seconds for each frequency band [125, 250, 500, 1000, 2000, 4000] Hz
pub fn calculate_rt60_sabine(geometry: &RoomGeometry, surfaces: &RoomSurfaces) -> [f32; 6] {
    let volume = geometry.volume();
    let floor_area = geometry.width * geometry.depth;
    let ceiling_area = floor_area;
    let wall_area = 2.0 * (geometry.width * geometry.height + geometry.depth * geometry.height);

    let mut rt60 = [0.0; 6];

    for i in 0..6 {
        // Total absorption at this frequency
        let floor_abs = surfaces.floor.0.absorption[i] * surfaces.floor.1 * floor_area;
        let ceiling_abs = surfaces.ceiling.absorption[i] * ceiling_area;
        let wall_abs = surfaces.walls.absorption[i] * wall_area;
        let total_abs = floor_abs + ceiling_abs + wall_abs;

        // Sabine formula: RT60 = 0.161 * V / A
        rt60[i] = 0.161 * volume / total_abs.max(0.001);
    }

    rt60
}

/// Calculates RT60 using Eyring's formula (more accurate for high absorption).
pub fn calculate_rt60_eyring(geometry: &RoomGeometry, surfaces: &RoomSurfaces) -> [f32; 6] {
    let volume = geometry.volume();
    let surface_area = geometry.surface_area();
    let floor_area = geometry.width * geometry.depth;
    let ceiling_area = floor_area;
    let wall_area = 2.0 * (geometry.width * geometry.height + geometry.depth * geometry.height);

    let mut rt60 = [0.0; 6];

    for i in 0..6 {
        // Average absorption coefficient at this frequency
        let floor_abs = surfaces.floor.0.absorption[i] * surfaces.floor.1;
        let ceiling_abs = surfaces.ceiling.absorption[i];
        let wall_abs = surfaces.walls.absorption[i];

        let avg_abs = (floor_area * floor_abs + ceiling_area * ceiling_abs + wall_area * wall_abs)
            / surface_area;

        // Eyring formula: RT60 = 0.161 * V / (-S * ln(1 - α))
        let absorption_term = if avg_abs >= 0.99 {
            100.0 // Avoid log(0)
        } else {
            -(1.0 - avg_abs).ln()
        };

        rt60[i] = 0.161 * volume / (surface_area * absorption_term).max(0.001);
    }

    rt60
}

/// Room modes (resonant frequencies) for a rectangular room.
#[derive(Debug, Clone)]
pub struct RoomModes {
    /// Axial modes (single dimension).
    pub axial: Vec<RoomMode>,
    /// Tangential modes (two dimensions).
    pub tangential: Vec<RoomMode>,
    /// Oblique modes (three dimensions).
    pub oblique: Vec<RoomMode>,
}

/// A single room mode.
#[derive(Debug, Clone)]
pub struct RoomMode {
    /// Frequency in Hz.
    pub frequency: f32,
    /// Mode indices (nx, ny, nz).
    pub indices: (u32, u32, u32),
    /// Relative amplitude (lower modes are stronger).
    pub amplitude: f32,
}

/// Calculates room modes up to a maximum frequency.
///
/// Room modes are the resonant frequencies determined by room dimensions.
/// They cause uneven bass response and coloration.
pub fn calculate_room_modes(geometry: &RoomGeometry, max_freq: f32) -> RoomModes {
    let speed_of_sound = 343.0; // m/s at 20°C
    let mut axial = Vec::new();
    let mut tangential = Vec::new();
    let mut oblique = Vec::new();

    // Calculate maximum mode indices
    let max_n = (2.0 * max_freq * geometry.width / speed_of_sound).ceil() as u32 + 1;

    for nx in 0..=max_n {
        for ny in 0..=max_n {
            for nz in 0..=max_n {
                if nx == 0 && ny == 0 && nz == 0 {
                    continue;
                }

                // Mode frequency: f = c/2 * sqrt((nx/Lx)^2 + (ny/Ly)^2 + (nz/Lz)^2)
                let fx = nx as f32 / geometry.width;
                let fy = ny as f32 / geometry.height;
                let fz = nz as f32 / geometry.depth;
                let freq = speed_of_sound / 2.0 * (fx * fx + fy * fy + fz * fz).sqrt();

                if freq > max_freq {
                    continue;
                }

                // Count non-zero indices
                let non_zero = [nx, ny, nz].iter().filter(|&&n| n > 0).count();

                // Amplitude decreases with mode number
                let mode_order = nx + ny + nz;
                let amplitude = 1.0 / (mode_order as f32 + 1.0);

                let mode = RoomMode {
                    frequency: freq,
                    indices: (nx, ny, nz),
                    amplitude,
                };

                match non_zero {
                    1 => axial.push(mode),
                    2 => tangential.push(mode),
                    3 => oblique.push(mode),
                    _ => {}
                }
            }
        }
    }

    // Sort by frequency
    axial.sort_by(|a, b| a.frequency.partial_cmp(&b.frequency).unwrap());
    tangential.sort_by(|a, b| a.frequency.partial_cmp(&b.frequency).unwrap());
    oblique.sort_by(|a, b| a.frequency.partial_cmp(&b.frequency).unwrap());

    RoomModes {
        axial,
        tangential,
        oblique,
    }
}

/// An early reflection from the image-source method.
#[derive(Debug, Clone)]
pub struct EarlyReflection {
    /// Delay time in seconds.
    pub delay: f32,
    /// Amplitude (after absorption losses).
    pub amplitude: f32,
    /// Image source position.
    pub source_position: Vec3,
    /// Reflection order (1 = first reflection, 2 = second, etc.).
    pub order: u32,
}

/// Calculates early reflections using the image-source method.
///
/// # Arguments
/// * `geometry` - Room dimensions
/// * `source` - Sound source position
/// * `listener` - Listener position
/// * `max_order` - Maximum reflection order (1-3 recommended)
/// * `absorption` - Wall absorption coefficient (0-1)
///
/// # Returns
/// List of early reflections sorted by delay time
pub fn calculate_early_reflections(
    geometry: &RoomGeometry,
    source: Vec3,
    listener: Vec3,
    max_order: u32,
    absorption: f32,
) -> Vec<EarlyReflection> {
    let speed_of_sound = 343.0;
    let mut reflections = Vec::new();

    // Direct sound
    let direct_distance = (source - listener).length();
    let direct_delay = direct_distance / speed_of_sound;

    // Generate image sources for each order
    fn generate_images(
        geometry: &RoomGeometry,
        source: Vec3,
        listener: Vec3,
        order: u32,
        max_order: u32,
        absorption: f32,
        reflections: &mut Vec<EarlyReflection>,
        speed_of_sound: f32,
        direct_delay: f32,
    ) {
        if order > max_order {
            return;
        }

        let reflection_coeff = (1.0 - absorption).powi(order as i32);

        // Iterate through all possible reflection combinations
        for ix in -(order as i32)..=(order as i32) {
            for iy in -(order as i32)..=(order as i32) {
                for iz in -(order as i32)..=(order as i32) {
                    // Check if this is exactly the right order
                    let this_order = ix.unsigned_abs() + iy.unsigned_abs() + iz.unsigned_abs();
                    if this_order != order {
                        continue;
                    }

                    // Calculate image source position
                    let image = Vec3::new(
                        mirror_position(source.x, geometry.width, ix),
                        mirror_position(source.y, geometry.height, iy),
                        mirror_position(source.z, geometry.depth, iz),
                    );

                    let distance = (image - listener).length();
                    let delay = distance / speed_of_sound;
                    let amplitude = reflection_coeff / distance.max(0.1);

                    // Relative delay to direct sound
                    let relative_delay = delay - direct_delay;

                    if relative_delay > 0.0 {
                        reflections.push(EarlyReflection {
                            delay: relative_delay,
                            amplitude,
                            source_position: image,
                            order,
                        });
                    }
                }
            }
        }

        generate_images(
            geometry,
            source,
            listener,
            order + 1,
            max_order,
            absorption,
            reflections,
            speed_of_sound,
            direct_delay,
        );
    }

    generate_images(
        geometry,
        source,
        listener,
        1,
        max_order,
        absorption,
        &mut reflections,
        speed_of_sound,
        direct_delay,
    );

    // Sort by delay
    reflections.sort_by(|a, b| a.delay.partial_cmp(&b.delay).unwrap());

    reflections
}

/// Mirrors a position across room boundaries.
fn mirror_position(pos: f32, size: f32, reflection_index: i32) -> f32 {
    if reflection_index == 0 {
        return pos;
    }

    if reflection_index > 0 {
        // Positive reflections: alternate between (size - pos) and pos
        let n = reflection_index as f32;
        if reflection_index % 2 == 1 {
            n * size + (size - pos)
        } else {
            n * size + pos
        }
    } else {
        // Negative reflections
        let n = (-reflection_index) as f32;
        if (-reflection_index) % 2 == 1 {
            -n * size + (size - pos)
        } else {
            -n * size + pos
        }
    }
}

/// Complete room acoustics simulation.
#[derive(Debug, Clone)]
pub struct RoomAcoustics {
    /// Room geometry.
    pub geometry: RoomGeometry,
    /// Surface materials.
    pub surfaces: RoomSurfaces,
    /// Calculated RT60 values.
    pub rt60: [f32; 6],
    /// Room modes.
    pub modes: RoomModes,
}

impl RoomAcoustics {
    /// Creates a new room acoustics simulation.
    pub fn new(geometry: RoomGeometry, surfaces: RoomSurfaces) -> Self {
        let rt60 = calculate_rt60_sabine(&geometry, &surfaces);
        let modes = calculate_room_modes(&geometry, 300.0);

        Self {
            geometry,
            surfaces,
            rt60,
            modes,
        }
    }

    /// Returns the average RT60 across all frequency bands.
    pub fn average_rt60(&self) -> f32 {
        self.rt60.iter().sum::<f32>() / 6.0
    }

    /// Calculates early reflections for a source/listener pair.
    pub fn early_reflections(
        &self,
        source: Vec3,
        listener: Vec3,
        max_order: u32,
    ) -> Vec<EarlyReflection> {
        let avg_absorption = self.surfaces.average_absorption(&self.geometry);
        calculate_early_reflections(&self.geometry, source, listener, max_order, avg_absorption)
    }

    /// Generates an impulse response for this room.
    pub fn generate_ir(
        &self,
        source: Vec3,
        listener: Vec3,
        duration: f32,
        sample_rate: f32,
    ) -> Vec<f32> {
        let samples = (duration * sample_rate) as usize;
        let mut ir = vec![0.0; samples];

        // Direct sound
        let direct_distance = (source - listener).length();
        let direct_delay = direct_distance / 343.0;
        let direct_idx = (direct_delay * sample_rate) as usize;
        if direct_idx < samples {
            ir[direct_idx] = 1.0 / direct_distance.max(0.1);
        }

        // Early reflections
        let reflections = self.early_reflections(source, listener, 3);
        for refl in &reflections {
            let idx = ((direct_delay + refl.delay) * sample_rate) as usize;
            if idx < samples {
                ir[idx] += refl.amplitude;
            }
        }

        // Diffuse tail
        let avg_rt60 = self.average_rt60();
        let decay_rate = 6.91 / avg_rt60; // ln(1000) ≈ 6.91

        // Start diffuse tail after early reflections
        let tail_start = reflections.last().map(|r| r.delay + 0.02).unwrap_or(0.05);
        let tail_start_idx = ((direct_delay + tail_start) * sample_rate) as usize;

        let mut rng_state = 12345u32;
        for i in tail_start_idx..samples {
            let t = (i - tail_start_idx) as f32 / sample_rate;
            let decay = (-decay_rate * t).exp();

            // Simple noise
            rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
            let noise = (rng_state as f32 / u32::MAX as f32) * 2.0 - 1.0;

            ir[i] += noise * decay * 0.02;
        }

        // Normalize
        let max = ir.iter().map(|&x| x.abs()).fold(0.0f32, f32::max);
        if max > 0.0 {
            for sample in &mut ir {
                *sample /= max;
            }
        }

        ir
    }

    /// Preset: small untreated room.
    pub fn small_room() -> Self {
        Self::new(RoomGeometry::small_room(), RoomSurfaces::living_room())
    }

    /// Preset: treated studio.
    pub fn studio() -> Self {
        Self::new(RoomGeometry::studio(), RoomSurfaces::treated_studio())
    }

    /// Preset: concert hall.
    pub fn concert_hall() -> Self {
        Self::new(RoomGeometry::large_hall(), RoomSurfaces::concert_hall())
    }
}

impl Default for RoomAcoustics {
    fn default() -> Self {
        Self::small_room()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_room_geometry() {
        let room = RoomGeometry::new(5.0, 3.0, 4.0);
        assert_eq!(room.volume(), 60.0);
        assert_eq!(room.surface_area(), 94.0);
    }

    #[test]
    fn test_room_geometry_presets() {
        let small = RoomGeometry::small_room();
        let large = RoomGeometry::large_hall();

        assert!(small.volume() < large.volume());
    }

    #[test]
    fn test_room_material_absorption() {
        let concrete = RoomMaterial::concrete();
        let foam = RoomMaterial::acoustic_foam();

        // Foam should absorb more
        assert!(foam.average_absorption() > concrete.average_absorption());
    }

    #[test]
    fn test_rt60_sabine() {
        let geometry = RoomGeometry::medium_room();
        let surfaces = RoomSurfaces::living_room();

        let rt60 = calculate_rt60_sabine(&geometry, &surfaces);

        // RT60 should be positive
        for &t in &rt60 {
            assert!(t > 0.0);
        }

        // Low frequencies should have longer RT60 (less absorption)
        assert!(rt60[0] > rt60[5]);
    }

    #[test]
    fn test_rt60_eyring() {
        let geometry = RoomGeometry::studio();
        let surfaces = RoomSurfaces::treated_studio();

        let rt60 = calculate_rt60_eyring(&geometry, &surfaces);

        // Should be shorter than Sabine for high absorption
        let sabine_rt60 = calculate_rt60_sabine(&geometry, &surfaces);

        // Eyring typically gives shorter RT60 for high absorption rooms
        for i in 0..6 {
            assert!(rt60[i] > 0.0);
            // Allow some tolerance
            assert!(rt60[i] <= sabine_rt60[i] * 1.1);
        }
    }

    #[test]
    fn test_room_modes() {
        let geometry = RoomGeometry::new(5.0, 3.0, 4.0);
        let modes = calculate_room_modes(&geometry, 200.0);

        // Should have modes in all categories
        assert!(!modes.axial.is_empty());
        assert!(!modes.tangential.is_empty());
        assert!(!modes.oblique.is_empty());

        // First axial mode should be lowest
        let first_axial = &modes.axial[0];
        assert!(first_axial.frequency < 50.0);
    }

    #[test]
    fn test_room_modes_sorted() {
        let geometry = RoomGeometry::medium_room();
        let modes = calculate_room_modes(&geometry, 150.0);

        // Check axial modes are sorted
        for window in modes.axial.windows(2) {
            assert!(window[0].frequency <= window[1].frequency);
        }
    }

    #[test]
    fn test_early_reflections() {
        let geometry = RoomGeometry::new(5.0, 3.0, 4.0);
        let source = Vec3::new(1.0, 1.5, 1.0);
        let listener = Vec3::new(4.0, 1.5, 3.0);

        let reflections = calculate_early_reflections(&geometry, source, listener, 2, 0.2);

        // Should have some reflections
        assert!(!reflections.is_empty());

        // All delays should be positive
        for refl in &reflections {
            assert!(refl.delay > 0.0);
        }

        // Should be sorted by delay
        for window in reflections.windows(2) {
            assert!(window[0].delay <= window[1].delay);
        }
    }

    #[test]
    fn test_early_reflections_order() {
        let geometry = RoomGeometry::medium_room();
        let source = Vec3::new(2.0, 1.5, 2.0);
        let listener = Vec3::new(4.0, 1.5, 3.0);

        let order1 = calculate_early_reflections(&geometry, source, listener, 1, 0.2);
        let order2 = calculate_early_reflections(&geometry, source, listener, 2, 0.2);

        // Higher order should have more reflections
        assert!(order2.len() > order1.len());
    }

    #[test]
    fn test_room_acoustics() {
        let room = RoomAcoustics::small_room();

        // Check RT60 is reasonable (0.1 - 5 seconds)
        let avg_rt60 = room.average_rt60();
        assert!(avg_rt60 > 0.1 && avg_rt60 < 5.0);

        // Check modes exist
        assert!(!room.modes.axial.is_empty());
    }

    #[test]
    fn test_room_acoustics_presets() {
        let small = RoomAcoustics::small_room();
        let hall = RoomAcoustics::concert_hall();

        // Hall should have longer RT60
        assert!(hall.average_rt60() > small.average_rt60());
    }

    #[test]
    fn test_generate_ir() {
        let room = RoomAcoustics::studio();
        let source = Vec3::new(1.0, 1.5, 1.0);
        let listener = Vec3::new(3.0, 1.5, 2.0);

        let ir = room.generate_ir(source, listener, 0.5, 44100.0);

        // Should have expected length
        assert_eq!(ir.len(), 22050);

        // Should be normalized
        let max = ir.iter().map(|&x| x.abs()).fold(0.0f32, f32::max);
        assert!((max - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_mirror_position() {
        // Test reflection at 0
        let pos = mirror_position(1.0, 5.0, 0);
        assert_eq!(pos, 1.0);

        // First positive reflection
        let pos = mirror_position(1.0, 5.0, 1);
        assert_eq!(pos, 9.0); // 5 + (5-1) = 9

        // First negative reflection (reflects across near wall at x=0)
        let pos = mirror_position(1.0, 5.0, -1);
        assert_eq!(pos, -1.0); // -n*size + (size-pos) = -5 + 4 = -1
    }

    #[test]
    fn test_room_surfaces_absorption() {
        let geometry = RoomGeometry::medium_room();
        let treated = RoomSurfaces::treated_studio();
        let untreated = RoomSurfaces::uniform(RoomMaterial::concrete());

        let treated_abs = treated.average_absorption(&geometry);
        let untreated_abs = untreated.average_absorption(&geometry);

        // Treated room should have higher absorption
        assert!(treated_abs > untreated_abs);
    }
}
