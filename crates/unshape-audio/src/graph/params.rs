use std::sync::atomic::{AtomicU32, Ordering};

/// An atomic f32 for lock-free parameter updates.
///
/// Use this for parameters that need to be updated from a UI thread
/// while audio is processing. Updates are wait-free and won't cause
/// audio glitches.
///
/// # Example
///
/// ```
/// use unshape_audio::graph::AtomicF32;
///
/// let param = AtomicF32::new(0.5);
///
/// // Audio thread reads
/// let value = param.get();
///
/// // UI thread writes (lock-free)
/// param.set(0.75);
/// ```
#[derive(Debug)]
pub struct AtomicF32(AtomicU32);

impl AtomicF32 {
    /// Creates a new atomic f32 with the given initial value.
    pub const fn new(value: f32) -> Self {
        Self(AtomicU32::new(value.to_bits()))
    }

    /// Gets the current value (relaxed ordering, suitable for audio).
    #[inline]
    pub fn get(&self) -> f32 {
        f32::from_bits(self.0.load(Ordering::Relaxed))
    }

    /// Sets the value (relaxed ordering, suitable for UI updates).
    #[inline]
    pub fn set(&self, value: f32) {
        self.0.store(value.to_bits(), Ordering::Relaxed);
    }

    /// Gets the current value with acquire ordering.
    #[inline]
    pub fn get_acquire(&self) -> f32 {
        f32::from_bits(self.0.load(Ordering::Acquire))
    }

    /// Sets the value with release ordering.
    #[inline]
    pub fn set_release(&self, value: f32) {
        self.0.store(value.to_bits(), Ordering::Release);
    }
}

impl Default for AtomicF32 {
    fn default() -> Self {
        Self::new(0.0)
    }
}

impl Clone for AtomicF32 {
    fn clone(&self) -> Self {
        Self::new(self.get())
    }
}

/// A set of named atomic parameters for lock-free audio control.
///
/// This provides a simple way to expose multiple parameters that can be
/// safely updated from any thread while audio is processing.
///
/// # Example
///
/// ```
/// use unshape_audio::graph::AtomicParams;
///
/// let mut params = AtomicParams::new();
/// params.add("cutoff", 1000.0);
/// params.add("resonance", 0.5);
///
/// // Audio thread reads
/// let cutoff = params.get("cutoff").unwrap();
///
/// // UI thread writes (lock-free)
/// params.set("cutoff", 2000.0);
/// ```
#[derive(Debug, Clone, Default)]
pub struct AtomicParams {
    names: Vec<&'static str>,
    values: Vec<AtomicF32>,
}

impl AtomicParams {
    /// Creates a new empty parameter set.
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a parameter with the given name and initial value.
    pub fn add(&mut self, name: &'static str, value: f32) {
        self.names.push(name);
        self.values.push(AtomicF32::new(value));
    }

    /// Gets a parameter value by name.
    pub fn get(&self, name: &str) -> Option<f32> {
        self.names
            .iter()
            .position(|&n| n == name)
            .map(|i| self.values[i].get())
    }

    /// Sets a parameter value by name.
    pub fn set(&self, name: &str, value: f32) -> bool {
        if let Some(i) = self.names.iter().position(|&n| n == name) {
            self.values[i].set(value);
            true
        } else {
            false
        }
    }

    /// Gets a parameter value by index.
    pub fn get_index(&self, index: usize) -> Option<f32> {
        self.values.get(index).map(|v| v.get())
    }

    /// Sets a parameter value by index.
    pub fn set_index(&self, index: usize, value: f32) -> bool {
        if let Some(v) = self.values.get(index) {
            v.set(value);
            true
        } else {
            false
        }
    }

    /// Returns the number of parameters.
    pub fn len(&self) -> usize {
        self.names.len()
    }

    /// Returns true if there are no parameters.
    pub fn is_empty(&self) -> bool {
        self.names.is_empty()
    }

    /// Returns an iterator over parameter names.
    pub fn names(&self) -> impl Iterator<Item = &'static str> + '_ {
        self.names.iter().copied()
    }
}

/// Audio processing context passed to nodes.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct AudioContext {
    /// Sample rate in Hz.
    pub sample_rate: f32,
    /// Current time in seconds.
    pub time: f32,
    /// Time delta (1 / sample_rate).
    pub dt: f32,
    /// Current sample index.
    pub sample_index: u64,
}

impl AudioContext {
    /// Creates a new audio context.
    pub fn new(sample_rate: f32) -> Self {
        let dt = 1.0 / sample_rate;
        Self {
            sample_rate,
            time: 0.0,
            dt,
            sample_index: 0,
        }
    }

    /// Advances the context by one sample.
    pub fn advance(&mut self) {
        self.sample_index += 1;
        self.time = self.sample_index as f32 * self.dt;
    }

    /// Resets the context to time zero.
    pub fn reset(&mut self) {
        self.sample_index = 0;
        self.time = 0.0;
    }
}

/// Describes a modulatable parameter on an audio node.
#[derive(Debug, Clone, Copy)]
pub struct ParamDescriptor {
    /// Parameter name.
    pub name: &'static str,
    /// Default value.
    pub default: f32,
    /// Minimum value.
    pub min: f32,
    /// Maximum value.
    pub max: f32,
}

impl ParamDescriptor {
    /// Creates a new parameter descriptor.
    pub const fn new(name: &'static str, default: f32, min: f32, max: f32) -> Self {
        Self {
            name,
            default,
            min,
            max,
        }
    }
}
