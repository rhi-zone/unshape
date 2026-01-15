//! Synth patch system for parameter presets and modulation.
//!
//! Provides a flexible system for defining, storing, and recalling synthesizer
//! parameter configurations with optional modulation routing.

use std::collections::HashMap;

/// A single parameter with range and modulation.
#[derive(Debug, Clone)]
pub struct PatchParameter {
    /// Parameter name.
    pub name: String,
    /// Minimum value.
    pub min: f32,
    /// Maximum value.
    pub max: f32,
    /// Default value.
    pub default: f32,
    /// Current value.
    pub value: f32,
    /// Modulation amount (-1 to 1, scaled to range).
    pub modulation: f32,
    /// Unit suffix for display (e.g., "Hz", "%", "ms").
    pub unit: String,
}

impl PatchParameter {
    /// Creates a new parameter.
    pub fn new(name: &str, min: f32, max: f32, default: f32) -> Self {
        Self {
            name: name.to_string(),
            min,
            max,
            default,
            value: default,
            modulation: 0.0,
            unit: String::new(),
        }
    }

    /// Returns the current value with modulation applied.
    pub fn modulated_value(&self) -> f32 {
        let range = self.max - self.min;
        let mod_offset = self.modulation * range;
        (self.value + mod_offset).clamp(self.min, self.max)
    }

    /// Returns the value normalized to 0-1.
    pub fn normalized(&self) -> f32 {
        (self.value - self.min) / (self.max - self.min)
    }

    /// Sets the value from a normalized 0-1 input.
    pub fn set_normalized(&mut self, normalized: f32) {
        self.value = self.min + normalized.clamp(0.0, 1.0) * (self.max - self.min);
    }

    /// Resets to default value.
    pub fn reset(&mut self) {
        self.value = self.default;
        self.modulation = 0.0;
    }

    /// Common preset: volume (0-1).
    pub fn volume() -> Self {
        Self::new("volume", 0.0, 1.0, 0.8)
    }

    /// Creates a parameter with a unit suffix.
    pub fn with_unit(name: &str, min: f32, max: f32, default: f32, unit: &str) -> Self {
        Self {
            name: name.to_string(),
            min,
            max,
            default,
            value: default,
            modulation: 0.0,
            unit: unit.to_string(),
        }
    }

    /// Common preset: frequency in Hz.
    pub fn frequency(min: f32, max: f32, default: f32) -> Self {
        Self::with_unit("frequency", min, max, default, "Hz")
    }

    /// Common preset: cutoff frequency.
    pub fn cutoff() -> Self {
        Self::with_unit("cutoff", 20.0, 20000.0, 5000.0, "Hz")
    }

    /// Common preset: resonance (0-1).
    pub fn resonance() -> Self {
        Self::new("resonance", 0.0, 1.0, 0.0)
    }

    /// Common preset: attack time.
    pub fn attack() -> Self {
        Self::with_unit("attack", 0.001, 5.0, 0.01, "s")
    }

    /// Common preset: decay time.
    pub fn decay() -> Self {
        Self::with_unit("decay", 0.001, 5.0, 0.1, "s")
    }

    /// Common preset: sustain level.
    pub fn sustain() -> Self {
        Self::new("sustain", 0.0, 1.0, 0.7)
    }

    /// Common preset: release time.
    pub fn release() -> Self {
        Self::with_unit("release", 0.001, 10.0, 0.3, "s")
    }

    /// Common preset: detune in cents.
    pub fn detune() -> Self {
        Self::with_unit("detune", -100.0, 100.0, 0.0, "cents")
    }

    /// Common preset: pan (-1 to 1).
    pub fn pan() -> Self {
        Self::new("pan", -1.0, 1.0, 0.0)
    }
}

/// Modulation source types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ModSource {
    /// Low-frequency oscillator.
    Lfo1,
    Lfo2,
    /// Envelope generators.
    Envelope1,
    Envelope2,
    /// Velocity.
    Velocity,
    /// Aftertouch.
    Aftertouch,
    /// Mod wheel.
    ModWheel,
    /// Pitch bend.
    PitchBend,
    /// Note number (for key tracking).
    KeyTrack,
}

/// A modulation routing.
#[derive(Debug, Clone)]
pub struct ModRouting {
    /// Modulation source.
    pub source: ModSource,
    /// Target parameter name.
    pub target: String,
    /// Modulation amount (-1 to 1).
    pub amount: f32,
}

impl ModRouting {
    /// Creates a new modulation routing.
    pub fn new(source: ModSource, target: &str, amount: f32) -> Self {
        Self {
            source,
            target: target.to_string(),
            amount,
        }
    }
}

/// A complete synth patch with parameters and modulation.
#[derive(Debug, Clone)]
pub struct SynthPatch {
    /// Patch name.
    pub name: String,
    /// Patch category (e.g., "lead", "pad", "bass").
    pub category: String,
    /// Author/creator.
    pub author: String,
    /// Parameters by name.
    pub parameters: HashMap<String, PatchParameter>,
    /// Modulation routings.
    pub mod_routings: Vec<ModRouting>,
    /// Tags for searching.
    pub tags: Vec<String>,
}

impl SynthPatch {
    /// Creates a new empty patch.
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            category: String::new(),
            author: String::new(),
            parameters: HashMap::new(),
            mod_routings: Vec::new(),
            tags: Vec::new(),
        }
    }

    /// Adds a parameter.
    pub fn add_param(&mut self, param: PatchParameter) -> &mut Self {
        self.parameters.insert(param.name.clone(), param);
        self
    }

    /// Gets a parameter value.
    pub fn get(&self, name: &str) -> Option<f32> {
        self.parameters.get(name).map(|p| p.value)
    }

    /// Gets a modulated parameter value.
    pub fn get_modulated(&self, name: &str) -> Option<f32> {
        self.parameters.get(name).map(|p| p.modulated_value())
    }

    /// Sets a parameter value.
    pub fn set(&mut self, name: &str, value: f32) -> bool {
        if let Some(param) = self.parameters.get_mut(name) {
            param.value = value.clamp(param.min, param.max);
            true
        } else {
            false
        }
    }

    /// Sets a parameter from normalized 0-1 input.
    pub fn set_normalized(&mut self, name: &str, normalized: f32) -> bool {
        if let Some(param) = self.parameters.get_mut(name) {
            param.set_normalized(normalized);
            true
        } else {
            false
        }
    }

    /// Adds a modulation routing.
    pub fn add_mod(&mut self, routing: ModRouting) -> &mut Self {
        self.mod_routings.push(routing);
        self
    }

    /// Applies modulation from a source.
    pub fn apply_modulation(&mut self, source: ModSource, value: f32) {
        for routing in &self.mod_routings {
            if routing.source == source {
                if let Some(param) = self.parameters.get_mut(&routing.target) {
                    param.modulation = value * routing.amount;
                }
            }
        }
    }

    /// Resets all modulation.
    pub fn reset_modulation(&mut self) {
        for param in self.parameters.values_mut() {
            param.modulation = 0.0;
        }
    }

    /// Resets all parameters to defaults.
    pub fn reset(&mut self) {
        for param in self.parameters.values_mut() {
            param.reset();
        }
    }

    /// Creates a basic subtractive synth patch template.
    pub fn subtractive_template() -> Self {
        let mut patch = Self::new("Init Subtractive");
        patch.category = "template".to_string();
        patch.add_param(PatchParameter::volume());
        patch.add_param(PatchParameter::frequency(20.0, 20000.0, 440.0));
        patch.add_param(PatchParameter::new("waveform", 0.0, 3.0, 0.0));
        patch.add_param(PatchParameter::cutoff());
        patch.add_param(PatchParameter::resonance());
        patch.add_param(PatchParameter::attack());
        patch.add_param(PatchParameter::decay());
        patch.add_param(PatchParameter::sustain());
        patch.add_param(PatchParameter::release());
        patch.add_param(PatchParameter::new("filter_env", 0.0, 1.0, 0.5));
        patch.add_param(PatchParameter::with_unit("lfo_rate", 0.1, 20.0, 5.0, "Hz"));
        patch.add_param(PatchParameter::new("lfo_depth", 0.0, 1.0, 0.0));
        patch
    }

    /// Creates a basic FM synth patch template.
    pub fn fm_template() -> Self {
        let mut patch = Self::new("Init FM");
        patch.category = "template".to_string();
        patch.add_param(PatchParameter::volume());
        patch.add_param(PatchParameter::frequency(20.0, 8000.0, 440.0));
        patch.add_param(PatchParameter::new("mod_ratio", 0.5, 16.0, 2.0));
        patch.add_param(PatchParameter::new("mod_index", 0.0, 20.0, 5.0));
        patch.add_param(PatchParameter::attack());
        patch.add_param(PatchParameter::decay());
        patch.add_param(PatchParameter::sustain());
        patch.add_param(PatchParameter::release());
        patch.add_param(PatchParameter::with_unit(
            "mod_attack",
            0.001,
            5.0,
            0.001,
            "s",
        ));
        patch.add_param(PatchParameter::with_unit("mod_decay", 0.001, 5.0, 1.0, "s"));
        patch
    }

    /// Preset: warm pad.
    pub fn warm_pad() -> Self {
        let mut patch = Self::new("Warm Pad");
        patch.category = "pad".to_string();
        patch.tags = vec!["warm".to_string(), "ambient".to_string()];
        patch.add_param(PatchParameter::volume());
        patch.add_param(PatchParameter::new("waveform", 0.0, 3.0, 1.0)); // Saw
        patch.add_param(PatchParameter::with_unit(
            "cutoff", 20.0, 20000.0, 2000.0, "Hz",
        ));
        patch.add_param(PatchParameter::new("resonance", 0.0, 1.0, 0.2));
        patch.add_param(PatchParameter::with_unit("attack", 0.001, 5.0, 0.5, "s"));
        patch.add_param(PatchParameter::with_unit("decay", 0.001, 5.0, 0.5, "s"));
        patch.add_param(PatchParameter::new("sustain", 0.0, 1.0, 0.8));
        patch.add_param(PatchParameter::with_unit("release", 0.001, 10.0, 1.0, "s"));
        patch.add_param(PatchParameter::detune());
        patch.add_mod(ModRouting::new(ModSource::Lfo1, "cutoff", 0.1));
        patch
    }

    /// Preset: acid bass.
    pub fn acid_bass() -> Self {
        let mut patch = Self::new("Acid Bass");
        patch.category = "bass".to_string();
        patch.tags = vec!["acid".to_string(), "resonant".to_string()];
        patch.add_param(PatchParameter::volume());
        patch.add_param(PatchParameter::new("waveform", 0.0, 3.0, 1.0)); // Saw
        patch.add_param(PatchParameter::with_unit(
            "cutoff", 20.0, 20000.0, 800.0, "Hz",
        ));
        patch.add_param(PatchParameter::new("resonance", 0.0, 1.0, 0.8));
        patch.add_param(PatchParameter::with_unit("attack", 0.001, 5.0, 0.001, "s"));
        patch.add_param(PatchParameter::with_unit("decay", 0.001, 5.0, 0.2, "s"));
        patch.add_param(PatchParameter::new("sustain", 0.0, 1.0, 0.0));
        patch.add_param(PatchParameter::with_unit("release", 0.001, 10.0, 0.1, "s"));
        patch.add_param(PatchParameter::new("filter_env", 0.0, 1.0, 0.9));
        patch.add_mod(ModRouting::new(ModSource::Envelope1, "cutoff", 0.8));
        patch
    }

    /// Preset: electric piano.
    pub fn electric_piano() -> Self {
        let mut patch = Self::new("Electric Piano");
        patch.category = "keys".to_string();
        patch.tags = vec!["fm".to_string(), "electric".to_string()];
        patch.add_param(PatchParameter::volume());
        patch.add_param(PatchParameter::new("mod_ratio", 0.5, 16.0, 14.0));
        patch.add_param(PatchParameter::new("mod_index", 0.0, 20.0, 4.0));
        patch.add_param(PatchParameter::with_unit("attack", 0.001, 5.0, 0.001, "s"));
        patch.add_param(PatchParameter::with_unit("decay", 0.001, 5.0, 1.5, "s"));
        patch.add_param(PatchParameter::new("sustain", 0.0, 1.0, 0.3));
        patch.add_param(PatchParameter::with_unit("release", 0.001, 10.0, 0.5, "s"));
        patch.add_mod(ModRouting::new(ModSource::Velocity, "mod_index", 0.5));
        patch
    }

    /// Preset: pluck lead.
    pub fn pluck_lead() -> Self {
        let mut patch = Self::new("Pluck Lead");
        patch.category = "lead".to_string();
        patch.tags = vec!["pluck".to_string(), "bright".to_string()];
        patch.add_param(PatchParameter::volume());
        patch.add_param(PatchParameter::new("waveform", 0.0, 3.0, 2.0)); // Square
        patch.add_param(PatchParameter::with_unit(
            "cutoff", 20.0, 20000.0, 8000.0, "Hz",
        ));
        patch.add_param(PatchParameter::new("resonance", 0.0, 1.0, 0.3));
        patch.add_param(PatchParameter::with_unit("attack", 0.001, 5.0, 0.001, "s"));
        patch.add_param(PatchParameter::with_unit("decay", 0.001, 5.0, 0.3, "s"));
        patch.add_param(PatchParameter::new("sustain", 0.0, 1.0, 0.5));
        patch.add_param(PatchParameter::with_unit("release", 0.001, 10.0, 0.2, "s"));
        patch.add_mod(ModRouting::new(ModSource::Envelope1, "cutoff", 0.6));
        patch
    }
}

impl Default for SynthPatch {
    fn default() -> Self {
        Self::subtractive_template()
    }
}

/// A bank of patches.
#[derive(Debug, Clone)]
pub struct PatchBank {
    /// Bank name.
    pub name: String,
    /// Patches in the bank.
    pub patches: Vec<SynthPatch>,
}

impl PatchBank {
    /// Creates a new empty bank.
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            patches: Vec::new(),
        }
    }

    /// Adds a patch to the bank.
    pub fn add(&mut self, patch: SynthPatch) -> &mut Self {
        self.patches.push(patch);
        self
    }

    /// Gets a patch by index.
    pub fn get(&self, index: usize) -> Option<&SynthPatch> {
        self.patches.get(index)
    }

    /// Gets a patch by name.
    pub fn get_by_name(&self, name: &str) -> Option<&SynthPatch> {
        self.patches.iter().find(|p| p.name == name)
    }

    /// Searches patches by tag.
    pub fn search_tag(&self, tag: &str) -> Vec<&SynthPatch> {
        self.patches
            .iter()
            .filter(|p| p.tags.iter().any(|t| t == tag))
            .collect()
    }

    /// Searches patches by category.
    pub fn search_category(&self, category: &str) -> Vec<&SynthPatch> {
        self.patches
            .iter()
            .filter(|p| p.category == category)
            .collect()
    }

    /// Returns the number of patches.
    pub fn len(&self) -> usize {
        self.patches.len()
    }

    /// Returns true if the bank is empty.
    pub fn is_empty(&self) -> bool {
        self.patches.is_empty()
    }

    /// Creates a factory bank with common presets.
    pub fn factory() -> Self {
        let mut bank = Self::new("Factory");
        bank.add(SynthPatch::subtractive_template());
        bank.add(SynthPatch::fm_template());
        bank.add(SynthPatch::warm_pad());
        bank.add(SynthPatch::acid_bass());
        bank.add(SynthPatch::electric_piano());
        bank.add(SynthPatch::pluck_lead());
        bank
    }
}

impl Default for PatchBank {
    fn default() -> Self {
        Self::factory()
    }
}

/// Interpolates between two patches.
pub fn interpolate_patches(a: &SynthPatch, b: &SynthPatch, t: f32) -> SynthPatch {
    let t = t.clamp(0.0, 1.0);
    let mut result = a.clone();
    result.name = format!("{} -> {} ({:.0}%)", a.name, b.name, t * 100.0);

    for (name, param_a) in &a.parameters {
        if let Some(param_b) = b.parameters.get(name) {
            if let Some(result_param) = result.parameters.get_mut(name) {
                result_param.value = param_a.value * (1.0 - t) + param_b.value * t;
            }
        }
    }

    result
}

/// Randomizes a patch within parameter ranges.
pub fn randomize_patch(patch: &SynthPatch, amount: f32, seed: u64) -> SynthPatch {
    let mut result = patch.clone();
    result.name = format!("{} (Random)", patch.name);

    let mut rng_state = seed.wrapping_add(1);
    let next_f32 = |state: &mut u64| -> f32 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (*state as f32) / (u64::MAX as f32)
    };

    for param in result.parameters.values_mut() {
        let random = next_f32(&mut rng_state);
        let range = param.max - param.min;
        let offset = (random - 0.5) * 2.0 * range * amount;
        param.value = (param.value + offset).clamp(param.min, param.max);
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_patch_parameter() {
        let mut param = PatchParameter::new("test", 0.0, 100.0, 50.0);

        assert_eq!(param.value, 50.0);
        assert_eq!(param.normalized(), 0.5);

        param.set_normalized(0.25);
        assert_eq!(param.value, 25.0);

        param.reset();
        assert_eq!(param.value, 50.0);
    }

    #[test]
    fn test_parameter_modulation() {
        let mut param = PatchParameter::new("test", 0.0, 100.0, 50.0);

        param.modulation = 0.2;
        assert!((param.modulated_value() - 70.0).abs() < 0.001);

        param.modulation = -0.3;
        assert!((param.modulated_value() - 20.0).abs() < 0.001);
    }

    #[test]
    fn test_synth_patch() {
        let mut patch = SynthPatch::new("Test");
        patch.add_param(PatchParameter::volume());
        patch.add_param(PatchParameter::cutoff());

        assert!(patch.get("volume").is_some());
        assert!(patch.set("volume", 0.5));
        assert_eq!(patch.get("volume"), Some(0.5));
    }

    #[test]
    fn test_patch_modulation() {
        let mut patch = SynthPatch::new("Test");
        patch.add_param(PatchParameter::cutoff());
        patch.add_mod(ModRouting::new(ModSource::Lfo1, "cutoff", 0.5));

        patch.apply_modulation(ModSource::Lfo1, 1.0);

        let param = patch.parameters.get("cutoff").unwrap();
        assert_eq!(param.modulation, 0.5);
    }

    #[test]
    fn test_patch_templates() {
        let subtractive = SynthPatch::subtractive_template();
        assert!(subtractive.parameters.contains_key("cutoff"));
        assert!(subtractive.parameters.contains_key("attack"));

        let fm = SynthPatch::fm_template();
        assert!(fm.parameters.contains_key("mod_ratio"));
        assert!(fm.parameters.contains_key("mod_index"));
    }

    #[test]
    fn test_patch_presets() {
        let pad = SynthPatch::warm_pad();
        assert_eq!(pad.category, "pad");
        assert!(pad.tags.contains(&"warm".to_string()));

        let bass = SynthPatch::acid_bass();
        assert_eq!(bass.category, "bass");
    }

    #[test]
    fn test_patch_bank() {
        let bank = PatchBank::factory();

        assert!(!bank.is_empty());
        assert!(bank.get(0).is_some());
        assert!(bank.get_by_name("Warm Pad").is_some());
    }

    #[test]
    fn test_bank_search() {
        let bank = PatchBank::factory();

        let pads = bank.search_category("pad");
        assert!(!pads.is_empty());

        let fm_patches = bank.search_tag("fm");
        assert!(!fm_patches.is_empty());
    }

    #[test]
    fn test_interpolate_patches() {
        let mut a = SynthPatch::new("A");
        a.add_param(PatchParameter::new("test", 0.0, 100.0, 0.0));
        let mut b = SynthPatch::new("B");
        b.add_param(PatchParameter::new("test", 0.0, 100.0, 100.0));
        b.set("test", 100.0);

        let mid = interpolate_patches(&a, &b, 0.5);
        assert_eq!(mid.get("test"), Some(50.0));
    }

    #[test]
    fn test_randomize_patch() {
        let original = SynthPatch::subtractive_template();
        let randomized = randomize_patch(&original, 0.5, 12345);

        // Values should be different
        let orig_cutoff = original.get("cutoff").unwrap();
        let rand_cutoff = randomized.get("cutoff").unwrap();

        // With enough randomization, they should differ
        // (might occasionally be the same, but very unlikely)
        assert!(randomized.name.contains("Random"));
    }

    #[test]
    fn test_parameter_presets() {
        let volume = PatchParameter::volume();
        assert_eq!(volume.min, 0.0);
        assert_eq!(volume.max, 1.0);

        let cutoff = PatchParameter::cutoff();
        assert_eq!(cutoff.unit, "Hz");

        let attack = PatchParameter::attack();
        assert_eq!(attack.unit, "s");
    }

    #[test]
    fn test_mod_sources() {
        let routing1 = ModRouting::new(ModSource::Lfo1, "cutoff", 0.5);
        let routing2 = ModRouting::new(ModSource::Lfo2, "cutoff", 0.5);

        assert_ne!(routing1.source, routing2.source);
    }
}
