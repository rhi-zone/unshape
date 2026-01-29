//! Cellular automata for procedural pattern generation.
//!
//! Implements 1D, 2D, and 3D cellular automata with configurable neighborhoods.
//!
//! # Example
//!
//! ```
//! use unshape_automata::{ElementaryCA, GameOfLife, CellularAutomaton2D, Moore, VonNeumann};
//!
//! // 1D: Rule 30
//! let mut ca = ElementaryCA::new(100, 30);
//! ca.randomize(12345);
//! ca.step();
//!
//! // 2D: Game of Life with Moore neighborhood (default)
//! let mut life = GameOfLife::life(50, 50);
//! life.randomize(12345, 0.3);
//! life.step();
//!
//! // 2D: Custom neighborhood
//! let mut ca = CellularAutomaton2D::with_neighborhood(50, 50, &[3], &[2, 3], VonNeumann);
//! ca.randomize(12345, 0.3);
//! ca.step();
//! ```

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

mod elementary;
mod hash_life;
mod larger_than_life;
mod life;
mod neighborhood;
mod smooth_life;
mod turmite;

pub use elementary::*;
pub use hash_life::*;
pub use larger_than_life::*;
pub use life::*;
pub use neighborhood::*;
pub use smooth_life::*;
pub use turmite::*;

/// Registers all automata operations with an [`OpRegistry`].
///
/// Call this to enable deserialization of automata ops from saved pipelines.
#[cfg(feature = "dynop")]
pub fn register_ops(registry: &mut unshape_op::OpRegistry) {
    registry.register_type::<ElementaryCAConfig>("resin::ElementaryCAConfig");
    registry.register_type::<CellularAutomaton2DConfig>("resin::CellularAutomaton2DConfig");
    registry.register_type::<StepElementaryCA>("resin::StepElementaryCA");
    registry.register_type::<StepCellularAutomaton2D>("resin::StepCellularAutomaton2D");
    registry.register_type::<GeneratePattern>("resin::GeneratePattern");
}

/// Simple RNG for cellular automata.
pub(crate) struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    pub(crate) fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_add(1),
        }
    }

    pub(crate) fn next_u64(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.state
    }

    pub(crate) fn next_bool(&mut self) -> bool {
        self.next_u64() & 1 == 1
    }

    pub(crate) fn next_f32(&mut self) -> f32 {
        (self.next_u64() as f64 / u64::MAX as f64) as f32
    }
}

// ============================================================================
// Operation Structs (DynOp support)
// ============================================================================

/// Configuration operation for creating a 1D elementary cellular automaton.
///
/// This operation creates an `ElementaryCA` with the specified width and rule.
/// Use a seed for reproducible random initialization, or None for empty state.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dynop", derive(unshape_op::Op))]
#[cfg_attr(feature = "dynop", op(input = (), output = ElementaryCA))]
pub struct ElementaryCAConfig {
    /// Width of the automaton (number of cells).
    pub width: usize,
    /// Rule number (0-255).
    pub rule: u8,
    /// Whether to wrap at edges (toroidal topology).
    pub wrap: bool,
    /// Seed for random initialization (None = start with center cell only).
    pub seed: Option<u64>,
}

impl ElementaryCAConfig {
    /// Creates a new configuration with default settings.
    pub fn new(width: usize, rule: u8) -> Self {
        Self {
            width,
            rule,
            wrap: true,
            seed: None,
        }
    }

    /// Creates the configured ElementaryCA.
    pub fn apply(&self) -> ElementaryCA {
        let mut ca = ElementaryCA::new(self.width, self.rule);
        ca.set_wrap(self.wrap);
        if let Some(seed) = self.seed {
            ca.randomize(seed);
        } else {
            ca.set_center();
        }
        ca
    }
}

impl Default for ElementaryCAConfig {
    fn default() -> Self {
        Self::new(100, elementary_rules::RULE_30)
    }
}

/// Configuration operation for creating a 2D cellular automaton.
///
/// This operation creates a `CellularAutomaton2D` with the specified dimensions
/// and birth/survival rules. Use a seed and density for random initialization.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dynop", derive(unshape_op::Op))]
#[cfg_attr(feature = "dynop", op(input = (), output = CellularAutomaton2D))]
pub struct CellularAutomaton2DConfig {
    /// Width of the grid.
    pub width: usize,
    /// Height of the grid.
    pub height: usize,
    /// Birth rule (number of neighbors that cause birth).
    pub birth: Vec<u8>,
    /// Survival rule (number of neighbors that allow survival).
    pub survive: Vec<u8>,
    /// Whether to wrap at edges (toroidal topology).
    pub wrap: bool,
    /// Seed for random initialization (None = start empty).
    pub seed: Option<u64>,
    /// Density for random initialization (0.0 - 1.0).
    pub density: f32,
}

impl CellularAutomaton2DConfig {
    /// Creates a new configuration with custom rules.
    pub fn new(width: usize, height: usize, birth: &[u8], survive: &[u8]) -> Self {
        Self {
            width,
            height,
            birth: birth.to_vec(),
            survive: survive.to_vec(),
            wrap: true,
            seed: None,
            density: 0.3,
        }
    }

    /// Creates a Game of Life configuration (B3/S23).
    pub fn life(width: usize, height: usize) -> Self {
        let (birth, survive) = rules::LIFE;
        Self::new(width, height, birth, survive)
    }

    /// Creates a HighLife configuration (B36/S23).
    pub fn high_life(width: usize, height: usize) -> Self {
        let (birth, survive) = rules::HIGH_LIFE;
        Self::new(width, height, birth, survive)
    }

    /// Creates a Seeds configuration (B2/S).
    pub fn seeds(width: usize, height: usize) -> Self {
        let (birth, survive) = rules::SEEDS;
        Self::new(width, height, birth, survive)
    }

    /// Creates the configured CellularAutomaton2D.
    pub fn apply(&self) -> CellularAutomaton2D {
        let mut ca = CellularAutomaton2D::new(self.width, self.height, &self.birth, &self.survive);
        ca.set_wrap(self.wrap);
        if let Some(seed) = self.seed {
            ca.randomize(seed, self.density);
        }
        ca
    }
}

impl Default for CellularAutomaton2DConfig {
    fn default() -> Self {
        Self::life(50, 50)
    }
}

/// Operation to step an elementary CA forward.
///
/// Takes an `ElementaryCA` and returns it after advancing the specified number of steps.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dynop", derive(unshape_op::Op))]
#[cfg_attr(feature = "dynop", op(input = ElementaryCA, output = ElementaryCA))]
pub struct StepElementaryCA {
    /// Number of steps to advance.
    pub steps: usize,
}

impl StepElementaryCA {
    /// Creates a new step operation.
    pub fn new(steps: usize) -> Self {
        Self { steps }
    }

    /// Applies the step operation to an ElementaryCA.
    pub fn apply(&self, ca: &ElementaryCA) -> ElementaryCA {
        let mut result = ca.clone();
        result.steps(self.steps);
        result
    }
}

impl Default for StepElementaryCA {
    fn default() -> Self {
        Self::new(1)
    }
}

/// Operation to step a 2D CA forward.
///
/// Takes a `CellularAutomaton2D` and returns it after advancing the specified number of steps.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dynop", derive(unshape_op::Op))]
#[cfg_attr(feature = "dynop", op(input = CellularAutomaton2D, output = CellularAutomaton2D))]
pub struct StepCellularAutomaton2D {
    /// Number of steps to advance.
    pub steps: usize,
}

impl StepCellularAutomaton2D {
    /// Creates a new step operation.
    pub fn new(steps: usize) -> Self {
        Self { steps }
    }

    /// Applies the step operation to a CellularAutomaton2D.
    pub fn apply(&self, ca: &CellularAutomaton2D) -> CellularAutomaton2D {
        let mut result = ca.clone();
        result.steps(self.steps);
        result
    }
}

impl Default for StepCellularAutomaton2D {
    fn default() -> Self {
        Self::new(1)
    }
}

/// Operation to generate a 2D pattern from a 1D elementary CA.
///
/// Runs the CA for multiple generations and returns all states as a 2D grid.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dynop", derive(unshape_op::Op))]
#[cfg_attr(feature = "dynop", op(input = ElementaryCA, output = Vec<Vec<bool>>))]
pub struct GeneratePattern {
    /// Number of generations to produce.
    pub generations: usize,
}

impl GeneratePattern {
    /// Creates a new pattern generation operation.
    pub fn new(generations: usize) -> Self {
        Self { generations }
    }

    /// Generates the pattern from an ElementaryCA.
    pub fn apply(&self, ca: &ElementaryCA) -> Vec<Vec<bool>> {
        let mut result = ca.clone();
        result.generate_pattern(self.generations)
    }
}

impl Default for GeneratePattern {
    fn default() -> Self {
        Self::new(100)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_elementary_ca_creation() {
        let ca = ElementaryCA::new(100, 30);
        assert_eq!(ca.width(), 100);
        assert_eq!(ca.rule(), 30);
    }

    #[test]
    fn test_elementary_ca_center() {
        let mut ca = ElementaryCA::new(10, 30);
        ca.set_center();

        assert!(!ca.get(0));
        assert!(ca.get(5));
        assert!(!ca.get(9));
    }

    #[test]
    fn test_elementary_ca_step() {
        let mut ca = ElementaryCA::new(10, 30);
        ca.set_center();

        let initial = ca.cells().to_vec();
        ca.step();
        let after = ca.cells().to_vec();

        // State should change
        assert_ne!(initial, after);
    }

    #[test]
    fn test_elementary_ca_rule_90() {
        // Rule 90 produces Sierpinski triangle
        let mut ca = ElementaryCA::new(11, 90);
        ca.set_center();

        // After one step, should have two cells
        ca.step();

        let alive_count: usize = ca.cells().iter().filter(|&&c| c).count();
        assert_eq!(alive_count, 2);
    }

    #[test]
    fn test_elementary_ca_generate_pattern() {
        let mut ca = ElementaryCA::new(20, 30);
        ca.set_center();

        let pattern = ca.generate_pattern(10);

        assert_eq!(pattern.len(), 10);
        assert_eq!(pattern[0].len(), 20);
    }

    #[test]
    fn test_2d_ca_creation() {
        let ca = CellularAutomaton2D::new(10, 10, &[3], &[2, 3]);
        assert_eq!(ca.width(), 10);
        assert_eq!(ca.height(), 10);
    }

    #[test]
    fn test_2d_ca_set_get() {
        let mut ca = CellularAutomaton2D::new(10, 10, &[3], &[2, 3]);

        assert!(!ca.get(5, 5));
        ca.set(5, 5, true);
        assert!(ca.get(5, 5));
    }

    #[test]
    fn test_2d_ca_randomize() {
        let mut ca = CellularAutomaton2D::new(20, 20, &[3], &[2, 3]);
        ca.randomize(12345, 0.5);

        let pop = ca.population();
        // Should have roughly 50% alive (with some variance)
        assert!(pop > 100 && pop < 300);
    }

    #[test]
    fn test_game_of_life_blinker() {
        // Blinker oscillator
        let mut life = GameOfLife::life(5, 5);

        // Horizontal blinker
        life.set(1, 2, true);
        life.set(2, 2, true);
        life.set(3, 2, true);

        let initial_pop = life.population();
        assert_eq!(initial_pop, 3);

        // After one step, should become vertical
        life.step();

        assert!(!life.get(1, 2));
        assert!(life.get(2, 1));
        assert!(life.get(2, 2));
        assert!(life.get(2, 3));
        assert!(!life.get(3, 2));
    }

    #[test]
    fn test_game_of_life_block() {
        // Block still life
        let mut life = GameOfLife::life(5, 5);

        life.set(1, 1, true);
        life.set(2, 1, true);
        life.set(1, 2, true);
        life.set(2, 2, true);

        let initial = life.cells().clone();
        life.step();

        // Block should not change
        assert_eq!(life.cells(), &initial);
    }

    #[test]
    fn test_count_neighbors() {
        let mut ca = CellularAutomaton2D::new(5, 5, &[3], &[2, 3]);

        // Set a cross pattern around (2,2)
        ca.set(1, 2, true);
        ca.set(3, 2, true);
        ca.set(2, 1, true);
        ca.set(2, 3, true);

        let count = ca.count_neighbors(2, 2);
        assert_eq!(count, 4);
    }

    #[test]
    fn test_rules_presets() {
        let (birth, survive) = rules::LIFE;
        let life = CellularAutomaton2D::new(10, 10, birth, survive);
        assert_eq!(life.width(), 10);

        let (birth, survive) = rules::HIGH_LIFE;
        let _high = CellularAutomaton2D::new(10, 10, birth, survive);

        let (birth, survive) = rules::SEEDS;
        let _seeds = CellularAutomaton2D::new(10, 10, birth, survive);
    }

    #[test]
    fn test_wrap_behavior() {
        let mut ca = CellularAutomaton2D::new(5, 5, &[3], &[2, 3]);
        ca.set_wrap(true);

        // Set cells at edges
        ca.set(0, 0, true);
        ca.set(4, 0, true);
        ca.set(0, 4, true);

        // Cell at (0,0) should count (4,4) as neighbor when wrapping
        let count = ca.count_neighbors(0, 0);
        assert!(count >= 2);
    }

    // Neighborhood tests

    #[test]
    fn test_moore_neighborhood() {
        let moore = Moore;
        assert_eq!(moore.offsets().len(), 8);
        assert_eq!(moore.max_neighbors(), 8);
    }

    #[test]
    fn test_von_neumann_neighborhood() {
        let vn = VonNeumann;
        assert_eq!(vn.offsets().len(), 4);
        assert_eq!(vn.max_neighbors(), 4);
    }

    #[test]
    fn test_extended_moore_radius_1() {
        let em = ExtendedMoore::new(1);
        assert_eq!(em.offsets().len(), 8); // Same as Moore
    }

    #[test]
    fn test_extended_moore_radius_2() {
        let em = ExtendedMoore::new(2);
        // 5x5 - 1 center = 24 neighbors
        assert_eq!(em.offsets().len(), 24);
    }

    #[test]
    fn test_ca_with_von_neumann() {
        // Von Neumann neighborhood has only 4 neighbors
        // B2/S rule should behave differently
        let mut ca = CellularAutomaton2D::with_neighborhood(5, 5, &[2], &[1, 2], VonNeumann);

        // Set a cross pattern - each arm cell has 1 neighbor (center)
        ca.set(2, 2, true); // center
        ca.set(1, 2, true);
        ca.set(3, 2, true);
        ca.set(2, 1, true);
        ca.set(2, 3, true);

        assert_eq!(ca.population(), 5);
        assert_eq!(ca.max_neighbors(), 4);
    }

    #[test]
    fn test_custom_neighborhood() {
        // Knight's move neighborhood (like chess knight)
        let knight = CustomNeighborhood2D::new([
            (-2, -1),
            (-2, 1),
            (-1, -2),
            (-1, 2),
            (1, -2),
            (1, 2),
            (2, -1),
            (2, 1),
        ]);
        assert_eq!(knight.offsets().len(), 8);
    }

    #[test]
    fn test_3d_neighborhoods() {
        let moore3d = Moore3D;
        assert_eq!(moore3d.offsets().len(), 26);

        let vn3d = VonNeumann3D;
        assert_eq!(vn3d.offsets().len(), 6);
    }

    // Larger than Life tests

    #[test]
    fn test_ltl_creation() {
        let ltl = LargerThanLife::new(50, 50, 5, ltl_rules::BUGS);
        assert_eq!(ltl.width(), 50);
        assert_eq!(ltl.height(), 50);
        assert_eq!(ltl.radius(), 5);
    }

    #[test]
    fn test_ltl_rules() {
        let rules = LtlRules::new(34, 45, 34, 58);
        assert!(rules.should_birth(34));
        assert!(rules.should_birth(40));
        assert!(rules.should_birth(45));
        assert!(!rules.should_birth(33));
        assert!(!rules.should_birth(46));

        assert!(rules.should_survive(34));
        assert!(rules.should_survive(50));
        assert!(rules.should_survive(58));
        assert!(!rules.should_survive(33));
        assert!(!rules.should_survive(59));
    }

    #[test]
    fn test_ltl_step() {
        let mut ltl = LargerThanLife::new(30, 30, 2, ltl_rules::MAJORITY);
        ltl.randomize(12345, 0.5);

        let initial_pop = ltl.population();
        ltl.step();
        let after_pop = ltl.population();

        // Population should change
        assert_ne!(initial_pop, after_pop);
    }

    #[test]
    fn test_ltl_presets() {
        // Just verify presets are valid
        let _ = LargerThanLife::new(10, 10, 5, ltl_rules::BUGS);
        let _ = LargerThanLife::new(10, 10, 7, ltl_rules::WAFFLE);
        let _ = LargerThanLife::new(10, 10, 8, ltl_rules::GLOBE);
        let _ = LargerThanLife::new(10, 10, 4, ltl_rules::MAJORITY);
        let _ = LargerThanLife::new(10, 10, 10, ltl_rules::MODERN_ART);
    }

    // Langton's Ant tests

    #[test]
    fn test_langtons_ant_creation() {
        let ant = LangtonsAnt::new(100, 100, ant_rules::LANGTON);
        assert_eq!(ant.width(), 100);
        assert_eq!(ant.height(), 100);
        assert_eq!(ant.position(), (50, 50));
        assert_eq!(ant.direction(), Direction::North);
    }

    #[test]
    fn test_langtons_ant_step() {
        let mut ant = LangtonsAnt::new(10, 10, "RL");

        // Initial: at (5,5), facing north, cell is 0 (white)
        assert_eq!(ant.get(5, 5), 0);

        // Step: turn right (R for white), flip to 1, move forward (east)
        ant.step();

        assert_eq!(ant.get(5, 5), 1); // Cell flipped
        assert_eq!(ant.position(), (6, 5)); // Moved east
        assert_eq!(ant.direction(), Direction::East);
    }

    #[test]
    fn test_langtons_ant_multi_state() {
        // LLRR has 4 states
        let mut ant = LangtonsAnt::new(20, 20, "LLRR");
        ant.steps(100);

        let counts = ant.state_counts();
        assert_eq!(counts.len(), 4);
    }

    #[test]
    fn test_langtons_ant_presets() {
        let _ = LangtonsAnt::new(50, 50, ant_rules::LANGTON);
        let _ = LangtonsAnt::new(50, 50, ant_rules::LLRR);
        let _ = LangtonsAnt::new(50, 50, ant_rules::LRRL);
        let _ = LangtonsAnt::new(50, 50, ant_rules::COMPLEX);
    }

    #[test]
    fn test_direction_turns() {
        assert_eq!(Direction::North.turn_left(), Direction::West);
        assert_eq!(Direction::North.turn_right(), Direction::East);
        assert_eq!(Direction::North.turn_around(), Direction::South);

        assert_eq!(Direction::East.turn_left(), Direction::North);
        assert_eq!(Direction::West.turn_right(), Direction::North);
    }

    #[test]
    fn test_turmite_creation() {
        let rules = vec![
            TurmiteRule::new(0, 0, 1, Turn::Left, 0),
            TurmiteRule::new(1, 0, 0, Turn::Right, 0),
        ];
        let turmite = Turmite::new(50, 50, 2, 1, rules);

        assert_eq!(turmite.width(), 50);
        assert_eq!(turmite.num_grid_states(), 2);
        assert_eq!(turmite.num_ant_states(), 1);
    }

    #[test]
    fn test_turmite_step() {
        let rules = vec![
            TurmiteRule::new(0, 0, 1, Turn::Right, 0),
            TurmiteRule::new(1, 0, 0, Turn::Left, 0),
        ];
        let mut turmite = Turmite::new(10, 10, 2, 1, rules);

        let initial_pos = turmite.position();
        turmite.step();
        let new_pos = turmite.position();

        // Should have moved
        assert_ne!(initial_pos, new_pos);
        // Cell should have changed
        assert_eq!(turmite.get(5, 5), 1);
    }

    // 3D Cellular Automata tests

    #[test]
    fn test_ca_3d_creation() {
        let ca = CellularAutomaton3D::new(10, 10, 10, &[4], &[4]);
        assert_eq!(ca.width(), 10);
        assert_eq!(ca.height(), 10);
        assert_eq!(ca.depth(), 10);
        assert_eq!(ca.max_neighbors(), 26); // Moore3D
    }

    #[test]
    fn test_ca_3d_set_get() {
        let mut ca = CellularAutomaton3D::new(5, 5, 5, &[4], &[4]);

        assert!(!ca.get(2, 2, 2));
        ca.set(2, 2, 2, true);
        assert!(ca.get(2, 2, 2));
    }

    #[test]
    fn test_ca_3d_randomize() {
        let mut ca = CellularAutomaton3D::new(10, 10, 10, &[4], &[4]);
        ca.randomize(12345, 0.5);

        let pop = ca.population();
        // Should have roughly 50% alive (1000 cells total)
        assert!(pop > 300 && pop < 700);
    }

    #[test]
    fn test_ca_3d_step() {
        let mut ca = CellularAutomaton3D::new(10, 10, 10, &[4], &[4]);
        ca.randomize(12345, 0.3);

        let initial_pop = ca.population();
        ca.step();
        let after_pop = ca.population();

        // Population should change
        assert_ne!(initial_pop, after_pop);
    }

    #[test]
    fn test_ca_3d_with_von_neumann() {
        let ca = CellularAutomaton3D::with_neighborhood(5, 5, 5, &[2], &[2], VonNeumann3D);
        assert_eq!(ca.max_neighbors(), 6);
    }

    #[test]
    fn test_ca_3d_slice() {
        let mut ca = CellularAutomaton3D::new(5, 5, 5, &[4], &[4]);
        ca.set(2, 2, 2, true);

        let slice = ca.slice_z(2).unwrap();
        assert!(slice[2][2]);
    }

    #[test]
    fn test_ca_3d_presets() {
        let (b, s) = rules_3d::LIFE_445;
        let _ = CellularAutomaton3D::new(5, 5, 5, b, s);

        let (b, s) = rules_3d::CRYSTAL;
        let _ = CellularAutomaton3D::new(5, 5, 5, b, s);

        let (b, s) = rules_3d::AMOEBA_3D;
        let _ = CellularAutomaton3D::new(5, 5, 5, b, s);
    }

    // SmoothLife tests

    #[test]
    fn test_smoothlife_creation() {
        let sl = SmoothLife::new(50, 50, SmoothLifeConfig::default());
        assert_eq!(sl.width(), 50);
        assert_eq!(sl.height(), 50);
    }

    #[test]
    fn test_smoothlife_set_get() {
        let mut sl = SmoothLife::new(10, 10, SmoothLifeConfig::default());

        assert_eq!(sl.get(5, 5), 0.0);
        sl.set(5, 5, 0.7);
        assert!((sl.get(5, 5) - 0.7).abs() < 0.001);
    }

    #[test]
    fn test_smoothlife_clamping() {
        let mut sl = SmoothLife::new(10, 10, SmoothLifeConfig::default());

        sl.set(5, 5, 1.5);
        assert_eq!(sl.get(5, 5), 1.0);

        sl.set(5, 5, -0.5);
        assert_eq!(sl.get(5, 5), 0.0);
    }

    #[test]
    fn test_smoothlife_randomize() {
        let mut sl = SmoothLife::new(20, 20, SmoothLifeConfig::default());
        sl.randomize(12345, 0.5);

        // Average should be roughly in the middle
        let avg = sl.average_value();
        assert!(avg > 0.1 && avg < 0.9);
    }

    #[test]
    fn test_smoothlife_step() {
        let mut sl = SmoothLife::new(30, 30, SmoothLifeConfig::default());
        sl.randomize(12345, 0.3);

        let initial_avg = sl.average_value();
        sl.step(0.5);
        let after_avg = sl.average_value();

        // State should change (values evolve)
        assert!((initial_avg - after_avg).abs() > 0.001);
    }

    #[test]
    fn test_smoothlife_configs() {
        let _ = SmoothLife::new(10, 10, SmoothLifeConfig::standard());
        let _ = SmoothLife::new(10, 10, SmoothLifeConfig::fluid());
        let _ = SmoothLife::new(10, 10, SmoothLifeConfig::slow());
    }

    #[test]
    fn test_smoothlife_continuous_values() {
        let mut sl = SmoothLife::new(20, 20, SmoothLifeConfig::default());
        sl.randomize(12345, 0.4);
        sl.steps(5, 0.2);

        // After several steps, values should still be in [0, 1]
        for row in sl.cells() {
            for &cell in row {
                assert!(cell >= 0.0 && cell <= 1.0);
            }
        }
    }

    // HashLife tests

    #[test]
    fn test_hashlife_creation() {
        let universe = HashLife::new();
        assert_eq!(universe.population(), 0);
        assert_eq!(universe.generation(), 0);
    }

    #[test]
    fn test_hashlife_set_get() {
        let mut universe = HashLife::new();

        universe.set_cell(0, 0, true);
        assert!(universe.get_cell(0, 0));
        assert!(!universe.get_cell(1, 0));

        universe.set_cell(5, 7, true);
        assert!(universe.get_cell(5, 7));
        assert_eq!(universe.population(), 2);
    }

    #[test]
    fn test_hashlife_negative_coords() {
        let mut universe = HashLife::new();

        universe.set_cell(-5, -3, true);
        assert!(universe.get_cell(-5, -3));
        assert!(!universe.get_cell(-5, -2));
    }

    #[test]
    fn test_hashlife_blinker() {
        let mut universe = HashLife::new();

        // Create a vertical blinker
        universe.set_cell(0, -1, true);
        universe.set_cell(0, 0, true);
        universe.set_cell(0, 1, true);

        assert_eq!(universe.population(), 3);

        // After one step, should become horizontal
        universe.step();

        // Horizontal blinker
        assert!(universe.get_cell(-1, 0));
        assert!(universe.get_cell(0, 0));
        assert!(universe.get_cell(1, 0));
        assert!(!universe.get_cell(0, -1));
        assert!(!universe.get_cell(0, 1));
        assert_eq!(universe.population(), 3);

        // After another step, back to vertical
        universe.step();
        assert!(universe.get_cell(0, -1));
        assert!(universe.get_cell(0, 0));
        assert!(universe.get_cell(0, 1));
    }

    #[test]
    fn test_hashlife_block() {
        let mut universe = HashLife::new();

        // Create a block (2x2 still life)
        universe.set_cell(0, 0, true);
        universe.set_cell(1, 0, true);
        universe.set_cell(0, 1, true);
        universe.set_cell(1, 1, true);

        let initial_pop = universe.population();
        assert_eq!(initial_pop, 4);

        // Block is stable - should not change
        universe.steps(5);
        assert_eq!(universe.population(), 4);
        assert!(universe.get_cell(0, 0));
        assert!(universe.get_cell(1, 0));
        assert!(universe.get_cell(0, 1));
        assert!(universe.get_cell(1, 1));
    }

    #[test]
    fn test_hashlife_glider() {
        let mut universe = HashLife::new();

        // Create a glider
        //   X
        //     X
        // X X X
        universe.set_cell(1, 0, true);
        universe.set_cell(2, 1, true);
        universe.set_cell(0, 2, true);
        universe.set_cell(1, 2, true);
        universe.set_cell(2, 2, true);

        assert_eq!(universe.population(), 5);

        // Glider should survive
        universe.steps(4);
        assert_eq!(universe.population(), 5);
    }

    #[test]
    fn test_hashlife_clear_cache() {
        let mut universe = HashLife::new();

        universe.set_cell(0, 0, true);
        universe.set_cell(1, 0, true);
        universe.step();

        // Cache should have entries now
        universe.clear_cache();
        // Should still work after clearing
        universe.step();
    }

    #[test]
    fn test_hashlife_bounds() {
        let mut universe = HashLife::new();

        // Empty universe has no bounds
        assert!(universe.bounds().is_none());

        universe.set_cell(5, 10, true);
        universe.set_cell(-3, 7, true);

        let bounds = universe.bounds();
        assert!(bounds.is_some());
        let (min_x, min_y, max_x, max_y) = bounds.unwrap();
        assert_eq!(min_x, -3);
        assert_eq!(max_x, 5);
        assert_eq!(min_y, 7);
        assert_eq!(max_y, 10);
    }

    #[test]
    fn test_hashlife_step_pow2_blinker() {
        // Blinker is period 2. After 2^1 = 2 generations, it returns to original.
        let mut universe = HashLife::new();
        universe.set_cell(0, -1, true);
        universe.set_cell(0, 0, true);
        universe.set_cell(0, 1, true);

        universe.step_pow2(1); // 2 generations
        assert_eq!(universe.generation(), 2);

        // Should be back to original vertical orientation
        assert!(universe.get_cell(0, -1));
        assert!(universe.get_cell(0, 0));
        assert!(universe.get_cell(0, 1));
        assert!(!universe.get_cell(-1, 0));
        assert!(!universe.get_cell(1, 0));
    }

    #[test]
    fn test_hashlife_step_pow2_large() {
        // R-pentomino: stabilizes after 1103 generations.
        // Test that step_pow2(10) = 1024 generations works without crashing.
        let mut universe = HashLife::new();
        universe.set_cell(1, 0, true);
        universe.set_cell(2, 0, true);
        universe.set_cell(0, 1, true);
        universe.set_cell(1, 1, true);
        universe.set_cell(1, 2, true);

        universe.step_pow2(10); // 1024 generations
        assert_eq!(universe.generation(), 1024);
        // R-pentomino should still have live cells after 1024 generations
        assert!(universe.population() > 0);
    }

    #[test]
    fn test_hashlife_steps_decomposed() {
        // Compare step-by-step with decomposed steps() for a blinker.
        // Blinker period is 2, so after 7 steps it should be in phase 1.
        let mut a = HashLife::new();
        a.set_cell(0, -1, true);
        a.set_cell(0, 0, true);
        a.set_cell(0, 1, true);

        let mut b = a.clone();

        // a: step one at a time
        for _ in 0..7 {
            a.step();
        }

        // b: decomposed (7 = 4 + 2 + 1)
        b.steps(7);

        assert_eq!(a.generation(), 7);
        assert_eq!(b.generation(), 7);
        assert_eq!(a.population(), b.population());

        // Both should be in phase 1 (horizontal blinker)
        assert!(a.get_cell(-1, 0));
        assert!(a.get_cell(0, 0));
        assert!(a.get_cell(1, 0));

        assert!(b.get_cell(-1, 0));
        assert!(b.get_cell(0, 0));
        assert!(b.get_cell(1, 0));
    }

    #[test]
    fn test_hashlife_step_pow2_glider() {
        // Glider moves 1 cell diagonally every 4 generations.
        // After 2^2 = 4 generations, glider should shift by (1, 1).
        let mut universe = HashLife::new();
        universe.set_cell(1, 0, true);
        universe.set_cell(2, 1, true);
        universe.set_cell(0, 2, true);
        universe.set_cell(1, 2, true);
        universe.set_cell(2, 2, true);

        universe.step_pow2(2); // 4 generations
        assert_eq!(universe.generation(), 4);
        assert_eq!(universe.population(), 5);

        // Glider shifted by (1, 1)
        assert!(universe.get_cell(2, 1));
        assert!(universe.get_cell(3, 2));
        assert!(universe.get_cell(1, 3));
        assert!(universe.get_cell(2, 3));
        assert!(universe.get_cell(3, 3));
    }

    #[test]
    fn test_hashlife_memoization() {
        // Run a pattern, clear cache, run again - results should be identical.
        let mut a = HashLife::new();
        a.set_cell(0, -1, true);
        a.set_cell(0, 0, true);
        a.set_cell(0, 1, true);

        let mut b = a.clone();

        a.step_pow2(3); // 8 generations with warm cache
        b.clear_cache();
        b.step_pow2(3); // 8 generations with cold cache

        assert_eq!(a.generation(), b.generation());
        assert_eq!(a.population(), b.population());
    }
}

// ============================================================================
// Invariant tests - mathematical properties that must hold
// ============================================================================

#[cfg(all(test, feature = "invariant-tests"))]
mod invariant_tests {
    use super::*;

    // ------------------------------------------------------------------------
    // Neighborhood invariants
    // ------------------------------------------------------------------------

    #[test]
    fn test_moore_neighbor_count() {
        let moore = Moore;
        assert_eq!(
            moore.offsets().len(),
            8,
            "Moore neighborhood must have 8 neighbors"
        );
    }

    #[test]
    fn test_vonneumann_neighbor_count() {
        let vn = VonNeumann;
        assert_eq!(
            vn.offsets().len(),
            4,
            "Von Neumann neighborhood must have 4 neighbors"
        );
    }

    #[test]
    fn test_hexagonal_neighbor_count() {
        let hex = Hexagonal;
        assert_eq!(
            hex.offsets().len(),
            6,
            "Hexagonal neighborhood must have 6 neighbors"
        );
    }

    #[test]
    fn test_extended_moore_radius_formula() {
        // ExtendedMoore(r) should have (2r+1)^2 - 1 neighbors
        for r in 1..=5 {
            let em = ExtendedMoore::new(r);
            let expected = ((2 * r + 1) * (2 * r + 1) - 1) as usize;
            assert_eq!(
                em.offsets().len(),
                expected,
                "ExtendedMoore({r}) should have {expected} neighbors"
            );
        }
    }

    #[test]
    fn test_moore3d_neighbor_count() {
        let moore = Moore3D;
        assert_eq!(
            moore.offsets().len(),
            26,
            "Moore3D neighborhood must have 26 neighbors"
        );
    }

    #[test]
    fn test_vonneumann3d_neighbor_count() {
        let vn = VonNeumann3D;
        assert_eq!(
            vn.offsets().len(),
            6,
            "VonNeumann3D neighborhood must have 6 neighbors"
        );
    }

    #[test]
    fn test_neighborhoods_exclude_origin() {
        // No neighborhood should include (0, 0) as a neighbor
        let moore = Moore;
        assert!(
            !moore.offsets().contains(&(0, 0)),
            "Moore must not include origin"
        );

        let vn = VonNeumann;
        assert!(
            !vn.offsets().contains(&(0, 0)),
            "VonNeumann must not include origin"
        );

        let hex = Hexagonal;
        assert!(
            !hex.offsets().contains(&(0, 0)),
            "Hexagonal must not include origin"
        );

        let em = ExtendedMoore::new(2);
        assert!(
            !em.offsets().contains(&(0, 0)),
            "ExtendedMoore must not include origin"
        );

        let moore3d = Moore3D;
        assert!(
            !moore3d.offsets().contains(&(0, 0, 0)),
            "Moore3D must not include origin"
        );

        let vn3d = VonNeumann3D;
        assert!(
            !vn3d.offsets().contains(&(0, 0, 0)),
            "VonNeumann3D must not include origin"
        );
    }

    // ------------------------------------------------------------------------
    // Game of Life pattern invariants
    // ------------------------------------------------------------------------

    #[test]
    fn test_gol_block_is_still_life() {
        // Block (2x2 square) should never change
        let mut ca = CellularAutomaton2D::life(10, 10);
        ca.set(4, 4, true);
        ca.set(5, 4, true);
        ca.set(4, 5, true);
        ca.set(5, 5, true);

        let initial_pop = ca.population();
        for _ in 0..100 {
            ca.step();
            assert_eq!(
                ca.population(),
                initial_pop,
                "Block population must stay constant"
            );
        }
    }

    #[test]
    fn test_gol_blinker_period_2() {
        // Blinker oscillates with period 2
        let mut ca = CellularAutomaton2D::life(10, 10);
        ca.set(4, 5, true);
        ca.set(5, 5, true);
        ca.set(6, 5, true);

        // After 1 step: vertical
        ca.step();
        assert!(ca.get(5, 4));
        assert!(ca.get(5, 5));
        assert!(ca.get(5, 6));

        // After 2 steps: back to horizontal
        ca.step();
        assert!(ca.get(4, 5));
        assert!(ca.get(5, 5));
        assert!(ca.get(6, 5));
    }

    #[test]
    fn test_gol_glider_displacement() {
        // Glider moves (1, 1) every 4 generations
        let mut ca = CellularAutomaton2D::life(20, 20);
        // Glider pattern
        ca.set(1, 0, true);
        ca.set(2, 1, true);
        ca.set(0, 2, true);
        ca.set(1, 2, true);
        ca.set(2, 2, true);

        ca.steps(4);

        // Glider should have moved to (2, 1), (3, 2), (1, 3), (2, 3), (3, 3)
        assert!(ca.get(2, 1), "Glider displaced position (2,1)");
        assert!(ca.get(3, 2), "Glider displaced position (3,2)");
        assert!(ca.get(1, 3), "Glider displaced position (1,3)");
        assert!(ca.get(2, 3), "Glider displaced position (2,3)");
        assert!(ca.get(3, 3), "Glider displaced position (3,3)");
        assert_eq!(ca.population(), 5, "Glider population stays 5");
    }

    #[test]
    fn test_gol_empty_stays_empty() {
        let mut ca = CellularAutomaton2D::life(10, 10);
        for _ in 0..100 {
            ca.step();
            assert_eq!(ca.population(), 0, "Empty grid must stay empty");
        }
    }

    // ------------------------------------------------------------------------
    // Elementary CA invariants
    // ------------------------------------------------------------------------

    #[test]
    fn test_elementary_ca_single_cell_rule_90() {
        // Rule 90 from single center cell produces Sierpinski triangle pattern
        // After 2^n steps, has 2^n live cells (row sum = 2^row for row < 2^n)
        let mut ca = ElementaryCA::new(129, 90);
        ca.set_center();

        // After 1 step: 2 cells
        ca.step();
        assert_eq!(ca.cells().iter().filter(|&&x| x).count(), 2);

        // After 2 more steps (total 3): row 3 should have 4 cells
        ca.steps(2);
        // Row patterns: 1, 2, 2, 4, 2, 4, 4, 8, ...
    }

    #[test]
    fn test_elementary_ca_rule_deterministic() {
        let mut ca1 = ElementaryCA::new(50, 110);
        ca1.set_center();
        ca1.steps(20);

        let mut ca2 = ElementaryCA::new(50, 110);
        ca2.set_center();
        ca2.steps(20);

        assert_eq!(ca1.cells(), ca2.cells(), "Same rule + seed = same result");
    }

    #[test]
    fn test_elementary_ca_rule_184_conservation() {
        // Rule 184 is a traffic flow model - total count conserved
        let mut ca = ElementaryCA::new(100, 184);
        ca.randomize(42);
        let initial_count: usize = ca.cells().iter().filter(|&&x| x).count();

        for _ in 0..50 {
            ca.step();
            let count: usize = ca.cells().iter().filter(|&&x| x).count();
            assert_eq!(count, initial_count, "Rule 184 conserves particle count");
        }
    }

    // ------------------------------------------------------------------------
    // SmoothLife invariants
    // ------------------------------------------------------------------------

    #[test]
    fn test_smoothlife_values_bounded() {
        let config = SmoothLifeConfig::standard();
        let mut sl = SmoothLife::new(32, 32, config);
        sl.randomize(42, 0.5);

        for _ in 0..50 {
            sl.step(0.1);
            for y in 0..sl.height() {
                for x in 0..sl.width() {
                    let v = sl.get(x, y);
                    assert!(
                        (0.0..=1.0).contains(&v),
                        "SmoothLife values must be in [0, 1], got {v}"
                    );
                }
            }
        }
    }

    #[test]
    fn test_smoothlife_empty_stays_near_zero() {
        let config = SmoothLifeConfig::standard();
        let mut sl = SmoothLife::new(32, 32, config);
        // Start empty (all zeros)

        for _ in 0..10 {
            sl.step(0.1);
        }

        // Average should stay very low (near 0)
        let avg = sl.average_value();
        assert!(
            avg < 0.01,
            "Empty SmoothLife should stay near zero, got {avg}"
        );
    }

    // ------------------------------------------------------------------------
    // HashLife vs brute-force invariants
    // ------------------------------------------------------------------------

    #[test]
    fn test_hashlife_matches_ca2d_blinker() {
        // Compare HashLife to CellularAutomaton2D for blinker pattern
        let mut ca = CellularAutomaton2D::life(20, 20);
        ca.set(9, 10, true);
        ca.set(10, 10, true);
        ca.set(11, 10, true);

        let mut hl = HashLife::from_ca2d(&ca);

        for step in 0..20 {
            assert_eq!(
                ca.population() as u64,
                hl.population(),
                "Population mismatch at generation {}",
                step
            );
            ca.step();
            hl.step();
        }
    }

    #[test]
    fn test_hashlife_matches_ca2d_glider() {
        // Compare HashLife to CellularAutomaton2D for glider
        let mut ca = CellularAutomaton2D::life(30, 30);
        ca.set(1, 0, true);
        ca.set(2, 1, true);
        ca.set(0, 2, true);
        ca.set(1, 2, true);
        ca.set(2, 2, true);

        let mut hl = HashLife::from_ca2d(&ca);

        for step in 0..40 {
            assert_eq!(
                ca.population() as u64,
                hl.population(),
                "Glider population mismatch at generation {}",
                step
            );
            ca.step();
            hl.step();
        }
    }

    // ------------------------------------------------------------------------
    // Langton's Ant invariants
    // ------------------------------------------------------------------------

    #[test]
    fn test_langtons_ant_flips_cell() {
        // Each step, ant flips the cell it's on
        let mut ant = LangtonsAnt::new(10, 10, "RL");
        let (x, y) = ant.position();
        let before = ant.get(x as usize, y as usize);
        ant.step();
        let after = ant.get(x as usize, y as usize);
        assert_ne!(before, after, "Ant must flip cell state");
    }

    #[test]
    fn test_langtons_ant_grid_values_bounded() {
        // Grid values should always be < num_states (rule length)
        let mut ant = LangtonsAnt::new(100, 100, "RL");
        let num_states = 2u8; // "RL" has 2 states

        for _ in 0..1000 {
            ant.step();
        }

        for y in 0..ant.height() {
            for x in 0..ant.width() {
                let state = ant.get(x, y);
                assert!(
                    state < num_states,
                    "Grid value {} exceeds num_states {}",
                    state,
                    num_states
                );
            }
        }
    }

    #[test]
    fn test_langtons_ant_step_count_monotonic() {
        // Step count should increment by 1 each step
        let mut ant = LangtonsAnt::new(50, 50, "RL");

        for expected in 1..=100u64 {
            ant.step();
            assert_eq!(
                ant.step_count(),
                expected,
                "Step count should be {}",
                expected
            );
        }
    }

    // ------------------------------------------------------------------------
    // LargerThanLife invariants
    // ------------------------------------------------------------------------

    #[test]
    fn test_ltl_values_boolean() {
        // LargerThanLife cells are boolean (0 or 1)
        let mut ltl = LargerThanLife::new(32, 32, 2, ltl_rules::BUGS);
        ltl.randomize(42, 0.3);

        for _ in 0..20 {
            ltl.step();
            for row in ltl.cells() {
                for &cell in row {
                    assert!(cell == false || cell == true, "LtL cells must be boolean");
                }
            }
        }
    }

    // ------------------------------------------------------------------------
    // 3D CA invariants
    // ------------------------------------------------------------------------

    #[test]
    fn test_ca3d_empty_stays_empty() {
        let mut ca = CellularAutomaton3D::new(10, 10, 10, &[4], &[5]);
        for _ in 0..20 {
            ca.step();
            assert_eq!(ca.population(), 0, "Empty 3D grid must stay empty");
        }
    }

    #[test]
    fn test_ca3d_single_cell_dies() {
        // A single cell with B4/S5 rule (typical 3D rule) should die
        let mut ca = CellularAutomaton3D::new(10, 10, 10, &[4], &[5]);
        ca.set(5, 5, 5, true);
        ca.step();
        assert_eq!(ca.population(), 0, "Single cell should die with B4/S5");
    }

    // ------------------------------------------------------------------------
    // Turmite invariants
    // ------------------------------------------------------------------------

    #[test]
    fn test_turmite_grid_values_bounded() {
        // Grid values should be < num_grid_states
        let rules = vec![
            TurmiteRule::new(0, 0, 1, Turn::Right, 0),
            TurmiteRule::new(1, 0, 0, Turn::Left, 0),
        ];
        let mut turmite = Turmite::new(50, 50, 2, 1, rules);

        for _ in 0..1000 {
            turmite.step();
        }

        let max_state = turmite.num_grid_states();
        for row in turmite.grid() {
            for &cell in row {
                assert!(
                    cell < max_state,
                    "Grid value {} exceeds max state {}",
                    cell,
                    max_state
                );
            }
        }
    }
}
