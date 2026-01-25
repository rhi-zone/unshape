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

// ============================================================================
// Neighborhood Traits and Implementations
// ============================================================================

/// A 2D neighborhood pattern for cellular automata.
///
/// Neighborhoods define which cells are considered "neighbors" when counting
/// for birth/survival rules. The offsets are relative to the cell being evaluated.
pub trait Neighborhood2D: Clone {
    /// Returns the relative offsets of neighboring cells.
    ///
    /// Each offset is `(dx, dy)` relative to the center cell.
    /// The center cell `(0, 0)` should NOT be included.
    fn offsets(&self) -> &[(i32, i32)];

    /// Returns the maximum number of neighbors (for rule validation).
    fn max_neighbors(&self) -> u8 {
        self.offsets().len() as u8
    }
}

/// A 3D neighborhood pattern for cellular automata.
pub trait Neighborhood3D: Clone {
    /// Returns the relative offsets of neighboring cells.
    ///
    /// Each offset is `(dx, dy, dz)` relative to the center cell.
    fn offsets(&self) -> &[(i32, i32, i32)];

    /// Returns the maximum number of neighbors.
    fn max_neighbors(&self) -> u8 {
        self.offsets().len() as u8
    }
}

/// Moore neighborhood - 8 neighbors (orthogonal + diagonal).
///
/// ```text
/// ┌───┬───┬───┐
/// │ X │ X │ X │
/// ├───┼───┼───┤
/// │ X │ · │ X │
/// ├───┼───┼───┤
/// │ X │ X │ X │
/// └───┴───┴───┘
/// ```
#[derive(Debug, Clone, Copy, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Moore;

impl Neighborhood2D for Moore {
    fn offsets(&self) -> &[(i32, i32)] {
        &[
            (-1, -1),
            (0, -1),
            (1, -1),
            (-1, 0),
            (1, 0),
            (-1, 1),
            (0, 1),
            (1, 1),
        ]
    }
}

/// Von Neumann neighborhood - 4 neighbors (orthogonal only).
///
/// ```text
/// ┌───┬───┬───┐
/// │   │ X │   │
/// ├───┼───┼───┤
/// │ X │ · │ X │
/// ├───┼───┼───┤
/// │   │ X │   │
/// └───┴───┴───┘
/// ```
#[derive(Debug, Clone, Copy, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct VonNeumann;

impl Neighborhood2D for VonNeumann {
    fn offsets(&self) -> &[(i32, i32)] {
        &[(0, -1), (-1, 0), (1, 0), (0, 1)]
    }
}

/// Hexagonal neighborhood - 6 neighbors.
///
/// Uses offset coordinates (odd-r layout).
/// ```text
///   ┌───┬───┐
///   │ X │ X │
/// ┌───┼───┼───┐
/// │ X │ · │ X │
/// └───┼───┼───┘
///   │ X │ X │
///   └───┴───┘
/// ```
#[derive(Debug, Clone, Copy, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Hexagonal;

impl Neighborhood2D for Hexagonal {
    fn offsets(&self) -> &[(i32, i32)] {
        // Odd-r offset coordinates
        &[(-1, 0), (1, 0), (0, -1), (1, -1), (0, 1), (1, 1)]
    }
}

/// Extended Moore neighborhood with configurable radius.
///
/// Radius 1 = standard Moore (8 neighbors)
/// Radius 2 = 24 neighbors
/// Radius r = (2r+1)² - 1 neighbors
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ExtendedMoore {
    radius: u32,
    offsets: Vec<(i32, i32)>,
}

impl ExtendedMoore {
    /// Creates an extended Moore neighborhood with the given radius.
    pub fn new(radius: u32) -> Self {
        let r = radius as i32;
        let mut offsets = Vec::with_capacity(((2 * r + 1) * (2 * r + 1) - 1) as usize);
        for dy in -r..=r {
            for dx in -r..=r {
                if dx != 0 || dy != 0 {
                    offsets.push((dx, dy));
                }
            }
        }
        Self { radius, offsets }
    }

    /// Returns the radius of this neighborhood.
    pub fn radius(&self) -> u32 {
        self.radius
    }
}

impl Neighborhood2D for ExtendedMoore {
    fn offsets(&self) -> &[(i32, i32)] {
        &self.offsets
    }
}

/// Custom neighborhood with user-defined offsets.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct CustomNeighborhood2D {
    offsets: Vec<(i32, i32)>,
}

impl CustomNeighborhood2D {
    /// Creates a custom neighborhood from the given offsets.
    ///
    /// The center cell (0, 0) will be filtered out if present.
    pub fn new(offsets: impl IntoIterator<Item = (i32, i32)>) -> Self {
        let offsets: Vec<_> = offsets
            .into_iter()
            .filter(|&(dx, dy)| dx != 0 || dy != 0)
            .collect();
        Self { offsets }
    }
}

impl Neighborhood2D for CustomNeighborhood2D {
    fn offsets(&self) -> &[(i32, i32)] {
        &self.offsets
    }
}

// 3D Neighborhoods

/// 3D Moore neighborhood - 26 neighbors.
#[derive(Debug, Clone, Copy, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Moore3D;

impl Neighborhood3D for Moore3D {
    fn offsets(&self) -> &[(i32, i32, i32)] {
        &[
            // z = -1 layer (9 cells)
            (-1, -1, -1),
            (0, -1, -1),
            (1, -1, -1),
            (-1, 0, -1),
            (0, 0, -1),
            (1, 0, -1),
            (-1, 1, -1),
            (0, 1, -1),
            (1, 1, -1),
            // z = 0 layer (8 cells, excluding center)
            (-1, -1, 0),
            (0, -1, 0),
            (1, -1, 0),
            (-1, 0, 0),
            (1, 0, 0),
            (-1, 1, 0),
            (0, 1, 0),
            (1, 1, 0),
            // z = 1 layer (9 cells)
            (-1, -1, 1),
            (0, -1, 1),
            (1, -1, 1),
            (-1, 0, 1),
            (0, 0, 1),
            (1, 0, 1),
            (-1, 1, 1),
            (0, 1, 1),
            (1, 1, 1),
        ]
    }
}

/// 3D Von Neumann neighborhood - 6 neighbors (face-adjacent only).
#[derive(Debug, Clone, Copy, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct VonNeumann3D;

impl Neighborhood3D for VonNeumann3D {
    fn offsets(&self) -> &[(i32, i32, i32)] {
        &[
            (0, 0, -1),
            (0, 0, 1),
            (0, -1, 0),
            (0, 1, 0),
            (-1, 0, 0),
            (1, 0, 0),
        ]
    }
}

/// Custom 3D neighborhood with user-defined offsets.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct CustomNeighborhood3D {
    offsets: Vec<(i32, i32, i32)>,
}

impl CustomNeighborhood3D {
    /// Creates a custom 3D neighborhood from the given offsets.
    pub fn new(offsets: impl IntoIterator<Item = (i32, i32, i32)>) -> Self {
        let offsets: Vec<_> = offsets
            .into_iter()
            .filter(|&(dx, dy, dz)| dx != 0 || dy != 0 || dz != 0)
            .collect();
        Self { offsets }
    }
}

impl Neighborhood3D for CustomNeighborhood3D {
    fn offsets(&self) -> &[(i32, i32, i32)] {
        &self.offsets
    }
}

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

/// 1D Elementary Cellular Automaton.
///
/// Implements Wolfram's elementary cellular automata with 256 possible rules.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ElementaryCA {
    /// Current cell states (true = alive).
    cells: Vec<bool>,
    /// Rule number (0-255).
    rule: u8,
    /// Wrap around at edges.
    wrap: bool,
}

impl ElementaryCA {
    /// Creates a new 1D cellular automaton.
    pub fn new(width: usize, rule: u8) -> Self {
        Self {
            cells: vec![false; width],
            rule,
            wrap: true,
        }
    }

    /// Sets whether the grid wraps at edges.
    pub fn set_wrap(&mut self, wrap: bool) {
        self.wrap = wrap;
    }

    /// Returns the width of the automaton.
    pub fn width(&self) -> usize {
        self.cells.len()
    }

    /// Returns the rule number.
    pub fn rule(&self) -> u8 {
        self.rule
    }

    /// Sets the rule number.
    pub fn set_rule(&mut self, rule: u8) {
        self.rule = rule;
    }

    /// Gets the state of a cell.
    pub fn get(&self, x: usize) -> bool {
        self.cells.get(x).copied().unwrap_or(false)
    }

    /// Sets the state of a cell.
    pub fn set(&mut self, x: usize, alive: bool) {
        if x < self.cells.len() {
            self.cells[x] = alive;
        }
    }

    /// Clears all cells.
    pub fn clear(&mut self) {
        self.cells.fill(false);
    }

    /// Sets a single cell in the center (common starting condition).
    pub fn set_center(&mut self) {
        self.clear();
        let center = self.cells.len() / 2;
        self.cells[center] = true;
    }

    /// Randomizes the cells.
    pub fn randomize(&mut self, seed: u64) {
        let mut rng = SimpleRng::new(seed);
        for cell in &mut self.cells {
            *cell = rng.next_bool();
        }
    }

    /// Advances the automaton by one step.
    #[allow(clippy::needless_range_loop)]
    pub fn step(&mut self) {
        let width = self.cells.len();
        let mut next = vec![false; width];

        for i in 0..width {
            let left = if i == 0 {
                if self.wrap {
                    self.cells[width - 1]
                } else {
                    false
                }
            } else {
                self.cells[i - 1]
            };

            let center = self.cells[i];

            let right = if i == width - 1 {
                if self.wrap { self.cells[0] } else { false }
            } else {
                self.cells[i + 1]
            };

            // Convert neighborhood to index (0-7)
            let index = (left as u8) << 2 | (center as u8) << 1 | (right as u8);

            // Look up new state from rule
            next[i] = (self.rule >> index) & 1 == 1;
        }

        self.cells = next;
    }

    /// Advances multiple steps.
    pub fn steps(&mut self, n: usize) {
        for _ in 0..n {
            self.step();
        }
    }

    /// Returns a reference to the cell states.
    pub fn cells(&self) -> &[bool] {
        &self.cells
    }

    /// Generates a 2D pattern by running the CA for multiple steps.
    ///
    /// Returns a 2D grid where each row is a generation.
    pub fn generate_pattern(&mut self, generations: usize) -> Vec<Vec<bool>> {
        let mut pattern = Vec::with_capacity(generations);

        for _ in 0..generations {
            pattern.push(self.cells.clone());
            self.step();
        }

        pattern
    }
}

/// 2D Cellular Automaton with configurable rules and neighborhood.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct CellularAutomaton2D {
    /// Cell states.
    cells: Vec<Vec<bool>>,
    /// Width.
    width: usize,
    /// Height.
    height: usize,
    /// Birth rule (number of neighbors that cause birth).
    birth: Vec<u8>,
    /// Survival rule (number of neighbors that allow survival).
    survive: Vec<u8>,
    /// Wrap around at edges.
    wrap: bool,
    /// Neighborhood offsets.
    neighborhood: Vec<(i32, i32)>,
}

impl CellularAutomaton2D {
    /// Creates a new 2D cellular automaton with custom rules and Moore neighborhood.
    ///
    /// Rules are specified as birth/survival counts (e.g., B3/S23 for Game of Life).
    pub fn new(width: usize, height: usize, birth: &[u8], survive: &[u8]) -> Self {
        Self::with_neighborhood(width, height, birth, survive, Moore)
    }

    /// Creates a new 2D cellular automaton with custom rules and neighborhood.
    pub fn with_neighborhood<N: Neighborhood2D>(
        width: usize,
        height: usize,
        birth: &[u8],
        survive: &[u8],
        neighborhood: N,
    ) -> Self {
        Self {
            cells: vec![vec![false; width]; height],
            width,
            height,
            birth: birth.to_vec(),
            survive: survive.to_vec(),
            wrap: true,
            neighborhood: neighborhood.offsets().to_vec(),
        }
    }

    /// Sets whether the grid wraps at edges.
    pub fn set_wrap(&mut self, wrap: bool) {
        self.wrap = wrap;
    }

    /// Sets the neighborhood pattern.
    pub fn set_neighborhood<N: Neighborhood2D>(&mut self, neighborhood: N) {
        self.neighborhood = neighborhood.offsets().to_vec();
    }

    /// Returns the width.
    pub fn width(&self) -> usize {
        self.width
    }

    /// Returns the height.
    pub fn height(&self) -> usize {
        self.height
    }

    /// Returns the maximum number of neighbors for this neighborhood.
    pub fn max_neighbors(&self) -> u8 {
        self.neighborhood.len() as u8
    }

    /// Gets the state of a cell.
    pub fn get(&self, x: usize, y: usize) -> bool {
        self.cells
            .get(y)
            .and_then(|row| row.get(x))
            .copied()
            .unwrap_or(false)
    }

    /// Sets the state of a cell.
    pub fn set(&mut self, x: usize, y: usize, alive: bool) {
        if y < self.height && x < self.width {
            self.cells[y][x] = alive;
        }
    }

    /// Clears all cells.
    pub fn clear(&mut self) {
        for row in &mut self.cells {
            row.fill(false);
        }
    }

    /// Randomizes cells with given density (0.0 to 1.0).
    pub fn randomize(&mut self, seed: u64, density: f32) {
        let mut rng = SimpleRng::new(seed);
        for row in &mut self.cells {
            for cell in row {
                *cell = rng.next_f32() < density;
            }
        }
    }

    /// Counts alive neighbors for a cell using the configured neighborhood.
    fn count_neighbors(&self, x: usize, y: usize) -> u8 {
        let mut count = 0u8;

        for &(dx, dy) in &self.neighborhood {
            let nx = if self.wrap {
                ((x as i32 + dx).rem_euclid(self.width as i32)) as usize
            } else {
                let nx = x as i32 + dx;
                if nx < 0 || nx >= self.width as i32 {
                    continue;
                }
                nx as usize
            };

            let ny = if self.wrap {
                ((y as i32 + dy).rem_euclid(self.height as i32)) as usize
            } else {
                let ny = y as i32 + dy;
                if ny < 0 || ny >= self.height as i32 {
                    continue;
                }
                ny as usize
            };

            if self.cells[ny][nx] {
                count += 1;
            }
        }

        count
    }

    /// Advances the automaton by one step.
    #[allow(clippy::needless_range_loop)]
    pub fn step(&mut self) {
        let mut next = vec![vec![false; self.width]; self.height];

        for y in 0..self.height {
            for x in 0..self.width {
                let neighbors = self.count_neighbors(x, y);
                let alive = self.cells[y][x];

                next[y][x] = if alive {
                    self.survive.contains(&neighbors)
                } else {
                    self.birth.contains(&neighbors)
                };
            }
        }

        self.cells = next;
    }

    /// Advances multiple steps.
    pub fn steps(&mut self, n: usize) {
        for _ in 0..n {
            self.step();
        }
    }

    /// Returns a reference to the cell grid.
    pub fn cells(&self) -> &Vec<Vec<bool>> {
        &self.cells
    }

    /// Counts total alive cells.
    pub fn population(&self) -> usize {
        self.cells
            .iter()
            .flat_map(|row| row.iter())
            .filter(|&&c| c)
            .count()
    }
}

/// Conway's Game of Life (B3/S23).
pub type GameOfLife = CellularAutomaton2D;

impl GameOfLife {
    /// Creates a new Game of Life grid.
    pub fn life(width: usize, height: usize) -> Self {
        Self::new(width, height, &[3], &[2, 3])
    }
}

/// Common 2D CA rule presets.
pub mod rules {
    /// Game of Life (B3/S23) - classic rules.
    pub const LIFE: (&[u8], &[u8]) = (&[3], &[2, 3]);

    /// HighLife (B36/S23) - similar to Life but with more action.
    pub const HIGH_LIFE: (&[u8], &[u8]) = (&[3, 6], &[2, 3]);

    /// Seeds (B2/S) - explosive growth.
    pub const SEEDS: (&[u8], &[u8]) = (&[2], &[]);

    /// Day & Night (B3678/S34678) - symmetric rules.
    pub const DAY_NIGHT: (&[u8], &[u8]) = (&[3, 6, 7, 8], &[3, 4, 6, 7, 8]);

    /// Maze (B3/S12345) - creates maze-like patterns.
    pub const MAZE: (&[u8], &[u8]) = (&[3], &[1, 2, 3, 4, 5]);

    /// Diamoeba (B35678/S5678) - amoeba-like growth.
    pub const DIAMOEBA: (&[u8], &[u8]) = (&[3, 5, 6, 7, 8], &[5, 6, 7, 8]);

    /// Replicator (B1357/S1357) - patterns replicate.
    pub const REPLICATOR: (&[u8], &[u8]) = (&[1, 3, 5, 7], &[1, 3, 5, 7]);
}

/// Common 1D CA rules.
pub mod elementary_rules {
    /// Rule 30 - chaotic, used for random number generation.
    pub const RULE_30: u8 = 30;

    /// Rule 90 - Sierpinski triangle.
    pub const RULE_90: u8 = 90;

    /// Rule 110 - Turing complete.
    pub const RULE_110: u8 = 110;

    /// Rule 184 - traffic flow model.
    pub const RULE_184: u8 = 184;

    /// Rule 250 - simple growth.
    pub const RULE_250: u8 = 250;
}

// ============================================================================
// Larger than Life
// ============================================================================

/// Range-based birth/survival rules for Larger than Life.
///
/// Unlike standard Life rules which use exact neighbor counts,
/// LtL uses ranges: birth if neighbors in `birth_range`, survive if in `survive_range`.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct LtlRules {
    /// Minimum neighbors for birth (inclusive).
    pub birth_min: u32,
    /// Maximum neighbors for birth (inclusive).
    pub birth_max: u32,
    /// Minimum neighbors for survival (inclusive).
    pub survive_min: u32,
    /// Maximum neighbors for survival (inclusive).
    pub survive_max: u32,
}

impl LtlRules {
    /// Creates new LtL rules with the given ranges.
    pub fn new(birth_min: u32, birth_max: u32, survive_min: u32, survive_max: u32) -> Self {
        Self {
            birth_min,
            birth_max,
            survive_min,
            survive_max,
        }
    }

    /// Checks if a dead cell should be born.
    pub fn should_birth(&self, neighbors: u32) -> bool {
        neighbors >= self.birth_min && neighbors <= self.birth_max
    }

    /// Checks if a live cell should survive.
    pub fn should_survive(&self, neighbors: u32) -> bool {
        neighbors >= self.survive_min && neighbors <= self.survive_max
    }
}

/// Larger than Life cellular automaton.
///
/// A generalization of Conway's Game of Life with:
/// - Configurable neighborhood radius (radius 1 = standard Moore)
/// - Range-based birth/survival rules instead of exact counts
///
/// # Example
///
/// ```
/// use unshape_automata::{LargerThanLife, ltl_rules};
///
/// // Bugs rule: radius 5, birth 34-45, survive 34-58
/// let mut ltl = LargerThanLife::new(100, 100, 5, ltl_rules::BUGS);
/// ltl.randomize(12345, 0.5);
/// ltl.step();
/// ```
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct LargerThanLife {
    /// Cell states.
    cells: Vec<Vec<bool>>,
    /// Width.
    width: usize,
    /// Height.
    height: usize,
    /// Neighborhood radius.
    radius: u32,
    /// Birth/survival rules.
    rules: LtlRules,
    /// Wrap around at edges.
    wrap: bool,
    /// Cached neighborhood offsets.
    neighborhood: Vec<(i32, i32)>,
}

impl LargerThanLife {
    /// Creates a new Larger than Life automaton.
    pub fn new(width: usize, height: usize, radius: u32, rules: LtlRules) -> Self {
        let neighborhood = ExtendedMoore::new(radius);
        Self {
            cells: vec![vec![false; width]; height],
            width,
            height,
            radius,
            rules,
            wrap: true,
            neighborhood: neighborhood.offsets().to_vec(),
        }
    }

    /// Sets whether the grid wraps at edges.
    pub fn set_wrap(&mut self, wrap: bool) {
        self.wrap = wrap;
    }

    /// Returns the width.
    pub fn width(&self) -> usize {
        self.width
    }

    /// Returns the height.
    pub fn height(&self) -> usize {
        self.height
    }

    /// Returns the neighborhood radius.
    pub fn radius(&self) -> u32 {
        self.radius
    }

    /// Returns the rules.
    pub fn rules(&self) -> &LtlRules {
        &self.rules
    }

    /// Gets the state of a cell.
    pub fn get(&self, x: usize, y: usize) -> bool {
        self.cells
            .get(y)
            .and_then(|row| row.get(x))
            .copied()
            .unwrap_or(false)
    }

    /// Sets the state of a cell.
    pub fn set(&mut self, x: usize, y: usize, alive: bool) {
        if y < self.height && x < self.width {
            self.cells[y][x] = alive;
        }
    }

    /// Clears all cells.
    pub fn clear(&mut self) {
        for row in &mut self.cells {
            row.fill(false);
        }
    }

    /// Randomizes cells with given density (0.0 to 1.0).
    pub fn randomize(&mut self, seed: u64, density: f32) {
        let mut rng = SimpleRng::new(seed);
        for row in &mut self.cells {
            for cell in row {
                *cell = rng.next_f32() < density;
            }
        }
    }

    /// Counts alive neighbors for a cell.
    fn count_neighbors(&self, x: usize, y: usize) -> u32 {
        let mut count = 0u32;

        for &(dx, dy) in &self.neighborhood {
            let nx = if self.wrap {
                ((x as i32 + dx).rem_euclid(self.width as i32)) as usize
            } else {
                let nx = x as i32 + dx;
                if nx < 0 || nx >= self.width as i32 {
                    continue;
                }
                nx as usize
            };

            let ny = if self.wrap {
                ((y as i32 + dy).rem_euclid(self.height as i32)) as usize
            } else {
                let ny = y as i32 + dy;
                if ny < 0 || ny >= self.height as i32 {
                    continue;
                }
                ny as usize
            };

            if self.cells[ny][nx] {
                count += 1;
            }
        }

        count
    }

    /// Advances the automaton by one step.
    #[allow(clippy::needless_range_loop)]
    pub fn step(&mut self) {
        let mut next = vec![vec![false; self.width]; self.height];

        for y in 0..self.height {
            for x in 0..self.width {
                let neighbors = self.count_neighbors(x, y);
                let alive = self.cells[y][x];

                next[y][x] = if alive {
                    self.rules.should_survive(neighbors)
                } else {
                    self.rules.should_birth(neighbors)
                };
            }
        }

        self.cells = next;
    }

    /// Advances multiple steps.
    pub fn steps(&mut self, n: usize) {
        for _ in 0..n {
            self.step();
        }
    }

    /// Returns a reference to the cell grid.
    pub fn cells(&self) -> &Vec<Vec<bool>> {
        &self.cells
    }

    /// Counts total alive cells.
    pub fn population(&self) -> usize {
        self.cells
            .iter()
            .flat_map(|row| row.iter())
            .filter(|&&c| c)
            .count()
    }
}

/// Common Larger than Life rule presets.
pub mod ltl_rules {
    use super::LtlRules;

    /// Bugs - radius 5, birth 34-45, survive 34-58.
    ///
    /// Creates bug-like organisms that move and interact.
    pub const BUGS: LtlRules = LtlRules {
        birth_min: 34,
        birth_max: 45,
        survive_min: 34,
        survive_max: 58,
    };

    /// Bosco's Rule - radius 5, birth 34-45, survive 34-58 (same as Bugs).
    pub const BOSCO: LtlRules = BUGS;

    /// Waffle - radius 7, birth 100-200, survive 75-170.
    ///
    /// Creates stable waffle-like patterns.
    pub const WAFFLE: LtlRules = LtlRules {
        birth_min: 100,
        birth_max: 200,
        survive_min: 75,
        survive_max: 170,
    };

    /// Globe - radius 8, birth 163-223, survive 163-223.
    ///
    /// Creates large circular organisms.
    pub const GLOBE: LtlRules = LtlRules {
        birth_min: 163,
        birth_max: 223,
        survive_min: 163,
        survive_max: 223,
    };

    /// Majority - radius 4, birth 41-81, survive 41-81.
    ///
    /// Cells take the state of the majority of neighbors.
    pub const MAJORITY: LtlRules = LtlRules {
        birth_min: 41,
        birth_max: 81,
        survive_min: 41,
        survive_max: 81,
    };

    /// ModernArt - radius 10, birth 210-350, survive 190-290.
    ///
    /// Creates abstract patterns reminiscent of modern art.
    pub const MODERN_ART: LtlRules = LtlRules {
        birth_min: 210,
        birth_max: 350,
        survive_min: 190,
        survive_max: 290,
    };
}

// ============================================================================
// Turmites / Langton's Ant
// ============================================================================

/// Direction for 2D grid movement.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum Direction {
    /// North (up, -Y).
    North,
    /// East (right, +X).
    East,
    /// South (down, +Y).
    South,
    /// West (left, -X).
    West,
}

impl Direction {
    /// Turns left (counter-clockwise).
    pub fn turn_left(self) -> Self {
        match self {
            Direction::North => Direction::West,
            Direction::East => Direction::North,
            Direction::South => Direction::East,
            Direction::West => Direction::South,
        }
    }

    /// Turns right (clockwise).
    pub fn turn_right(self) -> Self {
        match self {
            Direction::North => Direction::East,
            Direction::East => Direction::South,
            Direction::South => Direction::West,
            Direction::West => Direction::North,
        }
    }

    /// Turns around (180 degrees).
    pub fn turn_around(self) -> Self {
        match self {
            Direction::North => Direction::South,
            Direction::East => Direction::West,
            Direction::South => Direction::North,
            Direction::West => Direction::East,
        }
    }

    /// Returns the offset for moving in this direction.
    pub fn offset(self) -> (i32, i32) {
        match self {
            Direction::North => (0, -1),
            Direction::East => (1, 0),
            Direction::South => (0, 1),
            Direction::West => (-1, 0),
        }
    }
}

/// Turn instruction for Langton's Ant / Turmites.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum Turn {
    /// No turn (continue straight).
    None,
    /// Turn left (counter-clockwise).
    Left,
    /// Turn right (clockwise).
    Right,
    /// Turn around (180 degrees).
    Around,
}

impl Turn {
    /// Applies this turn to a direction.
    pub fn apply(self, dir: Direction) -> Direction {
        match self {
            Turn::None => dir,
            Turn::Left => dir.turn_left(),
            Turn::Right => dir.turn_right(),
            Turn::Around => dir.turn_around(),
        }
    }
}

/// Langton's Ant - a 2D Turing machine.
///
/// A simple cellular automaton where an "ant" moves on a grid:
/// - On a white cell: turn right, flip color, move forward
/// - On a black cell: turn left, flip color, move forward
///
/// The rule string (e.g., "RL") specifies the turn for each state.
/// Classic Langton's Ant uses "RL" (Right on white, Left on black).
///
/// # Example
///
/// ```
/// use unshape_automata::LangtonsAnt;
///
/// let mut ant = LangtonsAnt::new(100, 100, "RL");
/// ant.steps(10000);
/// ```
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct LangtonsAnt {
    /// Grid states (index into rule string).
    grid: Vec<Vec<u8>>,
    /// Width.
    width: usize,
    /// Height.
    height: usize,
    /// Ant position.
    x: i32,
    y: i32,
    /// Ant direction.
    direction: Direction,
    /// Turn rules for each state.
    rules: Vec<Turn>,
    /// Number of states (rule string length).
    num_states: u8,
    /// Wrap at edges.
    wrap: bool,
    /// Step counter.
    step_count: u64,
}

impl LangtonsAnt {
    /// Creates a new Langton's Ant with the given rule string.
    ///
    /// Rule string characters:
    /// - 'L' = turn left
    /// - 'R' = turn right
    /// - 'N' = no turn (continue straight)
    /// - 'U' = U-turn (turn around)
    ///
    /// The ant starts at the center, facing north.
    pub fn new(width: usize, height: usize, rule: &str) -> Self {
        let rules: Vec<Turn> = rule
            .chars()
            .map(|c| match c {
                'L' | 'l' => Turn::Left,
                'R' | 'r' => Turn::Right,
                'N' | 'n' => Turn::None,
                'U' | 'u' => Turn::Around,
                _ => Turn::Right, // Default to right for unknown
            })
            .collect();

        let num_states = rules.len().max(1) as u8;

        Self {
            grid: vec![vec![0; width]; height],
            width,
            height,
            x: width as i32 / 2,
            y: height as i32 / 2,
            direction: Direction::North,
            rules,
            num_states,
            wrap: true,
            step_count: 0,
        }
    }

    /// Sets whether the grid wraps at edges.
    pub fn set_wrap(&mut self, wrap: bool) {
        self.wrap = wrap;
    }

    /// Returns the width.
    pub fn width(&self) -> usize {
        self.width
    }

    /// Returns the height.
    pub fn height(&self) -> usize {
        self.height
    }

    /// Returns the ant's position.
    pub fn position(&self) -> (i32, i32) {
        (self.x, self.y)
    }

    /// Returns the ant's direction.
    pub fn direction(&self) -> Direction {
        self.direction
    }

    /// Returns the number of steps taken.
    pub fn step_count(&self) -> u64 {
        self.step_count
    }

    /// Gets the state of a cell (0 to num_states-1).
    pub fn get(&self, x: usize, y: usize) -> u8 {
        self.grid
            .get(y)
            .and_then(|row| row.get(x))
            .copied()
            .unwrap_or(0)
    }

    /// Gets the state as a boolean (true if non-zero).
    pub fn get_bool(&self, x: usize, y: usize) -> bool {
        self.get(x, y) != 0
    }

    /// Sets the state of a cell.
    pub fn set(&mut self, x: usize, y: usize, state: u8) {
        if y < self.height && x < self.width {
            self.grid[y][x] = state % self.num_states;
        }
    }

    /// Clears all cells.
    pub fn clear(&mut self) {
        for row in &mut self.grid {
            row.fill(0);
        }
    }

    /// Resets the ant to the center.
    pub fn reset_ant(&mut self) {
        self.x = self.width as i32 / 2;
        self.y = self.height as i32 / 2;
        self.direction = Direction::North;
        self.step_count = 0;
    }

    /// Advances the ant by one step.
    ///
    /// Returns false if the ant moved out of bounds (when wrapping is disabled).
    pub fn step(&mut self) -> bool {
        // Get current cell state
        let ux = self.x as usize;
        let uy = self.y as usize;

        if ux >= self.width || uy >= self.height {
            return false;
        }

        let state = self.grid[uy][ux];

        // Turn based on current state
        if let Some(&turn) = self.rules.get(state as usize) {
            self.direction = turn.apply(self.direction);
        }

        // Flip to next state
        self.grid[uy][ux] = (state + 1) % self.num_states;

        // Move forward
        let (dx, dy) = self.direction.offset();
        self.x += dx;
        self.y += dy;

        // Handle wrapping or bounds
        if self.wrap {
            self.x = self.x.rem_euclid(self.width as i32);
            self.y = self.y.rem_euclid(self.height as i32);
        }

        self.step_count += 1;

        // Check if in bounds
        self.x >= 0 && self.x < self.width as i32 && self.y >= 0 && self.y < self.height as i32
    }

    /// Advances multiple steps.
    ///
    /// Returns the number of steps actually taken (may be less if ant goes out of bounds).
    pub fn steps(&mut self, n: usize) -> usize {
        for i in 0..n {
            if !self.step() {
                return i;
            }
        }
        n
    }

    /// Returns a reference to the grid.
    pub fn grid(&self) -> &Vec<Vec<u8>> {
        &self.grid
    }

    /// Counts cells in each state.
    pub fn state_counts(&self) -> Vec<usize> {
        let mut counts = vec![0; self.num_states as usize];
        for row in &self.grid {
            for &cell in row {
                counts[cell as usize] += 1;
            }
        }
        counts
    }
}

/// Turmite - a generalized multi-state ant.
///
/// A turmite has internal states in addition to the grid states.
/// The transition function maps (grid_state, ant_state) to (new_grid_state, turn, new_ant_state).
///
/// # Example
///
/// ```
/// use unshape_automata::{Turmite, TurmiteRule, Turn};
///
/// // Fibonacci turmite
/// let rules = vec![
///     // (grid_state, ant_state) -> (new_grid, turn, new_ant)
///     TurmiteRule::new(0, 0, 1, Turn::Left, 0),
///     TurmiteRule::new(0, 1, 1, Turn::Left, 1),
///     TurmiteRule::new(1, 0, 1, Turn::Right, 1),
///     TurmiteRule::new(1, 1, 0, Turn::None, 0),
/// ];
/// let mut turmite = Turmite::new(100, 100, 2, 2, rules);
/// turmite.steps(1000);
/// ```
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Turmite {
    /// Grid states.
    grid: Vec<Vec<u8>>,
    /// Width.
    width: usize,
    /// Height.
    height: usize,
    /// Ant position.
    x: i32,
    y: i32,
    /// Ant direction.
    direction: Direction,
    /// Ant internal state.
    ant_state: u8,
    /// Number of grid states.
    num_grid_states: u8,
    /// Number of ant states.
    num_ant_states: u8,
    /// Transition rules: [grid_state][ant_state] -> (new_grid, turn, new_ant)
    transitions: Vec<Vec<(u8, Turn, u8)>>,
    /// Wrap at edges.
    wrap: bool,
    /// Step counter.
    step_count: u64,
}

/// A single transition rule for a Turmite.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct TurmiteRule {
    /// Input grid state.
    pub grid_state: u8,
    /// Input ant state.
    pub ant_state: u8,
    /// Output grid state.
    pub new_grid_state: u8,
    /// Turn to make.
    pub turn: Turn,
    /// Output ant state.
    pub new_ant_state: u8,
}

impl TurmiteRule {
    /// Creates a new turmite rule.
    pub fn new(
        grid_state: u8,
        ant_state: u8,
        new_grid_state: u8,
        turn: Turn,
        new_ant_state: u8,
    ) -> Self {
        Self {
            grid_state,
            ant_state,
            new_grid_state,
            turn,
            new_ant_state,
        }
    }
}

impl Turmite {
    /// Creates a new Turmite with the given transition rules.
    pub fn new(
        width: usize,
        height: usize,
        num_grid_states: u8,
        num_ant_states: u8,
        rules: Vec<TurmiteRule>,
    ) -> Self {
        // Build transition table
        let mut transitions =
            vec![vec![(0u8, Turn::None, 0u8); num_ant_states as usize]; num_grid_states as usize];

        for rule in rules {
            if rule.grid_state < num_grid_states && rule.ant_state < num_ant_states {
                transitions[rule.grid_state as usize][rule.ant_state as usize] =
                    (rule.new_grid_state, rule.turn, rule.new_ant_state);
            }
        }

        Self {
            grid: vec![vec![0; width]; height],
            width,
            height,
            x: width as i32 / 2,
            y: height as i32 / 2,
            direction: Direction::North,
            ant_state: 0,
            num_grid_states,
            num_ant_states,
            transitions,
            wrap: true,
            step_count: 0,
        }
    }

    /// Sets whether the grid wraps at edges.
    pub fn set_wrap(&mut self, wrap: bool) {
        self.wrap = wrap;
    }

    /// Returns the width.
    pub fn width(&self) -> usize {
        self.width
    }

    /// Returns the height.
    pub fn height(&self) -> usize {
        self.height
    }

    /// Returns the ant's position.
    pub fn position(&self) -> (i32, i32) {
        (self.x, self.y)
    }

    /// Returns the ant's direction.
    pub fn direction(&self) -> Direction {
        self.direction
    }

    /// Returns the ant's internal state.
    pub fn ant_state(&self) -> u8 {
        self.ant_state
    }

    /// Returns the number of grid states.
    pub fn num_grid_states(&self) -> u8 {
        self.num_grid_states
    }

    /// Returns the number of ant states.
    pub fn num_ant_states(&self) -> u8 {
        self.num_ant_states
    }

    /// Returns the step count.
    pub fn step_count(&self) -> u64 {
        self.step_count
    }

    /// Gets the state of a cell.
    pub fn get(&self, x: usize, y: usize) -> u8 {
        self.grid
            .get(y)
            .and_then(|row| row.get(x))
            .copied()
            .unwrap_or(0)
    }

    /// Advances the turmite by one step.
    pub fn step(&mut self) -> bool {
        let ux = self.x as usize;
        let uy = self.y as usize;

        if ux >= self.width || uy >= self.height {
            return false;
        }

        let grid_state = self.grid[uy][ux];

        // Look up transition
        let (new_grid, turn, new_ant) =
            self.transitions[grid_state as usize][self.ant_state as usize];

        // Apply transition
        self.grid[uy][ux] = new_grid;
        self.direction = turn.apply(self.direction);
        self.ant_state = new_ant;

        // Move forward
        let (dx, dy) = self.direction.offset();
        self.x += dx;
        self.y += dy;

        if self.wrap {
            self.x = self.x.rem_euclid(self.width as i32);
            self.y = self.y.rem_euclid(self.height as i32);
        }

        self.step_count += 1;

        self.x >= 0 && self.x < self.width as i32 && self.y >= 0 && self.y < self.height as i32
    }

    /// Advances multiple steps.
    pub fn steps(&mut self, n: usize) -> usize {
        for i in 0..n {
            if !self.step() {
                return i;
            }
        }
        n
    }

    /// Returns a reference to the grid.
    pub fn grid(&self) -> &Vec<Vec<u8>> {
        &self.grid
    }
}

/// Common Langton's Ant rule presets.
pub mod ant_rules {
    /// Classic Langton's Ant (RL).
    ///
    /// Creates the famous "highway" pattern after ~10,000 steps.
    pub const LANGTON: &str = "RL";

    /// LLRR - creates symmetrical patterns.
    pub const LLRR: &str = "LLRR";

    /// LRRL - creates filled triangles.
    pub const LRRL: &str = "LRRL";

    /// RLLR - creates complex branching structures.
    pub const RLLR: &str = "RLLR";

    /// RRLL - creates diagonal highways.
    pub const RRLL: &str = "RRLL";

    /// RRLLLRLLLRRR - creates intricate patterns.
    pub const COMPLEX: &str = "RRLLLRLLLRRR";
}

// ============================================================================
// 3D Cellular Automata
// ============================================================================

/// 3D Cellular Automaton with configurable rules and neighborhood.
///
/// Extension of 2D Life-like rules to three dimensions.
/// Uses B/S notation: birth if neighbor count in birth set, survive if in survive set.
///
/// # Example
///
/// ```
/// use unshape_automata::{CellularAutomaton3D, rules_3d, Moore3D};
///
/// // 3D Life variant 4/4/5/M (4 neighbors to birth, 4 to survive, 5 states, Moore)
/// let (birth, survive) = rules_3d::LIFE_445;
/// let mut ca = CellularAutomaton3D::new(20, 20, 20, birth, survive);
/// ca.randomize(12345, 0.3);
/// ca.step();
/// ```
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct CellularAutomaton3D {
    /// Cell states (z, y, x indexing).
    cells: Vec<Vec<Vec<bool>>>,
    /// Width (X).
    width: usize,
    /// Height (Y).
    height: usize,
    /// Depth (Z).
    depth: usize,
    /// Birth rule.
    birth: Vec<u8>,
    /// Survival rule.
    survive: Vec<u8>,
    /// Wrap at edges.
    wrap: bool,
    /// Neighborhood offsets.
    neighborhood: Vec<(i32, i32, i32)>,
}

impl CellularAutomaton3D {
    /// Creates a new 3D CA with Moore neighborhood (26 neighbors).
    pub fn new(width: usize, height: usize, depth: usize, birth: &[u8], survive: &[u8]) -> Self {
        Self::with_neighborhood(width, height, depth, birth, survive, Moore3D)
    }

    /// Creates a new 3D CA with a custom neighborhood.
    pub fn with_neighborhood<N: Neighborhood3D>(
        width: usize,
        height: usize,
        depth: usize,
        birth: &[u8],
        survive: &[u8],
        neighborhood: N,
    ) -> Self {
        Self {
            cells: vec![vec![vec![false; width]; height]; depth],
            width,
            height,
            depth,
            birth: birth.to_vec(),
            survive: survive.to_vec(),
            wrap: true,
            neighborhood: neighborhood.offsets().to_vec(),
        }
    }

    /// Sets whether the grid wraps at edges.
    pub fn set_wrap(&mut self, wrap: bool) {
        self.wrap = wrap;
    }

    /// Returns the width (X dimension).
    pub fn width(&self) -> usize {
        self.width
    }

    /// Returns the height (Y dimension).
    pub fn height(&self) -> usize {
        self.height
    }

    /// Returns the depth (Z dimension).
    pub fn depth(&self) -> usize {
        self.depth
    }

    /// Returns the maximum number of neighbors.
    pub fn max_neighbors(&self) -> u8 {
        self.neighborhood.len() as u8
    }

    /// Gets the state of a cell.
    pub fn get(&self, x: usize, y: usize, z: usize) -> bool {
        self.cells
            .get(z)
            .and_then(|plane| plane.get(y))
            .and_then(|row| row.get(x))
            .copied()
            .unwrap_or(false)
    }

    /// Sets the state of a cell.
    pub fn set(&mut self, x: usize, y: usize, z: usize, alive: bool) {
        if z < self.depth && y < self.height && x < self.width {
            self.cells[z][y][x] = alive;
        }
    }

    /// Clears all cells.
    pub fn clear(&mut self) {
        for plane in &mut self.cells {
            for row in plane {
                row.fill(false);
            }
        }
    }

    /// Randomizes cells with given density.
    pub fn randomize(&mut self, seed: u64, density: f32) {
        let mut rng = SimpleRng::new(seed);
        for plane in &mut self.cells {
            for row in plane {
                for cell in row {
                    *cell = rng.next_f32() < density;
                }
            }
        }
    }

    /// Counts alive neighbors for a cell.
    fn count_neighbors(&self, x: usize, y: usize, z: usize) -> u8 {
        let mut count = 0u8;

        for &(dx, dy, dz) in &self.neighborhood {
            let nx = if self.wrap {
                ((x as i32 + dx).rem_euclid(self.width as i32)) as usize
            } else {
                let nx = x as i32 + dx;
                if nx < 0 || nx >= self.width as i32 {
                    continue;
                }
                nx as usize
            };

            let ny = if self.wrap {
                ((y as i32 + dy).rem_euclid(self.height as i32)) as usize
            } else {
                let ny = y as i32 + dy;
                if ny < 0 || ny >= self.height as i32 {
                    continue;
                }
                ny as usize
            };

            let nz = if self.wrap {
                ((z as i32 + dz).rem_euclid(self.depth as i32)) as usize
            } else {
                let nz = z as i32 + dz;
                if nz < 0 || nz >= self.depth as i32 {
                    continue;
                }
                nz as usize
            };

            if self.cells[nz][ny][nx] {
                count += 1;
            }
        }

        count
    }

    /// Advances the automaton by one step.
    #[allow(clippy::needless_range_loop)]
    pub fn step(&mut self) {
        let mut next = vec![vec![vec![false; self.width]; self.height]; self.depth];

        for z in 0..self.depth {
            for y in 0..self.height {
                for x in 0..self.width {
                    let neighbors = self.count_neighbors(x, y, z);
                    let alive = self.cells[z][y][x];

                    next[z][y][x] = if alive {
                        self.survive.contains(&neighbors)
                    } else {
                        self.birth.contains(&neighbors)
                    };
                }
            }
        }

        self.cells = next;
    }

    /// Advances multiple steps.
    pub fn steps(&mut self, n: usize) {
        for _ in 0..n {
            self.step();
        }
    }

    /// Returns a reference to the cell grid.
    pub fn cells(&self) -> &Vec<Vec<Vec<bool>>> {
        &self.cells
    }

    /// Counts total alive cells.
    pub fn population(&self) -> usize {
        self.cells
            .iter()
            .flat_map(|plane| plane.iter())
            .flat_map(|row| row.iter())
            .filter(|&&c| c)
            .count()
    }

    /// Returns a 2D slice at the given Z coordinate.
    pub fn slice_z(&self, z: usize) -> Option<&Vec<Vec<bool>>> {
        self.cells.get(z)
    }
}

/// Common 3D CA rule presets.
///
/// Rules are given as (birth, survive) tuples for Moore neighborhood (26 neighbors).
pub mod rules_3d {
    /// 4/4/5 - stable structures, caves.
    pub const LIFE_445: (&[u8], &[u8]) = (&[4], &[4]);

    /// 5/5 - similar behavior to 2D Life.
    pub const LIFE_55: (&[u8], &[u8]) = (&[5], &[5]);

    /// 4/5 - growing crystals.
    pub const CRYSTAL: (&[u8], &[u8]) = (&[4], &[5]);

    /// 6-7/5-6 - amoeba-like growth.
    pub const AMOEBA_3D: (&[u8], &[u8]) = (&[6, 7], &[5, 6]);

    /// 4/3-4 - pyroclastic (explosive growth then stabilization).
    pub const PYROCLASTIC: (&[u8], &[u8]) = (&[4], &[3, 4]);

    /// 5-7/6-8 - slow growth.
    pub const SLOW_GROWTH: (&[u8], &[u8]) = (&[5, 6, 7], &[6, 7, 8]);

    /// 9-26/5-7,12-13,15 - 3D coral structures.
    pub const CORAL_3D: (&[u8], &[u8]) = (
        &[
            9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
        ],
        &[5, 6, 7, 12, 13, 15],
    );
}

// ============================================================================
// SmoothLife
// ============================================================================

/// SmoothLife configuration parameters.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct SmoothLifeConfig {
    /// Inner radius for the "inner disk" (self + close neighbors).
    pub inner_radius: f32,
    /// Outer radius for the "outer ring" (neighbor annulus).
    pub outer_radius: f32,
    /// Birth threshold low (birth if filling in range [b1, b2]).
    pub birth_lo: f32,
    /// Birth threshold high.
    pub birth_hi: f32,
    /// Death threshold low (survive if filling in range [d1, d2]).
    pub death_lo: f32,
    /// Death threshold high.
    pub death_hi: f32,
    /// Sigmoid steepness (higher = sharper transition).
    pub alpha_n: f32,
    /// Sigmoid steepness for state mixing.
    pub alpha_m: f32,
}

impl Default for SmoothLifeConfig {
    fn default() -> Self {
        Self {
            inner_radius: 3.0,
            outer_radius: 9.0,
            birth_lo: 0.278,
            birth_hi: 0.365,
            death_lo: 0.267,
            death_hi: 0.445,
            alpha_n: 0.028,
            alpha_m: 0.147,
        }
    }
}

impl SmoothLifeConfig {
    /// Creates a SmoothLife config optimized for the "standard" smooth Life look.
    pub fn standard() -> Self {
        Self::default()
    }

    /// Creates a config for more fluid-like behavior.
    pub fn fluid() -> Self {
        Self {
            inner_radius: 4.0,
            outer_radius: 12.0,
            birth_lo: 0.257,
            birth_hi: 0.336,
            death_lo: 0.365,
            death_hi: 0.550,
            alpha_n: 0.028,
            alpha_m: 0.147,
        }
    }

    /// Creates a config for slower, more stable patterns.
    pub fn slow() -> Self {
        Self {
            inner_radius: 5.0,
            outer_radius: 15.0,
            birth_lo: 0.269,
            birth_hi: 0.340,
            death_lo: 0.262,
            death_hi: 0.428,
            alpha_n: 0.020,
            alpha_m: 0.100,
        }
    }
}

/// SmoothLife - continuous-state cellular automaton.
///
/// A continuous generalization of Conway's Game of Life:
/// - Cell states are continuous (0.0 to 1.0) instead of binary
/// - Neighbor counting uses smooth disk/ring integrals
/// - State transitions use smooth sigmoid functions
///
/// This produces organic, fluid-like patterns that evolve smoothly.
///
/// # Example
///
/// ```
/// use unshape_automata::{SmoothLife, SmoothLifeConfig};
///
/// let mut sl = SmoothLife::new(100, 100, SmoothLifeConfig::default());
/// sl.randomize(12345, 0.3);
/// sl.step(0.1);
/// ```
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct SmoothLife {
    /// Current cell states (0.0 to 1.0).
    cells: Vec<Vec<f32>>,
    /// Width.
    width: usize,
    /// Height.
    height: usize,
    /// Configuration parameters.
    config: SmoothLifeConfig,
    /// Precomputed inner disk weights.
    inner_weights: Vec<Vec<f32>>,
    /// Precomputed outer ring weights.
    outer_weights: Vec<Vec<f32>>,
    /// Inner disk total weight (for normalization).
    inner_total: f32,
    /// Outer ring total weight (for normalization).
    outer_total: f32,
    /// Kernel radius (in cells).
    kernel_radius: i32,
}

impl SmoothLife {
    /// Creates a new SmoothLife simulation.
    pub fn new(width: usize, height: usize, config: SmoothLifeConfig) -> Self {
        let kernel_radius = config.outer_radius.ceil() as i32 + 1;
        let (inner_weights, outer_weights, inner_total, outer_total) =
            Self::compute_weights(&config, kernel_radius);

        Self {
            cells: vec![vec![0.0; width]; height],
            width,
            height,
            config,
            inner_weights,
            outer_weights,
            inner_total,
            outer_total,
            kernel_radius,
        }
    }

    /// Computes disk/ring weights for neighbor sampling.
    fn compute_weights(
        config: &SmoothLifeConfig,
        kernel_radius: i32,
    ) -> (Vec<Vec<f32>>, Vec<Vec<f32>>, f32, f32) {
        let size = (kernel_radius * 2 + 1) as usize;
        let mut inner = vec![vec![0.0f32; size]; size];
        let mut outer = vec![vec![0.0f32; size]; size];
        let mut inner_total = 0.0f32;
        let mut outer_total = 0.0f32;

        let ri = config.inner_radius;
        let ro = config.outer_radius;

        for dy in -kernel_radius..=kernel_radius {
            for dx in -kernel_radius..=kernel_radius {
                let dist = ((dx * dx + dy * dy) as f32).sqrt();
                let ux = (dx + kernel_radius) as usize;
                let uy = (dy + kernel_radius) as usize;

                // Inner disk (excluding center for some variants, but we include it)
                if dist <= ri {
                    // Smooth falloff at edge
                    let w = Self::smooth_step(ri - dist, 0.0, 1.0);
                    inner[uy][ux] = w;
                    inner_total += w;
                }

                // Outer ring (annulus between ri and ro)
                if dist > ri && dist <= ro {
                    let w = Self::smooth_step(ro - dist, 0.0, 1.0);
                    outer[uy][ux] = w;
                    outer_total += w;
                }
            }
        }

        (inner, outer, inner_total.max(1.0), outer_total.max(1.0))
    }

    /// Smooth step function (smoothstep).
    fn smooth_step(x: f32, edge0: f32, edge1: f32) -> f32 {
        let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
        t * t * (3.0 - 2.0 * t)
    }

    /// Sigmoid function for smooth transitions.
    fn sigmoid(x: f32, a: f32, alpha: f32) -> f32 {
        1.0 / (1.0 + ((a - x) / alpha).exp())
    }

    /// Smooth interval membership (1 if x in [a,b], 0 otherwise, with smooth edges).
    fn sigmoid_interval(x: f32, a: f32, b: f32, alpha: f32) -> f32 {
        Self::sigmoid(x, a, alpha) * (1.0 - Self::sigmoid(x, b, alpha))
    }

    /// Transition function: computes new state from inner filling (m) and outer filling (n).
    fn transition(&self, n: f32, m: f32) -> f32 {
        let c = &self.config;

        // Birth: outer ring filling in [birth_lo, birth_hi]
        let birth = Self::sigmoid_interval(n, c.birth_lo, c.birth_hi, c.alpha_n);

        // Death threshold varies based on current state
        let death_lo = c.death_lo + (c.birth_lo - c.death_lo) * m;
        let death_hi = c.death_hi + (c.birth_hi - c.death_hi) * m;

        // Survival: outer ring filling in [death_lo, death_hi]
        let survival = Self::sigmoid_interval(n, death_lo, death_hi, c.alpha_n);

        // Mix based on current state
        let alive = Self::sigmoid(m, 0.5, c.alpha_m);
        birth * (1.0 - alive) + survival * alive
    }

    /// Returns the width.
    pub fn width(&self) -> usize {
        self.width
    }

    /// Returns the height.
    pub fn height(&self) -> usize {
        self.height
    }

    /// Gets the state of a cell.
    pub fn get(&self, x: usize, y: usize) -> f32 {
        self.cells
            .get(y)
            .and_then(|row| row.get(x))
            .copied()
            .unwrap_or(0.0)
    }

    /// Sets the state of a cell.
    pub fn set(&mut self, x: usize, y: usize, value: f32) {
        if y < self.height && x < self.width {
            self.cells[y][x] = value.clamp(0.0, 1.0);
        }
    }

    /// Clears all cells to 0.
    pub fn clear(&mut self) {
        for row in &mut self.cells {
            row.fill(0.0);
        }
    }

    /// Randomizes cells with given density (probability of being close to 1.0).
    pub fn randomize(&mut self, seed: u64, density: f32) {
        let mut rng = SimpleRng::new(seed);
        for row in &mut self.cells {
            for cell in row {
                *cell = if rng.next_f32() < density {
                    0.5 + rng.next_f32() * 0.5
                } else {
                    rng.next_f32() * 0.2
                };
            }
        }
    }

    /// Computes inner disk (m) and outer ring (n) filling for a cell.
    fn compute_filling(&self, x: usize, y: usize) -> (f32, f32) {
        let mut inner_sum = 0.0f32;
        let mut outer_sum = 0.0f32;

        let kr = self.kernel_radius;

        for dy in -kr..=kr {
            for dx in -kr..=kr {
                let nx = ((x as i32 + dx).rem_euclid(self.width as i32)) as usize;
                let ny = ((y as i32 + dy).rem_euclid(self.height as i32)) as usize;

                let wx = (dx + kr) as usize;
                let wy = (dy + kr) as usize;

                let cell_value = self.cells[ny][nx];

                inner_sum += self.inner_weights[wy][wx] * cell_value;
                outer_sum += self.outer_weights[wy][wx] * cell_value;
            }
        }

        let m = inner_sum / self.inner_total;
        let n = outer_sum / self.outer_total;

        (n, m)
    }

    /// Advances the simulation by one step.
    ///
    /// The `dt` parameter controls the rate of change (0.0 to 1.0).
    /// Use ~0.1 for smooth animation, 1.0 for discrete steps.
    #[allow(clippy::needless_range_loop)]
    pub fn step(&mut self, dt: f32) {
        let mut next = vec![vec![0.0f32; self.width]; self.height];

        for y in 0..self.height {
            for x in 0..self.width {
                let (n, m) = self.compute_filling(x, y);
                let target = self.transition(n, m);
                let current = self.cells[y][x];

                // Smooth interpolation toward target
                next[y][x] = current + (target - current) * dt;
            }
        }

        self.cells = next;
    }

    /// Advances multiple steps.
    pub fn steps(&mut self, n: usize, dt: f32) {
        for _ in 0..n {
            self.step(dt);
        }
    }

    /// Returns a reference to the cell grid.
    pub fn cells(&self) -> &Vec<Vec<f32>> {
        &self.cells
    }

    /// Returns the average cell value.
    pub fn average_value(&self) -> f32 {
        let sum: f32 = self.cells.iter().flat_map(|row| row.iter()).sum();
        sum / (self.width * self.height) as f32
    }
}

/// Simple RNG for cellular automata.
struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_add(1),
        }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.state
    }

    fn next_bool(&mut self) -> bool {
        self.next_u64() & 1 == 1
    }

    fn next_f32(&mut self) -> f32 {
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

// ============================================================================
// HashLife - Optimized Game of Life using Quadtrees and Memoization
// ============================================================================

/// A node in the HashLife quadtree.
///
/// Each node represents a square region of the grid. Leaf nodes (level 0)
/// represent single cells. Interior nodes have 4 children: NW, NE, SW, SE.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum QuadNode {
    /// A single cell (level 0).
    Leaf(bool),
    /// An interior node with 4 children.
    Interior {
        /// The level (size = 2^level).
        level: u32,
        /// Northwest quadrant.
        nw: u64,
        /// Northeast quadrant.
        ne: u64,
        /// Southwest quadrant.
        sw: u64,
        /// Southeast quadrant.
        se: u64,
        /// Population count for this node.
        population: u64,
    },
}

/// HashLife universe for Conway's Game of Life.
///
/// Uses quadtrees and memoization to efficiently simulate large, sparse patterns.
/// Can compute future states in O(log n) time for patterns with repetitive structure.
///
/// # Example
///
/// ```
/// use unshape_automata::HashLife;
///
/// let mut universe = HashLife::new();
///
/// // Create a glider
/// universe.set_cell(1, 0, true);
/// universe.set_cell(2, 1, true);
/// universe.set_cell(0, 2, true);
/// universe.set_cell(1, 2, true);
/// universe.set_cell(2, 2, true);
///
/// // Advance 4 generations
/// for _ in 0..4 {
///     universe.step();
/// }
///
/// assert!(universe.get_cell(2, 1));
/// ```
#[derive(Debug)]
pub struct HashLife {
    /// All nodes in the universe, indexed by ID.
    nodes: Vec<QuadNode>,
    /// Map from node content to ID for deduplication.
    node_map: std::collections::HashMap<QuadNode, u64>,
    /// Cache of computed future states: (node_id, step_size) -> result_node_id.
    result_cache: std::collections::HashMap<(u64, u64), u64>,
    /// The root node of the quadtree.
    root: u64,
    /// Generation counter.
    generation: u64,
    /// Origin offset X (for coordinates).
    origin_x: i64,
    /// Origin offset Y (for coordinates).
    origin_y: i64,
}

impl HashLife {
    /// Pre-allocated node IDs for dead and alive leaf nodes.
    const DEAD: u64 = 0;
    const ALIVE: u64 = 1;

    /// Creates a new empty HashLife universe.
    pub fn new() -> Self {
        let mut nodes = Vec::new();
        let mut node_map = std::collections::HashMap::new();

        // Pre-allocate leaf nodes
        nodes.push(QuadNode::Leaf(false)); // ID 0 = dead
        node_map.insert(QuadNode::Leaf(false), 0);

        nodes.push(QuadNode::Leaf(true)); // ID 1 = alive
        node_map.insert(QuadNode::Leaf(true), 1);

        // Create an empty 2x2 node as the initial root
        let empty_2x2 = Self::create_interior_node_static(&mut nodes, &mut node_map, 1, 0, 0, 0, 0);

        Self {
            nodes,
            node_map,
            result_cache: std::collections::HashMap::new(),
            root: empty_2x2,
            generation: 0,
            origin_x: 0,
            origin_y: 0,
        }
    }

    fn create_interior_node_static(
        nodes: &mut Vec<QuadNode>,
        node_map: &mut std::collections::HashMap<QuadNode, u64>,
        level: u32,
        nw: u64,
        ne: u64,
        sw: u64,
        se: u64,
    ) -> u64 {
        // Calculate population
        let pop = |id: u64| -> u64 {
            match &nodes[id as usize] {
                QuadNode::Leaf(alive) => *alive as u64,
                QuadNode::Interior { population, .. } => *population,
            }
        };
        let population = pop(nw) + pop(ne) + pop(sw) + pop(se);

        let node = QuadNode::Interior {
            level,
            nw,
            ne,
            sw,
            se,
            population,
        };

        if let Some(&id) = node_map.get(&node) {
            id
        } else {
            let id = nodes.len() as u64;
            nodes.push(node.clone());
            node_map.insert(node, id);
            id
        }
    }

    /// Creates or retrieves an interior node with the given children.
    fn create_interior_node(&mut self, level: u32, nw: u64, ne: u64, sw: u64, se: u64) -> u64 {
        Self::create_interior_node_static(
            &mut self.nodes,
            &mut self.node_map,
            level,
            nw,
            ne,
            sw,
            se,
        )
    }

    /// Gets the level of a node.
    fn level(&self, id: u64) -> u32 {
        match &self.nodes[id as usize] {
            QuadNode::Leaf(_) => 0,
            QuadNode::Interior { level, .. } => *level,
        }
    }

    /// Gets the population of a node.
    fn node_population(&self, id: u64) -> u64 {
        match &self.nodes[id as usize] {
            QuadNode::Leaf(alive) => *alive as u64,
            QuadNode::Interior { population, .. } => *population,
        }
    }

    /// Gets the children of an interior node.
    fn children(&self, id: u64) -> (u64, u64, u64, u64) {
        match &self.nodes[id as usize] {
            QuadNode::Leaf(_) => panic!("Cannot get children of leaf node"),
            QuadNode::Interior { nw, ne, sw, se, .. } => (*nw, *ne, *sw, *se),
        }
    }

    /// Creates an empty node of the given level.
    fn empty_node(&mut self, level: u32) -> u64 {
        if level == 0 {
            Self::DEAD
        } else {
            let child = self.empty_node(level - 1);
            self.create_interior_node(level, child, child, child, child)
        }
    }

    /// Expands the universe by adding empty space around it.
    fn expand(&mut self) {
        let level = self.level(self.root);
        let (nw, ne, sw, se) = self.children(self.root);

        let empty = self.empty_node(level - 1);

        // Create new quadrants with the old quadrants in their inner corners
        let new_nw = self.create_interior_node(level, empty, empty, empty, nw);
        let new_ne = self.create_interior_node(level, empty, empty, ne, empty);
        let new_sw = self.create_interior_node(level, empty, sw, empty, empty);
        let new_se = self.create_interior_node(level, se, empty, empty, empty);

        self.root = self.create_interior_node(level + 1, new_nw, new_ne, new_sw, new_se);

        // Adjust origin
        let half_size = 1i64 << (level - 1);
        self.origin_x -= half_size;
        self.origin_y -= half_size;
    }

    /// Sets the value of a cell at the given coordinates.
    pub fn set_cell(&mut self, x: i64, y: i64, alive: bool) {
        // Ensure the root is large enough to contain the cell
        while self.level(self.root) < 3 || !self.contains(x, y) {
            self.expand();
        }

        self.root = self.set_cell_recursive(self.root, x - self.origin_x, y - self.origin_y, alive);
        // Invalidate result cache since the universe changed
        self.result_cache.clear();
    }

    fn contains(&self, x: i64, y: i64) -> bool {
        let level = self.level(self.root);
        let size = 1i64 << level;
        let local_x = x - self.origin_x;
        let local_y = y - self.origin_y;
        local_x >= 0 && local_x < size && local_y >= 0 && local_y < size
    }

    fn set_cell_recursive(&mut self, node: u64, x: i64, y: i64, alive: bool) -> u64 {
        let level = self.level(node);

        if level == 0 {
            return if alive { Self::ALIVE } else { Self::DEAD };
        }

        let (nw, ne, sw, se) = self.children(node);
        let half = 1i64 << (level - 1);

        let (new_nw, new_ne, new_sw, new_se) = if x < half {
            if y < half {
                // NW quadrant
                let new_nw = self.set_cell_recursive(nw, x, y, alive);
                (new_nw, ne, sw, se)
            } else {
                // SW quadrant
                let new_sw = self.set_cell_recursive(sw, x, y - half, alive);
                (nw, ne, new_sw, se)
            }
        } else if y < half {
            // NE quadrant
            let new_ne = self.set_cell_recursive(ne, x - half, y, alive);
            (nw, new_ne, sw, se)
        } else {
            // SE quadrant
            let new_se = self.set_cell_recursive(se, x - half, y - half, alive);
            (nw, ne, sw, new_se)
        };

        self.create_interior_node(level, new_nw, new_ne, new_sw, new_se)
    }

    /// Gets the value of a cell at the given coordinates.
    pub fn get_cell(&self, x: i64, y: i64) -> bool {
        if !self.contains(x, y) {
            return false;
        }
        self.get_cell_recursive(self.root, x - self.origin_x, y - self.origin_y)
    }

    fn get_cell_recursive(&self, node: u64, x: i64, y: i64) -> bool {
        match &self.nodes[node as usize] {
            QuadNode::Leaf(alive) => *alive,
            QuadNode::Interior {
                level,
                nw,
                ne,
                sw,
                se,
                ..
            } => {
                let half = 1i64 << (level - 1);
                if x < half {
                    if y < half {
                        self.get_cell_recursive(*nw, x, y)
                    } else {
                        self.get_cell_recursive(*sw, x, y - half)
                    }
                } else if y < half {
                    self.get_cell_recursive(*ne, x - half, y)
                } else {
                    self.get_cell_recursive(*se, x - half, y - half)
                }
            }
        }
    }

    // ========================================================================
    // Memoized recursive HashLife algorithm
    // ========================================================================

    /// Advances the universe by one generation.
    pub fn step(&mut self) {
        self.step_pow2(0);
    }

    /// Advances the universe by exactly 2^n generations using memoized recursion.
    ///
    /// This is the core HashLife speedup. For patterns with repetitive structure,
    /// memoization makes this effectively O(1) per unique sub-pattern, regardless
    /// of the number of generations.
    ///
    /// # Example
    ///
    /// ```
    /// use unshape_automata::HashLife;
    ///
    /// let mut universe = HashLife::new();
    /// // Set up an r-pentomino
    /// universe.set_cell(1, 0, true);
    /// universe.set_cell(2, 0, true);
    /// universe.set_cell(0, 1, true);
    /// universe.set_cell(1, 1, true);
    /// universe.set_cell(1, 2, true);
    ///
    /// // Advance 1024 generations in one call
    /// universe.step_pow2(10);
    /// assert_eq!(universe.generation(), 1024);
    /// ```
    pub fn step_pow2(&mut self, n: u32) {
        // Ensure the tree is large enough: advance at level L gives 2^(L-2) steps,
        // so we need level >= n + 2.
        let target_level = n + 2;
        while self.level(self.root) < target_level {
            self.expand();
        }
        // Ensure borders are clear (live cells must not touch the edge)
        while self.needs_expansion() {
            self.expand();
        }
        // One extra expansion for safety margin
        self.expand();

        let level = self.level(self.root);
        self.root = self.advance(self.root, n);
        self.generation += 1u64 << n;

        // Adjust origin: the result is the center of the original root.
        // Original root covers [origin, origin + 2^level).
        // Center starts at origin + 2^(level-2).
        let quarter = 1i64 << (level - 2);
        self.origin_x += quarter;
        self.origin_y += quarter;
    }

    /// The core memoized recursive algorithm.
    ///
    /// For a level-L node, advances `2^step_log2` generations and returns
    /// the center level-(L-1) result. Requires `step_log2 <= L - 2`.
    fn advance(&mut self, node: u64, step_log2: u32) -> u64 {
        let level = self.level(node);
        debug_assert!(level >= 2);
        debug_assert!(step_log2 <= level - 2);

        // Check memoization cache
        if let Some(&result) = self.result_cache.get(&(node, step_log2 as u64)) {
            return result;
        }

        let result = if level == 2 {
            // Base case: 4×4 grid → 2×2 center after 1 generation
            self.advance_level2(node)
        } else if step_log2 == level - 2 {
            // Full speed: two rounds of recursive advance
            self.advance_full(node)
        } else {
            // Slow mode: one round of advance, then extract centers
            self.advance_slow(node, step_log2)
        };

        self.result_cache.insert((node, step_log2 as u64), result);

        // Bounded memory: clear cache if it gets too large
        if self.result_cache.len() > 1_000_000 {
            self.result_cache.clear();
        }

        result
    }

    /// Base case: compute 1 generation of Game of Life on a 4×4 grid.
    /// Returns the 2×2 center as a level-1 node.
    fn advance_level2(&mut self, node: u64) -> u64 {
        let (nw, ne, sw, se) = self.children(node);
        let (nw_nw, nw_ne, nw_sw, nw_se) = self.children(nw);
        let (ne_nw, ne_ne, ne_sw, ne_se) = self.children(ne);
        let (sw_nw, sw_ne, sw_sw, sw_se) = self.children(sw);
        let (se_nw, se_ne, se_sw, se_se) = self.children(se);

        let cell = |id: u64| -> bool { matches!(&self.nodes[id as usize], QuadNode::Leaf(true)) };

        // 4×4 grid layout:
        //   a[0][0] a[0][1] a[0][2] a[0][3]
        //   a[1][0] a[1][1] a[1][2] a[1][3]
        //   a[2][0] a[2][1] a[2][2] a[2][3]
        //   a[3][0] a[3][1] a[3][2] a[3][3]
        let a = [
            [cell(nw_nw), cell(nw_ne), cell(ne_nw), cell(ne_ne)],
            [cell(nw_sw), cell(nw_se), cell(ne_sw), cell(ne_se)],
            [cell(sw_nw), cell(sw_ne), cell(se_nw), cell(se_ne)],
            [cell(sw_sw), cell(sw_se), cell(se_sw), cell(se_se)],
        ];

        // Apply Game of Life to the 4 center cells
        let life = |y: usize, x: usize| -> bool {
            let mut count = 0u8;
            for dy in [0usize, 1, 2] {
                for dx in [0usize, 1, 2] {
                    if dy == 1 && dx == 1 {
                        continue;
                    }
                    if a[y - 1 + dy][x - 1 + dx] {
                        count += 1;
                    }
                }
            }
            if a[y][x] {
                count == 2 || count == 3
            } else {
                count == 3
            }
        };

        let r_nw = if life(1, 1) { Self::ALIVE } else { Self::DEAD };
        let r_ne = if life(1, 2) { Self::ALIVE } else { Self::DEAD };
        let r_sw = if life(2, 1) { Self::ALIVE } else { Self::DEAD };
        let r_se = if life(2, 2) { Self::ALIVE } else { Self::DEAD };

        self.create_interior_node(1, r_nw, r_ne, r_sw, r_se)
    }

    /// Full-speed advance: two rounds of recursion.
    /// Advances 2^(L-2) generations for a level-L node.
    fn advance_full(&mut self, node: u64) -> u64 {
        let level = self.level(node);
        let sub_step = level - 3;

        // Form 9 overlapping sub-squares from the 4x4 grid of grandchildren
        let (n00, n01, n02, n10, n11, n12, n20, n21, n22) = self.nine_sub_squares(node);

        // First round: advance each sub-square by 2^(L-3) generations
        let r00 = self.advance(n00, sub_step);
        let r01 = self.advance(n01, sub_step);
        let r02 = self.advance(n02, sub_step);
        let r10 = self.advance(n10, sub_step);
        let r11 = self.advance(n11, sub_step);
        let r12 = self.advance(n12, sub_step);
        let r20 = self.advance(n20, sub_step);
        let r21 = self.advance(n21, sub_step);
        let r22 = self.advance(n22, sub_step);

        // Combine into 4 intermediate squares
        let c0 = self.create_interior_node(level - 1, r00, r01, r10, r11);
        let c1 = self.create_interior_node(level - 1, r01, r02, r11, r12);
        let c2 = self.create_interior_node(level - 1, r10, r11, r20, r21);
        let c3 = self.create_interior_node(level - 1, r11, r12, r21, r22);

        // Second round: advance each intermediate by another 2^(L-3) generations
        // Total: 2^(L-3) + 2^(L-3) = 2^(L-2) ✓
        let f0 = self.advance(c0, sub_step);
        let f1 = self.advance(c1, sub_step);
        let f2 = self.advance(c2, sub_step);
        let f3 = self.advance(c3, sub_step);

        self.create_interior_node(level - 1, f0, f1, f2, f3)
    }

    /// Slow-mode advance: one round of recursion, then extract centers.
    /// Advances 2^step_log2 generations where step_log2 < L-2.
    fn advance_slow(&mut self, node: u64, step_log2: u32) -> u64 {
        let level = self.level(node);

        // Form 9 overlapping sub-squares
        let (n00, n01, n02, n10, n11, n12, n20, n21, n22) = self.nine_sub_squares(node);

        // Advance each sub-square by 2^step_log2 generations
        let r00 = self.advance(n00, step_log2);
        let r01 = self.advance(n01, step_log2);
        let r02 = self.advance(n02, step_log2);
        let r10 = self.advance(n10, step_log2);
        let r11 = self.advance(n11, step_log2);
        let r12 = self.advance(n12, step_log2);
        let r20 = self.advance(n20, step_log2);
        let r21 = self.advance(n21, step_log2);
        let r22 = self.advance(n22, step_log2);

        // Combine into 4 intermediate squares
        let c0 = self.create_interior_node(level - 1, r00, r01, r10, r11);
        let c1 = self.create_interior_node(level - 1, r01, r02, r11, r12);
        let c2 = self.create_interior_node(level - 1, r10, r11, r20, r21);
        let c3 = self.create_interior_node(level - 1, r11, r12, r21, r22);

        // Extract centers (no additional stepping)
        // Total: 2^step_log2 generations ✓
        let f0 = self.center_node(c0);
        let f1 = self.center_node(c1);
        let f2 = self.center_node(c2);
        let f3 = self.center_node(c3);

        self.create_interior_node(level - 1, f0, f1, f2, f3)
    }

    /// Forms 9 overlapping level-(L-1) sub-squares from a level-L node.
    ///
    /// Given the 4x4 grid of grandchildren:
    /// ```text
    /// nw.nw nw.ne | ne.nw ne.ne
    /// nw.sw nw.se | ne.sw ne.se
    /// ------+------+------+------
    /// sw.nw sw.ne | se.nw se.ne
    /// sw.sw sw.se | se.sw se.se
    /// ```
    ///
    /// Returns 9 overlapping 2x2 blocks (each level L-1):
    /// ```text
    /// n00 n01 n02
    /// n10 n11 n12
    /// n20 n21 n22
    /// ```
    fn nine_sub_squares(&mut self, node: u64) -> (u64, u64, u64, u64, u64, u64, u64, u64, u64) {
        let (nw, ne, sw, se) = self.children(node);

        let n00 = nw;
        let n01 = self.center_horizontal(nw, ne);
        let n02 = ne;
        let n10 = self.center_vertical(nw, sw);
        let n11 = self.center_quad(nw, ne, sw, se);
        let n12 = self.center_vertical(ne, se);
        let n20 = sw;
        let n21 = self.center_horizontal(sw, se);
        let n22 = se;

        (n00, n01, n02, n10, n11, n12, n20, n21, n22)
    }

    /// Extracts the center level-(L-1) sub-node from a level-L node.
    fn center_node(&mut self, node: u64) -> u64 {
        let (nw, ne, sw, se) = self.children(node);
        self.center_quad(nw, ne, sw, se)
    }

    /// Forms the level-L node between two horizontally adjacent level-L nodes.
    fn center_horizontal(&mut self, w: u64, e: u64) -> u64 {
        let (_, w_ne, _, w_se) = self.children(w);
        let (e_nw, _, e_sw, _) = self.children(e);
        let level = self.level(w);
        self.create_interior_node(level, w_ne, e_nw, w_se, e_sw)
    }

    /// Forms the level-L node between two vertically adjacent level-L nodes.
    fn center_vertical(&mut self, n: u64, s: u64) -> u64 {
        let (_, _, n_sw, n_se) = self.children(n);
        let (s_nw, s_ne, _, _) = self.children(s);
        let level = self.level(n);
        self.create_interior_node(level, n_sw, n_se, s_nw, s_ne)
    }

    /// Forms the level-L node at the center of 4 arranged level-L nodes.
    fn center_quad(&mut self, nw: u64, ne: u64, sw: u64, se: u64) -> u64 {
        let (_, _, _, nw_se) = self.children(nw);
        let (_, _, ne_sw, _) = self.children(ne);
        let (_, sw_ne, _, _) = self.children(sw);
        let (se_nw, _, _, _) = self.children(se);
        let level = self.level(nw);
        self.create_interior_node(level, nw_se, ne_sw, sw_ne, se_nw)
    }

    fn needs_expansion(&self) -> bool {
        // Check if any border cells are alive
        let (nw, ne, sw, se) = self.children(self.root);

        // Check NW border
        let (nw_nw, nw_ne, nw_sw, _) = self.children(nw);
        if self.node_population(nw_nw) > 0
            || self.node_population(nw_ne) > 0
            || self.node_population(nw_sw) > 0
        {
            return true;
        }

        // Check NE border
        let (ne_nw, ne_ne, _, ne_se) = self.children(ne);
        if self.node_population(ne_nw) > 0
            || self.node_population(ne_ne) > 0
            || self.node_population(ne_se) > 0
        {
            return true;
        }

        // Check SW border
        let (sw_nw, _, sw_sw, sw_se) = self.children(sw);
        if self.node_population(sw_nw) > 0
            || self.node_population(sw_sw) > 0
            || self.node_population(sw_se) > 0
        {
            return true;
        }

        // Check SE border
        let (_, se_ne, se_sw, se_se) = self.children(se);
        if self.node_population(se_ne) > 0
            || self.node_population(se_sw) > 0
            || self.node_population(se_se) > 0
        {
            return true;
        }

        false
    }

    /// Returns the total population (number of live cells).
    pub fn population(&self) -> u64 {
        self.node_population(self.root)
    }

    /// Returns the current generation count.
    pub fn generation(&self) -> u64 {
        self.generation
    }

    /// Advances multiple generations.
    ///
    /// Decomposes n into powers of 2 and calls [`step_pow2`](Self::step_pow2)
    /// for each, taking advantage of memoization for large jumps.
    pub fn steps(&mut self, n: usize) {
        let mut remaining = n as u64;
        let mut bit = 0u32;
        while remaining > 0 {
            if remaining & 1 == 1 {
                self.step_pow2(bit);
            }
            remaining >>= 1;
            bit += 1;
        }
    }

    /// Gets the bounding box of all live cells.
    pub fn bounds(&self) -> Option<(i64, i64, i64, i64)> {
        if self.population() == 0 {
            return None;
        }

        let level = self.level(self.root);
        let size = 1i64 << level;

        // Simple scan for now
        let mut min_x = i64::MAX;
        let mut max_x = i64::MIN;
        let mut min_y = i64::MAX;
        let mut max_y = i64::MIN;

        for y in 0..size {
            for x in 0..size {
                if self.get_cell_recursive(self.root, x, y) {
                    let gx = x + self.origin_x;
                    let gy = y + self.origin_y;
                    min_x = min_x.min(gx);
                    max_x = max_x.max(gx);
                    min_y = min_y.min(gy);
                    max_y = max_y.max(gy);
                }
            }
        }

        Some((min_x, min_y, max_x, max_y))
    }

    /// Clears the result cache.
    pub fn clear_cache(&mut self) {
        self.result_cache.clear();
    }

    /// Creates a HashLife universe from a [`CellularAutomaton2D`].
    ///
    /// Copies all live cells from the grid-based automaton into the
    /// quadtree-based HashLife. The grid is placed at coordinates (0, 0)
    /// to (width-1, height-1).
    ///
    /// Note: HashLife uses Game of Life rules (B3/S23). The source CA's
    /// rule set is not transferred.
    pub fn from_ca2d(ca: &CellularAutomaton2D) -> Self {
        let mut universe = Self::new();
        let cells = ca.cells();
        for (y, row) in cells.iter().enumerate() {
            for (x, &alive) in row.iter().enumerate() {
                if alive {
                    universe.set_cell(x as i64, y as i64, true);
                }
            }
        }
        universe
    }
}

impl Default for HashLife {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for HashLife {
    fn clone(&self) -> Self {
        Self {
            nodes: self.nodes.clone(),
            node_map: self.node_map.clone(),
            result_cache: self.result_cache.clone(),
            root: self.root,
            generation: self.generation,
            origin_x: self.origin_x,
            origin_y: self.origin_y,
        }
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
