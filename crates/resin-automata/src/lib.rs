//! Cellular automata for procedural pattern generation.
//!
//! Implements both 1D (elementary) and 2D cellular automata.
//!
//! # Example
//!
//! ```
//! use resin_automata::{ElementaryCA, GameOfLife};
//!
//! // 1D: Rule 30
//! let mut ca = ElementaryCA::new(100, 30);
//! ca.randomize(12345);
//! ca.step();
//!
//! // 2D: Game of Life
//! let mut life = GameOfLife::life(50, 50);
//! life.randomize(12345, 0.3);
//! life.step();
//! ```

/// 1D Elementary Cellular Automaton.
///
/// Implements Wolfram's elementary cellular automata with 256 possible rules.
#[derive(Debug, Clone)]
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

/// 2D Cellular Automaton with configurable rules.
#[derive(Debug, Clone)]
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
}

impl CellularAutomaton2D {
    /// Creates a new 2D cellular automaton with custom rules.
    ///
    /// Rules are specified as birth/survival counts (e.g., B3/S23 for Game of Life).
    pub fn new(width: usize, height: usize, birth: &[u8], survive: &[u8]) -> Self {
        Self {
            cells: vec![vec![false; width]; height],
            width,
            height,
            birth: birth.to_vec(),
            survive: survive.to_vec(),
            wrap: true,
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
    fn count_neighbors(&self, x: usize, y: usize) -> u8 {
        let mut count = 0u8;

        for dy in -1i32..=1 {
            for dx in -1i32..=1 {
                if dx == 0 && dy == 0 {
                    continue;
                }

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
        }

        count
    }

    /// Advances the automaton by one step.
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
}
