#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::{ExtendedMoore, Neighborhood2D, SimpleRng};

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
