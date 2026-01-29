#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::SimpleRng;

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
