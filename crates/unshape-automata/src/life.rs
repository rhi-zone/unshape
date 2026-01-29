#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::{Moore, Moore3D, Neighborhood2D, Neighborhood3D, SimpleRng};

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
    pub(crate) fn count_neighbors(&self, x: usize, y: usize) -> u8 {
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
