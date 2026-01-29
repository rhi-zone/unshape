#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

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
