//! Classic maze generation algorithms.
//!
//! Provides several algorithms for generating perfect mazes (mazes with exactly
//! one path between any two points):
//!
//! - Recursive backtracker (depth-first search)
//! - Prim's algorithm (minimum spanning tree)
//! - Kruskal's algorithm (minimum spanning tree)
//! - Eller's algorithm (row-by-row generation)
//!
//! # Example
//!
//! ```
//! use rhizome_resin_procgen::maze::{Maze, MazeAlgorithm, generate_maze};
//!
//! let maze = generate_maze(10, 10, MazeAlgorithm::RecursiveBacktracker, 12345);
//!
//! // Check if a cell is a passage (not a wall)
//! assert!(maze.is_passage(1, 1));
//!
//! // Get the maze as a 2D grid
//! let grid = maze.to_grid();
//! ```

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// A generated maze.
#[derive(Debug, Clone)]
pub struct Maze {
    /// Width in cells (not including walls).
    width: usize,
    /// Height in cells (not including walls).
    height: usize,
    /// Grid of cells (true = passage, false = wall).
    /// Dimensions are (2*width+1) x (2*height+1) to include walls.
    grid: Vec<bool>,
    /// Grid width including walls.
    grid_width: usize,
    /// Grid height including walls.
    grid_height: usize,
}

impl Maze {
    /// Creates a new maze filled with walls.
    pub fn new(width: usize, height: usize) -> Self {
        let grid_width = 2 * width + 1;
        let grid_height = 2 * height + 1;
        Self {
            width,
            height,
            grid: vec![false; grid_width * grid_height],
            grid_width,
            grid_height,
        }
    }

    /// Returns the maze width in cells.
    pub fn width(&self) -> usize {
        self.width
    }

    /// Returns the maze height in cells.
    pub fn height(&self) -> usize {
        self.height
    }

    /// Returns the grid width (including walls).
    pub fn grid_width(&self) -> usize {
        self.grid_width
    }

    /// Returns the grid height (including walls).
    pub fn grid_height(&self) -> usize {
        self.grid_height
    }

    /// Converts cell coordinates to grid coordinates.
    fn cell_to_grid(&self, x: usize, y: usize) -> (usize, usize) {
        (2 * x + 1, 2 * y + 1)
    }

    /// Carves a passage at the given cell coordinates.
    pub fn carve(&mut self, x: usize, y: usize) {
        let (gx, gy) = self.cell_to_grid(x, y);
        if gx < self.grid_width && gy < self.grid_height {
            self.grid[gy * self.grid_width + gx] = true;
        }
    }

    /// Carves a wall between two adjacent cells.
    pub fn carve_wall(&mut self, x1: usize, y1: usize, x2: usize, y2: usize) {
        let (gx1, gy1) = self.cell_to_grid(x1, y1);
        let (gx2, gy2) = self.cell_to_grid(x2, y2);

        // Wall is at the midpoint
        let wx = (gx1 + gx2) / 2;
        let wy = (gy1 + gy2) / 2;

        if wx < self.grid_width && wy < self.grid_height {
            self.grid[wy * self.grid_width + wx] = true;
        }
    }

    /// Checks if a grid position is a passage.
    pub fn is_passage(&self, gx: usize, gy: usize) -> bool {
        if gx >= self.grid_width || gy >= self.grid_height {
            return false;
        }
        self.grid[gy * self.grid_width + gx]
    }

    /// Checks if a grid position is a wall.
    pub fn is_wall(&self, gx: usize, gy: usize) -> bool {
        !self.is_passage(gx, gy)
    }

    /// Sets a grid position.
    pub fn set(&mut self, gx: usize, gy: usize, passage: bool) {
        if gx < self.grid_width && gy < self.grid_height {
            self.grid[gy * self.grid_width + gx] = passage;
        }
    }

    /// Returns the maze as a 2D grid of booleans.
    ///
    /// True = passage, False = wall.
    pub fn to_grid(&self) -> Vec<Vec<bool>> {
        let mut result = vec![vec![false; self.grid_width]; self.grid_height];
        for y in 0..self.grid_height {
            for x in 0..self.grid_width {
                result[y][x] = self.grid[y * self.grid_width + x];
            }
        }
        result
    }

    /// Converts the maze to ASCII art.
    pub fn to_string_art(&self) -> String {
        let mut s = String::new();
        for y in 0..self.grid_height {
            for x in 0..self.grid_width {
                if self.is_passage(x, y) {
                    s.push(' ');
                } else {
                    s.push('#');
                }
            }
            s.push('\n');
        }
        s
    }

    /// Creates an entrance at the top-left.
    pub fn add_entrance(&mut self) {
        // Carve entrance on the left side of row 0
        self.set(0, 1, true);
    }

    /// Creates an exit at the bottom-right.
    pub fn add_exit(&mut self) {
        // Carve exit on the right side of last row
        self.set(self.grid_width - 1, self.grid_height - 2, true);
    }
}

/// Available maze generation algorithms.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum MazeAlgorithm {
    /// Depth-first search with backtracking.
    /// Creates long, winding passages.
    #[default]
    RecursiveBacktracker,
    /// Randomized Prim's algorithm.
    /// Creates many short dead ends.
    Prim,
    /// Randomized Kruskal's algorithm.
    /// Creates evenly distributed passages.
    Kruskal,
    /// Eller's row-by-row algorithm.
    /// Memory efficient, generates row by row.
    Eller,
    /// Binary tree algorithm.
    /// Simple and fast, but has diagonal bias.
    BinaryTree,
    /// Sidewinder algorithm.
    /// Similar to binary tree but with horizontal bias.
    Sidewinder,
}

/// Generates a maze using the specified algorithm.
pub fn generate_maze(width: usize, height: usize, algorithm: MazeAlgorithm, seed: u64) -> Maze {
    match algorithm {
        MazeAlgorithm::RecursiveBacktracker => recursive_backtracker(width, height, seed),
        MazeAlgorithm::Prim => prim(width, height, seed),
        MazeAlgorithm::Kruskal => kruskal(width, height, seed),
        MazeAlgorithm::Eller => eller(width, height, seed),
        MazeAlgorithm::BinaryTree => binary_tree(width, height, seed),
        MazeAlgorithm::Sidewinder => sidewinder(width, height, seed),
    }
}

/// Simple PRNG for maze generation.
struct Rng {
    state: u64,
}

impl Rng {
    fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_add(1),
        }
    }

    fn next(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.state
    }

    fn range(&mut self, max: usize) -> usize {
        (self.next() as usize) % max.max(1)
    }

    fn shuffle<T>(&mut self, slice: &mut [T]) {
        for i in (1..slice.len()).rev() {
            let j = self.range(i + 1);
            slice.swap(i, j);
        }
    }

    fn coin_flip(&mut self) -> bool {
        self.next() % 2 == 0
    }
}

/// Generates a maze using recursive backtracker (DFS).
pub fn recursive_backtracker(width: usize, height: usize, seed: u64) -> Maze {
    let mut maze = Maze::new(width, height);
    let mut rng = Rng::new(seed);
    let mut visited = vec![vec![false; width]; height];

    // Start from a random cell
    let start_x = rng.range(width);
    let start_y = rng.range(height);

    let mut stack = vec![(start_x, start_y)];
    visited[start_y][start_x] = true;
    maze.carve(start_x, start_y);

    let directions: [(i32, i32); 4] = [(0, -1), (1, 0), (0, 1), (-1, 0)];

    while let Some(&(x, y)) = stack.last() {
        // Find unvisited neighbors
        let mut neighbors = Vec::new();
        for (dx, dy) in directions {
            let nx = x as i32 + dx;
            let ny = y as i32 + dy;

            if nx >= 0 && ny >= 0 && (nx as usize) < width && (ny as usize) < height {
                let nx = nx as usize;
                let ny = ny as usize;
                if !visited[ny][nx] {
                    neighbors.push((nx, ny));
                }
            }
        }

        if neighbors.is_empty() {
            stack.pop();
        } else {
            // Choose random neighbor
            let idx = rng.range(neighbors.len());
            let (nx, ny) = neighbors[idx];

            // Carve passage to neighbor
            visited[ny][nx] = true;
            maze.carve(nx, ny);
            maze.carve_wall(x, y, nx, ny);

            stack.push((nx, ny));
        }
    }

    maze
}

/// Generates a maze using Prim's algorithm.
pub fn prim(width: usize, height: usize, seed: u64) -> Maze {
    let mut maze = Maze::new(width, height);
    let mut rng = Rng::new(seed);
    let mut in_maze = vec![vec![false; width]; height];

    // Start from a random cell
    let start_x = rng.range(width);
    let start_y = rng.range(height);

    in_maze[start_y][start_x] = true;
    maze.carve(start_x, start_y);

    // Frontier: walls that could be removed
    let mut frontier: Vec<(usize, usize, usize, usize)> = Vec::new();

    let directions: [(i32, i32); 4] = [(0, -1), (1, 0), (0, 1), (-1, 0)];

    // Add initial frontier
    for (dx, dy) in directions {
        let nx = start_x as i32 + dx;
        let ny = start_y as i32 + dy;
        if nx >= 0 && ny >= 0 && (nx as usize) < width && (ny as usize) < height {
            frontier.push((start_x, start_y, nx as usize, ny as usize));
        }
    }

    while !frontier.is_empty() {
        // Pick random wall from frontier
        let idx = rng.range(frontier.len());
        let (x1, y1, x2, y2) = frontier.swap_remove(idx);

        if in_maze[y2][x2] {
            continue;
        }

        // Add the cell
        in_maze[y2][x2] = true;
        maze.carve(x2, y2);
        maze.carve_wall(x1, y1, x2, y2);

        // Add new frontier
        for (dx, dy) in directions {
            let nx = x2 as i32 + dx;
            let ny = y2 as i32 + dy;
            if nx >= 0 && ny >= 0 && (nx as usize) < width && (ny as usize) < height {
                let nx = nx as usize;
                let ny = ny as usize;
                if !in_maze[ny][nx] {
                    frontier.push((x2, y2, nx, ny));
                }
            }
        }
    }

    maze
}

/// Generates a maze using Kruskal's algorithm.
pub fn kruskal(width: usize, height: usize, seed: u64) -> Maze {
    let mut maze = Maze::new(width, height);
    let mut rng = Rng::new(seed);

    // Union-find data structure
    let num_cells = width * height;
    let mut parent: Vec<usize> = (0..num_cells).collect();
    let mut rank = vec![0usize; num_cells];

    fn find(parent: &mut [usize], x: usize) -> usize {
        if parent[x] != x {
            parent[x] = find(parent, parent[x]);
        }
        parent[x]
    }

    fn union(parent: &mut [usize], rank: &mut [usize], x: usize, y: usize) -> bool {
        let px = find(parent, x);
        let py = find(parent, y);
        if px == py {
            return false;
        }

        if rank[px] < rank[py] {
            parent[px] = py;
        } else if rank[px] > rank[py] {
            parent[py] = px;
        } else {
            parent[py] = px;
            rank[px] += 1;
        }
        true
    }

    // Carve all cells
    for y in 0..height {
        for x in 0..width {
            maze.carve(x, y);
        }
    }

    // Create list of all internal walls
    let mut walls: Vec<(usize, usize, usize, usize)> = Vec::new();

    for y in 0..height {
        for x in 0..width {
            // Right wall
            if x + 1 < width {
                walls.push((x, y, x + 1, y));
            }
            // Bottom wall
            if y + 1 < height {
                walls.push((x, y, x, y + 1));
            }
        }
    }

    // Shuffle walls
    rng.shuffle(&mut walls);

    // Process walls
    for (x1, y1, x2, y2) in walls {
        let cell1 = y1 * width + x1;
        let cell2 = y2 * width + x2;

        if union(&mut parent, &mut rank, cell1, cell2) {
            maze.carve_wall(x1, y1, x2, y2);
        }
    }

    maze
}

/// Generates a maze using Eller's algorithm.
pub fn eller(width: usize, height: usize, seed: u64) -> Maze {
    let mut maze = Maze::new(width, height);
    let mut rng = Rng::new(seed);

    // Each cell belongs to a set (identified by a unique ID)
    let mut row_sets: Vec<usize> = vec![0; width];
    let mut next_set_id = 0;

    // Initialize first row - each cell in its own set
    for x in 0..width {
        row_sets[x] = next_set_id;
        next_set_id += 1;
        maze.carve(x, 0);
    }

    for y in 0..height {
        // Randomly join adjacent cells in the same row
        for x in 0..width.saturating_sub(1) {
            if row_sets[x] != row_sets[x + 1] && (y == height - 1 || rng.coin_flip()) {
                // Join the sets
                let old_set = row_sets[x + 1];
                let new_set = row_sets[x];
                for s in &mut row_sets {
                    if *s == old_set {
                        *s = new_set;
                    }
                }
                maze.carve_wall(x, y, x + 1, y);
            }
        }

        // If not the last row, create vertical connections
        if y < height - 1 {
            // Group cells by set
            let mut set_cells: std::collections::HashMap<usize, Vec<usize>> =
                std::collections::HashMap::new();
            for x in 0..width {
                set_cells.entry(row_sets[x]).or_default().push(x);
            }

            // Each set must have at least one vertical connection
            let mut next_row_sets = vec![0; width];
            for (set_id, cells) in set_cells {
                // Choose at least one cell to connect down
                let mut connected = false;
                for &x in &cells {
                    // Last cell in set must connect if none have
                    let is_last = x == *cells.last().unwrap();
                    if !connected && is_last || rng.coin_flip() {
                        next_row_sets[x] = set_id;
                        maze.carve(x, y + 1);
                        maze.carve_wall(x, y, x, y + 1);
                        connected = true;
                    }
                }
            }

            // Cells not connected to above get new set IDs
            for x in 0..width {
                if next_row_sets[x] == 0 && !maze.is_passage(2 * x + 1, 2 * (y + 1) + 1) {
                    next_row_sets[x] = next_set_id;
                    next_set_id += 1;
                    maze.carve(x, y + 1);
                }
            }

            row_sets = next_row_sets;
        }
    }

    maze
}

/// Generates a maze using binary tree algorithm.
///
/// Simple and fast, but produces mazes with diagonal bias
/// (passages tend toward north-west corner).
fn binary_tree(width: usize, height: usize, seed: u64) -> Maze {
    let mut maze = Maze::new(width, height);
    let mut rng = Rng::new(seed);

    for y in 0..height {
        for x in 0..width {
            maze.carve(x, y);

            // Choose to carve north or west (if possible)
            let can_north = y > 0;
            let can_west = x > 0;

            if can_north && can_west {
                if rng.coin_flip() {
                    maze.carve_wall(x, y, x, y - 1);
                } else {
                    maze.carve_wall(x, y, x - 1, y);
                }
            } else if can_north {
                maze.carve_wall(x, y, x, y - 1);
            } else if can_west {
                maze.carve_wall(x, y, x - 1, y);
            }
        }
    }

    maze
}

/// Generates a maze using sidewinder algorithm.
///
/// Similar to binary tree but with horizontal runs.
/// Produces mazes with horizontal bias.
fn sidewinder(width: usize, height: usize, seed: u64) -> Maze {
    let mut maze = Maze::new(width, height);
    let mut rng = Rng::new(seed);

    for y in 0..height {
        let mut run_start = 0;

        for x in 0..width {
            maze.carve(x, y);

            let at_east_boundary = x == width - 1;
            let at_north_boundary = y == 0;

            let should_close = at_east_boundary || (!at_north_boundary && rng.coin_flip());

            if should_close {
                if !at_north_boundary {
                    // Carve north from a random cell in the run
                    let run_length = x - run_start + 1;
                    let carve_x = run_start + (rng.next() as usize % run_length);
                    maze.carve_wall(carve_x, y, carve_x, y - 1);
                }
                run_start = x + 1;
            } else {
                // Carve east
                maze.carve_wall(x, y, x + 1, y);
            }
        }
    }

    maze
}

/// Operation for maze generation.
///
/// Takes a seed (u64) and produces a Maze.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dynop", derive(rhizome_resin_op::Op))]
#[cfg_attr(feature = "dynop", op(input = u64, output = Maze))]
pub struct GenerateMaze {
    /// Width in cells.
    pub width: usize,
    /// Height in cells.
    pub height: usize,
    /// Algorithm to use.
    pub algorithm: MazeAlgorithm,
    /// Whether to add entrance at top-left.
    pub add_entrance: bool,
    /// Whether to add exit at bottom-right.
    pub add_exit: bool,
}

impl Default for GenerateMaze {
    fn default() -> Self {
        Self {
            width: 10,
            height: 10,
            algorithm: MazeAlgorithm::RecursiveBacktracker,
            add_entrance: true,
            add_exit: true,
        }
    }
}

impl GenerateMaze {
    /// Creates a new maze config.
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            ..Default::default()
        }
    }

    /// Applies this operation to generate a maze.
    pub fn apply(&self, seed: &u64) -> Maze {
        let mut maze = generate_maze(self.width, self.height, self.algorithm, *seed);

        if self.add_entrance {
            maze.add_entrance();
        }
        if self.add_exit {
            maze.add_exit();
        }

        maze
    }
}

/// Backwards-compatible type alias.
pub type MazeConfig = GenerateMaze;

/// Generates a maze using the given configuration.
pub fn generate_maze_with_config(config: &MazeConfig, seed: u64) -> Maze {
    config.apply(&seed)
}

/// Finds a path through the maze using BFS.
///
/// Returns the path as grid coordinates, or None if no path exists.
pub fn solve_maze(
    maze: &Maze,
    start: (usize, usize),
    end: (usize, usize),
) -> Option<Vec<(usize, usize)>> {
    use std::collections::{HashMap, VecDeque};

    let mut visited = HashMap::new();
    let mut queue = VecDeque::new();

    queue.push_back(start);
    visited.insert(start, start);

    let directions: [(i32, i32); 4] = [(0, -1), (1, 0), (0, 1), (-1, 0)];

    while let Some(pos) = queue.pop_front() {
        if pos == end {
            // Reconstruct path
            let mut path = Vec::new();
            let mut current = end;
            while current != start {
                path.push(current);
                current = visited[&current];
            }
            path.push(start);
            path.reverse();
            return Some(path);
        }

        for (dx, dy) in directions {
            let nx = pos.0 as i32 + dx;
            let ny = pos.1 as i32 + dy;

            if nx >= 0 && ny >= 0 {
                let nx = nx as usize;
                let ny = ny as usize;

                if maze.is_passage(nx, ny) && !visited.contains_key(&(nx, ny)) {
                    visited.insert((nx, ny), pos);
                    queue.push_back((nx, ny));
                }
            }
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_maze_creation() {
        let maze = Maze::new(5, 5);
        assert_eq!(maze.width(), 5);
        assert_eq!(maze.height(), 5);
        assert_eq!(maze.grid_width(), 11);
        assert_eq!(maze.grid_height(), 11);
    }

    #[test]
    fn test_maze_carve() {
        let mut maze = Maze::new(3, 3);
        maze.carve(0, 0);
        assert!(maze.is_passage(1, 1)); // Cell (0,0) maps to grid (1,1)
    }

    #[test]
    fn test_maze_carve_wall() {
        let mut maze = Maze::new(3, 3);
        maze.carve(0, 0);
        maze.carve(1, 0);
        maze.carve_wall(0, 0, 1, 0);

        // Wall between cells should be carved
        assert!(maze.is_passage(2, 1)); // Wall at grid (2,1)
    }

    #[test]
    fn test_recursive_backtracker() {
        let maze = recursive_backtracker(5, 5, 12345);

        // All cells should be carved
        for y in 0..5 {
            for x in 0..5 {
                let (gx, gy) = (2 * x + 1, 2 * y + 1);
                assert!(maze.is_passage(gx, gy), "Cell ({}, {}) not carved", x, y);
            }
        }
    }

    #[test]
    fn test_prim() {
        let maze = prim(5, 5, 12345);

        // All cells should be carved
        for y in 0..5 {
            for x in 0..5 {
                let (gx, gy) = (2 * x + 1, 2 * y + 1);
                assert!(maze.is_passage(gx, gy), "Cell ({}, {}) not carved", x, y);
            }
        }
    }

    #[test]
    fn test_kruskal() {
        let maze = kruskal(5, 5, 12345);

        // All cells should be carved
        for y in 0..5 {
            for x in 0..5 {
                let (gx, gy) = (2 * x + 1, 2 * y + 1);
                assert!(maze.is_passage(gx, gy), "Cell ({}, {}) not carved", x, y);
            }
        }
    }

    #[test]
    fn test_eller() {
        let maze = eller(5, 5, 12345);

        // All cells should be carved
        for y in 0..5 {
            for x in 0..5 {
                let (gx, gy) = (2 * x + 1, 2 * y + 1);
                assert!(maze.is_passage(gx, gy), "Cell ({}, {}) not carved", x, y);
            }
        }
    }

    #[test]
    fn test_generate_maze() {
        for algorithm in [
            MazeAlgorithm::RecursiveBacktracker,
            MazeAlgorithm::Prim,
            MazeAlgorithm::Kruskal,
            MazeAlgorithm::Eller,
        ] {
            let maze = generate_maze(5, 5, algorithm, 12345);
            assert_eq!(maze.width(), 5);
            assert_eq!(maze.height(), 5);
        }
    }

    #[test]
    fn test_maze_config() {
        let config = MazeConfig {
            width: 10,
            height: 10,
            algorithm: MazeAlgorithm::Prim,
            add_entrance: true,
            add_exit: true,
        };

        let maze = generate_maze_with_config(&config, 12345);
        assert_eq!(maze.width(), 10);

        // Entrance should be open
        assert!(maze.is_passage(0, 1));
    }

    #[test]
    fn test_maze_to_string() {
        let maze = recursive_backtracker(3, 3, 12345);
        let s = maze.to_string_art();
        assert!(!s.is_empty());
        assert!(s.contains('#')); // Has walls
        assert!(s.contains(' ')); // Has passages
    }

    #[test]
    fn test_maze_to_grid() {
        let maze = recursive_backtracker(3, 3, 12345);
        let grid = maze.to_grid();

        assert_eq!(grid.len(), maze.grid_height());
        assert_eq!(grid[0].len(), maze.grid_width());
    }

    #[test]
    fn test_solve_maze() {
        let mut maze = recursive_backtracker(5, 5, 12345);
        maze.add_entrance();
        maze.add_exit();

        // Find path from entrance to exit
        let start = (1, 1); // Top-left cell
        let end = (maze.grid_width() - 2, maze.grid_height() - 2); // Bottom-right cell

        let path = solve_maze(&maze, start, end);
        assert!(path.is_some(), "Should find a path through the maze");

        let path = path.unwrap();
        assert!(path.len() > 1);
        assert_eq!(path[0], start);
        assert_eq!(*path.last().unwrap(), end);
    }

    #[test]
    fn test_maze_is_connected() {
        // Test that all generated mazes are connected (perfect mazes)
        for algorithm in [
            MazeAlgorithm::RecursiveBacktracker,
            MazeAlgorithm::Prim,
            MazeAlgorithm::Kruskal,
            MazeAlgorithm::Eller,
        ] {
            let maze = generate_maze(5, 5, algorithm, 54321);

            // Check that we can reach all cells from (1,1)
            let start = (1, 1);
            let mut reachable = std::collections::HashSet::new();
            let mut stack = vec![start];

            while let Some(pos) = stack.pop() {
                if reachable.contains(&pos) {
                    continue;
                }
                reachable.insert(pos);

                let directions: [(i32, i32); 4] = [(0, -1), (1, 0), (0, 1), (-1, 0)];
                for (dx, dy) in directions {
                    let nx = pos.0 as i32 + dx;
                    let ny = pos.1 as i32 + dy;
                    if nx >= 0 && ny >= 0 && maze.is_passage(nx as usize, ny as usize) {
                        stack.push((nx as usize, ny as usize));
                    }
                }
            }

            // All cells should be reachable
            for y in 0..5 {
                for x in 0..5 {
                    let (gx, gy) = (2 * x + 1, 2 * y + 1);
                    assert!(
                        reachable.contains(&(gx, gy)),
                        "Cell ({}, {}) not reachable with {:?}",
                        x,
                        y,
                        algorithm
                    );
                }
            }
        }
    }

    #[test]
    fn test_deterministic() {
        // Same seed should produce same maze
        let maze1 = recursive_backtracker(5, 5, 99999);
        let maze2 = recursive_backtracker(5, 5, 99999);

        assert_eq!(maze1.to_grid(), maze2.to_grid());
    }

    #[test]
    fn test_different_seeds() {
        // Different seeds should produce different mazes
        let maze1 = recursive_backtracker(5, 5, 11111);
        let maze2 = recursive_backtracker(5, 5, 22222);

        assert_ne!(maze1.to_grid(), maze2.to_grid());
    }
}
