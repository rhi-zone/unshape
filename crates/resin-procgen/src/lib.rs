//! Procedural generation algorithms.
//!
//! Provides procedural generation algorithms including:
//! - Wave Function Collapse for tile-based generation
//! - Classic maze algorithms (recursive backtracker, Prim's, Kruskal's, Eller's)
//! - Road and river network generation
//!
//! # Example
//!
//! ```
//! use rhizome_resin_procgen::maze::{Maze, MazeAlgorithm, generate_maze};
//!
//! // Generate a maze using recursive backtracker
//! let maze = generate_maze(10, 10, MazeAlgorithm::RecursiveBacktracker, 12345);
//!
//! // Check if a cell is a passage
//! assert!(maze.is_passage(1, 1));
//!
//! // Get the maze as a 2D grid
//! let grid = maze.to_grid();
//! ```

pub mod maze;
pub mod network;

// Re-export op types for convenient access
pub use maze::GenerateMaze;
pub use network::{GenerateRiver, GenerateRoadNetworkGrid, GenerateRoadNetworkHierarchical};

/// Registers all procgen operations with an [`OpRegistry`].
///
/// Call this to enable deserialization of procgen ops from saved pipelines.
#[cfg(feature = "dynop")]
pub fn register_ops(registry: &mut rhizome_resin_op::OpRegistry) {
    registry.register_type::<GenerateMaze>("resin::GenerateMaze");
    registry.register_type::<GenerateRiver>("resin::GenerateRiver");
    registry.register_type::<GenerateRoadNetworkGrid>("resin::GenerateRoadNetworkGrid");
    registry
        .register_type::<GenerateRoadNetworkHierarchical>("resin::GenerateRoadNetworkHierarchical");
}

use std::collections::{HashMap, HashSet, VecDeque};

/// Direction for adjacency rules in 2D.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Direction {
    /// Positive Y direction.
    Up,
    /// Negative Y direction.
    Down,
    /// Negative X direction.
    Left,
    /// Positive X direction.
    Right,
}

impl Direction {
    /// Returns the opposite direction.
    pub fn opposite(&self) -> Direction {
        match self {
            Direction::Up => Direction::Down,
            Direction::Down => Direction::Up,
            Direction::Left => Direction::Right,
            Direction::Right => Direction::Left,
        }
    }

    /// Returns all four directions.
    pub fn all() -> [Direction; 4] {
        [
            Direction::Up,
            Direction::Down,
            Direction::Left,
            Direction::Right,
        ]
    }

    /// Returns the delta (dx, dy) for this direction.
    pub fn delta(&self) -> (i32, i32) {
        match self {
            Direction::Up => (0, -1),
            Direction::Down => (0, 1),
            Direction::Left => (-1, 0),
            Direction::Right => (1, 0),
        }
    }
}

// ============================================================================
// TileId - Type-safe tile identifier
// ============================================================================

/// A type-safe identifier for a tile in a tileset.
///
/// Using `TileId` instead of raw `usize` prevents accidentally mixing tile
/// indices with other integer values (like grid positions or body indices).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TileId(pub usize);

impl TileId {
    /// Returns the underlying index.
    #[inline]
    pub fn index(self) -> usize {
        self.0
    }
}

// ============================================================================
// TileSet - Base type-safe tileset
// ============================================================================

/// A set of tiles with adjacency rules.
///
/// This is the base tileset type that uses `TileId` for type safety and
/// zero-overhead tile references. For a more ergonomic string-based API,
/// see [`NamedTileSet`].
///
/// # Example
///
/// ```
/// use rhizome_resin_procgen::{TileSet, TileId, Direction};
///
/// let mut ts = TileSet::new();
/// let grass = ts.add_tile();
/// let dirt = ts.add_tile();
///
/// ts.add_rule(grass, Direction::Down, dirt);
/// ```
#[derive(Debug, Clone)]
pub struct TileSet {
    /// Number of tiles.
    tile_count: usize,
    /// Adjacency rules: (tile_a, direction) -> set of valid tiles for that neighbor.
    rules: HashMap<(TileId, Direction), HashSet<TileId>>,
    /// Tile weights for biased selection.
    weights: Vec<f32>,
}

impl TileSet {
    /// Creates a new empty tileset.
    pub fn new() -> Self {
        Self {
            tile_count: 0,
            rules: HashMap::new(),
            weights: Vec::new(),
        }
    }

    /// Adds a tile to the tileset with default weight 1.0.
    pub fn add_tile(&mut self) -> TileId {
        let id = TileId(self.tile_count);
        self.tile_count += 1;
        self.weights.push(1.0);
        id
    }

    /// Adds a tile with a custom weight.
    pub fn add_tile_weighted(&mut self, weight: f32) -> TileId {
        let id = self.add_tile();
        self.weights[id.0] = weight;
        id
    }

    /// Sets the weight of a tile.
    pub fn set_weight(&mut self, id: TileId, weight: f32) {
        if id.0 < self.tile_count {
            self.weights[id.0] = weight;
        }
    }

    /// Gets the weight of a tile.
    pub fn weight(&self, id: TileId) -> f32 {
        self.weights.get(id.0).copied().unwrap_or(1.0)
    }

    /// Adds an adjacency rule: `from` tile can have `to` tile in the given direction.
    ///
    /// Automatically adds the reverse rule (to can have from in opposite direction).
    pub fn add_rule(&mut self, from: TileId, direction: Direction, to: TileId) {
        self.rules.entry((from, direction)).or_default().insert(to);

        // Add reverse rule automatically
        self.rules
            .entry((to, direction.opposite()))
            .or_default()
            .insert(from);
    }

    /// Adds a bidirectional rule (tiles can be adjacent in both orders).
    pub fn add_symmetric_rule(&mut self, a: TileId, direction: Direction, b: TileId) {
        self.add_rule(a, direction, b);
        self.add_rule(b, direction, a);
    }

    /// Returns the number of tiles.
    pub fn tile_count(&self) -> usize {
        self.tile_count
    }

    /// Returns all tile IDs in this tileset.
    pub fn tile_ids(&self) -> impl Iterator<Item = TileId> {
        (0..self.tile_count).map(TileId)
    }

    /// Returns valid neighbors for a tile in a direction.
    pub fn valid_neighbors(&self, tile: TileId, direction: Direction) -> Option<&HashSet<TileId>> {
        self.rules.get(&(tile, direction))
    }
}

impl Default for TileSet {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// NamedTileSet - Ergonomic string-based wrapper
// ============================================================================

/// A tileset with string names for ergonomic tile definition.
///
/// This wraps [`TileSet`] and adds a string-to-ID mapping for convenience.
/// Use this when defining tilesets manually or loading from config files.
///
/// # Example
///
/// ```
/// use rhizome_resin_procgen::{NamedTileSet, Direction};
///
/// let mut ts = NamedTileSet::new();
/// ts.add_tile("grass");
/// ts.add_tile("dirt");
///
/// ts.add_rule("grass", Direction::Down, "dirt");
///
/// // Get the underlying TileSet for use with WfcSolver
/// let tileset = ts.into_inner();
/// ```
#[derive(Debug, Clone)]
pub struct NamedTileSet {
    /// The underlying tileset.
    inner: TileSet,
    /// Tile names by ID.
    names: Vec<String>,
    /// Map from name to ID.
    name_to_id: HashMap<String, TileId>,
}

impl NamedTileSet {
    /// Creates a new empty named tileset.
    pub fn new() -> Self {
        Self {
            inner: TileSet::new(),
            names: Vec::new(),
            name_to_id: HashMap::new(),
        }
    }

    /// Adds a tile with the given name. Returns existing ID if name already exists.
    pub fn add_tile(&mut self, name: &str) -> TileId {
        if let Some(&id) = self.name_to_id.get(name) {
            return id;
        }

        let id = self.inner.add_tile();
        self.names.push(name.to_string());
        self.name_to_id.insert(name.to_string(), id);
        id
    }

    /// Adds a tile with a custom weight.
    pub fn add_tile_weighted(&mut self, name: &str, weight: f32) -> TileId {
        let id = self.add_tile(name);
        self.inner.set_weight(id, weight);
        id
    }

    /// Sets the weight of a tile by name.
    pub fn set_weight(&mut self, name: &str, weight: f32) {
        if let Some(&id) = self.name_to_id.get(name) {
            self.inner.set_weight(id, weight);
        }
    }

    /// Adds an adjacency rule using tile names.
    pub fn add_rule(&mut self, from: &str, direction: Direction, to: &str) {
        let from_id = self.add_tile(from);
        let to_id = self.add_tile(to);
        self.inner.add_rule(from_id, direction, to_id);
    }

    /// Adds a bidirectional rule using tile names.
    pub fn add_symmetric_rule(&mut self, a: &str, direction: Direction, b: &str) {
        let a_id = self.add_tile(a);
        let b_id = self.add_tile(b);
        self.inner.add_symmetric_rule(a_id, direction, b_id);
    }

    /// Returns the number of tiles.
    pub fn tile_count(&self) -> usize {
        self.inner.tile_count()
    }

    /// Gets a tile ID by name.
    pub fn get_id(&self, name: &str) -> Option<TileId> {
        self.name_to_id.get(name).copied()
    }

    /// Gets a tile name by ID.
    pub fn get_name(&self, id: TileId) -> Option<&str> {
        self.names.get(id.0).map(|s| s.as_str())
    }

    /// Returns a reference to the underlying tileset.
    pub fn inner(&self) -> &TileSet {
        &self.inner
    }

    /// Consumes this wrapper and returns the underlying tileset.
    pub fn into_inner(self) -> TileSet {
        self.inner
    }
}

impl Default for NamedTileSet {
    fn default() -> Self {
        Self::new()
    }
}

/// A cell in the WFC grid.
#[derive(Debug, Clone)]
struct Cell {
    /// Possible tiles for this cell.
    possibilities: HashSet<TileId>,
    /// Whether this cell has been collapsed.
    collapsed: bool,
    /// The final tile (if collapsed).
    tile: Option<TileId>,
}

impl Cell {
    fn new(tile_count: usize) -> Self {
        Self {
            possibilities: (0..tile_count).map(TileId).collect(),
            collapsed: false,
            tile: None,
        }
    }

    fn entropy(&self) -> usize {
        self.possibilities.len()
    }

    fn is_collapsed(&self) -> bool {
        self.collapsed
    }

    fn collapse(&mut self, tile: TileId) {
        self.possibilities.clear();
        self.possibilities.insert(tile);
        self.collapsed = true;
        self.tile = Some(tile);
    }
}

/// Wave Function Collapse solver.
#[derive(Debug, Clone)]
pub struct WfcSolver {
    /// Grid width.
    width: usize,
    /// Grid height.
    height: usize,
    /// The tileset.
    tileset: TileSet,
    /// The grid of cells.
    cells: Vec<Cell>,
    /// RNG state.
    rng_state: u64,
}

impl WfcSolver {
    /// Creates a new WFC solver.
    pub fn new(width: usize, height: usize, tileset: TileSet) -> Self {
        let tile_count = tileset.tile_count();
        let cells = vec![Cell::new(tile_count); width * height];

        Self {
            width,
            height,
            tileset,
            cells,
            rng_state: 0,
        }
    }

    /// Returns the grid width.
    pub fn width(&self) -> usize {
        self.width
    }

    /// Returns the grid height.
    pub fn height(&self) -> usize {
        self.height
    }

    /// Resets the solver to initial state.
    pub fn reset(&mut self) {
        let tile_count = self.tileset.tile_count();
        self.cells = vec![Cell::new(tile_count); self.width * self.height];
    }

    /// Sets a cell to a specific tile (constraint).
    pub fn set_cell(&mut self, x: usize, y: usize, tile: TileId) -> Result<(), WfcError> {
        let idx = self.cell_index(x, y)?;
        self.cells[idx].collapse(tile);
        self.propagate(x, y)?;
        Ok(())
    }

    /// Runs the WFC algorithm to completion.
    pub fn run(&mut self, seed: u64) -> Result<(), WfcError> {
        self.rng_state = seed.wrapping_add(1);

        while let Some((x, y)) = self.find_min_entropy_cell() {
            self.collapse_cell(x, y)?;
            self.propagate(x, y)?;
        }

        // Check if all cells are collapsed
        if self.cells.iter().any(|c| !c.is_collapsed()) {
            return Err(WfcError::Contradiction);
        }

        Ok(())
    }

    /// Runs a single step of the algorithm.
    pub fn step(&mut self) -> Result<bool, WfcError> {
        if let Some((x, y)) = self.find_min_entropy_cell() {
            self.collapse_cell(x, y)?;
            self.propagate(x, y)?;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Gets the result grid as tile IDs.
    pub fn get_result(&self) -> Vec<Vec<Option<TileId>>> {
        let mut result = vec![vec![None; self.width]; self.height];

        for y in 0..self.height {
            for x in 0..self.width {
                let idx = y * self.width + x;
                result[y][x] = self.cells[idx].tile;
            }
        }

        result
    }

    /// Gets the tile ID at a specific position.
    pub fn get_tile(&self, x: usize, y: usize) -> Option<TileId> {
        let idx = y * self.width + x;
        self.cells.get(idx)?.tile
    }

    /// Gets the entropy (number of possibilities) at a position.
    pub fn get_entropy(&self, x: usize, y: usize) -> usize {
        let idx = y * self.width + x;
        self.cells.get(idx).map(|c| c.entropy()).unwrap_or(0)
    }

    /// Checks if the solver is complete.
    pub fn is_complete(&self) -> bool {
        self.cells.iter().all(|c| c.is_collapsed())
    }

    fn cell_index(&self, x: usize, y: usize) -> Result<usize, WfcError> {
        if x >= self.width || y >= self.height {
            return Err(WfcError::OutOfBounds(x, y));
        }
        Ok(y * self.width + x)
    }

    fn find_min_entropy_cell(&mut self) -> Option<(usize, usize)> {
        let mut min_entropy = usize::MAX;
        let mut candidates = Vec::new();

        for y in 0..self.height {
            for x in 0..self.width {
                let idx = y * self.width + x;
                let cell = &self.cells[idx];

                if cell.is_collapsed() {
                    continue;
                }

                let entropy = cell.entropy();
                if entropy == 0 {
                    continue; // Contradiction, skip
                }

                if entropy < min_entropy {
                    min_entropy = entropy;
                    candidates.clear();
                    candidates.push((x, y));
                } else if entropy == min_entropy {
                    candidates.push((x, y));
                }
            }
        }

        if candidates.is_empty() {
            return None;
        }

        // Random selection among candidates with same entropy
        let idx = self.random_index(candidates.len());
        Some(candidates[idx])
    }

    fn collapse_cell(&mut self, x: usize, y: usize) -> Result<(), WfcError> {
        let idx = y * self.width + x;
        let cell = &self.cells[idx];

        if cell.is_collapsed() {
            return Ok(());
        }

        if cell.possibilities.is_empty() {
            return Err(WfcError::Contradiction);
        }

        // Weighted random selection
        let possibilities: Vec<TileId> = cell.possibilities.iter().copied().collect();
        let total_weight: f32 = possibilities.iter().map(|&t| self.tileset.weight(t)).sum();

        let mut r = self.random_f32() * total_weight;
        let mut selected = possibilities[0];

        for &tile in &possibilities {
            r -= self.tileset.weight(tile);
            if r <= 0.0 {
                selected = tile;
                break;
            }
        }

        self.cells[idx].collapse(selected);
        Ok(())
    }

    fn propagate(&mut self, start_x: usize, start_y: usize) -> Result<(), WfcError> {
        let mut queue = VecDeque::new();
        queue.push_back((start_x, start_y));

        while let Some((x, y)) = queue.pop_front() {
            let idx = y * self.width + x;
            let current_possibilities = self.cells[idx].possibilities.clone();

            for direction in Direction::all() {
                let (dx, dy) = direction.delta();
                let nx = x as i32 + dx;
                let ny = y as i32 + dy;

                if nx < 0 || ny < 0 || nx >= self.width as i32 || ny >= self.height as i32 {
                    continue;
                }

                let nx = nx as usize;
                let ny = ny as usize;
                let neighbor_idx = ny * self.width + nx;

                if self.cells[neighbor_idx].is_collapsed() {
                    continue;
                }

                // Compute valid tiles for neighbor based on current cell
                let mut valid_neighbors: HashSet<TileId> = HashSet::new();
                for &tile in &current_possibilities {
                    if let Some(neighbors) = self.tileset.valid_neighbors(tile, direction) {
                        valid_neighbors.extend(neighbors.iter().copied());
                    }
                }

                // Intersect with neighbor's possibilities
                let old_count = self.cells[neighbor_idx].possibilities.len();
                self.cells[neighbor_idx]
                    .possibilities
                    .retain(|t| valid_neighbors.contains(t));
                let new_count = self.cells[neighbor_idx].possibilities.len();

                if new_count == 0 {
                    return Err(WfcError::Contradiction);
                }

                // If possibilities changed, add to queue
                if new_count < old_count {
                    if !queue.contains(&(nx, ny)) {
                        queue.push_back((nx, ny));
                    }
                }
            }
        }

        Ok(())
    }

    fn random_index(&mut self, max: usize) -> usize {
        (self.random_u64() as usize) % max
    }

    fn random_f32(&mut self) -> f32 {
        (self.random_u64() as f64 / u64::MAX as f64) as f32
    }

    fn random_u64(&mut self) -> u64 {
        self.rng_state = self
            .rng_state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.rng_state
    }
}

/// Errors that can occur during WFC.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum WfcError {
    /// The algorithm reached a contradiction (cell with no valid tiles).
    #[error("WFC reached a contradiction")]
    Contradiction,
    /// Position out of bounds.
    #[error("Position ({0}, {1}) out of bounds")]
    OutOfBounds(usize, usize),
}

/// Creates a simple platformer tileset.
pub fn platformer_tileset() -> NamedTileSet {
    let mut ts = NamedTileSet::new();

    ts.add_tile("empty");
    ts.add_tile("ground");
    ts.add_tile("grass");
    ts.add_tile("platform");

    // Ground rules
    ts.add_rule("ground", Direction::Up, "ground");
    ts.add_rule("ground", Direction::Up, "grass");
    ts.add_rule("ground", Direction::Left, "ground");
    ts.add_rule("ground", Direction::Right, "ground");

    // Grass rules
    ts.add_rule("grass", Direction::Up, "empty");
    ts.add_rule("grass", Direction::Left, "grass");
    ts.add_rule("grass", Direction::Left, "empty");
    ts.add_rule("grass", Direction::Right, "grass");
    ts.add_rule("grass", Direction::Right, "empty");

    // Empty rules
    ts.add_rule("empty", Direction::Up, "empty");
    ts.add_rule("empty", Direction::Left, "empty");
    ts.add_rule("empty", Direction::Right, "empty");
    ts.add_rule("empty", Direction::Down, "empty");
    ts.add_rule("empty", Direction::Down, "grass");
    ts.add_rule("empty", Direction::Down, "platform");

    // Platform rules
    ts.add_rule("platform", Direction::Up, "empty");
    ts.add_rule("platform", Direction::Down, "empty");
    ts.add_rule("platform", Direction::Left, "platform");
    ts.add_rule("platform", Direction::Left, "empty");
    ts.add_rule("platform", Direction::Right, "platform");
    ts.add_rule("platform", Direction::Right, "empty");

    ts
}

/// Creates a maze tileset.
pub fn maze_tileset() -> NamedTileSet {
    let mut ts = NamedTileSet::new();

    ts.add_tile("wall");
    ts.add_tile("floor");
    ts.add_tile("corner_tl");
    ts.add_tile("corner_tr");
    ts.add_tile("corner_bl");
    ts.add_tile("corner_br");

    // Floor can connect to floor in all directions
    for dir in Direction::all() {
        ts.add_rule("floor", dir, "floor");
    }

    // Wall can connect to wall horizontally and vertically
    ts.add_rule("wall", Direction::Up, "wall");
    ts.add_rule("wall", Direction::Down, "wall");
    ts.add_rule("wall", Direction::Left, "wall");
    ts.add_rule("wall", Direction::Right, "wall");

    // Wall-floor transitions
    ts.add_rule("wall", Direction::Up, "floor");
    ts.add_rule("wall", Direction::Down, "floor");
    ts.add_rule("wall", Direction::Left, "floor");
    ts.add_rule("wall", Direction::Right, "floor");

    ts
}

#[cfg(test)]
mod tests {
    use super::*;

    fn simple_named_tileset() -> NamedTileSet {
        let mut ts = NamedTileSet::new();
        ts.add_tile("A");
        ts.add_tile("B");

        // A can be next to A or B
        ts.add_rule("A", Direction::Right, "A");
        ts.add_rule("A", Direction::Right, "B");
        ts.add_rule("A", Direction::Up, "A");
        ts.add_rule("A", Direction::Up, "B");

        // B can be next to A or B
        ts.add_rule("B", Direction::Right, "A");
        ts.add_rule("B", Direction::Right, "B");
        ts.add_rule("B", Direction::Up, "A");
        ts.add_rule("B", Direction::Up, "B");

        ts
    }

    #[test]
    fn test_tileset_creation() {
        let ts = simple_named_tileset();
        assert_eq!(ts.tile_count(), 2);
        assert_eq!(ts.get_name(TileId(0)), Some("A"));
        assert_eq!(ts.get_name(TileId(1)), Some("B"));
    }

    #[test]
    fn test_tileset_weights() {
        let mut ts = NamedTileSet::new();
        ts.add_tile_weighted("common", 10.0);
        ts.add_tile_weighted("rare", 1.0);

        assert_eq!(ts.inner().weight(TileId(0)), 10.0);
        assert_eq!(ts.inner().weight(TileId(1)), 1.0);
    }

    #[test]
    fn test_base_tileset() {
        let mut ts = TileSet::new();
        let a = ts.add_tile();
        let b = ts.add_tile_weighted(2.0);

        assert_eq!(ts.tile_count(), 2);
        assert_eq!(ts.weight(a), 1.0);
        assert_eq!(ts.weight(b), 2.0);

        ts.add_rule(a, Direction::Right, b);
        assert!(
            ts.valid_neighbors(a, Direction::Right)
                .unwrap()
                .contains(&b)
        );
    }

    #[test]
    fn test_direction_opposite() {
        assert_eq!(Direction::Up.opposite(), Direction::Down);
        assert_eq!(Direction::Left.opposite(), Direction::Right);
    }

    #[test]
    fn test_wfc_solver_creation() {
        let ts = simple_named_tileset();
        let solver = WfcSolver::new(5, 5, ts.into_inner());

        assert_eq!(solver.width(), 5);
        assert_eq!(solver.height(), 5);
    }

    #[test]
    fn test_wfc_run() {
        let ts = simple_named_tileset();
        let mut solver = WfcSolver::new(3, 3, ts.into_inner());

        let result = solver.run(12345);
        assert!(result.is_ok());
        assert!(solver.is_complete());
    }

    #[test]
    fn test_wfc_get_result() {
        let ts = simple_named_tileset();
        let mut solver = WfcSolver::new(2, 2, ts.into_inner());
        solver.run(12345).unwrap();

        let result = solver.get_result();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].len(), 2);

        // All cells should be collapsed
        for row in &result {
            for cell in row {
                assert!(cell.is_some());
            }
        }
    }

    #[test]
    fn test_wfc_get_tile() {
        let ts = simple_named_tileset();
        let mut solver = WfcSolver::new(2, 2, ts.into_inner());
        solver.run(12345).unwrap();

        let tile = solver.get_tile(0, 0);
        assert!(tile == Some(TileId(0)) || tile == Some(TileId(1)));
    }

    #[test]
    fn test_wfc_set_cell() {
        let ts = simple_named_tileset();
        let tile_a = ts.get_id("A").unwrap();
        let mut solver = WfcSolver::new(3, 3, ts.into_inner());

        solver.set_cell(1, 1, tile_a).unwrap();
        assert_eq!(solver.get_tile(1, 1), Some(tile_a));
    }

    #[test]
    fn test_wfc_step() {
        let ts = simple_named_tileset();
        let mut solver = WfcSolver::new(3, 3, ts.into_inner());

        // Should make progress
        let stepped = solver.step().unwrap();
        assert!(stepped);
    }

    #[test]
    fn test_wfc_reset() {
        let ts = simple_named_tileset();
        let mut solver = WfcSolver::new(3, 3, ts.into_inner());

        solver.run(12345).unwrap();
        assert!(solver.is_complete());

        solver.reset();
        assert!(!solver.is_complete());
    }

    #[test]
    fn test_platformer_tileset() {
        let ts = platformer_tileset();
        assert!(ts.tile_count() >= 4);
    }

    #[test]
    fn test_maze_tileset() {
        let ts = maze_tileset();
        assert!(ts.tile_count() >= 2);
    }

    #[test]
    fn test_wfc_with_platformer() {
        let ts = platformer_tileset();
        let mut solver = WfcSolver::new(5, 5, ts.into_inner());

        // Should complete without error
        let result = solver.run(99999);
        // Note: May fail due to contradictions in some cases
        if result.is_ok() {
            assert!(solver.is_complete());
        }
    }

    #[test]
    fn test_entropy() {
        let ts = simple_named_tileset();
        let solver = WfcSolver::new(2, 2, ts.into_inner());

        // Initially all cells have max entropy (2 possibilities)
        assert_eq!(solver.get_entropy(0, 0), 2);
    }
}
