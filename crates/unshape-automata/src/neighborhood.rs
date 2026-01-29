#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

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
