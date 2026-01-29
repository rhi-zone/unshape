use glam::Vec3;

/// An entry in the spatial hash grid.
#[derive(Debug, Clone)]
pub(crate) struct SpatialHashEntry<T> {
    pub position: Vec3,
    pub data: T,
}

/// A spatial hash grid for broad-phase collision detection.
///
/// Divides space into a uniform grid and maps objects to cells based on their
/// positions. Efficient for uniformly distributed objects.
///
/// # Type Parameters
///
/// * `T` - The type of data associated with each entry.
///
/// # Example
///
/// ```
/// use unshape_spatial::SpatialHash;
/// use glam::Vec3;
///
/// let mut hash = SpatialHash::new(10.0); // 10 unit cell size
///
/// hash.insert(Vec3::new(5.0, 5.0, 5.0), "A");
/// hash.insert(Vec3::new(15.0, 5.0, 5.0), "B");
/// hash.insert(Vec3::new(5.5, 5.5, 5.5), "C"); // Same cell as A
///
/// // Query nearby objects
/// let nearby: Vec<_> = hash.query_cell(Vec3::new(5.0, 5.0, 5.0)).collect();
/// assert_eq!(nearby.len(), 2); // A and C
/// ```
#[derive(Debug)]
pub struct SpatialHash<T> {
    cell_size: f32,
    inv_cell_size: f32,
    cells: std::collections::HashMap<(i32, i32, i32), Vec<SpatialHashEntry<T>>>,
}

impl<T> SpatialHash<T> {
    /// Creates a new spatial hash with the given cell size.
    pub fn new(cell_size: f32) -> Self {
        Self {
            cell_size,
            inv_cell_size: 1.0 / cell_size,
            cells: std::collections::HashMap::new(),
        }
    }

    /// Returns the cell size.
    pub fn cell_size(&self) -> f32 {
        self.cell_size
    }

    fn cell_key(&self, position: Vec3) -> (i32, i32, i32) {
        (
            (position.x * self.inv_cell_size).floor() as i32,
            (position.y * self.inv_cell_size).floor() as i32,
            (position.z * self.inv_cell_size).floor() as i32,
        )
    }

    /// Inserts an entry at the given position.
    pub fn insert(&mut self, position: Vec3, data: T) {
        let key = self.cell_key(position);
        self.cells
            .entry(key)
            .or_default()
            .push(SpatialHashEntry { position, data });
    }

    /// Queries all entries in the same cell as the given position.
    pub fn query_cell(&self, position: Vec3) -> impl Iterator<Item = (Vec3, &T)> {
        let key = self.cell_key(position);
        self.cells
            .get(&key)
            .map(|entries| {
                entries
                    .iter()
                    .map(|e| (e.position, &e.data))
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default()
            .into_iter()
    }

    /// Queries all entries in the cell containing position and all 26 neighboring cells.
    pub fn query_neighbors(&self, position: Vec3) -> impl Iterator<Item = (Vec3, &T)> {
        let (cx, cy, cz) = self.cell_key(position);
        let mut results = Vec::new();

        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    let key = (cx + dx, cy + dy, cz + dz);
                    if let Some(entries) = self.cells.get(&key) {
                        for e in entries {
                            results.push((e.position, &e.data));
                        }
                    }
                }
            }
        }

        results.into_iter()
    }

    /// Queries all entries within the given radius of a position.
    pub fn query_radius(&self, position: Vec3, radius: f32) -> impl Iterator<Item = (Vec3, &T)> {
        let radius_sq = radius * radius;
        let (cx, cy, cz) = self.cell_key(position);
        let cell_radius = (radius * self.inv_cell_size).ceil() as i32;

        let mut results = Vec::new();

        for dx in -cell_radius..=cell_radius {
            for dy in -cell_radius..=cell_radius {
                for dz in -cell_radius..=cell_radius {
                    let key = (cx + dx, cy + dy, cz + dz);
                    if let Some(entries) = self.cells.get(&key) {
                        for e in entries {
                            if e.position.distance_squared(position) <= radius_sq {
                                results.push((e.position, &e.data));
                            }
                        }
                    }
                }
            }
        }

        results.into_iter()
    }

    /// Returns the total number of entries.
    pub fn len(&self) -> usize {
        self.cells.values().map(|v| v.len()).sum()
    }

    /// Returns `true` if empty.
    pub fn is_empty(&self) -> bool {
        self.cells.is_empty()
    }

    /// Clears all entries.
    pub fn clear(&mut self) {
        self.cells.clear();
    }
}
