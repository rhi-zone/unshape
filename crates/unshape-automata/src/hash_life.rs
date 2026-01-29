use crate::CellularAutomaton2D;

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
