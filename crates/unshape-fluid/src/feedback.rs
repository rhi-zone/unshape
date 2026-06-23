//! Recurrent-graph integration for the fluid simulations.
//!
//! Ports the stateful fluid sims ([`FluidGrid2D`]/[`FluidGrid3D`] Jos Stam stable
//! fluids, [`SmokeGrid2D`]/[`SmokeGrid3D`] buoyant smoke, [`Sph2D`]/[`Sph3D`]
//! smoothed-particle hydrodynamics) onto the feedback-edge mechanism in
//! `unshape-core` (`docs/design/recurrent-graphs.md`). The entire mutable sim
//! state (fields/particles plus the embedded config) is carried on a feedback
//! wire as an opaque [`Value`], so each step node is a pure `&self` [`DynNode`]:
//! it clones the previous state, advances one step, and returns the new state.
//! No state lives in the node.
//!
//! # Tick-0 state
//!
//! The initial state is produced by an in-graph `*Init` source node (a pure
//! `&self` `DynNode` taking no inputs) wired into the matching `*Step`'s state
//! port with a *direct* edge, alongside the step's feedback self-loop. On tick 0
//! the `Init` node seeds the state; later ticks use the carried feedback value.
//! This makes the sims rewindable via
//! [`Graph::run_to_tick`](unshape_core::Graph::run_to_tick) with no manual seed.

use std::any::Any;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use unshape_core::{
    DataLocation, DynNode, EvalContext, GraphError, GraphValue, PortDescriptor, Value, ValueType,
};

use crate::{
    FluidConfig, FluidGrid2D, FluidGrid3D, SmokeConfig, SmokeGrid2D, SmokeGrid3D, Sph2D, Sph3D,
    SphConfig, SphConfig3D,
};

/// The opaque value type name for [`FluidGrid2D`] state on a wire.
pub const FLUID_GRID_2D_NAME: &str = "FluidGrid2D";

impl GraphValue for FluidGrid2D {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn type_name(&self) -> &'static str {
        FLUID_GRID_2D_NAME
    }

    fn location(&self) -> DataLocation {
        DataLocation::Cpu
    }
}

/// Returns the [`ValueType`] used for [`FluidGrid2D`] state on a wire.
pub fn fluid_grid_2d_type() -> ValueType {
    ValueType::of::<FluidGrid2D>(FLUID_GRID_2D_NAME)
}

/// A pure-data source applied to a fresh grid by [`FluidInit`].
///
/// Mirrors the imperative `add_density` / `add_velocity` helpers on
/// [`FluidGrid2D`].
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum FluidSource {
    /// Inject density at a cell.
    Density {
        /// Cell x.
        x: usize,
        /// Cell y.
        y: usize,
        /// Amount of density to add.
        amount: f32,
    },
    /// Inject velocity at a cell.
    Velocity {
        /// Cell x.
        x: usize,
        /// Cell y.
        y: usize,
        /// X velocity.
        vx: f32,
        /// Y velocity.
        vy: f32,
    },
}

/// Pure in-graph **source** node producing the initial [`FluidGrid2D`].
///
/// Takes no inputs; outputs a fresh grid of `width`×`height` with the given
/// [`FluidConfig`], after applying each [`FluidSource`] in order. Deterministic.
///
/// Wire this into a [`Step`]'s state port with a *direct* edge (tick-0 seed),
/// alongside the [`Step`]'s feedback self-loop.
///
/// # Ports
/// - Output `0` `"state"`: `Custom(FluidGrid2D)` — initial grid.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct FluidInit {
    /// Grid width.
    pub width: usize,
    /// Grid height.
    pub height: usize,
    /// Simulation configuration (diffusion, iterations, dt).
    pub config: FluidConfig,
    /// Sources applied to the fresh grid, in order.
    pub sources: Vec<FluidSource>,
}

impl FluidInit {
    /// Creates an init node with the given size and config and no sources.
    pub fn new(width: usize, height: usize, config: FluidConfig) -> Self {
        Self {
            width,
            height,
            config,
            sources: Vec::new(),
        }
    }

    /// Adds a density source.
    pub fn with_density(mut self, x: usize, y: usize, amount: f32) -> Self {
        self.sources.push(FluidSource::Density { x, y, amount });
        self
    }

    /// Adds a velocity source.
    pub fn with_velocity(mut self, x: usize, y: usize, vx: f32, vy: f32) -> Self {
        self.sources.push(FluidSource::Velocity { x, y, vx, vy });
        self
    }

    /// Builds the initial [`FluidGrid2D`] from this config (pure).
    pub fn build(&self) -> FluidGrid2D {
        let mut g = FluidGrid2D::new(self.width, self.height, self.config.clone());
        for source in &self.sources {
            match *source {
                FluidSource::Density { x, y, amount } => g.add_density(x, y, amount),
                FluidSource::Velocity { x, y, vx, vy } => g.add_velocity(x, y, vx, vy),
            }
        }
        g
    }
}

impl DynNode for FluidInit {
    fn type_name(&self) -> &'static str {
        "fluid::feedback::FluidInit"
    }

    fn inputs(&self) -> Vec<PortDescriptor> {
        vec![]
    }

    fn outputs(&self) -> Vec<PortDescriptor> {
        vec![PortDescriptor::new("state", fluid_grid_2d_type())]
    }

    fn execute(&self, _inputs: &[Value], _ctx: &EvalContext) -> Result<Vec<Value>, GraphError> {
        Ok(vec![Value::opaque(self.build())])
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Pure per-tick step node for the 2D fluid simulation.
///
/// State lives on the feedback edge, not in the node. `execute` clones the
/// previous [`FluidGrid2D`], advances it one step (clone-and-advance, mirroring
/// [`FluidGrid2D::step`]), and returns the new grid. Simulation parameters
/// (diffusion, iterations, dt) live in the grid's embedded config.
///
/// # Ports
/// - Input `0` `"state"`: `Custom(FluidGrid2D)` — previous-tick grid.
/// - Output `0` `"state"`: `Custom(FluidGrid2D)` — advanced grid.
#[derive(Debug, Clone, Copy, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Step;

impl DynNode for Step {
    fn type_name(&self) -> &'static str {
        "fluid::feedback::Step"
    }

    fn inputs(&self) -> Vec<PortDescriptor> {
        vec![PortDescriptor::new("state", fluid_grid_2d_type())]
    }

    fn outputs(&self) -> Vec<PortDescriptor> {
        vec![PortDescriptor::new("state", fluid_grid_2d_type())]
    }

    fn execute(&self, inputs: &[Value], _ctx: &EvalContext) -> Result<Vec<Value>, GraphError> {
        let prev = inputs[0].downcast_ref::<FluidGrid2D>().ok_or_else(|| {
            GraphError::ExecutionError(
                "fluid::feedback::Step expects a FluidGrid2D state input".to_string(),
            )
        })?;
        let mut next = prev.clone();
        next.step();
        Ok(vec![Value::opaque(next)])
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

// ===========================================================================
// FluidGrid3D
// ===========================================================================

/// The opaque value type name for [`FluidGrid3D`] state on a wire.
pub const FLUID_GRID_3D_NAME: &str = "FluidGrid3D";

impl GraphValue for FluidGrid3D {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn type_name(&self) -> &'static str {
        FLUID_GRID_3D_NAME
    }

    fn location(&self) -> DataLocation {
        DataLocation::Cpu
    }
}

/// Returns the [`ValueType`] used for [`FluidGrid3D`] state on a wire.
pub fn fluid_grid_3d_type() -> ValueType {
    ValueType::of::<FluidGrid3D>(FLUID_GRID_3D_NAME)
}

/// A pure-data source applied to a fresh 3D grid by [`Fluid3DInit`].
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum Fluid3DSource {
    /// Inject density at a cell.
    Density {
        /// Cell x.
        x: usize,
        /// Cell y.
        y: usize,
        /// Cell z.
        z: usize,
        /// Amount of density to add.
        amount: f32,
    },
    /// Inject velocity at a cell.
    Velocity {
        /// Cell x.
        x: usize,
        /// Cell y.
        y: usize,
        /// Cell z.
        z: usize,
        /// Velocity to add.
        vel: glam::Vec3,
    },
}

/// Pure in-graph **source** node producing the initial [`FluidGrid3D`].
///
/// Wire this into a [`Fluid3DStep`]'s state port with a *direct* edge (tick-0
/// seed), alongside the step's feedback self-loop.
///
/// # Ports
/// - Output `0` `"state"`: `Custom(FluidGrid3D)` — initial grid.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Fluid3DInit {
    /// Grid width.
    pub width: usize,
    /// Grid height.
    pub height: usize,
    /// Grid depth.
    pub depth: usize,
    /// Simulation configuration (diffusion, iterations, dt).
    pub config: FluidConfig,
    /// Sources applied to the fresh grid, in order.
    pub sources: Vec<Fluid3DSource>,
}

impl Fluid3DInit {
    /// Creates an init node with the given size and config and no sources.
    pub fn new(width: usize, height: usize, depth: usize, config: FluidConfig) -> Self {
        Self {
            width,
            height,
            depth,
            config,
            sources: Vec::new(),
        }
    }

    /// Adds a density source.
    pub fn with_density(mut self, x: usize, y: usize, z: usize, amount: f32) -> Self {
        self.sources
            .push(Fluid3DSource::Density { x, y, z, amount });
        self
    }

    /// Adds a velocity source.
    pub fn with_velocity(mut self, x: usize, y: usize, z: usize, vel: glam::Vec3) -> Self {
        self.sources.push(Fluid3DSource::Velocity { x, y, z, vel });
        self
    }

    /// Builds the initial [`FluidGrid3D`] from this config (pure).
    pub fn build(&self) -> FluidGrid3D {
        let mut g = FluidGrid3D::new(self.width, self.height, self.depth, self.config.clone());
        for source in &self.sources {
            match *source {
                Fluid3DSource::Density { x, y, z, amount } => g.add_density(x, y, z, amount),
                Fluid3DSource::Velocity { x, y, z, vel } => g.add_velocity(x, y, z, vel),
            }
        }
        g
    }
}

impl DynNode for Fluid3DInit {
    fn type_name(&self) -> &'static str {
        "fluid::feedback::Fluid3DInit"
    }

    fn inputs(&self) -> Vec<PortDescriptor> {
        vec![]
    }

    fn outputs(&self) -> Vec<PortDescriptor> {
        vec![PortDescriptor::new("state", fluid_grid_3d_type())]
    }

    fn execute(&self, _inputs: &[Value], _ctx: &EvalContext) -> Result<Vec<Value>, GraphError> {
        Ok(vec![Value::opaque(self.build())])
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Pure per-tick step node for the 3D fluid simulation.
///
/// State lives on the feedback edge, not in the node. `execute` clones the
/// previous [`FluidGrid3D`], advances it one step, and returns the new grid.
///
/// # Ports
/// - Input `0` `"state"`: `Custom(FluidGrid3D)` — previous-tick grid.
/// - Output `0` `"state"`: `Custom(FluidGrid3D)` — advanced grid.
#[derive(Debug, Clone, Copy, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Fluid3DStep;

impl DynNode for Fluid3DStep {
    fn type_name(&self) -> &'static str {
        "fluid::feedback::Fluid3DStep"
    }

    fn inputs(&self) -> Vec<PortDescriptor> {
        vec![PortDescriptor::new("state", fluid_grid_3d_type())]
    }

    fn outputs(&self) -> Vec<PortDescriptor> {
        vec![PortDescriptor::new("state", fluid_grid_3d_type())]
    }

    fn execute(&self, inputs: &[Value], _ctx: &EvalContext) -> Result<Vec<Value>, GraphError> {
        let prev = inputs[0].downcast_ref::<FluidGrid3D>().ok_or_else(|| {
            GraphError::ExecutionError(
                "fluid::feedback::Fluid3DStep expects a FluidGrid3D state input".to_string(),
            )
        })?;
        let mut next = prev.clone();
        next.step();
        Ok(vec![Value::opaque(next)])
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

// ===========================================================================
// SmokeGrid2D
// ===========================================================================

/// The opaque value type name for [`SmokeGrid2D`] state on a wire.
pub const SMOKE_GRID_2D_NAME: &str = "SmokeGrid2D";

impl GraphValue for SmokeGrid2D {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn type_name(&self) -> &'static str {
        SMOKE_GRID_2D_NAME
    }

    fn location(&self) -> DataLocation {
        DataLocation::Cpu
    }
}

/// Returns the [`ValueType`] used for [`SmokeGrid2D`] state on a wire.
pub fn smoke_grid_2d_type() -> ValueType {
    ValueType::of::<SmokeGrid2D>(SMOKE_GRID_2D_NAME)
}

/// A pure-data source applied to a fresh 2D smoke grid by [`Smoke2DInit`].
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum Smoke2DSource {
    /// Inject smoke (density + temperature) at a cell.
    Smoke {
        /// Cell x.
        x: usize,
        /// Cell y.
        y: usize,
        /// Density to add.
        density: f32,
        /// Temperature to add.
        temperature: f32,
    },
    /// Inject velocity at a cell.
    Velocity {
        /// Cell x.
        x: usize,
        /// Cell y.
        y: usize,
        /// X velocity.
        vx: f32,
        /// Y velocity.
        vy: f32,
    },
}

/// Pure in-graph **source** node producing the initial [`SmokeGrid2D`].
///
/// Wire this into a [`Smoke2DStep`]'s state port with a *direct* edge (tick-0
/// seed), alongside the step's feedback self-loop.
///
/// # Ports
/// - Output `0` `"state"`: `Custom(SmokeGrid2D)` — initial grid.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Smoke2DInit {
    /// Grid width.
    pub width: usize,
    /// Grid height.
    pub height: usize,
    /// Simulation configuration.
    pub config: SmokeConfig,
    /// Sources applied to the fresh grid, in order.
    pub sources: Vec<Smoke2DSource>,
}

impl Smoke2DInit {
    /// Creates an init node with the given size and config and no sources.
    pub fn new(width: usize, height: usize, config: SmokeConfig) -> Self {
        Self {
            width,
            height,
            config,
            sources: Vec::new(),
        }
    }

    /// Adds a smoke source.
    pub fn with_smoke(mut self, x: usize, y: usize, density: f32, temperature: f32) -> Self {
        self.sources.push(Smoke2DSource::Smoke {
            x,
            y,
            density,
            temperature,
        });
        self
    }

    /// Adds a velocity source.
    pub fn with_velocity(mut self, x: usize, y: usize, vx: f32, vy: f32) -> Self {
        self.sources.push(Smoke2DSource::Velocity { x, y, vx, vy });
        self
    }

    /// Builds the initial [`SmokeGrid2D`] from this config (pure).
    pub fn build(&self) -> SmokeGrid2D {
        let mut g = SmokeGrid2D::new(self.width, self.height, self.config.clone());
        for source in &self.sources {
            match *source {
                Smoke2DSource::Smoke {
                    x,
                    y,
                    density,
                    temperature,
                } => g.add_smoke(x, y, density, temperature),
                Smoke2DSource::Velocity { x, y, vx, vy } => g.add_velocity(x, y, vx, vy),
            }
        }
        g
    }
}

impl DynNode for Smoke2DInit {
    fn type_name(&self) -> &'static str {
        "fluid::feedback::Smoke2DInit"
    }

    fn inputs(&self) -> Vec<PortDescriptor> {
        vec![]
    }

    fn outputs(&self) -> Vec<PortDescriptor> {
        vec![PortDescriptor::new("state", smoke_grid_2d_type())]
    }

    fn execute(&self, _inputs: &[Value], _ctx: &EvalContext) -> Result<Vec<Value>, GraphError> {
        Ok(vec![Value::opaque(self.build())])
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Pure per-tick step node for the 2D smoke simulation.
///
/// # Ports
/// - Input `0` `"state"`: `Custom(SmokeGrid2D)` — previous-tick grid.
/// - Output `0` `"state"`: `Custom(SmokeGrid2D)` — advanced grid.
#[derive(Debug, Clone, Copy, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Smoke2DStep;

impl DynNode for Smoke2DStep {
    fn type_name(&self) -> &'static str {
        "fluid::feedback::Smoke2DStep"
    }

    fn inputs(&self) -> Vec<PortDescriptor> {
        vec![PortDescriptor::new("state", smoke_grid_2d_type())]
    }

    fn outputs(&self) -> Vec<PortDescriptor> {
        vec![PortDescriptor::new("state", smoke_grid_2d_type())]
    }

    fn execute(&self, inputs: &[Value], _ctx: &EvalContext) -> Result<Vec<Value>, GraphError> {
        let prev = inputs[0].downcast_ref::<SmokeGrid2D>().ok_or_else(|| {
            GraphError::ExecutionError(
                "fluid::feedback::Smoke2DStep expects a SmokeGrid2D state input".to_string(),
            )
        })?;
        let mut next = prev.clone();
        next.step();
        Ok(vec![Value::opaque(next)])
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

// ===========================================================================
// SmokeGrid3D
// ===========================================================================

/// The opaque value type name for [`SmokeGrid3D`] state on a wire.
pub const SMOKE_GRID_3D_NAME: &str = "SmokeGrid3D";

impl GraphValue for SmokeGrid3D {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn type_name(&self) -> &'static str {
        SMOKE_GRID_3D_NAME
    }

    fn location(&self) -> DataLocation {
        DataLocation::Cpu
    }
}

/// Returns the [`ValueType`] used for [`SmokeGrid3D`] state on a wire.
pub fn smoke_grid_3d_type() -> ValueType {
    ValueType::of::<SmokeGrid3D>(SMOKE_GRID_3D_NAME)
}

/// A pure-data source applied to a fresh 3D smoke grid by [`Smoke3DInit`].
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum Smoke3DSource {
    /// Inject smoke (density + temperature) at a cell.
    Smoke {
        /// Cell x.
        x: usize,
        /// Cell y.
        y: usize,
        /// Cell z.
        z: usize,
        /// Density to add.
        density: f32,
        /// Temperature to add.
        temperature: f32,
    },
    /// Inject velocity at a cell.
    Velocity {
        /// Cell x.
        x: usize,
        /// Cell y.
        y: usize,
        /// Cell z.
        z: usize,
        /// Velocity to add.
        vel: glam::Vec3,
    },
}

/// Pure in-graph **source** node producing the initial [`SmokeGrid3D`].
///
/// Wire this into a [`Smoke3DStep`]'s state port with a *direct* edge (tick-0
/// seed), alongside the step's feedback self-loop.
///
/// # Ports
/// - Output `0` `"state"`: `Custom(SmokeGrid3D)` — initial grid.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Smoke3DInit {
    /// Grid width.
    pub width: usize,
    /// Grid height.
    pub height: usize,
    /// Grid depth.
    pub depth: usize,
    /// Simulation configuration.
    pub config: SmokeConfig,
    /// Sources applied to the fresh grid, in order.
    pub sources: Vec<Smoke3DSource>,
}

impl Smoke3DInit {
    /// Creates an init node with the given size and config and no sources.
    pub fn new(width: usize, height: usize, depth: usize, config: SmokeConfig) -> Self {
        Self {
            width,
            height,
            depth,
            config,
            sources: Vec::new(),
        }
    }

    /// Adds a smoke source.
    pub fn with_smoke(
        mut self,
        x: usize,
        y: usize,
        z: usize,
        density: f32,
        temperature: f32,
    ) -> Self {
        self.sources.push(Smoke3DSource::Smoke {
            x,
            y,
            z,
            density,
            temperature,
        });
        self
    }

    /// Adds a velocity source.
    pub fn with_velocity(mut self, x: usize, y: usize, z: usize, vel: glam::Vec3) -> Self {
        self.sources.push(Smoke3DSource::Velocity { x, y, z, vel });
        self
    }

    /// Builds the initial [`SmokeGrid3D`] from this config (pure).
    pub fn build(&self) -> SmokeGrid3D {
        let mut g = SmokeGrid3D::new(self.width, self.height, self.depth, self.config.clone());
        for source in &self.sources {
            match *source {
                Smoke3DSource::Smoke {
                    x,
                    y,
                    z,
                    density,
                    temperature,
                } => g.add_smoke(x, y, z, density, temperature),
                Smoke3DSource::Velocity { x, y, z, vel } => g.add_velocity(x, y, z, vel),
            }
        }
        g
    }
}

impl DynNode for Smoke3DInit {
    fn type_name(&self) -> &'static str {
        "fluid::feedback::Smoke3DInit"
    }

    fn inputs(&self) -> Vec<PortDescriptor> {
        vec![]
    }

    fn outputs(&self) -> Vec<PortDescriptor> {
        vec![PortDescriptor::new("state", smoke_grid_3d_type())]
    }

    fn execute(&self, _inputs: &[Value], _ctx: &EvalContext) -> Result<Vec<Value>, GraphError> {
        Ok(vec![Value::opaque(self.build())])
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Pure per-tick step node for the 3D smoke simulation.
///
/// # Ports
/// - Input `0` `"state"`: `Custom(SmokeGrid3D)` — previous-tick grid.
/// - Output `0` `"state"`: `Custom(SmokeGrid3D)` — advanced grid.
#[derive(Debug, Clone, Copy, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Smoke3DStep;

impl DynNode for Smoke3DStep {
    fn type_name(&self) -> &'static str {
        "fluid::feedback::Smoke3DStep"
    }

    fn inputs(&self) -> Vec<PortDescriptor> {
        vec![PortDescriptor::new("state", smoke_grid_3d_type())]
    }

    fn outputs(&self) -> Vec<PortDescriptor> {
        vec![PortDescriptor::new("state", smoke_grid_3d_type())]
    }

    fn execute(&self, inputs: &[Value], _ctx: &EvalContext) -> Result<Vec<Value>, GraphError> {
        let prev = inputs[0].downcast_ref::<SmokeGrid3D>().ok_or_else(|| {
            GraphError::ExecutionError(
                "fluid::feedback::Smoke3DStep expects a SmokeGrid3D state input".to_string(),
            )
        })?;
        let mut next = prev.clone();
        next.step();
        Ok(vec![Value::opaque(next)])
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

// ===========================================================================
// Sph2D
// ===========================================================================

/// The opaque value type name for [`Sph2D`] state on a wire.
pub const SPH_2D_NAME: &str = "Sph2D";

impl GraphValue for Sph2D {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn type_name(&self) -> &'static str {
        SPH_2D_NAME
    }

    fn location(&self) -> DataLocation {
        DataLocation::Cpu
    }
}

/// Returns the [`ValueType`] used for [`Sph2D`] state on a wire.
pub fn sph_2d_type() -> ValueType {
    ValueType::of::<Sph2D>(SPH_2D_NAME)
}

/// A pure-data source seeding particles into a fresh [`Sph2D`] by [`Sph2DInit`].
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum Sph2DSource {
    /// A single particle.
    Particle {
        /// Particle position.
        position: glam::Vec2,
        /// Particle mass.
        mass: f32,
    },
    /// A grid-filled block of particles.
    Block {
        /// Block min corner.
        min: glam::Vec2,
        /// Block max corner.
        max: glam::Vec2,
        /// Spacing between particles.
        spacing: f32,
        /// Per-particle mass.
        mass: f32,
    },
}

/// Pure in-graph **source** node producing the initial [`Sph2D`].
///
/// Wire this into a [`Sph2DStep`]'s state port with a *direct* edge (tick-0
/// seed), alongside the step's feedback self-loop.
///
/// # Ports
/// - Output `0` `"state"`: `Custom(Sph2D)` — initial simulation.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Sph2DInit {
    /// Simulation parameters.
    pub config: SphConfig,
    /// Simulation bounds (min corner).
    pub bounds_min: glam::Vec2,
    /// Simulation bounds (max corner).
    pub bounds_max: glam::Vec2,
    /// Particle sources applied to the fresh simulation, in order.
    pub sources: Vec<Sph2DSource>,
}

impl Sph2DInit {
    /// Creates an init node with the given config and bounds and no particles.
    pub fn new(config: SphConfig, bounds_min: glam::Vec2, bounds_max: glam::Vec2) -> Self {
        Self {
            config,
            bounds_min,
            bounds_max,
            sources: Vec::new(),
        }
    }

    /// Adds a single particle.
    pub fn with_particle(mut self, position: glam::Vec2, mass: f32) -> Self {
        self.sources.push(Sph2DSource::Particle { position, mass });
        self
    }

    /// Adds a grid-filled block of particles.
    pub fn with_block(mut self, min: glam::Vec2, max: glam::Vec2, spacing: f32, mass: f32) -> Self {
        self.sources.push(Sph2DSource::Block {
            min,
            max,
            spacing,
            mass,
        });
        self
    }

    /// Builds the initial [`Sph2D`] from this config (pure).
    pub fn build(&self) -> Sph2D {
        let mut sim = Sph2D::new(self.config.clone(), (self.bounds_min, self.bounds_max));
        for source in &self.sources {
            match *source {
                Sph2DSource::Particle { position, mass } => sim.add_particle(position, mass),
                Sph2DSource::Block {
                    min,
                    max,
                    spacing,
                    mass,
                } => sim.add_block(min, max, spacing, mass),
            }
        }
        sim
    }
}

impl DynNode for Sph2DInit {
    fn type_name(&self) -> &'static str {
        "fluid::feedback::Sph2DInit"
    }

    fn inputs(&self) -> Vec<PortDescriptor> {
        vec![]
    }

    fn outputs(&self) -> Vec<PortDescriptor> {
        vec![PortDescriptor::new("state", sph_2d_type())]
    }

    fn execute(&self, _inputs: &[Value], _ctx: &EvalContext) -> Result<Vec<Value>, GraphError> {
        Ok(vec![Value::opaque(self.build())])
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Pure per-tick step node for the 2D SPH simulation.
///
/// # Ports
/// - Input `0` `"state"`: `Custom(Sph2D)` — previous-tick simulation.
/// - Output `0` `"state"`: `Custom(Sph2D)` — advanced simulation.
#[derive(Debug, Clone, Copy, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Sph2DStep;

impl DynNode for Sph2DStep {
    fn type_name(&self) -> &'static str {
        "fluid::feedback::Sph2DStep"
    }

    fn inputs(&self) -> Vec<PortDescriptor> {
        vec![PortDescriptor::new("state", sph_2d_type())]
    }

    fn outputs(&self) -> Vec<PortDescriptor> {
        vec![PortDescriptor::new("state", sph_2d_type())]
    }

    fn execute(&self, inputs: &[Value], _ctx: &EvalContext) -> Result<Vec<Value>, GraphError> {
        let prev = inputs[0].downcast_ref::<Sph2D>().ok_or_else(|| {
            GraphError::ExecutionError(
                "fluid::feedback::Sph2DStep expects a Sph2D state input".to_string(),
            )
        })?;
        let mut next = prev.clone();
        next.step();
        Ok(vec![Value::opaque(next)])
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

// ===========================================================================
// Sph3D
// ===========================================================================

/// The opaque value type name for [`Sph3D`] state on a wire.
pub const SPH_3D_NAME: &str = "Sph3D";

impl GraphValue for Sph3D {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn type_name(&self) -> &'static str {
        SPH_3D_NAME
    }

    fn location(&self) -> DataLocation {
        DataLocation::Cpu
    }
}

/// Returns the [`ValueType`] used for [`Sph3D`] state on a wire.
pub fn sph_3d_type() -> ValueType {
    ValueType::of::<Sph3D>(SPH_3D_NAME)
}

/// A pure-data source seeding particles into a fresh [`Sph3D`] by [`Sph3DInit`].
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum Sph3DSource {
    /// A single particle.
    Particle {
        /// Particle position.
        position: glam::Vec3,
        /// Particle mass.
        mass: f32,
    },
    /// A grid-filled block of particles.
    Block {
        /// Block min corner.
        min: glam::Vec3,
        /// Block max corner.
        max: glam::Vec3,
        /// Spacing between particles.
        spacing: f32,
        /// Per-particle mass.
        mass: f32,
    },
}

/// Pure in-graph **source** node producing the initial [`Sph3D`].
///
/// Wire this into a [`Sph3DStep`]'s state port with a *direct* edge (tick-0
/// seed), alongside the step's feedback self-loop.
///
/// # Ports
/// - Output `0` `"state"`: `Custom(Sph3D)` — initial simulation.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Sph3DInit {
    /// Simulation parameters.
    pub config: SphConfig3D,
    /// Simulation bounds (min corner).
    pub bounds_min: glam::Vec3,
    /// Simulation bounds (max corner).
    pub bounds_max: glam::Vec3,
    /// Particle sources applied to the fresh simulation, in order.
    pub sources: Vec<Sph3DSource>,
}

impl Sph3DInit {
    /// Creates an init node with the given config and bounds and no particles.
    pub fn new(config: SphConfig3D, bounds_min: glam::Vec3, bounds_max: glam::Vec3) -> Self {
        Self {
            config,
            bounds_min,
            bounds_max,
            sources: Vec::new(),
        }
    }

    /// Adds a single particle.
    pub fn with_particle(mut self, position: glam::Vec3, mass: f32) -> Self {
        self.sources.push(Sph3DSource::Particle { position, mass });
        self
    }

    /// Adds a grid-filled block of particles.
    pub fn with_block(mut self, min: glam::Vec3, max: glam::Vec3, spacing: f32, mass: f32) -> Self {
        self.sources.push(Sph3DSource::Block {
            min,
            max,
            spacing,
            mass,
        });
        self
    }

    /// Builds the initial [`Sph3D`] from this config (pure).
    pub fn build(&self) -> Sph3D {
        let mut sim = Sph3D::new(self.config.clone(), (self.bounds_min, self.bounds_max));
        for source in &self.sources {
            match *source {
                Sph3DSource::Particle { position, mass } => sim.add_particle(position, mass),
                Sph3DSource::Block {
                    min,
                    max,
                    spacing,
                    mass,
                } => sim.add_block(min, max, spacing, mass),
            }
        }
        sim
    }
}

impl DynNode for Sph3DInit {
    fn type_name(&self) -> &'static str {
        "fluid::feedback::Sph3DInit"
    }

    fn inputs(&self) -> Vec<PortDescriptor> {
        vec![]
    }

    fn outputs(&self) -> Vec<PortDescriptor> {
        vec![PortDescriptor::new("state", sph_3d_type())]
    }

    fn execute(&self, _inputs: &[Value], _ctx: &EvalContext) -> Result<Vec<Value>, GraphError> {
        Ok(vec![Value::opaque(self.build())])
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Pure per-tick step node for the 3D SPH simulation.
///
/// # Ports
/// - Input `0` `"state"`: `Custom(Sph3D)` — previous-tick simulation.
/// - Output `0` `"state"`: `Custom(Sph3D)` — advanced simulation.
#[derive(Debug, Clone, Copy, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Sph3DStep;

impl DynNode for Sph3DStep {
    fn type_name(&self) -> &'static str {
        "fluid::feedback::Sph3DStep"
    }

    fn inputs(&self) -> Vec<PortDescriptor> {
        vec![PortDescriptor::new("state", sph_3d_type())]
    }

    fn outputs(&self) -> Vec<PortDescriptor> {
        vec![PortDescriptor::new("state", sph_3d_type())]
    }

    fn execute(&self, inputs: &[Value], _ctx: &EvalContext) -> Result<Vec<Value>, GraphError> {
        let prev = inputs[0].downcast_ref::<Sph3D>().ok_or_else(|| {
            GraphError::ExecutionError(
                "fluid::feedback::Sph3DStep expects a Sph3D state input".to_string(),
            )
        })?;
        let mut next = prev.clone();
        next.step();
        Ok(vec![Value::opaque(next)])
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::FluidConfig;
    use unshape_core::{Graph, Latch, LatchSnapshot};

    fn init_node() -> FluidInit {
        FluidInit::new(32, 32, FluidConfig::default())
            .with_density(16, 16, 100.0)
            .with_velocity(16, 16, 5.0, 2.0)
    }

    fn seeded_grid() -> FluidGrid2D {
        init_node().build()
    }

    /// A built feedback graph: `Init --direct--> Step.state`, `Step --feedback--> Step.state`.
    struct Built {
        graph: Graph,
        step: u32,
    }

    fn build() -> Built {
        let mut graph = Graph::new();
        let init = graph.add_node(init_node());
        let latch = graph.add_node(Latch::new(fluid_grid_2d_type()));
        let step = graph.add_node(Step);
        graph.connect(init, 0, latch, 0).unwrap(); // Init -> latch.init (seed)
        graph.connect(latch, 0, step, 0).unwrap(); // latch.out -> step.state
        graph.connect(step, 0, latch, 1).unwrap(); // step.state -> latch.signal
        Built { graph, step }
    }

    fn density_sum(g: &FluidGrid2D) -> f64 {
        g.density_field().iter().map(|&x| x as f64).sum()
    }

    #[test]
    fn evolves_like_mut_step_loop() {
        // (a) feedback stepping N times matches the &mut step loop N times.
        let n = 15u64;

        let mut reference = seeded_grid();
        for _ in 0..n {
            reference.step();
        }

        let Built { mut graph, step } = build();
        let mut state = LatchSnapshot::new();
        let mut last = None;
        for t in 0..n {
            let r = graph
                .tick_latched(t, &mut state, &EvalContext::new())
                .unwrap();
            last = Some(
                r.get(step, 0)
                    .unwrap()
                    .downcast_ref::<FluidGrid2D>()
                    .unwrap()
                    .clone(),
            );
        }
        let evolved = last.unwrap();

        for (a, b) in evolved
            .density_field()
            .iter()
            .zip(reference.density_field())
        {
            assert_eq!(a, b);
        }
        let (evx, evy) = evolved.velocity_field();
        let (rvx, rvy) = reference.velocity_field();
        for (a, b) in evx.iter().zip(rvx) {
            assert_eq!(a, b);
        }
        for (a, b) in evy.iter().zip(rvy) {
            assert_eq!(a, b);
        }
    }

    #[test]
    fn node_is_pure_fresh_state_restarts() {
        // (b) the node holds no state: a fresh seed restarts from one step (the
        // Init source re-seeds tick 0).
        let Built { mut graph, step } = build();
        let mut state = LatchSnapshot::new();
        for t in 0..5 {
            graph
                .tick_latched(t, &mut state, &EvalContext::new())
                .unwrap();
        }

        let mut fresh = LatchSnapshot::new();
        let r = graph
            .tick_latched(0, &mut fresh, &EvalContext::new())
            .unwrap();
        let one = r
            .get(step, 0)
            .unwrap()
            .downcast_ref::<FluidGrid2D>()
            .unwrap()
            .clone();

        let mut reference = seeded_grid();
        reference.step();
        for (a, b) in one.density_field().iter().zip(reference.density_field()) {
            assert_eq!(a, b);
        }
    }

    #[test]
    fn deterministic() {
        // (c) same seed + inputs + N -> identical output.
        let run = || {
            let Built { mut graph, step } = build();
            let mut state = LatchSnapshot::new();
            let mut last = 0.0;
            for t in 0..12 {
                let r = graph
                    .tick_latched(t, &mut state, &EvalContext::new())
                    .unwrap();
                last = density_sum(
                    r.get(step, 0)
                        .unwrap()
                        .downcast_ref::<FluidGrid2D>()
                        .unwrap(),
                );
            }
            last
        };
        assert_eq!(run(), run());
    }

    #[test]
    fn run_to_tick_matches_manual_stepping() {
        // (d) run_to_tick now SUCCEEDS: the in-graph FluidInit source re-seeds
        // tick 0 after run_to_tick clears state. Resimulated == manual stepping.
        let target = 10u64;

        let manual = {
            let Built { mut graph, step } = build();
            let mut state = LatchSnapshot::new();
            let mut last = None;
            for t in 0..=target {
                let r = graph
                    .tick_latched(t, &mut state, &EvalContext::new())
                    .unwrap();
                last = Some(
                    r.get(step, 0)
                        .unwrap()
                        .downcast_ref::<FluidGrid2D>()
                        .unwrap()
                        .clone(),
                );
            }
            last.unwrap()
        };

        let Built { mut graph, step } = build();
        let mut state = LatchSnapshot::new();
        let r = graph
            .run_to_tick_latched(target, &mut state, |_t| EvalContext::new())
            .unwrap();
        let resimulated = r
            .get(step, 0)
            .unwrap()
            .downcast_ref::<FluidGrid2D>()
            .unwrap();

        for (a, b) in resimulated
            .density_field()
            .iter()
            .zip(manual.density_field())
        {
            assert_eq!(a, b);
        }
        let (rvx, rvy) = resimulated.velocity_field();
        let (mvx, mvy) = manual.velocity_field();
        for (a, b) in rvx.iter().zip(mvx) {
            assert_eq!(a, b);
        }
        for (a, b) in rvy.iter().zip(mvy) {
            assert_eq!(a, b);
        }
    }

    #[test]
    fn seek_resimulate_is_deterministic() {
        // (e) seek(Resimulate) works and reproduces.
        let seek_to = |target: u64| {
            let Built { mut graph, step } = build();
            let mut state = LatchSnapshot::new();
            let r = graph
                .seek_latched(
                    target,
                    0,
                    unshape_core::SeekBehavior::Resimulate,
                    &mut state,
                    |_t| EvalContext::new(),
                )
                .unwrap();
            density_sum(
                r.get(step, 0)
                    .unwrap()
                    .downcast_ref::<FluidGrid2D>()
                    .unwrap(),
            )
        };
        assert_eq!(seek_to(7), seek_to(7));
    }

    // -----------------------------------------------------------------------
    // FluidGrid3D
    // -----------------------------------------------------------------------

    #[test]
    fn fluid_3d_run_to_tick_matches_manual_stepping() {
        let init = Fluid3DInit::new(12, 12, 12, FluidConfig::default())
            .with_density(6, 6, 6, 100.0)
            .with_velocity(6, 6, 6, glam::Vec3::new(1.0, 0.5, 0.0));
        let target = 8u64;

        let mut reference = init.build();
        for _ in 0..=target {
            reference.step();
        }

        let mut graph = Graph::new();
        let i = graph.add_node(init);
        let latch = graph.add_node(Latch::new(fluid_grid_3d_type()));
        let s = graph.add_node(Fluid3DStep);
        graph.connect(i, 0, latch, 0).unwrap(); // Init -> latch.init
        graph.connect(latch, 0, s, 0).unwrap(); // latch.out -> step.state
        graph.connect(s, 0, latch, 1).unwrap(); // step.state -> latch.signal

        let mut state = LatchSnapshot::new();
        let r = graph
            .run_to_tick_latched(target, &mut state, |_t| EvalContext::new())
            .unwrap();
        let resim = r.get(s, 0).unwrap().downcast_ref::<FluidGrid3D>().unwrap();
        assert_eq!(resim.density_field(), reference.density_field());

        // determinism: a second resimulation matches.
        let mut state2 = LatchSnapshot::new();
        let r2 = graph
            .run_to_tick_latched(target, &mut state2, |_t| EvalContext::new())
            .unwrap();
        let resim2 = r2.get(s, 0).unwrap().downcast_ref::<FluidGrid3D>().unwrap();
        assert_eq!(resim.density_field(), resim2.density_field());
    }

    // -----------------------------------------------------------------------
    // SmokeGrid2D
    // -----------------------------------------------------------------------

    #[test]
    fn smoke_2d_run_to_tick_matches_manual_stepping() {
        let init = Smoke2DInit::new(32, 32, SmokeConfig::default())
            .with_smoke(16, 5, 100.0, 100.0)
            .with_velocity(16, 5, 0.0, 1.0);
        let target = 10u64;

        let mut reference = init.build();
        for _ in 0..=target {
            reference.step();
        }

        let mut graph = Graph::new();
        let i = graph.add_node(init);
        let latch = graph.add_node(Latch::new(smoke_grid_2d_type()));
        let s = graph.add_node(Smoke2DStep);
        graph.connect(i, 0, latch, 0).unwrap(); // Init -> latch.init
        graph.connect(latch, 0, s, 0).unwrap(); // latch.out -> step.state
        graph.connect(s, 0, latch, 1).unwrap(); // step.state -> latch.signal

        let mut state = LatchSnapshot::new();
        let r = graph
            .run_to_tick_latched(target, &mut state, |_t| EvalContext::new())
            .unwrap();
        let resim = r.get(s, 0).unwrap().downcast_ref::<SmokeGrid2D>().unwrap();
        assert_eq!(resim.density_field(), reference.density_field());
        assert_eq!(resim.temperature_field(), reference.temperature_field());

        let mut state2 = LatchSnapshot::new();
        let r2 = graph
            .run_to_tick_latched(target, &mut state2, |_t| EvalContext::new())
            .unwrap();
        let resim2 = r2.get(s, 0).unwrap().downcast_ref::<SmokeGrid2D>().unwrap();
        assert_eq!(resim.density_field(), resim2.density_field());
    }

    // -----------------------------------------------------------------------
    // SmokeGrid3D
    // -----------------------------------------------------------------------

    #[test]
    fn smoke_3d_run_to_tick_matches_manual_stepping() {
        let init = Smoke3DInit::new(12, 12, 12, SmokeConfig::default())
            .with_smoke(6, 3, 6, 100.0, 100.0)
            .with_velocity(6, 3, 6, glam::Vec3::new(0.0, 1.0, 0.0));
        let target = 6u64;

        let mut reference = init.build();
        for _ in 0..=target {
            reference.step();
        }

        let mut graph = Graph::new();
        let i = graph.add_node(init);
        let latch = graph.add_node(Latch::new(smoke_grid_3d_type()));
        let s = graph.add_node(Smoke3DStep);
        graph.connect(i, 0, latch, 0).unwrap(); // Init -> latch.init
        graph.connect(latch, 0, s, 0).unwrap(); // latch.out -> step.state
        graph.connect(s, 0, latch, 1).unwrap(); // step.state -> latch.signal

        let mut state = LatchSnapshot::new();
        let r = graph
            .run_to_tick_latched(target, &mut state, |_t| EvalContext::new())
            .unwrap();
        let resim = r.get(s, 0).unwrap().downcast_ref::<SmokeGrid3D>().unwrap();
        assert_eq!(resim.density_field(), reference.density_field());
        assert_eq!(resim.temperature_field(), reference.temperature_field());

        let mut state2 = LatchSnapshot::new();
        let r2 = graph
            .run_to_tick_latched(target, &mut state2, |_t| EvalContext::new())
            .unwrap();
        let resim2 = r2.get(s, 0).unwrap().downcast_ref::<SmokeGrid3D>().unwrap();
        assert_eq!(resim.density_field(), resim2.density_field());
    }

    // -----------------------------------------------------------------------
    // Sph2D
    // -----------------------------------------------------------------------

    #[test]
    fn sph_2d_run_to_tick_matches_manual_stepping() {
        let init = Sph2DInit::new(
            SphConfig::default(),
            glam::Vec2::ZERO,
            glam::Vec2::new(100.0, 100.0),
        )
        .with_block(
            glam::Vec2::new(20.0, 50.0),
            glam::Vec2::new(40.0, 70.0),
            8.0,
            1.0,
        );
        let target = 12u64;

        let mut reference = init.build();
        for _ in 0..=target {
            reference.step();
        }
        let reference_pos = reference.positions();

        let mut graph = Graph::new();
        let i = graph.add_node(init);
        let latch = graph.add_node(Latch::new(sph_2d_type()));
        let s = graph.add_node(Sph2DStep);
        graph.connect(i, 0, latch, 0).unwrap(); // Init -> latch.init
        graph.connect(latch, 0, s, 0).unwrap(); // latch.out -> step.state
        graph.connect(s, 0, latch, 1).unwrap(); // step.state -> latch.signal

        let mut state = LatchSnapshot::new();
        let r = graph
            .run_to_tick_latched(target, &mut state, |_t| EvalContext::new())
            .unwrap();
        let resim = r.get(s, 0).unwrap().downcast_ref::<Sph2D>().unwrap();
        assert_eq!(resim.positions(), reference_pos);

        let mut state2 = LatchSnapshot::new();
        let r2 = graph
            .run_to_tick_latched(target, &mut state2, |_t| EvalContext::new())
            .unwrap();
        let resim2 = r2.get(s, 0).unwrap().downcast_ref::<Sph2D>().unwrap();
        assert_eq!(resim.positions(), resim2.positions());
    }

    // -----------------------------------------------------------------------
    // Sph3D
    // -----------------------------------------------------------------------

    #[test]
    fn sph_3d_run_to_tick_matches_manual_stepping() {
        let init = Sph3DInit::new(
            SphConfig3D::default(),
            glam::Vec3::ZERO,
            glam::Vec3::new(1.0, 1.0, 1.0),
        )
        .with_block(
            glam::Vec3::new(0.2, 0.5, 0.2),
            glam::Vec3::new(0.4, 0.7, 0.4),
            0.05,
            0.001,
        );
        let target = 10u64;

        let mut reference = init.build();
        for _ in 0..=target {
            reference.step();
        }
        let reference_pos = reference.positions();

        let mut graph = Graph::new();
        let i = graph.add_node(init);
        let latch = graph.add_node(Latch::new(sph_3d_type()));
        let s = graph.add_node(Sph3DStep);
        graph.connect(i, 0, latch, 0).unwrap(); // Init -> latch.init
        graph.connect(latch, 0, s, 0).unwrap(); // latch.out -> step.state
        graph.connect(s, 0, latch, 1).unwrap(); // step.state -> latch.signal

        let mut state = LatchSnapshot::new();
        let r = graph
            .run_to_tick_latched(target, &mut state, |_t| EvalContext::new())
            .unwrap();
        let resim = r.get(s, 0).unwrap().downcast_ref::<Sph3D>().unwrap();
        assert_eq!(resim.positions(), reference_pos);

        let mut state2 = LatchSnapshot::new();
        let r2 = graph
            .run_to_tick_latched(target, &mut state2, |_t| EvalContext::new())
            .unwrap();
        let resim2 = r2.get(s, 0).unwrap().downcast_ref::<Sph3D>().unwrap();
        assert_eq!(resim.positions(), resim2.positions());
    }
}
