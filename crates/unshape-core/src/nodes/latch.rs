//! [`Latch`] — a seeded unit-delay (1-tick memory) node.
//!
//! A latch is the sole recurrence primitive: the visible delay element on a
//! cycle (the same primitive as Lustre `fby`, Pd `[z~]`, Faust `~`, a hardware
//! register). It makes the back-edge of a recurrent graph a *node* rather than
//! an invisible wire flag, so cycle validation becomes structural ("every cycle
//! must contain a latch") and no per-wire feedback machinery is needed.
//!
//! See `docs/design/recurrent-graphs.md`.
//!
//! # Ports (built from `ty`)
//!
//! - input `0` `"init"` — **required wired** seed; used only when no stored
//!   value exists yet (cold / tick 0). There is **no** zero-default fallback:
//!   opaque sim state has no zero value, so the seed must be a wired source.
//! - input `1` `"signal"` — the value captured for the *next* tick.
//! - output `0` `"out"` — the stored value: `init` at tick 0, the
//!   previously-captured `signal` thereafter.
//!
//! # Scheduling
//!
//! The latch is a **driver-recognized** node: [`Graph::tick`](crate::Graph::tick)
//! downcasts to it (like [`GraphInput`](crate::GraphInput)) and special-cases it.
//! Within a tick the latch `out` port is a pure *source* (emits the stored value)
//! and the `signal` input is a *sink* (captured at tick end), so edges into a
//! latch's `signal` input are excluded from the within-tick topological order —
//! that is what lets a recurrent graph remain acyclic per tick.
//!
//! The plain [`DynNode::execute`] path is a degenerate identity (`out = init`)
//! for the non-recurrent case; the real unit-delay behaviour lives in the tick
//! driver reading/writing the latch snapshot.

use std::any::Any;

use crate::error::GraphError;
use crate::eval::EvalContext;
use crate::node::{DynNode, PortDescriptor};
use crate::value::{Value, ValueType};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Input port index of a [`Latch`]'s `init` (seed) port.
pub const LATCH_INIT_PORT: usize = 0;
/// Input port index of a [`Latch`]'s `signal` (capture) port.
pub const LATCH_SIGNAL_PORT: usize = 1;
/// Output port index of a [`Latch`]'s `out` (stored value) port.
pub const LATCH_OUT_PORT: usize = 0;

/// How often a [`Latch`] advances (captures its `signal` and updates `out`).
///
/// `Tick` (the default) advances every base tick. `Every(n)` advances once
/// every `n` base ticks and holds its output between advances (zero-order
/// hold) — the integer sub-rate case (`Every(1)` ≡ `Tick`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum Rate {
    /// Advance every base tick (≡ `Every(1)`).
    #[default]
    Tick,
    /// Advance once every `n` base ticks, holding between advances.
    Every(u32),
}

impl Rate {
    /// The integer divisor: `Tick` ⇒ 1, `Every(n)` ⇒ `n`.
    pub fn divisor(self) -> u32 {
        match self {
            Rate::Tick => 1,
            Rate::Every(n) => n.max(1),
        }
    }

    /// Whether this latch advances (captures) on the given base `tick`.
    pub fn advances_on(self, tick: u64) -> bool {
        tick.is_multiple_of(self.divisor() as u64)
    }
}

/// A seeded unit-delay (1-tick memory) node. See the [module docs](self).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Latch {
    /// The value type flowing through the latch; determines all three port
    /// types (`init`, `signal`, `out`).
    pub ty: ValueType,
    /// How often the latch advances. Defaults to [`Rate::Tick`].
    #[cfg_attr(feature = "serde", serde(default))]
    pub rate: Rate,
}

impl Latch {
    /// Creates a per-tick latch carrying values of type `ty`.
    pub fn new(ty: ValueType) -> Self {
        Self {
            ty,
            rate: Rate::Tick,
        }
    }

    /// Sets the latch's advance rate.
    pub fn with_rate(mut self, rate: Rate) -> Self {
        self.rate = rate;
        self
    }
}

impl DynNode for Latch {
    fn type_name(&self) -> &'static str {
        "core::Latch"
    }

    fn inputs(&self) -> Vec<PortDescriptor> {
        vec![
            PortDescriptor::new("init", self.ty),
            PortDescriptor::new("signal", self.ty),
        ]
    }

    fn outputs(&self) -> Vec<PortDescriptor> {
        vec![PortDescriptor::new("out", self.ty)]
    }

    fn execute(&self, inputs: &[Value], _ctx: &EvalContext) -> Result<Vec<Value>, GraphError> {
        // Degenerate identity for the non-recurrent (plain DAG) execute path:
        // `out = init`. The real unit-delay behaviour is in `Graph::tick`, which
        // recognizes the latch and resolves `out` from its snapshot.
        let init = inputs
            .first()
            .cloned()
            .ok_or(GraphError::UnconnectedInput {
                node: 0,
                port: LATCH_INIT_PORT,
            })?;
        Ok(vec![init])
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn latch_ports_built_from_ty() {
        let latch = Latch::new(ValueType::F32);
        let inputs = latch.inputs();
        assert_eq!(inputs.len(), 2);
        assert_eq!(inputs[0].name, "init");
        assert_eq!(inputs[0].value_type, ValueType::F32);
        assert_eq!(inputs[1].name, "signal");
        assert_eq!(inputs[1].value_type, ValueType::F32);
        let outputs = latch.outputs();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].name, "out");
        assert_eq!(outputs[0].value_type, ValueType::F32);
    }

    #[test]
    fn rate_default_is_tick() {
        assert_eq!(Rate::default(), Rate::Tick);
        assert_eq!(Latch::new(ValueType::F32).rate, Rate::Tick);
    }

    #[test]
    fn rate_advances_on() {
        assert!(Rate::Tick.advances_on(0));
        assert!(Rate::Tick.advances_on(7));
        assert!(Rate::Every(3).advances_on(0));
        assert!(!Rate::Every(3).advances_on(1));
        assert!(!Rate::Every(3).advances_on(2));
        assert!(Rate::Every(3).advances_on(3));
    }

    #[test]
    fn execute_is_identity_on_init() {
        let latch = Latch::new(ValueType::F32);
        let out = latch
            .execute(&[Value::F32(5.0), Value::F32(99.0)], &EvalContext::new())
            .unwrap();
        assert_eq!(out, vec![Value::F32(5.0)]);
    }
}
