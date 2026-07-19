//! [`unshape_op::DynOp`] implementations for the single-input, single-output table ops.
//!
//! `Join` takes two tables and is a "Construction" shape (see `docs/design/op-shapes.md`),
//! not the `Transform` shape `DynOp::apply_dyn` models (single `OpValue` in, single
//! `OpValue` out) — it is intentionally not wired into this module. It can still be called
//! directly via [`crate::Join::apply`], or wired into a graph via a node type that accepts
//! two named inputs, same as other multi-input ops in the workspace.

use crate::{AddColumn, Filter, GroupBy, Pivot, Select, Sort, Table};
use unshape_op::{DynOp, OpError, OpType, OpValue};

fn table_type() -> OpType {
    OpType::of::<Table>("Table")
}

macro_rules! impl_table_dyn_op {
    ($ty:ty, $name:literal) => {
        impl DynOp for $ty {
            fn type_name(&self) -> &'static str {
                $name
            }

            fn input_type(&self) -> OpType {
                table_type()
            }

            fn output_type(&self) -> OpType {
                table_type()
            }

            fn apply_dyn(&self, input: OpValue) -> Result<OpValue, OpError> {
                let table: Table = input.downcast()?;
                let result = self
                    .apply(&table)
                    .map_err(|e| OpError::ExecutionError(e.to_string()))?;
                Ok(OpValue::new(table_type(), result))
            }

            fn params(&self) -> serde_json::Value {
                serde_json::to_value(self).unwrap_or(serde_json::Value::Null)
            }
        }
    };
}

impl_table_dyn_op!(Filter, "unshape::table::Filter");
impl_table_dyn_op!(Sort, "unshape::table::Sort");
impl_table_dyn_op!(Select, "unshape::table::Select");
impl_table_dyn_op!(GroupBy, "unshape::table::GroupBy");
impl_table_dyn_op!(AddColumn, "unshape::table::AddColumn");
impl_table_dyn_op!(Pivot, "unshape::table::Pivot");
