//! [`Pivot`]: reshape a "long" table into a "wide" one.

use crate::Table;
use crate::column::Column;
use crate::error::TableError;
use crate::groupby::{AggregateFn, Aggregation, GroupBy};
use crate::util::{scalar_display, scalar_key, zero_value};
use std::cmp::Ordering;
use std::collections::HashMap;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

const PIVOT_VALUE_COLUMN: &str = "__pivot_value";

/// Reshape rows into columns: one output row per distinct `index` value, one output column
/// per distinct `columns` value, cells filled by aggregating `values` with `agg`.
///
/// Equivalent to [`GroupBy`] with `group_by: [index, columns]` followed by a wide reshape.
/// `index` and `columns` must be orderable column types (not `Vec2`/`Vec3`/`Vec4`). A cell
/// with no matching input rows is filled with the value column's zero value — see the same
/// caveat on [`crate::join::JoinKind::Left`] about the lack of a nullable column type yet.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Pivot {
    /// Column whose distinct values become output rows.
    pub index: String,
    /// Column whose distinct values become output columns.
    pub columns: String,
    /// Column whose values are aggregated into each cell.
    pub values: String,
    /// Aggregation function applied to `values` within each `(index, columns)` cell.
    pub agg: AggregateFn,
}

impl Pivot {
    /// Apply this pivot, returning the reshaped table.
    pub fn apply(&self, table: &Table) -> Result<Table, TableError> {
        let grouped = GroupBy {
            group_by: vec![self.index.clone(), self.columns.clone()],
            aggregations: vec![Aggregation {
                column: self.values.clone(),
                function: self.agg,
                output_name: PIVOT_VALUE_COLUMN.to_string(),
            }],
        }
        .apply(table)?;

        let index_col = grouped.require_column(&self.index)?;
        let columns_col = grouped.require_column(&self.columns)?;
        let value_col = grouped.require_column(PIVOT_VALUE_COLUMN)?;

        // Distinct index values, sorted (grouped's output is already sorted by index
        // primarily, so this is a plain dedup pass).
        let mut index_reps: Vec<usize> = Vec::new();
        for i in 0..grouped.len() {
            let is_new = match index_reps.last() {
                Some(&r) => index_col.data.compare_rows(r, i) != Some(Ordering::Equal),
                None => true,
            };
            if is_new {
                index_reps.push(i);
            }
        }

        // Distinct pivot-column values, sorted independently (they are not globally sorted
        // within `grouped`, since it is sorted primarily by index).
        let mut pivot_order: Vec<usize> = (0..grouped.len()).collect();
        pivot_order.sort_by(|&a, &b| {
            columns_col
                .data
                .compare_rows(a, b)
                .expect("columns column orderability already enforced by GroupBy")
        });
        let mut pivot_reps: Vec<usize> = Vec::new();
        for &i in &pivot_order {
            let is_new = match pivot_reps.last() {
                Some(&r) => columns_col.data.compare_rows(r, i) != Some(Ordering::Equal),
                None => true,
            };
            if is_new {
                pivot_reps.push(i);
            }
        }

        let mut cells: HashMap<(String, String), usize> = HashMap::new();
        for i in 0..grouped.len() {
            let index_value = index_col.data.get(i).expect("row index in bounds");
            let pivot_value = columns_col.data.get(i).expect("row index in bounds");
            cells.insert((scalar_key(&index_value), scalar_key(&pivot_value)), i);
        }

        let index_indices: Vec<usize> = index_reps.clone();
        let mut columns = vec![Column::new(
            self.index.clone(),
            index_col.data.take_rows(&index_indices),
        )];
        for &prep in &pivot_reps {
            let pivot_value = columns_col.data.get(prep).expect("row index in bounds");
            let column_name = scalar_display(&pivot_value);
            let mut data = value_col.data.empty_like();
            for &irep in &index_reps {
                let index_value = index_col.data.get(irep).expect("row index in bounds");
                let key = (scalar_key(&index_value), scalar_key(&pivot_value));
                let cell_value = match cells.get(&key) {
                    Some(&row) => value_col.data.get(row).expect("row index in bounds"),
                    None => zero_value(value_col.column_type()),
                };
                data.push(cell_value)
                    .expect("value type matches value column");
            }
            columns.push(Column::new(column_name, data));
        }
        Table::new(columns)
    }
}

impl Table {
    /// Reshape rows into columns. Sugar for [`Pivot::apply`].
    pub fn pivot(
        &self,
        index: impl Into<String>,
        columns: impl Into<String>,
        values: impl Into<String>,
        agg: AggregateFn,
    ) -> Result<Table, TableError> {
        Pivot {
            index: index.into(),
            columns: columns.into(),
            values: values.into(),
            agg,
        }
        .apply(self)
    }
}
