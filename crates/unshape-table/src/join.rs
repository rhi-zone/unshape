//! [`Join`]: combine two tables by matching a key column.

use crate::Table;
use crate::column::{Column, ColumnData};
use crate::error::TableError;
use crate::util::{scalar_key, zero_value};
use std::collections::HashMap;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Which rows to keep when a key does not match on one side.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum JoinKind {
    /// Keep only rows whose key matches on both sides.
    Inner,
    /// Keep all left rows; unmatched right-side columns are filled with each column type's
    /// zero value (`0`, `false`, `""`, or a zero vector). There is no nullable column type
    /// yet, so unmatched cells are not distinguishable from a genuine zero — see
    /// `docs/design/domain-subsumption.md` for the `Table` design and revisit this once
    /// nullability lands.
    Left,
    /// Keep all right rows; unmatched left-side columns are filled the same way as [`Left`](JoinKind::Left).
    Right,
    /// Keep all rows from both sides; unmatched columns on either side are filled the same
    /// way as [`Left`](JoinKind::Left).
    Full,
}

/// Combine two tables by matching `left_key` in the left table against `right_key` in the
/// right table. Output columns are the left table's columns followed by the right table's
/// columns, excluding `right_key` (its values are identical to `left_key` for matched rows).
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Join {
    /// Key column in the left table.
    pub left_key: String,
    /// Key column in the right table.
    pub right_key: String,
    /// Which unmatched rows to keep.
    pub kind: JoinKind,
}

/// Build an output column by gathering values from `source` at `indices`, where `None`
/// means "no match" (filled with the column's zero value).
fn gather_or_zero(source: &Column, indices: &[Option<usize>]) -> ColumnData {
    let mut data = source.data.empty_like();
    for &index in indices {
        let value = match index {
            Some(i) => source.data.get(i).expect("row index in bounds"),
            None => zero_value(source.column_type()),
        };
        data.push(value).expect("value type matches source column");
    }
    data
}

impl Join {
    /// Apply this join, returning the combined table.
    pub fn apply(&self, left: &Table, right: &Table) -> Result<Table, TableError> {
        let left_key_col = left.require_column(&self.left_key)?;
        let right_key_col = right.require_column(&self.right_key)?;

        let mut right_by_key: HashMap<String, Vec<usize>> = HashMap::new();
        for i in 0..right.len() {
            let value = right_key_col.data.get(i).expect("row index in bounds");
            right_by_key.entry(scalar_key(&value)).or_default().push(i);
        }

        // (left_row, right_row) pairs; either side is `None` for an unmatched row.
        let mut left_rows: Vec<Option<usize>> = Vec::new();
        let mut right_rows: Vec<Option<usize>> = Vec::new();
        let mut matched_right: Vec<bool> = vec![false; right.len()];

        for i in 0..left.len() {
            let value = left_key_col.data.get(i).expect("row index in bounds");
            let key = scalar_key(&value);
            match right_by_key.get(&key) {
                Some(matches) => {
                    for &j in matches {
                        left_rows.push(Some(i));
                        right_rows.push(Some(j));
                        matched_right[j] = true;
                    }
                }
                None => {
                    if matches!(self.kind, JoinKind::Left | JoinKind::Full) {
                        left_rows.push(Some(i));
                        right_rows.push(None);
                    }
                }
            }
        }

        if matches!(self.kind, JoinKind::Right | JoinKind::Full) {
            for (j, matched) in matched_right.iter().enumerate() {
                if !matched {
                    left_rows.push(None);
                    right_rows.push(Some(j));
                }
            }
        }

        let left_indices: Vec<usize> = left_rows.iter().map(|r| r.unwrap_or(0)).collect();
        let mut columns = Vec::with_capacity(left.num_columns() + right.num_columns());
        for column in left.columns() {
            let data = if left_rows.iter().all(|r| r.is_some()) {
                column.data.take_rows(&left_indices)
            } else {
                gather_or_zero(column, &left_rows)
            };
            columns.push(Column::new(column.name.clone(), data));
        }
        for column in right.columns() {
            if column.name == self.right_key {
                continue;
            }
            let right_indices: Vec<usize> = right_rows.iter().map(|r| r.unwrap_or(0)).collect();
            let data = if right_rows.iter().all(|r| r.is_some()) {
                column.data.take_rows(&right_indices)
            } else {
                gather_or_zero(column, &right_rows)
            };
            let name = if left.column(&column.name).is_some() {
                format!("{}_right", column.name)
            } else {
                column.name.clone()
            };
            columns.push(Column::new(name, data));
        }
        Table::new(columns)
    }
}

impl Table {
    /// Inner-join this table (left) with `right`, matching `left_key` against `right_key`.
    /// Sugar for [`Join::apply`].
    pub fn join(
        &self,
        right: &Table,
        left_key: impl Into<String>,
        right_key: impl Into<String>,
    ) -> Result<Table, TableError> {
        Join {
            left_key: left_key.into(),
            right_key: right_key.into(),
            kind: JoinKind::Inner,
        }
        .apply(self, right)
    }
}
