//! [`Sort`]: reorder rows by one or more columns.

use crate::Table;
use crate::error::TableError;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// A single sort key: a column name and direction.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct SortKey {
    /// Column to sort by.
    pub column: String,
    /// Sort in descending order (default is ascending).
    pub descending: bool,
}

impl SortKey {
    /// Sort ascending by `column`.
    pub fn ascending(column: impl Into<String>) -> Self {
        Self {
            column: column.into(),
            descending: false,
        }
    }

    /// Sort descending by `column`.
    pub fn descending(column: impl Into<String>) -> Self {
        Self {
            column: column.into(),
            descending: true,
        }
    }
}

/// Reorder rows by one or more columns, in priority order (first key is primary).
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Sort {
    /// Sort keys, evaluated in order (ties on the first key are broken by the second, etc).
    pub keys: Vec<SortKey>,
}

impl Sort {
    /// Sort ascending by a single column.
    pub fn by(column: impl Into<String>) -> Self {
        Self {
            keys: vec![SortKey::ascending(column)],
        }
    }

    /// Sort descending by a single column.
    pub fn by_descending(column: impl Into<String>) -> Self {
        Self {
            keys: vec![SortKey::descending(column)],
        }
    }

    /// Apply this sort, returning a new table with rows reordered. The sort is stable: rows
    /// that compare equal on every key keep their relative input order.
    pub fn apply(&self, table: &Table) -> Result<Table, TableError> {
        let mut columns = Vec::with_capacity(self.keys.len());
        for key in &self.keys {
            columns.push((table.require_column(&key.column)?, key.descending));
        }
        let mut indices: Vec<usize> = (0..table.len()).collect();
        let mut error = None;
        indices.sort_by(|&a, &b| {
            for (column, descending) in &columns {
                let Some(ordering) = column.data.compare_rows(a, b) else {
                    error.get_or_insert_with(|| TableError::NotOrderable(column.name.clone()));
                    return std::cmp::Ordering::Equal;
                };
                let ordering = if *descending {
                    ordering.reverse()
                } else {
                    ordering
                };
                if ordering != std::cmp::Ordering::Equal {
                    return ordering;
                }
            }
            std::cmp::Ordering::Equal
        });
        if let Some(error) = error {
            return Err(error);
        }
        Ok(table.take_rows(&indices))
    }
}

impl Table {
    /// Sort ascending by a single column. Sugar for [`Sort::apply`].
    pub fn sort_by(&self, column: impl Into<String>) -> Result<Table, TableError> {
        Sort::by(column).apply(self)
    }
}
