//! [`Select`]: project a table down to a subset of columns.

use crate::Table;
use crate::column::Column;
use crate::error::TableError;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Keep only the named columns, in the given order.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Select {
    /// Names of the columns to keep, in output order.
    pub columns: Vec<String>,
}

impl Select {
    /// Select these columns, in order.
    pub fn columns(columns: impl IntoIterator<Item = impl Into<String>>) -> Self {
        Self {
            columns: columns.into_iter().map(Into::into).collect(),
        }
    }

    /// Apply this selection, returning a new table with only the requested columns.
    pub fn apply(&self, table: &Table) -> Result<Table, TableError> {
        let columns: Result<Vec<Column>, TableError> = self
            .columns
            .iter()
            .map(|name| table.require_column(name).cloned())
            .collect();
        Table::new(columns?)
    }
}

impl Table {
    /// Keep only the named columns, in the given order. Sugar for [`Select::apply`].
    pub fn select(
        &self,
        columns: impl IntoIterator<Item = impl Into<String>>,
    ) -> Result<Table, TableError> {
        Select::columns(columns).apply(self)
    }
}
