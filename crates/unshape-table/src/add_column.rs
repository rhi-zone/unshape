//! [`AddColumn`]: compute a new column from existing ones.

use crate::Table;
use crate::column::{Column, ColumnData};
use crate::error::TableError;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// A numeric expression tree over existing columns and constants. Evaluated row-by-row in
/// `f64`, widening `F32`/`I32`/`I64` inputs. This is a typed Rust value, not a query string —
/// see the "no DSLs" constraint in `CLAUDE.md`.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum ColumnExpr {
    /// Read a numeric column by name.
    Column(String),
    /// A constant value.
    Const(f64),
    /// `a + b`.
    Add(Box<ColumnExpr>, Box<ColumnExpr>),
    /// `a - b`.
    Sub(Box<ColumnExpr>, Box<ColumnExpr>),
    /// `a * b`.
    Mul(Box<ColumnExpr>, Box<ColumnExpr>),
    /// `a / b`.
    Div(Box<ColumnExpr>, Box<ColumnExpr>),
    /// `-a`.
    Neg(Box<ColumnExpr>),
    /// `|a|`.
    Abs(Box<ColumnExpr>),
    /// `min(a, b)`.
    Min(Box<ColumnExpr>, Box<ColumnExpr>),
    /// `max(a, b)`.
    Max(Box<ColumnExpr>, Box<ColumnExpr>),
}

impl ColumnExpr {
    /// Reference a numeric column by name.
    pub fn column(name: impl Into<String>) -> Self {
        ColumnExpr::Column(name.into())
    }

    /// A constant value.
    pub fn constant(value: f64) -> Self {
        ColumnExpr::Const(value)
    }

    fn eval(&self, table: &Table, row: usize, cache: &mut ExprCache) -> Result<f64, TableError> {
        match self {
            ColumnExpr::Column(name) => {
                let values = cache.numeric_column(table, name)?;
                Ok(values[row])
            }
            ColumnExpr::Const(value) => Ok(*value),
            ColumnExpr::Add(a, b) => Ok(a.eval(table, row, cache)? + b.eval(table, row, cache)?),
            ColumnExpr::Sub(a, b) => Ok(a.eval(table, row, cache)? - b.eval(table, row, cache)?),
            ColumnExpr::Mul(a, b) => Ok(a.eval(table, row, cache)? * b.eval(table, row, cache)?),
            ColumnExpr::Div(a, b) => Ok(a.eval(table, row, cache)? / b.eval(table, row, cache)?),
            ColumnExpr::Neg(a) => Ok(-a.eval(table, row, cache)?),
            ColumnExpr::Abs(a) => Ok(a.eval(table, row, cache)?.abs()),
            ColumnExpr::Min(a, b) => Ok(a.eval(table, row, cache)?.min(b.eval(table, row, cache)?)),
            ColumnExpr::Max(a, b) => Ok(a.eval(table, row, cache)?.max(b.eval(table, row, cache)?)),
        }
    }
}

/// Caches widened `f64` values for each referenced column, so a multi-node expression
/// doesn't re-widen the same column once per row.
#[derive(Default)]
struct ExprCache {
    columns: std::collections::HashMap<String, Vec<f64>>,
}

impl ExprCache {
    fn numeric_column(&mut self, table: &Table, name: &str) -> Result<&Vec<f64>, TableError> {
        if !self.columns.contains_key(name) {
            let column = table.require_column(name)?;
            let values = column
                .data
                .as_f64_slice()
                .ok_or_else(|| TableError::NotNumeric(name.to_string()))?;
            self.columns.insert(name.to_string(), values);
        }
        Ok(&self.columns[name])
    }
}

/// Compute a new column from an expression over existing columns, and append it to the
/// table. The result is always stored as `f64`.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct AddColumn {
    /// Name of the new column.
    pub name: String,
    /// Expression computing the new column's values.
    pub expr: ColumnExpr,
}

impl AddColumn {
    /// Apply this op, returning a new table with the computed column appended.
    pub fn apply(&self, table: &Table) -> Result<Table, TableError> {
        let mut cache = ExprCache::default();
        let mut values = Vec::with_capacity(table.len());
        for row in 0..table.len() {
            values.push(self.expr.eval(table, row, &mut cache)?);
        }
        table.with_column(Column::new(self.name.clone(), ColumnData::F64(values)))
    }
}

impl Table {
    /// Compute a new column from an expression over existing columns. Sugar for
    /// [`AddColumn::apply`].
    pub fn add_column(
        &self,
        name: impl Into<String>,
        expr: ColumnExpr,
    ) -> Result<Table, TableError> {
        AddColumn {
            name: name.into(),
            expr,
        }
        .apply(self)
    }
}
