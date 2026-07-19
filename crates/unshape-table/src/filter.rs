//! [`Filter`]: select a subset of rows matching a [`Predicate`].

use crate::Table;
use crate::column::{ColumnType, ScalarValue};
use crate::error::TableError;
use std::cmp::Ordering;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// A row predicate, built from typed comparisons and logical combinators (no query strings).
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum Predicate {
    /// `column == value`.
    Eq {
        /// Column to compare.
        column: String,
        /// Value to compare against.
        value: ScalarValue,
    },
    /// `column != value`.
    Ne {
        /// Column to compare.
        column: String,
        /// Value to compare against.
        value: ScalarValue,
    },
    /// `column < value`. Errors at apply time if the column type has no total order.
    Lt {
        /// Column to compare.
        column: String,
        /// Value to compare against.
        value: ScalarValue,
    },
    /// `column <= value`.
    Le {
        /// Column to compare.
        column: String,
        /// Value to compare against.
        value: ScalarValue,
    },
    /// `column > value`.
    Gt {
        /// Column to compare.
        column: String,
        /// Value to compare against.
        value: ScalarValue,
    },
    /// `column >= value`.
    Ge {
        /// Column to compare.
        column: String,
        /// Value to compare against.
        value: ScalarValue,
    },
    /// String column contains `value` as a substring.
    Contains {
        /// Column to search.
        column: String,
        /// Substring to search for.
        value: String,
    },
    /// Bool column is `true`.
    IsTrue {
        /// Column to check.
        column: String,
    },
    /// All sub-predicates hold.
    And(Vec<Predicate>),
    /// At least one sub-predicate holds.
    Or(Vec<Predicate>),
    /// The sub-predicate does not hold.
    Not(Box<Predicate>),
}

impl Predicate {
    fn compare(
        table: &Table,
        column: &str,
        row: usize,
        value: &ScalarValue,
    ) -> Result<Ordering, TableError> {
        let col = table.require_column(column)?;
        let actual = col.data.get(row).expect("row index within table bounds");
        if actual.column_type() != value.column_type() {
            return Err(TableError::ScalarTypeMismatch {
                column: column.to_string(),
                expected: actual.column_type(),
                actual: value.column_type(),
            });
        }
        let ordering = match (&actual, value) {
            (ScalarValue::F32(a), ScalarValue::F32(b)) => a.total_cmp(b),
            (ScalarValue::F64(a), ScalarValue::F64(b)) => a.total_cmp(b),
            (ScalarValue::I32(a), ScalarValue::I32(b)) => a.cmp(b),
            (ScalarValue::I64(a), ScalarValue::I64(b)) => a.cmp(b),
            (ScalarValue::Bool(a), ScalarValue::Bool(b)) => a.cmp(b),
            (ScalarValue::String(a), ScalarValue::String(b)) => a.cmp(b),
            _ => return Err(TableError::NotOrderable(column.to_string())),
        };
        Ok(ordering)
    }

    fn eval(&self, table: &Table, row: usize) -> Result<bool, TableError> {
        match self {
            Predicate::Eq { column, value } => {
                let col = table.require_column(column)?;
                let actual = col.data.get(row).expect("row index within table bounds");
                Ok(&actual == value)
            }
            Predicate::Ne { column, value } => {
                let col = table.require_column(column)?;
                let actual = col.data.get(row).expect("row index within table bounds");
                Ok(&actual != value)
            }
            Predicate::Lt { column, value } => {
                Ok(Self::compare(table, column, row, value)? == Ordering::Less)
            }
            Predicate::Le { column, value } => {
                Ok(Self::compare(table, column, row, value)? != Ordering::Greater)
            }
            Predicate::Gt { column, value } => {
                Ok(Self::compare(table, column, row, value)? == Ordering::Greater)
            }
            Predicate::Ge { column, value } => {
                Ok(Self::compare(table, column, row, value)? != Ordering::Less)
            }
            Predicate::Contains { column, value } => {
                let col = table.require_column(column)?;
                match col.data.get(row).expect("row index within table bounds") {
                    ScalarValue::String(s) => Ok(s.contains(value.as_str())),
                    other => Err(TableError::ScalarTypeMismatch {
                        column: column.clone(),
                        expected: other.column_type(),
                        actual: ColumnType::String,
                    }),
                }
            }
            Predicate::IsTrue { column } => {
                let col = table.require_column(column)?;
                match col.data.get(row).expect("row index within table bounds") {
                    ScalarValue::Bool(b) => Ok(b),
                    other => Err(TableError::ScalarTypeMismatch {
                        column: column.clone(),
                        expected: other.column_type(),
                        actual: ColumnType::Bool,
                    }),
                }
            }
            Predicate::And(preds) => {
                for p in preds {
                    if !p.eval(table, row)? {
                        return Ok(false);
                    }
                }
                Ok(true)
            }
            Predicate::Or(preds) => {
                for p in preds {
                    if p.eval(table, row)? {
                        return Ok(true);
                    }
                }
                Ok(false)
            }
            Predicate::Not(p) => Ok(!p.eval(table, row)?),
        }
    }
}

/// Keep only rows matching `predicate`.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Filter {
    /// The predicate rows must satisfy to be kept.
    pub predicate: Predicate,
}

impl Filter {
    /// Apply this filter, returning a new table containing only matching rows.
    pub fn apply(&self, table: &Table) -> Result<Table, TableError> {
        let mut indices = Vec::new();
        for row in 0..table.len() {
            if self.predicate.eval(table, row)? {
                indices.push(row);
            }
        }
        Ok(table.take_rows(&indices))
    }
}

impl Table {
    /// Keep only rows matching `predicate`. Sugar for [`Filter::apply`].
    pub fn filter(&self, predicate: Predicate) -> Result<Table, TableError> {
        Filter { predicate }.apply(self)
    }
}
