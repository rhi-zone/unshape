//! Error types for table operations.

use crate::column::ColumnType;
use thiserror::Error;

/// Errors that can occur while constructing or operating on a [`Table`](crate::Table).
#[derive(Debug, Clone, PartialEq, Error)]
pub enum TableError {
    /// A referenced column does not exist in the table.
    #[error("column not found: {0}")]
    ColumnNotFound(String),

    /// A column being added has a different length than the table's existing columns.
    #[error("column length mismatch: expected {expected} rows, got {actual}")]
    LengthMismatch {
        /// Expected row count (the table's current length).
        expected: usize,
        /// Actual length of the column being added.
        actual: usize,
    },

    /// Two columns being combined (e.g. in a join) have incompatible types.
    #[error("type mismatch on column {column}: expected {expected:?}, got {actual:?}")]
    TypeMismatch {
        /// Name of the offending column.
        column: String,
        /// Type that was expected.
        expected: ColumnType,
        /// Type that was actually found.
        actual: ColumnType,
    },

    /// A table was constructed with two columns sharing the same name.
    #[error("duplicate column name: {0}")]
    DuplicateColumn(String),

    /// A column's type does not support ordering comparisons (e.g. vector columns).
    #[error("column {0} does not support ordering")]
    NotOrderable(String),

    /// A column's type is not numeric, but a numeric aggregation or computation was requested.
    #[error("column {0} is not numeric")]
    NotNumeric(String),

    /// A scalar value's type did not match the column it was compared/computed against.
    #[error("scalar type mismatch on column {column}: expected {expected:?}, got {actual:?}")]
    ScalarTypeMismatch {
        /// Name of the offending column.
        column: String,
        /// Type that was expected.
        expected: ColumnType,
        /// Type that was actually found.
        actual: ColumnType,
    },
}
