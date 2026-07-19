//! Columnar value types: [`ColumnData`], [`Column`], [`Table`].

use crate::error::TableError;
use glam::{Vec2, Vec3, Vec4};
use std::cmp::Ordering;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// The type of data stored in a [`Column`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum ColumnType {
    /// 32-bit float.
    F32,
    /// 64-bit float.
    F64,
    /// 32-bit signed integer.
    I32,
    /// 64-bit signed integer.
    I64,
    /// Boolean.
    Bool,
    /// UTF-8 string.
    String,
    /// 2D vector, for spatial tables.
    Vec2,
    /// 3D vector, for spatial tables.
    Vec3,
    /// 4D vector, for spatial tables.
    Vec4,
}

/// A single typed value, used for predicates, computed columns, and row access.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum ScalarValue {
    /// 32-bit float.
    F32(f32),
    /// 64-bit float.
    F64(f64),
    /// 32-bit signed integer.
    I32(i32),
    /// 64-bit signed integer.
    I64(i64),
    /// Boolean.
    Bool(bool),
    /// UTF-8 string.
    String(String),
    /// 2D vector.
    Vec2(Vec2),
    /// 3D vector.
    Vec3(Vec3),
    /// 4D vector.
    Vec4(Vec4),
}

impl ScalarValue {
    /// The [`ColumnType`] of this value.
    pub fn column_type(&self) -> ColumnType {
        match self {
            ScalarValue::F32(_) => ColumnType::F32,
            ScalarValue::F64(_) => ColumnType::F64,
            ScalarValue::I32(_) => ColumnType::I32,
            ScalarValue::I64(_) => ColumnType::I64,
            ScalarValue::Bool(_) => ColumnType::Bool,
            ScalarValue::String(_) => ColumnType::String,
            ScalarValue::Vec2(_) => ColumnType::Vec2,
            ScalarValue::Vec3(_) => ColumnType::Vec3,
            ScalarValue::Vec4(_) => ColumnType::Vec4,
        }
    }

    /// Widen this value to `f64` if it is numeric (`F32`, `F64`, `I32`, `I64`).
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            ScalarValue::F32(v) => Some(*v as f64),
            ScalarValue::F64(v) => Some(*v),
            ScalarValue::I32(v) => Some(*v as f64),
            ScalarValue::I64(v) => Some(*v as f64),
            _ => None,
        }
    }
}

/// Columnar storage for a single column: a homogeneous, contiguous array of values.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum ColumnData {
    /// 32-bit float column.
    F32(Vec<f32>),
    /// 64-bit float column.
    F64(Vec<f64>),
    /// 32-bit signed integer column.
    I32(Vec<i32>),
    /// 64-bit signed integer column.
    I64(Vec<i64>),
    /// Boolean column.
    Bool(Vec<bool>),
    /// UTF-8 string column.
    String(Vec<String>),
    /// 2D vector column.
    Vec2(Vec<Vec2>),
    /// 3D vector column.
    Vec3(Vec<Vec3>),
    /// 4D vector column.
    Vec4(Vec<Vec4>),
}

impl ColumnData {
    /// Number of rows (elements) in this column.
    pub fn len(&self) -> usize {
        match self {
            ColumnData::F32(v) => v.len(),
            ColumnData::F64(v) => v.len(),
            ColumnData::I32(v) => v.len(),
            ColumnData::I64(v) => v.len(),
            ColumnData::Bool(v) => v.len(),
            ColumnData::String(v) => v.len(),
            ColumnData::Vec2(v) => v.len(),
            ColumnData::Vec3(v) => v.len(),
            ColumnData::Vec4(v) => v.len(),
        }
    }

    /// Whether this column has zero rows.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The [`ColumnType`] of this column.
    pub fn column_type(&self) -> ColumnType {
        match self {
            ColumnData::F32(_) => ColumnType::F32,
            ColumnData::F64(_) => ColumnType::F64,
            ColumnData::I32(_) => ColumnType::I32,
            ColumnData::I64(_) => ColumnType::I64,
            ColumnData::Bool(_) => ColumnType::Bool,
            ColumnData::String(_) => ColumnType::String,
            ColumnData::Vec2(_) => ColumnType::Vec2,
            ColumnData::Vec3(_) => ColumnType::Vec3,
            ColumnData::Vec4(_) => ColumnType::Vec4,
        }
    }

    /// An empty column of the same type as `self`.
    pub fn empty_like(&self) -> ColumnData {
        match self {
            ColumnData::F32(_) => ColumnData::F32(Vec::new()),
            ColumnData::F64(_) => ColumnData::F64(Vec::new()),
            ColumnData::I32(_) => ColumnData::I32(Vec::new()),
            ColumnData::I64(_) => ColumnData::I64(Vec::new()),
            ColumnData::Bool(_) => ColumnData::Bool(Vec::new()),
            ColumnData::String(_) => ColumnData::String(Vec::new()),
            ColumnData::Vec2(_) => ColumnData::Vec2(Vec::new()),
            ColumnData::Vec3(_) => ColumnData::Vec3(Vec::new()),
            ColumnData::Vec4(_) => ColumnData::Vec4(Vec::new()),
        }
    }

    /// An empty column of the given type.
    pub fn empty(column_type: ColumnType) -> ColumnData {
        match column_type {
            ColumnType::F32 => ColumnData::F32(Vec::new()),
            ColumnType::F64 => ColumnData::F64(Vec::new()),
            ColumnType::I32 => ColumnData::I32(Vec::new()),
            ColumnType::I64 => ColumnData::I64(Vec::new()),
            ColumnType::Bool => ColumnData::Bool(Vec::new()),
            ColumnType::String => ColumnData::String(Vec::new()),
            ColumnType::Vec2 => ColumnData::Vec2(Vec::new()),
            ColumnType::Vec3 => ColumnData::Vec3(Vec::new()),
            ColumnType::Vec4 => ColumnData::Vec4(Vec::new()),
        }
    }

    /// Get the value at row `index` as a [`ScalarValue`], or `None` if out of bounds.
    pub fn get(&self, index: usize) -> Option<ScalarValue> {
        match self {
            ColumnData::F32(v) => v.get(index).copied().map(ScalarValue::F32),
            ColumnData::F64(v) => v.get(index).copied().map(ScalarValue::F64),
            ColumnData::I32(v) => v.get(index).copied().map(ScalarValue::I32),
            ColumnData::I64(v) => v.get(index).copied().map(ScalarValue::I64),
            ColumnData::Bool(v) => v.get(index).copied().map(ScalarValue::Bool),
            ColumnData::String(v) => v.get(index).cloned().map(ScalarValue::String),
            ColumnData::Vec2(v) => v.get(index).copied().map(ScalarValue::Vec2),
            ColumnData::Vec3(v) => v.get(index).copied().map(ScalarValue::Vec3),
            ColumnData::Vec4(v) => v.get(index).copied().map(ScalarValue::Vec4),
        }
    }

    /// Append a scalar value. Errors if the value's type does not match this column's type.
    pub fn push(&mut self, value: ScalarValue) -> Result<(), TableError> {
        match (self, value) {
            (ColumnData::F32(v), ScalarValue::F32(x)) => v.push(x),
            (ColumnData::F64(v), ScalarValue::F64(x)) => v.push(x),
            (ColumnData::I32(v), ScalarValue::I32(x)) => v.push(x),
            (ColumnData::I64(v), ScalarValue::I64(x)) => v.push(x),
            (ColumnData::Bool(v), ScalarValue::Bool(x)) => v.push(x),
            (ColumnData::String(v), ScalarValue::String(x)) => v.push(x),
            (ColumnData::Vec2(v), ScalarValue::Vec2(x)) => v.push(x),
            (ColumnData::Vec3(v), ScalarValue::Vec3(x)) => v.push(x),
            (ColumnData::Vec4(v), ScalarValue::Vec4(x)) => v.push(x),
            (col, value) => {
                return Err(TableError::ScalarTypeMismatch {
                    column: String::new(),
                    expected: col.column_type(),
                    actual: value.column_type(),
                });
            }
        }
        Ok(())
    }

    /// Build a new column by selecting `indices` (with repetition/reordering allowed) from
    /// this one. Panics if any index is out of bounds; callers are expected to validate
    /// indices against `self.len()` first (all in-crate call sites do).
    pub fn take_rows(&self, indices: &[usize]) -> ColumnData {
        match self {
            ColumnData::F32(v) => ColumnData::F32(indices.iter().map(|&i| v[i]).collect()),
            ColumnData::F64(v) => ColumnData::F64(indices.iter().map(|&i| v[i]).collect()),
            ColumnData::I32(v) => ColumnData::I32(indices.iter().map(|&i| v[i]).collect()),
            ColumnData::I64(v) => ColumnData::I64(indices.iter().map(|&i| v[i]).collect()),
            ColumnData::Bool(v) => ColumnData::Bool(indices.iter().map(|&i| v[i]).collect()),
            ColumnData::String(v) => {
                ColumnData::String(indices.iter().map(|&i| v[i].clone()).collect())
            }
            ColumnData::Vec2(v) => ColumnData::Vec2(indices.iter().map(|&i| v[i]).collect()),
            ColumnData::Vec3(v) => ColumnData::Vec3(indices.iter().map(|&i| v[i]).collect()),
            ColumnData::Vec4(v) => ColumnData::Vec4(indices.iter().map(|&i| v[i]).collect()),
        }
    }

    /// Compare rows `a` and `b`. Returns `None` for column types with no total order
    /// (`Vec2`, `Vec3`, `Vec4`).
    pub fn compare_rows(&self, a: usize, b: usize) -> Option<Ordering> {
        match self {
            ColumnData::F32(v) => Some(v[a].total_cmp(&v[b])),
            ColumnData::F64(v) => Some(v[a].total_cmp(&v[b])),
            ColumnData::I32(v) => Some(v[a].cmp(&v[b])),
            ColumnData::I64(v) => Some(v[a].cmp(&v[b])),
            ColumnData::Bool(v) => Some(v[a].cmp(&v[b])),
            ColumnData::String(v) => Some(v[a].cmp(&v[b])),
            ColumnData::Vec2(_) | ColumnData::Vec3(_) | ColumnData::Vec4(_) => None,
        }
    }

    /// Extract this column as `f64`s, widening `F32`/`I32`/`I64`. Returns `None` for
    /// non-numeric column types.
    pub fn as_f64_slice(&self) -> Option<Vec<f64>> {
        match self {
            ColumnData::F32(v) => Some(v.iter().map(|&x| x as f64).collect()),
            ColumnData::F64(v) => Some(v.clone()),
            ColumnData::I32(v) => Some(v.iter().map(|&x| x as f64).collect()),
            ColumnData::I64(v) => Some(v.iter().map(|&x| x as f64).collect()),
            _ => None,
        }
    }
}

/// A named, typed column: `data` holds one value per row.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Column {
    /// The column's name, unique within its [`Table`].
    pub name: String,
    /// The column's values.
    pub data: ColumnData,
}

impl Column {
    /// Create a new named column.
    pub fn new(name: impl Into<String>, data: ColumnData) -> Self {
        Self {
            name: name.into(),
            data,
        }
    }

    /// Number of rows in this column.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Whether this column has zero rows.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// The [`ColumnType`] of this column.
    pub fn column_type(&self) -> ColumnType {
        self.data.column_type()
    }
}

/// A single entry in a [`Table`]'s schema: a column's name and type.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ColumnSchema {
    /// Column name.
    pub name: String,
    /// Column type.
    pub column_type: ColumnType,
}

/// Columnar tabular data: named, typed columns, all with the same number of rows.
///
/// `Table` is a value, not an op — see `docs/design/ops-as-values.md`. Transformations
/// (filter, sort, group-by, join, select, ...) are separate op structs that take a `Table`
/// (or two, for join) and produce a new `Table`.
#[derive(Debug, Clone, PartialEq, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Table {
    columns: Vec<Column>,
}

impl Table {
    /// Create a new table from columns. Errors if columns have mismatched lengths or
    /// duplicate names.
    pub fn new(columns: Vec<Column>) -> Result<Self, TableError> {
        let mut seen = std::collections::HashSet::new();
        for column in &columns {
            if !seen.insert(column.name.as_str()) {
                return Err(TableError::DuplicateColumn(column.name.clone()));
            }
        }
        if let Some(expected) = columns.first().map(Column::len) {
            for column in &columns[1..] {
                if column.len() != expected {
                    return Err(TableError::LengthMismatch {
                        expected,
                        actual: column.len(),
                    });
                }
            }
        }
        Ok(Self { columns })
    }

    /// An empty table (no columns, no rows).
    pub fn empty() -> Self {
        Self {
            columns: Vec::new(),
        }
    }

    /// Number of rows in the table (0 if the table has no columns).
    pub fn len(&self) -> usize {
        self.columns.first().map(Column::len).unwrap_or(0)
    }

    /// Whether the table has zero rows.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Number of columns.
    pub fn num_columns(&self) -> usize {
        self.columns.len()
    }

    /// All columns, in order.
    pub fn columns(&self) -> &[Column] {
        &self.columns
    }

    /// Look up a column by name.
    pub fn column(&self, name: &str) -> Option<&Column> {
        self.columns.iter().find(|c| c.name == name)
    }

    /// Look up a column by name, returning an error naming the missing column if absent.
    pub fn require_column(&self, name: &str) -> Result<&Column, TableError> {
        self.column(name)
            .ok_or_else(|| TableError::ColumnNotFound(name.to_string()))
    }

    /// Index of a column by name.
    pub fn column_index(&self, name: &str) -> Option<usize> {
        self.columns.iter().position(|c| c.name == name)
    }

    /// Names of all columns, in order.
    pub fn column_names(&self) -> Vec<&str> {
        self.columns.iter().map(|c| c.name.as_str()).collect()
    }

    /// The table's schema: each column's name and type, in order.
    pub fn schema(&self) -> Vec<ColumnSchema> {
        self.columns
            .iter()
            .map(|c| ColumnSchema {
                name: c.name.clone(),
                column_type: c.column_type(),
            })
            .collect()
    }

    /// Return a new table with `column` appended. Errors if the name is already in use, or
    /// its length does not match the table's existing row count.
    pub fn with_column(&self, column: Column) -> Result<Self, TableError> {
        if self.column(&column.name).is_some() {
            return Err(TableError::DuplicateColumn(column.name.clone()));
        }
        if !self.columns.is_empty() && column.len() != self.len() {
            return Err(TableError::LengthMismatch {
                expected: self.len(),
                actual: column.len(),
            });
        }
        let mut columns = self.columns.clone();
        columns.push(column);
        Ok(Self { columns })
    }

    /// Build a new table by selecting `indices` (with repetition/reordering allowed) from
    /// each column. Rows referencing an index `>= self.len()` are not produced by any
    /// in-crate op; this is a low-level building block for [`crate::filter::Filter`],
    /// [`crate::sort::Sort`], and [`crate::join::Join`].
    pub fn take_rows(&self, indices: &[usize]) -> Self {
        let columns = self
            .columns
            .iter()
            .map(|c| Column::new(c.name.clone(), c.data.take_rows(indices)))
            .collect();
        Self { columns }
    }
}
