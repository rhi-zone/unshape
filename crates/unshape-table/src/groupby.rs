//! [`GroupBy`]: group rows by one or more columns and aggregate the rest.

use crate::Table;
use crate::column::{Column, ColumnData, ColumnType};
use crate::error::TableError;
use std::cmp::Ordering;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// An aggregation function applied to a column within each group.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum AggregateFn {
    /// Sum of numeric values in the group. Requires a numeric column.
    Sum,
    /// Arithmetic mean of numeric values in the group. Requires a numeric column.
    Mean,
    /// Number of rows in the group. Works for any column type.
    Count,
    /// Minimum value in the group. Requires an orderable column type.
    Min,
    /// Maximum value in the group. Requires an orderable column type.
    Max,
}

/// One aggregation to compute per group: `output_name = function(column)`.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Aggregation {
    /// Source column to aggregate.
    pub column: String,
    /// Aggregation function to apply.
    pub function: AggregateFn,
    /// Name of the resulting output column.
    pub output_name: String,
}

impl Aggregation {
    /// Create an aggregation, naming the output column `{column}_{function}`.
    pub fn new(column: impl Into<String>, function: AggregateFn) -> Self {
        let column = column.into();
        let suffix = match function {
            AggregateFn::Sum => "sum",
            AggregateFn::Mean => "mean",
            AggregateFn::Count => "count",
            AggregateFn::Min => "min",
            AggregateFn::Max => "max",
        };
        let output_name = format!("{column}_{suffix}");
        Self {
            column,
            function,
            output_name,
        }
    }
}

/// Group rows by one or more columns, then compute one or more aggregations per group.
/// Output has one row per distinct combination of group-by values, plus one column per
/// aggregation.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct GroupBy {
    /// Columns to group by.
    pub group_by: Vec<String>,
    /// Aggregations to compute per group.
    pub aggregations: Vec<Aggregation>,
}

fn extreme_index(
    column_name: &str,
    data: &ColumnData,
    chunk: &[usize],
    want_min: bool,
) -> Result<usize, TableError> {
    let mut best = chunk[0];
    for &i in &chunk[1..] {
        let ordering = data
            .compare_rows(i, best)
            .ok_or_else(|| TableError::NotOrderable(column_name.to_string()))?;
        let better = if want_min {
            ordering == Ordering::Less
        } else {
            ordering == Ordering::Greater
        };
        if better {
            best = i;
        }
    }
    Ok(best)
}

impl GroupBy {
    /// Apply this group-by, returning a new table with one row per group.
    pub fn apply(&self, table: &Table) -> Result<Table, TableError> {
        let group_columns: Vec<&Column> = self
            .group_by
            .iter()
            .map(|name| table.require_column(name))
            .collect::<Result<_, _>>()?;
        for column in &group_columns {
            if matches!(
                column.column_type(),
                ColumnType::Vec2 | ColumnType::Vec3 | ColumnType::Vec4
            ) {
                return Err(TableError::NotOrderable(column.name.clone()));
            }
        }
        // Validate aggregation source columns up front too, so a bad reference fails
        // regardless of whether the table has any rows.
        for aggregation in &self.aggregations {
            table.require_column(&aggregation.column)?;
        }

        let mut indices: Vec<usize> = (0..table.len()).collect();
        indices.sort_by(|&a, &b| {
            for column in &group_columns {
                match column.data.compare_rows(a, b) {
                    Some(Ordering::Equal) | None => continue,
                    Some(other) => return other,
                }
            }
            Ordering::Equal
        });

        let chunks: Vec<&[usize]> = {
            let mut chunks = Vec::new();
            let mut start = 0;
            for i in 1..indices.len() {
                let same_group = group_columns.iter().all(|column| {
                    column.data.compare_rows(indices[i - 1], indices[i]) == Some(Ordering::Equal)
                });
                if !same_group {
                    chunks.push(&indices[start..i]);
                    start = i;
                }
            }
            if start < indices.len() {
                chunks.push(&indices[start..]);
            }
            chunks
        };

        let mut group_output: Vec<ColumnData> =
            group_columns.iter().map(|c| c.data.empty_like()).collect();
        for chunk in &chunks {
            for (output, column) in group_output.iter_mut().zip(&group_columns) {
                let value = column.data.get(chunk[0]).expect("chunk index in bounds");
                output
                    .push(value)
                    .expect("value type matches source column");
            }
        }

        let mut aggregation_output: Vec<ColumnData> = Vec::with_capacity(self.aggregations.len());
        for aggregation in &self.aggregations {
            let source = table.require_column(&aggregation.column)?;
            let data = match aggregation.function {
                AggregateFn::Count => {
                    ColumnData::I64(chunks.iter().map(|chunk| chunk.len() as i64).collect())
                }
                AggregateFn::Sum | AggregateFn::Mean => {
                    let values = source
                        .data
                        .as_f64_slice()
                        .ok_or_else(|| TableError::NotNumeric(aggregation.column.clone()))?;
                    let sums: Vec<f64> = chunks
                        .iter()
                        .map(|chunk| chunk.iter().map(|&i| values[i]).sum())
                        .collect();
                    if aggregation.function == AggregateFn::Sum {
                        ColumnData::F64(sums)
                    } else {
                        ColumnData::F64(
                            sums.into_iter()
                                .zip(chunks.iter())
                                .map(|(sum, chunk)| sum / chunk.len() as f64)
                                .collect(),
                        )
                    }
                }
                AggregateFn::Min | AggregateFn::Max => {
                    let want_min = aggregation.function == AggregateFn::Min;
                    let extreme_indices: Vec<usize> = chunks
                        .iter()
                        .map(|chunk| {
                            extreme_index(&aggregation.column, &source.data, chunk, want_min)
                        })
                        .collect::<Result<_, _>>()?;
                    source.data.take_rows(&extreme_indices)
                }
            };
            aggregation_output.push(data);
        }

        let mut columns = Vec::with_capacity(self.group_by.len() + self.aggregations.len());
        for (name, data) in self.group_by.iter().zip(group_output) {
            columns.push(Column::new(name.clone(), data));
        }
        for (aggregation, data) in self.aggregations.iter().zip(aggregation_output) {
            columns.push(Column::new(aggregation.output_name.clone(), data));
        }
        Table::new(columns)
    }
}

impl Table {
    /// Group by these columns and compute these aggregations. Sugar for [`GroupBy::apply`].
    pub fn group_by(
        &self,
        group_by: impl IntoIterator<Item = impl Into<String>>,
        aggregations: Vec<Aggregation>,
    ) -> Result<Table, TableError> {
        GroupBy {
            group_by: group_by.into_iter().map(Into::into).collect(),
            aggregations,
        }
        .apply(self)
    }
}
