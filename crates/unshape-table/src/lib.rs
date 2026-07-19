//! Columnar tabular data for unshape: `Table`/`Column` values, plus ops for filtering,
//! sorting, grouping/aggregating, joining, selecting, computing columns, and pivoting.
//!
//! `Table` is a value (columnar storage: each column is a contiguous array), not an op — see
//! `docs/design/ops-as-values.md`. Every transformation is a small op struct with an
//! `apply(&self, ...)` method, following the "operations as values" convention: op structs
//! carry all their parameters, derive `Serialize`/`Deserialize` behind the `serde` feature,
//! and (behind the `dynop` feature) implement [`unshape_op::DynOp`] for use in serialized
//! pipelines and node graphs.
//!
//! This is "Excel + ClickHouse without the UI": a computation substrate for tabular data,
//! domain-agnostic (numeric/bool/string columns cover most cases, plus `glam` vector columns
//! for spatial tables), designed to compose with other unshape domains via the node graph.

mod add_column;
mod column;
mod error;
mod filter;
mod groupby;
mod join;
mod pivot;
mod select;
mod sort;
mod util;

#[cfg(feature = "dynop")]
mod dynop;

pub use add_column::{AddColumn, ColumnExpr};
pub use column::{Column, ColumnData, ColumnSchema, ColumnType, ScalarValue, Table};
pub use error::TableError;
pub use filter::{Filter, Predicate};
pub use groupby::{AggregateFn, Aggregation, GroupBy};
pub use join::{Join, JoinKind};
pub use pivot::Pivot;
pub use select::Select;
pub use sort::{Sort, SortKey};

#[cfg(test)]
mod tests {
    use super::*;

    fn people_table() -> Table {
        Table::new(vec![
            Column::new(
                "name",
                ColumnData::String(vec![
                    "alice".into(),
                    "bob".into(),
                    "carol".into(),
                    "dave".into(),
                    "erin".into(),
                ]),
            ),
            Column::new("age", ColumnData::I32(vec![30, 25, 35, 25, 40])),
            Column::new(
                "dept",
                ColumnData::String(vec![
                    "eng".into(),
                    "eng".into(),
                    "sales".into(),
                    "sales".into(),
                    "eng".into(),
                ]),
            ),
            Column::new(
                "salary",
                ColumnData::F64(vec![90_000.0, 80_000.0, 95_000.0, 70_000.0, 110_000.0]),
            ),
        ])
        .unwrap()
    }

    #[test]
    fn table_new_rejects_mismatched_lengths() {
        let result = Table::new(vec![
            Column::new("a", ColumnData::I32(vec![1, 2, 3])),
            Column::new("b", ColumnData::I32(vec![1, 2])),
        ]);
        assert!(matches!(
            result,
            Err(TableError::LengthMismatch {
                expected: 3,
                actual: 2
            })
        ));
    }

    #[test]
    fn table_new_rejects_duplicate_names() {
        let result = Table::new(vec![
            Column::new("a", ColumnData::I32(vec![1])),
            Column::new("a", ColumnData::I32(vec![2])),
        ]);
        assert!(matches!(result, Err(TableError::DuplicateColumn(name)) if name == "a"));
    }

    #[test]
    fn table_schema_reports_names_and_types() {
        let table = people_table();
        let schema = table.schema();
        assert_eq!(schema.len(), 4);
        assert_eq!(schema[0].name, "name");
        assert_eq!(schema[0].column_type, ColumnType::String);
        assert_eq!(schema[1].column_type, ColumnType::I32);
    }

    #[test]
    fn filter_keeps_matching_rows() {
        let table = people_table();
        let filtered = table
            .filter(Predicate::Eq {
                column: "dept".into(),
                value: ScalarValue::String("eng".into()),
            })
            .unwrap();
        assert_eq!(filtered.len(), 3);
        let names = filtered.column("name").unwrap();
        assert_eq!(
            names.data,
            ColumnData::String(vec!["alice".into(), "bob".into(), "erin".into()])
        );
    }

    #[test]
    fn filter_and_or_not_compose() {
        let table = people_table();
        let filtered = table
            .filter(Predicate::And(vec![
                Predicate::Eq {
                    column: "dept".into(),
                    value: ScalarValue::String("eng".into()),
                },
                Predicate::Not(Box::new(Predicate::Lt {
                    column: "age".into(),
                    value: ScalarValue::I32(30),
                })),
            ]))
            .unwrap();
        // eng rows: alice(30), bob(25), erin(40); age >= 30 -> alice, erin
        assert_eq!(filtered.len(), 2);
    }

    #[test]
    fn filter_missing_column_errors() {
        let table = people_table();
        let result = table.filter(Predicate::IsTrue {
            column: "nope".into(),
        });
        assert!(matches!(result, Err(TableError::ColumnNotFound(name)) if name == "nope"));
    }

    #[test]
    fn sort_ascending_and_descending() {
        let table = people_table();
        let sorted = table.sort_by("age").unwrap();
        let ages = match &sorted.column("age").unwrap().data {
            ColumnData::I32(v) => v.clone(),
            _ => panic!("expected i32 column"),
        };
        assert_eq!(ages, vec![25, 25, 30, 35, 40]);

        let sorted_desc = Sort::by_descending("age").apply(&table).unwrap();
        let ages_desc = match &sorted_desc.column("age").unwrap().data {
            ColumnData::I32(v) => v.clone(),
            _ => panic!("expected i32 column"),
        };
        assert_eq!(ages_desc, vec![40, 35, 30, 25, 25]);
    }

    #[test]
    fn sort_is_stable_and_supports_multiple_keys() {
        let table = people_table();
        let sorted = Sort {
            keys: vec![SortKey::ascending("dept"), SortKey::ascending("age")],
        }
        .apply(&table)
        .unwrap();
        let names = match &sorted.column("name").unwrap().data {
            ColumnData::String(v) => v.clone(),
            _ => panic!("expected string column"),
        };
        // eng: bob(25), alice(30), erin(40); sales: dave(25), carol(35)
        assert_eq!(names, vec!["bob", "alice", "erin", "dave", "carol"]);
    }

    #[test]
    fn select_projects_columns_in_order() {
        let table = people_table();
        let selected = table.select(["salary", "name"]).unwrap();
        assert_eq!(selected.column_names(), vec!["salary", "name"]);
        assert_eq!(selected.num_columns(), 2);
    }

    #[test]
    fn select_missing_column_errors() {
        let table = people_table();
        assert!(table.select(["nope"]).is_err());
    }

    #[test]
    fn add_column_computes_expression() {
        let table = people_table();
        let with_bonus = table
            .add_column(
                "salary_with_bonus",
                ColumnExpr::Add(
                    Box::new(ColumnExpr::column("salary")),
                    Box::new(ColumnExpr::Mul(
                        Box::new(ColumnExpr::column("salary")),
                        Box::new(ColumnExpr::constant(0.1)),
                    )),
                ),
            )
            .unwrap();
        let bonus = match &with_bonus.column("salary_with_bonus").unwrap().data {
            ColumnData::F64(v) => v.clone(),
            _ => panic!("expected f64 column"),
        };
        assert!((bonus[0] - 99_000.0).abs() < 1e-6);
    }

    #[test]
    fn group_by_sum_mean_count_min_max() {
        let table = people_table();
        let grouped = table
            .group_by(
                ["dept"],
                vec![
                    Aggregation::new("salary", AggregateFn::Sum),
                    Aggregation::new("salary", AggregateFn::Mean),
                    Aggregation::new("name", AggregateFn::Count),
                    Aggregation::new("age", AggregateFn::Min),
                    Aggregation::new("age", AggregateFn::Max),
                ],
            )
            .unwrap();
        assert_eq!(grouped.len(), 2);

        let dept_index = |dept: &str| -> usize {
            match &grouped.column("dept").unwrap().data {
                ColumnData::String(v) => v.iter().position(|d| d == dept).unwrap(),
                _ => panic!("expected string column"),
            }
        };
        let eng = dept_index("eng");
        let sales = dept_index("sales");

        let sums = match &grouped.column("salary_sum").unwrap().data {
            ColumnData::F64(v) => v.clone(),
            _ => panic!("expected f64 column"),
        };
        assert!((sums[eng] - 280_000.0).abs() < 1e-6); // 90k + 80k + 110k
        assert!((sums[sales] - 165_000.0).abs() < 1e-6); // 95k + 70k

        let counts = match &grouped.column("name_count").unwrap().data {
            ColumnData::I64(v) => v.clone(),
            _ => panic!("expected i64 column"),
        };
        assert_eq!(counts[eng], 3);
        assert_eq!(counts[sales], 2);

        let mins = match &grouped.column("age_min").unwrap().data {
            ColumnData::I32(v) => v.clone(),
            _ => panic!("expected i32 column"),
        };
        assert_eq!(mins[eng], 25);
        assert_eq!(mins[sales], 25);

        let maxes = match &grouped.column("age_max").unwrap().data {
            ColumnData::I32(v) => v.clone(),
            _ => panic!("expected i32 column"),
        };
        assert_eq!(maxes[eng], 40);
        assert_eq!(maxes[sales], 35);
    }

    #[test]
    fn group_by_rejects_non_numeric_sum() {
        let table = people_table();
        let result = table.group_by(["dept"], vec![Aggregation::new("name", AggregateFn::Sum)]);
        assert!(matches!(result, Err(TableError::NotNumeric(name)) if name == "name"));
    }

    #[test]
    fn join_inner_matches_rows() {
        let depts = Table::new(vec![
            Column::new(
                "dept",
                ColumnData::String(vec!["eng".into(), "sales".into(), "hr".into()]),
            ),
            Column::new("floor", ColumnData::I32(vec![3, 5, 1])),
        ])
        .unwrap();
        let table = people_table();
        let joined = table.join(&depts, "dept", "dept").unwrap();
        assert_eq!(joined.len(), 5); // all people match eng or sales; hr has no people
        assert_eq!(joined.num_columns(), 5); // name, age, dept, salary, floor (dept_right dropped)
        assert!(joined.column("floor").is_some());
    }

    #[test]
    fn join_left_fills_unmatched_with_zero_value() {
        let depts = Table::new(vec![
            Column::new("dept", ColumnData::String(vec!["eng".into()])),
            Column::new("floor", ColumnData::I32(vec![3])),
        ])
        .unwrap();
        let table = people_table();
        let joined = Join {
            left_key: "dept".into(),
            right_key: "dept".into(),
            kind: JoinKind::Left,
        }
        .apply(&table, &depts)
        .unwrap();
        assert_eq!(joined.len(), 5); // every left row kept, sales rows unmatched
        let floors = match &joined.column("floor").unwrap().data {
            ColumnData::I32(v) => v.clone(),
            _ => panic!("expected i32 column"),
        };
        // sales rows (carol, dave) get the zero fill value
        assert!(floors.contains(&0));
        assert!(floors.contains(&3));
    }

    #[test]
    fn pivot_reshapes_long_to_wide() {
        let table = people_table();
        let pivoted = table
            .pivot("dept", "name", "salary", AggregateFn::Sum)
            .unwrap();
        assert_eq!(pivoted.len(), 2); // eng, sales
        // one column per distinct name, plus the index column
        assert_eq!(pivoted.num_columns(), 1 + 5);
        assert!(pivoted.column("alice").is_some());
        assert!(pivoted.column("carol").is_some());
    }

    #[test]
    fn take_rows_reorders_all_columns_together() {
        let table = people_table();
        let reordered = table.take_rows(&[4, 0]);
        let names = match &reordered.column("name").unwrap().data {
            ColumnData::String(v) => v.clone(),
            _ => panic!("expected string column"),
        };
        assert_eq!(names, vec!["erin", "alice"]);
    }
}
