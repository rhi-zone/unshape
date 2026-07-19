# unshape-table

Columnar tabular data: `Table`/`Column` values, plus ops for filtering, sorting,
grouping/aggregating, joining, selecting, computing columns, and pivoting.

## Purpose

"Excel + ClickHouse without the UI" — a computation substrate for tabular data. `Table` is a
value (columnar storage: each column is a contiguous typed array), not an op. Every
transformation — `Filter`, `Sort`, `GroupBy`, `Join`, `Select`, `AddColumn`, `Pivot` — is a
serializable op struct following the workspace's "operations as values" convention, so table
pipelines can be saved as project files, replayed, and composed into node graphs the same
way mesh or image ops are.

Column types cover the common cases (`f32`, `f64`, `i32`, `i64`, `bool`, `String`) plus
`glam::Vec2`/`Vec3`/`Vec4` for spatial tables (e.g. scattered point data, per-instance
attributes).

Filter predicates and computed-column expressions are typed Rust trees (`Predicate`,
`ColumnExpr`), not query strings — consistent with the workspace's "no DSLs" constraint.

## Related Crates

- **unshape-op** - `DynOp` trait; behind the `dynop` feature, `Filter`/`Sort`/`Select`/
  `GroupBy`/`AddColumn`/`Pivot` implement it for use in serialized pipelines and node graphs
  (`Join` is a two-input "Construction"-shaped op and is called directly instead — see
  `docs/design/op-shapes.md`)
- **unshape-serde** - Graph serialization; a `Table` can flow through the same `SerialGraph`
  machinery as other value types once wired into `Value`
- **unshape-spatial** - Spatial data structures; a `Table` with `Vec2`/`Vec3` columns is a
  natural input to spatial queries (nearest-neighbor, range queries) over point data

## Use Cases

### Filtering and sorting

```rust
let adults = table
    .filter(Predicate::Ge { column: "age".into(), value: ScalarValue::I32(18) })?
    .sort_by("age")?;
```

### Group-by aggregation

```rust
let by_dept = table.group_by(
    ["dept"],
    vec![
        Aggregation::new("salary", AggregateFn::Sum),
        Aggregation::new("salary", AggregateFn::Mean),
        Aggregation::new("name", AggregateFn::Count),
    ],
)?;
```

### Joining two tables

```rust
let enriched = people.join(&departments, "dept", "dept")?; // inner join
let full = Join { left_key: "dept".into(), right_key: "dept".into(), kind: JoinKind::Left }
    .apply(&people, &departments)?;
```

### Computed columns

```rust
let with_bonus = table.add_column(
    "total_comp",
    ColumnExpr::Add(
        Box::new(ColumnExpr::column("salary")),
        Box::new(ColumnExpr::column("bonus")),
    ),
)?;
```

### Pivoting long to wide

```rust
let wide = table.pivot("dept", "quarter", "revenue", AggregateFn::Sum)?;
```

## Known Limitations

There is no nullable column type yet. `Join` (`Left`/`Right`/`Full`) and `Pivot` fill
unmatched cells with each column type's zero value (`0`, `false`, `""`, a zero vector)
rather than a true null — a cell filled this way is not distinguishable from a genuine zero.
See `docs/design/domain-subsumption.md` for the `Table` design and revisit this once
nullability is added to the value system.

## Compositions

### With unshape-spatial

Use `Vec2`/`Vec3` columns to hold point positions, feed them into spatial queries, then
join the results back onto the table by row index.

### With unshape-core

Wire `Filter`/`Sort`/`GroupBy`/`Select`/`AddColumn`/`Pivot` into a node graph via their
`DynOp` impls (behind the `dynop` feature) for visual, replayable table pipelines.
