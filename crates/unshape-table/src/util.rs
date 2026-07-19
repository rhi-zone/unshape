//! Internal helpers shared by [`crate::join`] and [`crate::pivot`].

use crate::column::{ColumnType, ScalarValue};

/// A canonical, deterministic string key for a scalar value, suitable for hash-map lookups.
/// Floats are keyed by their bit pattern (not their textual form) so the key is exact and
/// total, including for `NaN`.
pub(crate) fn scalar_key(value: &ScalarValue) -> String {
    match value {
        ScalarValue::F32(v) => format!("f32:{:x}", v.to_bits()),
        ScalarValue::F64(v) => format!("f64:{:x}", v.to_bits()),
        ScalarValue::I32(v) => format!("i32:{v}"),
        ScalarValue::I64(v) => format!("i64:{v}"),
        ScalarValue::Bool(v) => format!("bool:{v}"),
        ScalarValue::String(v) => format!("str:{v}"),
        ScalarValue::Vec2(v) => format!("vec2:{:x}:{:x}", v.x.to_bits(), v.y.to_bits()),
        ScalarValue::Vec3(v) => format!(
            "vec3:{:x}:{:x}:{:x}",
            v.x.to_bits(),
            v.y.to_bits(),
            v.z.to_bits()
        ),
        ScalarValue::Vec4(v) => format!(
            "vec4:{:x}:{:x}:{:x}:{:x}",
            v.x.to_bits(),
            v.y.to_bits(),
            v.z.to_bits(),
            v.w.to_bits()
        ),
    }
}

/// The "zero" value for a column type: `0`, `false`, `""`, or a zero vector. Used to fill
/// cells with no corresponding data (unmatched join rows, absent pivot combinations), since
/// there is no nullable column type yet — see `docs/design/domain-subsumption.md`.
pub(crate) fn zero_value(column_type: ColumnType) -> ScalarValue {
    match column_type {
        ColumnType::F32 => ScalarValue::F32(0.0),
        ColumnType::F64 => ScalarValue::F64(0.0),
        ColumnType::I32 => ScalarValue::I32(0),
        ColumnType::I64 => ScalarValue::I64(0),
        ColumnType::Bool => ScalarValue::Bool(false),
        ColumnType::String => ScalarValue::String(String::new()),
        ColumnType::Vec2 => ScalarValue::Vec2(glam::Vec2::ZERO),
        ColumnType::Vec3 => ScalarValue::Vec3(glam::Vec3::ZERO),
        ColumnType::Vec4 => ScalarValue::Vec4(glam::Vec4::ZERO),
    }
}

/// Render a scalar as a human-readable string, for use as a generated column name (e.g. in
/// [`crate::pivot::Pivot`]). Panics on vector values — callers must reject vector-typed
/// columns as pivot keys first (via [`crate::groupby::GroupBy`]'s orderability check).
pub(crate) fn scalar_display(value: &ScalarValue) -> String {
    match value {
        ScalarValue::F32(v) => v.to_string(),
        ScalarValue::F64(v) => v.to_string(),
        ScalarValue::I32(v) => v.to_string(),
        ScalarValue::I64(v) => v.to_string(),
        ScalarValue::Bool(v) => v.to_string(),
        ScalarValue::String(v) => v.clone(),
        ScalarValue::Vec2(_) | ScalarValue::Vec3(_) | ScalarValue::Vec4(_) => {
            unreachable!("vector-typed pivot columns are rejected upstream by GroupBy")
        }
    }
}
