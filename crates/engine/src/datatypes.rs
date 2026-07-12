//! The engine's type system (book: "Type System").
//!
//! Operators never touch arrow's concrete array types directly — they see
//! [`ColumnVector`], the engine's own column abstraction. [`ArrowFieldVector`]
//! adapts an arrow [`ArrayRef`] to it; commit 1.4 adds a second impl for
//! scalar literals that never materializes a full array.

use std::sync::Arc;

use arrow::array::{
    Array, ArrayRef, BooleanArray, Float32Array, Float64Array, Int8Array, Int16Array, Int32Array,
    Int64Array, StringArray, UInt8Array, UInt16Array, UInt32Array, UInt64Array,
};
use arrow::datatypes::{DataType, Schema, SchemaRef};
use arrow::error::ArrowError;
use arrow::record_batch::RecordBatch as ArrowRecordBatch;

/// One value read out of a column. KQuery returns Kotlin's `Any?` here; Rust
/// has no `Any?`, so the engine's supported scalar types are a closed enum.
/// SQL NULL is represented outside this enum, as `Option<ScalarValue>::None`.
#[derive(Debug, Clone, PartialEq)]
pub enum ScalarValue {
    Boolean(bool),
    Int8(i8),
    Int16(i16),
    Int32(i32),
    Int64(i64),
    UInt8(u8),
    UInt16(u16),
    UInt32(u32),
    UInt64(u64),
    Float32(f32),
    Float64(f64),
    Utf8(String),
}

/// A read-only column, `size()` rows long.
pub trait ColumnVector {
    fn data_type(&self) -> &DataType;

    /// Value at row `i`; `None` means SQL NULL.
    ///
    /// Panics if `i >= size()`, like slice indexing.
    fn value(&self, i: usize) -> Option<ScalarValue>;

    fn size(&self) -> usize;
}

/// Adapter exposing any arrow array as a [`ColumnVector`].
pub struct ArrowFieldVector {
    array: ArrayRef,
}

impl ArrowFieldVector {
    pub fn new(array: ArrayRef) -> Self {
        Self { array }
    }
}

impl ColumnVector for ArrowFieldVector {
    fn data_type(&self) -> &DataType {
        self.array.data_type()
    }

    fn value(&self, i: usize) -> Option<ScalarValue> {
        assert!(
            i < self.array.len(),
            "row {i} out of bounds for column of {} rows",
            self.array.len()
        );
        if self.array.is_null(i) {
            return None;
        }
        // One arm per supported type: downcast to the concrete array, read
        // row i, wrap in the matching ScalarValue variant.
        macro_rules! read {
            ($array_ty:ty, $variant:ident) => {{
                let a = self.array.as_any().downcast_ref::<$array_ty>().unwrap();
                ScalarValue::$variant(a.value(i))
            }};
        }
        Some(match self.array.data_type() {
            DataType::Boolean => read!(BooleanArray, Boolean),
            DataType::Int8 => read!(Int8Array, Int8),
            DataType::Int16 => read!(Int16Array, Int16),
            DataType::Int32 => read!(Int32Array, Int32),
            DataType::Int64 => read!(Int64Array, Int64),
            DataType::UInt8 => read!(UInt8Array, UInt8),
            DataType::UInt16 => read!(UInt16Array, UInt16),
            DataType::UInt32 => read!(UInt32Array, UInt32),
            DataType::UInt64 => read!(UInt64Array, UInt64),
            DataType::Float32 => read!(Float32Array, Float32),
            DataType::Float64 => read!(Float64Array, Float64),
            // Utf8 can't use the macro: a.value(i) is &str, we store String.
            DataType::Utf8 => {
                let a = self.array.as_any().downcast_ref::<StringArray>().unwrap();
                ScalarValue::Utf8(a.value(i).to_string())
            }
            other => panic!("ArrowFieldVector: unsupported data type {other:?}"),
        })
    }

    fn size(&self) -> usize {
        self.array.len()
    }
}

/// A scalar pretending to be a column: every row reads the same value.
///
/// Physical literal expressions (commit 2.4) return this so a constant in
/// e.g. `price * 0.9` is stored once, not materialized into an N-row array.
pub struct LiteralValueVector {
    data_type: DataType,
    value: Option<ScalarValue>,
    size: usize,
}

impl LiteralValueVector {
    /// `value: None` makes a column of NULLs.
    pub fn new(data_type: DataType, value: Option<ScalarValue>, size: usize) -> Self {
        Self {
            data_type,
            value,
            size,
        }
    }
}

impl ColumnVector for LiteralValueVector {
    fn data_type(&self) -> &DataType {
        &self.data_type
    }

    fn value(&self, i: usize) -> Option<ScalarValue> {
        assert!(
            i < self.size,
            "row {i} out of bounds for column of {} rows",
            self.size
        );
        self.value.clone()
    }

    fn size(&self) -> usize {
        self.size
    }
}

/// A batch of equal-length engine columns plus the schema describing them —
/// the unit of data flowing between operators.
///
/// Mirrors arrow's `RecordBatch`, but the columns are `dyn ColumnVector`, so
/// a literal column travels through operators exactly like an arrow-backed one.
/// Cloning is cheap: the schema and every column are behind `Arc`s.
#[derive(Clone)]
pub struct RecordBatch {
    schema: SchemaRef,
    columns: Vec<Arc<dyn ColumnVector>>,
}

impl RecordBatch {
    /// Panics if the column count doesn't match the schema, or if the
    /// columns have differing lengths.
    pub fn new(schema: SchemaRef, columns: Vec<Arc<dyn ColumnVector>>) -> Self {
        assert_eq!(
            schema.fields().len(),
            columns.len(),
            "schema has {} fields but {} columns were provided",
            schema.fields().len(),
            columns.len()
        );
        if let Some(first) = columns.first() {
            assert!(
                columns.iter().all(|c| c.size() == first.size()),
                "columns have differing lengths"
            );
        }
        Self { schema, columns }
    }

    pub fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }

    pub fn row_count(&self) -> usize {
        self.columns.first().map_or(0, |c| c.size())
    }

    pub fn column_count(&self) -> usize {
        self.columns.len()
    }

    pub fn column(&self, i: usize) -> &Arc<dyn ColumnVector> {
        &self.columns[i]
    }
}

impl From<ArrowRecordBatch> for RecordBatch {
    /// Wrap every arrow column in an [`ArrowFieldVector`]. Data sources
    /// (1.7–1.9) use this to hand arrow reader output to the engine.
    fn from(batch: ArrowRecordBatch) -> Self {
        let columns = batch
            .columns()
            .iter()
            .map(|a| Arc::new(ArrowFieldVector::new(Arc::clone(a))) as Arc<dyn ColumnVector>)
            .collect();
        Self::new(batch.schema(), columns)
    }
}

/// A schema containing only `names`, in the order given. Errors on names the
/// schema doesn't have. (The by-index case is arrow's own `Schema::project`.)
pub fn project_by_name(schema: &Schema, names: &[&str]) -> Result<Schema, ArrowError> {
    let indices = names
        .iter()
        .map(|name| schema.index_of(name))
        .collect::<Result<Vec<_>, _>>()?;
    schema.project(&indices)
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::datatypes::Field;
    use std::sync::Arc;

    #[test]
    fn arrow_field_vector_reads_values_and_nulls() {
        let array = Int64Array::from(vec![Some(10), None, Some(30)]);
        let col = ArrowFieldVector::new(Arc::new(array));

        assert_eq!(col.size(), 3);
        assert_eq!(col.data_type(), &DataType::Int64);
        assert_eq!(col.value(0), Some(ScalarValue::Int64(10)));
        assert_eq!(col.value(1), None);
        assert_eq!(col.value(2), Some(ScalarValue::Int64(30)));
    }

    #[test]
    fn arrow_field_vector_reads_strings() {
        let array = StringArray::from(vec![Some("a"), None]);
        let col = ArrowFieldVector::new(Arc::new(array));

        assert_eq!(col.value(0), Some(ScalarValue::Utf8("a".to_string())));
        assert_eq!(col.value(1), None);
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn arrow_field_vector_panics_past_the_end() {
        let array = Int64Array::from(vec![1]);
        let col = ArrowFieldVector::new(Arc::new(array));
        col.value(1);
    }

    #[test]
    fn literal_vector_repeats_one_value() {
        let col = LiteralValueVector::new(DataType::Int64, Some(ScalarValue::Int64(7)), 1000);

        assert_eq!(col.size(), 1000);
        assert_eq!(col.data_type(), &DataType::Int64);
        assert_eq!(col.value(0), Some(ScalarValue::Int64(7)));
        assert_eq!(col.value(999), Some(ScalarValue::Int64(7)));
    }

    #[test]
    fn literal_vector_of_nulls() {
        let col = LiteralValueVector::new(DataType::Utf8, None, 3);
        assert_eq!(col.value(2), None);
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn literal_vector_panics_past_the_end() {
        let col = LiteralValueVector::new(DataType::Int64, Some(ScalarValue::Int64(7)), 2);
        col.value(2);
    }

    #[test]
    fn works_through_a_trait_object() {
        // Operators will hold columns as Box/Arc<dyn ColumnVector>.
        let array = Float64Array::from(vec![1.5]);
        let col: Box<dyn ColumnVector> = Box::new(ArrowFieldVector::new(Arc::new(array)));
        assert_eq!(col.value(0), Some(ScalarValue::Float64(1.5)));
    }

    #[test]
    fn record_batch_mixes_arrow_and_literal_columns() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("bonus", DataType::Float64, true),
        ]));
        let id: Arc<dyn ColumnVector> =
            Arc::new(ArrowFieldVector::new(Arc::new(Int64Array::from(vec![1, 2]))));
        let bonus: Arc<dyn ColumnVector> = Arc::new(LiteralValueVector::new(
            DataType::Float64,
            Some(ScalarValue::Float64(0.1)),
            2,
        ));
        let batch = RecordBatch::new(schema, vec![id, bonus]);

        assert_eq!(batch.row_count(), 2);
        assert_eq!(batch.column_count(), 2);
        assert_eq!(batch.column(0).value(1), Some(ScalarValue::Int64(2)));
        assert_eq!(batch.column(1).value(1), Some(ScalarValue::Float64(0.1)));
    }

    #[test]
    fn record_batch_from_arrow_preserves_values_and_nulls() {
        let arrow_batch = ArrowRecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new("x", DataType::Int64, true)])),
            vec![Arc::new(Int64Array::from(vec![Some(5), None]))],
        )
        .unwrap();

        let batch: RecordBatch = arrow_batch.into();
        assert_eq!(batch.row_count(), 2);
        assert_eq!(batch.column(0).value(0), Some(ScalarValue::Int64(5)));
        assert_eq!(batch.column(0).value(1), None);
    }

    #[test]
    #[should_panic(expected = "differing lengths")]
    fn record_batch_rejects_ragged_columns() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int64, false),
            Field::new("b", DataType::Int64, false),
        ]));
        let two: Arc<dyn ColumnVector> =
            Arc::new(ArrowFieldVector::new(Arc::new(Int64Array::from(vec![1, 2]))));
        let three: Arc<dyn ColumnVector> = Arc::new(ArrowFieldVector::new(Arc::new(
            Int64Array::from(vec![1, 2, 3]),
        )));
        RecordBatch::new(schema, vec![two, three]);
    }

    #[test]
    fn project_by_name_subsets_and_reorders() {
        let schema = Schema::new(vec![
            Field::new("a", DataType::Int64, false),
            Field::new("b", DataType::Utf8, false),
            Field::new("c", DataType::Float64, false),
        ]);

        let projected = project_by_name(&schema, &["c", "a"]).unwrap();
        assert_eq!(projected.fields().len(), 2);
        assert_eq!(projected.field(0).name(), "c");
        assert_eq!(projected.field(1).name(), "a");

        assert!(project_by_name(&schema, &["nope"]).is_err());
    }
}
