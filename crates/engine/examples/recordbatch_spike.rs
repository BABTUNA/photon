//! Spike: the Arrow memory model, hands-on.
//!
//! Run with: `cargo run --example recordbatch_spike`
//! Book chapter: "Apache Arrow".

use std::sync::Arc;

use arrow::array::{Array, ArrayRef, Float64Array, Int64Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use arrow::util::pretty::pretty_format_batches;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Columns are Arrays: contiguous, typed buffers. Building from
    //    Vec<Option<_>> gives a validity bitmap; None becomes a null
    //    without needing a sentinel value in the data itself.
    let ids = Int64Array::from(vec![1, 2, 3, 4]);
    let cities = StringArray::from(vec![
        Some("oakland"),
        Some("fresno"),
        None,
        Some("san jose"),
    ]);
    let temps = Float64Array::from(vec![Some(21.5), Some(35.1), Some(18.0), None]);

    // Nulls live in a separate bitmap; the values buffer still has a slot.
    assert_eq!(cities.null_count(), 1);
    assert!(cities.is_null(2));

    // 2. A Schema is the table shape: named, typed, nullability-aware fields.
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("city", DataType::Utf8, true),
        Field::new("temp_c", DataType::Float64, true),
    ]));

    // 3. ArrayRef = Arc<dyn Array> — type-erased so one Vec can hold columns
    //    of different types. Cloning is a refcount bump, not a data copy.
    let columns: Vec<ArrayRef> = vec![Arc::new(ids), Arc::new(cities), Arc::new(temps)];

    // 4. RecordBatch = schema + equal-length columns. This is the unit of
    //    data that flows through every operator we build from here on.
    let batch = RecordBatch::try_new(schema, columns)?;
    println!("{}", pretty_format_batches(std::slice::from_ref(&batch))?);

    // 5. Getting a typed view back: downcast through Any. This is the raw
    //    pattern our ColumnVector wrapper (commit 1.3) exists to hide.
    let ids = batch
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .expect("column 0 is Int64");
    let total: i64 = ids.iter().flatten().sum();
    println!("sum(id) = {total}");

    // 6. Slicing is zero-copy: same underlying buffers, new offset + length.
    let tail = batch.slice(2, 2);
    println!(
        "slice(2, 2) -> {} rows sharing the original buffers",
        tail.num_rows()
    );

    Ok(())
}
