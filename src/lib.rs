use pyo3::prelude::*;

mod api;
mod object_store;
mod runtime;
mod segment_allocator;

/// Native Rust extension for photon.
///
/// Lives at `photon._native`; the Python package re-exports symbols
/// from here through `photon/__init__.py`.
#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(api::hello, m)?)?;
    m.add_function(wrap_pyfunction!(api::execute_task, m)?)?;
    m.add_class::<api::TaskHandle>()?;
    Ok(())
}
