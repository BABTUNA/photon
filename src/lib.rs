use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

/// Native Rust extension for photon.
///
/// Lives at `photon._native`; the Python package re-exports symbols
/// from here through `photon/__init__.py`.
#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(hello, m)?)?;
    m.add_function(wrap_pyfunction!(execute_task, m)?)?;
    Ok(())
}

/// Returns a greeting from the Rust extension.
///
/// Smoke test that the Python/Rust boundary marshals strings both ways.
#[pyfunction]
fn hello(name: &str) -> String {
    format!("Hello from photon, {}!", name)
}

/// Execute a Python callable synchronously and return its result.
///
/// First cut: runs in the caller's thread while holding the GIL.
/// Future commits will move execution onto a Tokio worker pool.
#[pyfunction]
fn execute_task<'py>(
    func: &Bound<'py, PyAny>,
    args: &Bound<'py, PyTuple>,
    kwargs: &Bound<'py, PyDict>,
) -> PyResult<Bound<'py, PyAny>> {
    func.call(args, Some(kwargs))
}
