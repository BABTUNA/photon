use pyo3::prelude::*;

/// Native Rust extension for photon.
///
/// Lives at `photon._native`; the Python package re-exports symbols
/// from here through `photon/__init__.py`.
#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(hello, m)?)?;
    Ok(())
}

/// Returns a greeting from the Rust extension.
///
/// Smoke test that the Python/Rust boundary marshals strings both ways.
#[pyfunction]
fn hello(name: &str) -> String {
    format!("Hello from photon, {}!", name)
}
