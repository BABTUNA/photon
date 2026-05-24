use pyo3::prelude::*;

/// Native Rust extension for photon.
///
/// Lives at `photon._native`; the Python package re-exports symbols
/// from here through `photon/__init__.py`.
#[pymodule]
fn _native(_m: &Bound<'_, PyModule>) -> PyResult<()> {
    Ok(())
}
