//! PyO3-exposed surface: top-level functions and classes visible from Python.

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use std::sync::mpsc;
use std::sync::Mutex;

use crate::runtime::runtime;

/// Returns a greeting from the Rust extension.
///
/// Smoke test that the Python/Rust boundary marshals strings both ways.
#[pyfunction]
pub(crate) fn hello(name: &str) -> String {
    format!("Hello from photon, {}!", name)
}

/// Handle to a task running on the Tokio blocking pool.
///
/// Internally holds a one-shot mpsc receiver that delivers the pickled
/// result bytes once the worker thread finishes. `get()` may only be called once.
#[pyclass]
pub(crate) struct TaskHandle {
    receiver: Mutex<Option<mpsc::Receiver<PyResult<Vec<u8>>>>>,
}

#[pymethods]
impl TaskHandle {
    /// Block until the task completes, then return the pickled result bytes.
    ///
    /// Releases the GIL while waiting so the worker thread can acquire it.
    fn get<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let receiver = self
            .receiver
            .lock()
            .map_err(|_| PyRuntimeError::new_err("task handle mutex poisoned"))?
            .take()
            .ok_or_else(|| PyRuntimeError::new_err("get() may only be called once"))?;

        let bytes_vec: Vec<u8> = py
            .allow_threads(move || receiver.recv())
            .map_err(|_| PyRuntimeError::new_err("task worker disconnected"))??;
        Ok(PyBytes::new_bound(py, &bytes_vec))
    }
}

/// Submit a pickled (func, args, kwargs) payload to the Tokio blocking pool.
///
/// The worker imports `photon._worker.run_pickled_task`, which unpickles
/// the payload, runs the callable, and returns the pickled result.
#[pyfunction]
pub(crate) fn execute_task(payload: Vec<u8>) -> PyResult<TaskHandle> {
    let (sender, receiver) = mpsc::channel();

    runtime().spawn_blocking(move || {
        let result = Python::with_gil(|py| -> PyResult<Vec<u8>> {
            let worker = py.import_bound("photon._worker")?;
            let result_obj = worker
                .getattr("run_pickled_task")?
                .call1((payload.as_slice(),))?;
            result_obj.extract()
        });
        let _ = sender.send(result);
    });

    Ok(TaskHandle {
        receiver: Mutex::new(Some(receiver)),
    })
}
