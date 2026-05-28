use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};
use std::sync::mpsc;
use std::sync::{Mutex, OnceLock};
use tokio::runtime::Runtime;

/// Native Rust extension for photon.
///
/// Lives at `photon._native`; the Python package re-exports symbols
/// from here through `photon/__init__.py`.
#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(hello, m)?)?;
    m.add_function(wrap_pyfunction!(execute_task, m)?)?;
    m.add_class::<TaskHandle>()?;
    Ok(())
}

/// Returns a greeting from the Rust extension.
///
/// Smoke test that the Python/Rust boundary marshals strings both ways.
#[pyfunction]
fn hello(name: &str) -> String {
    format!("Hello from photon, {}!", name)
}

static RUNTIME: OnceLock<Runtime> = OnceLock::new();

/// Lazily-initialized Tokio multi-threaded runtime. Lives for the duration
/// of the Python process.
fn runtime() -> &'static Runtime {
    RUNTIME.get_or_init(|| Runtime::new().expect("failed to start Tokio runtime"))
}

/// Handle to a task running on the Tokio blocking pool.
///
/// Internally holds a one-shot mpsc receiver that delivers the task's
/// result once the worker thread finishes. `get()` may only be called once.
#[pyclass]
struct TaskHandle {
    receiver: Mutex<Option<mpsc::Receiver<PyResult<PyObject>>>>,
}

#[pymethods]
impl TaskHandle {
    /// Block until the task completes, then return its result.
    ///
    /// Releases the GIL while waiting so the worker thread can acquire it.
    fn get(&self, py: Python<'_>) -> PyResult<PyObject> {
        let receiver = self
            .receiver
            .lock()
            .map_err(|_| PyRuntimeError::new_err("task handle mutex poisoned"))?
            .take()
            .ok_or_else(|| PyRuntimeError::new_err("get() may only be called once"))?;

        py.allow_threads(move || receiver.recv())
            .map_err(|_| PyRuntimeError::new_err("task worker disconnected"))?
    }
}

/// Submit a Python callable to the Tokio blocking pool.
///
/// Returns a `TaskHandle` that resolves to the callable's result via `.get()`.
/// The callable runs on a worker thread which acquires the GIL when ready.
#[pyfunction]
fn execute_task(func: Py<PyAny>, args: Py<PyTuple>, kwargs: Py<PyDict>) -> PyResult<TaskHandle> {
    let (sender, receiver) = mpsc::channel();

    runtime().spawn_blocking(move || {
        let result = Python::with_gil(|py| {
            let func = func.bind(py);
            let args = args.bind(py);
            let kwargs = kwargs.bind(py);
            func.call(args, Some(kwargs)).map(|b| b.unbind())
        });
        let _ = sender.send(result);
    });

    Ok(TaskHandle {
        receiver: Mutex::new(Some(receiver)),
    })
}
