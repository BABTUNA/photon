//! Tokio runtime singleton.

use std::sync::OnceLock;
use tokio::runtime::Runtime;

static RUNTIME: OnceLock<Runtime> = OnceLock::new();

/// Lazily-initialized Tokio multi-threaded runtime. Lives for the duration
/// of the Python process.
pub(crate) fn runtime() -> &'static Runtime {
    RUNTIME.get_or_init(|| Runtime::new().expect("failed to start Tokio runtime"))
}
