"""Future / handle to a task result."""

from typing import Any

import cloudpickle

from photon._native import TaskHandle


class Future:
    """A handle to a value produced by a remote function call.

    Wraps a Rust-side ``TaskHandle``. ``get()`` blocks until the worker
    finishes and returns the unpickled result.
    """

    def __init__(self, handle: TaskHandle) -> None:
        self._handle = handle

    def get(self) -> Any:
        result_bytes = self._handle.get()
        return cloudpickle.loads(result_bytes)
