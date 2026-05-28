"""Future / handle to a task result."""

from typing import Any

from photon._native import TaskHandle


class Future:
    """A handle to a value produced by a remote function call.

    Wraps a Rust-side ``TaskHandle``. ``get()`` blocks until the underlying
    worker thread finishes and returns the result.
    """

    def __init__(self, handle: TaskHandle) -> None:
        self._handle = handle

    def get(self) -> Any:
        return self._handle.get()
