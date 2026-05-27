"""Future / handle to a task result."""

from typing import Any


class Future:
    """A handle to a value produced by a remote function call.

    For now the value is materialized eagerly when the Future is constructed.
    Once async execution lands, ``get()`` will block until the value is ready.
    """

    def __init__(self, value: Any) -> None:
        self._value = value

    def get(self) -> Any:
        return self._value
