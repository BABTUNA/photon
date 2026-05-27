"""Future / handle to a task result."""

from typing import Any


class Future:
    """A handle to a value that will be available later.

    Currently a stub — ``get()`` raises until the runtime is wired in.
    """

    def get(self) -> Any:
        # TODO(1.5): block until result is ready, then return it
        raise NotImplementedError("Future.get() is not implemented yet")
