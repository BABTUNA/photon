"""Decorator and remote-function wrapper."""

from typing import Any, Callable

from photon._native import execute_task
from photon.future import Future


class RemoteFunction:
    """A function that has been wrapped with ``@remote``.

    Calling it directly raises — use ``.remote(...)`` to submit it for execution.
    """

    def __init__(self, func: Callable[..., Any]) -> None:
        self._func = func
        self.__name__ = getattr(func, "__name__", "remote_function")

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError(
            f"Remote function {self.__name__!r} must be invoked via .remote(); "
            "direct calls are not supported."
        )

    def remote(self, *args: Any, **kwargs: Any) -> Future:
        """Submit this function for execution and return a Future."""
        handle = execute_task(self._func, args, kwargs)
        return Future(handle)


def remote(func: Callable[..., Any]) -> RemoteFunction:
    """Decorator that turns a function into a remote function."""
    return RemoteFunction(func)
