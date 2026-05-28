"""Worker-side helpers — runs inside the Tokio blocking-pool thread."""

import cloudpickle


def run_pickled_task(payload: bytes) -> bytes:
    """Unpickle a (func, args, kwargs) tuple, run it, return the pickled result."""
    func, args, kwargs = cloudpickle.loads(payload)
    result = func(*args, **kwargs)
    return cloudpickle.dumps(result)
