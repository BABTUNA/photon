"""Smoke tests for the @remote / .remote() / .get() round trip."""

import pytest

import photon


@photon.remote
def double(x):
    return x * 2


@photon.remote
def add(a, b=10):
    return a + b


@photon.remote
def boom():
    raise ValueError("expected")


def test_round_trip_positional():
    assert double.remote(3).get() == 6


def test_round_trip_kwargs():
    assert add.remote(5, b=7).get() == 12


def test_direct_call_raises():
    with pytest.raises(RuntimeError):
        double(3)


def test_exception_propagates():
    # Exceptions surface at .get(), not .remote(), because execution is
    # deferred onto the Tokio blocking pool.
    ref = boom.remote()
    with pytest.raises(ValueError):
        ref.get()
