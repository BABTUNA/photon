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


def test_lambda_works():
    square = photon.remote(lambda x: x * x)
    assert square.remote(4).get() == 16


def test_closure_works():
    factor = 10
    multiply = photon.remote(lambda x: x * factor)
    assert multiply.remote(3).get() == 30
