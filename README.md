# photon

A small distributed Python runtime, built in Rust + PyO3 to understand the abstractions Ray provides. If Ray is a beam, photon is the smallest unit of one.

## Status

Early work in progress.

## Requirements

- Rust 1.75+ (stable)
- Python 3.10+
- Linux or macOS (no Windows support — uses `mmap` and Unix sockets)

## Build

First-time setup:

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install maturin
maturin develop
```

Iterate (after the venv exists and is activated):

```sh
maturin develop
```

`maturin develop` compiles the Rust extension and installs it into the active venv as an editable package. Re-run it after any change to Rust source.

## Usage

```python
import photon

print(photon.hello("world"))
# Hello from photon, world!
```

More to come.
