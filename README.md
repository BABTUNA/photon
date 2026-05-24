# photon

A small distributed Python runtime, built in Rust + PyO3 to understand the abstractions Ray provides. If Ray is a beam, photon is the smallest unit of one.

## Status

Early work in progress.

## Build

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install maturin
maturin develop
python -c "import photon"
```
