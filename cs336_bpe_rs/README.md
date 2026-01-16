# cs336_bpe_rs

Rust-accelerated pretokenization for the CS336 BPE assignment. This crate exposes a PyO3 module that replaces the
slowest part of training: GPT-2 regex pretokenization + token counting.

## Build & install (editable)

```bash
uv tool install maturin
maturin develop --release -m cs336_bpe_rs/Cargo.toml
```

If you prefer a wheel build:

```bash
maturin build --release -m cs336_bpe_rs/Cargo.toml
```

## Usage (Python)

Once installed, `cs336_basics/train_bpe.py` will automatically import the Rust module and use it for
pretokenization when available. You can keep the Python fallback by uninstalling the module.

## Parallelism

The Rust `train_bpe` function accepts an optional `num_threads` argument. The Python wrapper forwards
`num_workers` into Rust when present:

- `num_workers=0` (default): use Rayon default thread count
- `num_workers>0`: use that many threads for pretokenization
