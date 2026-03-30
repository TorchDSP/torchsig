# Rust Functions for TorchSig

Rust implementations for various DSP computations for TorchSig. The code is built as part of the package using [maturin](https://www.maturin.rs/).

## Rust File Structure
```
rust/
├── Cargo.toml
├── README.md
├── src
│   └── lib.rs
```


## For Developers/Advanced Users
Most users do not need to do anything to this code in order to use TorchSig. However, if you want to see how we have implemented our code or want to override it for a specific purpose, see below.


### Build Code for TorchSig

1. Code your changes/functions inside `src/`. Most of our code is in `src/lib.rs`

2. Bind your Rust functions inside `src/lib.rs` in order to use in TorchSig. See [PyO3 user guide](https://pyo3.rs/v0.25.1/rust-from-python.html) to learn more about exposing Rust code to Python.

3. Reinstall TorchSig. `maturin` will compile the extension module and make it available within `torchsig`.
```bash
pip install -e .
```

4. Now you can call the Rust functions in Python:
```python
from torchsig.utils.rust_functions import <insert_rust_function_name>
```
