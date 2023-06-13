## Summary

Describe your changes here specifying if the change is a bug fix, enhancement, new feature, etc.

## Test Plan

Describe how you tested and verified your changes here (changes captured in existing tests, built and ran new tests, etc.).

## Before Submitting
- [ ] Check mypy locally
    - `pip3 install mypy==1.2.0`
    - `mypy --ignore-missing-imports torchsig`
    - Address any error messages
- [ ] Lint check locally
    - `pip3 install flake8`
    - `flake8 --select=E9,F63,F7,F82 torchsig`
    - Address any error messages
- [ ] Run formatter if needed
    - `pip3 install git+https://github.com/GooeeIOT/pyfmt.git`
    - `pyfmt torchsig`
- [ ] Run test suite locally
    - `pytest --ignore-glob=*_figures.py --ignore-glob=*_benchmark.py`
    - Ensure tests are successful prior to submitting PR

